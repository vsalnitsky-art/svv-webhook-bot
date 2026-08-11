"""
poc_setup — САМОСТІЙНИЙ моніторинг-рушій «POC-сетап» (НЕЗАЛЕЖНИЙ від Fuel
Auto-Filter; працює навіть коли FF повністю вимкнено).

Аналізує монети WATCHLIST і будує 5-шаровий конфлюенс навколо POC (Volume
Profile). Напрямок сетапу задає ПЕРШИЙ шар (POC).

Шари (кожен має колір напрямку: 🟢 LONG / 🔴 SHORT):
  L1 POC     — |ціна↔POC| ≥ poc_setup_pct% на вікні `window_days` (ТФ `tf`,
               ринок `market`). Задає НАПРЯМОК: ціна нижче POC → LONG, вище → SHORT.
  L2 1H      — прогноз 1H (дублює бейдж «🔮 1H»).
  L3 4H      — прогноз 4H (дублює бейдж «🔮 4H»).
  L4 Вердикт — Decision Center «SHORT 88% сильний».
  L5 OB      — Require OB на `ob_tf` (дефолт 15м). Активується ЛИШЕ коли L1..L4
               збіглись кольором (одним напрямком). Коли всі 5 збіглись →
               негайне відкриття угоди + видалення монети зі списку.

Колонки 7D/14D/30D — POC-дистанція(%) + ціна POC на цих вікнах (колір за напрямком).

Джерела: compute_poc (Binance SPOT/FUTURES, авто-фолбек Bybit) · forecast_engine
(1H/4H) · TradeManager.compute_decision (вердикт) · smc_ob_state (OB) ·
market_data (жива ціна). Відкриття — TradeManager.manual_open(bypass_gates=True),
тож на вході стампляться показники (у т.ч. POC) у хронологію угоди.

Тротлінг: за тік рахуємо `max_per_cycle` монет (round-robin), кеш пер-монета `ttl`.
"""

import time
import json
import threading
from typing import Dict, List, Optional, Callable

CYCLE_SECS = 12
_DB_SETTINGS = 'poc_setup_settings'
_DB_STATE = 'poc_setup_state'      # персист таблиці/озброєння між рестартами

DEFAULTS = {
    'enabled': False,          # майстер-тумблер
    'pct': 1.0,                 # поріг |ціна↔POC|, % (редаговане)
    'market': 'spot',           # spot | futures
    'tf': '1h',                 # ТФ POC
    'window_days': 7,           # вікно POC-сетапу (дні)
    'ob_tf': '15m',             # Require OB таймфрейм (L5, менший — edge-тригер)
    'ob_htf': '1h',             # Require OB СТАРШИЙ ТФ: якщо ОСТАННІЙ OB на ньому
                                #   у наш бік — відкриваємо ОДРАЗУ (SL з нього),
                                #   не чекаючи нового меншого. Порожньо = вимкнено.
    'auto_open': True,          # авто-відкриття коли всі 5 шарів
    'max_per_cycle': 5,         # тротл: монет за тік (безпечно для біржі)
    'ttl': 120,                 # TTL кешу пер-монета (с)
    'sl_buffer_pct': 0.10,      # буфер SL за межу OB-блоку, %
}

_WINDOWS = (7, 14, 30)          # колонки 7D/14D/30D
_REOPEN_GUARD = 300.0           # анти-повтор авто-відкриття (с)


def _strength_word(conf) -> str:
    try:
        c = float(conf or 0)
    except (TypeError, ValueError):
        c = 0.0
    return 'сильний' if c >= 66 else ('помірний' if c >= 40 else 'слабкий')


def _decision_text(dec: Optional[Dict]) -> (str, Optional[str]):
    """(«🧠 🔴 SHORT 88% сильний», dir) з вердикту Decision Center."""
    if not dec:
        return '', None
    rec = dec.get('recommended')
    vw = {'good': 'сильний', 'marginal': 'помірний', 'poor': 'слабкий'}
    word = vw.get(dec.get('verdict'), '')
    try:
        if rec == 'LONG':
            pct = round(float(dec.get('prob_long') or 0) * 100)
        elif rec == 'SHORT':
            pct = round(float(dec.get('prob_short') or 0) * 100)
        else:
            pct = round(max(float(dec.get('prob_long') or 0),
                            float(dec.get('prob_short') or 0)) * 100)
    except (TypeError, ValueError):
        pct = 0
    d = rec if rec in ('LONG', 'SHORT') else None
    icon = '🟢' if d == 'LONG' else ('🔴' if d == 'SHORT' else '⚖️')
    txt = f"{icon} {rec or 'WAIT'} {pct}%{(' ' + word) if word else ''}"
    return txt, d


class PocSetupDaemon:
    def __init__(self, db, get_watchlist: Callable, get_trade_manager: Callable):
        self._db = db
        self._get_watchlist = get_watchlist
        self._get_tm = get_trade_manager
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rows: Dict[str, Dict] = {}       # sym → row
        self._at: Dict[str, float] = {}         # sym → last compute ts
        self._rr: int = 0                       # round-robin cursor
        self._opened: Dict[str, float] = {}     # sym → last auto-open ts
        # L5 «озброєння»: коли L1–L4 збіглись — чекаємо РАЗОВОГО НОВОГО OB.
        # {sym: {'dir','baseline'(ob bar_time на момент озброєння),'since','ob_box'}}
        self._armed: Dict[str, Dict] = {}

    # ── settings ────────────────────────────────────────────────────────
    def get_settings(self) -> Dict:
        s = dict(DEFAULTS)
        try:
            raw = self._db.get_setting(_DB_SETTINGS, {}) or {}
            if isinstance(raw, str):
                raw = json.loads(raw or '{}')
            if isinstance(raw, dict):
                for k in DEFAULTS:
                    if k in raw:
                        s[k] = raw[k]
        except Exception:
            pass
        return s

    def update_settings(self, patch: Dict) -> Dict:
        s = self.get_settings()
        if isinstance(patch, dict):
            for k in DEFAULTS:
                if k in patch:
                    s[k] = patch[k]
        try:
            self._db.set_setting(_DB_SETTINGS, s)
        except Exception as e:
            print(f"[POC-setup] settings persist error: {e}")
        if s.get('enabled'):
            self.start()
        return s

    # ── персист таблиці/стану (переживає рестарт) ───────────────────────
    def _persist_state(self):
        try:
            with self._lock:
                blob = {'rows': self._rows, 'armed': self._armed,
                        'at': self._at, 'opened': self._opened}
            self._db.set_setting(_DB_STATE, blob)
        except Exception as e:
            print(f"[POC-setup] persist error: {e}")

    def _restore_state(self):
        try:
            raw = self._db.get_setting(_DB_STATE, {}) or {}
            if isinstance(raw, str):
                raw = json.loads(raw or '{}')
            if not isinstance(raw, dict):
                return
            with self._lock:
                self._rows = dict(raw.get('rows') or {})
                self._armed = dict(raw.get('armed') or {})
                self._at = {k: float(v) for k, v in (raw.get('at') or {}).items()}
                self._opened = {k: float(v) for k, v in (raw.get('opened') or {}).items()}
            print(f"[POC-setup] restored {len(self._rows)} rows from DB")
        except Exception as e:
            print(f"[POC-setup] restore error: {e}")

    def is_enabled(self) -> bool:
        return bool(self.get_settings().get('enabled'))

    def set_enabled(self, on: bool) -> Dict:
        return self.update_settings({'enabled': bool(on)})

    # ── helpers ─────────────────────────────────────────────────────────
    def _watchlist(self) -> List[str]:
        try:
            from detection.smc_scanner import get_smc_scanner
            sc = get_smc_scanner()
            if sc:
                return [s.upper() for s in (sc.get_watchlist() or [])]
        except Exception:
            pass
        try:
            return [s.upper() for s in (self._get_watchlist() or [])]
        except Exception:
            return []

    def _price(self, symbol: str) -> Optional[float]:
        """Запасне джерело ціни — ОСТАННІЙ close klines (MarketData НЕ має
        get_ticker!). Основний шлях — last_close із compute_poc."""
        try:
            from detection.market_data import get_market_data
            md = get_market_data()
            if md and hasattr(md, 'fetch_klines'):
                kl = md.fetch_klines(symbol, limit=2)
                if kl:
                    last = kl[-1]
                    # market_data повертає dict {p,o,h,l,v}: p = close/price.
                    p = last.get('p') if isinstance(last, dict) else None
                    if p and float(p) > 0:
                        return float(p)
        except Exception:
            pass
        return None

    def _has_position(self, symbol: str) -> bool:
        """Чи вже є відкрита позиція (real/paper) по монеті — щоб не показувати
        її в таблиці й не відкривати повторно."""
        try:
            tm = self._get_tm() if self._get_tm else None
            if not tm:
                return False
            for book in ('_positions', '_shadow_positions'):
                d = getattr(tm, book, None)
                if isinstance(d, dict) and symbol in d:
                    return True
        except Exception:
            pass
        return False

    def _forecast(self, symbol: str):
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            if not fe:
                return {}, {}
            fc = fe.ensure_fresh(symbol, max_age=300) or fe.get(symbol) or {}
            return (fc.get('forecast_1h') or {}), (fc.get('forecast_4h') or {})
        except Exception:
            return {}, {}

    def _ob_info(self, sym: str, ob_tf: str) -> Optional[Dict]:
        """Останній валідний OB на `ob_tf`: {dir, top, bottom, bar_time} або None.
        ON-DEMAND (як сканер у _update_smc_ob), бо сканер тримає OB лише на своєму
        ob_filter_timeframe. Fallback — DB smc_ob_state (без точного bar_time-edge)."""
        try:
            from detection.market_data import get_market_data
            from detection.smc_structure import detect_smc_structure
            from detection.ob_detector import detect_last_order_block
            md = get_market_data()
            kl = None
            if md and hasattr(md, 'fetch_klines'):
                try:
                    kl = md.fetch_klines(sym, limit=700, interval=ob_tf)
                except Exception:
                    kl = md.fetch_klines(sym, limit=700)
            if kl and len(kl) >= 220:
                closed = kl[:-1] if len(kl) >= 2 else kl
                isize, ssize = 5, 50
                try:
                    from detection.smc_scanner import get_smc_scanner
                    sc = get_smc_scanner()
                    if sc:
                        isize = sc.get_internal_size()
                        ssize = int(sc._settings.get('swing_size', 50))
                except Exception:
                    pass
                res = detect_smc_structure(closed, internal_size=isize, swing_size=ssize)
                internal = res.get('internal', {})
                ob = detect_last_order_block(klines=closed,
                                             pivots=internal.get('pivots', []),
                                             events=internal.get('events', []))
                bias = (ob or {}).get('bias')
                if bias in ('BULLISH', 'BEARISH'):
                    return {'dir': 'LONG' if bias == 'BULLISH' else 'SHORT',
                            'top': ob.get('bar_high'), 'bottom': ob.get('bar_low'),
                            'bar_time': ob.get('bar_time')}
                return None
        except Exception:
            pass
        # Fallback — те, що встиг порахувати сканер у БД.
        try:
            from storage.db_operations import get_db as _gdb
            row = _gdb().get_smc_ob_state(sym, ob_tf)
            bias = (row or {}).get('bias')
            if bias in ('BULLISH', 'BEARISH'):
                return {'dir': 'LONG' if bias == 'BULLISH' else 'SHORT',
                        'top': row.get('bar_high'), 'bottom': row.get('bar_low'),
                        'bar_time': row.get('bar_time')}
        except Exception:
            pass
        return None

    @staticmethod
    def _fc_layer(fc: Dict):
        """(dir, «SHORT -58% · 75%») з forecast-бейджа."""
        if not fc or fc.get('side') not in (1, -1):
            return None, '—'
        d = 'LONG' if fc['side'] == 1 else 'SHORT'
        pct = fc.get('pct')
        conf = fc.get('confidence')
        sign = '+' if (isinstance(pct, (int, float)) and pct > 0) else ''
        if pct is not None:
            return d, f"{d} {sign}{pct}% · {conf}%"
        return d, f"{d} · {conf}%"

    # ── per-coin compute ────────────────────────────────────────────────
    def _compute_one(self, sym: str, s: Dict) -> Optional[Dict]:
        from detection.volume_profile import compute_poc
        market = s.get('market', 'spot')
        tf = s.get('tf', '1h')
        win = int(s.get('window_days', 7) or 7)
        thr = float(s.get('pct', 1.0) or 1.0)
        ob_tf = s.get('ob_tf', '15m')
        # Монета вже в угоді → не показуємо (і не рахуємо зайве).
        if self._has_position(sym):
            self._armed.pop(sym, None)
            return None
        # POC per window (setup window + 7/14/30 columns). Поточну ціну беремо з
        # last_close результату compute_poc (БЕЗ окремого ticker-API — його немає).
        windows = sorted(set([win] + list(_WINDOWS)))
        raw: Dict[int, Dict] = {}
        price = None
        for d in windows:
            try:
                r = compute_poc(sym, hours=d * 24, bins=150, interval=tf, market=market)
            except Exception:
                r = None
            if r and r.get('ok') and r.get('poc'):
                raw[d] = r
                if price is None and r.get('last_close'):
                    price = float(r['last_close'])
        if price is None:
            price = self._price(sym)   # запасний шлях (klines)
        if price is None or win not in raw:
            return None
        pocs: Dict[int, Optional[Dict]] = {}
        for d, r in raw.items():
            poc = float(r['poc'])
            dist = (price - poc) / poc * 100.0
            wdir = 'LONG' if price < poc else ('SHORT' if price > poc else None)
            pocs[d] = {'poc': poc, 'dist_pct': round(dist, 2), 'dir': wdir,
                       'exchange': r.get('exchange')}
        base = pocs.get(win)
        if not base:
            return None
        # L1 — POC (задає напрямок). ВІДБІР У ТАБЛИЦЮ: показуємо ЛИШЕ монети, що
        # відповідають порогу % (|ціна↔POC| ≥ pct). Інакше монету не додаємо.
        l1_lit = (base['dist_pct'] is not None and abs(base['dist_pct']) >= thr
                  and base['dir'] in ('LONG', 'SHORT'))
        if not l1_lit:
            return None
        setup_dir = base['dir']
        l1_val = (f"{base['dir']} {base['dist_pct']:+.2f}% ≥ {thr:g}%" if l1_lit
                  else (f"{base['dist_pct']:+.2f}% < {thr:g}%"
                        if base['dist_pct'] is not None else '—'))
        # L2/L3 — forecast 1H/4H
        f1, f4 = self._forecast(sym)
        l2_dir, l2_val = self._fc_layer(f1)
        l3_dir, l3_val = self._fc_layer(f4)
        # L4 — Decision Center вердикт
        l4_dir, l4_val = None, '—'
        try:
            tm = self._get_tm() if self._get_tm else None
            if tm and hasattr(tm, 'compute_decision'):
                dec = tm.compute_decision(sym, price)
                l4_val, l4_dir = _decision_text(dec)
                if not l4_val:
                    l4_val = '—'
        except Exception:
            pass
        aligned4 = bool(setup_dir) and (l2_dir == setup_dir) and \
            (l3_dir == setup_dir) and (l4_dir == setup_dir)
        # L5 — Require OB як РАЗОВИЙ (edge) тригер. Коли L1–L4 збіглись —
        # «озброюємось» і чекаємо НОВИЙ OB (bar_time новіший за момент озброєння)
        # у той самий бік. Якщо L1–L4 порушились — роззброюємось (новий цикл
        # → нове очікування нового OB). Спрацьовує РАЗ.
        ob_htf = (s.get('ob_htf', '1h') or '').strip()
        now_ts = time.time()
        l5_dir, l5_lit, l5_val = None, False, 'чекає L1–L4'
        ob_box = None
        if not aligned4:
            self._armed.pop(sym, None)
        else:
            # 0) СПОЧАТКУ — СТАРШИЙ OB (HTF, дефолт 1H): якщо ОСТАННІЙ OB на ньому
            #    вже у наш бік — відкриваємо ОДРАЗУ (SL з цього HTF-OB), НЕ чекаючи
            #    нового меншого TF.
            htf = self._ob_info(sym, ob_htf) if ob_htf else None
            if htf and htf.get('dir') == setup_dir:
                l5_dir, l5_lit = setup_dir, True
                l5_val = f"OB {ob_htf.upper()} {setup_dir} ✓ (старший)"
                ob_box = {'top': htf.get('top'), 'bottom': htf.get('bottom')}
                self._armed.pop(sym, None)   # HTF-шлях — LTF-озброєння не потрібне
                info = None
            else:
                # 1) LTF-шлях: чекаємо НОВИЙ OB на меншому TF (edge-latch).
                info = self._ob_info(sym, ob_tf)
                cur_t = (info or {}).get('bar_time')
                ob_dir = (info or {}).get('dir')
                armed = self._armed.get(sym)
                if not armed or armed.get('dir') != setup_dir:
                    # (пере)озброєння: baseline = поточний OB, чекаємо НОВІШИЙ
                    self._armed[sym] = {'dir': setup_dir, 'baseline': cur_t,
                                        'since': now_ts}
                    l5_val = f"озброєно · чекаємо новий OB {ob_tf.upper()}"
                else:
                    base_t = armed.get('baseline')
                    is_new = (cur_t is not None and (base_t is None or cur_t > base_t))
                    if is_new and ob_dir == setup_dir:
                        l5_dir, l5_lit = ob_dir, True
                        l5_val = f"OB {ob_tf.upper()} {ob_dir} ✓ (новий)"
                        ob_box = {'top': (info or {}).get('top'),
                                  'bottom': (info or {}).get('bottom')}
                        armed['ob_box'] = ob_box
                    elif is_new and ob_dir and ob_dir != setup_dir:
                        # новий OB проти напрямку → переозброюємось на цей baseline
                        self._armed[sym] = {'dir': setup_dir, 'baseline': cur_t,
                                            'since': now_ts}
                        l5_val = f"новий OB проти — переозброєно"
                    else:
                        l5_val = f"озброєно · чекаємо новий OB {ob_tf.upper()}"
        layers = [
            {'n': 1, 'lit': bool(l1_lit), 'dir': (base['dir'] if l1_lit else None), 'val': l1_val},
            {'n': 2, 'lit': bool(l2_dir and l2_dir == setup_dir), 'dir': l2_dir, 'val': l2_val},
            {'n': 3, 'lit': bool(l3_dir and l3_dir == setup_dir), 'dir': l3_dir, 'val': l3_val},
            {'n': 4, 'lit': bool(l4_dir and l4_dir == setup_dir), 'dir': l4_dir, 'val': l4_val},
            {'n': 5, 'lit': bool(l5_lit), 'dir': l5_dir, 'val': l5_val},
        ]
        match_count = sum(1 for L in layers if L['lit'])
        all5 = bool(aligned4 and l5_lit)
        return {
            'symbol': sym, 'dir': setup_dir, 'price': price,
            'layers': layers, 'match_count': match_count,
            'aligned4': aligned4, 'all5': all5,
            'poc7': pocs.get(7), 'poc14': pocs.get(14), 'poc30': pocs.get(30),
            'setup_win': win, 'poc_pct': thr,
            'exchange': base.get('exchange'),   # біржа джерела POC (L1)
            'decision_text': l4_val, 'decision_dir': l4_dir,
            # для відкриття: SL = межа OB-блоку + буфер; TP = POC СЕТАП-вікна
            # (= напрямок L1 виводиться з неї, тож TP ГАРАНТОВАНО з правильного
            # боку ціни; за замовч. вікно=7 → це і є поле «7D»).
            'ob_box': ob_box, 'tp': base.get('poc'),
            'ts': time.time(),
        }

    # ── loop ────────────────────────────────────────────────────────────
    def _run(self):
        self._stop.wait(8)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[POC-setup] tick error: {e}")
            self._stop.wait(CYCLE_SECS)

    def _tick(self):
        s = self.get_settings()
        if not s.get('enabled'):
            with self._lock:
                if self._rows:
                    self._rows.clear()
            return
        wl = self._watchlist()
        if not wl:
            return
        cap = max(1, int(s.get('max_per_cycle', 6) or 6))
        ttl = float(s.get('ttl', 120) or 120)
        now = time.time()
        n = len(wl)
        picked, scanned, i = [], 0, self._rr
        while len(picked) < cap and scanned < n:
            sym = wl[i % n]
            i += 1
            scanned += 1
            if now - self._at.get(sym, 0) >= ttl:
                picked.append(sym)
        self._rr = i % n
        for sym in picked:
            try:
                row = self._compute_one(sym, s)
            except Exception as e:
                print(f"[POC-setup] {sym} compute error: {e}")
                row = None
            self._at[sym] = now
            with self._lock:
                if row:
                    self._rows[sym] = row
                else:
                    self._rows.pop(sym, None)
        # drop coins no longer in WATCHLIST
        wlset = set(wl)
        with self._lock:
            for k in list(self._rows.keys()):
                if k not in wlset:
                    self._rows.pop(k, None)
        # auto-open
        if s.get('auto_open', True):
            self._auto_open(s, now)
        # Зберігаємо таблицю/стан — переживе рестарт (не рахуємо все з нуля).
        self._persist_state()

    def _auto_open(self, s: Dict, now: float):
        with self._lock:
            ready = [(sym, r['dir']) for sym, r in self._rows.items()
                     if r.get('all5') and r.get('dir') in ('LONG', 'SHORT')]
        for sym, side in ready:
            if now - self._opened.get(sym, 0) < _REOPEN_GUARD:
                continue
            if self._open_symbol(sym, side, manual=False):
                self._opened[sym] = now

    def _open_symbol(self, sym: str, side: str, manual: bool = False) -> bool:
        """Відкриваємо угоду через TradeManager (bypass_gates=True), СТАВИМО
        Manual SL (межа OB-блоку + буфер) і Manual TP (ціна POC 7D), прибираємо
        монету зі списку. Мітка opened_by = «🎯 POC-сетап». Працює однаково для
        авто- і ручного (кнопка) відкриття."""
        try:
            tm = self._get_tm() if self._get_tm else None
            if not tm or not hasattr(tm, 'manual_open'):
                return False
            s = self.get_settings()
            with self._lock:
                row = dict(self._rows.get(sym) or {})
            tag = f"🎯 POC-сетап{' (ручне)' if manual else ''} {side}"
            res = tm.manual_open(sym, side, bypass_gates=True, opened_by=tag) or {}
            ok = bool(res.get('real_opened') or res.get('shadow_opened')
                      or res.get('status') == 'opened' or res.get('ok'))
            if not ok:
                return False
            is_paper = bool(res.get('is_paper'))
            # ── Manual SL = межа OB-блоку + буфер; Manual TP = ціна POC 7D ──
            try:
                self._apply_sl_tp(tm, sym, side, row, s, is_paper)
            except Exception as e:
                print(f"[POC-setup] SL/TP set warn {sym}: {e}")
            with self._lock:
                self._rows.pop(sym, None)
                self._armed.pop(sym, None)
            self._at[sym] = time.time()
            try:
                from detection.activity_log import log_activity
                log_activity(sym, 'opened',
                             f'{tag} — {"ручне" if manual else "авто"} відкриття (POC-сетап)',
                             side=side, source='POC')
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[POC-setup] open error {sym}: {e}")
            return False

    def _apply_sl_tp(self, tm, sym: str, side: str, row: Dict, s: Dict, is_paper: bool):
        """SL = межа OB-блоку ± буфер (SHORT→top×(1+buf), LONG→bottom×(1−buf)),
        TP = POC сетап-вікна. OB-бокс беремо зі спрацьованого L5; для ручної
        кнопки (L5 ще не спалахнув) — рахуємо OB наживо, але ЛИШЕ у бік угоди.

        ВАЖЛИВО: update_manual_sl_tp відхиляє ОБИДВА рівні атомарно, якщо хоч один
        не з того боку ціни. Тому шлемо SL і TP ОКРЕМИМИ викликами (щоб один
        невдалий не блокував інший) і логуємо причину відмови."""
        if not hasattr(tm, 'update_manual_sl_tp'):
            return
        try:
            buf = float(s.get('sl_buffer_pct', 0.10) or 0.10) / 100.0
        except (TypeError, ValueError):
            buf = 0.001
        box = row.get('ob_box')
        if not box or box.get('top') is None or box.get('bottom') is None:
            # Фолбек: беремо ОСТАННІЙ OB у БІК угоди (старший → менший TF).
            box = None
            for tf in [(s.get('ob_htf', '1h') or '').strip(), s.get('ob_tf', '15m')]:
                if not tf:
                    continue
                info = self._ob_info(sym, tf)
                if info and info.get('dir') == side and info.get('top') and info.get('bottom'):
                    box = {'top': info['top'], 'bottom': info['bottom']}
                    break
        sl = None
        if box and box.get('top') and box.get('bottom'):
            if side == 'SHORT':
                sl = float(box['top']) * (1.0 + buf)
            else:
                sl = float(box['bottom']) * (1.0 - buf)
        tp = row.get('tp')
        try:
            from detection.activity_log import log_activity
        except Exception:
            log_activity = lambda *a, **k: None
        # SL і TP — ОКРЕМИМИ викликами (анти-атомарна відмова).
        if sl is not None and sl > 0:
            r = tm.update_manual_sl_tp(sym, manual_sl=sl, is_shadow=is_paper) or {}
            if r.get('ok'):
                log_activity(sym, 'autosl', f"POC-сетап SL → {self._fmt(sl)} (OB+буфер {buf*100:g}%)",
                             side=side, source='POC')
            else:
                log_activity(sym, 'skipped', f"POC-сетап SL не встановлено: {r.get('reason', '?')}",
                             side=side, source='POC')
        if tp is not None and tp > 0:
            r = tm.update_manual_sl_tp(sym, manual_tp=tp, is_shadow=is_paper) or {}
            if r.get('ok'):
                log_activity(sym, 'autosl', f"POC-сетап TP → {self._fmt(tp)} (POC вікна)",
                             side=side, source='POC')
            else:
                log_activity(sym, 'skipped', f"POC-сетап TP не встановлено: {r.get('reason', '?')}",
                             side=side, source='POC')

    @staticmethod
    def _fmt(x):
        try:
            x = float(x)
            return f"{x:.6g}"
        except (TypeError, ValueError):
            return str(x)

    def open_symbol(self, symbol: str) -> Dict:
        """Ручне відкриття з кнопки таблиці."""
        sym = (symbol or '').upper().strip()
        if sym.endswith('.P'):
            sym = sym[:-2]
        with self._lock:
            r = self._rows.get(sym)
        side = (r or {}).get('dir')
        if side not in ('LONG', 'SHORT'):
            # напрямок задає L1 (POC); якщо його ще нема — не відкриваємо наосліп
            return {'ok': False, 'reason': 'немає напрямку POC (L1) для монети'}
        ok = self._open_symbol(sym, side, manual=True)
        return {'ok': bool(ok), 'symbol': sym, 'side': side}

    def _tm_ready(self):
        """(ready, reason) — чи здатний TradeManager відкрити угоду ЗАРАЗ:
        увімкнено real (enabled) АБО paper (test_mode). Інакше POC-сетап нікуди
        не відкриє — і це головна причина «5/5, а угоди немає»."""
        try:
            tm = self._get_tm() if self._get_tm else None
            if not tm:
                return False, 'TradeManager недоступний'
            st = tm.get_settings() if hasattr(tm, 'get_settings') else {}
            real = bool(st.get('enabled'))
            paper = bool(st.get('test_mode'))
            if real:
                return True, 'real'
            if paper:
                return True, 'paper'
            return False, 'Trade Manager і Test Mode вимкнені'
        except Exception:
            return False, 'н/д'

    # ── state for UI ────────────────────────────────────────────────────
    def get_state(self) -> Dict:
        s = self.get_settings()
        with self._lock:
            rows = [dict(r) for r in self._rows.values()]
        # Сортуємо за «найкращим» (найближчим до відкриття): більше збігів шарів
        # вище; тайбрейк — aligned4, потім менша |дистанція 7D|.
        def _key(r):
            d7 = ((r.get('poc7') or {}).get('dist_pct'))
            return (-int(r.get('match_count') or 0),
                    0 if r.get('aligned4') else 1,
                    abs(d7) if isinstance(d7, (int, float)) else 9e9)
        rows.sort(key=_key)
        tm_ready, tm_reason = self._tm_ready()
        return {'enabled': bool(s.get('enabled')), 'settings': s,
                'rows': rows, 'count': len(rows),
                'auto_open': bool(s.get('auto_open', True)),
                'tm_ready': tm_ready, 'tm_reason': tm_reason,
                'ready5': sum(1 for r in rows if r.get('all5')),
                'running': bool(self._thread and self._thread.is_alive())}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='poc-setup')
        self._thread.start()
        print("[POC-setup] daemon started")


_instance: Optional[PocSetupDaemon] = None


def init_poc_setup(db, get_watchlist, get_trade_manager) -> PocSetupDaemon:
    global _instance
    if _instance is None:
        _instance = PocSetupDaemon(db, get_watchlist, get_trade_manager)
        try:
            _instance._restore_state()   # відновити таблицю після рестарту
        except Exception:
            pass
        try:
            if _instance.is_enabled():
                _instance.start()
                print("[POC-setup] restored ON state from DB — loop running")
        except Exception:
            pass
    return _instance


def get_poc_setup() -> Optional[PocSetupDaemon]:
    return _instance
