"""
🔔 Session Band Reversal (SBR) — рушій сигналів 1H + 4H (вимога 29.10).

Дослівно: «Візьми скрипт для TradingView "Session Band Reversal Signals" на
100% адаптуй його для роботи бота … Сигнал має відслідковуватись як на 1Н так
і на 4Н таймфреймах. Відтвори сигнали один в один … як у TradingView.»

ДЖЕРЕЛО ПРАВДИ — `tools/tradingview_session_band_signals.pine` (v4). Тут той
самий розрахунок, бар у бар:

    mid   = ta.ema(hlc3, 5)        atr = ta.atr(14)   (RMA від true range)
    upper = mid + 0.5·atr          lower = mid − 0.5·atr
    upP, loP, atrP = рівні ПОПЕРЕДНЬОГО закритого бару
    shortScore = clamp(47.8 + 41.2·(high − upP)/atrP)
    longScore  = clamp(47.8 + 41.2·(loP − low)/atrP)
    SHORT: high ≥ upP · score ≥ 55 · червона свічка · close > upP
    LONG : low  ≤ loP · score ≥ 55 · зелена свічка  · close < loP
    Trigger = high/low бару · SL = trigger ± 3·atrP · Risk за Score
    «#N» — нумерація в межах доби (Europe/Kyiv), окремо SHORT і LONG.

⚠️ ТОЧНІСТЬ «ОДИН В ОДИН». Pine-функції рекурсивні (EMA/RMA залежать від
усієї історії). Вплив стартової точки затухає як (13/14)^N для ATR і (2/3)^N
для EMA — на 1000 барах це ~1e-32, тобто числа збігаються з TradingView до
останнього знака, ЯКЩО свічки ті самі. Тому бари — Bybit (`fetch_klines`
ходить спершу в Bybit), рівно ті, що `BYBIT:<SYM>.P` на графіку TV.
⚠️ Сигнал рахується ЛИШЕ на ЗАКРИТОМУ барі (у Pine `onClose` = true за
замовчуванням: колір свічки остаточний тільки на закритті). Живий бар
відкидається — інакше мітка зʼявлялась би і зникала всередині години.

ШВИДКІСТЬ. Рушій не чекає загального скану: окремий потік щоразу через
`SETTLE_SEC` після закриття бару 1H (і 4H на межах 00/04/08… UTC) паралельно
(`WORKERS`) тягне `BARS` свічок КОЖНОЇ монети й одразу шле Telegram. Монета,
по якій біржа ще не віддала закритий бар, переопитується кожні `RETRY_SEC`.

ЛИШЕ ЧИСТІ ФУНКЦІЇ ВГОРІ (юніт-тести), I/O — у класі `SbrEngine`.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _KYIV = ZoneInfo('Europe/Kyiv')
except Exception:          # pragma: no cover — без tzdata
    _KYIV = None

# ── Параметри індикатора — ДЕФОЛТИ Pine-скрипта (не міняти без нього!) ──────
DEFAULTS = {
    'mid_len': 5, 'atr_len': 14, 'band_k': 0.5,
    'sc_base': 47.8, 'sc_slope': 41.2, 'min_score': 55.0,
    'sl_k': 3.0, 'th_add': 90.0, 'th_1r': 70.0, 'th_half': 60.0,
    'rev_candle': True, 'close_out': True,
}

TFS = ('1h', '4h')
TF_SEC = {'1h': 3600, '4h': 14400}
TF_LABEL = {'1h': '1H', '4h': '4H'}
BARS = 1000            # «без економії»: 1 сторінка Bybit = 1000 барів на TF
SETTLE_SEC = 2.0       # пауза після закриття бару — біржа встигає його закрити
RETRY_SEC = 5.0        # повтор для монет, по яких закритого бару ще немає
FAIL_BACKOFF_MAX = 300.0
WORKERS = 8
LOOP_SEC = 1.0
TAIL = 12              # скільки останніх сигналів TF тримаємо на монету
SEND_MAX_LAG_FRAC = 0.25   # сигнал старший за ¼ бару після закриття — тиха база
DB_KEY_STATE = 'sbr_state_v1'
DB_KEY_SETTINGS = 'sbr_settings'
DEFAULT_TG_SYMBOLS = ['BTCUSDT', 'ETHUSDT']
TG_CATEGORY = 'sbr'

DEFAULT_SETTINGS = {
    'enabled': True,          # рушій (≈ 2 запити на монету на годину)
    'tg_enabled': True,       # майстер-вимикач 📨 у тему «🔔 Сигнали»
    'tg_symbols': list(DEFAULT_TG_SYMBOLS),
    # 📶 TF сигналів у Telegram (вимога 29.10): 1H / 4H / обидва.
    # Обидва вимкнені = обидва (інакше тумблер мовчки гасив би всю тему).
    'tg_1h': True,
    'tg_4h': True,
}


def tg_tfs_of(settings) -> tuple:
    """TF, сигнали яких ідуть у Telegram. ЧИСТА."""
    s = settings or {}
    out = tuple(tf for tf, k in (('1h', 'tg_1h'), ('4h', 'tg_4h')) if bool(s.get(k, True)))
    return out or TFS


# ═══════════════════════════ ЧИСТІ ФУНКЦІЇ ═══════════════════════════════════

def _clamp(x: float) -> float:
    return max(0.0, min(100.0, x))


def pine_ema(src: List[float], length: int) -> List[Optional[float]]:
    """`ta.ema`: alpha = 2/(len+1), старт — SMA перших `len` значень."""
    out: List[Optional[float]] = [None] * len(src)
    if length <= 0 or len(src) < length:
        return out
    a = 2.0 / (length + 1)
    v = sum(src[:length]) / length
    out[length - 1] = v
    for i in range(length, len(src)):
        v = a * src[i] + (1 - a) * v
        out[i] = v
    return out


def pine_rma(src: List[float], length: int) -> List[Optional[float]]:
    """`ta.rma`: alpha = 1/len, старт — SMA перших `len` значень."""
    out: List[Optional[float]] = [None] * len(src)
    if length <= 0 or len(src) < length:
        return out
    a = 1.0 / length
    v = sum(src[:length]) / length
    out[length - 1] = v
    for i in range(length, len(src)):
        v = a * src[i] + (1 - a) * v
        out[i] = v
    return out


def pine_atr(h: List[float], l: List[float], c: List[float],
             length: int) -> List[Optional[float]]:
    """`ta.atr(len)` = `ta.rma(ta.tr(true), len)`; перший TR = high − low."""
    tr = []
    for i in range(len(h)):
        if i == 0:
            tr.append(h[i] - l[i])
        else:
            tr.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    return pine_rma(tr, length)


def risk_text(score: float, cfg: Dict = None) -> str:
    c = cfg or DEFAULTS
    if score >= c['th_add']:
        return '1R + Add'
    if score >= c['th_1r']:
        return '1R'
    if score >= c['th_half']:
        return '0.5R'
    return '0.25R'


def kyiv_dt(ts_sec: float) -> datetime:
    if _KYIV is not None:
        return datetime.fromtimestamp(ts_sec, _KYIV)
    return datetime.utcfromtimestamp(ts_sec + 2 * 3600)   # грубий фолбек


def closed_bars(bars: List[Dict], tf_sec: int, now: float) -> List[Dict]:
    """Лише ЗАКРИТІ бари (час відкриття + тривалість ≤ зараз)."""
    out = []
    for b in bars or []:
        try:
            t = float(b['t']) / 1000.0
        except Exception:
            continue
        if t + tf_sec <= now + 0.5:
            out.append(b)
    return out


def compute_signals(bars: List[Dict], tf_sec: int, cfg: Dict = None) -> List[Dict]:
    """ВСІ сигнали SBR на ЗАКРИТИХ барах `bars` (найстаріший першим).

    Бар: {t (мс, відкриття), o, h, l, p (close)} — формат `fetch_klines`;
    `c`/`close` теж приймаються. Повертає [{side, t, close_t, trigger, score,
    risk, sl, n, time_kyiv}] у хронологічному порядку.
    """
    c = dict(DEFAULTS)
    c.update(cfg or {})
    n = len(bars or [])
    if n < max(c['mid_len'], c['atr_len']) + 2:
        return []
    o, h, l, cl, t = [], [], [], [], []
    for b in bars:
        o.append(float(b['o']))
        h.append(float(b['h']))
        l.append(float(b['l']))
        cl.append(float(b.get('p', b.get('c', b.get('close')))))
        t.append(int(b['t']))
    hlc3 = [(h[i] + l[i] + cl[i]) / 3.0 for i in range(n)]
    mid = pine_ema(hlc3, int(c['mid_len']))
    atr = pine_atr(h, l, cl, int(c['atr_len']))
    out: List[Dict] = []
    s_cnt = l_cnt = 0
    last_day = None
    for i in range(n):
        ts = t[i] / 1000.0
        d = kyiv_dt(ts).day
        if last_day is None or d != last_day:
            s_cnt = l_cnt = 0
            last_day = d
        if i == 0 or mid[i - 1] is None or atr[i - 1] is None:
            continue
        atr_p = atr[i - 1]
        if not atr_p or atr_p <= 0:
            continue
        up_p = mid[i - 1] + c['band_k'] * atr_p
        lo_p = mid[i - 1] - c['band_k'] * atr_p
        s_sc = _clamp(c['sc_base'] + c['sc_slope'] * (h[i] - up_p) / atr_p)
        l_sc = _clamp(c['sc_base'] + c['sc_slope'] * (lo_p - l[i]) / atr_p)
        short_sig = (h[i] >= up_p and s_sc >= c['min_score']
                     and (not c['rev_candle'] or cl[i] < o[i])
                     and (not c['close_out'] or cl[i] > up_p))
        long_sig = (l[i] <= lo_p and l_sc >= c['min_score']
                    and (not c['rev_candle'] or cl[i] > o[i])
                    and (not c['close_out'] or cl[i] < lo_p))
        tk = kyiv_dt(ts).strftime('%H:%M')
        if short_sig:
            s_cnt += 1
            out.append({'side': 'SHORT', 't': t[i], 'close_t': ts + tf_sec,
                        'trigger': h[i], 'score': round(s_sc, 2),
                        'risk': risk_text(s_sc, c), 'sl': h[i] + c['sl_k'] * atr_p,
                        'n': s_cnt, 'time_kyiv': tk})
        if long_sig:
            l_cnt += 1
            out.append({'side': 'LONG', 't': t[i], 'close_t': ts + tf_sec,
                        'trigger': l[i], 'score': round(l_sc, 2),
                        'risk': risk_text(l_sc, c), 'sl': l[i] - c['sl_k'] * atr_p,
                        'n': l_cnt, 'time_kyiv': tk})
    return out


def tf_snapshot(bars: List[Dict], tf: str, now: float, cfg: Dict = None) -> Optional[Dict]:
    """Стан одного TF по монеті: останній закритий бар + хвіст сигналів."""
    tf_sec = TF_SEC[tf]
    cb = closed_bars(bars, tf_sec, now)
    if not cb:
        return None
    sigs = compute_signals(cb, tf_sec, cfg)
    last_t = int(cb[-1]['t'])
    tail = sigs[-TAIL:]
    return {'closed_t': last_t, 'bars': len(cb),
            'price': float(cb[-1].get('p', cb[-1].get('c', 0)) or 0),
            'last': tail[-1] if tail else None, 'tail': tail}


def combine(s1: Optional[Dict], s4: Optional[Dict]) -> Optional[Dict]:
    """🔀 ОДИН рядок на монету з двох TF (вимога 6: «якщо сигнал є на 1Н і
    потім повторюється на 4Н — додатковий запис не потрібен, колонка показує,
    який сигнал прийшов, або обидва»). ЧИСТА.

    • Напрямок = бік НАЙСВІЖІШОГО сигналу (за моментом закриття бару).
    • Колонка TF = усі TF, чий ОСТАННІЙ сигнал у тому самому боці
      (`'1H'` / `'4H'` / `'1H+4H'`).
    • Одночасне закриття з протилежними боками → вирішує 4H (старший TF).
    • Сила = Score вирішального сигналу (при збігу двох — більший).
    Немає жодного сигналу → None.
    """
    c = [(sig['close_t'], tf, sig) for tf, sig in (('1h', s1), ('4h', s4)) if sig]
    if not c:
        return None
    c.sort(key=lambda x: (x[0], 1 if x[1] == '4h' else 0))
    _, tf_d, sig_d = c[-1]
    side = sig_d['side']
    agree = [(tf, sig) for _, tf, sig in c if sig['side'] == side]
    tfs = [tf for tf in TFS if any(a[0] == tf for a in agree)]
    return {'side': side, 'tfs': tfs, 'tf_label': '+'.join(TF_LABEL[x] for x in tfs),
            'strength': int(round(max(sig['score'] for _, sig in agree))),
            'decider': tf_d, 'signal': sig_d,
            'by_tf': {tf: sig for tf, sig in agree}}


def is_fresh(snap: Optional[Dict]) -> bool:
    """Сигнал стоїть САМЕ на останньому закритому барі TF (щойно зʼявився)."""
    if not snap or not snap.get('last'):
        return False
    return int(snap['last']['t']) == int(snap.get('closed_t') or -1)


def fmt_price(p) -> str:
    try:
        from detection.ob_alert import fmt_price as _fp
        return _fp(p)
    except Exception:
        try:
            v = float(p)
        except Exception:
            return '—'
        return f'{v:,.2f}' if v >= 100 else (f'{v:.4f}' if v >= 1 else f'{v:.6f}')


def tg_text(symbol: str, items: List[tuple]) -> str:
    """Текст Telegram у форматі мітки Pine. `items` = [(tf, signal)] одного
    боку — 1H і 4H, що зійшлись на одному закритті, ідуть ОДНИМ повідомленням."""
    tf_lbl = '+'.join(TF_LABEL[tf] for tf, _ in items)
    side = items[0][1]['side']
    head_sig = max(items, key=lambda x: TF_SEC[x[0]])[1]    # старший TF — головний
    emo = '🟢' if side == 'LONG' else '🔴'
    lines = [f"🔔 <b>SBR · #{symbol}</b> {emo} <b>{side} #{head_sig['n']}</b> · {tf_lbl}"]
    for tf, sig in sorted(items, key=lambda x: TF_SEC[x[0]]):
        pre = f"[{TF_LABEL[tf]}] " if len(items) > 1 else ''
        lines.append(f"{pre}Time: {sig['time_kyiv']} Kyiv · Trigger: {fmt_price(sig['trigger'])}")
        lines.append(f"{pre}Score: {sig['score']:.2f} · Risk: {sig['risk']} · "
                     f"SL: {fmt_price(sig['sl'])}")
    return '\n'.join(lines)


# ═══════════════════════════ РУШІЙ (I/O) ═════════════════════════════════════

class SbrEngine:
    """Фоновий потік: тягне бари 1H/4H одразу після закриття, рахує сигнали,
    шле Telegram по монетах, відмічених кнопкою 🔔 у WATCHLIST."""

    def __init__(self, db=None, get_watchlist=None, fetch=None, notify=None,
                 clock=None):
        self.db = db
        self._get_watchlist = get_watchlist or (lambda: [])
        self._fetch = fetch            # (sym, tf, limit) → bars | None
        self._notify = notify          # (text) → (ok, why)
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._state: Dict[str, Dict] = {}      # {SYM: {'1h': snap, '4h': snap}}
        self._sent: Dict[str, int] = {}        # {'SYM|tf': t бару, про який уже сказано}
        self._fail: Dict[str, tuple] = {}      # {'SYM|tf': (k, наступна спроба)}
        self._attempt: Dict[str, float] = {}   # {tf: остання спроба}
        self._events: List[Dict] = []          # останні сигнали (для UI)
        self._ts = 0.0
        self._last_scan: Dict[str, Dict] = {}
        self._thread = None
        self._running = False
        self._settings = self._load_settings()
        self._restore()

    # ── налаштування ──────────────────────────────────────────────────────
    def _load_settings(self) -> Dict:
        s = dict(DEFAULT_SETTINGS)
        s['tg_symbols'] = list(DEFAULT_TG_SYMBOLS)
        try:
            raw = self.db.get_setting(DB_KEY_SETTINGS, None) if self.db else None
            if isinstance(raw, dict):
                s.update({k: v for k, v in raw.items() if k in DEFAULT_SETTINGS})
        except Exception:
            pass
        s['tg_symbols'] = sorted({str(x).upper() for x in (s.get('tg_symbols') or []) if x})
        s['enabled'] = bool(s.get('enabled', True))
        s['tg_enabled'] = bool(s.get('tg_enabled', True))
        s['tg_1h'] = bool(s.get('tg_1h', True))
        s['tg_4h'] = bool(s.get('tg_4h', True))
        if not (s['tg_1h'] or s['tg_4h']):
            s['tg_1h'] = s['tg_4h'] = True
        return s

    def get_settings(self) -> Dict:
        with self._lock:
            return {**self._settings, 'tg_symbols': list(self._settings['tg_symbols'])}

    def _save_settings(self):
        try:
            if self.db:
                self.db.set_setting(DB_KEY_SETTINGS, self.get_settings())
        except Exception as e:
            print(f"[SBR] settings save error: {e}")

    def update_settings(self, patch: Dict) -> Dict:
        with self._lock:
            if 'enabled' in patch:
                self._settings['enabled'] = bool(patch['enabled'])
            if 'tg_enabled' in patch:
                self._settings['tg_enabled'] = bool(patch['tg_enabled'])
            for k in ('tg_1h', 'tg_4h'):
                if k in patch:
                    self._settings[k] = bool(patch[k])
            if not (self._settings.get('tg_1h', True) or self._settings.get('tg_4h', True)):
                self._settings['tg_1h'] = self._settings['tg_4h'] = True
            if 'tg_symbols' in patch and isinstance(patch['tg_symbols'], list):
                self._settings['tg_symbols'] = sorted(
                    {str(x).upper() for x in patch['tg_symbols'] if x})
        self._save_settings()
        if self._settings['enabled']:
            self.start()
        return self.get_settings()

    def set_tg_symbol(self, symbol: str, on: bool) -> Dict:
        S = str(symbol or '').upper()
        if not S:
            return self.get_settings()
        with self._lock:
            cur = set(self._settings['tg_symbols'])
            (cur.add if on else cur.discard)(S)
            self._settings['tg_symbols'] = sorted(cur)
        self._save_settings()
        return self.get_settings()

    def is_enabled(self) -> bool:
        return bool(self._settings.get('enabled', True))

    # ── збереження стану ─────────────────────────────────────────────────
    def _restore(self):
        try:
            blob = self.db.get_setting(DB_KEY_STATE, None) if self.db else None
        except Exception:
            blob = None
        if not isinstance(blob, dict):
            return
        st = blob.get('state') or {}
        if isinstance(st, dict):
            self._state = {str(k).upper(): v for k, v in st.items() if isinstance(v, dict)}
        sent = blob.get('sent') or {}
        if isinstance(sent, dict):
            self._sent = {str(k): int(v) for k, v in sent.items()
                          if isinstance(v, (int, float))}
        ev = blob.get('events') or []
        if isinstance(ev, list):
            self._events = [e for e in ev if isinstance(e, dict)][-50:]
        self._ts = float(blob.get('ts') or 0.0)

    def _persist(self):
        try:
            if not self.db:
                return
            with self._lock:
                blob = {'ts': self._ts, 'state': dict(self._state),
                        'sent': dict(self._sent), 'events': list(self._events[-50:])}
            self.db.set_setting(DB_KEY_STATE, blob)
        except Exception as e:
            print(f"[SBR] persist error: {e}")

    # ── читання ──────────────────────────────────────────────────────────
    def rows(self) -> Dict[str, Dict]:
        """{SYM: combine(...) + snaps} — читає книга 🔔 МММ-SBR."""
        with self._lock:
            st = {k: dict(v) for k, v in self._state.items()}
        out = {}
        for S, v in st.items():
            s1, s4 = v.get('1h'), v.get('4h')
            cmb = combine((s1 or {}).get('last'), (s4 or {}).get('last'))
            out[S] = {'combo': cmb, '1h': s1, '4h': s4,
                      'fresh': {tf: is_fresh(v.get(tf)) for tf in TFS},
                      'price': (s1 or s4 or {}).get('price')}
        return out

    def state(self) -> Dict:
        now = self._clock()
        with self._lock:
            ls = {k: dict(v) for k, v in self._last_scan.items()}
            ev = list(self._events[-30:])
            n = len(self._state)
        nxt = {tf: (now // TF_SEC[tf] + 1) * TF_SEC[tf] + SETTLE_SEC for tf in TFS}
        return {'settings': self.get_settings(), 'running': self._running,
                'coins': n, 'ts': self._ts, 'last_scan': ls, 'next_at': nxt,
                'events': ev, 'server_now': now}

    # ── потік ────────────────────────────────────────────────────────────
    def start(self):
        if self._running or not self.is_enabled():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name='sbr-engine', daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        while self._running:
            try:
                if self.is_enabled():
                    self.tick()
            except Exception as e:
                print(f"[SBR] tick error: {e}")
            time.sleep(LOOP_SEC)

    def _universe(self) -> List[str]:
        try:
            wl = [str(x).upper() for x in (self._get_watchlist() or []) if x]
        except Exception:
            wl = []
        tg = self.get_settings()['tg_symbols']
        return list(dict.fromkeys(wl + tg))

    def _due(self, syms: List[str], tf: str, now: float) -> List[str]:
        """Монети, яким бракує ОСТАННЬОГО закритого бару TF."""
        tf_sec = TF_SEC[tf]
        bound = int(now // tf_sec) * tf_sec           # відкриття живого бару
        if now < bound + SETTLE_SEC:
            bound -= tf_sec                           # ще чекаємо закриття
        need_t = (bound - tf_sec) * 1000              # відкриття останнього закритого
        out = []
        with self._lock:
            for S in syms:
                snap = (self._state.get(S) or {}).get(tf)
                if snap and int(snap.get('closed_t') or 0) >= need_t:
                    continue
                f = self._fail.get(f'{S}|{tf}')
                if f and now < f[1]:
                    continue
                out.append(S)
        return out

    def tick(self) -> Dict:
        """Один прохід: по кожному TF — дотягнути монети без останнього
        закритого бару. Повертає {tf: [символи, що оновились]}."""
        now = self._clock()
        syms = self._universe()
        if not syms or self._fetch is None:
            return {}
        done: Dict[str, List[str]] = {}
        jobs = []
        for tf in TFS:
            due = self._due(syms, tf, now)
            if not due:
                continue
            if now - self._attempt.get(tf, 0.0) < RETRY_SEC and \
                    any((self._state.get(S) or {}).get(tf) for S in due):
                continue
            self._attempt[tf] = now
            jobs += [(S, tf) for S in due]
        if not jobs:
            return {}
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda j: (j, self._safe_fetch(*j)), jobs))
        now = self._clock()
        fresh: Dict[str, List[tuple]] = {}
        need_by_tf = {}
        for tf in TFS:
            tf_sec = TF_SEC[tf]
            bound = int(now // tf_sec) * tf_sec
            if now < bound + SETTLE_SEC:
                bound -= tf_sec
            need_by_tf[tf] = (bound - tf_sec) * 1000
        for (S, tf), bars in res:
            key = f'{S}|{tf}'
            snap = tf_snapshot(bars, tf, now) if bars else None
            if not snap:
                k = (self._fail.get(key) or (0, 0))[0] + 1
                self._fail[key] = (k, now + min(FAIL_BACKOFF_MAX, RETRY_SEC * 2 ** k))
                continue
            self._fail.pop(key, None)
            with self._lock:
                self._state.setdefault(S, {})[tf] = snap
            done.setdefault(tf, []).append(S)
            if snap['closed_t'] < need_by_tf[tf]:
                continue          # біржа ще не закрила бар → повтор за RETRY_SEC
            last = snap.get('last')
            if not is_fresh(snap):
                continue
            if self._sent.get(key) == int(last['t']):
                continue
            self._sent[key] = int(last['t'])
            lag = now - float(last['close_t'])
            if lag > TF_SEC[tf] * SEND_MAX_LAG_FRAC:
                continue          # бот лежав на закритті — тиха база, не спам
            fresh.setdefault(S, []).append((tf, last))
        with self._lock:
            self._ts = now
            for tf in TFS:
                if tf in done or any(j[1] == tf for j in jobs):
                    self._last_scan[tf] = {'at': now, 'took': round(time.time() - t0, 2),
                                           'asked': sum(1 for j in jobs if j[1] == tf),
                                           'ok': len(done.get(tf, []))}
        for S, items in fresh.items():
            self._on_signals(S, items, now)
        if done:
            self._kick_book()
            self._persist()
        return done

    def _kick_book(self):
        """Нові бари → книга 🔔 МММ-SBR перебудовується ОДРАЗУ (не чекаючи
        такту двигуна): вкладки LONG/SHORT змінюються в мить закриття бару."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            if ff is not None and hasattr(ff, 'on_sbr_update'):
                ff.on_sbr_update()
        except Exception:
            pass

    def _safe_fetch(self, S: str, tf: str):
        try:
            return self._fetch(S, tf, BARS)
        except Exception as e:
            print(f"[SBR] fetch {S} {tf} error: {e}")
            return None

    def _on_signals(self, S: str, items: List[tuple], now: float):
        """Свіжі сигнали монети: подія в журнал + Telegram (якщо відмічена)."""
        by_side: Dict[str, List[tuple]] = {}
        for tf, sig in items:
            by_side.setdefault(sig['side'], []).append((tf, sig))
        s = self.get_settings()
        for side, grp in by_side.items():
            tf_lbl = '+'.join(TF_LABEL[tf] for tf, _ in sorted(grp, key=lambda x: TF_SEC[x[0]]))
            ev = {'symbol': S, 'side': side, 'tfs': tf_lbl, 'at': now,
                  'score': max(g[1]['score'] for g in grp),
                  'trigger': grp[-1][1]['trigger'], 'tg': None}
            _tg_grp = [g for g in grp if g[0] in tg_tfs_of(s)]
            if s['tg_enabled'] and S in s['tg_symbols'] and self._notify and _tg_grp:
                try:
                    ok, why = self._notify(tg_text(S, _tg_grp))
                except Exception as e:
                    ok, why = False, str(e)
                ev['tg'] = bool(ok)
                if not ok:
                    ev['tg_err'] = str(why)[:160]
            with self._lock:
                self._events.append(ev)
                self._events = self._events[-50:]
            try:
                from detection.activity_log import log_activity
                txt = (f"🔔 SBR {tf_lbl} {side} · Score {ev['score']:.2f} · "
                       f"Trigger {fmt_price(ev['trigger'])}")
                if ev['tg'] is True:
                    txt += ' · 📨 надіслано в «🔔 Сигнали»'
                elif ev['tg'] is False:
                    txt += f" · 📨 Telegram НЕ прийняв: {ev.get('tg_err')}"
                if ev['tg'] is not None:
                    log_activity(S, 'signal', txt, side=side, source='SBR',
                                 extra={'tfs': tf_lbl, 'score': ev['score']})
            except Exception:
                pass


_instance: Optional[SbrEngine] = None


def _default_fetch(sym: str, tf: str, limit: int):
    from detection.market_data import get_market_data
    return get_market_data().fetch_klines(sym, limit=limit, interval=tf)


def _default_notify(text: str):
    from web.tg_bot import notify_category, last_send_error
    ok = bool(notify_category(TG_CATEGORY, text))
    return ok, ('' if ok else (last_send_error() or 'не надіслано'))


def init_sbr_engine(db=None, get_watchlist=None, fetch=None, notify=None) -> SbrEngine:
    global _instance
    if _instance is None:
        _instance = SbrEngine(db, get_watchlist, fetch or _default_fetch,
                              notify or _default_notify)
        if _instance.is_enabled():
            _instance.start()
    return _instance


def get_sbr_engine() -> Optional[SbrEngine]:
    return _instance
