"""🔭 «Пружина — скан біржі» (вимога 08.11).

Дослівно: «Організуй повністю ідентичний блок і алгоритм як "Пружина" тільки
працюватимемо не із WATCHLIST, а повністю в ручному режимі скануватимемо
повністю біржу (на вибір Bybit або Binance) на виявлення саме вже заряджених
на даний момент монети… відбиратимемо по об'єму за 24 години (за
замовчуванням 50М). І в таблицю додаємо лише заряджені на момент сканування
монети.» + «Організуй красиву візуалізацію процесу сканування.»

Принципи:
- Розрахунок — ТОЙ САМИЙ `spring_coil.analyze` і той самий `shape_row`, що й
  панелі 🌀 Пружина / 💰 Funding — Пружина (одна метрика — одне джерело).
- Лише РУЧНИЙ запуск. Монети = USDT-перпетуали обраної біржі з обігом за
  24 год ≥ порогу (без стейблів/обгорток). ЗАКРИТІ 1h-бари (живий бар
  відкидається), ОДИН запит на монету, паралельно.
- У таблицю потрапляють ЛИШЕ монети зі станом `charged` на момент скану.
- Через спільну `scan_queue` — за раз працює лише один скан біржі.
- Прогрес (етап, N/M, поточна монета, знайдені) живе в памʼяті й віддається
  сторінці — це і є візуалізація процесу.
- Результат персиститься (`DB_KEY`) і переживає рестарт.
- ЛИШЕ ПОКАЗ: нічого не відкриває. Telegram — лише попередження «монету
  взято в роботу» (кнопки 1H/4H у рядку + збіг сигналу SBR з напрямком).

Доповнення 09.11: фільтр заряду `min_charge` (деф. 80) · автоскан
(`auto_on`, раз на `auto_min`=30 хв) — повторний скан ДОПОВНЮЄ таблицю ·
перерахунок заряду ЛИШЕ монет таблиці на кожному закритому 1h-барі: статус
уже не «🌀 заряджена» → монета вибуває · 1H/4H → TG при сигналі SBR.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional

EXCHANGES = ('bybit', 'binance')
DEFAULT_EXCHANGE = 'bybit'
DEFAULT_MIN_VOL = 50_000_000.0
MIN_VOL_FLOOR = 1_000_000.0
LIMIT = 300                 # ≥ spring_coil.MIN_BARS (220) з запасом; 1 сторінка біржі
WORKERS = 8
MAX_SYMBOLS = 400           # стеля на випадок дуже низького порогу обігу
RECENT_N = 14               # стрічка «щойно перевірені»
DB_KEY = 'spring_scan_v1'
DB_SETTINGS = 'spring_scan_settings'
QUEUE_NAME = 'spring-scan'

PHASES = ('queued', 'tickers', 'bars', 'done')
DEFAULT_MIN_CHARGE = 80.0
DEFAULT_AUTO_MIN = 30
AUTO_MIN_RANGE = (5, 1440)
FAIL_RETRY_SEC = 300          # невдалий автоскан → повтор через 5 хв
RECALC_SETTLE_SEC = 8         # після закриття 1h-бару даємо біржі кілька секунд
TICK_SEC = 5
QUEUE_RECALC = 'spring-scan-recalc'
SBR_TFS = ('1h', '4h')
REMOVED_KEEP = 20
SENT_CAP = 400
TG_CATEGORY = 'sbr'


def _fs():
    try:
        from detection import funding_spring
        return funding_spring
    except Exception:
        import importlib.util
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'funding_spring.py')
        spec = importlib.util.spec_from_file_location('_ss_funding_spring', p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


def _excluded() -> set:
    try:
        from detection.tickr_core import WL_LIKE_EXCLUDE
        return set(WL_LIKE_EXCLUDE)
    except Exception:
        return {'USDC', 'FDUSD', 'TUSD', 'DAI', 'USDE', 'BUSD', 'USD1', 'WBTC',
                'WETH', 'PAXG', 'XAUT'}


def norm_exchange(x) -> str:
    x = str(x or '').strip().lower()
    return x if x in EXCHANGES else DEFAULT_EXCHANGE


def norm_min_vol(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return DEFAULT_MIN_VOL
    if v != v or v <= 0:
        return DEFAULT_MIN_VOL
    return max(MIN_VOL_FLOOR, v)


def norm_min_charge(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return DEFAULT_MIN_CHARGE
    if v != v:
        return DEFAULT_MIN_CHARGE
    return max(0.0, min(100.0, v))


def norm_auto_min(x) -> int:
    try:
        v = int(float(x))
    except (TypeError, ValueError):
        return DEFAULT_AUTO_MIN
    return max(AUTO_MIN_RANGE[0], min(AUTO_MIN_RANGE[1], v))


def norm_tfs(tfs) -> List[str]:
    """Кнопки 1H/4H рядка → впорядкований список (порожньо = вимкнено)."""
    out = [str(t).lower() for t in (tfs or []) if str(t).lower() in SBR_TFS]
    return [t for t in SBR_TFS if t in out]


def takes_in_work(rec: Optional[Dict], min_charge: float) -> bool:
    """ЧИСТА: монету ВЗЯТО в таблицю — 🌀 заряджена І заряд ≥ фільтра."""
    if not rec or rec.get('state') != 'charged':
        return False
    try:
        return float(rec.get('charge') or 0) >= float(min_charge)
    except (TypeError, ValueError):
        return False


def stays_in_table(rec: Optional[Dict]) -> bool:
    """ЧИСТА: монета ЛИШАЄТЬСЯ в таблиці, поки її статус «🌀 заряджена».
    Фільтр заряду діє лише на ВХІД (дослівна вимога: «якщо статус монети не
    "🌀 заряджена" … монету видаляємо»)."""
    return bool(rec) and rec.get('state') == 'charged'


def row_side(row: Optional[Dict]) -> Optional[str]:
    """Напрямок монети в таблиці: куди ймовірніше вистрілить (`dir`), інакше
    бік більшого натягу (`lean`). ⚖ — None."""
    if not row:
        return None
    d = row.get('dir') or row.get('lean')
    return d if d in ('LONG', 'SHORT') else None


def sbr_hits(row: Optional[Dict], tfs, items) -> List[tuple]:
    """ЧИСТА: сигнали SBR [(tf, sig)], що відповідають кнопкам рядка (1H/4H)
    і НАПРЯМКУ монети в таблиці."""
    side = row_side(row)
    tfs = norm_tfs(tfs)
    if not side or not tfs:
        return []
    return [(tf, sig) for tf, sig in (items or [])
            if tf in tfs and (sig or {}).get('side') == side]


def next_bar_due(now: float, bar_sec: int = 3600, settle: float = RECALC_SETTLE_SEC) -> int:
    """Час відкриття ОСТАННЬОГО закритого 1h-бару (із запасом settle)."""
    return int((now - settle) // bar_sec) * bar_sec - bar_sec


def tg_text(symbol: str, row: Dict, hits: List[tuple]) -> str:
    """Попередження в Telegram: монету з таблиці 🔭 взято в роботу."""
    side = hits[0][1]['side']
    emo = '🟢' if side == 'LONG' else '🔴'
    tfs = '+'.join(tf.upper() for tf, _ in hits)
    sig = hits[-1][1]
    try:
        from detection.sbr_signals import fmt_price
    except Exception:
        fmt_price = lambda v: f"{float(v):g}"
    ch = row.get('charge')
    return '\n'.join([
        f"🔭 <b>Пружина → в роботу</b> · {emo} <b>{side}</b>",
        f"<b>#{symbol}</b> · 🔔 SBR <code>{tfs}</code> · 🌀 заряд <b>{float(ch or 0):.0f}</b>",
        f"🎯 Trigger: <code>{fmt_price(sig.get('trigger'))}</code> · 🛑 SL: <code>{fmt_price(sig.get('sl'))}</code>",
    ])


def pick_symbols(tickers: Dict[str, Dict], min_vol: float, cap: int = MAX_SYMBOLS):
    """ЧИСТА: тікери біржі → (список монет за обігом ↓, розклад відсіву).

    Беремо лише `<BASE>USDT` (перпетуал; поставочні контракти з датою —
    відсіюються), без стейблів/обгорток, з ціною > 0 і обігом ≥ порогу.
    """
    excl = _excluded()
    total = 0
    not_usdt = stable = low = 0
    keep = []
    for sym, t in (tickers or {}).items():
        total += 1
        s = str(sym or '').upper()
        if not s.endswith('USDT') or not s[:-4].isalnum():
            not_usdt += 1
            continue
        base = s[:-4]
        if base in excl:
            stable += 1
            continue
        try:
            vol = float((t or {}).get('vol_usd') or 0)
            last = float((t or {}).get('last') or 0)
        except (TypeError, ValueError):
            vol, last = 0.0, 0.0
        if last <= 0 or vol < min_vol:
            low += 1
            continue
        keep.append((s, vol))
    keep.sort(key=lambda x: (-x[1], x[0]))
    capped = max(0, len(keep) - cap)
    keep = keep[:cap]
    return ([s for s, _ in keep], {s: v for s, v in keep},
            {'total': total, 'not_usdt': not_usdt, 'stable': stable,
             'low_vol': low, 'capped': capped, 'kept': len(keep)})


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _klines_bybit(sess, sym: str, limit: int = LIMIT) -> List[Dict]:
    r = sess.get('https://api.bybit.com/v5/market/kline',
                 params={'category': 'linear', 'symbol': sym, 'interval': '60',
                         'limit': limit}, timeout=12)
    r.raise_for_status()
    lst = ((r.json() or {}).get('result') or {}).get('list') or []
    out = [{'t': int(_f(x[0])), 'o': _f(x[1]), 'h': _f(x[2]), 'l': _f(x[3]),
            'p': _f(x[4]), 'v': _f(x[6]) if len(x) > 6 else _f(x[5])} for x in lst]
    out.reverse()
    return out


def _klines_binance(sess, sym: str, limit: int = LIMIT) -> List[Dict]:
    r = sess.get('https://fapi.binance.com/fapi/v1/klines',
                 params={'symbol': sym, 'interval': '1h', 'limit': limit}, timeout=12)
    r.raise_for_status()
    return [{'t': int(_f(x[0])), 'o': _f(x[1]), 'h': _f(x[2]), 'l': _f(x[3]),
             'p': _f(x[4]), 'v': _f(x[7])} for x in (r.json() or [])]


_KLINES = {'bybit': _klines_bybit, 'binance': _klines_binance}


def _default_tickers(exchange: str) -> Dict[str, Dict]:
    from detection import tickr_core
    return tickr_core._ACTIVITY[exchange](tickr_core.MARKET_SWAP)


def _default_fetch(exchange: str):
    import requests
    sess = requests.Session()
    fn = _KLINES[exchange]
    return lambda sym: fn(sess, sym)


def _default_run(name, fn, source):
    from detection import scan_queue
    return scan_queue.run(name, fn, source, timeout=900)


class SpringScan:
    def __init__(self, db=None, tickers: Callable = None, fetch_factory: Callable = None,
                 run_queued: Callable = None, workers: int = WORKERS,
                 notify: Callable = None, clock: Callable = None):
        self._db = db
        self._tickers = tickers or _default_tickers
        self._fetch_factory = fetch_factory or _default_fetch
        self._run_queued = run_queued or _default_run
        self._notify = notify
        self._clock = clock or time.time
        self._workers = workers
        self._lock = threading.Lock()
        self._settings = {'exchange': DEFAULT_EXCHANGE, 'min_vol_usd': DEFAULT_MIN_VOL,
                          'min_charge': DEFAULT_MIN_CHARGE, 'auto_on': False,
                          'auto_min': DEFAULT_AUTO_MIN, 'tg': {}}
        self._result: Dict = {}
        self._table: Dict[str, Dict] = {}      # S → рядок таблиці
        self._removed: List[Dict] = []          # хто вибув і чому
        self._recalc: Dict = {}                 # останній перерахунок
        self._sent: List[str] = []              # 'S|tf|close_t' уже надіслано
        self._events: List[Dict] = []
        self._prog: Dict = {'running': False}
        self._auto_fail_at = 0.0
        self._thread = None
        self._stop = threading.Event()
        self._load()

    # ── налаштування / персист ───────────────────────────────────────────
    def _load(self):
        if not self._db:
            return
        try:
            s = self._db.get_setting(DB_SETTINGS, None)
            if isinstance(s, dict):
                self._apply_settings(s)
        except Exception:
            pass
        try:
            r = self._db.get_setting(DB_KEY, None)
            if isinstance(r, dict) and r.get('ts'):
                rows = r.get('rows') or []
                self._table = {str(x.get('symbol')).upper(): dict(x) for x in rows
                               if isinstance(x, dict) and x.get('symbol')}
                self._removed = list(r.get('removed') or [])[-REMOVED_KEEP:]
                self._recalc = dict(r.get('recalc') or {})
                self._sent = list(r.get('sent') or [])[-SENT_CAP:]
                r['restored'] = True
                self._result = r
        except Exception as e:
            print(f"[SpringScan] restore error: {e}")

    def _apply_settings(self, s: Dict):
        if 'exchange' in s:
            self._settings['exchange'] = norm_exchange(s.get('exchange'))
        if 'min_vol_usd' in s:
            self._settings['min_vol_usd'] = norm_min_vol(s.get('min_vol_usd'))
        if 'min_charge' in s:
            self._settings['min_charge'] = norm_min_charge(s.get('min_charge'))
        if 'auto_on' in s:
            self._settings['auto_on'] = bool(s.get('auto_on'))
        if 'auto_min' in s:
            self._settings['auto_min'] = norm_auto_min(s.get('auto_min'))
        if isinstance(s.get('tg'), dict):
            self._settings['tg'] = {str(k).upper(): norm_tfs(v) for k, v in s['tg'].items()
                                    if norm_tfs(v)}

    def settings(self) -> Dict:
        out = dict(self._settings)
        out['tg'] = {k: list(v) for k, v in self._settings['tg'].items()}
        return out

    def _save_settings(self):
        if self._db:
            try:
                self._db.set_setting(DB_SETTINGS, self.settings())
            except Exception:
                pass

    def update_settings(self, patch: Dict) -> Dict:
        self._apply_settings({k: v for k, v in (patch or {}).items() if k != 'tg'})
        self._save_settings()
        return self.settings()

    def set_tg(self, symbol: str, tfs) -> Dict:
        """Кнопки 1H/4H рядка. Лише монети таблиці."""
        S = str(symbol or '').upper()
        if S not in self._table:
            return {'ok': False, 'reason': f'{S} немає в таблиці'}
        t = norm_tfs(tfs)
        if t:
            self._settings['tg'][S] = t
        else:
            self._settings['tg'].pop(S, None)
        self._save_settings()
        return {'ok': True, 'symbol': S, 'tfs': t}

    def remove(self, symbol: str, now: Optional[float] = None) -> Dict:
        """✕ Прибрати монету з таблиці вручну (разом із кнопками 1H/4H).
        Наступний повний скан знову додасть її, якщо вона ще заряджена ≥ фільтра."""
        S = str(symbol or '').upper()
        if S not in self._table:
            return {'ok': False, 'reason': f'{S} немає в таблиці'}
        now = now if now is not None else self._clock()
        self._drop(S, self._table.get(S), '✕ прибрано вручну', now)
        self._save_settings()
        self._persist()
        return {'ok': True, 'symbol': S, 'left': len(self._table)}

    def clear(self, now: Optional[float] = None) -> Dict:
        """🗑 Очистити всю таблицю (і кнопки 1H/4H усіх монет)."""
        n = len(self._table)
        now = now if now is not None else self._clock()
        self._table = {}
        self._settings['tg'] = {}
        if n:
            self._removed.append({'symbol': f'усі ({n})', 'at': now,
                                  'why': '🗑 таблицю очищено вручну', 'state': None, 'charge': None})
            self._removed = self._removed[-REMOVED_KEEP:]
        self._save_settings()
        self._persist()
        return {'ok': True, 'cleared': n}

    def tg_symbols(self) -> List[str]:
        """Монети з увімкненими 1H/4H — рушій SBR рахує їх і поза WATCHLIST."""
        return [s for s in self._settings['tg'] if s in self._table]

    def _persist(self):
        if not self._db:
            return
        try:
            with self._lock:
                res = dict(self._result)
                res['rows'] = [dict(r) for r in self._table.values()]
                res['removed'] = list(self._removed)
                res['recalc'] = dict(self._recalc)
                res['sent'] = list(self._sent)
                res.pop('restored', None)
            self._db.set_setting(DB_KEY, res)
        except Exception as e:
            print(f"[SpringScan] persist error: {e}")

    # ── прогрес (візуалізація) ───────────────────────────────────────────
    def _p(self, **kw):
        with self._lock:
            self._prog.update(kw)

    def _p_coin(self, sym: str, rec: Optional[Dict], err: str = ''):
        with self._lock:
            p = self._prog
            p['done'] = int(p.get('done') or 0) + 1
            p['cur'] = sym
            if err:
                p['errors'] = int(p.get('errors') or 0) + 1
                st, ch = 'err', None
            elif not rec:
                p['short'] = int(p.get('short') or 0) + 1
                st, ch = 'short', None
            else:
                st, ch = rec.get('state') or 'flat', rec.get('charge')
                if takes_in_work(rec, self._settings['min_charge']):
                    p['found'] = int(p.get('found') or 0) + 1
                    p.setdefault('found_syms', []).append(sym)
                bs = p.setdefault('by_state', {})
                bs[st] = int(bs.get(st) or 0) + 1
            rec_l = p.setdefault('recent', [])
            rec_l.insert(0, {'symbol': sym, 'state': st,
                             'charge': round(float(ch), 1) if ch is not None else None})
            del rec_l[RECENT_N:]

    def progress(self) -> Dict:
        with self._lock:
            p = dict(self._prog)
            for k in ('recent', 'found_syms', 'gone_syms'):
                if k in p:
                    p[k] = list(p[k])
            if 'by_state' in p:
                p['by_state'] = dict(p['by_state'])
        return p

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._prog.get('running'))

    # ── запуск ───────────────────────────────────────────────────────────
    def _begin(self, kind: str) -> bool:
        with self._lock:
            if self._prog.get('running'):
                return False
            self._prog = {'running': True, 'kind': kind, 'phase': 'queued',
                          'exchange': self._settings['exchange'],
                          'min_vol_usd': self._settings['min_vol_usd'],
                          'min_charge': self._settings['min_charge'],
                          'started_at': self._clock(), 'total': 0, 'done': 0,
                          'found': 0, 'errors': 0, 'short': 0, 'by_state': {},
                          'recent': [], 'found_syms': [], 'gone_syms': [], 'cur': ''}
            return True

    def start(self, exchange=None, min_vol_usd=None, min_charge=None, source='ui') -> Dict:
        """Скан біржі у фоні. Повторний клік під час роботи нічого не дублює."""
        patch = {k: v for k, v in (('exchange', exchange), ('min_vol_usd', min_vol_usd),
                                   ('min_charge', min_charge)) if v is not None}
        if patch:
            self.update_settings(patch)
        if not self._begin('scan'):
            return {'ok': True, 'started': False, 'reason': 'скан уже йде'}
        ex, mv = self._settings['exchange'], self._settings['min_vol_usd']
        threading.Thread(target=self._bg, args=(QUEUE_NAME, lambda: self.scan(ex, mv), source),
                         daemon=True, name='spring-scan').start()
        return {'ok': True, 'started': True}

    def start_recalc(self, source='bar') -> Dict:
        if not self._table:
            return {'ok': True, 'started': False, 'reason': 'таблиця порожня'}
        if not self._begin('recalc'):
            return {'ok': True, 'started': False, 'reason': 'скан уже йде'}
        threading.Thread(target=self._bg, args=(QUEUE_RECALC, self.recalc, source),
                         daemon=True, name='spring-recalc').start()
        return {'ok': True, 'started': True}

    def _bg(self, name: str, fn: Callable, source: str):
        try:
            res = self._run_queued(name, fn, source)
            if isinstance(res, dict) and res.get('error') and 'rows' not in res and 'checked' not in res:
                raise RuntimeError(res.get('error'))
        except Exception as e:
            self._auto_fail_at = self._clock()
            self._p(running=False, phase='error', error=str(e)[:200],
                    finished_at=self._clock())

    def _analyze_many(self, syms: List[str], fetch, now: float):
        fs = _fs()
        sc = fs._sc()
        recs: Dict[str, Dict] = {}
        errors: List[str] = []

        def one(sym):
            bars = fs.closed_only(fetch(sym), now)
            if len(bars) < getattr(sc, 'MIN_BARS', 220):
                return sym, None
            return sym, sc.analyze(bars)

        with ThreadPoolExecutor(max_workers=max(1, self._workers)) as pool:
            futs = {pool.submit(one, s): s for s in syms}
            for fu in as_completed(futs):
                s = futs[fu]
                try:
                    _, rec = fu.result()
                except Exception as e:
                    errors.append(f"{s}: {str(e)[:80]}")
                    self._p_coin(s, None, err=str(e))
                    continue
                if rec:
                    recs[s] = rec
                self._p_coin(s, rec)
        return recs, errors

    def _row(self, S: str, rec: Dict, now: float, vol=None) -> Dict:
        fs = _fs()
        row = fs.shape_row(rec)
        row['symbol'] = S
        old = self._table.get(S) or {}
        row['added_at'] = old.get('added_at') or now
        row['charge_in'] = old.get('charge_in', rec.get('charge'))
        row['vol_usd'] = vol if vol is not None else old.get('vol_usd')
        row['calc_at'] = now
        return row

    def _drop(self, S: str, rec: Optional[Dict], why: str, now: float):
        """Монета вибуває з таблиці — і її кнопки 1H/4H разом із нею."""
        self._table.pop(S, None)
        self._settings['tg'].pop(S, None)
        self._removed.append({'symbol': S, 'at': now, 'why': why,
                              'state': (rec or {}).get('state'),
                              'charge': (rec or {}).get('charge')})
        self._removed = self._removed[-REMOVED_KEEP:]
        with self._lock:
            self._prog.setdefault('gone_syms', []).append(S)

    def scan(self, exchange: str, min_vol: float, now: Optional[float] = None) -> Dict:
        """Повний скан. Нові заряджені (≥ фільтра) ДОПОВНЮЮТЬ таблицю; монети
        таблиці, які скан щойно перерахував і вони вже не заряджені, — вибувають."""
        t0 = self._clock()
        self._p(phase='tickers', scan_started=t0)
        tickers = self._tickers(exchange) or {}
        syms, vols, brk = pick_symbols(tickers, min_vol)
        self._p(phase='bars', total=len(syms), breakdown=brk)
        now = now if now is not None else self._clock()
        recs, errors = self._analyze_many(syms, self._fetch_factory(exchange), now)
        mc = self._settings['min_charge']
        added = 0
        for S, rec in recs.items():
            if S in self._table:
                if stays_in_table(rec):
                    self._table[S] = self._row(S, rec, now, vols.get(S))
                else:
                    self._drop(S, rec, 'скан: статус уже не «🌀 заряджена»', now)
            elif takes_in_work(rec, mc):
                self._table[S] = self._row(S, rec, now, vols.get(S))
                added += 1
        by_state: Dict[str, int] = {}
        for rec in recs.values():
            k = rec.get('state') or 'flat'
            by_state[k] = by_state.get(k, 0) + 1
        try:
            agg = _fs()._sc().aggregate(recs, total=len(recs)) if recs else None
        except Exception:
            agg = None
        took = round(self._clock() - t0, 1)
        res = {'ts': self._clock(), 'exchange': exchange, 'min_vol_usd': min_vol,
               'min_charge': mc, 'agg': agg, 'breakdown': brk, 'scanned': len(syms),
               'analyzed': len(recs), 'charged': by_state.get('charged', 0),
               'added': added, 'by_state': by_state, 'errors': len(errors),
               'error_samples': errors[:5], 'took': took}
        with self._lock:
            self._result = res
            self._prog.update({'running': False, 'phase': 'done', 'finished_at': self._clock(),
                               'took': took, 'cur': '', 'added': added})
        self._auto_fail_at = 0.0
        self._save_settings()
        self._persist()
        print(f"[SpringScan] {exchange} ≥{min_vol/1e6:.0f}M: {len(syms)} монет за {took}с, "
              f"нових у таблиці {added}, у таблиці {len(self._table)}, збоїв {len(errors)}")
        return dict(res, rows=self.rows())

    def recalc(self, now: Optional[float] = None) -> Dict:
        """Перерахунок заряду ЛИШЕ монет таблиці. Статус не «🌀 заряджена» →
        монета вибуває. Збій завантаження монету НЕ видаляє."""
        t0 = self._clock()
        syms = list(self._table)
        self._p(phase='bars', total=len(syms))
        now = now if now is not None else self._clock()
        recs, errors = self._analyze_many(syms, self._fetch_factory(self._settings['exchange']), now)
        gone = []
        for S in syms:
            rec = recs.get(S)
            if rec is None:
                continue
            if S not in self._table:        # прибрали вручну під час перерахунку
                continue
            if stays_in_table(rec):
                self._table[S] = self._row(S, rec, now)
            else:
                self._drop(S, rec, 'перерахунок: статус уже не «🌀 заряджена»', now)
                gone.append(S)
        took = round(self._clock() - t0, 1)
        self._recalc = {'at': self._clock(), 'bar': next_bar_due(now), 'checked': len(recs),
                        'removed': gone, 'errors': len(errors), 'took': took}
        with self._lock:
            self._prog.update({'running': False, 'phase': 'done', 'finished_at': self._clock(),
                               'took': took, 'cur': ''})
        self._save_settings()
        self._persist()
        if gone:
            print(f"[SpringScan] перерахунок: вибули {', '.join(gone)}")
        return dict(self._recalc)

    # ── 🔔 сигнали SBR → Telegram ───────────────────────────────────────
    def on_sbr(self, symbol: str, items: List[tuple]) -> int:
        """Свіжі сигнали рушія SBR. Монета в таблиці + кнопка TF + сигнал у бік
        монети → одне попередження в Telegram (кожен сигнал — один раз)."""
        S = str(symbol or '').upper()
        row = self._table.get(S)
        hits = sbr_hits(row, self._settings['tg'].get(S), items)
        keys = [f"{S}|{tf}|{int(float(sig.get('close_t') or sig.get('t') or 0))}" for tf, sig in hits]
        fresh = [h for h, k in zip(hits, keys) if k not in self._sent]
        if not fresh:
            return 0
        ok, why = False, 'Telegram не налаштовано'
        if self._notify:
            try:
                ok, why = self._notify(tg_text(S, row, fresh))
            except Exception as e:
                ok, why = False, str(e)
        for h, k in zip(hits, keys):
            if h in fresh:
                self._sent.append(k)
        self._sent = self._sent[-SENT_CAP:]
        side = fresh[0][1]['side']
        tfs = '+'.join(tf.upper() for tf, _ in fresh)
        ev = {'symbol': S, 'side': side, 'tfs': tfs, 'at': self._clock(), 'tg': bool(ok),
              'err': '' if ok else str(why)[:160], 'charge': row.get('charge')}
        with self._lock:
            self._events.append(ev)
            self._events = self._events[-30:]
        try:
            from detection.activity_log import log_activity
            txt = (f"🔭 Пружина → в роботу: 🔔 SBR {tfs} {side} · 🌀 заряд "
                   f"{float(row.get('charge') or 0):.0f} · "
                   + ('📨 надіслано в «🔔 Сигнали»' if ok else f'📨 Telegram НЕ прийняв: {why}'))
            log_activity(S, 'signal', txt, side=side, source='XSPR', extra={'tfs': tfs})
        except Exception:
            pass
        self._persist()
        return 1

    # ── потік: автоскан + перерахунок на закритті 1h-бару ───────────────
    def tick(self, now: Optional[float] = None) -> Optional[str]:
        now = now if now is not None else self._clock()
        if self.is_running():
            return None
        bar = next_bar_due(now)
        if self._table and int(self._recalc.get('bar') or 0) < bar:
            if self.start_recalc('bar').get('started'):
                return 'recalc'
        s = self._settings
        if s['auto_on']:
            last = float(self._result.get('ts') or 0)
            if self._auto_fail_at and now - self._auto_fail_at < FAIL_RETRY_SEC:
                return None
            if now - last >= s['auto_min'] * 60:
                if self.start(source='auto').get('started'):
                    return 'scan'
        return None

    def next_auto_at(self) -> Optional[float]:
        if not self._settings['auto_on']:
            return None
        last = float(self._result.get('ts') or 0)
        nxt = last + self._settings['auto_min'] * 60
        if self._auto_fail_at:
            nxt = max(nxt, self._auto_fail_at + FAIL_RETRY_SEC)
        return nxt

    def _loop(self):
        while not self._stop.wait(TICK_SEC):
            try:
                self.tick()
            except Exception as e:
                print(f"[SpringScan] tick error: {e}")

    def run_forever(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name='spring-scan-loop')
        self._thread.start()

    def rows(self) -> List[Dict]:
        with self._lock:
            rows = [dict(r) for r in self._table.values()]
        tg = self._settings['tg']
        for r in rows:
            r['tg'] = list(tg.get(r['symbol']) or [])
        rows.sort(key=lambda r: -(float(r.get('charge') or 0)))
        return rows

    def state(self) -> Dict:
        with self._lock:
            res = dict(self._result)
            removed = list(self._removed)
            recalc = dict(self._recalc)
            events = list(self._events)
        res['rows'] = self.rows()
        now = self._clock()
        return {'ok': True, 'settings': self.settings(), 'result': res,
                'progress': self.progress(), 'server_now': now,
                'exchanges': list(EXCHANGES), 'removed': removed[::-1],
                'recalc': recalc,
                'recalc_next': (int(now // 3600) + 1) * 3600 + RECALC_SETTLE_SEC,
                'auto_next': self.next_auto_at(), 'events': events[::-1],
                'running_loop': bool(self._thread and self._thread.is_alive())}


_instance: Optional[SpringScan] = None
_ilock = threading.Lock()


def _default_notify(text: str):
    from web.tg_bot import notify_category, last_send_error
    ok = bool(notify_category(TG_CATEGORY, text))
    return ok, ('' if ok else (last_send_error() or 'не надіслано'))


def init_spring_scan(db=None, start_loop: bool = False, **kw) -> SpringScan:
    global _instance
    with _ilock:
        if _instance is None:
            kw.setdefault('notify', _default_notify)
            _instance = SpringScan(db=db, **kw)
    if start_loop:
        _instance.run_forever()
    return _instance


def get_spring_scan() -> Optional[SpringScan]:
    return _instance


def sbr_symbols() -> List[str]:
    """Для рушія SBR: монети таблиці з увімкненими 1H/4H."""
    return _instance.tg_symbols() if _instance else []


def on_sbr_signals(symbol: str, items: List[tuple]) -> int:
    return _instance.on_sbr(symbol, items) if _instance else 0
