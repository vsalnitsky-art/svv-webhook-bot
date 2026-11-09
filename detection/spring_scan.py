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
- ЛИШЕ ПОКАЗ: нічого не відкриває і не шле в Telegram.
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
                 run_queued: Callable = None, workers: int = WORKERS):
        self._db = db
        self._tickers = tickers or _default_tickers
        self._fetch_factory = fetch_factory or _default_fetch
        self._run_queued = run_queued or _default_run
        self._workers = workers
        self._lock = threading.Lock()
        self._settings = {'exchange': DEFAULT_EXCHANGE, 'min_vol_usd': DEFAULT_MIN_VOL}
        self._result: Dict = {}
        self._prog: Dict = {'running': False}
        self._load()

    # ── налаштування / персист ───────────────────────────────────────────
    def _load(self):
        if not self._db:
            return
        try:
            s = self._db.get_setting(DB_SETTINGS, None)
            if isinstance(s, dict):
                self._settings['exchange'] = norm_exchange(s.get('exchange'))
                self._settings['min_vol_usd'] = norm_min_vol(s.get('min_vol_usd'))
        except Exception:
            pass
        try:
            r = self._db.get_setting(DB_KEY, None)
            if isinstance(r, dict) and r.get('ts'):
                r['restored'] = True
                self._result = r
        except Exception as e:
            print(f"[SpringScan] restore error: {e}")

    def settings(self) -> Dict:
        return dict(self._settings)

    def update_settings(self, patch: Dict) -> Dict:
        if 'exchange' in (patch or {}):
            self._settings['exchange'] = norm_exchange(patch.get('exchange'))
        if 'min_vol_usd' in (patch or {}):
            self._settings['min_vol_usd'] = norm_min_vol(patch.get('min_vol_usd'))
        if self._db:
            try:
                self._db.set_setting(DB_SETTINGS, dict(self._settings))
            except Exception:
                pass
        return self.settings()

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
                if st == 'charged':
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
            for k in ('recent', 'found_syms'):
                if k in p:
                    p[k] = list(p[k])
            if 'by_state' in p:
                p['by_state'] = dict(p['by_state'])
        return p

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._prog.get('running'))

    # ── скан ─────────────────────────────────────────────────────────────
    def start(self, exchange=None, min_vol_usd=None) -> Dict:
        """Ручний запуск у фоні. Повторний клік під час скану нічого не дублює."""
        patch = {}
        if exchange is not None:
            patch['exchange'] = exchange
        if min_vol_usd is not None:
            patch['min_vol_usd'] = min_vol_usd
        if patch:
            self.update_settings(patch)
        with self._lock:
            if self._prog.get('running'):
                return {'ok': True, 'started': False, 'reason': 'скан уже йде'}
            ex, mv = self._settings['exchange'], self._settings['min_vol_usd']
            self._prog = {'running': True, 'phase': 'queued', 'exchange': ex,
                          'min_vol_usd': mv, 'started_at': time.time(), 'total': 0,
                          'done': 0, 'found': 0, 'errors': 0, 'short': 0,
                          'by_state': {}, 'recent': [], 'found_syms': [], 'cur': ''}
        t = threading.Thread(target=self._bg, args=(ex, mv), daemon=True, name='spring-scan')
        t.start()
        return {'ok': True, 'started': True}

    def _bg(self, ex: str, mv: float):
        try:
            res = self._run_queued(QUEUE_NAME, lambda: self.scan(ex, mv), 'ui')
            if isinstance(res, dict) and 'rows' not in res and res.get('error'):
                raise RuntimeError(res.get('error'))
        except Exception as e:
            self._p(running=False, phase='error', error=str(e)[:200],
                    finished_at=time.time())

    def scan(self, exchange: str, min_vol: float, now: Optional[float] = None) -> Dict:
        fs = _fs()
        sc = fs._sc()
        t0 = time.time()
        self._p(phase='tickers', scan_started=t0)
        tickers = self._tickers(exchange) or {}
        syms, vols, brk = pick_symbols(tickers, min_vol)
        self._p(phase='bars', total=len(syms), breakdown=brk)
        fetch = self._fetch_factory(exchange)
        now = now if now is not None else time.time()
        recs: Dict[str, Dict] = {}

        def one(sym):
            bars = fs.closed_only(fetch(sym), now)
            if len(bars) < getattr(sc, 'MIN_BARS', 220):
                return sym, None
            return sym, sc.analyze(bars)

        errors: List[str] = []
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
        rows = []
        for s, rec in recs.items():
            if rec.get('state') != 'charged':
                continue
            row = fs.shape_row(rec)
            row['symbol'] = s
            row['vol_usd'] = vols.get(s)
            rows.append(row)
        rows.sort(key=lambda r: -(float(r.get('charge') or 0)))
        by_state: Dict[str, int] = {}
        for rec in recs.values():
            k = rec.get('state') or 'flat'
            by_state[k] = by_state.get(k, 0) + 1
        try:
            agg = sc.aggregate(recs, total=len(recs)) if recs else None
        except Exception:
            agg = None
        took = round(time.time() - t0, 1)
        res = {'ts': time.time(), 'exchange': exchange, 'min_vol_usd': min_vol,
               'rows': rows, 'agg': agg, 'breakdown': brk, 'scanned': len(syms),
               'analyzed': len(recs), 'charged': len(rows), 'by_state': by_state,
               'errors': len(errors), 'error_samples': errors[:5], 'took': took}
        with self._lock:
            self._result = res
            self._prog.update({'running': False, 'phase': 'done', 'finished_at': time.time(),
                               'took': took, 'cur': ''})
        if self._db:
            try:
                self._db.set_setting(DB_KEY, res)
            except Exception as e:
                print(f"[SpringScan] persist error: {e}")
        print(f"[SpringScan] {exchange} ≥{min_vol/1e6:.0f}M: {len(syms)} монет за {took}с, "
              f"заряджені {len(rows)}, збоїв {len(errors)}")
        return res

    def state(self) -> Dict:
        with self._lock:
            res = dict(self._result)
            res['rows'] = [dict(r) for r in (res.get('rows') or [])]
        return {'ok': True, 'settings': self.settings(), 'result': res,
                'progress': self.progress(), 'server_now': time.time(),
                'exchanges': list(EXCHANGES)}


_instance: Optional[SpringScan] = None
_ilock = threading.Lock()


def init_spring_scan(db=None, **kw) -> SpringScan:
    global _instance
    with _ilock:
        if _instance is None:
            _instance = SpringScan(db=db, **kw)
    return _instance


def get_spring_scan() -> Optional[SpringScan]:
    return _instance
