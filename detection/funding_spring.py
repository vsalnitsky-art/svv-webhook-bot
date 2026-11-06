"""💰🌀 «Funding — Пружина»: алгоритм «Пружина» для КОЖНОЇ монети 💰 Funding
Rate Scanner (вимога 06.11, ЗАМІЩУЄ стару таблицю «💰 Funding — МММ по
монетах» з її шарами, скальпом, VOB→Черга-4, авто-відкриттям і Telegram).

Принципи:
- Розрахунок — ТОЙ САМИЙ `spring_coil.analyze`, що й панель 🌀 Пружина;
  свого правила тут немає (урок PD-зони: одна метрика — одне джерело).
- Монета з WATCHLIST → береться ГОТОВИЙ запис сканера (`spring_for`): нуль
  запитів, і число в обох таблицях одне й те саме.
- Решта монет → власне завантаження ЗАКРИТИХ 1h-барів (живий бар
  відкидається), ОДИН запит на монету на закритий бар (~1/год), паралельно.
- Біржа ще не закрила бар → повтор через `RETRY_SEC`; збій → backoff.
- Записи персистяться (`DB_KEY`): після рестарту таблиця не порожня, а
  запис, чий бар уже закрився, позначається ↻ до перерахунку.
- ЛИШЕ ПОКАЗ: нічого не відкриває і не шле в Telegram.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

TF = '1h'
BAR_SEC = 3600
LIMIT = 300               # ≥ spring_coil.MIN_BARS (220) з запасом; < 1 сторінки біржі
SETTLE_SEC = 5            # після закриття бару даємо біржі кілька секунд
RETRY_SEC = 10            # біржа ще не віддала закритий бар → повтор
TICK_SEC = 5
WORKERS = 4
MAX_BACKOFF = 600
RESTORE_MAX_AGE = 6 * 3600
DB_KEY = 'funding_spring_v1'
DB_SETTINGS = 'funding_spring_settings'
ROW_KEYS = ('state', 'label', 'charge', 'bias', 'dir', 'pot_long', 'pot_short',
            'squeeze_bars', 'band', 'box_pct', 'vol_ratio', 'atr_rank',
            'fired', 'price', 'bar_t', 'parts',
            'lean', 'released', 'lean_pp', 'prob_long', 'tgt_long', 'tgt_short',
            'move_long_pct', 'move_short_pct', 'tension_long', 'tension_short',
            'phase', 'atr_peak_ratio', 'move24_pct', 'unit_r')


def _sc():
    try:
        from detection import spring_coil
        return spring_coil
    except Exception:
        import importlib.util
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spring_coil.py')
        spec = importlib.util.spec_from_file_location('_fs_spring_coil', p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m


def bar_t(b) -> int:
    """Час відкриття бару в СЕКУНДАХ (біржі віддають мс)."""
    try:
        t = int(float((b or {}).get('t') or 0))
    except (TypeError, ValueError):
        return 0
    return t // 1000 if t > 10 ** 12 else t


def expected_last(now: float, settle: float = SETTLE_SEC) -> int:
    """Час ВІДКРИТТЯ останнього бару, який уже мав закритись (із запасом settle)."""
    return int((now - settle) // BAR_SEC) * BAR_SEC - BAR_SEC


def closed_only(klines, now: float) -> List[Dict]:
    """Лише ЗАКРИТІ бари: живий (той, що ще не закрився) відкидається."""
    return [b for b in (klines or []) if bar_t(b) and bar_t(b) + BAR_SEC <= now]


def shape_row(rec: Dict) -> Dict:
    """Запис «Пружини» → рядок таблиці (ті самі правила, що /api/spring/state)."""
    r = dict(rec or {})
    if 'tension_long' not in r and r.get('box_hi') is not None:
        try:
            r.update(_sc().projection(r.get('price'), r.get('box_hi'), r.get('box_lo'),
                                      r.get('atr') or 0.0, r.get('pot_long'),
                                      r.get('pot_short'), r.get('fired'), r.get('charge')))
        except Exception:
            pass
    if r.get('state') == 'flat':
        r['dir'] = None
    return {k: r.get(k) for k in ROW_KEYS}


class FundingSpring:
    def __init__(self, db=None, get_symbols: Callable = None, fetch: Callable = None,
                 scanner_recs: Callable = None, funding_info: Callable = None):
        self._db = db
        self._get_symbols = get_symbols or (lambda: [])
        self._fetch = fetch
        self._scanner_recs = scanner_recs or (lambda syms: {})
        self._funding_info = funding_info or (lambda: {})
        self._lock = threading.Lock()
        self._recs: Dict[str, Dict] = {}
        self._fail: Dict[str, tuple] = {}      # S → (кількість, наступна спроба)
        self._restored: set = set()
        self._stats = {'tick_at': 0.0, 'calc_at': 0.0, 'calc_n': 0, 'calc_bar': 0,
                       'fetched': 0, 'own': 0, 'watchlist': 0, 'errors': 0, 'last_err': ''}
        self._dirty = False
        self._thread = None
        self._stop = threading.Event()
        self._settings = {'enabled': True}
        self._load()

    # ── налаштування / персист ───────────────────────────────────────────
    def _load(self):
        if not self._db:
            return
        try:
            s = self._db.get_setting(DB_SETTINGS, None)
            if isinstance(s, dict):
                self._settings.update({k: s[k] for k in ('enabled',) if k in s})
        except Exception:
            pass
        try:
            st = self._db.get_setting(DB_KEY, None) or {}
            now = time.time()
            for sym, rec in (st.get('recs') or {}).items():
                if not isinstance(rec, dict) or now - float(rec.get('ts') or 0) > RESTORE_MAX_AGE:
                    continue
                self._recs[str(sym).upper()] = rec
                if int(rec.get('bar_t') or 0) < expected_last(now):
                    self._restored.add(str(sym).upper())
        except Exception as e:
            print(f"[FundSpring] restore error: {e}")

    def _persist(self):
        if not self._db or not self._dirty:
            return
        try:
            with self._lock:
                recs = {k: v for k, v in self._recs.items() if v.get('src') == 'own'}
            self._db.set_setting(DB_KEY, {'recs': recs, 'ts': time.time()})
            self._dirty = False
        except Exception as e:
            print(f"[FundSpring] persist error: {e}")

    def is_enabled(self) -> bool:
        return bool(self._settings.get('enabled', True))

    def update_settings(self, patch: Dict) -> Dict:
        if 'enabled' in (patch or {}):
            self._settings['enabled'] = bool(patch['enabled'])
            if not self._settings['enabled']:
                with self._lock:
                    self._recs.clear()
                    self._restored.clear()
                    self._fail.clear()
                self._dirty = True
                self._persist()
            elif not (self._thread and self._thread.is_alive()):
                self.start()
        if self._db:
            try:
                self._db.set_setting(DB_SETTINGS, dict(self._settings))
            except Exception:
                pass
        return dict(self._settings)

    # ── робочий такт ─────────────────────────────────────────────────────
    def _symbols(self) -> List[str]:
        try:
            return list(dict.fromkeys(str(s).upper() for s in (self._get_symbols() or []) if s))
        except Exception:
            return []

    def _calc_one(self, sym: str, exp: int, now: float):
        """→ ('ok', rec) | ('wait', None) | ('err', причина)."""
        try:
            kl = self._fetch(sym, TF, LIMIT)
        except Exception as e:
            return 'err', f'{type(e).__name__}: {e}'
        if not kl:
            return 'err', 'біржа не віддала бари'
        closed = closed_only(kl, now)
        if not closed or bar_t(closed[-1]) < exp:
            return 'wait', None
        rec = _sc().analyze(closed)
        rec['bar_t'] = bar_t(closed[-1])
        return 'ok', rec

    def tick(self, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self._stats['tick_at'] = now
        if not self.is_enabled():
            return
        syms = self._symbols()
        exp = expected_last(now)
        try:
            wl = self._scanner_recs(syms) or {}
        except Exception:
            wl = {}
        need = []
        with self._lock:
            for s in list(self._recs):
                if s not in syms:
                    self._recs.pop(s, None)
                    self._restored.discard(s)
                    self._fail.pop(s, None)
            for s in syms:
                w = wl.get(s)
                if w:
                    # WATCHLIST: готовий запис сканера — без запитів до біржі.
                    self._recs[s] = {**w, 'src': 'watchlist'}
                    self._restored.discard(s)
                    continue
                cur = self._recs.get(s)
                if cur and cur.get('src') == 'own' and int(cur.get('bar_t') or 0) >= exp:
                    continue
                f = self._fail.get(s)
                if f and f[1] > now:
                    continue
                need.append(s)
        n_ok = 0
        if need and self._fetch:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                res = list(ex.map(lambda s: (s, self._calc_one(s, exp, now)), need))
            with self._lock:
                for s, (kind, val) in res:
                    if kind == 'ok':
                        val['ts'], val['src'] = now, 'own'
                        self._recs[s] = val
                        self._restored.discard(s)
                        self._fail.pop(s, None)
                        n_ok += 1
                        self._dirty = True
                    elif kind == 'wait':
                        self._fail[s] = (0, now + RETRY_SEC)
                    else:
                        n = (self._fail.get(s) or (0, 0))[0] + 1
                        self._fail[s] = (n, now + min(MAX_BACKOFF, RETRY_SEC * (2 ** n)))
                        self._stats['errors'] += 1
                        self._stats['last_err'] = f'{s}: {val}'
            self._stats['fetched'] += len(need)
        if n_ok:
            if self._stats.get('calc_bar') != exp:
                self._stats['calc_bar'], self._stats['calc_n'] = exp, 0
            self._stats['calc_n'] += n_ok
            self._stats['calc_at'] = now
        with self._lock:
            self._stats['own'] = sum(1 for r in self._recs.values() if r.get('src') == 'own')
            self._stats['watchlist'] = sum(1 for r in self._recs.values() if r.get('src') == 'watchlist')
        self._persist()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as e:
                print(f"[FundSpring] tick error: {e}")
            self._stop.wait(TICK_SEC)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='FundingSpring', daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    # ── читання для сторінки ─────────────────────────────────────────────
    def state(self, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else now
        syms = self._symbols()
        try:
            info = self._funding_info() or {}
        except Exception:
            info = {}
        sc = _sc()
        rows, full, pending = [], {}, []
        with self._lock:
            recs = {s: dict(self._recs[s]) for s in syms if s in self._recs}
            restored = set(self._restored)
            fails = dict(self._fail)
        for s in syms:
            rec = recs.get(s)
            fi = info.get(s) or {}
            fund = {'rate': fi.get('rate'), 'next_ms': fi.get('next_ms'),
                    'vol24h': fi.get('vol24h'), 'f_trend': fi.get('f_trend')}
            if not rec:
                pending.append(s)
                f = fails.get(s)
                rows.append({'symbol': s, 'pending': True, 'funding': fund,
                             'reason': ('повтор після збою' if f and f[0] else 'рахується')})
                continue
            if not rec.get('ok'):
                rows.append({'symbol': s, 'pending': True, 'funding': fund,
                             'reason': rec.get('reason') or 'немає даних'})
                continue
            full[s] = rec
            row = {'symbol': s, **shape_row(rec), 'funding': fund,
                   'src': rec.get('src'), 'restored': s in restored}
            rows.append(row)
        rows.sort(key=lambda x: (bool(x.get('pending')),
                                 -max(x.get('pot_long') or 0, x.get('pot_short') or 0)))
        try:
            agg = sc.aggregate({k: {**v, **shape_row(v)} for k, v in full.items()},
                               total=len(full))
        except Exception:
            agg = None
        return {'ok': True, 'enabled': self.is_enabled(), 'rows': rows, 'agg': agg,
                'coins': len(syms), 'computed': len(full), 'pending': len(pending),
                'stats': dict(self._stats), 'tf': TF, 'server_now': now,
                'next_at': (int(now // BAR_SEC) + 1) * BAR_SEC,
                'running': bool(self._thread and self._thread.is_alive())}


# ── синглтон ─────────────────────────────────────────────────────────────
_instance: Optional[FundingSpring] = None


def _default_symbols() -> List[str]:
    try:
        from detection.funding_monitor import get_funding_monitor
        fm = get_funding_monitor()
        if not fm or not fm.is_enabled() or not getattr(fm, '_running', False):
            return []
        return [s.upper() for s in (fm.get_symbols() if hasattr(fm, 'get_symbols') else [])]
    except Exception:
        return []


def _default_fetch(sym: str, tf: str, limit: int):
    from detection.market_data import get_market_data
    return get_market_data().fetch_klines(sym, limit=limit, interval=tf)


def _default_scanner_recs(syms) -> Dict[str, Dict]:
    try:
        from detection.smc_scanner import get_smc_scanner
        sc = get_smc_scanner()
        if sc is None or not hasattr(sc, 'spring_for'):
            return {}
        return {k.upper(): v for k, v in (sc.spring_for(syms) or {}).items()
                if isinstance(v, dict) and v.get('ok', True)}
    except Exception:
        return {}


def _default_funding_info() -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    try:
        from detection.funding_monitor import get_funding_monitor
        fm = get_funding_monitor()
        if not fm:
            return out
        g = lambda n: ({str(k).upper(): v for k, v in (getattr(fm, n)() or {}).items()}
                       if hasattr(fm, n) else {})
        rates, nxt, vols, tr = g('get_rates'), g('get_next_funding'), g('get_volumes'), g('get_trends')
        for s in set(rates) | set(nxt) | set(vols):
            out[s] = {'rate': rates.get(s), 'next_ms': nxt.get(s),
                      'vol24h': vols.get(s), 'f_trend': tr.get(s)}
    except Exception:
        pass
    return out


def init_funding_spring(db=None, get_symbols=None, fetch=None,
                        scanner_recs=None, funding_info=None) -> FundingSpring:
    global _instance
    if _instance is None:
        _instance = FundingSpring(db, get_symbols or _default_symbols,
                                  fetch or _default_fetch,
                                  scanner_recs or _default_scanner_recs,
                                  funding_info or _default_funding_info)
        if _instance.is_enabled():
            _instance.start()
    return _instance


def get_funding_spring() -> Optional[FundingSpring]:
    return _instance
