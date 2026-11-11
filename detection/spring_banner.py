"""⚖ Банер «SMC Forecast» — куди накручено обидві «Пружини» (вимога 11.11).

Дослівно: «Розроби цей банер для бота і вже з бота на інфосайт його транслюй
показники» + «Таймер запамʼятовуй при рестарті бота».

- Сторони: 🔭 скан біржі (банер останнього скану, підпис = біржа скану) і
  🌀 WATCHLIST (бари Bybit). Джерело — ті самі `spring_coil.aggregate`, що й
  банери «Куди накручено» на /smart-money (`eng.banner_agg()`,
  `sc.spring_banner()`); своїх розрахунків тут немає.
- Таймер «відколи тримається напрямок» — ВЕДЕ БОТ безперервно (тік із потоку
  🔭 скану раз на 5с), а не сторінка на запит. Стан — у БД (`DB_KEY`), тож
  рестарт його не обнуляє.
- ⚠️ «Немає даних» ≠ «рівновага»: сторона без даних таймер НЕ чіпає; у перші
  `WARMUP_SEC` після старту СПІЛЬНИЙ напрямок не змінюється, поки хоч одна
  сторона ще без даних (інакше кожен рестарт скидав таймер, кейс 11.11).
"""
import json
import threading
import time
from typing import Dict, List, Optional

DB_KEY = 'info_spring_banner_v1'
EX_LABEL = {'binance': 'Binance', 'bybit': 'Bybit'}
SIDES = ('LONG', 'SHORT')
MIN_GAP_SEC = 3.0
WARMUP_SEC = 600.0

_lock = threading.Lock()
_state: Dict[str, Dict] = {}
_loaded = [False]
_cache = {'ts': 0.0, 'out': None}
_BOOT = time.time()


# ── ЧИСТІ правила ───────────────────────────────────────────────────────────
def lean_of(agg) -> Optional[str]:
    """LONG / SHORT / FLAT (рівновага або замало заряджених); None — даних немає."""
    if not isinstance(agg, dict):
        return None
    try:
        if int(agg.get('total') or 0) <= 0 and int(agg.get('coins') or 0) <= 0:
            return None
    except (TypeError, ValueError):
        pass
    return agg.get('lean') if agg.get('lean') in SIDES else 'FLAT'


def track(prev: Optional[Dict], lean: Optional[str], now: float, data_ts=None) -> Optional[Dict]:
    """Той самий напрямок → `since` не рухається; зміна → відлік від моменту
    ДАНИХ, що її показали; немає даних (`lean` None) → без змін."""
    if lean is None:
        return dict(prev) if prev else None
    if prev and prev.get('lean') == lean and prev.get('since'):
        return dict(prev)
    try:
        ts = float(data_ts) if data_ts and 0 < float(data_ts) <= now else now
    except (TypeError, ValueError):
        ts = now
    if prev and prev.get('since') and float(prev['since']) > ts:
        ts = now
    return {'lean': lean, 'since': float(ts)}


def combined(leans: List[Optional[str]], warm: bool = False) -> Optional[str]:
    """Спільний напрямок (ОДИН таймер): обидві в один бік → він; одна з
    напрямком, друга ⚖ → її бік; протилежні → FLAT. Сторона без даних:
    у прогріві → None (не чіпаємо), далі — судимо за тими, що є."""
    if warm and any(x is None for x in leans):
        return None
    have = [x for x in leans if x is not None]
    if not have:
        return None
    ds = [x for x in have if x in SIDES]
    if not ds:
        return 'FLAT'
    return ds[0] if all(x == ds[0] for x in ds) else 'FLAT'


def view(key: str, label: str, agg, st) -> Dict:
    """Рядок банера для сторінки."""
    lean = lean_of(agg)
    if lean is None:
        return {'key': key, 'label': label, 'ok': False}
    pl = float(agg.get('prob_long') or 50.0)
    win = 'LONG' if pl >= 50 else 'SHORT'
    return {'key': key, 'label': label, 'ok': True, 'lean': lean,
            'pct': round(pl if win == 'LONG' else 100.0 - pl, 1), 'win': win,
            'prob_long': round(pl, 1), 'enough': bool(agg.get('enough')),
            'coins': int(agg.get('coins') or 0), 'total': int(agg.get('total') or 0),
            'need': int(agg.get('need') or 0),
            'n_long': int(agg.get('n_long') or 0), 'n_short': int(agg.get('n_short') or 0),
            'fired_long': int(agg.get('fired_long') or 0),
            'fired_short': int(agg.get('fired_short') or 0),
            'since': (st or {}).get('since') if (st or {}).get('lean') == lean else None}


# ── Стан + персист ──────────────────────────────────────────────────────────
def _db():
    try:
        from storage.db_operations import get_db
        return get_db()
    except Exception:
        return None


def _load():
    if _loaded[0]:
        return
    _loaded[0] = True
    db = _db()
    try:
        raw = db.get_setting(DB_KEY, None) if db else None
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, dict):
            for k in ('scan', 'wl', 'all'):
                v = raw.get(k)
                if isinstance(v, dict) and v.get('lean') and v.get('since'):
                    _state[k] = {'lean': v['lean'], 'since': float(v['since'])}
    except Exception:
        pass


def _scanner():
    try:
        from detection.smc_scanner import get_smc_scanner
        return get_smc_scanner()
    except Exception:
        return None


def _scan_engine():
    import sys
    m = sys.modules.get('detection.spring_scan')
    try:
        return m.get_spring_scan() if m is not None else None
    except Exception:
        return None


def sources(sc, eng) -> List[tuple]:
    """[(key, exchange, agg, ts)] — сторони банера з їх даними."""
    src = []
    try:
        if eng is not None:
            st = eng.state().get('result') or {}
            ex = str(st.get('exchange') or 'binance').lower()
            src.append(('scan', ex, eng.banner_agg(), st.get('ts')))
    except Exception:
        pass
    try:
        if sc is not None and hasattr(sc, 'spring_banner'):
            ts = None
            try:
                ts = (sc.spring_stats() or {}).get('calc_at')
            except Exception:
                pass
            src.append(('wl', 'bybit', sc.spring_banner(), ts))
    except Exception:
        pass
    return src


def compute(sc=None, eng=None, now: Optional[float] = None, force: bool = False) -> List[Dict]:
    """Банер: сторони + `{'key':'all', lean, since}`. Відстеження й персист."""
    now = time.time() if now is None else now
    with _lock:
        if not force and _cache['out'] is not None and now - _cache['ts'] < MIN_GAP_SEC:
            return [dict(x) for x in _cache['out']]
    sc = sc if sc is not None else _scanner()
    eng = eng if eng is not None else _scan_engine()
    src = sources(sc, eng)
    with _lock:
        _load()
        out, changed, leans, tss = [], False, [], []
        for key, ex, agg, ts in src:
            prev = _state.get(key)
            ln = lean_of(agg)
            leans.append(ln)
            try:
                if ts:
                    tss.append(float(ts))
            except (TypeError, ValueError):
                pass
            nxt = track(prev, ln, now, ts)
            if nxt and nxt != prev:
                _state[key] = nxt
                changed = True
            v = view(key, EX_LABEL.get(ex, ex.title()), agg, _state.get(key))
            v['ex'] = ex
            out.append(v)
        if src:
            warm = now - _BOOT < WARMUP_SEC
            prev = _state.get('all')
            nxt = track(prev, combined(leans, warm), now, max(tss) if tss else None)
            if nxt and nxt != prev:
                _state['all'] = nxt
                changed = True
            a = _state.get('all') or {}
            out.append({'key': 'all', 'ok': bool(a), 'lean': a.get('lean'), 'since': a.get('since')})
        _cache.update(ts=now, out=[dict(x) for x in out])
        snap = dict(_state) if changed else None
    if snap is not None:
        db = _db()
        try:
            if db:
                db.set_setting(DB_KEY, snap)
        except Exception:
            pass
    return out


def lean_for(key: str) -> Optional[str]:
    """Поточний напрямок сторони / спільний ('scan' | 'wl' | 'all')."""
    for v in compute():
        if v.get('key') == key:
            return v.get('lean') if v.get('ok') else None
    return None


def payload(now: Optional[float] = None) -> Dict:
    now = time.time() if now is None else now
    return {'ok': True, 'banners': compute(now=now), 'ts': now}
