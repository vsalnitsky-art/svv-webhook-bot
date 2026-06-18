"""
exchange_router — picks the best available exchange for analytics.

Policy: prefer Binance for analytics (deeper liquidity / volume → more
representative market read), but FALL BACK to Bybit when Binance is
unreachable (e.g. HTTP 418 geo-block on shared Render IPs). Every result is
tagged with the source so the UI can label it honestly.

Critical design point: we NEVER probe Binance inline on each calculation —
a blocked request can hang for the full timeout and freeze the UI. Instead a
cached health flag is refreshed at most once per CHECK_INTERVAL with a SHORT
timeout; all calculations read the cached flag instantly.
"""

import time
import threading
from typing import Optional, Dict, List, Tuple

CHECK_INTERVAL = 300      # re-test Binance availability at most every 5 min
PROBE_TIMEOUT = 3         # short timeout for the health ping (not 10s)

_lock = threading.Lock()
_state = {
    'binance_ok': None,       # None = never checked, True/False = last result
    'checked_at': 0.0,
    'last_reason': '',
}


def _probe_binance() -> Tuple[bool, str]:
    """One short, cheap request to confirm Binance Futures is reachable.
    Returns (ok, reason). Never raises."""
    try:
        import requests
        # /fapi/v1/ping is the lightest endpoint Binance exposes.
        r = requests.get('https://fapi.binance.com/fapi/v1/ping',
                         timeout=PROBE_TIMEOUT)
        if r.status_code == 200:
            return True, 'ok'
        return False, f'HTTP {r.status_code}'
    except Exception as e:
        return False, type(e).__name__


def _probe_bybit() -> Tuple[bool, str]:
    """Light reachability check for Bybit's public API. Never raises."""
    try:
        import requests
        r = requests.get('https://api.bybit.com/v5/market/time',
                         timeout=PROBE_TIMEOUT)
        if r.status_code == 200:
            return True, 'ok'
        return False, f'HTTP {r.status_code}'
    except Exception as e:
        return False, type(e).__name__


# Bybit health is cached the same way as Binance.
_bybit_state = {'ok': None, 'checked_at': 0.0, 'reason': ''}


def bybit_available(force: bool = False) -> bool:
    now = time.time()
    with _lock:
        fresh = (now - _bybit_state['checked_at']) < CHECK_INTERVAL
        if _bybit_state['ok'] is not None and fresh and not force:
            return _bybit_state['ok']
    ok, reason = _probe_bybit()
    with _lock:
        _bybit_state['ok'] = ok
        _bybit_state['checked_at'] = time.time()
        _bybit_state['reason'] = reason
    return ok


def binance_available(force: bool = False) -> bool:
    """Cached availability flag. Refreshes at most once per CHECK_INTERVAL.
    Reads are instant; only the periodic refresh touches the network."""
    now = time.time()
    with _lock:
        fresh = (now - _state['checked_at']) < CHECK_INTERVAL
        if _state['binance_ok'] is not None and fresh and not force:
            return _state['binance_ok']
    # Refresh (outside the lock so the probe's latency doesn't block readers
    # that arrive during it; worst case two probes fire, harmless).
    ok, reason = _probe_binance()
    with _lock:
        _state['binance_ok'] = ok
        _state['checked_at'] = time.time()
        _state['last_reason'] = reason
    return ok


def health() -> Dict:
    """Current router health for diagnostics/UI — both exchanges."""
    with _lock:
        return {
            'binance_ok': _state['binance_ok'],
            'binance_reason': _state['last_reason'],
            'binance_age_secs': round(time.time() - _state['checked_at'], 1)
                        if _state['checked_at'] else None,
            'bybit_ok': _bybit_state['ok'],
            'bybit_reason': _bybit_state['reason'],
            'bybit_age_secs': round(time.time() - _bybit_state['checked_at'], 1)
                        if _bybit_state['checked_at'] else None,
        }


def _binance():
    try:
        from core.binance_connector import get_binance_connector
        return get_binance_connector()
    except Exception:
        return None


def _bybit():
    try:
        from core.bybit_connector import get_connector
        return get_connector()
    except Exception:
        return None


# Bybit kline interval codes ↔ Binance codes differ; the Binance connector
# already maps via TIMEFRAME_MAP, so we pass the Bybit-style code to both.

def get_klines(symbol: str, interval: str = "240", limit: int = 120) -> Tuple[List[Dict], str]:
    """Return (klines, source). Tries Binance when available, else Bybit.
    Both connectors return the same dict shape {open,high,low,close,
    volume,timestamp}."""
    if binance_available():
        bn = _binance()
        if bn:
            try:
                k = bn.get_klines(symbol, interval=interval, limit=limit)
                if k and len(k) >= 2:
                    return k, 'binance'
            except Exception:
                pass
    bb = _bybit()
    if bb:
        try:
            k = bb.get_klines(symbol, interval=interval, limit=limit)
            if k:
                return k, 'bybit'
        except Exception:
            pass
    return [], 'none'


def get_funding_rate(symbol: str) -> Tuple[Optional[float], str]:
    """Return (funding_rate, source)."""
    if binance_available():
        bn = _binance()
        if bn:
            try:
                fd = bn.get_funding_rate(symbol)
                if fd and fd.get('funding_rate') is not None:
                    return fd['funding_rate'], 'binance'
            except Exception:
                pass
    bb = _bybit()
    if bb:
        try:
            fd = bb.get_funding_rate(symbol)
            if fd and fd.get('funding_rate') is not None:
                return fd['funding_rate'], 'bybit'
        except Exception:
            pass
    return None, 'none'


def get_open_interest(symbol: str, interval: str = "4h", limit: int = 12) -> Tuple[List[Dict], str]:
    """Return (oi_history, source)."""
    if binance_available():
        bn = _binance()
        if bn:
            try:
                oi = bn.get_open_interest(symbol, interval=interval, limit=limit)
                if oi:
                    return oi, 'binance'
            except Exception:
                pass
    bb = _bybit()
    if bb:
        try:
            oi = bb.get_open_interest(symbol, interval=interval, limit=limit)
            if oi:
                return oi, 'bybit'
        except Exception:
            pass
    return [], 'none'
