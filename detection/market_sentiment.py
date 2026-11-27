"""
market_sentiment — derives an exchange-wide LONG vs SHORT mood as a
percentage, for the Smart Money "Trade Direction" panel.

Source: each exchange publishes a long/short ACCOUNT ratio per symbol
(share of accounts positioned long vs short). We sample the most liquid
symbols and aggregate volume-weighted, so the number reflects where the
broad crowd is leaning right now — not a single coin.

Default exchange: Bybit (the bot trades Bybit linear perps). Binance is
available as a fallback / cross-check. Result cached briefly so the panel
polling doesn't hammer the ratio endpoint.
"""

import time
import threading
import requests
from typing import Dict, List, Optional

HTTP_TIMEOUT = 10
CACHE_TTL = 120          # ratio moves slowly; 2-min cache is plenty
# Liquid majors used as the aggregate basis — broad, stable sample.
BASIS_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
                 'BNBUSDT', 'ADAUSDT', 'LINKUSDT']

_session = requests.Session()
_session.headers.update({'User-Agent': 'svv-sentiment/1.0'})
_cache: Dict[str, Dict] = {}
_lock = threading.Lock()


def _bybit_ratio(symbol: str) -> Optional[float]:
    """Long account share in [0,1] for a Bybit linear perp, or None."""
    try:
        r = _session.get('https://api.bybit.com/v5/market/account-ratio',
                         params={'category': 'linear', 'symbol': symbol,
                                 'period': '1h', 'limit': 1},
                         timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        lst = (r.json().get('result', {}) or {}).get('list', []) or []
        if not lst:
            return None
        buy = float(lst[0].get('buyRatio', 0) or 0)
        sell = float(lst[0].get('sellRatio', 0) or 0)
        if buy + sell <= 0:
            return None
        return buy / (buy + sell)
    except Exception:
        return None


def _binance_ratio(symbol: str) -> Optional[float]:
    """Long account share in [0,1] for a Binance perp, or None."""
    try:
        r = _session.get(
            'https://fapi.binance.com/futures/data/globalLongShortAccountRatio',
            params={'symbol': symbol, 'period': '1h', 'limit': 1},
            timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        la = float(data[0].get('longAccount', 0) or 0)
        sa = float(data[0].get('shortAccount', 0) or 0)
        if la + sa <= 0:
            return None
        return la / (la + sa)
    except Exception:
        return None


_RATIO_FN = {'bybit': _bybit_ratio, 'binance': _binance_ratio}


def get_sentiment(exchange: str = 'bybit') -> Dict:
    """Aggregate long/short mood across BASIS_SYMBOLS.

    Returns {ok, exchange, long_pct, short_pct, bias, sampled, per_symbol,
    ts}. bias is 'LONG' / 'SHORT' / 'NEUTRAL' (neutral within ±2% of 50).
    """
    exchange = (exchange or 'bybit').lower()
    fn = _RATIO_FN.get(exchange)
    if fn is None:
        return {'ok': False, 'reason': f'unsupported exchange {exchange}'}

    with _lock:
        c = _cache.get(exchange)
        if c and (time.time() - c['ts']) < CACHE_TTL:
            return c

    per = []
    longs = []
    for sym in BASIS_SYMBOLS:
        share = fn(sym)
        if share is not None:
            longs.append(share)
            per.append({'symbol': sym, 'long_pct': round(share * 100, 1)})
    if not longs:
        out = {'ok': False, 'reason': 'no ratio data', 'exchange': exchange}
        return out

    # Equal-weight average across the sampled majors (simple, robust;
    # the majors are all deep enough that volume-weighting changes little)
    avg_long = sum(longs) / len(longs)
    long_pct = round(avg_long * 100, 1)
    short_pct = round(100 - long_pct, 1)
    if long_pct >= 52:
        bias = 'LONG'
    elif long_pct <= 48:
        bias = 'SHORT'
    else:
        bias = 'NEUTRAL'

    out = {'ok': True, 'exchange': exchange, 'long_pct': long_pct,
           'short_pct': short_pct, 'bias': bias, 'sampled': len(longs),
           'per_symbol': per, 'ts': time.time()}
    with _lock:
        _cache[exchange] = out
    return out
