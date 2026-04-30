"""Momentum Strength — Pine-accurate replica of momStrengthRaw.

This module computes the same multi-timeframe momentum strength score that
the SMC_PRO_BOT Pine indicator uses to filter CTR signals. Without it, the
Python CTR shows raw STC crossovers — but Pine only emits a CTR signal when
strength agrees with the direction (LONG when raw > 0, SHORT when raw < 0).

Pine reference (lines 1297-1313 of SMC_PRO_BOT__47_.pine):

    [momEma1m,  momVwap1m]  = request.security(syminfo.tickerid, '1',
                                ta.ema(close, 20), ta.vwap(hlc3))
    [momEma5m,  momVwap5m]  = request.security(syminfo.tickerid, '5',  ...)
    [momEma15m, momVwap15m] = request.security(syminfo.tickerid, '15', ...)
    [momEma30m, momVwap30m] = request.security(syminfo.tickerid, '30', ...)
    [momEma1h,  momVwap1h]  = request.security(syminfo.tickerid, '60', ...)
    [momEma4h,  momVwap4h]  = request.security(syminfo.tickerid, '240',...)
    [momEmaD,   momVwapD]   = request.security(syminfo.tickerid, 'D',  ...)
    
    momT*  = close > ema AND close > vwap ? +1 :
             close < ema AND close < vwap ? -1 : 0
    
    momStrengthRaw = momT1m + momT5m + momT15m + momT30m + momT1h + momT4h + momTD
                   ∈ [-7, +7]

Key Pine semantics replicated here:

    1. ta.vwap(hlc3) defaults to anchor='session'. For crypto on Bybit the
       session anchors to 00:00 UTC. So ALL seven momVwap* values are the
       same daily VWAP — they only differ in the granularity at which they
       are sampled. We compute one daily VWAP per symbol from 1m candles
       (highest fidelity available cheaply) and reuse it.
    
    2. ta.ema(close, 20) is computed per timeframe. 20 bars on 1m is 20
       minutes; 20 bars on 1D is 20 days. They are genuinely different
       series and must be fetched independently.
    
    3. The current bar of each timeframe is the same physical close (the
       symbol's most recent trade) — Pine's request.security() returns
       same-bar value when the requested TF is ≤ chart TF. We mirror that
       by using a single fresh price from the highest-fidelity klines and
       comparing against each TF's EMA.

Caching is critical: we recompute strength at most once per CACHE_TTL_SEC
to keep API load reasonable. Each compute hits 7 endpoints which would
murder rate limits if done per-tick.
"""

from typing import Dict, List, Optional, Tuple
import threading
import time

# Per-TF (interval_str, lookback_bars) — enough history for ta.ema(close, 20)
# to have its SMA seed and ~80 bars of EMA recurrence.
# 100 bars × 1m = 100 minutes (~1.7h).  Enough for a 20-period EMA seed.
# 100 bars × 5m = 500 minutes (~8h).
# 100 bars × 15m = 25h.
# 100 bars × 30m = 50h.
# 100 bars × 60m = ~4 days.
# 100 bars × 240m = ~17 days.
# 100 bars × 1d = ~100 days. (Will trim for newer symbols.)
TF_LIST = [
    ('1m',  100),
    ('5m',  100),
    ('15m', 100),
    ('30m', 100),
    ('1h',  100),
    ('4h',  100),
    ('1d',  100),
]

# Map our interval strings to a friendly key for the result dict
TF_KEYS = {
    '1m': 'momT1m',  '5m': 'momT5m',  '15m': 'momT15m',
    '30m': 'momT30m', '1h': 'momT1h', '4h': 'momT4h', '1d': 'momTD',
}

# Result is good for ~60s — strength evolves slowly relative to fetch cost.
# CTR signal generation runs once per scan cycle (~30-60s) so this lines up.
CACHE_TTL_SEC = 60.0

# How long to wait before retrying after a hard failure on one TF (e.g. rate
# limited). We keep the stale value in cache and just don't refresh.
RETRY_BACKOFF_SEC = 30.0


def _ema_pine(values: List[float], period: int) -> List[Optional[float]]:
    """Pine-accurate ta.ema (mirrors detection.forecast_engine._ema)."""
    n = len(values) if values else 0
    if n == 0 or period < 1:
        return []
    out: List = [None] * n
    if n < period:
        return out
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _session_vwap_hlc3(klines: List[Dict]) -> Optional[float]:
    """Compute Pine's ta.vwap(hlc3) with anchor='session' (UTC daily reset).
    
    Returns the rolling VWAP value at the last (most recent) bar in `klines`.
    The session is bounded by the UTC midnight that falls at-or-before the
    last bar's open time. Bars before that midnight are ignored.
    
    Returns None if no bars fall in the current session (shouldn't happen
    in practice — even at 00:00:01 UTC there's at least one bar today).
    """
    if not klines:
        return None
    
    # Compute today's UTC-midnight ms timestamp from the most recent bar.
    last_t_ms = klines[-1].get('t', 0)
    if not last_t_ms:
        return None
    
    # 86400000 ms = 1 day. Floor to the start of the UTC day.
    DAY_MS = 86_400_000
    session_start_ms = (last_t_ms // DAY_MS) * DAY_MS
    
    cum_pv = 0.0
    cum_v = 0.0
    for k in klines:
        if k.get('t', 0) < session_start_ms:
            continue
        h = k.get('h', k.get('p', 0))
        l = k.get('l', k.get('p', 0))
        c = k.get('p', 0)
        v = k.get('v', 0)
        hlc3 = (h + l + c) / 3.0
        cum_pv += hlc3 * v
        cum_v += v
    
    if cum_v <= 0:
        return None
    return cum_pv / cum_v


def _direction_for_tf(close: float, ema: Optional[float],
                      vwap: Optional[float]) -> int:
    """Pine: close > ema AND close > vwap ? +1 :
             close < ema AND close < vwap ? -1 : 0"""
    if ema is None or vwap is None:
        return 0  # treat na as neutral
    if close > ema and close > vwap:
        return 1
    if close < ema and close < vwap:
        return -1
    return 0


class MomentumStrength:
    """Per-symbol momentum strength cache.
    
    Designed to be a singleton accessed from forecast_engine when computing
    CTR signals. The .get_strength() method blocks while fetching the seven
    timeframes the first time around (~1-3 seconds), but is essentially free
    after that for the next CACHE_TTL_SEC.
    """
    
    def __init__(self, market_data=None):
        self._md = market_data
        self._lock = threading.RLock()
        # symbol → {'raw': int, 'breakdown': dict, 'computed_at': float}
        self._cache: Dict[str, Dict] = {}
        # symbol → epoch when we last failed to refresh (so we back off)
        self._failed_at: Dict[str, float] = {}
    
    def set_market_data(self, market_data):
        self._md = market_data
    
    def _fetch_tf_pair(self, symbol: str, interval: str,
                       limit: int) -> Tuple[Optional[float], Optional[float]]:
        """Fetch one timeframe and return (ema20_last, vwap_session_last).
        
        Returns (None, None) on any error so the caller can substitute 0
        (neutral) for that TF's contribution.
        """
        if not self._md or not hasattr(self._md, 'fetch_klines'):
            return None, None
        try:
            klines = self._md.fetch_klines(symbol, limit=limit, interval=interval)
        except Exception:
            return None, None
        if not klines or len(klines) < 21:  # need at least 20 bars for EMA seed
            return None, None
        
        # Drop the in-progress bar so EMA matches Pine's barclose semantics
        # (same reasoning as in CTR proper — see smc_scanner closing-bar comment)
        klines_closed = klines[:-1] if len(klines) > 1 else klines
        if len(klines_closed) < 20:
            return None, None
        
        closes = [k.get('p', 0) for k in klines_closed]
        ema_series = _ema_pine(closes, 20)
        ema_last = ema_series[-1]
        
        # VWAP uses the FULL kline series including the in-progress bar —
        # ta.vwap(hlc3) accumulates by tick on the live bar. For our daily-
        # session VWAP we want the bar's hlc3 to count as soon as it opens.
        vwap_last = _session_vwap_hlc3(klines)
        
        return ema_last, vwap_last
    
    def _compute_now(self, symbol: str, current_close: float) -> Dict:
        """Hit all seven TFs and assemble the strength breakdown."""
        breakdown = {}
        raw = 0
        for interval, limit in TF_LIST:
            ema, vwap = self._fetch_tf_pair(symbol, interval, limit)
            t = _direction_for_tf(current_close, ema, vwap)
            breakdown[TF_KEYS[interval]] = {
                'dir': t, 'ema': ema, 'vwap': vwap,
            }
            raw += t
        
        return {
            'raw': raw,           # int in [-7, +7]
            'breakdown': breakdown,
            'computed_at': time.time(),
            'price': current_close,
        }
    
    def get_strength(self, symbol: str,
                     current_close: float) -> Optional[Dict]:
        """Return current strength for `symbol`, refreshing if stale.
        
        Args:
            symbol: e.g. 'BTCUSDT'
            current_close: latest known close price (used as the `close`
                value in Pine's `close > ema` checks). Caller already has
                this from the LTF klines they're scanning.
        
        Returns:
            {raw: int, breakdown: dict, computed_at: float, price: float}
            or None if market_data is unavailable.
        """
        if not self._md:
            return None
        
        now = time.time()
        with self._lock:
            cached = self._cache.get(symbol)
            failed_at = self._failed_at.get(symbol, 0)
        
        # Use cache if fresh
        if cached and now - cached['computed_at'] < CACHE_TTL_SEC:
            return dict(cached)
        
        # Backoff after recent failure
        if now - failed_at < RETRY_BACKOFF_SEC:
            if cached:
                return dict(cached)
            return None
        
        try:
            result = self._compute_now(symbol, current_close)
        except Exception as e:
            with self._lock:
                self._failed_at[symbol] = now
            print(f"[MomStrength] {symbol} compute error: {e}")
            if cached:
                return dict(cached)
            return None
        
        with self._lock:
            self._cache[symbol] = result
            self._failed_at.pop(symbol, None)
        return dict(result)
    
    def clear(self, symbol: Optional[str] = None):
        with self._lock:
            if symbol:
                self._cache.pop(symbol, None)
                self._failed_at.pop(symbol, None)
            else:
                self._cache.clear()
                self._failed_at.clear()


# ============================================================
# Singleton accessor (mirrors forecast_engine pattern)
# ============================================================

_instance: Optional[MomentumStrength] = None


def get_momentum_strength() -> Optional[MomentumStrength]:
    return _instance


def init_momentum_strength(market_data) -> MomentumStrength:
    global _instance
    if _instance is None:
        _instance = MomentumStrength(market_data=market_data)
    else:
        _instance.set_market_data(market_data)
    return _instance
