"""
detection/forecast_engine.py — Pine SMC PRO BOT indicators ported 1:1.

Implements 2 of 3 Pine indicators (Strength postponed by user):
  - Forecast 1H (Прогноз 1Н) — 6 Fibonacci horizons aggregated into ±100% prediction
  - CTR (Cyclic Trend Reversal) — Schaff Trend Cycle momentum signals

Pine reference: SMC_PRO_BOT__47_.pine (lines 1294-1668)

Data fetching:
  - Forecast 1H: 300 1H bars per symbol (~12 days, covers all 6 horizons + ATR averaging)
  - CTR: uses LTF klines passed by caller (scanner already fetches them)

Real-time: includes the live (unclosed) bar — no waiting for bar close.
This matches Pine `request.security(... lookahead=barmerge.lookahead_off)` semantics.
"""

import time
import threading
from typing import Dict, List, Optional


# ============================================================
# Pine TA helpers — 1:1 implementations
# ============================================================

def _ema(values: List[float], period: int) -> List[float]:
    """Pine-accurate ta.ema().
    
    Pine semantics for ta.ema(src, length):
        - Bars 0 .. length-2:   value is na (we model this as None)
        - Bar  length-1:        SMA seed = mean(values[0..length-1])
        - Bars length+:         alpha*src + (1-alpha)*prev_ema
    
    Why it matters for CTR specifically:
        STC chains 4 EMAs on top of one another. A naive EMA that just sets
        out[0] = values[0] (without the SMA seed phase) drifts noticeably for
        the first ~3-5*period bars. Through 4 chained EMAs that drift
        compounds — early STC values can sit at saturated 0 or 100 for many
        bars longer than they should, producing crossovers that don't exist
        in TradingView.
    
    Returned shape: same length as input. None entries for the na region;
    callers should treat None as "not yet defined" in stoch/cross logic.
    """
    n = len(values) if values else 0
    if n == 0 or period < 1:
        return []
    
    out: List = [None] * n
    if n < period:
        return out  # entirely na — caller must guard length
    
    # Bar period-1: SMA seed
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    
    # Bar period+: standard EMA recurrence
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(values: List[float], period: int) -> List[float]:
    """Pine ta.sma()."""
    if not values:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - period + 1)
        w = values[start:i + 1]
        out.append(sum(w) / len(w))
    return out


def _atr(klines: List[Dict], period: int) -> List[float]:
    """Pine ta.atr() — Wilder's RMA of True Range."""
    if not klines:
        return []
    if len(klines) < 2:
        return [0.0]
    
    trs = [0.0]
    for i in range(1, len(klines)):
        h = klines[i].get('h', klines[i]['p'])
        l = klines[i].get('l', klines[i]['p'])
        prev_c = klines[i - 1].get('p', 0)
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
    
    # Wilder's RMA
    out = [trs[0]]
    alpha = 1.0 / period
    for i in range(1, len(trs)):
        out.append(alpha * trs[i] + (1 - alpha) * out[-1])
    return out


def _stoch_pct(values: List, length: int) -> List:
    """Pine-accurate proCTR_stoch(src, length).
    
        lo = ta.lowest(src, length)
        hi = ta.highest(src, length)
        d  = hi - lo
        result = d == 0 ? 50 : (src - lo) / d * 100
    
    Pine ta.lowest / ta.highest are na-aware: if any of the last `length`
    values is na the whole window result is na. We mirror that — if the
    src value at i is None, or if the window hasn't seen `length`
    non-None inputs yet, we emit None.
    
    Returned list is parallel to `values` and may contain None entries.
    """
    n = len(values) if values else 0
    if n == 0:
        return []
    
    out: List = [None] * n
    for i in range(n):
        v = values[i]
        if v is None:
            continue
        start = max(0, i - length + 1)
        # Pine wants a full window of length `length`. Before that we keep
        # emitting na to match the platform exactly.
        if i - start + 1 < length:
            continue
        w = values[start:i + 1]
        if any(x is None for x in w):
            continue
        lo = min(w)
        hi = max(w)
        d = hi - lo
        out[i] = 50.0 if d == 0 else (v - lo) / d * 100.0
    return out


# ============================================================
# Forecast 1H — 6 Fibonacci horizons
# Pine reference lines 1550-1585
# ============================================================

# Per Pine: ATR period mapping per horizon
_FORECAST_ATR_PERIODS = {5: 5, 8: 7, 13: 10, 21: 14, 34: 21, 55: 28}
# (horizon, momentum_lookback) tuples
_FORECAST_HORIZONS = [(5, 1), (8, 2), (13, 3), (21, 5), (34, 8), (55, 13)]


def calc_forecast_1h(klines_1h: List[Dict]) -> Dict:
    """Compute Прогноз 1H — 6 Fibonacci horizons aggregated.
    
    Per horizon score (-2 to +2):
        trend      ±1 / 0  (close vs (EMA(N) + SMA(hlc3, N))/2)
        momentum   ±0.5 / 0  (close - close[lookback])
        volatility +0.5 / 0  (ATR(P) > SMA(ATR(P), 20))
    
    Sum range: -12 to +12, normalized to ±100%.
    
    Returns dict with: sum, pct, side (-1/0/1), confidence (50/60/75/90),
    scores per horizon, full horizon breakdown.
    """
    if not klines_1h:
        return _empty_forecast('no klines')
    if len(klines_1h) < 80:
        return _empty_forecast(f'need 80+ 1H bars, got {len(klines_1h)}')
    
    closes = [k['p'] for k in klines_1h]
    hlc3 = [(k.get('h', k['p']) + k.get('l', k['p']) + k['p']) / 3 for k in klines_1h]
    last_close = closes[-1]
    
    total_sum = 0.0
    scores = {}
    horizon_details = {}
    
    for n, mom_lookback in _FORECAST_HORIZONS:
        ema_n = _ema(closes, n)
        sma_hlc3_n = _sma(hlc3, n)
        # With pine-accurate EMA, ema_n[-1] is None when len(closes) < n.
        # _FORECAST_HORIZONS goes up to 55 and we already required 55+ bars,
        # so this should not happen, but guard anyway.
        ema_last = ema_n[-1] if ema_n[-1] is not None else last_close
        trend_ref = (ema_last + sma_hlc3_n[-1]) / 2
        if last_close > trend_ref:
            t = 1
        elif last_close < trend_ref:
            t = -1
        else:
            t = 0
        
        if len(closes) > mom_lookback:
            mom = last_close - closes[-1 - mom_lookback]
        else:
            mom = 0
        if mom > 0:
            mom_score = 0.5
        elif mom < 0:
            mom_score = -0.5
        else:
            mom_score = 0.0
        
        atr_p = _FORECAST_ATR_PERIODS[n]
        atr_n = _atr(klines_1h, atr_p)
        atr_avg_n = _sma(atr_n, 20)
        vol_score = 0.5 if atr_n[-1] > atr_avg_n[-1] else 0.0
        
        sc = t + mom_score + vol_score
        scores[n] = round(sc, 2)
        total_sum += sc
        horizon_details[n] = {
            'trend': t,
            'momentum': mom_score,
            'volatility': vol_score,
            'score': round(sc, 2),
        }
    
    pct = int((total_sum / 12.0) * 100)
    abs_sum = abs(total_sum)
    if abs_sum >= 10:
        conf = 90
    elif abs_sum >= 7:
        conf = 75
    elif abs_sum >= 4:
        conf = 60
    else:
        conf = 50
    side = 1 if total_sum > 0 else (-1 if total_sum < 0 else 0)
    
    return {
        'sum': round(total_sum, 2),
        'pct': pct,
        'side': side,
        'confidence': conf,
        'scores': scores,
        'horizons': horizon_details,
    }


def _empty_forecast(reason: str = '') -> Dict:
    return {
        'sum': 0, 'pct': 0, 'side': 0, 'confidence': 0,
        'scores': {}, 'horizons': {}, 'reason': reason,
    }


# ============================================================
# CTR — Cyclic Trend Reversal (Schaff Trend Cycle)
# Pine reference lines 1622-1668
# ============================================================

def calc_ctr(closes: List[float],
             fast_len: int = 21,
             slow_len: int = 50,
             cycle_len: int = 10,
             d1_len: int = 3,
             d2_len: int = 3,
             upper: int = 75,
             lower: int = 25,
             strength_raw: Optional[int] = None) -> Dict:
    """CTR via STC (Schaff Trend Cycle) — Pine-accurate replica.
    
    Pipeline (matches Pine SMC_PRO_BOT lines 1634-1642 exactly):
        ctrFastEMA = ta.ema(close, fast_len)
        ctrSlowEMA = ta.ema(close, slow_len)
        ctrMACD    = ctrFastEMA - ctrSlowEMA
        ctrK   = proCTR_stoch(ctrMACD, cycle_len)
        ctrD   = ta.ema(ctrK, d1_len)
        ctrKD  = proCTR_stoch(ctrD, cycle_len)
        ctrSTC = ta.ema(ctrKD, d2_len)
        ctrSTC := math.max(math.min(ctrSTC, 100), 0)
    
    Raw signals (Pine lines 1645-1646):
        ctrRawBuy  = ta.crossover(ctrSTC, lower)
        ctrRawSell = ta.crossunder(ctrSTC, upper)
    
    Strength filter (Pine lines 1651-1652):
        ctrLongFiltered  = ctrRawBuy  and momStrengthRaw > 0
        ctrShortFiltered = ctrRawSell and momStrengthRaw < 0
    
    The strength_raw argument is the multi-TF momentum strength score
    computed by detection.momentum_strength. When None, we fall back to
    raw STC crossovers (informational only — caller should warn the user
    that strength filtering is unavailable).
    
    Returns the latest signal observed in the available history.
    """
    needed = slow_len + cycle_len + d1_len + d2_len + 5
    if not closes or len(closes) < needed:
        return {
            'stc': None, 'last_dir': None, 'last_signal_idx': None,
            'last_signal_age_bars': None,
            'reason': f'need {needed}+ bars, got {len(closes) if closes else 0}',
        }
    
    fast_ema = _ema(closes, fast_len)
    slow_ema = _ema(closes, slow_len)
    # MACD is na while either component is na — mirror Pine na propagation
    macd = []
    for f, s in zip(fast_ema, slow_ema):
        if f is None or s is None:
            macd.append(None)
        else:
            macd.append(f - s)
    
    k = _stoch_pct(macd, cycle_len)
    d = _ema_skip_none(k, d1_len)
    kd = _stoch_pct(d, cycle_len)
    stc_raw = _ema_skip_none(kd, d2_len)
    # Clamp to [0, 100], preserving None
    stc = []
    for v in stc_raw:
        if v is None:
            stc.append(None)
        else:
            stc.append(max(0.0, min(100.0, v)))
    
    # Find the last raw STC crossover in history. Pine ta.crossover requires
    # both [i] and [i-1] to be non-na — we mirror that.
    last_raw_dir = None
    last_raw_idx = None
    for i in range(1, len(stc)):
        prev, cur = stc[i - 1], stc[i]
        if prev is None or cur is None:
            continue
        if prev <= lower and cur > lower:
            last_raw_dir = 'LONG'
            last_raw_idx = i
        elif prev >= upper and cur < upper:
            last_raw_dir = 'SHORT'
            last_raw_idx = i
    
    # Apply Pine's Strength filter on the last raw signal. If strength
    # disagrees with the raw direction we treat it as no-signal, matching
    # Pine where ctrLongFiltered/ctrShortFiltered would have been false on
    # that bar — Pine simply wouldn't have plotted the marker.
    last_dir = None
    last_idx = None
    strength_filter_applied = False
    if last_raw_dir is not None and strength_raw is not None:
        strength_filter_applied = True
        if last_raw_dir == 'LONG' and strength_raw > 0:
            last_dir, last_idx = last_raw_dir, last_raw_idx
        elif last_raw_dir == 'SHORT' and strength_raw < 0:
            last_dir, last_idx = last_raw_dir, last_raw_idx
        # else: signal filtered out (strength disagrees) → last_dir stays None
    elif last_raw_dir is not None:
        # No strength info → return raw signal as a best-effort, but flag it
        last_dir, last_idx = last_raw_dir, last_raw_idx
    
    stc_last = stc[-1] if stc and stc[-1] is not None else None
    
    return {
        'stc': round(stc_last, 2) if stc_last is not None else None,
        'last_dir': last_dir,
        'last_signal_idx': last_idx,
        'last_signal_age_bars': (len(stc) - 1 - last_idx) if last_idx is not None else None,
        'strength_raw': strength_raw,
        'strength_filter_applied': strength_filter_applied,
        # Diagnostic: also expose the raw (pre-filter) signal so the UI can
        # show "would have fired but strength disagreed" if we want.
        'raw_dir': last_raw_dir,
        'raw_idx': last_raw_idx,
    }


def _ema_skip_none(values: List, period: int) -> List:
    """EMA over a series that may contain Nones at the start. We treat the
    leading na region as absent — the SMA seed and the recurrence both
    operate on the trailing non-None segment.
    
    This matches Pine: when ta.ema is fed a series whose initial values are
    na, ta.ema only starts producing output once the first `length`
    non-na inputs have arrived.
    """
    n = len(values) if values else 0
    if n == 0 or period < 1:
        return []
    out: List = [None] * n
    
    # Find the index of the first non-None value
    start_idx = None
    for i, v in enumerate(values):
        if v is not None:
            start_idx = i
            break
    if start_idx is None:
        return out
    
    # We need `period` consecutive non-None values starting at start_idx
    # (in practice the upstream is contiguous after its own seed phase, so
    # this is just a length check).
    seed_end = start_idx + period
    if seed_end > n:
        return out
    seed_window = values[start_idx:seed_end]
    if any(x is None for x in seed_window):
        # Discontinuity — punt; this should not happen in practice but we
        # don't want to silently corrupt EMA if the upstream has gaps.
        return out
    
    seed = sum(seed_window) / period
    out[seed_end - 1] = seed
    alpha = 2.0 / (period + 1)
    for i in range(seed_end, n):
        v = values[i]
        prev = out[i - 1]
        if v is None or prev is None:
            # gap — propagate na
            continue
        out[i] = alpha * v + (1 - alpha) * prev
    return out


# ============================================================
# ForecastEngine — caches per-symbol forecasts
# ============================================================

FETCH_LIMIT_1H = 300  # ~12.5 days at 1H — plenty for horizon 55 + ATR(28) avg(20)


class ForecastEngine:
    """Per-symbol forecast cache. Updated by SMC scanner each scan cycle."""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._cache: Dict[str, Dict] = {}
    
    def update(self, symbol: str, ltf_klines: Optional[List[Dict]] = None) -> Dict:
        """Compute fresh forecast + CTR. Always overwrites cache."""
        from detection.market_data import get_market_data
        
        result = {
            'symbol': symbol,
            'computed_at': time.time(),
            'forecast_1h': _empty_forecast('not computed yet'),
            'ctr': {'stc': None, 'last_dir': None, 'reason': 'not computed yet'},
        }
        
        # ---- Forecast 1H ----
        try:
            md = get_market_data()
            klines_1h = None
            if md and hasattr(md, 'fetch_klines'):
                if 'interval' in md.fetch_klines.__code__.co_varnames:
                    klines_1h = md.fetch_klines(symbol, limit=FETCH_LIMIT_1H, interval='1h')
            if klines_1h and len(klines_1h) >= 80:
                result['forecast_1h'] = calc_forecast_1h(klines_1h)
            else:
                got = len(klines_1h) if klines_1h else 0
                result['forecast_1h'] = _empty_forecast(f'insufficient 1H data ({got})')
        except Exception as e:
            result['forecast_1h'] = _empty_forecast(f'fetch error: {e}')
        
        # ---- CTR — uses LTF klines from scanner ----
        # We feed momStrengthRaw from the multi-TF momentum module so that
        # only signals matching Pine's ctrLongFiltered/ctrShortFiltered
        # propagate. Without strength filter the Python CTR shows raw STC
        # crossovers — twice as many signals as TradingView, with the
        # opposite direction sometimes "winning" between scan cycles.
        if ltf_klines:
            try:
                closes = [k['p'] for k in ltf_klines]
                
                # Get momStrengthRaw — best-effort, falls through to raw STC
                # if the strength module is unavailable or fetch failed.
                strength_raw = None
                try:
                    from detection.momentum_strength import get_momentum_strength
                    ms = get_momentum_strength()
                    if ms and closes:
                        s = ms.get_strength(symbol, closes[-1])
                        if s is not None:
                            strength_raw = s.get('raw')
                except Exception as e:
                    print(f"[ForecastEngine] strength fetch error for {symbol}: {e}")
                
                result['ctr'] = calc_ctr(closes, strength_raw=strength_raw)
            except Exception as e:
                result['ctr'] = {'stc': None, 'last_dir': None,
                                  'reason': f'compute error: {e}'}
        
        with self._lock:
            self._cache[symbol] = result
        return result
    
    def get(self, symbol: str) -> Optional[Dict]:
        """Return cached forecast (may be stale)."""
        with self._lock:
            return dict(self._cache[symbol]) if symbol in self._cache else None
    
    def clear(self, symbol: Optional[str] = None):
        with self._lock:
            if symbol:
                self._cache.pop(symbol, None)
            else:
                self._cache.clear()


# Singleton
_instance: Optional[ForecastEngine] = None


def get_forecast_engine() -> Optional[ForecastEngine]:
    return _instance


def init_forecast_engine() -> ForecastEngine:
    global _instance
    if _instance is None:
        _instance = ForecastEngine()
    return _instance
