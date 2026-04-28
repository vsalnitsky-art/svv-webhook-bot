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
    """Pine ta.ema()."""
    if not values or period < 1:
        return []
    alpha = 2.0 / (period + 1)
    out = [values[0]]
    for i in range(1, len(values)):
        out.append(alpha * values[i] + (1 - alpha) * out[-1])
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


def _stoch_pct(values: List[float], length: int) -> List[float]:
    """Pine custom proCTR_stoch():
       lo = lowest(src, length), hi = highest(src, length)
       result = d == 0 ? 50 : (src - lo) / d * 100
    """
    if not values:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - length + 1)
        w = values[start:i + 1]
        lo = min(w)
        hi = max(w)
        d = hi - lo
        out.append(50.0 if d == 0 else (values[i] - lo) / d * 100.0)
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
        trend_ref = (ema_n[-1] + sma_hlc3_n[-1]) / 2
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
             lower: int = 25) -> Dict:
    """CTR via STC (Schaff Trend Cycle).
    
    Pipeline:
        MACD = EMA(close, fast_len) - EMA(close, slow_len)
        K    = stoch_pct(MACD, cycle_len)
        D    = EMA(K, d1_len)
        KD   = stoch_pct(D, cycle_len)
        STC  = clamp(EMA(KD, d2_len), 0, 100)
    
    Signals:
        BUY  = STC crossover lower (was ≤lower, now >lower)
        SELL = STC crossunder upper (was ≥upper, now <upper)
    
    Returns the LATEST signal in history for informational display.
    """
    needed = slow_len + cycle_len + d1_len + d2_len + 5
    if not closes or len(closes) < needed:
        return {
            'stc': None, 'last_dir': None,
            'reason': f'need {needed}+ bars, got {len(closes) if closes else 0}',
        }
    
    fast_ema = _ema(closes, fast_len)
    slow_ema = _ema(closes, slow_len)
    macd = [f - s for f, s in zip(fast_ema, slow_ema)]
    
    k = _stoch_pct(macd, cycle_len)
    d = _ema(k, d1_len)
    kd = _stoch_pct(d, cycle_len)
    stc_raw = _ema(kd, d2_len)
    stc = [max(0.0, min(100.0, v)) for v in stc_raw]
    
    last_dir = None
    last_idx = None
    for i in range(1, len(stc)):
        if stc[i - 1] <= lower and stc[i] > lower:
            last_dir = 'LONG'
            last_idx = i
        elif stc[i - 1] >= upper and stc[i] < upper:
            last_dir = 'SHORT'
            last_idx = i
    
    return {
        'stc': round(stc[-1], 2),
        'last_dir': last_dir,
        'last_signal_idx': last_idx,
        'last_signal_age_bars': (len(stc) - 1 - last_idx) if last_idx is not None else None,
    }


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
        if ltf_klines:
            try:
                closes = [k['p'] for k in ltf_klines]
                result['ctr'] = calc_ctr(closes)
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
