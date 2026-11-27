"""
squeeze.py — TTM Squeeze (Bollinger-inside-Keltner) volatility compression.

The classic John Carter / LazyBear construction:
    BB  = SMA(close, 20) ± 2.0 × stdev(close, 20)
    KC  = SMA(close, 20) ± 1.5 × ATR(20)            (Wilder-smoothed ATR)
    Squeeze ON  when BB sits fully INSIDE KC (both bands)
    Squeeze OFF when BB expands back outside

Interpretation: volatility is coiling; the longer/deeper the squeeze, the
more violent the eventual expansion tends to be. Direction is NOT given by
the squeeze itself — we report a momentum reading alongside so the UI can
hint at likely expansion direction, but the squeeze % is direction-neutral.

`probability` mapping (what the UI shows as "SQUEEZE 95%"):
    ratio = BB_width / KC_width
    ratio >= 1.0      → 0%   (no squeeze)
    ratio  = 0.5      → 100% (BB half the width of KC — deep compression)
    linear in between, clamped.

Pure functions, no external deps beyond stdlib — easy to unit-test.
Klines are the bot-standard dicts from market_data.fetch_klines:
{p (close), o, h, l, v, t, ...}.
"""

import math
from typing import List, Dict, Optional


# === Multi-band TTM (2026-06-11, v3 — final) ===
# Problem history: the binary 1.5-Keltner cross almost never fires on
# 1h/4h crypto (stdev/ATR proportions there sit structurally above 1.5),
# so the gauge showed OFF forever. Two adaptive legs were prototyped and
# REJECTED with statistical tests: a percentile rank (fires a fixed ~15%
# of the time on stationary data by construction) and a median-relative
# width threshold (BB width over random-walk closes is too noisy — ~35%
# false-fire rate in steady regimes).
#
# Final design — the standard practitioner multi-band TTM:
#   unit_r = BB_width / (2 × ATR)   (Keltner multiplier as a unit)
#   unit_r < 1.0  → HIGH compression
#   unit_r < 1.5  → MID  (the classic squeeze)
#   unit_r < 2.0  → LOW  (meaningful on higher TFs)
#   unit_r ≥ 2.0  → OFF
# probability maps unit_r 2.0 → 0%, 0.5 → 100% (linear, clamped). Same
# formula on every timeframe — the GRADING is what adapts: 1h/4h setups
# now read "ON · low/mid compression" instead of a meaningless OFF, and
# the absolute ATR-unit threshold carries no built-in false-fire rate.
LOW_BAND = 2.0    # ON boundary (unit_r), and the 0% probability anchor
PROB_FLOOR_R = 0.5  # unit_r at which probability saturates to 100%


def _sma(vals: List[float], n: int) -> Optional[float]:
    if len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def _stdev(vals: List[float], n: int) -> Optional[float]:
    if len(vals) < n:
        return None
    window = vals[-n:]
    m = sum(window) / n
    var = sum((v - m) ** 2 for v in window) / n  # population, as TV does
    return math.sqrt(var)


def _atr_wilder(highs: List[float], lows: List[float],
                 closes: List[float], n: int) -> Optional[float]:
    """Wilder-smoothed ATR over the full series; returns the last value.
    Matches TradingView's ta.atr (RMA of true range)."""
    if len(closes) < n + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    # RMA: seed with SMA of first n TRs, then recursive
    atr = sum(trs[:n]) / n
    for tr in trs[n:]:
        atr = (atr * (n - 1) + tr) / n
    return atr


def calc_squeeze(klines: List[Dict],
                  bb_len: int = 20, bb_mult: float = 2.0,
                  kc_len: int = 20, kc_mult: float = 1.5) -> Dict:
    """Compute squeeze state from chronological klines (oldest first).
    
    Returns:
      {
        'ok': bool,
        'squeeze_on': bool,
        'probability': 0..100,      # compression intensity for the gauge
        'bars_in_squeeze': int,     # consecutive ON bars ending now
        'momentum': float,          # detrended momentum value (TTM-style)
        'momentum_rising': bool,
        'bb_width': float, 'kc_width': float,
        'reason': str (when not ok)
      }
    """
    need = max(bb_len, kc_len) + kc_len + 2  # ATR warmup headroom
    if not klines or len(klines) < need:
        return {'ok': False, 'squeeze_on': False, 'probability': 0,
                'bars_in_squeeze': 0, 'momentum': 0.0,
                'momentum_rising': False, 'bb_width': 0, 'kc_width': 0,
                'reason': f'insufficient data ({len(klines or [])} bars, need {need})'}
    
    closes = [float(k.get('p') or k.get('c') or k.get('close') or 0) for k in klines]
    highs = [float(k.get('h') or 0) for k in klines]
    lows = [float(k.get('l') or 0) for k in klines]
    if not all(closes):
        return {'ok': False, 'squeeze_on': False, 'probability': 0,
                'bars_in_squeeze': 0, 'momentum': 0.0,
                'momentum_rising': False, 'bb_width': 0, 'kc_width': 0,
                'reason': 'zero closes in data'}
    
    def state_at(end_idx: int):
        """Squeeze ratio at bar index end_idx (inclusive). Returns
        (squeeze_on, ratio) or None when warmup insufficient."""
        c = closes[:end_idx + 1]
        h = highs[:end_idx + 1]
        l = lows[:end_idx + 1]
        basis = _sma(c, bb_len)
        sd = _stdev(c, bb_len)
        atr = _atr_wilder(h, l, c, kc_len)
        if basis is None or sd is None or atr is None or atr <= 0:
            return None
        bb_w = 2 * bb_mult * sd
        kc_w = 2 * kc_mult * atr
        if kc_w <= 0:
            return None
        ratio = bb_w / kc_w
        return (ratio < 1.0, ratio, bb_w, kc_w)
    
    last = state_at(len(closes) - 1)
    if last is None:
        return {'ok': False, 'squeeze_on': False, 'probability': 0,
                'bars_in_squeeze': 0, 'momentum': 0.0,
                'momentum_rising': False, 'bb_width': 0, 'kc_width': 0,
                'reason': 'indicator warmup failed'}
    
    squeeze_on, ratio, bb_w, kc_w = last
    # Multi-band grading in ATR units: ratio is bbw/kcw at kc_mult, so
    # unit_r = ratio × kc_mult = bbw / (2·ATR) — multiplier-independent.
    unit_r = ratio * kc_mult
    squeeze_on = unit_r < LOW_BAND
    probability = max(0.0, min(1.0, (LOW_BAND - unit_r)
                                     / (LOW_BAND - PROB_FLOOR_R))) * 100.0
    band = ('high' if unit_r < 1.0 else
            'mid' if unit_r < 1.5 else
            'low' if unit_r < LOW_BAND else 'off')
    
    # Count consecutive ON bars ending at the last bar (walk back until OFF).
    # Capped at 50 walks — beyond that "50+" is plenty of information.
    # A bar counts as ON if EITHER detector fires: classic band cross OR
    # its relative width within the percentile-ON threshold.
    bars_in = 0
    if squeeze_on:
        for back in range(0, 50):
            idx = len(closes) - 1 - back
            st = state_at(idx)
            if st is None or (st[1] * kc_mult) >= LOW_BAND:
                break
            bars_in += 1
    
    # TTM momentum (LazyBear variant): close detrended against the average
    # of (donchian midline, sma), i.e. where price sits vs its recent
    # equilibrium. Sign hints at likely expansion direction.
    def momentum_at(end_idx: int) -> Optional[float]:
        c = closes[:end_idx + 1]
        h = highs[:end_idx + 1]
        l = lows[:end_idx + 1]
        if len(c) < kc_len:
            return None
        hh = max(h[-kc_len:])
        ll = min(l[-kc_len:])
        sma = _sma(c, kc_len)
        if sma is None:
            return None
        midline = ((hh + ll) / 2 + sma) / 2
        return c[-1] - midline
    
    mom_now = momentum_at(len(closes) - 1) or 0.0
    mom_prev = momentum_at(len(closes) - 2) or 0.0
    
    return {
        'ok': True,
        'squeeze_on': bool(squeeze_on),
        'band': band,
        'probability': round(probability, 1),
        'bars_in_squeeze': bars_in,
        'momentum': mom_now,
        'momentum_rising': mom_now > mom_prev,
        'bb_width': bb_w,
        'kc_width': kc_w,
        'reason': '',
    }
