"""Volumized Order Blocks — Python port of the TradingView Pine indicator.

Source: SVV Charts "Volumized Order Blocks | SVV Charts with Alerts" by vsalnitsky.
Lines 247-334 of the source contain the core detection algorithm; this module is
a one-to-one port (state machine over bars, identical swing/OB rules, same ATR
filter and same invalidation/removal semantics).

Public API:
    detect_volumized_obs(klines, **settings) -> dict
    get_latest_ob_trend(klines, **settings) -> dict

Returned dict shape::

    {
        'bullish_obs': [...],          # active+breaker bullish OBs (newest first), trimmed to zone_count
        'bearish_obs': [...],          # symmetric for bearish
        'all_bullish': [...],          # untrimmed (for debugging/UI history)
        'all_bearish': [...],
        'latest_ob': {...} | None,     # most recent OB across both lists (by start_time)
        'trend': 'LONG' | 'SHORT' | None,
        'trend_meta': {...},           # info about the OB that determined trend
    }

Each OB dict::

    {
        'type': 'Bull' | 'Bear',
        'top': float,                # OB box top price
        'bottom': float,             # OB box bottom price
        'start_time': int,           # ms epoch (matches Pine `time` semantics)
        'start_idx': int,            # bar index for debugging
        'ob_volume': float,          # sum of 3-bar impulse volume
        'ob_low_volume': float,      # vol of bars i-2 (bull) or i+i-1 (bear)
        'ob_high_volume': float,     # vol of bars i+i-1 (bull) or i-2 (bear)
        'breaker': bool,             # True once invalidated (price closed/wicked through)
        'break_time': int | None,    # ms when invalidated
        'bb_volume': float,          # volume on the breaker bar
        'combined': bool,            # True if produced by combine_obs
    }

Design notes:
  * The algorithm is a per-bar state machine: swing detection, OB invalidation,
    OB formation. We walk through klines in order and maintain four pieces
    of state — current swing type (0=high/1=low), top swing point, bottom
    swing point, and the lists of active OBs.
  * Pine's `unshift` adds to FRONT of list; Python equivalent is `list.insert(0, ...)`.
  * Pine's `bullishOrderBlocksList.size() > maxOrderBlocks → pop()` trims the OLDEST
    (back of list). Python: `list.pop()` removes from end. ✓ Same behaviour.
  * ATR(10) uses Wilder's smoothing (RMA), not SMA. Pine's `ta.atr` matches this.
  * Pine reads `klines.h/l/o/c/v/t`; we accept dicts with keys 'h','l','o','c'
    (or 'p' for close), 'v', 't'. This matches the project's existing klines shape.
"""
from typing import Any, Dict, List, Optional


# ============================================================
# ATR calculation — Wilder's RMA smoothing (matches Pine ta.atr)
# ============================================================
def _compute_atr(klines: List[Dict], period: int = 10) -> List[Optional[float]]:
    """Compute ATR(period) with Wilder's smoothing.
    
    Returns a list aligned with klines; first `period` entries are None
    (warmup). Pine's `ta.atr(10)` uses RMA (running moving average),
    equivalent to EMA with alpha=1/N. We implement the textbook version:
      - True Range[i] = max(high[i]-low[i], |high[i]-close[i-1]|, |low[i]-close[i-1]|)
      - ATR[period] = simple mean of TR[1..period]
      - ATR[i>period] = (ATR[i-1] * (period-1) + TR[i]) / period
    
    Pine often "seeds" ATR differently on first bars; this is close enough
    for OB-size filtering (which is what we use ATR for).
    """
    n = len(klines)
    atrs: List[Optional[float]] = [None] * n
    if n < period + 1:
        return atrs
    
    tr_values: List[Optional[float]] = [None] * n
    for i in range(1, n):
        h = float(klines[i].get('h', 0))
        l = float(klines[i].get('l', 0))
        prev_c = float(klines[i - 1].get('c', klines[i - 1].get('p', 0)))
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        tr_values[i] = tr
    
    # First ATR at index `period`: simple mean of TR[1..period]
    valid_initial = [t for t in tr_values[1:period + 1] if t is not None]
    if len(valid_initial) < period:
        return atrs
    atrs[period] = sum(valid_initial) / period
    
    # Wilder smoothing for subsequent bars
    for i in range(period + 1, n):
        if tr_values[i] is None:
            atrs[i] = atrs[i - 1]
            continue
        atrs[i] = (atrs[i - 1] * (period - 1) + tr_values[i]) / period
    
    return atrs


# ============================================================
# Main detection — one-pass state machine over klines
# ============================================================
def detect_volumized_obs(
    klines: List[Dict],
    swing_length: int = 10,
    ob_end_method: str = 'Wick',  # 'Wick' | 'Close'
    max_atr_mult: float = 3.5,
    zone_count: str = 'Low',  # 'One' | 'Low' | 'Medium' | 'High'
    combine_obs: bool = True,
    atr_period: int = 10,
) -> Dict[str, Any]:
    """Run Volumized OB detection on `klines`.
    
    Mirrors Pine `findOrderBlocks()` (lines 247-334) plus `findOBSwings()`
    (lines 229-245) and `combineOBsFunc()` (lines 372-406).
    
    Args mirror Pine input names:
      * swing_length      → Pine `swingLength`
      * ob_end_method     → Pine `obEndMethod`
      * max_atr_mult      → Pine `maxATRMult`
      * zone_count        → Pine `zoneCount` (controls trimming below)
      * combine_obs       → Pine `combineOBs`
      * atr_period        → Pine `ta.atr(10)` literal — kept as kwarg for tests
    
    Returns dict with bullish/bearish lists and a derived `trend`.
    """
    if not klines or len(klines) < swing_length + 3:
        return {
            'bullish_obs': [], 'bearish_obs': [],
            'all_bullish': [], 'all_bearish': [],
            'latest_ob': None, 'trend': None, 'trend_meta': {},
        }
    
    n = len(klines)
    
    # Pre-extract arrays for fast indexed access (avoids dict overhead in
    # the hot loop). Klines may use 'c' or 'p' for close — accept both.
    highs = [float(k.get('h', 0)) for k in klines]
    lows = [float(k.get('l', 0)) for k in klines]
    opens = [float(k.get('o', 0)) for k in klines]
    closes = [float(k.get('c', k.get('p', 0))) for k in klines]
    volumes = [float(k.get('v', 0)) for k in klines]
    times = [int(k.get('t', 0)) for k in klines]
    
    atrs = _compute_atr(klines, period=atr_period)
    
    # === State (Pine `var` equivalents) ===
    # swing_type: 0 = high swing active, 1 = low swing active
    # IMPORTANT: Pine initialises with `var swingType = 0` (line 230 of source).
    # That's NOT a "no swing detected yet" marker — it means "treat first
    # state as if a high swing already exists, only trigger on transitions".
    # The trigger condition `swingType == 0 and swingType[1] != 0` evaluates
    # false at startup (both 0), so the first top swing is registered only
    # AFTER a 1→0 transition (i.e., low swing happens first, then a high).
    # If we'd started from -1 (our previous bug), the first high swing would
    # fire immediately, producing a different OB sequence than Pine — the
    # latest_ob direction could then disagree with TradingView, which is
    # exactly what we saw on the NEARUSDT chart.
    swing_type = 0
    prev_swing_type = 0
    # Each swing point: {'x': bar_idx, 'y': price, 'volume': vol, 'crossed': bool}
    top_swing: Optional[Dict[str, Any]] = None
    bottom_swing: Optional[Dict[str, Any]] = None
    
    bullish_obs: List[Dict[str, Any]] = []
    bearish_obs: List[Dict[str, Any]] = []
    
    max_ob_count = 30  # Pine constant `maxOrderBlocks`
    
    # === Main loop ===
    # Bar `i` corresponds to Pine's "current bar". We need at least 2 bars
    # of lookback for volume[1]/volume[2], hence i starts at swing_length
    # (which is >= 3 by Pine's minval, so >= 3 lookback is satisfied).
    for i in range(swing_length, n):
        # ---- 1. Swing detection (findOBSwings) ----
        # Pine: upper = ta.highest(len), lower = ta.lowest(len) at current bar
        # i.e., max/min over the last `swing_length` bars (inclusive).
        upper = max(highs[i - swing_length + 1 : i + 1])
        lower = min(lows[i - swing_length + 1 : i + 1])
        
        prev_swing_type = swing_type
        # Pine: swingType := high[len] > upper ? 0 : low[len] < lower ? 1 : swingType
        # high[len] = highs[i - swing_length]
        cand_high = highs[i - swing_length]
        cand_low = lows[i - swing_length]
        if cand_high > upper:
            swing_type = 0
        elif cand_low < lower:
            swing_type = 1
        # else: swing_type unchanged
        
        # Pine: if swingType == 0 and swingType[1] != 0 → new top
        if swing_type == 0 and prev_swing_type != 0:
            top_swing = {
                'x': i - swing_length,
                'y': highs[i - swing_length],
                'volume': volumes[i - swing_length],
                'crossed': False,
            }
        # Pine: if swingType == 1 and swingType[1] != 1 → new bottom
        if swing_type == 1 and prev_swing_type != 1:
            bottom_swing = {
                'x': i - swing_length,
                'y': lows[i - swing_length],
                'volume': volumes[i - swing_length],
                'crossed': False,
            }
        
        # ---- 2. Bullish OB invalidation/removal ----
        # Pine iterates in reverse and modifies in-place. Python collects
        # indices to pop after the loop to avoid mid-iteration mutation.
        bull_idx_to_remove = []
        for ob_i in range(len(bullish_obs) - 1, -1, -1):
            cur_ob = bullish_obs[ob_i]
            if not cur_ob['breaker']:
                # Trigger: low (Wick) or min(open, close) (Close)
                if ob_end_method == 'Wick':
                    trigger = lows[i]
                else:
                    trigger = min(opens[i], closes[i])
                if trigger < cur_ob['bottom']:
                    cur_ob['breaker'] = True
                    cur_ob['break_time'] = times[i]
                    cur_ob['bb_volume'] = volumes[i]
            else:
                # Already a breaker — remove if price returns above
                if highs[i] > cur_ob['top']:
                    bull_idx_to_remove.append(ob_i)
        for idx in bull_idx_to_remove:
            bullish_obs.pop(idx)
        
        # ---- 3. Bullish OB formation ----
        # Trigger: close > top.y AND top hasn't been crossed yet
        if (top_swing is not None
                and closes[i] > top_swing['y']
                and not top_swing['crossed']):
            top_swing['crossed'] = True
            
            # Walk back from bar i-1 to top.x finding the LOWEST low.
            # That low is the OB bottom; the SAME bar's high is the OB top.
            box_btm = lows[i - 1]
            box_top = highs[i - 1]
            box_loc = times[i - 1]
            
            distance = (i - top_swing['x']) - 1
            for j in range(1, distance + 1):
                src_idx = i - j
                if src_idx < 0:
                    break
                if lows[src_idx] < box_btm:
                    box_btm = lows[src_idx]
                    box_top = highs[src_idx]
                    box_loc = times[src_idx]
            
            ob_volume = volumes[i] + volumes[i - 1] + volumes[i - 2]
            ob_low_volume = volumes[i - 2]
            ob_high_volume = volumes[i] + volumes[i - 1]
            
            ob_size = abs(box_top - box_btm)
            current_atr = atrs[i] if atrs[i] is not None else 0
            
            # Filter: OB box must not exceed `max_atr_mult` × ATR. This
            # rejects oversized OBs spanning ranges that exceed normal
            # volatility — usually crash bars where the "OB" is actually
            # the whole impulse, not a real institutional zone.
            if current_atr > 0 and ob_size <= current_atr * max_atr_mult:
                new_ob = {
                    'type': 'Bull',
                    'top': box_top,
                    'bottom': box_btm,
                    'start_time': box_loc,
                    'start_idx': i,
                    'ob_volume': ob_volume,
                    'ob_low_volume': ob_low_volume,
                    'ob_high_volume': ob_high_volume,
                    'breaker': False,
                    'break_time': None,
                    'bb_volume': 0.0,
                    'combined': False,
                }
                bullish_obs.insert(0, new_ob)
                if len(bullish_obs) > max_ob_count:
                    bullish_obs.pop()
        
        # ---- 4. Bearish OB invalidation/removal (symmetric) ----
        bear_idx_to_remove = []
        for ob_i in range(len(bearish_obs) - 1, -1, -1):
            cur_ob = bearish_obs[ob_i]
            if not cur_ob['breaker']:
                if ob_end_method == 'Wick':
                    trigger = highs[i]
                else:
                    trigger = max(opens[i], closes[i])
                if trigger > cur_ob['top']:
                    cur_ob['breaker'] = True
                    cur_ob['break_time'] = times[i]
                    cur_ob['bb_volume'] = volumes[i]
            else:
                if lows[i] < cur_ob['bottom']:
                    bear_idx_to_remove.append(ob_i)
        for idx in bear_idx_to_remove:
            bearish_obs.pop(idx)
        
        # ---- 5. Bearish OB formation (symmetric) ----
        if (bottom_swing is not None
                and closes[i] < bottom_swing['y']
                and not bottom_swing['crossed']):
            bottom_swing['crossed'] = True
            
            # Walk back finding the HIGHEST high. That high = OB top;
            # the same bar's low = OB bottom.
            box_btm = lows[i - 1]
            box_top = highs[i - 1]
            box_loc = times[i - 1]
            
            distance = (i - bottom_swing['x']) - 1
            for j in range(1, distance + 1):
                src_idx = i - j
                if src_idx < 0:
                    break
                if highs[src_idx] > box_top:
                    box_top = highs[src_idx]
                    box_btm = lows[src_idx]
                    box_loc = times[src_idx]
            
            ob_volume = volumes[i] + volumes[i - 1] + volumes[i - 2]
            # Note Pine's flip for bearish: low/high volume mapping inverted
            ob_low_volume = volumes[i] + volumes[i - 1]
            ob_high_volume = volumes[i - 2]
            
            ob_size = abs(box_top - box_btm)
            current_atr = atrs[i] if atrs[i] is not None else 0
            
            if current_atr > 0 and ob_size <= current_atr * max_atr_mult:
                new_ob = {
                    'type': 'Bear',
                    'top': box_top,
                    'bottom': box_btm,
                    'start_time': box_loc,
                    'start_idx': i,
                    'ob_volume': ob_volume,
                    'ob_low_volume': ob_low_volume,
                    'ob_high_volume': ob_high_volume,
                    'breaker': False,
                    'break_time': None,
                    'bb_volume': 0.0,
                    'combined': False,
                }
                bearish_obs.insert(0, new_ob)
                if len(bearish_obs) > max_ob_count:
                    bearish_obs.pop()
    
    # === Post-processing: combine overlapping OBs of the same type ===
    if combine_obs:
        bullish_obs = _combine_obs_func(bullish_obs)
        bearish_obs = _combine_obs_func(bearish_obs)
    
    # Trim to visible count per zone_count setting
    zone_count_map = {'One': 1, 'Low': 3, 'Medium': 5, 'High': 10}
    visible_count = zone_count_map.get(zone_count, 3)
    visible_bull = bullish_obs[:visible_count]
    visible_bear = bearish_obs[:visible_count]
    
    # === Latest OB across both lists (most recent start_time) ===
    # User's principle: "trend = direction of the latest formed OB".
    # We consider ALL OBs (including breakers) because Pine doesn't filter
    # by breaker state when determining the most recent impulse.
    all_visible = visible_bull + visible_bear
    latest_ob: Optional[Dict[str, Any]] = None
    if all_visible:
        latest_ob = max(all_visible, key=lambda ob: ob['start_time'])
    
    trend: Optional[str] = None
    trend_meta: Dict[str, Any] = {}
    if latest_ob is not None:
        trend = 'LONG' if latest_ob['type'] == 'Bull' else 'SHORT'
        trend_meta = {
            'ob_type': latest_ob['type'],
            'top': latest_ob['top'],
            'bottom': latest_ob['bottom'],
            'start_time': latest_ob['start_time'],
            'ob_volume': latest_ob['ob_volume'],
            'breaker': latest_ob['breaker'],
            'combined': latest_ob.get('combined', False),
        }
    
    return {
        'bullish_obs': visible_bull,
        'bearish_obs': visible_bear,
        'all_bullish': bullish_obs,
        'all_bearish': bearish_obs,
        'latest_ob': latest_ob,
        'trend': trend,
        'trend_meta': trend_meta,
    }


# ============================================================
# OB merging — Pine `combineOBsFunc()`
# ============================================================
def _obs_overlap(ob1: Dict, ob2: Dict) -> bool:
    """Check if two same-type OBs overlap in BOTH price and time.
    
    Pine uses an area-based overlap percentage with a threshold of 0
    (line 8: `overlapThresholdPercentage = 0`), which effectively means
    "any overlap at all". So we use a strict overlap check.
    """
    # Y-axis (price) overlap
    y_overlap = (ob1['top'] > ob2['bottom']) and (ob2['top'] > ob1['bottom'])
    # X-axis (time) overlap. Active OBs (no break_time) extend "forever";
    # use a large sentinel so two active OBs of same type with overlapping
    # price ranges always merge.
    SENTINEL = 9999999999999  # year ~2286 in ms
    end1 = ob1.get('break_time') or SENTINEL
    end2 = ob2.get('break_time') or SENTINEL
    x_overlap = (ob1['start_time'] < end2) and (ob2['start_time'] < end1)
    return y_overlap and x_overlap


def _combine_obs_func(obs: List[Dict]) -> List[Dict]:
    """Merge overlapping OBs of the same type until no overlaps remain.
    
    Pine's loop: `while lastCombinations > 0` — repeats until pass
    finds no more merges. Python equivalent uses a flag.
    Merged OB takes:
      * top    = max(both tops)
      * bottom = min(both bottoms)
      * start_time = min(both starts)
      * break_time = max(both break_times) or None if both active
      * volume fields = sum
      * breaker = OR
      * combined = True (badge for UI)
    """
    if len(obs) < 2:
        return obs
    
    # Keep merging until a full pass produces no changes
    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(obs)):
            if i >= len(obs):
                break
            for j in range(i + 1, len(obs)):
                if j >= len(obs):
                    break
                if obs[i]['type'] != obs[j]['type']:
                    continue
                if not _obs_overlap(obs[i], obs[j]):
                    continue
                ob1 = obs[i]
                ob2 = obs[j]
                bt1 = ob1.get('break_time')
                bt2 = ob2.get('break_time')
                if bt1 is None and bt2 is None:
                    new_break = None
                else:
                    new_break = max(bt1 or 0, bt2 or 0) or None
                merged = {
                    'type': ob1['type'],
                    'top': max(ob1['top'], ob2['top']),
                    'bottom': min(ob1['bottom'], ob2['bottom']),
                    'start_time': min(ob1['start_time'], ob2['start_time']),
                    'start_idx': min(ob1.get('start_idx', 0), ob2.get('start_idx', 0)),
                    'ob_volume': ob1['ob_volume'] + ob2['ob_volume'],
                    'ob_low_volume': ob1.get('ob_low_volume', 0) + ob2.get('ob_low_volume', 0),
                    'ob_high_volume': ob1.get('ob_high_volume', 0) + ob2.get('ob_high_volume', 0),
                    'breaker': ob1['breaker'] or ob2['breaker'],
                    'break_time': new_break,
                    'bb_volume': ob1.get('bb_volume', 0) + ob2.get('bb_volume', 0),
                    'combined': True,
                }
                # Remove both originals, insert merged at front
                # (Pine's `unshift` semantics — newest visible first).
                obs = [o for k, o in enumerate(obs) if k != i and k != j]
                obs.insert(0, merged)
                merged_any = True
                break
            if merged_any:
                break
    
    return obs


# ============================================================
# Convenience wrapper — just the trend
# ============================================================
def get_latest_ob_trend(klines: List[Dict], **settings) -> Dict[str, Any]:
    """Convenience wrapper: run detect_volumized_obs and return only the
    trend fields. Callers that don't need the full OB lists save a few
    dict allocations and serialization bytes.
    
    Returns: {'trend': 'LONG'|'SHORT'|None, 'trend_meta': {...}}.
    """
    result = detect_volumized_obs(klines, **settings)
    return {
        'trend': result['trend'],
        'trend_meta': result['trend_meta'],
        'latest_ob': result['latest_ob'],
    }
