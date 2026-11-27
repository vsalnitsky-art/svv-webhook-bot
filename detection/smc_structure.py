"""
Smart Money Concepts Engine — direct port of LuxAlgo SMC Pine Script.

Pine reference: SMC_PRO_BOT__47_.pine (lines 463-738)

Algorithm — matches Pine main loop exactly:

  For each bar i, in this exact order:
    1. SNAPSHOT pivot levels as they stood at end of bar i-1 (used for
       Pine `level[1]` history semantics in ta.crossover/crossunder).
    
    2. SWING leg detection (Pine: getCurrentStructure(swing_size) FIRST):
       - candidate = bar i-swing_size
       - newLegHigh = candidate.high > max(high in [i-swing_size+1, i])
       - newLegLow  = candidate.low  < min(low  in [i-swing_size+1, i])
       - On leg transition → set new swing pivot at candidate, crossed=false
    
    3. INTERNAL leg detection (Pine: getCurrentStructure(internal_size)):
       - Same as above but with internal_size
       - On leg transition → set new internal pivot
    
    4. INTERNAL events check (Pine: displayStructure(true) FIRST):
       - Bullish: ta.crossover(close, internalHigh.level)
                  AND not internalHigh.crossed
                  AND internalHigh.level != swingHigh.level   ← extraCondition
       - Bearish: ta.crossunder(close, internalLow.level)
                  AND not internalLow.crossed
                  AND internalLow.level != swingLow.level     ← extraCondition
       - Tag: CHoCH if trend was opposite, else BOS
       - Update internal trend bias on event
    
    5. SWING events check (Pine: displayStructure() AFTER):
       - Bullish: ta.crossover(close, swingHigh.level)
                  AND not swingHigh.crossed
                  (no extraCondition for swing — extraCondition=true for non-internal)
       - Bearish: ta.crossunder(close, swingLow.level)
                  AND not swingLow.crossed
       - Tag and trend update analogous

Key Pine semantics this implementation preserves:

  • `var leg = 0` initialization (BEARISH_LEG). The very first newLegHigh
    after start does NOT fire a pivot (leg already 0); the first newLegLow
    DOES fire (0→1 transition). This asymmetric quirk is matched exactly
    to avoid 1-2 pivot offset on freshly-opened symbols.
  
  • `ta.crossover(close, level)` semantics: fires when close[0]>level[0]
    AND close[1]<=level[1]. The `[1]` version uses the level as it was at
    end of prior bar — when a pivot updates on bar i, level[0] is NEW but
    level[1] is OLD. We snapshot at start-of-bar to preserve this.
  
  • `extraCondition` for internal events: when internalHigh/Low level
    equals swingHigh/Low level, suppress the internal event. This filter
    fires often in clear trends where the swing peak is dominant enough
    that internal pivots can't form between it and now. Without this
    filter, internal CHoCH/BOS events double-up with swing events,
    producing extra alerts that don't appear on TradingView.
  
  • Pine `var trend = trend.new(0)` — initial trend bias = 0. First event
    is always BOS (not CHoCH) because the conditional `bias == BEARISH ?
    CHoCH : BOS` evaluates to BOS when bias is the initial 0.
  
  • Pivot type classification (HH/HL/LH/LL) compares newPivot.level vs
    pivot.lastLevel (= old level before this update). The `var` semantics
    of Pine UDTs persist this state across bars.

Confluence filter (Pine `internalFilterConfluenceInput`) is NOT
implemented — it defaults OFF in Pine and we don't expose it in our UI.
If a future need arises, the candle-structure check would gate `bullishBar`
and `bearishBar` per Pine lines 681-683, which then AND into
extraCondition for internal events.
"""

from typing import Dict, List, Optional


# Pine constants (lines 9-13 of SMC_PRO_BOT__47_.pine)
BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = 1
BEARISH = -1
CHOCH = "CHoCH"
BOS = "BOS"


def detect_smc_structure(klines: List[Dict], internal_size: int = 5,
                          swing_size: int = 50) -> Dict:
    """Detect SMC structure on klines using a combined Pine-faithful pass.
    
    Args:
        klines: list of bar dicts with 'h', 'l', 'p' (close), 't' (ms epoch).
        internal_size: pivot lookback for internal structure (Pine default 5)
        swing_size: pivot lookback for swing structure (Pine default 50)
    
    Returns:
        {
            'internal': {pivots, events, trend, last_bos, last_choch},
            'swing':    {pivots, events, trend, last_bos, last_choch},
            'klines_count': N,
        }
    """
    if not klines or len(klines) < internal_size + 2:
        return _empty_result(klines)
    
    return _process_combined(klines, internal_size, swing_size)


def _empty_result(klines):
    return {
        'internal': _empty_struct(),
        'swing': _empty_struct(),
        'klines_count': len(klines) if klines else 0,
    }


def _empty_struct():
    return {
        'pivots': [],
        'events': [],
        'trend': 0,
        'last_bos': None,
        'last_choch': None,
    }


def _new_pivot_state():
    """Pine `var pivot pivotName = pivot.new(...)` — initial state with all
    fields uninitialized (na in Pine, None here)."""
    return {'level': None, 'last_level': None, 'crossed': False,
            't': None, 'idx': None}


def _process_combined(klines: List[Dict], internal_size: int,
                       swing_size: int) -> Dict:
    """Single-pass SMC detection running internal and swing pivots in
    parallel — matches Pine's `displayStructure(true)` then
    `displayStructure()` ordering with cross-pivot extraCondition.
    """
    n = len(klines)
    
    # Pine: var leg = 0 (BEARISH_LEG) for both internal and swing
    internal_leg = BEARISH_LEG
    swing_leg = BEARISH_LEG
    
    # Pine: var pivot internalHigh = pivot.new(...) etc
    internal_high = _new_pivot_state()
    internal_low  = _new_pivot_state()
    swing_high    = _new_pivot_state()
    swing_low     = _new_pivot_state()
    
    # Pine: var trend internalTrend = trend.new(0); var trend swingTrend
    internal_trend = 0
    swing_trend = 0
    
    internal_pivots: List[Dict] = []
    swing_pivots: List[Dict] = []
    internal_events: List[Dict] = []
    swing_events: List[Dict] = []
    
    # Pre-extract h/l/p arrays once — repeated dict.get in inner loops is
    # measurably slower on large series (700+ bars × 50-bar windows).
    highs = [float(k.get('h', k.get('p', 0))) for k in klines]
    lows  = [float(k.get('l', k.get('p', 0))) for k in klines]
    closes = [float(k.get('p', 0)) for k in klines]
    times = [k.get('t', 0) for k in klines]
    
    # Iterate from bar 0 — Pine evaluates every bar but leg() returns NA-safe
    # values when high[size] is out of bounds. Equivalent: skip leg detection
    # for bars i < size, but still allow event-check from bar 1 onward (in
    # case a pivot somehow exists earlier — it can't, but cheap to be safe).
    for i in range(n):
        # ┌─────────────────────────────────────────────────────────────┐
        # │ STEP 1: Snapshot pivot levels at start of bar (for [1])    │
        # │ Pine ta.crossover(close, level) ≡ close[0]>level[0] AND     │
        # │                                    close[1]<=level[1]       │
        # │ The `[1]` indexing returns the variable's value at END of   │
        # │ prior bar = START of this bar (state hasn't been touched).  │
        # └─────────────────────────────────────────────────────────────┘
        prev_internal_high_level = internal_high['level']
        prev_internal_low_level  = internal_low['level']
        prev_swing_high_level    = swing_high['level']
        prev_swing_low_level     = swing_low['level']
        
        close_now = closes[i]
        close_prev = closes[i - 1] if i >= 1 else None
        bar_t = times[i]
        
        # ┌─────────────────────────────────────────────────────────────┐
        # │ STEP 2: SWING leg/pivot update (Pine getCurrentStructure 1) │
        # │ Pine processes swing FIRST, internal SECOND. Ordering       │
        # │ matters because the extraCondition in step 4 reads          │
        # │ swingHigh/Low.currentLevel which must reflect THIS bar's    │
        # │ swing update.                                                │
        # └─────────────────────────────────────────────────────────────┘
        if i >= swing_size:
            cand_idx = i - swing_size
            c_high = highs[cand_idx]
            c_low  = lows[cand_idx]
            # Pine: ta.highest(size) on bar i = max(high in last `size` bars
            # INCLUDING bar i) = max(highs[i-size+1 .. i]). Same for ta.lowest.
            max_h = max(highs[j] for j in range(i - swing_size + 1, i + 1))
            min_l = min(lows[j]  for j in range(i - swing_size + 1, i + 1))
            new_leg_high = c_high > max_h
            new_leg_low  = c_low  < min_l
            
            prev_swing_leg = swing_leg
            if new_leg_high:
                swing_leg = BEARISH_LEG
            elif new_leg_low:
                swing_leg = BULLISH_LEG
            
            if swing_leg != prev_swing_leg:
                if swing_leg == BULLISH_LEG:
                    swing_low['last_level'] = swing_low['level']
                    swing_low['level']      = c_low
                    swing_low['crossed']    = False
                    swing_low['t']          = times[cand_idx]
                    swing_low['idx']        = cand_idx
                    pt = ('HL' if (swing_low['last_level'] is not None
                                   and c_low > swing_low['last_level']) else 'LL')
                    swing_pivots.append({
                        't': swing_low['t'], 'idx': cand_idx,
                        'price': c_low, 'type': pt,
                    })
                else:  # BEARISH_LEG (high pivot)
                    swing_high['last_level'] = swing_high['level']
                    swing_high['level']      = c_high
                    swing_high['crossed']    = False
                    swing_high['t']          = times[cand_idx]
                    swing_high['idx']        = cand_idx
                    pt = ('HH' if (swing_high['last_level'] is not None
                                   and c_high > swing_high['last_level']) else 'LH')
                    swing_pivots.append({
                        't': swing_high['t'], 'idx': cand_idx,
                        'price': c_high, 'type': pt,
                    })
        
        # ┌─────────────────────────────────────────────────────────────┐
        # │ STEP 3: INTERNAL leg/pivot update                            │
        # └─────────────────────────────────────────────────────────────┘
        if i >= internal_size:
            cand_idx = i - internal_size
            c_high = highs[cand_idx]
            c_low  = lows[cand_idx]
            max_h = max(highs[j] for j in range(i - internal_size + 1, i + 1))
            min_l = min(lows[j]  for j in range(i - internal_size + 1, i + 1))
            new_leg_high = c_high > max_h
            new_leg_low  = c_low  < min_l
            
            prev_internal_leg = internal_leg
            if new_leg_high:
                internal_leg = BEARISH_LEG
            elif new_leg_low:
                internal_leg = BULLISH_LEG
            
            if internal_leg != prev_internal_leg:
                if internal_leg == BULLISH_LEG:
                    internal_low['last_level'] = internal_low['level']
                    internal_low['level']      = c_low
                    internal_low['crossed']    = False
                    internal_low['t']          = times[cand_idx]
                    internal_low['idx']        = cand_idx
                    pt = ('HL' if (internal_low['last_level'] is not None
                                   and c_low > internal_low['last_level']) else 'LL')
                    internal_pivots.append({
                        't': internal_low['t'], 'idx': cand_idx,
                        'price': c_low, 'type': pt,
                    })
                else:  # BEARISH_LEG
                    internal_high['last_level'] = internal_high['level']
                    internal_high['level']      = c_high
                    internal_high['crossed']    = False
                    internal_high['t']          = times[cand_idx]
                    internal_high['idx']        = cand_idx
                    pt = ('HH' if (internal_high['last_level'] is not None
                                   and c_high > internal_high['last_level']) else 'LH')
                    internal_pivots.append({
                        't': internal_high['t'], 'idx': cand_idx,
                        'price': c_high, 'type': pt,
                    })
        
        if i < 1:
            continue  # Need close_prev for crossover; bar 0 has none
        
        # ┌─────────────────────────────────────────────────────────────┐
        # │ STEP 4: INTERNAL events (Pine: displayStructure(true))      │
        # │ Bullish: crossover with extraCondition (level != swing.lvl) │
        # │ Bearish: crossunder with extraCondition                      │
        # │ Crossing flips internal_trend; tag = CHoCH if trend was      │
        # │ opposite, else BOS.                                          │
        # └─────────────────────────────────────────────────────────────┘
        # Bullish internal
        if (internal_high['level'] is not None
                and not internal_high['crossed']
                and prev_internal_high_level is not None
                and close_now > internal_high['level']
                and close_prev <= prev_internal_high_level
                # extraCondition: internalHigh.level != swingHigh.level.
                # If swing_high.level is None (no swing pivot yet) we treat
                # the inequality as TRUE — Pine `na != value` evaluates
                # to true (na compares unequal to any concrete value).
                and (swing_high['level'] is None
                     or internal_high['level'] != swing_high['level'])):
            tag = CHOCH if internal_trend == BEARISH else BOS
            internal_events.append({
                'from_t': internal_high['t'],
                'from_idx': internal_high['idx'],
                'to_t': bar_t, 'to_idx': i,
                'level': internal_high['level'],
                'tag': tag, 'dir': 'bull',
            })
            internal_high['crossed'] = True
            internal_trend = BULLISH
        
        # Bearish internal
        if (internal_low['level'] is not None
                and not internal_low['crossed']
                and prev_internal_low_level is not None
                and close_now < internal_low['level']
                and close_prev >= prev_internal_low_level
                and (swing_low['level'] is None
                     or internal_low['level'] != swing_low['level'])):
            tag = CHOCH if internal_trend == BULLISH else BOS
            internal_events.append({
                'from_t': internal_low['t'],
                'from_idx': internal_low['idx'],
                'to_t': bar_t, 'to_idx': i,
                'level': internal_low['level'],
                'tag': tag, 'dir': 'bear',
            })
            internal_low['crossed'] = True
            internal_trend = BEARISH
        
        # ┌─────────────────────────────────────────────────────────────┐
        # │ STEP 5: SWING events (Pine: displayStructure())              │
        # │ extraCondition is `true` for non-internal — no level filter. │
        # └─────────────────────────────────────────────────────────────┘
        # Bullish swing
        if (swing_high['level'] is not None
                and not swing_high['crossed']
                and prev_swing_high_level is not None
                and close_now > swing_high['level']
                and close_prev <= prev_swing_high_level):
            tag = CHOCH if swing_trend == BEARISH else BOS
            swing_events.append({
                'from_t': swing_high['t'],
                'from_idx': swing_high['idx'],
                'to_t': bar_t, 'to_idx': i,
                'level': swing_high['level'],
                'tag': tag, 'dir': 'bull',
            })
            swing_high['crossed'] = True
            swing_trend = BULLISH
        
        # Bearish swing
        if (swing_low['level'] is not None
                and not swing_low['crossed']
                and prev_swing_low_level is not None
                and close_now < swing_low['level']
                and close_prev >= prev_swing_low_level):
            tag = CHOCH if swing_trend == BULLISH else BOS
            swing_events.append({
                'from_t': swing_low['t'],
                'from_idx': swing_low['idx'],
                'to_t': bar_t, 'to_idx': i,
                'level': swing_low['level'],
                'tag': tag, 'dir': 'bear',
            })
            swing_low['crossed'] = True
            swing_trend = BEARISH
    
    # Last events for downstream consumers (alerts, UI status badges).
    # `next` with default None gives O(1) on tail-most match in reverse iter.
    last_internal_bos   = next((e for e in reversed(internal_events) if e['tag'] == BOS), None)
    last_internal_choch = next((e for e in reversed(internal_events) if e['tag'] == CHOCH), None)
    last_swing_bos      = next((e for e in reversed(swing_events)    if e['tag'] == BOS), None)
    last_swing_choch    = next((e for e in reversed(swing_events)    if e['tag'] == CHOCH), None)
    
    return {
        'internal': {
            'pivots': internal_pivots,
            'events': internal_events,
            'trend': internal_trend,
            'last_bos': last_internal_bos,
            'last_choch': last_internal_choch,
        },
        'swing': {
            'pivots': swing_pivots,
            'events': swing_events,
            'trend': swing_trend,
            'last_bos': last_swing_bos,
            'last_choch': last_swing_choch,
        },
        'klines_count': n,
    }
