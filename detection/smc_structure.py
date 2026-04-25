"""
Smart Money Concepts Engine v1.0 — 100% replication of LuxAlgo SMC Pine Script logic.

Implements:
  - leg(size) — bearish/bullish leg detection
  - getCurrentStructure() — pivot point storage (HH, HL, LH, LL)
  - displayStructure() — BOS / CHoCH detection on close crossover/crossunder

Pine reference: SMC_PRO_BOT__47_.pine, lines 463-738
  Internal Structure: size = 5 (fixed in Pine)
  Swing Structure:    size = 50 (default, configurable)

This module currently implements ONLY Internal Structure (per user request).
Swing Structure scaffolding is present but disabled.

Constants match Pine exactly:
  BULLISH_LEG = 1, BEARISH_LEG = 0
  BULLISH = 1, BEARISH = -1
  CHOCH = "CHoCH", BOS = "BOS"
"""

from typing import Dict, List, Optional, Tuple


# Pine constants
BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = 1
BEARISH = -1
CHOCH = "CHoCH"
BOS = "BOS"


def detect_smc_structure(klines: List[Dict], internal_size: int = 5,
                          swing_size: int = 50) -> Dict:
    """Run full SMC structure detection on a klines array.
    
    Args:
        klines: list of {t, p, h, l, v, ...} where:
            t = open time (ms or sec)
            p = close
            h = high
            l = low
        internal_size: pivot lookback for Internal Structure (Pine default 5)
        swing_size: pivot lookback for Swing Structure (Pine default 50)
    
    Returns:
        {
            'internal': {
                'pivots': [{t, idx, price, type}, ...]   # type ∈ HH, HL, LH, LL
                'events': [{from_t, from_idx, to_t, to_idx, level, tag, dir}, ...]
                                                         # tag ∈ BOS, CHoCH
                                                         # dir ∈ bull, bear
                'trend':   1 | -1 | 0  (final state)
                'last_bos':   {...} or None
                'last_choch': {...} or None
            },
            'swing': { ... }   # SAME structure (computed but not used yet)
            'klines_count': N,
            'symbol': '',
            'interval': '',
        }
    
    Crucial Pine semantics this implements verbatim:
      - leg(size) returns BEARISH_LEG when high[size] is the highest of the
        previous (size+1) bars including itself; BULLISH_LEG analogously.
        Note: Pine's `ta.highest(size)` excludes current bar offset, so we use
        the [size]-back bar as candidate and compare to bars [0..size-1].
      - On `startOfNewLeg`, the pivot at offset `size` becomes the new pivot.
      - BOS/CHoCH triggers on `close > pivot.level` (crossover) ONLY ONCE per
        pivot (`pivot.crossed` flag), with tag = CHoCH if trend was opposite,
        else BOS. Trend then flips to the new direction.
    """
    if not klines or len(klines) < internal_size + 2:
        return _empty_result(klines)
    
    # Pine processes Internal and Swing in parallel because Internal events
    # are filtered against current Swing levels (line 691 of Pine):
    #   extraCondition = internalHigh.currentLevel != swingHigh.currentLevel
    #                    and bullishBar
    # This is why we run them together rather than two independent passes.
    has_swing = len(klines) >= swing_size + 2
    internal, swing = _process_structures_parallel(
        klines, internal_size, swing_size if has_swing else None
    )
    
    return {
        'internal': internal,
        'swing': swing if has_swing else _empty_struct(),
        'klines_count': len(klines),
    }


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


def _process_structures_parallel(klines: List[Dict], internal_size: int,
                                  swing_size):
    """Run Internal and Swing structure detection bar-by-bar in lockstep.
    
    This is required to match Pine: Internal BOS/CHoCH events are filtered
    against the CURRENT Swing pivot level (extraCondition in Pine line 691),
    plus a bullishBar/bearishBar filter on the crossover bar.
    
    Without this filter, our algorithm produces extra Internal events at the
    same price levels as Swing pivots (e.g. the spurious CHoCH bull + BOS bear
    seen between the genuine TV events).
    """
    n = len(klines)
    
    # Per-structure state
    state_int = _new_struct_state()
    state_sw = _new_struct_state() if swing_size else None
    
    pivots_int = []
    events_int = []
    pivots_sw = []
    events_sw = []
    
    for i in range(max(internal_size, swing_size or 0), n):
        bar = klines[i]
        
        # === Update Swing state first (so Internal can reference current Swing levels) ===
        if swing_size and i >= swing_size:
            _update_leg_and_pivots(klines, i, swing_size, state_sw, pivots_sw)
            _check_crossover_events(klines, i, state_sw, events_sw,
                                     extra_filter=False, bar_char_filter=False,
                                     swing_state=None)
        
        # === Then update Internal ===
        if i >= internal_size:
            _update_leg_and_pivots(klines, i, internal_size, state_int, pivots_int)
            # Internal applies extra filters: dedup vs swing level + bar character
            _check_crossover_events(klines, i, state_int, events_int,
                                     extra_filter=True, bar_char_filter=True,
                                     swing_state=state_sw)
    
    internal = {
        'pivots': pivots_int,
        'events': events_int,
        'trend': state_int['trend_bias'],
        'last_bos': _last_with_tag(events_int, 'BOS'),
        'last_choch': _last_with_tag(events_int, 'CHoCH'),
    }
    
    swing = None
    if state_sw is not None:
        swing = {
            'pivots': pivots_sw,
            'events': events_sw,
            'trend': state_sw['trend_bias'],
            'last_bos': _last_with_tag(events_sw, 'BOS'),
            'last_choch': _last_with_tag(events_sw, 'CHoCH'),
        }
    
    return internal, swing


def _new_struct_state():
    return {
        'leg_state': None,
        'high_pivot': {'level': None, 'last_level': None, 'crossed': False,
                        't': None, 'idx': None},
        'low_pivot': {'level': None, 'last_level': None, 'crossed': False,
                       't': None, 'idx': None},
        'trend_bias': 0,
    }


def _update_leg_and_pivots(klines, i, size, state, pivots_out):
    """Detect new leg/pivot at bar i with given size."""
    candidate = klines[i - size]
    c_high = candidate.get('h', candidate['p'])
    c_low = candidate.get('l', candidate['p'])
    
    # Pine: ta.highest(size) on bar i = max of high[i-size+1 .. i]
    window_highs = [klines[j].get('h', klines[j]['p']) for j in range(i - size + 1, i + 1)]
    window_lows = [klines[j].get('l', klines[j]['p']) for j in range(i - size + 1, i + 1)]
    max_h = max(window_highs) if window_highs else c_high
    min_l = min(window_lows) if window_lows else c_low
    
    new_leg_high = c_high > max_h
    new_leg_low = c_low < min_l
    
    prev_leg = state['leg_state']
    if new_leg_high:
        state['leg_state'] = BEARISH_LEG
    elif new_leg_low:
        state['leg_state'] = BULLISH_LEG
    
    # Pivot forms on any leg state transition (including from None)
    new_pivot = (prev_leg is not None and 
                 state['leg_state'] is not None and 
                 state['leg_state'] != prev_leg)
    
    if not new_pivot:
        return
    
    if state['leg_state'] == BULLISH_LEG:
        lp = state['low_pivot']
        lp['last_level'] = lp['level']
        lp['level'] = c_low
        lp['crossed'] = False
        lp['t'] = candidate.get('t')
        lp['idx'] = i - size
        pt = 'HL' if (lp['last_level'] is not None and c_low > lp['last_level']) else 'LL'
        pivots_out.append({
            't': lp['t'], 'idx': lp['idx'], 'price': c_low, 'type': pt,
        })
    elif state['leg_state'] == BEARISH_LEG:
        hp = state['high_pivot']
        hp['last_level'] = hp['level']
        hp['level'] = c_high
        hp['crossed'] = False
        hp['t'] = candidate.get('t')
        hp['idx'] = i - size
        pt = 'HH' if (hp['last_level'] is not None and c_high > hp['last_level']) else 'LH'
        pivots_out.append({
            't': hp['t'], 'idx': hp['idx'], 'price': c_high, 'type': pt,
        })


def _check_crossover_events(klines, i, state, events_out,
                              extra_filter=False, bar_char_filter=False,
                              swing_state=None):
    """Check for BOS/CHoCH crossovers at bar i."""
    if i < 1:
        return
    bar = klines[i]
    close_now = bar['p']
    close_prev = klines[i - 1]['p']
    
    # Bar character (Pine: bullishBar/bearishBar with internalFilterConfluenceInput=true)
    # bullishBar: top wick smaller than bottom wick → buyer-dominated
    # bearishBar: top wick larger than bottom wick → seller-dominated
    h = bar.get('h', close_now)
    l = bar.get('l', close_now)
    o = bar.get('o', close_now)
    top_wick = h - max(close_now, o)
    bottom_wick = min(close_now, o) - l
    bullish_bar = top_wick < bottom_wick
    bearish_bar = top_wick > bottom_wick
    
    hp = state['high_pivot']
    lp = state['low_pivot']
    
    # Bullish crossover
    if hp['level'] is not None and not hp['crossed']:
        if close_now > hp['level'] and close_prev <= hp['level']:
            # Internal filter (Pine line 691):
            # extraCondition = internalHigh.currentLevel != swingHigh.currentLevel
            #                  AND bullishBar
            ok = True
            if extra_filter and swing_state is not None:
                sw_h = swing_state['high_pivot']['level']
                if sw_h is not None and abs(hp['level'] - sw_h) < 1e-12:
                    ok = False
            if bar_char_filter and not bullish_bar:
                ok = False
            
            if ok:
                tag = CHOCH if state['trend_bias'] == BEARISH else BOS
                ev = {
                    'from_t': hp['t'], 'from_idx': hp['idx'],
                    'to_t': bar.get('t'), 'to_idx': i,
                    'level': hp['level'], 'tag': tag, 'dir': 'bull',
                }
                events_out.append(ev)
                hp['crossed'] = True
                state['trend_bias'] = BULLISH
            else:
                # Mark crossed even when filtered, so we don't re-trigger
                hp['crossed'] = True
                # Trend bias still flips per Pine (line 705 unconditional)
                state['trend_bias'] = BULLISH
    
    # Bearish crossover
    if lp['level'] is not None and not lp['crossed']:
        if close_now < lp['level'] and close_prev >= lp['level']:
            ok = True
            if extra_filter and swing_state is not None:
                sw_l = swing_state['low_pivot']['level']
                if sw_l is not None and abs(lp['level'] - sw_l) < 1e-12:
                    ok = False
            if bar_char_filter and not bearish_bar:
                ok = False
            
            if ok:
                tag = CHOCH if state['trend_bias'] == BULLISH else BOS
                ev = {
                    'from_t': lp['t'], 'from_idx': lp['idx'],
                    'to_t': bar.get('t'), 'to_idx': i,
                    'level': lp['level'], 'tag': tag, 'dir': 'bear',
                }
                events_out.append(ev)
                lp['crossed'] = True
                state['trend_bias'] = BEARISH
            else:
                lp['crossed'] = True
                state['trend_bias'] = BEARISH


def _last_with_tag(events, tag):
    for e in reversed(events):
        if e.get('tag') == tag:
            return e
    return None


def _process_structure(klines: List[Dict], size: int, label: str = '') -> Dict:
    """Walk klines and detect structure.
    
    Mirrors Pine getCurrentStructure() + displayStructure() called bar-by-bar.
    
    IMPORTANT: Pine uses BULLISH_LEG=1 and BEARISH_LEG=0 (where 0 is a VALID
    state, not "uninitialized"). To match Pine semantics exactly, we use
    `leg_state = None` to mean "uninitialized" so we don't accidentally treat
    BEARISH_LEG transitions as "first time".
    """
    n = len(klines)
    
    # leg state — None means uninitialized (Pine uses var leg = 0 but that's
    # also BEARISH_LEG; we keep them distinct here)
    leg_state = None
    prev_leg_state = None
    
    # pivot storage (current = most recent confirmed pivot)
    high_pivot = {'level': None, 'last_level': None, 'crossed': False,
                   't': None, 'idx': None}
    low_pivot = {'level': None, 'last_level': None, 'crossed': False,
                  't': None, 'idx': None}
    
    trend_bias = 0  # 0 / +1 BULLISH / -1 BEARISH
    
    pivots = []  # all detected pivots with classification
    events = []  # BOS/CHoCH events
    last_bos = None
    last_choch = None
    
    # Walk bars from index `size` to end (need `size` bars of history for leg)
    for i in range(size, n):
        bar = klines[i]
        candidate = klines[i - size]  # bar at offset 'size' bars back
        c_high = candidate.get('h', candidate['p'])
        c_low = candidate.get('l', candidate['p'])
        
        # Pine: ta.highest(size) on bar i = max of high[i-size+1 .. i]
        # (size bars including current, excluding the candidate at i-size)
        # newLegHigh = high[size] > ta.highest(size)
        window_highs = [klines[j].get('h', klines[j]['p']) for j in range(i - size + 1, i + 1)]
        window_lows = [klines[j].get('l', klines[j]['p']) for j in range(i - size + 1, i + 1)]
        
        max_h = max(window_highs) if window_highs else c_high
        min_l = min(window_lows) if window_lows else c_low
        
        new_leg_high = c_high > max_h
        new_leg_low = c_low < min_l
        
        prev_leg_state = leg_state
        if new_leg_high:
            leg_state = BEARISH_LEG  # Pine: bearish leg = a new high formed (=0)
        elif new_leg_low:
            leg_state = BULLISH_LEG  # =1
        # else: leg_state unchanged (Pine: leg keeps prior value)
        
        # Pine startOfNewLeg: ta.change(leg) != 0
        # In our case: pivot is created on EVERY transition — including the
        # very first transition from None (uninitialized) to a valid state,
        # AND any transition between BEARISH_LEG (0) and BULLISH_LEG (1).
        new_pivot = (prev_leg_state is not None and 
                     leg_state is not None and 
                     leg_state != prev_leg_state)
        
        if new_pivot:
            if leg_state == BULLISH_LEG:
                # New low pivot at offset size
                low_pivot['last_level'] = low_pivot['level']
                low_pivot['level'] = c_low
                low_pivot['crossed'] = False
                low_pivot['t'] = candidate.get('t')
                low_pivot['idx'] = i - size
                
                # Classify HL vs LL
                pt = 'HL' if (low_pivot['last_level'] is not None and
                               c_low > low_pivot['last_level']) else 'LL'
                pivots.append({
                    't': low_pivot['t'],
                    'idx': low_pivot['idx'],
                    'price': c_low,
                    'type': pt,
                })
            elif leg_state == BEARISH_LEG:
                # New high pivot
                high_pivot['last_level'] = high_pivot['level']
                high_pivot['level'] = c_high
                high_pivot['crossed'] = False
                high_pivot['t'] = candidate.get('t')
                high_pivot['idx'] = i - size
                
                pt = 'HH' if (high_pivot['last_level'] is not None and
                               c_high > high_pivot['last_level']) else 'LH'
                pivots.append({
                    't': high_pivot['t'],
                    'idx': high_pivot['idx'],
                    'price': c_high,
                    'type': pt,
                })
        
        # === BOS / CHoCH detection ===
        # Bullish: ta.crossover(close, p_ivot.currentLevel) and not crossed
        if high_pivot['level'] is not None and not high_pivot['crossed']:
            close_now = bar['p']
            close_prev = klines[i - 1]['p']
            # crossover = close_now > level AND close_prev <= level
            if close_now > high_pivot['level'] and close_prev <= high_pivot['level']:
                tag = CHOCH if trend_bias == BEARISH else BOS
                ev = {
                    'from_t': high_pivot['t'],
                    'from_idx': high_pivot['idx'],
                    'to_t': bar.get('t'),
                    'to_idx': i,
                    'level': high_pivot['level'],
                    'tag': tag,
                    'dir': 'bull',
                }
                events.append(ev)
                if tag == BOS:
                    last_bos = ev
                else:
                    last_choch = ev
                
                high_pivot['crossed'] = True
                trend_bias = BULLISH
        
        # Bearish: crossunder(close, low_pivot.level)
        if low_pivot['level'] is not None and not low_pivot['crossed']:
            close_now = bar['p']
            close_prev = klines[i - 1]['p']
            if close_now < low_pivot['level'] and close_prev >= low_pivot['level']:
                tag = CHOCH if trend_bias == BULLISH else BOS
                ev = {
                    'from_t': low_pivot['t'],
                    'from_idx': low_pivot['idx'],
                    'to_t': bar.get('t'),
                    'to_idx': i,
                    'level': low_pivot['level'],
                    'tag': tag,
                    'dir': 'bear',
                }
                events.append(ev)
                if tag == BOS:
                    last_bos = ev
                else:
                    last_choch = ev
                
                low_pivot['crossed'] = True
                trend_bias = BEARISH
    
    return {
        'pivots': pivots,
        'events': events,
        'trend': trend_bias,
        'last_bos': last_bos,
        'last_choch': last_choch,
    }
