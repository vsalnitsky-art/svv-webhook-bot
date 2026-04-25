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
    
    internal = _process_structure(klines, internal_size, label='internal')
    
    # Swing computed but currently disabled in UI — keep logic ready
    swing = _process_structure(klines, swing_size, label='swing') \
        if len(klines) >= swing_size + 2 else _empty_struct()
    
    return {
        'internal': internal,
        'swing': swing,
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


def _process_structure(klines: List[Dict], size: int, label: str = '') -> Dict:
    """Walk klines and detect structure.
    
    Mirrors Pine getCurrentStructure() + displayStructure() called bar-by-bar.
    """
    n = len(klines)
    
    # leg state
    leg_state = 0  # 0 = none yet (Pine: var leg = 0)
    prev_leg_state = 0
    
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
        
        # ta.highest(size) at bar i ≈ max of high[i-size..i-1]
        # newLegHigh = high[size] > ta.highest(size)
        # In Pine: high[size] is the bar `size` back; ta.highest(size) returns
        # the highest of the previous `size` bars (offsets 0..size-1).
        # So we compare candidate.high to max(high[i-size+1 .. i])
        window_highs = [klines[j].get('h', klines[j]['p']) for j in range(i - size + 1, i + 1)]
        window_lows = [klines[j].get('l', klines[j]['p']) for j in range(i - size + 1, i + 1)]
        
        max_h = max(window_highs) if window_highs else c_high
        min_l = min(window_lows) if window_lows else c_low
        
        new_leg_high = c_high > max_h
        new_leg_low = c_low < min_l
        
        prev_leg_state = leg_state
        if new_leg_high:
            leg_state = BEARISH_LEG  # Pine: bearish leg = a new high formed
        elif new_leg_low:
            leg_state = BULLISH_LEG
        
        new_pivot = (leg_state != prev_leg_state) and prev_leg_state != 0 or \
                    (prev_leg_state == 0 and leg_state != 0)
        
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
