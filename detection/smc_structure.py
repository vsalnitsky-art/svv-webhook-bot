"""
Smart Money Concepts Engine — direct port of LuxAlgo SMC Pine Script.

Pine reference: SMC_PRO_BOT__47_.pine (lines 463-738)

Algorithm:
  For each bar i ≥ size:
    1. Run leg(size) on bar i, comparing high[i-size]/low[i-size] (the candidate
       bar 'size' bars back) against ta.highest(size)/ta.lowest(size) — i.e.
       max/min of high/low across the next 'size' bars (from i-size+1 to i).
    
    2. If candidate.high > max(next size bars): leg becomes BEARISH_LEG (=0)
       If candidate.low  < min(next size bars):  leg becomes BULLISH_LEG (=1)
       Otherwise: leg stays unchanged.
    
    3. On any leg transition (startOfNewLeg in Pine = ta.change(leg) != 0),
       create a new pivot at the candidate bar:
         - BULLISH_LEG transition → new LOW pivot
         - BEARISH_LEG transition → new HIGH pivot
       Classify pivot vs previous same-side pivot:
         - new low > prev low  → HL,  else LL
         - new high > prev high → HH, else LH
    
    4. On EVERY bar, check close-price crossovers/crossunders of the most
       recent pivot levels:
         - close > internalHigh.level AND not crossed → bullish event
            tag = CHoCH if trend was BEARISH, else BOS; trend → BULLISH
         - close < internalLow.level AND not crossed  → bearish event
            tag = CHoCH if trend was BULLISH, else BOS; trend → BEARISH
"""

from typing import Dict, List, Optional


# Pine constants
BULLISH_LEG = 1
BEARISH_LEG = 0
BULLISH = 1
BEARISH = -1
CHOCH = "CHoCH"
BOS = "BOS"


def detect_smc_structure(klines: List[Dict], internal_size: int = 5,
                          swing_size: int = 50) -> Dict:
    """Detect SMC structure on klines.
    
    Returns:
        {
            'internal': {pivots, events, trend, last_bos, last_choch},
            'swing':    {pivots, events, trend, last_bos, last_choch},
            'klines_count': N,
        }
    """
    if not klines or len(klines) < internal_size + 2:
        return _empty_result(klines)
    
    internal = _process(klines, internal_size)
    swing = _process(klines, swing_size) if len(klines) >= swing_size + 2 \
            else _empty_struct()
    
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


def _process(klines: List[Dict], size: int) -> Dict:
    """Single-pass SMC detection at given pivot size."""
    n = len(klines)
    
    # Pine `var leg = 0` — but 0 collides with BEARISH_LEG. We use None to mean
    # "uninitialized" so the very first leg detection doesn't fire a phantom
    # pivot transition.
    leg_state = None
    
    high_pivot = {'level': None, 'last_level': None, 'crossed': False,
                   't': None, 'idx': None}
    low_pivot = {'level': None, 'last_level': None, 'crossed': False,
                  't': None, 'idx': None}
    
    trend_bias = 0
    pivots = []
    events = []
    
    for i in range(size, n):
        bar = klines[i]
        candidate = klines[i - size]
        c_high = candidate.get('h', candidate['p'])
        c_low = candidate.get('l', candidate['p'])
        
        # Pine: ta.highest(size) on bar i = max(high[i-size+1..i])
        max_h = max(klines[j].get('h', klines[j]['p']) for j in range(i - size + 1, i + 1))
        min_l = min(klines[j].get('l', klines[j]['p']) for j in range(i - size + 1, i + 1))
        
        new_leg_high = c_high > max_h
        new_leg_low = c_low < min_l
        
        prev_leg = leg_state
        if new_leg_high:
            leg_state = BEARISH_LEG
        elif new_leg_low:
            leg_state = BULLISH_LEG
        
        # Pivot only forms on a real leg transition (not on first init)
        new_pivot = (prev_leg is not None and 
                     leg_state is not None and 
                     leg_state != prev_leg)
        
        if new_pivot:
            if leg_state == BULLISH_LEG:
                low_pivot['last_level'] = low_pivot['level']
                low_pivot['level'] = c_low
                low_pivot['crossed'] = False
                low_pivot['t'] = candidate.get('t')
                low_pivot['idx'] = i - size
                pt = 'HL' if (low_pivot['last_level'] is not None and 
                              c_low > low_pivot['last_level']) else 'LL'
                pivots.append({
                    't': low_pivot['t'], 'idx': low_pivot['idx'],
                    'price': c_low, 'type': pt,
                })
            else:  # BEARISH_LEG
                high_pivot['last_level'] = high_pivot['level']
                high_pivot['level'] = c_high
                high_pivot['crossed'] = False
                high_pivot['t'] = candidate.get('t')
                high_pivot['idx'] = i - size
                pt = 'HH' if (high_pivot['last_level'] is not None and 
                              c_high > high_pivot['last_level']) else 'LH'
                pivots.append({
                    't': high_pivot['t'], 'idx': high_pivot['idx'],
                    'price': c_high, 'type': pt,
                })
        
        # === BOS / CHoCH detection ===
        if i >= 1:
            close_now = bar['p']
            close_prev = klines[i - 1]['p']
            
            # Bullish crossover
            if (high_pivot['level'] is not None and not high_pivot['crossed']
                and close_now > high_pivot['level'] and close_prev <= high_pivot['level']):
                tag = CHOCH if trend_bias == BEARISH else BOS
                events.append({
                    'from_t': high_pivot['t'], 'from_idx': high_pivot['idx'],
                    'to_t': bar.get('t'), 'to_idx': i,
                    'level': high_pivot['level'], 'tag': tag, 'dir': 'bull',
                })
                high_pivot['crossed'] = True
                trend_bias = BULLISH
            
            # Bearish crossunder
            if (low_pivot['level'] is not None and not low_pivot['crossed']
                and close_now < low_pivot['level'] and close_prev >= low_pivot['level']):
                tag = CHOCH if trend_bias == BULLISH else BOS
                events.append({
                    'from_t': low_pivot['t'], 'from_idx': low_pivot['idx'],
                    'to_t': bar.get('t'), 'to_idx': i,
                    'level': low_pivot['level'], 'tag': tag, 'dir': 'bear',
                })
                low_pivot['crossed'] = True
                trend_bias = BEARISH
    
    last_bos = next((e for e in reversed(events) if e['tag'] == BOS), None)
    last_choch = next((e for e in reversed(events) if e['tag'] == CHOCH), None)
    
    return {
        'pivots': pivots,
        'events': events,
        'trend': trend_bias,
        'last_bos': last_bos,
        'last_choch': last_choch,
    }
