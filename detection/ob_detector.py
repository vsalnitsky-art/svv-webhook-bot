"""Order Block Detector — exact port of SMC_PRO_BOT__47_.pine logic.

Pine reference (lines numbered exactly):
  437-449  : Per-bar parsedHigh / parsedLow computation (volatility filter)
  450-456  : Push to parallel arrays (parsedHighs, parsedLows, highs, lows, times)
  605-626  : deleteOrderBlocks — mitigation logic
  628-651  : storeOrdeBlock — when BOS/CHoCH fires, find the extreme
             parsed value in the slice from pivot to current bar and store
             that bar as the order block.
  712-713  : storeOrdeBlock(p_ivot, internal, BULLISH) called on bullish
             BOS or CHoCH (close crossover of swing high)
  737-738  : storeOrdeBlock(p_ivot, internal, BEARISH) on bearish event

Algorithm summary:
  Each bar:
    atrMeasure = ta.atr(200)
    volatilityMeasure = atrMeasure  (default 'ATR' filter; alternative is
                                      cumulative TR / bar_index)
    highVolatilityBar = (high - low) >= 2 * volatilityMeasure
    parsedHigh = highVolatilityBar ? low : high
    parsedLow  = highVolatilityBar ? high : low
    parsedHighs.push(parsedHigh)
    parsedLows.push(parsedLow)
    highs.push(high), lows.push(low), times.push(time)
    
    For each order block already stored:
      if bias == BEARISH and high > OB.barHigh: mitigated → remove
      if bias == BULLISH and low  < OB.barLow:  mitigated → remove
    (mitigation source = HIGHLOW by default; alternative is CLOSE)
  
  When a BULLISH BOS or CHoCH event fires at bar `bar_index`:
    pivot = the swing/internal high pivot that got crossed
    slice parsedLows from pivot.barIndex to bar_index (exclusive of current)
    find min in that slice; let parsedIndex = pivot.barIndex + slice.argmin()
    OB = (barHigh = parsedHighs[parsedIndex],
          barLow  = parsedLows[parsedIndex],
          barTime = times[parsedIndex],
          bias    = BULLISH)
    unshift OB into internalOrderBlocks (newest at index 0)
  
  Mirror for BEARISH events: slice parsedHighs, find max instead.

User requirement: only the LAST valid (unmitigated) OB matters for display.
That is internalOrderBlocks[0] after all delete-mitigation passes have run
through to the latest bar. Returns None if every detected OB has been
mitigated, which is normal — strong trends mitigate counter-trend OBs.
"""

from typing import Dict, List, Optional


# Pine constants (mirroring smc_structure.py)
BULLISH = 1
BEARISH = -1


def detect_last_order_block(
    klines: List[Dict],
    pivots: List[Dict],
    events: List[Dict],
    atr_period: int = 200,
    filter_method: str = 'ATR',
    mitigation_method: str = 'HIGHLOW',
) -> Optional[Dict]:
    """Run the full Pine OB algorithm on a kline series and return only the
    most recent unmitigated order block (or None if none survives).
    
    Args:
        klines: list of bar dicts with keys 'h', 'l', 'p' (close), 't' (ms epoch).
            We expect closed bars only — caller drops the in-progress bar.
        pivots: pivot list from smc_structure._process(); each pivot has
            'idx' (= bar index of the pivot bar = Pine p_ivot.barIndex),
            'price', 'type' (HH/HL/LH/LL), 't'.
        events: BOS/CHoCH event list from smc_structure._process(). Each event
            has 'tag' (BOS/CHoCH), 'dir' (bull/bear), 'to_idx' (= bar_index
            at the moment the event fired), 'from_idx' (= pivot.barIndex
            of the crossed pivot), 'level'.
        atr_period: Pine's ta.atr(200) period — keep at 200 for exact match
        filter_method: 'ATR' (default) or 'RANGE' (cumulative TR / bar_index)
        mitigation_method: 'HIGHLOW' (default) or 'CLOSE'
    
    Returns:
        dict with the last unmitigated OB, or None.
        Shape:
            {
                'bias': 'BULLISH' | 'BEARISH',
                'bar_high': float,    # Pine barHigh
                'bar_low': float,     # Pine barLow
                'bar_time': int,      # Pine barTime (ms epoch)
                'bar_idx': int,       # the bar index in klines where OB sits
                'created_at_idx': int,  # idx of the BOS/CHoCH that created it
                'created_at_t': int,    # timestamp of that triggering event
            }
    """
    n = len(klines)
    if n < atr_period + 2:
        # Not enough history for ATR-based filter; without a stable ATR
        # the parsedHigh/Low classification flips wildly.
        return None
    
    # === Pre-compute ATR(200) (Wilder smoothing — Pine ta.atr semantics) ===
    # ta.atr(N) in Pine: TR_i = max(high-low, |high-prev_close|, |low-prev_close|)
    # then Wilder smoothing: ATR_0 = SMA(TR, N), ATR_i = (ATR_{i-1}*(N-1) + TR_i) / N
    # We compute the full series so we can index per-bar.
    atr_series = _compute_atr_wilder(klines, atr_period)
    
    # === Cumulative TR for the alternative 'RANGE' filter ===
    cum_tr_avg = None
    if filter_method == 'RANGE':
        cum_tr_avg = _compute_cum_tr_avg(klines)
    
    # === Walk bars chronologically, building parsed arrays + OB lifecycle ===
    parsed_highs: List[float] = []
    parsed_lows: List[float] = []
    highs: List[float] = []
    lows: List[float] = []
    times: List[int] = []
    
    # Order blocks list — newest at index 0 (Pine .unshift semantics).
    # Only the last (newest unmitigated) one matters for our use case but
    # we maintain the full list because mitigation passes touch all of them.
    obs: List[Dict] = []
    
    # Pre-index events by their `to_idx` so we can run storeOrdeBlock at
    # the bar where the BOS/CHoCH fired. Multiple events at the same bar
    # are unusual but possible (a CHoCH and a BOS on the very same close
    # if both directions resolve in one bar, which Pine treats sequentially).
    events_by_idx: Dict[int, List[Dict]] = {}
    for ev in events:
        if ev.get('tag') not in ('BOS', 'CHoCH'):
            continue
        idx = ev.get('to_idx')
        if idx is None:
            continue
        events_by_idx.setdefault(idx, []).append(ev)
    
    for i in range(n):
        bar = klines[i]
        high = float(bar.get('h', bar.get('p', 0)))
        low = float(bar.get('l', bar.get('p', 0)))
        t = bar.get('t', 0)
        
        # === Pine 437-449: parsed values via volatility filter ===
        # The trick: on a high-volatility bar, the high/low roles flip —
        # this avoids treating spike bars as the OB origin (they'd be
        # poor zones because price already moved through them).
        if filter_method == 'ATR':
            vol = atr_series[i] if i < len(atr_series) else None
        else:  # RANGE
            vol = cum_tr_avg[i] if cum_tr_avg and i < len(cum_tr_avg) else None
        
        if vol is None or vol <= 0:
            # Before ATR is seeded, treat as low-volatility (no flip)
            parsed_high = high
            parsed_low = low
        else:
            high_vol_bar = (high - low) >= 2 * vol
            parsed_high = low if high_vol_bar else high
            parsed_low = high if high_vol_bar else low
        
        parsed_highs.append(parsed_high)
        parsed_lows.append(parsed_low)
        highs.append(high)
        lows.append(low)
        times.append(t)
        
        # === Pine 605-626: deleteOrderBlocks (mitigation) ===
        # Run BEFORE store so that the OB created on this bar isn't
        # incorrectly mitigated by its own bar's high/low. Pine evaluates
        # delete on the new bar's high/low against existing OBs, then
        # store runs on the same bar and adds the new one.
        if mitigation_method == 'HIGHLOW':
            bear_src = high   # bearishOrderBlockMitigationSource
            bull_src = low    # bullishOrderBlockMitigationSource
        else:  # CLOSE
            close_v = float(bar.get('p', 0))
            bear_src = close_v
            bull_src = close_v
        
        # Iterate in reverse so .pop(idx) doesn't shift unprocessed indices.
        # Pine `for [index, eachOrderBlock] in orderBlocks` walks low-to-high
        # but uses `orderBlocks.remove(index)` which in Pine slices the array;
        # in Python pop-by-index works the same when iterated in reverse.
        for ob_idx in range(len(obs) - 1, -1, -1):
            ob = obs[ob_idx]
            if ob['bias'] == BEARISH and bear_src > ob['bar_high']:
                obs.pop(ob_idx)
            elif ob['bias'] == BULLISH and bull_src < ob['bar_low']:
                obs.pop(ob_idx)
        
        # === Pine 633-651: storeOrdeBlock — for any event ending at this bar ===
        for ev in events_by_idx.get(i, []):
            pivot_idx = ev.get('from_idx')
            if pivot_idx is None or pivot_idx >= i:
                # No pivot to slice from, or pivot is at/after current bar
                # (shouldn't happen — events fire AFTER pivot exists)
                continue
            
            ev_dir = ev.get('dir')
            if ev_dir == 'bull':
                # BULLISH event — slice parsedLows from pivot to current bar
                # Pine: a_rray = parsedLows.slice(pivot.barIndex, bar_index)
                # Pine slice is exclusive of end index so we use [pivot_idx:i]
                # which gives bars pivot_idx, pivot_idx+1, ..., i-1.
                slice_lows = parsed_lows[pivot_idx:i]
                if not slice_lows:
                    continue
                # Pine: parsedIndex = pivot.barIndex + a_rray.indexof(a_rray.min())
                min_val = min(slice_lows)
                rel_idx = slice_lows.index(min_val)
                parsed_idx = pivot_idx + rel_idx
                bias = BULLISH
            elif ev_dir == 'bear':
                # BEARISH — slice parsedHighs, find max
                slice_highs = parsed_highs[pivot_idx:i]
                if not slice_highs:
                    continue
                max_val = max(slice_highs)
                rel_idx = slice_highs.index(max_val)
                parsed_idx = pivot_idx + rel_idx
                bias = BEARISH
            else:
                continue
            
            ob = {
                'bias': bias,
                'bar_high': parsed_highs[parsed_idx],
                'bar_low': parsed_lows[parsed_idx],
                'bar_time': times[parsed_idx],
                'bar_idx': parsed_idx,
                'created_at_idx': i,
                'created_at_t': t,
                # Track which event tag created this OB. Pine semantics:
                # storeOrdeBlock fires on BOTH BOS and CHoCH equally.
                # We expose the tag so downstream filters can distinguish:
                #   CHoCH-created = fresh trend reversal OB (first signal of new trend)
                #   BOS-created   = continuation OB (trend already in motion)
                # Trading the CHoCH-created OB only is a much tighter setup —
                # you enter at the pivot point of a confirmed trend change,
                # not a continuation move that's already extended.
                'created_by_tag': ev.get('tag', ''),  # 'CHoCH' or 'BOS'
            }
            
            # Pine: if orderBlocks.size() >= 100: orderBlocks.pop()
            # then orderBlocks.unshift(o_rderBlock)
            # i.e. cap at 100, drop oldest, prepend newest.
            if len(obs) >= 100:
                obs.pop()  # drop oldest (last index)
            obs.insert(0, ob)  # unshift = prepend
    
    # User wants only the latest valid OB. After the full pass, if the list
    # is non-empty, index 0 is the most recently stored AND not yet mitigated
    # (because mitigation removes from the list, not just marks).
    if not obs:
        return None
    
    last_ob = obs[0]
    # Convert Pine-internal ints to readable strings for the JSON layer
    return {
        'bias': 'BULLISH' if last_ob['bias'] == BULLISH else 'BEARISH',
        'bar_high': float(last_ob['bar_high']),
        'bar_low': float(last_ob['bar_low']),
        'bar_time': int(last_ob['bar_time']),
        'bar_idx': int(last_ob['bar_idx']),
        'created_at_idx': int(last_ob['created_at_idx']),
        'created_at_t': int(last_ob['created_at_t']),
        # 'CHoCH' or 'BOS' — used downstream to gate "fresh-only" trades
        'created_by_tag': str(last_ob.get('created_by_tag', '')),
    }


def _compute_atr_wilder(klines: List[Dict], period: int) -> List[Optional[float]]:
    """Wilder's ATR — exact match for Pine ta.atr(period).
    
    For bars 0..period-1 returns None (insufficient data). On bar period-1
    the seed is SMA(TR, period). Subsequent bars use Wilder recurrence:
        ATR_i = (ATR_{i-1} * (period - 1) + TR_i) / period
    """
    n = len(klines)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    
    # Compute TR series
    tr: List[float] = [0.0] * n
    for i in range(n):
        high = float(klines[i].get('h', klines[i].get('p', 0)))
        low = float(klines[i].get('l', klines[i].get('p', 0)))
        if i == 0:
            tr[i] = high - low
        else:
            prev_close = float(klines[i - 1].get('p', 0))
            tr[i] = max(high - low,
                         abs(high - prev_close),
                         abs(low - prev_close))
    
    # Seed ATR at bar period (1-indexed) = index period-1 with SMA of first
    # `period` TR values. Pine's ta.atr stabilizes here.
    seed_idx = period - 1
    seed = sum(tr[0:period]) / period
    out[seed_idx] = seed
    
    # Wilder recurrence for the rest
    for i in range(seed_idx + 1, n):
        prev = out[i - 1] or seed
        out[i] = (prev * (period - 1) + tr[i]) / period
    
    return out


def _compute_cum_tr_avg(klines: List[Dict]) -> List[Optional[float]]:
    """Cumulative-TR average per bar — Pine: ta.cum(ta.tr) / bar_index.
    
    bar_index in Pine is 0 on the first bar; ta.cum starts at 0 and adds
    TR each bar. The division by bar_index produces a running mean of TR
    that's resistant to outliers but slower to react than ATR.
    """
    n = len(klines)
    out: List[Optional[float]] = [None] * n
    cum = 0.0
    for i in range(n):
        high = float(klines[i].get('h', klines[i].get('p', 0)))
        low = float(klines[i].get('l', klines[i].get('p', 0)))
        if i == 0:
            tr = high - low
            cum += tr
            # bar_index=0 → division undefined in Pine; treat as None
            continue
        prev_close = float(klines[i - 1].get('p', 0))
        tr = max(high - low,
                  abs(high - prev_close),
                  abs(low - prev_close))
        cum += tr
        out[i] = cum / (i + 1)  # +1 because we want N bars including current
    return out


def format_ob_label(ob: Optional[Dict]) -> str:
    """Render a one-line label for the chart header / Telegram.
    
    Format: '🟢 OB Long' or '🔴 OB Short' or '' when no valid OB.
    """
    if not ob:
        return ''
    if ob['bias'] == 'BULLISH':
        return '🟢 OB Long'
    if ob['bias'] == 'BEARISH':
        return '🔴 OB Short'
    return ''
