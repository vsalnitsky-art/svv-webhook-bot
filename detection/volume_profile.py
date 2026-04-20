"""
Volume Profile Engine v1.0 — Market Profile analytics

Builds standard Volume Profile from 1-minute klines:
  - POC (Point of Control): price level with maximum volume
  - VAH / VAL (Value Area High / Low): boundaries of area containing 70% of volume
  - Volume bars per price level (buy vs sell split via taker volume)
  - HVN / LVN (High / Low Volume Nodes)

Methodology:
  1. Fetch 1m klines for requested period (1h → 7d, max Binance limit 1500)
  2. For each candle, distribute its volume across price levels between high and low
  3. Aggregate by price buckets (auto-resolution based on price range)
  4. Find POC, then expand around it until 70% of total volume is covered (VA)

Data source: MarketData (Binance → OKX → Bybit fallback chain)
"""

from typing import Dict, List, Optional, Tuple


# Value Area percentage (TPO standard = 70%)
VALUE_AREA_PCT = 0.70

# Target number of price buckets — UI friendly (more buckets = finer detail)
TARGET_BUCKETS = 50

# Minimum klines required for a meaningful profile
MIN_KLINES = 5


def build_volume_profile(symbol: str, hours: int = 24, buckets: int = TARGET_BUCKETS) -> Dict:
    """Build Volume Profile for a symbol over the given time window.
    
    Args:
        symbol: e.g. 'BTCUSDT'
        hours: time window in hours (1-168)
        buckets: target number of price levels (default 50)
    
    Returns:
        {
            'symbol': 'BTCUSDT',
            'hours': 24,
            'klines_count': 1440,
            'price_min': 74000, 'price_max': 76000,
            'total_volume': 3_500_000_000,
            'poc_price': 75300,
            'poc_volume': 180_000_000,
            'vah': 75800,          # Value Area High
            'val': 74900,          # Value Area Low
            'va_volume_pct': 70.2, # actual achieved VA percentage
            'levels': [             # sorted high → low for UI
                {
                    'price': 75800,
                    'total': 45_000_000,
                    'buy': 27_000_000,
                    'sell': 18_000_000,
                    'is_poc': false,
                    'in_va': true,
                    'is_hvn': false,
                    'is_lvn': false,
                },
                ...
            ],
            'source': 'Binance',
        }
    """
    try:
        from detection.market_data import get_market_data
        md = get_market_data()
        
        # Cap limit: Binance max 1500, 1m candles → ~25h max usable
        limit = min(1500, max(MIN_KLINES, int(hours * 60)))
        klines = md.fetch_klines(symbol, limit=limit)
        
        if not klines or len(klines) < MIN_KLINES:
            return _empty_profile(symbol, hours, reason='Not enough data')
        
        source = md._sources.get('klines', 'Unknown')
        
        # Determine price range from all candle highs/lows
        all_highs = [c.get('h', c['p']) for c in klines]
        all_lows = [c.get('l', c['p']) for c in klines]
        p_max = max(all_highs)
        p_min = min(all_lows)
        
        if p_max <= p_min:
            return _empty_profile(symbol, hours, reason='Invalid price range')
        
        # Calculate bucket size — round to sensible tick
        bucket_size = _calc_bucket_size(p_min, p_max, buckets)
        
        # Aggregate volume per bucket
        # For each candle, distribute its buy/sell volume uniformly across its H-L range
        bucket_data: Dict[float, Dict[str, float]] = {}
        
        for c in klines:
            h = c.get('h', c['p'])
            l = c.get('l', c['p'])
            buy = c.get('b', 0)
            sell = c.get('s', 0)
            
            if h < l:
                h, l = l, h
            
            # Find all buckets that overlap [l, h]
            low_bucket = _price_to_bucket(l, bucket_size)
            high_bucket = _price_to_bucket(h, bucket_size)
            
            if low_bucket == high_bucket:
                # Whole candle fits in one bucket
                _add_to_bucket(bucket_data, low_bucket, buy, sell)
            else:
                # Distribute uniformly across buckets
                n_buckets = int((high_bucket - low_bucket) / bucket_size) + 1
                if n_buckets <= 0:
                    n_buckets = 1
                buy_per = buy / n_buckets
                sell_per = sell / n_buckets
                b = low_bucket
                while b <= high_bucket:
                    _add_to_bucket(bucket_data, round(b, 8), buy_per, sell_per)
                    b += bucket_size
        
        if not bucket_data:
            return _empty_profile(symbol, hours, reason='No volume data')
        
        # Build sorted level list (high → low for UI)
        sorted_prices = sorted(bucket_data.keys(), reverse=True)
        
        # Find POC (max total volume)
        poc_price = max(bucket_data.keys(), key=lambda p: bucket_data[p]['buy'] + bucket_data[p]['sell'])
        poc_volume = bucket_data[poc_price]['buy'] + bucket_data[poc_price]['sell']
        
        # Calculate Value Area: expand from POC until we cover 70% of total
        total_volume = sum(d['buy'] + d['sell'] for d in bucket_data.values())
        target_va = total_volume * VALUE_AREA_PCT
        
        vah, val, va_pct = _find_value_area(bucket_data, poc_price, bucket_size, target_va, total_volume)
        
        # HVN / LVN detection: compare to average
        avg_vol = total_volume / len(bucket_data)
        hvn_threshold = avg_vol * 1.8
        lvn_threshold = avg_vol * 0.3
        
        # Build levels list
        levels = []
        for p in sorted_prices:
            d = bucket_data[p]
            total = d['buy'] + d['sell']
            levels.append({
                'price': round(p, 2) if p < 1000 else round(p),
                'total': round(total),
                'buy': round(d['buy']),
                'sell': round(d['sell']),
                'is_poc': (p == poc_price),
                'in_va': (val <= p <= vah),
                'is_hvn': (total >= hvn_threshold and p != poc_price),
                'is_lvn': (total <= lvn_threshold),
            })
        
        return {
            'symbol': symbol,
            'hours': hours,
            'klines_count': len(klines),
            'bucket_size': bucket_size,
            'price_min': round(p_min, 2) if p_min < 1000 else round(p_min),
            'price_max': round(p_max, 2) if p_max < 1000 else round(p_max),
            'total_volume': round(total_volume),
            'poc_price': round(poc_price, 2) if poc_price < 1000 else round(poc_price),
            'poc_volume': round(poc_volume),
            'vah': round(vah, 2) if vah < 1000 else round(vah),
            'val': round(val, 2) if val < 1000 else round(val),
            'va_volume_pct': round(va_pct * 100, 1),
            'levels': levels,
            'source': source,
            'current_price': klines[-1]['p'],
        }
    except Exception as e:
        print(f"[VP] Error building profile: {type(e).__name__}: {e}")
        return _empty_profile(symbol, hours, reason=str(e))


def _calc_bucket_size(p_min: float, p_max: float, target_buckets: int) -> float:
    """Choose a sensible bucket size that produces ~target_buckets levels."""
    rng = p_max - p_min
    raw_size = rng / target_buckets
    
    # Round to a clean number
    if raw_size >= 100:
        return round(raw_size / 50) * 50
    elif raw_size >= 10:
        return round(raw_size / 5) * 5
    elif raw_size >= 1:
        return round(raw_size)
    elif raw_size >= 0.1:
        return round(raw_size * 10) / 10
    elif raw_size >= 0.01:
        return round(raw_size * 100) / 100
    else:
        return max(raw_size, 0.0001)


def _price_to_bucket(price: float, bucket_size: float) -> float:
    return round(price / bucket_size) * bucket_size


def _add_to_bucket(bucket_data: Dict, price: float, buy: float, sell: float):
    if price not in bucket_data:
        bucket_data[price] = {'buy': 0.0, 'sell': 0.0}
    bucket_data[price]['buy'] += buy
    bucket_data[price]['sell'] += sell


def _find_value_area(bucket_data: Dict, poc_price: float, bucket_size: float,
                      target: float, total: float) -> Tuple[float, float, float]:
    """Expand from POC alternately up/down until we cover `target` volume.
    
    Returns (VAH, VAL, actual_va_pct).
    """
    if total <= 0:
        return poc_price, poc_price, 0.0
    
    sorted_prices = sorted(bucket_data.keys())
    try:
        poc_idx = sorted_prices.index(poc_price)
    except ValueError:
        return poc_price, poc_price, 0.0
    
    accumulated = bucket_data[poc_price]['buy'] + bucket_data[poc_price]['sell']
    high_idx = poc_idx
    low_idx = poc_idx
    
    while accumulated < target and (high_idx < len(sorted_prices) - 1 or low_idx > 0):
        # Compare next candidate above and below, pick the bigger one
        above_vol = 0
        below_vol = 0
        if high_idx < len(sorted_prices) - 1:
            ap = sorted_prices[high_idx + 1]
            above_vol = bucket_data[ap]['buy'] + bucket_data[ap]['sell']
        if low_idx > 0:
            bp = sorted_prices[low_idx - 1]
            below_vol = bucket_data[bp]['buy'] + bucket_data[bp]['sell']
        
        if above_vol == 0 and below_vol == 0:
            break
        
        if above_vol >= below_vol:
            high_idx += 1
            accumulated += above_vol
        else:
            low_idx -= 1
            accumulated += below_vol
    
    vah = sorted_prices[high_idx]
    val = sorted_prices[low_idx]
    return vah, val, (accumulated / total if total > 0 else 0)


def _empty_profile(symbol: str, hours: int, reason: str = '') -> Dict:
    return {
        'symbol': symbol,
        'hours': hours,
        'klines_count': 0,
        'price_min': 0, 'price_max': 0,
        'total_volume': 0,
        'poc_price': 0, 'poc_volume': 0,
        'vah': 0, 'val': 0, 'va_volume_pct': 0,
        'levels': [],
        'source': 'none',
        'current_price': 0,
        'error': reason,
    }
