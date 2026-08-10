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


# ═══════════════════════════════════════════════════════════════════════════
# POC (біла лінія «як у MobChart») з Binance SPOT/FUTURES — окремий, легкий шлях
# поряд із build_volume_profile(). Використовується ендпоінтом
# /api/volume-profile/poc: біла горизонтальна лінія POC на графіку + вердикт
# «ціна↔POC» для напрямку сигналу (LONG ОК коли POC вище ціни; SHORT — коли нижче).
# ТОЧНІСТЬ: 1m-klines у вікні, обсяг КОЖНОЇ свічки рівномірно по бінах [low..high].
# ═══════════════════════════════════════════════════════════════════════════
import time as _vp_time
import threading as _vp_threading
import requests as _vp_requests

_SPOT_KLINES = 'https://api.binance.com/api/v3/klines'
_FUT_KLINES = 'https://fapi.binance.com/fapi/v1/klines'
_DEFAULT_BINS = 150
_DEFAULT_HOURS = 72.0
_MAX_REQUESTS = 10
_POC_VALUE_AREA = 0.70

_poc_cache = {}
_poc_cache_lock = _vp_threading.Lock()
_POC_TTL = 25.0


def _poc_klines_url(market: str) -> str:
    return _FUT_KLINES if (market or 'spot').lower().startswith('fut') else _SPOT_KLINES


def _poc_norm_symbol(symbol: str) -> str:
    s = (symbol or '').upper().strip()
    if s.endswith('.P'):
        s = s[:-2]
    return s


def _poc_fetch_klines(symbol, interval, limit=1000, start_ms=None, end_ms=None, market='spot'):
    params = {'symbol': symbol, 'interval': interval, 'limit': min(1000, int(limit))}
    if start_ms is not None:
        params['startTime'] = int(start_ms)
    if end_ms is not None:
        params['endTime'] = int(end_ms)
    r = _vp_requests.get(_poc_klines_url(market), params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else []


def _poc_fetch_window(symbol, interval, start_ms, end_ms, market='spot'):
    out = []
    cur = int(start_ms)
    for _ in range(_MAX_REQUESTS):
        batch = _poc_fetch_klines(symbol, interval, 1000, start_ms=cur, end_ms=end_ms, market=market)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        nxt = int(batch[-1][0]) + 1
        if nxt >= end_ms or nxt <= cur:
            break
        cur = nxt
    return out


# ── Bybit-фолбек: якщо монети немає на Binance — шукаємо на Bybit (v5 kline). ──
_BYBIT_KLINE = 'https://api.bybit.com/v5/market/kline'
_BYBIT_IV = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
             '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
             '1d': 'D', '1w': 'W'}


def _poc_fetch_window_bybit(symbol, interval, start_ms, end_ms):
    """(klines, category) з Bybit v5. Пробуємо linear (perp) → потім spot.
    Формат рядка Bybit: [start, open, high, low, close, volume, turnover] —
    індекси high[2]/low[3]/volume[5] ЗБІГАЮТЬСЯ з Binance, тож _poc_profile
    приймає їх напряму. Відповідь newest-first; пагінація зсувом `end`."""
    iv = _BYBIT_IV.get((interval or '1m').lower(), '60')
    for category in ('linear', 'spot'):
        rows = []
        cur_end = int(end_ms)
        try:
            for _ in range(_MAX_REQUESTS):
                params = {'category': category, 'symbol': symbol, 'interval': iv,
                          'start': int(start_ms), 'end': int(cur_end), 'limit': 1000}
                r = _vp_requests.get(_BYBIT_KLINE, params=params, timeout=12)
                r.raise_for_status()
                j = r.json()
                lst = (((j or {}).get('result') or {}).get('list')) or []
                if not lst:
                    break
                rows.extend(lst)
                if len(lst) < 1000:
                    break
                oldest = int(lst[-1][0])   # newest-first → останній = найстаріший
                if oldest <= start_ms:
                    break
                cur_end = oldest - 1
        except Exception:
            rows = []
        if rows:
            return rows, category
    return [], None


def _poc_profile(klines, bins):
    """(vol_by_bin, lo, hi, binw) — обсяг кожної свічки рівномірно по бінах [low..high]."""
    parsed, lows, highs = [], [], []
    for k in klines:
        try:
            hi = float(k[2]); lo = float(k[3]); vol = float(k[5])
        except (IndexError, TypeError, ValueError):
            continue
        if hi <= 0 or lo <= 0 or vol < 0 or hi < lo:
            continue
        parsed.append((lo, hi, vol)); lows.append(lo); highs.append(hi)
    if not parsed:
        return None, 0.0, 0.0, 0.0
    lo = min(lows); hi = max(highs)
    if hi <= lo:
        hi = lo * 1.0001 + 1e-9
    bins = max(10, int(bins))
    binw = (hi - lo) / bins
    if binw <= 0:
        return None, lo, hi, 0.0
    vol_by_bin = [0.0] * bins
    for (kl, kh, kv) in parsed:
        if kv <= 0:
            continue
        i0 = max(0, int((kl - lo) / binw))
        i1 = min(bins - 1, int((kh - lo) / binw))
        if i1 < i0:
            i1 = i0
        share = kv / (i1 - i0 + 1)
        for i in range(i0, i1 + 1):
            vol_by_bin[i] += share
    return vol_by_bin, lo, hi, binw


def _poc_value_area(vol_by_bin, poc_idx, target_frac):
    total = sum(vol_by_bin)
    if total <= 0:
        return poc_idx, poc_idx
    acc = vol_by_bin[poc_idx]; lo_i = hi_i = poc_idx
    n = len(vol_by_bin); need = total * target_frac
    while acc < need and (lo_i > 0 or hi_i < n - 1):
        below = vol_by_bin[lo_i - 1] if lo_i > 0 else -1.0
        above = vol_by_bin[hi_i + 1] if hi_i < n - 1 else -1.0
        if above >= below:
            hi_i += 1; acc += max(0.0, above)
        else:
            lo_i -= 1; acc += max(0.0, below)
    return lo_i, hi_i


def compute_poc(symbol, from_sec=None, to_sec=None, hours=None,
                bins=_DEFAULT_BINS, interval='1m', market='spot'):
    """POC (+VAH/VAL) з Binance (SPOT/FUTURES) за вікном.
    Вікно: [from_sec,to_sec] (видимий діапазон графіка) → інакше останні `hours`."""
    sym = _poc_norm_symbol(symbol)
    mkt = 'futures' if (market or 'spot').lower().startswith('fut') else 'spot'
    out = {'ok': False, 'symbol': sym, 'market': mkt, 'exchange': 'binance',
           'poc': None, 'vah': None,
           'val': None, 'price_high': None, 'price_low': None, 'bins': int(bins),
           'klines': 0, 'src': f'binance_{mkt}_{interval}',
           'value_area_pct': int(_POC_VALUE_AREA * 100), 'computed_at': None, 'reason': ''}
    if not sym:
        out['reason'] = 'no symbol'; return out
    now_ms = int(_vp_time.time() * 1000)
    if from_sec and to_sec and to_sec > from_sec:
        start_ms = int(float(from_sec) * 1000); end_ms = int(float(to_sec) * 1000)
    else:
        h = float(hours or _DEFAULT_HOURS)
        end_ms = now_ms; start_ms = end_ms - int(h * 3600 * 1000)
    if end_ms > now_ms:
        end_ms = now_ms
    ck = f"{mkt}|{sym}|{interval}|{start_ms // 1000}|{end_ms // 1000}|{int(bins)}"
    with _poc_cache_lock:
        hit = _poc_cache.get(ck)
        if hit and (_vp_time.time() - hit[0]) < _POC_TTL:
            return hit[1]
    # 1) Основне джерело — Binance (SPOT/FUTURES за вибором).
    klines = []
    try:
        klines = _poc_fetch_window(sym, interval, start_ms, end_ms, market=mkt)
    except Exception:
        klines = []
    # 2) ФОЛБЕК: монети немає на Binance → автоматично шукаємо на Bybit
    #    (linear perp → spot). Намагаємось визначити POC максимально.
    if not klines:
        try:
            bkl, bcat = _poc_fetch_window_bybit(sym, interval, start_ms, end_ms)
        except Exception:
            bkl, bcat = [], None
        if bkl:
            klines = bkl
            out['exchange'] = 'bybit'
            out['market'] = bcat or 'linear'
            out['src'] = f'bybit_{bcat or "linear"}_{interval}'
    if not klines:
        out['reason'] = 'no klines (немає ні на Binance, ні на Bybit)'; return out
    vol_by_bin, lo, hi, binw = _poc_profile(klines, bins)
    if not vol_by_bin or binw <= 0:
        out['reason'] = 'insufficient data'; return out
    poc_idx = max(range(len(vol_by_bin)), key=lambda i: vol_by_bin[i])
    poc = lo + (poc_idx + 0.5) * binw
    val_i, vah_i = _poc_value_area(vol_by_bin, poc_idx, _POC_VALUE_AREA)
    # Поточна ціна = close ОСТАННЬОЇ свічки вікна (index 4 у Binance/Bybit
    # klines). Віддаємо, щоб споживачі (POC-сетап) не залежали від окремого
    # ticker-API (його немає в MarketData).
    last_close = None
    try:
        last_close = round(float(klines[-1][4]), 10)
    except (IndexError, TypeError, ValueError):
        last_close = None
    out.update({'ok': True, 'poc': round(poc, 10),
                'vah': round(lo + (vah_i + 1.0) * binw, 10),
                'val': round(lo + val_i * binw, 10),
                'price_high': round(hi, 10), 'price_low': round(lo, 10),
                'last_close': last_close,
                'klines': len(klines), 'computed_at': _vp_time.time()})
    with _poc_cache_lock:
        _poc_cache[ck] = (_vp_time.time(), out)
    return out


def price_vs_poc(poc, price, side=None, tol_pct=0.05):
    """Вердикт «ціна↔POC»: LONG ОК коли POC вище ціни; SHORT ОК коли POC нижче."""
    res = {'ok': None, 'rel': None, 'reason': 'немає даних', 'dist_pct': None}
    try:
        if poc is None or price is None or poc <= 0 or price <= 0:
            return res
        dist_pct = (price - poc) / poc * 100.0
        res['dist_pct'] = round(dist_pct, 3)
        band = abs(float(tol_pct))
        rel = 'at' if abs(dist_pct) <= band else ('above' if price > poc else 'below')
        res['rel'] = rel
        s = (side or '').upper()
        if s == 'LONG':
            res['ok'] = (rel == 'below')
            res['reason'] = ('POC вище ціни — LONG ОК' if rel == 'below'
                             else ('ціна на POC' if rel == 'at' else 'POC нижче ціни — LONG проти вартості'))
        elif s == 'SHORT':
            res['ok'] = (rel == 'above')
            res['reason'] = ('POC нижче ціни — SHORT ОК' if rel == 'above'
                             else ('ціна на POC' if rel == 'at' else 'POC вище ціни — SHORT проти вартості'))
        else:
            res['reason'] = ('ціна вище POC' if rel == 'above' else ('ціна нижче POC' if rel == 'below' else 'ціна на POC'))
        return res
    except Exception:
        return res
