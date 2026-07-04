"""
tickr_core — Python port of the Go `tickr` project, embedded as a module
of the SVV bot. Fetches and NORMALIZES trading instruments from four
exchanges (Binance, MEXC, BingX, Bybit) into one common Symbol shape, so
they can be filtered, compared (diff), and turned into TradingView
watchlists.

Mirrors the original's three capabilities:
  • fetch  — pull instruments for an exchange + category set
  • diff   — compare two snapshots (added / removed / changed)
  • watch  — here: manual fetch+diff against a stored snapshot (the
             background daemon is intentionally NOT ported per the user's
             choice; alerts fire on user action through the bot's own
             Telegram notifier)

Each exchange is a small adapter behind fetch_symbols(); adding a fifth
is one function + a registry entry, exactly like the Go `Adapter`.
"""

import time
import requests
from typing import Dict, List, Optional

HTTP_TIMEOUT = 12
USER_AGENT = 'svv-tickr/1.0'

# Canonical market types
MARKET_SPOT = 'spot'
MARKET_SWAP = 'swap'

EXCHANGES = ['binance', 'bybit', 'mexc', 'bingx']

# Categories the UI offers (subset of the Go version that makes sense here)
CATEGORIES = ['spot', 'swap', 'usdt', 'usdc']

_session = requests.Session()
_session.headers.update({'User-Agent': USER_AGENT})


# ----------------------------------------------------------------------
# Common Symbol shape (dict — keeps it JSON-trivial for the API layer)
# ----------------------------------------------------------------------

def _mk_symbol(exchange, symbol, base, quote, market_type,
               status, is_active, tick_size='', min_qty='',
               contract_type='', exchange_symbol=''):
    sym = {
        'exchange': exchange,
        'symbol': symbol,                       # canonical e.g. BTCUSDT
        'exchange_symbol': exchange_symbol or symbol,
        'base_asset': base,
        'quote_asset': quote,
        'market_type': market_type,
        'contract_type': contract_type,
        'status': status,
        'is_active': bool(is_active),
        'is_spot': market_type == MARKET_SPOT,
        'is_swap': market_type == MARKET_SWAP,
        'is_usdt': quote == 'USDT',
        'is_usdc': quote == 'USDC',
        'tick_size': tick_size,
        'min_qty': min_qty,
    }
    sym['tradingview_symbol'] = _tv_symbol(sym)
    return sym


def symbol_key(s: Dict) -> str:
    """Stable identity for diff/state — exchange|market|symbol."""
    return f"{s['exchange']}|{s['market_type']}|{s['symbol']}"


# ----------------------------------------------------------------------
# TradingView symbol mapping (ports internal/tv/tradingview.go)
# ----------------------------------------------------------------------

_TV_PREFIX = {'binance': 'BINANCE', 'bybit': 'BYBIT',
              'mexc': 'MEXC', 'bingx': 'BINGX'}


def _tv_symbol(s: Dict, suffix_perp: str = '.P') -> str:
    prefix = _TV_PREFIX.get(s['exchange'].lower(), s['exchange'].upper())
    if s['market_type'] == MARKET_SWAP:
        return f"{prefix}:{s['symbol']}{suffix_perp}"
    return f"{prefix}:{s['symbol']}"


def _get_json(url: str, params: dict = None) -> dict:
    r = _session.get(url, params=params or {}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------
# Adapters — one per exchange. Each returns a list of common Symbols.
# ----------------------------------------------------------------------

def _fetch_binance(markets: List[str]) -> List[Dict]:
    out = []
    if MARKET_SPOT in markets:
        data = _get_json('https://api.binance.com/api/v3/exchangeInfo')
        for s in data.get('symbols', []):
            tick = mn = ''
            for f in s.get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    tick = f.get('tickSize', '')
                elif f.get('filterType') == 'LOT_SIZE':
                    mn = f.get('minQty', '')
            out.append(_mk_symbol(
                'binance', s['symbol'], s.get('baseAsset', ''),
                s.get('quoteAsset', ''), MARKET_SPOT, s.get('status', ''),
                s.get('status') == 'TRADING', tick, mn))
    if MARKET_SWAP in markets:
        data = _get_json('https://fapi.binance.com/fapi/v1/exchangeInfo')
        for s in data.get('symbols', []):
            tick = mn = ''
            for f in s.get('filters', []):
                if f.get('filterType') == 'PRICE_FILTER':
                    tick = f.get('tickSize', '')
                elif f.get('filterType') == 'LOT_SIZE':
                    mn = f.get('minQty', '')
            out.append(_mk_symbol(
                'binance', s['symbol'], s.get('baseAsset', ''),
                s.get('quoteAsset', ''), MARKET_SWAP, s.get('status', ''),
                s.get('status') == 'TRADING', tick, mn,
                contract_type=s.get('contractType', '')))
    return out


def _fetch_bybit(markets: List[str]) -> List[Dict]:
    out = []
    cats = []
    if MARKET_SPOT in markets:
        cats.append(('spot', MARKET_SPOT))
    if MARKET_SWAP in markets:
        cats.append(('linear', MARKET_SWAP))
    for cat, mt in cats:
        cursor = ''
        for _ in range(20):  # paginate defensively
            params = {'category': cat, 'limit': 1000}
            if cursor:
                params['cursor'] = cursor
            data = _get_json('https://api.bybit.com/v5/market/instruments-info', params)
            result = data.get('result', {}) or {}
            for ins in result.get('list', []) or []:
                pf = ins.get('priceFilter', {}) or {}
                lf = ins.get('lotSizeFilter', {}) or {}
                out.append(_mk_symbol(
                    'bybit', ins.get('symbol', ''), ins.get('baseCoin', ''),
                    ins.get('quoteCoin', ''), mt, ins.get('status', ''),
                    ins.get('status') == 'Trading',
                    pf.get('tickSize', ''),
                    lf.get('minOrderQty', lf.get('minTradingQty', '')),
                    contract_type=ins.get('contractType', '')))
            cursor = result.get('nextPageCursor', '')
            if not cursor:
                break
    return out


def _fetch_mexc(markets: List[str]) -> List[Dict]:
    out = []
    if MARKET_SPOT in markets:
        data = _get_json('https://api.mexc.com/api/v3/exchangeInfo')
        for s in data.get('symbols', []):
            out.append(_mk_symbol(
                'mexc', s.get('symbol', ''), s.get('baseAsset', ''),
                s.get('quoteAsset', ''), MARKET_SPOT, s.get('status', ''),
                str(s.get('status', '')).upper() in ('ENABLED', '1', 'TRADING'),
                s.get('quotePrecision', '') and '', ''))
    if MARKET_SWAP in markets:
        data = _get_json('https://contract.mexc.com/api/v1/contract/detail')
        for s in data.get('data', []) or []:
            name = s.get('symbol', '')  # e.g. BTC_USDT
            canon = name.replace('_', '')
            out.append(_mk_symbol(
                'mexc', canon, s.get('baseCoin', ''), s.get('quoteCoin', ''),
                MARKET_SWAP, str(s.get('state', '')),
                s.get('state') in (0, '0'), '', '',
                exchange_symbol=name))
    return out


def _fetch_bingx(markets: List[str]) -> List[Dict]:
    out = []
    if MARKET_SPOT in markets:
        data = _get_json('https://open-api.bingx.com/openApi/spot/v1/common/symbols')
        syms = ((data.get('data') or {}).get('symbols')) or []
        for s in syms:
            name = s.get('symbol', '')        # e.g. BTC-USDT
            canon = name.replace('-', '')
            base, _, quote = name.partition('-')
            out.append(_mk_symbol(
                'bingx', canon, base, quote, MARKET_SPOT,
                str(s.get('status', '')), s.get('status') in (1, '1'),
                '', '', exchange_symbol=name))
    if MARKET_SWAP in markets:
        data = _get_json('https://open-api.bingx.com/openApi/swap/v2/quote/contracts')
        for s in data.get('data', []) or []:
            name = s.get('symbol', '')        # e.g. BTC-USDT
            canon = name.replace('-', '')
            out.append(_mk_symbol(
                'bingx', canon, s.get('asset', ''), s.get('currency', ''),
                MARKET_SWAP, str(s.get('status', '')),
                s.get('status') in (1, '1'), '', '',
                exchange_symbol=name))
    return out


_ADAPTERS = {
    'binance': _fetch_binance,
    'bybit': _fetch_bybit,
    'mexc': _fetch_mexc,
    'bingx': _fetch_bingx,
}


# ----------------------------------------------------------------------
# Activity metrics — 24h volume / trade count / price change, plus OI &
# funding for swap. One bulk ticker request per (exchange, market), keyed
# by canonical symbol, then merged onto the fetched instruments.
# ----------------------------------------------------------------------

def _canon(name: str) -> str:
    return name.replace('-', '').replace('_', '').upper()


def _activity_binance(market: str) -> Dict[str, Dict]:
    """canonical symbol → {vol_usd, change_pct, trades, oi_usd, funding}."""
    out = {}
    if market == MARKET_SPOT:
        data = _get_json('https://api.binance.com/api/v3/ticker/24hr')
        for t in data:
            out[_canon(t['symbol'])] = {
                'vol_usd': float(t.get('quoteVolume', 0) or 0),
                'change_pct': float(t.get('priceChangePercent', 0) or 0),
                'trades': int(t.get('count', 0) or 0),
                'last': float(t.get('lastPrice', 0) or 0),
            }
    else:
        data = _get_json('https://fapi.binance.com/fapi/v1/ticker/24hr')
        for t in data:
            out[_canon(t['symbol'])] = {
                'vol_usd': float(t.get('quoteVolume', 0) or 0),
                'change_pct': float(t.get('priceChangePercent', 0) or 0),
                'trades': int(t.get('count', 0) or 0),
                'last': float(t.get('lastPrice', 0) or 0),
            }
        # OI + funding (separate bulk endpoints on Binance futures)
        try:
            prem = _get_json('https://fapi.binance.com/fapi/v1/premiumIndex')
            for p in prem:
                k = _canon(p['symbol'])
                if k in out:
                    out[k]['funding'] = float(p.get('lastFundingRate', 0) or 0)
        except Exception:
            pass
    return out


def _activity_bybit(market: str) -> Dict[str, Dict]:
    out = {}
    cat = 'spot' if market == MARKET_SPOT else 'linear'
    data = _get_json('https://api.bybit.com/v5/market/tickers', {'category': cat})
    for t in (data.get('result', {}) or {}).get('list', []) or []:
        k = _canon(t.get('symbol', ''))
        last = float(t.get('lastPrice', 0) or 0)
        hi = float(t.get('highPrice24h', 0) or 0)
        lo = float(t.get('lowPrice24h', 0) or 0)
        rec = {
            'vol_usd': float(t.get('turnover24h', 0) or 0),  # turnover = quote vol
            'change_pct': float(t.get('price24hPcnt', 0) or 0) * 100.0,
            'last': last,
            # 24h range as % of price — small range = compressed (coiling)
            'range_pct': ((hi - lo) / last * 100.0) if last > 0 else 0.0,
        }
        if market == MARKET_SWAP:
            oi_contracts = float(t.get('openInterest', 0) or 0)
            rec['oi_usd'] = oi_contracts * rec['last']
            rec['funding'] = float(t.get('fundingRate', 0) or 0)
        out[k] = rec
    return out


def _activity_mexc(market: str) -> Dict[str, Dict]:
    out = {}
    if market == MARKET_SPOT:
        data = _get_json('https://api.mexc.com/api/v3/ticker/24hr')
        for t in (data if isinstance(data, list) else []):
            out[_canon(t.get('symbol', ''))] = {
                'vol_usd': float(t.get('quoteVolume', 0) or 0),
                'change_pct': float(t.get('priceChangePercent', 0) or 0) * 100.0,
                'trades': int(t.get('count', 0) or 0),
                'last': float(t.get('lastPrice', 0) or 0),
            }
    else:
        data = _get_json('https://contract.mexc.com/api/v1/contract/ticker')
        for t in data.get('data', []) or []:
            k = _canon(t.get('symbol', ''))
            last = float(t.get('lastPrice', 0) or 0)
            out[k] = {
                'vol_usd': float(t.get('amount24', 0) or 0),  # quote turnover
                'change_pct': float(t.get('riseFallRate', 0) or 0) * 100.0,
                'last': last,
                'oi_usd': float(t.get('holdVol', 0) or 0) * last,
                'funding': float(t.get('fundingRate', 0) or 0),
            }
    return out


def _activity_bingx(market: str) -> Dict[str, Dict]:
    out = {}
    if market == MARKET_SPOT:
        data = _get_json('https://open-api.bingx.com/openApi/spot/v1/ticker/24hr')
        rows = data.get('data', []) or []
        for t in rows:
            out[_canon(t.get('symbol', ''))] = {
                'vol_usd': float(t.get('quoteVolume', 0) or 0),
                'change_pct': float(t.get('priceChangePercent', 0) or 0),
                'last': float(t.get('lastPrice', 0) or 0),
            }
    else:
        data = _get_json('https://open-api.bingx.com/openApi/swap/v2/quote/ticker')
        rows = data.get('data', []) or []
        for t in rows:
            k = _canon(t.get('symbol', ''))
            out[k] = {
                'vol_usd': float(t.get('quoteVolume', 0) or 0),
                'change_pct': float(t.get('priceChangePercent', 0) or 0),
                'last': float(t.get('lastPrice', 0) or 0),
            }
    return out


_ACTIVITY = {
    'binance': _activity_binance,
    'bybit': _activity_bybit,
    'mexc': _activity_mexc,
    'bingx': _activity_bingx,
}


# ----------------------------------------------------------------------
# Public API: fetch / filter / diff
# ----------------------------------------------------------------------

def _markets_from_categories(cats: List[str]) -> List[str]:
    markets = []
    if 'spot' in cats:
        markets.append(MARKET_SPOT)
    if 'swap' in cats:
        markets.append(MARKET_SWAP)
    if not markets:                 # no explicit market → both
        markets = [MARKET_SPOT, MARKET_SWAP]
    return markets


def fetch(exchange: str, categories: List[str],
          active_only: bool = True, reverse: bool = False) -> Dict:
    """Fetch + filter instruments. Returns an envelope dict with
    {ok, exchange, count, symbols, warnings, fetched_at}."""
    exchange = (exchange or '').lower()
    adapter = _ADAPTERS.get(exchange)
    if adapter is None:
        return {'ok': False, 'reason': f'unknown exchange {exchange}'}
    cats = [c.lower() for c in (categories or [])]
    markets = _markets_from_categories(cats)
    warnings = []
    try:
        syms = adapter(markets)
    except Exception as e:
        return {'ok': False, 'reason': f'{exchange} fetch error: {e}',
                'exchange': exchange}

    # Quote-asset filters from categories
    want_usdt = 'usdt' in cats
    want_usdc = 'usdc' in cats
    def keep(s):
        if active_only and not s['is_active']:
            return False
        if want_usdt or want_usdc:
            if not ((want_usdt and s['is_usdt']) or (want_usdc and s['is_usdc'])):
                return False
        return True
    syms = [s for s in syms if keep(s)]
    syms.sort(key=lambda s: (s['market_type'], s['symbol']))
    if reverse:
        syms.reverse()
    return {'ok': True, 'exchange': exchange, 'categories': cats,
            'count': len(syms), 'symbols': syms, 'warnings': warnings,
            'fetched_at': time.time()}


_WATCHED_FIELDS = ['status', 'quote_asset', 'contract_type',
                   'tick_size', 'min_qty']


def diff(old_syms: List[Dict], new_syms: List[Dict]) -> Dict:
    """Compare two symbol lists by key → added / removed / changed.
    Ports internal/app/diff.go (same watched fields)."""
    old_idx = {symbol_key(s): s for s in (old_syms or [])}
    new_idx = {symbol_key(s): s for s in (new_syms or [])}
    added, removed, changed = [], [], []
    for k in sorted(new_idx):
        n = new_idx[k]
        o = old_idx.get(k)
        if o is None:
            added.append(n)
            continue
        field_diffs = []
        for f in _WATCHED_FIELDS:
            if str(o.get(f, '')) != str(n.get(f, '')):
                field_diffs.append({'field': f, 'old': str(o.get(f, '')),
                                    'new': str(n.get(f, ''))})
        if field_diffs:
            changed.append({'key': k, 'symbol': n['symbol'],
                            'changes': field_diffs})
    for k in sorted(old_idx):
        if k not in new_idx:
            removed.append(old_idx[k])
    return {'added': added, 'removed': removed, 'changed': changed}


def to_tv_list(symbols: List[Dict], separator: str = 'newline') -> str:
    """Build a TradingView watchlist text from symbols."""
    items = [s['tradingview_symbol'] for s in symbols]
    return (',' if separator == 'comma' else '\n').join(items)


# ----------------------------------------------------------------------
# Top Active — rank instruments by live trader activity
# ----------------------------------------------------------------------

# Sort keys the UI offers. 'spike' needs a baseline (see below).
SORT_KEYS = ['vol_usd', 'spike', 'change_abs', 'trades', 'oi_usd', 'funding_abs']


def top_active(exchange: str, categories: List[str], sort_by: str = 'vol_usd',
               top_n: int = 20, active_only: bool = True,
               baseline_key: str = '', min_vol_usd: float = 0.0) -> Dict:
    """Fetch instruments + 24h activity metrics, rank, return top N.

    sort_by:
      vol_usd     — 24h quote volume (who's biggest right now)
      spike       — vol now / stored-baseline vol (anomaly: who just lit up)
      change_abs  — |24h % change| (biggest movers, up or down)
      trades      — trade count (retail churn)
      oi_usd      — open interest in USD (swap only — capital committed)
      funding_abs — |funding rate| (swap only — crowded/overheated side)

    For 'spike' the caller passes baseline_key; we read the stored
    snapshot's per-symbol vol from DB-side (handled in the route) — here
    baseline arrives already merged via the 'baseline' field if present.
    """
    exchange = (exchange or '').lower()
    if exchange not in _ADAPTERS:
        return {'ok': False, 'reason': f'unknown exchange {exchange}'}
    cats = [c.lower() for c in (categories or [])]
    base = fetch(exchange, categories, active_only=active_only)
    if not base.get('ok'):
        return base
    symbols = base['symbols']

    # Pull activity per market present in the result, merge by canon symbol
    markets = sorted({s['market_type'] for s in symbols})
    act_fn = _ACTIVITY[exchange]
    metrics = {}
    warnings = []
    for mt in markets:
        try:
            m = act_fn(mt)
            for k, v in m.items():
                metrics[(mt, k)] = v
        except Exception as e:
            warnings.append(f'{exchange} {mt} activity: {e}')

    try:
        min_vol_usd = max(0.0, float(min_vol_usd or 0.0))
    except (TypeError, ValueError):
        min_vol_usd = 0.0

    enriched = []
    for s in symbols:
        # Skip dated/delivery futures (ETHUSDT-26MAR27, BTCUSDT-31JUL26 …) —
        # they carry a '-' suffix and are not the perpetual/spot we want.
        if '-' in str(s.get('symbol', '')):
            continue
        m = metrics.get((s['market_type'], _canon(s['symbol']))) or {}
        row = dict(s)
        row['vol_usd'] = m.get('vol_usd', 0.0)
        row['change_pct'] = m.get('change_pct', 0.0)
        row['change_abs'] = abs(m.get('change_pct', 0.0))
        row['trades'] = m.get('trades', 0)
        row['oi_usd'] = m.get('oi_usd', 0.0)
        row['funding'] = m.get('funding', 0.0)
        row['funding_abs'] = abs(m.get('funding', 0.0))
        row['last'] = m.get('last', 0.0)
        # Filter out illiquid coins (small 24h volume).
        if min_vol_usd > 0 and (row['vol_usd'] or 0) < min_vol_usd:
            continue
        enriched.append(row)

    sort_by = sort_by if sort_by in SORT_KEYS else 'vol_usd'
    if sort_by == 'spike':
        # spike requires a baseline; without it fall back to vol_usd so the
        # panel still ranks something meaningful instead of erroring.
        if not any(r.get('baseline_vol') for r in enriched):
            sort_by = 'vol_usd'
        else:
            for r in enriched:
                bv = r.get('baseline_vol') or 0
                r['spike'] = (r['vol_usd'] / bv) if bv > 0 else 0.0
    enriched.sort(key=lambda r: r.get(sort_by, 0) or 0, reverse=True)
    return {'ok': True, 'exchange': exchange, 'categories': cats,
            'sort_by': sort_by, 'count': len(enriched),
            'symbols': enriched[:max(1, min(top_n, 100))],
            'warnings': warnings, 'fetched_at': time.time()}


# ----------------------------------------------------------------------
# Opportunity Scanner — "coiled spring" pre-move selection for watchlist
# ----------------------------------------------------------------------
# Philosophy (user choice): pick coins where a move is BUILDING, not one
# already spent. A good watchlist candidate for the SMC bot has:
#   • liquidity        — enough volume + OI to enter/exit (HARD GATE)
#   • OI accumulation  — open interest rising vs a stored baseline
#                        (fresh positions = fuel for the coming move)
#   • compression      — tight 24h range (coiled, not already exploded)
#   • mild volume spike — money arriving, but the move hasn't run yet
#   • funding not extreme — extreme funding = crowded/late (PENALTY)
#
# This is a STATISTICAL front-filter: it narrows ~600 coins to a short
# list of high-potential names, which are then handed to the SMC bot.
# It does NOT see structure (OB/CHoCH) — that's the bot's job. It raises
# the odds by focusing the bot on promising coins, not random ones.

# Default gates / weights — all overridable from the API/UI.
OPP_DEFAULTS = {
    'min_vol_usd': 5_000_000,     # 24h turnover floor (liquidity)
    'min_oi_usd': 2_000_000,      # open-interest floor (swap depth)
    'max_funding_abs': 0.001,     # |funding| above this = crowded → drop
    'max_change_abs': 25.0,       # already moved >25% in 24h = likely late
    'w_oi_accel': 35,             # OI rising vs baseline
    'w_compression': 30,          # tight range = coiled
    'w_vol_spike': 20,            # mild volume pickup
    'w_funding_room': 15,         # neutral funding = room to move
    'top_n': 20,
}

# FF-optimized preset: a separate "coiled-spring for the Fuel Auto-Filter"
# selection. FF reads liq-fuel off OPEN INTEREST, so candidates must have real
# depth (reliable fuel) and enough turnover (clean fills) — NOT the tiny coins
# the default coiled-spring score over-ranks. So: much higher liquidity floors,
# OI-acceleration weighted heavier (rising OI = fuel building), compression
# de-emphasised (a breakout-timing metric, not a fuel metric). Liquid coins
# score lower on this scale, so the firing threshold should be lower too.
FF_OPP_PARAMS = {
    'min_vol_usd': 30_000_000,    # liquid enough for clean fills
    'min_oi_usd': 10_000_000,     # real swap depth → reliable liq-fuel
    'max_funding_abs': 0.001,     # extreme funding = crowded → drop
    'max_change_abs': 25.0,       # already ran >25% in 24h = late → drop
    'w_oi_accel': 45,             # rising OI = fuel accumulating (heaviest)
    'w_compression': 15,          # breakout metric — less relevant to fuel
    'w_vol_spike': 20,            # money arriving
    'w_funding_room': 20,         # neutral funding = room to move
    'top_n': 60,
}


def opportunity_scan(exchange: str = 'bybit', params: dict = None,
                     oi_baseline: dict = None) -> dict:
    """Rank Bybit swap-USDT coins by 'coiled-spring' opportunity score.

    params      — overrides for OPP_DEFAULTS (gates + weights)
    oi_baseline — {symbol: oi_usd} captured earlier, for OI acceleration.
                  When absent, the OI-accel component is neutral (50).
    Returns {ok, count_in, count_out, symbols:[...], params, fetched_at}.
    """
    p = dict(OPP_DEFAULTS)
    if params:
        for k, v in params.items():
            if k in p and v is not None:
                p[k] = v
    exchange = (exchange or 'bybit').lower()
    if exchange not in _ACTIVITY:
        return {'ok': False, 'reason': f'unknown exchange {exchange}'}

    # Bybit swap + USDT universe (the bot trades these)
    base = fetch(exchange, ['swap', 'usdt'], active_only=True)
    if not base.get('ok'):
        return base
    syms = [s for s in base['symbols'] if s['is_swap'] and s['is_usdt']]
    try:
        metrics = _ACTIVITY[exchange](MARKET_SWAP)
    except Exception as e:
        return {'ok': False, 'reason': f'{exchange} activity error: {e}'}

    oi_baseline = oi_baseline or {}
    count_in = len(syms)
    scored = []
    for s in syms:
        m = metrics.get(_canon(s['symbol']))
        if not m:
            continue
        vol = m.get('vol_usd', 0)
        oi = m.get('oi_usd', 0)
        funding = m.get('funding', 0)
        change_abs = abs(m.get('change_pct', 0))
        range_pct = m.get('range_pct', 0)

        # --- HARD GATES (the spent / illiquid / crowded are dropped) ---
        if vol < p['min_vol_usd']:
            continue
        if oi < p['min_oi_usd']:
            continue
        if abs(funding) > p['max_funding_abs']:
            continue
        if change_abs > p['max_change_abs']:
            continue

        # --- COMPONENT SCORES (0..1 each) ---
        # OI acceleration: now vs baseline. >0 means positions building.
        base_oi = oi_baseline.get(s['symbol'], 0)
        if base_oi > 0:
            oi_growth = (oi - base_oi) / base_oi
            # map -10%..+30% growth → 0..1, clamp
            oi_accel = max(0.0, min(1.0, (oi_growth + 0.10) / 0.40))
        else:
            oi_accel = 0.5            # neutral when no baseline yet

        # Compression: tighter 24h range = higher score. 2% range → ~1.0,
        # 20%+ range → ~0. Coiled springs sit at the low end.
        compression = max(0.0, min(1.0, 1.0 - (range_pct - 2.0) / 18.0))

        # Mild volume spike: reward elevated-but-not-blown-out volume.
        # We don't have a per-coin vol baseline here, so use OI-relative
        # turnover as a proxy: high turnover vs OI = active churn building.
        turn_ratio = (vol / oi) if oi > 0 else 0
        vol_spike = max(0.0, min(1.0, turn_ratio / 5.0))

        # Funding room: closer to neutral = more room before crowding.
        funding_room = max(0.0, 1.0 - abs(funding) / p['max_funding_abs'])

        score = (p['w_oi_accel'] * oi_accel
                 + p['w_compression'] * compression
                 + p['w_vol_spike'] * vol_spike
                 + p['w_funding_room'] * funding_room)
        max_score = (p['w_oi_accel'] + p['w_compression']
                     + p['w_vol_spike'] + p['w_funding_room'])
        score_100 = round(score / max_score * 100, 1) if max_score else 0

        scored.append({
            'symbol': s['symbol'],
            'tradingview_symbol': s['tradingview_symbol'],
            'opportunity': score_100,
            'vol_usd': round(vol, 0), 'oi_usd': round(oi, 0),
            'funding': funding, 'change_pct': round(m.get('change_pct', 0), 2),
            'range_pct': round(range_pct, 2),
            'oi_accel': round(oi_accel, 2),
            'compression': round(compression, 2),
            'vol_spike': round(vol_spike, 2),
            'funding_room': round(funding_room, 2),
            'has_baseline': base_oi > 0,
        })

    scored.sort(key=lambda r: r['opportunity'], reverse=True)
    top = scored[:max(1, min(p['top_n'], 100))]
    return {'ok': True, 'exchange': exchange,
            'count_in': count_in, 'count_passed': len(scored),
            'count_out': len(top), 'symbols': top,
            'params': p, 'fetched_at': time.time()}


def opportunity_oi_snapshot(exchange: str = 'bybit') -> dict:
    """Capture {symbol: oi_usd} now, to use later as OI-accel baseline."""
    exchange = (exchange or 'bybit').lower()
    if exchange not in _ACTIVITY:
        return {}
    try:
        m = _ACTIVITY[exchange](MARKET_SWAP)
    except Exception:
        return {}
    return {sym: rec.get('oi_usd', 0) for sym, rec in m.items()
            if rec.get('oi_usd', 0) > 0}
