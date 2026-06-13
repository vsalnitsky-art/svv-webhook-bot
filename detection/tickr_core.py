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
