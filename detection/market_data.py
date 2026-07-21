"""
Market Data Provider v1.2 — Multi-Exchange with Fallback + Pagination

KLINES priority: Bybit → Binance → OKX
  Bybit is primary because the bot trades there and TV's *.P reference
  charts pull from Bybit. Same exchange = identical OHLC = identical
  pivot detection.

Other data types (OI, depth, taker ratio) keep Binance Futures as primary
where the endpoint surface is richer.

Tracks which exchange provided data for UI display.
"""

import requests
import time
from typing import Dict, List, Optional, Tuple

REQUEST_TIMEOUT = 10
DELAY = 0.3

# Binance endpoints
BN_KLINE   = 'https://fapi.binance.com/fapi/v1/klines'
BN_OI      = 'https://fapi.binance.com/fapi/v1/openInterest'
BN_LS      = 'https://fapi.binance.com/futures/data/globalLongShortAccountRatio'
BN_TOP_LS  = 'https://fapi.binance.com/futures/data/topLongShortAccountRatio'
BN_TAKER   = 'https://fapi.binance.com/futures/data/takerlongshortRatio'
BN_DEPTH   = 'https://fapi.binance.com/fapi/v1/depth'

# OKX endpoints
OKX_KLINE  = 'https://www.okx.com/api/v5/market/candles'
OKX_OI     = 'https://www.okx.com/api/v5/rubik/stat/contracts-open-interest-history'
OKX_LS     = 'https://www.okx.com/api/v5/rubik/stat/contracts-long-short-account-ratio-contract-top-trader'
OKX_TAKER  = 'https://www.okx.com/api/v5/rubik/stat/taker-volume-contract'
OKX_DEPTH  = 'https://www.okx.com/api/v5/market/books'

# Bybit endpoints (public, no key)
BB_KLINE   = 'https://api.bybit.com/v5/market/kline'
BB_OI      = 'https://api.bybit.com/v5/market/open-interest'
BB_TAKER   = 'https://api.bybit.com/v5/market/account-ratio'


def _okx_symbol(symbol: str) -> str:
    """Convert BTCUSDT → BTC-USDT-SWAP (OKX format)."""
    base = symbol.replace('USDT', '')
    return f'{base}-USDT-SWAP'


def _to_okx_interval(binance_interval: str) -> str:
    """Map Binance interval to OKX bar."""
    return {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1H', '4h': '4H', '1d': '1D',
    }.get(binance_interval, '1m')


def _to_bybit_interval(binance_interval: str) -> str:
    """Map Binance interval to Bybit numeric interval."""
    return {
        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '4h': '240', '1d': 'D',
    }.get(binance_interval, '1')


class MarketData:
    """Fetches market data with Binance→OKX fallback."""
    
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'VSV-Bot/1.0'})
        
        # Track source per data type: {'klines': 'Binance', 'oi': 'OKX', ...}
        self._sources: Dict[str, str] = {}
        # Track errors for monitoring
        self._bn_errors: int = 0
        self._okx_errors: int = 0
        self._bb_errors: int = 0
    
    @property
    def sources(self) -> Dict[str, str]:
        return dict(self._sources)
    
    @property
    def source_summary(self) -> str:
        """Human-readable summary: 'Binance' or 'Binance+OKX' or 'OKX'."""
        vals = set(self._sources.values())
        if not vals:
            return '—'
        return '+'.join(sorted(vals))
    
    # ========================================
    # KLINES (taker buy/sell volumes)
    # ========================================
    
    # ────────────────────────────────────────────────────────────────────
    # Provider-specific page-size limits (per single API call).
    # Binance Futures: max 1500 per call (we use 1000 for safety margin).
    # Bybit V5:       max 1000 per call.
    # OKX:            max 300 per call (history endpoint).
    # ────────────────────────────────────────────────────────────────────
    _BN_PAGE_MAX  = 1000
    _BB_PAGE_MAX  = 1000
    _OKX_PAGE_MAX = 300
    _PAGE_DELAY   = 0.1   # Small pause between paginated calls to be polite
    _MAX_PAGES    = 20    # Safety cap — at 1000/page that's 20k bars max
    
    def fetch_klines(self, symbol: str, limit: int = 60,
                       interval: str = '1m') -> Optional[List[Dict]]:
        """Fetch klines with taker buy/sell. Returns [{p, v, b, s, h, l, o, t}, ...]
        in chronological order (oldest first).
        
        interval: '1m', '5m', '15m', '30m', '1h', '4h', '1d'  (Binance format)
        
        ─── Provider priority: Bybit → Binance → OKX ───
        Bybit is queried first because the bot trades on Bybit and the user's
        TradingView reference charts show `*USDT.P` symbols which are Bybit
        perpetuals. Using Bybit data here means our SMC structure detection
        runs on the exact same OHLC the user sees on TV — no exchange-to-
        exchange wick/close mismatches that would shift pivot bars by 1-2
        and produce subtly different CHoCH/BOS sequences.
        
        Binance Futures stays as the first fallback because it has the
        deepest history and the highest single-page limit (1500). OKX is
        the last resort for symbols that exist on neither Bybit nor Binance.
        
        When `limit` exceeds a provider's single-page maximum, the method
        transparently paginates via `end`/`endTime`/`before` API parameters
        and concatenates pages chronologically. Pagination stays within ONE
        provider to avoid mixing prices from different exchanges. If the
        primary provider yields less than requested due to history depth,
        the function returns what's available rather than failing — callers
        can detect short returns by checking len() vs limit.
        """
        # Try each provider in order. Each provider call returns either a
        # full chronological array (oldest→newest) or None on failure.
        candles = self._fetch_bybit(symbol, limit, interval)
        if candles:
            self._sources['klines'] = 'Bybit'
            return candles
        
        time.sleep(DELAY)
        candles = self._fetch_binance(symbol, limit, interval)
        if candles:
            self._sources['klines'] = 'Binance'
            return candles
        
        time.sleep(DELAY)
        candles = self._fetch_okx(symbol, limit, interval)
        if candles:
            self._sources['klines'] = 'OKX'
            return candles
        
        return None
    
    # ────────────────────────────────────────────────────────────────────
    # Per-provider paginated fetchers
    # All return oldest-first chronological array, or None on hard failure.
    # ────────────────────────────────────────────────────────────────────
    
    def _fetch_binance(self, symbol: str, limit: int,
                          interval: str) -> Optional[List[Dict]]:
        """Binance Futures with endTime pagination (newest→older chained)."""
        all_bars: List[Dict] = []
        remaining = limit
        end_time: Optional[int] = None
        pages = 0
        
        while remaining > 0 and pages < self._MAX_PAGES:
            pages += 1
            page_limit = min(remaining, self._BN_PAGE_MAX)
            
            try:
                params = {'symbol': symbol, 'interval': interval,
                          'limit': page_limit}
                if end_time is not None:
                    params['endTime'] = end_time
                r = self._session.get(BN_KLINE, params=params,
                                        timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    if pages == 1:
                        self._bn_errors += 1
                        return None  # First-page failure → try fallback
                    break  # Mid-pagination failure → keep what we have
                
                page = []
                for k in r.json():
                    try:
                        tv = float(k[7]); tb = float(k[10])
                        page.append({'t': int(k[0]), 'p': float(k[4]),
                                     'v': round(tv), 'b': round(tb),
                                     's': round(tv - tb),
                                     'h': float(k[2]), 'l': float(k[3]),
                                     'o': float(k[1])})
                    except Exception:
                        continue
                
                if not page:
                    break  # Empty page → no more history available
                
                # Binance returns oldest→newest within a page. Detect overlap
                # with already-fetched data to avoid infinite loops.
                if all_bars and page[-1]['t'] >= all_bars[0]['t']:
                    break
                
                all_bars = page + all_bars  # older + newer
                
                # Stop if we got fewer than requested — no older data exists
                if len(page) < page_limit:
                    break
                
                # Next page: end strictly before oldest bar we have
                end_time = all_bars[0]['t'] - 1
                remaining = limit - len(all_bars)
                
                if remaining > 0:
                    time.sleep(self._PAGE_DELAY)
            except Exception:
                if pages == 1:
                    self._bn_errors += 1
                    return None
                break
        
        return all_bars if all_bars else None
    
    def _fetch_okx(self, symbol: str, limit: int,
                     interval: str) -> Optional[List[Dict]]:
        """OKX with `before` pagination. OKX returns newest→oldest within a
        page; we reverse per-page for chronological consistency."""
        all_bars: List[Dict] = []
        remaining = limit
        before: Optional[int] = None
        pages = 0
        okx_sym = _okx_symbol(symbol)
        okx_interval = _to_okx_interval(interval)
        
        while remaining > 0 and pages < self._MAX_PAGES:
            pages += 1
            page_limit = min(remaining, self._OKX_PAGE_MAX)
            
            try:
                params = {'instId': okx_sym, 'bar': okx_interval,
                          'limit': str(page_limit)}
                if before is not None:
                    # OKX `before` returns bars with ts < before (older)
                    params['before'] = str(before)
                r = self._session.get(OKX_KLINE, params=params,
                                        timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    if pages == 1:
                        self._okx_errors += 1
                        return None
                    break
                
                data = r.json().get('data', [])
                page = []
                for k in data:
                    try:
                        tv = float(k[7])
                        o = float(k[1]); c = float(k[4])
                        buy_ratio = 0.6 if c > o else 0.4 if c < o else 0.5
                        tb = tv * buy_ratio
                        page.append({'t': int(k[0]), 'p': c, 'v': round(tv),
                                     'b': round(tb), 's': round(tv - tb),
                                     'h': float(k[2]), 'l': float(k[3]), 'o': o})
                    except Exception:
                        continue
                
                if not page:
                    break
                
                page.reverse()  # OKX returns newest-first; flip to oldest-first
                
                if all_bars and page[-1]['t'] >= all_bars[0]['t']:
                    break
                
                all_bars = page + all_bars
                
                if len(page) < page_limit:
                    break
                
                before = all_bars[0]['t']
                remaining = limit - len(all_bars)
                
                if remaining > 0:
                    time.sleep(self._PAGE_DELAY)
            except Exception:
                if pages == 1:
                    self._okx_errors += 1
                    return None
                break
        
        return all_bars if all_bars else None
    
    def _fetch_bybit(self, symbol: str, limit: int,
                       interval: str) -> Optional[List[Dict]]:
        """Bybit V5 with `end` pagination. Bybit returns newest→oldest."""
        all_bars: List[Dict] = []
        remaining = limit
        end_ms: Optional[int] = None
        pages = 0
        bb_interval = _to_bybit_interval(interval)
        
        while remaining > 0 and pages < self._MAX_PAGES:
            pages += 1
            page_limit = min(remaining, self._BB_PAGE_MAX)
            
            try:
                params = {'category': 'linear', 'symbol': symbol,
                          'interval': bb_interval, 'limit': page_limit}
                if end_ms is not None:
                    params['end'] = end_ms
                r = self._session.get(BB_KLINE, params=params,
                                        timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    if pages == 1:
                        self._bb_errors += 1
                        return None
                    break
                
                result = r.json().get('result', {})
                data = result.get('list', [])
                page = []
                for k in data:
                    try:
                        tv = float(k[6])
                        o = float(k[1]); c = float(k[4])
                        buy_ratio = 0.6 if c > o else 0.4 if c < o else 0.5
                        tb = tv * buy_ratio
                        page.append({'t': int(k[0]), 'p': c, 'v': round(tv),
                                     'b': round(tb), 's': round(tv - tb),
                                     'h': float(k[2]), 'l': float(k[3]), 'o': o})
                    except Exception:
                        continue
                
                if not page:
                    break
                
                page.reverse()  # Bybit returns newest-first; flip
                
                if all_bars and page[-1]['t'] >= all_bars[0]['t']:
                    break
                
                all_bars = page + all_bars
                
                if len(page) < page_limit:
                    break
                
                end_ms = all_bars[0]['t'] - 1
                remaining = limit - len(all_bars)
                
                if remaining > 0:
                    time.sleep(self._PAGE_DELAY)
            except Exception:
                if pages == 1:
                    self._bb_errors += 1
                    return None
                break
        
        return all_bars if all_bars else None
    
    # ========================================
    # TOP-N PERP UNIVERSE (Binance only)
    # ========================================
    
    def fetch_top_perp_symbols(self, n: int = 100, min_quote_volume_usd: float = 0.0
                                ) -> Optional[List[Dict]]:
        """Fetch top-N USDT-perpetual symbols from Binance Futures, sorted by
        24h quote volume (USD turnover) descending.
        
        Returns list of dicts: [{'symbol', 'quote_volume', 'last_price',
        'price_change_pct'}, ...]  ordered by quote_volume desc, length<=n,
        all entries have quote_volume >= min_quote_volume_usd.
        
        Only Binance is queried — no fallback. The TOP-100 universe should
        come from a single canonical source so the list stays stable
        between scans (different exchanges rank differently).
        
        Filters out:
          - Non-USDT pairs (BUSD, USDC, BTC quote, etc.)
          - Symbols ending in _<date> (delivery contracts, not perpetuals)
          - Inactive / suspended symbols (price_change == 0 with 0 volume)
        """
        try:
            r = self._session.get(
                'https://fapi.binance.com/fapi/v1/ticker/24hr',
                timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                return None
            tickers = r.json()
        except Exception as e:
            print(f"[MarketData] fetch_top_perp_symbols error: {e}")
            return None
        
        rows = []
        for t in tickers:
            symbol = t.get('symbol', '')
            # USDT-perp filter: must end with USDT and not have an underscore
            # (BTCUSDT_240927 = quarterly delivery, BTCUSDT = perpetual).
            if not symbol.endswith('USDT') or '_' in symbol:
                continue
            try:
                quote_vol = float(t.get('quoteVolume', 0))
                last_price = float(t.get('lastPrice', 0))
                pct_change = float(t.get('priceChangePercent', 0))
            except (TypeError, ValueError):
                continue
            # Drop dead pairs — zero turnover means delisted or stale
            if quote_vol <= 0 or last_price <= 0:
                continue
            if quote_vol < min_quote_volume_usd:
                continue
            rows.append({
                'symbol': symbol,
                'quote_volume': quote_vol,  # USD turnover 24h
                'last_price': last_price,
                'price_change_pct': pct_change,
            })
        
        # Sort by quote volume descending, take top N
        rows.sort(key=lambda x: x['quote_volume'], reverse=True)
        return rows[:n]
    
    def fetch_top_by_marketcap(self, n: int = 150,
                                min_quote_volume_usd: float = 0.0
                                ) -> Optional[List[Dict]]:
        """Fetch USDT-perp symbols filtered/sorted by CoinGecko MARKET CAP
        rank instead of 24h volume. Returns the perps whose underlying coin
        is in the top-N by market cap, intersected with what actually trades
        as a USDT-perp on Binance Futures.
        
        Returns list of dicts with the SAME shape as fetch_top_perp_symbols
        ({'symbol', 'quote_volume', 'last_price', 'price_change_pct'}) PLUS
        a 'market_cap_rank' field, sorted by market_cap_rank ascending
        (highest market cap first).
        
        Why intersect with Binance perps: a coin can be top-150 by market
        cap but have no perpetual to trade (or not on our exchange). We can
        only scan things that actually have a tradeable perp + live price.
        
        Falls back to None on CoinGecko unavailability — caller should then
        use fetch_top_perp_symbols (volume-based) as a safety net.
        """
        # 1. Get the full perp universe (with prices/volume) — pull a wide
        #    net (top 400 by volume) so we don't miss mid-cap coins that
        #    rank high by market cap but lower by volume.
        perp_rows = self.fetch_top_perp_symbols(
            n=400, min_quote_volume_usd=min_quote_volume_usd)
        if not perp_rows:
            return None
        
        # 2. Resolve market cap ranks via CoinGecko
        try:
            from detection.coingecko_client import get_coingecko_client
            cg = get_coingecko_client()
            if not cg.is_available():
                print('[MarketData] CoinGecko unavailable — '
                      'cannot apply market cap filter')
                return None
        except Exception as e:
            print(f'[MarketData] CoinGecko client error: {e}')
            return None
        
        # 3. Annotate each perp with its market cap rank, keep top-N
        annotated = []
        by_symbol = {r['symbol']: r for r in perp_rows}
        for sym, row in by_symbol.items():
            rank = cg.get_rank(sym)
            if rank is not None and rank <= n:
                enriched = dict(row)
                enriched['market_cap_rank'] = rank
                annotated.append(enriched)
        
        # 4. Sort by market cap rank ascending (biggest coins first)
        annotated.sort(key=lambda x: x['market_cap_rank'])
        print(f'[MarketData] Market-cap universe: {len(annotated)} perps '
              f'in top-{n} by market cap')
        return annotated
    
    # ========================================
    # OPEN INTEREST
    # ========================================
    
    def fetch_oi(self, symbol: str, price: float = 0) -> Tuple[Optional[float], str]:
        """Fetch Open Interest in USD. Returns (oi_usd, source)."""
        # Binance
        try:
            r = self._session.get(BN_OI,
                params={'symbol': symbol}, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                qty = float(r.json().get('openInterest', 0))
                oi_usd = qty * price if price else qty
                self._sources['oi'] = 'Binance'
                return oi_usd, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX
        try:
            time.sleep(DELAY)
            okx_sym = _okx_symbol(symbol)
            r = self._session.get(OKX_OI,
                params={'instType': 'SWAP', 'ccy': symbol.replace('USDT', ''),
                        'period': '5m', 'begin': '', 'end': ''},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    oi_val = float(data[0].get('oi', 0))
                    oi_usd = oi_val * price if price else oi_val
                    self._sources['oi'] = 'OKX'
                    return oi_usd, 'OKX'
        except:
            self._okx_errors += 1
        
        # Bybit
        try:
            time.sleep(DELAY)
            r = self._session.get(BB_OI,
                params={'category': 'linear', 'symbol': symbol, 'intervalTime': '5min', 'limit': 1},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                result = r.json().get('result', {})
                rows = result.get('list', [])
                if rows:
                    oi_val = float(rows[0].get('openInterest', 0))
                    oi_usd = oi_val * price if price else oi_val
                    self._sources['oi'] = 'Bybit'
                    return oi_usd, 'Bybit'
        except:
            self._bb_errors += 1
        
        return None, ''
    
    # ========================================
    # LONG/SHORT RATIO
    # ========================================
    
    def fetch_ls_ratio(self, symbol: str) -> Tuple[Optional[Dict], str]:
        """Fetch L/S ratio. Returns ({ls_ratio, ls_long}, source)."""
        # Binance
        try:
            r = self._session.get(BN_LS,
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data:
                    latest = data[-1]
                    result = {
                        'ls_ratio': float(latest.get('longShortRatio', 1)),
                        'ls_long': round(float(latest.get('longAccount', 0.5)) * 100, 1),
                    }
                    self._sources['ls'] = 'Binance'
                    return result, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX
        try:
            time.sleep(DELAY)
            ccy = symbol.replace('USDT', '')
            r = self._session.get(OKX_LS,
                params={'instType': 'SWAP', 'ccy': ccy, 'period': '5m'},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    ratio = float(data[0].get('ratio', 1))
                    long_pct = round(ratio / (1 + ratio) * 100, 1) if ratio > 0 else 50
                    result = {'ls_ratio': ratio, 'ls_long': long_pct}
                    self._sources['ls'] = 'OKX'
                    return result, 'OKX'
        except:
            self._okx_errors += 1
        
        # Bybit (account ratio — closest to L/S)
        try:
            time.sleep(DELAY)
            r = self._session.get(BB_TAKER,
                params={'category': 'linear', 'symbol': symbol, 'period': '1d', 'limit': 1},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                rows = r.json().get('result', {}).get('list', [])
                if rows:
                    buy_ratio = float(rows[0].get('buyRatio', 0.5))
                    sell_ratio = float(rows[0].get('sellRatio', 0.5))
                    ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1
                    long_pct = round(buy_ratio * 100, 1)
                    result = {'ls_ratio': round(ratio, 3), 'ls_long': long_pct}
                    self._sources['ls'] = 'Bybit'
                    return result, 'Bybit'
        except:
            self._bb_errors += 1
        
        return None, ''
    
    # ========================================
    # TOP TRADER L/S RATIO
    # ========================================
    
    def fetch_top_ls(self, symbol: str) -> Tuple[Optional[Dict], str]:
        """Fetch Top Trader L/S. Returns ({top_ls, top_long}, source)."""
        # Binance
        try:
            r = self._session.get(BN_TOP_LS,
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data:
                    latest = data[-1]
                    result = {
                        'top_ls': float(latest.get('longShortRatio', 1)),
                        'top_long': round(float(latest.get('longAccount', 0.5)) * 100, 1),
                    }
                    self._sources['top_ls'] = 'Binance'
                    return result, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX (uses same endpoint with different params)
        try:
            time.sleep(DELAY)
            ccy = symbol.replace('USDT', '')
            r = self._session.get(OKX_LS,
                params={'instType': 'SWAP', 'ccy': ccy, 'period': '5m'},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    ratio = float(data[0].get('ratio', 1))
                    long_pct = round(ratio / (1 + ratio) * 100, 1) if ratio > 0 else 50
                    result = {'top_ls': ratio, 'top_long': long_pct}
                    self._sources['top_ls'] = 'OKX'
                    return result, 'OKX'
        except:
            self._okx_errors += 1
        
        # Bybit (uses same account-ratio as ls)
        try:
            time.sleep(DELAY)
            r = self._session.get(BB_TAKER,
                params={'category': 'linear', 'symbol': symbol, 'period': '1d', 'limit': 1},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                rows = r.json().get('result', {}).get('list', [])
                if rows:
                    buy_ratio = float(rows[0].get('buyRatio', 0.5))
                    sell_ratio = float(rows[0].get('sellRatio', 0.5))
                    ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1
                    long_pct = round(buy_ratio * 100, 1)
                    result = {'top_ls': round(ratio, 3), 'top_long': long_pct}
                    self._sources['top_ls'] = 'Bybit'
                    return result, 'Bybit'
        except:
            self._bb_errors += 1
        
        return None, ''
    
    # ========================================
    # TAKER BUY/SELL RATIO
    # ========================================
    
    def fetch_taker_ratio(self, symbol: str) -> Tuple[Optional[float], str]:
        """Fetch taker buy/sell ratio. Returns (ratio, source)."""
        # Binance
        try:
            r = self._session.get(BN_TAKER,
                params={'symbol': symbol, 'period': '5m', 'limit': 6},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                if data:
                    ratio = float(data[-1].get('buySellRatio', 1))
                    self._sources['taker'] = 'Binance'
                    return ratio, 'Binance'
        except:
            self._bn_errors += 1
        
        # OKX
        try:
            time.sleep(DELAY)
            ccy = symbol.replace('USDT', '')
            r = self._session.get(OKX_TAKER,
                params={'instType': 'SWAP', 'ccy': ccy, 'period': '5m'},
                timeout=REQUEST_TIMEOUT)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    sell = float(data[0].get('sellVol', 1))
                    buy = float(data[0].get('buyVol', 1))
                    ratio = buy / sell if sell > 0 else 1
                    self._sources['taker'] = 'OKX'
                    return ratio, 'OKX'
        except:
            self._okx_errors += 1
        
        return None, ''
    
    # ========================================
    # FULL SENTIMENT (all 4 sentiment endpoints)
    # ========================================
    
    def fetch_sentiment(self, symbol: str) -> Dict:
        """Fetch OI + LS + TopLS + Taker with fallback. Returns dict with source info."""
        sent = {}
        sources = []
        
        # LS Ratio
        ls_data, ls_src = self.fetch_ls_ratio(symbol)
        if ls_data:
            sent.update(ls_data)
            if ls_src not in sources: sources.append(ls_src)
        time.sleep(DELAY)
        
        # Top Trader
        top_data, top_src = self.fetch_top_ls(symbol)
        if top_data:
            sent.update(top_data)
            if top_src not in sources: sources.append(top_src)
        time.sleep(DELAY)
        
        # Taker
        taker, taker_src = self.fetch_taker_ratio(symbol)
        if taker is not None:
            sent['taker'] = taker
            if taker_src not in sources: sources.append(taker_src)
        
        sent['_sources'] = '+'.join(sources) if sources else '—'
        return sent
    
    def get_status(self) -> Dict:
        return {
            'sources': self._sources,
            'summary': self.source_summary,
            'bn_errors': self._bn_errors,
            'okx_errors': self._okx_errors,
            'bb_errors': self._bb_errors,
        }


# Singleton
_instance: Optional[MarketData] = None

def get_market_data() -> MarketData:
    global _instance
    if _instance is None:
        _instance = MarketData()
    return _instance
