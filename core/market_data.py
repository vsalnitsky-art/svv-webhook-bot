"""
Market Data - Data retrieval and caching for Sleeper OB Bot
Використовує Binance Futures для сканування/аналізу
"""
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from config import API_LIMITS, TIMEFRAME_MAP
from core.binance_connector import get_binance_connector

class MarketDataFetcher:
    """Fetches and caches market data from Binance Futures"""
    
    def __init__(self):
        self.connector = get_binance_connector()
        self._ticker_cache = {}
        self._ticker_cache_time = 0
        self._cache_ttl = 10  # 10 seconds cache
        
    def get_top_symbols(self, limit: int = 100, min_volume: float = 20000000) -> List[Dict]:
        """Get top symbols by 24h volume"""
        tickers = self._get_tickers()
        
        # Prefixes/patterns to EXCLUDE (leveraged, premium, problematic tokens)
        EXCLUDED_PREFIXES = (
            '1000',      # 1000PEPE, 1000SHIB, etc. (denomination tokens)
            '1M',        # 1MBABYDOGE, etc.
            '10000',     # 10000SATS, etc.
        )
        
        EXCLUDED_SUFFIXES = (
            'DOWNUSDT',  # Short leveraged tokens
            'UPUSDT',    # Long leveraged tokens
            'BULLUSDT',
            'BEARUSDT',
        )
        
        # Low-liquidity / problematic / delivering tokens to exclude
        EXCLUDED_SYMBOLS = {
            # Low liquidity / delisting candidates
            'BSWUSDT', 'UNFIUSDT', 'RENUSDT', 'STRKUSDT', 'VIDTUSDT',
            'AGIXUSDT', 'ALPHAUSDT', 'UNAUSDT', 'LEVERUSDT', 'MDTUSDT',
            'OAXUSDT', 'RAREUSDT', 'VGXUSDT', 'FIROUSDT', 'MOBUSDT',
            'OOKIUSDT', 'FORTHUSDT', 'BONDUSDT', 'MLNUSDT', 'AMBUSDT',
            # Meme coins with unreliable data
            'NEIROETHUSDT', 'MEMEFUSDT', 'MEMEFIUSDT',
            # Symbols in delivering/settling state (API error -4108)
            'PORT3USDT', 'UXLINKUSDT', 'LTAUSDT', 'FTMUSDT',
        }
        
        # Filter and sort by volume
        usdt_tickers = []
        for t in tickers:
            symbol = t.get('symbol', '')
            if not symbol.endswith('USDT'):
                continue
            
            # Skip leveraged/premium tokens
            if symbol.startswith(EXCLUDED_PREFIXES):
                continue
            if symbol.endswith(EXCLUDED_SUFFIXES):
                continue
            
            # Skip known problematic symbols
            if symbol in EXCLUDED_SYMBOLS:
                continue
            
            try:
                volume_24h = float(t.get('turnover24h', 0))
                if volume_24h >= min_volume:
                    usdt_tickers.append({
                        'symbol': symbol,
                        'volume_24h': volume_24h,
                        'last_price': float(t.get('lastPrice', 0)),
                        'price_change_24h': float(t.get('price24hPcnt', 0)) * 100,
                        'high_24h': float(t.get('highPrice24h', 0)),
                        'low_24h': float(t.get('lowPrice24h', 0)),
                        'bid': float(t.get('bid1Price', 0)),
                        'ask': float(t.get('ask1Price', 0)),
                    })
            except (ValueError, TypeError):
                continue
        
        # Sort by volume descending
        usdt_tickers.sort(key=lambda x: x['volume_24h'], reverse=True)
        return usdt_tickers[:limit]
    
    def _get_tickers(self) -> List[Dict]:
        """Get tickers with caching"""
        now = time.time()
        if now - self._ticker_cache_time < self._cache_ttl and self._ticker_cache:
            return self._ticker_cache
        
        self._ticker_cache = self.connector.get_tickers()
        self._ticker_cache_time = now
        return self._ticker_cache
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get single ticker data for symbol"""
        return self.connector.get_ticker(symbol)
    
    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> List[Dict]:
        """Get candlestick data"""
        return self.connector.get_klines(symbol, interval, limit)
    
    def get_multi_tf_klines(self, symbol: str, 
                            timeframes: List[str] = ['4h', '15m', '5m', '1m'],
                            limits: Dict[str, int] = None) -> Dict[str, List[Dict]]:
        """Get klines for multiple timeframes"""
        limits = limits or {'4h': 100, '15m': 100, '5m': 100, '1m': 100}
        result = {}
        
        for tf in timeframes:
            limit = limits.get(tf, 100)
            try:
                result[tf] = self.get_klines(symbol, tf, limit)
                time.sleep(API_LIMITS['rate_limit_delay'])
            except Exception as e:
                print(f"Error fetching {tf} for {symbol}: {e}")
                result[tf] = []
        
        return result
    
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate"""
        try:
            data = self.connector.get_funding_rate(symbol)
            if data:
                # Binance connector returns 'funding_rate' key
                return float(data.get('funding_rate', data.get('fundingRate', 0)))
            return None
        except Exception as e:
            print(f"[MARKET_DATA] Error getting funding rate for {symbol}: {e}")
            return None
    
    def get_oi_change(self, symbol: str, hours: int = 4) -> Optional[float]:
        """Calculate OI change over specified hours"""
        try:
            # Binance OI history - отримуємо достатньо даних
            oi_data = self.connector.get_open_interest(symbol, '1h', hours + 2)
            
            if not oi_data or len(oi_data) < 2:
                # Fallback: спробуємо отримати поточний OI
                current_oi = self.connector.get_current_open_interest(symbol)
                if current_oi:
                    return 0.0  # Немає історії для порівняння
                return None
            
            # Binance повертає в хронологічному порядку (старі -> нові)
            # Не потрібно реверсувати
            old_oi = float(oi_data[0].get('open_interest', oi_data[0].get('openInterest', 0)))
            new_oi = float(oi_data[-1].get('open_interest', oi_data[-1].get('openInterest', 0)))
            
            if old_oi == 0:
                return None
            
            change_pct = ((new_oi - old_oi) / old_oi) * 100
            return change_pct
            
        except Exception as e:
            print(f"[MARKET_DATA] Error getting OI change for {symbol}: {e}")
            return None
    
    def get_oi_history(self, symbol: str, limit: int = 200, interval: str = '1h') -> List[Dict]:
        """Get OI history for accumulation analysis"""
        try:
            oi_data = self.connector.get_open_interest(symbol, interval, limit)
            if not oi_data:
                return []
            
            # Convert to standard format
            result = []
            for item in oi_data:
                result.append({
                    'timestamp': item.get('timestamp'),
                    'open_interest': float(item.get('open_interest', item.get('openInterest', 0)))
                })
            
            return result
        except:
            return []
    
    def get_orderbook_imbalance(self, symbol: str, depth: int = 25) -> Dict:
        """Calculate orderbook imbalance"""
        try:
            orderbook = self.connector.get_orderbook(symbol, depth)
            
            bids = orderbook.get('b', [])
            asks = orderbook.get('a', [])
            
            bid_volume = sum(float(b[1]) for b in bids)
            ask_volume = sum(float(a[1]) for a in asks)
            
            total = bid_volume + ask_volume
            if total == 0:
                return {'bid_pct': 50, 'ask_pct': 50, 'imbalance': 0}
            
            bid_pct = (bid_volume / total) * 100
            ask_pct = (ask_volume / total) * 100
            imbalance = bid_pct - ask_pct  # Positive = more buyers
            
            return {
                'bid_pct': bid_pct,
                'ask_pct': ask_pct,
                'imbalance': imbalance,
                'bid_volume': bid_volume,
                'ask_volume': ask_volume
            }
        except:
            return {'bid_pct': 50, 'ask_pct': 50, 'imbalance': 0}
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol trading info"""
        try:
            info = self.connector.get_instrument_info(symbol)
            if info:
                # Binance connector returns a dict directly, not a list
                return {
                    'symbol': info.get('symbol', symbol),
                    'tick_size': float(info.get('priceFilter', {}).get('tickSize', 0.01)),
                    'min_qty': float(info.get('lotSizeFilter', {}).get('minOrderQty', 0.001)),
                    'qty_step': float(info.get('lotSizeFilter', {}).get('qtyStep', 0.001)),
                    'max_leverage': 20,  # Binance default
                    'pricePrecision': info.get('pricePrecision', 2),
                    'quantityPrecision': info.get('quantityPrecision', 3),
                }
            return None
        except Exception as e:
            print(f"[MARKET_DATA] Error getting symbol info for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price"""
        tickers = self._get_tickers()
        for t in tickers:
            if t.get('symbol') == symbol:
                return float(t.get('lastPrice', 0))
        return None
    
    def batch_get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple symbols at once"""
        tickers = self._get_tickers()
        prices = {}
        
        ticker_map = {t.get('symbol'): float(t.get('lastPrice', 0)) for t in tickers}
        
        for symbol in symbols:
            prices[symbol] = ticker_map.get(symbol, 0)
        
        return prices


class DataCache:
    """Simple in-memory cache for market data"""
    
    def __init__(self, default_ttl: int = 60):
        self._cache = {}
        self._timestamps = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[any]:
        """Get cached value if not expired"""
        if key not in self._cache:
            return None
        
        if time.time() - self._timestamps.get(key, 0) > self.default_ttl:
            del self._cache[key]
            del self._timestamps[key]
            return None
        
        return self._cache[key]
    
    def set(self, key: str, value: any, ttl: int = None):
        """Set cached value"""
        self._cache[key] = value
        self._timestamps[key] = time.time()
    
    def delete(self, key: str):
        """Delete cached value"""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def clear(self):
        """Clear all cache"""
        self._cache.clear()
        self._timestamps.clear()


# Singleton instances
_fetcher_instance = None
_cache_instance = None

def get_fetcher() -> MarketDataFetcher:
    """Get market data fetcher instance"""
    global _fetcher_instance
    if _fetcher_instance is None:
        _fetcher_instance = MarketDataFetcher()
    return _fetcher_instance

def get_cache() -> DataCache:
    """Get data cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance
