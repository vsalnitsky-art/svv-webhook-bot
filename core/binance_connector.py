"""
Binance Connector - Futures API для сканування та аналізу
Використовується для: Sleeper Scanner, OB Scanner, Market Data
Торгівля залишається на Bybit
"""

import os
import time
import hmac
import hashlib
from typing import Optional, Dict, Any, List
from datetime import datetime
import requests

# Спробуємо імпортувати binance, якщо немає - працюємо через requests
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_LIB_AVAILABLE = True
except ImportError:
    BINANCE_LIB_AVAILABLE = False
    print("[BINANCE] Warning: python-binance not installed, using requests fallback")


class BinanceConnector:
    """
    Клієнт для роботи з Binance Futures API
    Використовується ТІЛЬКИ для сканування/аналізу, не для торгівлі
    """
    
    # API Endpoints
    FUTURES_BASE_URL = "https://fapi.binance.com"
    SPOT_BASE_URL = "https://api.binance.com"
    
    # Timeframe mapping (Bybit format -> Binance format)
    TIMEFRAME_MAP = {
        '1': '1m',
        '3': '3m', 
        '5': '5m',
        '15': '15m',
        '30': '30m',
        '60': '1h',
        '120': '2h',
        '240': '4h',
        '360': '6h',
        '720': '12h',
        'D': '1d',
        '1440': '1d',
        'W': '1w',
        'M': '1M',
        # Also accept Binance format directly
        '1m': '1m',
        '3m': '3m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        '6h': '6h',
        '12h': '12h',
        '1d': '1d',
        '1w': '1w',
        '1M': '1M',
    }
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """
        Ініціалізація конектора
        API ключі опціональні для публічних endpoints
        """
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY', '')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET', '')
        
        # Використовуємо бібліотеку python-binance якщо доступна
        if BINANCE_LIB_AVAILABLE and self.api_key:
            self.client = Client(self.api_key, self.api_secret)
            self._use_library = True
            print("[BINANCE] Initialized with python-binance library")
        else:
            self.client = None
            self._use_library = False
            print("[BINANCE] Initialized with requests (public API only)")
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.05  # 50ms between requests (safe margin)
        
        # Request session for better performance
        self._session = requests.Session()
        self._session.headers.update({
            'X-MBX-APIKEY': self.api_key
        })
    
    def _rate_limit(self):
        """Simple rate limiter"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _safe_float(self, value, default: float = 0.0) -> float:
        """Безпечне перетворення в float"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def _convert_timeframe(self, interval: str) -> str:
        """Конвертує таймфрейм з Bybit формату в Binance"""
        interval = str(interval).strip()
        # Try exact match first
        result = self.TIMEFRAME_MAP.get(interval)
        if result:
            return result
        # Try case-insensitive match (e.g. '4H' → '4h')
        lower = interval.lower()
        result = self.TIMEFRAME_MAP.get(lower)
        if result:
            return result
        print(f"[BINANCE] ⚠️ Unknown interval '{interval}', passing as-is")
        return interval
    
    def _make_request(self, endpoint: str, params: dict = None, base_url: str = None) -> Optional[Dict]:
        """Виконує HTTP запит до API"""
        self._rate_limit()
        
        url = (base_url or self.FUTURES_BASE_URL) + endpoint
        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[BINANCE] Request error: {e}")
            return None
    
    # ===== PUBLIC API - MARKET DATA =====
    
    def get_tickers(self) -> List[Dict]:
        """
        Отримати всі тікери Futures
        Повертає формат сумісний з Bybit для мінімальних змін в market_data.py
        """
        try:
            if self._use_library:
                tickers = self.client.futures_ticker()
            else:
                tickers = self._make_request("/fapi/v1/ticker/24hr")
            
            if not tickers:
                return []
            
            # Конвертуємо в Bybit-сумісний формат
            result = []
            for t in tickers:
                result.append({
                    'symbol': t.get('symbol', ''),
                    'lastPrice': t.get('lastPrice', '0'),
                    'turnover24h': t.get('quoteVolume', '0'),  # Bybit: turnover24h
                    'volume24h': t.get('volume', '0'),
                    'price24hPcnt': str(self._safe_float(t.get('priceChangePercent', 0)) / 100),  # Convert to decimal
                    'highPrice24h': t.get('highPrice', '0'),
                    'lowPrice24h': t.get('lowPrice', '0'),
                    'bid1Price': t.get('bidPrice', '0'),
                    'ask1Price': t.get('askPrice', '0'),
                    # Binance specific
                    'weightedAvgPrice': t.get('weightedAvgPrice', '0'),
                    'openPrice': t.get('openPrice', '0'),
                    'closeTime': t.get('closeTime', 0),
                })
            
            return result
            
        except Exception as e:
            print(f"[BINANCE] get_tickers error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Отримати тікер для одного символу"""
        try:
            if self._use_library:
                ticker = self.client.futures_ticker(symbol=symbol)
            else:
                ticker = self._make_request("/fapi/v1/ticker/24hr", {'symbol': symbol})
            
            if not ticker:
                return None
            
            # Handle both single ticker and list response
            if isinstance(ticker, list):
                ticker = ticker[0] if ticker else None
            
            if not ticker:
                return None
            
            return {
                'symbol': ticker.get('symbol', ''),
                'lastPrice': ticker.get('lastPrice', '0'),
                'turnover24h': ticker.get('quoteVolume', '0'),
                'volume24h': ticker.get('volume', '0'),
                'price24hPcnt': str(self._safe_float(ticker.get('priceChangePercent', 0)) / 100),
                'highPrice24h': ticker.get('highPrice', '0'),
                'lowPrice24h': ticker.get('lowPrice', '0'),
                'bid1Price': ticker.get('bidPrice', '0'),
                'ask1Price': ticker.get('askPrice', '0'),
            }
            
        except Exception as e:
            print(f"[BINANCE] get_ticker error for {symbol}: {e}")
            return None
    
    def get_price(self, symbol: str) -> float:
        """Отримати поточну ціну"""
        try:
            if self._use_library:
                price = self.client.futures_symbol_ticker(symbol=symbol)
                return self._safe_float(price.get('price'))
            else:
                data = self._make_request("/fapi/v1/ticker/price", {'symbol': symbol})
                return self._safe_float(data.get('price')) if data else 0.0
        except Exception as e:
            print(f"[BINANCE] get_price error: {e}")
            return 0.0
    
    def get_klines(self, symbol: str, interval: str = "60", limit: int = 200) -> List[Dict]:
        """
        Отримати свічки (klines)
        interval: Приймає як Bybit формат (60, 240, D) так і Binance (1h, 4h, 1d)
        """
        try:
            binance_interval = self._convert_timeframe(interval)
            
            # Debug: log first call per interval to help trace Invalid interval errors
            _debug_key = f"_logged_{binance_interval}"
            if not getattr(self, _debug_key, False):
                print(f"[BINANCE] get_klines interval: '{interval}' → '{binance_interval}'")
                setattr(self, _debug_key, True)
            
            if self._use_library:
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=binance_interval,
                    limit=limit
                )
            else:
                klines = self._make_request("/fapi/v1/klines", {
                    'symbol': symbol,
                    'interval': binance_interval,
                    'limit': limit
                })
            
            if not klines:
                return []
            
            # Конвертуємо в стандартний формат (Bybit-сумісний)
            # Binance klines: [open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore]
            result = []
            for k in klines:
                result.append({
                    'timestamp': int(k[0]),
                    'open': self._safe_float(k[1]),
                    'high': self._safe_float(k[2]),
                    'low': self._safe_float(k[3]),
                    'close': self._safe_float(k[4]),
                    'volume': self._safe_float(k[5]),
                    'turnover': self._safe_float(k[7]),  # quote_volume
                    'close_time': int(k[6]),
                    'trades': int(k[8]),
                    'taker_buy_volume': self._safe_float(k[9]),
                    'taker_buy_quote_volume': self._safe_float(k[10]),
                })
            
            return result
            
        except Exception as e:
            print(f"[BINANCE] get_klines error for {symbol}: {e}")
            return []
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """
        Отримати orderbook
        Повертає у Bybit-сумісному форматі
        """
        try:
            if self._use_library:
                book = self.client.futures_order_book(symbol=symbol, limit=limit)
            else:
                book = self._make_request("/fapi/v1/depth", {
                    'symbol': symbol,
                    'limit': limit
                })
            
            if not book:
                return None
            
            # Конвертуємо в Bybit формат: 'b' для bids, 'a' для asks
            return {
                'b': book.get('bids', []),  # [[price, qty], ...]
                'a': book.get('asks', []),
                # Також зберігаємо оригінальні ключі для сумісності
                'bids': book.get('bids', []),
                'asks': book.get('asks', []),
                'lastUpdateId': book.get('lastUpdateId'),
            }
            
        except Exception as e:
            print(f"[BINANCE] get_orderbook error: {e}")
            return None
    
    def get_instrument_info(self, symbol: str) -> Optional[Dict]:
        """Отримати інформацію про інструмент"""
        try:
            if self._use_library:
                info = self.client.futures_exchange_info()
            else:
                info = self._make_request("/fapi/v1/exchangeInfo")
            
            if not info:
                return None
            
            # Знаходимо потрібний символ
            for s in info.get('symbols', []):
                if s.get('symbol') == symbol:
                    # Витягуємо фільтри
                    price_filter = next((f for f in s.get('filters', []) if f['filterType'] == 'PRICE_FILTER'), {})
                    lot_filter = next((f for f in s.get('filters', []) if f['filterType'] == 'LOT_SIZE'), {})
                    
                    return {
                        'symbol': s.get('symbol'),
                        'status': s.get('status'),
                        'baseAsset': s.get('baseAsset'),
                        'quoteAsset': s.get('quoteAsset'),
                        'pricePrecision': s.get('pricePrecision'),
                        'quantityPrecision': s.get('quantityPrecision'),
                        # Bybit-сумісний формат
                        'priceFilter': {
                            'tickSize': price_filter.get('tickSize', '0.01'),
                            'minPrice': price_filter.get('minPrice', '0'),
                            'maxPrice': price_filter.get('maxPrice', '0'),
                        },
                        'lotSizeFilter': {
                            'minOrderQty': lot_filter.get('minQty', '0'),
                            'maxOrderQty': lot_filter.get('maxQty', '0'),
                            'qtyStep': lot_filter.get('stepSize', '0'),
                        },
                    }
            
            return None
            
        except Exception as e:
            print(f"[BINANCE] get_instrument_info error: {e}")
            return None
    
    # ===== FUTURES SPECIFIC =====
    
    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Отримати поточний funding rate
        """
        try:
            if self._use_library:
                # Отримуємо premium index який містить funding rate
                data = self.client.futures_mark_price(symbol=symbol)
            else:
                data = self._make_request("/fapi/v1/premiumIndex", {'symbol': symbol})
            
            if not data:
                return None
            
            # Handle list response
            if isinstance(data, list):
                data = data[0] if data else None
            
            if not data:
                return None
            
            return {
                'symbol': symbol,
                'funding_rate': self._safe_float(data.get('lastFundingRate', 0)),
                'fundingRate': self._safe_float(data.get('lastFundingRate', 0)),  # Bybit compat
                'next_funding_time': data.get('nextFundingTime'),
                'mark_price': self._safe_float(data.get('markPrice', 0)),
                'index_price': self._safe_float(data.get('indexPrice', 0)),
            }
            
        except Exception as e:
            print(f"[BINANCE] get_funding_rate error: {e}")
            return None
    
    def get_open_interest(self, symbol: str, interval: str = "1h", limit: int = 50) -> List[Dict]:
        """
        Отримати історію open interest
        
        ВАЖЛИВО: Binance має обмеження на історію OI
        interval: 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d
        limit: max 500, доступно за останні 30 днів
        """
        try:
            # Конвертуємо interval
            binance_interval = self._convert_timeframe(interval)
            
            # Валідні інтервали для OI
            valid_intervals = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d']
            if binance_interval not in valid_intervals:
                binance_interval = '1h'  # fallback
            
            if self._use_library:
                data = self.client.futures_open_interest_hist(
                    symbol=symbol,
                    period=binance_interval,
                    limit=min(limit, 500)
                )
            else:
                data = self._make_request("/futures/data/openInterestHist", {
                    'symbol': symbol,
                    'period': binance_interval,
                    'limit': min(limit, 500)
                })
            
            if not data:
                return []
            
            # Конвертуємо в стандартний формат
            result = []
            for item in data:
                result.append({
                    'timestamp': int(item.get('timestamp', 0)),
                    'open_interest': self._safe_float(item.get('sumOpenInterest', 0)),
                    'openInterest': self._safe_float(item.get('sumOpenInterest', 0)),  # Bybit compat
                    'open_interest_value': self._safe_float(item.get('sumOpenInterestValue', 0)),
                })
            
            return result
            
        except Exception as e:
            error_str = str(e)
            # Code -4108: Symbol is on delivering/delivered/settling/closed/pre-trading
            if '-4108' in error_str:
                return []
            print(f"[BINANCE] get_open_interest error: {e}")
            return []
    
    def get_current_open_interest(self, symbol: str) -> Optional[Dict]:
        """Отримати поточний OI (не історію)"""
        try:
            if self._use_library:
                data = self.client.futures_open_interest(symbol=symbol)
            else:
                data = self._make_request("/fapi/v1/openInterest", {'symbol': symbol})
            
            if not data:
                return None
            
            return {
                'symbol': symbol,
                'open_interest': self._safe_float(data.get('openInterest', 0)),
                'time': data.get('time', 0),
            }
            
        except Exception as e:
            error_str = str(e)
            # Code -4108: Symbol is on delivering/delivered/settling/closed/pre-trading
            # This is normal for some futures contracts, just skip silently
            if '-4108' in error_str:
                return None
            print(f"[BINANCE] get_current_open_interest error: {e}")
            return None
    
    def get_long_short_ratio(self, symbol: str, period: str = "1h", limit: int = 30) -> List[Dict]:
        """
        Отримати співвідношення Long/Short позицій
        Корисно для аналізу настроїв ринку
        """
        try:
            if self._use_library:
                data = self.client.futures_global_longshort_ratio(
                    symbol=symbol,
                    period=period,
                    limit=limit
                )
            else:
                data = self._make_request("/futures/data/globalLongShortAccountRatio", {
                    'symbol': symbol,
                    'period': period,
                    'limit': limit
                })
            
            if not data:
                return []
            
            return [
                {
                    'timestamp': int(item.get('timestamp', 0)),
                    'long_ratio': self._safe_float(item.get('longAccount', 0)),
                    'short_ratio': self._safe_float(item.get('shortAccount', 0)),
                    'long_short_ratio': self._safe_float(item.get('longShortRatio', 0)),
                }
                for item in data
            ]
            
        except Exception as e:
            print(f"[BINANCE] get_long_short_ratio error: {e}")
            return []
    
    def get_top_trader_positions(self, symbol: str, period: str = "1h", limit: int = 30) -> List[Dict]:
        """
        Позиції топ-трейдерів (accounts)
        """
        try:
            if self._use_library:
                data = self.client.futures_top_longshort_account_ratio(
                    symbol=symbol,
                    period=period,
                    limit=limit
                )
            else:
                data = self._make_request("/futures/data/topLongShortAccountRatio", {
                    'symbol': symbol,
                    'period': period,
                    'limit': limit
                })
            
            if not data:
                return []
            
            return [
                {
                    'timestamp': int(item.get('timestamp', 0)),
                    'long_ratio': self._safe_float(item.get('longAccount', 0)),
                    'short_ratio': self._safe_float(item.get('shortAccount', 0)),
                    'long_short_ratio': self._safe_float(item.get('longShortRatio', 0)),
                }
                for item in data
            ]
            
        except Exception as e:
            print(f"[BINANCE] get_top_trader_positions error: {e}")
            return []
    
    def get_taker_buy_sell_volume(self, symbol: str, period: str = "1h", limit: int = 30) -> List[Dict]:
        """
        Taker Buy/Sell Volume - показує агресивність покупців/продавців
        """
        try:
            if self._use_library:
                data = self.client.futures_taker_long_short_ratio(
                    symbol=symbol,
                    period=period,
                    limit=limit
                )
            else:
                data = self._make_request("/futures/data/takerlongshortRatio", {
                    'symbol': symbol,
                    'period': period,
                    'limit': limit
                })
            
            if not data:
                return []
            
            return [
                {
                    'timestamp': int(item.get('timestamp', 0)),
                    'buy_sell_ratio': self._safe_float(item.get('buySellRatio', 0)),
                    'buy_vol': self._safe_float(item.get('buyVol', 0)),
                    'sell_vol': self._safe_float(item.get('sellVol', 0)),
                }
                for item in data
            ]
            
        except Exception as e:
            print(f"[BINANCE] get_taker_buy_sell_volume error: {e}")
            return []
    
    # ===== UTILITY METHODS =====
    
    def get_all_symbols(self) -> List[str]:
        """Отримати список всіх USDT-M Futures символів"""
        try:
            if self._use_library:
                info = self.client.futures_exchange_info()
            else:
                info = self._make_request("/fapi/v1/exchangeInfo")
            
            if not info:
                return []
            
            symbols = [
                s['symbol'] for s in info.get('symbols', [])
                if s.get('status') == 'TRADING' and s.get('quoteAsset') == 'USDT'
            ]
            
            return symbols
            
        except Exception as e:
            print(f"[BINANCE] get_all_symbols error: {e}")
            return []
    
    def test_connection(self) -> bool:
        """Перевірка з'єднання з API"""
        try:
            if self._use_library:
                self.client.futures_ping()
            else:
                result = self._make_request("/fapi/v1/ping")
                if result is None:
                    return False
            return True
        except Exception as e:
            print(f"[BINANCE] Connection test failed: {e}")
            return False
    
    def get_server_time(self) -> Optional[int]:
        """Отримати час сервера"""
        try:
            if self._use_library:
                return self.client.futures_time().get('serverTime')
            else:
                data = self._make_request("/fapi/v1/time")
                return data.get('serverTime') if data else None
        except Exception as e:
            print(f"[BINANCE] get_server_time error: {e}")
            return None


# ===== Singleton =====
_connector: Optional[BinanceConnector] = None

def get_binance_connector() -> BinanceConnector:
    """Отримати singleton instance"""
    global _connector
    if _connector is None:
        _connector = BinanceConnector()
    return _connector

def reset_connector():
    """Скинути конектор (для тестів або переконфігурації)"""
    global _connector
    _connector = None
