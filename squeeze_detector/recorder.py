#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SQUEEZE DETECTOR v1.0 - RECORDER                         ║
║                                                                              ║
║  Компонент для збору ринкових даних:                                         ║
║  • WebSocket - real-time оновлення tickers                                   ║
║  • REST API - історичні дані OI, Funding                                     ║
║  • Hybrid mode - REST для історії + WebSocket для real-time                  ║
║                                                                              ║
║  Bybit API v5:                                                               ║
║  • GET /v5/market/tickers - поточні ціни                                     ║
║  • GET /v5/market/open-interest - Open Interest                              ║
║  • GET /v5/market/funding/history - Funding Rate                             ║
║                                                                              ║
║  Автор: SVV Webhook Bot                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import json
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import traceback

logger = logging.getLogger(__name__)

# === SAFE MODEL IMPORTS ===
try:
    from .models import MarketSnapshot
except ImportError:
    try:
        from squeeze_detector.models import MarketSnapshot
    except ImportError:
        logger.error("Failed to import MarketSnapshot")
        MarketSnapshot = None

# WebSocket
try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    websocket = None

# HTTP requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

# Pybit (optional, for authenticated requests)
try:
    from pybit.unified_trading import HTTP
    HAS_PYBIT = True
except ImportError:
    HAS_PYBIT = False
    HTTP = None


# ============================================================================
#                              CONSTANTS
# ============================================================================

# Bybit API endpoints
BYBIT_REST_URL = "https://api.bybit.com"
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"

# API Limits
MAX_SYMBOLS_PER_WS = 10  # Bybit рекомендує не більше 10 підписок на з'єднання
MAX_OI_LIMIT = 200       # Максимум записів OI за запит
MAX_FUNDING_LIMIT = 200  # Максимум записів Funding за запит

# Timeouts
REST_TIMEOUT = 10        # секунд
WS_PING_INTERVAL = 20    # секунд
WS_RECONNECT_DELAY = 5   # секунд

# Rate limiting
REST_RATE_LIMIT_DELAY = 0.1  # 100ms між запитами


# ============================================================================
#                              DATA CLASSES
# ============================================================================

@dataclass
class TickerData:
    """Дані тікера з Bybit"""
    symbol: str
    last_price: float = 0.0
    mark_price: float = 0.0
    index_price: float = 0.0
    bid1_price: float = 0.0
    bid1_size: float = 0.0
    ask1_price: float = 0.0
    ask1_size: float = 0.0
    volume_24h: float = 0.0
    turnover_24h: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    price_change_24h: float = 0.0
    open_interest: float = 0.0
    open_interest_value: float = 0.0
    funding_rate: float = 0.0
    next_funding_time: Optional[datetime] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def spread_percent(self) -> float:
        """Розраховує спред у відсотках"""
        if self.bid1_price > 0 and self.ask1_price > 0:
            return ((self.ask1_price - self.bid1_price) / self.bid1_price) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'last_price': self.last_price,
            'mark_price': self.mark_price,
            'index_price': self.index_price,
            'bid1_price': self.bid1_price,
            'bid1_size': self.bid1_size,
            'ask1_price': self.ask1_price,
            'ask1_size': self.ask1_size,
            'volume_24h': self.volume_24h,
            'turnover_24h': self.turnover_24h,
            'high_24h': self.high_24h,
            'low_24h': self.low_24h,
            'price_change_24h': self.price_change_24h,
            'open_interest': self.open_interest,
            'open_interest_value': self.open_interest_value,
            'funding_rate': self.funding_rate,
            'next_funding_time': self.next_funding_time.isoformat() if self.next_funding_time else None,
            'spread_percent': self.spread_percent,
            'timestamp': self.timestamp.isoformat(),
        }


@dataclass
class OIData:
    """Open Interest дані"""
    symbol: str
    timestamp: datetime
    open_interest: float      # В монетах
    open_interest_value: float  # В USD
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open_interest': self.open_interest,
            'open_interest_value': self.open_interest_value,
        }


@dataclass 
class FundingData:
    """Funding Rate дані"""
    symbol: str
    timestamp: datetime
    funding_rate: float
    funding_rate_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'funding_rate': self.funding_rate,
        }


# ============================================================================
#                              REST API CLIENT
# ============================================================================

class BybitRestClient:
    """
    REST API клієнт для Bybit v5.
    Використовується для:
    - Отримання списку монет
    - Історичних даних OI
    - Історичних даних Funding
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = BYBIT_REST_URL
        self.session = requests.Session() if HAS_REQUESTS else None
        
        # Rate limiting
        self._last_request_time = 0
        self._request_lock = threading.Lock()
        
        # Pybit session (для authenticated endpoints)
        self.pybit_session = None
        if HAS_PYBIT and api_key and api_secret:
            try:
                self.pybit_session = HTTP(
                    testnet=False,
                    api_key=api_key,
                    api_secret=api_secret,
                )
                logger.info("✅ Pybit session initialized")
            except Exception as e:
                logger.warning(f"⚠️ Pybit init failed: {e}")
    
    def _rate_limit(self):
        """Простий rate limiter"""
        with self._request_lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < REST_RATE_LIMIT_DELAY:
                time.sleep(REST_RATE_LIMIT_DELAY - elapsed)
            self._last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Виконує GET запит до Bybit API"""
        if not self.session:
            logger.error("Requests library not available")
            return None
        
        self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=REST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('retCode') != 0:
                logger.warning(f"API error: {data.get('retMsg')} for {endpoint}")
                return None
            
            return data.get('result', {})
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout for {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error for {endpoint}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error for {endpoint}: {e}")
            return None
    
    def get_usdt_perpetuals(self, min_volume: float = 5_000_000) -> List[Dict]:
        """
        Отримує список USDT Perpetual пар відсортованих за об'ємом.
        
        Args:
            min_volume: Мінімальний 24h об'єм в USD
            
        Returns:
            List[Dict] з інформацією про монети
        """
        # limit=1000 щоб отримати ВСІ тікери (Bybit за замовчуванням може повертати менше)
        result = self._request("/v5/market/tickers", {"category": "linear", "limit": 1000})
        
        if not result:
            return []
        
        tickers = result.get('list', [])
        
        symbols = []
        for t in tickers:
            symbol = t.get('symbol', '')
            
            # Фільтруємо тільки USDT пари
            if not symbol.endswith('USDT'):
                continue
            
            try:
                turnover = float(t.get('turnover24h', 0))
                
                if turnover >= min_volume:
                    symbols.append({
                        'symbol': symbol,
                        'last_price': float(t.get('lastPrice', 0)),
                        'turnover_24h': turnover,
                        'volume_24h': float(t.get('volume24h', 0)),
                        'price_change_24h': float(t.get('price24hPcnt', 0)) * 100,
                        'open_interest': float(t.get('openInterest', 0)),
                        'open_interest_value': float(t.get('openInterestValue', 0)),
                        'funding_rate': float(t.get('fundingRate', 0)),
                        'bid1_price': float(t.get('bid1Price', 0)),
                        'ask1_price': float(t.get('ask1Price', 0)),
                    })
            except (ValueError, TypeError) as e:
                logger.debug(f"Parse error for {symbol}: {e}")
                continue
        
        # Сортуємо за об'ємом (найбільші першими)
        symbols.sort(key=lambda x: x['turnover_24h'], reverse=True)
        
        return symbols
    
    def get_ticker(self, symbol: str) -> Optional[TickerData]:
        """Отримує поточні дані тікера"""
        result = self._request("/v5/market/tickers", {
            "category": "linear",
            "symbol": symbol
        })
        
        if not result:
            return None
        
        tickers = result.get('list', [])
        if not tickers:
            return None
        
        t = tickers[0]
        
        try:
            # Parse next funding time
            next_funding = None
            nft = t.get('nextFundingTime')
            if nft:
                try:
                    next_funding = datetime.fromtimestamp(int(nft) / 1000)
                except (ValueError, TypeError):
                    pass
            
            return TickerData(
                symbol=symbol,
                last_price=float(t.get('lastPrice', 0)),
                mark_price=float(t.get('markPrice', 0)),
                index_price=float(t.get('indexPrice', 0)),
                bid1_price=float(t.get('bid1Price', 0)),
                bid1_size=float(t.get('bid1Size', 0)),
                ask1_price=float(t.get('ask1Price', 0)),
                ask1_size=float(t.get('ask1Size', 0)),
                volume_24h=float(t.get('volume24h', 0)),
                turnover_24h=float(t.get('turnover24h', 0)),
                high_24h=float(t.get('highPrice24h', 0)),
                low_24h=float(t.get('lowPrice24h', 0)),
                price_change_24h=float(t.get('price24hPcnt', 0)) * 100,
                open_interest=float(t.get('openInterest', 0)),
                open_interest_value=float(t.get('openInterestValue', 0)),
                funding_rate=float(t.get('fundingRate', 0)),
                next_funding_time=next_funding,
                timestamp=datetime.utcnow(),
            )
        except (ValueError, TypeError) as e:
            logger.warning(f"Parse ticker error for {symbol}: {e}")
            return None
    
    def get_all_tickers(self) -> Dict[str, TickerData]:
        """Отримує всі тікери за один запит"""
        # limit=1000 щоб отримати ВСІ тікери
        result = self._request("/v5/market/tickers", {"category": "linear", "limit": 1000})
        
        if not result:
            return {}
        
        tickers = {}
        for t in result.get('list', []):
            symbol = t.get('symbol', '')
            if not symbol.endswith('USDT'):
                continue
            
            try:
                next_funding = None
                nft = t.get('nextFundingTime')
                if nft:
                    try:
                        next_funding = datetime.fromtimestamp(int(nft) / 1000)
                    except (ValueError, TypeError):
                        pass
                
                tickers[symbol] = TickerData(
                    symbol=symbol,
                    last_price=float(t.get('lastPrice', 0)),
                    mark_price=float(t.get('markPrice', 0)),
                    index_price=float(t.get('indexPrice', 0)),
                    bid1_price=float(t.get('bid1Price', 0)),
                    bid1_size=float(t.get('bid1Size', 0)),
                    ask1_price=float(t.get('ask1Price', 0)),
                    ask1_size=float(t.get('ask1Size', 0)),
                    volume_24h=float(t.get('volume24h', 0)),
                    turnover_24h=float(t.get('turnover24h', 0)),
                    high_24h=float(t.get('highPrice24h', 0)),
                    low_24h=float(t.get('lowPrice24h', 0)),
                    price_change_24h=float(t.get('price24hPcnt', 0)) * 100,
                    open_interest=float(t.get('openInterest', 0)),
                    open_interest_value=float(t.get('openInterestValue', 0)),
                    funding_rate=float(t.get('fundingRate', 0)),
                    next_funding_time=next_funding,
                    timestamp=datetime.utcnow(),
                )
            except (ValueError, TypeError):
                continue
        
        return tickers
    
    def get_open_interest_history(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 48
    ) -> List[OIData]:
        """
        Отримує історію Open Interest.
        
        Args:
            symbol: Торгова пара
            interval: 5min, 15min, 30min, 1h, 4h, 1d
            limit: Кількість записів (max 200)
            
        Returns:
            List[OIData] відсортований від старих до нових
        """
        result = self._request("/v5/market/open-interest", {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": interval,
            "limit": min(limit, MAX_OI_LIMIT)
        })
        
        if not result:
            return []
        
        data_list = result.get('list', [])
        
        oi_history = []
        for item in data_list:
            try:
                timestamp = datetime.fromtimestamp(int(item.get('timestamp', 0)) / 1000)
                oi_history.append(OIData(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_interest=float(item.get('openInterest', 0)),
                    open_interest_value=float(item.get('openInterestValue', 0)),
                ))
            except (ValueError, TypeError) as e:
                logger.debug(f"OI parse error: {e}")
                continue
        
        # Bybit повертає від нових до старих, реверсуємо
        oi_history.reverse()
        
        return oi_history
    
    def get_funding_history(
        self,
        symbol: str,
        limit: int = 48
    ) -> List[FundingData]:
        """
        Отримує історію Funding Rate.
        
        Args:
            symbol: Торгова пара
            limit: Кількість записів (max 200)
            
        Returns:
            List[FundingData] відсортований від старих до нових
        """
        result = self._request("/v5/market/funding/history", {
            "category": "linear",
            "symbol": symbol,
            "limit": min(limit, MAX_FUNDING_LIMIT)
        })
        
        if not result:
            return []
        
        data_list = result.get('list', [])
        
        funding_history = []
        for item in data_list:
            try:
                timestamp = datetime.fromtimestamp(int(item.get('fundingRateTimestamp', 0)) / 1000)
                funding_history.append(FundingData(
                    symbol=symbol,
                    timestamp=datetime.utcnow(),
                    funding_rate=float(item.get('fundingRate', 0)),
                    funding_rate_timestamp=timestamp,
                ))
            except (ValueError, TypeError) as e:
                logger.debug(f"Funding parse error: {e}")
                continue
        
        # Bybit повертає від нових до старих, реверсуємо
        funding_history.reverse()
        
        return funding_history
    
    def get_long_short_ratio(
        self,
        symbol: str,
        period: str = "1h",
        limit: int = 24
    ) -> Optional[Dict]:
        """
        Отримує Long/Short Ratio.
        
        Args:
            symbol: Торгова пара
            period: 5min, 15min, 30min, 1h, 4h, 1d
            limit: Кількість записів
        """
        result = self._request("/v5/market/account-ratio", {
            "category": "linear",
            "symbol": symbol,
            "period": period,
            "limit": limit
        })
        
        if not result:
            return None
        
        data_list = result.get('list', [])
        if not data_list:
            return None
        
        # Повертаємо найсвіжіший
        latest = data_list[0]
        
        try:
            buy_ratio = float(latest.get('buyRatio', 0.5))
            sell_ratio = float(latest.get('sellRatio', 0.5))
            
            return {
                'symbol': symbol,
                'buy_ratio': buy_ratio,
                'sell_ratio': sell_ratio,
                'ls_ratio': buy_ratio / sell_ratio if sell_ratio > 0 else 1.0,
                'timestamp': datetime.fromtimestamp(int(latest.get('timestamp', 0)) / 1000),
            }
        except (ValueError, TypeError):
            return None


# ============================================================================
#                              WEBSOCKET CLIENT
# ============================================================================

class BybitWebSocketClient:
    """
    WebSocket клієнт для real-time даних.
    
    Підписується на:
    - tickers.{symbol} - ціни, OI, funding
    """
    
    def __init__(
        self,
        symbols: List[str],
        on_ticker: Callable[[TickerData], None] = None,
        on_error: Callable[[Exception], None] = None,
    ):
        self.symbols = symbols
        self.on_ticker = on_ticker
        self.on_error = on_error
        
        self.ws: Optional[websocket.WebSocketApp] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._reconnect_count = 0
        self._max_reconnects = 10
        
        # Останні отримані дані
        self._latest_tickers: Dict[str, TickerData] = {}
        self._lock = threading.Lock()
    
    def start(self):
        """Запускає WebSocket з'єднання в окремому потоці"""
        if not HAS_WEBSOCKET:
            logger.error("websocket-client library not installed")
            return False
        
        if self._running:
            logger.warning("WebSocket already running")
            return True
        
        self._running = True
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()
        
        logger.info(f"🔌 WebSocket started for {len(self.symbols)} symbols")
        return True
    
    def stop(self):
        """Зупиняє WebSocket"""
        self._running = False
        
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        
        logger.info("🔌 WebSocket stopped")
    
    def _run_forever(self):
        """Основний цикл з reconnect логікою"""
        while self._running and self._reconnect_count < self._max_reconnects:
            try:
                self._connect()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if self.on_error:
                    self.on_error(e)
            
            if self._running:
                self._reconnect_count += 1
                logger.info(f"🔄 Reconnecting ({self._reconnect_count}/{self._max_reconnects})...")
                time.sleep(WS_RECONNECT_DELAY)
        
        if self._reconnect_count >= self._max_reconnects:
            logger.error("❌ Max reconnects reached")
    
    def _connect(self):
        """Створює WebSocket з'єднання"""
        self.ws = websocket.WebSocketApp(
            BYBIT_WS_URL,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_ws_error,
            on_close=self._on_close,
        )
        
        # Run with ping/pong
        self.ws.run_forever(
            ping_interval=WS_PING_INTERVAL,
            ping_timeout=10,
        )
    
    def _on_open(self, ws):
        """Викликається при відкритті з'єднання"""
        logger.info("✅ WebSocket connected")
        self._reconnect_count = 0
        
        # Підписуємось на тікери
        # Bybit дозволяє підписатись на декілька символів одразу
        subscribe_msg = {
            "op": "subscribe",
            "args": [f"tickers.{symbol}" for symbol in self.symbols]
        }
        
        ws.send(json.dumps(subscribe_msg))
        logger.info(f"📡 Subscribed to {len(self.symbols)} tickers")
    
    def _on_message(self, ws, message: str):
        """Обробляє вхідні повідомлення"""
        try:
            data = json.loads(message)
            
            # Перевіряємо тип повідомлення
            topic = data.get('topic', '')
            
            if topic.startswith('tickers.'):
                self._handle_ticker(data)
            elif data.get('success') is False:
                logger.warning(f"WS error: {data.get('ret_msg')}")
            elif data.get('op') == 'subscribe':
                if data.get('success'):
                    logger.debug("Subscription confirmed")
                    
        except json.JSONDecodeError as e:
            logger.warning(f"WS JSON error: {e}")
        except Exception as e:
            logger.error(f"WS message error: {e}")
    
    def _handle_ticker(self, data: Dict):
        """Обробляє ticker update"""
        try:
            ticker_data = data.get('data', {})
            symbol = ticker_data.get('symbol', '')
            
            if not symbol:
                return
            
            # Parse next funding time
            next_funding = None
            nft = ticker_data.get('nextFundingTime')
            if nft:
                try:
                    next_funding = datetime.fromtimestamp(int(nft) / 1000)
                except (ValueError, TypeError):
                    pass
            
            ticker = TickerData(
                symbol=symbol,
                last_price=float(ticker_data.get('lastPrice', 0)),
                mark_price=float(ticker_data.get('markPrice', 0)),
                index_price=float(ticker_data.get('indexPrice', 0)),
                bid1_price=float(ticker_data.get('bid1Price', 0)),
                bid1_size=float(ticker_data.get('bid1Size', 0)),
                ask1_price=float(ticker_data.get('ask1Price', 0)),
                ask1_size=float(ticker_data.get('ask1Size', 0)),
                volume_24h=float(ticker_data.get('volume24h', 0)),
                turnover_24h=float(ticker_data.get('turnover24h', 0)),
                high_24h=float(ticker_data.get('highPrice24h', 0)),
                low_24h=float(ticker_data.get('lowPrice24h', 0)),
                price_change_24h=float(ticker_data.get('price24hPcnt', 0)) * 100,
                open_interest=float(ticker_data.get('openInterest', 0)),
                open_interest_value=float(ticker_data.get('openInterestValue', 0)),
                funding_rate=float(ticker_data.get('fundingRate', 0)),
                next_funding_time=next_funding,
                timestamp=datetime.utcnow(),
            )
            
            # Зберігаємо
            with self._lock:
                self._latest_tickers[symbol] = ticker
            
            # Callback
            if self.on_ticker:
                self.on_ticker(ticker)
                
        except (ValueError, TypeError) as e:
            logger.debug(f"Ticker parse error: {e}")
    
    def _on_ws_error(self, ws, error):
        """Обробляє WebSocket помилки"""
        logger.warning(f"WS error: {error}")
        if self.on_error:
            self.on_error(error)
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Викликається при закритті з'єднання"""
        logger.info(f"🔌 WebSocket closed: {close_status_code} - {close_msg}")
    
    def get_latest_ticker(self, symbol: str) -> Optional[TickerData]:
        """Повертає останні дані для символу"""
        with self._lock:
            return self._latest_tickers.get(symbol)
    
    def get_all_latest_tickers(self) -> Dict[str, TickerData]:
        """Повертає всі останні дані"""
        with self._lock:
            return self._latest_tickers.copy()


# ============================================================================
#                              DATA RECORDER
# ============================================================================

class DataRecorder:
    """
    Головний клас для запису ринкових даних.
    
    Використовує гібридний підхід:
    - REST API для періодичного збору повних даних
    - WebSocket для real-time оновлень (опціонально)
    """
    
    def __init__(
        self,
        db_session_factory,
        api_key: str = None,
        api_secret: str = None,
        use_websocket: bool = False,
    ):
        self.db_session_factory = db_session_factory
        self.rest_client = BybitRestClient(api_key, api_secret)
        self.ws_client: Optional[BybitWebSocketClient] = None
        self.use_websocket = use_websocket
        
        # Список монет для моніторингу
        self.monitored_symbols: List[str] = []
        
        # Статистика
        self.stats = {
            'snapshots_recorded': 0,
            'last_record_time': None,
            'errors': 0,
        }
        
        # Recording state
        self._recording = False
        self._record_thread: Optional[threading.Thread] = None
        self._record_interval = 300  # 5 хвилин
        
        logger.info("📊 DataRecorder initialized")
    
    def set_symbols(self, symbols: List[str]):
        """Встановлює список символів для моніторингу"""
        # Підтримуємо до 500 монет (для Aggressive mode 400)
        self.monitored_symbols = symbols[:500]
        logger.info(f"📋 Monitoring {len(self.monitored_symbols)} symbols")
        
        # Перезапускаємо WebSocket якщо активний
        if self.ws_client:
            self.ws_client.stop()
            self.ws_client = None
        
        if self.use_websocket and self.monitored_symbols:
            self._start_websocket()
    
    def update_symbols_from_api(self, min_volume: float = 5_000_000, limit: int = 400):
        """Оновлює список символів з API"""
        symbols_data = self.rest_client.get_usdt_perpetuals(min_volume)
        symbols = [s['symbol'] for s in symbols_data[:limit]]
        self.set_symbols(symbols)
        return symbols
    
    def _start_websocket(self):
        """Запускає WebSocket для real-time даних"""
        if not HAS_WEBSOCKET:
            logger.warning("WebSocket library not available")
            return
        
        self.ws_client = BybitWebSocketClient(
            symbols=self.monitored_symbols,
            on_ticker=self._on_ticker_update,
            on_error=self._on_ws_error,
        )
        self.ws_client.start()
    
    def _on_ticker_update(self, ticker: TickerData):
        """Callback для WebSocket ticker updates"""
        # WebSocket дані можна використовувати для real-time UI
        # Запис в БД робиться через periodic recording
        pass
    
    def _on_ws_error(self, error: Exception):
        """Callback для WebSocket помилок"""
        self.stats['errors'] += 1
        logger.warning(f"WS error in recorder: {error}")
    
    def record_snapshot(self, symbol: str = None) -> int:
        """
        Записує snapshot для одного або всіх символів.
        
        Args:
            symbol: Якщо None, записує для всіх monitored_symbols
            
        Returns:
            Кількість записаних snapshots
        """
        # MarketSnapshot вже імпортований глобально
        
        if symbol:
            symbols = [symbol]
        else:
            symbols = self.monitored_symbols
        
        if not symbols:
            logger.warning("No symbols to record")
            return 0
        
        session = self.db_session_factory()
        recorded = 0
        
        try:
            # Отримуємо всі тікери за один запит
            all_tickers = self.rest_client.get_all_tickers()
            
            for sym in symbols:
                ticker = all_tickers.get(sym)
                if not ticker:
                    continue
                
                try:
                    snapshot = MarketSnapshot(
                        symbol=sym,
                        timestamp=datetime.utcnow(),
                        open_interest=ticker.open_interest,
                        open_interest_qty=ticker.open_interest,
                        mark_price=ticker.mark_price,
                        index_price=ticker.index_price,
                        last_price=ticker.last_price,
                        funding_rate=ticker.funding_rate,
                        next_funding_time=ticker.next_funding_time,
                        volume_24h=ticker.volume_24h,
                        turnover_24h=ticker.turnover_24h,
                        bid1_price=ticker.bid1_price,
                        bid1_size=ticker.bid1_size,
                        ask1_price=ticker.ask1_price,
                        ask1_size=ticker.ask1_size,
                        spread_percent=ticker.spread_percent,
                        high_24h=ticker.high_24h,
                        low_24h=ticker.low_24h,
                        price_change_24h=ticker.price_change_24h,
                    )
                    
                    session.add(snapshot)
                    recorded += 1
                    
                except Exception as e:
                    logger.debug(f"Snapshot error for {sym}: {e}")
                    continue
            
            session.commit()
            
            self.stats['snapshots_recorded'] += recorded
            self.stats['last_record_time'] = datetime.utcnow()
            
            logger.info(f"📸 Recorded {recorded} snapshots")
            
        except Exception as e:
            logger.error(f"Record snapshot error: {e}")
            session.rollback()
            self.stats['errors'] += 1
        finally:
            session.close()
        
        return recorded
    
    def start_periodic_recording(self, interval: int = 300):
        """
        Запускає періодичний запис snapshots.
        
        Args:
            interval: Інтервал в секундах (default 5 хвилин)
        """
        if self._recording:
            logger.warning("Recording already running")
            return
        
        self._recording = True
        self._record_interval = interval
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()
        
        logger.info(f"🔄 Started periodic recording every {interval}s")
    
    def stop_periodic_recording(self):
        """Зупиняє періодичний запис"""
        self._recording = False
        
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join(timeout=10)
        
        logger.info("⏹️ Stopped periodic recording")
    
    def _record_loop(self):
        """Цикл періодичного запису"""
        last_cleanup = time.time()
        cleanup_interval = 3600  # Раз на годину
        
        while self._recording:
            try:
                self.record_snapshot()
                
                # === АВТООЧИСТКА БД (раз на годину) ===
                if time.time() - last_cleanup > cleanup_interval:
                    try:
                        from .models import cleanup_old_snapshots
                        session = self.db_session_factory()
                        try:
                            # Зберігаємо тільки 3 дні історії
                            deleted = cleanup_old_snapshots(session, days=3)
                            if deleted > 0:
                                logger.info(f"🧹 Cleanup: deleted {deleted} old snapshots")
                            last_cleanup = time.time()
                        finally:
                            session.close()
                    except Exception as ce:
                        logger.error(f"Cleanup error: {ce}")
                
            except Exception as e:
                logger.error(f"Record loop error: {e}")
                self.stats['errors'] += 1
            
            # Чекаємо з можливістю переривання
            for _ in range(self._record_interval):
                if not self._recording:
                    break
                time.sleep(1)
    
    def load_historical_data(self, symbol: str, hours: int = 24) -> bool:
        """
        Завантажує історичні дані для монети.
        Використовується для ініціалізації нових монет.
        
        Args:
            symbol: Торгова пара
            hours: Скільки годин історії завантажити
            
        Returns:
            True якщо успішно
        """
        # MarketSnapshot вже імпортований глобально
        
        session = self.db_session_factory()
        
        try:
            # Отримуємо OI історію
            oi_history = self.rest_client.get_open_interest_history(
                symbol,
                interval="1h" if hours >= 4 else "5min",
                limit=min(hours * 12, 200)  # ~12 записів на годину для 5min
            )
            
            if not oi_history:
                logger.warning(f"No OI history for {symbol}")
                return False
            
            # Отримуємо поточний тікер
            ticker = self.rest_client.get_ticker(symbol)
            
            if not ticker:
                logger.warning(f"No ticker for {symbol}")
                return False
            
            # Записуємо історичні точки
            recorded = 0
            for oi_data in oi_history:
                try:
                    # Перевіряємо чи вже є запис
                    existing = session.query(MarketSnapshot).filter(
                        MarketSnapshot.symbol == symbol,
                        MarketSnapshot.timestamp == oi_data.timestamp
                    ).first()
                    
                    if existing:
                        continue
                    
                    snapshot = MarketSnapshot(
                        symbol=symbol,
                        timestamp=oi_data.timestamp,
                        open_interest=oi_data.open_interest,
                        open_interest_qty=oi_data.open_interest,
                        mark_price=ticker.mark_price,  # Approximation
                        last_price=ticker.last_price,
                        funding_rate=ticker.funding_rate,
                        volume_24h=ticker.volume_24h,
                        turnover_24h=ticker.turnover_24h,
                    )
                    
                    session.add(snapshot)
                    recorded += 1
                    
                except Exception as e:
                    logger.debug(f"Historical snapshot error: {e}")
                    continue
            
            session.commit()
            logger.info(f"📥 Loaded {recorded} historical snapshots for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Load historical error for {symbol}: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Повертає статистику"""
        return {
            **self.stats,
            'monitored_symbols': len(self.monitored_symbols),
            'websocket_active': self.ws_client is not None,
            'recording_active': self._recording,
            'record_interval': self._record_interval,
        }
    
    def shutdown(self):
        """Зупиняє всі процеси"""
        self.stop_periodic_recording()
        
        if self.ws_client:
            self.ws_client.stop()
        
        logger.info("📊 DataRecorder shutdown complete")
