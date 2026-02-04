"""
CTR Fast Scanner v2.0 - Maximum Speed Edition

Архітектура для максимальної швидкості:
1. Попереднє завантаження історії при старті (1000 свічок)
2. WebSocket для real-time оновлення свічок
3. In-memory кеш - без затримок на I/O
4. Сканування кожні 5 секунд
5. Миттєві сигнали в Telegram

Результат: Сигнали за 1-5 секунд після формування свічки
(vs 30-60 секунд у старій версії)
"""

import numpy as np
import threading
import time
import json
import websocket
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Kline:
    """Одна свічка"""
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    is_closed: bool = True
    
    @classmethod
    def from_binance(cls, data: list) -> 'Kline':
        """Parse Binance kline data"""
        return cls(
            open_time=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
            close_time=int(data[6]),
            is_closed=True
        )
    
    @classmethod
    def from_websocket(cls, data: dict) -> 'Kline':
        """Parse WebSocket kline data"""
        k = data['k']
        return cls(
            open_time=int(k['t']),
            open=float(k['o']),
            high=float(k['h']),
            low=float(k['l']),
            close=float(k['c']),
            volume=float(k['v']),
            close_time=int(k['T']),
            is_closed=k['x']
        )


@dataclass
class SymbolCache:
    """Кеш даних для одного символу"""
    symbol: str
    timeframe: str
    klines: List[Kline] = field(default_factory=list)
    last_update: float = 0
    last_stc: float = 50.0
    prev_stc: float = 50.0
    is_ready: bool = False
    
    def get_closes(self) -> np.ndarray:
        """Отримати масив close prices"""
        return np.array([k.close for k in self.klines])
    
    def update_kline(self, kline: Kline):
        """Оновити або додати свічку"""
        if not self.klines:
            self.klines.append(kline)
            return
        
        # Якщо це та сама свічка - оновити
        if self.klines[-1].open_time == kline.open_time:
            self.klines[-1] = kline
        # Якщо нова свічка - додати
        elif kline.open_time > self.klines[-1].open_time:
            self.klines.append(kline)
            # Обмежуємо розмір кешу
            if len(self.klines) > 1500:
                self.klines = self.klines[-1000:]
        
        self.last_update = time.time()


# ============================================
# STC CALCULATOR (Optimized)
# ============================================

class STCCalculator:
    """
    Оптимізований розрахунок STC (Schaff Trend Cycle)
    
    Використовує векторизовані операції numpy для швидкості.
    """
    
    def __init__(
        self,
        fast_length: int = 21,
        slow_length: int = 50,
        cycle_length: int = 10,
        d1_length: int = 3,
        d2_length: int = 3,
        upper: float = 75,
        lower: float = 25
    ):
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.cycle_length = cycle_length
        self.d1_length = d1_length
        self.d2_length = d2_length
        self.upper = upper
        self.lower = lower
        
        # Мінімальна кількість свічок для стабільного розрахунку
        self.min_candles = slow_length + cycle_length * 2 + d1_length + d2_length + 100
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average - векторизована версія"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        
        # SMA для першого значення
        ema[period-1] = np.mean(data[:period])
        
        # EMA для решти
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        ema[:period-1] = np.nan
        return ema
    
    def _stochastic(self, data: np.ndarray, length: int) -> np.ndarray:
        """Stochastic oscillator"""
        result = np.full(len(data), 50.0)
        
        for i in range(length - 1, len(data)):
            window = data[i - length + 1:i + 1]
            valid = window[~np.isnan(window)]
            
            if len(valid) < 2:
                continue
            
            lowest = np.min(valid)
            highest = np.max(valid)
            denom = highest - lowest
            
            if denom > 0 and not np.isnan(data[i]):
                result[i] = (data[i] - lowest) / denom * 100
        
        return result
    
    def calculate(self, closes: np.ndarray) -> Tuple[float, float]:
        """
        Розрахувати STC для останніх двох значень
        
        Returns:
            (current_stc, prev_stc)
        """
        if len(closes) < self.min_candles:
            return 50.0, 50.0
        
        # MACD
        fast_ema = self._ema(closes, self.fast_length)
        slow_ema = self._ema(closes, self.slow_length)
        macd = fast_ema - slow_ema
        
        # Перший стохастик
        k = self._stochastic(macd, self.cycle_length)
        
        # Перше згладжування (D1)
        d = self._ema(k, self.d1_length)
        
        # Другий стохастик
        kd = self._stochastic(d, self.cycle_length)
        
        # Друге згладжування (D2) = STC
        stc = self._ema(kd, self.d2_length)
        
        # Clamp to 0-100
        stc = np.clip(stc, 0, 100)
        
        current = stc[-1] if not np.isnan(stc[-1]) else 50.0
        prev = stc[-2] if len(stc) > 1 and not np.isnan(stc[-2]) else current
        
        return float(current), float(prev)
    
    def detect_signal(self, closes: np.ndarray) -> Tuple[bool, bool, float, str]:
        """
        Детекція сигналів crossover/crossunder
        
        Returns:
            (buy_signal, sell_signal, current_stc, status)
        """
        current_stc, prev_stc = self.calculate(closes)
        
        # Crossover/Crossunder detection
        buy_signal = prev_stc <= self.lower and current_stc > self.lower
        sell_signal = prev_stc >= self.upper and current_stc < self.upper
        
        # Status
        if current_stc >= self.upper:
            status = "Overbought"
        elif current_stc <= self.lower:
            status = "Oversold"
        else:
            status = "Neutral"
        
        return buy_signal, sell_signal, current_stc, status


# ============================================
# FAST CTR SCANNER
# ============================================

class CTRFastScanner:
    """
    Швидкий CTR Scanner з WebSocket та in-memory кешем
    
    Особливості:
    - Попереднє завантаження історії
    - Real-time оновлення через WebSocket
    - Сканування без API запитів
    - Сигнали за 1-5 секунд
    """
    
    # Binance WebSocket endpoints
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    REST_BASE_URL = "https://api.binance.com/api/v3"
    
    # Timeframe to Binance interval mapping
    TIMEFRAME_MAP = {
        '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m',
        '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
        '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d'
    }
    
    def __init__(
        self,
        timeframe: str = '15m',
        fast_length: int = 21,
        slow_length: int = 50,
        cycle_length: int = 10,
        d1_length: int = 3,
        d2_length: int = 3,
        upper: float = 75,
        lower: float = 25,
        on_signal: Callable = None
    ):
        self.timeframe = timeframe
        self.on_signal = on_signal  # Callback для сигналів
        
        # STC Calculator
        self.stc = STCCalculator(
            fast_length, slow_length, cycle_length,
            d1_length, d2_length, upper, lower
        )
        
        # In-memory cache
        self._cache: Dict[str, SymbolCache] = {}
        self._lock = threading.RLock()
        
        # WebSocket
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_connected = False
        
        # Scanner state
        self._running = False
        self._scan_thread: Optional[threading.Thread] = None
        self._watchlist: List[str] = []
        
        # Signal tracking (deduplication)
        self._last_signals: Dict[str, Tuple[str, float]] = {}  # symbol -> (signal_type, timestamp)
        self._signal_cooldown = 3600  # 1 година між однаковими сигналами
        
        # Statistics
        self._stats = {
            'scans': 0,
            'signals_sent': 0,
            'ws_messages': 0,
            'last_scan_time': 0,
            'avg_scan_ms': 0
        }
        
        print(f"[CTR Fast] Initialized: TF={timeframe}, Upper={upper}, Lower={lower}")
    
    # ========================================
    # DATA LOADING
    # ========================================
    
    def _load_history(self, symbol: str) -> bool:
        """Завантажити історичні дані для символу"""
        import requests
        
        try:
            url = f"{self.REST_BASE_URL}/klines"
            params = {
                'symbol': symbol,
                'interval': self.TIMEFRAME_MAP.get(self.timeframe, '15m'),
                'limit': 1000
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"[CTR Fast] ❌ Failed to load {symbol}: {response.status_code}")
                return False
            
            data = response.json()
            
            if not data:
                print(f"[CTR Fast] ❌ No data for {symbol}")
                return False
            
            # Parse klines
            klines = [Kline.from_binance(k) for k in data]
            
            # Create cache entry
            with self._lock:
                cache = SymbolCache(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    klines=klines,
                    last_update=time.time(),
                    is_ready=len(klines) >= self.stc.min_candles
                )
                self._cache[symbol] = cache
            
            print(f"[CTR Fast] ✅ Loaded {symbol}: {len(klines)} candles")
            return True
            
        except Exception as e:
            print(f"[CTR Fast] ❌ Error loading {symbol}: {e}")
            return False
    
    def preload_watchlist(self, symbols: List[str]) -> int:
        """
        Попереднє завантаження даних для всіх символів
        
        Returns: кількість успішно завантажених
        """
        print(f"[CTR Fast] Preloading {len(symbols)} symbols...")
        
        loaded = 0
        for symbol in symbols:
            if self._load_history(symbol):
                loaded += 1
            time.sleep(0.1)  # Невелика затримка між запитами
        
        print(f"[CTR Fast] Preloaded {loaded}/{len(symbols)} symbols")
        return loaded
    
    # ========================================
    # WEBSOCKET
    # ========================================
    
    def _get_ws_url(self, symbols: List[str]) -> str:
        """Створити WebSocket URL для підписки на кілька символів"""
        streams = [f"{s.lower()}@kline_{self.timeframe}" for s in symbols]
        return f"{self.WS_BASE_URL}/{'/'.join(streams)}"
    
    def _on_ws_message(self, ws, message):
        """Обробка WebSocket повідомлення"""
        try:
            data = json.loads(message)
            
            # Визначаємо формат повідомлення
            if 'stream' in data:
                # Combined stream format
                stream_data = data['data']
            else:
                # Single stream format
                stream_data = data
            
            if 'e' not in stream_data or stream_data['e'] != 'kline':
                return
            
            symbol = stream_data['s']
            kline = Kline.from_websocket(stream_data)
            
            with self._lock:
                if symbol in self._cache:
                    self._cache[symbol].update_kline(kline)
                    self._stats['ws_messages'] += 1
                    
                    # Якщо свічка закрилась - негайно сканувати
                    if kline.is_closed:
                        self._scan_symbol_immediate(symbol)
                        
        except Exception as e:
            logger.error(f"[CTR Fast] WS message error: {e}")
    
    def _on_ws_error(self, ws, error):
        """Обробка WebSocket помилки"""
        print(f"[CTR Fast] WS Error: {error}")
        self._ws_connected = False
    
    def _on_ws_close(self, ws, close_status, close_msg):
        """Обробка закриття WebSocket"""
        print(f"[CTR Fast] WS Closed: {close_status} {close_msg}")
        self._ws_connected = False
        
        # Автоматичне перепідключення
        if self._running:
            print("[CTR Fast] Reconnecting WebSocket in 5 seconds...")
            time.sleep(5)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        """Обробка відкриття WebSocket"""
        print(f"[CTR Fast] ✅ WebSocket connected")
        self._ws_connected = True
        
        # Підписка на символи (для combined stream)
        if len(self._watchlist) > 1:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{s.lower()}@kline_{self.timeframe}" for s in self._watchlist],
                "id": 1
            }
            ws.send(json.dumps(subscribe_msg))
    
    def _start_websocket(self):
        """Запустити WebSocket підключення"""
        if not self._watchlist:
            return
        
        # Для кількох символів використовуємо combined stream
        if len(self._watchlist) > 1:
            streams = "/".join([f"{s.lower()}@kline_{self.timeframe}" for s in self._watchlist])
            ws_url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        else:
            ws_url = f"{self.WS_BASE_URL}/{self._watchlist[0].lower()}@kline_{self.timeframe}"
        
        self._ws = websocket.WebSocketApp(
            ws_url,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
            on_open=self._on_ws_open
        )
        
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            daemon=True
        )
        self._ws_thread.start()
    
    def _stop_websocket(self):
        """Зупинити WebSocket"""
        if self._ws:
            self._ws.close()
            self._ws = None
        self._ws_connected = False
    
    # ========================================
    # SCANNING
    # ========================================
    
    def _scan_symbol_immediate(self, symbol: str):
        """Негайне сканування символу (при закритті свічки)"""
        with self._lock:
            cache = self._cache.get(symbol)
            if not cache or not cache.is_ready:
                return
            
            closes = cache.get_closes()
        
        if len(closes) < self.stc.min_candles:
            return
        
        buy, sell, stc_value, status = self.stc.detect_signal(closes)
        
        # Оновлюємо STC в кеші
        with self._lock:
            cache.prev_stc = cache.last_stc
            cache.last_stc = stc_value
        
        # Перевіряємо сигнал
        if buy or sell:
            self._process_signal(symbol, 'BUY' if buy else 'SELL', stc_value, closes[-1])
    
    def _scan_all(self):
        """Сканування всіх символів"""
        start_time = time.time()
        
        with self._lock:
            symbols = list(self._cache.keys())
        
        results = []
        
        for symbol in symbols:
            with self._lock:
                cache = self._cache.get(symbol)
                if not cache or not cache.is_ready:
                    continue
                closes = cache.get_closes()
            
            if len(closes) < self.stc.min_candles:
                continue
            
            buy, sell, stc_value, status = self.stc.detect_signal(closes)
            
            # Оновлюємо кеш
            with self._lock:
                cache.prev_stc = cache.last_stc
                cache.last_stc = stc_value
            
            results.append({
                'symbol': symbol,
                'stc': round(stc_value, 2),
                'status': status,
                'price': closes[-1],
                'buy_signal': buy,
                'sell_signal': sell
            })
            
            # Обробка сигналу
            if buy or sell:
                self._process_signal(symbol, 'BUY' if buy else 'SELL', stc_value, closes[-1])
        
        # Статистика
        scan_time = (time.time() - start_time) * 1000
        self._stats['scans'] += 1
        self._stats['last_scan_time'] = scan_time
        self._stats['avg_scan_ms'] = (
            (self._stats['avg_scan_ms'] * (self._stats['scans'] - 1) + scan_time) 
            / self._stats['scans']
        )
        
        return results
    
    def _process_signal(self, symbol: str, signal_type: str, stc_value: float, price: float):
        """Обробка та відправка сигналу"""
        now = time.time()
        
        # Перевірка дедуплікації
        last = self._last_signals.get(symbol)
        if last:
            last_type, last_time = last
            if last_type == signal_type and (now - last_time) < self._signal_cooldown:
                return  # Пропускаємо дублікат
        
        # Зберігаємо сигнал
        self._last_signals[symbol] = (signal_type, now)
        self._stats['signals_sent'] += 1
        
        # Формуємо повідомлення
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        action = "ПОКУПКА" if signal_type == "BUY" else "ПРОДАЖ"
        cross = f"STC перетнув {self.stc.lower} знизу" if signal_type == "BUY" else f"STC перетнув {self.stc.upper} зверху"
        
        message = f"""{emoji} CTR: Сигнал {action}

Монета: {symbol}
Ціна: ${price:,.4f}
STC: {stc_value:.2f}
Таймфрейм: {self.timeframe}

{cross}

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"""
        
        print(f"\n{'='*50}")
        print(message)
        print(f"{'='*50}\n")
        
        # Callback
        if self.on_signal:
            try:
                self.on_signal({
                    'symbol': symbol,
                    'type': signal_type,
                    'price': price,
                    'stc': stc_value,
                    'timeframe': self.timeframe,
                    'message': message
                })
            except Exception as e:
                print(f"[CTR Fast] Signal callback error: {e}")
    
    def _scan_loop(self):
        """Головний цикл сканування"""
        print("[CTR Fast] Scan loop started")
        
        scan_interval = 5  # Сканування кожні 5 секунд
        
        while self._running:
            try:
                results = self._scan_all()
                
                # Логування
                ready_count = sum(1 for r in results if r['status'] != 'Neutral')
                if ready_count > 0 or self._stats['scans'] % 12 == 0:  # Кожну хвилину
                    print(f"[CTR Fast] Scan #{self._stats['scans']}: "
                          f"{len(results)} symbols, "
                          f"{self._stats['last_scan_time']:.1f}ms, "
                          f"WS msgs: {self._stats['ws_messages']}")
                
            except Exception as e:
                print(f"[CTR Fast] Scan error: {e}")
            
            time.sleep(scan_interval)
        
        print("[CTR Fast] Scan loop stopped")
    
    # ========================================
    # PUBLIC API
    # ========================================
    
    def start(self, watchlist: List[str]):
        """
        Запустити сканер
        
        Args:
            watchlist: список символів для моніторингу
        """
        if self._running:
            print("[CTR Fast] Already running")
            return
        
        self._watchlist = [s.upper() for s in watchlist]
        
        print(f"[CTR Fast] Starting with {len(self._watchlist)} symbols...")
        
        # 1. Завантажити історію
        loaded = self.preload_watchlist(self._watchlist)
        
        if loaded == 0:
            print("[CTR Fast] ❌ Failed to load any symbols")
            return
        
        # 2. Запустити WebSocket
        self._start_websocket()
        
        # Чекаємо підключення
        for _ in range(10):
            if self._ws_connected:
                break
            time.sleep(0.5)
        
        # 3. Запустити сканування
        self._running = True
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        
        print(f"[CTR Fast] ✅ Started successfully")
    
    def stop(self):
        """Зупинити сканер"""
        print("[CTR Fast] Stopping...")
        
        self._running = False
        self._stop_websocket()
        
        if self._scan_thread:
            self._scan_thread.join(timeout=5)
        
        print("[CTR Fast] ✅ Stopped")
    
    def add_symbol(self, symbol: str) -> bool:
        """Додати символ до watchlist"""
        symbol = symbol.upper()
        
        if symbol in self._watchlist:
            return False
        
        # Завантажити дані
        if not self._load_history(symbol):
            return False
        
        self._watchlist.append(symbol)
        
        # Перепідключити WebSocket з новим символом
        if self._ws_connected:
            self._stop_websocket()
            self._start_websocket()
        
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        """Видалити символ з watchlist"""
        symbol = symbol.upper()
        
        if symbol not in self._watchlist:
            return False
        
        self._watchlist.remove(symbol)
        
        with self._lock:
            if symbol in self._cache:
                del self._cache[symbol]
        
        # Перепідключити WebSocket
        if self._ws_connected and self._watchlist:
            self._stop_websocket()
            self._start_websocket()
        
        return True
    
    def get_status(self) -> Dict:
        """Отримати статус сканера"""
        with self._lock:
            cache_status = {
                symbol: {
                    'candles': len(cache.klines),
                    'stc': round(cache.last_stc, 2),
                    'ready': cache.is_ready,
                    'last_update': cache.last_update
                }
                for symbol, cache in self._cache.items()
            }
        
        return {
            'running': self._running,
            'ws_connected': self._ws_connected,
            'watchlist': self._watchlist,
            'timeframe': self.timeframe,
            'cache': cache_status,
            'stats': self._stats.copy()
        }
    
    def get_results(self) -> List[Dict]:
        """Отримати поточні результати для всіх символів"""
        results = []
        
        with self._lock:
            for symbol, cache in self._cache.items():
                if not cache.is_ready:
                    continue
                
                closes = cache.get_closes()
                if len(closes) < 2:
                    continue
                
                # Визначаємо статус
                stc = cache.last_stc
                if stc >= self.stc.upper:
                    status = "Overbought"
                elif stc <= self.stc.lower:
                    status = "Oversold"
                else:
                    status = "Neutral"
                
                results.append({
                    'symbol': symbol,
                    'price': closes[-1],
                    'stc': round(stc, 2),
                    'prev_stc': round(cache.prev_stc, 2),
                    'status': status,
                    'candles': len(cache.klines),
                    'timeframe': self.timeframe
                })
        
        return sorted(results, key=lambda x: x['symbol'])
    
    def reload_settings(self, settings: Dict):
        """Оновити налаштування"""
        if 'timeframe' in settings:
            new_tf = settings['timeframe']
            if new_tf != self.timeframe:
                self.timeframe = new_tf
                # Потрібно перезавантажити дані
                if self._running:
                    self.stop()
                    self.start(self._watchlist)
        
        if 'upper' in settings:
            self.stc.upper = float(settings['upper'])
        if 'lower' in settings:
            self.stc.lower = float(settings['lower'])
        if 'fast_length' in settings:
            self.stc.fast_length = int(settings['fast_length'])
        if 'slow_length' in settings:
            self.stc.slow_length = int(settings['slow_length'])
        
        print(f"[CTR Fast] Settings reloaded: TF={self.timeframe}, "
              f"Upper={self.stc.upper}, Lower={self.stc.lower}")


# ============================================
# SINGLETON
# ============================================

_ctr_fast_instance: Optional[CTRFastScanner] = None
_ctr_fast_lock = threading.Lock()


def get_ctr_fast_scanner(
    timeframe: str = '15m',
    on_signal: Callable = None,
    **kwargs
) -> CTRFastScanner:
    """Отримати singleton екземпляр CTR Fast Scanner"""
    global _ctr_fast_instance
    
    with _ctr_fast_lock:
        if _ctr_fast_instance is None:
            _ctr_fast_instance = CTRFastScanner(
                timeframe=timeframe,
                on_signal=on_signal,
                **kwargs
            )
        return _ctr_fast_instance


def reset_ctr_fast_scanner():
    """Скинути singleton (для тестів)"""
    global _ctr_fast_instance
    
    with _ctr_fast_lock:
        if _ctr_fast_instance:
            _ctr_fast_instance.stop()
            _ctr_fast_instance = None
