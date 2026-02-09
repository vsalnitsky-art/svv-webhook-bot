"""
CTR Fast Scanner v2.5 - Production Release (Anti-Oscillation Fix)

Based on v2.4 + v2.5 fixes:
- Trend Guard grace period (5 min after signal)
- Minimum cycle travel requirement for Trend Guard
- Standard cooldown 60s → 300s

Changes from v2.1:
1. High-Water Mark: Cycle extremes tracking (cycle_high/cycle_low) to detect
   hidden peaks between 5-second scan intervals.
2. Gap Detection: If STC crossed upper/lower between scans, signal is generated
   retroactively with auto-reset to prevent infinite loops.
3. Trend Guard: Emergency exit when trend fails mid-cycle (STC returns to entry
   zone without reaching target). Only triggers if target zone WAS NOT reached.
4. CPU Optimization: STC calculation throttled to max 1/sec per symbol on WS ticks.
5. Thread Safety: State mutations inside locks, I/O outside locks.
6. Cooldown: 60s standard, Trend Guard bypasses cooldowns and SMC filter.

Architecture:
1. Preload 1000 candles at startup
2. WebSocket for real-time candle updates
3. In-memory cache - zero I/O latency
4. Scan every 5 seconds + immediate on candle close
5. Smart signal detection with gap/trend guard logic
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

# SMC Filter import
try:
    from detection.smc_structure_filter import SMCSignalFilter, TrendBias
    SMC_AVAILABLE = True
except ImportError:
    SMC_AVAILABLE = False
    print("[CTR Fast] Warning: SMC Structure Filter not available")


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
    
    # STC State
    last_stc: float = 50.0
    prev_stc: float = 50.0
    last_calc_time: float = 0.0  # CPU throttle timestamp
    last_signal_time: float = 0.0  # When last signal was generated (for grace period)
    
    # Cycle Memory (High-Water Mark) — v2.4
    cycle_high: float = 0.0    # Max STC since last signal reset
    cycle_low: float = 100.0   # Min STC since last signal reset
    
    # Trend State — v2.4
    last_signal_type: Optional[str] = None  # 'BUY' or 'SELL'
    
    is_ready: bool = False
    smc_filter: Optional['SMCSignalFilter'] = None
    
    def get_closes(self) -> np.ndarray:
        return np.array([k.close for k in self.klines])
    
    def get_highs(self) -> np.ndarray:
        return np.array([k.high for k in self.klines])
    
    def get_lows(self) -> np.ndarray:
        return np.array([k.low for k in self.klines])
    
    def get_ohlc(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        opens = np.array([k.open for k in self.klines])
        highs = np.array([k.high for k in self.klines])
        lows = np.array([k.low for k in self.klines])
        closes = np.array([k.close for k in self.klines])
        return opens, highs, lows, closes
    
    def update_kline(self, kline: Kline):
        if not self.klines:
            self.klines.append(kline)
            return
        
        if self.klines[-1].open_time == kline.open_time:
            self.klines[-1] = kline
        elif kline.open_time > self.klines[-1].open_time:
            self.klines.append(kline)
            if len(self.klines) > 1500:
                self.klines = self.klines[-1000:]
        
        self.last_update = time.time()
    
    def update_cycle_extremes(self, current_stc: float):
        """Track STC peaks between scans"""
        self.cycle_high = max(self.cycle_high, current_stc)
        self.cycle_low = min(self.cycle_low, current_stc)
    
    def reset_cycle_extremes(self, signal_type: str):
        """Reset extremes after signal is confirmed"""
        if signal_type == 'BUY':
            self.cycle_high = 0.0
        elif signal_type == 'SELL':
            self.cycle_low = 100.0


# ============================================
# STC CALCULATOR (Optimized)
# ============================================

class STCCalculator:
    """Оптимізований розрахунок STC (Schaff Trend Cycle)"""
    
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
        self.min_candles = slow_length + cycle_length * 2 + d1_length + d2_length + 100
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        if len(data) < period:
            return np.full(len(data), np.nan)
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        ema[:period-1] = np.nan
        return ema
    
    def _stochastic(self, data: np.ndarray, length: int) -> np.ndarray:
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
        """Returns: (current_stc, prev_stc)"""
        if len(closes) < self.min_candles:
            return 50.0, 50.0
        
        fast_ema = self._ema(closes, self.fast_length)
        slow_ema = self._ema(closes, self.slow_length)
        macd = fast_ema - slow_ema
        k = self._stochastic(macd, self.cycle_length)
        d = self._ema(k, self.d1_length)
        kd = self._stochastic(d, self.cycle_length)
        stc = self._ema(kd, self.d2_length)
        stc = np.clip(stc, 0, 100)
        
        current = stc[-1] if not np.isnan(stc[-1]) else 50.0
        prev = stc[-2] if len(stc) > 1 and not np.isnan(stc[-2]) else current
        return float(current), float(prev)


# ============================================
# FAST CTR SCANNER
# ============================================

class CTRFastScanner:
    """Швидкий CTR Scanner з WebSocket та Smart Reversal Detection"""
    
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"
    REST_BASE_URL = "https://api.binance.com/api/v3"
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
        on_signal: Callable = None,
        # SMC Filter settings
        smc_filter_enabled: bool = False,
        smc_swing_length: int = 50,
        smc_zone_threshold: float = 1.0,
        smc_require_trend: bool = True,
    ):
        self.timeframe = timeframe
        self.on_signal = on_signal
        
        # STC Calculator
        self.stc = STCCalculator(
            fast_length, slow_length, cycle_length,
            d1_length, d2_length, upper, lower
        )
        
        # SMC Filter settings
        self.smc_filter_enabled = smc_filter_enabled and SMC_AVAILABLE
        self.smc_swing_length = smc_swing_length
        self.smc_zone_threshold = smc_zone_threshold
        self.smc_require_trend = smc_require_trend
        
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
        
        # Signal tracking — v2.5: 300s cooldown (was 60 in v2.4)
        self._last_signals: Dict[str, Tuple[str, float]] = {}
        self._signal_cooldown = 300
        # Trend Guard anti-oscillation: grace period + min travel
        self._trend_guard_grace = 300  # seconds before TG can fire after any signal
        self._trend_guard_min_travel = 0.30  # STC must travel 30% of range before TG triggers
        
        # Statistics
        self._stats = {
            'scans': 0,
            'signals_sent': 0,
            'signals_filtered': 0,
            'ws_messages': 0,
            'last_scan_time': 0,
            'avg_scan_ms': 0
        }
        
        smc_status = "ON" if self.smc_filter_enabled else "OFF"
        print(f"[CTR Fast v2.5] Initialized: TF={timeframe}, Upper={upper}, Lower={lower}, SMC={smc_status}")
    
    # ========================================
    # DATA LOADING
    # ========================================
    
    def _create_smc_filter(self) -> Optional['SMCSignalFilter']:
        """Створити SMC фільтр для символу"""
        if not self.smc_filter_enabled or not SMC_AVAILABLE:
            return None
        
        try:
            return SMCSignalFilter(
                swing_length=self.smc_swing_length,
                zone_threshold_percent=self.smc_zone_threshold,
                require_trend_for_zones=self.smc_require_trend,
            )
        except TypeError:
            # Fallback: старий SMCSignalFilter без require_trend_for_zones
            return SMCSignalFilter(
                swing_length=self.smc_swing_length,
                zone_threshold_percent=self.smc_zone_threshold,
            )
    
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
            
            klines = [Kline.from_binance(k) for k in data]
            smc_filter = self._create_smc_filter()
            
            with self._lock:
                cache = SymbolCache(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    klines=klines,
                    last_update=time.time(),
                    is_ready=len(klines) >= self.stc.min_candles,
                    smc_filter=smc_filter
                )
                self._cache[symbol] = cache
                
                if smc_filter and len(klines) > 100:
                    highs = cache.get_highs()
                    lows = cache.get_lows()
                    closes = cache.get_closes()
                    smc_filter.update_structure(highs, lows, closes)
            
            smc_tag = "SMC✓" if smc_filter else ""
            print(f"[CTR Fast] ✅ Loaded {symbol}: {len(klines)} candles {smc_tag}")
            return True
            
        except Exception as e:
            print(f"[CTR Fast] ❌ Error loading {symbol}: {e}")
            return False
    
    def preload_watchlist(self, symbols: List[str]) -> int:
        """Попереднє завантаження з повторною спробою"""
        print(f"[CTR Fast] Preloading {len(symbols)} symbols: {symbols}")
        
        loaded = 0
        failed = []
        
        for symbol in symbols:
            if self._load_history(symbol):
                loaded += 1
            else:
                failed.append(symbol)
            time.sleep(0.2)
        
        if failed:
            print(f"[CTR Fast] Retrying failed symbols in 2 seconds: {failed}")
            time.sleep(2)
            
            retry_failed = []
            for symbol in failed:
                if self._load_history(symbol):
                    loaded += 1
                    print(f"[CTR Fast] ✅ Retry successful: {symbol}")
                else:
                    retry_failed.append(symbol)
                time.sleep(0.3)
            
            failed = retry_failed
        
        print(f"[CTR Fast] ✅ Preloaded {loaded}/{len(symbols)} symbols")
        if failed:
            print(f"[CTR Fast] ⚠️ Failed to load after retry: {failed}")
        
        return loaded
    
    # ========================================
    # WEBSOCKET
    # ========================================
    
    def _get_ws_url(self, symbols: List[str]) -> str:
        streams = [f"{s.lower()}@kline_{self.timeframe}" for s in symbols]
        return f"{self.WS_BASE_URL}/{'/'.join(streams)}"
    
    def _on_ws_message(self, ws, message):
        """Обробка WebSocket повідомлення"""
        try:
            data = json.loads(message)
            
            stream_data = data['data'] if 'stream' in data else data
            
            if stream_data.get('e') != 'kline':
                return
            
            symbol = stream_data['s']
            kline = Kline.from_websocket(stream_data)
            
            with self._lock:
                if symbol in self._cache:
                    cache = self._cache[symbol]
                    cache.update_kline(kline)
                    self._stats['ws_messages'] += 1
                    
                    # v2.4: CPU-throttled STC calc for cycle extreme tracking
                    # Max 1 calc per second per symbol (saves 30-75x CPU)
                    now = time.time()
                    if cache.is_ready and (now - cache.last_calc_time >= 1.0 or kline.is_closed):
                        closes = cache.get_closes()
                        curr_stc, _ = self.stc.calculate(closes)
                        cache.update_cycle_extremes(curr_stc)
                        cache.last_calc_time = now
                    
                    # Негайне сканування при закритті свічки
                    if kline.is_closed:
                        self._scan_symbol_immediate(symbol)
                        
        except Exception as e:
            pass  # WS messages come fast, don't spam logs
    
    def _on_ws_error(self, ws, error):
        self._ws_connected = False
        print(f"[CTR Fast] WS Error: {error}")
    
    def _on_ws_close(self, ws, close_status, close_msg):
        self._ws_connected = False
        print(f"[CTR Fast] WS Closed: {close_status} {close_msg}")
        
        if self._running:
            print("[CTR Fast] Reconnecting WebSocket in 5 seconds...")
            time.sleep(5)
            self._start_websocket()
    
    def _on_ws_open(self, ws):
        self._ws_connected = True
        print(f"[CTR Fast] ✅ WebSocket connected")
        
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
        
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()
    
    def _stop_websocket(self):
        """Зупинити WebSocket"""
        if self._ws:
            self._ws.close()
        self._ws = None
    
    # ========================================
    # SMART SIGNAL DETECTION — v2.4
    # ========================================
    
    def _detect_smart_signals(self, symbol: str, cache: SymbolCache) -> Tuple[bool, bool, float, str, str]:
        """
        Smart detection v2.4:
        1. Standard crossover (prev_stc crosses upper/lower)
        2. Gap detection (cycle_high/low crossed between scans)
        3. Trend Guard (emergency exit if target NOT reached)
        
        Returns: (buy_signal, sell_signal, current_stc, status, reason)
        """
        closes = cache.get_closes()
        if len(closes) < self.stc.min_candles:
            return False, False, 50.0, "Loading", ""
        
        current_stc, prev_stc = self.stc.calculate(closes)
        
        # === 1. Standard Crossover ===
        buy_cross = prev_stc <= self.stc.lower and current_stc > self.stc.lower
        sell_cross = prev_stc >= self.stc.upper and current_stc < self.stc.upper
        
        # === 2. Gap Detection (Hidden Peak) ===
        # STC crossed upper/lower between scans but we missed the crossover
        gap_sell = (not sell_cross) and (cache.cycle_high >= self.stc.upper) and (current_stc < self.stc.upper)
        gap_buy = (not buy_cross) and (cache.cycle_low <= self.stc.lower) and (current_stc > self.stc.lower)
        
        # === 3. Trend Guard (Emergency Exit) ===
        # v2.5: Added grace period + minimum travel requirement to prevent oscillation
        # Trend Guard only triggers if:
        #   a) Enough time passed since last signal (grace period)
        #   b) STC actually traveled meaningfully before reversing (min travel)
        
        now = time.time()
        tg_range = self.stc.upper - self.stc.lower  # e.g. 75 - 25 = 50
        tg_min_high = self.stc.lower + tg_range * self._trend_guard_min_travel  # e.g. 25 + 15 = 40
        tg_max_low = self.stc.upper - tg_range * self._trend_guard_min_travel   # e.g. 75 - 15 = 60
        tg_grace_ok = (now - cache.last_signal_time) >= self._trend_guard_grace
        
        trend_fail_sell = False
        # After BUY: expected to reach UPPER. If fell back below LOWER without reaching UPPER → exit
        if cache.last_signal_type == 'BUY' and current_stc < self.stc.lower:
            if cache.prev_stc > self.stc.lower:  # Just crossed down
                if cache.cycle_high < self.stc.upper:  # Never reached target
                    # v2.5: Only if STC traveled above midpoint AND grace period passed
                    if cache.cycle_high >= tg_min_high and tg_grace_ok:
                        trend_fail_sell = True
        
        trend_fail_buy = False
        # After SELL: expected to reach LOWER. If rose back above UPPER without reaching LOWER → exit
        if cache.last_signal_type == 'SELL' and current_stc > self.stc.upper:
            if cache.prev_stc < self.stc.upper:  # Just crossed up
                if cache.cycle_low > self.stc.lower:  # Never reached target
                    # v2.5: Only if STC traveled below midpoint AND grace period passed
                    if cache.cycle_low <= tg_max_low and tg_grace_ok:
                        trend_fail_buy = True
        
        # === Combine ===
        final_buy = buy_cross or gap_buy or trend_fail_buy
        final_sell = sell_cross or gap_sell or trend_fail_sell
        
        # Determine reason
        reason = ""
        if final_buy:
            if buy_cross: reason = "Crossover"
            elif gap_buy: reason = "Gap Fill (Rapid Dip)"
            elif trend_fail_buy: reason = "Trend Guard (Short Fail)"
        elif final_sell:
            if sell_cross: reason = "Crossunder"
            elif gap_sell: reason = "Gap Fill (Rapid Pump)"
            elif trend_fail_sell: reason = "Trend Guard (Long Fail)"
        
        # Status string
        if current_stc >= self.stc.upper:
            status = "Overbought"
        elif current_stc <= self.stc.lower:
            status = "Oversold"
        else:
            status = "Neutral"
        
        # Loop Protection: reset immediately after gap detection
        # Prevents infinite loop if SMC filters the resulting signal
        if gap_sell:
            cache.cycle_high = 0.0
        if gap_buy:
            cache.cycle_low = 100.0
        
        return final_buy, final_sell, current_stc, status, reason
    
    # ========================================
    # SCANNING
    # ========================================
    
    def _scan_symbol_immediate(self, symbol: str):
        """Негайне сканування при закритті свічки (thread-safe)"""
        signal_data = None
        
        with self._lock:
            cache = self._cache.get(symbol)
            if not cache or not cache.is_ready:
                return
            
            # Update cycle extremes on candle close
            closes = cache.get_closes()
            stc_val, _ = self.stc.calculate(closes)
            cache.update_cycle_extremes(stc_val)
            cache.prev_stc = cache.last_stc
            cache.last_stc = stc_val
            
            buy, sell, _, status, reason = self._detect_smart_signals(symbol, cache)
            
            # Update SMC structure
            if cache.smc_filter:
                highs = cache.get_highs()
                lows = cache.get_lows()
                cache.smc_filter.update_structure(highs, lows, closes)
            
            if buy or sell:
                signal_type = 'BUY' if buy else 'SELL'
                # State update inside lock
                cache.last_signal_type = signal_type
                cache.last_signal_time = time.time()
                cache.reset_cycle_extremes(signal_type)
                
                signal_data = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'stc_value': stc_val,
                    'price': closes[-1],
                    'cache': cache,
                    'reason': reason
                }
        
        # I/O outside lock
        if signal_data:
            self._process_signal(**signal_data)
    
    def _scan_all(self):
        """Регулярне сканування всіх символів"""
        start_time = time.time()
        
        with self._lock:
            symbols = list(self._cache.keys())
        
        results = []
        
        for symbol in symbols:
            signal_data = None
            
            with self._lock:
                cache = self._cache.get(symbol)
                if not cache or not cache.is_ready:
                    continue
                
                buy, sell, stc_val, status, reason = self._detect_smart_signals(symbol, cache)
                
                # Update cache state
                cache.prev_stc = cache.last_stc
                cache.last_stc = stc_val
                cache.update_cycle_extremes(stc_val)
                
                # Update SMC structure
                if cache.smc_filter:
                    highs = cache.get_highs()
                    lows = cache.get_lows()
                    closes = cache.get_closes()
                    cache.smc_filter.update_structure(highs, lows, closes)
                
                # SMC info for results
                smc_status = None
                if cache.smc_filter:
                    smc_data = cache.smc_filter.get_status()
                    smc_status = {
                        'trend': smc_data['trend_bias'],
                        'near_support': self._is_near_smc_level(cache.get_closes()[-1], smc_data, 'support'),
                        'near_resistance': self._is_near_smc_level(cache.get_closes()[-1], smc_data, 'resistance'),
                    }
                
                results.append({
                    'symbol': symbol,
                    'stc': round(stc_val, 2),
                    'status': status,
                    'price': cache.get_closes()[-1],
                    'buy_signal': buy,
                    'sell_signal': sell,
                    'smc': smc_status
                })
                
                if buy or sell:
                    signal_type = 'BUY' if buy else 'SELL'
                    cache.last_signal_type = signal_type
                    cache.last_signal_time = time.time()
                    cache.reset_cycle_extremes(signal_type)
                    signal_data = {
                        'symbol': symbol,
                        'signal_type': signal_type,
                        'stc_value': stc_val,
                        'price': cache.get_closes()[-1],
                        'cache': cache,
                        'reason': reason
                    }
            
            # I/O outside lock
            if signal_data:
                self._process_signal(**signal_data)
        
        # Statistics
        scan_time = (time.time() - start_time) * 1000
        self._stats['scans'] += 1
        self._stats['last_scan_time'] = scan_time
        self._stats['avg_scan_ms'] = (
            (self._stats['avg_scan_ms'] * (self._stats['scans'] - 1) + scan_time)
            / self._stats['scans']
        )
        
        return results
    
    def _is_near_smc_level(self, price: float, smc_data: Dict, level_type: str) -> bool:
        """Перевірка чи ціна біля SMC рівня"""
        threshold = price * (self.smc_zone_threshold / 100)
        
        if level_type == 'support':
            levels = [smc_data.get('strong_low'), smc_data.get('last_hl'), smc_data.get('swing_low')]
        else:
            levels = [smc_data.get('weak_high'), smc_data.get('last_lh'), smc_data.get('swing_high')]
        
        for level in levels:
            if level and abs(price - level) <= threshold:
                return True
        return False
    
    def _process_signal(self, symbol: str, signal_type: str, stc_value: float,
                        price: float, cache: SymbolCache = None, reason: str = ""):
        """
        Process and send signal notification.
        State updates happen in caller (inside lock). This method is purely I/O.
        """
        now = time.time()
        
        # Priority: Trend Guard bypasses SMC filter but NOT cooldown
        is_priority = "Trend Guard" in reason
        
        # Cooldown check — v2.5: per-symbol cooldown for ANY signal type
        last = self._last_signals.get(symbol)
        if last:
            last_type, last_time = last
            time_since = now - last_time
            
            if is_priority:
                # Trend Guard: needs its own grace period (same as _trend_guard_grace)
                if time_since < self._trend_guard_grace:
                    return
            else:
                # Regular signals: same-type blocked by cooldown
                if last_type == signal_type:
                    if time_since < self._signal_cooldown:
                        return
                # Different-type: minimum 30s gap to prevent rapid flip-flop
                else:
                    if time_since < 30:
                        return
        
        # SMC Filter check (Trend Guard bypasses)
        smc_info = ""
        if cache and cache.smc_filter and self.smc_filter_enabled:
            if signal_type == "BUY":
                is_valid, smc_reason = cache.smc_filter.validate_buy_signal(price)
            else:
                is_valid, smc_reason = cache.smc_filter.validate_sell_signal(price)
            
            if not is_valid and not is_priority:
                self._stats['signals_filtered'] += 1
                print(f"[CTR Fast] 🚫 Signal FILTERED by SMC: {symbol} {signal_type}")
                print(f"           Reason: {smc_reason}")
                return
            
            smc_status = cache.smc_filter.get_status()
            trend = smc_status['trend_bias']
            smc_info = f"\n\n📊 SMC Filter: ✅ PASSED\nTrend: {trend}\nReason: {smc_reason}"
        
        # Update signal tracking
        self._last_signals[symbol] = (signal_type, now)
        self._stats['signals_sent'] += 1
        
        # Format message
        emoji = "🟢" if signal_type == "BUY" else "🔴"
        action = "ПОКУПКА" if signal_type == "BUY" else "ПРОДАЖ"
        
        if "Gap Fill" in reason:
            cross_desc = f"⚠️ Швидкий розворот! (Hidden Peak)\nSTC перетнув рівні миттєво"
        elif "Trend Guard" in reason:
            cross_desc = f"🛡️ Trend Guard (Stop Loss)\nТренд не дійшов до цілі, аварійний вихід!"
        else:
            level = self.stc.lower if signal_type == "BUY" else self.stc.upper
            direction = "знизу" if signal_type == "BUY" else "зверху"
            cross_desc = f"STC перетнув {level} {direction}"
        
        message = f"""{emoji} CTR: Сигнал {action}

Монета: {symbol}
Ціна: ${price:,.4f}
STC: {stc_value:.2f}
Таймфрейм: {self.timeframe}

{cross_desc}{smc_info}

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
                    'message': message,
                    'smc_filtered': self.smc_filter_enabled,
                    'reason': reason
                })
            except Exception as e:
                print(f"[CTR Fast] Signal callback error: {e}")
    
    def _scan_loop(self):
        """Головний цикл сканування"""
        print("[CTR Fast] Scan loop started")
        
        scan_interval = 5
        
        while self._running:
            try:
                results = self._scan_all()
                
                # Periodic logging
                ready_count = sum(1 for r in results if r['status'] != 'Neutral')
                if ready_count > 0 or self._stats['scans'] % 12 == 0:
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
        """Запустити сканер"""
        if self._running:
            print("[CTR Fast] Already running")
            return
        
        requested_symbols = [s.upper() for s in watchlist]
        
        print(f"[CTR Fast] Starting with {len(requested_symbols)} symbols: {requested_symbols}")
        
        loaded = self.preload_watchlist(requested_symbols)
        
        if loaded == 0:
            print("[CTR Fast] ❌ Failed to load any symbols")
            return
        
        with self._lock:
            self._watchlist = list(self._cache.keys())
        
        print(f"[CTR Fast] Active watchlist: {self._watchlist}")
        
        self._start_websocket()
        
        for _ in range(10):
            if self._ws_connected:
                break
            time.sleep(0.5)
        
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
        symbol = symbol.upper()
        if symbol in self._watchlist:
            return False
        if not self._load_history(symbol):
            return False
        self._watchlist.append(symbol)
        if self._ws_connected:
            self._stop_websocket()
            self._start_websocket()
        return True
    
    def remove_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol not in self._watchlist:
            return False
        self._watchlist.remove(symbol)
        with self._lock:
            if symbol in self._cache:
                del self._cache[symbol]
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
                
                stc = cache.last_stc
                if stc >= self.stc.upper:
                    status = "Overbought"
                elif stc <= self.stc.lower:
                    status = "Oversold"
                else:
                    status = "Neutral"
                
                result = {
                    'symbol': symbol,
                    'price': closes[-1],
                    'stc': round(stc, 2),
                    'prev_stc': round(cache.prev_stc, 2),
                    'status': status,
                    'candles': len(cache.klines),
                    'timeframe': self.timeframe
                }
                
                # Full SMC data (preserved from production v2.1)
                if cache.smc_filter:
                    smc_status = cache.smc_filter.get_status()
                    result['smc'] = {
                        'trend': smc_status['trend_bias'],
                        'swing_high': round(smc_status['swing_high'], 4) if smc_status['swing_high'] else None,
                        'swing_low': round(smc_status['swing_low'], 4) if smc_status['swing_low'] else None,
                        'last_hh': round(smc_status['last_hh'], 4) if smc_status['last_hh'] else None,
                        'last_hl': round(smc_status['last_hl'], 4) if smc_status['last_hl'] else None,
                        'last_lh': round(smc_status['last_lh'], 4) if smc_status['last_lh'] else None,
                        'last_ll': round(smc_status['last_ll'], 4) if smc_status['last_ll'] else None,
                    }
                
                results.append(result)
        
        return sorted(results, key=lambda x: x['symbol'])
    
    def reload_settings(self, settings: Dict):
        """Оновити налаштування"""
        if 'timeframe' in settings:
            new_tf = settings['timeframe']
            if new_tf != self.timeframe:
                self.timeframe = new_tf
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
        
        # SMC Filter settings
        if 'smc_filter_enabled' in settings:
            self.smc_filter_enabled = bool(settings['smc_filter_enabled']) and SMC_AVAILABLE
            with self._lock:
                for cache in self._cache.values():
                    if self.smc_filter_enabled and cache.smc_filter is None:
                        cache.smc_filter = self._create_smc_filter()
                    elif not self.smc_filter_enabled:
                        cache.smc_filter = None
        
        if 'smc_swing_length' in settings:
            self.smc_swing_length = int(settings['smc_swing_length'])
        if 'smc_zone_threshold' in settings:
            self.smc_zone_threshold = float(settings['smc_zone_threshold'])
        
        smc_status = "ON" if self.smc_filter_enabled else "OFF"
        print(f"[CTR Fast] Settings reloaded: TF={self.timeframe}, "
              f"Upper={self.stc.upper}, Lower={self.stc.lower}, SMC={smc_status}")
    
    def get_smc_status(self, symbol: str) -> Optional[Dict]:
        """Отримати SMC статус для символу"""
        with self._lock:
            cache = self._cache.get(symbol)
            if not cache or not cache.smc_filter:
                return None
            return cache.smc_filter.get_status()


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
    """Скинути singleton"""
    global _ctr_fast_instance
    
    with _ctr_fast_lock:
        if _ctr_fast_instance:
            _ctr_fast_instance.stop()
            _ctr_fast_instance = None
