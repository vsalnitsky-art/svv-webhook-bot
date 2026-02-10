"""
CTR Fast Scanner v2.6 - Incremental STC (Performance Release)

Based on v2.5 + incremental optimization:
- INCREMENTAL STC: Full calculate once at startup, then O(1) per candle
- ~7000x faster per scan cycle (1 operation vs 7000 loop iterations)
- Tentative updates for in-progress candles (no state pollution)
- Mathematically identical results to full recalculation

v2.5 features preserved:
- Trend Guard grace period (5 min after signal)
- Minimum cycle travel requirement for Trend Guard
- Standard cooldown 300s
- Gap Detection, SMC Filter, Anti-Oscillation

Architecture:
1. Preload 1000 candles → full STC calculate → save STCState
2. WebSocket for real-time candle updates
3. Candle in progress → tentative STC (copy state, don't commit)
4. Candle closed → commit incremental update (O(1))
5. Signal detection uses cached STC values (zero recalculation)
"""

import numpy as np
import threading
import time
import json
import websocket
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import deque
import logging
import copy

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
class STCState:
    """
    Інкрементальний стан STC розрахунку.
    Зберігає все необхідне для O(1) оновлення на нову свічку.
    """
    # EMA running values
    fast_ema: float = 0.0
    slow_ema: float = 0.0
    d_ema: float = 0.0       # EMA(k, d1_length) 
    stc_ema: float = 0.0     # EMA(kd, d2_length)
    
    # Stochastic rolling windows
    macd_window: deque = field(default_factory=lambda: deque(maxlen=10))
    d_window: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # STC values for crossover detection
    prev_stc: float = 50.0
    current_stc: float = 50.0
    
    # Initialized flag
    ready: bool = False
    
    def copy(self) -> 'STCState':
        """Швидка копія для tentative updates"""
        s = STCState(
            fast_ema=self.fast_ema,
            slow_ema=self.slow_ema,
            d_ema=self.d_ema,
            stc_ema=self.stc_ema,
            macd_window=deque(self.macd_window, maxlen=self.macd_window.maxlen),
            d_window=deque(self.d_window, maxlen=self.d_window.maxlen),
            prev_stc=self.prev_stc,
            current_stc=self.current_stc,
            ready=self.ready
        )
        return s


@dataclass
class SymbolCache:
    """Кеш даних для одного символу"""
    symbol: str
    timeframe: str
    klines: List[Kline] = field(default_factory=list)
    last_update: float = 0
    
    # STC State — v2.6: incremental
    stc_state: Optional[STCState] = None
    last_stc: float = 50.0
    prev_stc: float = 50.0
    last_calc_time: float = 0.0
    last_signal_time: float = 0.0
    
    # Cycle Memory (High-Water Mark) — v2.4
    cycle_high: float = 0.0
    cycle_low: float = 100.0
    
    # Trend State — v2.4
    last_signal_type: Optional[str] = None
    
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
        self.cycle_high = max(self.cycle_high, current_stc)
        self.cycle_low = min(self.cycle_low, current_stc)
    
    def reset_cycle_extremes(self, signal_type: str):
        if signal_type == 'BUY':
            self.cycle_high = 0.0
        elif signal_type == 'SELL':
            self.cycle_low = 100.0


# ============================================
# STC CALCULATOR with Incremental Support
# ============================================

class STCCalculator:
    """
    STC Calculator v2.6 — Dual mode:
    - full_calculate(): повний розрахунок по масиву (startup/preload)
    - incremental_update(): O(1) оновлення на одну нову свічку
    - tentative_update(): O(1) прогноз без зміни стану (in-progress candle)
    
    Математично ідентичний результат в обох режимах.
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
        self.min_candles = slow_length + cycle_length * 2 + d1_length + d2_length + 100
        
        # EMA multipliers (precomputed)
        self.alpha_fast = 2.0 / (fast_length + 1)
        self.alpha_slow = 2.0 / (slow_length + 1)
        self.alpha_d1 = 2.0 / (d1_length + 1)
        self.alpha_d2 = 2.0 / (d2_length + 1)
    
    # ------------------------------------------
    # FULL CALCULATE (startup / preload)
    # ------------------------------------------
    
    def _ema_full(self, data: np.ndarray, period: int) -> np.ndarray:
        """Full EMA over entire array"""
        if len(data) < period:
            return np.full(len(data), np.nan)
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        ema[:period - 1] = np.nan
        return ema
    
    def _stochastic_full(self, data: np.ndarray, length: int) -> np.ndarray:
        """Full stochastic over entire array"""
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
    
    def full_calculate(self, closes: np.ndarray) -> Tuple[STCState, float, float]:
        """
        Повний розрахунок STC + збереження стану для інкрементальних оновлень.
        
        Returns: (state, current_stc, prev_stc)
        """
        state = STCState()
        
        if len(closes) < self.min_candles:
            return state, 50.0, 50.0
        
        # Full pipeline
        fast_ema = self._ema_full(closes, self.fast_length)
        slow_ema = self._ema_full(closes, self.slow_length)
        macd = fast_ema - slow_ema
        k = self._stochastic_full(macd, self.cycle_length)
        d = self._ema_full(k, self.d1_length)
        kd = self._stochastic_full(d, self.cycle_length)
        stc = self._ema_full(kd, self.d2_length)
        stc = np.clip(stc, 0, 100)
        
        # Extract final state for incremental updates
        state.fast_ema = float(fast_ema[-1])
        state.slow_ema = float(slow_ema[-1])
        state.d_ema = float(d[-1]) if not np.isnan(d[-1]) else 50.0
        state.stc_ema = float(stc[-1]) if not np.isnan(stc[-1]) else 50.0
        
        # Fill stochastic windows (last cycle_length values)
        state.macd_window = deque(maxlen=self.cycle_length)
        state.d_window = deque(maxlen=self.cycle_length)
        
        for i in range(max(0, len(macd) - self.cycle_length), len(macd)):
            val = float(macd[i]) if not np.isnan(macd[i]) else 0.0
            state.macd_window.append(val)
        
        for i in range(max(0, len(d) - self.cycle_length), len(d)):
            val = float(d[i]) if not np.isnan(d[i]) else 50.0
            state.d_window.append(val)
        
        current = float(stc[-1]) if not np.isnan(stc[-1]) else 50.0
        prev = float(stc[-2]) if len(stc) > 1 and not np.isnan(stc[-2]) else current
        
        state.current_stc = current
        state.prev_stc = prev
        state.ready = True
        
        return state, current, prev
    
    def calculate(self, closes: np.ndarray) -> Tuple[float, float]:
        """Legacy compatibility: повний розрахунок без стану"""
        _, current, prev = self.full_calculate(closes)
        return current, prev
    
    # ------------------------------------------
    # INCREMENTAL UPDATE (O(1) per candle)
    # ------------------------------------------
    
    def _stoch_from_deque(self, window: deque, current_val: float) -> float:
        """Stochastic з deque window — O(window_size) ≈ O(10) = O(1)"""
        if len(window) < 2:
            return 50.0
        lowest = min(window)
        highest = max(window)
        denom = highest - lowest
        if denom > 0:
            return max(0.0, min(100.0, (current_val - lowest) / denom * 100))
        return 50.0
    
    def incremental_update(self, state: STCState, new_close: float) -> Tuple[float, float]:
        """
        O(1) оновлення STC на одну нову свічку.
        МОДИФІКУЄ state in-place.
        
        Returns: (current_stc, prev_stc)
        """
        if not state.ready:
            return 50.0, 50.0
        
        # Step 1: Update EMAs — O(1)
        state.fast_ema = self.alpha_fast * new_close + (1 - self.alpha_fast) * state.fast_ema
        state.slow_ema = self.alpha_slow * new_close + (1 - self.alpha_slow) * state.slow_ema
        macd = state.fast_ema - state.slow_ema
        
        # Step 2: Stoch(macd, cycle_length) — O(cycle_length) ≈ O(10)
        state.macd_window.append(macd)
        k = self._stoch_from_deque(state.macd_window, macd)
        
        # Step 3: EMA(k, d1_length) — O(1)
        state.d_ema = self.alpha_d1 * k + (1 - self.alpha_d1) * state.d_ema
        
        # Step 4: Stoch(d, cycle_length) — O(cycle_length) ≈ O(10)
        state.d_window.append(state.d_ema)
        kd = self._stoch_from_deque(state.d_window, state.d_ema)
        
        # Step 5: EMA(kd, d2_length) + clamp — O(1)
        state.prev_stc = state.current_stc
        raw_stc = self.alpha_d2 * kd + (1 - self.alpha_d2) * state.stc_ema
        state.stc_ema = raw_stc
        state.current_stc = max(0.0, min(100.0, raw_stc))
        
        return state.current_stc, state.prev_stc
    
    def tentative_update(self, state: STCState, new_close: float) -> Tuple[float, float]:
        """
        O(1) прогноз STC БЕЗ зміни стану (для in-progress candle).
        Використовує копію стану.
        
        Returns: (current_stc, prev_stc)
        """
        if not state.ready:
            return 50.0, 50.0
        
        # Compute on copies without modifying state
        fast_ema = self.alpha_fast * new_close + (1 - self.alpha_fast) * state.fast_ema
        slow_ema = self.alpha_slow * new_close + (1 - self.alpha_slow) * state.slow_ema
        macd = fast_ema - slow_ema
        
        # Temporary window for stoch
        tmp_macd = deque(state.macd_window, maxlen=self.cycle_length)
        tmp_macd.append(macd)
        k = self._stoch_from_deque(tmp_macd, macd)
        
        d_ema = self.alpha_d1 * k + (1 - self.alpha_d1) * state.d_ema
        
        tmp_d = deque(state.d_window, maxlen=self.cycle_length)
        tmp_d.append(d_ema)
        kd = self._stoch_from_deque(tmp_d, d_ema)
        
        raw_stc = self.alpha_d2 * kd + (1 - self.alpha_d2) * state.stc_ema
        current = max(0.0, min(100.0, raw_stc))
        
        return current, state.current_stc


# ============================================
# FAST CTR SCANNER
# ============================================

class CTRFastScanner:
    """Швидкий CTR Scanner v2.6 — Incremental STC"""
    
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
        
        # STC Calculator v2.6
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
        
        # Signal tracking — v2.5: 300s cooldown
        self._last_signals: Dict[str, Tuple[str, float]] = {}
        self._signal_cooldown = 300
        self._trend_guard_grace = 300
        self._trend_guard_min_travel = 0.30
        
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
        print(f"[CTR Fast v2.6] Initialized: TF={timeframe}, Upper={upper}, Lower={lower}, SMC={smc_status}")
    
    # ========================================
    # DATA LOADING
    # ========================================
    
    def _create_smc_filter(self) -> Optional['SMCSignalFilter']:
        if not self.smc_filter_enabled or not SMC_AVAILABLE:
            return None
        try:
            return SMCSignalFilter(
                swing_length=self.smc_swing_length,
                zone_threshold_percent=self.smc_zone_threshold,
                require_trend_for_zones=self.smc_require_trend,
            )
        except TypeError:
            return SMCSignalFilter(
                swing_length=self.smc_swing_length,
                zone_threshold_percent=self.smc_zone_threshold,
            )
    
    def _load_history(self, symbol: str) -> bool:
        """Завантажити історичні дані + full STC calculate + save state"""
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
            
            # v2.6: Full STC calculate at startup → save state
            closes = np.array([k.close for k in klines])
            stc_state, current_stc, prev_stc = self.stc.full_calculate(closes)
            
            with self._lock:
                cache = SymbolCache(
                    symbol=symbol,
                    timeframe=self.timeframe,
                    klines=klines,
                    last_update=time.time(),
                    is_ready=stc_state.ready,
                    smc_filter=smc_filter,
                    stc_state=stc_state,
                    last_stc=current_stc,
                    prev_stc=prev_stc,
                )
                self._cache[symbol] = cache
                
                if smc_filter and len(klines) > 100:
                    highs = cache.get_highs()
                    lows = cache.get_lows()
                    smc_filter.update_structure(highs, lows, closes)
            
            smc_tag = "SMC✓" if smc_filter else ""
            print(f"[CTR Fast] ✅ Loaded {symbol}: {len(klines)} candles, "
                  f"STC={current_stc:.2f} {smc_tag}")
            return True
            
        except Exception as e:
            print(f"[CTR Fast] ❌ Error loading {symbol}: {e}")
            return False
    
    def preload_watchlist(self, symbols: List[str]) -> int:
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
        """v2.6: Incremental STC on WS message"""
        try:
            data = json.loads(message)
            stream_data = data['data'] if 'stream' in data else data
            
            if stream_data.get('e') != 'kline':
                return
            
            symbol = stream_data['s']
            kline = Kline.from_websocket(stream_data)
            
            with self._lock:
                if symbol not in self._cache:
                    return
                    
                cache = self._cache[symbol]
                cache.update_kline(kline)
                self._stats['ws_messages'] += 1
                
                # v2.6: Tentative STC update for cycle extreme tracking
                # O(1) instead of O(7000) — no CPU throttle needed
                if cache.is_ready and cache.stc_state and cache.stc_state.ready:
                    if kline.is_closed:
                        # Candle CLOSED → commit incremental update
                        curr_stc, _ = self.stc.incremental_update(
                            cache.stc_state, kline.close
                        )
                        cache.update_cycle_extremes(curr_stc)
                        cache.prev_stc = cache.last_stc
                        cache.last_stc = curr_stc
                        cache.last_calc_time = time.time()
                    else:
                        # Candle in progress → tentative (no state change)
                        now = time.time()
                        if now - cache.last_calc_time >= 1.0:
                            curr_stc, _ = self.stc.tentative_update(
                                cache.stc_state, kline.close
                            )
                            cache.update_cycle_extremes(curr_stc)
                            cache.last_calc_time = now
                
                # Негайне сканування при закритті свічки
                if kline.is_closed:
                    # Release lock before scan (scan acquires lock internally)
                    pass
            
            # Outside lock: immediate scan on candle close
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
        if self._ws:
            self._ws.close()
        self._ws = None
    
    # ========================================
    # SMART SIGNAL DETECTION — v2.4/v2.5/v2.6
    # ========================================
    
    def _detect_signals(self, symbol: str, cache: SymbolCache) -> Tuple[bool, bool, float, str, str]:
        """
        v2.6: Signal detection using cached STC values (ZERO recalculation).
        STC is already computed by WS handler or _scan_all.
        
        Returns: (buy_signal, sell_signal, current_stc, status, reason)
        """
        current_stc = cache.last_stc
        prev_stc = cache.prev_stc
        
        # === 1. Standard Crossover ===
        buy_cross = prev_stc <= self.stc.lower and current_stc > self.stc.lower
        sell_cross = prev_stc >= self.stc.upper and current_stc < self.stc.upper
        
        # === 2. Gap Detection (Hidden Peak) ===
        gap_sell = (not sell_cross) and (cache.cycle_high >= self.stc.upper) and (current_stc < self.stc.upper)
        gap_buy = (not buy_cross) and (cache.cycle_low <= self.stc.lower) and (current_stc > self.stc.lower)
        
        # === 3. Trend Guard (Emergency Exit) ===
        now = time.time()
        tg_range = self.stc.upper - self.stc.lower
        tg_min_high = self.stc.lower + tg_range * self._trend_guard_min_travel
        tg_max_low = self.stc.upper - tg_range * self._trend_guard_min_travel
        tg_grace_ok = (now - cache.last_signal_time) >= self._trend_guard_grace
        
        trend_fail_sell = False
        if cache.last_signal_type == 'BUY' and current_stc < self.stc.lower:
            if prev_stc > self.stc.lower:
                if cache.cycle_high < self.stc.upper:
                    if cache.cycle_high >= tg_min_high and tg_grace_ok:
                        trend_fail_sell = True
        
        trend_fail_buy = False
        if cache.last_signal_type == 'SELL' and current_stc > self.stc.upper:
            if prev_stc < self.stc.upper:
                if cache.cycle_low > self.stc.lower:
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
        
        # Loop Protection
        if gap_sell:
            cache.cycle_high = 0.0
        if gap_buy:
            cache.cycle_low = 100.0
        
        return final_buy, final_sell, current_stc, status, reason
    
    # ========================================
    # SCANNING
    # ========================================
    
    def _scan_symbol_immediate(self, symbol: str):
        """v2.6: Негайне сканування — STC вже оновлений інкрементально"""
        signal_data = None
        
        with self._lock:
            cache = self._cache.get(symbol)
            if not cache or not cache.is_ready:
                return
            
            # v2.6: STC already updated by WS handler (incremental_update)
            # Just detect signals using cached values
            buy, sell, _, status, reason = self._detect_signals(symbol, cache)
            
            # Update SMC structure (only on candle close — infrequent)
            if cache.smc_filter:
                highs = cache.get_highs()
                lows = cache.get_lows()
                closes = cache.get_closes()
                cache.smc_filter.update_structure(highs, lows, closes)
            
            if buy or sell:
                signal_type = 'BUY' if buy else 'SELL'
                cache.last_signal_type = signal_type
                cache.last_signal_time = time.time()
                cache.reset_cycle_extremes(signal_type)
                
                signal_data = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'stc_value': cache.last_stc,
                    'price': cache.klines[-1].close if cache.klines else 0,
                    'cache': cache,
                    'reason': reason
                }
        
        # I/O outside lock
        if signal_data:
            self._process_signal(**signal_data)
    
    def _scan_all(self):
        """v2.6: Periodic scan — uses cached STC, no full recalculation"""
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
                
                # v2.6: If no WS update happened, do tentative from last closed candle
                if cache.stc_state and cache.stc_state.ready and cache.klines:
                    current_close = cache.klines[-1].close
                    # Only tentative — don't modify state during periodic scan
                    tent_stc, _ = self.stc.tentative_update(cache.stc_state, current_close)
                    cache.update_cycle_extremes(tent_stc)
                
                buy, sell, stc_val, status, reason = self._detect_signals(symbol, cache)
                
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
                    price = cache.klines[-1].close if cache.klines else 0
                    smc_status = {
                        'trend': smc_data['trend_bias'],
                        'near_support': self._is_near_smc_level(price, smc_data, 'support'),
                        'near_resistance': self._is_near_smc_level(price, smc_data, 'resistance'),
                    }
                
                results.append({
                    'symbol': symbol,
                    'stc': round(stc_val, 2),
                    'status': status,
                    'price': cache.klines[-1].close if cache.klines else 0,
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
                        'price': cache.klines[-1].close if cache.klines else 0,
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
        """Process and send signal notification (unchanged from v2.5)"""
        now = time.time()
        
        is_priority = "Trend Guard" in reason
        
        # Cooldown check
        last = self._last_signals.get(symbol)
        if last:
            last_type, last_time = last
            time_since = now - last_time
            
            if is_priority:
                if time_since < self._trend_guard_grace:
                    return
            else:
                if last_type == signal_type:
                    if time_since < self._signal_cooldown:
                        return
                else:
                    if time_since < 30:
                        return
        
        # SMC Filter check
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
        print("[CTR Fast] Scan loop started")
        
        scan_interval = 5
        
        while self._running:
            try:
                results = self._scan_all()
                
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
        results = []
        
        with self._lock:
            for symbol, cache in self._cache.items():
                if not cache.is_ready:
                    continue
                
                if not cache.klines:
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
                    'price': cache.klines[-1].close,
                    'stc': round(stc, 2),
                    'prev_stc': round(cache.prev_stc, 2),
                    'status': status,
                    'candles': len(cache.klines),
                    'timeframe': self.timeframe
                }
                
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
        """Оновити налаштування. v2.6: recalculate STC state if params changed."""
        stc_params_changed = False
        
        if 'timeframe' in settings:
            new_tf = settings['timeframe']
            if new_tf != self.timeframe:
                self.timeframe = new_tf
                if self._running:
                    self.stop()
                    self.start(self._watchlist)
                return  # Full restart handles everything
        
        if 'upper' in settings:
            new_val = float(settings['upper'])
            if new_val != self.stc.upper:
                self.stc.upper = new_val
        if 'lower' in settings:
            new_val = float(settings['lower'])
            if new_val != self.stc.lower:
                self.stc.lower = new_val
        
        # STC core params — need full recalculation
        for param in ('fast_length', 'slow_length', 'cycle_length', 'd1_length', 'd2_length'):
            if param in settings:
                new_val = int(settings[param])
                old_val = getattr(self.stc, param)
                if new_val != old_val:
                    setattr(self.stc, param, new_val)
                    stc_params_changed = True
        
        # Recalculate alphas if params changed
        if stc_params_changed:
            self.stc.alpha_fast = 2.0 / (self.stc.fast_length + 1)
            self.stc.alpha_slow = 2.0 / (self.stc.slow_length + 1)
            self.stc.alpha_d1 = 2.0 / (self.stc.d1_length + 1)
            self.stc.alpha_d2 = 2.0 / (self.stc.d2_length + 1)
            self.stc.min_candles = (self.stc.slow_length + self.stc.cycle_length * 2 
                                    + self.stc.d1_length + self.stc.d2_length + 100)
            
            # Full recalculate STC state for all symbols
            with self._lock:
                for symbol, cache in self._cache.items():
                    if cache.klines:
                        closes = cache.get_closes()
                        state, current, prev = self.stc.full_calculate(closes)
                        cache.stc_state = state
                        cache.last_stc = current
                        cache.prev_stc = prev
                        cache.is_ready = state.ready
            print(f"[CTR Fast] STC params changed — full recalculation done for all symbols")
        
        # SMC Filter settings — v2.6: recreate filters with new params
        smc_changed = False
        if 'smc_filter_enabled' in settings:
            new_enabled = bool(settings['smc_filter_enabled']) and SMC_AVAILABLE
            if new_enabled != self.smc_filter_enabled:
                self.smc_filter_enabled = new_enabled
                smc_changed = True
        
        if 'smc_swing_length' in settings:
            new_val = int(settings['smc_swing_length'])
            if new_val != self.smc_swing_length:
                self.smc_swing_length = new_val
                smc_changed = True
        if 'smc_zone_threshold' in settings:
            new_val = float(settings['smc_zone_threshold'])
            if new_val != self.smc_zone_threshold:
                self.smc_zone_threshold = new_val
                smc_changed = True
        if 'smc_require_trend' in settings:
            new_val = bool(settings['smc_require_trend'])
            if new_val != self.smc_require_trend:
                self.smc_require_trend = new_val
                smc_changed = True
        
        # v2.6 fix: recreate SMC filters with updated params
        if smc_changed:
            with self._lock:
                for symbol, cache in self._cache.items():
                    if self.smc_filter_enabled:
                        cache.smc_filter = self._create_smc_filter()
                        # Re-feed historical data
                        if cache.klines and len(cache.klines) > 100:
                            highs = cache.get_highs()
                            lows = cache.get_lows()
                            closes = cache.get_closes()
                            cache.smc_filter.update_structure(highs, lows, closes)
                    else:
                        cache.smc_filter = None
            print(f"[CTR Fast] SMC filters recreated with new params")
        
        smc_status = "ON" if self.smc_filter_enabled else "OFF"
        print(f"[CTR Fast] Settings reloaded: TF={self.timeframe}, "
              f"Upper={self.stc.upper}, Lower={self.stc.lower}, SMC={smc_status}")
    
    def get_smc_status(self, symbol: str) -> Optional[Dict]:
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
    global _ctr_fast_instance
    
    with _ctr_fast_lock:
        if _ctr_fast_instance:
            _ctr_fast_instance.stop()
            _ctr_fast_instance = None
