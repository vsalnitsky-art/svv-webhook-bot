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
    from detection.smc_structure_filter import SMCSignalFilter, SMCStructureDetector, TrendBias
    SMC_AVAILABLE = True
except ImportError:
    try:
        from smc_structure_filter import SMCSignalFilter, SMCStructureDetector, TrendBias
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
# SMC TREND FILTER (HTF 4h / 1h)
# ============================================

class SMCTrendFilter:
    """
    SMC Trend Filter — самостійний фільтр тренду на HTF (4h і 1h).
    
    Окремо завантажує свічки з Binance REST API для кожного TF.
    Визначає тренд через HH/HL/LH/LL + BOS/CHoCH (SMCStructureDetector).
    Фільтрує CTR сигнали: BUY тільки при BULLISH, SELL тільки при BEARISH.
    
    Режими (mode):
        both  — обидва TF мають підтвердити напрямок
        any   — достатньо одного TF
        4h    — тільки 4h тренд
        1h    — тільки 1h тренд
    
    NEUTRAL тренд на будь-якому TF = пропускає сигнал (не блокує).
    """
    
    REST_BASE_URL = "https://api.binance.com/api/v3"
    CANDLES_4H = 500   # 500 × 4h ≈ 83 дні — достатньо для swing_length=50
    CANDLES_1H = 500   # 500 × 1h ≈ 21 день
    CANDLES_15M = 300  # 300 × 15m ≈ 3 дні — достатньо для swing_length=20
    
    def __init__(
        self,
        enabled: bool = False,
        swing_length_4h: int = 50,
        swing_length_1h: int = 50,
        mode: str = "both",
        refresh_interval: int = 900,  # 15 хв
        block_neutral: bool = False,  # Block ALL signals when trend is NEUTRAL
        early_warning: bool = False,  # Detect 15m divergence from HTF
        swing_length_15m: int = 20,   # Swing length for 15m structure
    ):
        self.enabled = enabled and SMC_AVAILABLE
        self.swing_length_4h = swing_length_4h
        self.swing_length_1h = swing_length_1h
        self.swing_length_15m = swing_length_15m
        self.mode = mode  # both / any / 4h / 1h
        self.refresh_interval = refresh_interval
        self.block_neutral = block_neutral
        self.early_warning = early_warning
        
        # Per-symbol: { symbol: { '4h': TrendBias.name, '1h': TrendBias.name } }
        self._trends: Dict[str, Dict[str, str]] = {}
        self._detectors: Dict[str, Dict[str, SMCStructureDetector]] = {}
        self._lock = threading.RLock()
        
        # Previous trends for change detection
        self._prev_trends: Dict[str, Dict[str, str]] = {}
        
        # Background refresh
        self._refresh_thread: Optional[threading.Thread] = None
        self._refresh_running = False
        self._watchlist: List[str] = []
        
        # Telegram notifier (lazy init)
        self._notifier = None
        
        # Stats
        self._stats = {
            'signals_passed': 0,
            'signals_blocked': 0,
            'last_refresh_ms': 0,
            'symbols_loaded': 0,
            'trend_changes': 0,
            'early_warnings': 0,
        }
        
        if self.enabled:
            bn = ", block_neutral=ON" if self.block_neutral else ""
            ew = ", early_warning=ON" if self.early_warning else ""
            print(f"[SMC Trend] ✅ Initialized: mode={mode}, "
                  f"4h_swing={swing_length_4h}, 1h_swing={swing_length_1h}, "
                  f"refresh={refresh_interval}s{bn}{ew}")
    
    # ---- DATA LOADING ----
    
    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> Optional[tuple]:
        """
        Завантажити свічки з Binance REST API.
        Returns: (highs, lows, closes) numpy arrays або None при помилці.
        """
        import requests as req
        try:
            url = f"{self.REST_BASE_URL}/klines"
            params = {'symbol': symbol, 'interval': interval, 'limit': limit}
            resp = req.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                print(f"[SMC Trend] ❌ {symbol} {interval}: HTTP {resp.status_code}")
                return None
            data = resp.json()
            if not data or len(data) < 100:
                print(f"[SMC Trend] ❌ {symbol} {interval}: only {len(data)} candles")
                return None
            highs = np.array([float(k[2]) for k in data])
            lows = np.array([float(k[3]) for k in data])
            closes = np.array([float(k[4]) for k in data])
            return highs, lows, closes
        except Exception as e:
            print(f"[SMC Trend] ❌ {symbol} {interval}: {e}")
            return None
    
    def _detect_trend(self, highs: np.ndarray, lows: np.ndarray, 
                      closes: np.ndarray, swing_length: int) -> str:
        """
        Повний bar-by-bar аналіз SMC структури для точного визначення тренду.
        Обробляє кожен бар послідовно (як Pine Script), не тільки останній.
        Returns: 'BULLISH' / 'BEARISH' / 'NEUTRAL'
        """
        detector = SMCStructureDetector(swing_length=swing_length)
        start_idx = swing_length + 10
        
        # Прогоняємо всі бари послідовно для точного визначення
        for i in range(start_idx, len(highs)):
            detector.update(highs[:i+1], lows[:i+1], closes[:i+1])
        
        return detector.structure.trend_bias.name, detector
    
    def load_symbol(self, symbol: str) -> bool:
        """Завантажити 4h і 1h дані для одного символу"""
        if not self.enabled or not SMC_AVAILABLE:
            return False
        
        trends = {}
        detectors = {}
        
        # 4h
        if self.mode in ('both', 'any', '4h'):
            data_4h = self._fetch_klines(symbol, '4h', self.CANDLES_4H)
            if data_4h:
                trend_4h, det_4h = self._detect_trend(*data_4h, self.swing_length_4h)
                trends['4h'] = trend_4h
                detectors['4h'] = det_4h
            else:
                trends['4h'] = 'NEUTRAL'
                detectors['4h'] = None
        
        # 1h
        if self.mode in ('both', 'any', '1h'):
            data_1h = self._fetch_klines(symbol, '1h', self.CANDLES_1H)
            if data_1h:
                trend_1h, det_1h = self._detect_trend(*data_1h, self.swing_length_1h)
                trends['1h'] = trend_1h
                detectors['1h'] = det_1h
            else:
                trends['1h'] = 'NEUTRAL'
                detectors['1h'] = None
        
        with self._lock:
            self._trends[symbol] = trends
            self._detectors[symbol] = detectors
        
        t4 = trends.get('4h', '-')
        t1 = trends.get('1h', '-')
        print(f"[SMC Trend] {symbol}: 4h={t4}, 1h={t1}")
        return True
    
    def load_symbols(self, symbols: List[str]):
        """Завантажити дані для всіх символів"""
        start = time.time()
        loaded = 0
        for symbol in symbols:
            if self.load_symbol(symbol):
                loaded += 1
            time.sleep(0.15)  # Rate limit
        
        elapsed = (time.time() - start) * 1000
        self._stats['symbols_loaded'] = loaded
        self._stats['last_refresh_ms'] = elapsed
        print(f"[SMC Trend] ✅ Loaded {loaded}/{len(symbols)} symbols in {elapsed:.0f}ms")
    
    def remove_symbol(self, symbol: str):
        with self._lock:
            self._trends.pop(symbol, None)
            self._detectors.pop(symbol, None)
    
    # ---- BACKGROUND REFRESH ----
    
    def _refresh_loop(self):
        """Фоновий потік оновлення HTF даних + early warning"""
        while self._refresh_running:
            time.sleep(self.refresh_interval)
            if not self._refresh_running:
                break
            
            # Save previous trends for change detection
            with self._lock:
                prev = {s: dict(t) for s, t in self._trends.items()}
            
            start = time.time()
            symbols = list(self._watchlist)
            for symbol in symbols:
                if not self._refresh_running:
                    break
                self.load_symbol(symbol)
                time.sleep(0.15)
            
            elapsed = (time.time() - start) * 1000
            self._stats['last_refresh_ms'] = elapsed
            print(f"[SMC Trend] 🔄 Refreshed {len(symbols)} symbols in {elapsed:.0f}ms")
            
            # Detect HTF trend changes
            self._detect_htf_changes(prev, symbols)
            
            # Early warning: 15m divergence from HTF
            if self.early_warning:
                self._check_early_warnings(symbols)
    
    def _get_notifier(self):
        """Lazy-init Telegram notifier"""
        if self._notifier is None:
            try:
                from alerts.telegram_notifier import get_notifier
                self._notifier = get_notifier()
            except:
                self._notifier = False  # Mark as unavailable
        return self._notifier if self._notifier else None
    
    def _send_notification(self, message: str):
        """Send Telegram notification"""
        notifier = self._get_notifier()
        if notifier:
            try:
                notifier.send_message(message)
            except Exception as e:
                print(f"[SMC Trend] ❌ Notification error: {e}")
    
    def _detect_htf_changes(self, prev_trends: Dict, symbols: List[str]):
        """Detect and notify about HTF trend changes"""
        for symbol in symbols:
            prev = prev_trends.get(symbol, {})
            curr = self._trends.get(symbol, {})
            
            if not prev or not curr:
                continue
            
            changes = []
            for tf in ('4h', '1h'):
                old_t = prev.get(tf)
                new_t = curr.get(tf)
                if old_t and new_t and old_t != new_t and old_t != 'N/A':
                    emoji_map = {'BULLISH': '🟢', 'BEARISH': '🔴', 'NEUTRAL': '⚪'}
                    old_e = emoji_map.get(old_t, '❓')
                    new_e = emoji_map.get(new_t, '❓')
                    changes.append(f"  {tf}: {old_e}{old_t} → {new_e}{new_t}")
            
            if changes:
                self._stats['trend_changes'] += 1
                msg = (
                    f"🔄 HTF TREND CHANGE: {symbol}\n"
                    + "\n".join(changes)
                )
                print(f"[SMC Trend] {msg}")
                self._send_notification(msg)
    
    def _check_early_warnings(self, symbols: List[str]):
        """
        Fetch 15m structure and check for divergence with HTF trend.
        If 15m trend contradicts HTF → early warning notification.
        """
        for symbol in symbols:
            if not self._refresh_running:
                break
            
            htf = self._trends.get(symbol, {})
            t4h = htf.get('4h', 'NEUTRAL')
            t1h = htf.get('1h', 'NEUTRAL')
            
            # Determine dominant HTF trend
            if t4h == t1h and t4h in ('BULLISH', 'BEARISH'):
                htf_dominant = t4h
            elif self.mode == '4h' and t4h in ('BULLISH', 'BEARISH'):
                htf_dominant = t4h
            elif self.mode == '1h' and t1h in ('BULLISH', 'BEARISH'):
                htf_dominant = t1h
            elif t4h in ('BULLISH', 'BEARISH') and t1h == 'NEUTRAL':
                htf_dominant = t4h
            elif t1h in ('BULLISH', 'BEARISH') and t4h == 'NEUTRAL':
                htf_dominant = t1h
            else:
                # No clear HTF trend or conflicting → skip
                continue
            
            # Fetch 15m klines
            data_15m = self._fetch_klines(symbol, '15m', self.CANDLES_15M)
            if not data_15m:
                continue
            
            trend_15m, _ = self._detect_trend(*data_15m, self.swing_length_15m)
            
            # Check divergence: 15m opposite to HTF dominant
            if (htf_dominant == 'BULLISH' and trend_15m == 'BEARISH') or \
               (htf_dominant == 'BEARISH' and trend_15m == 'BULLISH'):
                self._stats['early_warnings'] += 1
                
                htf_emoji = '🟢' if htf_dominant == 'BULLISH' else '🔴'
                m15_emoji = '🟢' if trend_15m == 'BULLISH' else '🔴'
                
                msg = (
                    f"⚠️ EARLY WARNING: {symbol}\n"
                    f"  HTF: {htf_emoji}{htf_dominant} (4h={t4h}, 1h={t1h})\n"
                    f"  15m: {m15_emoji}{trend_15m} ← структура протилежна!\n"
                    f"  Потенційна зміна тренду"
                )
                print(f"[SMC Trend] {msg}")
                self._send_notification(msg)
            
            time.sleep(0.15)  # Rate limit
    
    def start_refresh(self, watchlist: List[str]):
        """Запустити фоновий потік оновлення"""
        self._watchlist = list(watchlist)
        if self._refresh_thread and self._refresh_thread.is_alive():
            return
        self._refresh_running = True
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
        print(f"[SMC Trend] 🔄 Background refresh started (every {self.refresh_interval}s)")
    
    def stop_refresh(self):
        """Зупинити фоновий потік"""
        self._refresh_running = False
        if self._refresh_thread:
            self._refresh_thread.join(timeout=5)
        self._refresh_thread = None
    
    # ---- SIGNAL VALIDATION ----
    
    def validate_signal(self, symbol: str, signal_type: str) -> Tuple[bool, str]:
        """
        Перевірити чи сигнал відповідає HTF тренду.
        
        block_neutral=False: BUY при BULLISH або NEUTRAL, SELL при BEARISH або NEUTRAL
        block_neutral=True:  BUY ТІЛЬКИ при BULLISH, SELL ТІЛЬКИ при BEARISH
        
        Returns: (is_valid, reason)
        """
        with self._lock:
            trends = self._trends.get(symbol)
        
        if not trends:
            return True, "No HTF data — passed"
        
        t4h = trends.get('4h', 'NEUTRAL')
        t1h = trends.get('1h', 'NEUTRAL')
        
        # Визначаємо допустимий напрямок
        required = 'BULLISH' if signal_type == 'BUY' else 'BEARISH'
        opposite = 'BEARISH' if signal_type == 'BUY' else 'BULLISH'
        
        if self.mode == '4h':
            if t4h == opposite:
                self._stats['signals_blocked'] += 1
                return False, f"4h trend is {t4h} — {signal_type} blocked"
            if self.block_neutral and t4h == 'NEUTRAL':
                self._stats['signals_blocked'] += 1
                return False, f"4h trend is NEUTRAL — {signal_type} blocked (block_neutral)"
            self._stats['signals_passed'] += 1
            return True, f"4h={t4h}"
        
        elif self.mode == '1h':
            if t1h == opposite:
                self._stats['signals_blocked'] += 1
                return False, f"1h trend is {t1h} — {signal_type} blocked"
            if self.block_neutral and t1h == 'NEUTRAL':
                self._stats['signals_blocked'] += 1
                return False, f"1h trend is NEUTRAL — {signal_type} blocked (block_neutral)"
            self._stats['signals_passed'] += 1
            return True, f"1h={t1h}"
        
        elif self.mode == 'both':
            # Блок якщо хоча б один = opposite
            if t4h == opposite or t1h == opposite:
                blocked_by = []
                if t4h == opposite: blocked_by.append(f"4h={t4h}")
                if t1h == opposite: blocked_by.append(f"1h={t1h}")
                self._stats['signals_blocked'] += 1
                return False, f"{', '.join(blocked_by)} — {signal_type} blocked"
            # Блок якщо хоча б один = NEUTRAL (при block_neutral)
            if self.block_neutral and (t4h == 'NEUTRAL' or t1h == 'NEUTRAL'):
                neutral_by = []
                if t4h == 'NEUTRAL': neutral_by.append("4h=NEUTRAL")
                if t1h == 'NEUTRAL': neutral_by.append("1h=NEUTRAL")
                self._stats['signals_blocked'] += 1
                return False, f"{', '.join(neutral_by)} — {signal_type} blocked (block_neutral)"
            self._stats['signals_passed'] += 1
            return True, f"4h={t4h}, 1h={t1h}"
        
        elif self.mode == 'any':
            # Блокуємо тільки якщо ОБИДВА = opposite
            if t4h == opposite and t1h == opposite:
                self._stats['signals_blocked'] += 1
                return False, f"Both 4h={t4h}, 1h={t1h} — {signal_type} blocked"
            # При block_neutral: блокуємо якщо ОБИДВА = opposite або NEUTRAL (жоден не required)
            if self.block_neutral:
                t4h_ok = (t4h == required)
                t1h_ok = (t1h == required)
                if not t4h_ok and not t1h_ok:
                    self._stats['signals_blocked'] += 1
                    return False, f"4h={t4h}, 1h={t1h} — no {required} trend (block_neutral)"
            self._stats['signals_passed'] += 1
            return True, f"4h={t4h}, 1h={t1h}"
        
        return True, "Unknown mode — passed"
    
    def get_symbol_trends(self, symbol: str) -> Dict[str, str]:
        """Отримати тренди для символу"""
        with self._lock:
            return dict(self._trends.get(symbol, {'4h': 'N/A', '1h': 'N/A'}))
    
    def get_all_trends(self) -> Dict[str, Dict[str, str]]:
        """Отримати тренди для всіх символів"""
        with self._lock:
            return {s: dict(t) for s, t in self._trends.items()}
    
    def get_status(self) -> Dict:
        """Повний статус фільтра"""
        return {
            'enabled': self.enabled,
            'mode': self.mode,
            'block_neutral': self.block_neutral,
            'early_warning': self.early_warning,
            'swing_length_4h': self.swing_length_4h,
            'swing_length_1h': self.swing_length_1h,
            'swing_length_15m': self.swing_length_15m,
            'refresh_interval': self.refresh_interval,
            'symbols_loaded': self._stats['symbols_loaded'],
            'signals_passed': self._stats['signals_passed'],
            'signals_blocked': self._stats['signals_blocked'],
            'trend_changes': self._stats['trend_changes'],
            'early_warnings': self._stats['early_warnings'],
            'last_refresh_ms': self._stats['last_refresh_ms'],
            'trends': self.get_all_trends(),
        }
    
    def update_settings(self, **kwargs):
        """Оновити налаштування (hot-reload)"""
        need_reload = False
        
        if 'enabled' in kwargs:
            new_val = bool(kwargs['enabled']) and SMC_AVAILABLE
            if new_val != self.enabled:
                self.enabled = new_val
                if new_val and self._watchlist:
                    need_reload = True
                elif not new_val:
                    self.stop_refresh()
        
        if 'swing_length_4h' in kwargs:
            new_val = int(kwargs['swing_length_4h'])
            if new_val != self.swing_length_4h:
                self.swing_length_4h = new_val
                need_reload = True
        
        if 'swing_length_1h' in kwargs:
            new_val = int(kwargs['swing_length_1h'])
            if new_val != self.swing_length_1h:
                self.swing_length_1h = new_val
                need_reload = True
        
        if 'mode' in kwargs:
            new_val = str(kwargs['mode'])
            if new_val != self.mode:
                self.mode = new_val
                need_reload = True
        
        if 'refresh_interval' in kwargs:
            new_val = int(kwargs['refresh_interval'])
            if new_val != self.refresh_interval:
                self.refresh_interval = new_val
                # Restart refresh thread with new interval
                if self._refresh_running:
                    self.stop_refresh()
                    if self.enabled and self._watchlist:
                        self.start_refresh(self._watchlist)
        
        if 'block_neutral' in kwargs:
            self.block_neutral = bool(kwargs['block_neutral'])
        
        if 'early_warning' in kwargs:
            self.early_warning = bool(kwargs['early_warning'])
        
        if 'swing_length_15m' in kwargs:
            new_val = int(kwargs['swing_length_15m'])
            if new_val != self.swing_length_15m:
                self.swing_length_15m = new_val
        
        # Reload data if structure params changed
        if need_reload and self.enabled and self._watchlist:
            threading.Thread(
                target=self._reload_all_data, daemon=True
            ).start()
    
    def _reload_all_data(self):
        """Перезавантажити дані у фоновому потоці"""
        print(f"[SMC Trend] 🔄 Reloading data after settings change...")
        self.load_symbols(self._watchlist)
        if not self._refresh_running:
            self.start_refresh(self._watchlist)


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
        # Optional signal filters (OFF = 100% Pine Script original)
        use_trend_guard: bool = False,
        use_gap_detection: bool = False,
        use_cooldown: bool = True,
        cooldown_seconds: int = 300,
        # SMC Filter settings (per-symbol structure levels)
        smc_filter_enabled: bool = False,
        smc_swing_length: int = 50,
        smc_zone_threshold: float = 1.0,
        smc_require_trend: bool = True,
        # SMC Trend Filter (HTF direction — 4h/1h)
        smc_trend_enabled: bool = False,
        smc_trend_swing_4h: int = 50,
        smc_trend_swing_1h: int = 50,
        smc_trend_mode: str = "both",
        smc_trend_refresh: int = 900,
        smc_trend_block_neutral: bool = False,
        smc_trend_early_warning: bool = False,
        smc_trend_swing_15m: int = 20,
    ):
        self.timeframe = timeframe
        self.on_signal = on_signal
        
        # STC Calculator v2.6
        self.stc = STCCalculator(
            fast_length, slow_length, cycle_length,
            d1_length, d2_length, upper, lower
        )
        
        # Optional signal filters
        self.use_trend_guard = use_trend_guard
        self.use_gap_detection = use_gap_detection
        self.use_cooldown = use_cooldown
        
        # SMC Filter settings (existing — per-symbol structure levels)
        self.smc_filter_enabled = smc_filter_enabled and SMC_AVAILABLE
        self.smc_swing_length = smc_swing_length
        self.smc_zone_threshold = smc_zone_threshold
        self.smc_require_trend = smc_require_trend
        
        # SMC Trend Filter (HTF 4h/1h)
        self._smc_trend_filter = SMCTrendFilter(
            enabled=smc_trend_enabled,
            swing_length_4h=smc_trend_swing_4h,
            swing_length_1h=smc_trend_swing_1h,
            mode=smc_trend_mode,
            refresh_interval=smc_trend_refresh,
            block_neutral=smc_trend_block_neutral,
            early_warning=smc_trend_early_warning,
            swing_length_15m=smc_trend_swing_15m,
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
        
        # Signal tracking
        self._last_signals: Dict[str, Tuple[str, float]] = {}
        self._signal_cooldown = cooldown_seconds
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
        
        # Log filter state
        filters = []
        if self.use_trend_guard: filters.append("TG")
        if self.use_gap_detection: filters.append("GAP")
        if self.use_cooldown: filters.append(f"CD={cooldown_seconds}s")
        if self.smc_filter_enabled: filters.append("SMC")
        if self._smc_trend_filter.enabled: filters.append(f"SMC-Trend({smc_trend_mode})")
        filter_str = "+".join(filters) if filters else "ORIGINAL (no filters)"
        print(f"[CTR Fast v2.6] Initialized: TF={timeframe}, Upper={upper}, Lower={lower}, Filters: {filter_str}")
    
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
    # SIGNAL DETECTION — v2.6 (Original + Optional Filters)
    # ========================================
    
    def _detect_signals(self, symbol: str, cache: SymbolCache) -> Tuple[bool, bool, float, str, str]:
        """
        v2.6: Signal detection — Primary = 100% Pine Script crossover/crossunder.
        Gap Detection, Trend Guard — only when enabled via flags.
        STC is already computed by WS handler (ZERO recalculation).
        
        Returns: (buy_signal, sell_signal, current_stc, status, reason)
        """
        current_stc = cache.last_stc
        prev_stc = cache.prev_stc
        
        # === PRIMARY: Standard Crossover (100% Pine Script original) ===
        # ta.crossover(stc, lower): prev <= lower AND current > lower
        # ta.crossunder(stc, upper): prev >= upper AND current < upper
        buy_cross = prev_stc <= self.stc.lower and current_stc > self.stc.lower
        sell_cross = prev_stc >= self.stc.upper and current_stc < self.stc.upper
        
        # Guard: prevent same signal type firing twice (race between WS and periodic scan)
        if buy_cross and cache.last_signal_type == 'BUY':
            buy_cross = False
        if sell_cross and cache.last_signal_type == 'SELL':
            sell_cross = False
        
        # === OPTIONAL FILTER: Gap Detection ===
        gap_sell = False
        gap_buy = False
        if self.use_gap_detection:
            gap_sell = (not sell_cross) and (cache.cycle_high >= self.stc.upper) and (current_stc < self.stc.upper)
            gap_buy = (not buy_cross) and (cache.cycle_low <= self.stc.lower) and (current_stc > self.stc.lower)
        
        # === OPTIONAL FILTER: Trend Guard ===
        trend_fail_sell = False
        trend_fail_buy = False
        if self.use_trend_guard:
            now = time.time()
            tg_range = self.stc.upper - self.stc.lower
            tg_min_high = self.stc.lower + tg_range * self._trend_guard_min_travel
            tg_max_low = self.stc.upper - tg_range * self._trend_guard_min_travel
            tg_grace_ok = (now - cache.last_signal_time) >= self._trend_guard_grace
            
            # After BUY: STC fell back below lower without reaching upper
            if cache.last_signal_type == 'BUY' and current_stc < self.stc.lower:
                if prev_stc > self.stc.lower:
                    if cache.cycle_high < self.stc.upper:
                        if cache.cycle_high >= tg_min_high and tg_grace_ok:
                            trend_fail_sell = True
            
            # After SELL: STC rose back above upper without reaching lower
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
        
        # Loop Protection (only relevant when gap detection is on)
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
                
                # SMC Trend info for results
                smc_trend = None
                if self._smc_trend_filter and self._smc_trend_filter.enabled:
                    smc_trend = self._smc_trend_filter.get_symbol_trends(symbol)
                
                results.append({
                    'symbol': symbol,
                    'stc': round(stc_val, 2),
                    'status': status,
                    'price': cache.klines[-1].close if cache.klines else 0,
                    'buy_signal': buy,
                    'sell_signal': sell,
                    'smc': smc_status,
                    'smc_trend': smc_trend
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
        """Process and send signal notification. Cooldown is optional (v2.6)."""
        now = time.time()
        
        is_priority = "Trend Guard" in reason
        
        # OPTIONAL: Cooldown check (OFF = Pine Script original, no cooldown)
        if self.use_cooldown:
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
        
        # SMC Trend Filter (HTF direction — 4h/1h)
        smc_trend_info = ""
        if self._smc_trend_filter and self._smc_trend_filter.enabled:
            trend_valid, trend_reason = self._smc_trend_filter.validate_signal(symbol, signal_type)
            
            if not trend_valid and not is_priority:
                self._stats['signals_filtered'] += 1
                print(f"[CTR Fast] 🚫 Signal FILTERED by SMC Trend: {symbol} {signal_type}")
                print(f"           Reason: {trend_reason}")
                return
            
            trends = self._smc_trend_filter.get_symbol_trends(symbol)
            smc_trend_info = f"\n\n🔭 SMC Trend: ✅ PASSED\n4h: {trends.get('4h','N/A')} | 1h: {trends.get('1h','N/A')}"
        
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

{cross_desc}{smc_info}{smc_trend_info}

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
                    'smc_trend_filtered': bool(self._smc_trend_filter and self._smc_trend_filter.enabled),
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
    
    def get_smc_trend_filter(self) -> 'SMCTrendFilter':
        """Отримати SMC Trend Filter для зовнішнього доступу (API/UI)"""
        return self._smc_trend_filter
    
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
        
        # SMC Trend Filter: load HTF data and start refresh
        if self._smc_trend_filter and self._smc_trend_filter.enabled:
            print(f"[SMC Trend] Loading HTF data for {len(self._watchlist)} symbols...")
            self._smc_trend_filter.load_symbols(self._watchlist)
            self._smc_trend_filter.start_refresh(self._watchlist)
        
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
        # Stop SMC Trend refresh
        if self._smc_trend_filter:
            self._smc_trend_filter.stop_refresh()
        print("[CTR Fast] ✅ Stopped")
    
    def add_symbol(self, symbol: str) -> bool:
        symbol = symbol.upper()
        if symbol in self._watchlist:
            return False
        if not self._load_history(symbol):
            return False
        self._watchlist.append(symbol)
        # Load SMC Trend data for new symbol
        if self._smc_trend_filter and self._smc_trend_filter.enabled:
            self._smc_trend_filter.load_symbol(symbol)
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
        # Cleanup SMC Trend data
        if self._smc_trend_filter:
            self._smc_trend_filter.remove_symbol(symbol)
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
                
                # SMC Trend Filter (HTF 4h/1h)
                if self._smc_trend_filter and self._smc_trend_filter.enabled:
                    result['smc_trend'] = self._smc_trend_filter.get_symbol_trends(symbol)
                
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
        
        # Optional signal filter settings
        if 'use_trend_guard' in settings:
            self.use_trend_guard = bool(settings['use_trend_guard'])
        if 'use_gap_detection' in settings:
            self.use_gap_detection = bool(settings['use_gap_detection'])
        if 'use_cooldown' in settings:
            self.use_cooldown = bool(settings['use_cooldown'])
        if 'cooldown_seconds' in settings:
            self._signal_cooldown = int(settings['cooldown_seconds'])
        
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
        
        # SMC Trend Filter settings — hot-reload
        smc_trend_kwargs = {}
        for key in ('smc_trend_enabled', 'smc_trend_swing_4h', 'smc_trend_swing_1h', 
                     'smc_trend_mode', 'smc_trend_refresh', 'smc_trend_block_neutral',
                     'smc_trend_early_warning', 'smc_trend_swing_15m'):
            if key in settings:
                mapped = key.replace('smc_trend_', '')
                if mapped in ('enabled', 'block_neutral', 'early_warning'):
                    smc_trend_kwargs[mapped] = bool(settings[key])
                elif mapped == 'mode':
                    smc_trend_kwargs[mapped] = str(settings[key])
                else:
                    smc_trend_kwargs[mapped] = int(settings[key])
        
        if smc_trend_kwargs and self._smc_trend_filter:
            self._smc_trend_filter.update_settings(**smc_trend_kwargs)
            print(f"[CTR Fast] SMC Trend settings updated: {smc_trend_kwargs}")
        
        filters = []
        if self.use_trend_guard: filters.append("TG")
        if self.use_gap_detection: filters.append("GAP")
        if self.use_cooldown: filters.append(f"CD={self._signal_cooldown}s")
        if self.smc_filter_enabled: filters.append("SMC")
        if self._smc_trend_filter and self._smc_trend_filter.enabled: 
            filters.append(f"SMC-Trend({self._smc_trend_filter.mode})")
        filter_str = "+".join(filters) if filters else "ORIGINAL"
        print(f"[CTR Fast] Settings reloaded: TF={self.timeframe}, "
              f"Upper={self.stc.upper}, Lower={self.stc.lower}, Filters: {filter_str}")
    
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
