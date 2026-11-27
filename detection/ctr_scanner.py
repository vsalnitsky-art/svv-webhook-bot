"""
CTR Scanner - Cyclic Trend Reversal (STC-based)
================================================

100% відтворення логіки з Pine Script індикатора:
OB + RSI/MFI + SMC + CTR Unified v4.0

CTR Algorithm (Schaff Trend Cycle):
1. MACD = EMA(close, fast) - EMA(close, slow)
2. k = Stochastic(MACD, cycle_length)
3. d = EMA(k, d1_length)  
4. kd = Stochastic(d, cycle_length)
5. stc = EMA(kd, d2_length)
6. stc = clamp(stc, 0, 100)

Signals:
- BUY: stc crossover lower (25)
- SELL: stc crossunder upper (75)

v1.0 - Initial implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import time
import threading

# Binance connector
from core.binance_connector import get_binance_connector


class CTRScanner:
    """
    CTR (Cyclic Trend Reversal) Scanner
    
    Точне відтворення логіки з TradingView Pine Script.
    """
    
    # Default parameters (matching Pine Script)
    DEFAULT_PARAMS = {
        'fast_length': 21,      # Довжина швидкого періоду MACD
        'slow_length': 50,      # Довжина повільного періоду MACD
        'cycle_length': 10,     # Довжина циклу
        'd1_length': 3,         # Довжина першого %D
        'd2_length': 3,         # Довжина другого %D
        'upper': 75,            # Верхня межа (overbought)
        'lower': 25,            # Нижня межа (oversold)
    }
    
    def __init__(self, db=None):
        """Initialize CTR Scanner"""
        self.db = db
        self.fetcher = get_binance_connector()
        
        # Load settings from DB or use defaults
        self._load_settings()
        
        # State tracking for each symbol
        self.symbol_states: Dict[str, Dict] = {}
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        print(f"[CTR] Scanner initialized with params:")
        print(f"  MACD: fast={self.fast_length}, slow={self.slow_length}")
        print(f"  Cycle: {self.cycle_length}, D1: {self.d1_length}, D2: {self.d2_length}")
        print(f"  Levels: upper={self.upper}, lower={self.lower}")
    
    def _load_settings(self):
        """Load settings from database"""
        if self.db:
            self.fast_length = int(self.db.get_setting('ctr_fast_length', self.DEFAULT_PARAMS['fast_length']))
            self.slow_length = int(self.db.get_setting('ctr_slow_length', self.DEFAULT_PARAMS['slow_length']))
            self.cycle_length = int(self.db.get_setting('ctr_cycle_length', self.DEFAULT_PARAMS['cycle_length']))
            self.d1_length = int(self.db.get_setting('ctr_d1_length', self.DEFAULT_PARAMS['d1_length']))
            self.d2_length = int(self.db.get_setting('ctr_d2_length', self.DEFAULT_PARAMS['d2_length']))
            self.upper = int(self.db.get_setting('ctr_upper', self.DEFAULT_PARAMS['upper']))
            self.lower = int(self.db.get_setting('ctr_lower', self.DEFAULT_PARAMS['lower']))
            self.timeframe = self.db.get_setting('ctr_timeframe', '15m')
        else:
            self.fast_length = self.DEFAULT_PARAMS['fast_length']
            self.slow_length = self.DEFAULT_PARAMS['slow_length']
            self.cycle_length = self.DEFAULT_PARAMS['cycle_length']
            self.d1_length = self.DEFAULT_PARAMS['d1_length']
            self.d2_length = self.DEFAULT_PARAMS['d2_length']
            self.upper = self.DEFAULT_PARAMS['upper']
            self.lower = self.DEFAULT_PARAMS['lower']
            self.timeframe = '15m'
    
    def reload_settings(self):
        """Reload settings from database"""
        self._load_settings()
        print(f"[CTR] Settings reloaded: TF={self.timeframe}, upper={self.upper}, lower={self.lower}")
    
    # ========================================
    # CORE CTR CALCULATIONS (matching Pine)
    # ========================================
    
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> np.ndarray:
        """
        Calculate EMA (Exponential Moving Average)
        Matches TradingView ta.ema()
        """
        if len(data) < period:
            return np.full(len(data), np.nan)
        
        alpha = 2 / (period + 1)
        ema = np.zeros(len(data))
        
        # Initialize with SMA for first value
        ema[period-1] = np.mean(data[:period])
        
        # Calculate EMA
        for i in range(period, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        # Fill initial values
        ema[:period-1] = np.nan
        
        return ema
    
    @staticmethod
    def _stochastic(data: np.ndarray, length: int) -> np.ndarray:
        """
        Calculate Stochastic
        Matches Pine Script ctr_stoch() function:
        
        ctr_stoch(src_val, length) =>
            lowest_val = ta.lowest(src_val, length)
            highest_val = ta.highest(src_val, length)
            denominator = highest_val - lowest_val
            denominator == 0 ? 50 : (src_val - lowest_val) / denominator * 100
        """
        if len(data) < length:
            return np.full(len(data), 50.0)
        
        result = np.zeros(len(data))
        
        for i in range(len(data)):
            if i < length - 1:
                result[i] = 50.0
            else:
                window = data[i - length + 1:i + 1]
                
                # Handle NaN values in window
                valid_window = window[~np.isnan(window)]
                if len(valid_window) < 2:
                    result[i] = 50.0
                    continue
                
                lowest = np.min(valid_window)
                highest = np.max(valid_window)
                denominator = highest - lowest
                
                current_val = data[i]
                if np.isnan(current_val) or denominator == 0:
                    result[i] = 50.0
                else:
                    result[i] = (current_val - lowest) / denominator * 100
        
        return result
    
    def calculate_stc(self, closes: np.ndarray) -> np.ndarray:
        """
        Calculate STC (Schaff Trend Cycle)
        
        Exact implementation from Pine Script:
        
        ctr_fastEMA = ta.ema(ctr_src, ctr_fastLength)
        ctr_slowEMA = ta.ema(ctr_src, ctr_slowLength)
        ctr_macd = ctr_fastEMA - ctr_slowEMA
        
        ctr_k = ctr_stoch(ctr_macd, ctr_cycleLength)
        ctr_d = ta.ema(ctr_k, ctr_d1Length)
        ctr_kd = ctr_stoch(ctr_d, ctr_cycleLength)
        ctr_stc = ta.ema(ctr_kd, ctr_d2Length)
        ctr_stc := math.max(math.min(ctr_stc, 100), 0)
        """
        # Step 1: Calculate MACD
        fast_ema = self._ema(closes, self.fast_length)
        slow_ema = self._ema(closes, self.slow_length)
        macd = fast_ema - slow_ema
        
        # Step 2: Stochastic of MACD
        k = self._stochastic(macd, self.cycle_length)
        
        # Step 3: EMA of k
        d = self._ema(k, self.d1_length)
        
        # Step 4: Stochastic of d
        kd = self._stochastic(d, self.cycle_length)
        
        # Step 5: EMA of kd = STC
        stc = self._ema(kd, self.d2_length)
        
        # Step 6: Clamp to 0-100
        stc = np.clip(stc, 0, 100)
        
        return stc
    
    def detect_signals(self, closes: np.ndarray) -> Tuple[bool, bool, float, str]:
        """
        Detect CTR signals
        
        Returns:
            (buy_signal, sell_signal, current_stc, status)
        
        Pine Script logic:
        ctr_buySignal = ctr_enabled and ta.crossover(ctr_stc, ctr_lower)
        ctr_sellSignal = ctr_enabled and ta.crossunder(ctr_stc, ctr_upper)
        """
        if len(closes) < self.slow_length + self.cycle_length + 10:
            return False, False, 50.0, "Недостатньо даних"
        
        stc = self.calculate_stc(closes)
        
        current_stc = stc[-1]
        prev_stc = stc[-2] if len(stc) > 1 else current_stc
        
        # Handle NaN values
        if np.isnan(current_stc):
            return False, False, 50.0, "Недостатньо даних"
        if np.isnan(prev_stc):
            prev_stc = current_stc
        
        # Crossover detection (matching ta.crossover / ta.crossunder)
        # Convert to Python bool for JSON serialization
        buy_signal = bool(prev_stc <= self.lower and current_stc > self.lower)
        sell_signal = bool(prev_stc >= self.upper and current_stc < self.upper)
        
        # Status
        if current_stc >= self.upper:
            status = "Overbought"
        elif current_stc <= self.lower:
            status = "Oversold"
        else:
            status = "Neutral"
        
        return buy_signal, sell_signal, float(current_stc), status
    
    # ========================================
    # DATA FETCHING
    # ========================================
    
    def get_required_candles(self) -> int:
        """Calculate minimum candles needed for accurate STC"""
        # Need enough for: slow_ema + cycle + d1 + d2 + buffer
        return self.slow_length + self.cycle_length * 2 + self.d1_length + self.d2_length + 50
    
    def fetch_data(self, symbol: str, timeframe: str = None) -> Optional[np.ndarray]:
        """
        Fetch klines from Binance and return close prices
        """
        tf = timeframe or self.timeframe
        limit = self.get_required_candles()
        
        try:
            klines = self.fetcher.get_klines(symbol, tf, limit=limit)
            if not klines or len(klines) < limit // 2:
                return None
            
            closes = np.array([float(k['close']) for k in klines])
            return closes
        except Exception as e:
            print(f"[CTR] Error fetching {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from Binance ticker"""
        try:
            ticker = self.fetcher.get_ticker(symbol)
            if ticker:
                return float(ticker.get('lastPrice', 0))
        except Exception as e:
            print(f"[CTR] Error getting price for {symbol}: {e}")
        return None
    
    # ========================================
    # SCANNING
    # ========================================
    
    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Scan single symbol for CTR signals
        
        Returns dict with signal info or None
        """
        closes = self.fetch_data(symbol)
        if closes is None:
            return None
        
        buy_signal, sell_signal, stc_value, status = self.detect_signals(closes)
        
        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            current_price = closes[-1]
        
        # Track state changes
        with self._lock:
            prev_state = self.symbol_states.get(symbol, {})
            prev_stc = prev_state.get('stc', 50.0)
            
            # Update state
            self.symbol_states[symbol] = {
                'stc': stc_value,
                'status': status,
                'price': current_price,
                'updated_at': datetime.now(timezone.utc),
            }
        
        result = {
            'symbol': symbol,
            'stc': round(float(stc_value), 2) if not np.isnan(stc_value) else 50.0,
            'prev_stc': round(float(prev_stc), 2) if not np.isnan(prev_stc) else 50.0,
            'status': status,
            'price': float(current_price),
            'buy_signal': bool(buy_signal),
            'sell_signal': bool(sell_signal),
            'timeframe': self.timeframe,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        
        return result
    
    def scan_watchlist(self, symbols: List[str]) -> List[Dict]:
        """
        Scan list of symbols for CTR signals
        
        Returns list of results with signals
        """
        results = []
        signals = []
        
        for symbol in symbols:
            result = self.scan_symbol(symbol)
            if result:
                results.append(result)
                
                if result['buy_signal'] or result['sell_signal']:
                    signals.append(result)
        
        return results, signals
    
    # ========================================
    # SIGNAL FORMATTING
    # ========================================
    
    def format_telegram_signal(self, signal: Dict) -> str:
        """Format signal for Telegram notification"""
        symbol = signal['symbol']
        price = signal['price']
        stc = signal['stc']
        timeframe = signal['timeframe']
        
        if signal['buy_signal']:
            emoji = "🟢"
            direction = "ПОКУПКА"
            level = f"STC перетнув {self.lower} знизу"
        else:
            emoji = "🔴"
            direction = "ПРОДАЖ"
            level = f"STC перетнув {self.upper} зверху"
        
        # Format price with appropriate precision
        if price >= 1000:
            price_str = f"{price:,.2f}"
        elif price >= 1:
            price_str = f"{price:.4f}"
        else:
            price_str = f"{price:.8f}"
        
        message = f"""
{emoji} <b>CTR: Сигнал {direction}</b>

<b>Монета:</b> {symbol}
<b>Ціна:</b> ${price_str}
<b>STC:</b> {stc:.2f}
<b>Таймфрейм:</b> {timeframe}

<i>{level}</i>

⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return message.strip()


# ========================================
# SINGLETON
# ========================================

_scanner = None

def get_ctr_scanner(db=None) -> CTRScanner:
    """Get CTR Scanner instance (singleton)"""
    global _scanner
    if _scanner is None:
        _scanner = CTRScanner(db)
    return _scanner
