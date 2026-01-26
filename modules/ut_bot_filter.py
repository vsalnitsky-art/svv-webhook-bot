"""
UT Bot Filter Module - ATR Trailing Stop Implementation

Точна реалізація TradingView індикатора 'UT Bot Alerts'
Використовує тільки BYBIT API

Pine Script логіка:
- ATR Trailing Stop змінюється тільки при зміні напрямку
- Heikin Ashi для згладжування (опціонально)
- Сигнали на перетині ціни та trailing stop

Автор: SVV Bot Team
Версія: 1.0 (2026-01-26)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from core.bybit_connector import get_connector


class UTSignalType(Enum):
    """Типи сигналів UT Bot"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class UTBotSignal:
    """Результат аналізу UT Bot"""
    symbol: str
    signal: UTSignalType
    direction: str                # 'LONG' or 'SHORT' or None
    price: float
    atr_trailing_stop: float
    atr_value: float
    position: int                 # 1 = long, -1 = short, 0 = neutral
    confidence: float
    heikin_ashi_used: bool
    timeframe: str
    timestamp: datetime
    
    # Додаткова інформація
    buy_condition: bool = False
    sell_condition: bool = False
    bar_color: str = "neutral"    # 'green', 'red', 'neutral'
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'signal': self.signal.value,
            'direction': self.direction,
            'price': round(self.price, 8),
            'atr_trailing_stop': round(self.atr_trailing_stop, 8),
            'atr_value': round(self.atr_value, 8),
            'position': self.position,
            'confidence': round(self.confidence, 1),
            'heikin_ashi_used': self.heikin_ashi_used,
            'timeframe': self.timeframe,
            'buy_condition': self.buy_condition,
            'sell_condition': self.sell_condition,
            'bar_color': self.bar_color,
            'timestamp': self.timestamp.isoformat()
        }


class UTBotFilter:
    """
    UT Bot Filter - ATR Trailing Stop Implementation
    
    Реалізація:
    1. ATR Trailing Stop = динамічний стоп на основі ATR
    2. Heikin Ashi = опціональне згладжування
    3. Сигнали = перетин ціни через trailing stop
    
    Використовує ТІЛЬКИ Bybit API!
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        'key_value': 1.0,            # ATR multiplier (sensitivity)
        'atr_period': 10,            # ATR period
        'use_heikin_ashi': True,     # Use Heikin Ashi candles
        'timeframe': '15m',          # Default timeframe
        'required_candles': 100,     # Minimum candles for analysis
    }
    
    # Available timeframes
    TIMEFRAMES = ['4h', '1h', '15m']
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize UT Bot Filter"""
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
        
        self.bybit = get_connector()
        
        # Cache for trailing stop values
        self._trailing_stop_cache: Dict[str, float] = {}
        self._position_cache: Dict[str, int] = {}
    
    def analyze(self, 
                symbol: str, 
                klines: List[Dict] = None,
                timeframe: str = None) -> UTBotSignal:
        """
        Analyze symbol and generate UT Bot signal
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            klines: Optional pre-fetched klines
            timeframe: Timeframe to use (default from config)
            
        Returns:
            UTBotSignal with analysis result
        """
        tf = timeframe or self.config['timeframe']
        
        # Fetch klines if not provided
        if klines is None:
            klines = self._fetch_klines(symbol, tf, self.config['required_candles'])
        
        if not klines or len(klines) < 20:
            return self._no_signal(symbol, tf, "Insufficient data")
        
        # Prepare data
        opens = np.array([float(k['open']) for k in klines])
        highs = np.array([float(k['high']) for k in klines])
        lows = np.array([float(k['low']) for k in klines])
        closes = np.array([float(k['close']) for k in klines])
        volumes = np.array([float(k.get('volume', 0)) for k in klines])
        
        # Calculate ATR
        atr = self._calculate_atr(highs, lows, closes, self.config['atr_period'])
        n_loss = self.config['key_value'] * atr
        
        # Get source (Heikin Ashi or regular close)
        if self.config['use_heikin_ashi']:
            src = self._calculate_heikin_ashi_close(opens, highs, lows, closes)
        else:
            src = closes.copy()
        
        # Calculate ATR Trailing Stop (EXACT Pine Script logic)
        x_atr_trailing_stop = self._calculate_trailing_stop(src, n_loss)
        
        # Calculate position (EXACT Pine Script logic)
        pos = self._calculate_position(src, x_atr_trailing_stop)
        
        # Calculate signals
        ema = src.copy()  # EMA(src, 1) = src itself
        above = np.zeros(len(src), dtype=bool)
        below = np.zeros(len(src), dtype=bool)
        
        for i in range(1, len(src)):
            above[i] = ema[i] > x_atr_trailing_stop[i] and ema[i-1] <= x_atr_trailing_stop[i-1]
            below[i] = x_atr_trailing_stop[i] > ema[i] and x_atr_trailing_stop[i-1] <= ema[i-1]
        
        # Buy/Sell conditions (EXACT Pine Script logic)
        buy = (src > x_atr_trailing_stop) & above
        sell = (src < x_atr_trailing_stop) & below
        
        # Bar color conditions
        bar_buy = src > x_atr_trailing_stop
        bar_sell = src < x_atr_trailing_stop
        
        # Get last values
        last_idx = len(src) - 1
        current_price = closes[last_idx]
        current_stop = x_atr_trailing_stop[last_idx]
        current_pos = pos[last_idx]
        current_atr = atr[last_idx] if last_idx < len(atr) else atr[-1]
        
        # Determine signal
        signal_type = UTSignalType.HOLD
        direction = None
        
        if buy[last_idx]:
            signal_type = UTSignalType.BUY
            direction = "LONG"
        elif sell[last_idx]:
            signal_type = UTSignalType.SELL
            direction = "SHORT"
        
        # Determine bar color
        bar_color = "neutral"
        if bar_buy[last_idx]:
            bar_color = "green"
        elif bar_sell[last_idx]:
            bar_color = "red"
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            current_price, current_stop, current_atr, 
            volumes[-5:].mean(), volumes[-20:].mean()
        )
        
        return UTBotSignal(
            symbol=symbol,
            signal=signal_type,
            direction=direction,
            price=current_price,
            atr_trailing_stop=current_stop,
            atr_value=current_atr,
            position=int(current_pos),
            confidence=confidence,
            heikin_ashi_used=self.config['use_heikin_ashi'],
            timeframe=tf,
            timestamp=datetime.now(),
            buy_condition=bool(buy[last_idx]),
            sell_condition=bool(sell[last_idx]),
            bar_color=bar_color
        )
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                       closes: np.ndarray, period: int) -> np.ndarray:
        """Calculate Average True Range"""
        tr = np.zeros(len(highs))
        
        for i in range(len(highs)):
            if i == 0:
                tr[i] = highs[i] - lows[i]
            else:
                tr[i] = max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i-1]),
                    abs(lows[i] - closes[i-1])
                )
        
        # Simple moving average of TR
        atr = np.zeros(len(tr))
        for i in range(len(tr)):
            if i < period:
                atr[i] = tr[:i+1].mean()
            else:
                atr[i] = tr[i-period+1:i+1].mean()
        
        return atr
    
    def _calculate_heikin_ashi_close(self, opens: np.ndarray, highs: np.ndarray,
                                      lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
        """
        Calculate Heikin Ashi close prices
        
        HA Close = (Open + High + Low + Close) / 4
        """
        return (opens + highs + lows + closes) / 4
    
    def _calculate_trailing_stop(self, src: np.ndarray, n_loss: np.ndarray) -> np.ndarray:
        """
        Calculate ATR Trailing Stop (EXACT Pine Script logic)
        
        Pine Script:
        xATRTrailingStop := 
            iff(src > nz(xATRTrailingStop[1], 0) and src[1] > nz(xATRTrailingStop[1], 0), 
                max(nz(xATRTrailingStop[1]), src - nLoss),
            iff(src < nz(xATRTrailingStop[1], 0) and src[1] < nz(xATRTrailingStop[1], 0), 
                min(nz(xATRTrailingStop[1]), src + nLoss), 
            iff(src > nz(xATRTrailingStop[1], 0), 
                src - nLoss, 
                src + nLoss)))
        """
        x_atr_trailing_stop = np.zeros(len(src))
        
        for i in range(len(src)):
            if i == 0:
                x_atr_trailing_stop[i] = src[i] - n_loss[i]
                continue
            
            prev_stop = x_atr_trailing_stop[i-1]
            
            # Condition 1: Price above stop and previous price above stop
            if src[i] > prev_stop and src[i-1] > prev_stop:
                x_atr_trailing_stop[i] = max(prev_stop, src[i] - n_loss[i])
            
            # Condition 2: Price below stop and previous price below stop
            elif src[i] < prev_stop and src[i-1] < prev_stop:
                x_atr_trailing_stop[i] = min(prev_stop, src[i] + n_loss[i])
            
            # Condition 3: Price crosses above stop
            elif src[i] > prev_stop:
                x_atr_trailing_stop[i] = src[i] - n_loss[i]
            
            # Condition 4: Price crosses below stop
            else:
                x_atr_trailing_stop[i] = src[i] + n_loss[i]
        
        return x_atr_trailing_stop
    
    def _calculate_position(self, src: np.ndarray, trailing_stop: np.ndarray) -> np.ndarray:
        """
        Calculate position based on price crossing trailing stop
        
        Pine Script:
        pos := iff(src[1] < nz(xATRTrailingStop[1], 0) and src > nz(xATRTrailingStop[1], 0), 1,
               iff(src[1] > nz(xATRTrailingStop[1], 0) and src < nz(xATRTrailingStop[1], 0), -1, 
               nz(pos[1], 0)))
        """
        pos = np.zeros(len(src), dtype=int)
        
        for i in range(1, len(src)):
            prev_stop = trailing_stop[i-1]
            
            # Cross above (buy signal)
            if src[i-1] < prev_stop and src[i] > prev_stop:
                pos[i] = 1
            # Cross below (sell signal)
            elif src[i-1] > prev_stop and src[i] < prev_stop:
                pos[i] = -1
            else:
                pos[i] = pos[i-1]
        
        return pos
    
    def _calculate_confidence(self, price: float, stop: float, atr: float,
                              recent_vol: float, avg_vol: float) -> float:
        """Calculate signal confidence"""
        confidence = 50.0
        
        # Distance from stop (further = stronger signal)
        if price != 0:
            distance_pct = abs(price - stop) / price * 100
            confidence += min(20, distance_pct * 10)
        
        # Volume confirmation
        if avg_vol > 0 and recent_vol > avg_vol * 1.5:
            confidence += 15
        
        # ATR relative to price (volatility)
        if price > 0:
            atr_pct = atr / price * 100
            if atr_pct > 1.5:
                confidence += 10
        
        return min(95, confidence)
    
    def _fetch_klines(self, symbol: str, timeframe: str, limit: int) -> List[Dict]:
        """Fetch klines from Bybit"""
        try:
            return self.bybit.get_klines(symbol, timeframe, limit)
        except Exception as e:
            print(f"[UT BOT] Error fetching klines for {symbol}: {e}")
            return []
    
    def _no_signal(self, symbol: str, timeframe: str, reason: str) -> UTBotSignal:
        """Return no signal result"""
        return UTBotSignal(
            symbol=symbol,
            signal=UTSignalType.NO_SIGNAL,
            direction=None,
            price=0.0,
            atr_trailing_stop=0.0,
            atr_value=0.0,
            position=0,
            confidence=0.0,
            heikin_ashi_used=self.config['use_heikin_ashi'],
            timeframe=timeframe,
            timestamp=datetime.now()
        )
    
    def check_signal_with_bias(self, symbol: str, bias: str, 
                                klines: List[Dict] = None,
                                timeframe: str = None) -> Dict:
        """
        Check UT Bot signal with Direction Engine bias alignment
        
        Args:
            symbol: Trading symbol
            bias: Direction bias ('LONG', 'SHORT', 'NEUTRAL')
            klines: Optional pre-fetched klines
            timeframe: Timeframe to use
            
        Returns:
            Dict with signal and alignment info
        """
        signal = self.analyze(symbol, klines, timeframe)
        
        result = signal.to_dict()
        result['bias'] = bias
        result['aligned'] = False
        result['action'] = 'HOLD'
        
        # Check alignment
        if bias == 'LONG' and signal.signal == UTSignalType.BUY:
            result['aligned'] = True
            result['action'] = 'ENTER_LONG'
        elif bias == 'SHORT' and signal.signal == UTSignalType.SELL:
            result['aligned'] = True
            result['action'] = 'ENTER_SHORT'
        elif signal.signal in [UTSignalType.BUY, UTSignalType.SELL] and bias == 'NEUTRAL':
            result['action'] = 'SIGNAL_NO_BIAS'
        
        return result
    
    def set_config(self, key: str, value) -> None:
        """Update config value"""
        if key in self.config:
            self.config[key] = value
    
    def get_config(self) -> Dict:
        """Get current config"""
        return self.config.copy()


# Factory
_ut_bot_instance = None

def get_ut_bot_filter(config: Dict = None) -> UTBotFilter:
    """Get singleton instance of UT Bot Filter"""
    global _ut_bot_instance
    if _ut_bot_instance is None:
        _ut_bot_instance = UTBotFilter(config)
    return _ut_bot_instance
