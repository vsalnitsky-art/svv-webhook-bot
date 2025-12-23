#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         RSI SNIPER PRO v1.0                                   ║
║                                                                               ║
║  Професійна стратегія на основі RSI, MFI Cloud, Bollinger Bands              ║
║  з Market Structure та Divergence Detection                                   ║
║                                                                               ║
║  3 Типи сигналів:                                                            ║
║  • SNIPER REVERSAL - Контртренд (BB Extreme + RSI Zone)                      ║
║  • SMART DIVERGENCE - Розворот (Price/RSI Divergence)                        ║
║  • TREND FLOW - За трендом (MFI Cloud confirmation)                          ║
║                                                                               ║
║  Based on: "Professional RSI & MFI Sniper Strategy [Structure]"              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text

from models import db_manager, Base
from settings_manager import settings
from bot import bot_instance

logger = logging.getLogger("RSISniperPro")


# ============================================================================
#                              ENUMS & CONSTANTS
# ============================================================================

class SignalType(Enum):
    """Типи сигналів"""
    SNIPER = "SNIPER"           # BB Extreme + RSI Zone
    DIVERGENCE = "DIVERGENCE"   # Price/RSI Divergence
    FLOW = "FLOW"               # Trend continuation
    ROYAL = "ROYAL"             # Divergence + Sniper + Structure (Королівський)


class Direction(Enum):
    """Напрямок угоди"""
    LONG = "LONG"
    SHORT = "SHORT"


class StructureType(Enum):
    """Тип структури"""
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low


class TradeStatus(Enum):
    """Статус угоди"""
    PENDING = "Pending"         # Чекає підтвердження (для Divergence)
    OPEN = "Open"
    TP1_HIT = "TP1 Hit"
    CLOSED = "Closed"
    CANCELLED = "Cancelled"
    EXPIRED = "Expired"


# Параметри за замовчуванням (з оригіналу Pine Script)
DEFAULT_CONFIG = {
    # RSI Settings
    'rsp_rsi_length': 14,
    'rsp_oversold': 30,
    'rsp_overbought': 70,
    
    # MFI Cloud Settings
    'rsp_mfi_length': 20,
    'rsp_fast_mfi_ema': 5,
    'rsp_slow_mfi_ema': 13,
    
    # Signal Configuration
    'rsp_require_volume': True,  # ✅ Увімкнено за замовчуванням
    'rsp_trend_confirmation': False,
    'rsp_min_peak_strength': 2,
    
    # Bollinger Bands
    'rsp_use_bb': True,
    'rsp_bb_length': 20,
    'rsp_bb_mult': 2.0,
    
    # Structure & Divergence
    'rsp_show_structure': True,
    'rsp_show_divergence': True,
    'rsp_pivot_left': 5,
    'rsp_pivot_right': 2,
    
    # Timeframes
    'rsp_main_tf': '15',
    'rsp_htf': '60',
    'rsp_htf_auto': True,
    
    # Volume Filter
    'rsp_min_volume_24h': 10_000_000,
    'rsp_scan_limit': 50,
    
    # Signal Types Enable/Disable
    'rsp_enable_sniper': True,
    'rsp_enable_divergence': True,
    'rsp_enable_flow': True,
    'rsp_enable_royal': True,
    
    # Risk Management
    'rsp_max_daily_trades': 10,
    'rsp_max_open_positions': 5,
    'rsp_position_size_percent': 5.0,
    'rsp_max_daily_loss': 3.0,
    'rsp_leverage': 10,
    
    # Execution
    'rsp_paper_trading': True,
    'rsp_auto_execute': True,  # ✅ Автовиконання для імітації
    'rsp_telegram_signals': False,
    'rsp_close_on_opposite': True,  # ✅ Закривати при протилежному сигналі
    
    # Auto Mode
    'rsp_auto_mode': True,
    'rsp_scan_interval': 1,  # хвилини
}

# HTF Mapping
HTF_MAPPING = {
    '1': '5', '5': '15', '15': '60',
    '30': '60', '60': '240', '240': 'D', 'D': 'W'
}

# Пояснення параметрів
PARAM_HELP = {
    'rsp_rsi_length': "Період RSI (стандарт: 14)",
    'rsp_oversold': "Рівень перепроданості RSI",
    'rsp_overbought': "Рівень перекупленості RSI",
    'rsp_mfi_length': "Період MFI",
    'rsp_fast_mfi_ema': "Швидка EMA для MFI Cloud",
    'rsp_slow_mfi_ema': "Повільна EMA для MFI Cloud",
    'rsp_require_volume': "Вимагати підтвердження об'ємом",
    'rsp_trend_confirmation': "Вимагати підтвердження тренду (EMA 20)",
    'rsp_min_peak_strength': "Мінімальна сила піку RSI (1-5)",
    'rsp_use_bb': "Використовувати Bollinger Bands для SNIPER",
    'rsp_bb_length': "Період Bollinger Bands",
    'rsp_bb_mult': "Множник стандартного відхилення BB",
    'rsp_pivot_left': "Ліві бари для визначення Pivot",
    'rsp_pivot_right': "Праві бари для визначення Pivot",
}


# ============================================================================
#                              DATABASE MODEL
# ============================================================================

class RSISniperTrade(Base):
    """Модель для збереження угод з повною аналітикою"""
    __tablename__ = 'rsi_sniper_trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    direction = Column(String(10))           # LONG / SHORT
    signal_type = Column(String(20))         # SNIPER / DIVERGENCE / FLOW / ROYAL
    status = Column(String(20), default='Open')
    
    # Prices
    signal_price = Column(Float)             # Ціна на момент сигналу
    entry_price = Column(Float)              # Ціна входу
    current_price = Column(Float)            # Поточна ціна
    highest_price = Column(Float)            # Найвища ціна за час угоди
    lowest_price = Column(Float)             # Найнижча ціна за час угоди
    sl_price = Column(Float)
    tp1_price = Column(Float)
    tp2_price = Column(Float)
    exit_price = Column(Float)
    
    # BB Levels (for SNIPER)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    
    # Indicators at entry
    rsi_value = Column(Float)
    mfi_value = Column(Float)
    mfi_cloud = Column(String(10))           # BULLISH / BEARISH
    structure = Column(String(10))           # HH / HL / LH / LL
    volume_ratio = Column(Float)             # Volume / SMA(20)
    atr_value = Column(Float)
    
    # Trade Management
    tp1_hit = Column(Boolean, default=False)
    tp1_exit_price = Column(Float)
    tp1_pnl = Column(Float)
    tp2_hit = Column(Boolean, default=False)
    moved_to_be = Column(Boolean, default=False)  # SL moved to breakeven
    
    # P&L
    pnl_percent = Column(Float)
    pnl_usdt = Column(Float)
    max_profit_percent = Column(Float)       # Максимальний прибуток (для аналізу)
    max_drawdown_percent = Column(Float)     # Максимальна просадка
    
    # Exit Info
    exit_reason = Column(String(50))         # TP1, TP2, SL, OPPOSITE_SIGNAL, MANUAL, EXPIRED
    
    # Meta
    timeframe = Column(String(10))
    paper_trade = Column(Boolean, default=True)
    leverage = Column(Integer, default=10)
    position_size = Column(Float)
    
    # Timestamps
    signal_time = Column(DateTime)
    entry_time = Column(DateTime)
    tp1_time = Column(DateTime)
    exit_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    hold_time_minutes = Column(Float)        # Час утримання позиції
    
    # Analysis tags
    notes = Column(Text)
    tags = Column(String(200))               # Для аналізу: "reversal,high_volume,divergence"


# ============================================================================
#                              INDICATOR ENGINE
# ============================================================================

class IndicatorEngine:
    """Розрахунок всіх індикаторів"""
    
    @staticmethod
    def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
        """RSI з Wilder's Smoothing (як в TradingView)"""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        # Wilder's smoothing
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                      volume: pd.Series, length: int = 20) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        raw_mf = typical_price * volume
        
        positive_mf = pd.Series(0.0, index=close.index)
        negative_mf = pd.Series(0.0, index=close.index)
        
        tp_diff = typical_price.diff()
        positive_mf = raw_mf.where(tp_diff > 0, 0.0)
        negative_mf = raw_mf.where(tp_diff < 0, 0.0)
        
        positive_sum = positive_mf.rolling(window=length).sum()
        negative_sum = negative_mf.rolling(window=length).sum()
        
        mfr = positive_sum / negative_sum.replace(0, np.inf)
        mfi = 100 - (100 / (1 + mfr))
        return mfi
    
    @staticmethod
    def calculate_ema(series: pd.Series, length: int) -> pd.Series:
        """Exponential Moving Average"""
        return series.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def calculate_sma(series: pd.Series, length: int) -> pd.Series:
        """Simple Moving Average"""
        return series.rolling(window=length).mean()
    
    @staticmethod
    def calculate_bollinger_bands(close: pd.Series, length: int = 20, 
                                   mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: returns (middle, upper, lower)"""
        middle = close.rolling(window=length).mean()
        std = close.rolling(window=length).std()
        upper = middle + (std * mult)
        lower = middle - (std * mult)
        return middle, upper, lower
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, 
                      length: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr


# ============================================================================
#                              PIVOT & STRUCTURE DETECTOR
# ============================================================================

@dataclass
class PivotPoint:
    """Точка Pivot"""
    index: int
    bar_index: int
    price: float
    rsi_value: float
    pivot_type: str  # 'high' or 'low'
    structure: Optional[StructureType] = None


class StructureDetector:
    """Виявлення Market Structure та Divergence"""
    
    def __init__(self, pivot_left: int = 5, pivot_right: int = 2):
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        
        # Зберігаємо останні pivot points
        self.last_pivot_low: Optional[PivotPoint] = None
        self.last_pivot_high: Optional[PivotPoint] = None
    
    def find_pivots(self, df: pd.DataFrame, rsi: pd.Series) -> Tuple[List[PivotPoint], List[PivotPoint]]:
        """
        Знаходить Pivot High та Pivot Low точки.
        Returns: (pivot_highs, pivot_lows)
        """
        n = len(df)
        pivot_highs = []
        pivot_lows = []
        
        for i in range(self.pivot_left, n - self.pivot_right):
            # Check Pivot Low (in RSI)
            is_pivot_low = True
            rsi_val = rsi.iloc[i]
            for j in range(1, self.pivot_left + 1):
                if rsi.iloc[i - j] <= rsi_val:
                    is_pivot_low = False
                    break
            if is_pivot_low:
                for j in range(1, self.pivot_right + 1):
                    if i + j < n and rsi.iloc[i + j] <= rsi_val:
                        is_pivot_low = False
                        break
            
            if is_pivot_low:
                pivot_lows.append(PivotPoint(
                    index=i,
                    bar_index=i,
                    price=df['low'].iloc[i],
                    rsi_value=rsi_val,
                    pivot_type='low'
                ))
            
            # Check Pivot High (in RSI)
            is_pivot_high = True
            for j in range(1, self.pivot_left + 1):
                if rsi.iloc[i - j] >= rsi_val:
                    is_pivot_high = False
                    break
            if is_pivot_high:
                for j in range(1, self.pivot_right + 1):
                    if i + j < n and rsi.iloc[i + j] >= rsi_val:
                        is_pivot_high = False
                        break
            
            if is_pivot_high:
                pivot_highs.append(PivotPoint(
                    index=i,
                    bar_index=i,
                    price=df['high'].iloc[i],
                    rsi_value=rsi_val,
                    pivot_type='high'
                ))
        
        return pivot_highs, pivot_lows
    
    def detect_structure(self, pivot_highs: List[PivotPoint], 
                         pivot_lows: List[PivotPoint]) -> Tuple[Optional[StructureType], Optional[PivotPoint]]:
        """
        Визначає поточну структуру ринку.
        Returns: (structure_type, latest_pivot)
        """
        latest_structure = None
        latest_pivot = None
        
        # Аналіз Pivot Lows (для HL/LL)
        if len(pivot_lows) >= 2:
            current = pivot_lows[-1]
            previous = pivot_lows[-2]
            
            if current.price > previous.price:
                current.structure = StructureType.HL
                latest_structure = StructureType.HL
            else:
                current.structure = StructureType.LL
                latest_structure = StructureType.LL
            
            self.last_pivot_low = current
            latest_pivot = current
        
        # Аналіз Pivot Highs (для HH/LH)
        if len(pivot_highs) >= 2:
            current = pivot_highs[-1]
            previous = pivot_highs[-2]
            
            if current.price > previous.price:
                current.structure = StructureType.HH
                latest_structure = StructureType.HH
            else:
                current.structure = StructureType.LH
                latest_structure = StructureType.LH
            
            self.last_pivot_high = current
            latest_pivot = current
        
        return latest_structure, latest_pivot
    
    def detect_divergence(self, pivot_highs: List[PivotPoint], 
                          pivot_lows: List[PivotPoint]) -> Tuple[bool, bool, Optional[str]]:
        """
        Виявляє дивергенції.
        Returns: (bullish_divergence, bearish_divergence, description)
        """
        bullish_div = False
        bearish_div = False
        description = None
        
        # Bullish Divergence: Price LL + RSI HL
        if len(pivot_lows) >= 2:
            current = pivot_lows[-1]
            previous = pivot_lows[-2]
            
            if current.price < previous.price and current.rsi_value > previous.rsi_value:
                bullish_div = True
                description = f"Bullish Div: Price LL ({previous.price:.4f} → {current.price:.4f}), RSI HL ({previous.rsi_value:.1f} → {current.rsi_value:.1f})"
        
        # Bearish Divergence: Price HH + RSI LH
        if len(pivot_highs) >= 2:
            current = pivot_highs[-1]
            previous = pivot_highs[-2]
            
            if current.price > previous.price and current.rsi_value < previous.rsi_value:
                bearish_div = True
                description = f"Bearish Div: Price HH ({previous.price:.4f} → {current.price:.4f}), RSI LH ({previous.rsi_value:.1f} → {current.rsi_value:.1f})"
        
        return bullish_div, bearish_div, description


# ============================================================================
#                              SIGNAL GENERATOR
# ============================================================================

@dataclass
class Signal:
    """Сигнал для торгівлі"""
    symbol: str
    direction: Direction
    signal_type: SignalType
    
    # Prices
    price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    
    # BB Levels
    bb_upper: float = 0
    bb_middle: float = 0
    bb_lower: float = 0
    
    # Indicators
    rsi: float = 0
    mfi: float = 0
    mfi_cloud: str = "NEUTRAL"
    structure: Optional[StructureType] = None
    
    # Flags
    is_valid: bool = True
    requires_confirmation: bool = False  # For DIVERGENCE
    is_royal: bool = False
    
    # Meta
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeframe: str = "15"
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'signal_type': self.signal_type.value,
            'price': self.price,
            'sl_price': self.sl_price,
            'tp1_price': self.tp1_price,
            'tp2_price': self.tp2_price,
            'bb_upper': self.bb_upper,
            'bb_middle': self.bb_middle,
            'bb_lower': self.bb_lower,
            'rsi': self.rsi,
            'mfi': self.mfi,
            'mfi_cloud': self.mfi_cloud,
            'structure': self.structure.value if self.structure else None,
            'is_valid': self.is_valid,
            'requires_confirmation': self.requires_confirmation,
            'is_royal': self.is_royal,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'notes': self.notes,
        }


class SignalGenerator:
    """Генератор сигналів"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicator_engine = IndicatorEngine()
        self.structure_detector = StructureDetector(
            pivot_left=config.get('rsp_pivot_left', 5),
            pivot_right=config.get('rsp_pivot_right', 2)
        )
    
    def update_config(self, config: Dict):
        self.config = config
        self.structure_detector = StructureDetector(
            pivot_left=config.get('rsp_pivot_left', 5),
            pivot_right=config.get('rsp_pivot_right', 2)
        )
    
    def analyze(self, df: pd.DataFrame, symbol: str) -> List[Signal]:
        """
        Аналізує символ та генерує сигнали.
        Returns: List of signals
        """
        signals = []
        
        if df is None or len(df) < 50:
            return signals
        
        # Extract data
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        current_price = close.iloc[-1]
        
        # Calculate indicators
        rsi = self.indicator_engine.calculate_rsi(close, self.config['rsp_rsi_length'])
        mfi = self.indicator_engine.calculate_mfi(high, low, close, volume, self.config['rsp_mfi_length'])
        
        fast_mfi = self.indicator_engine.calculate_ema(mfi, self.config['rsp_fast_mfi_ema'])
        slow_mfi = self.indicator_engine.calculate_ema(mfi, self.config['rsp_slow_mfi_ema'])
        
        bb_middle, bb_upper, bb_lower = self.indicator_engine.calculate_bollinger_bands(
            close, self.config['rsp_bb_length'], self.config['rsp_bb_mult']
        )
        
        atr = self.indicator_engine.calculate_atr(high, low, close, 14)
        ema20 = self.indicator_engine.calculate_ema(close, 20)
        
        # Current values
        current_rsi = rsi.iloc[-1]
        current_mfi = mfi.iloc[-1]
        current_fast_mfi = fast_mfi.iloc[-1]
        current_slow_mfi = slow_mfi.iloc[-1]
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_middle = bb_middle.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        current_atr = atr.iloc[-1]
        current_ema20 = ema20.iloc[-1]
        
        # MFI Cloud
        mfi_cloud = "BULLISH" if current_fast_mfi > current_slow_mfi else "BEARISH"
        
        # Volume check
        volume_sma = volume.rolling(20).mean().iloc[-1]
        volume_ok = not self.config['rsp_require_volume'] or volume.iloc[-1] > volume_sma
        
        # Trend check
        trend_ok_buy = not self.config['rsp_trend_confirmation'] or current_price > current_ema20
        trend_ok_sell = not self.config['rsp_trend_confirmation'] or current_price < current_ema20
        
        # BB Extremes
        is_extreme_low = current_price <= current_bb_lower
        is_extreme_high = current_price >= current_bb_upper
        
        # RSI Zones
        oversold = self.config['rsp_oversold']
        overbought = self.config['rsp_overbought']
        
        # Peak detection
        min_strength = self.config['rsp_min_peak_strength']
        is_peak = self._is_falling(rsi, min_strength) and current_rsi >= overbought
        is_dip = self._is_rising(rsi, min_strength) and current_rsi <= oversold
        
        # RSI changes
        rsi_rising = rsi.iloc[-1] > rsi.iloc[-2] and rsi.iloc[-2] > rsi.iloc[-3]
        rsi_falling = rsi.iloc[-1] < rsi.iloc[-2] and rsi.iloc[-2] < rsi.iloc[-3]
        
        # Crossovers
        cross_over_sold = rsi.iloc[-2] < oversold and rsi.iloc[-1] >= oversold
        cross_under_bought = rsi.iloc[-2] > overbought and rsi.iloc[-1] <= overbought
        
        # Structure & Divergence
        pivot_highs, pivot_lows = self.structure_detector.find_pivots(df, rsi)
        structure, latest_pivot = self.structure_detector.detect_structure(pivot_highs, pivot_lows)
        bullish_div, bearish_div, div_description = self.structure_detector.detect_divergence(pivot_highs, pivot_lows)
        
        # =====================================================================
        # SIGNAL GENERATION
        # =====================================================================
        
        # Base signals
        base_buy = (is_dip and volume_ok and trend_ok_buy and rsi_rising) or \
                   (not self.config['rsp_require_volume'] and not self.config['rsp_trend_confirmation'] and cross_over_sold)
        
        base_sell = (is_peak and volume_ok and trend_ok_sell and rsi_falling) or \
                    (not self.config['rsp_require_volume'] and not self.config['rsp_trend_confirmation'] and cross_under_bought)
        
        # =====================================================================
        # 1. SNIPER SIGNALS (Контртренд)
        # =====================================================================
        if self.config.get('rsp_enable_sniper', True) and self.config.get('rsp_use_bb', True):
            
            # SNIPER LONG: Base Buy + BB Extreme Low
            if base_buy and is_extreme_low:
                # SL за хвіст свічки або BB lower
                sl_price = min(low.iloc[-1], current_bb_lower) - (current_atr * 0.2)
                # TP1: Middle BB (закриваємо 50%, потім SL в BE)
                tp1_price = current_bb_middle
                # TP2: Upper BB
                tp2_price = current_bb_upper
                
                # Check for ROYAL setup
                is_royal = bullish_div and (structure == StructureType.LL)
                
                signal = Signal(
                    symbol=symbol,
                    direction=Direction.LONG,
                    signal_type=SignalType.ROYAL if is_royal else SignalType.SNIPER,
                    price=current_price,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    bb_upper=current_bb_upper,
                    bb_middle=current_bb_middle,
                    bb_lower=current_bb_lower,
                    rsi=current_rsi,
                    mfi=current_mfi,
                    mfi_cloud=mfi_cloud,
                    structure=structure,
                    is_royal=is_royal,
                    timeframe=self.config.get('rsp_main_tf', '15'),
                    notes=f"SNIPER LONG: RSI={current_rsi:.1f}, BB Low Touch" + (f" + {div_description}" if is_royal else "")
                )
                signals.append(signal)
            
            # SNIPER SHORT: Base Sell + BB Extreme High
            if base_sell and is_extreme_high:
                sl_price = max(high.iloc[-1], current_bb_upper) + (current_atr * 0.2)
                tp1_price = current_bb_middle
                tp2_price = current_bb_lower
                
                is_royal = bearish_div and (structure == StructureType.HH)
                
                signal = Signal(
                    symbol=symbol,
                    direction=Direction.SHORT,
                    signal_type=SignalType.ROYAL if is_royal else SignalType.SNIPER,
                    price=current_price,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    bb_upper=current_bb_upper,
                    bb_middle=current_bb_middle,
                    bb_lower=current_bb_lower,
                    rsi=current_rsi,
                    mfi=current_mfi,
                    mfi_cloud=mfi_cloud,
                    structure=structure,
                    is_royal=is_royal,
                    timeframe=self.config.get('rsp_main_tf', '15'),
                    notes=f"SNIPER SHORT: RSI={current_rsi:.1f}, BB High Touch" + (f" + {div_description}" if is_royal else "")
                )
                signals.append(signal)
        
        # =====================================================================
        # 2. DIVERGENCE SIGNALS (Розворот - потребує підтвердження)
        # =====================================================================
        if self.config.get('rsp_enable_divergence', True):
            
            # Bullish Divergence
            if bullish_div and not is_extreme_low:  # Не дублювати SNIPER
                # SL за найближчий Swing Low
                swing_low = pivot_lows[-1].price if pivot_lows else low.min()
                sl_price = swing_low - (current_atr * 0.3)
                
                # TP: тримаємо до зміни MFI Cloud
                tp1_price = current_price * 1.015  # +1.5%
                tp2_price = current_price * 1.03   # +3%
                
                signal = Signal(
                    symbol=symbol,
                    direction=Direction.LONG,
                    signal_type=SignalType.DIVERGENCE,
                    price=current_price,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    rsi=current_rsi,
                    mfi=current_mfi,
                    mfi_cloud=mfi_cloud,
                    structure=structure,
                    requires_confirmation=True,  # Чекаємо підтвердження!
                    timeframe=self.config.get('rsp_main_tf', '15'),
                    notes=div_description or "Bullish Divergence detected"
                )
                signals.append(signal)
            
            # Bearish Divergence
            if bearish_div and not is_extreme_high:
                swing_high = pivot_highs[-1].price if pivot_highs else high.max()
                sl_price = swing_high + (current_atr * 0.3)
                
                tp1_price = current_price * 0.985  # -1.5%
                tp2_price = current_price * 0.97   # -3%
                
                signal = Signal(
                    symbol=symbol,
                    direction=Direction.SHORT,
                    signal_type=SignalType.DIVERGENCE,
                    price=current_price,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    rsi=current_rsi,
                    mfi=current_mfi,
                    mfi_cloud=mfi_cloud,
                    structure=structure,
                    requires_confirmation=True,
                    timeframe=self.config.get('rsp_main_tf', '15'),
                    notes=div_description or "Bearish Divergence detected"
                )
                signals.append(signal)
        
        # =====================================================================
        # 3. FLOW SIGNALS (За трендом)
        # =====================================================================
        if self.config.get('rsp_enable_flow', True):
            
            # FLOW LONG: Base Buy + MFI Cloud BULLISH + структура HL
            if base_buy and mfi_cloud == "BULLISH" and not is_extreme_low:
                # Ідеально якщо структура HL
                if structure != StructureType.HL:
                    pass  # Можна пропустити або дати менший пріоритет
                
                sl_price = current_price - (current_atr * 1.5)  # За MFI Cloud
                tp1_price = current_price * (1 + (overbought - current_rsi) / 1000)  # До RSI 70
                tp2_price = current_bb_upper
                
                signal = Signal(
                    symbol=symbol,
                    direction=Direction.LONG,
                    signal_type=SignalType.FLOW,
                    price=current_price,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    bb_middle=current_bb_middle,
                    rsi=current_rsi,
                    mfi=current_mfi,
                    mfi_cloud=mfi_cloud,
                    structure=structure,
                    timeframe=self.config.get('rsp_main_tf', '15'),
                    notes=f"FLOW LONG: MFI Cloud Bullish, RSI={current_rsi:.1f}"
                )
                signals.append(signal)
            
            # FLOW SHORT: Base Sell + MFI Cloud BEARISH + структура LH
            if base_sell and mfi_cloud == "BEARISH" and not is_extreme_high:
                sl_price = current_price + (current_atr * 1.5)
                tp1_price = current_price * (1 - (current_rsi - oversold) / 1000)
                tp2_price = current_bb_lower
                
                signal = Signal(
                    symbol=symbol,
                    direction=Direction.SHORT,
                    signal_type=SignalType.FLOW,
                    price=current_price,
                    sl_price=sl_price,
                    tp1_price=tp1_price,
                    tp2_price=tp2_price,
                    bb_middle=current_bb_middle,
                    rsi=current_rsi,
                    mfi=current_mfi,
                    mfi_cloud=mfi_cloud,
                    structure=structure,
                    timeframe=self.config.get('rsp_main_tf', '15'),
                    notes=f"FLOW SHORT: MFI Cloud Bearish, RSI={current_rsi:.1f}"
                )
                signals.append(signal)
        
        return signals
    
    def _is_falling(self, series: pd.Series, length: int) -> bool:
        """Перевіряє чи серія падає протягом length барів"""
        if len(series) < length + 1:
            return False
        for i in range(1, length + 1):
            if series.iloc[-i] >= series.iloc[-i-1]:
                return False
        return True
    
    def _is_rising(self, series: pd.Series, length: int) -> bool:
        """Перевіряє чи серія зростає протягом length барів"""
        if len(series) < length + 1:
            return False
        for i in range(1, length + 1):
            if series.iloc[-i] <= series.iloc[-i-1]:
                return False
        return True


# ============================================================================
#                              TRADE EXECUTOR
# ============================================================================

class TradeExecutor:
    """Виконання угод на біржі"""
    
    def __init__(self):
        pass
    
    def execute_trade(self, signal: Signal, config: Dict) -> Optional[Dict]:
        """Виконує угоду на основі сигналу"""
        try:
            if config.get('rsp_paper_trading', True):
                return self._paper_trade(signal, config)
            else:
                return self._real_trade(signal, config)
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None
    
    def _paper_trade(self, signal: Signal, config: Dict) -> Dict:
        """Paper trading - симуляція"""
        return {
            'success': True,
            'paper': True,
            'symbol': signal.symbol,
            'direction': signal.direction.value,
            'entry_price': signal.price,
            'size': config.get('rsp_position_size_percent', 5.0),
            'leverage': config.get('rsp_leverage', 10),
        }
    
    def _real_trade(self, signal: Signal, config: Dict) -> Dict:
        """Real trading через Bybit API"""
        try:
            symbol = signal.symbol
            side = "Buy" if signal.direction == Direction.LONG else "Sell"
            
            # Get balance
            balance = bot_instance.session.get_wallet_balance(
                accountType="UNIFIED", coin="USDT"
            )
            usdt_balance = float(balance['result']['list'][0]['totalEquity'])
            
            # Calculate position size
            size_percent = config.get('rsp_position_size_percent', 5.0)
            leverage = config.get('rsp_leverage', 10)
            position_value = usdt_balance * (size_percent / 100) * leverage
            
            # Get symbol info
            ticker = bot_instance.session.get_tickers(category="linear", symbol=symbol)
            last_price = float(ticker['result']['list'][0]['lastPrice'])
            
            # Calculate quantity
            qty = position_value / last_price
            
            # Get precision
            info = bot_instance.session.get_instruments_info(category="linear", symbol=symbol)
            qty_step = float(info['result']['list'][0]['lotSizeFilter']['qtyStep'])
            qty = round(qty / qty_step) * qty_step
            
            # Set leverage
            bot_instance.session.set_leverage(
                category="linear", symbol=symbol,
                buyLeverage=str(leverage), sellLeverage=str(leverage)
            )
            
            # Place order
            result = bot_instance.session.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
            )
            
            if result['retCode'] == 0:
                order_id = result['result']['orderId']
                
                # Set TP/SL
                self._set_tp_sl(symbol, signal, qty, side, config)
                
                return {
                    'success': True,
                    'paper': False,
                    'order_id': order_id,
                    'symbol': symbol,
                    'direction': signal.direction.value,
                    'entry_price': last_price,
                    'qty': qty,
                    'leverage': leverage,
                }
            else:
                logger.error(f"Order failed: {result}")
                return {'success': False, 'error': result.get('retMsg', 'Unknown')}
                
        except Exception as e:
            logger.error(f"Real trade error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _set_tp_sl(self, symbol: str, signal: Signal, qty: float, side: str, config: Dict):
        """Встановлює TP та SL"""
        try:
            # TP1 (50% позиції)
            tp_side = "Sell" if side == "Buy" else "Buy"
            tp1_qty = qty * 0.5
            
            bot_instance.session.place_order(
                category="linear",
                symbol=symbol,
                side=tp_side,
                orderType="Limit",
                qty=str(tp1_qty),
                price=str(signal.tp1_price),
                reduceOnly=True,
            )
            
            # SL (повна позиція)
            bot_instance.session.set_trading_stop(
                category="linear",
                symbol=symbol,
                stopLoss=str(signal.sl_price),
                slTriggerBy="LastPrice",
            )
            
        except Exception as e:
            logger.error(f"TP/SL setup error: {e}")


# ============================================================================
#                              POSITION MONITOR
# ============================================================================

class PositionMonitor:
    """Моніторинг відкритих позицій з повним відстеженням через біржу"""
    
    def __init__(self, scalper):
        self.scalper = scalper
        self.running = False
        self._thread = None
        self._check_interval = 10  # секунд
    
    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("📊 Position monitor started")
    
    def stop(self):
        self.running = False
    
    def _monitor_loop(self):
        while self.running:
            try:
                self._check_positions()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(self._check_interval)
    
    def _check_positions(self):
        """Перевіряє та оновлює всі відкриті позиції"""
        session = db_manager.get_session()
        try:
            open_trades = session.query(RSISniperTrade).filter(
                RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
            ).all()
            
            for trade in open_trades:
                try:
                    self._update_trade(trade, session)
                except Exception as e:
                    logger.debug(f"Check {trade.symbol}: {e}")
            
            session.commit()
                    
        except Exception as e:
            logger.error(f"Position check error: {e}")
        finally:
            session.close()
    
    def _update_trade(self, trade: RSISniperTrade, session):
        """Оновлює одну угоду"""
        # Отримуємо поточну ціну з біржі
        try:
            ticker = bot_instance.session.get_tickers(
                category="linear", symbol=trade.symbol
            )
            current_price = float(ticker['result']['list'][0]['lastPrice'])
        except:
            return
        
        trade.current_price = current_price
        
        # Оновлюємо найвищу/найнижчу ціну
        if trade.highest_price is None or current_price > trade.highest_price:
            trade.highest_price = current_price
        if trade.lowest_price is None or current_price < trade.lowest_price:
            trade.lowest_price = current_price
        
        # Розраховуємо поточний P&L
        if trade.direction == 'LONG':
            current_pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100 * trade.leverage
            max_profit = ((trade.highest_price - trade.entry_price) / trade.entry_price) * 100 * trade.leverage
            max_dd = ((trade.lowest_price - trade.entry_price) / trade.entry_price) * 100 * trade.leverage
        else:
            current_pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100 * trade.leverage
            max_profit = ((trade.entry_price - trade.lowest_price) / trade.entry_price) * 100 * trade.leverage
            max_dd = ((trade.entry_price - trade.highest_price) / trade.entry_price) * 100 * trade.leverage
        
        trade.max_profit_percent = max_profit
        trade.max_drawdown_percent = min(max_dd, 0)  # Drawdown завжди негативний
        
        # Перевіряємо TP1
        if not trade.tp1_hit and trade.tp1_price:
            tp1_hit = (trade.direction == 'LONG' and current_price >= trade.tp1_price) or \
                      (trade.direction == 'SHORT' and current_price <= trade.tp1_price)
            
            if tp1_hit:
                trade.tp1_hit = True
                trade.tp1_exit_price = current_price
                trade.tp1_time = datetime.utcnow()
                trade.status = 'TP1 Hit'
                
                # Розраховуємо P&L для TP1 (50% позиції)
                if trade.direction == 'LONG':
                    trade.tp1_pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100 * trade.leverage * 0.5
                else:
                    trade.tp1_pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100 * trade.leverage * 0.5
                
                # Переносимо SL в беззбиток
                trade.sl_price = trade.entry_price
                trade.moved_to_be = True
                logger.info(f"✅ {trade.symbol} TP1 hit! P&L: {trade.tp1_pnl:.2f}%, SL moved to BE")
        
        # Перевіряємо TP2
        if trade.tp1_hit and trade.tp2_price:
            tp2_hit = (trade.direction == 'LONG' and current_price >= trade.tp2_price) or \
                      (trade.direction == 'SHORT' and current_price <= trade.tp2_price)
            
            if tp2_hit:
                trade.tp2_hit = True
                self._close_trade(trade, current_price, 'TP2')
                return
        
        # Перевіряємо SL
        if trade.sl_price:
            sl_hit = (trade.direction == 'LONG' and current_price <= trade.sl_price) or \
                     (trade.direction == 'SHORT' and current_price >= trade.sl_price)
            
            if sl_hit:
                reason = 'SL_BE' if trade.moved_to_be else 'SL'
                self._close_trade(trade, current_price, reason)
                return
    
    def _close_trade(self, trade: RSISniperTrade, exit_price: float, reason: str):
        """Закриває угоду з повним розрахунком P&L"""
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.status = 'Closed'
        trade.exit_reason = reason
        
        # Час утримання позиції
        if trade.entry_time:
            delta = trade.exit_time - trade.entry_time
            trade.hold_time_minutes = delta.total_seconds() / 60
        
        # Розрахунок фінального P&L
        if trade.direction == 'LONG':
            base_pnl = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            base_pnl = ((trade.entry_price - exit_price) / trade.entry_price) * 100
        
        # Якщо TP1 був досягнутий, враховуємо часткове закриття
        if trade.tp1_hit and trade.tp1_pnl:
            # 50% закрито по TP1 + 50% по exit_price
            remaining_pnl = base_pnl * trade.leverage * 0.5
            trade.pnl_percent = trade.tp1_pnl + remaining_pnl
        else:
            trade.pnl_percent = base_pnl * trade.leverage
        
        # Генеруємо теги для аналізу
        tags = [trade.signal_type]
        if trade.tp1_hit:
            tags.append('tp1_hit')
        if trade.tp2_hit:
            tags.append('tp2_hit')
        if trade.pnl_percent > 0:
            tags.append('win')
        else:
            tags.append('loss')
        if trade.hold_time_minutes and trade.hold_time_minutes < 30:
            tags.append('quick_trade')
        trade.tags = ','.join(tags)
        
        logger.info(f"📊 {trade.symbol} CLOSED by {reason}: P&L {trade.pnl_percent:.2f}%, "
                   f"Hold: {trade.hold_time_minutes:.1f}m, Max Profit: {trade.max_profit_percent:.2f}%")


# ============================================================================
#                              RSI SNIPER PRO MAIN CLASS
# ============================================================================

class RSISniperPro:
    """
    🎯 RSI Sniper PRO - Головний клас
    
    Особливості:
    - Одна позиція на символ (без спаму)
    - Закриття протилежної позиції при новому сигналі
    - Автоматичне відстеження через біржу
    - Повна аналітика для оптимізації
    """
    
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status = "Idle"
        self.last_scan_time = None
        self.scan_results: List[Signal] = []
        self.auto_running = False
        self.today_trades = 0
        
        # Components
        self.signal_generator = SignalGenerator(self._load_config())
        self.executor = TradeExecutor()
        self.position_monitor = PositionMonitor(self)
        
        # Threading
        self._stop_scan = threading.Event()
        self._auto_thread = None
        
        # Cache для відкритих позицій
        self._open_positions: Dict[str, RSISniperTrade] = {}
        
        # Init
        self._ensure_table()
        self._load_open_positions()
        self.position_monitor.start()
        
        # Auto start
        config = self._load_config()
        if config.get('rsp_auto_mode', True):
            interval = config.get('rsp_scan_interval', 1)
            self.start_auto_mode(interval)
            logger.info(f"🎯 RSI Sniper PRO auto mode enabled (interval: {interval} min)")
        else:
            logger.info("🎯 RSI Sniper PRO initialized")
    
    def _ensure_table(self):
        """Створює таблицю"""
        try:
            RSISniperTrade.__table__.create(db_manager.engine, checkfirst=True)
            
            # Migrate existing table
            from sqlalchemy import text
            new_columns = [
                ("highest_price", "FLOAT"),
                ("lowest_price", "FLOAT"),
                ("volume_ratio", "FLOAT"),
                ("atr_value", "FLOAT"),
                ("tp2_hit", "BOOLEAN DEFAULT FALSE"),
                ("max_profit_percent", "FLOAT"),
                ("max_drawdown_percent", "FLOAT"),
                ("exit_reason", "VARCHAR(50)"),
                ("tp1_time", "DATETIME"),
                ("hold_time_minutes", "FLOAT"),
                ("tags", "VARCHAR(200)"),
            ]
            
            with db_manager.engine.connect() as conn:
                for col_name, col_type in new_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE rsi_sniper_trades ADD COLUMN {col_name} {col_type}"))
                        conn.commit()
                    except:
                        pass
        except Exception as e:
            logger.debug(f"Table: {e}")
    
    def _load_open_positions(self):
        """Завантажує відкриті позиції в кеш"""
        session = db_manager.get_session()
        try:
            open_trades = session.query(RSISniperTrade).filter(
                RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
            ).all()
            
            for trade in open_trades:
                self._open_positions[trade.symbol] = trade
            
            logger.info(f"📋 Loaded {len(self._open_positions)} open positions")
        except Exception as e:
            logger.error(f"Load positions error: {e}")
        finally:
            session.close()
    
    def _load_config(self) -> Dict:
        """Завантажує конфігурацію"""
        
        def to_bool(val, default=True):
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 'on')
            return bool(val) if val is not None else default
        
        def to_int(val, default=0):
            try:
                return int(float(val)) if val is not None else default
            except:
                return default
        
        def to_float(val, default=0.0):
            try:
                return float(val) if val is not None else default
            except:
                return default
        
        config = {}
        for key, default in DEFAULT_CONFIG.items():
            raw = settings.get(key)
            if isinstance(default, bool):
                config[key] = to_bool(raw, default)
            elif isinstance(default, int):
                config[key] = to_int(raw, default)
            elif isinstance(default, float):
                config[key] = to_float(raw, default)
            else:
                config[key] = raw if raw is not None else default
        
        return config
    
    def get_config(self) -> Dict:
        return self._load_config()
    
    def save_config(self, data: Dict):
        """Зберігає конфігурацію"""
        settings.save_settings(data)
        self.signal_generator.update_config(self._load_config())
        logger.info("✅ Config saved")
    
    def get_status(self) -> Dict:
        """Повертає статус"""
        config = self._load_config()
        return {
            'is_scanning': self.is_scanning,
            'progress': self.progress,
            'status': self.status,
            'last_scan_time': self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else None,
            'results_count': len(self.scan_results),
            'auto_running': self.auto_running,
            'scan_interval': config.get('rsp_scan_interval', 1),
            'today_trades': self.today_trades,
            'paper_mode': config.get('rsp_paper_trading', True),
            'open_positions': len(self._open_positions),
        }
    
    def has_open_position(self, symbol: str) -> bool:
        """Перевіряє чи є відкрита позиція по символу"""
        return symbol in self._open_positions
    
    def get_open_position(self, symbol: str) -> Optional[RSISniperTrade]:
        """Повертає відкриту позицію по символу"""
        return self._open_positions.get(symbol)
    
    def close_position_by_opposite(self, symbol: str, new_direction: str, current_price: float) -> bool:
        """Закриває позицію через протилежний сигнал"""
        if symbol not in self._open_positions:
            return False
        
        session = db_manager.get_session()
        try:
            trade = session.query(RSISniperTrade).filter(
                RSISniperTrade.symbol == symbol,
                RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
            ).first()
            
            if trade and trade.direction != new_direction:
                # Закриваємо протилежну позицію
                trade.exit_price = current_price
                trade.exit_time = datetime.utcnow()
                trade.status = 'Closed'
                trade.exit_reason = 'OPPOSITE_SIGNAL'
                
                if trade.entry_time:
                    delta = trade.exit_time - trade.entry_time
                    trade.hold_time_minutes = delta.total_seconds() / 60
                
                if trade.direction == 'LONG':
                    base_pnl = ((current_price - trade.entry_price) / trade.entry_price) * 100
                else:
                    base_pnl = ((trade.entry_price - current_price) / trade.entry_price) * 100
                
                if trade.tp1_hit and trade.tp1_pnl:
                    trade.pnl_percent = trade.tp1_pnl + (base_pnl * trade.leverage * 0.5)
                else:
                    trade.pnl_percent = base_pnl * trade.leverage
                
                # Теги
                tags = [trade.signal_type, 'opposite_close']
                if trade.pnl_percent > 0:
                    tags.append('win')
                else:
                    tags.append('loss')
                trade.tags = ','.join(tags)
                
                session.commit()
                
                # Видаляємо з кешу
                self._open_positions.pop(symbol, None)
                
                logger.info(f"🔄 {symbol} CLOSED by opposite signal: P&L {trade.pnl_percent:.2f}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Close opposite error: {e}")
            session.rollback()
            return False
        finally:
            session.close()
    
    def start_scan(self) -> Dict:
        """Запускає сканування"""
        if self.is_scanning:
            return {'status': 'error', 'error': 'Already scanning'}
        
        self._stop_scan.clear()
        thread = threading.Thread(target=self._scan_thread, daemon=True)
        thread.start()
        
        return {'status': 'started'}
    
    def stop_scan(self):
        """Зупиняє сканування"""
        self._stop_scan.set()
        self.is_scanning = False
        return {'status': 'stopped'}
    
    def _scan_thread(self):
        """Потік сканування"""
        self.is_scanning = True
        self.progress = 0
        self.status = "Starting..."
        self.scan_results = []
        
        config = self._load_config()
        self.signal_generator.update_config(config)
        
        try:
            # Get symbols
            self.status = "Fetching markets..."
            self.progress = 5
            tickers = bot_instance.get_all_tickers()
            
            min_volume = config.get('rsp_min_volume_24h', 10_000_000)
            scan_limit = config.get('rsp_scan_limit', 50)
            
            targets = [
                t for t in tickers
                if t['symbol'].endswith('USDT')
                and float(t.get('turnover24h', 0)) > min_volume
            ]
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            targets = targets[:scan_limit]
            
            total = len(targets)
            self.status = f"Scanning {total} coins..."
            
            found_signals = []
            tf = config.get('rsp_main_tf', '15')
            
            for i, t in enumerate(targets):
                if self._stop_scan.is_set():
                    break
                
                symbol = t['symbol']
                self.status = f"Analyzing {symbol}... ({i+1}/{total})"
                self.progress = 5 + int((i / total) * 90)
                
                try:
                    df = self._fetch_klines(symbol, tf, 200)
                    if df is None or len(df) < 50:
                        continue
                    
                    signals = self.signal_generator.analyze(df, symbol)
                    
                    for signal in signals:
                        if signal.is_valid:
                            # Фільтруємо сигнали по увімкненим типам
                            if signal.signal_type == SignalType.SNIPER and not config.get('rsp_enable_sniper', True):
                                continue
                            if signal.signal_type == SignalType.DIVERGENCE and not config.get('rsp_enable_divergence', True):
                                continue
                            if signal.signal_type == SignalType.FLOW and not config.get('rsp_enable_flow', True):
                                continue
                            if signal.signal_type == SignalType.ROYAL and not config.get('rsp_enable_royal', True):
                                continue
                            
                            found_signals.append(signal)
                            logger.info(f"✅ {signal.symbol} {signal.signal_type.value} {signal.direction.value}: "
                                       f"RSI={signal.rsi:.1f}, MFI={signal.mfi_cloud}")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Scan {symbol}: {e}")
                    continue
            
            # Sort by signal type priority: ROYAL > SNIPER > DIVERGENCE > FLOW
            priority = {SignalType.ROYAL: 0, SignalType.SNIPER: 1, SignalType.DIVERGENCE: 2, SignalType.FLOW: 3}
            found_signals.sort(key=lambda x: priority.get(x.signal_type, 99))
            
            self.scan_results = found_signals
            self.progress = 100
            self.status = f"Done! Found {len(self.scan_results)} signals"
            self.last_scan_time = datetime.now()
            
            logger.info(f"📊 Scan complete: {len(self.scan_results)} signals "
                       f"(ROYAL: {sum(1 for s in found_signals if s.signal_type == SignalType.ROYAL)}, "
                       f"SNIPER: {sum(1 for s in found_signals if s.signal_type == SignalType.SNIPER)}, "
                       f"DIV: {sum(1 for s in found_signals if s.signal_type == SignalType.DIVERGENCE)}, "
                       f"FLOW: {sum(1 for s in found_signals if s.signal_type == SignalType.FLOW)})")
            
            # Auto execute (для імітації)
            if config.get('rsp_auto_execute', True) and found_signals:
                self._auto_execute_signals(found_signals, config)
                
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.status = f"Error: {str(e)}"
        finally:
            self.is_scanning = False
    
    def _fetch_klines(self, symbol: str, tf: str, limit: int = 200) -> Optional[pd.DataFrame]:
        """Завантажує свічки"""
        try:
            klines = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=tf, limit=limit
            )
            if not klines or 'result' not in klines:
                return None
            
            data = klines['result']['list']
            if len(data) < 50:
                return None
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype({
                'open': 'float64', 'high': 'float64', 'low': 'float64',
                'close': 'float64', 'volume': 'float64'
            })
            return df.iloc[::-1].reset_index(drop=True)
        except:
            return None
    
    def _auto_execute_signals(self, signals: List[Signal], config: Dict):
        """Автоматичне виконання сигналів (імітація)"""
        max_trades = config.get('rsp_max_daily_trades', 10)
        max_positions = config.get('rsp_max_open_positions', 5)
        close_on_opposite = config.get('rsp_close_on_opposite', True)
        
        executed_count = 0
        
        for signal in signals:
            # Перевіряємо ліміти
            if self.today_trades >= max_trades:
                logger.debug(f"Daily limit reached: {self.today_trades}/{max_trades}")
                break
            
            if len(self._open_positions) >= max_positions:
                # Перевіряємо чи можна закрити протилежну
                if not (close_on_opposite and self.has_open_position(signal.symbol)):
                    logger.debug(f"Max positions reached: {len(self._open_positions)}/{max_positions}")
                    continue
            
            # Divergence потребує підтвердження - пропускаємо автовиконання
            if signal.requires_confirmation:
                continue
            
            # Перевіряємо чи є вже позиція по цьому символу
            if self.has_open_position(signal.symbol):
                existing = self.get_open_position(signal.symbol)
                
                # Якщо той же напрямок - пропускаємо (не спамимо)
                if existing and existing.direction == signal.direction.value:
                    logger.debug(f"Skip {signal.symbol}: already have {existing.direction} position")
                    continue
                
                # Якщо протилежний напрямок - закриваємо стару
                if close_on_opposite:
                    self.close_position_by_opposite(signal.symbol, signal.direction.value, signal.price)
            
            # Виконуємо сигнал
            result = self._execute_signal(signal, config)
            if result and result.get('success'):
                self.today_trades += 1
                executed_count += 1
        
        if executed_count > 0:
            logger.info(f"⚡ Auto-executed {executed_count} trades")
    
    def _execute_signal(self, signal: Signal, config: Dict) -> Optional[Dict]:
        """Виконує сигнал та зберігає в БД"""
        # Execute trade
        result = self.executor.execute_trade(signal, config)
        
        if result and result.get('success'):
            # Save to DB
            session = db_manager.get_session()
            try:
                trade = RSISniperTrade(
                    symbol=signal.symbol,
                    direction=signal.direction.value,
                    signal_type=signal.signal_type.value,
                    status='Open',
                    signal_price=signal.price,
                    entry_price=result.get('entry_price', signal.price),
                    current_price=signal.price,
                    highest_price=signal.price,
                    lowest_price=signal.price,
                    sl_price=signal.sl_price,
                    tp1_price=signal.tp1_price,
                    tp2_price=signal.tp2_price,
                    bb_upper=signal.bb_upper,
                    bb_middle=signal.bb_middle,
                    bb_lower=signal.bb_lower,
                    rsi_value=signal.rsi,
                    mfi_value=signal.mfi,
                    mfi_cloud=signal.mfi_cloud,
                    structure=signal.structure.value if signal.structure else None,
                    timeframe=signal.timeframe,
                    paper_trade=config.get('rsp_paper_trading', True),
                    leverage=config.get('rsp_leverage', 10),
                    position_size=config.get('rsp_position_size_percent', 5.0),
                    signal_time=signal.timestamp,
                    entry_time=datetime.utcnow(),
                    notes=signal.notes,
                )
                session.add(trade)
                session.commit()
                
                # Додаємо в кеш
                self._open_positions[signal.symbol] = trade
                
                logger.info(f"📈 TRADE OPENED: {signal.symbol} {signal.direction.value} ({signal.signal_type.value}) "
                           f"@ {signal.price:.4f} | SL: {signal.sl_price:.4f} | TP1: {signal.tp1_price:.4f}")
            except Exception as e:
                logger.error(f"Save trade error: {e}")
                session.rollback()
            finally:
                session.close()
        
        return result
    
    def execute_manual(self, signal_data: Dict) -> Dict:
        """Ручне виконання сигналу"""
        config = self._load_config()
        
        # Create signal from data
        signal = Signal(
            symbol=signal_data['symbol'],
            direction=Direction[signal_data['direction']],
            signal_type=SignalType[signal_data['signal_type']],
            price=signal_data['price'],
            sl_price=signal_data['sl_price'],
            tp1_price=signal_data['tp1_price'],
            tp2_price=signal_data['tp2_price'],
            rsi=signal_data.get('rsi', 0),
            mfi=signal_data.get('mfi', 0),
            mfi_cloud=signal_data.get('mfi_cloud', 'NEUTRAL'),
        )
        
        # Перевіряємо чи є вже позиція
        if self.has_open_position(signal.symbol):
            existing = self.get_open_position(signal.symbol)
            if existing and existing.direction == signal.direction.value:
                return {'success': False, 'error': f'Already have {existing.direction} position on {signal.symbol}'}
            
            # Закриваємо протилежну
            if config.get('rsp_close_on_opposite', True):
                self.close_position_by_opposite(signal.symbol, signal.direction.value, signal.price)
        
        result = self._execute_signal(signal, config)
        return result or {'success': False, 'error': 'Execution failed'}
    
    def get_trades(self, limit: int = 50, status: str = None) -> List[Dict]:
        """Отримує історію угод"""
        session = db_manager.get_session()
        try:
            query = session.query(RSISniperTrade)
            
            if status:
                query = query.filter(RSISniperTrade.status == status)
            
            trades = query.order_by(RSISniperTrade.created_at.desc()).limit(limit).all()
            
            return [{
                'id': t.id,
                'symbol': t.symbol,
                'direction': t.direction,
                'signal_type': t.signal_type,
                'status': t.status,
                'entry_price': t.entry_price,
                'current_price': t.current_price,
                'exit_price': t.exit_price,
                'sl_price': t.sl_price,
                'tp1_price': t.tp1_price,
                'tp2_price': t.tp2_price,
                'tp1_hit': t.tp1_hit,
                'tp2_hit': t.tp2_hit,
                'pnl_percent': t.pnl_percent,
                'max_profit': t.max_profit_percent,
                'max_drawdown': t.max_drawdown_percent,
                'exit_reason': t.exit_reason,
                'rsi': t.rsi_value,
                'mfi_cloud': t.mfi_cloud,
                'structure': t.structure,
                'timeframe': t.timeframe,
                'paper': t.paper_trade,
                'hold_time': t.hold_time_minutes,
                'tags': t.tags,
                'time': t.created_at.strftime('%d.%m %H:%M') if t.created_at else '',
            } for t in trades]
        except Exception as e:
            logger.error(f"Get trades error: {e}")
            return []
        finally:
            session.close()
    
    def get_analytics(self) -> Dict:
        """Отримує аналітику по угодам"""
        session = db_manager.get_session()
        try:
            from sqlalchemy import func
            
            # Загальна статистика
            total = session.query(RSISniperTrade).filter(RSISniperTrade.status == 'Closed').count()
            wins = session.query(RSISniperTrade).filter(
                RSISniperTrade.status == 'Closed',
                RSISniperTrade.pnl_percent > 0
            ).count()
            
            total_pnl = session.query(func.sum(RSISniperTrade.pnl_percent)).filter(
                RSISniperTrade.status == 'Closed'
            ).scalar() or 0
            
            # По типам сигналів
            by_type = {}
            for signal_type in ['SNIPER', 'DIVERGENCE', 'FLOW', 'ROYAL']:
                type_trades = session.query(RSISniperTrade).filter(
                    RSISniperTrade.status == 'Closed',
                    RSISniperTrade.signal_type == signal_type
                ).all()
                
                type_total = len(type_trades)
                type_wins = sum(1 for t in type_trades if t.pnl_percent and t.pnl_percent > 0)
                type_pnl = sum(t.pnl_percent or 0 for t in type_trades)
                
                by_type[signal_type] = {
                    'total': type_total,
                    'wins': type_wins,
                    'win_rate': (type_wins / type_total * 100) if type_total > 0 else 0,
                    'total_pnl': type_pnl,
                    'avg_pnl': type_pnl / type_total if type_total > 0 else 0,
                }
            
            # По причинам закриття
            by_exit = {}
            for reason in ['TP1', 'TP2', 'SL', 'SL_BE', 'OPPOSITE_SIGNAL']:
                count = session.query(RSISniperTrade).filter(
                    RSISniperTrade.status == 'Closed',
                    RSISniperTrade.exit_reason == reason
                ).count()
                by_exit[reason] = count
            
            return {
                'total_trades': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / total if total > 0 else 0,
                'by_signal_type': by_type,
                'by_exit_reason': by_exit,
                'open_positions': len(self._open_positions),
            }
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return {}
        finally:
            session.close()
    
    def start_auto_mode(self, interval: int = 1):
        """Запускає авто-режим"""
        if self.auto_running:
            return
        self.auto_running = True
        self._auto_thread = threading.Thread(target=self._auto_loop, args=(interval,), daemon=True)
        self._auto_thread.start()
        logger.info(f"⏰ Auto mode started (interval: {interval} min)")
    
    def stop_auto_mode(self):
        """Зупиняє авто-режим"""
        self.auto_running = False
    
    def _auto_loop(self, interval: int):
        """Цикл авто-режиму"""
        while self.auto_running:
            if not self.is_scanning:
                self.start_scan()
            
            # Wait for interval
            for _ in range(interval * 60):
                if not self.auto_running:
                    break
                time.sleep(1)


# ============================================================================
#                              FLASK ROUTES
# ============================================================================

rsi_sniper_pro = RSISniperPro()


def register_rsi_sniper_routes(app):
    """Реєструє Flask routes"""
    from flask import render_template, request, jsonify
    
    @app.route('/rsi_sniper')
    def rsi_sniper_page():
        config = rsi_sniper_pro.get_config()
        trades = rsi_sniper_pro.get_trades(30)
        open_positions = rsi_sniper_pro.get_trades(10, status='Open')
        analytics = rsi_sniper_pro.get_analytics()
        return render_template('rsi_sniper_pro.html',
                              config=config,
                              trades=trades,
                              open_positions=open_positions,
                              analytics=analytics,
                              help=PARAM_HELP)
    
    @app.route('/rsi_sniper/status')
    def rsi_sniper_status():
        status = rsi_sniper_pro.get_status()
        status['results'] = [s.to_dict() for s in rsi_sniper_pro.scan_results]
        return jsonify(status)
    
    @app.route('/rsi_sniper/scan', methods=['POST'])
    def rsi_sniper_scan():
        return jsonify(rsi_sniper_pro.start_scan())
    
    @app.route('/rsi_sniper/stop', methods=['POST'])
    def rsi_sniper_stop():
        return jsonify(rsi_sniper_pro.stop_scan())
    
    @app.route('/rsi_sniper/config', methods=['GET', 'POST'])
    def rsi_sniper_config():
        if request.method == 'POST':
            data = request.json or {}
            rsi_sniper_pro.save_config(data)
            return jsonify({'status': 'ok'})
        return jsonify(rsi_sniper_pro.get_config())
    
    @app.route('/rsi_sniper/auto', methods=['POST'])
    def rsi_sniper_auto():
        data = request.json or {}
        enabled = data.get('enabled', False)
        interval = int(data.get('interval', 1))
        
        if enabled:
            rsi_sniper_pro.start_auto_mode(interval)
        else:
            rsi_sniper_pro.stop_auto_mode()
        
        # Save to config
        rsi_sniper_pro.save_config({
            'rsp_auto_mode': enabled,
            'rsp_scan_interval': interval
        })
        
        return jsonify({'status': 'ok', 'auto_running': rsi_sniper_pro.auto_running})
    
    @app.route('/rsi_sniper/execute', methods=['POST'])
    def rsi_sniper_execute():
        data = request.json or {}
        result = rsi_sniper_pro.execute_manual(data)
        return jsonify(result)
    
    @app.route('/rsi_sniper/trades')
    def rsi_sniper_trades():
        limit = request.args.get('limit', 50, type=int)
        status = request.args.get('status', None)
        return jsonify(rsi_sniper_pro.get_trades(limit, status))
    
    @app.route('/rsi_sniper/analytics')
    def rsi_sniper_analytics():
        return jsonify(rsi_sniper_pro.get_analytics())
    
    @app.route('/rsi_sniper/positions')
    def rsi_sniper_positions():
        """Повертає відкриті позиції"""
        positions = rsi_sniper_pro.get_trades(50, status='Open')
        tp1_positions = rsi_sniper_pro.get_trades(50, status='TP1 Hit')
        return jsonify({
            'open': positions,
            'tp1_hit': tp1_positions,
            'total': len(positions) + len(tp1_positions)
        })
    
    logger.info("🎯 RSI Sniper PRO routes registered")
