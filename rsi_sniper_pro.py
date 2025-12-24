#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         RSI SNIPER PRO v2.0                                   ║
║                                                                               ║
║  Професійна стратегія на основі RSI, MFI Cloud, Bollinger Bands              ║
║  з Market Structure та Divergence Detection                                   ║
║                                                                               ║
║  3 Типи сигналів:                                                            ║
║  • SNIPER REVERSAL - Контртренд (BB Extreme + RSI Zone)                      ║
║  • SMART DIVERGENCE - Розворот (Price/RSI Divergence)                        ║
║  • TREND FLOW - За трендом (MFI Cloud confirmation)                          ║
║                                                                               ║
║  Особливості:                                                                ║
║  • Одна позиція на символ (без спаму)                                        ║
║  • Закриття протилежної позиції при новому сигналі                           ║
║  • Повна аналітика для оптимізації                                           ║
║  • Автоматичне збереження налаштувань                                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Налаштування логера
logger = logging.getLogger("RSISniperPro")

# ============================================================================
#                              SAFE IMPORTS
# ============================================================================

try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    np = None
    pd = None

try:
    from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, text
    from models import db_manager, Base
    HAS_DB = True
except ImportError as e:
    logger.warning(f"Database not available: {e}")
    HAS_DB = False
    Base = object
    db_manager = None

try:
    from settings_manager import settings
    HAS_SETTINGS = True
except ImportError as e:
    logger.warning(f"Settings manager not available: {e}")
    HAS_SETTINGS = False
    settings = None

try:
    from bot import bot_instance
    HAS_BOT = True
except ImportError as e:
    logger.warning(f"Bot instance not available: {e}")
    HAS_BOT = False
    bot_instance = None


# ============================================================================
#                              ENUMS & CONSTANTS
# ============================================================================

class SignalType(Enum):
    """Типи сигналів"""
    SNIPER = "SNIPER"
    DIVERGENCE = "DIVERGENCE"
    FLOW = "FLOW"
    ROYAL = "ROYAL"


class Direction(Enum):
    """Напрямок угоди"""
    LONG = "LONG"
    SHORT = "SHORT"


class StructureType(Enum):
    """Тип структури"""
    HH = "HH"
    HL = "HL"
    LH = "LH"
    LL = "LL"


class TradeStatus(Enum):
    """Статус угоди"""
    OPEN = "Open"
    TP1_HIT = "TP1 Hit"
    CLOSED = "Closed"


# ============================================================================
#                              DEFAULT CONFIG
# ============================================================================

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
    'rsp_require_volume': False,
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
    'rsp_auto_execute': True,
    'rsp_telegram_signals': False,
    'rsp_close_on_opposite': True,
    
    # Auto Mode
    'rsp_auto_mode': True,
    'rsp_scan_interval': 1,
    
    # Trade Management - SNIPER
    'rsp_sniper_sl_atr': 0.2,
    'rsp_sniper_tp1': 'BB_Middle',
    'rsp_sniper_tp2': 'BB_Opposite',
    
    # Trade Management - FLOW
    'rsp_flow_sl_atr': 1.5,
    'rsp_flow_tp1': 1.0,
    'rsp_flow_tp2': 'BB_Opposite',
    
    # Trade Management - DIVERGENCE
    'rsp_div_sl_atr': 1.0,
    'rsp_div_tp1': 1.5,
    'rsp_div_tp2': 3.0,
    
    # Trade Management - ROYAL
    'rsp_royal_sl_atr': 0.5,
    'rsp_royal_tp1': 2.0,
    'rsp_royal_tp2': 4.0,
}


# ============================================================================
#                              DATABASE MODEL
# ============================================================================

if HAS_DB:
    class RSISniperTrade(Base):
        """Модель для збереження угод - сумісна з існуючою БД"""
        __tablename__ = 'rsi_sniper_trades'
        __table_args__ = {'extend_existing': True}
        
        id = Column(Integer, primary_key=True)
        symbol = Column(String(20), index=True)
        direction = Column(String(10))
        signal_type = Column(String(20))
        status = Column(String(20), default='Open')
        
        # Prices
        signal_price = Column(Float)
        entry_price = Column(Float)
        current_price = Column(Float)
        sl_price = Column(Float)
        tp1_price = Column(Float)
        tp2_price = Column(Float)
        exit_price = Column(Float)
        
        # BB Levels
        bb_upper = Column(Float)
        bb_middle = Column(Float)
        bb_lower = Column(Float)
        
        # Indicators
        rsi_value = Column(Float)
        mfi_value = Column(Float)
        mfi_cloud = Column(String(10))
        structure = Column(String(10))
        
        # Trade Management
        tp1_hit = Column(Boolean, default=False)
        moved_to_be = Column(Boolean, default=False)
        
        # P&L
        pnl_percent = Column(Float)
        
        # Exit Info
        exit_reason = Column(String(50))
        
        # Meta
        timeframe = Column(String(10))
        paper_trade = Column(Boolean, default=True)
        leverage = Column(Integer, default=10)
        position_size = Column(Float)
        
        # Timestamps
        signal_time = Column(DateTime)
        entry_time = Column(DateTime)
        exit_time = Column(DateTime)
        created_at = Column(DateTime, default=datetime.utcnow)
        
        # Notes
        notes = Column(Text)
else:
    RSISniperTrade = None


# ============================================================================
#                              INDICATOR ENGINE
# ============================================================================

class IndicatorEngine:
    """Розрахунок індикаторів"""
    
    @staticmethod
    def calculate_rsi(close, length: int = 14):
        """RSI з Wilder's Smoothing"""
        if pd is None:
            return None
        
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, float('inf'))
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_mfi(high, low, close, volume, length: int = 20):
        """Money Flow Index"""
        if pd is None:
            return None
        
        typical_price = (high + low + close) / 3
        raw_mf = typical_price * volume
        
        tp_diff = typical_price.diff()
        positive_mf = raw_mf.where(tp_diff > 0, 0.0)
        negative_mf = raw_mf.where(tp_diff < 0, 0.0)
        
        positive_sum = positive_mf.rolling(window=length).sum()
        negative_sum = negative_mf.rolling(window=length).sum()
        
        mfr = positive_sum / negative_sum.replace(0, float('inf'))
        mfi = 100 - (100 / (1 + mfr))
        return mfi
    
    @staticmethod
    def calculate_ema(series, length: int):
        """Exponential Moving Average"""
        if pd is None:
            return None
        return series.ewm(span=length, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(close, length: int = 20, mult: float = 2.0):
        """Bollinger Bands: returns (middle, upper, lower)"""
        if pd is None:
            return None, None, None
        
        middle = close.rolling(window=length).mean()
        std = close.rolling(window=length).std()
        upper = middle + (std * mult)
        lower = middle - (std * mult)
        return middle, upper, lower
    
    @staticmethod
    def calculate_atr(high, low, close, length: int = 14):
        """Average True Range"""
        if pd is None:
            return None
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
        return atr


# ============================================================================
#                              SIGNAL DATACLASS
# ============================================================================

@dataclass
class Signal:
    """Сигнал для торгівлі"""
    symbol: str
    direction: Direction
    signal_type: SignalType
    
    price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    
    bb_upper: float = 0
    bb_middle: float = 0
    bb_lower: float = 0
    
    rsi: float = 0
    mfi: float = 0
    mfi_cloud: str = "NEUTRAL"
    structure: Optional[StructureType] = None
    
    is_valid: bool = True
    requires_confirmation: bool = False
    is_royal: bool = False
    
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
            'rsi': round(self.rsi, 1) if self.rsi else 0,
            'mfi': round(self.mfi, 1) if self.mfi else 0,
            'mfi_cloud': self.mfi_cloud,
            'structure': self.structure.value if self.structure else None,
            'is_valid': self.is_valid,
            'requires_confirmation': self.requires_confirmation,
            'is_royal': self.is_royal,
            'timestamp': self.timestamp.isoformat(),
            'timeframe': self.timeframe,
            'notes': self.notes,
        }


# ============================================================================
#                              SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Генератор сигналів"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicator_engine = IndicatorEngine()
    
    def update_config(self, config: Dict):
        self.config = config
    
    def analyze(self, df, symbol: str) -> List[Signal]:
        """Аналізує символ та генерує сигнали"""
        signals = []
        
        if pd is None or df is None or len(df) < 50:
            return signals
        
        try:
            # Extract data
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df['volume']
            current_price = float(close.iloc[-1])
            
            # Calculate indicators
            rsi = self.indicator_engine.calculate_rsi(close, self.config.get('rsp_rsi_length', 14))
            mfi = self.indicator_engine.calculate_mfi(high, low, close, volume, self.config.get('rsp_mfi_length', 20))
            
            fast_mfi = self.indicator_engine.calculate_ema(mfi, self.config.get('rsp_fast_mfi_ema', 5))
            slow_mfi = self.indicator_engine.calculate_ema(mfi, self.config.get('rsp_slow_mfi_ema', 13))
            
            bb_middle, bb_upper, bb_lower = self.indicator_engine.calculate_bollinger_bands(
                close, self.config.get('rsp_bb_length', 20), self.config.get('rsp_bb_mult', 2.0)
            )
            
            atr = self.indicator_engine.calculate_atr(high, low, close, 14)
            ema20 = self.indicator_engine.calculate_ema(close, 20)
            
            if rsi is None or mfi is None or bb_middle is None:
                return signals
            
            # Current values
            current_rsi = float(rsi.iloc[-1])
            current_mfi = float(mfi.iloc[-1])
            current_fast_mfi = float(fast_mfi.iloc[-1])
            current_slow_mfi = float(slow_mfi.iloc[-1])
            current_bb_upper = float(bb_upper.iloc[-1])
            current_bb_middle = float(bb_middle.iloc[-1])
            current_bb_lower = float(bb_lower.iloc[-1])
            current_atr = float(atr.iloc[-1])
            current_ema20 = float(ema20.iloc[-1])
            
            # MFI Cloud
            mfi_cloud = "BULLISH" if current_fast_mfi > current_slow_mfi else "BEARISH"
            
            # Volume check
            volume_sma = float(volume.rolling(20).mean().iloc[-1])
            volume_ok = not self.config.get('rsp_require_volume', False) or float(volume.iloc[-1]) > volume_sma
            
            # Trend check
            trend_ok_buy = not self.config.get('rsp_trend_confirmation', False) or current_price > current_ema20
            trend_ok_sell = not self.config.get('rsp_trend_confirmation', False) or current_price < current_ema20
            
            # BB Extremes
            is_extreme_low = current_price <= current_bb_lower
            is_extreme_high = current_price >= current_bb_upper
            
            # RSI Zones
            oversold = self.config.get('rsp_oversold', 30)
            overbought = self.config.get('rsp_overbought', 70)
            
            # RSI changes (for FLOW signals)
            rsi_rising = rsi.iloc[-1] > rsi.iloc[-2]
            rsi_falling = rsi.iloc[-1] < rsi.iloc[-2]
            
            # Base conditions
            is_oversold = current_rsi <= oversold
            is_overbought = current_rsi >= overbought
            
            # ===============================================================
            # SNIPER SIGNALS (BB Extreme + RSI Zone)
            # Trade Book: Entry immediately, SL behind candle wick or BB edge
            # ===============================================================
            if self.config.get('rsp_enable_sniper', True) and self.config.get('rsp_use_bb', True):
                
                # SNIPER LONG: RSI oversold + BB lower touch
                if is_oversold and is_extreme_low and volume_ok:
                    sl_price = min(float(low.iloc[-1]), current_bb_lower) - (current_atr * 0.2)
                    tp1_price = current_bb_middle
                    tp2_price = current_bb_upper
                    
                    signal = Signal(
                        symbol=symbol,
                        direction=Direction.LONG,
                        signal_type=SignalType.SNIPER,
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
                        timeframe=self.config.get('rsp_main_tf', '15'),
                        notes=f"SNIPER LONG: RSI={current_rsi:.1f}, BB Low Touch"
                    )
                    signals.append(signal)
                
                # SNIPER SHORT: RSI overbought + BB upper touch
                if is_overbought and is_extreme_high and volume_ok:
                    sl_price = max(float(high.iloc[-1]), current_bb_upper) + (current_atr * 0.2)
                    tp1_price = current_bb_middle
                    tp2_price = current_bb_lower
                    
                    signal = Signal(
                        symbol=symbol,
                        direction=Direction.SHORT,
                        signal_type=SignalType.SNIPER,
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
                        timeframe=self.config.get('rsp_main_tf', '15'),
                        notes=f"SNIPER SHORT: RSI={current_rsi:.1f}, BB High Touch"
                    )
                    signals.append(signal)
            
            # ===============================================================
            # FLOW SIGNALS (Trend Continuation)
            # Trade Book: MFI Cloud color must match signal direction
            # ===============================================================
            if self.config.get('rsp_enable_flow', True):
                
                # FLOW LONG: RSI oversold + MFI Cloud BULLISH + NOT at BB extreme
                if is_oversold and mfi_cloud == "BULLISH" and not is_extreme_low and volume_ok and rsi_rising:
                    sl_price = current_price - (current_atr * 1.5)
                    tp1_price = current_price * 1.01
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
                        timeframe=self.config.get('rsp_main_tf', '15'),
                        notes=f"FLOW LONG: MFI Cloud Bullish, RSI={current_rsi:.1f}"
                    )
                    signals.append(signal)
                
                # FLOW SHORT: RSI overbought + MFI Cloud BEARISH + NOT at BB extreme
                if is_overbought and mfi_cloud == "BEARISH" and not is_extreme_high and volume_ok and rsi_falling:
                    sl_price = current_price + (current_atr * 1.5)
                    tp1_price = current_price * 0.99
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
                        timeframe=self.config.get('rsp_main_tf', '15'),
                        notes=f"FLOW SHORT: MFI Cloud Bearish, RSI={current_rsi:.1f}"
                    )
                    signals.append(signal)
            
        except Exception as e:
            logger.error(f"Signal analysis error for {symbol}: {e}")
        
        return signals


# ============================================================================
#                              RSI SNIPER PRO MAIN CLASS
# ============================================================================

class RSISniperPro:
    """RSI Sniper PRO - Головний клас"""
    
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status = "Ready"
        self.last_scan_time = None
        self.scan_results: List[Signal] = []
        self.auto_running = False
        self.today_trades = 0
        
        # Components
        self.signal_generator = SignalGenerator(self._load_config())
        
        # Threading
        self._stop_scan = threading.Event()
        self._auto_thread = None
        self._monitor_thread = None
        self._monitor_running = False
        
        # Cache
        self._open_positions: Dict[str, Any] = {}
        
        # Init
        self._ensure_table()
        self._load_open_positions()
        self._start_monitor()
        
        # Auto start if enabled
        config = self._load_config()
        if config.get('rsp_auto_mode', True):
            interval = config.get('rsp_scan_interval', 1)
            self.start_auto_mode(interval)
            logger.info(f"🎯 RSI Sniper PRO v2.0 auto mode enabled (interval: {interval} min)")
        else:
            logger.info("🎯 RSI Sniper PRO v2.0 initialized")
    
    def _ensure_table(self):
        """Створює таблицю в БД"""
        if not HAS_DB or db_manager is None:
            return
        
        try:
            RSISniperTrade.__table__.create(db_manager.engine, checkfirst=True)
            logger.info("✅ RSI Sniper trades table ready")
        except Exception as e:
            logger.debug(f"Table creation: {e}")
    
    def _load_open_positions(self):
        """Завантажує відкриті позиції в кеш"""
        if not HAS_DB or db_manager is None:
            return
        
        session = None
        try:
            session = db_manager.get_session()
            open_trades = session.query(RSISniperTrade).filter(
                RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
            ).all()
            
            for trade in open_trades:
                self._open_positions[trade.symbol] = {
                    'id': trade.id,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'signal_type': trade.signal_type
                }
            
            logger.info(f"📋 Loaded {len(self._open_positions)} open positions")
        except Exception as e:
            logger.error(f"Load positions error: {e}")
        finally:
            if session:
                session.close()
    
    def _start_monitor(self):
        """Запускає моніторинг позицій"""
        if self._monitor_running:
            return
        
        self._monitor_running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("📊 Position monitor started")
    
    def _monitor_loop(self):
        """Цикл моніторингу позицій"""
        while self._monitor_running:
            try:
                self._check_positions()
            except Exception as e:
                logger.error(f"Monitor error: {e}")
            time.sleep(30)  # Check every 30 seconds
    
    def _check_positions(self):
        """Перевіряє відкриті позиції"""
        if not HAS_DB or not HAS_BOT or db_manager is None or bot_instance is None:
            return
        
        session = None
        try:
            session = db_manager.get_session()
            open_trades = session.query(RSISniperTrade).filter(
                RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
            ).all()
            
            if not open_trades:
                return  # No open positions, skip silently
            
            logger.debug(f"Checking {len(open_trades)} open positions...")
            
            for trade in open_trades:
                try:
                    # Get current price
                    ticker = bot_instance.session.get_tickers(
                        category="linear", symbol=trade.symbol
                    )
                    if not ticker or 'result' not in ticker:
                        continue
                    
                    current_price = float(ticker['result']['list'][0]['lastPrice'])
                    trade.current_price = current_price
                    
                    # Check TP1
                    if not trade.tp1_hit and trade.tp1_price:
                        tp1_hit = (trade.direction == 'LONG' and current_price >= trade.tp1_price) or \
                                  (trade.direction == 'SHORT' and current_price <= trade.tp1_price)
                        
                        if tp1_hit:
                            trade.tp1_hit = True
                            trade.status = 'TP1 Hit'
                            trade.sl_price = trade.entry_price
                            trade.moved_to_be = True
                            logger.info(f"✅ {trade.symbol} TP1 hit! SL moved to BE")
                    
                    # Check TP2
                    if trade.tp1_hit and trade.tp2_price:
                        tp2_hit = (trade.direction == 'LONG' and current_price >= trade.tp2_price) or \
                                  (trade.direction == 'SHORT' and current_price <= trade.tp2_price)
                        
                        if tp2_hit:
                            self._close_trade(session, trade, current_price, 'TP2')
                            continue
                    
                    # Check SL
                    if trade.sl_price:
                        sl_hit = (trade.direction == 'LONG' and current_price <= trade.sl_price) or \
                                 (trade.direction == 'SHORT' and current_price >= trade.sl_price)
                        
                        if sl_hit:
                            reason = 'SL_BE' if trade.moved_to_be else 'SL'
                            self._close_trade(session, trade, current_price, reason)
                    
                except Exception as e:
                    logger.debug(f"Check {trade.symbol}: {e}")
            
            session.commit()
            
        except Exception as e:
            logger.error(f"Position check error: {e}")
        finally:
            if session:
                session.close()
    
    def _close_trade(self, session, trade, exit_price: float, reason: str):
        """Закриває угоду"""
        trade.exit_price = exit_price
        trade.exit_time = datetime.utcnow()
        trade.status = 'Closed'
        trade.exit_reason = reason
        
        # P&L
        if trade.direction == 'LONG':
            pnl = ((exit_price - trade.entry_price) / trade.entry_price) * 100 * trade.leverage
        else:
            pnl = ((trade.entry_price - exit_price) / trade.entry_price) * 100 * trade.leverage
        
        trade.pnl_percent = pnl
        
        # Remove from cache
        self._open_positions.pop(trade.symbol, None)
        
        logger.info(f"📊 {trade.symbol} CLOSED by {reason}: P&L {pnl:.2f}%")
    
    def _load_config(self) -> Dict:
        """Завантажує конфігурацію"""
        config = DEFAULT_CONFIG.copy()
        
        if not HAS_SETTINGS or settings is None:
            return config
        
        for key, default in DEFAULT_CONFIG.items():
            try:
                raw = settings.get(key)
                if raw is not None:
                    if isinstance(default, bool):
                        config[key] = str(raw).lower() in ('true', '1', 'yes', 'on')
                    elif isinstance(default, int):
                        config[key] = int(float(raw))
                    elif isinstance(default, float):
                        config[key] = float(raw)
                    else:
                        config[key] = raw
            except:
                pass
        
        return config
    
    def get_config(self) -> Dict:
        """Повертає конфігурацію"""
        return self._load_config()
    
    def save_config(self, data: Dict):
        """Зберігає конфігурацію"""
        if HAS_SETTINGS and settings is not None:
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
        """Перевіряє чи є відкрита позиція"""
        return symbol in self._open_positions
    
    def start_scan(self) -> Dict:
        """Запускає сканування"""
        if self.is_scanning:
            return {'status': 'error', 'error': 'Already scanning'}
        
        if not HAS_BOT or bot_instance is None:
            return {'status': 'error', 'error': 'Bot not available'}
        
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
            # Get tickers
            self.status = "Fetching markets..."
            self.progress = 5
            
            tickers = bot_instance.get_all_tickers()
            if not tickers:
                self.status = "No tickers found"
                self.is_scanning = False
                return
            
            min_volume = config.get('rsp_min_volume_24h', 10_000_000)
            scan_limit = config.get('rsp_scan_limit', 50)
            
            targets = [
                t for t in tickers
                if t.get('symbol', '').endswith('USDT')
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
                    df = self._fetch_klines(symbol, tf, 1000)
                    if df is None or len(df) < 50:
                        continue
                    
                    signals = self.signal_generator.analyze(df, symbol)
                    
                    for signal in signals:
                        if signal.is_valid:
                            # Filter by enabled types
                            if signal.signal_type == SignalType.SNIPER and not config.get('rsp_enable_sniper', True):
                                continue
                            if signal.signal_type == SignalType.DIVERGENCE and not config.get('rsp_enable_divergence', True):
                                continue
                            if signal.signal_type == SignalType.FLOW and not config.get('rsp_enable_flow', True):
                                continue
                            if signal.signal_type == SignalType.ROYAL and not config.get('rsp_enable_royal', True):
                                continue
                            
                            found_signals.append(signal)
                            logger.info(f"✅ {signal.symbol} {signal.signal_type.value} {signal.direction.value}")
                    
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Scan {symbol}: {e}")
            
            # Sort by priority
            priority = {SignalType.ROYAL: 0, SignalType.SNIPER: 1, SignalType.DIVERGENCE: 2, SignalType.FLOW: 3}
            found_signals.sort(key=lambda x: priority.get(x.signal_type, 99))
            
            self.scan_results = found_signals
            self.progress = 100
            self.status = f"Done! Found {len(found_signals)} signals"
            self.last_scan_time = datetime.now()
            
            logger.info(f"📊 Scan complete: {len(found_signals)} signals")
            
            # Auto execute
            if config.get('rsp_auto_execute', True) and found_signals:
                self._auto_execute_signals(found_signals, config)
                
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.status = f"Error: {str(e)}"
        finally:
            self.is_scanning = False
    
    def _fetch_klines(self, symbol: str, tf: str, limit: int = 1000):
        """Завантажує свічки"""
        if not HAS_BOT or bot_instance is None or pd is None:
            return None
        
        try:
            # Convert timeframe format (1m -> 1, 5m -> 5, etc.)
            tf_clean = tf.replace('m', '').replace('h', '').replace('d', 'D').replace('w', 'W')
            if tf_clean == '1h':
                tf_clean = '60'
            elif tf_clean == '4h':
                tf_clean = '240'
            
            klines = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=tf_clean, limit=limit
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
        except Exception as e:
            logger.debug(f"Klines fetch error for {symbol}: {e}")
            return None
    
    def _auto_execute_signals(self, signals: List[Signal], config: Dict):
        """Автоматичне виконання сигналів"""
        max_trades = config.get('rsp_max_daily_trades', 10)
        max_positions = config.get('rsp_max_open_positions', 5)
        close_on_opposite = config.get('rsp_close_on_opposite', True)
        
        executed = 0
        
        for signal in signals:
            if self.today_trades >= max_trades:
                break
            
            if len(self._open_positions) >= max_positions:
                if not (close_on_opposite and self.has_open_position(signal.symbol)):
                    continue
            
            # Skip if same direction position exists
            if self.has_open_position(signal.symbol):
                existing = self._open_positions.get(signal.symbol)
                if existing and existing.get('direction') == signal.direction.value:
                    continue
                
                # Close opposite
                if close_on_opposite:
                    self._close_by_opposite(signal.symbol, signal.direction.value, signal.price)
            
            # Execute
            result = self._execute_signal(signal, config)
            if result and result.get('success'):
                self.today_trades += 1
                executed += 1
        
        if executed > 0:
            logger.info(f"⚡ Auto-executed {executed} trades")
    
    def _close_by_opposite(self, symbol: str, new_direction: str, current_price: float):
        """Закриває позицію протилежним сигналом"""
        if not HAS_DB or db_manager is None:
            return
        
        session = None
        try:
            session = db_manager.get_session()
            trade = session.query(RSISniperTrade).filter(
                RSISniperTrade.symbol == symbol,
                RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
            ).first()
            
            if trade and trade.direction != new_direction:
                self._close_trade(session, trade, current_price, 'OPPOSITE_SIGNAL')
                session.commit()
                logger.info(f"🔄 {symbol} closed by opposite signal")
                
        except Exception as e:
            logger.error(f"Close opposite error: {e}")
        finally:
            if session:
                session.close()
    
    def _execute_signal(self, signal: Signal, config: Dict) -> Optional[Dict]:
        """Виконує сигнал"""
        if not HAS_DB or db_manager is None:
            return {'success': True, 'paper': True}
        
        session = None
        try:
            session = db_manager.get_session()
            
            trade = RSISniperTrade(
                symbol=signal.symbol,
                direction=signal.direction.value,
                signal_type=signal.signal_type.value,
                status='Open',
                signal_price=signal.price,
                entry_price=signal.price,
                current_price=signal.price,
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
            
            # Add to cache
            self._open_positions[signal.symbol] = {
                'id': trade.id,
                'direction': signal.direction.value,
                'entry_price': signal.price,
                'signal_type': signal.signal_type.value
            }
            
            logger.info(f"📈 TRADE OPENED: {signal.symbol} {signal.direction.value} @ {signal.price:.6f} | SL: {signal.sl_price:.6f} | TP1: {signal.tp1_price:.6f}")
            
            return {'success': True, 'paper': config.get('rsp_paper_trading', True)}
            
        except Exception as e:
            logger.error(f"Execute error: {e}")
            if session:
                session.rollback()
            return {'success': False, 'error': str(e)}
        finally:
            if session:
                session.close()
    
    def execute_manual(self, signal_data: Dict) -> Dict:
        """Ручне виконання"""
        config = self._load_config()
        
        try:
            signal = Signal(
                symbol=signal_data['symbol'],
                direction=Direction[signal_data['direction']],
                signal_type=SignalType[signal_data['signal_type']],
                price=float(signal_data['price']),
                sl_price=float(signal_data['sl_price']),
                tp1_price=float(signal_data['tp1_price']),
                tp2_price=float(signal_data['tp2_price']),
                rsi=float(signal_data.get('rsi', 0)),
                mfi=float(signal_data.get('mfi', 0)),
                mfi_cloud=signal_data.get('mfi_cloud', 'NEUTRAL'),
            )
            
            # Check existing position
            if self.has_open_position(signal.symbol):
                existing = self._open_positions.get(signal.symbol)
                if existing and existing.get('direction') == signal.direction.value:
                    return {'success': False, 'error': f'Already have {existing["direction"]} position'}
                
                if config.get('rsp_close_on_opposite', True):
                    self._close_by_opposite(signal.symbol, signal.direction.value, signal.price)
            
            return self._execute_signal(signal, config) or {'success': False, 'error': 'Execution failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_trades(self, limit: int = 50, status: str = None) -> List[Dict]:
        """Отримує угоди"""
        if not HAS_DB or db_manager is None:
            return []
        
        session = None
        try:
            session = db_manager.get_session()
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
                'pnl_percent': t.pnl_percent,
                'exit_reason': t.exit_reason,
                'rsi': t.rsi_value,
                'mfi_cloud': t.mfi_cloud,
                'timeframe': t.timeframe,
                'paper': t.paper_trade,
                'time': t.created_at.strftime('%d.%m %H:%M') if t.created_at else '',
            } for t in trades]
        except Exception as e:
            logger.error(f"Get trades error: {e}")
            return []
        finally:
            if session:
                session.close()
    
    def get_analytics(self) -> Dict:
        """Отримує аналітику"""
        if not HAS_DB or db_manager is None:
            return {}
        
        session = None
        try:
            session = db_manager.get_session()
            from sqlalchemy import func
            
            total = session.query(RSISniperTrade).filter(RSISniperTrade.status == 'Closed').count()
            wins = session.query(RSISniperTrade).filter(
                RSISniperTrade.status == 'Closed',
                RSISniperTrade.pnl_percent > 0
            ).count()
            
            total_pnl = session.query(func.sum(RSISniperTrade.pnl_percent)).filter(
                RSISniperTrade.status == 'Closed'
            ).scalar() or 0
            
            by_type = {}
            for signal_type in ['SNIPER', 'DIVERGENCE', 'FLOW', 'ROYAL']:
                type_total = session.query(RSISniperTrade).filter(
                    RSISniperTrade.status == 'Closed',
                    RSISniperTrade.signal_type == signal_type
                ).count()
                
                type_wins = session.query(RSISniperTrade).filter(
                    RSISniperTrade.status == 'Closed',
                    RSISniperTrade.signal_type == signal_type,
                    RSISniperTrade.pnl_percent > 0
                ).count()
                
                type_pnl = session.query(func.sum(RSISniperTrade.pnl_percent)).filter(
                    RSISniperTrade.status == 'Closed',
                    RSISniperTrade.signal_type == signal_type
                ).scalar() or 0
                
                by_type[signal_type] = {
                    'total': type_total,
                    'wins': type_wins,
                    'win_rate': (type_wins / type_total * 100) if type_total > 0 else 0,
                    'total_pnl': type_pnl,
                }
            
            return {
                'total_trades': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_pnl': total_pnl,
                'by_signal_type': by_type,
                'open_positions': len(self._open_positions),
            }
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return {}
        finally:
            if session:
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
            
            for _ in range(interval * 60):
                if not self.auto_running:
                    break
                time.sleep(1)


# ============================================================================
#                              FLASK ROUTES
# ============================================================================

# Singleton instance
_rsi_sniper_pro: Optional[RSISniperPro] = None


def get_rsi_sniper_pro() -> RSISniperPro:
    """Отримує або створює екземпляр"""
    global _rsi_sniper_pro
    if _rsi_sniper_pro is None:
        _rsi_sniper_pro = RSISniperPro()
    return _rsi_sniper_pro


def register_rsi_sniper_routes(app):
    """Реєструє Flask routes"""
    from flask import render_template, request, jsonify
    
    @app.route('/rsi_sniper')
    def rsi_sniper_page():
        rsp = get_rsi_sniper_pro()
        config = rsp.get_config()
        trades = rsp.get_trades(30)
        open_positions = rsp.get_trades(10, status='Open')
        analytics = rsp.get_analytics()
        return render_template('rsi_sniper_pro.html',
                              config=config,
                              trades=trades,
                              open_positions=open_positions,
                              analytics=analytics)
    
    @app.route('/rsi_sniper/status')
    def rsi_sniper_status():
        rsp = get_rsi_sniper_pro()
        status = rsp.get_status()
        status['results'] = [s.to_dict() for s in rsp.scan_results]
        return jsonify(status)
    
    @app.route('/rsi_sniper/scan', methods=['POST'])
    def rsi_sniper_scan():
        rsp = get_rsi_sniper_pro()
        return jsonify(rsp.start_scan())
    
    @app.route('/rsi_sniper/stop', methods=['POST'])
    def rsi_sniper_stop():
        rsp = get_rsi_sniper_pro()
        return jsonify(rsp.stop_scan())
    
    @app.route('/rsi_sniper/config', methods=['GET', 'POST'])
    def rsi_sniper_config():
        rsp = get_rsi_sniper_pro()
        if request.method == 'POST':
            data = request.json or {}
            rsp.save_config(data)
            return jsonify({'status': 'ok'})
        return jsonify(rsp.get_config())
    
    @app.route('/rsi_sniper/auto', methods=['POST'])
    def rsi_sniper_auto():
        rsp = get_rsi_sniper_pro()
        data = request.json or {}
        enabled = data.get('enabled', False)
        interval = int(data.get('interval', 1))
        
        if enabled:
            rsp.start_auto_mode(interval)
        else:
            rsp.stop_auto_mode()
        
        rsp.save_config({
            'rsp_auto_mode': enabled,
            'rsp_scan_interval': interval
        })
        
        return jsonify({'status': 'ok', 'auto_running': rsp.auto_running})
    
    @app.route('/rsi_sniper/execute', methods=['POST'])
    def rsi_sniper_execute():
        rsp = get_rsi_sniper_pro()
        data = request.json or {}
        result = rsp.execute_manual(data)
        return jsonify(result)
    
    @app.route('/rsi_sniper/trades')
    def rsi_sniper_trades():
        rsp = get_rsi_sniper_pro()
        limit = request.args.get('limit', 50, type=int)
        status = request.args.get('status', None)
        return jsonify(rsp.get_trades(limit, status))
    
    @app.route('/rsi_sniper/analytics')
    def rsi_sniper_analytics():
        rsp = get_rsi_sniper_pro()
        return jsonify(rsp.get_analytics())
    
    @app.route('/rsi_sniper/positions')
    def rsi_sniper_positions():
        rsp = get_rsi_sniper_pro()
        positions = rsp.get_trades(50, status='Open')
        tp1_positions = rsp.get_trades(50, status='TP1 Hit')
        return jsonify({
            'open': positions,
            'tp1_hit': tp1_positions,
            'total': len(positions) + len(tp1_positions)
        })
    
    @app.route('/rsi_sniper/defaults', methods=['POST'])
    def rsi_sniper_defaults():
        """Завантажити налаштування за замовчуванням"""
        rsp = get_rsi_sniper_pro()
        try:
            # Зберігаємо default конфіг
            default_config = {k: v for k, v in DEFAULT_CONFIG.items()}
            rsp.save_config(default_config)
            return jsonify({'success': True})
        except Exception as e:
            logger.error(f"Load defaults error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/rsi_sniper/trades/clear', methods=['POST'])
    def rsi_sniper_clear_trades():
        """Очистити всю історію угод"""
        rsp = get_rsi_sniper_pro()
        try:
            if HAS_DB and db_manager:
                session = db_manager.get_session()
                try:
                    deleted = session.query(RSISniperTrade).filter(
                        RSISniperTrade.status == 'Closed'
                    ).delete()
                    session.commit()
                    logger.info(f"🗑️ Cleared {deleted} closed trades")
                    return jsonify({'success': True, 'deleted': deleted})
                finally:
                    session.close()
            return jsonify({'success': False, 'error': 'Database not available'})
        except Exception as e:
            logger.error(f"Clear trades error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/rsi_sniper/trades/<int:trade_id>', methods=['DELETE'])
    def rsi_sniper_delete_trade(trade_id):
        """Видалити одну угоду"""
        try:
            if HAS_DB and db_manager:
                session = db_manager.get_session()
                try:
                    trade = session.query(RSISniperTrade).filter(
                        RSISniperTrade.id == trade_id
                    ).first()
                    if trade:
                        session.delete(trade)
                        session.commit()
                        logger.info(f"🗑️ Deleted trade #{trade_id}")
                        return jsonify({'success': True})
                    return jsonify({'success': False, 'error': 'Trade not found'})
                finally:
                    session.close()
            return jsonify({'success': False, 'error': 'Database not available'})
        except Exception as e:
            logger.error(f"Delete trade error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/rsi_sniper/positions/<int:trade_id>/close', methods=['POST'])
    def rsi_sniper_close_position(trade_id):
        """Закрити позицію вручну"""
        rsp = get_rsi_sniper_pro()
        try:
            if HAS_DB and db_manager:
                session = db_manager.get_session()
                try:
                    trade = session.query(RSISniperTrade).filter(
                        RSISniperTrade.id == trade_id,
                        RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
                    ).first()
                    if trade:
                        rsp._close_trade(session, trade, trade.current_price or trade.entry_price, 'MANUAL')
                        session.commit()
                        # Remove from cache
                        if trade.symbol in rsp._open_positions:
                            del rsp._open_positions[trade.symbol]
                        logger.info(f"✋ Manually closed position #{trade_id}")
                        return jsonify({'success': True})
                    return jsonify({'success': False, 'error': 'Position not found'})
                finally:
                    session.close()
            return jsonify({'success': False, 'error': 'Database not available'})
        except Exception as e:
            logger.error(f"Close position error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    @app.route('/rsi_sniper/positions/close-all', methods=['POST'])
    def rsi_sniper_close_all():
        """Закрити всі позиції"""
        rsp = get_rsi_sniper_pro()
        try:
            if HAS_DB and db_manager:
                session = db_manager.get_session()
                try:
                    trades = session.query(RSISniperTrade).filter(
                        RSISniperTrade.status.in_(['Open', 'TP1 Hit'])
                    ).all()
                    closed = 0
                    for trade in trades:
                        rsp._close_trade(session, trade, trade.current_price or trade.entry_price, 'MANUAL')
                        closed += 1
                    session.commit()
                    rsp._open_positions.clear()
                    logger.info(f"✋ Manually closed {closed} positions")
                    return jsonify({'success': True, 'closed': closed})
                finally:
                    session.close()
            return jsonify({'success': False, 'error': 'Database not available'})
        except Exception as e:
            logger.error(f"Close all error: {e}")
            return jsonify({'success': False, 'error': str(e)})
    
    logger.info("🎯 RSI Sniper PRO routes registered")
    
    # Initialize singleton when routes are registered
    get_rsi_sniper_pro()
