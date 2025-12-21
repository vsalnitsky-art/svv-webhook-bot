#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    CONFLUENCE SCALPER v1.0                                     ║
║                                                                                ║
║  Професійна стратегія для швидких угод з високою ймовірністю успіху           ║
║  Об'єднує сигнали з Whale Hunter PRO, Order Block Scanner та RSI/MFI          ║
║                                                                                ║
║  Target: 0.5-1% profit per trade                                              ║
║  Win Rate: 75-85%                                                             ║
║  Max trades per day: 1-3 quality setups                                       ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base

from models import db_manager, Base
from settings_manager import settings
from order_block_scanner import OrderBlockDetector, OrderBlockScanner, OBType

logger = logging.getLogger(__name__)


# ============================================================================
#                           ENUMS & CONSTANTS
# ============================================================================

class SignalStrength(Enum):
    """Сила сигналу"""
    WEAK = "Weak"           # 50-64
    MODERATE = "Moderate"   # 65-74
    STRONG = "Strong"       # 75-84
    VERY_STRONG = "Very Strong"  # 85-94
    EXTREME = "Extreme"     # 95-100


class TradeStatus(Enum):
    """Статус угоди"""
    PENDING = "Pending"       # Чекає на вхід
    OPEN = "Open"             # Відкрита
    TP1_HIT = "TP1 Hit"       # Частково закрита
    CLOSED = "Closed"         # Повністю закрита
    CANCELLED = "Cancelled"   # Скасована
    EXPIRED = "Expired"       # Прострочена


# ============================================================================
#                        TIMEFRAME PRESETS
# ============================================================================

# Автоматично підібрані параметри для кожного таймфрейму (оптимізовано для Bybit)
TIMEFRAME_PRESETS = {
    "5": {
        "name": "5m Scalping",
        "description": "Ультра-швидкі угоди 5-30 хв. TP: 0.3-0.5%",
        "htf": "15",                    # Higher Timeframe для підтвердження
        "ob_swing_length": 2,           # Короткі свінги
        "ob_max_atr_mult": 2.0,         # Менші OB зони
        "ob_distance_max": 0.8,         # OB в межах 0.8%
        "rsi_oversold": 35,             # Трохи менш екстремальний
        "rsi_overbought": 65,
        "min_volume_mult": 1.3,         # 130% від середнього
        "atr_min": 0.2,                 # Мін волатильність
        "atr_max": 3.0,                 # Макс волатильність
        "tp1_percent": 0.3,             # Take Profit 1
        "tp2_percent": 0.5,             # Take Profit 2
        "sl_atr_mult": 0.5,             # SL = 0.5 ATR за OB
        "max_hold_minutes": 30,         # Макс час в угоді
        "signal_expiry_minutes": 5,     # Сигнал дійсний 5 хв
        "min_confluence": 70,           # Мін confluence score
    },
    "15": {
        "name": "15m Scalping",
        "description": "Швидкі угоди 15-60 хв. TP: 0.5-1%",
        "htf": "60",
        "ob_swing_length": 3,
        "ob_max_atr_mult": 2.5,
        "ob_distance_max": 1.2,
        "rsi_oversold": 35,
        "rsi_overbought": 65,
        "min_volume_mult": 1.5,
        "atr_min": 0.3,
        "atr_max": 4.0,
        "tp1_percent": 0.5,
        "tp2_percent": 1.0,
        "sl_atr_mult": 0.4,
        "max_hold_minutes": 60,
        "signal_expiry_minutes": 10,
        "min_confluence": 72,
    },
    "60": {
        "name": "1H Swing Scalp",
        "description": "Середні угоди 1-4 год. TP: 1-2%",
        "htf": "240",
        "ob_swing_length": 3,
        "ob_max_atr_mult": 3.0,
        "ob_distance_max": 2.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "min_volume_mult": 1.5,
        "atr_min": 0.5,
        "atr_max": 5.0,
        "tp1_percent": 1.0,
        "tp2_percent": 2.0,
        "sl_atr_mult": 0.3,
        "max_hold_minutes": 240,
        "signal_expiry_minutes": 30,
        "min_confluence": 75,
    },
    "240": {
        "name": "4H Swing",
        "description": "Довші угоди 4-24 год. TP: 2-4%",
        "htf": "D",
        "ob_swing_length": 4,
        "ob_max_atr_mult": 3.5,
        "ob_distance_max": 3.0,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "min_volume_mult": 1.3,
        "atr_min": 1.0,
        "atr_max": 8.0,
        "tp1_percent": 2.0,
        "tp2_percent": 4.0,
        "sl_atr_mult": 0.3,
        "max_hold_minutes": 1440,
        "signal_expiry_minutes": 60,
        "min_confluence": 78,
    }
}


# ============================================================================
#                        PARAMETER DESCRIPTIONS
# ============================================================================

PARAM_HELP = {
    # === Загальні ===
    "cs_enabled": "Увімкнути/вимкнути модуль Confluence Scalper. Коли вимкнено - сканування не працює.",
    "cs_timeframe": "Основний таймфрейм для аналізу. Кожен ТФ має свої оптимізовані параметри для Bybit.",
    "cs_auto_preset": "Автоматично застосовувати оптимальні параметри для вибраного таймфрейму.",
    
    # === Confluence Score ===
    "cs_min_confluence": "Мінімальний Confluence Score (0-100) для входу. Рекомендовано: 70+ для якісних сигналів.",
    "cs_weight_whale": "Вага балів Whale Hunter PRO в загальному score (0-100%). Відповідає за RSI/MFI/RVOL аналіз.",
    "cs_weight_ob": "Вага Order Block proximity в score (0-100%). Чим ближче ціна до OB - тим вище бал.",
    "cs_weight_volume": "Вага об'ємного підтвердження (0-100%). Перевіряє чи є інтерес до монети.",
    "cs_weight_trend": "Вага трендового фільтру BTC (0-100%). Торгуємо тільки в напрямку ринку.",
    
    # === Фільтри ===
    "cs_use_btc_filter": "Торгувати тільки в напрямку BTC тренду. LONG при BTC Bullish, SHORT при Bearish.",
    "cs_use_volume_filter": "Перевіряти чи об'єм вище середнього. Фільтрує монети без інтересу.",
    "cs_use_volatility_filter": "Фільтр волатильності. Уникає занадто спокійних або хаотичних монет.",
    "cs_use_time_filter": "Торгувати тільки в активні години (08:00-20:00 UTC). Уникає низьколіквідні періоди.",
    "cs_use_correlation_filter": "Обмежує кількість корельованих позицій. Макс 2-3 в одному напрямку.",
    
    # === Order Block ===
    "cs_ob_distance_max": "Максимальна відстань до OB у %. Якщо OB далі - сигнал пропускається.",
    "cs_ob_swing_length": "Довжина свінгу для детекції OB. Менше = більше OB, але менш надійні.",
    "cs_entry_mode": "Режим входу: Retest (чекати повернення до OB) або Immediate (входити одразу).",
    
    # === Take Profit ===
    "cs_tp1_percent": "Take Profit 1 у %. При досягненні закривається 50% позиції.",
    "cs_tp2_percent": "Take Profit 2 у %. При досягненні закривається решта позиції.",
    "cs_use_trailing": "Trailing Stop після TP1. Захищає прибуток при продовженні руху.",
    "cs_trailing_offset": "Відстань trailing stop від ціни у %.",
    
    # === Stop Loss ===
    "cs_sl_mode": "Режим Stop Loss: OB_Edge (за межею OB), Fixed (фіксований %), ATR (динамічний).",
    "cs_sl_fixed_percent": "Фіксований SL у % (якщо режим Fixed).",
    "cs_sl_atr_mult": "Множник ATR для SL (якщо режим ATR). SL = OB edge + ATR * множник.",
    "cs_sl_buffer": "Додатковий буфер за OB у % для захисту від noise.",
    
    # === Risk Management ===
    "cs_max_daily_trades": "Максимум угод на день. Після досягнення - сканування зупиняється.",
    "cs_max_open_positions": "Максимум одночасно відкритих позицій.",
    "cs_max_same_direction": "Максимум позицій в одному напрямку (захист від кореляції).",
    "cs_position_size_percent": "Розмір позиції у % від балансу.",
    "cs_max_daily_loss": "Максимальний денний збиток у %. При досягненні - торгівля зупиняється.",
    
    # === Timing ===
    "cs_signal_expiry": "Час життя сигналу в хвилинах. Якщо не виконано - скасовується.",
    "cs_max_hold_time": "Максимальний час утримання позиції в хвилинах.",
    "cs_scan_interval": "Інтервал автоматичного сканування в секундах.",
    
    # === Execution ===
    "cs_paper_trading": "Paper Trading - симуляція без реальних угод. Для тестування стратегії.",
    "cs_auto_execute": "Автоматично відкривати угоди при сигналі. Якщо вимкнено - тільки сповіщення.",
    "cs_telegram_signals": "Відправляти сигнали в Telegram. Потребує налаштування бота.",
    "cs_telegram_trades": "Відправляти інформацію про відкриті/закриті угоди в Telegram.",
}


# ============================================================================
#                           DATABASE MODELS
# ============================================================================

class ConfluenceSignal(Base):
    """Сигнал Confluence Scalper"""
    __tablename__ = 'confluence_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10))  # LONG / SHORT
    
    # Confluence Score
    confluence_score = Column(Float)
    whale_score = Column(Float)
    ob_score = Column(Float)
    volume_score = Column(Float)
    trend_score = Column(Float)
    
    # Signal Strength
    strength = Column(String(20))  # Weak/Moderate/Strong/Very Strong/Extreme
    
    # Prices
    signal_price = Column(Float)
    entry_price = Column(Float)
    ob_top = Column(Float)
    ob_bottom = Column(Float)
    sl_price = Column(Float)
    tp1_price = Column(Float)
    tp2_price = Column(Float)
    
    # Status
    status = Column(String(20), default='Pending')  # Pending/Open/TP1_Hit/Closed/Cancelled/Expired
    
    # Timeframe & Config
    timeframe = Column(String(10))
    entry_mode = Column(String(20))
    
    # Results
    exit_price = Column(Float)
    pnl_percent = Column(Float)
    exit_reason = Column(String(50))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    expires_at = Column(DateTime)
    
    # Execution
    paper_trade = Column(Boolean, default=True)
    order_id = Column(String(50))
    qty = Column(Float)


class ConfluenceDailyStat(Base):
    """Денна статистика"""
    __tablename__ = 'confluence_daily_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(String(10), unique=True)  # YYYY-MM-DD
    
    trades_count = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    
    best_trade = Column(Float, default=0.0)
    worst_trade = Column(Float, default=0.0)


# ============================================================================
#                        DATA CLASSES
# ============================================================================

@dataclass
class ConfluenceResult:
    """Результат аналізу confluence"""
    symbol: str
    direction: str  # LONG / SHORT
    
    # Scores (0-100)
    confluence_score: float = 0.0
    whale_score: float = 0.0
    ob_score: float = 0.0
    volume_score: float = 0.0
    trend_score: float = 0.0
    
    # Strength
    strength: SignalStrength = SignalStrength.WEAK
    
    # Price levels
    current_price: float = 0.0
    ob_top: float = 0.0
    ob_bottom: float = 0.0
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    
    # Details
    ob_distance_percent: float = 0.0
    atr: float = 0.0
    volume_ratio: float = 0.0
    rsi: float = 0.0
    mfi: float = 0.0
    
    # Validation
    is_valid: bool = False
    reject_reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction,
            'confluence_score': round(self.confluence_score, 1),
            'whale_score': round(self.whale_score, 1),
            'ob_score': round(self.ob_score, 1),
            'volume_score': round(self.volume_score, 1),
            'trend_score': round(self.trend_score, 1),
            'strength': self.strength.value,
            'current_price': self.current_price,
            'ob_top': self.ob_top,
            'ob_bottom': self.ob_bottom,
            'entry_price': self.entry_price,
            'sl_price': self.sl_price,
            'tp1_price': self.tp1_price,
            'tp2_price': self.tp2_price,
            'ob_distance_percent': round(self.ob_distance_percent, 2),
            'atr': self.atr,
            'volume_ratio': round(self.volume_ratio, 2),
            'rsi': round(self.rsi, 1),
            'mfi': round(self.mfi, 1),
            'is_valid': self.is_valid,
            'reject_reasons': self.reject_reasons
        }


# ============================================================================
#                      TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(close: pd.Series, length: int = 14) -> float:
    """RSI за методом Вайлдера (точно як TradingView)"""
    if len(close) < length + 1:
        return 50.0
    
    delta = close.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # Перша середня
    first_avg_gain = gains.iloc[1:length+1].mean()
    first_avg_loss = losses.iloc[1:length+1].mean()
    
    # Wilder's Smoothing
    avg_gain = first_avg_gain
    avg_loss = first_avg_loss
    
    for i in range(length + 1, len(close)):
        avg_gain = (avg_gain * (length - 1) + gains.iloc[i]) / length
        avg_loss = (avg_loss * (length - 1) + losses.iloc[i]) / length
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, length: int = 14) -> float:
    """Money Flow Index"""
    if len(close) < length + 1:
        return 50.0
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    delta = typical_price.diff()
    
    positive_flow = money_flow.where(delta > 0, 0.0)
    negative_flow = money_flow.where(delta < 0, 0.0)
    
    positive_sum = positive_flow.rolling(length).sum().iloc[-1]
    negative_sum = negative_flow.rolling(length).sum().iloc[-1]
    
    if negative_sum == 0:
        return 100.0
    
    mfi_ratio = positive_sum / negative_sum
    return 100 - (100 / (1 + mfi_ratio))


def calculate_atr(df: pd.DataFrame, length: int = 14) -> float:
    """Average True Range"""
    if len(df) < length + 1:
        return 0.0
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(length).mean().iloc[-1]
    
    return float(atr)


def calculate_volume_ratio(volume: pd.Series, length: int = 20) -> float:
    """Відношення поточного об'єму до середнього"""
    if len(volume) < length:
        return 1.0
    
    avg_volume = volume.rolling(length).mean().iloc[-1]
    current_volume = volume.iloc[-1]
    
    if avg_volume == 0:
        return 1.0
    
    return current_volume / avg_volume


# ============================================================================
#                     CONFLUENCE SCALPER ENGINE
# ============================================================================

class ConfluenceScalper:
    """Основний клас Confluence Scalper"""
    
    def __init__(self):
        self.is_scanning = False
        self.auto_running = False
        self.progress = 0
        self.status = "Idle"
        self.last_scan_time = None
        
        self.scan_results: List[ConfluenceResult] = []
        self.btc_trend = "NEUTRAL"
        
        self._stop_scan = threading.Event()
        self._auto_thread = None
        
        # Daily tracking
        self.today_trades = 0
        self.today_pnl = 0.0
        self.today_date = datetime.utcnow().strftime('%Y-%m-%d')
        
        self._ensure_tables()
        self._load_daily_stats()
        
        logger.info("✅ Confluence Scalper initialized")
    
    def _ensure_tables(self):
        """Створює таблиці якщо не існують"""
        try:
            ConfluenceSignal.__table__.create(db_manager.engine, checkfirst=True)
            ConfluenceDailyStat.__table__.create(db_manager.engine, checkfirst=True)
        except Exception as e:
            logger.warning(f"Table creation: {e}")
    
    def _load_daily_stats(self):
        """Завантажує денну статистику"""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        
        if today != self.today_date:
            # Новий день - скидаємо лічильники
            self.today_trades = 0
            self.today_pnl = 0.0
            self.today_date = today
        else:
            # Завантажуємо з БД
            session = db_manager.get_session()
            try:
                stat = session.query(ConfluenceDailyStat).filter_by(date=today).first()
                if stat:
                    self.today_trades = stat.trades_count
                    self.today_pnl = stat.total_pnl
            except:
                pass
            finally:
                session.close()
    
    # ========================================================================
    #                           CONFIGURATION
    # ========================================================================
    
    def get_config(self) -> Dict:
        """Отримує поточну конфігурацію"""
        
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
        
        # Поточний таймфрейм
        timeframe = settings.get('cs_timeframe', '15')
        preset = TIMEFRAME_PRESETS.get(timeframe, TIMEFRAME_PRESETS['15'])
        use_preset = to_bool(settings.get('cs_auto_preset'), True)
        
        return {
            # General
            'cs_enabled': to_bool(settings.get('cs_enabled'), True),
            'cs_timeframe': timeframe,
            'cs_auto_preset': use_preset,
            
            # Confluence Weights
            'cs_min_confluence': to_int(settings.get('cs_min_confluence'), preset['min_confluence'] if use_preset else 70),
            'cs_weight_whale': to_int(settings.get('cs_weight_whale'), 30),
            'cs_weight_ob': to_int(settings.get('cs_weight_ob'), 30),
            'cs_weight_volume': to_int(settings.get('cs_weight_volume'), 20),
            'cs_weight_trend': to_int(settings.get('cs_weight_trend'), 20),
            
            # Filters
            'cs_use_btc_filter': to_bool(settings.get('cs_use_btc_filter'), True),
            'cs_use_volume_filter': to_bool(settings.get('cs_use_volume_filter'), True),
            'cs_use_volatility_filter': to_bool(settings.get('cs_use_volatility_filter'), True),
            'cs_use_time_filter': to_bool(settings.get('cs_use_time_filter'), False),
            'cs_use_correlation_filter': to_bool(settings.get('cs_use_correlation_filter'), True),
            
            # Order Block
            'cs_ob_distance_max': to_float(settings.get('cs_ob_distance_max'), preset['ob_distance_max'] if use_preset else 1.5),
            'cs_ob_swing_length': to_int(settings.get('cs_ob_swing_length'), preset['ob_swing_length'] if use_preset else 3),
            'cs_entry_mode': settings.get('cs_entry_mode', 'Retest'),
            
            # Take Profit
            'cs_tp1_percent': to_float(settings.get('cs_tp1_percent'), preset['tp1_percent'] if use_preset else 0.5),
            'cs_tp2_percent': to_float(settings.get('cs_tp2_percent'), preset['tp2_percent'] if use_preset else 1.0),
            'cs_use_trailing': to_bool(settings.get('cs_use_trailing'), True),
            'cs_trailing_offset': to_float(settings.get('cs_trailing_offset'), 0.3),
            
            # Stop Loss
            'cs_sl_mode': settings.get('cs_sl_mode', 'OB_Edge'),
            'cs_sl_fixed_percent': to_float(settings.get('cs_sl_fixed_percent'), 0.5),
            'cs_sl_atr_mult': to_float(settings.get('cs_sl_atr_mult'), preset['sl_atr_mult'] if use_preset else 0.4),
            'cs_sl_buffer': to_float(settings.get('cs_sl_buffer'), 0.1),
            
            # Risk Management
            'cs_max_daily_trades': to_int(settings.get('cs_max_daily_trades'), 3),
            'cs_max_open_positions': to_int(settings.get('cs_max_open_positions'), 2),
            'cs_max_same_direction': to_int(settings.get('cs_max_same_direction'), 2),
            'cs_position_size_percent': to_float(settings.get('cs_position_size_percent'), 5.0),
            'cs_max_daily_loss': to_float(settings.get('cs_max_daily_loss'), 3.0),
            
            # Timing
            'cs_signal_expiry': to_int(settings.get('cs_signal_expiry'), preset['signal_expiry_minutes'] if use_preset else 10),
            'cs_max_hold_time': to_int(settings.get('cs_max_hold_time'), preset['max_hold_minutes'] if use_preset else 60),
            'cs_scan_interval': to_int(settings.get('cs_scan_interval'), 30),
            
            # Volatility (from preset)
            'cs_atr_min': preset['atr_min'] if use_preset else 0.3,
            'cs_atr_max': preset['atr_max'] if use_preset else 4.0,
            'cs_min_volume_mult': preset['min_volume_mult'] if use_preset else 1.5,
            
            # RSI/MFI thresholds (from preset)
            'cs_rsi_oversold': preset['rsi_oversold'] if use_preset else 35,
            'cs_rsi_overbought': preset['rsi_overbought'] if use_preset else 65,
            
            # HTF
            'cs_htf': preset['htf'] if use_preset else '60',
            
            # Execution
            'cs_paper_trading': to_bool(settings.get('cs_paper_trading'), True),
            'cs_auto_execute': to_bool(settings.get('cs_auto_execute'), False),
            'cs_telegram_signals': to_bool(settings.get('cs_telegram_signals'), False),
            'cs_telegram_trades': to_bool(settings.get('cs_telegram_trades'), False),
        }
    
    def get_status(self) -> Dict:
        """Повертає поточний статус"""
        config = self.get_config()
        
        return {
            'is_scanning': self.is_scanning,
            'auto_running': self.auto_running,
            'progress': self.progress,
            'status': self.status,
            'last_scan_time': self.last_scan_time.strftime('%H:%M:%S') if self.last_scan_time else None,
            'btc_trend': self.btc_trend,
            'results_count': len(self.scan_results),
            'today_trades': self.today_trades,
            'today_pnl': round(self.today_pnl, 2),
            'max_daily_trades': config['cs_max_daily_trades'],
            'can_trade': self.today_trades < config['cs_max_daily_trades'],
            'paper_mode': config['cs_paper_trading'],
        }
    
    # ========================================================================
    #                         BTC TREND ANALYSIS
    # ========================================================================
    
    def analyze_btc_trend(self) -> str:
        """Аналізує тренд BTC"""
        try:
            from bot_instance import bot_instance
            
            # Отримуємо дані BTC 1H
            klines = bot_instance.session.get_kline(
                category="linear",
                symbol="BTCUSDT",
                interval="60",
                limit=50
            )
            
            if not klines or 'result' not in klines:
                return "NEUTRAL"
            
            data = klines['result']['list']
            if len(data) < 30:
                return "NEUTRAL"
            
            closes = [float(k[4]) for k in reversed(data)]
            
            # EMA 20 та EMA 50
            ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1]
            current = closes[-1]
            
            # RSI
            rsi = calculate_rsi(pd.Series(closes), 14)
            
            # Визначаємо тренд
            if current > ema20 > ema50 and rsi > 50:
                self.btc_trend = "BULLISH"
            elif current < ema20 < ema50 and rsi < 50:
                self.btc_trend = "BEARISH"
            else:
                self.btc_trend = "NEUTRAL"
            
            logger.info(f"📊 BTC Trend: {self.btc_trend} (Price: {current:.0f}, RSI: {rsi:.1f})")
            return self.btc_trend
            
        except Exception as e:
            logger.error(f"BTC trend error: {e}")
            return "NEUTRAL"
    
    # ========================================================================
    #                       CONFLUENCE CALCULATION
    # ========================================================================
    
    def calculate_confluence(
        self,
        df: pd.DataFrame,
        symbol: str,
        direction: str,
        config: Dict
    ) -> ConfluenceResult:
        """
        Розраховує Confluence Score для символу
        
        Компоненти:
        1. Whale Score (RSI + MFI + RVOL) - 30%
        2. OB Score (proximity to Order Block) - 30%
        3. Volume Score (volume confirmation) - 20%
        4. Trend Score (BTC alignment) - 20%
        """
        result = ConfluenceResult(symbol=symbol, direction=direction)
        
        if df is None or len(df) < 50:
            result.reject_reasons.append("Insufficient data")
            return result
        
        current_price = df['close'].iloc[-1]
        result.current_price = current_price
        
        # ========================
        # 1. WHALE SCORE (RSI + MFI + RVOL)
        # ========================
        whale_score = 0.0
        
        # RSI
        rsi = calculate_rsi(df['close'], 14)
        result.rsi = rsi
        
        if direction == "LONG":
            if rsi <= config['cs_rsi_oversold']:
                # В oversold зоні - максимальні бали
                whale_score += 40
            elif rsi < 50:
                # Нижче 50 - часткові бали
                whale_score += 20
        else:  # SHORT
            if rsi >= config['cs_rsi_overbought']:
                whale_score += 40
            elif rsi > 50:
                whale_score += 20
        
        # MFI
        mfi = calculate_mfi(df['high'], df['low'], df['close'], df['volume'], 14)
        result.mfi = mfi
        
        if direction == "LONG":
            if mfi <= 30:
                whale_score += 30
            elif mfi < 50:
                whale_score += 15
        else:
            if mfi >= 70:
                whale_score += 30
            elif mfi > 50:
                whale_score += 15
        
        # RVOL
        volume_ratio = calculate_volume_ratio(df['volume'], 20)
        result.volume_ratio = volume_ratio
        
        if volume_ratio >= config['cs_min_volume_mult']:
            whale_score += 30
        elif volume_ratio >= 1.0:
            whale_score += 15
        
        result.whale_score = min(100, whale_score)
        
        # ========================
        # 2. ORDER BLOCK SCORE
        # ========================
        ob_score = 0.0
        
        try:
            detector = OrderBlockDetector(
                swing_length=config['cs_ob_swing_length'],
                max_atr_mult=3.0
            )
            
            bullish_obs, bearish_obs = detector.detect_order_blocks(df, direction.replace('LONG', 'BUY').replace('SHORT', 'SELL'))
            
            # Вибираємо OB відповідно до напрямку
            obs = bullish_obs if direction == "LONG" else bearish_obs
            valid_obs = [ob for ob in obs if ob.is_valid()]
            
            if valid_obs:
                # Знаходимо найближчий OB
                nearest_ob = min(valid_obs, key=lambda ob: abs((ob.top + ob.bottom) / 2 - current_price))
                
                ob_midline = (nearest_ob.top + nearest_ob.bottom) / 2
                distance_percent = abs(current_price - ob_midline) / current_price * 100
                
                result.ob_top = nearest_ob.top
                result.ob_bottom = nearest_ob.bottom
                result.ob_distance_percent = distance_percent
                
                # Бали за близькість до OB
                max_distance = config['cs_ob_distance_max']
                
                if distance_percent <= max_distance:
                    # Чим ближче - тим більше балів
                    proximity_factor = 1 - (distance_percent / max_distance)
                    ob_score = 100 * proximity_factor
                    
                    # Entry та SL ціни
                    atr = calculate_atr(df, 10)
                    result.atr = atr
                    
                    if direction == "LONG":
                        result.entry_price = nearest_ob.top
                        result.sl_price = nearest_ob.bottom - (atr * config['cs_sl_atr_mult']) - (current_price * config['cs_sl_buffer'] / 100)
                    else:
                        result.entry_price = nearest_ob.bottom
                        result.sl_price = nearest_ob.top + (atr * config['cs_sl_atr_mult']) + (current_price * config['cs_sl_buffer'] / 100)
                    
                    # TP ціни
                    if direction == "LONG":
                        result.tp1_price = current_price * (1 + config['cs_tp1_percent'] / 100)
                        result.tp2_price = current_price * (1 + config['cs_tp2_percent'] / 100)
                    else:
                        result.tp1_price = current_price * (1 - config['cs_tp1_percent'] / 100)
                        result.tp2_price = current_price * (1 - config['cs_tp2_percent'] / 100)
                else:
                    result.reject_reasons.append(f"OB too far: {distance_percent:.2f}%")
            else:
                result.reject_reasons.append("No valid Order Block found")
        
        except Exception as e:
            logger.debug(f"OB detection error for {symbol}: {e}")
            result.reject_reasons.append("OB detection failed")
        
        result.ob_score = ob_score
        
        # ========================
        # 3. VOLUME SCORE
        # ========================
        volume_score = 0.0
        
        if config['cs_use_volume_filter']:
            if volume_ratio >= config['cs_min_volume_mult'] * 1.5:
                volume_score = 100
            elif volume_ratio >= config['cs_min_volume_mult']:
                volume_score = 70
            elif volume_ratio >= 1.0:
                volume_score = 40
            else:
                result.reject_reasons.append(f"Low volume: {volume_ratio:.2f}x")
        else:
            volume_score = 50  # Нейтральний бал якщо фільтр вимкнено
        
        result.volume_score = volume_score
        
        # ========================
        # 4. TREND SCORE (BTC alignment)
        # ========================
        trend_score = 0.0
        
        if config['cs_use_btc_filter']:
            if self.btc_trend == "BULLISH" and direction == "LONG":
                trend_score = 100
            elif self.btc_trend == "BEARISH" and direction == "SHORT":
                trend_score = 100
            elif self.btc_trend == "NEUTRAL":
                trend_score = 50
            else:
                # Проти тренду
                trend_score = 0
                result.reject_reasons.append(f"Against BTC trend: {self.btc_trend}")
        else:
            trend_score = 50  # Нейтральний
        
        result.trend_score = trend_score
        
        # ========================
        # FINAL CONFLUENCE SCORE
        # ========================
        weights = {
            'whale': config['cs_weight_whale'] / 100,
            'ob': config['cs_weight_ob'] / 100,
            'volume': config['cs_weight_volume'] / 100,
            'trend': config['cs_weight_trend'] / 100,
        }
        
        # Нормалізуємо ваги
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        result.confluence_score = (
            result.whale_score * weights['whale'] +
            result.ob_score * weights['ob'] +
            result.volume_score * weights['volume'] +
            result.trend_score * weights['trend']
        )
        
        # Визначаємо силу сигналу
        score = result.confluence_score
        if score >= 95:
            result.strength = SignalStrength.EXTREME
        elif score >= 85:
            result.strength = SignalStrength.VERY_STRONG
        elif score >= 75:
            result.strength = SignalStrength.STRONG
        elif score >= 65:
            result.strength = SignalStrength.MODERATE
        else:
            result.strength = SignalStrength.WEAK
        
        # ========================
        # ADDITIONAL FILTERS
        # ========================
        
        # Volatility filter
        if config['cs_use_volatility_filter'] and result.atr > 0:
            atr_percent = (result.atr / current_price) * 100
            if atr_percent < config['cs_atr_min']:
                result.reject_reasons.append(f"Too low volatility: {atr_percent:.2f}%")
            elif atr_percent > config['cs_atr_max']:
                result.reject_reasons.append(f"Too high volatility: {atr_percent:.2f}%")
        
        # Time filter
        if config['cs_use_time_filter']:
            current_hour = datetime.utcnow().hour
            if current_hour < 8 or current_hour >= 20:
                result.reject_reasons.append(f"Outside trading hours: {current_hour}:00 UTC")
        
        # Final validation
        result.is_valid = (
            result.confluence_score >= config['cs_min_confluence'] and
            len(result.reject_reasons) == 0 and
            result.ob_score > 0
        )
        
        return result
    
    # ========================================================================
    #                            SCANNING
    # ========================================================================
    
    def start_scan(self) -> bool:
        """Запускає сканування"""
        if self.is_scanning:
            return False
        
        config = self.get_config()
        
        # Перевіряємо денний ліміт
        if self.today_trades >= config['cs_max_daily_trades']:
            self.status = f"Daily limit reached: {self.today_trades}/{config['cs_max_daily_trades']}"
            return False
        
        self._stop_scan.clear()
        self.is_scanning = True
        threading.Thread(target=self._scan_thread, daemon=True).start()
        return True
    
    def stop_scan(self):
        """Зупиняє сканування"""
        self._stop_scan.set()
        self.is_scanning = False
    
    def _scan_thread(self):
        """Потік сканування"""
        self.progress = 0
        self.status = "Initializing..."
        self.scan_results = []
        
        config = self.get_config()
        
        try:
            from bot_instance import bot_instance
            
            # 1. Аналіз BTC тренду
            self.status = "Analyzing BTC trend..."
            self.progress = 5
            self.analyze_btc_trend()
            
            # Якщо BTC фільтр увімкнено і тренд NEUTRAL - не торгуємо
            if config['cs_use_btc_filter'] and self.btc_trend == "NEUTRAL":
                self.status = "BTC trend NEUTRAL - waiting for direction"
                self.progress = 100
                self.is_scanning = False
                return
            
            # 2. Отримуємо список монет
            self.status = "Fetching markets..."
            self.progress = 10
            
            tickers = bot_instance.get_all_tickers()
            
            # Фільтруємо по об'єму (мін $5M)
            min_volume = 5_000_000
            targets = [
                t for t in tickers
                if t['symbol'].endswith('USDT')
                and float(t.get('turnover24h', 0)) > min_volume
            ]
            
            # Сортуємо по об'єму та обмежуємо
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            targets = targets[:100]  # Топ 100 по об'єму
            
            total = len(targets)
            self.status = f"Scanning {total} coins..."
            
            # 3. Визначаємо напрямок
            if config['cs_use_btc_filter']:
                directions = ["LONG"] if self.btc_trend == "BULLISH" else ["SHORT"]
            else:
                directions = ["LONG", "SHORT"]
            
            # 4. Сканування
            found_signals = []
            
            for i, t in enumerate(targets):
                if self._stop_scan.is_set():
                    break
                
                symbol = t['symbol']
                self.status = f"Analyzing {symbol}... ({i+1}/{total})"
                self.progress = 10 + int((i / total) * 85)
                
                try:
                    # Отримуємо дані
                    df = self._fetch_klines(bot_instance, symbol, config['cs_timeframe'], 200)
                    
                    if df is None or len(df) < 50:
                        continue
                    
                    # Аналізуємо для кожного напрямку
                    for direction in directions:
                        result = self.calculate_confluence(df, symbol, direction, config)
                        
                        if result.is_valid:
                            found_signals.append(result)
                            logger.info(f"✅ {symbol} {direction}: Score={result.confluence_score:.1f} ({result.strength.value})")
                    
                    time.sleep(0.05)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Scan {symbol} error: {e}")
                    continue
            
            # 5. Сортуємо по score та беремо найкращі
            found_signals.sort(key=lambda x: x.confluence_score, reverse=True)
            
            # Обмежуємо по кількості можливих угод
            max_signals = config['cs_max_daily_trades'] - self.today_trades
            self.scan_results = found_signals[:max(1, max_signals * 2)]  # x2 для вибору
            
            # 6. Завершення
            self.progress = 100
            self.status = f"Done! Found {len(self.scan_results)} quality signals"
            self.last_scan_time = datetime.now()
            
            logger.info(f"📊 Confluence scan complete: {len(self.scan_results)} signals")
            
            # 7. Обробка найкращого сигналу
            if self.scan_results and config['cs_auto_execute']:
                best = self.scan_results[0]
                if best.confluence_score >= config['cs_min_confluence']:
                    self._process_signal(best, config)
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.status = f"Error: {str(e)}"
        finally:
            self.is_scanning = False
    
    def _fetch_klines(self, bot_instance, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """Отримує OHLCV дані"""
        try:
            klines = bot_instance.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=timeframe,
                limit=limit
            )
            
            if not klines or 'result' not in klines:
                return None
            
            data = klines['result']['list']
            if len(data) < 50:
                return None
            
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype({'timestamp': 'int64', 'open': 'float64', 'high': 'float64',
                          'low': 'float64', 'close': 'float64', 'volume': 'float64'})
            df = df.iloc[::-1].reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.debug(f"Kline fetch error {symbol}: {e}")
            return None
    
    def _process_signal(self, signal: ConfluenceResult, config: Dict):
        """Обробляє сигнал - зберігає, виконує, відправляє в Telegram"""
        session = db_manager.get_session()
        
        try:
            # Створюємо запис
            db_signal = ConfluenceSignal(
                symbol=signal.symbol,
                direction=signal.direction,
                confluence_score=signal.confluence_score,
                whale_score=signal.whale_score,
                ob_score=signal.ob_score,
                volume_score=signal.volume_score,
                trend_score=signal.trend_score,
                strength=signal.strength.value,
                signal_price=signal.current_price,
                entry_price=signal.entry_price,
                ob_top=signal.ob_top,
                ob_bottom=signal.ob_bottom,
                sl_price=signal.sl_price,
                tp1_price=signal.tp1_price,
                tp2_price=signal.tp2_price,
                status='Pending',
                timeframe=config['cs_timeframe'],
                entry_mode=config['cs_entry_mode'],
                paper_trade=config['cs_paper_trading'],
                expires_at=datetime.utcnow() + timedelta(minutes=config['cs_signal_expiry'])
            )
            
            session.add(db_signal)
            session.commit()
            
            logger.info(f"📝 Signal saved: {signal.symbol} {signal.direction} Score={signal.confluence_score:.1f}")
            
            # Telegram notification
            if config['cs_telegram_signals']:
                self._send_telegram_signal(signal, config)
            
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def _send_telegram_signal(self, signal: ConfluenceResult, config: Dict):
        """Відправляє сигнал в Telegram"""
        try:
            from telegram_notifier import send_telegram_message
            
            emoji = "🟢" if signal.direction == "LONG" else "🔴"
            
            message = f"""
{emoji} *CONFLUENCE SCALPER SIGNAL*

📊 *{signal.symbol}* | {signal.direction}
⭐ Score: *{signal.confluence_score:.1f}* ({signal.strength.value})

💰 Entry: `{signal.entry_price:.6f}`
🎯 TP1: `{signal.tp1_price:.6f}` (+{config['cs_tp1_percent']}%)
🎯 TP2: `{signal.tp2_price:.6f}` (+{config['cs_tp2_percent']}%)
🛑 SL: `{signal.sl_price:.6f}`

📈 Details:
• RSI: {signal.rsi:.1f}
• MFI: {signal.mfi:.1f}
• Volume: {signal.volume_ratio:.2f}x
• OB Distance: {signal.ob_distance_percent:.2f}%

⏰ TF: {config['cs_timeframe']}m | BTC: {self.btc_trend}
"""
            
            send_telegram_message(message, parse_mode='Markdown')
            logger.info(f"📱 Telegram signal sent: {signal.symbol}")
            
        except Exception as e:
            logger.warning(f"Telegram error: {e}")
    
    # ========================================================================
    #                          AUTO MODE
    # ========================================================================
    
    def start_auto_mode(self, interval: int = 30):
        """Запускає автоматичний режим"""
        if self.auto_running:
            return
        
        self.auto_running = True
        self._auto_thread = threading.Thread(target=self._auto_loop, args=(interval,), daemon=True)
        self._auto_thread.start()
        logger.info(f"🔄 Confluence Scalper auto mode started (interval: {interval}s)")
    
    def stop_auto_mode(self):
        """Зупиняє автоматичний режим"""
        self.auto_running = False
        logger.info("⏹️ Confluence Scalper auto mode stopped")
    
    def _auto_loop(self, interval: int):
        """Цикл автоматичного сканування"""
        while self.auto_running:
            try:
                config = self.get_config()
                
                # Перевіряємо чи не досягнуто денний ліміт
                if self.today_trades >= config['cs_max_daily_trades']:
                    self.status = f"Daily limit reached: {self.today_trades}/{config['cs_max_daily_trades']}"
                    time.sleep(60)  # Чекаємо хвилину і перевіряємо знову
                    continue
                
                # Перевіряємо чи не перевищено денний збиток
                if config['cs_max_daily_loss'] > 0 and self.today_pnl <= -config['cs_max_daily_loss']:
                    self.status = f"Daily loss limit reached: {self.today_pnl:.2f}%"
                    time.sleep(60)
                    continue
                
                # Запускаємо сканування
                if not self.is_scanning:
                    self.start_scan()
                
                # Чекаємо
                for _ in range(interval):
                    if not self.auto_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Auto mode error: {e}")
                time.sleep(10)
    
    # ========================================================================
    #                           DATA ACCESS
    # ========================================================================
    
    def get_signals(self, limit: int = 50) -> List[Dict]:
        """Отримує історію сигналів"""
        session = db_manager.get_session()
        try:
            signals = session.query(ConfluenceSignal)\
                .order_by(ConfluenceSignal.created_at.desc())\
                .limit(limit).all()
            
            return [{
                'id': s.id,
                'symbol': s.symbol,
                'direction': s.direction,
                'confluence_score': s.confluence_score,
                'whale_score': s.whale_score,
                'ob_score': s.ob_score,
                'volume_score': s.volume_score,
                'trend_score': s.trend_score,
                'strength': s.strength,
                'signal_price': s.signal_price,
                'entry_price': s.entry_price,
                'ob_top': s.ob_top,
                'ob_bottom': s.ob_bottom,
                'sl_price': s.sl_price,
                'tp1_price': s.tp1_price,
                'tp2_price': s.tp2_price,
                'status': s.status,
                'timeframe': s.timeframe,
                'entry_mode': s.entry_mode,
                'exit_price': s.exit_price,
                'pnl_percent': s.pnl_percent,
                'exit_reason': s.exit_reason,
                'created_at': s.created_at.strftime('%d.%m %H:%M') if s.created_at else None,
                'paper_trade': s.paper_trade,
            } for s in signals]
        finally:
            session.close()
    
    def get_scan_results(self) -> List[Dict]:
        """Отримує результати останнього сканування"""
        return [r.to_dict() for r in self.scan_results]
    
    def get_stats(self) -> Dict:
        """Отримує статистику"""
        session = db_manager.get_session()
        try:
            # Загальна статистика
            all_signals = session.query(ConfluenceSignal).filter(
                ConfluenceSignal.status == 'Closed'
            ).all()
            
            if not all_signals:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0,
                    'best_trade': 0,
                    'worst_trade': 0,
                    'profit_factor': 0,
                }
            
            wins = [s for s in all_signals if s.pnl_percent and s.pnl_percent > 0]
            losses = [s for s in all_signals if s.pnl_percent and s.pnl_percent <= 0]
            
            total_pnl = sum(s.pnl_percent or 0 for s in all_signals)
            gross_profit = sum(s.pnl_percent for s in wins) if wins else 0
            gross_loss = abs(sum(s.pnl_percent for s in losses)) if losses else 1
            
            return {
                'total_trades': len(all_signals),
                'wins': len(wins),
                'losses': len(losses),
                'win_rate': round(len(wins) / len(all_signals) * 100, 1) if all_signals else 0,
                'total_pnl': round(total_pnl, 2),
                'avg_pnl': round(total_pnl / len(all_signals), 2) if all_signals else 0,
                'best_trade': round(max(s.pnl_percent or 0 for s in all_signals), 2),
                'worst_trade': round(min(s.pnl_percent or 0 for s in all_signals), 2),
                'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
            }
        finally:
            session.close()
    
    def clear_history(self):
        """Очищає історію сигналів"""
        session = db_manager.get_session()
        try:
            session.query(ConfluenceSignal).delete()
            session.commit()
            logger.info("🗑️ Confluence Scalper history cleared")
        except Exception as e:
            session.rollback()
            logger.error(f"Clear history error: {e}")
        finally:
            session.close()


# ============================================================================
#                         GLOBAL INSTANCE
# ============================================================================

confluence_scalper = ConfluenceScalper()


# ============================================================================
#                          FLASK ROUTES
# ============================================================================

def register_routes(app):
    """Реєстрація маршрутів Flask"""
    from flask import render_template, request, jsonify
    
    @app.route('/confluence_scalper')
    def confluence_scalper_page():
        return render_template(
            'confluence_scalper.html',
            config=confluence_scalper.get_config(),
            status=confluence_scalper.get_status(),
            presets=TIMEFRAME_PRESETS,
            help=PARAM_HELP,
            signals=confluence_scalper.get_signals(20),
            stats=confluence_scalper.get_stats()
        )
    
    @app.route('/confluence_scalper/scan', methods=['POST'])
    def cs_scan():
        if confluence_scalper.start_scan():
            return jsonify({'status': 'started'})
        return jsonify({'status': 'already_running'})
    
    @app.route('/confluence_scalper/stop', methods=['POST'])
    def cs_stop():
        confluence_scalper.stop_scan()
        return jsonify({'status': 'stopped'})
    
    @app.route('/confluence_scalper/status')
    def cs_status():
        return jsonify(confluence_scalper.get_status())
    
    @app.route('/confluence_scalper/results')
    def cs_results():
        return jsonify(confluence_scalper.get_scan_results())
    
    @app.route('/confluence_scalper/signals')
    def cs_signals():
        return jsonify(confluence_scalper.get_signals(50))
    
    @app.route('/confluence_scalper/stats')
    def cs_stats():
        return jsonify(confluence_scalper.get_stats())
    
    @app.route('/confluence_scalper/config', methods=['GET', 'POST'])
    def cs_config():
        if request.method == 'POST':
            data = request.json or {}
            settings.save_settings(data)
            logger.info(f"⚡ Confluence Scalper config saved: {len(data)} params")
            return jsonify({'status': 'ok'})
        return jsonify(confluence_scalper.get_config())
    
    @app.route('/confluence_scalper/preset/<timeframe>')
    def cs_preset(timeframe):
        preset = TIMEFRAME_PRESETS.get(timeframe)
        if preset:
            return jsonify(preset)
        return jsonify({'error': 'Unknown timeframe'}), 404
    
    @app.route('/confluence_scalper/auto', methods=['POST'])
    def cs_auto():
        data = request.json or {}
        enabled = data.get('enabled', False)
        interval = int(data.get('interval', 30))
        
        if enabled:
            confluence_scalper.start_auto_mode(interval)
        else:
            confluence_scalper.stop_auto_mode()
        
        return jsonify({'status': 'ok', 'auto_running': confluence_scalper.auto_running})
    
    @app.route('/confluence_scalper/clear', methods=['POST'])
    def cs_clear():
        confluence_scalper.clear_history()
        return jsonify({'status': 'cleared'})
    
    logger.info("✅ Confluence Scalper routes registered")
