#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     COILING SPRING SCANNER v1.0                              ║
║                                                                              ║
║  Стратегія "Стиснута Пружина" - пошук монет у консолідації                   ║
║  перед вибуховим рухом (памп/дамп)                                           ║
║                                                                              ║
║  Ключові метрики:                                                            ║
║  • Open Interest (OI) - "паливо" для руху                                    ║
║  • CVD (Cumulative Volume Delta) - приховане накопичення                     ║
║  • Funding Rate - Short/Long Squeeze сигнали                                 ║
║  • Ichimoku Cloud - зона консолідації                                        ║
║  • Volatility Contraction - стиснення перед вибухом                          ║
║                                                                              ║
║  Автор: SVV Webhook Bot                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
import threading
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Flask
from flask import Blueprint, render_template, jsonify, request

# Database
try:
    from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
    from sqlalchemy.ext.declarative import declarative_base
    from database_manager import DatabaseManager
    Base = declarative_base()
    HAS_DB = True
except ImportError:
    HAS_DB = False
    Base = object

# Bot instance
try:
    from main_app import bot_instance, db_manager
    HAS_BOT = True
except ImportError:
    bot_instance = None
    db_manager = None
    HAS_BOT = False

# Pandas
try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    pd = None
    np = None
    HAS_PANDAS = False

# Settings
try:
    from settings_manager import settings
except ImportError:
    settings = None

logger = logging.getLogger(__name__)


# ============================================================================
#                              ENUMS & CONSTANTS
# ============================================================================

class SpringSignal(Enum):
    """Тип сигналу пружини"""
    COILING = "COILING"           # Стиснення (підготовка)
    READY_PUMP = "READY_PUMP"     # Готовий до пампу
    READY_DUMP = "READY_DUMP"     # Готовий до дампу
    BREAKOUT_UP = "BREAKOUT_UP"   # Пробій вгору
    BREAKOUT_DOWN = "BREAKOUT_DOWN"  # Пробій вниз


class TrendBias(Enum):
    """Ухил тренду"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


# ============================================================================
#                              DEFAULT CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    # General
    'css_enabled': True,
    'css_scan_interval': 300,       # 5 хвилин між сканами
    'css_min_volume_24h': 10_000_000,  # Мін об'єм $10M
    'css_max_spread_percent': 0.3,  # Макс спред 0.3%
    'css_scan_limit': 100,          # Скільки монет сканувати
    
    # Open Interest
    'css_use_oi': True,
    'css_oi_change_threshold': 5.0,  # OI має зрости на 5%+
    'css_oi_lookback_hours': 1,      # За останню годину
    'css_oi_weight': 25,             # Вага в скорингу
    
    # Funding Rate
    'css_use_funding': True,
    'css_funding_extreme_pos': 0.03,  # Екстремальний позитивний (3%)
    'css_funding_extreme_neg': -0.03, # Екстремальний негативний (-3%)
    'css_funding_weight': 15,
    
    # CVD (Volume Delta)
    'css_use_cvd': True,
    'css_cvd_divergence_threshold': 0.5,  # % розбіжності
    'css_cvd_weight': 20,
    
    # Volatility Contraction
    'css_use_volatility': True,
    'css_bb_width_percentile': 20,   # BB Width у нижніх 20%
    'css_atr_contraction': 0.7,      # ATR < 70% від середнього
    'css_volatility_weight': 15,
    
    # Ichimoku Cloud
    'css_use_ichimoku': True,
    'css_ichimoku_tenkan': 9,
    'css_ichimoku_kijun': 26,
    'css_ichimoku_senkou_b': 52,
    'css_ichimoku_weight': 15,
    
    # Price Action
    'css_price_change_max': 1.0,     # Ціна змінилась < 1% (флет)
    'css_price_lookback_hours': 4,   # За останні 4 години
    
    # Scoring
    'css_min_score': 60,             # Мінімальний скор для сигналу
    'css_strong_signal_score': 80,   # Сильний сигнал
    
    # Timeframe
    'css_main_tf': '15',             # Основний таймфрейм
    'css_htf': '60',                 # Higher timeframe для підтвердження
    
    # Execution
    'css_paper_trading': True,
    'css_telegram_alerts': True,
    'css_auto_mode': False,
}

PARAM_HELP = {
    'css_oi_change_threshold': 'Open Interest має зрости на X% при флеті ціни',
    'css_funding_extreme_pos': 'Екстремально високий Funding = готовність до Long Squeeze',
    'css_funding_extreme_neg': 'Екстремально низький Funding = готовність до Short Squeeze',
    'css_bb_width_percentile': 'Bollinger Bands Width у нижніх X% = сильне стиснення',
    'css_price_change_max': 'Максимальна зміна ціни для визначення флету',
}


# ============================================================================
#                              DATABASE MODEL
# ============================================================================

if HAS_DB:
    class CoilingSpringSignal(Base):
        """Модель для збереження сигналів"""
        __tablename__ = 'coiling_spring_signals'
        __table_args__ = {'extend_existing': True}
        
        id = Column(Integer, primary_key=True)
        symbol = Column(String(20), index=True)
        signal_type = Column(String(20))  # COILING, READY_PUMP, etc.
        trend_bias = Column(String(10))   # BULLISH, BEARISH, NEUTRAL
        
        # Scores
        total_score = Column(Integer)
        oi_score = Column(Integer, default=0)
        funding_score = Column(Integer, default=0)
        cvd_score = Column(Integer, default=0)
        volatility_score = Column(Integer, default=0)
        ichimoku_score = Column(Integer, default=0)
        
        # Raw metrics
        oi_change_percent = Column(Float)
        funding_rate = Column(Float)
        cvd_divergence = Column(Float)
        bb_width_percentile = Column(Float)
        atr_ratio = Column(Float)
        price_change_percent = Column(Float)
        
        # Ichimoku
        price_vs_cloud = Column(String(20))  # ABOVE, INSIDE, BELOW
        tenkan_kijun_cross = Column(String(10))  # BULLISH, BEARISH, NONE
        
        # Market data
        current_price = Column(Float)
        volume_24h = Column(Float)
        open_interest = Column(Float)
        
        # Meta
        timeframe = Column(String(10))
        created_at = Column(DateTime, default=datetime.utcnow)
        notes = Column(Text)
else:
    CoilingSpringSignal = None


# ============================================================================
#                              INDICATOR CALCULATIONS
# ============================================================================

class IndicatorEngine:
    """Рушій для розрахунку індикаторів"""
    
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict:
        """
        Розраховує Ichimoku Cloud
        
        Returns:
            Dict з компонентами Ichimoku та статусом
        """
        if df is None or len(df) < senkou_b + 26:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (Conversion Line)
        tenkan_high = high.rolling(window=tenkan).max()
        tenkan_low = low.rolling(window=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = high.rolling(window=kijun).max()
        kijun_low = low.rolling(window=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_b_high = high.rolling(window=senkou_b).max()
        senkou_b_low = low.rolling(window=senkou_b).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun)
        
        # Current values
        current_price = float(close.iloc[-1])
        current_tenkan = float(tenkan_sen.iloc[-1])
        current_kijun = float(kijun_sen.iloc[-1])
        current_senkou_a = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else current_price
        current_senkou_b = float(senkou_span_b.iloc[-1]) if not pd.isna(senkou_span_b.iloc[-1]) else current_price
        
        # Cloud boundaries
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)
        
        # Price vs Cloud
        if current_price > cloud_top:
            price_vs_cloud = "ABOVE"
        elif current_price < cloud_bottom:
            price_vs_cloud = "BELOW"
        else:
            price_vs_cloud = "INSIDE"
        
        # Tenkan/Kijun cross
        prev_tenkan = float(tenkan_sen.iloc[-2])
        prev_kijun = float(kijun_sen.iloc[-2])
        
        if prev_tenkan <= prev_kijun and current_tenkan > current_kijun:
            tk_cross = "BULLISH"
        elif prev_tenkan >= prev_kijun and current_tenkan < current_kijun:
            tk_cross = "BEARISH"
        else:
            tk_cross = "NONE"
        
        # Cloud color (future)
        cloud_bullish = current_senkou_a > current_senkou_b
        
        # Flatness detection (Tenkan ~ Kijun)
        tk_diff_percent = abs(current_tenkan - current_kijun) / current_price * 100
        is_flat = tk_diff_percent < 0.5  # < 0.5% difference = flat
        
        return {
            'tenkan_sen': current_tenkan,
            'kijun_sen': current_kijun,
            'senkou_a': current_senkou_a,
            'senkou_b': current_senkou_b,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'price_vs_cloud': price_vs_cloud,
            'tk_cross': tk_cross,
            'cloud_bullish': cloud_bullish,
            'is_flat': is_flat,
            'tk_diff_percent': tk_diff_percent,
        }
    
    @staticmethod
    def calculate_bb_width(df: pd.DataFrame, length: int = 20, mult: float = 2.0) -> Tuple[float, float]:
        """
        Розраховує Bollinger Bands Width та його перцентиль
        
        Returns:
            (current_width, percentile)
        """
        if df is None or len(df) < length + 100:
            return None, None
        
        close = df['close']
        
        # BB
        sma = close.rolling(window=length).mean()
        std = close.rolling(window=length).std()
        
        upper = sma + (std * mult)
        lower = sma - (std * mult)
        
        # Width = (Upper - Lower) / Middle
        bb_width = (upper - lower) / sma * 100
        
        current_width = float(bb_width.iloc[-1])
        
        # Percentile over last 100 bars
        historical_width = bb_width.iloc[-100:]
        percentile = (historical_width < current_width).sum() / len(historical_width) * 100
        
        return current_width, percentile
    
    @staticmethod
    def calculate_atr_ratio(df: pd.DataFrame, period: int = 14, lookback: int = 50) -> float:
        """
        Розраховує відношення поточного ATR до середнього
        
        Returns:
            ATR ratio (< 1 = contraction, > 1 = expansion)
        """
        if df is None or len(df) < period + lookback:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean()
        
        current_atr = float(atr.iloc[-1])
        avg_atr = float(atr.iloc[-lookback:].mean())
        
        return current_atr / avg_atr if avg_atr > 0 else 1.0
    
    @staticmethod
    def calculate_cvd(df: pd.DataFrame) -> Tuple[float, float]:
        """
        Розраховує Cumulative Volume Delta (спрощена версія)
        
        На основі candle body direction:
        - Green candle = Buy volume
        - Red candle = Sell volume
        
        Returns:
            (cvd_change_percent, price_change_percent) for divergence detection
        """
        if df is None or len(df) < 20:
            return None, None
        
        close = df['close']
        open_price = df['open']
        volume = df['volume']
        
        # Volume Delta per candle
        delta = volume.copy()
        bullish = close > open_price
        delta[~bullish] = -delta[~bullish]
        
        # CVD
        cvd = delta.cumsum()
        
        # Changes over last 20 bars
        cvd_change = (cvd.iloc[-1] - cvd.iloc[-20]) / abs(cvd.iloc[-20]) * 100 if cvd.iloc[-20] != 0 else 0
        price_change = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] * 100
        
        return float(cvd_change), float(price_change)
    
    @staticmethod
    def calculate_price_change(df: pd.DataFrame, bars: int) -> float:
        """Розраховує зміну ціни за N барів"""
        if df is None or len(df) < bars:
            return None
        
        close = df['close']
        return ((close.iloc[-1] - close.iloc[-bars]) / close.iloc[-bars]) * 100


# ============================================================================
#                              BYBIT API HELPERS
# ============================================================================

class BybitDataFetcher:
    """Отримання даних з Bybit API"""
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
    
    def get_open_interest(self, symbol: str, interval: str = "1h", limit: int = 24) -> Optional[Dict]:
        """
        Отримує Open Interest історію
        
        Returns:
            Dict з поточним OI та зміною
        """
        if not self.bot:
            return None
        
        try:
            result = self.bot.session.get_open_interest(
                category="linear",
                symbol=symbol,
                intervalTime=interval,
                limit=limit
            )
            
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if len(data) >= 2:
                    current_oi = float(data[0].get('openInterest', 0))
                    old_oi = float(data[-1].get('openInterest', 0))
                    
                    change_percent = ((current_oi - old_oi) / old_oi * 100) if old_oi > 0 else 0
                    
                    return {
                        'current': current_oi,
                        'change_percent': change_percent,
                        'history': data
                    }
        except Exception as e:
            logger.debug(f"OI fetch error for {symbol}: {e}")
        
        return None
    
    def get_funding_rate(self, symbol: str) -> Optional[Dict]:
        """
        Отримує поточний Funding Rate
        
        Returns:
            Dict з funding rate та історією
        """
        if not self.bot:
            return None
        
        try:
            result = self.bot.session.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                limit=10
            )
            
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if data:
                    current_funding = float(data[0].get('fundingRate', 0))
                    
                    # Average funding
                    avg_funding = sum(float(d.get('fundingRate', 0)) for d in data) / len(data)
                    
                    return {
                        'current': current_funding,
                        'average': avg_funding,
                        'is_extreme_positive': current_funding > 0.0003,  # 0.03%
                        'is_extreme_negative': current_funding < -0.0003,
                    }
        except Exception as e:
            logger.debug(f"Funding fetch error for {symbol}: {e}")
        
        return None
    
    def get_ticker_info(self, symbol: str) -> Optional[Dict]:
        """Отримує інформацію про тікер"""
        if not self.bot:
            return None
        
        try:
            result = self.bot.session.get_tickers(
                category="linear",
                symbol=symbol
            )
            
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if data:
                    ticker = data[0]
                    return {
                        'price': float(ticker.get('lastPrice', 0)),
                        'volume_24h': float(ticker.get('turnover24h', 0)),
                        'price_change_24h': float(ticker.get('price24hPcnt', 0)) * 100,
                        'high_24h': float(ticker.get('highPrice24h', 0)),
                        'low_24h': float(ticker.get('lowPrice24h', 0)),
                        'bid': float(ticker.get('bid1Price', 0)),
                        'ask': float(ticker.get('ask1Price', 0)),
                    }
        except Exception as e:
            logger.debug(f"Ticker fetch error for {symbol}: {e}")
        
        return None
    
    def get_klines(self, symbol: str, interval: str = "15", limit: int = 200) -> Optional[pd.DataFrame]:
        """Отримує свічки"""
        if not self.bot or not HAS_PANDAS:
            return None
        
        try:
            result = self.bot.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if data:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    df = df.astype({
                        'open': float, 'high': float, 'low': float,
                        'close': float, 'volume': float
                    })
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    return df
        except Exception as e:
            logger.debug(f"Klines fetch error for {symbol}: {e}")
        
        return None
    
    def get_usdt_perpetuals(self, min_volume: float = 10_000_000) -> List[str]:
        """Отримує список USDT Perpetual пар"""
        if not self.bot:
            return []
        
        try:
            result = self.bot.session.get_tickers(category="linear")
            
            if result and result.get('retCode') == 0:
                tickers = result.get('result', {}).get('list', [])
                
                symbols = []
                for t in tickers:
                    symbol = t.get('symbol', '')
                    if not symbol.endswith('USDT'):
                        continue
                    
                    volume = float(t.get('turnover24h', 0))
                    if volume >= min_volume:
                        symbols.append(symbol)
                
                return sorted(symbols, key=lambda x: x)
        except Exception as e:
            logger.error(f"Get perpetuals error: {e}")
        
        return []


# ============================================================================
#                              SCORING ENGINE
# ============================================================================

@dataclass
class SpringScore:
    """Детальний скор для сигналу"""
    oi: int = 0
    funding: int = 0
    cvd: int = 0
    volatility: int = 0
    ichimoku: int = 0
    
    # Raw values
    oi_change: float = 0
    funding_rate: float = 0
    cvd_divergence: float = 0
    bb_width_percentile: float = 0
    atr_ratio: float = 0
    price_change: float = 0
    
    # Ichimoku details
    price_vs_cloud: str = "UNKNOWN"
    tk_cross: str = "NONE"
    is_flat: bool = False
    
    @property
    def total(self) -> int:
        return self.oi + self.funding + self.cvd + self.volatility + self.ichimoku
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SpringScoringEngine:
    """Двигун скорингу для Coiling Spring"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators = IndicatorEngine()
    
    def update_config(self, config: Dict):
        self.config = config
    
    def _get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def calculate_score(
        self,
        df: pd.DataFrame,
        oi_data: Optional[Dict],
        funding_data: Optional[Dict],
        ticker_data: Optional[Dict]
    ) -> Tuple[SpringScore, TrendBias]:
        """
        Розраховує загальний скор та ухил тренду
        
        Returns:
            (SpringScore, TrendBias)
        """
        score = SpringScore()
        bullish_signals = 0
        bearish_signals = 0
        
        if df is None or len(df) < 100:
            return score, TrendBias.NEUTRAL
        
        # ========== 1. OPEN INTEREST SCORE ==========
        if self._get('css_use_oi', True) and oi_data:
            oi_change = oi_data.get('change_percent', 0)
            score.oi_change = oi_change
            
            threshold = self._get('css_oi_change_threshold', 5.0)
            weight = self._get('css_oi_weight', 25)
            
            # OI росте при флеті = накопичення
            price_change = self.indicators.calculate_price_change(df, 12)  # ~1 hour for 15m
            
            if oi_change >= threshold and abs(price_change or 0) < self._get('css_price_change_max', 1.0):
                # Strong signal: OI up, price flat
                intensity = min(1.0, oi_change / (threshold * 2))
                score.oi = int(weight * (0.7 + 0.3 * intensity))
            elif oi_change >= threshold * 0.5:
                score.oi = int(weight * 0.4)
        
        # ========== 2. FUNDING RATE SCORE ==========
        if self._get('css_use_funding', True) and funding_data:
            funding = funding_data.get('current', 0)
            score.funding_rate = funding * 100  # Convert to percentage
            
            weight = self._get('css_funding_weight', 15)
            extreme_pos = self._get('css_funding_extreme_pos', 0.03) / 100
            extreme_neg = self._get('css_funding_extreme_neg', -0.03) / 100
            
            if funding >= extreme_pos:
                # Екстремально високий funding = багато лонгів = Long Squeeze ймовірний
                score.funding = weight
                bearish_signals += 1
            elif funding <= extreme_neg:
                # Екстремально низький funding = багато шортів = Short Squeeze ймовірний
                score.funding = weight
                bullish_signals += 1
            elif abs(funding) > abs(extreme_pos) * 0.5:
                score.funding = int(weight * 0.5)
        
        # ========== 3. CVD DIVERGENCE SCORE ==========
        if self._get('css_use_cvd', True):
            cvd_change, price_change = self.indicators.calculate_cvd(df)
            
            if cvd_change is not None and price_change is not None:
                score.cvd_divergence = cvd_change - price_change
                
                weight = self._get('css_cvd_weight', 20)
                
                # Bullish divergence: price down/flat, CVD up
                if price_change < 0.5 and cvd_change > 2:
                    score.cvd = weight
                    bullish_signals += 1
                # Bearish divergence: price up/flat, CVD down
                elif price_change > -0.5 and cvd_change < -2:
                    score.cvd = weight
                    bearish_signals += 1
                elif abs(cvd_change - price_change) > 3:
                    score.cvd = int(weight * 0.5)
        
        # ========== 4. VOLATILITY CONTRACTION SCORE ==========
        if self._get('css_use_volatility', True):
            bb_width, bb_percentile = self.indicators.calculate_bb_width(df)
            atr_ratio = self.indicators.calculate_atr_ratio(df)
            
            if bb_percentile is not None:
                score.bb_width_percentile = bb_percentile
            if atr_ratio is not None:
                score.atr_ratio = atr_ratio
            
            weight = self._get('css_volatility_weight', 15)
            target_percentile = self._get('css_bb_width_percentile', 20)
            target_atr = self._get('css_atr_contraction', 0.7)
            
            vol_score = 0
            if bb_percentile is not None and bb_percentile <= target_percentile:
                vol_score += 0.5
            if atr_ratio is not None and atr_ratio <= target_atr:
                vol_score += 0.5
            
            score.volatility = int(weight * vol_score)
        
        # ========== 5. ICHIMOKU CLOUD SCORE ==========
        if self._get('css_use_ichimoku', True):
            ichimoku = self.indicators.calculate_ichimoku(
                df,
                tenkan=self._get('css_ichimoku_tenkan', 9),
                kijun=self._get('css_ichimoku_kijun', 26),
                senkou_b=self._get('css_ichimoku_senkou_b', 52)
            )
            
            if ichimoku:
                score.price_vs_cloud = ichimoku['price_vs_cloud']
                score.tk_cross = ichimoku['tk_cross']
                score.is_flat = ichimoku['is_flat']
                
                weight = self._get('css_ichimoku_weight', 15)
                
                # Inside cloud = consolidation zone
                if ichimoku['price_vs_cloud'] == "INSIDE":
                    score.ichimoku = int(weight * 0.7)
                    
                    # TK cross inside cloud = potential breakout direction
                    if ichimoku['tk_cross'] == "BULLISH":
                        bullish_signals += 1
                    elif ichimoku['tk_cross'] == "BEARISH":
                        bearish_signals += 1
                
                # Flat TK = strong consolidation
                if ichimoku['is_flat']:
                    score.ichimoku = weight
                
                # Price at cloud edge = ready for breakout
                if ichimoku['price_vs_cloud'] == "ABOVE" and ichimoku['tk_cross'] == "BULLISH":
                    bullish_signals += 1
                elif ichimoku['price_vs_cloud'] == "BELOW" and ichimoku['tk_cross'] == "BEARISH":
                    bearish_signals += 1
        
        # ========== CALCULATE PRICE CHANGE ==========
        price_change = self.indicators.calculate_price_change(df, 16)  # ~4 hours for 15m
        if price_change is not None:
            score.price_change = price_change
        
        # ========== DETERMINE TREND BIAS ==========
        if bullish_signals > bearish_signals:
            trend_bias = TrendBias.BULLISH
        elif bearish_signals > bullish_signals:
            trend_bias = TrendBias.BEARISH
        else:
            trend_bias = TrendBias.NEUTRAL
        
        return score, trend_bias


# ============================================================================
#                              MAIN SCANNER CLASS
# ============================================================================

class CoilingSpringScanner:
    """
    🎯 Coiling Spring Scanner - Головний клас
    
    Сканує ринок на предмет монет у стані "стиснутої пружини":
    - Низька волатильність
    - Зростаючий Open Interest
    - Аномалії Funding Rate
    - CVD дивергенції
    - Ichimoku консолідація
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.is_scanning = False
        self.progress = 0
        self.status = "Ready"
        self.last_scan_time = None
        self.scan_results: List[Dict] = []
        
        # Auto mode
        self.auto_running = False
        self._auto_thread = None
        
        # Components
        self.data_fetcher = BybitDataFetcher(bot_instance)
        self.scoring_engine = SpringScoringEngine(self._load_config())
        
        # Ensure table
        self._ensure_table()
        
        self._initialized = True
        logger.info("🌀 Coiling Spring Scanner v1.0 initialized")
    
    def _ensure_table(self):
        """Створює таблицю в БД"""
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            return
        
        try:
            CoilingSpringSignal.__table__.create(db_manager.engine, checkfirst=True)
            logger.info("✅ Coiling Spring signals table ready")
        except Exception as e:
            logger.debug(f"Table creation: {e}")
    
    def _load_config(self) -> Dict:
        """Завантажує конфігурацію"""
        config = DEFAULT_CONFIG.copy()
        
        if settings:
            saved = settings.get_all()
            for key in config.keys():
                if key in saved:
                    config[key] = saved[key]
        
        return config
    
    def get_config(self) -> Dict:
        """Отримує поточну конфігурацію"""
        return self._load_config()
    
    def save_config(self, data: Dict):
        """Зберігає конфігурацію"""
        if settings:
            settings.save_settings(data)
            self.scoring_engine.update_config(self._load_config())
    
    def start_scan(self, custom_config: Optional[Dict] = None):
        """Запускає сканування"""
        if self.is_scanning:
            return {'status': 'already_running'}
        
        self.is_scanning = True
        self.progress = 0
        self.status = "Starting..."
        
        thread = threading.Thread(target=self._scan_thread, args=(custom_config,), daemon=True)
        thread.start()
        
        return {'status': 'started'}
    
    def _scan_thread(self, custom_config: Optional[Dict] = None):
        """Потік сканування"""
        try:
            config = custom_config or self._load_config()
            self.scoring_engine.update_config(config)
            
            min_volume = config.get('css_min_volume_24h', 10_000_000)
            min_score = config.get('css_min_score', 60)
            main_tf = config.get('css_main_tf', '15')
            scan_limit = config.get('css_scan_limit', 100)
            
            # Get symbols
            self.status = "Fetching symbols..."
            symbols = self.data_fetcher.get_usdt_perpetuals(min_volume)[:scan_limit]
            
            if not symbols:
                self.status = "No symbols found"
                self.is_scanning = False
                return
            
            total = len(symbols)
            results = []
            
            for i, symbol in enumerate(symbols):
                try:
                    self.progress = int((i + 1) / total * 100)
                    self.status = f"Scanning {symbol} ({i+1}/{total})"
                    
                    # Fetch data
                    df = self.data_fetcher.get_klines(symbol, main_tf, 200)
                    if df is None or len(df) < 100:
                        continue
                    
                    oi_data = self.data_fetcher.get_open_interest(symbol)
                    funding_data = self.data_fetcher.get_funding_rate(symbol)
                    ticker_data = self.data_fetcher.get_ticker_info(symbol)
                    
                    # Calculate score
                    score, trend_bias = self.scoring_engine.calculate_score(
                        df, oi_data, funding_data, ticker_data
                    )
                    
                    if score.total >= min_score:
                        # Determine signal type
                        signal_type = self._determine_signal_type(score, trend_bias)
                        
                        result = {
                            'symbol': symbol,
                            'signal_type': signal_type.value,
                            'trend_bias': trend_bias.value,
                            'total_score': score.total,
                            'scores': score.to_dict(),
                            'ticker': ticker_data,
                            'timestamp': datetime.utcnow().isoformat(),
                        }
                        
                        results.append(result)
                        
                        # Save to DB
                        self._save_signal(result, score, ticker_data)
                        
                        logger.info(f"🌀 {symbol}: {signal_type.value} | Score: {score.total} | Bias: {trend_bias.value}")
                    
                    # Rate limiting
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.debug(f"Scan error for {symbol}: {e}")
            
            # Sort by score
            results.sort(key=lambda x: x['total_score'], reverse=True)
            self.scan_results = results
            
            self.status = f"Complete: {len(results)} signals found"
            self.last_scan_time = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Scan thread error: {e}")
            self.status = f"Error: {e}"
        finally:
            self.is_scanning = False
            self.progress = 100
    
    def _determine_signal_type(self, score: SpringScore, trend_bias: TrendBias) -> SpringSignal:
        """Визначає тип сигналу"""
        # High volatility contraction + high OI = ready for breakout
        if score.volatility >= 10 and score.oi >= 15:
            if trend_bias == TrendBias.BULLISH:
                return SpringSignal.READY_PUMP
            elif trend_bias == TrendBias.BEARISH:
                return SpringSignal.READY_DUMP
        
        # Inside cloud with TK cross = potential breakout
        if score.price_vs_cloud == "INSIDE" and score.tk_cross != "NONE":
            if score.tk_cross == "BULLISH":
                return SpringSignal.READY_PUMP
            else:
                return SpringSignal.READY_DUMP
        
        # Default: still coiling
        return SpringSignal.COILING
    
    def _save_signal(self, result: Dict, score: SpringScore, ticker: Optional[Dict]):
        """Зберігає сигнал в БД"""
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            return
        
        session = None
        try:
            session = db_manager.get_session()
            
            signal = CoilingSpringSignal(
                symbol=result['symbol'],
                signal_type=result['signal_type'],
                trend_bias=result['trend_bias'],
                total_score=score.total,
                oi_score=score.oi,
                funding_score=score.funding,
                cvd_score=score.cvd,
                volatility_score=score.volatility,
                ichimoku_score=score.ichimoku,
                oi_change_percent=score.oi_change,
                funding_rate=score.funding_rate,
                cvd_divergence=score.cvd_divergence,
                bb_width_percentile=score.bb_width_percentile,
                atr_ratio=score.atr_ratio,
                price_change_percent=score.price_change,
                price_vs_cloud=score.price_vs_cloud,
                tenkan_kijun_cross=score.tk_cross,
                current_price=ticker.get('price') if ticker else None,
                volume_24h=ticker.get('volume_24h') if ticker else None,
            )
            
            session.add(signal)
            session.commit()
            
        except Exception as e:
            logger.error(f"Save signal error: {e}")
            if session:
                session.rollback()
        finally:
            if session:
                session.close()
    
    def get_signals(self, limit: int = 50) -> List[Dict]:
        """Отримує історію сигналів"""
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            return self.scan_results[:limit]
        
        session = None
        try:
            session = db_manager.get_session()
            signals = session.query(CoilingSpringSignal).order_by(
                CoilingSpringSignal.created_at.desc()
            ).limit(limit).all()
            
            return [{
                'id': s.id,
                'symbol': s.symbol,
                'signal_type': s.signal_type,
                'trend_bias': s.trend_bias,
                'total_score': s.total_score,
                'oi_score': s.oi_score,
                'funding_score': s.funding_score,
                'cvd_score': s.cvd_score,
                'volatility_score': s.volatility_score,
                'ichimoku_score': s.ichimoku_score,
                'oi_change': s.oi_change_percent,
                'funding_rate': s.funding_rate,
                'price_vs_cloud': s.price_vs_cloud,
                'current_price': s.current_price,
                'volume_24h': s.volume_24h,
                'time': s.created_at.strftime('%d.%m %H:%M') if s.created_at else '',
            } for s in signals]
            
        except Exception as e:
            logger.error(f"Get signals error: {e}")
            return []
        finally:
            if session:
                session.close()
    
    def get_status(self) -> Dict:
        """Отримує статус сканера"""
        return {
            'is_scanning': self.is_scanning,
            'progress': self.progress,
            'status': self.status,
            'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'results_count': len(self.scan_results),
            'auto_running': self.auto_running,
        }
    
    def start_auto_mode(self, interval: int = 300):
        """Запускає автоматичний режим"""
        if self.auto_running:
            return
        
        self.auto_running = True
        self._auto_thread = threading.Thread(target=self._auto_loop, args=(interval,), daemon=True)
        self._auto_thread.start()
        logger.info(f"🔄 Auto mode started ({interval}s interval)")
    
    def stop_auto_mode(self):
        """Зупиняє автоматичний режим"""
        self.auto_running = False
        logger.info("⏹️ Auto mode stopped")
    
    def _auto_loop(self, interval: int):
        """Цикл автоматичного сканування"""
        while self.auto_running:
            if not self.is_scanning:
                self.start_scan()
            
            # Wait for scan to complete
            while self.is_scanning and self.auto_running:
                time.sleep(1)
            
            # Wait interval
            for _ in range(interval):
                if not self.auto_running:
                    break
                time.sleep(1)
    
    def clear_history(self):
        """Очищає історію сигналів"""
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            self.scan_results = []
            return
        
        session = None
        try:
            session = db_manager.get_session()
            session.query(CoilingSpringSignal).delete()
            session.commit()
            self.scan_results = []
            logger.info("🗑️ Signal history cleared")
        except Exception as e:
            logger.error(f"Clear history error: {e}")
        finally:
            if session:
                session.close()


# ============================================================================
#                              FLASK ROUTES
# ============================================================================

def get_coiling_spring_scanner() -> CoilingSpringScanner:
    """Отримує singleton instance"""
    return CoilingSpringScanner()


def register_coiling_spring_routes(app):
    """Реєструє Flask routes"""
    
    @app.route('/coiling_spring')
    def coiling_spring_page():
        scanner = get_coiling_spring_scanner()
        return render_template(
            'coiling_spring.html',
            config=scanner.get_config(),
            status=scanner.get_status(),
            signals=scanner.get_signals(50),
            param_help=PARAM_HELP,
        )
    
    @app.route('/coiling_spring/scan', methods=['POST'])
    def coiling_spring_scan():
        scanner = get_coiling_spring_scanner()
        result = scanner.start_scan()
        return jsonify(result)
    
    @app.route('/coiling_spring/status')
    def coiling_spring_status():
        scanner = get_coiling_spring_scanner()
        return jsonify(scanner.get_status())
    
    @app.route('/coiling_spring/signals')
    def coiling_spring_signals():
        scanner = get_coiling_spring_scanner()
        limit = request.args.get('limit', 50, type=int)
        return jsonify(scanner.get_signals(limit))
    
    @app.route('/coiling_spring/config', methods=['GET', 'POST'])
    def coiling_spring_config():
        scanner = get_coiling_spring_scanner()
        if request.method == 'POST':
            data = request.json or {}
            scanner.save_config(data)
            return jsonify({'status': 'ok'})
        return jsonify(scanner.get_config())
    
    @app.route('/coiling_spring/auto', methods=['POST'])
    def coiling_spring_auto():
        scanner = get_coiling_spring_scanner()
        data = request.json or {}
        enabled = data.get('enabled', False)
        interval = int(data.get('interval', 300))
        
        if enabled:
            scanner.start_auto_mode(interval)
        else:
            scanner.stop_auto_mode()
        
        return jsonify({'status': 'ok', 'auto_running': scanner.auto_running})
    
    @app.route('/coiling_spring/clear', methods=['POST'])
    def coiling_spring_clear():
        scanner = get_coiling_spring_scanner()
        scanner.clear_history()
        return jsonify({'status': 'cleared'})
    
    logger.info("✅ Coiling Spring Scanner routes registered")


# ============================================================================
#                              ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Test mode
    print("🌀 Coiling Spring Scanner v1.0")
    print("=" * 50)
    
    scanner = CoilingSpringScanner()
    print(f"Config: {scanner.get_config()}")
    print(f"Status: {scanner.get_status()}")
