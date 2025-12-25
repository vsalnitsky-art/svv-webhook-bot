#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     COILING SPRING SCANNER v2.0                              ║
║                                                                              ║
║  Стратегія "Стиснута Пружина" - пошук монет у консолідації                   ║
║  перед вибуховим рухом (памп/дамп)                                           ║
║                                                                              ║
║  Ключові метрики:                                                            ║
║  • Open Interest (OI) - "паливо" для руху                                    ║
║  • CVD (Cumulative Volume Delta) - приховане накопичення                     ║
║  • Funding Rate - Short/Long Squeeze сигнали                                 ║
║  • Long/Short Ratio - позиціонування натовпу                                 ║
║  • Ichimoku Cloud - зона консолідації                                        ║
║  • Volatility Contraction - стиснення перед вибухом                          ║
║                                                                              ║
║  v2.0: Розширені дані (500+ свічок), Long/Short Ratio, Live Updates          ║
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
from collections import deque

from flask import Blueprint, render_template, jsonify, request

try:
    from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
    from sqlalchemy.ext.declarative import declarative_base
    from database_manager import DatabaseManager
    Base = declarative_base()
    HAS_DB = True
except ImportError:
    HAS_DB = False
    Base = object

try:
    from main_app import bot_instance, db_manager
    HAS_BOT = True
except ImportError:
    bot_instance = None
    db_manager = None
    HAS_BOT = False

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    pd = None
    np = None
    HAS_PANDAS = False

try:
    from settings_manager import settings
except ImportError:
    settings = None

logger = logging.getLogger(__name__)

# ============================================================================
#                              ENUMS & CONSTANTS
# ============================================================================

class SpringSignal(Enum):
    COILING = "COILING"
    READY_PUMP = "READY_PUMP"
    READY_DUMP = "READY_DUMP"
    BREAKOUT_UP = "BREAKOUT_UP"
    BREAKOUT_DOWN = "BREAKOUT_DOWN"

class TrendBias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

BYBIT_KLINE_LIMIT = 1000
BYBIT_OI_LIMIT = 200
BYBIT_FUNDING_LIMIT = 200

# ============================================================================
#                              DEFAULT CONFIG
# ============================================================================

DEFAULT_CONFIG = {
    'css_enabled': True,
    'css_scan_interval': 300,
    'css_min_volume_24h': 10_000_000,
    'css_max_spread_percent': 0.3,
    'css_scan_limit': 100,
    'css_kline_limit': 500,
    'css_oi_history_hours': 24,
    'css_funding_history': 48,
    'css_use_oi': True,
    'css_oi_change_threshold': 5.0,
    'css_oi_lookback_hours': 1,
    'css_oi_weight': 25,
    'css_use_funding': True,
    'css_funding_extreme_pos': 0.03,
    'css_funding_extreme_neg': -0.03,
    'css_funding_weight': 15,
    'css_use_ls_ratio': True,
    'css_ls_extreme_long': 2.0,
    'css_ls_extreme_short': 0.5,
    'css_ls_weight': 10,
    'css_use_cvd': True,
    'css_cvd_divergence_threshold': 0.5,
    'css_cvd_lookback': 20,
    'css_cvd_weight': 20,
    'css_use_volatility': True,
    'css_bb_width_percentile': 20,
    'css_atr_contraction': 0.7,
    'css_volatility_lookback': 100,
    'css_volatility_weight': 15,
    'css_use_ichimoku': True,
    'css_ichimoku_tenkan': 9,
    'css_ichimoku_kijun': 26,
    'css_ichimoku_senkou_b': 52,
    'css_ichimoku_weight': 15,
    'css_price_change_max': 1.0,
    'css_price_lookback_hours': 4,
    'css_min_score': 60,
    'css_strong_signal_score': 80,
    'css_main_tf': '15',
    'css_htf': '60',
    'css_paper_trading': True,
    'css_telegram_alerts': True,
    'css_auto_mode': False,
}

PARAM_HELP = {
    'css_kline_limit': 'Кількість свічок для аналізу (50-1000)',
    'css_oi_change_threshold': 'Open Interest має зрости на X% при флеті',
    'css_funding_extreme_pos': 'Екстремально високий Funding = Long Squeeze',
    'css_ls_extreme_long': 'L/S Ratio > X = занадто багато лонгів',
    'css_bb_width_percentile': 'BB Width у нижніх X% = стиснення',
}

# ============================================================================
#                              DATABASE MODEL
# ============================================================================

if HAS_DB:
    class CoilingSpringSignal(Base):
        __tablename__ = 'coiling_spring_signals'
        __table_args__ = {'extend_existing': True}
        
        id = Column(Integer, primary_key=True)
        symbol = Column(String(20), index=True)
        signal_type = Column(String(20))
        trend_bias = Column(String(10))
        total_score = Column(Integer)
        oi_score = Column(Integer, default=0)
        funding_score = Column(Integer, default=0)
        ls_ratio_score = Column(Integer, default=0)
        cvd_score = Column(Integer, default=0)
        volatility_score = Column(Integer, default=0)
        ichimoku_score = Column(Integer, default=0)
        oi_change_percent = Column(Float)
        funding_rate = Column(Float)
        long_short_ratio = Column(Float)
        cvd_divergence = Column(Float)
        bb_width_percentile = Column(Float)
        atr_ratio = Column(Float)
        price_change_percent = Column(Float)
        price_vs_cloud = Column(String(20))
        tenkan_kijun_cross = Column(String(10))
        current_price = Column(Float)
        volume_24h = Column(Float)
        open_interest = Column(Float)
        timeframe = Column(String(10))
        candles_analyzed = Column(Integer)
        created_at = Column(DateTime, default=datetime.utcnow)
        notes = Column(Text)
else:
    CoilingSpringSignal = None

# ============================================================================
#                              INDICATOR CALCULATIONS
# ============================================================================

class IndicatorEngine:
    
    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict:
        min_bars = senkou_b + kijun + 10
        if df is None or len(df) < min_bars:
            return None
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        tenkan_high = high.rolling(window=tenkan).max()
        tenkan_low = low.rolling(window=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        kijun_high = high.rolling(window=kijun).max()
        kijun_low = low.rolling(window=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_b_high = high.rolling(window=senkou_b).max()
        senkou_b_low = low.rolling(window=senkou_b).min()
        senkou_span_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun)
        
        current_price = float(close.iloc[-1])
        current_tenkan = float(tenkan_sen.iloc[-1])
        current_kijun = float(kijun_sen.iloc[-1])
        current_senkou_a = float(senkou_a.iloc[-1]) if not pd.isna(senkou_a.iloc[-1]) else current_price
        current_senkou_b = float(senkou_span_b.iloc[-1]) if not pd.isna(senkou_span_b.iloc[-1]) else current_price
        
        cloud_top = max(current_senkou_a, current_senkou_b)
        cloud_bottom = min(current_senkou_a, current_senkou_b)
        
        if current_price > cloud_top:
            price_vs_cloud = "ABOVE"
        elif current_price < cloud_bottom:
            price_vs_cloud = "BELOW"
        else:
            price_vs_cloud = "INSIDE"
        
        tk_cross = "NONE"
        for i in range(1, min(4, len(df))):
            prev_tenkan = float(tenkan_sen.iloc[-i-1])
            prev_kijun = float(kijun_sen.iloc[-i-1])
            curr_tenkan = float(tenkan_sen.iloc[-i])
            curr_kijun = float(kijun_sen.iloc[-i])
            
            if prev_tenkan <= prev_kijun and curr_tenkan > curr_kijun:
                tk_cross = "BULLISH"
                break
            elif prev_tenkan >= prev_kijun and curr_tenkan < curr_kijun:
                tk_cross = "BEARISH"
                break
        
        cloud_bullish = current_senkou_a > current_senkou_b
        tk_diff_percent = abs(current_tenkan - current_kijun) / current_price * 100
        is_flat = tk_diff_percent < 0.3
        
        kumo_twist = False
        if len(senkou_a) > 5:
            for i in range(1, 5):
                if not pd.isna(senkou_a.iloc[-i]) and not pd.isna(senkou_span_b.iloc[-i]):
                    if not pd.isna(senkou_a.iloc[-i-1]) and not pd.isna(senkou_span_b.iloc[-i-1]):
                        prev_bullish = senkou_a.iloc[-i-1] > senkou_span_b.iloc[-i-1]
                        curr_bullish = senkou_a.iloc[-i] > senkou_span_b.iloc[-i]
                        if prev_bullish != curr_bullish:
                            kumo_twist = True
                            break
        
        return {
            'tenkan_sen': current_tenkan, 'kijun_sen': current_kijun,
            'senkou_a': current_senkou_a, 'senkou_b': current_senkou_b,
            'cloud_top': cloud_top, 'cloud_bottom': cloud_bottom,
            'price_vs_cloud': price_vs_cloud, 'tk_cross': tk_cross,
            'cloud_bullish': cloud_bullish, 'is_flat': is_flat,
            'tk_diff_percent': tk_diff_percent, 'kumo_twist': kumo_twist,
        }
    
    @staticmethod
    def calculate_bb_width(df: pd.DataFrame, length: int = 20, mult: float = 2.0, lookback: int = 100) -> Tuple[float, float, float]:
        min_bars = max(length, lookback) + 10
        if df is None or len(df) < min_bars:
            return None, None, None
        
        close = df['close']
        sma = close.rolling(window=length).mean()
        std = close.rolling(window=length).std()
        upper = sma + (std * mult)
        lower = sma - (std * mult)
        bb_width = (upper - lower) / sma * 100
        
        current_width = float(bb_width.iloc[-1])
        historical_width = bb_width.iloc[-lookback:].dropna()
        if len(historical_width) < 20:
            return current_width, 50.0, 0.5
        
        percentile = (historical_width < current_width).sum() / len(historical_width) * 100
        min_width = historical_width.min()
        max_width = historical_width.max()
        squeeze_intensity = 1 - (current_width - min_width) / (max_width - min_width) if max_width > min_width else 0.5
        
        return current_width, percentile, squeeze_intensity
    
    @staticmethod
    def calculate_atr_ratio(df: pd.DataFrame, period: int = 14, lookback: int = 50) -> Tuple[float, float]:
        min_bars = period + lookback + 10
        if df is None or len(df) < min_bars:
            return None, None
        
        high, low, close = df['high'], df['low'], df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        current_atr = float(atr.iloc[-1])
        avg_atr = float(atr.iloc[-lookback:].mean())
        ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        
        return ratio, current_atr
    
    @staticmethod
    def calculate_cvd(df: pd.DataFrame, lookback: int = 20) -> Tuple[float, float, float]:
        min_bars = lookback + 10
        if df is None or len(df) < min_bars:
            return None, None, None
        
        close, open_price = df['close'], df['open']
        high, low, volume = df['high'], df['low'], df['volume']
        
        body = close - open_price
        range_hl = high - low
        
        delta = volume.copy()
        for i in range(len(df)):
            if range_hl.iloc[i] > 0:
                body_pct = abs(body.iloc[i]) / range_hl.iloc[i]
                if body.iloc[i] > 0:
                    delta.iloc[i] = volume.iloc[i] * (0.5 + body_pct * 0.5)
                else:
                    delta.iloc[i] = -volume.iloc[i] * (0.5 + body_pct * 0.5)
        
        cvd = delta.cumsum()
        cvd_change = (cvd.iloc[-1] - cvd.iloc[-lookback]) / abs(cvd.iloc[-lookback]) * 100 if cvd.iloc[-lookback] != 0 else 0
        price_change = (close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback] * 100
        
        if price_change < 0.5 and cvd_change > 2:
            divergence_score = min(100, abs(cvd_change - price_change))
        elif price_change > -0.5 and cvd_change < -2:
            divergence_score = -min(100, abs(cvd_change - price_change))
        else:
            divergence_score = 0
        
        return float(cvd_change), float(price_change), float(divergence_score)
    
    @staticmethod
    def calculate_price_change(df: pd.DataFrame, bars: int) -> float:
        if df is None or len(df) < bars + 1:
            return None
        close = df['close']
        return ((close.iloc[-1] - close.iloc[-bars]) / close.iloc[-bars]) * 100

# ============================================================================
#                              BYBIT API HELPERS
# ============================================================================

class BybitDataFetcher:
    
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self._cache = {}
        self._cache_ttl = 30
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache:
            return False
        timestamp, _ = self._cache[key]
        return (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl
    
    def get_open_interest(self, symbol: str, interval: str = "1h", limit: int = 48) -> Optional[Dict]:
        if not self.bot:
            return None
        
        cache_key = f"oi_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        
        try:
            result = self.bot.session.get_open_interest(
                category="linear", symbol=symbol, intervalTime=interval,
                limit=min(limit, BYBIT_OI_LIMIT)
            )
            
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if len(data) >= 2:
                    current_oi = float(data[0].get('openInterest', 0))
                    changes = {}
                    for period_name, hours in {'1h': 1, '4h': 4, '24h': 24}.items():
                        idx = min(hours, len(data) - 1)
                        if idx > 0:
                            old_oi = float(data[idx].get('openInterest', current_oi))
                            changes[period_name] = ((current_oi - old_oi) / old_oi * 100) if old_oi > 0 else 0
                    
                    oi_values = [float(d.get('openInterest', 0)) for d in data[:24]]
                    oi_trend = "INCREASING" if oi_values[0] > oi_values[-1] * 1.02 else \
                               "DECREASING" if oi_values[0] < oi_values[-1] * 0.98 else "STABLE"
                    
                    result_data = {
                        'current': current_oi, 'change_1h': changes.get('1h', 0),
                        'change_4h': changes.get('4h', 0), 'change_24h': changes.get('24h', 0),
                        'trend': oi_trend, 'history': data[:24],
                    }
                    self._cache[cache_key] = (datetime.utcnow(), result_data)
                    return result_data
        except Exception as e:
            logger.debug(f"OI fetch error for {symbol}: {e}")
        return None
    
    def get_funding_rate(self, symbol: str, limit: int = 48) -> Optional[Dict]:
        if not self.bot:
            return None
        
        cache_key = f"funding_{symbol}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        
        try:
            result = self.bot.session.get_funding_rate_history(
                category="linear", symbol=symbol, limit=min(limit, BYBIT_FUNDING_LIMIT)
            )
            
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if data:
                    current_funding = float(data[0].get('fundingRate', 0))
                    funding_values = [float(d.get('fundingRate', 0)) for d in data]
                    avg_funding = sum(funding_values) / len(funding_values)
                    
                    recent_avg = sum(funding_values[:8]) / min(8, len(funding_values))
                    old_avg = sum(funding_values[8:24]) / max(1, min(16, len(funding_values) - 8))
                    funding_trend = "INCREASING" if recent_avg > old_avg * 1.2 else \
                                   "DECREASING" if recent_avg < old_avg * 0.8 else "STABLE"
                    
                    std_funding = (sum((f - avg_funding) ** 2 for f in funding_values) / len(funding_values)) ** 0.5
                    is_anomaly = abs(current_funding - avg_funding) > 2 * std_funding
                    
                    result_data = {
                        'current': current_funding, 'current_percent': current_funding * 100,
                        'average': avg_funding, 'trend': funding_trend, 'is_anomaly': is_anomaly,
                        'is_extreme_positive': current_funding > 0.0003,
                        'is_extreme_negative': current_funding < -0.0003,
                    }
                    self._cache[cache_key] = (datetime.utcnow(), result_data)
                    return result_data
        except Exception as e:
            logger.debug(f"Funding fetch error for {symbol}: {e}")
        return None
    
    def get_long_short_ratio(self, symbol: str, period: str = "1h") -> Optional[Dict]:
        if not self.bot:
            return None
        
        cache_key = f"lsratio_{symbol}_{period}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key][1]
        
        try:
            result = self.bot.session.get_long_short_ratio(
                category="linear", symbol=symbol, period=period, limit=24
            )
            
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if data:
                    current = data[0]
                    buy_ratio = float(current.get('buyRatio', 0.5))
                    sell_ratio = float(current.get('sellRatio', 0.5))
                    ls_ratio = buy_ratio / sell_ratio if sell_ratio > 0 else 1.0
                    
                    result_data = {
                        'ratio': ls_ratio, 'buy_ratio': buy_ratio, 'sell_ratio': sell_ratio,
                        'is_extreme_long': ls_ratio > 2.0, 'is_extreme_short': ls_ratio < 0.5,
                        'crowd_position': "LONG" if ls_ratio > 1.2 else "SHORT" if ls_ratio < 0.8 else "NEUTRAL",
                    }
                    self._cache[cache_key] = (datetime.utcnow(), result_data)
                    return result_data
        except Exception as e:
            logger.debug(f"L/S Ratio fetch error for {symbol}: {e}")
        return None
    
    def get_ticker_info(self, symbol: str) -> Optional[Dict]:
        if not self.bot:
            return None
        try:
            result = self.bot.session.get_tickers(category="linear", symbol=symbol)
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if data:
                    ticker = data[0]
                    price = float(ticker.get('lastPrice', 0))
                    bid = float(ticker.get('bid1Price', 0))
                    ask = float(ticker.get('ask1Price', 0))
                    spread = ((ask - bid) / price * 100) if price > 0 else 0
                    return {
                        'price': price, 'volume_24h': float(ticker.get('turnover24h', 0)),
                        'price_change_24h': float(ticker.get('price24hPcnt', 0)) * 100,
                        'high_24h': float(ticker.get('highPrice24h', 0)),
                        'low_24h': float(ticker.get('lowPrice24h', 0)),
                        'bid': bid, 'ask': ask, 'spread': spread,
                        'open_interest': float(ticker.get('openInterest', 0)),
                        'funding_rate': float(ticker.get('fundingRate', 0)),
                    }
        except Exception as e:
            logger.debug(f"Ticker fetch error for {symbol}: {e}")
        return None
    
    def get_klines(self, symbol: str, interval: str = "15", limit: int = 500) -> Optional[pd.DataFrame]:
        if not self.bot or not HAS_PANDAS:
            return None
        try:
            actual_limit = min(limit, BYBIT_KLINE_LIMIT)
            result = self.bot.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=actual_limit
            )
            if result and result.get('retCode') == 0:
                data = result.get('result', {}).get('list', [])
                if data:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                    df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'turnover': float})
                    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    return df
        except Exception as e:
            logger.debug(f"Klines fetch error for {symbol}: {e}")
        return None
    
    def get_usdt_perpetuals(self, min_volume: float = 10_000_000) -> List[Dict]:
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
                        price = float(t.get('lastPrice', 0))
                        bid = float(t.get('bid1Price', 0))
                        ask = float(t.get('ask1Price', 0))
                        spread = ((ask - bid) / price * 100) if price > 0 else 999
                        symbols.append({
                            'symbol': symbol, 'volume_24h': volume, 'price': price,
                            'spread': spread, 'price_change': float(t.get('price24hPcnt', 0)) * 100,
                        })
                symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
                return symbols
        except Exception as e:
            logger.error(f"Get perpetuals error: {e}")
        return []

# ============================================================================
#                              SCORING ENGINE
# ============================================================================

@dataclass
class SpringScore:
    oi: int = 0
    funding: int = 0
    ls_ratio: int = 0
    cvd: int = 0
    volatility: int = 0
    ichimoku: int = 0
    
    oi_change: float = 0
    oi_trend: str = "UNKNOWN"
    funding_rate: float = 0
    funding_trend: str = "UNKNOWN"
    long_short_ratio: float = 1.0
    crowd_position: str = "NEUTRAL"
    cvd_change: float = 0
    cvd_divergence: float = 0
    bb_width_percentile: float = 50
    squeeze_intensity: float = 0
    atr_ratio: float = 1.0
    price_change: float = 0
    
    price_vs_cloud: str = "UNKNOWN"
    tk_cross: str = "NONE"
    cloud_bullish: bool = False
    is_flat: bool = False
    kumo_twist: bool = False
    
    candles_analyzed: int = 0
    
    @property
    def total(self) -> int:
        return self.oi + self.funding + self.ls_ratio + self.cvd + self.volatility + self.ichimoku
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SpringScoringEngine:
    
    def __init__(self, config: Dict):
        self.config = config
        self.indicators = IndicatorEngine()
    
    def update_config(self, config: Dict):
        self.config = config
    
    def _get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def calculate_score(self, df: pd.DataFrame, oi_data: Optional[Dict], funding_data: Optional[Dict],
                        ls_ratio_data: Optional[Dict], ticker_data: Optional[Dict]) -> Tuple[SpringScore, TrendBias]:
        score = SpringScore()
        bullish_signals = 0
        bearish_signals = 0
        
        if df is not None:
            score.candles_analyzed = len(df)
        
        if df is None or len(df) < 100:
            return score, TrendBias.NEUTRAL
        
        # 1. OPEN INTEREST
        if self._get('css_use_oi', True) and oi_data:
            oi_change = oi_data.get('change_1h', 0)
            score.oi_change = oi_change
            score.oi_trend = oi_data.get('trend', 'UNKNOWN')
            
            threshold = self._get('css_oi_change_threshold', 5.0)
            weight = self._get('css_oi_weight', 25)
            price_change = self.indicators.calculate_price_change(df, 12)
            score.price_change = price_change or 0
            
            if oi_change >= threshold and abs(price_change or 0) < self._get('css_price_change_max', 1.0):
                intensity = min(1.0, oi_change / (threshold * 2))
                score.oi = int(weight * (0.7 + 0.3 * intensity))
            elif oi_change >= threshold * 0.5:
                score.oi = int(weight * 0.4)
        
        # 2. FUNDING RATE
        if self._get('css_use_funding', True) and funding_data:
            funding = funding_data.get('current', 0)
            score.funding_rate = funding * 100
            score.funding_trend = funding_data.get('trend', 'UNKNOWN')
            
            weight = self._get('css_funding_weight', 15)
            extreme_pos = self._get('css_funding_extreme_pos', 0.03) / 100
            extreme_neg = self._get('css_funding_extreme_neg', -0.03) / 100
            
            if funding >= extreme_pos or funding_data.get('is_anomaly'):
                score.funding = weight
                bearish_signals += 1
            elif funding <= extreme_neg or funding_data.get('is_anomaly'):
                score.funding = weight
                bullish_signals += 1
            elif abs(funding) > abs(extreme_pos) * 0.5:
                score.funding = int(weight * 0.5)
        
        # 3. LONG/SHORT RATIO
        if self._get('css_use_ls_ratio', True) and ls_ratio_data:
            ls_ratio = ls_ratio_data.get('ratio', 1.0)
            score.long_short_ratio = ls_ratio
            score.crowd_position = ls_ratio_data.get('crowd_position', 'NEUTRAL')
            
            weight = self._get('css_ls_weight', 10)
            if ls_ratio >= self._get('css_ls_extreme_long', 2.0):
                score.ls_ratio = weight
                bearish_signals += 1
            elif ls_ratio <= self._get('css_ls_extreme_short', 0.5):
                score.ls_ratio = weight
                bullish_signals += 1
            elif ls_ratio > 1.3 or ls_ratio < 0.7:
                score.ls_ratio = int(weight * 0.5)
        
        # 4. CVD DIVERGENCE
        if self._get('css_use_cvd', True):
            lookback = self._get('css_cvd_lookback', 20)
            cvd_change, price_change, divergence_score = self.indicators.calculate_cvd(df, lookback)
            
            if cvd_change is not None:
                score.cvd_change = cvd_change
                score.cvd_divergence = divergence_score
                weight = self._get('css_cvd_weight', 20)
                
                if divergence_score > 20:
                    score.cvd = weight
                    bullish_signals += 1
                elif divergence_score < -20:
                    score.cvd = weight
                    bearish_signals += 1
                elif abs(divergence_score) > 10:
                    score.cvd = int(weight * 0.5)
        
        # 5. VOLATILITY CONTRACTION
        if self._get('css_use_volatility', True):
            lookback = self._get('css_volatility_lookback', 100)
            bb_width, bb_percentile, squeeze_intensity = self.indicators.calculate_bb_width(df, lookback=lookback)
            atr_ratio, _ = self.indicators.calculate_atr_ratio(df)
            
            if bb_percentile is not None:
                score.bb_width_percentile = bb_percentile
                score.squeeze_intensity = squeeze_intensity or 0
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
        
        # 6. ICHIMOKU CLOUD
        if self._get('css_use_ichimoku', True):
            ichimoku = self.indicators.calculate_ichimoku(
                df, tenkan=self._get('css_ichimoku_tenkan', 9),
                kijun=self._get('css_ichimoku_kijun', 26),
                senkou_b=self._get('css_ichimoku_senkou_b', 52)
            )
            
            if ichimoku:
                score.price_vs_cloud = ichimoku['price_vs_cloud']
                score.tk_cross = ichimoku['tk_cross']
                score.cloud_bullish = ichimoku['cloud_bullish']
                score.is_flat = ichimoku['is_flat']
                score.kumo_twist = ichimoku.get('kumo_twist', False)
                
                weight = self._get('css_ichimoku_weight', 15)
                
                if ichimoku['price_vs_cloud'] == "INSIDE":
                    score.ichimoku = int(weight * 0.8)
                    if ichimoku['tk_cross'] == "BULLISH":
                        bullish_signals += 1
                    elif ichimoku['tk_cross'] == "BEARISH":
                        bearish_signals += 1
                
                if ichimoku['is_flat']:
                    score.ichimoku = weight
                
                if ichimoku['kumo_twist']:
                    score.ichimoku = max(score.ichimoku, int(weight * 0.9))
        
        # DETERMINE TREND BIAS
        if bullish_signals > bearish_signals + 0.5:
            trend_bias = TrendBias.BULLISH
        elif bearish_signals > bullish_signals + 0.5:
            trend_bias = TrendBias.BEARISH
        else:
            trend_bias = TrendBias.NEUTRAL
        
        return score, trend_bias

# ============================================================================
#                              MAIN SCANNER CLASS
# ============================================================================

class CoilingSpringScanner:
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
        self.current_symbol = ""
        self.last_scan_time = None
        self.scan_results: List[Dict] = []
        self.activity_log: deque = deque(maxlen=100)
        
        self.auto_running = False
        self.auto_scan_count = 0
        self._auto_thread = None
        
        self.stats = {
            'total_scans': 0, 'total_signals': 0, 'bullish_signals': 0,
            'bearish_signals': 0, 'last_scan_duration': 0, 'avg_scan_duration': 0,
        }
        
        self.data_fetcher = BybitDataFetcher(bot_instance)
        self.scoring_engine = SpringScoringEngine(self._load_config())
        self._ensure_table()
        
        self._initialized = True
        logger.info("🌀 Coiling Spring Scanner v2.0 initialized")
    
    def _ensure_table(self):
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            return
        try:
            CoilingSpringSignal.__table__.create(db_manager.engine, checkfirst=True)
            from sqlalchemy import inspect, text
            inspector = inspect(db_manager.engine)
            existing = {col['name'] for col in inspector.get_columns('coiling_spring_signals')}
            new_cols = {'ls_ratio_score': 'INTEGER DEFAULT 0', 'long_short_ratio': 'FLOAT', 'candles_analyzed': 'INTEGER'}
            with db_manager.engine.connect() as conn:
                for col, typ in new_cols.items():
                    if col not in existing:
                        try:
                            conn.execute(text(f"ALTER TABLE coiling_spring_signals ADD COLUMN {col} {typ}"))
                        except: pass
                conn.commit()
            logger.info("✅ Coiling Spring signals table ready")
        except Exception as e:
            logger.debug(f"Table creation: {e}")
    
    def _load_config(self) -> Dict:
        config = DEFAULT_CONFIG.copy()
        if settings:
            saved = settings.get_all()
            for key in config.keys():
                if key in saved:
                    config[key] = saved[key]
        return config
    
    def get_config(self) -> Dict:
        return self._load_config()
    
    def save_config(self, data: Dict):
        if settings:
            settings.save_settings(data)
            self.scoring_engine.update_config(self._load_config())
    
    def _log_activity(self, message: str, level: str = "info"):
        self.activity_log.append({'time': datetime.utcnow().strftime('%H:%M:%S'), 'message': message, 'level': level})
    
    def start_scan(self, custom_config: Optional[Dict] = None):
        if self.is_scanning:
            return {'status': 'already_running'}
        self.is_scanning = True
        self.progress = 0
        self.status = "Starting..."
        self.current_symbol = ""
        threading.Thread(target=self._scan_thread, args=(custom_config,), daemon=True).start()
        return {'status': 'started'}
    
    def _scan_thread(self, custom_config: Optional[Dict] = None):
        scan_start = time.time()
        try:
            config = custom_config or self._load_config()
            self.scoring_engine.update_config(config)
            
            min_volume = config.get('css_min_volume_24h', 10_000_000)
            min_score = config.get('css_min_score', 60)
            main_tf = config.get('css_main_tf', '15')
            kline_limit = config.get('css_kline_limit', 500)
            scan_limit = config.get('css_scan_limit', 100)
            max_spread = config.get('css_max_spread_percent', 0.3)
            
            self.status = "Fetching market data..."
            self._log_activity("📡 Fetching USDT perpetuals...")
            
            symbols_data = self.data_fetcher.get_usdt_perpetuals(min_volume)
            symbols_data = [s for s in symbols_data if s.get('spread', 999) <= max_spread][:scan_limit]
            
            if not symbols_data:
                self.status = "No symbols found"
                self._log_activity("❌ No symbols match criteria", "error")
                self.is_scanning = False
                return
            
            total = len(symbols_data)
            self._log_activity(f"🎯 Found {total} symbols to scan")
            results = []
            
            for i, sym_data in enumerate(symbols_data):
                symbol = sym_data['symbol']
                try:
                    self.progress = int((i + 1) / total * 100)
                    self.status = f"Analyzing {symbol}"
                    self.current_symbol = symbol
                    
                    df = self.data_fetcher.get_klines(symbol, main_tf, kline_limit)
                    if df is None or len(df) < 100:
                        continue
                    
                    oi_data = self.data_fetcher.get_open_interest(symbol, "1h", 48)
                    funding_data = self.data_fetcher.get_funding_rate(symbol, 48)
                    ls_ratio_data = self.data_fetcher.get_long_short_ratio(symbol, "1h")
                    ticker_data = self.data_fetcher.get_ticker_info(symbol)
                    
                    score, trend_bias = self.scoring_engine.calculate_score(df, oi_data, funding_data, ls_ratio_data, ticker_data)
                    
                    if score.total >= min_score:
                        signal_type = self._determine_signal_type(score, trend_bias)
                        result = {
                            'symbol': symbol, 'signal_type': signal_type.value, 'trend_bias': trend_bias.value,
                            'total_score': score.total, 'scores': score.to_dict(), 'ticker': ticker_data,
                            'timestamp': datetime.utcnow().isoformat(),
                        }
                        results.append(result)
                        self._save_signal(result, score, ticker_data)
                        self._log_activity(f"🌀 {symbol}: {signal_type.value} | Score: {score.total} | {trend_bias.value}", "success")
                    
                    time.sleep(0.05)
                except Exception as e:
                    logger.debug(f"Scan error for {symbol}: {e}")
            
            results.sort(key=lambda x: x['total_score'], reverse=True)
            self.scan_results = results
            
            scan_duration = time.time() - scan_start
            self.stats['total_scans'] += 1
            self.stats['total_signals'] += len(results)
            self.stats['bullish_signals'] += sum(1 for r in results if r['trend_bias'] == 'BULLISH')
            self.stats['bearish_signals'] += sum(1 for r in results if r['trend_bias'] == 'BEARISH')
            self.stats['last_scan_duration'] = round(scan_duration, 1)
            
            self.status = f"Complete: {len(results)} signals found"
            self.last_scan_time = datetime.utcnow()
            self._log_activity(f"✅ Scan complete: {len(results)} signals in {scan_duration:.1f}s", "success")
            
        except Exception as e:
            logger.error(f"Scan thread error: {e}")
            self.status = f"Error: {e}"
            self._log_activity(f"❌ Scan error: {e}", "error")
        finally:
            self.is_scanning = False
            self.progress = 100
            self.current_symbol = ""
    
    def _determine_signal_type(self, score: SpringScore, trend_bias: TrendBias) -> SpringSignal:
        is_squeezed = score.volatility >= 10 or score.squeeze_intensity > 0.7
        has_fuel = score.oi >= 15 or score.oi_change > 3
        
        if is_squeezed and has_fuel:
            if trend_bias == TrendBias.BULLISH:
                return SpringSignal.READY_PUMP
            elif trend_bias == TrendBias.BEARISH:
                return SpringSignal.READY_DUMP
        
        if score.price_vs_cloud == "INSIDE":
            if score.tk_cross == "BULLISH" and score.cvd_divergence > 0:
                return SpringSignal.READY_PUMP
            elif score.tk_cross == "BEARISH" and score.cvd_divergence < 0:
                return SpringSignal.READY_DUMP
        
        if score.kumo_twist:
            return SpringSignal.READY_PUMP if score.cloud_bullish else SpringSignal.READY_DUMP
        
        return SpringSignal.COILING
    
    def _save_signal(self, result: Dict, score: SpringScore, ticker: Optional[Dict]):
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            return
        session = None
        try:
            session = db_manager.get_session()
            signal = CoilingSpringSignal(
                symbol=result['symbol'], signal_type=result['signal_type'], trend_bias=result['trend_bias'],
                total_score=score.total, oi_score=score.oi, funding_score=score.funding, ls_ratio_score=score.ls_ratio,
                cvd_score=score.cvd, volatility_score=score.volatility, ichimoku_score=score.ichimoku,
                oi_change_percent=score.oi_change, funding_rate=score.funding_rate, long_short_ratio=score.long_short_ratio,
                cvd_divergence=score.cvd_divergence, bb_width_percentile=score.bb_width_percentile, atr_ratio=score.atr_ratio,
                price_change_percent=score.price_change, price_vs_cloud=score.price_vs_cloud, tenkan_kijun_cross=score.tk_cross,
                current_price=ticker.get('price') if ticker else None, volume_24h=ticker.get('volume_24h') if ticker else None,
                candles_analyzed=score.candles_analyzed,
            )
            session.add(signal)
            session.commit()
        except Exception as e:
            logger.error(f"Save signal error: {e}")
            if session: session.rollback()
        finally:
            if session: session.close()
    
    def get_signals(self, limit: int = 50) -> List[Dict]:
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            return self.scan_results[:limit]
        session = None
        try:
            session = db_manager.get_session()
            signals = session.query(CoilingSpringSignal).order_by(CoilingSpringSignal.created_at.desc()).limit(limit).all()
            return [{
                'id': s.id, 'symbol': s.symbol, 'signal_type': s.signal_type, 'trend_bias': s.trend_bias,
                'total_score': s.total_score, 'oi_score': s.oi_score, 'funding_score': s.funding_score,
                'ls_ratio_score': s.ls_ratio_score or 0, 'cvd_score': s.cvd_score, 'volatility_score': s.volatility_score,
                'ichimoku_score': s.ichimoku_score, 'oi_change': s.oi_change_percent, 'funding_rate': s.funding_rate,
                'long_short_ratio': s.long_short_ratio or 1.0, 'cvd_divergence': s.cvd_divergence,
                'price_vs_cloud': s.price_vs_cloud, 'current_price': s.current_price, 'volume_24h': s.volume_24h,
                'candles_analyzed': s.candles_analyzed or 0, 'time': s.created_at.strftime('%d.%m %H:%M') if s.created_at else '',
            } for s in signals]
        except Exception as e:
            logger.error(f"Get signals error: {e}")
            return []
        finally:
            if session: session.close()
    
    def get_status(self) -> Dict:
        return {
            'is_scanning': self.is_scanning, 'progress': self.progress, 'status': self.status,
            'current_symbol': self.current_symbol, 'last_scan': self.last_scan_time.isoformat() if self.last_scan_time else None,
            'results_count': len(self.scan_results), 'auto_running': self.auto_running,
            'auto_scan_count': self.auto_scan_count, 'stats': self.stats,
        }
    
    def get_activity_log(self, limit: int = 50) -> List[Dict]:
        return list(self.activity_log)[-limit:]
    
    def start_auto_mode(self, interval: int = 300):
        if self.auto_running:
            return
        self.auto_running = True
        self.auto_scan_count = 0
        self._auto_thread = threading.Thread(target=self._auto_loop, args=(interval,), daemon=True)
        self._auto_thread.start()
        self._log_activity(f"🔄 Auto mode started ({interval}s interval)", "info")
    
    def stop_auto_mode(self):
        self.auto_running = False
        self._log_activity("⏹️ Auto mode stopped", "info")
    
    def _auto_loop(self, interval: int):
        while self.auto_running:
            if not self.is_scanning:
                self.auto_scan_count += 1
                self._log_activity(f"🔄 Auto scan #{self.auto_scan_count} starting...", "info")
                self.start_scan()
            while self.is_scanning and self.auto_running:
                time.sleep(1)
            for _ in range(interval):
                if not self.auto_running:
                    break
                time.sleep(1)
    
    def clear_history(self):
        if not HAS_DB or db_manager is None or CoilingSpringSignal is None:
            self.scan_results = []
            return
        session = None
        try:
            session = db_manager.get_session()
            session.query(CoilingSpringSignal).delete()
            session.commit()
            self.scan_results = []
            self._log_activity("🗑️ History cleared", "info")
        except Exception as e:
            logger.error(f"Clear history error: {e}")
        finally:
            if session: session.close()

# ============================================================================
#                              FLASK ROUTES
# ============================================================================

def get_coiling_spring_scanner() -> CoilingSpringScanner:
    return CoilingSpringScanner()

def register_coiling_spring_routes(app):
    
    @app.route('/coiling_spring')
    def coiling_spring_page():
        scanner = get_coiling_spring_scanner()
        return render_template('coiling_spring.html', config=scanner.get_config(), status=scanner.get_status(),
                               signals=scanner.get_signals(50), param_help=PARAM_HELP)
    
    @app.route('/coiling_spring/scan', methods=['POST'])
    def coiling_spring_scan():
        return jsonify(get_coiling_spring_scanner().start_scan())
    
    @app.route('/coiling_spring/status')
    def coiling_spring_status():
        return jsonify(get_coiling_spring_scanner().get_status())
    
    @app.route('/coiling_spring/signals')
    def coiling_spring_signals():
        return jsonify(get_coiling_spring_scanner().get_signals(request.args.get('limit', 50, type=int)))
    
    @app.route('/coiling_spring/activity')
    def coiling_spring_activity():
        return jsonify(get_coiling_spring_scanner().get_activity_log())
    
    @app.route('/coiling_spring/config', methods=['GET', 'POST'])
    def coiling_spring_config():
        scanner = get_coiling_spring_scanner()
        if request.method == 'POST':
            scanner.save_config(request.json or {})
            return jsonify({'status': 'ok'})
        return jsonify(scanner.get_config())
    
    @app.route('/coiling_spring/auto', methods=['POST'])
    def coiling_spring_auto():
        scanner = get_coiling_spring_scanner()
        data = request.json or {}
        if data.get('enabled', False):
            scanner.start_auto_mode(int(data.get('interval', 300)))
        else:
            scanner.stop_auto_mode()
        return jsonify({'status': 'ok', 'auto_running': scanner.auto_running})
    
    @app.route('/coiling_spring/clear', methods=['POST'])
    def coiling_spring_clear():
        get_coiling_spring_scanner().clear_history()
        return jsonify({'status': 'cleared'})
    
    logger.info("✅ Coiling Spring Scanner v2.0 routes registered")

if __name__ == "__main__":
    print("🌀 Coiling Spring Scanner v2.0")
    scanner = CoilingSpringScanner()
    print(f"Status: {scanner.get_status()}")
