#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🐋 WHALE HUNTER PRO v3.0
========================
Професійна бальна система пошуку торгових можливостей.
Замість жорстких AND-фільтрів — гнучкий scoring engine.

Особливості:
- Бальна система (0-100 очок)
- Multi-Timeframe Analysis
- Адаптивні режими (BULL/BEAR/NEUTRAL)
- 7 незалежних фільтрів з вагами
- Авто та ручний режими
- Повна кастомізація параметрів

Автор: SVV Webhook Bot Team
Версія: 3.0.0
"""

import threading
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# Імпорти з проекту
from bot import bot_instance
from settings_manager import settings
from models import db_manager, Base
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Text

# Індикатори
from indicators import (
    simple_rsi, calculate_rsi_series,
    simple_atr, calculate_atr_series,
    calculate_ema, calculate_obv, calculate_slope,
    calculate_bollinger_bands
)
from indicators_pro import calculate_rvol, check_ttm_squeeze, calculate_adx
from rsi_screener import calculate_mfi_series

logger = logging.getLogger("WhaleHunterPro")


# ============================================================================
#                              ENUMS & CONSTANTS
# ============================================================================

class MarketMode(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SignalStrength(Enum):
    WEAK = "WEAK"           # 40-54
    MODERATE = "MODERATE"   # 55-69
    STRONG = "STRONG"       # 70-84
    PREMIUM = "PREMIUM"     # 85-100


# Пояснення параметрів (для UI)
PARAM_HELP = {
    # Загальні
    "whp_enabled": "Увімкнути/вимкнути модуль Whale Hunter PRO",
    "whp_auto_mode": "Автоматичне сканування за розкладом",
    "whp_auto_interval": "Інтервал авто-сканування (хвилини)",
    "whp_min_score": "Мінімальний бал для сигналу (0-100). Рекомендовано 50+",
    
    # Ліквідність
    "whp_min_volume": "Мінімальний 24h об'єм торгів в USDT. Фільтрує неліквідні монети",
    "whp_scan_limit": "Кількість топ-монет для сканування (по об'єму)",
    
    # Таймфрейми
    "whp_main_tf": "Основний таймфрейм аналізу. 15m-1H для скальпінгу, 4H для свінгу",
    "whp_htf": "Старший таймфрейм для підтвердження тренду. Зазвичай 4x від основного",
    "whp_htf_auto": "Автоматичний вибір HTF на основі основного TF",
    
    # Фільтри та ваги
    "whp_use_rsi": "RSI фільтр: виявляє перекупленість/перепроданість",
    "whp_rsi_weight": "Вага RSI у загальному балі (0-30)",
    "whp_rsi_oversold": "Рівень перепроданості RSI (buy zone)",
    "whp_rsi_overbought": "Рівень перекупленості RSI (sell zone)",
    
    "whp_use_mfi": "MFI Cloud: аналіз грошового потоку з об'ємом",
    "whp_mfi_weight": "Вага MFI у загальному балі (0-20)",
    
    "whp_use_rvol": "RVOL: відносний об'єм порівняно із середнім",
    "whp_rvol_weight": "Вага RVOL у загальному балі (0-20)",
    "whp_rvol_threshold": "Поріг RVOL (1.2 = на 20% вище середнього)",
    
    "whp_use_ob": "Order Block: пошук інституційних зон",
    "whp_ob_weight": "Вага OB у загальному балі (0-25)",
    "whp_ob_distance": "Максимальна відстань до OB у % від ціни",
    
    "whp_use_btc": "BTC Trend: фільтр загального стану ринку",
    "whp_btc_weight": "Вага BTC тренду у загальному балі (0-20)",
    
    "whp_use_adx": "ADX: індикатор сили тренду",
    "whp_adx_weight": "Вага ADX у загальному балі (0-15)",
    "whp_adx_threshold": "Поріг ADX для визначення тренду (зазвичай 25)",
    
    "whp_use_squeeze": "TTM Squeeze: стиснення волатильності перед рухом",
    "whp_squeeze_weight": "Вага Squeeze у загальному балі (0-10)",
    
    # Додаткові
    "whp_use_divergence": "OBV Divergence: розходження ціни та об'єму",
    "whp_divergence_weight": "Вага дивергенції у загальному балі (0-15)",
}

# HTF Auto Mapping
HTF_MAPPING = {
    "5": "60", "15": "60", "30": "240",
    "60": "240", "120": "240", "240": "D",
    "D": "W"
}


# ============================================================================
#                              DATABASE MODEL
# ============================================================================

class WhaleHunterSignal(Base):
    """Модель для збереження сигналів"""
    __tablename__ = 'whale_hunter_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    price = Column(Float)
    direction = Column(String(10))  # LONG / SHORT
    score = Column(Integer)
    strength = Column(String(20))   # WEAK/MODERATE/STRONG/PREMIUM
    
    # Деталі скорингу
    rsi_score = Column(Integer, default=0)
    mfi_score = Column(Integer, default=0)
    rvol_score = Column(Integer, default=0)
    ob_score = Column(Integer, default=0)
    btc_score = Column(Integer, default=0)
    adx_score = Column(Integer, default=0)
    squeeze_score = Column(Integer, default=0)
    divergence_score = Column(Integer, default=0)
    
    # Значення індикаторів
    rsi_value = Column(Float, default=0)
    mfi_value = Column(Float, default=0)
    rvol_value = Column(Float, default=0)
    adx_value = Column(Float, default=0)
    
    # OB Info
    ob_distance = Column(Float, default=0)
    ob_type = Column(String(10), default="")
    
    # Meta
    btc_trend = Column(String(20), default="")
    market_mode = Column(String(20), default="")
    timeframe = Column(String(10), default="")
    details = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
#                              SCORING ENGINE
# ============================================================================

@dataclass
class ScoreBreakdown:
    """Деталізація балів"""
    rsi: int = 0
    mfi: int = 0
    rvol: int = 0
    ob: int = 0
    btc: int = 0
    adx: int = 0
    squeeze: int = 0
    divergence: int = 0
    
    # Значення індикаторів
    rsi_value: float = 0
    mfi_value: float = 0
    rvol_value: float = 0
    adx_value: float = 0
    
    # OB
    ob_distance: float = 0
    ob_type: str = ""
    
    @property
    def total(self) -> int:
        return self.rsi + self.mfi + self.rvol + self.ob + self.btc + self.adx + self.squeeze + self.divergence
    
    @property
    def strength(self) -> SignalStrength:
        t = self.total
        if t >= 85:
            return SignalStrength.PREMIUM
        elif t >= 70:
            return SignalStrength.STRONG
        elif t >= 55:
            return SignalStrength.MODERATE
        else:
            return SignalStrength.WEAK
    
    def to_dict(self) -> Dict:
        return {
            'rsi': self.rsi, 'mfi': self.mfi, 'rvol': self.rvol,
            'ob': self.ob, 'btc': self.btc, 'adx': self.adx,
            'squeeze': self.squeeze, 'divergence': self.divergence,
            'total': self.total, 'strength': self.strength.value,
            'rsi_value': self.rsi_value, 'mfi_value': self.mfi_value,
            'rvol_value': self.rvol_value, 'adx_value': self.adx_value,
            'ob_distance': self.ob_distance, 'ob_type': self.ob_type
        }


class ScoringEngine:
    """
    🎯 Двигун скорингу - серце Whale Hunter PRO
    
    Кожен фільтр має:
    - enabled: bool - увімкнений чи ні
    - weight: int - максимальна кількість балів
    - threshold: float - поріг спрацювання
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def update_config(self, config: Dict):
        self.config = config
    
    def _get(self, key: str, default: Any = None) -> Any:
        """Отримати значення з конфігу"""
        return self.config.get(key, default)
    
    def calculate_score(
        self,
        df: pd.DataFrame,
        direction: str,  # "LONG" or "SHORT"
        btc_trend: MarketMode,
        ob_info: Optional[Dict] = None
    ) -> ScoreBreakdown:
        """
        Розраховує загальний бал для монети
        
        Args:
            df: DataFrame з OHLCV даними
            direction: Напрямок сигналу
            btc_trend: Поточний тренд BTC
            ob_info: Інформація про найближчий Order Block
        
        Returns:
            ScoreBreakdown з деталізацією балів
        """
        score = ScoreBreakdown()
        
        if df is None or len(df) < 50:
            return score
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        current_price = close.iloc[-1]
        
        # ========== 1. RSI SCORE ==========
        if self._get('whp_use_rsi', True):
            try:
                rsi = simple_rsi(close, 14)
                score.rsi_value = rsi
                
                weight = self._get('whp_rsi_weight', 20)
                oversold = self._get('whp_rsi_oversold', 40)
                overbought = self._get('whp_rsi_overbought', 60)
                
                if direction == "LONG" and rsi <= oversold:
                    # Чим нижче RSI, тим більше балів
                    intensity = (oversold - rsi) / oversold
                    score.rsi = int(weight * min(1.0, 0.5 + intensity))
                elif direction == "SHORT" and rsi >= overbought:
                    intensity = (rsi - overbought) / (100 - overbought)
                    score.rsi = int(weight * min(1.0, 0.5 + intensity))
                elif direction == "LONG" and rsi < 50:
                    score.rsi = int(weight * 0.3)  # Частковий бал
                elif direction == "SHORT" and rsi > 50:
                    score.rsi = int(weight * 0.3)
            except Exception as e:
                logger.debug(f"RSI calc error: {e}")
        
        # ========== 2. MFI SCORE ==========
        if self._get('whp_use_mfi', True):
            try:
                mfi_series = calculate_mfi_series(high, low, close, volume, 20)
                mfi = float(mfi_series.iloc[-1])
                score.mfi_value = mfi
                
                # MFI Cloud: Fast vs Slow
                fast_mfi = calculate_ema(mfi_series, 5).iloc[-1]
                slow_mfi = calculate_ema(mfi_series, 13).iloc[-1]
                
                weight = self._get('whp_mfi_weight', 15)
                
                if direction == "LONG" and fast_mfi > slow_mfi:
                    score.mfi = weight
                elif direction == "SHORT" and fast_mfi < slow_mfi:
                    score.mfi = weight
                elif direction == "LONG" and mfi < 40:
                    score.mfi = int(weight * 0.5)
                elif direction == "SHORT" and mfi > 60:
                    score.mfi = int(weight * 0.5)
            except Exception as e:
                logger.debug(f"MFI calc error: {e}")
        
        # ========== 3. RVOL SCORE ==========
        if self._get('whp_use_rvol', True):
            try:
                rvol_series = calculate_rvol(volume, 20)
                rvol = float(rvol_series.iloc[-1])
                score.rvol_value = round(rvol, 2)
                
                weight = self._get('whp_rvol_weight', 15)
                threshold = self._get('whp_rvol_threshold', 1.2)
                
                if rvol >= threshold:
                    # Більший об'єм = більше балів (до 2x threshold)
                    intensity = min(1.0, (rvol - 1) / (threshold - 1 + 0.5))
                    score.rvol = int(weight * intensity)
            except Exception as e:
                logger.debug(f"RVOL calc error: {e}")
        
        # ========== 4. ORDER BLOCK SCORE ==========
        if self._get('whp_use_ob', True) and ob_info:
            try:
                weight = self._get('whp_ob_weight', 20)
                max_distance = self._get('whp_ob_distance', 3.0)
                
                ob_mid = (ob_info['top'] + ob_info['bottom']) / 2
                distance_pct = abs(current_price - ob_mid) / current_price * 100
                score.ob_distance = round(distance_pct, 2)
                score.ob_type = ob_info.get('type', '')
                
                if distance_pct <= max_distance:
                    # Чим ближче до OB, тим більше балів
                    proximity = 1 - (distance_pct / max_distance)
                    score.ob = int(weight * proximity)
            except Exception as e:
                logger.debug(f"OB calc error: {e}")
        
        # ========== 5. BTC TREND SCORE ==========
        if self._get('whp_use_btc', True):
            weight = self._get('whp_btc_weight', 15)
            
            if direction == "LONG" and btc_trend == MarketMode.BULLISH:
                score.btc = weight
            elif direction == "SHORT" and btc_trend == MarketMode.BEARISH:
                score.btc = weight
            elif btc_trend == MarketMode.NEUTRAL:
                score.btc = int(weight * 0.5)
        
        # ========== 6. ADX SCORE ==========
        if self._get('whp_use_adx', True):
            try:
                adx_series, plus_di, minus_di = calculate_adx(high, low, close, 14)
                adx = float(adx_series.iloc[-1])
                score.adx_value = round(adx, 1)
                
                weight = self._get('whp_adx_weight', 10)
                threshold = self._get('whp_adx_threshold', 25)
                
                # ADX < threshold = consolidation (good for entry)
                # ADX > threshold and rising = trend starting
                if adx < threshold:
                    score.adx = int(weight * 0.7)  # Консолідація
                elif adx > threshold:
                    # Перевіряємо напрямок DI
                    if direction == "LONG" and plus_di.iloc[-1] > minus_di.iloc[-1]:
                        score.adx = weight
                    elif direction == "SHORT" and minus_di.iloc[-1] > plus_di.iloc[-1]:
                        score.adx = weight
            except Exception as e:
                logger.debug(f"ADX calc error: {e}")
        
        # ========== 7. TTM SQUEEZE SCORE ==========
        if self._get('whp_use_squeeze', True):
            try:
                squeeze_series = check_ttm_squeeze(df)
                is_squeeze = bool(squeeze_series.iloc[-1])
                
                weight = self._get('whp_squeeze_weight', 5)
                
                if is_squeeze:
                    score.squeeze = weight
            except Exception as e:
                logger.debug(f"Squeeze calc error: {e}")
        
        # ========== 8. OBV DIVERGENCE SCORE ==========
        if self._get('whp_use_divergence', True):
            try:
                obv = calculate_obv(close, volume)
                price_slope = calculate_slope(close, 15)
                obv_slope = calculate_slope(obv, 15)
                
                # Нормалізація
                norm_price_slope = (price_slope / current_price) * 100 if current_price > 0 else 0
                
                weight = self._get('whp_divergence_weight', 10)
                
                # Bullish divergence: price down/flat, OBV up
                if direction == "LONG" and norm_price_slope < 0.5 and obv_slope > 0:
                    score.divergence = weight
                # Bearish divergence: price up/flat, OBV down
                elif direction == "SHORT" and norm_price_slope > -0.5 and obv_slope < 0:
                    score.divergence = weight
            except Exception as e:
                logger.debug(f"Divergence calc error: {e}")
        
        return score


# ============================================================================
#                           ORDER BLOCK FINDER (Simplified)
# ============================================================================

class SimpleOBFinder:
    """Спрощений пошук Order Blocks для скорингу"""
    
    def __init__(self, swing_length: int = 3):
        self.swing_length = swing_length
    
    def find_nearest_ob(
        self,
        df: pd.DataFrame,
        direction: str,
        max_distance_pct: float = 5.0
    ) -> Optional[Dict]:
        """
        Знаходить найближчий валідний Order Block
        
        Args:
            df: OHLCV DataFrame
            direction: "LONG" (шукаємо Bull OB) або "SHORT" (Bear OB)
            max_distance_pct: Максимальна відстань у %
        
        Returns:
            Dict з top, bottom, type або None
        """
        if df is None or len(df) < 50:
            return None
        
        current_price = df['close'].iloc[-1]
        obs = []
        
        # Аналізуємо останні 200 свічок
        subset = df.tail(200).reset_index(drop=True)
        n = len(subset)
        
        for i in range(self.swing_length, n - self.swing_length):
            # Перевірка Swing Low (для Bull OB)
            if direction == "LONG":
                is_swing = True
                for j in range(1, self.swing_length + 1):
                    if subset['low'].iloc[i] >= subset['low'].iloc[i - j] or \
                       subset['low'].iloc[i] >= subset['low'].iloc[i + j]:
                        is_swing = False
                        break
                
                if is_swing:
                    # Перевіряємо чи був BOS (Break of Structure) вгору
                    swing_high = subset['high'].iloc[i]
                    for k in range(i + 1, min(i + 30, n)):
                        if subset['close'].iloc[k] > swing_high:
                            # Bull OB знайдено
                            obs.append({
                                'top': subset['high'].iloc[i],
                                'bottom': subset['low'].iloc[i],
                                'type': 'Bull',
                                'idx': i
                            })
                            break
            
            # Перевірка Swing High (для Bear OB)
            else:
                is_swing = True
                for j in range(1, self.swing_length + 1):
                    if subset['high'].iloc[i] <= subset['high'].iloc[i - j] or \
                       subset['high'].iloc[i] <= subset['high'].iloc[i + j]:
                        is_swing = False
                        break
                
                if is_swing:
                    swing_low = subset['low'].iloc[i]
                    for k in range(i + 1, min(i + 30, n)):
                        if subset['close'].iloc[k] < swing_low:
                            obs.append({
                                'top': subset['high'].iloc[i],
                                'bottom': subset['low'].iloc[i],
                                'type': 'Bear',
                                'idx': i
                            })
                            break
        
        if not obs:
            return None
        
        # Фільтруємо валідні (не пробиті) та в межах відстані
        valid_obs = []
        for ob in obs:
            ob_mid = (ob['top'] + ob['bottom']) / 2
            distance = abs(current_price - ob_mid) / current_price * 100
            
            # Перевірка інвалідації
            is_valid = True
            if direction == "LONG" and current_price < ob['bottom']:
                is_valid = False
            elif direction == "SHORT" and current_price > ob['top']:
                is_valid = False
            
            if is_valid and distance <= max_distance_pct:
                ob['distance'] = distance
                valid_obs.append(ob)
        
        if not valid_obs:
            return None
        
        # Повертаємо найближчий
        return min(valid_obs, key=lambda x: x['distance'])


# ============================================================================
#                           WHALE HUNTER PRO CORE
# ============================================================================

class WhaleHunterPro:
    """
    🐋 Головний клас Whale Hunter PRO
    
    Функціонал:
    - Визначення режиму ринку (BTC Trend)
    - Сканування топ-монет
    - Бальна система оцінки
    - Авто та ручний режими
    - Збереження сигналів у БД
    """
    
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status = "Ready"
        self.last_scan_time = None
        self.btc_trend = MarketMode.NEUTRAL
        self.scan_results = []
        
        # Автоматизація
        self.auto_running = False
        self.auto_interval = 60  # хвилини
        self._stop_event = threading.Event()
        
        # Компоненти
        self.scoring_engine = ScoringEngine(self._load_config())
        self.ob_finder = SimpleOBFinder()
        
        # Запуск планувальника
        threading.Thread(target=self._scheduler_loop, daemon=True).start()
        
        # Ініціалізація таблиці
        self._ensure_table()
    
    def _ensure_table(self):
        """Створює таблицю якщо не існує"""
        try:
            WhaleHunterSignal.__table__.create(db_manager.engine, checkfirst=True)
        except Exception as e:
            logger.error(f"Table creation error: {e}")
    
    def _load_config(self) -> Dict:
        """Завантажує конфігурацію з settings"""
        
        def to_bool(val, default=True):
            """Конвертує значення в boolean"""
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.lower() in ('true', '1', 'yes', 'on')
            return bool(val) if val is not None else default
        
        def to_int(val, default=0):
            """Конвертує значення в int"""
            try:
                return int(float(val)) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        def to_float(val, default=0.0):
            """Конвертує значення в float"""
            try:
                return float(val) if val is not None else default
            except (ValueError, TypeError):
                return default
        
        return {
            # Загальні
            'whp_enabled': to_bool(settings.get('whp_enabled'), True),
            'whp_auto_mode': to_bool(settings.get('whp_auto_mode'), False),
            'whp_auto_interval': to_int(settings.get('whp_auto_interval'), 60),
            'whp_min_score': to_int(settings.get('whp_min_score'), 50),
            
            # Ліквідність
            'whp_min_volume': to_float(settings.get('whp_min_volume'), 5_000_000),
            'whp_scan_limit': to_int(settings.get('whp_scan_limit'), 50),
            
            # Таймфрейми
            'whp_main_tf': str(settings.get('whp_main_tf', '60')),
            'whp_htf': str(settings.get('whp_htf', '240')),
            'whp_htf_auto': to_bool(settings.get('whp_htf_auto'), True),
            
            # RSI
            'whp_use_rsi': to_bool(settings.get('whp_use_rsi'), True),
            'whp_rsi_weight': to_int(settings.get('whp_rsi_weight'), 20),
            'whp_rsi_oversold': to_int(settings.get('whp_rsi_oversold'), 40),
            'whp_rsi_overbought': to_int(settings.get('whp_rsi_overbought'), 60),
            
            # MFI
            'whp_use_mfi': to_bool(settings.get('whp_use_mfi'), True),
            'whp_mfi_weight': to_int(settings.get('whp_mfi_weight'), 15),
            
            # RVOL
            'whp_use_rvol': to_bool(settings.get('whp_use_rvol'), True),
            'whp_rvol_weight': to_int(settings.get('whp_rvol_weight'), 15),
            'whp_rvol_threshold': to_float(settings.get('whp_rvol_threshold'), 1.2),
            
            # Order Block
            'whp_use_ob': to_bool(settings.get('whp_use_ob'), True),
            'whp_ob_weight': to_int(settings.get('whp_ob_weight'), 20),
            'whp_ob_distance': to_float(settings.get('whp_ob_distance'), 3.0),
            
            # BTC Trend
            'whp_use_btc': to_bool(settings.get('whp_use_btc'), True),
            'whp_btc_weight': to_int(settings.get('whp_btc_weight'), 15),
            
            # ADX
            'whp_use_adx': to_bool(settings.get('whp_use_adx'), True),
            'whp_adx_weight': to_int(settings.get('whp_adx_weight'), 10),
            'whp_adx_threshold': to_int(settings.get('whp_adx_threshold'), 25),
            
            # TTM Squeeze
            'whp_use_squeeze': to_bool(settings.get('whp_use_squeeze'), True),
            'whp_squeeze_weight': to_int(settings.get('whp_squeeze_weight'), 5),
            
            # Divergence
            'whp_use_divergence': to_bool(settings.get('whp_use_divergence'), True),
            'whp_divergence_weight': to_int(settings.get('whp_divergence_weight'), 10),
            
            # Integration
            'whp_add_to_watchlist': to_bool(settings.get('whp_add_to_watchlist'), True),
        }
    
    def update_config(self):
        """Оновлює конфігурацію"""
        config = self._load_config()
        self.scoring_engine.update_config(config)
        self.auto_interval = config.get('whp_auto_interval', 60)
        logger.info("WhaleHunterPro config updated")
    
    def _get_htf(self, main_tf: str) -> str:
        """Отримує HTF на основі налаштувань"""
        config = self._load_config()
        if config.get('whp_htf_auto', True):
            return HTF_MAPPING.get(main_tf, '240')
        return config.get('whp_htf', '240')
    
    def fetch_data(self, symbol: str, timeframe: str, limit: int = 500) -> Optional[pd.DataFrame]:
        """Завантажує OHLCV дані"""
        try:
            tf_map = {'5': '5', '15': '15', '30': '30', '60': '60', 
                      '120': '120', '240': '240', 'D': 'D', 'W': 'W'}
            interval = tf_map.get(str(timeframe), '60')
            
            res = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=interval, limit=limit
            )
            
            if res['retCode'] == 0 and res['result']['list']:
                data = res['result']['list'][::-1]  # Oldest -> Newest
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype({'open': float, 'high': float, 'low': float, 
                               'close': float, 'volume': float})
                
                # Виключаємо останню незакриту свічку
                if len(df) > 1:
                    df = df.iloc[:-1].reset_index(drop=True)
                
                return df
        except Exception as e:
            logger.error(f"Fetch error {symbol}: {e}")
        return None
    
    def analyze_btc_trend(self) -> MarketMode:
        """Аналізує тренд BTC"""
        try:
            df = self.fetch_data("BTCUSDT", "240", 250)
            if df is None:
                return MarketMode.NEUTRAL
            
            close = df['close']
            ema200 = calculate_ema(close, 200).iloc[-1]
            ema50 = calculate_ema(close, 50).iloc[-1]
            price = close.iloc[-1]
            
            # Bullish: Price > EMA200 AND EMA50 > EMA200
            if price > ema200 and ema50 > ema200:
                self.btc_trend = MarketMode.BULLISH
            # Bearish: Price < EMA200 AND EMA50 < EMA200
            elif price < ema200 and ema50 < ema200:
                self.btc_trend = MarketMode.BEARISH
            else:
                self.btc_trend = MarketMode.NEUTRAL
            
            logger.info(f"BTC Trend: {self.btc_trend.value} (Price: {price:.0f}, EMA200: {ema200:.0f})")
            return self.btc_trend
            
        except Exception as e:
            logger.error(f"BTC trend error: {e}")
            return MarketMode.NEUTRAL
    
    def determine_direction(self) -> str:
        """Визначає напрямок на основі BTC тренду"""
        if self.btc_trend == MarketMode.BULLISH:
            return "LONG"
        elif self.btc_trend == MarketMode.BEARISH:
            return "SHORT"
        else:
            return "LONG"  # Default to LONG in neutral
    
    def analyze_symbol(self, symbol: str, direction: str) -> Optional[Dict]:
        """Аналізує один символ"""
        config = self._load_config()
        main_tf = config.get('whp_main_tf', '60')
        
        # Завантажуємо дані
        df = self.fetch_data(symbol, main_tf, 500)
        if df is None or len(df) < 100:
            return None
        
        current_price = df['close'].iloc[-1]
        
        # Шукаємо Order Block
        ob_info = None
        if config.get('whp_use_ob', True):
            ob_direction = "LONG" if direction == "LONG" else "SHORT"
            ob_info = self.ob_finder.find_nearest_ob(
                df, ob_direction, 
                config.get('whp_ob_distance', 3.0)
            )
        
        # Розраховуємо бали
        score = self.scoring_engine.calculate_score(df, direction, self.btc_trend, ob_info)
        
        # Перевіряємо мінімальний поріг
        min_score = config.get('whp_min_score', 50)
        if score.total < min_score:
            return None
        
        return {
            'symbol': symbol,
            'price': current_price,
            'direction': direction,
            'score': score.total,
            'strength': score.strength.value,
            'breakdown': score.to_dict(),
            'btc_trend': self.btc_trend.value,
            'timeframe': main_tf
        }
    
    def save_signal(self, result: Dict):
        """Зберігає сигнал у БД"""
        session = db_manager.get_session()
        try:
            breakdown = result['breakdown']
            
            sig = WhaleHunterSignal(
                symbol=result['symbol'],
                price=result['price'],
                direction=result['direction'],
                score=result['score'],
                strength=result['strength'],
                rsi_score=breakdown.get('rsi', 0),
                mfi_score=breakdown.get('mfi', 0),
                rvol_score=breakdown.get('rvol', 0),
                ob_score=breakdown.get('ob', 0),
                btc_score=breakdown.get('btc', 0),
                adx_score=breakdown.get('adx', 0),
                squeeze_score=breakdown.get('squeeze', 0),
                divergence_score=breakdown.get('divergence', 0),
                rsi_value=breakdown.get('rsi_value', 0),
                mfi_value=breakdown.get('mfi_value', 0),
                rvol_value=breakdown.get('rvol_value', 0),
                adx_value=breakdown.get('adx_value', 0),
                ob_distance=breakdown.get('ob_distance', 0),
                ob_type=breakdown.get('ob_type', ''),
                btc_trend=result['btc_trend'],
                market_mode=self.btc_trend.value,
                timeframe=result['timeframe'],
                created_at=datetime.utcnow()
            )
            session.add(sig)
            session.commit()
            logger.info(f"✅ Signal saved: {result['symbol']} Score={result['score']} ({result['strength']})")
            
            # 🆕 ІНТЕГРАЦІЯ: Додаємо до Smart Money Watchlist
            if settings.get('whp_add_to_watchlist', True):
                try:
                    from scanner_coordinator import add_to_smart_money_watchlist
                    
                    direction = 'BUY' if result['direction'] == 'LONG' else 'SELL'
                    add_result = add_to_smart_money_watchlist(
                        symbol=result['symbol'],
                        direction=direction,
                        source='Whale Hunter PRO'
                    )
                    
                    if add_result.get('status') == 'ok':
                        logger.info(f"📋 Added to SM Watchlist: {result['symbol']}")
                        
                except Exception as e:
                    logger.warning(f"Failed to add to watchlist: {e}")
                    
        except Exception as e:
            logger.error(f"Save error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def get_history(self, limit: int = 100) -> List[Dict]:
        """Отримує історію сигналів"""
        self._ensure_table()
        session = db_manager.get_session()
        try:
            results = session.query(WhaleHunterSignal)\
                .order_by(WhaleHunterSignal.created_at.desc())\
                .limit(limit).all()
            
            return [{
                'id': r.id,
                'symbol': r.symbol,
                'price': r.price,
                'direction': r.direction,
                'score': r.score,
                'strength': r.strength,
                'rsi': r.rsi_score,
                'mfi': r.mfi_score,
                'rvol': r.rvol_score,
                'ob': r.ob_score,
                'btc': r.btc_score,
                'adx': r.adx_score,
                'squeeze': r.squeeze_score,
                'divergence': r.divergence_score,
                'rsi_value': r.rsi_value,
                'mfi_value': r.mfi_value,
                'rvol_value': r.rvol_value,
                'adx_value': r.adx_value,
                'ob_distance': r.ob_distance,
                'btc_trend': r.btc_trend,
                'timeframe': r.timeframe,
                'time': r.created_at.strftime('%d.%m %H:%M')
            } for r in results]
        except Exception as e:
            logger.error(f"Get history error: {e}")
            return []
        finally:
            session.close()
    
    def clear_history(self):
        """Очищає історію сигналів"""
        session = db_manager.get_session()
        try:
            session.query(WhaleHunterSignal).delete()
            session.commit()
            logger.info("History cleared")
        except Exception as e:
            logger.error(f"Clear error: {e}")
            session.rollback()
        finally:
            session.close()
    
    def start_scan(self, custom_config: Optional[Dict] = None):
        """Запускає сканування"""
        if self.is_scanning:
            return False
        
        # Оновлюємо конфіг якщо передано
        if custom_config:
            for key, value in custom_config.items():
                settings.save_settings({key: value})
        
        self.update_config()
        threading.Thread(target=self._scan_thread).start()
        return True
    
    def _scan_thread(self):
        """Потік сканування"""
        self.is_scanning = True
        self.progress = 0
        self.status = "Initializing..."
        self.scan_results = []
        
        config = self._load_config()
        
        # Очищаємо старі результати
        self.clear_history()
        
        try:
            # 1. Аналіз BTC тренду
            self.status = "Analyzing BTC Trend..."
            self.progress = 5
            self.analyze_btc_trend()
            
            # 2. Визначаємо напрямок
            direction = self.determine_direction()
            self.status = f"Mode: {direction} ({self.btc_trend.value})"
            self.progress = 10
            
            # 3. Отримуємо список монет
            self.status = "Fetching markets..."
            tickers = bot_instance.get_all_tickers()
            
            # Фільтруємо по об'єму
            min_vol = config.get('whp_min_volume', 5_000_000)
            targets = [
                t for t in tickers 
                if t['symbol'].endswith('USDT') 
                and float(t.get('turnover24h', 0)) > min_vol
            ]
            
            # Сортуємо та обмежуємо
            targets.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            scan_limit = config.get('whp_scan_limit', 50)
            targets = targets[:scan_limit]
            
            total = len(targets)
            self.status = f"Scanning {total} coins..."
            
            # 4. Сканування
            for i, t in enumerate(targets):
                if not self.is_scanning:
                    break
                
                symbol = t['symbol']
                self.status = f"Analyzing {symbol}... ({i+1}/{total})"
                self.progress = 10 + int((i / total) * 85)
                
                result = self.analyze_symbol(symbol, direction)
                if result:
                    self.save_signal(result)
                    self.scan_results.append(result)
                
                time.sleep(0.1)
            
            # 5. Завершення
            self.progress = 100
            self.status = f"Done! Found {len(self.scan_results)} signals"
            self.last_scan_time = datetime.now().strftime("%H:%M:%S")
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.status = f"Error: {str(e)}"
        finally:
            self.is_scanning = False
    
    def set_auto_mode(self, enabled: bool, interval: int = 60):
        """Встановлює авто-режим"""
        self.auto_running = enabled
        self.auto_interval = interval
        settings.save_settings({
            'whp_auto_mode': enabled,
            'whp_auto_interval': interval
        })
        logger.info(f"Auto mode: {enabled}, interval: {interval}m")
    
    def _scheduler_loop(self):
        """Цикл планувальника для авто-режиму"""
        while not self._stop_event.is_set():
            if self.auto_running and not self.is_scanning:
                logger.info("⏰ Auto-scan triggered")
                self.start_scan()
                
                # Чекаємо інтервал
                for _ in range(self.auto_interval * 60):
                    if not self.auto_running:
                        break
                    time.sleep(1)
            else:
                time.sleep(5)
    
    def get_status(self) -> Dict:
        """Повертає поточний статус"""
        return {
            'is_scanning': self.is_scanning,
            'progress': self.progress,
            'status': self.status,
            'last_scan_time': self.last_scan_time,
            'btc_trend': self.btc_trend.value,
            'auto_running': self.auto_running,
            'auto_interval': self.auto_interval,
            'results_count': len(self.scan_results)
        }
    
    def get_config(self) -> Dict:
        """Повертає поточну конфігурацію"""
        return self._load_config()
    
    def get_param_help(self) -> Dict:
        """Повертає довідку по параметрах"""
        return PARAM_HELP


# Глобальний екземпляр
whale_hunter_pro = WhaleHunterPro()


# ============================================================================
#                    COORDINATOR INTEGRATION
# ============================================================================

def register_with_coordinator():
    """Реєструє Whale Hunter PRO з координатором сканерів"""
    try:
        from scanner_coordinator import scanner_coordinator, ScannerType
        
        def scan_wrapper():
            """Обгортка для сканування"""
            whale_hunter_pro.start_scan()
        
        scanner_coordinator.set_scan_function(ScannerType.WHALE_HUNTER, scan_wrapper)
        logger.info("✅ Whale Hunter PRO registered with Coordinator")
        
    except ImportError:
        logger.warning("Scanner Coordinator not available")
    except Exception as e:
        logger.error(f"Coordinator registration error: {e}")


# Автореєстрація при імпорті
register_with_coordinator()


# ============================================================================
#                              FLASK ROUTES
# ============================================================================

def register_routes(app):
    """Реєстрація маршрутів Flask"""
    from flask import render_template, request, jsonify
    
    @app.route('/whale_hunter')
    def whale_hunter_page():
        return render_template(
            'whale_hunter_pro.html',
            history=whale_hunter_pro.get_history(),
            status=whale_hunter_pro.get_status(),
            config=whale_hunter_pro.get_config(),
            help=PARAM_HELP
        )
    
    @app.route('/whale_hunter/scan', methods=['POST'])
    def whale_hunter_scan():
        data = request.json or {}
        whale_hunter_pro.start_scan(data)
        return jsonify({'status': 'started'})
    
    @app.route('/whale_hunter/stop', methods=['POST'])
    def whale_hunter_stop():
        whale_hunter_pro.is_scanning = False
        return jsonify({'status': 'stopped'})
    
    @app.route('/whale_hunter/status')
    def whale_hunter_status():
        return jsonify(whale_hunter_pro.get_status())
    
    @app.route('/whale_hunter/history')
    def whale_hunter_history():
        return jsonify(whale_hunter_pro.get_history())
    
    @app.route('/whale_hunter/config', methods=['GET', 'POST'])
    def whale_hunter_config():
        if request.method == 'POST':
            data = request.json or {}
            # Зберігаємо всі налаштування одразу
            settings.save_settings(data)
            whale_hunter_pro.update_config()
            logger.info(f"⚡ Whale Hunter PRO config saved: {len(data)} params")
            return jsonify({'status': 'ok'})
        return jsonify(whale_hunter_pro.get_config())
    
    @app.route('/whale_hunter/auto', methods=['POST'])
    def whale_hunter_auto():
        data = request.json or {}
        enabled = data.get('enabled', False)
        interval = int(data.get('interval', 60))
        whale_hunter_pro.set_auto_mode(enabled, interval)
        return jsonify({'status': 'ok'})
    
    @app.route('/whale_hunter/clear', methods=['POST'])
    def whale_hunter_clear():
        whale_hunter_pro.clear_history()
        return jsonify({'status': 'cleared'})
