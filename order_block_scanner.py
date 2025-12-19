#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📦 ORDER BLOCK SCANNER v1.0
============================
100% реалізація логіки Order Block з Pine Script індикатора.
Без спрощень та модифікацій алгоритму.

Алгоритм:
1. Swing Detection (ta.highest/ta.lowest)
2. Order Block Detection (Bullish/Bearish)
3. Invalidation Check (Wick/Close method)
4. Retest Detection

Автор: SVV Webhook Bot Team
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
#                              ENUMS & TYPES
# ============================================================================

class OBType(Enum):
    BULL = "Bull"
    BEAR = "Bear"


class InvalidationMethod(Enum):
    WICK = "Wick"
    CLOSE = "Close"


class EntryMode(Enum):
    IMMEDIATE = "Immediate"
    RETEST = "Retest"


class OBSelection(Enum):
    NEWEST = "Newest"
    CLOSEST = "Closest"


class OBStatus(Enum):
    VALID = "Valid"
    WAITING_RETEST = "Waiting Retest"
    TRIGGERED = "Triggered"
    INVALIDATED = "Invalidated"


# ============================================================================
#                           ORDER BLOCK INFO
# ============================================================================

@dataclass
class OrderBlockInfo:
    """Інформація про Order Block (відповідає Pine Script orderBlockInfo)"""
    top: float
    bottom: float
    ob_volume: float
    ob_type: OBType
    start_time: int  # timestamp в мілісекундах
    bb_volume: float = 0.0
    ob_low_volume: float = 0.0
    ob_high_volume: float = 0.0
    breaker: bool = False
    break_time: Optional[int] = None
    timeframe_str: str = ""
    disabled: bool = False
    combined: bool = False
    touched: bool = False
    
    def is_valid(self) -> bool:
        """Перевірка чи OB валідний (не breaker і не disabled)"""
        return not self.breaker and not self.disabled
    
    def get_size(self) -> float:
        """Розмір OB зони"""
        return abs(self.top - self.bottom)
    
    def contains_price(self, price: float) -> bool:
        """Чи ціна знаходиться в зоні OB"""
        return self.bottom <= price <= self.top
    
    def to_dict(self) -> Dict:
        """Конвертація в словник"""
        return {
            'top': self.top,
            'bottom': self.bottom,
            'ob_volume': self.ob_volume,
            'ob_type': self.ob_type.value,
            'start_time': self.start_time,
            'breaker': self.breaker,
            'break_time': self.break_time,
            'timeframe_str': self.timeframe_str,
            'disabled': self.disabled,
            'combined': self.combined,
            'touched': self.touched,
            'is_valid': self.is_valid(),
            'size': self.get_size()
        }


# ============================================================================
#                           SWING DETECTION
# ============================================================================

@dataclass
class OBSwing:
    """Swing point для Order Block detection"""
    x: int = 0  # bar_index
    y: float = 0.0  # price
    swing_volume: float = 0.0
    crossed: bool = False


class SwingDetector:
    """
    Детекція свінгів (100% логіка з Pine Script)
    
    Pine Script:
    findOBSwings(len) =>
        var swingType = 0
        upper = ta.highest(len)
        lower = ta.lowest(len)
        swingType := high[len] > upper ? 0 : low[len] < lower ? 1 : swingType
    
    swingType = 0 означає Swing High
    swingType = 1 означає Swing Low
    """
    
    def __init__(self, swing_length: int = 3):
        self.swing_length = swing_length
    
    def find_swings(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Знаходить swing type для кожного бару
        
        Returns:
            Tuple[swing_types, swing_indices] - масиви свінг типів та індексів
        """
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        swing_types = np.zeros(n, dtype=int)  # 0 = high, 1 = low
        swing_highs = np.zeros(n)  # Y-координата swing high
        swing_lows = np.zeros(n)   # Y-координата swing low
        
        swing_type = 0  # var swingType = 0 (persistent)
        
        for i in range(self.swing_length, n):
            # ta.highest(len) - максимум за останні len барів (не включаючи [len])
            # В Pine Script це high[1] to high[len]
            upper = np.max(highs[i - self.swing_length + 1:i + 1])
            lower = np.min(lows[i - self.swing_length + 1:i + 1])
            
            # high[len] - значення high на len барів назад
            idx_len = i - self.swing_length
            if idx_len >= 0:
                # swingType := high[len] > upper ? 0 : low[len] < lower ? 1 : swingType
                if highs[idx_len] > upper:
                    swing_type = 0  # Swing High detected
                    swing_highs[i] = highs[idx_len]
                elif lows[idx_len] < lower:
                    swing_type = 1  # Swing Low detected
                    swing_lows[i] = lows[idx_len]
            
            swing_types[i] = swing_type
        
        return swing_types, swing_highs, swing_lows


# ============================================================================
#                        ORDER BLOCK DETECTOR
# ============================================================================

class OrderBlockDetector:
    """
    Детекція Order Blocks (100% логіка з Pine Script)
    
    Алгоритм:
    1. Відстежуємо swing points (top/bottom)
    2. При пробитті swing high вгору → Bullish OB (остання ведмежа свічка)
    3. При пробитті swing low вниз → Bearish OB (остання бича свічка)
    4. Перевірка інвалідації на кожному барі
    """
    
    def __init__(
        self,
        swing_length: int = 10,
        max_atr_mult: float = 3.5,
        invalidation_method: InvalidationMethod = InvalidationMethod.WICK,
        combine_obs: bool = True,
        max_order_blocks: int = 30,
        zone_count: str = "Low"
    ):
        self.swing_length = swing_length
        self.max_atr_mult = max_atr_mult
        self.invalidation_method = invalidation_method
        self.combine_obs = combine_obs
        self.max_order_blocks = max_order_blocks
        
        # Zone count mapping
        zone_mapping = {"One": 1, "Low": 3, "Medium": 5, "High": 10}
        self.max_zones = zone_mapping.get(zone_count, 10)
        
        self.swing_detector = SwingDetector(swing_length)
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Розрахунок ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def detect_order_blocks(
        self,
        df: pd.DataFrame,
        direction: Optional[str] = None
    ) -> Tuple[List[OrderBlockInfo], List[OrderBlockInfo]]:
        """
        Детекція Order Blocks
        
        Args:
            df: DataFrame з OHLCV даними
            direction: "BUY" - тільки Bullish, "SELL" - тільки Bearish, None - обидва
            
        Returns:
            Tuple[bullish_obs, bearish_obs]
        """
        if len(df) < self.swing_length + 10:
            logger.warning(f"Not enough data: {len(df)} bars, need {self.swing_length + 10}")
            return [], []
        
        # Розрахунок ATR
        atr = self.calculate_atr(df, 10)
        
        # Отримуємо дані
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        volumes = df['volume'].values
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df))
        
        bullish_obs: List[OrderBlockInfo] = []
        bearish_obs: List[OrderBlockInfo] = []
        
        # Знаходимо свінги - тепер це persistent tracking
        swing_types, swing_highs, swing_lows = self.swing_detector.find_swings(df)
        
        # Поточний swing для відстеження (Pine Script var)
        current_top_y = 0.0
        current_top_x = 0
        current_top_crossed = False
        current_top_volume = 0.0
        
        current_bottom_y = 0.0
        current_bottom_x = 0
        current_bottom_crossed = False
        current_bottom_volume = 0.0
        
        # Проходимо по барах
        for i in range(self.swing_length + 1, len(df)):
            current_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else 0
            
            # Оновлюємо поточні свінги коли знаходимо нові
            idx_len = i - self.swing_length
            if idx_len >= 0:
                # Новий Swing High
                if swing_highs[i] > 0:
                    current_top_y = swing_highs[i]
                    current_top_x = idx_len
                    current_top_crossed = False
                    current_top_volume = volumes[idx_len]
                
                # Новий Swing Low  
                if swing_lows[i] > 0:
                    current_bottom_y = swing_lows[i]
                    current_bottom_x = idx_len
                    current_bottom_crossed = False
                    current_bottom_volume = volumes[idx_len]
            
            # ===== INVALIDATION CHECK FOR EXISTING OBs =====
            
            # Перевірка інвалідації Bullish OBs
            for ob in bullish_obs:
                if not ob.breaker:
                    if self.invalidation_method == InvalidationMethod.WICK:
                        if lows[i] < ob.bottom:
                            ob.breaker = True
                            ob.break_time = int(timestamps[i]) if isinstance(timestamps[i], (int, float, np.integer)) else i
                            ob.bb_volume = volumes[i]
                    else:  # Close method
                        if min(opens[i], closes[i]) < ob.bottom:
                            ob.breaker = True
                            ob.break_time = int(timestamps[i]) if isinstance(timestamps[i], (int, float, np.integer)) else i
                            ob.bb_volume = volumes[i]
            
            # Видаляємо breaker OBs
            bullish_obs = [ob for ob in bullish_obs if not ob.breaker]
            
            # Перевірка інвалідації Bearish OBs
            for ob in bearish_obs:
                if not ob.breaker:
                    if self.invalidation_method == InvalidationMethod.WICK:
                        if highs[i] > ob.top:
                            ob.breaker = True
                            ob.break_time = int(timestamps[i]) if isinstance(timestamps[i], (int, float, np.integer)) else i
                            ob.bb_volume = volumes[i]
                    else:  # Close method
                        if max(opens[i], closes[i]) > ob.top:
                            ob.breaker = True
                            ob.break_time = int(timestamps[i]) if isinstance(timestamps[i], (int, float, np.integer)) else i
                            ob.bb_volume = volumes[i]
            
            # Видаляємо breaker OBs
            bearish_obs = [ob for ob in bearish_obs if not ob.breaker]
            
            # ===== BULLISH OB DETECTION =====
            # Пробиття swing high вгору → шукаємо останню ведмежу свічку
            if direction in [None, "BUY"] and current_top_y > 0 and not current_top_crossed:
                if closes[i] > current_top_y:
                    current_top_crossed = True
                    
                    # Шукаємо свічку з найнижчим low між swing і пробиттям
                    # OB зона = весь діапазон свічки (high-low), не тільки body!
                    box_btm = lows[i-1]
                    box_top = highs[i-1]
                    box_loc = int(timestamps[i-1]) if isinstance(timestamps[i-1], (int, float, np.integer)) else i-1
                    
                    # Проходимо від поточного бару назад до swing
                    search_range = min(i - current_top_x, i)
                    for j in range(1, search_range):
                        idx = i - j
                        if idx < 0:
                            break
                        # Шукаємо свічку з найнижчим low (остання ведмежа перед пробиттям)
                        if lows[idx] < box_btm:
                            box_btm = lows[idx]
                            box_top = highs[idx]
                            box_loc = int(timestamps[idx]) if isinstance(timestamps[idx], (int, float, np.integer)) else idx
                    
                    # Перевірка розміру OB
                    ob_size = box_top - box_btm
                    if current_atr > 0 and ob_size <= current_atr * self.max_atr_mult:
                        # Створюємо Bullish OB
                        new_ob = OrderBlockInfo(
                            top=box_top,
                            bottom=box_btm,
                            ob_volume=volumes[i] + (volumes[i-1] if i >= 1 else 0) + (volumes[i-2] if i >= 2 else 0),
                            ob_type=OBType.BULL,
                            start_time=box_loc,
                            ob_low_volume=volumes[i-2] if i >= 2 else 0,
                            ob_high_volume=volumes[i] + (volumes[i-1] if i >= 1 else 0)
                        )
                        
                        bullish_obs.insert(0, new_ob)
                        if len(bullish_obs) > self.max_order_blocks:
                            bullish_obs.pop()
                        
                        logger.debug(f"Bullish OB detected: top={box_top:.6f}, bottom={box_btm:.6f}, size={ob_size:.6f}, ATR={current_atr:.6f}")
            
            # ===== BEARISH OB DETECTION =====
            # Пробиття swing low вниз → шукаємо останню бичу свічку
            if direction in [None, "SELL"] and current_bottom_y > 0 and not current_bottom_crossed:
                if closes[i] < current_bottom_y:
                    current_bottom_crossed = True
                    
                    # Шукаємо свічку з найвищим high між swing і пробиттям
                    # OB зона = весь діапазон свічки (high-low), не тільки body!
                    box_top = highs[i-1]
                    box_btm = lows[i-1]
                    box_loc = int(timestamps[i-1]) if isinstance(timestamps[i-1], (int, float, np.integer)) else i-1
                    
                    # Проходимо від поточного бару назад до swing
                    search_range = min(i - current_bottom_x, i)
                    for j in range(1, search_range):
                        idx = i - j
                        if idx < 0:
                            break
                        # Шукаємо свічку з найвищим high (остання бича перед пробиттям)
                        if highs[idx] > box_top:
                            box_top = highs[idx]
                            box_btm = lows[idx]
                            box_loc = int(timestamps[idx]) if isinstance(timestamps[idx], (int, float, np.integer)) else idx
                    
                    # Перевірка розміру OB
                    ob_size = box_top - box_btm
                    if current_atr > 0 and ob_size <= current_atr * self.max_atr_mult:
                        # Створюємо Bearish OB
                        new_ob = OrderBlockInfo(
                            top=box_top,
                            bottom=box_btm,
                            ob_volume=volumes[i] + (volumes[i-1] if i >= 1 else 0) + (volumes[i-2] if i >= 2 else 0),
                            ob_type=OBType.BEAR,
                            start_time=box_loc,
                            ob_low_volume=volumes[i-2] if i >= 2 else 0,
                            ob_high_volume=volumes[i] + (volumes[i-1] if i >= 1 else 0)
                        )
                        
                        bearish_obs.insert(0, new_ob)
                        if len(bearish_obs) > self.max_order_blocks:
                            bearish_obs.pop()
                        
                        logger.debug(f"Bearish OB detected: top={box_top:.6f}, bottom={box_btm:.6f}, size={ob_size:.6f}, ATR={current_atr:.6f}")
        
        # Фільтруємо тільки валідні OBs (не breaker)
        valid_bullish = [ob for ob in bullish_obs if not ob.breaker]
        valid_bearish = [ob for ob in bearish_obs if not ob.breaker]
        
        # Combine overlapping OBs якщо увімкнено
        if self.combine_obs:
            valid_bullish = self._combine_overlapping_obs(valid_bullish)
            valid_bearish = self._combine_overlapping_obs(valid_bearish)
            logger.debug(f"After combine: {len(valid_bullish)} bullish, {len(valid_bearish)} bearish")
        
        # Обмежуємо кількість по Zone Count
        valid_bullish = valid_bullish[:self.max_zones]
        valid_bearish = valid_bearish[:self.max_zones]
        
        logger.info(f"OB Detection complete: {len(valid_bullish)} bullish, {len(valid_bearish)} bearish")
        
        return valid_bullish, valid_bearish
    
    def _combine_overlapping_obs(self, obs: List[OrderBlockInfo]) -> List[OrderBlockInfo]:
        """Комбінування перекриваючих Order Blocks"""
        if len(obs) <= 1:
            return obs
        
        combined = []
        used = set()
        
        for i, ob1 in enumerate(obs):
            if i in used:
                continue
            
            current_ob = ob1
            
            for j, ob2 in enumerate(obs):
                if i == j or j in used:
                    continue
                
                # Перевірка перекриття
                if self._obs_overlap(current_ob, ob2):
                    # Об'єднуємо
                    current_ob = self._merge_obs(current_ob, ob2)
                    used.add(j)
            
            combined.append(current_ob)
            used.add(i)
        
        return combined
    
    def _obs_overlap(self, ob1: OrderBlockInfo, ob2: OrderBlockInfo) -> bool:
        """Перевірка чи два OB перекриваються"""
        return not (ob1.bottom > ob2.top or ob2.bottom > ob1.top)
    
    def _merge_obs(self, ob1: OrderBlockInfo, ob2: OrderBlockInfo) -> OrderBlockInfo:
        """Об'єднання двох OB"""
        return OrderBlockInfo(
            top=max(ob1.top, ob2.top),
            bottom=min(ob1.bottom, ob2.bottom),
            ob_volume=ob1.ob_volume + ob2.ob_volume,
            ob_type=ob1.ob_type,
            start_time=min(ob1.start_time, ob2.start_time),
            ob_low_volume=ob1.ob_low_volume + ob2.ob_low_volume,
            ob_high_volume=ob1.ob_high_volume + ob2.ob_high_volume,
            combined=True
        )
    
    def check_retest(self, ob: OrderBlockInfo, current_price: float, high: float, low: float) -> bool:
        """
        Перевірка ретесту OB зони
        
        Args:
            ob: Order Block
            current_price: поточна ціна
            high: high поточного бару
            low: low поточного бару
            
        Returns:
            True якщо ціна торкнулась зони
        """
        if not ob.is_valid() or ob.touched:
            return False
        
        # Ціна входить в зону
        price_enters_zone = (low <= ob.top and high >= ob.bottom)
        
        if price_enters_zone:
            ob.touched = True
            return True
        
        return False


# ============================================================================
#                        ORDER BLOCK SCANNER
# ============================================================================

class OrderBlockScanner:
    """
    Головний сканер Order Blocks для watchlist
    """
    
    def __init__(
        self,
        session,  # Bybit session
        settings: Dict[str, Any]
    ):
        self.session = session
        self.settings = settings
        
        # Параметри з налаштувань
        self.source_tf = settings.get('ob_source_tf', '15')
        self.swing_length = settings.get('ob_swing_length', 3)
        self.zone_count = settings.get('ob_zone_count', 'High')
        self.max_atr_mult = settings.get('ob_max_atr_mult', 3.5)
        self.invalidation_method = InvalidationMethod(settings.get('ob_invalidation_method', 'Wick'))
        self.combine_obs = settings.get('ob_combine_obs', True)
        self.entry_mode = EntryMode(settings.get('ob_entry_mode', 'Immediate'))
        self.ob_selection = OBSelection(settings.get('ob_selection', 'Newest'))
        self.sl_atr_mult = settings.get('ob_sl_atr_mult', 0.3)
        self.persistence_check = settings.get('ob_persistence_check', False)
        
        # Детектор
        self.detector = OrderBlockDetector(
            swing_length=self.swing_length,
            max_atr_mult=self.max_atr_mult,
            invalidation_method=self.invalidation_method,
            combine_obs=self.combine_obs,
            zone_count=self.zone_count
        )
    
    def _get_interval(self) -> str:
        """Конвертація таймфрейму для Bybit API"""
        tf_map = {
            '1': '1', '3': '3', '5': '5', '15': '15', '30': '30',
            '60': '60', '120': '120', '240': '240', '360': '360',
            '720': '720', 'D': 'D', 'W': 'W', 'M': 'M'
        }
        return tf_map.get(self.source_tf, '15')
    
    def _fetch_klines(self, symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
        """Отримання свічок з біржі"""
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=self._get_interval(),
                limit=limit
            )
            
            if response.get('retCode') != 0:
                logger.warning(f"Klines error for {symbol}: {response.get('retMsg')}")
                return None
            
            klines = response.get('result', {}).get('list', [])
            if not klines:
                return None
            
            # Конвертація в DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype({
                'timestamp': 'int64',
                'open': 'float64',
                'high': 'float64',
                'low': 'float64',
                'close': 'float64',
                'volume': 'float64'
            })
            
            # Сортування від старих до нових
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch klines for {symbol}: {e}")
            return None
    
    def scan_symbol(
        self,
        symbol: str,
        direction: str
    ) -> Optional[Dict[str, Any]]:
        """
        Сканування одного символу на наявність Order Block
        
        Args:
            symbol: Символ (напр. "BTCUSDT")
            direction: "BUY" або "SELL"
            
        Returns:
            Dict з інформацією про знайдений OB або None
        """
        # Отримуємо дані
        df = self._fetch_klines(symbol)
        if df is None or len(df) < 50:
            return None
        
        # Детекція OB
        bullish_obs, bearish_obs = self.detector.detect_order_blocks(df, direction)
        
        # Вибираємо потрібні OB за direction
        obs = bullish_obs if direction == "BUY" else bearish_obs
        
        # Фільтруємо тільки валідні
        valid_obs = [ob for ob in obs if ob.is_valid()]
        
        if not valid_obs:
            return None
        
        # Вибираємо OB за налаштуванням
        current_price = df['close'].iloc[-1]
        
        if self.ob_selection == OBSelection.NEWEST:
            selected_ob = valid_obs[0]  # Перший = найновіший
        else:  # CLOSEST
            selected_ob = min(valid_obs, key=lambda ob: abs((ob.top + ob.bottom) / 2 - current_price))
        
        # Розрахунок ATR для SL
        atr = self.detector.calculate_atr(df, 10).iloc[-1]
        
        # Розрахунок SL
        if direction == "BUY":
            entry_price = selected_ob.top
            sl_price = selected_ob.bottom - (atr * self.sl_atr_mult)
        else:
            entry_price = selected_ob.bottom
            sl_price = selected_ob.top + (atr * self.sl_atr_mult)
        
        # Перевірка Entry Mode
        if self.entry_mode == EntryMode.RETEST:
            # Перевіряємо чи ціна вже в зоні
            high = df['high'].iloc[-1]
            low = df['low'].iloc[-1]
            if not self.detector.check_retest(selected_ob, current_price, high, low):
                # OB знайдено, але чекаємо retest
                return {
                    'symbol': symbol,
                    'direction': direction,
                    'ob_type': selected_ob.ob_type.value,
                    'ob_top': selected_ob.top,
                    'ob_bottom': selected_ob.bottom,
                    'entry_price': entry_price,
                    'sl_price': sl_price,
                    'current_price': current_price,
                    'atr': atr,
                    'status': OBStatus.WAITING_RETEST.value,
                    'ob_info': selected_ob.to_dict()
                }
        
        # OB знайдено і готовий до входу
        return {
            'symbol': symbol,
            'direction': direction,
            'ob_type': selected_ob.ob_type.value,
            'ob_top': selected_ob.top,
            'ob_bottom': selected_ob.bottom,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'current_price': current_price,
            'atr': atr,
            'status': OBStatus.VALID.value if self.entry_mode == EntryMode.IMMEDIATE else OBStatus.TRIGGERED.value,
            'ob_info': selected_ob.to_dict()
        }
    
    def scan_watchlist(
        self,
        watchlist: List[Dict[str, Any]],
        delay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Сканування всього watchlist
        
        Args:
            watchlist: Список символів з direction
            delay: Затримка між запитами (секунди)
            
        Returns:
            Список знайдених OB
        """
        import time
        
        results = []
        
        for item in watchlist:
            symbol = item.get('symbol')
            direction = item.get('direction')
            
            if not symbol or not direction:
                continue
            
            result = self.scan_symbol(symbol, direction)
            
            if result:
                results.append(result)
                logger.info(f"OB found: {symbol} {direction} - {result['status']}")
            
            time.sleep(delay)
        
        return results


# ============================================================================
#                              UTILITY FUNCTIONS
# ============================================================================

def format_timeframe(tf: str) -> str:
    """Форматування таймфрейму для відображення"""
    if tf in ['D', 'W', 'M']:
        return {'D': '1D', 'W': '1W', 'M': '1M'}.get(tf, tf)
    
    try:
        minutes = int(tf)
        if minutes >= 60:
            hours = minutes // 60
            return f"{hours}H"
        return f"{minutes}m"
    except:
        return tf


def get_scan_interval_seconds(tf: str) -> int:
    """Отримання інтервалу сканування в секундах"""
    tf_seconds = {
        '1': 60, '3': 180, '5': 300, '15': 900, '30': 1800,
        '60': 3600, '120': 7200, '240': 14400, '360': 21600,
        '720': 43200, 'D': 86400, 'W': 604800
    }
    return tf_seconds.get(tf, 900)  # Default 15m
