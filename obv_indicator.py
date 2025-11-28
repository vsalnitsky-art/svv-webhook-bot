"""
OBV (On-Balance Volume) Indicator
Используется для подтверждения трендов при автозакрытии позиций
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OBVIndicator:
    """
    On-Balance Volume индикатор для анализа объемов
    """
    
    def __init__(self, ema_period: int = 20):
        """
        Args:
            ema_period: период для EMA расчёта (default: 20)
        """
        self.ema_period = ema_period
    
    def calculate_obv(self, candles: List[Dict]) -> np.ndarray:
        """
        Рассчитать On-Balance Volume
        
        OBV логика:
        - Если close > close_prev: OBV = OBV_prev + volume
        - Если close < close_prev: OBV = OBV_prev - volume
        - Если close == close_prev: OBV = OBV_prev
        
        Args:
            candles: список свечей с полями close, volume
            
        Returns:
            numpy array с значениями OBV
        """
        if not candles or len(candles) < 2:
            return np.array([0])
        
        obv = [0]  # Начальное значение
        
        for i in range(1, len(candles)):
            try:
                current_close = float(candles[i]['close'])
                prev_close = float(candles[i-1]['close'])
                volume = float(candles[i]['volume'])
                
                if current_close > prev_close:
                    # Цена выросла - прибавляем объем
                    obv.append(obv[-1] + volume)
                elif current_close < prev_close:
                    # Цена упала - вычитаем объем
                    obv.append(obv[-1] - volume)
                else:
                    # Цена не изменилась
                    obv.append(obv[-1])
                    
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Error calculating OBV at index {i}: {e}")
                obv.append(obv[-1] if obv else 0)
        
        return np.array(obv)
    
    def calculate_ema(self, data: np.ndarray, period: int = None) -> np.ndarray:
        """
        Рассчитать EMA (Exponential Moving Average)
        
        Args:
            data: массив данных
            period: период EMA (если None, использует self.ema_period)
            
        Returns:
            numpy array с значениями EMA
        """
        if period is None:
            period = self.ema_period
        
        if len(data) < period:
            logger.warning(f"Not enough data for EMA({period}): {len(data)} candles")
            return np.full(len(data), data[-1] if len(data) > 0 else 0)
        
        # Pandas EMA для точности
        df = pd.DataFrame({'obv': data})
        ema = df['obv'].ewm(span=period, adjust=False).mean()
        
        return ema.values
    
    def check_trend(self, 
                   candles: List[Dict], 
                   direction: str,
                   n_candles: int = 3,
                   sensitivity: str = 'medium') -> Tuple[bool, str, Dict]:
        """
        Проверить тренд OBV для закрытия позиции
        
        Args:
            candles: список свечей
            direction: 'down' для Long, 'up' для Short
            n_candles: количество свечей для проверки тренда
            sensitivity: 'low', 'medium', 'high'
            
        Returns:
            (should_close, reason, details)
            - should_close: True если нужно закрыть
            - reason: причина закрытия
            - details: детали для логирования
        """
        if len(candles) < self.ema_period + n_candles:
            return False, "Not enough data", {}
        
        # Рассчитать OBV и EMA
        obv = self.calculate_obv(candles)
        obv_ema = self.calculate_ema(obv)
        
        # Текущие значения
        current_obv = obv[-1]
        current_ema = obv_ema[-1]
        prev_obv = obv[-2]
        prev_ema = obv_ema[-2]
        
        # Детали для логирования
        details = {
            'current_obv': float(current_obv),
            'current_ema': float(current_ema),
            'obv_above_ema': current_obv > current_ema,
            'direction': direction,
            'n_candles': n_candles
        }
        
        # Проверка 1: Пересечение EMA
        crossover_detected = False
        
        if direction == 'down':
            # Long позиция: OBV пересекает EMA вниз
            if prev_obv >= prev_ema and current_obv < current_ema:
                crossover_detected = True
                details['crossover'] = 'down'
                logger.info(f"OBV crossed EMA down: {current_obv:.2f} < {current_ema:.2f}")
        
        elif direction == 'up':
            # Short позиция: OBV пересекает EMA вверх
            if prev_obv <= prev_ema and current_obv > current_ema:
                crossover_detected = True
                details['crossover'] = 'up'
                logger.info(f"OBV crossed EMA up: {current_obv:.2f} > {current_ema:.2f}")
        
        if crossover_detected:
            return True, f"OBV crossed EMA {direction}", details
        
        # Проверка 2: N свечей подряд
        # Адаптируем n_candles в зависимости от sensitivity
        candles_required = {
            'low': 5,
            'medium': 3,
            'high': 2
        }.get(sensitivity, 3)
        
        recent_obv = obv[-candles_required:]
        details['candles_checked'] = candles_required
        
        if direction == 'down':
            # Long: OBV падает N свечей подряд
            is_falling = all(
                recent_obv[i] > recent_obv[i+1] 
                for i in range(len(recent_obv)-1)
            )
            
            if is_falling:
                details['consecutive_down'] = candles_required
                logger.info(f"OBV falling {candles_required} candles in a row")
                return True, f"OBV falling {candles_required} candles", details
        
        elif direction == 'up':
            # Short: OBV растёт N свечей подряд
            is_rising = all(
                recent_obv[i] < recent_obv[i+1] 
                for i in range(len(recent_obv)-1)
            )
            
            if is_rising:
                details['consecutive_up'] = candles_required
                logger.info(f"OBV rising {candles_required} candles in a row")
                return True, f"OBV rising {candles_required} candles", details
        
        # Не сработало ни одно условие
        details['status'] = 'monitoring'
        return False, "OBV monitoring", details
    
    def get_current_state(self, candles: List[Dict]) -> Dict:
        """
        Получить текущее состояние OBV для отображения
        
        Returns:
            {
                'obv': current OBV value,
                'ema': current EMA value,
                'trend': 'bullish'/'bearish'/'neutral',
                'strength': 'weak'/'medium'/'strong'
            }
        """
        if len(candles) < self.ema_period + 1:
            return {
                'obv': 0,
                'ema': 0,
                'trend': 'neutral',
                'strength': 'weak'
            }
        
        obv = self.calculate_obv(candles)
        obv_ema = self.calculate_ema(obv)
        
        current_obv = obv[-1]
        current_ema = obv_ema[-1]
        
        # Определить тренд
        if current_obv > current_ema * 1.02:  # >2% выше
            trend = 'bullish'
        elif current_obv < current_ema * 0.98:  # >2% ниже
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Определить силу тренда
        diff_percent = abs((current_obv - current_ema) / current_ema * 100)
        
        if diff_percent > 10:
            strength = 'strong'
        elif diff_percent > 5:
            strength = 'medium'
        else:
            strength = 'weak'
        
        return {
            'obv': float(current_obv),
            'ema': float(current_ema),
            'trend': trend,
            'strength': strength,
            'diff_percent': float(diff_percent)
        }


# Удобная функция для быстрого использования
def check_obv_signal(candles: List[Dict], 
                     direction: str,
                     ema_period: int = 20,
                     n_candles: int = 3,
                     sensitivity: str = 'medium') -> Tuple[bool, str, Dict]:
    """
    Быстрая проверка OBV сигнала
    
    Args:
        candles: список свечей
        direction: 'down' для Long, 'up' для Short
        ema_period: период EMA (default: 20)
        n_candles: количество свечей (default: 3)
        sensitivity: чувствительность (default: 'medium')
        
    Returns:
        (should_close, reason, details)
    """
    indicator = OBVIndicator(ema_period=ema_period)
    return indicator.check_trend(candles, direction, n_candles, sensitivity)
