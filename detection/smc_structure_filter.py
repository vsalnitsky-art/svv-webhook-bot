"""
SMC Structure Filter v1.0

Точна реалізація логіки з TradingView індикатора:
- HH (Higher High)
- HL (Higher Low)  
- LH (Lower High)
- LL (Lower Low)
- Strong/Weak High/Low

Використовується як фільтр для CTR сигналів:
- BUY валідний біля Strong Low або HL
- SELL валідний біля Weak High або LH
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from enum import Enum


class TrendBias(Enum):
    """Напрямок тренду"""
    NEUTRAL = 0
    BULLISH = 1
    BEARISH = -1


class SwingType(Enum):
    """Тип свінгової точки"""
    HH = "HH"  # Higher High
    HL = "HL"  # Higher Low
    LH = "LH"  # Lower High
    LL = "LL"  # Lower Low


class ExtremeType(Enum):
    """Тип екстремуму"""
    STRONG_HIGH = "Strong High"
    WEAK_HIGH = "Weak High"
    STRONG_LOW = "Strong Low"
    WEAK_LOW = "Weak Low"


@dataclass
class SwingPoint:
    """Свінгова точка"""
    price: float
    bar_index: int
    swing_type: Optional[SwingType] = None
    extreme_type: Optional[ExtremeType] = None
    timestamp: Optional[datetime] = None


@dataclass
class SMCPivot:
    """Півот (аналог smcPivot з Pine Script)"""
    current_level: float = 0.0
    last_level: float = 0.0
    crossed: bool = False
    bar_index: int = 0


@dataclass
class SMCStructure:
    """Поточна SMC структура"""
    # Свінгові точки
    swing_high: SMCPivot = field(default_factory=SMCPivot)
    swing_low: SMCPivot = field(default_factory=SMCPivot)
    
    # Тренд
    trend_bias: TrendBias = TrendBias.NEUTRAL
    
    # Трейлінг екстремуми
    trailing_top: float = 0.0
    trailing_bottom: float = float('inf')
    
    # Останні точки
    last_hh: Optional[SwingPoint] = None
    last_hl: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None
    last_ll: Optional[SwingPoint] = None
    
    # Сильні/слабкі екстремуми
    strong_high: Optional[SwingPoint] = None
    weak_high: Optional[SwingPoint] = None
    strong_low: Optional[SwingPoint] = None
    weak_low: Optional[SwingPoint] = None
    
    # Історія свінгів для аналізу
    swing_history: List[SwingPoint] = field(default_factory=list)


class SMCStructureDetector:
    """
    Детектор SMC структури - точна реалізація з TradingView
    
    Алгоритм:
    1. Визначення свінгових точок (pivot high/low)
    2. Класифікація HH/HL/LH/LL
    3. Визначення тренду через BOS/CHoCH
    4. Маркування Strong/Weak екстремумів
    """
    
    # Константи (як в Pine Script)
    BULLISH_LEG = 1
    BEARISH_LEG = 0
    
    def __init__(self, swing_length: int = 50):
        """
        Args:
            swing_length: Довжина свінга для визначення структури (smcSwingsLengthInput)
        """
        self.swing_length = swing_length
        self.structure = SMCStructure()
        self._last_leg = 0
        self._initialized = False
    
    def _get_leg(self, highs: np.ndarray, lows: np.ndarray, size: int) -> int:
        """
        Визначення поточної "ноги" тренду (smcLeg з Pine Script)
        
        high[size] > ta.highest(size) → BEARISH_LEG
        low[size] < ta.lowest(size)  → BULLISH_LEG
        """
        if len(highs) <= size or len(lows) <= size:
            return self._last_leg
        
        # high[size] - це highs[-(size+1)] в Python (зсув на size свічок назад)
        pivot_high = highs[-(size + 1)]
        pivot_low = lows[-(size + 1)]
        
        # Останні size свічок (без pivot свічки)
        recent_highs = highs[-size:]
        recent_lows = lows[-size:]
        
        highest = np.max(recent_highs)
        lowest = np.min(recent_lows)
        
        # Визначаємо ногу
        if pivot_high > highest:
            return self.BEARISH_LEG  # Верхній півот
        elif pivot_low < lowest:
            return self.BULLISH_LEG  # Нижній півот
        
        return self._last_leg
    
    def _start_of_new_leg(self, current_leg: int) -> bool:
        """ta.change(leg) != 0"""
        return current_leg != self._last_leg
    
    def _start_of_bullish_leg(self, current_leg: int) -> bool:
        """ta.change(leg) == +1 (перехід до бичої ноги = нижній півот)"""
        return current_leg == self.BULLISH_LEG and self._last_leg == self.BEARISH_LEG
    
    def _start_of_bearish_leg(self, current_leg: int) -> bool:
        """ta.change(leg) == -1 (перехід до ведмежої ноги = верхній півот)"""
        return current_leg == self.BEARISH_LEG and self._last_leg == self.BULLISH_LEG
    
    def update(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """
        Оновити SMC структуру на основі нових даних
        
        Returns:
            Dict з результатами аналізу
        """
        result = {
            'new_swing': False,
            'swing_type': None,
            'extreme_type': None,
            'trend_bias': self.structure.trend_bias,
            'structure': self.structure
        }
        
        if len(highs) < self.swing_length + 10:
            return result
        
        size = self.swing_length
        current_leg = self._get_leg(highs, lows, size)
        
        # Ініціалізація при першому запуску
        if not self._initialized:
            self.structure.swing_high.current_level = np.max(highs[-size:])
            self.structure.swing_low.current_level = np.min(lows[-size:])
            self.structure.trailing_top = self.structure.swing_high.current_level
            self.structure.trailing_bottom = self.structure.swing_low.current_level
            self._initialized = True
            self._last_leg = current_leg
            return result
        
        # Перевіряємо чи є новий півот
        new_pivot = self._start_of_new_leg(current_leg)
        pivot_low = self._start_of_bullish_leg(current_leg)
        pivot_high = self._start_of_bearish_leg(current_leg)
        
        bar_index = len(highs) - size - 1
        
        if new_pivot:
            result['new_swing'] = True
            
            if pivot_low:
                # Новий нижній півот
                pivot_price = lows[-(size + 1)]
                
                # Визначаємо тип: LL чи HL
                is_ll = pivot_price < self.structure.swing_low.current_level
                swing_type = SwingType.LL if is_ll else SwingType.HL
                
                # Оновлюємо структуру
                self.structure.swing_low.last_level = self.structure.swing_low.current_level
                self.structure.swing_low.current_level = pivot_price
                self.structure.swing_low.crossed = False
                self.structure.swing_low.bar_index = bar_index
                
                # Оновлюємо trailing
                self.structure.trailing_bottom = pivot_price
                
                # Створюємо SwingPoint
                swing_point = SwingPoint(
                    price=pivot_price,
                    bar_index=bar_index,
                    swing_type=swing_type
                )
                
                # Зберігаємо
                if is_ll:
                    self.structure.last_ll = swing_point
                else:
                    self.structure.last_hl = swing_point
                
                self.structure.swing_history.append(swing_point)
                
                # Визначаємо Strong/Weak Low
                # Strong Low = trend is BULLISH
                if self.structure.trend_bias == TrendBias.BULLISH:
                    swing_point.extreme_type = ExtremeType.STRONG_LOW
                    self.structure.strong_low = swing_point
                else:
                    swing_point.extreme_type = ExtremeType.WEAK_LOW
                    self.structure.weak_low = swing_point
                
                result['swing_type'] = swing_type
                result['extreme_type'] = swing_point.extreme_type
                
            else:  # pivot_high
                # Новий верхній півот
                pivot_price = highs[-(size + 1)]
                
                # Визначаємо тип: HH чи LH
                is_hh = pivot_price > self.structure.swing_high.current_level
                swing_type = SwingType.HH if is_hh else SwingType.LH
                
                # Оновлюємо структуру
                self.structure.swing_high.last_level = self.structure.swing_high.current_level
                self.structure.swing_high.current_level = pivot_price
                self.structure.swing_high.crossed = False
                self.structure.swing_high.bar_index = bar_index
                
                # Оновлюємо trailing
                self.structure.trailing_top = pivot_price
                
                # Створюємо SwingPoint
                swing_point = SwingPoint(
                    price=pivot_price,
                    bar_index=bar_index,
                    swing_type=swing_type
                )
                
                # Зберігаємо
                if is_hh:
                    self.structure.last_hh = swing_point
                else:
                    self.structure.last_lh = swing_point
                
                self.structure.swing_history.append(swing_point)
                
                # Визначаємо Strong/Weak High
                # Strong High = trend is BEARISH
                if self.structure.trend_bias == TrendBias.BEARISH:
                    swing_point.extreme_type = ExtremeType.STRONG_HIGH
                    self.structure.strong_high = swing_point
                else:
                    swing_point.extreme_type = ExtremeType.WEAK_HIGH
                    self.structure.weak_high = swing_point
                
                result['swing_type'] = swing_type
                result['extreme_type'] = swing_point.extreme_type
        
        # Перевіряємо BOS/CHoCH для зміни тренду
        current_close = closes[-1]
        
        # Crossover swing_high → BULLISH
        if not self.structure.swing_high.crossed:
            if current_close > self.structure.swing_high.current_level:
                self.structure.swing_high.crossed = True
                self.structure.trend_bias = TrendBias.BULLISH
                result['trend_bias'] = TrendBias.BULLISH
        
        # Crossunder swing_low → BEARISH
        if not self.structure.swing_low.crossed:
            if current_close < self.structure.swing_low.current_level:
                self.structure.swing_low.crossed = True
                self.structure.trend_bias = TrendBias.BEARISH
                result['trend_bias'] = TrendBias.BEARISH
        
        # Оновлюємо trailing екстремуми
        current_high = highs[-1]
        current_low = lows[-1]
        
        if current_high > self.structure.trailing_top:
            self.structure.trailing_top = current_high
        if current_low < self.structure.trailing_bottom:
            self.structure.trailing_bottom = current_low
        
        # Зберігаємо поточну ногу для наступної ітерації
        self._last_leg = current_leg
        
        # Обмежуємо історію
        if len(self.structure.swing_history) > 100:
            self.structure.swing_history = self.structure.swing_history[-50:]
        
        result['structure'] = self.structure
        return result
    
    def get_nearest_levels(self) -> Dict[str, Optional[float]]:
        """
        Отримати найближчі рівні для фільтрації
        """
        return {
            'swing_high': self.structure.swing_high.current_level,
            'swing_low': self.structure.swing_low.current_level,
            'trailing_top': self.structure.trailing_top,
            'trailing_bottom': self.structure.trailing_bottom,
            'last_hh': self.structure.last_hh.price if self.structure.last_hh else None,
            'last_hl': self.structure.last_hl.price if self.structure.last_hl else None,
            'last_lh': self.structure.last_lh.price if self.structure.last_lh else None,
            'last_ll': self.structure.last_ll.price if self.structure.last_ll else None,
            'strong_high': self.structure.strong_high.price if self.structure.strong_high else None,
            'weak_high': self.structure.weak_high.price if self.structure.weak_high else None,
            'strong_low': self.structure.strong_low.price if self.structure.strong_low else None,
            'weak_low': self.structure.weak_low.price if self.structure.weak_low else None,
        }


class SMCSignalFilter:
    """
    Фільтр CTR сигналів на основі SMC структури
    
    Логіка:
    - BUY валідний якщо ціна біля Strong Low або HL (зона дисконту)
    - SELL валідний якщо ціна біля Weak High або LH (зона преміум)
    """
    
    def __init__(
        self,
        swing_length: int = 50,
        zone_threshold_percent: float = 1.0,  # Відсоток відхилення від рівня
        use_premium_discount: bool = True,    # Використовувати зони преміум/дисконт
    ):
        """
        Args:
            swing_length: Довжина свінга для SMC
            zone_threshold_percent: % від ціни для визначення "біля рівня"
            use_premium_discount: Чи використовувати логіку преміум/дисконт
        """
        self.detector = SMCStructureDetector(swing_length=swing_length)
        self.zone_threshold_percent = zone_threshold_percent
        self.use_premium_discount = use_premium_discount
        self._enabled = True
    
    @property
    def enabled(self) -> bool:
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = value
    
    def update_structure(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict:
        """Оновити SMC структуру"""
        return self.detector.update(highs, lows, closes)
    
    def _is_near_level(self, price: float, level: Optional[float]) -> bool:
        """Перевірити чи ціна біля рівня"""
        if level is None or level == 0:
            return False
        
        threshold = level * (self.zone_threshold_percent / 100)
        return abs(price - level) <= threshold
    
    def _is_in_discount_zone(self, price: float) -> bool:
        """
        Перевірити чи ціна в зоні дисконту (нижня половина діапазону)
        """
        structure = self.detector.structure
        range_size = structure.trailing_top - structure.trailing_bottom
        
        if range_size <= 0:
            return False
        
        equilibrium = structure.trailing_bottom + range_size * 0.5
        return price < equilibrium
    
    def _is_in_premium_zone(self, price: float) -> bool:
        """
        Перевірити чи ціна в зоні преміум (верхня половина діапазону)
        """
        structure = self.detector.structure
        range_size = structure.trailing_top - structure.trailing_bottom
        
        if range_size <= 0:
            return False
        
        equilibrium = structure.trailing_bottom + range_size * 0.5
        return price > equilibrium
    
    def validate_buy_signal(self, current_price: float) -> Tuple[bool, str]:
        """
        Валідувати BUY сигнал
        
        BUY валідний якщо:
        1. Ціна біля Strong Low (сильний мінімум при бичому тренді)
        2. АБО ціна біля HL (higher low)
        3. АБО ціна в зоні дисконту (якщо увімкнено)
        
        Returns:
            (is_valid, reason)
        """
        if not self._enabled:
            return True, "Filter disabled"
        
        structure = self.detector.structure
        levels = self.detector.get_nearest_levels()
        
        reasons = []
        
        # Перевірка Strong Low
        if self._is_near_level(current_price, levels['strong_low']):
            return True, f"Near Strong Low ({levels['strong_low']:.4f})"
        
        # Перевірка HL
        if self._is_near_level(current_price, levels['last_hl']):
            return True, f"Near HL ({levels['last_hl']:.4f})"
        
        # Перевірка Swing Low
        if self._is_near_level(current_price, levels['swing_low']):
            reasons.append(f"Near Swing Low ({levels['swing_low']:.4f})")
        
        # Перевірка зони дисконту
        if self.use_premium_discount and self._is_in_discount_zone(current_price):
            # Тренд має бути бичим для покупки в дисконті
            if structure.trend_bias == TrendBias.BULLISH:
                return True, "In Discount Zone + Bullish Trend"
            else:
                reasons.append("In Discount Zone (but not bullish trend)")
        
        # Якщо жодна умова не виконана
        if reasons:
            return False, f"Rejected: {'; '.join(reasons)}"
        
        return False, f"Not near support levels (Strong Low: {levels['strong_low']}, HL: {levels['last_hl']})"
    
    def validate_sell_signal(self, current_price: float) -> Tuple[bool, str]:
        """
        Валідувати SELL сигнал
        
        SELL валідний якщо:
        1. Ціна біля Weak High (слабкий максимум - буде пробитий)
        2. АБО ціна біля LH (lower high)
        3. АБО ціна в зоні преміум (якщо увімкнено)
        
        Returns:
            (is_valid, reason)
        """
        if not self._enabled:
            return True, "Filter disabled"
        
        structure = self.detector.structure
        levels = self.detector.get_nearest_levels()
        
        reasons = []
        
        # Перевірка Weak High (найкраща точка для шорту)
        if self._is_near_level(current_price, levels['weak_high']):
            return True, f"Near Weak High ({levels['weak_high']:.4f})"
        
        # Перевірка LH
        if self._is_near_level(current_price, levels['last_lh']):
            return True, f"Near LH ({levels['last_lh']:.4f})"
        
        # Перевірка Strong High (можна шортити якщо тренд ведмежий)
        if self._is_near_level(current_price, levels['strong_high']):
            if structure.trend_bias == TrendBias.BEARISH:
                return True, f"Near Strong High + Bearish Trend ({levels['strong_high']:.4f})"
            else:
                reasons.append(f"Near Strong High but not bearish trend")
        
        # Перевірка Swing High
        if self._is_near_level(current_price, levels['swing_high']):
            reasons.append(f"Near Swing High ({levels['swing_high']:.4f})")
        
        # Перевірка зони преміум
        if self.use_premium_discount and self._is_in_premium_zone(current_price):
            # Тренд має бути ведмежим для продажу в преміумі
            if structure.trend_bias == TrendBias.BEARISH:
                return True, "In Premium Zone + Bearish Trend"
            else:
                reasons.append("In Premium Zone (but not bearish trend)")
        
        # Якщо жодна умова не виконана
        if reasons:
            return False, f"Rejected: {'; '.join(reasons)}"
        
        return False, f"Not near resistance levels (Weak High: {levels['weak_high']}, LH: {levels['last_lh']})"
    
    def get_status(self) -> Dict:
        """Отримати поточний статус фільтра"""
        structure = self.detector.structure
        levels = self.detector.get_nearest_levels()
        
        return {
            'enabled': self._enabled,
            'trend_bias': structure.trend_bias.name,
            'swing_high': levels['swing_high'],
            'swing_low': levels['swing_low'],
            'trailing_top': levels['trailing_top'],
            'trailing_bottom': levels['trailing_bottom'],
            'last_hh': levels['last_hh'],
            'last_hl': levels['last_hl'],
            'last_lh': levels['last_lh'],
            'last_ll': levels['last_ll'],
            'strong_high': levels['strong_high'],
            'weak_high': levels['weak_high'],
            'strong_low': levels['strong_low'],
            'weak_low': levels['weak_low'],
            'swing_count': len(structure.swing_history),
        }


# ============================================
# HELPER FUNCTIONS
# ============================================

def create_smc_filter(
    swing_length: int = 50,
    zone_threshold_percent: float = 1.0,
    enabled: bool = True
) -> SMCSignalFilter:
    """
    Створити SMC фільтр з налаштуваннями
    """
    smc_filter = SMCSignalFilter(
        swing_length=swing_length,
        zone_threshold_percent=zone_threshold_percent
    )
    smc_filter.enabled = enabled
    return smc_filter


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    import random
    
    # Генеруємо тестові дані
    np.random.seed(42)
    n = 200
    
    # Симулюємо ціновий рух
    base_price = 100
    volatility = 2
    
    closes = [base_price]
    for i in range(1, n):
        change = np.random.normal(0, volatility)
        # Додаємо тренд
        if i < 100:
            change += 0.1  # Бичий тренд
        else:
            change -= 0.1  # Ведмежий тренд
        closes.append(closes[-1] + change)
    
    closes = np.array(closes)
    highs = closes + np.abs(np.random.normal(0, volatility/2, n))
    lows = closes - np.abs(np.random.normal(0, volatility/2, n))
    
    # Тестуємо детектор
    detector = SMCStructureDetector(swing_length=10)
    
    print("=" * 60)
    print("SMC Structure Detector Test")
    print("=" * 60)
    
    for i in range(50, n):
        result = detector.update(highs[:i+1], lows[:i+1], closes[:i+1])
        
        if result['new_swing']:
            print(f"\nBar {i}: New swing detected!")
            print(f"  Type: {result['swing_type'].value if result['swing_type'] else 'N/A'}")
            print(f"  Extreme: {result['extreme_type'].value if result['extreme_type'] else 'N/A'}")
            print(f"  Trend: {result['trend_bias'].name}")
    
    print("\n" + "=" * 60)
    print("Final Structure Status")
    print("=" * 60)
    
    status = SMCSignalFilter(swing_length=10)
    for i in range(50, n):
        status.update_structure(highs[:i+1], lows[:i+1], closes[:i+1])
    
    final_status = status.get_status()
    for key, value in final_status.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    # Тестуємо фільтр
    print("\n" + "=" * 60)
    print("Signal Filter Test")
    print("=" * 60)
    
    current_price = closes[-1]
    print(f"Current price: {current_price:.2f}")
    
    buy_valid, buy_reason = status.validate_buy_signal(current_price)
    print(f"BUY signal: {'✅ VALID' if buy_valid else '❌ REJECTED'}")
    print(f"  Reason: {buy_reason}")
    
    sell_valid, sell_reason = status.validate_sell_signal(current_price)
    print(f"SELL signal: {'✅ VALID' if sell_valid else '❌ REJECTED'}")
    print(f"  Reason: {sell_reason}")
