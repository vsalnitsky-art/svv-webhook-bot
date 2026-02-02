"""
SMC Analyzer - Smart Money Concepts Analysis Module

Реалізує концепції інституційної торгівлі:
- Market Structure (BOS/CHoCH)
- Order Blocks (Bullish/Bearish)
- Premium/Discount Zones
- Fair Value Gaps (FVG)
- EQH/EQL (Equal Highs/Lows)

Автор: SVV Bot Team
Версія: 1.0 (2026-02-02)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class StructureSignal(Enum):
    """Типи структурних сигналів"""
    BULLISH_BOS = "BULLISH_BOS"      # Break of Structure вгору (підтвердження тренду)
    BEARISH_BOS = "BEARISH_BOS"      # Break of Structure вниз
    BULLISH_CHOCH = "BULLISH_CHOCH"  # Change of Character вгору (зміна тренду)
    BEARISH_CHOCH = "BEARISH_CHOCH"  # Change of Character вниз
    NONE = "NONE"


class MarketBias(Enum):
    """Напрямок ринкового bias"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class PriceZone(Enum):
    """Зони ціни відносно свінгу"""
    PREMIUM = "PREMIUM"       # > 0.5 (дорого)
    EQUILIBRIUM = "EQUILIBRIUM"  # ~0.5
    DISCOUNT = "DISCOUNT"     # < 0.5 (дешево)


@dataclass
class SwingPoint:
    """Точка свінгу (pivot)"""
    price: float
    index: int
    timestamp: int = 0
    is_high: bool = True
    is_broken: bool = False
    
    def __repr__(self):
        t = "HH" if self.is_high else "LL"
        return f"{t}({self.price:.4f}@{self.index})"


@dataclass
class OrderBlock:
    """Order Block - зона інституційного інтересу"""
    high: float
    low: float
    index: int
    timestamp: int = 0
    is_bullish: bool = True
    is_mitigated: bool = False
    strength: float = 1.0  # Сила OB (0-1)
    
    @property
    def mid_price(self) -> float:
        return (self.high + self.low) / 2
    
    def contains_price(self, price: float) -> bool:
        return self.low <= price <= self.high
    
    def __repr__(self):
        t = "Bull" if self.is_bullish else "Bear"
        return f"{t}OB({self.low:.4f}-{self.high:.4f})"


@dataclass
class FairValueGap:
    """Fair Value Gap - гепи справедливої вартості"""
    high: float
    low: float
    index: int
    is_bullish: bool = True
    is_filled: bool = False


@dataclass 
class SMCAnalysisResult:
    """Результат SMC аналізу"""
    # Структура
    market_bias: MarketBias = MarketBias.NEUTRAL
    structure_signal: StructureSignal = StructureSignal.NONE
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows: List[SwingPoint] = field(default_factory=list)
    
    # Поточний стан структури
    last_hh: Optional[SwingPoint] = None
    last_hl: Optional[SwingPoint] = None
    last_lh: Optional[SwingPoint] = None
    last_ll: Optional[SwingPoint] = None
    
    # Зони
    price_zone: PriceZone = PriceZone.EQUILIBRIUM
    zone_level: float = 0.5  # 0-1, де 0 = low, 1 = high
    
    # Order Blocks
    active_bullish_obs: List[OrderBlock] = field(default_factory=list)
    active_bearish_obs: List[OrderBlock] = field(default_factory=list)
    nearest_bullish_ob: Optional[OrderBlock] = None
    nearest_bearish_ob: Optional[OrderBlock] = None
    price_at_bullish_ob: bool = False
    price_at_bearish_ob: bool = False
    
    # FVG
    active_fvgs: List[FairValueGap] = field(default_factory=list)
    
    # Scores для Direction Engine
    smc_score: float = 0.0  # -1 to +1
    structure_score: float = 0.0
    zone_score: float = 0.0
    ob_score: float = 0.0
    
    # Metadata
    reasons: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'market_bias': self.market_bias.value,
            'structure_signal': self.structure_signal.value,
            'price_zone': self.price_zone.value,
            'zone_level': round(self.zone_level, 3),
            'smc_score': round(self.smc_score, 3),
            'structure_score': round(self.structure_score, 3),
            'zone_score': round(self.zone_score, 3),
            'ob_score': round(self.ob_score, 3),
            'price_at_bullish_ob': self.price_at_bullish_ob,
            'price_at_bearish_ob': self.price_at_bearish_ob,
            'active_bullish_obs': len(self.active_bullish_obs),
            'active_bearish_obs': len(self.active_bearish_obs),
            'reasons': self.reasons,
        }


class SMCAnalyzer:
    """
    Smart Money Concepts Analyzer
    
    Аналізує ринкову структуру за принципами інституційної торгівлі.
    """
    
    # Параметри для пошуку свінгів
    SWING_WINDOW = 50      # Для великих свінгів (глобальна структура)
    INTERNAL_WINDOW = 10   # Для внутрішньої структури (CHoCH)
    
    # Параметри зон
    PREMIUM_THRESHOLD = 0.618   # Вище = Premium
    DISCOUNT_THRESHOLD = 0.382  # Нижче = Discount
    
    # Параметри Order Blocks
    OB_LOOKBACK = 100      # Скільки свічок назад шукати OB
    OB_MIN_STRENGTH = 0.5  # Мінімальна "сила" OB
    
    def __init__(self, config: Optional[Dict] = None):
        if config:
            self.SWING_WINDOW = config.get('swing_window', 50)
            self.INTERNAL_WINDOW = config.get('internal_window', 10)
            self.PREMIUM_THRESHOLD = config.get('premium_threshold', 0.618)
            self.DISCOUNT_THRESHOLD = config.get('discount_threshold', 0.382)
    
    def analyze(self, 
                klines: List[Dict],
                htf_klines: Optional[List[Dict]] = None) -> SMCAnalysisResult:
        """
        Основний метод аналізу
        
        Args:
            klines: Свічки робочого ТФ (1H)
            htf_klines: Свічки старшого ТФ (4H) для глобального bias
        
        Returns:
            SMCAnalysisResult з усіма компонентами аналізу
        """
        result = SMCAnalysisResult()
        
        if not klines or len(klines) < self.SWING_WINDOW + 10:
            result.reasons.append("Insufficient data")
            return result
        
        # Витягуємо ціни
        highs = [float(k['high']) for k in klines]
        lows = [float(k['low']) for k in klines]
        closes = [float(k['close']) for k in klines]
        current_price = closes[-1]
        
        # 1. Знаходимо свінг-точки (Swing Structure)
        swing_highs = self._find_swing_points(highs, lows, self.SWING_WINDOW, is_high=True)
        swing_lows = self._find_swing_points(highs, lows, self.SWING_WINDOW, is_high=False)
        
        result.swing_highs = swing_highs
        result.swing_lows = swing_lows
        
        # 2. Визначаємо структуру HH/HL/LH/LL
        self._classify_structure(result, swing_highs, swing_lows)
        
        # 3. Визначаємо Market Bias
        result.market_bias = self._determine_market_bias(result)
        
        # 4. Детектуємо BOS/CHoCH
        internal_highs = self._find_swing_points(highs, lows, self.INTERNAL_WINDOW, is_high=True)
        internal_lows = self._find_swing_points(highs, lows, self.INTERNAL_WINDOW, is_high=False)
        result.structure_signal = self._detect_structure_break(
            current_price, closes, internal_highs, internal_lows, result.market_bias
        )
        
        # 5. Premium/Discount Zone
        self._calculate_price_zone(result, current_price, swing_highs, swing_lows)
        
        # 6. Order Blocks
        self._find_order_blocks(result, klines, current_price)
        
        # 7. HTF Bias (якщо є дані)
        htf_bias_bonus = 0.0
        if htf_klines and len(htf_klines) >= self.SWING_WINDOW:
            htf_result = self.analyze(htf_klines)  # Рекурсивний виклик для HTF
            if htf_result.market_bias == MarketBias.BULLISH:
                htf_bias_bonus = 0.15
                result.reasons.append("HTF:BULL")
            elif htf_result.market_bias == MarketBias.BEARISH:
                htf_bias_bonus = -0.15
                result.reasons.append("HTF:BEAR")
        
        # 8. Розраховуємо фінальні scores
        self._calculate_scores(result, htf_bias_bonus)
        
        return result
    
    def _find_swing_points(self, 
                           highs: List[float], 
                           lows: List[float],
                           window: int,
                           is_high: bool = True) -> List[SwingPoint]:
        """
        Знаходить pivot points (свінг-точки)
        
        Свінг High: точка, яка вища за всі точки в межах window зліва і справа
        Свінг Low: точка, яка нижча за всі точки в межах window зліва і справа
        """
        pivots = []
        prices = highs if is_high else lows
        half_window = window // 2
        
        for i in range(half_window, len(prices) - half_window):
            is_pivot = True
            
            # Перевіряємо, чи це екстремум у вікні
            for j in range(1, half_window + 1):
                if is_high:
                    # Для high: має бути >= всіх сусідів
                    if prices[i] < prices[i - j] or prices[i] < prices[i + j]:
                        is_pivot = False
                        break
                else:
                    # Для low: має бути <= всіх сусідів
                    if prices[i] > prices[i - j] or prices[i] > prices[i + j]:
                        is_pivot = False
                        break
            
            if is_pivot:
                pivots.append(SwingPoint(
                    price=prices[i],
                    index=i,
                    is_high=is_high
                ))
        
        # Повертаємо останні 10 свінгів
        return pivots[-10:] if len(pivots) > 10 else pivots
    
    def _classify_structure(self, 
                            result: SMCAnalysisResult,
                            swing_highs: List[SwingPoint],
                            swing_lows: List[SwingPoint]):
        """
        Класифікує структуру: HH/HL або LH/LL
        """
        # Аналізуємо highs
        if len(swing_highs) >= 2:
            for i in range(1, len(swing_highs)):
                prev = swing_highs[i-1]
                curr = swing_highs[i]
                if curr.price > prev.price:
                    result.last_hh = curr
                else:
                    result.last_lh = curr
        
        # Аналізуємо lows
        if len(swing_lows) >= 2:
            for i in range(1, len(swing_lows)):
                prev = swing_lows[i-1]
                curr = swing_lows[i]
                if curr.price > prev.price:
                    result.last_hl = curr
                else:
                    result.last_ll = curr
    
    def _determine_market_bias(self, result: SMCAnalysisResult) -> MarketBias:
        """
        Визначає загальний напрямок ринку на основі структури
        
        Bullish: HH + HL (вищі максимуми + вищі мінімуми)
        Bearish: LH + LL (нижчі максимуми + нижчі мінімуми)
        """
        has_hh = result.last_hh is not None
        has_hl = result.last_hl is not None
        has_lh = result.last_lh is not None
        has_ll = result.last_ll is not None
        
        # Підраховуємо кількість бичачих/ведмежих свінгів
        bullish_count = 0
        bearish_count = 0
        
        if len(result.swing_highs) >= 2:
            for i in range(1, len(result.swing_highs)):
                if result.swing_highs[i].price > result.swing_highs[i-1].price:
                    bullish_count += 1
                else:
                    bearish_count += 1
        
        if len(result.swing_lows) >= 2:
            for i in range(1, len(result.swing_lows)):
                if result.swing_lows[i].price > result.swing_lows[i-1].price:
                    bullish_count += 1
                else:
                    bearish_count += 1
        
        # Визначаємо bias
        if bullish_count > bearish_count + 1:
            result.reasons.append(f"Structure:HH/HL({bullish_count})")
            return MarketBias.BULLISH
        elif bearish_count > bullish_count + 1:
            result.reasons.append(f"Structure:LH/LL({bearish_count})")
            return MarketBias.BEARISH
        else:
            result.reasons.append("Structure:RANGING")
            return MarketBias.NEUTRAL
    
    def _detect_structure_break(self,
                                current_price: float,
                                closes: List[float],
                                internal_highs: List[SwingPoint],
                                internal_lows: List[SwingPoint],
                                current_bias: MarketBias) -> StructureSignal:
        """
        Детектує BOS (Break of Structure) та CHoCH (Change of Character)
        
        BOS: Пробій структури в напрямку тренду (підтвердження)
        CHoCH: Пробій структури ПРОТИ тренду (перша ознака розвороту)
        """
        if len(internal_highs) < 2 or len(internal_lows) < 2:
            return StructureSignal.NONE
        
        # Останні два внутрішні свінги
        last_internal_high = internal_highs[-1]
        prev_internal_high = internal_highs[-2]
        last_internal_low = internal_lows[-1]
        prev_internal_low = internal_lows[-2]
        
        # Перевіряємо пробій вгору (Bullish)
        # Умова: поточна ціна закрилася вище останнього внутрішнього high
        if current_price > last_internal_high.price and not last_internal_high.is_broken:
            last_internal_high.is_broken = True
            
            if current_bias == MarketBias.BEARISH:
                # Це CHoCH - перший злам проти тренду!
                return StructureSignal.BULLISH_CHOCH
            else:
                # Це BOS - підтвердження бичачого тренду
                return StructureSignal.BULLISH_BOS
        
        # Перевіряємо пробій вниз (Bearish)
        if current_price < last_internal_low.price and not last_internal_low.is_broken:
            last_internal_low.is_broken = True
            
            if current_bias == MarketBias.BULLISH:
                # Це CHoCH - перший злам проти тренду!
                return StructureSignal.BEARISH_CHOCH
            else:
                # Це BOS - підтвердження ведмежого тренду
                return StructureSignal.BEARISH_BOS
        
        return StructureSignal.NONE
    
    def _calculate_price_zone(self,
                              result: SMCAnalysisResult,
                              current_price: float,
                              swing_highs: List[SwingPoint],
                              swing_lows: List[SwingPoint]):
        """
        Визначає, в якій зоні знаходиться ціна: Premium, Discount, або Equilibrium
        
        Використовуємо останній значний свінг для розрахунку.
        """
        if not swing_highs or not swing_lows:
            return
        
        # Знаходимо останній значущий range
        recent_high = max(s.price for s in swing_highs[-3:]) if swing_highs else current_price
        recent_low = min(s.price for s in swing_lows[-3:]) if swing_lows else current_price
        
        range_height = recent_high - recent_low
        if range_height <= 0:
            return
        
        # Позиція ціни в range (0 = low, 1 = high)
        result.zone_level = (current_price - recent_low) / range_height
        
        # Визначаємо зону
        if result.zone_level >= self.PREMIUM_THRESHOLD:
            result.price_zone = PriceZone.PREMIUM
            result.reasons.append(f"Zone:PREMIUM({result.zone_level:.2f})")
        elif result.zone_level <= self.DISCOUNT_THRESHOLD:
            result.price_zone = PriceZone.DISCOUNT
            result.reasons.append(f"Zone:DISCOUNT({result.zone_level:.2f})")
        else:
            result.price_zone = PriceZone.EQUILIBRIUM
    
    def _find_order_blocks(self,
                           result: SMCAnalysisResult,
                           klines: List[Dict],
                           current_price: float):
        """
        Знаходить Order Blocks
        
        Bullish OB: Остання ведмежа свічка перед імпульсним рухом вгору
        Bearish OB: Остання бичача свічка перед імпульсним рухом вниз
        """
        if len(klines) < 10:
            return
        
        lookback = min(self.OB_LOOKBACK, len(klines) - 5)
        
        for i in range(len(klines) - lookback, len(klines) - 3):
            candle = klines[i]
            next_candles = klines[i+1:i+4]
            
            c_open = float(candle['open'])
            c_close = float(candle['close'])
            c_high = float(candle['high'])
            c_low = float(candle['low'])
            
            is_bearish_candle = c_close < c_open
            is_bullish_candle = c_close > c_open
            
            # Перевіряємо імпульсний рух після свічки
            next_highs = [float(k['high']) for k in next_candles]
            next_lows = [float(k['low']) for k in next_candles]
            next_closes = [float(k['close']) for k in next_candles]
            
            # Bullish OB: ведмежа свічка + імпульс вгору
            if is_bearish_candle:
                impulse_up = all(c > c_high for c in next_closes)
                if impulse_up:
                    ob = OrderBlock(
                        high=c_high,
                        low=c_low,
                        index=i,
                        is_bullish=True,
                        is_mitigated=current_price < c_low  # Пробитий, якщо ціна нижче
                    )
                    if not ob.is_mitigated:
                        result.active_bullish_obs.append(ob)
            
            # Bearish OB: бичача свічка + імпульс вниз  
            if is_bullish_candle:
                impulse_down = all(c < c_low for c in next_closes)
                if impulse_down:
                    ob = OrderBlock(
                        high=c_high,
                        low=c_low,
                        index=i,
                        is_bullish=False,
                        is_mitigated=current_price > c_high  # Пробитий, якщо ціна вище
                    )
                    if not ob.is_mitigated:
                        result.active_bearish_obs.append(ob)
        
        # Знаходимо найближчі OB
        if result.active_bullish_obs:
            # Найближчий bullish OB нижче ціни
            below = [ob for ob in result.active_bullish_obs if ob.high < current_price]
            if below:
                result.nearest_bullish_ob = max(below, key=lambda ob: ob.high)
                # Чи ціна в зоні OB?
                if result.nearest_bullish_ob.contains_price(current_price):
                    result.price_at_bullish_ob = True
                    result.reasons.append("At Bullish OB")
        
        if result.active_bearish_obs:
            # Найближчий bearish OB вище ціни
            above = [ob for ob in result.active_bearish_obs if ob.low > current_price]
            if above:
                result.nearest_bearish_ob = min(above, key=lambda ob: ob.low)
                if result.nearest_bearish_ob.contains_price(current_price):
                    result.price_at_bearish_ob = True
                    result.reasons.append("At Bearish OB")
    
    def _calculate_scores(self, result: SMCAnalysisResult, htf_bias_bonus: float = 0.0):
        """
        Розраховує фінальні scores для інтеграції в Direction Engine
        
        Шкала: -1.0 (сильний SHORT) до +1.0 (сильний LONG)
        """
        # 1. Structure Score
        structure_score = 0.0
        
        if result.structure_signal == StructureSignal.BULLISH_CHOCH:
            structure_score = 0.45  # Найсильніший бичачий сигнал
            result.reasons.append("CHoCH:BULL(+45)")
        elif result.structure_signal == StructureSignal.BULLISH_BOS:
            structure_score = 0.25
            result.reasons.append("BOS:BULL(+25)")
        elif result.structure_signal == StructureSignal.BEARISH_CHOCH:
            structure_score = -0.45
            result.reasons.append("CHoCH:BEAR(-45)")
        elif result.structure_signal == StructureSignal.BEARISH_BOS:
            structure_score = -0.25
            result.reasons.append("BOS:BEAR(-25)")
        
        # Market bias contribution
        if result.market_bias == MarketBias.BULLISH:
            structure_score += 0.10
        elif result.market_bias == MarketBias.BEARISH:
            structure_score -= 0.10
        
        result.structure_score = np.clip(structure_score, -1.0, 1.0)
        
        # 2. Zone Score
        zone_score = 0.0
        if result.price_zone == PriceZone.DISCOUNT:
            zone_score = 0.15  # Хороша зона для LONG
            result.reasons.append("Zone:+15")
        elif result.price_zone == PriceZone.PREMIUM:
            zone_score = -0.15  # Хороша зона для SHORT
            result.reasons.append("Zone:-15")
        
        result.zone_score = zone_score
        
        # 3. Order Block Score
        ob_score = 0.0
        if result.price_at_bullish_ob:
            ob_score = 0.20  # Ціна на підтримці
            result.reasons.append("OB:+20")
        elif result.price_at_bearish_ob:
            ob_score = -0.20  # Ціна на опорі
            result.reasons.append("OB:-20")
        elif result.nearest_bullish_ob and result.price_zone == PriceZone.DISCOUNT:
            ob_score = 0.10  # Є підтримка нижче в discount
        elif result.nearest_bearish_ob and result.price_zone == PriceZone.PREMIUM:
            ob_score = -0.10  # Є опір вище в premium
        
        result.ob_score = ob_score
        
        # 4. Total SMC Score
        result.smc_score = np.clip(
            result.structure_score + result.zone_score + result.ob_score + htf_bias_bonus,
            -1.0, 1.0
        )


# Factory function
_analyzer = None

def get_smc_analyzer(config: Dict = None) -> SMCAnalyzer:
    """Get SMC Analyzer instance (singleton)"""
    global _analyzer
    if _analyzer is None:
        _analyzer = SMCAnalyzer(config)
    return _analyzer
