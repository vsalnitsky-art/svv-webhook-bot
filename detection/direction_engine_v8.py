"""
Direction Engine v8.0 - Smart Money Concepts Integration

КЛЮЧОВІ ВІДМІННОСТІ від v7:
- Інтеграція SMC Analyzer (BOS/CHoCH, Order Blocks, Premium/Discount)
- Підтримка MTF аналізу (4H для bias, 1H для сигналів)
- Нова система балів з урахуванням інституційних патернів
- Стратегія "Пробудження Сплячого" з SMC підтвердженням

Автор: SVV Bot Team
Версія: 8.0 (2026-02-02)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from detection.smc_analyzer import (
    SMCAnalyzer, get_smc_analyzer, SMCAnalysisResult,
    StructureSignal, MarketBias, PriceZone
)


class StructureType(Enum):
    HIGHER_HIGHS = "HH"
    HIGHER_LOWS = "HL"
    LOWER_HIGHS = "LH"
    LOWER_LOWS = "LL"
    RANGING = "RANGING"
    UNKNOWN = "UNKNOWN"


class BiasDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class StructureAnalysisV8:
    """Аналіз структури з SMC компонентами"""
    # Класичні метрики
    hh_count: int = 0
    hl_count: int = 0
    lh_count: int = 0
    ll_count: int = 0
    dominant_structure: StructureType = StructureType.UNKNOWN
    structure_bias: float = 0.0
    is_near_high: bool = False
    is_near_low: bool = False
    is_in_middle: bool = True
    vwap_price: float = 0.0
    price_vs_vwap_pct: float = 0.0
    
    # SMC компоненти
    smc_signal: StructureSignal = StructureSignal.NONE
    market_bias: MarketBias = MarketBias.NEUTRAL
    price_zone: PriceZone = PriceZone.EQUILIBRIUM
    zone_level: float = 0.5
    price_at_bullish_ob: bool = False
    price_at_bearish_ob: bool = False
    smc_score: float = 0.0


@dataclass
class MomentumAnalysisV8:
    ema_slope: float = 0.0
    macd_histogram: float = 0.0
    volume_bias: float = 0.0
    momentum_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class DerivativesAnalysisV8:
    oi_change_pct: float = 0.0
    funding_rate: float = 0.0
    ob_imbalance: float = 0.0
    derivatives_bias: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class DirectionResultV8:
    """Результат аналізу напрямку v8"""
    symbol: str
    direction: BiasDirection
    bias_score: float
    confidence: float
    adx_value: float = 0.0
    is_consolidation: bool = False
    structure: StructureAnalysisV8 = None
    momentum: MomentumAnalysisV8 = None
    derivatives: DerivativesAnalysisV8 = None
    smc_result: SMCAnalysisResult = None
    should_trade: bool = False
    reason: str = ""
    timestamp: datetime = None
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'bias_score': round(self.bias_score, 3),
            'confidence': round(self.confidence, 1),
            'adx_value': round(self.adx_value, 1),
            'is_consolidation': self.is_consolidation,
            'should_trade': self.should_trade,
            'reason': self.reason,
            'structure': {
                'hh_count': self.structure.hh_count if self.structure else 0,
                'hl_count': self.structure.hl_count if self.structure else 0,
                'lh_count': self.structure.lh_count if self.structure else 0,
                'll_count': self.structure.ll_count if self.structure else 0,
                'dominant': self.structure.dominant_structure.value if self.structure else 'UNKNOWN',
                'is_near_high': self.structure.is_near_high if self.structure else False,
                'is_near_low': self.structure.is_near_low if self.structure else False,
                'is_in_middle': self.structure.is_in_middle if self.structure else True,
                # SMC
                'smc_signal': self.structure.smc_signal.value if self.structure else 'NONE',
                'market_bias': self.structure.market_bias.value if self.structure else 'NEUTRAL',
                'price_zone': self.structure.price_zone.value if self.structure else 'EQUILIBRIUM',
                'zone_level': round(self.structure.zone_level, 3) if self.structure else 0.5,
                'price_at_bullish_ob': self.structure.price_at_bullish_ob if self.structure else False,
                'price_at_bearish_ob': self.structure.price_at_bearish_ob if self.structure else False,
                'smc_score': round(self.structure.smc_score, 3) if self.structure else 0,
            } if self.structure else {},
            'momentum_score': self.momentum.momentum_score if self.momentum else 0,
            'derivatives_bias': self.derivatives.derivatives_bias if self.derivatives else 0,
            'smc': self.smc_result.to_dict() if self.smc_result else {},
        }


class DirectionEngineV8:
    """
    Direction Engine v8 - Smart Money Concepts Integration
    
    Поєднує класичний аналіз з SMC для точнішого визначення напрямку.
    Оптимізований для стратегії "Пробудження Сплячого".
    """
    
    # Пороги ADX
    MAX_ADX_FOR_SLEEPER = 25
    
    # Пороги bias для прийняття рішень
    BIAS_THRESHOLD_LONG = 0.15   # v8.2.4: Знижено з 0.30 для кращої детекції
    BIAS_THRESHOLD_SHORT = -0.15  # v8.2.4: Знижено з -0.30
    
    # Ваги компонентів (оновлені для v8)
    # SMC отримує більшу вагу, оскільки це головний фокус
    WEIGHT_SMC = 0.40        # Smart Money Concepts
    WEIGHT_STRUCTURE = 0.20  # Класична структура
    WEIGHT_MOMENTUM = 0.20   # Momentum індикатори
    WEIGHT_DERIVATIVES = 0.20  # Деривативи (OI, Funding, OrderBook)
    
    def __init__(self, config: Optional[Dict] = None):
        self.smc_analyzer = get_smc_analyzer(config)
        
        if config:
            self.MAX_ADX_FOR_SLEEPER = config.get('max_adx', 25)
            self.BIAS_THRESHOLD_LONG = config.get('bias_threshold_long', 0.15)
            self.BIAS_THRESHOLD_SHORT = config.get('bias_threshold_short', -0.15)
            self.WEIGHT_SMC = config.get('weight_smc', 0.40)
            self.WEIGHT_STRUCTURE = config.get('weight_structure', 0.20)
            self.WEIGHT_MOMENTUM = config.get('weight_momentum', 0.20)
            self.WEIGHT_DERIVATIVES = config.get('weight_derivatives', 0.20)
    
    def analyze(self, 
                symbol: str,
                klines_4h: List[Dict],
                klines_1h: List[Dict] = None,
                oi_change_pct: float = None,
                funding_rate: float = None,
                ob_imbalance: float = None,
                poc_price: float = None) -> DirectionResultV8:
        """
        Головний метод аналізу
        
        Args:
            symbol: Символ монети
            klines_4h: Свічки 4H (для глобального тренду та SMC HTF)
            klines_1h: Свічки 1H (для сигналів)
            oi_change_pct: Зміна Open Interest (%)
            funding_rate: Funding Rate
            ob_imbalance: Дисбаланс Order Book (%)
            poc_price: Point of Control ціна
        
        Returns:
            DirectionResultV8 з повним аналізом
        """
        if not klines_4h or len(klines_4h) < 50:
            return self._neutral_result(symbol, "Insufficient data")
        
        if not klines_1h:
            klines_1h = klines_4h
        
        # 1. Перевіряємо ADX для визначення консолідації
        adx_value = self._calculate_adx(klines_4h)
        is_consolidation = adx_value < self.MAX_ADX_FOR_SLEEPER
        
        if not is_consolidation:
            return DirectionResultV8(
                symbol=symbol,
                direction=BiasDirection.NEUTRAL,
                bias_score=0.0,
                confidence=0.0,
                adx_value=adx_value,
                is_consolidation=False,
                should_trade=False,
                reason=f"Not sleeper: ADX={adx_value:.1f}",
                timestamp=datetime.now()
            )
        
        # 2. SMC Аналіз (ГОЛОВНИЙ КОМПОНЕНТ v8)
        smc_result = self.smc_analyzer.analyze(
            klines=klines_1h,
            htf_klines=klines_4h
        )
        
        # 3. Класичний аналіз структури (з SMC компонентами)
        structure = self._analyze_structure(klines_4h, poc_price, smc_result)
        
        # 4. Momentum аналіз
        momentum = self._analyze_momentum(klines_1h)
        
        # 5. Деривативи
        derivatives = self._analyze_derivatives(oi_change_pct, funding_rate, ob_imbalance)
        
        # 6. Розраховуємо загальний bias
        total_bias = (
            smc_result.smc_score * self.WEIGHT_SMC +
            structure.structure_bias * self.WEIGHT_STRUCTURE +
            momentum.momentum_score * self.WEIGHT_MOMENTUM +
            derivatives.derivatives_bias * self.WEIGHT_DERIVATIVES
        )
        total_bias = np.clip(total_bias, -1.0, 1.0)
        
        # 7. Визначаємо напрямок та умови для торгівлі
        direction, should_trade, confidence = self._determine_direction(
            total_bias, structure, smc_result
        )
        
        # 8. Формуємо причину
        reasons = self._build_reasons(structure, smc_result, momentum, derivatives)
        
        return DirectionResultV8(
            symbol=symbol,
            direction=direction,
            bias_score=round(total_bias, 3),
            confidence=confidence,
            adx_value=adx_value,
            is_consolidation=is_consolidation,
            structure=structure,
            momentum=momentum,
            derivatives=derivatives,
            smc_result=smc_result,
            should_trade=should_trade,
            reason=" | ".join(reasons) if reasons else "Mixed",
            timestamp=datetime.now()
        )
    
    def _analyze_structure(self, 
                           klines: List[Dict], 
                           poc_price: float = None,
                           smc_result: SMCAnalysisResult = None) -> StructureAnalysisV8:
        """Аналіз структури з інтеграцією SMC"""
        result = StructureAnalysisV8()
        
        if len(klines) < 20:
            return result
        
        highs = [float(k['high']) for k in klines]
        lows = [float(k['low']) for k in klines]
        closes = [float(k['close']) for k in klines]
        current_price = closes[-1]
        
        # Класичний pivot аналіз
        pivot_highs = self._find_pivot_points(highs, is_high=True)
        pivot_lows = self._find_pivot_points(lows, is_high=False)
        
        # Підраховуємо HH/HL/LH/LL
        for i in range(1, len(pivot_highs)):
            if pivot_highs[i] > pivot_highs[i-1]:
                result.hh_count += 1
            else:
                result.lh_count += 1
        
        for i in range(1, len(pivot_lows)):
            if pivot_lows[i] > pivot_lows[i-1]:
                result.hl_count += 1
            else:
                result.ll_count += 1
        
        # Визначаємо домінуючу структуру
        bullish_count = result.hh_count + result.hl_count
        bearish_count = result.lh_count + result.ll_count
        
        if bullish_count > bearish_count + 1:
            result.dominant_structure = StructureType.HIGHER_HIGHS if result.hh_count > result.hl_count else StructureType.HIGHER_LOWS
            result.structure_bias = min(1.0, bullish_count / 4)
        elif bearish_count > bullish_count + 1:
            result.dominant_structure = StructureType.LOWER_HIGHS if result.lh_count > result.ll_count else StructureType.LOWER_LOWS
            result.structure_bias = -min(1.0, bearish_count / 4)
        else:
            result.dominant_structure = StructureType.RANGING
            result.structure_bias = 0.0
        
        # Позиція в діапазоні
        recent_high = max(highs[-50:])
        recent_low = min(lows[-50:])
        range_height = recent_high - recent_low
        
        if range_height > 0:
            position = (current_price - recent_low) / range_height
            result.is_near_high = position > 0.85
            result.is_near_low = position < 0.15
            result.is_in_middle = 0.35 < position < 0.65
        
        # VWAP
        result.vwap_price = self._calculate_vwap(klines)
        if result.vwap_price > 0:
            result.price_vs_vwap_pct = (current_price - result.vwap_price) / result.vwap_price * 100
            vwap_component = np.clip(result.price_vs_vwap_pct / 3, -0.3, 0.3)
            result.structure_bias = np.clip(result.structure_bias * 0.7 + vwap_component * 0.3, -1.0, 1.0)
        
        # Інтегруємо SMC результати
        if smc_result:
            result.smc_signal = smc_result.structure_signal
            result.market_bias = smc_result.market_bias
            result.price_zone = smc_result.price_zone
            result.zone_level = smc_result.zone_level
            result.price_at_bullish_ob = smc_result.price_at_bullish_ob
            result.price_at_bearish_ob = smc_result.price_at_bearish_ob
            result.smc_score = smc_result.smc_score
        
        return result
    
    def _find_pivot_points(self, prices: List[float], is_high: bool = True, lookback: int = 2) -> List[float]:
        """Знаходить pivot points"""
        pivots = []
        for i in range(lookback, len(prices) - lookback):
            is_pivot = True
            for j in range(1, lookback + 1):
                if is_high:
                    if prices[i] <= prices[i-j] or prices[i] <= prices[i+j]:
                        is_pivot = False
                        break
                else:
                    if prices[i] >= prices[i-j] or prices[i] >= prices[i+j]:
                        is_pivot = False
                        break
            if is_pivot:
                pivots.append(prices[i])
        return pivots[-6:] if len(pivots) > 6 else pivots
    
    def _calculate_vwap(self, klines: List[Dict]) -> float:
        """Розраховує VWAP"""
        if not klines:
            return 0.0
        total_tp_vol = 0.0
        total_vol = 0.0
        for k in klines[-50:]:
            tp = (float(k['high']) + float(k['low']) + float(k['close'])) / 3
            vol = float(k.get('volume', 0))
            total_tp_vol += tp * vol
            total_vol += vol
        return total_tp_vol / total_vol if total_vol > 0 else 0.0
    
    def _analyze_momentum(self, klines: List[Dict]) -> MomentumAnalysisV8:
        """Аналіз momentum індикаторів"""
        result = MomentumAnalysisV8()
        if len(klines) < 30:
            return result
        
        closes = [float(k['close']) for k in klines]
        
        # EMA Slope
        result.ema_slope = self._calculate_ema_slope(closes, 9, 5)
        ema_component = np.clip(result.ema_slope / 2, -0.5, 0.5)
        
        # MACD
        result.macd_histogram = self._calculate_macd_histogram(closes)
        macd_component = np.clip(result.macd_histogram / closes[-1] * 500, -0.3, 0.3) if closes[-1] > 0 else 0
        
        # Volume Bias
        result.volume_bias = self._calculate_volume_bias(klines[-20:])
        vol_component = np.clip(result.volume_bias / 50, -0.3, 0.3)
        
        result.momentum_score = np.clip(ema_component * 0.4 + macd_component * 0.3 + vol_component * 0.3, -1.0, 1.0)
        
        if result.momentum_score > 0.2:
            result.reasons.append(f"Mom+{result.momentum_score:.2f}")
        elif result.momentum_score < -0.2:
            result.reasons.append(f"Mom{result.momentum_score:.2f}")
        
        return result
    
    def _calculate_ema_slope(self, closes: List[float], period: int = 9, lookback: int = 5) -> float:
        """Розраховує нахил EMA"""
        if len(closes) < period + lookback:
            return 0.0
        ema = []
        mult = 2 / (period + 1)
        ema.append(sum(closes[:period]) / period)
        for i in range(period, len(closes)):
            ema.append((closes[i] - ema[-1]) * mult + ema[-1])
        if len(ema) < lookback or ema[-lookback] == 0:
            return 0.0
        return (ema[-1] - ema[-lookback]) / ema[-lookback] * 100
    
    def _calculate_macd_histogram(self, closes: List[float]) -> float:
        """Розраховує MACD histogram"""
        if len(closes) < 35:
            return 0.0
        
        def ema(vals, period):
            if len(vals) < period:
                return vals[-1] if vals else 0
            m = 2 / (period + 1)
            e = sum(vals[:period]) / period
            for v in vals[period:]:
                e = (v - e) * m + e
            return e
        
        ema12 = ema(closes, 12)
        ema26 = ema(closes, 26)
        macd_line = ema12 - ema26
        
        macd_hist = []
        for i in range(26, len(closes)):
            e12 = ema(closes[:i+1], 12)
            e26 = ema(closes[:i+1], 26)
            macd_hist.append(e12 - e26)
        
        if len(macd_hist) >= 9:
            signal = sum(macd_hist[-9:]) / 9
            return macd_line - signal
        return 0.0
    
    def _calculate_volume_bias(self, klines: List[Dict]) -> float:
        """Розраховує bias об'єму"""
        up_vol = down_vol = 0.0
        for k in klines:
            vol = float(k.get('volume', 0))
            if float(k['close']) > float(k['open']):
                up_vol += vol
            else:
                down_vol += vol
        total = up_vol + down_vol
        return (up_vol - down_vol) / total * 100 if total > 0 else 0.0
    
    def _analyze_derivatives(self, 
                             oi_change: float = None, 
                             funding: float = None, 
                             ob_imb: float = None) -> DerivativesAnalysisV8:
        """Аналіз деривативів"""
        result = DerivativesAnalysisV8()
        components = []
        
        if oi_change is not None:
            result.oi_change_pct = oi_change
            if oi_change > 5:
                comp = 0.2 if ob_imb and ob_imb > 0 else -0.2 if ob_imb and ob_imb < 0 else 0.1
                components.append(comp)
                result.reasons.append(f"OI+{oi_change:.1f}%")
        
        if funding is not None:
            result.funding_rate = funding
            if funding > 0.03:
                components.append(-0.15)
                result.reasons.append("Fund+")
            elif funding < -0.03:
                components.append(0.15)
                result.reasons.append("Fund-")
        
        if ob_imb is not None:
            result.ob_imbalance = ob_imb
            if abs(ob_imb) > 10:
                components.append(np.clip(ob_imb / 50, -0.3, 0.3))
                result.reasons.append(f"OB{ob_imb:+.1f}%")
        
        if components:
            result.derivatives_bias = np.clip(sum(components), -1.0, 1.0)
        return result
    
    def _determine_direction(self,
                             total_bias: float,
                             structure: StructureAnalysisV8,
                             smc_result: SMCAnalysisResult) -> Tuple[BiasDirection, bool, float]:
        """
        Визначає напрямок та умови для торгівлі
        
        Стратегія "Пробудження Сплячого" з SMC:
        - LONG: CHoCH/BOS бичачий + Discount Zone + (опційно) біля Bullish OB
        - SHORT: CHoCH/BOS ведмежий + Premium Zone + (опційно) біля Bearish OB
        
        v8.1: Додано "м'який" напрямок для sleepers без CHoCH
        - Базується на market bias та price zone
        - Показує потенційний напрямок до появи CHoCH
        """
        direction = BiasDirection.NEUTRAL
        should_trade = False
        confidence = 0.0
        
        # ============================================
        # СИЛЬНІ СИГНАЛИ (з CHoCH/BOS)
        # ============================================
        
        # LONG з CHoCH/BOS
        if total_bias >= self.BIAS_THRESHOLD_LONG:
            direction = BiasDirection.LONG
            confidence = min(95, 50 + abs(total_bias) * 50)
            
            # Умови для торгівлі LONG
            smc_bullish = smc_result.structure_signal in [
                StructureSignal.BULLISH_CHOCH, 
                StructureSignal.BULLISH_BOS
            ]
            good_zone = smc_result.price_zone in [PriceZone.DISCOUNT, PriceZone.EQUILIBRIUM]
            near_support = structure.is_near_low or smc_result.price_at_bullish_ob
            
            # Найвищий пріоритет: CHoCH в Discount Zone
            if smc_result.structure_signal == StructureSignal.BULLISH_CHOCH and good_zone:
                should_trade = True
                confidence = min(95, confidence + 10)
            # BOS + хороша зона
            elif smc_bullish and good_zone:
                should_trade = True
            # Класичні умови (near low + bullish structure)
            elif near_support and structure.dominant_structure in [StructureType.HIGHER_HIGHS, StructureType.HIGHER_LOWS]:
                should_trade = True
        
        # SHORT з CHoCH/BOS
        elif total_bias <= self.BIAS_THRESHOLD_SHORT:
            direction = BiasDirection.SHORT
            confidence = min(95, 50 + abs(total_bias) * 50)
            
            smc_bearish = smc_result.structure_signal in [
                StructureSignal.BEARISH_CHOCH,
                StructureSignal.BEARISH_BOS
            ]
            good_zone = smc_result.price_zone in [PriceZone.PREMIUM, PriceZone.EQUILIBRIUM]
            near_resistance = structure.is_near_high or smc_result.price_at_bearish_ob
            
            # Найвищий пріоритет: CHoCH в Premium Zone
            if smc_result.structure_signal == StructureSignal.BEARISH_CHOCH and good_zone:
                should_trade = True
                confidence = min(95, confidence + 10)
            elif smc_bearish and good_zone:
                should_trade = True
            elif near_resistance and structure.dominant_structure in [StructureType.LOWER_HIGHS, StructureType.LOWER_LOWS]:
                should_trade = True
        
        # ============================================
        # v8.1: М'ЯКИЙ НАПРЯМОК ДЛЯ SLEEPERS (без CHoCH)
        # Показує потенційний напрямок на основі структури
        # ============================================
        else:
            # Визначаємо потенційний напрямок на основі комбінації факторів
            bullish_signals = 0
            bearish_signals = 0
            
            # Market Bias
            if smc_result.market_bias == MarketBias.BULLISH:
                bullish_signals += 2
            elif smc_result.market_bias == MarketBias.BEARISH:
                bearish_signals += 2
            
            # Price Zone
            if smc_result.price_zone == PriceZone.DISCOUNT:
                bullish_signals += 1
            elif smc_result.price_zone == PriceZone.PREMIUM:
                bearish_signals += 1
            
            # Order Block proximity
            if smc_result.price_at_bullish_ob or smc_result.nearest_bullish_ob:
                bullish_signals += 1
            if smc_result.price_at_bearish_ob or smc_result.nearest_bearish_ob:
                bearish_signals += 1
            
            # Structure pattern (HH/HL vs LH/LL)
            if structure.dominant_structure == StructureType.HIGHER_HIGHS:
                bullish_signals += 1
            elif structure.dominant_structure == StructureType.HIGHER_LOWS:
                bullish_signals += 1
            elif structure.dominant_structure == StructureType.LOWER_HIGHS:
                bearish_signals += 1
            elif structure.dominant_structure == StructureType.LOWER_LOWS:
                bearish_signals += 1
            
            # Position bias
            if structure.is_near_low:
                bullish_signals += 1
            if structure.is_near_high:
                bearish_signals += 1
            
            # Визначаємо м'який напрямок (мінімум 2 сигнали перевага)
            signal_diff = bullish_signals - bearish_signals
            
            if signal_diff >= 2:
                direction = BiasDirection.LONG
                confidence = min(60, 30 + signal_diff * 5)  # Низька впевненість (30-60%)
                # should_trade = False - не торгуємо без CHoCH!
            elif signal_diff <= -2:
                direction = BiasDirection.SHORT
                confidence = min(60, 30 + abs(signal_diff) * 5)
                # should_trade = False - не торгуємо без CHoCH!
            # else: залишаємо NEUTRAL
        
        return direction, should_trade, confidence
    
    def _build_reasons(self,
                       structure: StructureAnalysisV8,
                       smc_result: SMCAnalysisResult,
                       momentum: MomentumAnalysisV8,
                       derivatives: DerivativesAnalysisV8) -> List[str]:
        """Формує список причин для рішення"""
        reasons = []
        
        # SMC сигнали (головні)
        if structure.smc_signal != StructureSignal.NONE:
            signal_name = structure.smc_signal.value.replace("_", " ")
            reasons.append(signal_name)
        
        # Структура
        if structure.dominant_structure in [StructureType.HIGHER_HIGHS, StructureType.HIGHER_LOWS]:
            reasons.append(f"{structure.hh_count}HH/{structure.hl_count}HL")
        elif structure.dominant_structure in [StructureType.LOWER_HIGHS, StructureType.LOWER_LOWS]:
            reasons.append(f"{structure.lh_count}LH/{structure.ll_count}LL")
        
        # Зона
        if structure.price_zone == PriceZone.DISCOUNT:
            reasons.append(f"Discount({structure.zone_level:.2f})")
        elif structure.price_zone == PriceZone.PREMIUM:
            reasons.append(f"Premium({structure.zone_level:.2f})")
        
        # Order Blocks
        if structure.price_at_bullish_ob:
            reasons.append("@BullOB")
        elif structure.price_at_bearish_ob:
            reasons.append("@BearOB")
        
        # Позиція
        if structure.is_near_high:
            reasons.append("near HIGH")
        elif structure.is_near_low:
            reasons.append("near LOW")
        
        # SMC reasons
        reasons.extend(smc_result.reasons[:3])  # Обмежуємо щоб не переповнювати
        
        return reasons
    
    def _calculate_adx(self, klines: List[Dict], period: int = 14) -> float:
        """Розраховує ADX"""
        if len(klines) < period * 2:
            return 25.0
        
        highs = [float(k['high']) for k in klines]
        lows = [float(k['low']) for k in klines]
        closes = [float(k['close']) for k in klines]
        
        tr_list = []
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(klines)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            tr_list.append(tr)
            
            up = highs[i] - highs[i-1]
            down = lows[i-1] - lows[i]
            
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)
        
        def wilder(vals, p):
            if len(vals) < p:
                return sum(vals) / len(vals) if vals else 0
            s = sum(vals[:p]) / p
            for v in vals[p:]:
                s = (s * (p - 1) + v) / p
            return s
        
        atr = wilder(tr_list, period)
        if atr == 0:
            return 25.0
        
        plus_di = 100 * wilder(plus_dm, period) / atr
        minus_di = 100 * wilder(minus_dm, period) / atr
        
        if plus_di + minus_di == 0:
            return 25.0
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx
    
    def _neutral_result(self, symbol: str, reason: str) -> DirectionResultV8:
        """Повертає нейтральний результат"""
        return DirectionResultV8(
            symbol=symbol,
            direction=BiasDirection.NEUTRAL,
            bias_score=0.0,
            confidence=0.0,
            should_trade=False,
            reason=reason,
            timestamp=datetime.now()
        )


# Factory
_engine = None

def get_direction_engine_v8(config: Dict = None) -> DirectionEngineV8:
    """Get Direction Engine v8 instance (singleton)"""
    global _engine
    if _engine is None:
        _engine = DirectionEngineV8(config)
    return _engine
