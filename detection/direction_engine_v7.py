"""
Direction Engine v7.0 - Optimized for Sleeper Consolidation Analysis

КЛЮЧОВА ВІДМІННІСТЬ: Працює ТІЛЬКИ для монет в консолідації (ADX < 25)
Визначає напрямок "пробудження" - куди ймовірніше вийде ціна

Автор: SVV Bot Team
Версія: 7.0 (2026-01-26)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


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
class StructureAnalysisV7:
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


@dataclass
class MomentumAnalysisV7:
    ema_slope: float = 0.0
    macd_histogram: float = 0.0
    volume_bias: float = 0.0
    momentum_score: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class DerivativesAnalysisV7:
    oi_change_pct: float = 0.0
    funding_rate: float = 0.0
    ob_imbalance: float = 0.0
    derivatives_bias: float = 0.0
    reasons: List[str] = field(default_factory=list)


@dataclass
class DirectionResultV7:
    symbol: str
    direction: BiasDirection
    bias_score: float
    confidence: float
    adx_value: float = 0.0
    is_consolidation: bool = False
    structure: StructureAnalysisV7 = None
    momentum: MomentumAnalysisV7 = None
    derivatives: DerivativesAnalysisV7 = None
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
            } if self.structure else {},
            'momentum_score': self.momentum.momentum_score if self.momentum else 0,
            'derivatives_bias': self.derivatives.derivatives_bias if self.derivatives else 0,
        }


class DirectionEngineV7:
    """Direction Engine v7 - Optimized for Sleeper Consolidation"""
    
    MAX_ADX_FOR_SLEEPER = 25
    BIAS_THRESHOLD_LONG = 0.30
    BIAS_THRESHOLD_SHORT = -0.30
    
    WEIGHT_STRUCTURE = 0.35
    WEIGHT_MOMENTUM = 0.35
    WEIGHT_DERIVATIVES = 0.30
    
    def __init__(self, config: Optional[Dict] = None):
        if config:
            self.MAX_ADX_FOR_SLEEPER = config.get('max_adx', 25)
            self.BIAS_THRESHOLD_LONG = config.get('bias_threshold_long', 0.30)
            self.BIAS_THRESHOLD_SHORT = config.get('bias_threshold_short', -0.30)
    
    def analyze(self, 
                symbol: str,
                klines_4h: List[Dict],
                klines_1h: List[Dict] = None,
                oi_change_pct: float = None,
                funding_rate: float = None,
                ob_imbalance: float = None,
                poc_price: float = None) -> DirectionResultV7:
        
        if not klines_4h or len(klines_4h) < 50:
            return self._neutral_result(symbol, "Insufficient data")
        
        if not klines_1h:
            klines_1h = klines_4h
        
        # Check ADX for consolidation
        adx_value = self._calculate_adx(klines_4h)
        is_consolidation = adx_value < self.MAX_ADX_FOR_SLEEPER
        
        if not is_consolidation:
            return DirectionResultV7(
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
        
        # Analyze components
        structure = self._analyze_structure(klines_4h, poc_price)
        momentum = self._analyze_momentum(klines_1h)
        derivatives = self._analyze_derivatives(oi_change_pct, funding_rate, ob_imbalance)
        
        # Calculate total bias
        total_bias = (
            structure.structure_bias * self.WEIGHT_STRUCTURE +
            momentum.momentum_score * self.WEIGHT_MOMENTUM +
            derivatives.derivatives_bias * self.WEIGHT_DERIVATIVES
        )
        total_bias = np.clip(total_bias, -1.0, 1.0)
        
        # Determine direction
        direction = BiasDirection.NEUTRAL
        should_trade = False
        confidence = 0.0
        
        if total_bias >= self.BIAS_THRESHOLD_LONG:
            direction = BiasDirection.LONG
            confidence = min(90, 50 + abs(total_bias) * 50)
            # Trade only if near low (good entry) or at HH/HL structure
            if structure.is_near_low or (structure.dominant_structure in [StructureType.HIGHER_HIGHS, StructureType.HIGHER_LOWS]):
                should_trade = True
            
        elif total_bias <= self.BIAS_THRESHOLD_SHORT:
            direction = BiasDirection.SHORT
            confidence = min(90, 50 + abs(total_bias) * 50)
            # Trade only if near high (good entry) or at LH/LL structure
            if structure.is_near_high or (structure.dominant_structure in [StructureType.LOWER_HIGHS, StructureType.LOWER_LOWS]):
                should_trade = True
        
        # Build reason
        reasons = []
        if structure.dominant_structure in [StructureType.HIGHER_HIGHS, StructureType.HIGHER_LOWS]:
            reasons.append(f"{structure.hh_count}HH/{structure.hl_count}HL")
        elif structure.dominant_structure in [StructureType.LOWER_HIGHS, StructureType.LOWER_LOWS]:
            reasons.append(f"{structure.lh_count}LH/{structure.ll_count}LL")
        
        if structure.is_near_high:
            reasons.append("near HIGH")
        elif structure.is_near_low:
            reasons.append("near LOW")
        
        return DirectionResultV7(
            symbol=symbol,
            direction=direction,
            bias_score=round(total_bias, 3),
            confidence=confidence,
            adx_value=adx_value,
            is_consolidation=is_consolidation,
            structure=structure,
            momentum=momentum,
            derivatives=derivatives,
            should_trade=should_trade,
            reason=" | ".join(reasons) if reasons else "Mixed",
            timestamp=datetime.now()
        )
    
    def _analyze_structure(self, klines: List[Dict], poc_price: float = None) -> StructureAnalysisV7:
        result = StructureAnalysisV7()
        
        if len(klines) < 20:
            return result
        
        highs = [float(k['high']) for k in klines]
        lows = [float(k['low']) for k in klines]
        closes = [float(k['close']) for k in klines]
        volumes = [float(k.get('volume', 0)) for k in klines]
        
        current_price = closes[-1]
        
        # Find pivot points
        pivot_highs = self._find_pivot_points(highs, is_high=True)
        pivot_lows = self._find_pivot_points(lows, is_high=False)
        
        # Count HH/HL/LH/LL
        if len(pivot_highs) >= 2:
            for i in range(1, min(5, len(pivot_highs))):
                if pivot_highs[i] > pivot_highs[i-1]:
                    result.hh_count += 1
                elif pivot_highs[i] < pivot_highs[i-1]:
                    result.lh_count += 1
        
        if len(pivot_lows) >= 2:
            for i in range(1, min(5, len(pivot_lows))):
                if pivot_lows[i] > pivot_lows[i-1]:
                    result.hl_count += 1
                elif pivot_lows[i] < pivot_lows[i-1]:
                    result.ll_count += 1
        
        # Determine dominant structure
        bullish_count = result.hh_count + result.hl_count
        bearish_count = result.lh_count + result.ll_count
        
        if bullish_count >= 3 and bullish_count > bearish_count + 1:
            result.dominant_structure = StructureType.HIGHER_LOWS if result.hl_count >= result.hh_count else StructureType.HIGHER_HIGHS
            result.structure_bias = min(1.0, bullish_count / 4)
        elif bearish_count >= 3 and bearish_count > bullish_count + 1:
            result.dominant_structure = StructureType.LOWER_HIGHS if result.lh_count >= result.ll_count else StructureType.LOWER_LOWS
            result.structure_bias = -min(1.0, bearish_count / 4)
        else:
            result.dominant_structure = StructureType.RANGING
            result.structure_bias = 0.0
        
        # Position in range
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
        
        return result
    
    def _find_pivot_points(self, prices: List[float], is_high: bool = True, lookback: int = 2) -> List[float]:
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
    
    def _analyze_momentum(self, klines: List[Dict]) -> MomentumAnalysisV7:
        result = MomentumAnalysisV7()
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
        return result
    
    def _calculate_ema_slope(self, closes: List[float], period: int = 9, lookback: int = 5) -> float:
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
        
        # Build MACD history for signal
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
        up_vol = down_vol = 0.0
        for k in klines:
            vol = float(k.get('volume', 0))
            if float(k['close']) > float(k['open']):
                up_vol += vol
            else:
                down_vol += vol
        total = up_vol + down_vol
        return (up_vol - down_vol) / total * 100 if total > 0 else 0.0
    
    def _analyze_derivatives(self, oi_change: float = None, funding: float = None, ob_imb: float = None) -> DerivativesAnalysisV7:
        result = DerivativesAnalysisV7()
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
            elif funding < -0.03:
                components.append(0.15)
        
        if ob_imb is not None:
            result.ob_imbalance = ob_imb
            if abs(ob_imb) > 10:
                components.append(np.clip(ob_imb / 50, -0.3, 0.3))
                result.reasons.append(f"OB{ob_imb:+.1f}%")
        
        if components:
            result.derivatives_bias = np.clip(sum(components), -1.0, 1.0)
        return result
    
    def _calculate_adx(self, klines: List[Dict], period: int = 14) -> float:
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
    
    def _neutral_result(self, symbol: str, reason: str) -> DirectionResultV7:
        return DirectionResultV7(
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

def get_direction_engine_v7(config: Dict = None) -> DirectionEngineV7:
    global _engine
    if _engine is None:
        _engine = DirectionEngineV7(config)
    return _engine
