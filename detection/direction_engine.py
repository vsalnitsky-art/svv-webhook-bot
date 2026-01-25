"""
Direction Engine v1.0 - Professional Direction Resolution

Професійна парадигма:
1. Sleeper Detector → ЧИ є сенс торгувати
2. Direction Engine → В ЯКИЙ БІК (цей модуль)
3. Trigger Engine   → КОЛИ входити

Direction = HTF Bias (50%) + LTF Momentum (30%) + Derivatives (20%)

ВАЖЛИВО:
- Direction ≠ Signal
- NEUTRAL = валідний стан (не торгуємо)
- Direction не впливає на sleeper_score, тільки фільтрує/маркує
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

from core.market_data import get_fetcher
from core.tech_indicators import get_indicators
from config import DIRECTION_ENGINE


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


@dataclass
class DirectionResult:
    """Результат визначення напрямку"""
    direction: Direction
    score: float  # -1.0 to +1.0
    confidence: str  # HIGH, MEDIUM, LOW
    
    # Component scores
    htf_bias: float      # -1 to +1
    ltf_bias: float      # -1 to +1
    deriv_bias: float    # -1 to +1
    
    # Debug info
    htf_reason: str
    ltf_reason: str
    deriv_reason: str


class DirectionEngine:
    """
    3-Layer Direction Model:
    
    Layer 1: HTF Structural Bias (50%)
        - 1D Market Structure (price vs EMA50)
        - 4H EMA slope (EMA20 direction)
        
    Layer 2: LTF Momentum Shift (30%)
        - RSI divergence
        - BB position (above/below middle)
        - Price momentum
        
    Layer 3: Derivatives Positioning (20%)
        - OI change + Price change combination
        - Funding rate extremes
        - Institutional positioning logic
    """
    
    def __init__(self):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        
        # Load config
        self.weights = DIRECTION_ENGINE.get('weights', {
            'htf': 0.5,
            'ltf': 0.3,
            'derivatives': 0.2
        })
        self.long_threshold = DIRECTION_ENGINE.get('long_threshold', 0.5)
        self.short_threshold = DIRECTION_ENGINE.get('short_threshold', -0.5)
    
    def resolve(self, symbol: str, 
                klines_4h: List[Dict] = None,
                klines_1d: List[Dict] = None,
                oi_change: float = None,
                funding_rate: float = None,
                price_change_4h: float = None) -> DirectionResult:
        """
        Resolve direction for symbol.
        
        Can use pre-fetched data (during scan) or fetch fresh data.
        
        Returns DirectionResult with direction, score, and component breakdown.
        """
        
        # Fetch data if not provided
        if klines_4h is None:
            klines_4h = self.fetcher.get_klines(symbol, '4h', limit=50)
        if klines_1d is None:
            klines_1d = self.fetcher.get_klines(symbol, '1d', limit=30)
        
        # Calculate price change if not provided
        if price_change_4h is None and len(klines_4h) >= 2:
            price_change_4h = (klines_4h[-1]['close'] - klines_4h[-2]['close']) / klines_4h[-2]['close'] * 100
        
        # === LAYER 1: HTF Structural Bias (50%) ===
        htf_bias, htf_reason = self._calculate_htf_bias(klines_1d, klines_4h)
        
        # === LAYER 2: LTF Momentum Shift (30%) ===
        ltf_bias, ltf_reason = self._calculate_ltf_bias(klines_4h)
        
        # === LAYER 3: Derivatives Positioning (20%) ===
        deriv_bias, deriv_reason = self._calculate_derivatives_bias(
            price_change_4h or 0,
            oi_change or 0,
            funding_rate or 0
        )
        
        # === AGGREGATE ===
        direction_score = (
            htf_bias * self.weights['htf'] +
            ltf_bias * self.weights['ltf'] +
            deriv_bias * self.weights['derivatives']
        )
        
        # Determine direction
        if direction_score >= self.long_threshold:
            direction = Direction.LONG
        elif direction_score <= self.short_threshold:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL
        
        # Determine confidence
        abs_score = abs(direction_score)
        if abs_score >= 0.7:
            confidence = "HIGH"
        elif abs_score >= 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return DirectionResult(
            direction=direction,
            score=round(direction_score, 3),
            confidence=confidence,
            htf_bias=htf_bias,
            ltf_bias=ltf_bias,
            deriv_bias=deriv_bias,
            htf_reason=htf_reason,
            ltf_reason=ltf_reason,
            deriv_reason=deriv_reason
        )
    
    def _calculate_htf_bias(self, klines_1d: List[Dict], klines_4h: List[Dict]) -> Tuple[float, str]:
        """
        Layer 1: HTF Structural Bias
        
        Components:
        - 1D: Price position vs EMA50 (structure)
        - 4H: EMA20 slope (momentum)
        
        Returns: (bias: -1 to +1, reason: str)
        """
        if len(klines_1d) < 50 or len(klines_4h) < 20:
            return 0, "Insufficient data"
        
        # === 1D Structure ===
        closes_1d = [k['close'] for k in klines_1d]
        ema50_1d = self.indicators.ema(closes_1d, 50)
        
        current_price = closes_1d[-1]
        ema50_current = ema50_1d[-1]
        
        # Price position relative to EMA50
        price_vs_ema = (current_price - ema50_current) / ema50_current * 100
        
        # Structure bias based on price position
        if price_vs_ema > 3:
            structure_bias = 1.0
            structure_reason = f"Price {price_vs_ema:.1f}% above EMA50"
        elif price_vs_ema > 1:
            structure_bias = 0.5
            structure_reason = f"Price {price_vs_ema:.1f}% above EMA50"
        elif price_vs_ema < -3:
            structure_bias = -1.0
            structure_reason = f"Price {abs(price_vs_ema):.1f}% below EMA50"
        elif price_vs_ema < -1:
            structure_bias = -0.5
            structure_reason = f"Price {abs(price_vs_ema):.1f}% below EMA50"
        else:
            structure_bias = 0
            structure_reason = "Price near EMA50"
        
        # === 4H EMA Slope ===
        closes_4h = [k['close'] for k in klines_4h]
        ema20_4h = self.indicators.ema(closes_4h, 20)
        
        # Slope over last 5 candles (20 hours)
        if len(ema20_4h) >= 5:
            slope = ema20_4h[-1] - ema20_4h[-5]
            slope_pct = slope / ema20_4h[-5] * 100 if ema20_4h[-5] != 0 else 0
            
            if slope_pct > 1:
                slope_bias = 1.0
                slope_reason = f"EMA20 rising {slope_pct:.2f}%"
            elif slope_pct > 0.3:
                slope_bias = 0.5
                slope_reason = f"EMA20 slight up {slope_pct:.2f}%"
            elif slope_pct < -1:
                slope_bias = -1.0
                slope_reason = f"EMA20 falling {abs(slope_pct):.2f}%"
            elif slope_pct < -0.3:
                slope_bias = -0.5
                slope_reason = f"EMA20 slight down {abs(slope_pct):.2f}%"
            else:
                slope_bias = 0
                slope_reason = "EMA20 flat"
        else:
            slope_bias = 0
            slope_reason = "Insufficient EMA data"
        
        # Combine structure (60%) + slope (40%)
        combined_bias = structure_bias * 0.6 + slope_bias * 0.4
        combined_reason = f"Structure: {structure_reason} | Slope: {slope_reason}"
        
        return combined_bias, combined_reason
    
    def _calculate_ltf_bias(self, klines_4h: List[Dict]) -> Tuple[float, str]:
        """
        Layer 2: LTF Momentum Shift
        
        Components:
        - RSI divergence (bullish/bearish)
        - Price position vs BB middle
        - Short-term momentum
        
        Returns: (bias: -1 to +1, reason: str)
        """
        if len(klines_4h) < 20:
            return 0, "Insufficient data"
        
        # Calculate indicators
        indicators = self.indicators.calculate_all(klines_4h)
        
        closes = [k['close'] for k in klines_4h]
        current_price = closes[-1]
        
        reasons = []
        biases = []
        
        # === RSI Divergence ===
        divergence = indicators.get('divergence')
        if divergence == 'bullish':
            biases.append(1.0)
            reasons.append("RSI bullish divergence")
        elif divergence == 'bearish':
            biases.append(-1.0)
            reasons.append("RSI bearish divergence")
        else:
            # Use RSI level as weak signal
            rsi = indicators.get('rsi_current', 50)
            if rsi < 35:
                biases.append(0.3)
                reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi > 65:
                biases.append(-0.3)
                reasons.append(f"RSI overbought ({rsi:.0f})")
            else:
                biases.append(0)
                reasons.append(f"RSI neutral ({rsi:.0f})")
        
        # === BB Position ===
        bb = indicators.get('bb', {})
        bb_middle = bb.get('middle', [])
        bb_upper = bb.get('upper', [])
        bb_lower = bb.get('lower', [])
        
        if bb_middle and bb_upper and bb_lower:
            middle = bb_middle[-1]
            upper = bb_upper[-1]
            lower = bb_lower[-1]
            
            # Position in BB range (0 to 1)
            bb_range = upper - lower
            if bb_range > 0:
                bb_position = (current_price - lower) / bb_range
                
                if bb_position > 0.8:
                    biases.append(-0.5)  # Near upper = potential reversal
                    reasons.append("Near BB upper")
                elif bb_position < 0.2:
                    biases.append(0.5)   # Near lower = potential bounce
                    reasons.append("Near BB lower")
                elif current_price > middle:
                    biases.append(0.3)
                    reasons.append("Above BB middle")
                else:
                    biases.append(-0.3)
                    reasons.append("Below BB middle")
        
        # === Short-term Momentum ===
        if len(closes) >= 5:
            momentum = (closes[-1] - closes[-5]) / closes[-5] * 100
            if momentum > 3:
                biases.append(0.5)
                reasons.append(f"Strong momentum +{momentum:.1f}%")
            elif momentum > 1:
                biases.append(0.3)
                reasons.append(f"Positive momentum +{momentum:.1f}%")
            elif momentum < -3:
                biases.append(-0.5)
                reasons.append(f"Weak momentum {momentum:.1f}%")
            elif momentum < -1:
                biases.append(-0.3)
                reasons.append(f"Negative momentum {momentum:.1f}%")
        
        # Average biases
        if biases:
            combined_bias = sum(biases) / len(biases)
        else:
            combined_bias = 0
        
        combined_reason = " | ".join(reasons) if reasons else "No LTF signals"
        
        return combined_bias, combined_reason
    
    def _calculate_derivatives_bias(self, price_change: float, oi_change: float, 
                                     funding_rate: float) -> Tuple[float, str]:
        """
        Layer 3: Derivatives Positioning Bias
        
        Institutional logic table:
        Price | OI    | Funding | Meaning
        ↑     | ↑     | neutral | Longs building → LONG
        ↓     | ↑     | neutral | Shorts building → SHORT
        ↑     | ↑     | high +  | Crowded longs → CAUTION
        ↓     | ↑     | low -   | Crowded shorts → SQUEEZE potential
        
        Returns: (bias: -1 to +1, reason: str)
        """
        reasons = []
        bias = 0.0
        
        # Funding rate thresholds
        FUNDING_HIGH = 0.0005   # 0.05%
        FUNDING_LOW = -0.0003   # -0.03%
        
        # OI change thresholds
        OI_SIGNIFICANT = 5      # 5% change
        
        # Price change thresholds
        PRICE_MOVE = 1          # 1% move
        
        # === Main Logic Table ===
        
        if oi_change > OI_SIGNIFICANT:
            # OI is building
            if price_change > PRICE_MOVE:
                # Price up + OI up
                if funding_rate > FUNDING_HIGH:
                    # Crowded longs - caution
                    bias = 0.3  # Still bullish but reduced
                    reasons.append(f"Crowded longs (FR:{funding_rate*100:.3f}%)")
                else:
                    # Clean long accumulation
                    bias = 1.0
                    reasons.append(f"Longs building (OI+{oi_change:.1f}%, Price+{price_change:.1f}%)")
                    
            elif price_change < -PRICE_MOVE:
                # Price down + OI up
                if funding_rate < FUNDING_LOW:
                    # Crowded shorts - squeeze potential
                    bias = 0.7  # Bullish squeeze setup
                    reasons.append(f"Short squeeze setup (FR:{funding_rate*100:.3f}%)")
                else:
                    # Clean short accumulation
                    bias = -1.0
                    reasons.append(f"Shorts building (OI+{oi_change:.1f}%, Price{price_change:.1f}%)")
            else:
                # OI building but price flat
                reasons.append(f"Accumulation (OI+{oi_change:.1f}%), direction unclear")
                
        elif oi_change < -OI_SIGNIFICANT:
            # OI decreasing - positions closing
            if price_change > PRICE_MOVE:
                # Price up + OI down = shorts closing
                bias = 0.5
                reasons.append(f"Short covering (OI{oi_change:.1f}%)")
            elif price_change < -PRICE_MOVE:
                # Price down + OI down = longs closing
                bias = -0.5
                reasons.append(f"Long liquidation (OI{oi_change:.1f}%)")
        else:
            # OI stable
            if abs(funding_rate) > FUNDING_HIGH:
                # Extreme funding without OI change
                if funding_rate > 0:
                    bias = -0.3  # Expect mean reversion
                    reasons.append(f"High funding ({funding_rate*100:.3f}%) - longs crowded")
                else:
                    bias = 0.3
                    reasons.append(f"Negative funding ({funding_rate*100:.3f}%) - shorts crowded")
        
        if not reasons:
            reasons.append("Derivatives neutral")
        
        return bias, " | ".join(reasons)


# Singleton instance
_engine_instance = None

def get_direction_engine() -> DirectionEngine:
    """Get direction engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DirectionEngine()
    return _engine_instance


def resolve_direction(symbol: str, **kwargs) -> DirectionResult:
    """
    Convenience function to resolve direction.
    
    Usage:
        result = resolve_direction('BTCUSDT')
        print(f"Direction: {result.direction.value}, Score: {result.score}")
    """
    engine = get_direction_engine()
    return engine.resolve(symbol, **kwargs)
