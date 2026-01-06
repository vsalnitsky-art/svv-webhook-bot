"""
Trend Analyzer - Professional 4H Trend Context System
======================================================
Determines market regime using 4-component TrendScore:
- 30% Market Structure (swing HH/HL detection)
- 25% Volatility Expansion (ATR growth)
- 25% Acceptance (VWAP/price position)
- 20% Momentum Asymmetry (impulse imbalance)

TrendScore > 65 → Trending (BULLISH/BEARISH)
TrendScore 40-65 → Transition
TrendScore < 40 → Range/No Trade
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

from core.market_data import MarketDataFetcher


class TrendRegime(Enum):
    """Market regime classification"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NO_TRADE = "NO_TRADE"


class TrendDirection(Enum):
    """Trend direction for components"""
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


@dataclass
class SwingPoint:
    """Represents a swing high or low"""
    index: int
    price: float
    timestamp: datetime
    is_high: bool  # True = swing high, False = swing low


@dataclass
class TrendScore:
    """Complete trend analysis result"""
    symbol: str
    timeframe: str
    
    # Component scores (0-100)
    structure_score: float
    volatility_score: float
    acceptance_score: float
    momentum_score: float
    
    # Weighted total
    total_score: float
    
    # Direction
    structure_direction: TrendDirection
    overall_direction: TrendDirection
    
    # Regime
    regime: TrendRegime
    
    # Details for debugging
    details: Dict
    
    # Timestamp
    calculated_at: datetime


class TrendAnalyzer:
    """
    Professional trend analysis using 4-component system.
    
    Components:
    1. Market Structure (30%) - Swing HH/HL detection
    2. Volatility Expansion (25%) - ATR growth analysis
    3. Acceptance (25%) - VWAP and price position
    4. Momentum Asymmetry (20%) - Impulse imbalance
    """
    
    # Component weights
    WEIGHT_STRUCTURE = 0.30
    WEIGHT_VOLATILITY = 0.25
    WEIGHT_ACCEPTANCE = 0.25
    WEIGHT_MOMENTUM = 0.20
    
    # Thresholds
    TREND_THRESHOLD = 65      # Score above this = trending
    TRANSITION_LOW = 40       # Below this = range
    
    # Structure detection params
    SWING_LOOKBACK = 5        # Bars to confirm swing
    MIN_SWINGS = 3            # Minimum swings needed
    
    # Volatility params
    ATR_PERIOD = 14
    VOL_EXPANSION_MULT = 1.2  # ATR must be 1.2x median
    
    # Momentum params
    MOMENTUM_PERIOD = 14
    ASYMMETRY_THRESHOLD = 0.15  # 15% imbalance
    
    def __init__(self):
        self.fetcher = MarketDataFetcher()
        self._cache = {}
        self._cache_ttl = 300  # 5 min cache for 4H analysis
    
    def analyze(self, symbol: str, timeframe: str = "240") -> Optional[TrendScore]:
        """
        Main analysis method - returns complete TrendScore.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            timeframe: Chart timeframe (default 240 = 4H)
            
        Returns:
            TrendScore with all components and regime
        """
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache
        if cache_key in self._cache:
            cached, timestamp = self._cache[cache_key]
            if (datetime.utcnow() - timestamp).total_seconds() < self._cache_ttl:
                return cached
        
        try:
            # Get OHLCV data
            klines = self.fetcher.get_klines(symbol, timeframe, limit=200)
            if not klines or len(klines) < 50:
                print(f"[TREND] Not enough data for {symbol}")
                return None
            
            # Convert to numpy arrays
            opens = np.array([float(k['open']) for k in klines])
            highs = np.array([float(k['high']) for k in klines])
            lows = np.array([float(k['low']) for k in klines])
            closes = np.array([float(k['close']) for k in klines])
            volumes = np.array([float(k.get('volume', 0)) for k in klines])
            
            # Calculate all components
            structure_score, structure_dir, structure_details = self._analyze_structure(
                highs, lows, closes
            )
            
            volatility_score, vol_details = self._analyze_volatility(
                highs, lows, closes
            )
            
            acceptance_score, acceptance_details = self._analyze_acceptance(
                opens, highs, lows, closes, volumes
            )
            
            momentum_score, momentum_dir, momentum_details = self._analyze_momentum(
                opens, closes
            )
            
            # Calculate weighted total
            # Adjust scores based on direction alignment
            direction_mult = self._get_direction_multiplier(
                structure_dir, momentum_dir
            )
            
            total_score = (
                self.WEIGHT_STRUCTURE * structure_score +
                self.WEIGHT_VOLATILITY * volatility_score +
                self.WEIGHT_ACCEPTANCE * acceptance_score +
                self.WEIGHT_MOMENTUM * momentum_score
            ) * direction_mult
            
            # Determine overall direction
            if structure_dir != TrendDirection.NEUTRAL:
                overall_dir = structure_dir
            elif momentum_dir != TrendDirection.NEUTRAL:
                overall_dir = momentum_dir
            else:
                overall_dir = TrendDirection.NEUTRAL
            
            # Determine regime
            regime = self._determine_regime(total_score, overall_dir)
            
            # Build result
            result = TrendScore(
                symbol=symbol,
                timeframe=timeframe,
                structure_score=structure_score,
                volatility_score=volatility_score,
                acceptance_score=acceptance_score,
                momentum_score=momentum_score,
                total_score=total_score,
                structure_direction=structure_dir,
                overall_direction=overall_dir,
                regime=regime,
                details={
                    'structure': structure_details,
                    'volatility': vol_details,
                    'acceptance': acceptance_details,
                    'momentum': momentum_details,
                    'direction_mult': direction_mult
                },
                calculated_at=datetime.utcnow()
            )
            
            # Cache result
            self._cache[cache_key] = (result, datetime.utcnow())
            
            return result
            
        except Exception as e:
            print(f"[TREND] Error analyzing {symbol}: {e}")
            return None
    
    def get_regime(self, symbol: str, timeframe: str = "240") -> TrendRegime:
        """Quick method to get just the regime"""
        result = self.analyze(symbol, timeframe)
        return result.regime if result else TrendRegime.NO_TRADE
    
    def is_tradeable(self, symbol: str, direction: str, timeframe: str = "240") -> bool:
        """
        Check if trading is allowed in given direction.
        
        Args:
            symbol: Trading pair
            direction: "LONG" or "SHORT"
            timeframe: Analysis timeframe
            
        Returns:
            True if trade direction matches trend regime
        """
        regime = self.get_regime(symbol, timeframe)
        
        if regime == TrendRegime.NO_TRADE:
            return False
        elif regime == TrendRegime.BULLISH and direction.upper() == "LONG":
            return True
        elif regime == TrendRegime.BEARISH and direction.upper() == "SHORT":
            return True
        else:
            return False
    
    # =========================================================================
    # COMPONENT 1: MARKET STRUCTURE (30%)
    # =========================================================================
    
    def _analyze_structure(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Tuple[float, TrendDirection, Dict]:
        """
        Analyze market structure using swing detection.
        
        Bullish: Higher Highs + Higher Lows
        Bearish: Lower Highs + Lower Lows
        
        Returns:
            (score 0-100, direction, details dict)
        """
        # Detect swing points
        swing_highs = self._detect_swing_highs(highs)
        swing_lows = self._detect_swing_lows(lows)
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 50.0, TrendDirection.NEUTRAL, {
                'swing_highs': len(swing_highs),
                'swing_lows': len(swing_lows),
                'reason': 'insufficient_swings'
            }
        
        # Analyze swing sequence
        hh_count, lh_count = self._count_higher_lower_highs(swing_highs)
        hl_count, ll_count = self._count_higher_lower_lows(swing_lows)
        
        total_highs = hh_count + lh_count
        total_lows = hl_count + ll_count
        
        # Calculate bullish/bearish bias
        if total_highs > 0:
            high_ratio = hh_count / total_highs  # 1.0 = all HH
        else:
            high_ratio = 0.5
            
        if total_lows > 0:
            low_ratio = hl_count / total_lows    # 1.0 = all HL
        else:
            low_ratio = 0.5
        
        # Combined structure score
        bullish_score = (high_ratio + low_ratio) / 2  # 0-1
        
        # Determine direction
        if bullish_score > 0.65:
            direction = TrendDirection.UP
            score = 50 + (bullish_score - 0.5) * 100  # 65-100
        elif bullish_score < 0.35:
            direction = TrendDirection.DOWN
            score = 50 + (0.5 - bullish_score) * 100  # 65-100
        else:
            direction = TrendDirection.NEUTRAL
            score = 50 - abs(bullish_score - 0.5) * 50  # 25-50
        
        # Check structure integrity (last swing not broken)
        integrity_bonus = self._check_structure_integrity(
            swing_highs, swing_lows, closes[-1], direction
        )
        score = min(100, score + integrity_bonus)
        
        details = {
            'swing_highs': len(swing_highs),
            'swing_lows': len(swing_lows),
            'hh_count': hh_count,
            'lh_count': lh_count,
            'hl_count': hl_count,
            'll_count': ll_count,
            'bullish_score': round(bullish_score, 3),
            'integrity_bonus': integrity_bonus
        }
        
        return round(score, 1), direction, details
    
    def _detect_swing_highs(self, highs: np.ndarray) -> List[SwingPoint]:
        """Detect swing highs using lookback confirmation"""
        swings = []
        lookback = self.SWING_LOOKBACK
        
        for i in range(lookback, len(highs) - lookback):
            # Check if this is highest in window
            window = highs[i - lookback:i + lookback + 1]
            if highs[i] == np.max(window):
                swings.append(SwingPoint(
                    index=i,
                    price=highs[i],
                    timestamp=datetime.utcnow(),  # Placeholder
                    is_high=True
                ))
        
        return swings
    
    def _detect_swing_lows(self, lows: np.ndarray) -> List[SwingPoint]:
        """Detect swing lows using lookback confirmation"""
        swings = []
        lookback = self.SWING_LOOKBACK
        
        for i in range(lookback, len(lows) - lookback):
            # Check if this is lowest in window
            window = lows[i - lookback:i + lookback + 1]
            if lows[i] == np.min(window):
                swings.append(SwingPoint(
                    index=i,
                    price=lows[i],
                    timestamp=datetime.utcnow(),
                    is_high=False
                ))
        
        return swings
    
    def _count_higher_lower_highs(self, swings: List[SwingPoint]) -> Tuple[int, int]:
        """Count HH and LH in swing sequence"""
        hh = 0
        lh = 0
        for i in range(1, len(swings)):
            if swings[i].price > swings[i-1].price:
                hh += 1
            else:
                lh += 1
        return hh, lh
    
    def _count_higher_lower_lows(self, swings: List[SwingPoint]) -> Tuple[int, int]:
        """Count HL and LL in swing sequence"""
        hl = 0
        ll = 0
        for i in range(1, len(swings)):
            if swings[i].price > swings[i-1].price:
                hl += 1
            else:
                ll += 1
        return hl, ll
    
    def _check_structure_integrity(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint],
        current_close: float,
        direction: TrendDirection
    ) -> float:
        """
        Check if structure is intact (key levels not broken).
        Returns bonus score 0-15.
        """
        if not swing_highs or not swing_lows:
            return 0.0
        
        last_high = swing_highs[-1].price
        last_low = swing_lows[-1].price
        
        if direction == TrendDirection.UP:
            # In uptrend, last HL should not be broken
            if len(swing_lows) >= 2:
                protected_low = swing_lows[-1].price
                if current_close > protected_low:
                    return 15.0  # Structure intact
                else:
                    return -10.0  # Structure broken
        
        elif direction == TrendDirection.DOWN:
            # In downtrend, last LH should not be broken
            if len(swing_highs) >= 2:
                protected_high = swing_highs[-1].price
                if current_close < protected_high:
                    return 15.0
                else:
                    return -10.0
        
        return 0.0
    
    # =========================================================================
    # COMPONENT 2: VOLATILITY REGIME (25%)
    # =========================================================================
    
    def _analyze_volatility(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Analyze volatility regime - trend needs expansion.
        
        Uses:
        - ATR growth (current vs historical)
        - Range expansion
        
        Returns:
            (score 0-100, details dict)
        """
        # Calculate ATR
        atr = self._calculate_atr(highs, lows, closes, self.ATR_PERIOD)
        
        if len(atr) < 50:
            return 50.0, {'reason': 'insufficient_data'}
        
        # Current ATR vs median
        current_atr = atr[-1]
        median_atr = np.median(atr[-50:])
        atr_ratio = current_atr / median_atr if median_atr > 0 else 1.0
        
        # ATR trend (is it expanding?)
        atr_sma_short = np.mean(atr[-5:])
        atr_sma_long = np.mean(atr[-20:])
        atr_expanding = atr_sma_short > atr_sma_long
        
        # Range analysis
        ranges = highs - lows
        current_range = ranges[-1]
        median_range = np.median(ranges[-50:])
        range_ratio = current_range / median_range if median_range > 0 else 1.0
        
        # Calculate score
        # Expansion = high score, Compression = low score
        if atr_ratio >= self.VOL_EXPANSION_MULT:
            # Strong expansion
            score = 70 + min(30, (atr_ratio - 1.2) * 50)
        elif atr_ratio >= 1.0:
            # Mild expansion
            score = 50 + (atr_ratio - 1.0) * 100
        else:
            # Compression
            score = 50 * atr_ratio
        
        # Bonus for expanding trend
        if atr_expanding:
            score = min(100, score + 10)
        
        details = {
            'current_atr': round(current_atr, 6),
            'median_atr': round(median_atr, 6),
            'atr_ratio': round(atr_ratio, 3),
            'atr_expanding': atr_expanding,
            'range_ratio': round(range_ratio, 3)
        }
        
        return round(score, 1), details
    
    def _calculate_atr(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int
    ) -> np.ndarray:
        """Calculate ATR using Wilder's smoothing"""
        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - np.roll(closes, 1)),
                np.abs(lows - np.roll(closes, 1))
            )
        )
        tr[0] = highs[0] - lows[0]
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        
        multiplier = 1 / period
        for i in range(period, len(tr)):
            atr[i] = atr[i-1] * (1 - multiplier) + tr[i] * multiplier
        
        return atr
    
    # =========================================================================
    # COMPONENT 3: ACCEPTANCE / REJECTION (25%)
    # =========================================================================
    
    def _analyze_acceptance(
        self,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        Analyze price acceptance using VWAP and close positions.
        
        Acceptance = closes above/below VWAP consistently
        Rejection = wicks and returns
        
        Returns:
            (score 0-100, details dict)
        """
        # Calculate VWAP (session-based approximation for 4H)
        vwap = self._calculate_vwap(highs, lows, closes, volumes, period=20)
        
        if len(vwap) < 20:
            return 50.0, {'reason': 'insufficient_data'}
        
        # Recent closes vs VWAP
        recent_closes = closes[-20:]
        recent_vwap = vwap[-20:]
        
        above_vwap = np.sum(recent_closes > recent_vwap)
        below_vwap = np.sum(recent_closes < recent_vwap)
        total = len(recent_closes)
        
        # Close position in range (where in daily range does it close?)
        close_positions = (closes - lows) / (highs - lows + 1e-10)
        recent_positions = close_positions[-20:]
        avg_position = np.mean(recent_positions)  # 0.5 = middle, 1.0 = at high
        
        # Wick analysis (rejection detection)
        upper_wicks = highs - np.maximum(opens, closes)
        lower_wicks = np.minimum(opens, closes) - lows
        bodies = np.abs(closes - opens)
        
        # Wick ratio (high = rejection)
        recent_upper_wick_ratio = np.mean(upper_wicks[-10:]) / (np.mean(bodies[-10:]) + 1e-10)
        recent_lower_wick_ratio = np.mean(lower_wicks[-10:]) / (np.mean(bodies[-10:]) + 1e-10)
        
        # Calculate score
        if above_vwap > below_vwap:
            # Bullish acceptance
            acceptance_ratio = above_vwap / total
            base_score = 50 + acceptance_ratio * 40
            
            # Position bonus (closing near highs = stronger)
            position_bonus = (avg_position - 0.5) * 20
            
            # Wick penalty (upper wicks = rejection at highs)
            wick_penalty = min(15, recent_upper_wick_ratio * 10)
            
        elif below_vwap > above_vwap:
            # Bearish acceptance
            acceptance_ratio = below_vwap / total
            base_score = 50 + acceptance_ratio * 40
            
            # Position bonus (closing near lows = stronger)
            position_bonus = (0.5 - avg_position) * 20
            
            # Wick penalty
            wick_penalty = min(15, recent_lower_wick_ratio * 10)
        else:
            base_score = 50
            position_bonus = 0
            wick_penalty = 0
        
        score = max(0, min(100, base_score + position_bonus - wick_penalty))
        
        details = {
            'above_vwap': int(above_vwap),
            'below_vwap': int(below_vwap),
            'avg_close_position': round(avg_position, 3),
            'upper_wick_ratio': round(recent_upper_wick_ratio, 3),
            'lower_wick_ratio': round(recent_lower_wick_ratio, 3),
            'current_vs_vwap': 'above' if closes[-1] > vwap[-1] else 'below'
        }
        
        return round(score, 1), details
    
    def _calculate_vwap(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        period: int = 20
    ) -> np.ndarray:
        """Calculate rolling VWAP"""
        typical_price = (highs + lows + closes) / 3
        
        # Handle zero volume
        volumes = np.where(volumes == 0, 1, volumes)
        
        # Rolling VWAP
        vwap = np.zeros_like(closes)
        for i in range(period - 1, len(closes)):
            start = max(0, i - period + 1)
            tp_slice = typical_price[start:i+1]
            vol_slice = volumes[start:i+1]
            vwap[i] = np.sum(tp_slice * vol_slice) / np.sum(vol_slice)
        
        return vwap
    
    # =========================================================================
    # COMPONENT 4: MOMENTUM ASYMMETRY (20%)
    # =========================================================================
    
    def _analyze_momentum(
        self,
        opens: np.ndarray,
        closes: np.ndarray
    ) -> Tuple[float, TrendDirection, Dict]:
        """
        Analyze momentum asymmetry - are up moves stronger than down moves?
        
        NOT traditional RSI, but imbalance of impulses.
        
        Returns:
            (score 0-100, direction, details dict)
        """
        # Calculate returns
        returns = (closes[1:] - closes[:-1]) / closes[:-1]
        
        if len(returns) < self.MOMENTUM_PERIOD:
            return 50.0, TrendDirection.NEUTRAL, {'reason': 'insufficient_data'}
        
        recent_returns = returns[-self.MOMENTUM_PERIOD:]
        
        # Separate up and down moves
        up_moves = recent_returns[recent_returns > 0]
        down_moves = recent_returns[recent_returns < 0]
        
        # Average magnitude
        avg_up = np.mean(up_moves) if len(up_moves) > 0 else 0
        avg_down = abs(np.mean(down_moves)) if len(down_moves) > 0 else 0
        
        # Asymmetry ratio
        total_magnitude = avg_up + avg_down
        if total_magnitude > 0:
            asymmetry = (avg_up - avg_down) / total_magnitude  # -1 to +1
        else:
            asymmetry = 0
        
        # Count ratio
        up_count = len(up_moves)
        down_count = len(down_moves)
        count_ratio = up_count / (up_count + down_count) if (up_count + down_count) > 0 else 0.5
        
        # Impulse continuation (do moves continue or reverse?)
        continuation_rate = self._calculate_continuation_rate(returns)
        
        # Determine direction and score
        if asymmetry > self.ASYMMETRY_THRESHOLD:
            direction = TrendDirection.UP
            score = 50 + asymmetry * 50 + (count_ratio - 0.5) * 20
        elif asymmetry < -self.ASYMMETRY_THRESHOLD:
            direction = TrendDirection.DOWN
            score = 50 + abs(asymmetry) * 50 + (0.5 - count_ratio) * 20
        else:
            direction = TrendDirection.NEUTRAL
            score = 50 - abs(asymmetry) * 30
        
        # Continuation bonus
        if continuation_rate > 0.5:
            score = min(100, score + (continuation_rate - 0.5) * 20)
        
        score = max(0, min(100, score))
        
        details = {
            'avg_up_move': round(avg_up * 100, 4),  # As percentage
            'avg_down_move': round(avg_down * 100, 4),
            'asymmetry': round(asymmetry, 4),
            'up_count': up_count,
            'down_count': down_count,
            'count_ratio': round(count_ratio, 3),
            'continuation_rate': round(continuation_rate, 3)
        }
        
        return round(score, 1), direction, details
    
    def _calculate_continuation_rate(self, returns: np.ndarray) -> float:
        """Calculate how often moves continue vs reverse"""
        if len(returns) < 2:
            return 0.5
        
        continuations = 0
        for i in range(1, len(returns)):
            # Same sign = continuation
            if (returns[i] > 0 and returns[i-1] > 0) or \
               (returns[i] < 0 and returns[i-1] < 0):
                continuations += 1
        
        return continuations / (len(returns) - 1)
    
    # =========================================================================
    # HELPERS
    # =========================================================================
    
    def _get_direction_multiplier(
        self,
        structure_dir: TrendDirection,
        momentum_dir: TrendDirection
    ) -> float:
        """
        Adjust score based on direction alignment.
        
        Aligned = boost
        Conflicting = penalty
        """
        if structure_dir == TrendDirection.NEUTRAL or momentum_dir == TrendDirection.NEUTRAL:
            return 1.0
        
        if structure_dir == momentum_dir:
            return 1.1  # 10% boost for alignment
        else:
            return 0.85  # 15% penalty for conflict
    
    def _determine_regime(
        self,
        total_score: float,
        direction: TrendDirection
    ) -> TrendRegime:
        """Determine final regime based on score and direction"""
        if total_score >= self.TREND_THRESHOLD:
            if direction == TrendDirection.UP:
                return TrendRegime.BULLISH
            elif direction == TrendDirection.DOWN:
                return TrendRegime.BEARISH
            else:
                return TrendRegime.NO_TRADE
        elif total_score >= self.TRANSITION_LOW:
            # Transition zone - be cautious
            return TrendRegime.NO_TRADE
        else:
            return TrendRegime.NO_TRADE
    
    def clear_cache(self):
        """Clear analysis cache"""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'entries': len(self._cache),
            'ttl_seconds': self._cache_ttl
        }


# Singleton instance
_analyzer_instance = None

def get_trend_analyzer() -> TrendAnalyzer:
    """Get singleton TrendAnalyzer instance"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TrendAnalyzer()
    return _analyzer_instance
