"""
Direction Engine v6.0 - Professional Direction System with Market Structure Analysis

НОВІ ФІЧІ v6:
1. MSS (Market Structure Shift) - детектор HL/LH всередині консолідації
2. OI + Volume Delta Divergence - накопичення/розподіл
3. POC Positioning - ціна вище/нижче Point of Control
4. Покращений scoring з вагами для кожного сигналу

КЛЮЧОВІ ВІДМІННОСТІ від v5:
1. PHASE DETECTION - визначає фазу ринку (accumulation/markup/distribution/markdown)
2. EXHAUSTION DETECTION - бачить коли тренд вичерпався
3. REVERSAL SIGNALS - шукає розвороти на S/R рівнях
4. MSS DETECTION - Higher Lows = LONG, Lower Highs = SHORT
5. OI DIVERGENCE - OI↑ + Delta↑ = LONG, OI↑ + Delta↓ = SHORT

КРИТИЧНО: 
- На ДНІ → шукаємо LONG (не short!)
- На ТОПІ → шукаємо SHORT (не long!)
- В середині тренду → торгуємо з трендом

Автор: SVV Bot Team
Версія: 6.0 (2026-01-26)
"""

from typing import Dict, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import statistics

from core.market_data import get_fetcher
from core.tech_indicators import get_indicators


# ============================================
# ENUMS & DATA CLASSES
# ============================================

class MarketPhase(Enum):
    """4 фази ринку за Вайкоффом"""
    ACCUMULATION = "ACCUMULATION"   # Дно, великі гравці купують
    MARKUP = "MARKUP"               # Зростання
    DISTRIBUTION = "DISTRIBUTION"   # Топ, великі гравці продають
    MARKDOWN = "MARKDOWN"           # Падіння
    UNKNOWN = "UNKNOWN"


class PhaseMaturity(Enum):
    """Зрілість фази"""
    EARLY = "EARLY"       # Початок фази
    MIDDLE = "MIDDLE"     # Середина фази  
    LATE = "LATE"         # Кінець фази
    EXHAUSTED = "EXHAUSTED"  # Фаза вичерпана, очікується розворот


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class TrendStrength(Enum):
    STRONG = "STRONG"     # ADX > 30
    MODERATE = "MODERATE" # ADX 20-30
    WEAK = "WEAK"         # ADX < 20
    TRENDLESS = "TRENDLESS"  # ADX < 15


@dataclass
class ExhaustionSignals:
    """Сигнали вичерпання тренду"""
    price_exhaustion: float = 0.0     # 0-1: наскільки велике падіння/зростання
    volume_exhaustion: float = 0.0    # 0-1: declining volume = exhaustion
    rsi_divergence: str = "none"      # bullish/bearish/none
    rsi_extreme: bool = False         # RSI < 25 or > 75
    at_support: bool = False          # Біля support level
    at_resistance: bool = False       # Біля resistance level
    consecutive_candles: int = 0      # Кількість односпрямованих свічок
    
    @property
    def is_exhausted(self) -> bool:
        """Чи є ознаки вичерпання"""
        signals = 0
        if self.price_exhaustion > 0.6:
            signals += 1
        if self.volume_exhaustion > 0.5:
            signals += 1
        if self.rsi_divergence != "none":
            signals += 2  # Дивергенція - сильний сигнал
        if self.rsi_extreme:
            signals += 1
        if self.at_support or self.at_resistance:
            signals += 1
        if self.consecutive_candles >= 5:
            signals += 1
        return signals >= 3
    
    @property
    def exhaustion_score(self) -> float:
        """Score вичерпання 0-1"""
        score = 0.0
        score += self.price_exhaustion * 0.25
        score += self.volume_exhaustion * 0.2
        if self.rsi_divergence == "bullish" or self.rsi_divergence == "bearish":
            score += 0.3
        if self.rsi_extreme:
            score += 0.1
        if self.at_support or self.at_resistance:
            score += 0.1
        score += min(self.consecutive_candles / 10, 0.1)
        return min(score, 1.0)


@dataclass
class StructureAnalysis:
    """Аналіз структури ринку"""
    phase: MarketPhase = MarketPhase.UNKNOWN
    maturity: PhaseMaturity = PhaseMaturity.MIDDLE
    trend_direction: str = "neutral"  # up/down/neutral
    trend_strength: TrendStrength = TrendStrength.WEAK
    
    # Price levels
    current_price: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    
    # Key levels
    recent_high: float = 0.0
    recent_low: float = 0.0
    support_level: float = 0.0
    resistance_level: float = 0.0
    
    # Metrics
    price_change_5d: float = 0.0
    price_change_20d: float = 0.0
    distance_from_high: float = 0.0  # % від останнього хая
    distance_from_low: float = 0.0   # % від останнього лоу
    
    adx_value: float = 0.0
    
    # ===== v6: Market Structure Shift (MSS) =====
    mss_bias: str = "neutral"        # "long" / "short" / "neutral"
    higher_lows_count: int = 0       # Кількість Higher Lows в консолідації
    lower_highs_count: int = 0       # Кількість Lower Highs в консолідації
    mss_strength: float = 0.0        # 0-1: сила сигналу MSS
    
    # ===== v6: OI + Volume Delta =====
    oi_delta_bias: str = "neutral"   # "long" / "short" / "neutral"
    oi_growing: bool = False         # OI зростає?
    volume_delta: float = 0.0        # Positive = buyers, Negative = sellers
    
    # ===== v6: POC (Point of Control) =====
    poc_bias: str = "neutral"        # "long" / "short" / "neutral"
    poc_price: float = 0.0
    price_vs_poc: float = 0.0        # % вище/нижче POC


@dataclass
class DirectionResultV5:
    """Результат Direction Engine v5"""
    direction: Direction
    score: float  # -1.0 to +1.0
    confidence: str  # HIGH, MEDIUM, LOW
    
    # Phase info
    market_phase: MarketPhase
    phase_maturity: PhaseMaturity
    
    # Exhaustion
    exhaustion: ExhaustionSignals
    is_reversal_setup: bool
    
    # Structure
    structure: StructureAnalysis
    
    # Reasoning
    primary_reason: str
    secondary_reasons: List[str] = field(default_factory=list)
    
    # Trading style specific
    trading_style: str = "SWING"
    recommended_timeframe: str = "4h"


# ============================================
# DIRECTION ENGINE v5
# ============================================

class DirectionEngineV5:
    """
    Professional Direction Engine with:
    1. Market Phase Detection (Wyckoff-inspired)
    2. Exhaustion Detection (reversal anticipation)
    3. Multi-Timeframe Analysis
    4. Support/Resistance awareness
    5. Smart Direction Resolution
    
    ГОЛОВНА ЛОГІКА:
    - Trend EXHAUSTED at SUPPORT → LONG (не short!)
    - Trend EXHAUSTED at RESISTANCE → SHORT (не long!)
    - Strong trend NOT exhausted → trade WITH trend
    - Unclear → NEUTRAL
    """
    
    def __init__(self, trading_style: str = "SWING"):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.trading_style = trading_style
        
        # Timeframes based on trading style
        if trading_style == "SCALPING":
            self.htf = "1h"      # Higher timeframe
            self.mtf = "15m"     # Medium timeframe
            self.ltf = "5m"      # Lower timeframe
            self.lookback_days = 5
        else:  # SWING
            self.htf = "1d"      # Higher timeframe
            self.mtf = "4h"      # Medium timeframe
            self.ltf = "1h"      # Lower timeframe
            self.lookback_days = 20
    
    def resolve(self, symbol: str,
                klines_htf: List[Dict] = None,
                klines_mtf: List[Dict] = None,
                oi_change: float = None,
                funding_rate: float = None,
                ob_imbalance: float = None,  # v6: Order Book imbalance %
                poc_price: float = None) -> DirectionResultV5:  # v6: Point of Control price
        """
        Main entry point - resolve direction with full analysis
        
        v6 additions:
        - ob_imbalance: Order book bid/ask imbalance (positive = more bids)
        - poc_price: Volume Profile Point of Control price
        """
        
        # Fetch data if not provided
        if klines_htf is None:
            klines_htf = self.fetcher.get_klines(symbol, self.htf, limit=100)
        if klines_mtf is None:
            klines_mtf = self.fetcher.get_klines(symbol, self.mtf, limit=100)
        
        if not klines_htf or len(klines_htf) < 50:
            return self._neutral_result("Insufficient HTF data")
        
        if not klines_mtf or len(klines_mtf) < 30:
            return self._neutral_result("Insufficient MTF data")
        
        # === STEP 1: Structure Analysis ===
        structure = self._analyze_structure(klines_htf, klines_mtf)
        
        # === STEP 1.5 (v6): Additional Structure Signals ===
        # OI + Volume Delta divergence
        self._detect_oi_volume_divergence(structure, oi_change, ob_imbalance)
        # POC positioning
        self._detect_poc_positioning(structure, poc_price)
        
        # === STEP 2: Exhaustion Detection ===
        exhaustion = self._detect_exhaustion(klines_htf, klines_mtf, structure)
        
        # === STEP 3: Phase Detection ===
        phase, maturity = self._detect_phase(structure, exhaustion)
        structure.phase = phase
        structure.maturity = maturity
        
        # === STEP 4: Reversal Check ===
        is_reversal, reversal_direction = self._check_reversal_setup(
            structure, exhaustion, phase, maturity
        )
        
        # === STEP 5: Final Direction Resolution ===
        direction, score, reason = self._resolve_direction(
            structure, exhaustion, phase, maturity,
            is_reversal, reversal_direction,
            oi_change, funding_rate
        )
        
        # Confidence based on score
        abs_score = abs(score)
        if abs_score >= 0.7:
            confidence = "HIGH"
        elif abs_score >= 0.4:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return DirectionResultV5(
            direction=direction,
            score=round(score, 3),
            confidence=confidence,
            market_phase=phase,
            phase_maturity=maturity,
            exhaustion=exhaustion,
            is_reversal_setup=is_reversal,
            structure=structure,
            primary_reason=reason,
            secondary_reasons=self._get_secondary_reasons(structure, exhaustion),
            trading_style=self.trading_style,
            recommended_timeframe=self.mtf
        )
    
    # ============================================
    # STRUCTURE ANALYSIS
    # ============================================
    
    def _analyze_structure(self, klines_htf: List[Dict], 
                          klines_mtf: List[Dict]) -> StructureAnalysis:
        """Analyze market structure"""
        
        structure = StructureAnalysis()
        
        closes_htf = [k['close'] for k in klines_htf]
        highs_htf = [k['high'] for k in klines_htf]
        lows_htf = [k['low'] for k in klines_htf]
        
        structure.current_price = closes_htf[-1]
        
        # EMAs
        if len(closes_htf) >= 200:
            structure.ema_200 = self.indicators.ema(closes_htf, 200)[-1]
        if len(closes_htf) >= 50:
            structure.ema_50 = self.indicators.ema(closes_htf, 50)[-1]
        if len(closes_htf) >= 20:
            structure.ema_20 = self.indicators.ema(closes_htf, 20)[-1]
        
        # Recent high/low (20 periods)
        lookback = min(20, len(klines_htf))
        structure.recent_high = max(highs_htf[-lookback:])
        structure.recent_low = min(lows_htf[-lookback:])
        
        # Price changes
        if len(closes_htf) >= 5:
            structure.price_change_5d = (closes_htf[-1] - closes_htf[-5]) / closes_htf[-5] * 100
        if len(closes_htf) >= 20:
            structure.price_change_20d = (closes_htf[-1] - closes_htf[-20]) / closes_htf[-20] * 100
        
        # Distance from high/low
        price_range = structure.recent_high - structure.recent_low
        if price_range > 0:
            structure.distance_from_high = (structure.recent_high - structure.current_price) / price_range * 100
            structure.distance_from_low = (structure.current_price - structure.recent_low) / price_range * 100
        
        # Support/Resistance levels
        structure.support_level = self._find_support(klines_htf)
        structure.resistance_level = self._find_resistance(klines_htf)
        
        # Trend direction and strength
        structure.trend_direction = self._determine_trend_direction(closes_htf, structure)
        structure.trend_strength, structure.adx_value = self._calculate_trend_strength(klines_mtf)
        
        # v6: Detect Market Structure Shift (HL/LH pattern)
        self._detect_mss(klines_htf, structure)
        
        return structure
    
    def _find_support(self, klines: List[Dict]) -> float:
        """Find nearest support level using pivot lows"""
        lows = [k['low'] for k in klines]
        current_price = klines[-1]['close']
        
        # Find pivot lows (local minimums)
        pivot_lows = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivot_lows.append(lows[i])
        
        # Find nearest support below current price
        supports_below = [p for p in pivot_lows if p < current_price]
        if supports_below:
            return max(supports_below)  # Nearest support
        
        return min(lows[-20:]) if len(lows) >= 20 else min(lows)
    
    def _find_resistance(self, klines: List[Dict]) -> float:
        """Find nearest resistance level using pivot highs"""
        highs = [k['high'] for k in klines]
        current_price = klines[-1]['close']
        
        # Find pivot highs (local maximums)
        pivot_highs = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivot_highs.append(highs[i])
        
        # Find nearest resistance above current price
        resistances_above = [p for p in pivot_highs if p > current_price]
        if resistances_above:
            return min(resistances_above)  # Nearest resistance
        
        return max(highs[-20:]) if len(highs) >= 20 else max(highs)
    
    def _detect_mss(self, klines: List[Dict], structure: StructureAnalysis) -> None:
        """
        Detect Market Structure Shift (MSS) inside consolidation
        
        v6: Професійний аналіз структури:
        - Higher Lows (HL) = лімітний покупець витісняє продавців → LONG bias
        - Lower Highs (LH) = агресивний продавець тисне на ринок → SHORT bias
        
        Працює найкраще в фазі "сну" (низька волатильність, ADX < 25)
        """
        if len(klines) < 20:
            return
        
        highs = [k['high'] for k in klines[-50:]]  # Last 50 candles
        lows = [k['low'] for k in klines[-50:]]
        
        # Find pivot points (swing highs and swing lows)
        # Using 2-bar lookback/lookforward for simple pivot detection
        pivot_highs = []  # (index, price)
        pivot_lows = []   # (index, price)
        
        for i in range(2, len(highs) - 2):
            # Pivot High: higher than 2 bars on each side
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                pivot_highs.append((i, highs[i]))
            
            # Pivot Low: lower than 2 bars on each side
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                pivot_lows.append((i, lows[i]))
        
        if len(pivot_lows) < 2 or len(pivot_highs) < 2:
            return
        
        # Analyze last 3-4 pivot lows for Higher Lows pattern
        recent_lows = pivot_lows[-4:]  # Last 4 pivot lows
        hl_count = 0
        for i in range(1, len(recent_lows)):
            if recent_lows[i][1] > recent_lows[i-1][1]:
                hl_count += 1
        
        # Analyze last 3-4 pivot highs for Lower Highs pattern
        recent_highs = pivot_highs[-4:]  # Last 4 pivot highs
        lh_count = 0
        for i in range(1, len(recent_highs)):
            if recent_highs[i][1] < recent_highs[i-1][1]:
                lh_count += 1
        
        structure.higher_lows_count = hl_count
        structure.lower_highs_count = lh_count
        
        # Determine MSS bias
        # Need at least 2 consecutive HL or LH to be significant
        if hl_count >= 2 and lh_count <= 1:
            structure.mss_bias = "long"
            structure.mss_strength = min(1.0, hl_count / 3)  # Max at 3 HLs
        elif lh_count >= 2 and hl_count <= 1:
            structure.mss_bias = "short"
            structure.mss_strength = min(1.0, lh_count / 3)  # Max at 3 LHs
        else:
            structure.mss_bias = "neutral"
            structure.mss_strength = 0.0
    
    def _detect_oi_volume_divergence(self, structure: StructureAnalysis,
                                      oi_change: float = None,
                                      ob_imbalance: float = None) -> None:
        """
        Detect OI + Volume Delta divergence
        
        v6: Професійний аналіз накопичення/розподілу:
        - OI↑ + positive delta = накопичення (LONG)
        - OI↑ + negative delta = розподіл (SHORT)
        
        Використовуємо OB Imbalance як proxy для Volume Delta:
        - Bid > Ask = покупці агресивніші = positive delta
        - Ask > Bid = продавці агресивніші = negative delta
        """
        # OI growing?
        structure.oi_growing = oi_change is not None and oi_change > 2  # >2% growth
        
        # Use OB Imbalance as proxy for volume delta
        # Imbalance: positive = more bids, negative = more asks
        if ob_imbalance is not None:
            structure.volume_delta = ob_imbalance
        
        # Determine OI + Delta bias
        if structure.oi_growing:
            if structure.volume_delta > 10:  # >10% bid imbalance
                structure.oi_delta_bias = "long"
            elif structure.volume_delta < -10:  # >10% ask imbalance
                structure.oi_delta_bias = "short"
            else:
                structure.oi_delta_bias = "neutral"
        else:
            structure.oi_delta_bias = "neutral"
    
    def _detect_poc_positioning(self, structure: StructureAnalysis,
                                 poc_price: float = None) -> None:
        """
        Detect price positioning relative to POC (Point of Control)
        
        v6: Професійний аналіз Volume Profile:
        - Price > POC = POC як підтримка → LONG bias
        - Price < POC = POC як опір → SHORT bias
        """
        if poc_price is None or poc_price <= 0:
            return
        
        structure.poc_price = poc_price
        
        # Calculate distance from POC
        if structure.current_price > 0:
            structure.price_vs_poc = ((structure.current_price - poc_price) / poc_price) * 100
        
        # Determine POC bias (need >0.5% distance to be significant)
        if structure.price_vs_poc > 0.5:
            structure.poc_bias = "long"
        elif structure.price_vs_poc < -0.5:
            structure.poc_bias = "short"
        else:
            structure.poc_bias = "neutral"
    
    def _determine_trend_direction(self, closes: List[float], 
                                   structure: StructureAnalysis) -> str:
        """Determine overall trend direction"""
        
        # Check EMA alignment
        ema_bullish = 0
        ema_bearish = 0
        
        if structure.ema_20 and structure.ema_50:
            if structure.ema_20 > structure.ema_50:
                ema_bullish += 1
            else:
                ema_bearish += 1
        
        if structure.ema_50 and structure.ema_200:
            if structure.ema_50 > structure.ema_200:
                ema_bullish += 1
            else:
                ema_bearish += 1
        
        if structure.current_price > structure.ema_50 if structure.ema_50 else False:
            ema_bullish += 1
        else:
            ema_bearish += 1
        
        # Price changes
        if structure.price_change_20d > 5:
            ema_bullish += 1
        elif structure.price_change_20d < -5:
            ema_bearish += 1
        
        if ema_bullish > ema_bearish + 1:
            return "up"
        elif ema_bearish > ema_bullish + 1:
            return "down"
        return "neutral"
    
    def _calculate_trend_strength(self, klines: List[Dict]) -> Tuple[TrendStrength, float]:
        """Calculate ADX-based trend strength"""
        
        if len(klines) < 20:
            return TrendStrength.WEAK, 0
        
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        closes = [k['close'] for k in klines]
        
        try:
            adx = self.indicators.adx(highs, lows, closes, period=14)
            adx_value = adx[-1] if adx else 0
        except:
            adx_value = 0
        
        if adx_value >= 30:
            return TrendStrength.STRONG, adx_value
        elif adx_value >= 20:
            return TrendStrength.MODERATE, adx_value
        elif adx_value >= 15:
            return TrendStrength.WEAK, adx_value
        else:
            return TrendStrength.TRENDLESS, adx_value
    
    # ============================================
    # EXHAUSTION DETECTION
    # ============================================
    
    def _detect_exhaustion(self, klines_htf: List[Dict], 
                          klines_mtf: List[Dict],
                          structure: StructureAnalysis) -> ExhaustionSignals:
        """
        Detect trend exhaustion signals
        
        КРИТИЧНО: Це ключова функція для визначення чи тренд закінчується!
        """
        
        exhaustion = ExhaustionSignals()
        
        closes = [k['close'] for k in klines_htf]
        volumes = [k['volume'] for k in klines_htf]
        highs = [k['high'] for k in klines_htf]
        lows = [k['low'] for k in klines_htf]
        
        # === 1. PRICE EXHAUSTION ===
        # Великий рух без значної корекції = exhaustion
        
        # Calculate move magnitude
        lookback = min(20, len(closes) - 1)
        max_price = max(closes[-lookback:])
        min_price = min(closes[-lookback:])
        
        if max_price > min_price:
            total_range = (max_price - min_price) / min_price * 100
            
            # Current position in range
            if structure.trend_direction == "down":
                # В даунтренді - наскільки близько до мінімуму
                position = (closes[-1] - min_price) / (max_price - min_price)
                if position < 0.2 and total_range > 10:  # Близько до дна
                    exhaustion.price_exhaustion = min(total_range / 20, 1.0)
            elif structure.trend_direction == "up":
                # В аптренді - наскільки близько до максимуму
                position = (closes[-1] - min_price) / (max_price - min_price)
                if position > 0.8 and total_range > 10:  # Близько до топу
                    exhaustion.price_exhaustion = min(total_range / 20, 1.0)
        
        # === 2. VOLUME EXHAUSTION ===
        # Declining volume in trend = exhaustion
        
        if len(volumes) >= 10:
            # Compare recent volume to earlier volume
            recent_vol = statistics.mean(volumes[-5:])
            earlier_vol = statistics.mean(volumes[-10:-5])
            
            if earlier_vol > 0:
                vol_ratio = recent_vol / earlier_vol
                if vol_ratio < 0.7:  # Volume declining
                    exhaustion.volume_exhaustion = min((1 - vol_ratio) * 2, 1.0)
        
        # === 3. RSI DIVERGENCE ===
        # Price makes new low/high but RSI doesn't = divergence
        
        indicators = self.indicators.calculate_all(klines_mtf)
        rsi_values = indicators.get('rsi', [])
        
        if len(rsi_values) >= 10:
            # Check for bullish divergence (price lower low, RSI higher low)
            if structure.trend_direction == "down":
                if self._detect_bullish_divergence(closes[-20:], rsi_values[-20:]):
                    exhaustion.rsi_divergence = "bullish"
            
            # Check for bearish divergence (price higher high, RSI lower high)
            elif structure.trend_direction == "up":
                if self._detect_bearish_divergence(closes[-20:], rsi_values[-20:]):
                    exhaustion.rsi_divergence = "bearish"
        
        # RSI extreme levels
        current_rsi = rsi_values[-1] if rsi_values else 50
        if current_rsi < 25 or current_rsi > 75:
            exhaustion.rsi_extreme = True
        
        # === 4. SUPPORT/RESISTANCE PROXIMITY ===
        
        current_price = structure.current_price
        
        # Near support
        if structure.support_level > 0:
            support_distance = (current_price - structure.support_level) / current_price * 100
            if support_distance < 2:  # Within 2% of support
                exhaustion.at_support = True
        
        # Near resistance
        if structure.resistance_level > 0:
            resistance_distance = (structure.resistance_level - current_price) / current_price * 100
            if resistance_distance < 2:  # Within 2% of resistance
                exhaustion.at_resistance = True
        
        # === 5. CONSECUTIVE CANDLES ===
        # Count consecutive same-direction candles
        
        consecutive = 0
        for i in range(len(closes) - 1, 0, -1):
            if structure.trend_direction == "down":
                if closes[i] < closes[i-1]:
                    consecutive += 1
                else:
                    break
            elif structure.trend_direction == "up":
                if closes[i] > closes[i-1]:
                    consecutive += 1
                else:
                    break
        
        exhaustion.consecutive_candles = consecutive
        
        return exhaustion
    
    def _detect_bullish_divergence(self, prices: List[float], 
                                   rsi: List[float]) -> bool:
        """
        Bullish divergence: Price makes lower low, RSI makes higher low
        """
        if len(prices) < 10 or len(rsi) < 10:
            return False
        
        # Find two recent lows in price
        price_lows = []
        for i in range(2, len(prices) - 2):
            if prices[i] < prices[i-1] and prices[i] < prices[i-2] and \
               prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                price_lows.append((i, prices[i]))
        
        if len(price_lows) < 2:
            return False
        
        # Get last two lows
        first_low = price_lows[-2]
        second_low = price_lows[-1]
        
        # Price lower low
        if second_low[1] >= first_low[1]:
            return False
        
        # RSI higher low (check RSI at those price points)
        idx1, idx2 = first_low[0], second_low[0]
        if idx1 < len(rsi) and idx2 < len(rsi):
            if rsi[idx2] > rsi[idx1]:
                return True
        
        return False
    
    def _detect_bearish_divergence(self, prices: List[float], 
                                   rsi: List[float]) -> bool:
        """
        Bearish divergence: Price makes higher high, RSI makes lower high
        """
        if len(prices) < 10 or len(rsi) < 10:
            return False
        
        # Find two recent highs in price
        price_highs = []
        for i in range(2, len(prices) - 2):
            if prices[i] > prices[i-1] and prices[i] > prices[i-2] and \
               prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                price_highs.append((i, prices[i]))
        
        if len(price_highs) < 2:
            return False
        
        # Get last two highs
        first_high = price_highs[-2]
        second_high = price_highs[-1]
        
        # Price higher high
        if second_high[1] <= first_high[1]:
            return False
        
        # RSI lower high
        idx1, idx2 = first_high[0], second_high[0]
        if idx1 < len(rsi) and idx2 < len(rsi):
            if rsi[idx2] < rsi[idx1]:
                return True
        
        return False
    
    # ============================================
    # PHASE DETECTION
    # ============================================
    
    def _detect_phase(self, structure: StructureAnalysis,
                     exhaustion: ExhaustionSignals) -> Tuple[MarketPhase, PhaseMaturity]:
        """
        Detect market phase (Wyckoff-inspired)
        
        ACCUMULATION: At support, exhausted downtrend, volume building
        MARKUP: Uptrend, above EMAs, healthy volume
        DISTRIBUTION: At resistance, exhausted uptrend, volume building
        MARKDOWN: Downtrend, below EMAs
        """
        
        phase = MarketPhase.UNKNOWN
        maturity = PhaseMaturity.MIDDLE
        
        trend = structure.trend_direction
        exhausted = exhaustion.is_exhausted
        at_support = exhaustion.at_support
        at_resistance = exhaustion.at_resistance
        
        # === ACCUMULATION ===
        # Downtrend that's exhausted near support
        if trend == "down" and exhausted and at_support:
            phase = MarketPhase.ACCUMULATION
            maturity = PhaseMaturity.LATE
        elif trend == "down" and exhaustion.exhaustion_score > 0.4 and structure.distance_from_low < 20:
            phase = MarketPhase.ACCUMULATION
            maturity = PhaseMaturity.EARLY
        
        # === DISTRIBUTION ===
        # Uptrend that's exhausted near resistance
        elif trend == "up" and exhausted and at_resistance:
            phase = MarketPhase.DISTRIBUTION
            maturity = PhaseMaturity.LATE
        elif trend == "up" and exhaustion.exhaustion_score > 0.4 and structure.distance_from_high < 20:
            phase = MarketPhase.DISTRIBUTION
            maturity = PhaseMaturity.EARLY
        
        # === MARKUP (Uptrend) ===
        elif trend == "up" and not exhausted:
            phase = MarketPhase.MARKUP
            if structure.price_change_5d > 5:
                maturity = PhaseMaturity.EARLY
            elif structure.price_change_20d > 15:
                maturity = PhaseMaturity.LATE
            else:
                maturity = PhaseMaturity.MIDDLE
        
        # === MARKDOWN (Downtrend) ===
        elif trend == "down" and not exhausted:
            phase = MarketPhase.MARKDOWN
            if structure.price_change_5d < -5:
                maturity = PhaseMaturity.EARLY
            elif structure.price_change_20d < -15:
                maturity = PhaseMaturity.LATE
            else:
                maturity = PhaseMaturity.MIDDLE
        
        # Check if exhausted
        if exhausted:
            maturity = PhaseMaturity.EXHAUSTED
        
        return phase, maturity
    
    # ============================================
    # REVERSAL DETECTION
    # ============================================
    
    def _check_reversal_setup(self, structure: StructureAnalysis,
                              exhaustion: ExhaustionSignals,
                              phase: MarketPhase,
                              maturity: PhaseMaturity) -> Tuple[bool, Direction]:
        """
        Check if this is a reversal setup
        
        BULLISH REVERSAL (LONG):
        - Downtrend exhausted
        - At or near support
        - RSI bullish divergence or oversold
        
        BEARISH REVERSAL (SHORT):
        - Uptrend exhausted
        - At or near resistance
        - RSI bearish divergence or overbought
        """
        
        is_reversal = False
        reversal_direction = Direction.NEUTRAL
        
        # === BULLISH REVERSAL ===
        if phase in [MarketPhase.ACCUMULATION, MarketPhase.MARKDOWN]:
            if maturity in [PhaseMaturity.LATE, PhaseMaturity.EXHAUSTED]:
                bullish_signals = 0
                
                if exhaustion.at_support:
                    bullish_signals += 2
                if exhaustion.rsi_divergence == "bullish":
                    bullish_signals += 2
                if exhaustion.rsi_extreme and structure.trend_direction == "down":
                    bullish_signals += 1
                if exhaustion.volume_exhaustion > 0.5:
                    bullish_signals += 1
                if structure.distance_from_low < 15:  # Close to low
                    bullish_signals += 1
                
                if bullish_signals >= 3:
                    is_reversal = True
                    reversal_direction = Direction.LONG
        
        # === BEARISH REVERSAL ===
        if phase in [MarketPhase.DISTRIBUTION, MarketPhase.MARKUP]:
            if maturity in [PhaseMaturity.LATE, PhaseMaturity.EXHAUSTED]:
                bearish_signals = 0
                
                if exhaustion.at_resistance:
                    bearish_signals += 2
                if exhaustion.rsi_divergence == "bearish":
                    bearish_signals += 2
                if exhaustion.rsi_extreme and structure.trend_direction == "up":
                    bearish_signals += 1
                if exhaustion.volume_exhaustion > 0.5:
                    bearish_signals += 1
                if structure.distance_from_high < 15:  # Close to high
                    bearish_signals += 1
                
                if bearish_signals >= 3:
                    is_reversal = True
                    reversal_direction = Direction.SHORT
        
        return is_reversal, reversal_direction
    
    # ============================================
    # DIRECTION RESOLUTION
    # ============================================
    
    def _resolve_direction(self, structure: StructureAnalysis,
                          exhaustion: ExhaustionSignals,
                          phase: MarketPhase,
                          maturity: PhaseMaturity,
                          is_reversal: bool,
                          reversal_direction: Direction,
                          oi_change: float = None,
                          funding_rate: float = None) -> Tuple[Direction, float, str]:
        """
        MAIN DECISION LOGIC v6
        
        v6 scoring system with weighted signals:
        - MSS (Market Structure Shift): ±0.30 (30% weight) - LEADING indicator
        - OI + Delta Divergence: ±0.25 (25% weight) - accumulation/distribution
        - POC Positioning: ±0.20 (20% weight) - volume profile support/resistance
        - Phase/Trend: ±0.15 (15% weight) - background context
        - Exhaustion/Reversal: ±0.10 (10% weight) - extreme conditions
        
        Priority:
        1. REVERSAL SETUP → trade reversal direction
        2. MSS SIGNALS → strong leading indicator in consolidation
        3. OI + DELTA DIVERGENCE → accumulation/distribution
        4. POC POSITIONING → volume profile bias
        5. STRONG TREND → trade with trend (if not exhausted)
        6. UNCLEAR → NEUTRAL
        """
        
        score = 0.0
        reasons = []
        
        # === PRIORITY 1: REVERSAL SETUP (extreme conditions) ===
        if is_reversal:
            if reversal_direction == Direction.LONG:
                score = 0.6 + exhaustion.exhaustion_score * 0.3
                reason = f"REVERSAL: Downtrend exhausted at support, bullish setup"
                return reversal_direction, score, reason
            elif reversal_direction == Direction.SHORT:
                score = -0.6 - exhaustion.exhaustion_score * 0.3
                reason = f"REVERSAL: Uptrend exhausted at resistance, bearish setup"
                return reversal_direction, score, reason
        
        # === v6 SCORING SYSTEM ===
        
        # --- MSS (Market Structure Shift) - 30% weight ---
        # Best indicator in low volatility / consolidation
        if structure.mss_bias == "long":
            mss_score = 0.30 * structure.mss_strength
            score += mss_score
            reasons.append(f"MSS: {structure.higher_lows_count} Higher Lows (+{mss_score:.2f})")
        elif structure.mss_bias == "short":
            mss_score = -0.30 * structure.mss_strength
            score += mss_score
            reasons.append(f"MSS: {structure.lower_highs_count} Lower Highs ({mss_score:.2f})")
        
        # --- OI + Volume Delta Divergence - 25% weight ---
        # Shows accumulation (OI↑ + buying) or distribution (OI↑ + selling)
        if structure.oi_delta_bias == "long":
            oi_score = 0.25
            score += oi_score
            reasons.append(f"OI Divergence: Accumulation (OI↑ + Buy Delta)")
        elif structure.oi_delta_bias == "short":
            oi_score = -0.25
            score += oi_score
            reasons.append(f"OI Divergence: Distribution (OI↑ + Sell Delta)")
        
        # --- POC Positioning - 20% weight ---
        # Price above POC = support, below = resistance
        if structure.poc_bias == "long":
            poc_score = 0.20
            score += poc_score
            reasons.append(f"POC: Price {structure.price_vs_poc:+.1f}% above (support)")
        elif structure.poc_bias == "short":
            poc_score = -0.20
            score += poc_score
            reasons.append(f"POC: Price {structure.price_vs_poc:.1f}% below (resistance)")
        
        # --- Phase/Trend Context - 15% weight ---
        if maturity != PhaseMaturity.EXHAUSTED:
            if phase == MarketPhase.MARKUP and structure.trend_strength in [TrendStrength.STRONG, TrendStrength.MODERATE]:
                score += 0.15
                reasons.append(f"Trend: Uptrend (ADX {structure.adx_value:.0f})")
            elif phase == MarketPhase.MARKDOWN and structure.trend_strength in [TrendStrength.STRONG, TrendStrength.MODERATE]:
                score -= 0.15
                reasons.append(f"Trend: Downtrend (ADX {structure.adx_value:.0f})")
            elif phase == MarketPhase.ACCUMULATION:
                score += 0.10
                reasons.append("Phase: Accumulation")
            elif phase == MarketPhase.DISTRIBUTION:
                score -= 0.10
                reasons.append("Phase: Distribution")
        
        # --- Exhaustion Adjustment - 10% weight ---
        if maturity == PhaseMaturity.EXHAUSTED:
            if phase == MarketPhase.MARKDOWN:
                # Exhausted downtrend - don't short, lean long
                score += 0.10
                reasons.append("Exhaustion: Downtrend exhausted")
            elif phase == MarketPhase.MARKUP:
                # Exhausted uptrend - don't long, lean short
                score -= 0.10
                reasons.append("Exhaustion: Uptrend exhausted")
        
        # === DETERMINE FINAL DIRECTION ===
        # Threshold: need score >= 0.35 for direction, else NEUTRAL
        if score >= 0.35:
            direction = Direction.LONG
            primary_reason = " | ".join(reasons[:2]) if reasons else "Multiple bullish signals"
        elif score <= -0.35:
            direction = Direction.SHORT
            primary_reason = " | ".join(reasons[:2]) if reasons else "Multiple bearish signals"
        else:
            direction = Direction.NEUTRAL
            primary_reason = "No clear directional bias" if not reasons else f"Mixed signals: {reasons[0] if reasons else ''}"
        
        return direction, round(score, 3), primary_reason
    
    def _get_derivatives_bias(self, oi_change: float = None, 
                              funding_rate: float = None) -> float:
        """Get directional bias from derivatives data"""
        
        if oi_change is None and funding_rate is None:
            return 0.0
        
        bias = 0.0
        
        # OI change logic
        if oi_change:
            if oi_change > 10:
                bias += 0.2  # Big OI increase = activity
            elif oi_change < -10:
                bias -= 0.1  # OI decrease = positions closing
        
        # Funding rate extremes
        if funding_rate:
            if funding_rate > 0.0005:  # High positive funding
                bias -= 0.2  # Crowded longs, expect pullback
            elif funding_rate < -0.0003:  # Negative funding
                bias += 0.2  # Crowded shorts, expect squeeze
        
        return bias
    
    def _get_secondary_reasons(self, structure: StructureAnalysis,
                               exhaustion: ExhaustionSignals) -> List[str]:
        """Get secondary reasoning details"""
        
        reasons = []
        
        # Structure info
        if structure.price_change_5d != 0:
            reasons.append(f"5D change: {structure.price_change_5d:+.1f}%")
        if structure.price_change_20d != 0:
            reasons.append(f"20D change: {structure.price_change_20d:+.1f}%")
        
        # ADX
        reasons.append(f"ADX: {structure.adx_value:.0f} ({structure.trend_strength.value})")
        
        # Exhaustion signals
        if exhaustion.rsi_divergence != "none":
            reasons.append(f"RSI divergence: {exhaustion.rsi_divergence}")
        if exhaustion.at_support:
            reasons.append("At support level")
        if exhaustion.at_resistance:
            reasons.append("At resistance level")
        if exhaustion.volume_exhaustion > 0.3:
            reasons.append(f"Volume declining ({exhaustion.volume_exhaustion:.0%})")
        
        return reasons
    
    def _neutral_result(self, reason: str) -> DirectionResultV5:
        """Return neutral result with reason"""
        return DirectionResultV5(
            direction=Direction.NEUTRAL,
            score=0.0,
            confidence="LOW",
            market_phase=MarketPhase.UNKNOWN,
            phase_maturity=PhaseMaturity.MIDDLE,
            exhaustion=ExhaustionSignals(),
            is_reversal_setup=False,
            structure=StructureAnalysis(),
            primary_reason=reason,
            trading_style=self.trading_style,
            recommended_timeframe=self.mtf
        )


# ============================================
# SINGLETON & HELPERS
# ============================================

_engine_instances: Dict[str, DirectionEngineV5] = {}

def get_direction_engine_v5(trading_style: str = "SWING") -> DirectionEngineV5:
    """Get direction engine instance for trading style"""
    global _engine_instances
    if trading_style not in _engine_instances:
        _engine_instances[trading_style] = DirectionEngineV5(trading_style)
    return _engine_instances[trading_style]


def resolve_direction_v5(symbol: str, trading_style: str = "SWING", **kwargs) -> DirectionResultV5:
    """
    Convenience function to resolve direction with v5 engine.
    
    Usage:
        result = resolve_direction_v5('BTCUSDT', trading_style='SWING')
        print(f"Direction: {result.direction.value}")
        print(f"Phase: {result.market_phase.value} ({result.phase_maturity.value})")
        print(f"Reversal: {result.is_reversal_setup}")
        print(f"Reason: {result.primary_reason}")
    """
    engine = get_direction_engine_v5(trading_style)
    return engine.resolve(symbol, **kwargs)


# ============================================
# BACKWARD COMPATIBILITY
# ============================================

def convert_to_v1_format(result: DirectionResultV5) -> dict:
    """Convert v5 result to v1 format for backward compatibility"""
    from detection.direction_engine import DirectionResult, Direction as DirV1
    
    dir_map = {
        Direction.LONG: DirV1.LONG,
        Direction.SHORT: DirV1.SHORT,
        Direction.NEUTRAL: DirV1.NEUTRAL
    }
    
    return DirectionResult(
        direction=dir_map[result.direction],
        score=result.score,
        confidence=result.confidence,
        htf_bias=result.score * 0.5,
        ltf_bias=result.score * 0.3,
        deriv_bias=result.score * 0.2,
        htf_reason=result.primary_reason,
        ltf_reason=f"Phase: {result.market_phase.value}",
        deriv_reason=f"Exhaustion: {result.exhaustion.exhaustion_score:.0%}"
    )
