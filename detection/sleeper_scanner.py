"""
Sleeper Scanner - Detects coins building energy for potential moves
Sleeper Detector v2.0 with HP system and multi-factor scoring
"""
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from config import SLEEPER_THRESHOLDS, API_LIMITS, SleeperState
from core import get_fetcher, get_indicators
from storage import get_db

class SleeperScanner:
    """
    Sleeper Detector v2.0
    Detects coins in accumulation/distribution phase before explosive moves
    
    Score Components:
    - Fuel Score: Funding rate + OI change
    - Volatility Score: BB width compression
    - Price Score: Range tightness + position
    - Liquidity Score: Volume profile
    """
    
    def __init__(self):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.db = get_db()
        self.thresholds = SLEEPER_THRESHOLDS.copy()
        
    def _load_settings(self):
        """Load settings from DB, fallback to defaults"""
        # Score weights (should sum to 1.0)
        self.thresholds['weight_fuel'] = self.db.get_setting('weight_fuel', 30) / 100
        self.thresholds['weight_volatility'] = self.db.get_setting('weight_volatility', 25) / 100
        self.thresholds['weight_price'] = self.db.get_setting('weight_price', 25) / 100
        self.thresholds['weight_liquidity'] = self.db.get_setting('weight_liquidity', 20) / 100
        
        # Score thresholds
        self.min_score = self.db.get_setting('sleeper_min_score', 40)
        self.building_score = self.db.get_setting('sleeper_building_score', 55)
        self.ready_score = self.db.get_setting('sleeper_ready_score', 70)
        
        # Scan parameters
        self.scan_limit = int(self.db.get_setting('sleeper_scan_limit', 100))
        self.min_volume = float(self.db.get_setting('sleeper_min_volume', 20000000))
        
    def scan(self, max_symbols: int = None, min_volume: float = None) -> List[Dict]:
        """
        Run full sleeper scan on top symbols
        Returns list of detected sleeper candidates
        """
        # Load settings from DB
        self._load_settings()
        
        # Use DB settings if not overridden
        max_symbols = max_symbols or self.scan_limit
        min_volume = min_volume or self.min_volume
        
        print(f"\n{'='*50}")
        print(f"[SLEEPER SCAN] Starting scan: {max_symbols} symbols, min vol ${min_volume/1e6:.0f}M")
        print(f"[SLEEPER SCAN] Thresholds: min={self.min_score}, building={self.building_score}, ready={self.ready_score}")
        print(f"[SLEEPER SCAN] Weights: F={self.thresholds['weight_fuel']:.0%} V={self.thresholds['weight_volatility']:.0%} P={self.thresholds['weight_price']:.0%} L={self.thresholds['weight_liquidity']:.0%}")
        print(f"{'='*50}")
        
        self.db.log_event(
            f"Starting Sleeper scan: {max_symbols} symbols, min vol ${min_volume/1e6:.0f}M",
            level='INFO', category='SLEEPER'
        )
        
        # Get top symbols by volume
        symbols = self.fetcher.get_top_symbols(limit=max_symbols, min_volume=min_volume)
        print(f"[SLEEPER SCAN] Found {len(symbols)} symbols to analyze")
        
        candidates = []
        processed = 0
        passed = 0
        
        for symbol_data in symbols:
            try:
                symbol = symbol_data.get('symbol', 'UNKNOWN')
                result = self._analyze_symbol(symbol_data)
                
                processed += 1
                
                if result:
                    candidates.append(result)
                    passed += 1
                    
                    # Save to database
                    self.db.upsert_sleeper(result)
                    
                    print(f"[SLEEPER] ✓ {symbol}: Score={result['total_score']:.1f} Dir={result['direction']}")
                
                # Progress every 10 symbols
                if processed % 10 == 0:
                    print(f"[SLEEPER SCAN] Progress: {processed}/{len(symbols)} ({passed} candidates)")
                
                # Rate limiting
                if processed % API_LIMITS['symbols_per_batch'] == 0:
                    time.sleep(1)
                else:
                    time.sleep(API_LIMITS['rate_limit_delay'])
                    
            except Exception as e:
                print(f"[SLEEPER] ✗ {symbol_data.get('symbol')}: {e}")
                continue
        
        # Sort by score
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Update states based on scores
        self._update_states(candidates)
        
        # Remove dead sleepers (HP = 0)
        removed = self.db.remove_dead_sleepers()
        
        print(f"\n{'='*50}")
        print(f"[SLEEPER SCAN] Complete: {len(candidates)} candidates from {processed} symbols")
        print(f"[SLEEPER SCAN] Removed {removed} dead sleepers")
        print(f"{'='*50}\n")
        
        self.db.log_event(
            f"Sleeper scan complete: {len(candidates)} candidates found, {removed} removed",
            level='SUCCESS', category='SLEEPER'
        )
        
        return candidates
    
    def _analyze_symbol(self, symbol_data: Dict) -> Optional[Dict]:
        """Analyze a single symbol for sleeper characteristics with full data"""
        symbol = symbol_data['symbol']
        
        from config import DATA_REQUIREMENTS
        
        # === ЗАВАНТАЖЕННЯ ПОВНИХ ДАНИХ ===
        
        # 1. Multi-timeframe klines
        klines_4h = self.fetcher.get_klines(symbol, '240', limit=DATA_REQUIREMENTS['sleeper_klines_4h'])  # 500 x 4h = 83 days
        klines_1h = self.fetcher.get_klines(symbol, '60', limit=DATA_REQUIREMENTS['sleeper_klines_1h'])   # 200 x 1h = 8 days
        klines_1d = self.fetcher.get_klines(symbol, 'D', limit=DATA_REQUIREMENTS['sleeper_klines_1d'])    # 100 days
        
        if len(klines_4h) < 50:  # Minimum required
            return None
        
        # 2. Calculate indicators on multiple timeframes
        indicators_4h = self.indicators.calculate_all(klines_4h)
        indicators_1h = self.indicators.calculate_all(klines_1h) if len(klines_1h) > 20 else None
        indicators_1d = self.indicators.calculate_all(klines_1d) if len(klines_1d) > 20 else None
        
        # 3. Get funding rate
        funding_rate = self.fetcher.get_funding_rate(symbol) or 0
        
        # 4. Get Open Interest change
        oi_change = self.fetcher.get_oi_change(symbol, hours=4) or 0
        
        # 5. Get OI history for accumulation analysis
        oi_history = self.fetcher.get_oi_history(symbol, limit=API_LIMITS.get('oi_history_limit', 200))
        oi_accumulation = self._analyze_oi_accumulation(oi_history) if oi_history else 0
        
        # === РОЗРАХУНОК SCORES ===
        
        # Calculate component scores with enhanced data
        fuel_score = self._calculate_fuel_score(funding_rate, oi_change, oi_accumulation)
        volatility_score = self._calculate_volatility_score(indicators_4h, indicators_1d)
        price_score = self._calculate_price_score(indicators_4h, indicators_1d)
        liquidity_score = self._calculate_liquidity_score(
            indicators_4h['volume_profile'], 
            symbol_data['volume_24h']
        )
        
        # MTF confirmation bonus
        mtf_bonus = self._calculate_mtf_bonus(indicators_4h, indicators_1h, indicators_1d)
        
        # Total weighted score
        total_score = (
            fuel_score * self.thresholds['weight_fuel'] +
            volatility_score * self.thresholds['weight_volatility'] +
            price_score * self.thresholds['weight_price'] +
            liquidity_score * self.thresholds['weight_liquidity']
        ) + mtf_bonus
        
        # Determine direction bias
        direction = self._determine_direction(funding_rate, oi_change, indicators_4h, indicators_1d)
        
        if total_score < self.min_score:
            # Debug: show why rejected (only close misses)
            if total_score > self.min_score - 10:
                print(f"[SLEEPER] ⚠ {symbol}: Score {total_score:.1f} < {self.min_score} (F:{fuel_score:.0f} V:{volatility_score:.0f} P:{price_score:.0f} L:{liquidity_score:.0f})")
            return None
        
        # BB Width trend (is it compressing?)
        bb_width_change = self._calculate_bb_compression(klines_4h)
        
        # Helper to convert numpy types to Python native
        def to_float(val):
            try:
                return float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                return 0.0
        
        return {
            'symbol': symbol,
            'total_score': round(total_score, 2),
            'fuel_score': round(fuel_score, 2),
            'volatility_score': round(volatility_score, 2),
            'price_score': round(price_score, 2),
            'liquidity_score': round(liquidity_score, 2),
            'state': SleeperState.WATCHING.value,
            'hp': self.thresholds['hp_initial'],
            'direction': direction,
            'funding_rate': to_float(funding_rate),
            'oi_change_4h': to_float(oi_change),
            'bb_width': to_float(indicators_4h.get('bb_width_current')),
            'bb_width_change': to_float(bb_width_change),
            'volume_24h': to_float(symbol_data.get('volume_24h')),
            'volume_ratio': to_float(indicators_4h.get('volume_profile', {}).get('ratio')),
            'price_range_pct': to_float(indicators_4h.get('price_range', {}).get('range_pct')),
            'rsi': to_float(indicators_4h.get('rsi_current')),
        }
    
    def _analyze_oi_accumulation(self, oi_history: List[Dict]) -> float:
        """Analyze OI accumulation pattern (0-100)"""
        if not oi_history or len(oi_history) < 10:
            return 0
        
        # Get OI values
        oi_values = [h['open_interest'] for h in oi_history]
        
        # Calculate trend (linear regression slope)
        n = len(oi_values)
        x_mean = n / 2
        y_mean = sum(oi_values) / n
        
        numerator = sum((i - x_mean) * (oi_values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        slope = numerator / denominator
        
        # Normalize slope to 0-100 score
        # Positive slope = accumulation, negative = distribution
        oi_change_pct = (slope * n) / y_mean * 100 if y_mean > 0 else 0
        
        if oi_change_pct > 20:
            return 100
        elif oi_change_pct > 10:
            return 80
        elif oi_change_pct > 5:
            return 60
        elif oi_change_pct > 0:
            return 40
        else:
            return 20  # Distribution
    
    def _calculate_bb_compression(self, klines: List[Dict]) -> float:
        """Calculate BB width compression rate"""
        if len(klines) < 50:
            return 0
        
        # Calculate BB width for recent vs older periods
        recent_closes = [k['close'] for k in klines[-20:]]
        older_closes = [k['close'] for k in klines[-50:-30]]
        
        def bb_width(closes):
            if len(closes) < 5:
                return 0
            sma = sum(closes) / len(closes)
            variance = sum((c - sma) ** 2 for c in closes) / len(closes)
            std = variance ** 0.5
            return (std * 4) / sma * 100 if sma > 0 else 0
        
        recent_width = bb_width(recent_closes)
        older_width = bb_width(older_closes)
        
        if older_width > 0:
            compression = (older_width - recent_width) / older_width * 100
            return round(compression, 2)
        return 0
    
    def _calculate_mtf_bonus(self, ind_4h: Dict, ind_1h: Dict, ind_1d: Dict) -> float:
        """Calculate multi-timeframe alignment bonus"""
        bonus = 0
        
        # Check RSI alignment
        rsi_4h = ind_4h.get('rsi_current', 50)
        rsi_1h = ind_1h.get('rsi_current', 50) if ind_1h else 50
        rsi_1d = ind_1d.get('rsi_current', 50) if ind_1d else 50
        
        # All in oversold zone (potential long)
        if rsi_4h < 35 and rsi_1h < 40 and rsi_1d < 45:
            bonus += 5
        # All in overbought zone (potential short)
        elif rsi_4h > 65 and rsi_1h > 60 and rsi_1d > 55:
            bonus += 5
        
        # Check BB squeeze alignment
        bb_4h = ind_4h.get('bb_width_current', 5)
        bb_1d = ind_1d.get('bb_width_current', 5) if ind_1d else 5
        
        if bb_4h < 3 and bb_1d < 4:  # Compressed on both
            bonus += 5
        
        return bonus
    
    def _calculate_fuel_score(self, funding_rate: float, oi_change: float, oi_accumulation: float = 0) -> float:
        """
        Calculate Fuel Score based on funding rate, OI change, and OI accumulation
        High funding + increasing OI + accumulation = potential reversal setup
        """
        score = 0
        
        # Funding rate component (extreme funding = fuel for reversal)
        abs_funding = abs(funding_rate)
        if abs_funding >= self.thresholds['funding_rate_extreme']:
            score += 40
        elif abs_funding >= self.thresholds['funding_rate_moderate']:
            score += 25
        else:
            score += 10
        
        # OI change component (short-term accumulation signal)
        if oi_change >= self.thresholds['oi_change_high']:
            score += 30
        elif oi_change >= self.thresholds['oi_change_moderate']:
            score += 20
        elif oi_change > 0:
            score += 10
        else:
            score += 5
        
        # OI accumulation pattern (long-term signal) - NEW
        score += oi_accumulation * 0.3  # Max 30 points from OI accumulation
        
        return min(100, score)
    
    def _calculate_volatility_score(self, indicators: Dict, indicators_daily: Dict = None) -> float:
        """
        Calculate Volatility Score based on BB squeeze (MTF)
        Tight BBs = energy building
        """
        score = 0
        bb_width = indicators['bb_width_current']
        
        # BB squeeze detection
        if bb_width < 2:  # Very tight
            score = 100
        elif bb_width < 3:
            score = 80
        elif bb_width < 4:
            score = 60
        elif bb_width < 5:
            score = 40
        else:
            score = 20
        
        # Daily BB confirmation (MTF bonus)
        if indicators_daily:
            bb_daily = indicators_daily.get('bb_width_current', 5)
            if bb_daily < 4:  # Daily also squeezed
                score += 15
        
        # Divergence bonus
        if indicators.get('divergence'):
            score += 20
        
        return min(100, score)
    
    def _calculate_price_score(self, indicators: Dict, indicators_daily: Dict = None) -> float:
        """
        Calculate Price Score based on range and position (MTF)
        Tight range + neutral position = good setup
        """
        score = 0
        price_range = indicators['price_range']
        
        # Range tightness
        range_pct = price_range['range_pct']
        if range_pct < 2:
            score += 50
        elif range_pct < 3:
            score += 40
        elif range_pct < 4:
            score += 30
        elif range_pct < 5:
            score += 20
        else:
            score += 10
        
        # Position within range (middle is best for breakout potential)
        position = price_range['position']
        if 0.35 <= position <= 0.65:  # Middle
            score += 50
        elif 0.2 <= position <= 0.8:  # Near middle
            score += 35
        else:  # Extremes
            score += 20
        
        # Daily trend alignment (MTF bonus)
        if indicators_daily:
            daily_rsi = indicators_daily.get('rsi_current', 50)
            # RSI not overbought/oversold on daily = room to move
            if 30 < daily_rsi < 70:
                score += 10
        
        return min(100, score)
    
    def _calculate_liquidity_score(self, volume_profile: Dict, 
                                   volume_24h: float) -> float:
        """
        Calculate Liquidity Score
        Quiet accumulation is bullish, declining volume during consolidation
        """
        score = 0
        
        # Volume trend (declining = good for sleeper)
        if volume_profile['trend'] == 'decreasing':
            score += 40
        elif volume_profile['trend'] == 'neutral':
            score += 30
        else:
            score += 15
        
        # Volume ratio (low ratio = quiet)
        ratio = volume_profile['ratio']
        if ratio < 0.7:
            score += 60
        elif ratio < 1.0:
            score += 45
        elif ratio < 1.3:
            score += 30
        else:
            score += 15
        
        return score
    
    def _determine_direction(self, funding_rate: float, oi_change: float,
                            indicators: Dict, indicators_daily: Dict = None) -> str:
        """Determine probable direction based on metrics (MTF)"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Funding rate (negative = bullish bias for reversal)
        if funding_rate < -self.thresholds['funding_rate_moderate']:
            bullish_signals += 2
        elif funding_rate > self.thresholds['funding_rate_moderate']:
            bearish_signals += 2
        
        # RSI
        rsi = indicators['rsi_current']
        if rsi < 35:
            bullish_signals += 2
        elif rsi > 65:
            bearish_signals += 2
        elif rsi < 45:
            bullish_signals += 1
        elif rsi > 55:
            bearish_signals += 1
        
        # Price position in range
        position = indicators['price_range']['position']
        if position < 0.3:
            bullish_signals += 1
        elif position > 0.7:
            bearish_signals += 1
        
        # Divergence
        div = indicators.get('divergence')
        if div == 'bullish':
            bullish_signals += 2
        elif div == 'bearish':
            bearish_signals += 2
        
        # Daily RSI confirmation (MTF)
        if indicators_daily:
            daily_rsi = indicators_daily.get('rsi_current', 50)
            if daily_rsi < 40:
                bullish_signals += 1
            elif daily_rsi > 60:
                bearish_signals += 1
        
        if bullish_signals > bearish_signals + 1:
            return 'LONG'
        elif bearish_signals > bullish_signals + 1:
            return 'SHORT'
        return 'NEUTRAL'
    
    def _update_states(self, candidates: List[Dict]):
        """Update sleeper states based on scores"""
        for c in candidates:
            symbol = c['symbol']
            score = c['total_score']
            
            if score >= self.ready_score:
                new_state = SleeperState.READY.value
                self.db.log_event(
                    f"{symbol} READY (Score: {score:.1f})",
                    level='SUCCESS', category='SLEEPER', symbol=symbol
                )
            elif score >= self.building_score:
                new_state = SleeperState.BUILDING.value
            else:
                new_state = SleeperState.WATCHING.value
            
            self.db.update_sleeper_state(
                symbol=symbol,
                state=new_state,
                direction=c['direction']
            )
    
    def check_single(self, symbol: str) -> Optional[Dict]:
        """Quick check for a single symbol"""
        try:
            symbol_data = {
                'symbol': symbol,
                'volume_24h': 0
            }
            
            # Get ticker info
            for ticker in self.fetcher._get_tickers():
                if ticker.get('symbol') == symbol:
                    symbol_data['volume_24h'] = float(ticker.get('turnover24h', 0))
                    break
            
            return self._analyze_symbol(symbol_data)
        except Exception as e:
            print(f"Error checking {symbol}: {e}")
            return None
    
    def update_hp(self, symbol: str, score_change: float):
        """Update HP based on score changes"""
        if score_change > 5:
            hp_change = 1
        elif score_change < -5:
            hp_change = -1
        else:
            hp_change = 0
        
        if hp_change != 0:
            self.db.update_sleeper_state(symbol, state=None, hp_change=hp_change)


# Singleton instance
_scanner_instance = None

def get_sleeper_scanner() -> SleeperScanner:
    """Get sleeper scanner instance"""
    global _scanner_instance
    if _scanner_instance is None:
        _scanner_instance = SleeperScanner()
    return _scanner_instance
