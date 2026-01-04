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
    - Fuel Score (30%): Funding rate + OI change
    - Volatility Score (25%): BB width compression
    - Price Score (25%): Range tightness + position
    - Liquidity Score (20%): Volume profile
    """
    
    def __init__(self):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.db = get_db()
        self.thresholds = SLEEPER_THRESHOLDS
        
    def scan(self, max_symbols: int = 100, min_volume: float = 20000000) -> List[Dict]:
        """
        Run full sleeper scan on top symbols
        Returns list of detected sleeper candidates
        """
        self.db.log_event(
            f"Starting Sleeper scan: {max_symbols} symbols, min vol ${min_volume/1e6:.0f}M",
            level='INFO', category='SLEEPER'
        )
        
        # Get top symbols by volume
        symbols = self.fetcher.get_top_symbols(limit=max_symbols, min_volume=min_volume)
        
        candidates = []
        processed = 0
        
        for symbol_data in symbols:
            try:
                result = self._analyze_symbol(symbol_data)
                if result:
                    candidates.append(result)
                    
                    # Save to database
                    self.db.upsert_sleeper(result)
                    
                processed += 1
                
                # Rate limiting
                if processed % API_LIMITS['symbols_per_batch'] == 0:
                    time.sleep(1)
                else:
                    time.sleep(API_LIMITS['rate_limit_delay'])
                    
            except Exception as e:
                print(f"Error analyzing {symbol_data.get('symbol')}: {e}")
                continue
        
        # Sort by score
        candidates.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Update states based on scores
        self._update_states(candidates)
        
        # Remove dead sleepers (HP = 0)
        removed = self.db.remove_dead_sleepers()
        
        self.db.log_event(
            f"Sleeper scan complete: {len(candidates)} candidates found, {removed} removed",
            level='SUCCESS', category='SLEEPER'
        )
        
        return candidates
    
    def _analyze_symbol(self, symbol_data: Dict) -> Optional[Dict]:
        """Analyze a single symbol for sleeper characteristics"""
        symbol = symbol_data['symbol']
        
        # Get 4H klines for analysis
        klines = self.fetcher.get_klines(symbol, '4h', limit=100)
        if len(klines) < 20:
            return None
        
        # Calculate indicators
        indicators = self.indicators.calculate_all(klines)
        
        # Get additional metrics
        funding_rate = self.fetcher.get_funding_rate(symbol) or 0
        oi_change = self.fetcher.get_oi_change(symbol, hours=4) or 0
        
        # Calculate component scores
        fuel_score = self._calculate_fuel_score(funding_rate, oi_change)
        volatility_score = self._calculate_volatility_score(indicators)
        price_score = self._calculate_price_score(indicators)
        liquidity_score = self._calculate_liquidity_score(
            indicators['volume_profile'], 
            symbol_data['volume_24h']
        )
        
        # Total weighted score
        total_score = (
            fuel_score * self.thresholds['weight_fuel'] +
            volatility_score * self.thresholds['weight_volatility'] +
            price_score * self.thresholds['weight_price'] +
            liquidity_score * self.thresholds['weight_liquidity']
        )
        
        # Determine direction bias
        direction = self._determine_direction(funding_rate, oi_change, indicators)
        
        min_score = self.db.get_setting('sleeper_min_score', 60)
        if total_score < min_score:
            return None
        
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
            'funding_rate': funding_rate,
            'oi_change_4h': oi_change,
            'bb_width': indicators['bb_width_current'],
            'volume_24h': symbol_data['volume_24h'],
            'volume_ratio': indicators['volume_profile']['ratio'],
            'price_range_pct': indicators['price_range']['range_pct'],
            'rsi': indicators['rsi_current'],
        }
    
    def _calculate_fuel_score(self, funding_rate: float, oi_change: float) -> float:
        """
        Calculate Fuel Score based on funding rate and OI change
        High funding + increasing OI = potential reversal setup
        """
        score = 0
        
        # Funding rate component (extreme funding = fuel for reversal)
        abs_funding = abs(funding_rate)
        if abs_funding >= self.thresholds['funding_rate_extreme']:
            score += 50
        elif abs_funding >= self.thresholds['funding_rate_moderate']:
            score += 30
        else:
            score += 10
        
        # OI change component (accumulation signal)
        if oi_change >= self.thresholds['oi_change_high']:
            score += 50
        elif oi_change >= self.thresholds['oi_change_moderate']:
            score += 30
        elif oi_change > 0:
            score += 15
        else:
            score += 5
        
        return score
    
    def _calculate_volatility_score(self, indicators: Dict) -> float:
        """
        Calculate Volatility Score based on BB squeeze
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
        
        # Divergence bonus
        if indicators.get('divergence'):
            score += 20
        
        return min(100, score)
    
    def _calculate_price_score(self, indicators: Dict) -> float:
        """
        Calculate Price Score based on range and position
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
        
        return score
    
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
                            indicators: Dict) -> str:
        """Determine probable direction based on metrics"""
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
        
        if bullish_signals > bearish_signals + 1:
            return 'LONG'
        elif bearish_signals > bullish_signals + 1:
            return 'SHORT'
        return 'NEUTRAL'
    
    def _update_states(self, candidates: List[Dict]):
        """Update sleeper states based on scores"""
        ready_score = self.db.get_setting('sleeper_ready_score', 80)
        
        for c in candidates:
            symbol = c['symbol']
            score = c['total_score']
            
            if score >= ready_score:
                new_state = SleeperState.READY.value
                self.db.log_event(
                    f"{symbol} READY (Score: {score})",
                    level='SUCCESS', category='SLEEPER', symbol=symbol
                )
            elif score >= 70:
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
