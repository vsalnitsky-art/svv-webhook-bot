"""
Order Block Scanner - Detects institutional order blocks
Multi-timeframe OB detection with quality scoring
"""
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from config import OB_THRESHOLDS, API_LIMITS
from core import get_fetcher, get_indicators
from storage import get_db

class OBScanner:
    """
    Order Block Detector
    Detects institutional order blocks using multi-timeframe analysis
    
    Detection Logic:
    1. Find impulse candle (large move with volume)
    2. Mark the origin zone (last opposite candle before impulse)
    3. Calculate quality score
    4. Track price return to zone
    """
    
    def __init__(self):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.db = get_db()
        self.thresholds = OB_THRESHOLDS.copy()
    
    def _load_settings(self):
        """Load settings from DB"""
        self.min_quality = self.db.get_setting('ob_min_quality', 60)
        self.signal_quality = self.db.get_setting('ob_signal_quality', 70)
        self.volume_ratio = self.db.get_setting('ob_volume_ratio', 1.5)
        self.max_age_hours = self.db.get_setting('ob_max_age_hours', 48)
        
        # Parse timeframes
        tf_setting = self.db.get_setting('ob_timeframes', '15,5')
        if isinstance(tf_setting, str):
            self.timeframes = [f"{t.strip()}m" if t.strip().isdigit() else t.strip() for t in tf_setting.split(',')]
        else:
            self.timeframes = tf_setting
    
    def scan_symbol(self, symbol: str, timeframes: List[str] = None) -> List[Dict]:
        """
        Scan a symbol for order blocks on multiple timeframes
        """
        self._load_settings()
        timeframes = timeframes or self.timeframes
        all_obs = []
        
        for tf in timeframes:
            try:
                obs = self._detect_ob_on_timeframe(symbol, tf)
                all_obs.extend(obs)
                time.sleep(API_LIMITS['rate_limit_delay'])
            except Exception as e:
                print(f"Error scanning {symbol} {tf}: {e}")
                continue
        
        # Cross-timeframe confirmation
        all_obs = self._confirm_mtf(all_obs)
        
        # Save to database
        for ob in all_obs:
            self.db.add_orderblock(ob)
        
        return all_obs
    
    def _detect_ob_on_timeframe(self, symbol: str, timeframe: str) -> List[Dict]:
        """Detect order blocks on a single timeframe with full data"""
        from config import DATA_REQUIREMENTS
        
        # Get appropriate limit based on timeframe
        tf_limits = {
            '15': DATA_REQUIREMENTS['ob_klines_15m'],  # 500
            '5': DATA_REQUIREMENTS['ob_klines_5m'],    # 300
            '1': DATA_REQUIREMENTS['ob_klines_1m'],    # 200
        }
        limit = tf_limits.get(timeframe, 200)
        
        klines = self.fetcher.get_klines(symbol, timeframe, limit=limit)
        if len(klines) < 50:  # Need minimum data
            return []
        
        detected = []
        
        # Calculate volume average over larger sample
        volumes = [k['volume'] for k in klines]
        avg_volume = sum(volumes[:-10]) / len(volumes[:-10]) if len(volumes) > 10 else 1
        
        # Scan for OBs (skip last few candles)
        for i in range(20, len(klines) - 2):
            ob = self._check_ob_at_index(klines, i, avg_volume, symbol, timeframe)
            if ob:
                detected.append(ob)
        
        return detected
    
    def _check_ob_at_index(self, klines: List[Dict], idx: int, avg_volume: float,
                           symbol: str, timeframe: str) -> Optional[Dict]:
        """Check if there's an OB at given index"""
        current = klines[idx]
        next_candle = klines[idx + 1]
        
        # Impulse detection
        current_body = abs(current['close'] - current['open'])
        current_range = current['high'] - current['low']
        next_body = abs(next_candle['close'] - next_candle['open'])
        
        # Skip if current candle is too small
        if current_range == 0:
            return None
        
        body_ratio = current_body / current_range
        
        # Check for impulse candle (large body, high volume)
        is_impulse = (
            next_body > current_body * 1.5 and
            next_candle['volume'] > avg_volume * self.thresholds['ob_volume_ratio_min']
        )
        
        if not is_impulse:
            return None
        
        # Determine OB type
        if next_candle['close'] > next_candle['open']:  # Bullish impulse
            # Look for last bearish candle before
            ob_candle = self._find_opposite_candle(klines, idx, 'bearish')
            if not ob_candle:
                return None
            ob_type = 'BULLISH'
        else:  # Bearish impulse
            # Look for last bullish candle before
            ob_candle = self._find_opposite_candle(klines, idx, 'bullish')
            if not ob_candle:
                return None
            ob_type = 'BEARISH'
        
        # Calculate impulse percentage
        price_move = abs(next_candle['close'] - current['close'])
        impulse_pct = (price_move / current['close']) * 100
        
        if impulse_pct < self.thresholds['ob_impulse_min'] * 100:
            return None
        
        # Volume ratio
        volume_ratio = next_candle['volume'] / avg_volume if avg_volume > 0 else 1
        
        # Quality score
        quality = self._calculate_ob_quality(
            volume_ratio=volume_ratio,
            impulse_pct=impulse_pct,
            body_ratio=body_ratio,
            fresh=True
        )
        
        min_quality = self.db.get_setting('ob_min_quality', 65)
        if quality < min_quality:
            return None
        
        # OB zone
        ob_high = ob_candle['high']
        ob_low = ob_candle['low']
        ob_mid = (ob_high + ob_low) / 2
        
        # Expiry time
        max_age = self.db.get_setting('ob_max_age_minutes', 60)
        expires_at = datetime.utcnow() + timedelta(minutes=max_age)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'ob_type': ob_type,
            'ob_high': ob_high,
            'ob_low': ob_low,
            'ob_mid': ob_mid,
            'quality_score': round(quality, 2),
            'volume_ratio': round(volume_ratio, 2),
            'impulse_pct': round(impulse_pct, 4),
            'status': 'ACTIVE',
            'expires_at': expires_at,
        }
    
    def _find_opposite_candle(self, klines: List[Dict], idx: int, 
                              candle_type: str) -> Optional[Dict]:
        """Find the last opposite candle before index"""
        for i in range(idx, max(idx - 5, 0), -1):
            candle = klines[i]
            is_bullish = candle['close'] > candle['open']
            
            if candle_type == 'bullish' and is_bullish:
                return candle
            elif candle_type == 'bearish' and not is_bullish:
                return candle
        
        return None
    
    def _calculate_ob_quality(self, volume_ratio: float, impulse_pct: float,
                              body_ratio: float, fresh: bool) -> float:
        """Calculate OB quality score (0-100)"""
        score = 0
        
        # Volume component (35%)
        if volume_ratio >= self.thresholds['ob_volume_ratio_strong']:
            score += 35
        elif volume_ratio >= self.thresholds['ob_volume_ratio_min']:
            score += 25
        else:
            score += 10
        
        # Impulse component (30%)
        if impulse_pct >= self.thresholds['ob_impulse_strong'] * 100:
            score += 30
        elif impulse_pct >= self.thresholds['ob_impulse_min'] * 100:
            score += 20
        else:
            score += 10
        
        # Freshness component (20%)
        if fresh:
            score += 20
        else:
            score += 5
        
        # Structure component (15%) - body ratio
        if body_ratio >= 0.7:
            score += 15
        elif body_ratio >= 0.5:
            score += 10
        else:
            score += 5
        
        return score
    
    def _confirm_mtf(self, obs: List[Dict]) -> List[Dict]:
        """Add bonus for multi-timeframe confirmation"""
        if len(obs) < 2:
            return obs
        
        # Group by type
        bullish = [ob for ob in obs if ob['ob_type'] == 'BULLISH']
        bearish = [ob for ob in obs if ob['ob_type'] == 'BEARISH']
        
        # Check for overlapping zones
        for ob in obs:
            same_type = bullish if ob['ob_type'] == 'BULLISH' else bearish
            
            # Check if other timeframes have overlapping OBs
            for other in same_type:
                if other['timeframe'] != ob['timeframe']:
                    if self._zones_overlap(ob, other):
                        ob['quality_score'] = min(100, 
                            ob['quality_score'] + self.thresholds['mtf_confirmation_bonus'])
                        break
        
        return obs
    
    def _zones_overlap(self, ob1: Dict, ob2: Dict) -> bool:
        """Check if two OB zones overlap"""
        return not (ob1['ob_high'] < ob2['ob_low'] or ob2['ob_high'] < ob1['ob_low'])
    
    def check_price_at_ob(self, symbol: str) -> List[Dict]:
        """
        Check if current price is at any active OB zone
        Returns list of touched/triggered OBs
        """
        current_price = self.fetcher.get_current_price(symbol)
        if not current_price:
            return []
        
        active_obs = self.db.get_orderblocks(symbol=symbol, status='ACTIVE')
        touched = []
        
        for ob in active_obs:
            tolerance = current_price * self.thresholds['ob_touch_tolerance']
            
            # Check if price is in zone
            if ob['ob_low'] - tolerance <= current_price <= ob['ob_high'] + tolerance:
                # Update status
                self.db.update_ob_status(
                    ob_id=ob['id'],
                    status='TOUCHED',
                    touch_count=ob.get('touch_count', 0) + 1
                )
                
                ob['current_price'] = current_price
                ob['status'] = 'TOUCHED'
                touched.append(ob)
                
                self.db.log_event(
                    f"{symbol} price at {ob['ob_type']} OB zone ({ob['timeframe']})",
                    level='INFO', category='OB', symbol=symbol
                )
        
        return touched
    
    def get_entry_signal(self, symbol: str, sleeper_direction: str) -> Optional[Dict]:
        """
        Get entry signal when price touches OB matching sleeper direction
        """
        touched_obs = self.check_price_at_ob(symbol)
        
        for ob in touched_obs:
            # Match direction
            if sleeper_direction == 'LONG' and ob['ob_type'] == 'BULLISH':
                return self._create_signal(symbol, ob, 'LONG')
            elif sleeper_direction == 'SHORT' and ob['ob_type'] == 'BEARISH':
                return self._create_signal(symbol, ob, 'SHORT')
        
        return None
    
    def _create_signal(self, symbol: str, ob: Dict, direction: str) -> Dict:
        """Create trading signal from OB"""
        current_price = ob.get('current_price') or self.fetcher.get_current_price(symbol)
        
        # Calculate entry zone
        entry_price = ob['ob_mid']
        
        # SL below/above OB zone
        if direction == 'LONG':
            stop_loss = ob['ob_low'] * 0.998  # 0.2% below
            risk = entry_price - stop_loss
            take_profit = entry_price + (risk * 2)  # 2R
        else:
            stop_loss = ob['ob_high'] * 1.002  # 0.2% above
            risk = stop_loss - entry_price
            take_profit = entry_price - (risk * 2)  # 2R
        
        return {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ob_id': ob['id'],
            'ob_quality': ob['quality_score'],
            'ob_timeframe': ob['timeframe'],
            'risk_pct': abs(entry_price - stop_loss) / entry_price * 100,
            'distance_percent': abs(current_price - entry_price) / entry_price * 100,
        }
    
    def cleanup_expired(self) -> int:
        """Clean up expired order blocks"""
        max_age = self.db.get_setting('ob_max_age_minutes', 60)
        return self.db.expire_old_orderblocks(max_age)


# Singleton instance
_ob_scanner_instance = None

def get_ob_scanner() -> OBScanner:
    """Get OB scanner instance"""
    global _ob_scanner_instance
    if _ob_scanner_instance is None:
        _ob_scanner_instance = OBScanner()
    return _ob_scanner_instance
