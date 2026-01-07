"""
Order Block Scanner - Exact Pine Script Logic Implementation
Based on "Volumized Order Blocks | SVV Charts" indicator

Detection Logic (from Pine Script):
1. Find swing highs/lows using swingLength
2. When price crosses swing → find origin candle (lowest/highest point before impulse)
3. Mark OB zone with volume data
4. Track invalidation (breaker) when price breaks through zone
"""
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from config import API_LIMITS
from core import get_fetcher, get_indicators
from storage import get_db


@dataclass
class OBSwing:
    """Swing point for OB detection"""
    bar_index: int = 0
    price: float = 0.0
    volume: float = 0.0
    crossed: bool = False


@dataclass  
class OrderBlockInfo:
    """Order Block data structure matching Pine Script"""
    top: float
    bottom: float
    ob_volume: float  # Total volume (3 bars)
    ob_type: str  # "LONG" or "SHORT"
    start_time: int  # Unix timestamp
    ob_low_volume: float  # Volume at OB formation
    ob_high_volume: float  # Volume after impulse
    breaker: bool = False
    break_time: Optional[int] = None
    timeframe: str = ""
    symbol: str = ""
    quality: float = 0.0


class OBScanner:
    """
    Order Block Detector - Pine Script Logic
    
    Exact implementation of "Volumized Order Blocks | SVV Charts"
    
    Detection:
    1. Find swing high/low (swingLength bars)
    2. When close crosses swing → mark OB at origin candle
    3. OB zone = candle body/wick before impulse move
    4. Volume ratio = obHighVolume / obLowVolume
    """
    
    def __init__(self):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.db = get_db()
        
    def _load_settings(self):
        """Load settings from DB - matching Pine Script parameters"""
        # Pine Script parameters
        self.swing_length = int(self.db.get_setting('ob_swing_length', 5))
        self.max_atr_mult = float(self.db.get_setting('ob_max_atr_mult', 3.5))
        self.max_order_blocks = int(self.db.get_setting('ob_max_count', 30))
        self.ob_end_method = self.db.get_setting('ob_end_method', 'Wick')  # "Wick" or "Close"
        
        # Zone count: "High"=10, "Medium"=5, "Low"=3, "One"=1
        zone_count = self.db.get_setting('ob_zone_count', 'Low')
        self.bullish_ob_count = {'One': 1, 'Low': 3, 'Medium': 5, 'High': 10}.get(zone_count, 3)
        self.bearish_ob_count = self.bullish_ob_count
        
        # Quality thresholds
        self.min_quality = float(self.db.get_setting('ob_min_quality', 60))
        self.signal_quality = float(self.db.get_setting('ob_signal_quality', 70))
        
        # Timeframes
        tf_setting = self.db.get_setting('ob_timeframes', '15,5')
        if isinstance(tf_setting, str):
            self.timeframes = [t.strip() for t in tf_setting.split(',')]
        else:
            self.timeframes = tf_setting
    
    def scan_symbol(self, symbol: str, timeframes: List[str] = None) -> List[Dict]:
        """
        Scan a symbol for order blocks on multiple timeframes
        Returns list of detected OBs
        """
        self._load_settings()
        timeframes = timeframes or self.timeframes
        all_obs = []
        
        for tf in timeframes:
            try:
                # Convert timeframe format (15 -> 15, 5 -> 5, etc)
                interval = tf.replace('m', '') if 'm' in tf else tf
                obs = self._detect_order_blocks(symbol, interval)
                all_obs.extend(obs)
                time.sleep(API_LIMITS.get('rate_limit_delay', 0.1))
            except Exception as e:
                print(f"[OB] Error scanning {symbol} {tf}: {e}")
                continue
        
        # Filter by quality and save
        quality_obs = [ob for ob in all_obs if ob['quality'] >= self.min_quality]
        
        for ob in quality_obs:
            self.db.add_orderblock(ob)
        
        return quality_obs
    
    def _detect_order_blocks(self, symbol: str, interval: str) -> List[Dict]:
        """
        Detect order blocks using Pine Script swing logic
        
        Pine Script equivalent:
        - findOBSwings(swingLength)
        - findOrderBlocks()
        """
        # Get klines - need enough data for swing detection
        limit = 200  # Sufficient for swing analysis
        klines = self.fetcher.get_klines(symbol, interval, limit)
        
        if not klines or len(klines) < self.swing_length + 10:
            return []
        
        # Calculate ATR for size filter
        atr = self._calculate_atr(klines, 10)
        
        # Find swing points
        swings = self._find_swings(klines)
        
        # Detect OBs from swings
        bullish_obs = self._detect_bullish_obs(klines, swings['tops'], atr)
        bearish_obs = self._detect_bearish_obs(klines, swings['bottoms'], atr)
        
        # Limit count
        bullish_obs = bullish_obs[:self.bullish_ob_count]
        bearish_obs = bearish_obs[:self.bearish_ob_count]
        
        # Convert to dict format
        result = []
        for ob in bullish_obs + bearish_obs:
            result.append(self._ob_to_dict(ob, symbol, interval, klines))
        
        return result
    
    def _find_swings(self, klines: List[Dict]) -> Dict[str, List[OBSwing]]:
        """
        Find swing highs and lows - Pine Script logic
        
        Pine Script:
        upper = ta.highest(len)
        lower = ta.lowest(len)
        swingType := high[len] > upper ? 0 : low[len] < lower ? 1 : swingType
        """
        tops = []
        bottoms = []
        swing_type = -1
        
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        volumes = [k['volume'] for k in klines]
        
        for i in range(self.swing_length, len(klines) - self.swing_length):
            # Calculate highest/lowest over swing_length (looking forward)
            upper = max(highs[i+1:i+1+self.swing_length]) if i+1+self.swing_length <= len(highs) else highs[i]
            lower = min(lows[i+1:i+1+self.swing_length]) if i+1+self.swing_length <= len(lows) else lows[i]
            
            prev_swing_type = swing_type
            
            # Check if current bar is swing
            if highs[i] > upper:
                swing_type = 0  # Potential swing high
            elif lows[i] < lower:
                swing_type = 1  # Potential swing low
            
            # When swing type changes, record the swing point
            if swing_type == 0 and prev_swing_type != 0:
                tops.append(OBSwing(
                    bar_index=i,
                    price=highs[i],
                    volume=volumes[i],
                    crossed=False
                ))
            
            if swing_type == 1 and prev_swing_type != 1:
                bottoms.append(OBSwing(
                    bar_index=i,
                    price=lows[i],
                    volume=volumes[i],
                    crossed=False
                ))
        
        return {'tops': tops, 'bottoms': bottoms}
    
    def _detect_bullish_obs(self, klines: List[Dict], tops: List[OBSwing], atr: float) -> List[OrderBlockInfo]:
        """
        Detect Bullish Order Blocks - Pine Script logic
        
        When close > swing high and not crossed:
        - Find lowest point between swing and current bar
        - That's the OB zone
        """
        obs = []
        
        for top in tops:
            if top.crossed:
                continue
            
            # Look for cross after the swing
            for i in range(top.bar_index + 1, len(klines)):
                if klines[i]['close'] > top.price:
                    # Swing crossed! Find OB origin
                    top.crossed = True
                    
                    # Find lowest point between swing and current bar
                    box_btm = klines[i-1]['low'] if i > 0 else klines[i]['low']
                    box_top = klines[i-1]['high'] if i > 0 else klines[i]['high']
                    box_idx = i - 1
                    
                    for j in range(1, i - top.bar_index):
                        idx = i - j
                        if idx >= 0 and klines[idx]['low'] < box_btm:
                            box_btm = klines[idx]['low']
                            box_top = klines[idx]['high']
                            box_idx = idx
                    
                    # Check size vs ATR
                    ob_size = abs(box_top - box_btm)
                    if ob_size > atr * self.max_atr_mult:
                        continue
                    
                    # Volume calculation (3 bars)
                    vol_0 = klines[i]['volume'] if i < len(klines) else 0
                    vol_1 = klines[i-1]['volume'] if i-1 >= 0 else 0
                    vol_2 = klines[i-2]['volume'] if i-2 >= 0 else 0
                    
                    ob = OrderBlockInfo(
                        top=box_top,
                        bottom=box_btm,
                        ob_volume=vol_0 + vol_1 + vol_2,
                        ob_type="LONG",
                        start_time=klines[box_idx]['timestamp'],
                        ob_low_volume=vol_2,  # Volume before impulse
                        ob_high_volume=vol_0 + vol_1,  # Volume during impulse
                        timeframe=str(klines[0].get('interval', ''))
                    )
                    
                    # Check if already invalidated
                    ob = self._check_invalidation(ob, klines, i)
                    
                    obs.append(ob)
                    break
        
        return obs
    
    def _detect_bearish_obs(self, klines: List[Dict], bottoms: List[OBSwing], atr: float) -> List[OrderBlockInfo]:
        """
        Detect Bearish Order Blocks - Pine Script logic
        
        When close < swing low and not crossed:
        - Find highest point between swing and current bar
        - That's the OB zone
        """
        obs = []
        
        for btm in bottoms:
            if btm.crossed:
                continue
            
            # Look for cross after the swing
            for i in range(btm.bar_index + 1, len(klines)):
                if klines[i]['close'] < btm.price:
                    # Swing crossed! Find OB origin
                    btm.crossed = True
                    
                    # Find highest point between swing and current bar
                    box_top = klines[i-1]['high'] if i > 0 else klines[i]['high']
                    box_btm = klines[i-1]['low'] if i > 0 else klines[i]['low']
                    box_idx = i - 1
                    
                    for j in range(1, i - btm.bar_index):
                        idx = i - j
                        if idx >= 0 and klines[idx]['high'] > box_top:
                            box_top = klines[idx]['high']
                            box_btm = klines[idx]['low']
                            box_idx = idx
                    
                    # Check size vs ATR
                    ob_size = abs(box_top - box_btm)
                    if ob_size > atr * self.max_atr_mult:
                        continue
                    
                    # Volume calculation (reversed for bearish)
                    vol_0 = klines[i]['volume'] if i < len(klines) else 0
                    vol_1 = klines[i-1]['volume'] if i-1 >= 0 else 0
                    vol_2 = klines[i-2]['volume'] if i-2 >= 0 else 0
                    
                    ob = OrderBlockInfo(
                        top=box_top,
                        bottom=box_btm,
                        ob_volume=vol_0 + vol_1 + vol_2,
                        ob_type="SHORT",
                        start_time=klines[box_idx]['timestamp'],
                        ob_low_volume=vol_0 + vol_1,  # Volume during impulse
                        ob_high_volume=vol_2,  # Volume before impulse
                        timeframe=str(klines[0].get('interval', ''))
                    )
                    
                    # Check if already invalidated
                    ob = self._check_invalidation(ob, klines, i)
                    
                    obs.append(ob)
                    break
        
        return obs
    
    def _check_invalidation(self, ob: OrderBlockInfo, klines: List[Dict], start_idx: int) -> OrderBlockInfo:
        """
        Check if OB has been invalidated (breaker)
        
        Pine Script:
        Bullish: if (obEndMethod == "Wick" ? low : close) < OB.bottom → breaker
        Bearish: if (obEndMethod == "Wick" ? high : close) > OB.top → breaker
        """
        for i in range(start_idx + 1, len(klines)):
            if ob.ob_type == "LONG":
                # LONG OB invalidated when price goes below bottom
                check_price = klines[i]['low'] if self.ob_end_method == "Wick" else klines[i]['close']
                if check_price < ob.bottom:
                    ob.breaker = True
                    ob.break_time = klines[i]['timestamp']
                    break
            else:
                # SHORT OB invalidated when price goes above top
                check_price = klines[i]['high'] if self.ob_end_method == "Wick" else klines[i]['close']
                if check_price > ob.top:
                    ob.breaker = True
                    ob.break_time = klines[i]['timestamp']
                    break
        
        return ob
    
    def _calculate_atr(self, klines: List[Dict], period: int = 10) -> float:
        """Calculate ATR (Average True Range)"""
        if len(klines) < period + 1:
            return 0
        
        tr_values = []
        for i in range(1, min(period + 1, len(klines))):
            high = klines[i]['high']
            low = klines[i]['low']
            prev_close = klines[i-1]['close']
            
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)
        
        return sum(tr_values) / len(tr_values) if tr_values else 0
    
    def _calculate_quality(self, ob: OrderBlockInfo) -> float:
        """
        Calculate OB quality score (0-100)
        
        Based on:
        - Volume ratio (higher = better confirmation)
        - Size relative to ATR
        - Whether still valid (not breaker)
        """
        quality = 50.0  # Base score
        
        # Volume ratio bonus (Pine Script shows percentage)
        if ob.ob_high_volume > 0 and ob.ob_low_volume > 0:
            vol_ratio = min(ob.ob_high_volume, ob.ob_low_volume) / max(ob.ob_high_volume, ob.ob_low_volume)
            # Higher ratio = more balanced = better
            quality += vol_ratio * 30
        
        # Breaker penalty
        if ob.breaker:
            quality -= 20
        
        # Volume presence bonus
        if ob.ob_volume > 0:
            quality += 10
        
        return min(100, max(0, quality))
    
    def _ob_to_dict(self, ob: OrderBlockInfo, symbol: str, interval: str, klines: List[Dict] = None) -> Dict:
        """Convert OrderBlockInfo to dict for DB storage"""
        quality = self._calculate_quality(ob)
        
        # Volume ratio (як Pine Script показує у відсотках) 
        vol_ratio = 0.0
        if ob.ob_high_volume > 0 and ob.ob_low_volume > 0:
            vol_ratio = round(max(ob.ob_high_volume, ob.ob_low_volume) / 
                             min(ob.ob_high_volume, ob.ob_low_volume), 1)
        
        # Impulse percentage - зміна ціни від OB до найближчого екстремуму
        impulse_pct = 0.0
        if klines and ob.top > 0 and ob.bottom > 0:
            ob_mid = (ob.top + ob.bottom) / 2
            
            # Знаходимо max/min ціну після OB формування
            recent_prices = []
            for k in klines[-20:]:  # Останні 20 свічок
                recent_prices.append(float(k['high']))
                recent_prices.append(float(k['low']))
            
            if recent_prices:
                if ob.ob_type == 'LONG':
                    # Для LONG OB - рух вгору
                    max_price = max(recent_prices)
                    if ob_mid > 0:
                        impulse_pct = round((max_price - ob_mid) / ob_mid * 100, 2)
                else:
                    # Для SHORT OB - рух вниз
                    min_price = min(recent_prices)
                    if ob_mid > 0:
                        impulse_pct = round((ob_mid - min_price) / ob_mid * 100, 2)
        
        return {
            'symbol': symbol,
            'timeframe': interval,
            'ob_type': ob.ob_type,
            'top': float(ob.top),
            'bottom': float(ob.bottom),
            'volume': float(ob.ob_volume),
            'volume_ratio': vol_ratio,        # Відношення (1.5x, 2.3x, etc)
            'impulse_pct': abs(impulse_pct),  # Відсоток руху
            'ob_low_volume': float(ob.ob_low_volume),
            'ob_high_volume': float(ob.ob_high_volume),
            'quality': round(quality, 2),
            'start_time': ob.start_time,
            'breaker': ob.breaker,
            'break_time': ob.break_time,
            'mitigated': ob.breaker,
            'created_at': datetime.utcnow(),
        }
    
    def scan_ready_sleepers(self) -> List[Dict]:
        """Scan all ready sleepers for OBs"""
        sleepers = self.db.get_sleepers(state='READY')
        all_obs = []
        
        for sleeper in sleepers:
            try:
                obs = self.scan_symbol(sleeper['symbol'])
                all_obs.extend(obs)
            except Exception as e:
                print(f"[OB] Error scanning {sleeper['symbol']}: {e}")
        
        return all_obs
    
    def get_active_obs(self, symbol: str = None) -> List[Dict]:
        """Get active (non-mitigated) OBs"""
        return self.db.get_orderblocks(symbol=symbol, status='ACTIVE')
    
    def check_price_near_ob(self, symbol: str, current_price: float, 
                           proximity_pct: float = 0.5) -> Optional[Dict]:
        """
        Check if price is near any active OB
        Returns the OB if price is within proximity_pct of zone
        """
        active_obs = self.get_active_obs(symbol)
        
        for ob in active_obs:
            zone_size = ob['top'] - ob['bottom']
            proximity = zone_size * (proximity_pct / 100)
            
            # Check if price is near the zone
            if ob['ob_type'] == 'LONG':
                # For LONG OB, check if price approaching from above
                if ob['bottom'] - proximity <= current_price <= ob['top'] + proximity:
                    return ob
            else:
                # For SHORT OB, check if price approaching from below
                if ob['bottom'] - proximity <= current_price <= ob['top'] + proximity:
                    return ob
        
        return None
    
    def cleanup_expired(self) -> int:
        """
        Remove expired or mitigated order blocks
        Returns number of cleaned up OBs
        """
        from datetime import datetime, timedelta
        
        session = None
        try:
            from storage.db_models import OrderBlock, get_session
            session = get_session()
            
            # Mark expired OBs
            expired_count = session.query(OrderBlock).filter(
                OrderBlock.status == 'ACTIVE',
                OrderBlock.expires_at < datetime.utcnow()
            ).update({'status': 'EXPIRED'})
            
            # Delete very old OBs (older than 7 days)
            cutoff = datetime.utcnow() - timedelta(days=7)
            deleted_count = session.query(OrderBlock).filter(
                OrderBlock.created_at < cutoff
            ).delete()
            
            session.commit()
            
            total = expired_count + deleted_count
            if total > 0:
                print(f"[OB] Cleanup: {expired_count} expired, {deleted_count} deleted")
            
            return total
        except Exception as e:
            if session:
                session.rollback()
            print(f"[OB] Cleanup error: {e}")
            return 0
        finally:
            if session:
                session.close()
    
    def get_entry_signal(self, symbol: str, direction: str) -> Optional[Dict]:
        """
        Check if there's a valid entry signal for symbol
        
        Returns signal dict if:
        1. Active OB exists for symbol
        2. Current price is near OB zone
        3. Direction matches (LONG→Bullish OB, SHORT→Bearish OB)
        """
        # Get current price
        try:
            ticker = self.fetcher.get_ticker(symbol)
            if not ticker:
                return None
            current_price = float(ticker.get('lastPrice', 0))
            if current_price <= 0:
                return None
        except Exception as e:
            print(f"[OB] Error getting price for {symbol}: {e}")
            return None
        
        # Get active OBs for symbol
        active_obs = self.get_active_obs(symbol)
        if not active_obs:
            return None
        
        # Filter by direction (OB type matches direction: LONG→LONG OB, SHORT→SHORT OB)
        expected_ob_type = 'LONG' if direction == 'LONG' else 'SHORT'
        matching_obs = [ob for ob in active_obs if ob['ob_type'] == expected_ob_type]
        
        if not matching_obs:
            return None
        
        # Check proximity to each OB
        for ob in matching_obs:
            zone_size = ob['ob_high'] - ob['ob_low']
            proximity = zone_size * 0.5  # 50% of zone size
            
            # Check if price is in or near the zone
            if direction == 'LONG':
                # For LONG, price should be at or below bullish OB (support)
                if ob['ob_low'] - proximity <= current_price <= ob['ob_high'] + proximity:
                    return self._create_signal(symbol, direction, ob, current_price)
            else:
                # For SHORT, price should be at or above bearish OB (resistance)
                if ob['ob_low'] - proximity <= current_price <= ob['ob_high'] + proximity:
                    return self._create_signal(symbol, direction, ob, current_price)
        
        return None
    
    def _create_signal(self, symbol: str, direction: str, ob: Dict, current_price: float) -> Dict:
        """Create entry signal from OB"""
        zone_size = ob['ob_high'] - ob['ob_low']
        
        # Entry at zone edge
        if direction == 'LONG':
            entry_price = ob['ob_low']  # Enter at bottom of bullish OB
            stop_loss = entry_price - (zone_size * 1.5)  # SL below zone
            take_profit = entry_price + (zone_size * 3)  # 3:1 R:R
        else:
            entry_price = ob['ob_high']  # Enter at top of bearish OB
            stop_loss = entry_price + (zone_size * 1.5)  # SL above zone
            take_profit = entry_price - (zone_size * 3)  # 3:1 R:R
        
        return {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'current_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ob_id': ob['id'],
            'ob_type': ob['ob_type'],
            'ob_quality': ob.get('quality_score', 0),
            'zone_high': ob['ob_high'],
            'zone_low': ob['ob_low'],
            'timeframe': ob['timeframe'],
        }


# Singleton instance
_ob_scanner = None

def get_ob_scanner() -> OBScanner:
    """Get singleton OB scanner instance"""
    global _ob_scanner
    if _ob_scanner is None:
        _ob_scanner = OBScanner()
    return _ob_scanner
