"""
Sleeper Scanner v3.2 - 5-Day Strategy with VC-EXTREME Detection

CRITICAL FIX v3.2: BB width zero-filter bug fixed!
- Previous: bb_widths[:5] = [0,0,0,0,0] → bb_width_start = 0 → VC always = 20
- Fixed: Filter out zero values before calculation

Система оцінки:
- VOLATILITY_COMPRESSION: 40% - стиснення волатильності за 5 днів
- VOLUME_SUPPRESSION: 25% - пригнічення об'ємів
- OI_GROWTH: 20% - зростання Open Interest
- ORDER_BOOK_IMBALANCE: 15% - дисбаланс стакану

Для VC > 90% використовується modified formula:
- VC: 50%, VS: 15%, OI: 25%, OB: 10%

ДВА ШЛЯХИ ДО BUILDING:
1. Classic: Score > 55, VC > 50%, VS > 60%
2. Accelerated: VC > 90% + OI growth > 15% (пропускає VS requirement)

Стани:
- IDLE: Не відстежується
- WATCHING: Score > 40 OR VC > 95%
- BUILDING: Classic path OR Accelerated path (VC > 90% + OI growth)
- READY: Score > 65, compression > 70% OR VC > 95% + OI > 20%
- TRIGGERED: Volume > 200% + OI jump > 15%

Флаги:
- VC_EXTREME: VC > 95% при VOL < 1.2x (imminent breakout)
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from config import API_LIMITS
from config.bot_settings import SleeperState
from core.market_data import get_fetcher
from core.tech_indicators import get_indicators
from storage import get_db


class SleeperScannerV3:
    """
    5-Day Sleeper Detection Strategy
    
    Шукає монети що готуються до breakout:
    1. Волатильність стискається (BB squeeze)
    2. Об'єми знижуються (накопичення)
    3. OI росте (позиції накопичуються)
    4. Стакан показує напрямок (дисбаланс)
    """
    
    # === КОНФІГУРАЦІЯ ===
    
    # Scoring weights (must sum to 100)
    WEIGHTS = {
        'volatility_compression': 40,
        'volume_suppression': 25,
        'oi_growth': 20,
        'order_book_imbalance': 15,
    }
    
    # Data requirements
    KLINES_5D = 30   # 30 x 4H = 5 days
    KLINES_1D = 7    # 7 days for context
    OI_PERIODS = 30  # 30 hours of OI data
    
    # Score thresholds
    MIN_SCORE_WATCHING = 40
    MIN_SCORE_BUILDING = 55
    MIN_SCORE_READY = 65
    
    # Transition conditions
    COMPRESSION_BUILDING = 50   # % BB compression for BUILDING
    COMPRESSION_READY = 70      # % BB compression for READY
    VOLUME_SUPPRESSION_MIN = 60 # % volume below average for BUILDING
    
    # Trigger conditions
    VOLUME_SPIKE_THRESHOLD = 200  # % of average for trigger
    OI_JUMP_THRESHOLD = 15        # % OI jump for trigger
    
    # HP System
    HP_INITIAL = 5
    HP_MAX = 10
    HP_MIN = 0
    
    def __init__(self):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.db = get_db()
        
        # Scan settings from DB
        self.max_symbols = self.db.get_setting('sleeper_max_symbols', 150)
        self.min_volume = self.db.get_setting('sleeper_min_volume', 50_000_000)
        self.scan_interval = self.db.get_setting('sleeper_scan_interval', 240)  # 4H
    
    def run_scan(self) -> List[Dict]:
        """
        Run full sleeper scan
        Returns list of candidates
        """
        print(f"\n[SLEEPER v3] Starting 5-day strategy scan...")
        start_time = time.time()
        
        # 1. Get top symbols by volume
        symbols = self.fetcher.get_top_symbols(
            limit=self.max_symbols,
            min_volume=self.min_volume
        )
        
        if not symbols:
            print("[SLEEPER v3] No symbols found")
            return []
        
        print(f"[SLEEPER v3] Analyzing {len(symbols)} symbols...")
        
        candidates = []
        errors = 0
        vc_extreme_count = 0
        
        for i, sym_data in enumerate(symbols):
            try:
                result = self._analyze_symbol_5day(sym_data)
                
                if result:
                    # Save to database
                    saved = self.db.upsert_sleeper(result)
                    if saved:
                        candidates.append(result)
                        state_emoji = self._get_state_emoji(result['state'])
                        
                        # VC-Extreme flag
                        vc_flag = "🔥" if result.get('vc_extreme_detected') else ""
                        if result.get('vc_extreme_detected'):
                            vc_extreme_count += 1
                        
                        print(f"[SLEEPER v3] {state_emoji}{vc_flag} {result['symbol']}: "
                              f"Score={result['total_score']:.1f} "
                              f"Dir={result['direction']} "
                              f"(VC:{result['volatility_compression']:.0f} "
                              f"VS:{result['volume_suppression']:.0f} "
                              f"OI:{result['oi_growth']:.0f} "
                              f"OB:{result['order_book_imbalance']:.0f})"
                              f"{' [VC-EXTREME]' if vc_flag else ''}"
                              f" BB:{result.get('bb_compression_pct', 0):.1f}%")
                
                # Progress
                if (i + 1) % 20 == 0:
                    print(f"[SLEEPER v3] Progress: {i+1}/{len(symbols)} ({len(candidates)} candidates)")
                
                # Rate limiting
                time.sleep(API_LIMITS.get('rate_limit_delay', 0.1))
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"[SLEEPER v3] Error analyzing {sym_data.get('symbol')}: {e}")
        
        # Update HP for existing sleepers
        self._update_hp_scores(candidates)
        
        # Remove dead sleepers
        removed = self.db.remove_dead_sleepers()
        if removed > 0:
            print(f"[SLEEPER v3] Removed {removed} dead sleepers (HP=0)")
        
        elapsed = time.time() - start_time
        print(f"\n[SLEEPER v3.2] Scan complete in {elapsed:.1f}s")
        print(f"[SLEEPER v3.2] Results: {len(candidates)} candidates from {len(symbols)} symbols")
        
        # Count by state
        by_state = {}
        for c in candidates:
            state = c.get('state', 'UNKNOWN')
            by_state[state] = by_state.get(state, 0) + 1
        print(f"[SLEEPER v3.2] States: {by_state}")
        
        # VC-Extreme summary
        if vc_extreme_count > 0:
            print(f"[SLEEPER v3.2] 🔥 VC-EXTREME candidates: {vc_extreme_count}")
        
        return candidates
    
    def _analyze_symbol_5day(self, symbol_data: Dict) -> Optional[Dict]:
        """
        Analyze symbol using 5-day strategy
        """
        symbol = symbol_data['symbol']
        
        # === 1. GET 5-DAY DATA ===
        
        # 4H klines for 5 days (30 candles)
        klines_4h = self.fetcher.get_klines(symbol, '4h', limit=self.KLINES_5D)
        if len(klines_4h) < 20:
            return None
        
        # Daily klines for context
        klines_1d = self.fetcher.get_klines(symbol, '1d', limit=self.KLINES_1D)
        
        # OI history
        oi_history = self.fetcher.get_oi_history(symbol, limit=self.OI_PERIODS)
        
        # Current OI
        current_oi_data = self.fetcher.connector.get_current_open_interest(symbol)
        current_oi = current_oi_data.get('open_interest', 0) if current_oi_data else 0
        
        # Order book
        orderbook = self.fetcher.get_orderbook_imbalance(symbol, depth=20)
        
        # Funding rate
        funding_rate = self.fetcher.get_funding_rate(symbol) or 0
        
        # === 2. CALCULATE INDICATORS ===
        
        indicators_4h = self.indicators.calculate_all(klines_4h)
        indicators_1d = self.indicators.calculate_all(klines_1d) if len(klines_1d) > 5 else None
        
        # === 3. CALCULATE 5-DAY METRICS ===
        
        # Volatility Compression (40%)
        vc_data = self._calculate_volatility_compression(klines_4h, indicators_4h)
        
        # Volume Suppression (25%)
        vs_data = self._calculate_volume_suppression(klines_4h)
        
        # OI Growth (20%)
        oi_data = self._calculate_oi_growth(oi_history, current_oi)
        
        # Order Book Imbalance (15%)
        ob_data = self._calculate_order_book_imbalance(orderbook)
        
        # === 4. CALCULATE TOTAL SCORE ===
        
        total_score = (
            vc_data['score'] * (self.WEIGHTS['volatility_compression'] / 100) +
            vs_data['score'] * (self.WEIGHTS['volume_suppression'] / 100) +
            oi_data['score'] * (self.WEIGHTS['oi_growth'] / 100) +
            ob_data['score'] * (self.WEIGHTS['order_book_imbalance'] / 100)
        )
        
        # === 5. DATA QUALITY CHECK ===
        
        data_issues = 0
        if vc_data['bb_width_current'] == 0:
            data_issues += 1
        if vs_data['volume_current'] == 0:
            data_issues += 1
        if current_oi == 0:
            data_issues += 1
        
        if data_issues >= 2:
            return None
        
        # === 6. VC-ADJUSTED SCORE ===
        # For extreme compression (VC > 90%), use modified weights
        # This prioritizes volatility compression over volume suppression
        
        if vc_data['compression_pct'] >= 90:
            # Modified formula: VC*0.5 + VS*0.15 + OI*0.25 + OB*0.10
            adjusted_score = (
                vc_data['score'] * 0.50 +
                vs_data['score'] * 0.15 +
                oi_data['score'] * 0.25 +
                ob_data['score'] * 0.10
            )
            # Use higher of standard or adjusted score
            total_score = max(total_score, adjusted_score)
        
        # === 7. DETERMINE STATE ===
        
        state, vc_extreme = self._determine_state(
            total_score,
            vc_data['compression_pct'],
            vs_data['suppression_pct'],
            vs_data['volume_spike'],
            oi_data['oi_jump'],
            # Additional params for accelerated paths
            vc_score=vc_data['score'],
            oi_score=oi_data['score'],
            oi_growth_pct=oi_data['growth_pct'],
            volume_ratio=vs_data['volume_ratio']
        )
        
        # Minimum score filter (but keep VC-extreme candidates)
        if state == 'IDLE' and total_score < self.MIN_SCORE_WATCHING and not vc_extreme:
            return None
        
        # === 8. DETERMINE DIRECTION ===
        
        direction = self._determine_direction(
            funding_rate,
            ob_data['imbalance'],
            indicators_4h.get('rsi_current', 50),
            indicators_1d.get('rsi_current', 50) if indicators_1d else 50
        )
        
        # === 9. CALCULATE DYNAMIC HP ===
        hp = self._calculate_initial_hp(total_score, state)
        
        # Boost HP for VC-extreme candidates
        if vc_extreme:
            hp = min(self.HP_MAX, hp + 2)
        
        # === 10. BUILD RESULT ===
        
        return {
            'symbol': symbol,
            'total_score': round(total_score, 2),
            
            # 5-day scores
            'volatility_compression': round(vc_data['score'], 2),
            'volume_suppression': round(vs_data['score'], 2),
            'oi_growth': round(oi_data['score'], 2),
            'order_book_imbalance': round(ob_data['score'], 2),
            
            # Legacy scores (for compatibility)
            'fuel_score': round(oi_data['score'], 2),
            'volatility_score': round(vc_data['score'], 2),
            'price_score': round(vs_data['score'], 2),
            'liquidity_score': round(ob_data['score'], 2),
            
            # State
            'state': state,
            'hp': hp,  # Dynamic HP based on score
            'direction': direction,
            'vc_extreme_detected': vc_extreme,  # v3.1: VC-Extreme flag
            
            # 5-day metrics
            'bb_width_5d_start': vc_data['bb_width_start'],
            'bb_width_current': vc_data['bb_width_current'],
            'bb_compression_pct': vc_data['compression_pct'],
            'volume_5d_avg': vs_data['volume_avg'],
            'volume_current': vs_data['volume_current'],
            'volume_ratio': vs_data['volume_ratio'],
            'oi_5d_start': oi_data['oi_start'],
            'oi_current': oi_data['oi_current'],
            'oi_growth_pct': oi_data['growth_pct'],
            'bid_volume': ob_data['bid_volume'],
            'ask_volume': ob_data['ask_volume'],
            'ob_imbalance_pct': ob_data['imbalance'],
            
            # Trigger flags
            'volume_spike_detected': vs_data['volume_spike'],
            'oi_jump_detected': oi_data['oi_jump'],
            
            # Legacy metrics
            'funding_rate': funding_rate,
            'oi_change_4h': oi_data['growth_pct'],
            'bb_width': vc_data['bb_width_current'],
            'volume_24h': symbol_data.get('volume_24h', 0),
            'rsi': indicators_4h.get('rsi_current', 50),
        }
    
    def _calculate_initial_hp(self, total_score: float, state: str) -> int:
        """
        Calculate initial HP based on score and state
        
        HP Scale:
        - READY/TRIGGERED: 7-10 (high priority)
        - BUILDING: 5-7 (medium-high)
        - WATCHING: 3-5 (medium)
        - IDLE: 1-3 (low)
        """
        # Base HP from score (higher score = higher HP)
        if total_score >= 80:
            base_hp = 8
        elif total_score >= 70:
            base_hp = 7
        elif total_score >= 60:
            base_hp = 6
        elif total_score >= 50:
            base_hp = 5
        elif total_score >= 40:
            base_hp = 4
        else:
            base_hp = 3
        
        # State modifier
        if state in [SleeperState.READY.value, SleeperState.TRIGGERED.value]:
            state_bonus = 2
        elif state == SleeperState.BUILDING.value:
            state_bonus = 1
        elif state == SleeperState.WATCHING.value:
            state_bonus = 0
        else:
            state_bonus = -1
        
        hp = base_hp + state_bonus
        return max(self.HP_MIN, min(self.HP_MAX, hp))
    
    def _calculate_volatility_compression(self, klines: List[Dict], indicators: Dict) -> Dict:
        """
        Calculate BB width compression over 5 days
        Returns score 0-100 based on compression amount
        
        v3.1 FIX: Filter out zero values from BB width array
        """
        bb_widths_raw = indicators.get('bb', {}).get('width', [])
        
        # CRITICAL FIX: Filter out zero values (BB period warmup)
        bb_widths = [w for w in bb_widths_raw if w > 0]
        
        min_required = 5  # Need at least 5 valid BB width values
        
        if not bb_widths or len(bb_widths) < min_required:
            # Fallback: calculate from klines directly using range compression
            if klines and len(klines) >= 10:
                closes = [k['close'] for k in klines]
                highs = [k['high'] for k in klines]
                lows = [k['low'] for k in klines]
                
                # Calculate ATR-style range compression
                # First half vs second half of data
                half = len(klines) // 2
                
                old_ranges = []
                new_ranges = []
                
                for i in range(half):
                    if closes[i] > 0:
                        old_ranges.append((highs[i] - lows[i]) / closes[i] * 100)
                        
                for i in range(half, len(klines)):
                    if closes[i] > 0:
                        new_ranges.append((highs[i] - lows[i]) / closes[i] * 100)
                
                old_range = sum(old_ranges) / len(old_ranges) if old_ranges else 0
                new_range = sum(new_ranges) / len(new_ranges) if new_ranges else 0
                
                if old_range > 0:
                    compression_pct = ((old_range - new_range) / old_range) * 100
                else:
                    compression_pct = 0
                
                score = self._compression_to_score(compression_pct)
                
                return {
                    'score': score,
                    'bb_width_start': round(old_range, 4),
                    'bb_width_current': round(new_range, 4),
                    'compression_pct': round(compression_pct, 2),
                }
            
            return {
                'score': 50,  # Neutral when no data
                'bb_width_start': 0,
                'bb_width_current': 0,
                'compression_pct': 0,
            }
        
        # Now bb_widths contains only valid (non-zero) values
        available = len(bb_widths)
        
        # Take first 1/3 for start, last 1/3 for current
        third = max(1, available // 3)
        
        # Oldest valid BB widths (start of compression period)
        bb_width_start = sum(bb_widths[:third]) / third
        
        # Newest valid BB widths (current compression state)
        bb_width_current = sum(bb_widths[-third:]) / third
        
        # Calculate compression percentage
        if bb_width_start > 0:
            compression_pct = ((bb_width_start - bb_width_current) / bb_width_start) * 100
        else:
            compression_pct = 0
        
        score = self._compression_to_score(compression_pct)
        
        return {
            'score': score,
            'bb_width_start': round(bb_width_start, 4),
            'bb_width_current': round(bb_width_current, 4),
            'compression_pct': round(compression_pct, 2),
        }
    
    def _compression_to_score(self, compression_pct: float) -> float:
        """
        Convert compression percentage to score 0-100
        
        Positive compression = volatility decreasing = higher score
        Negative compression = volatility expanding = lower score but not zero
        """
        if compression_pct >= 70:
            return 100  # Extreme compression - ready to explode
        elif compression_pct >= 60:
            return 90
        elif compression_pct >= 50:
            return 80
        elif compression_pct >= 40:
            return 70
        elif compression_pct >= 30:
            return 60
        elif compression_pct >= 20:
            return 50
        elif compression_pct >= 10:
            return 45
        elif compression_pct >= 0:
            return 40  # No compression but stable
        elif compression_pct >= -10:
            return 35  # Slight expansion
        elif compression_pct >= -20:
            return 30  # Moderate expansion
        elif compression_pct >= -30:
            return 25  # Significant expansion
        else:
            return 20  # Strong expansion
    
    def _calculate_volume_suppression(self, klines: List[Dict]) -> Dict:
        """
        Calculate volume suppression over 5 days
        Low volume during consolidation = accumulation
        """
        if not klines or len(klines) < 10:
            return {
                'score': 50,
                'volume_avg': 0,
                'volume_current': 0,
                'volume_ratio': 1.0,
                'suppression_pct': 0,
                'volume_spike': False,
            }
        
        volumes = [k['volume'] for k in klines]
        
        # 5-day average (excluding last 3 candles)
        volume_avg = sum(volumes[:-3]) / len(volumes[:-3]) if len(volumes) > 3 else sum(volumes) / len(volumes)
        
        # Current volume (average of last 3 candles)
        volume_current = sum(volumes[-3:]) / 3 if len(volumes) >= 3 else volumes[-1]
        
        # Volume ratio
        volume_ratio = volume_current / volume_avg if volume_avg > 0 else 1.0
        
        # Suppression percentage (how much below average)
        suppression_pct = max(0, (1 - volume_ratio) * 100)
        
        # Volume spike detection
        volume_spike = volume_ratio >= (self.VOLUME_SPIKE_THRESHOLD / 100)
        
        # Score based on suppression
        # More suppression (lower volume) during consolidation = higher score
        if suppression_pct >= 70:
            score = 100  # Very low volume - strong accumulation signal
        elif suppression_pct >= 60:
            score = 90
        elif suppression_pct >= 50:
            score = 80
        elif suppression_pct >= 40:
            score = 70
        elif suppression_pct >= 30:
            score = 60
        elif suppression_pct >= 20:
            score = 50
        elif suppression_pct >= 10:
            score = 40
        elif volume_spike:
            score = 30  # Volume spike - might be triggering
        else:
            score = 35  # Normal volume
        
        return {
            'score': score,
            'volume_avg': round(volume_avg, 2),
            'volume_current': round(volume_current, 2),
            'volume_ratio': round(volume_ratio, 4),
            'suppression_pct': round(suppression_pct, 2),
            'volume_spike': volume_spike,
        }
    
    def _calculate_oi_growth(self, oi_history: List[Dict], current_oi: float) -> Dict:
        """
        Calculate OI growth over available history
        Growing OI during consolidation = positions accumulating
        """
        if not oi_history or current_oi == 0:
            return {
                'score': 50,
                'oi_start': 0,
                'oi_current': current_oi,
                'growth_pct': 0,
                'oi_jump': False,
            }
        
        # Get OI values
        oi_values = [h.get('open_interest', 0) for h in oi_history]
        oi_values = [v for v in oi_values if v > 0]
        
        if not oi_values:
            return {
                'score': 50,
                'oi_start': 0,
                'oi_current': current_oi,
                'growth_pct': 0,
                'oi_jump': False,
            }
        
        # OI at start
        oi_start = sum(oi_values[:5]) / min(5, len(oi_values))
        
        # Growth percentage
        if oi_start > 0:
            growth_pct = ((current_oi - oi_start) / oi_start) * 100
        else:
            growth_pct = 0
        
        # OI jump detection (sudden increase)
        recent_oi = oi_values[-3:] if len(oi_values) >= 3 else oi_values
        if len(recent_oi) >= 2:
            recent_change = ((recent_oi[-1] - recent_oi[0]) / recent_oi[0]) * 100 if recent_oi[0] > 0 else 0
            oi_jump = recent_change >= self.OI_JUMP_THRESHOLD
        else:
            oi_jump = False
        
        # Score based on OI growth
        # Positive growth = accumulation = higher score
        if growth_pct >= 30:
            score = 100
        elif growth_pct >= 20:
            score = 90
        elif growth_pct >= 15:
            score = 80
        elif growth_pct >= 10:
            score = 70
        elif growth_pct >= 5:
            score = 60
        elif growth_pct > 0:
            score = 50
        elif growth_pct > -5:
            score = 40  # Slight decrease
        else:
            score = 30  # Distribution
        
        return {
            'score': score,
            'oi_start': round(oi_start, 2),
            'oi_current': round(current_oi, 2),
            'growth_pct': round(growth_pct, 2),
            'oi_jump': oi_jump,
        }
    
    def _calculate_order_book_imbalance(self, orderbook: Dict) -> Dict:
        """
        Calculate order book imbalance
        Strong imbalance = directional pressure
        """
        bid_volume = orderbook.get('bid_volume', 0)
        ask_volume = orderbook.get('ask_volume', 0)
        
        total = bid_volume + ask_volume
        
        if total == 0:
            return {
                'score': 50,
                'bid_volume': 0,
                'ask_volume': 0,
                'imbalance': 0,
            }
        
        # Imbalance: positive = more bids (bullish), negative = more asks (bearish)
        imbalance = ((bid_volume - ask_volume) / total) * 100
        
        # Score based on absolute imbalance
        # Strong imbalance in either direction = higher score
        abs_imbalance = abs(imbalance)
        
        if abs_imbalance >= 40:
            score = 100  # Very strong imbalance
        elif abs_imbalance >= 30:
            score = 85
        elif abs_imbalance >= 20:
            score = 70
        elif abs_imbalance >= 10:
            score = 55
        else:
            score = 40  # Balanced book
        
        return {
            'score': score,
            'bid_volume': round(bid_volume, 2),
            'ask_volume': round(ask_volume, 2),
            'imbalance': round(imbalance, 2),
        }
    
    def _determine_state(self, total_score: float, compression_pct: float,
                         suppression_pct: float, volume_spike: bool, oi_jump: bool,
                         vc_score: float = 0, oi_score: float = 0, 
                         oi_growth_pct: float = 0, volume_ratio: float = 1.0) -> Tuple[str, bool]:
        """
        Determine sleeper state based on metrics
        
        Returns: (state, vc_extreme_flag)
        
        TWO PATHS TO BUILDING:
        1. Classic: Score > 55, VC > 50%, VS > 60%
        2. Accelerated: VC > 90% + OI growth > 15% (skip VS requirement)
        """
        vc_extreme = False
        
        # === TRIGGERED ===
        # Volume spike + OI jump (breakout happening)
        if volume_spike and oi_jump:
            return (SleeperState.TRIGGERED.value, vc_extreme)
        
        # === VC-EXTREME FLAG ===
        # Extreme compression with low volume = imminent breakout
        if compression_pct >= 95 and volume_ratio <= 1.2:
            vc_extreme = True
        
        # === READY ===
        # High compression, ready for breakout
        if (total_score >= self.MIN_SCORE_READY and 
            compression_pct >= self.COMPRESSION_READY):
            return (SleeperState.READY.value, vc_extreme)
        
        # === ACCELERATED PATH TO READY ===
        # VC > 95% + OI growth > 20% = big money entering during extreme squeeze
        if compression_pct >= 95 and oi_growth_pct >= 20:
            return (SleeperState.READY.value, True)
        
        # === BUILDING (Classic Path) ===
        # Good compression + volume suppressed
        if (total_score >= self.MIN_SCORE_BUILDING and 
            compression_pct >= self.COMPRESSION_BUILDING and
            suppression_pct >= self.VOLUME_SUPPRESSION_MIN):
            return (SleeperState.BUILDING.value, vc_extreme)
        
        # === BUILDING (Accelerated Path) ===
        # VC > 90% + OI growth > 15% = big money accumulating
        # Skip VS requirement - volume hasn't dropped but smart money is entering
        if compression_pct >= 90 and oi_growth_pct >= 15:
            return (SleeperState.BUILDING.value, True)
        
        # === BUILDING (VC-Priority Path) ===
        # VC > 90% + OI score > 70 = extreme squeeze with strong OI signal
        if compression_pct >= 90 and oi_score >= 70:
            return (SleeperState.BUILDING.value, True)
        
        # === WATCHING ===
        # Basic conditions met
        if total_score >= self.MIN_SCORE_WATCHING:
            return (SleeperState.WATCHING.value, vc_extreme)
        
        # === WATCHING (VC-Extreme Override) ===
        # VC > 95% alone is worth watching
        if compression_pct >= 95:
            return (SleeperState.WATCHING.value, True)
        
        return (SleeperState.IDLE.value, vc_extreme)
    
    def _determine_direction(self, funding_rate: float, ob_imbalance: float,
                             rsi_4h: float, rsi_1d: float) -> str:
        """
        Determine likely direction based on multiple signals
        """
        long_signals = 0
        short_signals = 0
        
        # Funding rate (contrarian)
        if funding_rate > 0.01:  # High positive = overleveraged longs
            short_signals += 1
        elif funding_rate < -0.01:  # Negative = overleveraged shorts
            long_signals += 1
        
        # Order book imbalance
        if ob_imbalance > 15:
            long_signals += 1
        elif ob_imbalance < -15:
            short_signals += 1
        
        # RSI 4H
        if rsi_4h < 35:
            long_signals += 1
        elif rsi_4h > 65:
            short_signals += 1
        
        # RSI 1D
        if rsi_1d < 40:
            long_signals += 1
        elif rsi_1d > 60:
            short_signals += 1
        
        if long_signals > short_signals:
            return 'LONG'
        elif short_signals > long_signals:
            return 'SHORT'
        else:
            return 'NEUTRAL'
    
    def _update_hp_scores(self, new_candidates: List[Dict]):
        """
        Update HP for existing sleepers based on score changes
        
        HP Changes:
        - Score improved significantly (+5): HP +2
        - Score improved slightly (+1): HP +1
        - Score stable: HP unchanged
        - Score dropped slightly (-1): HP -1
        - Score dropped significantly (-5): HP -2
        - Not in new scan: HP -2
        - State upgraded: HP +1
        - State downgraded: HP -1
        """
        existing = self.db.get_sleepers(limit=500)
        new_map = {c['symbol']: c for c in new_candidates}
        
        state_rank = {
            'TRIGGERED': 5,
            'READY': 4,
            'BUILDING': 3,
            'WATCHING': 2,
            'IDLE': 1
        }
        
        for old in existing:
            symbol = old['symbol']
            old_score = old.get('total_score', 0) or 0
            old_state = old.get('state', 'IDLE')
            
            if symbol not in new_map:
                # Not in new scan - decrease HP significantly
                self.db.update_sleeper_state(symbol, old_state, hp_change=-2)
            else:
                new = new_map[symbol]
                new_score = new.get('total_score', 0) or 0
                new_state = new.get('state', 'IDLE')
                
                hp_change = 0
                
                # Score change
                score_diff = new_score - old_score
                if score_diff >= 5:
                    hp_change += 2
                elif score_diff >= 1:
                    hp_change += 1
                elif score_diff <= -5:
                    hp_change -= 2
                elif score_diff <= -1:
                    hp_change -= 1
                
                # State change
                old_rank = state_rank.get(old_state, 1)
                new_rank = state_rank.get(new_state, 1)
                
                if new_rank > old_rank:
                    hp_change += 1  # State upgraded
                elif new_rank < old_rank:
                    hp_change -= 1  # State downgraded
                
                if hp_change != 0:
                    self.db.update_sleeper_state(symbol, new_state, hp_change=hp_change)
    
    def _get_state_emoji(self, state: str) -> str:
        """Get emoji for state"""
        return {
            'IDLE': '⚪',
            'WATCHING': '👀',
            'BUILDING': '🔨',
            'READY': '🎯',
            'TRIGGERED': '🚀',
        }.get(state, '❓')


# Singleton instance
_scanner_v3_instance: Optional[SleeperScannerV3] = None

def get_sleeper_scanner_v3() -> SleeperScannerV3:
    """Get sleeper scanner v3 instance"""
    global _scanner_v3_instance
    if _scanner_v3_instance is None:
        _scanner_v3_instance = SleeperScannerV3()
    return _scanner_v3_instance
