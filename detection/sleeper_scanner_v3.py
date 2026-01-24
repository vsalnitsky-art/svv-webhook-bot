"""
Sleeper Scanner v3.0 - 5-Day Strategy Implementation

Система оцінки:
- VOLATILITY_COMPRESSION: 40% - стиснення волатильності за 5 днів
- VOLUME_SUPPRESSION: 25% - пригнічення об'ємів
- OI_GROWTH: 20% - зростання Open Interest
- ORDER_BOOK_IMBALANCE: 15% - дисбаланс стакану

Стани:
- IDLE: Не відстежується
- WATCHING: Score > 40, базові умови виконані
- BUILDING: Score > 55, compression > 50%, volume < 40%
- READY: Score > 65, compression > 70%, volume spike imminent
- TRIGGERED: Volume > 200% + OI jump > 15%
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
        
        for i, sym_data in enumerate(symbols):
            try:
                result = self._analyze_symbol_5day(sym_data)
                
                if result:
                    # Save to database
                    saved = self.db.upsert_sleeper(result)
                    if saved:
                        candidates.append(result)
                        state_emoji = self._get_state_emoji(result['state'])
                        print(f"[SLEEPER v3] {state_emoji} {result['symbol']}: "
                              f"Score={result['total_score']:.1f} "
                              f"Dir={result['direction']} "
                              f"(VC:{result['volatility_compression']:.0f} "
                              f"VS:{result['volume_suppression']:.0f} "
                              f"OI:{result['oi_growth']:.0f} "
                              f"OB:{result['order_book_imbalance']:.0f})")
                
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
        print(f"\n[SLEEPER v3] Scan complete in {elapsed:.1f}s")
        print(f"[SLEEPER v3] Results: {len(candidates)} candidates from {len(symbols)} symbols")
        
        # Count by state
        by_state = {}
        for c in candidates:
            state = c.get('state', 'UNKNOWN')
            by_state[state] = by_state.get(state, 0) + 1
        print(f"[SLEEPER v3] States: {by_state}")
        
        return candidates
    
    def _analyze_symbol_5day(self, symbol_data: Dict) -> Optional[Dict]:
        """
        Analyze symbol using 5-day strategy
        """
        symbol = symbol_data['symbol']
        
        # === 1. GET 5-DAY DATA ===
        
        # 4H klines for 5 days (30 candles)
        klines_4h = self.fetcher.get_klines(symbol, '240', limit=self.KLINES_5D)
        if len(klines_4h) < 20:
            return None
        
        # Daily klines for context
        klines_1d = self.fetcher.get_klines(symbol, 'D', limit=self.KLINES_1D)
        
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
        
        # === 6. DETERMINE STATE ===
        
        state = self._determine_state(
            total_score,
            vc_data['compression_pct'],
            vs_data['suppression_pct'],
            vs_data['volume_spike'],
            oi_data['oi_jump']
        )
        
        # Minimum score filter
        if state == 'IDLE' and total_score < self.MIN_SCORE_WATCHING:
            return None
        
        # === 7. DETERMINE DIRECTION ===
        
        direction = self._determine_direction(
            funding_rate,
            ob_data['imbalance'],
            indicators_4h.get('rsi_current', 50),
            indicators_1d.get('rsi_current', 50) if indicators_1d else 50
        )
        
        # === 8. BUILD RESULT ===
        
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
            'hp': self.HP_INITIAL,
            'direction': direction,
            
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
    
    def _calculate_volatility_compression(self, klines: List[Dict], indicators: Dict) -> Dict:
        """
        Calculate BB width compression over 5 days
        Returns score 0-100 based on compression amount
        """
        bb_widths = indicators.get('bb', {}).get('width', [])
        
        if not bb_widths or len(bb_widths) < 20:
            return {
                'score': 50,
                'bb_width_start': 0,
                'bb_width_current': 0,
                'compression_pct': 0,
            }
        
        # Get BB width at start (5 days ago) and current
        bb_width_start = sum(bb_widths[:5]) / 5 if len(bb_widths) >= 5 else bb_widths[0]
        bb_width_current = sum(bb_widths[-3:]) / 3  # Average of last 3
        
        # Calculate compression percentage
        if bb_width_start > 0:
            compression_pct = ((bb_width_start - bb_width_current) / bb_width_start) * 100
        else:
            compression_pct = 0
        
        # Score based on compression
        # More compression = higher score
        if compression_pct >= 70:
            score = 100
        elif compression_pct >= 60:
            score = 90
        elif compression_pct >= 50:
            score = 80
        elif compression_pct >= 40:
            score = 70
        elif compression_pct >= 30:
            score = 60
        elif compression_pct >= 20:
            score = 50
        elif compression_pct >= 10:
            score = 40
        elif compression_pct > 0:
            score = 30
        else:
            # Expanding volatility - low score
            score = 20
        
        return {
            'score': score,
            'bb_width_start': round(bb_width_start, 4),
            'bb_width_current': round(bb_width_current, 4),
            'compression_pct': round(compression_pct, 2),
        }
    
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
                         suppression_pct: float, volume_spike: bool, oi_jump: bool) -> str:
        """
        Determine sleeper state based on metrics
        """
        # TRIGGERED: Volume spike + OI jump (breakout happening)
        if volume_spike and oi_jump:
            return SleeperState.TRIGGERED.value
        
        # READY: High compression, ready for breakout
        if (total_score >= self.MIN_SCORE_READY and 
            compression_pct >= self.COMPRESSION_READY):
            return SleeperState.READY.value
        
        # BUILDING: Good compression, volume suppressed
        if (total_score >= self.MIN_SCORE_BUILDING and 
            compression_pct >= self.COMPRESSION_BUILDING and
            suppression_pct >= self.VOLUME_SUPPRESSION_MIN):
            return SleeperState.BUILDING.value
        
        # WATCHING: Basic conditions met
        if total_score >= self.MIN_SCORE_WATCHING:
            return SleeperState.WATCHING.value
        
        return SleeperState.IDLE.value
    
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
        """
        existing = self.db.get_sleepers(limit=500)
        new_map = {c['symbol']: c for c in new_candidates}
        
        for old in existing:
            symbol = old['symbol']
            if symbol not in new_map:
                # Not in new scan - decrease HP
                self.db.update_sleeper_state(symbol, old['state'], hp_change=-1)
            else:
                new = new_map[symbol]
                # Compare scores
                if new['total_score'] > old.get('total_score', 0):
                    # Score improved - increase HP
                    self.db.update_sleeper_state(symbol, new['state'], hp_change=1)
    
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
