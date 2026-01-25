"""
Sleeper Scanner v4.1 - Professional 5-Day Strategy with Direction Engine

MAJOR FEATURES:
- v4.0: ADX filter, POC analysis, BTC correlation, Liquidity filter
- v4.1: Professional 3-Layer Direction Engine
  - HTF Structural Bias (50%): 1D structure + 4H EMA slope
  - LTF Momentum Shift (30%): RSI divergence + BB position
  - Derivatives Positioning (20%): OI + Funding + Price action

PROFESSIONAL PARADIGM:
1. Sleeper Detector → ЧИ є сенс торгувати (this module)
2. Direction Engine → В ЯКИЙ БІК (integrated)
3. Trigger Engine   → КОЛИ входити (future)

Direction = HTF Bias (50%) + LTF Momentum (30%) + Derivatives (20%)
NEUTRAL = валідний стан (не торгуємо)

CRITICAL FIXES:
- v3.2: BB width zero-filter bug
- v3.3: Rate limit 500ms + compression clamp

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
from detection.direction_engine import get_direction_engine, DirectionResult


class SleeperScannerV3:
    """
    5-Day Sleeper Detection Strategy
    
    Шукає монети що готуються до breakout:
    1. Волатильність стискається (BB squeeze)
    2. Об'єми знижуються (накопичення)
    3. OI росте (позиції накопичуються)
    4. Стакан показує напрямок (дисбаланс)
    """
    
    # === КОНФІГУРАЦІЯ v4 ===
    
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
    
    # === v4: PROTECTION FILTERS ===
    
    # Liquidity Filter (slippage protection)
    MIN_24H_VOLUME_USD = 50_000_000  # $50M minimum for safe entry/exit
    MIN_ORDERBOOK_DEPTH = 100_000    # $100K minimum in top 25 levels
    
    # Sleeper Timeout (prevents "eternal sleep")
    MAX_WATCHING_HOURS = 72          # 3 days max in WATCHING
    MAX_BUILDING_HOURS = 48          # 2 days max in BUILDING
    
    # POC (Point of Control) Filter
    POC_PROXIMITY_PCT = 2.0          # Price within 2% of POC = bonus
    POC_STRENGTH_MIN = 5.0           # Minimum POC strength for bonus
    
    def __init__(self):
        self.fetcher = get_fetcher()
        self.indicators = get_indicators()
        self.db = get_db()
        self.direction_engine = get_direction_engine()  # v4.1: Professional direction model
        
        # Scan settings from DB (reduced defaults for Binance rate limit protection)
        self.max_symbols = min(self.db.get_setting('sleeper_max_symbols', 30), 50)  # Max 50
        self.min_volume = self.db.get_setting('sleeper_min_volume', 75_000_000)  # Increased to 75M
        self.scan_interval = self.db.get_setting('sleeper_scan_interval', 240)  # 4H
        
        # Batch processing settings
        self.batch_size = 5   # Process 5 symbols at a time
        self.batch_delay = 3  # 3 seconds between batches
    
    def run_scan(self) -> List[Dict]:
        """
        Run full sleeper scan with rate limiting
        Returns list of candidates
        
        v4 features:
        - ADX filter (trendless markets get bonus)
        - BTC correlation check (warn if BTC volatile)
        - Batch processing with delays
        """
        print(f"\n[SLEEPER v4] Starting 5-day strategy scan...")
        start_time = time.time()
        
        # === BTC CORRELATION CHECK (v4) ===
        btc_warning = self._check_btc_volatility()
        if btc_warning:
            print(f"[SLEEPER v4] ⚠️ {btc_warning}")
        
        # 1. Get top symbols by volume (limited to prevent API abuse)
        symbols = self.fetcher.get_top_symbols(
            limit=self.max_symbols,
            min_volume=self.min_volume
        )
        
        if not symbols:
            print("[SLEEPER v4] No symbols found")
            return []
        
        print(f"[SLEEPER v4] Analyzing {len(symbols)} symbols (batch_size={self.batch_size}, delay={self.batch_delay}s)...")
        
        candidates = []
        errors = 0
        vc_extreme_count = 0
        trendless_count = 0
        poc_count = 0
        
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
                        
                        # ADX trendless flag (v4)
                        adx_flag = "📊" if result.get('adx_trendless') else ""
                        if result.get('adx_trendless'):
                            trendless_count += 1
                        
                        # POC flag (v4)
                        poc_flag = "🎯" if result.get('price_at_poc') else ""
                        if result.get('price_at_poc'):
                            poc_count += 1
                        
                        # Direction confidence flag (v4.1)
                        dir_confidence = result.get('direction_confidence', 'LOW')
                        dir_flag = "💪" if dir_confidence == "HIGH" else ""
                        
                        # Compact log with bonuses
                        bonuses = []
                        if result.get('adx_bonus', 0) > 0:
                            bonuses.append(f"+{result['adx_bonus']}ADX")
                        if result.get('poc_bonus', 0) > 0:
                            bonuses.append(f"+{result['poc_bonus']}POC")
                        bonus_str = f" ({','.join(bonuses)})" if bonuses else ""
                        
                        # Direction with confidence
                        direction = result['direction']
                        dir_score = result.get('direction_score', 0)
                        dir_str = f"{direction}[{dir_score:+.2f}]" if direction != "NEUTRAL" else "WAIT"
                        
                        print(f"[SLEEPER v4.1] {state_emoji}{vc_flag}{adx_flag}{poc_flag}{dir_flag} {result['symbol']}: "
                              f"Score={result['total_score']:.1f}{bonus_str} "
                              f"Dir={dir_str} "
                              f"(VC:{result['volatility_compression']:.0f} "
                              f"VS:{result['volume_suppression']:.0f} "
                              f"OI:{result['oi_growth']:.0f} "
                              f"OB:{result['order_book_imbalance']:.0f}) "
                              f"BB:{result.get('bb_compression_pct', 0):.1f}%")
                
                # Progress every 10 symbols
                if (i + 1) % 10 == 0:
                    print(f"[SLEEPER v4] Progress: {i+1}/{len(symbols)} ({len(candidates)} candidates)")
                
                # Rate limiting between individual symbols
                time.sleep(API_LIMITS.get('rate_limit_delay', 0.5))
                
                # Batch delay - longer pause every batch_size symbols
                if (i + 1) % self.batch_size == 0:
                    print(f"[SLEEPER v4] Batch complete, waiting {self.batch_delay}s...")
                    time.sleep(self.batch_delay)
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"[SLEEPER v4] Error analyzing {sym_data.get('symbol')}: {e}")
        
        # Update HP for existing sleepers
        self._update_hp_scores(candidates)
        
        # Remove dead sleepers
        removed = self.db.remove_dead_sleepers()
        if removed > 0:
            print(f"[SLEEPER v4] Removed {removed} dead sleepers (HP=0)")
        
        elapsed = time.time() - start_time
        print(f"\n[SLEEPER v4] Scan complete in {elapsed:.1f}s")
        print(f"[SLEEPER v4] Results: {len(candidates)} candidates from {len(symbols)} symbols")
        
        # Count by state
        by_state = {}
        for c in candidates:
            state = c.get('state', 'UNKNOWN')
            by_state[state] = by_state.get(state, 0) + 1
        print(f"[SLEEPER v4] States: {by_state}")
        
        # v4 Summary
        special_flags = []
        if vc_extreme_count > 0:
            special_flags.append(f"🔥VC-EXTREME:{vc_extreme_count}")
        if trendless_count > 0:
            special_flags.append(f"📊TRENDLESS:{trendless_count}")
        if poc_count > 0:
            special_flags.append(f"🎯POC:{poc_count}")
        
        if special_flags:
            print(f"[SLEEPER v4] Special: {' | '.join(special_flags)}")
        
        return candidates
    
    def _check_btc_volatility(self) -> Optional[str]:
        """
        Check BTC volatility to assess market conditions.
        If BTC is volatile (ADX > 30), altcoin sleeper signals may be unreliable.
        
        Returns warning message or None
        """
        try:
            # Get BTC 4h klines
            btc_klines = self.fetcher.get_klines('BTCUSDT', '4h', limit=30)
            if len(btc_klines) < 20:
                return None
            
            # Calculate BTC indicators
            btc_indicators = self.indicators.calculate_all(btc_klines)
            
            # Check BTC ADX
            btc_adx = btc_indicators.get('adx', {})
            btc_adx_value = btc_adx.get('adx', 25)
            
            # Check BTC BB width (volatility)
            btc_bb = btc_indicators.get('bb', {})
            btc_bb_widths = [w for w in btc_bb.get('width', []) if w > 0]
            btc_volatility = btc_bb_widths[-1] if btc_bb_widths else 0
            
            warnings = []
            
            # BTC trending strongly
            if btc_adx_value > 30:
                warnings.append(f"BTC ADX={btc_adx_value:.0f} (trending)")
            
            # BTC volatile
            if btc_volatility > 5:  # BB width > 5% = high volatility
                warnings.append(f"BTC Vol={btc_volatility:.1f}% (high)")
            
            if warnings:
                return f"Market warning: {', '.join(warnings)} - sleeper signals may be less reliable"
            
            return None
            
        except Exception as e:
            print(f"[SLEEPER v4] BTC check error: {e}")
            return None
    
    def _analyze_symbol_5day(self, symbol_data: Dict) -> Optional[Dict]:
        """
        Analyze symbol using 5-day strategy (v4)
        
        v4 enhancements:
        - ADX filter (trendless markets)
        - POC (Point of Control) bonus
        - Liquidity filter (slippage protection)
        """
        symbol = symbol_data['symbol']
        volume_24h = symbol_data.get('volume_24h', 0)
        
        # === 0. LIQUIDITY FILTER (v4) ===
        # Skip illiquid symbols to avoid slippage
        if volume_24h < self.MIN_24H_VOLUME_USD:
            return None
        
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
        
        # === 2.5 ADX FILTER (v4) ===
        # ADX < 20 = no trend = perfect for sleeper
        # ADX < 20 AND falling = even better
        adx_data = indicators_4h.get('adx', {})
        adx_value = adx_data.get('adx', 25)
        is_trendless = adx_data.get('is_trendless', False)
        adx_falling = adx_data.get('adx_falling', False)
        
        # ADX bonus for true sleepers
        adx_bonus = 0
        if is_trendless:  # ADX < 20
            adx_bonus = 10  # +10 to score
            if adx_falling:  # AND falling
                adx_bonus = 15  # +15 to score
        elif adx_value < 25:  # Weak trend
            adx_bonus = 5
        
        # === 2.6 POC (Point of Control) ANALYSIS (v4) ===
        poc_data = self.indicators.volume_profile_poc(klines_4h)
        poc_bonus = 0
        price_at_poc = poc_data.get('price_at_poc', False)
        poc_strength = poc_data.get('poc_strength', 0)
        poc_distance = poc_data.get('poc_distance_pct', 100)
        
        # Bonus if price is sleeping at POC (equilibrium zone)
        if poc_distance < self.POC_PROXIMITY_PCT and poc_strength >= self.POC_STRENGTH_MIN:
            poc_bonus = 8  # +8 to score (significant equilibrium)
            if poc_strength >= 10:  # Very strong POC
                poc_bonus = 12
        elif poc_distance < 3.0:  # Within 3% of POC
            poc_bonus = 4
        
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
        
        base_score = (
            vc_data['score'] * (self.WEIGHTS['volatility_compression'] / 100) +
            vs_data['score'] * (self.WEIGHTS['volume_suppression'] / 100) +
            oi_data['score'] * (self.WEIGHTS['oi_growth'] / 100) +
            ob_data['score'] * (self.WEIGHTS['order_book_imbalance'] / 100)
        )
        
        # Apply ADX bonus + POC bonus (capped at 100)
        total_score = min(100, base_score + adx_bonus + poc_bonus)
        
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
        
        # === 8. DETERMINE DIRECTION (v4.1: Professional Direction Engine) ===
        
        # Calculate price change for derivatives bias
        price_change_4h = 0
        if len(klines_4h) >= 2:
            price_change_4h = (klines_4h[-1]['close'] - klines_4h[-2]['close']) / klines_4h[-2]['close'] * 100
        
        # Use existing klines (no extra API calls!)
        # klines_4h already has 30 candles, klines_1d has 7 - enough for direction
        direction_result = self.direction_engine.resolve(
            symbol=symbol,
            klines_4h=klines_4h,      # Reuse existing data
            klines_1d=klines_1d,      # Reuse existing data  
            oi_change=oi_data['growth_pct'],
            funding_rate=funding_rate,
            price_change_4h=price_change_4h
        )
        
        direction = direction_result.direction.value  # LONG, SHORT, or NEUTRAL
        direction_score = direction_result.score
        direction_confidence = direction_result.confidence
        
        # === 9. CALCULATE DYNAMIC HP ===
        hp = self._calculate_initial_hp(total_score, state)
        
        # Boost HP for VC-extreme candidates
        if vc_extreme:
            hp = min(self.HP_MAX, hp + 2)
        
        # Boost HP for trendless markets (v4: ADX bonus)
        if is_trendless:
            hp = min(self.HP_MAX, hp + 1)
        
        # Boost HP for POC equilibrium (v4)
        if price_at_poc and poc_strength >= self.POC_STRENGTH_MIN:
            hp = min(self.HP_MAX, hp + 1)
        
        # Boost HP for high direction confidence (v4.1)
        if direction_confidence == "HIGH" and direction != "NEUTRAL":
            hp = min(self.HP_MAX, hp + 1)
        
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
            
            # v4.1: Direction Engine data
            'direction_score': round(direction_score, 3),
            'direction_confidence': direction_confidence,
            'direction_htf_bias': direction_result.htf_bias,
            'direction_ltf_bias': direction_result.ltf_bias,
            'direction_deriv_bias': direction_result.deriv_bias,
            'direction_reason': f"HTF:{direction_result.htf_reason[:30]}... | Deriv:{direction_result.deriv_reason[:30]}...",
            
            # v4: ADX data
            'adx_value': round(adx_value, 1),
            'adx_trendless': is_trendless,
            'adx_bonus': adx_bonus,
            
            # v4: POC data
            'poc_price': round(poc_data.get('poc_price', 0), 6),
            'poc_distance_pct': round(poc_distance, 2),
            'poc_strength': round(poc_strength, 1),
            'price_at_poc': price_at_poc,
            'poc_bonus': poc_bonus,
            
            # v4: Liquidity data
            'volume_24h_usd': volume_24h,
            
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
                    # Clamp to 0-100% (negative = expanding, show as 0)
                    compression_pct = max(0, min(100, compression_pct))
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
            # Clamp to 0-100% range for display
            # Negative = volatility expanding (not compression) → show as 0
            compression_pct = max(0, min(100, compression_pct))
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
