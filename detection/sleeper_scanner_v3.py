"""
Sleeper Scanner v8.0 - SMC (Smart Money Concepts) Strategy

ВЕРСІЯ v8.0 КЛЮЧОВІ ЗМІНИ:
- Direction Engine v8 з SMC інтеграцією
- BOS/CHoCH detection (Break of Structure / Change of Character)
- Order Blocks detection (Bullish/Bearish zones)
- Premium/Discount Zone analysis
- Multi-timeframe структура (4H/1H)

SMC КОМПОНЕНТИ:
1. Market Structure - HH/HL/LH/LL detection
2. BOS/CHoCH Signals - структурні зламі
3. Order Blocks - інституційні зони
4. Premium/Discount - зони відносно свінгу

DIRECTION ENGINE v8 ВАГИ:
- SMC Score: 40% (головний компонент)
- Structure: 20%
- Momentum: 20%
- Derivatives: 20%

PATHS TO READY (v8):
1. CHoCH + Discount Zone → найсильніший сигнал LONG
2. CHoCH + Premium Zone → найсильніший сигнал SHORT
3. BOS + Order Block → підтвердження напрямку
4. Classic: Score > 65, Direction confidence HIGH

DATA REQUIREMENTS:
- HTF (4H): 100 candles - for global bias & SMC HTF
- LTF (1H): 100 candles - for signals & CHoCH detection
- Cache TTL: 3 minutes (reduced API calls)
- Batch delay: 5 seconds (rate limit protection)
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

# v8: SMC Integration
from detection.direction_engine_v8 import (
    get_direction_engine_v8, DirectionResultV8, BiasDirection
)
from detection.smc_analyzer import StructureSignal, MarketBias, PriceZone

# v8.2.5: Trend Filter Integration
from detection.trend_analyzer import TrendAnalyzer, TrendRegime


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
    # === v8.2.5: Score Weights та Thresholds читаються з UI налаштувань ===
    # Дивись __init__ метод
    # WEIGHTS читаються з: weight_fuel, weight_volatility, weight_price, weight_liquidity
    # MIN_SCORE_* читаються з: sleeper_min_score, sleeper_building_score, sleeper_ready_score
    
    # Data requirements (константи)
    KLINES_5D = 60   # 60 x 4H = 10 days (enough for EMA20 + direction analysis)
    KLINES_1D = 60   # 60 days for EMA50 + structural analysis
    OI_PERIODS = 30  # 30 hours of OI data
    
    # Transition conditions (v4.1.2: relaxed for earlier signals)
    COMPRESSION_BUILDING = 35   # % BB compression for BUILDING (was 50)
    COMPRESSION_READY = 50      # % BB compression for READY (was 70)
    VOLUME_SUPPRESSION_MIN = 40 # % volume suppression for BUILDING (was 60)
    
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
        
        # Get trading style from settings
        self.trading_style = self.db.get_setting('trading_mode', 'SWING')
        
        # v8: SMC-integrated Direction Engine
        self.direction_engine = get_direction_engine_v8()
        
        # Scan settings from DB (reduced defaults for Binance rate limit protection)
        self.max_symbols = min(self.db.get_setting('sleeper_max_symbols', 30), 50)  # Max 50
        self.min_volume = self.db.get_setting('sleeper_min_volume', 75_000_000)  # Increased to 75M
        self.scan_interval = self.db.get_setting('sleeper_scan_interval', 240)  # 4H
        
        # === v8.2.5: ЧИТАЄМО НАЛАШТУВАННЯ З UI ===
        
        # Score Thresholds (Settings → Sleeper Detection → Score Thresholds)
        self.MIN_SCORE_WATCHING = int(self.db.get_setting('sleeper_min_score', 40))
        self.MIN_SCORE_BUILDING = int(self.db.get_setting('sleeper_building_score', 50))
        self.MIN_SCORE_READY = int(self.db.get_setting('sleeper_ready_score', 60))
        
        # Score Weights (Settings → Sleeper Detection → Score Weights)
        # UI: Fuel=OI Growth, Volatility=BB Squeeze, Price=Range Position, Liquidity=Volume Profile
        weight_fuel = int(self.db.get_setting('weight_fuel', 30))
        weight_volatility = int(self.db.get_setting('weight_volatility', 25))
        weight_price = int(self.db.get_setting('weight_price', 25))
        weight_liquidity = int(self.db.get_setting('weight_liquidity', 20))
        
        # Normalize weights to 100% if needed
        total_weight = weight_fuel + weight_volatility + weight_price + weight_liquidity
        if total_weight != 100 and total_weight > 0:
            weight_fuel = round(weight_fuel * 100 / total_weight)
            weight_volatility = round(weight_volatility * 100 / total_weight)
            weight_price = round(weight_price * 100 / total_weight)
            weight_liquidity = 100 - weight_fuel - weight_volatility - weight_price
        
        self.WEIGHTS = {
            'oi_growth': weight_fuel,                    # Fuel (%) = OI Growth
            'volatility_compression': weight_volatility,  # Volatility (%) = BB Squeeze
            'volume_suppression': weight_price,           # Price (%) = Range Position
            'order_book_imbalance': weight_liquidity,     # Liquidity (%) = Volume Profile
        }
        
        # Trend Context Filter (Settings → Trend Context)
        self.use_trend_filter = self.db.get_setting('use_trend_filter', True)
        if isinstance(self.use_trend_filter, str):
            self.use_trend_filter = self.use_trend_filter.lower() in ('1', 'true', 'yes')
        self.min_trend_score = int(self.db.get_setting('min_trend_score', 65))
        
        # Log loaded settings
        print(f"[SLEEPER v8.2.7] Initialized with UI settings")
        print(f"  Trend Filter: {'ON' if self.use_trend_filter else 'OFF'} (min={self.min_trend_score})")
        
        # v8.2.5: Trend Analyzer для 4H Context Filter
        self.trend_analyzer = TrendAnalyzer()
        self.trend_timeframe = self.db.get_setting('trend_timeframe', '240')  # 4H default
        self.allow_signals_without_trend = self.db.get_setting('allow_signals_without_trend', False)
        if isinstance(self.allow_signals_without_trend, str):
            self.allow_signals_without_trend = self.allow_signals_without_trend.lower() in ('1', 'true', 'yes')
        
        # v8.2.7: Primary Timeframe for sleeper analysis
        self.htf_timeframe = self.db.get_setting('sleeper_timeframe', '240')
        if self.htf_timeframe == '240':
            self.htf_timeframe_api = '4h'
        elif self.htf_timeframe == '60':
            self.htf_timeframe_api = '1h'
        elif self.htf_timeframe == 'D':
            self.htf_timeframe_api = '1d'
        else:
            self.htf_timeframe_api = '4h'
        
        # Batch processing settings (v4.1: increased delay for more data)
        self.batch_size = 5   # Process 5 symbols at a time
        self.batch_delay = 5  # 5 seconds between batches (was 3)
    
    def _reload_settings(self):
        """
        v8.2.5: Перечитуємо налаштування з UI перед кожним сканом.
        Це дозволяє змінювати налаштування без перезапуску бота.
        """
        # Score Thresholds
        self.MIN_SCORE_WATCHING = int(self.db.get_setting('sleeper_min_score', 40))
        self.MIN_SCORE_BUILDING = int(self.db.get_setting('sleeper_building_score', 50))
        self.MIN_SCORE_READY = int(self.db.get_setting('sleeper_ready_score', 60))
        
        # Score Weights
        weight_fuel = int(self.db.get_setting('weight_fuel', 30))
        weight_volatility = int(self.db.get_setting('weight_volatility', 25))
        weight_price = int(self.db.get_setting('weight_price', 25))
        weight_liquidity = int(self.db.get_setting('weight_liquidity', 20))
        
        # Normalize to 100%
        total_weight = weight_fuel + weight_volatility + weight_price + weight_liquidity
        if total_weight != 100 and total_weight > 0:
            weight_fuel = round(weight_fuel * 100 / total_weight)
            weight_volatility = round(weight_volatility * 100 / total_weight)
            weight_price = round(weight_price * 100 / total_weight)
            weight_liquidity = 100 - weight_fuel - weight_volatility - weight_price
        
        self.WEIGHTS = {
            'oi_growth': weight_fuel,
            'volatility_compression': weight_volatility,
            'volume_suppression': weight_price,
            'order_book_imbalance': weight_liquidity,
        }
        
        # Trend Filter
        self.use_trend_filter = self.db.get_setting('use_trend_filter', True)
        if isinstance(self.use_trend_filter, str):
            self.use_trend_filter = self.use_trend_filter.lower() in ('1', 'true', 'yes')
        self.min_trend_score = int(self.db.get_setting('min_trend_score', 65))
        self.allow_signals_without_trend = self.db.get_setting('allow_signals_without_trend', False)
        if isinstance(self.allow_signals_without_trend, str):
            self.allow_signals_without_trend = self.allow_signals_without_trend.lower() in ('1', 'true', 'yes')
        self.trend_timeframe = self.db.get_setting('trend_timeframe', '240')  # v8.2.7: Added to reload
        
        # Scan parameters
        self.max_symbols = min(int(self.db.get_setting('sleeper_max_symbols', 30)), 50)
        self.min_volume = int(self.db.get_setting('sleeper_min_volume', 75_000_000))
        
        # v8.2.7: Primary Timeframe for sleeper analysis
        self.htf_timeframe = self.db.get_setting('sleeper_timeframe', '240')  # 4H default
        # Convert to API format
        if self.htf_timeframe == '240':
            self.htf_timeframe_api = '4h'
        elif self.htf_timeframe == '60':
            self.htf_timeframe_api = '1h'
        elif self.htf_timeframe == 'D':
            self.htf_timeframe_api = '1d'
        else:
            self.htf_timeframe_api = '4h'
        
        # v8.2.6: Оновлюємо Direction Engine з UI налаштувань
        self.direction_engine.reload_from_db(self.db)
    
    def run_scan(self) -> List[Dict]:
        """
        Run full sleeper scan with rate limiting
        Returns list of candidates
        
        v4 features:
        - ADX filter (trendless markets get bonus)
        - BTC correlation check (warn if BTC volatile)
        - Batch processing with delays
        """
        # === v8.2.5: Перечитуємо налаштування з UI перед кожним сканом ===
        self._reload_settings()
        
        # v8.2.7: Логуємо поточні налаштування
        print(f"\n[SLEEPER v8.2.7] Starting SMC strategy scan ({self.htf_timeframe_api.upper()}+LTF)...")
        print(f"  Score: WATCH≥{self.MIN_SCORE_WATCHING}, BUILD≥{self.MIN_SCORE_BUILDING}, READY≥{self.MIN_SCORE_READY}")
        print(f"  Weights: Fuel={self.WEIGHTS['oi_growth']}%, Vol={self.WEIGHTS['volatility_compression']}%, Price={self.WEIGHTS['volume_suppression']}%, Liq={self.WEIGHTS['order_book_imbalance']}%")
        print(f"  Direction: SMC={int(self.direction_engine.WEIGHT_SMC*100)}%, Struct={int(self.direction_engine.WEIGHT_STRUCTURE*100)}%, Mom={int(self.direction_engine.WEIGHT_MOMENTUM*100)}%, Deriv={int(self.direction_engine.WEIGHT_DERIVATIVES*100)}%")
        print(f"  Bias: LONG≥{self.direction_engine.BIAS_THRESHOLD_LONG}, SHORT≤{self.direction_engine.BIAS_THRESHOLD_SHORT}")
        print(f"  Trend Filter: {'ON' if self.use_trend_filter else 'OFF'} (min={self.min_trend_score})")
        start_time = time.time()
        
        # === BTC CORRELATION CHECK (v4) ===
        btc_warning = self._check_btc_volatility()
        if btc_warning:
            print(f"[SLEEPER v8.2] ⚠️ {btc_warning}")
        
        # 1. Get top symbols by volume (limited to prevent API abuse)
        symbols = self.fetcher.get_top_symbols(
            limit=self.max_symbols * 2,  # v8.2: Get more to account for filtering
            min_volume=self.min_volume
        )
        
        if not symbols:
            print("[SLEEPER v8.2] No symbols found")
            return []
        
        # === v8.2: BLACKLIST + VOLATILITY FILTER ===
        blacklist = self.db.get_blacklist()
        min_volatility = self.db.get_setting('min_volatility_pct', 3.0)
        
        filtered_symbols = []
        blacklisted_count = 0
        low_vol_count = 0
        
        for sym_data in symbols:
            symbol = sym_data['symbol'] if isinstance(sym_data, dict) else sym_data
            
            # 1. Check blacklist
            if symbol in blacklist:
                blacklisted_count += 1
                continue
            
            # 2. Check 24h volatility (price range %)
            try:
                if isinstance(sym_data, dict):
                    high_24h = float(sym_data.get('high_24h', 0) or sym_data.get('highPrice24h', 0))
                    low_24h = float(sym_data.get('low_24h', 0) or sym_data.get('lowPrice24h', 0))
                    if low_24h > 0:
                        volatility_24h = ((high_24h - low_24h) / low_24h) * 100
                        if volatility_24h < min_volatility:
                            low_vol_count += 1
                            continue
            except (KeyError, TypeError, ValueError):
                pass  # If can't calculate, don't filter
            
            filtered_symbols.append(sym_data)
            
            if len(filtered_symbols) >= self.max_symbols:
                break
        
        if blacklisted_count > 0 or low_vol_count > 0:
            print(f"[SLEEPER v8.2] Filtered: {blacklisted_count} blacklisted, {low_vol_count} low volatility (<{min_volatility}%)")
        
        symbols = filtered_symbols
        # === END BLACKLIST + VOLATILITY FILTER ===
        
        print(f"[SLEEPER v8.2] Analyzing {len(symbols)} symbols (batch_size={self.batch_size}, delay={self.batch_delay}s)...")
        
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
                        
                        # v6: MSS indicator
                        mss_bias = result.get('mss_bias', 'neutral')
                        mss_str = ""
                        if mss_bias == 'long':
                            mss_str = f" MSS:↑{result.get('higher_lows_count', 0)}HL"
                        elif mss_bias == 'short':
                            mss_str = f" MSS:↓{result.get('lower_highs_count', 0)}LH"
                        
                        print(f"[SLEEPER v8.2] {state_emoji}{vc_flag}{adx_flag}{poc_flag}{dir_flag} {result['symbol']}: "
                              f"Score={result['total_score']:.1f}{bonus_str} "
                              f"Dir={dir_str}{mss_str} "
                              f"(VC:{result['volatility_compression']:.0f} "
                              f"VS:{result['volume_suppression']:.0f} "
                              f"OI:{result['oi_growth']:.0f} "
                              f"OB:{result['order_book_imbalance']:.0f}) "
                              f"BB:{result.get('bb_compression_pct', 0):.1f}%")
                
                # Progress every 10 symbols
                if (i + 1) % 10 == 0:
                    print(f"[SLEEPER v8.2] Progress: {i+1}/{len(symbols)} ({len(candidates)} candidates)")
                
                # Rate limiting between individual symbols
                time.sleep(API_LIMITS.get('rate_limit_delay', 0.5))
                
                # Batch delay - longer pause every batch_size symbols
                if (i + 1) % self.batch_size == 0:
                    print(f"[SLEEPER v8.2] Batch complete, waiting {self.batch_delay}s...")
                    time.sleep(self.batch_delay)
                
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"[SLEEPER v8.2] Error analyzing {sym_data.get('symbol')}: {e}")
        
        # Update HP for existing sleepers
        self._update_hp_scores(candidates)
        
        # Remove dead sleepers
        removed = self.db.remove_dead_sleepers()
        if removed > 0:
            print(f"[SLEEPER v8.2] Removed {removed} dead sleepers (HP=0)")
        
        elapsed = time.time() - start_time
        print(f"\n[SLEEPER v8.2] Scan complete in {elapsed:.1f}s")
        print(f"[SLEEPER v8.2] Results: {len(candidates)} candidates from {len(symbols)} symbols")
        
        # Count by state
        by_state = {}
        for c in candidates:
            state = c.get('state', 'UNKNOWN')
            by_state[state] = by_state.get(state, 0) + 1
        print(f"[SLEEPER v8.2] States: {by_state}")
        
        # v4 Summary
        special_flags = []
        if vc_extreme_count > 0:
            special_flags.append(f"🔥VC-EXTREME:{vc_extreme_count}")
        if trendless_count > 0:
            special_flags.append(f"📊TRENDLESS:{trendless_count}")
        if poc_count > 0:
            special_flags.append(f"🎯POC:{poc_count}")
        
        if special_flags:
            print(f"[SLEEPER v8.2] Special: {' | '.join(special_flags)}")
        
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
            print(f"[SLEEPER v8.2] BTC check error: {e}")
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
        
        # v8.2.7: HTF klines for sleeper analysis (configurable via UI)
        klines_4h = self.fetcher.get_klines(symbol, self.htf_timeframe_api, limit=self.KLINES_5D)
        if len(klines_4h) < 20:
            return None
        
        # v8.2: LTF data for entry signals (15M instead of 1D)
        # 15M gives 4x more detail than 1H for faster markets (ADX > 40)
        ltf_timeframe = self.db.get_setting('ltf_timeframe', '15m')
        ltf_limit = 100 if ltf_timeframe == '15m' else 60  # ~25h of 15m or 60h of 1h
        klines_ltf = self.fetcher.get_klines(symbol, ltf_timeframe, limit=ltf_limit)
        
        # Daily klines for context (still used for some metrics)
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
        
        # === 7. DETERMINE DIRECTION (v8: SMC-integrated) ===
        
        # v8.2: Direction Engine uses 4H (HTF) and 15M/1H (LTF) + SMC Analysis
        # klines_4h = HTF for global bias
        # klines_ltf = LTF for entry signals (15M by default for fast markets)
        direction_result = self.direction_engine.analyze(
            symbol=symbol,
            klines_4h=klines_4h,           # HTF data (4H)
            klines_1h=klines_ltf,          # LTF data (15M or 1H based on setting)
            oi_change_pct=oi_data['growth_pct'],
            funding_rate=funding_rate,
            ob_imbalance=ob_data['imbalance'],
            poc_price=poc_data.get('poc_price', 0)
        )
        
        direction = direction_result.direction.value  # LONG, SHORT, or NEUTRAL
        direction_score = direction_result.bias_score  # v8: bias_score
        
        # === v8.2.5: TREND FILTER (4H Context) ===
        trend_aligned = True
        trend_info = ""
        
        if self.use_trend_filter:
            try:
                trend_result = self.trend_analyzer.analyze(symbol, self.trend_timeframe)
                
                if trend_result:
                    trend_regime = trend_result.regime
                    trend_score_value = trend_result.total_score
                    trend_info = f"[Trend:{trend_regime.value}({trend_score_value:.0f})]"
                    
                    # Перевіряємо мінімальний score
                    if trend_score_value >= self.min_trend_score:
                        # Якщо direction=NEUTRAL, спробуємо визначити з trend
                        if direction == 'NEUTRAL':
                            if trend_regime == TrendRegime.BULLISH:
                                direction = 'LONG'
                                direction_score = 0.15  # Fallback score
                            elif trend_regime == TrendRegime.BEARISH:
                                direction = 'SHORT'
                                direction_score = -0.15
                        else:
                            # Перевіряємо alignment
                            if trend_regime == TrendRegime.BULLISH and direction == 'SHORT':
                                trend_aligned = False
                            elif trend_regime == TrendRegime.BEARISH and direction == 'LONG':
                                trend_aligned = False
                            elif trend_regime == TrendRegime.NO_TRADE:
                                trend_aligned = False
                    else:
                        # Score низький - не надійний trend
                        trend_aligned = self.allow_signals_without_trend
                else:
                    # Немає даних тренду
                    trend_aligned = self.allow_signals_without_trend
            except Exception as e:
                # Помилка аналізу - продовжуємо без trend filter
                pass
        
        # v8: Convert confidence float (0-100) to string
        conf_value = direction_result.confidence
        if conf_value >= 75:
            direction_confidence = "HIGH"
        elif conf_value >= 50:
            direction_confidence = "MEDIUM"
        else:
            direction_confidence = "LOW"
        
        # v8: Extract SMC info (mapped to v5-compatible format)
        smc_result = direction_result.smc_result
        structure = direction_result.structure
        
        # Map SMC market_bias to market_phase
        if smc_result and smc_result.market_bias == MarketBias.BULLISH:
            market_phase = "MARKUP"
        elif smc_result and smc_result.market_bias == MarketBias.BEARISH:
            market_phase = "MARKDOWN"
        else:
            market_phase = "ACCUMULATION"  # Neutral = potential accumulation
        
        # Map SMC price_zone to phase_maturity
        if structure:
            if structure.price_zone == PriceZone.PREMIUM:
                phase_maturity = "LATE" if market_phase == "MARKUP" else "EARLY"
            elif structure.price_zone == PriceZone.DISCOUNT:
                phase_maturity = "EARLY" if market_phase == "MARKUP" else "LATE"
            else:
                phase_maturity = "MIDDLE"
        else:
            phase_maturity = "MIDDLE"
        
        # v8: Detect reversal setup via CHoCH signal
        is_reversal_setup = False
        if smc_result:
            is_reversal_setup = smc_result.structure_signal in [
                StructureSignal.BULLISH_CHOCH, 
                StructureSignal.BEARISH_CHOCH
            ]
        
        # v8: Use SMC score as exhaustion proxy (higher SMC = stronger signal)
        exhaustion_score = abs(smc_result.smc_score) if smc_result else 0.0
        
        # Build primary reason from SMC
        primary_reason = " | ".join(smc_result.reasons[:3]) if smc_result else "SMC Analysis"
        
        # === 8. DETERMINE STATE (v5: with phase-aware logic) ===
        
        state, vc_extreme = self._determine_state_v5(
            total_score,
            vc_data['compression_pct'],
            vs_data['suppression_pct'],
            vs_data['volume_spike'],
            oi_data['oi_jump'],
            # Additional params for accelerated paths
            vc_score=vc_data['score'],
            oi_score=oi_data['score'],
            oi_growth_pct=oi_data['growth_pct'],
            volume_ratio=vs_data['volume_ratio'],
            # Direction params
            direction=direction,
            direction_score=direction_score,
            direction_confidence=direction_confidence,
            # v5: Phase params (mapped from SMC)
            market_phase=market_phase,
            phase_maturity=phase_maturity,
            is_reversal_setup=is_reversal_setup,
            exhaustion_score=exhaustion_score
        )
        
        # === v8.2.5: TREND FILTER STATE ADJUSTMENT ===
        # Якщо trend_aligned=False і Trend Filter увімкнений, не дозволяємо READY
        if self.use_trend_filter and not trend_aligned and state == 'READY':
            state = 'BUILDING'  # Понижуємо до BUILDING - чекаємо кращий момент
            print(f"[SLEEPER v8.2.5] {symbol}: READY→BUILDING (trend not aligned)")
        
        # Minimum score filter (but keep VC-extreme candidates)
        if state == 'IDLE' and total_score < self.MIN_SCORE_WATCHING and not vc_extreme:
            return None
        
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
        
        # Boost HP for high direction confidence (v5)
        if direction_confidence == "HIGH" and direction != "NEUTRAL":
            hp = min(self.HP_MAX, hp + 1)
        
        # Boost HP for reversal setup (v5)
        if is_reversal_setup:
            hp = min(self.HP_MAX, hp + 2)
        
        # === v8.2.5: FALLBACK DIRECTION для READY без напрямку ===
        # Якщо state=READY але direction=NEUTRAL, визначаємо fallback на основі SMC
        if state == 'READY' and direction == 'NEUTRAL' and structure:
            market_bias = structure.market_bias.value if structure.market_bias else 'NEUTRAL'
            price_zone = structure.price_zone.value if structure.price_zone else 'EQUILIBRIUM'
            smc_signal = structure.smc_signal.value if structure.smc_signal else 'NONE'
            
            # CHoCH має найвищий пріоритет
            if 'BULLISH_CHOCH' in smc_signal:
                direction = 'LONG'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback LONG (CHoCH)")
            elif 'BEARISH_CHOCH' in smc_signal:
                direction = 'SHORT'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback SHORT (CHoCH)")
            # BOS + market bias
            elif 'BULLISH_BOS' in smc_signal and market_bias != 'BEARISH':
                direction = 'LONG'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback LONG (BOS+bias)")
            elif 'BEARISH_BOS' in smc_signal and market_bias != 'BULLISH':
                direction = 'SHORT'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback SHORT (BOS+bias)")
            # Market bias + zone alignment
            elif market_bias == 'BULLISH' and price_zone in ['DISCOUNT', 'EQUILIBRIUM']:
                direction = 'LONG'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback LONG (bias={market_bias}, zone={price_zone})")
            elif market_bias == 'BEARISH' and price_zone in ['PREMIUM', 'EQUILIBRIUM']:
                direction = 'SHORT'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback SHORT (bias={market_bias}, zone={price_zone})")
            # Останній варіант: тільки zone
            elif price_zone == 'DISCOUNT':
                direction = 'LONG'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback LONG (zone={price_zone})")
            elif price_zone == 'PREMIUM':
                direction = 'SHORT'
                print(f"[SLEEPER v8.2.5] {symbol}: Fallback SHORT (zone={price_zone})")
        
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
            
            # v5: Direction Engine data
            'direction_score': round(direction_score, 3),
            'direction_confidence': direction_confidence,
            'market_phase': market_phase,
            'phase_maturity': phase_maturity,
            'is_reversal_setup': is_reversal_setup,
            'exhaustion_score': round(exhaustion_score, 2),
            'direction_reason': primary_reason,
            
            # v8.2.5: Trend Filter data
            'trend_aligned': trend_aligned,
            'trend_info': trend_info,
            
            # v8: SMC Structure data
            'smc_signal': structure.smc_signal.value if structure else 'NONE',
            'smc_market_bias': structure.market_bias.value if structure else 'NEUTRAL',
            'smc_price_zone': structure.price_zone.value if structure else 'EQUILIBRIUM',
            'smc_zone_level': round(structure.zone_level, 3) if structure else 0.5,
            'smc_score': round(structure.smc_score, 3) if structure else 0,
            'price_at_bullish_ob': structure.price_at_bullish_ob if structure else False,
            'price_at_bearish_ob': structure.price_at_bearish_ob if structure else False,
            
            # v8: Structure counts
            'hh_count': structure.hh_count if structure else 0,
            'hl_count': structure.hl_count if structure else 0,
            'lh_count': structure.lh_count if structure else 0,
            'll_count': structure.ll_count if structure else 0,
            'dominant_structure': structure.dominant_structure.value if structure else 'UNKNOWN',
            
            # v8: Position in range
            'is_near_high': structure.is_near_high if structure else False,
            'is_near_low': structure.is_near_low if structure else False,
            'is_in_middle': structure.is_in_middle if structure else True,
            'price_vs_vwap_pct': round(structure.price_vs_vwap_pct, 2) if structure else 0,
            
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
    
    def _determine_state_v5(self, total_score: float, compression_pct: float,
                           suppression_pct: float, volume_spike: bool, oi_jump: bool,
                           vc_score: float = 0, oi_score: float = 0, 
                           oi_growth_pct: float = 0, volume_ratio: float = 1.0,
                           direction: str = 'NEUTRAL', direction_score: float = 0,
                           direction_confidence: str = 'LOW',
                           market_phase: str = 'UNKNOWN', phase_maturity: str = 'MIDDLE',
                           is_reversal_setup: bool = False,
                           exhaustion_score: float = 0.0) -> Tuple[str, bool]:
        """
        Determine sleeper state based on metrics - v5 with phase awareness
        
        Returns: (state, vc_extreme_flag)
        
        v5 КРИТИЧНА ЛОГІКА:
        - EXHAUSTED фаза + at support/resistance → READY для розвороту
        - LATE фаза → НЕ торгуємо в напрямку тренду!
        - REVERSAL SETUP → високий пріоритет
        
        PATHS TO READY (v5):
        1. Reversal Setup: is_reversal_setup = True
        2. Exhausted + Support/Resistance: phase_maturity = EXHAUSTED
        3. Classic: Score > 65, VC > 50%
        4. Accelerated: VC > 95% + OI growth > 20%
        
        PATHS TO BUILDING (v5):
        1. Phase-aware: Early/Middle phase + clear direction
        2. Classic: Score > 55, VC > 35%, VS > 40%
        3. Accelerated: VC > 90% + OI growth > 15%
        
        BLOCKED PATHS (v5):
        - LATE phase downtrend → NO SHORT
        - LATE phase uptrend → NO LONG
        """
        vc_extreme = False
        
        # === VC-EXTREME FLAG ===
        if compression_pct >= 95 and volume_ratio <= 1.2:
            vc_extreme = True
        
        # === TRIGGERED ===
        # Volume spike + OI jump (breakout happening)
        if volume_spike and oi_jump:
            return (SleeperState.TRIGGERED.value, vc_extreme)
        
        # ============================================
        # v5: REVERSAL PRIORITY PATH
        # ============================================
        
        # READY якщо reversal setup виявлено
        if is_reversal_setup and exhaustion_score >= 0.5:
            return (SleeperState.READY.value, vc_extreme)
        
        # READY якщо фаза вичерпана (exhausted) і є чіткий напрямок
        if phase_maturity == 'EXHAUSTED' and direction != 'NEUTRAL':
            if abs(direction_score) >= 0.4:
                return (SleeperState.READY.value, vc_extreme)
        
        # ============================================
        # v5: PHASE-AWARE BLOCKING
        # Не даємо сигнали в кінці тренду в напрямку тренду!
        # ============================================
        
        phase_blocked = False
        
        if phase_maturity == 'LATE':
            # LATE MARKDOWN → блокуємо SHORT (тренд закінчується)
            if market_phase == 'MARKDOWN' and direction == 'SHORT':
                phase_blocked = True
            # LATE MARKUP → блокуємо LONG (тренд закінчується)
            elif market_phase == 'MARKUP' and direction == 'LONG':
                phase_blocked = True
        
        if phase_maturity == 'EXHAUSTED':
            # EXHAUSTED фаза → не даємо сигнали з трендом, тільки проти
            if market_phase == 'MARKDOWN' and direction == 'SHORT':
                phase_blocked = True
            elif market_phase == 'MARKUP' and direction == 'LONG':
                phase_blocked = True
        
        # Якщо фаза заблокувала напрямок, повертаємо WATCHING замість READY/BUILDING
        if phase_blocked:
            if total_score >= self.MIN_SCORE_WATCHING:
                return (SleeperState.WATCHING.value, vc_extreme)
            return (SleeperState.IDLE.value, vc_extreme)
        
        # ============================================
        # CLASSIC PATHS (with phase awareness)
        # ============================================
        
        # === READY (Classic) ===
        if (total_score >= self.MIN_SCORE_READY and 
            compression_pct >= self.COMPRESSION_READY):
            return (SleeperState.READY.value, vc_extreme)
        
        # === READY (Accelerated) ===
        if compression_pct >= 95 and oi_growth_pct >= 20:
            return (SleeperState.READY.value, True)
        
        # === READY (High Confidence Direction) ===
        if (total_score >= 75 and 
            direction != 'NEUTRAL' and 
            direction_confidence == 'HIGH' and
            abs(direction_score) >= 0.5):
            return (SleeperState.READY.value, vc_extreme)
        
        # === BUILDING (Phase-aware) ===
        # В EARLY або MIDDLE фазі з чітким напрямком
        if (phase_maturity in ['EARLY', 'MIDDLE'] and
            direction != 'NEUTRAL' and
            direction_confidence in ['HIGH', 'MEDIUM'] and
            total_score >= 60):
            return (SleeperState.BUILDING.value, vc_extreme)
        
        # === BUILDING (Classic) ===
        if (total_score >= self.MIN_SCORE_BUILDING and 
            compression_pct >= self.COMPRESSION_BUILDING and
            suppression_pct >= self.VOLUME_SUPPRESSION_MIN):
            return (SleeperState.BUILDING.value, vc_extreme)
        
        # === BUILDING (Accelerated) ===
        if compression_pct >= 90 and oi_growth_pct >= 15:
            return (SleeperState.BUILDING.value, True)
        
        # === BUILDING (VC-Priority) ===
        if compression_pct >= 90 and oi_score >= 70:
            return (SleeperState.BUILDING.value, True)
        
        # === BUILDING (Direction-based) ===
        if (total_score >= 65 and 
            direction != 'NEUTRAL' and 
            direction_confidence in ['HIGH', 'MEDIUM'] and
            abs(direction_score) >= 0.4):
            return (SleeperState.BUILDING.value, vc_extreme)
        
        # === WATCHING ===
        if total_score >= self.MIN_SCORE_WATCHING:
            return (SleeperState.WATCHING.value, vc_extreme)
        
        # === WATCHING (VC-Extreme Override) ===
        if compression_pct >= 95:
            return (SleeperState.WATCHING.value, True)
        
        # === WATCHING (Reversal potential) ===
        # Якщо є ознаки вичерпання, варто слідкувати
        if exhaustion_score >= 0.4:
            return (SleeperState.WATCHING.value, vc_extreme)
        
        return (SleeperState.IDLE.value, vc_extreme)
    
    def _determine_state(self, total_score: float, compression_pct: float,
                         suppression_pct: float, volume_spike: bool, oi_jump: bool,
                         vc_score: float = 0, oi_score: float = 0, 
                         oi_growth_pct: float = 0, volume_ratio: float = 1.0,
                         direction: str = 'NEUTRAL', direction_score: float = 0,
                         direction_confidence: str = 'LOW') -> Tuple[str, bool]:
        """
        Determine sleeper state based on metrics
        
        Returns: (state, vc_extreme_flag)
        
        PATHS TO BUILDING (v4.1.2):
        1. Classic: Score > 55, VC > 35%, VS > 40%
        2. Accelerated: VC > 90% + OI growth > 15%
        3. Direction-based: Score > 70 + clear direction (HIGH/MEDIUM confidence)
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
        
        # === DIRECTION-BASED READY (v4.1.2) ===
        # High score + clear direction + high confidence = ready even without extreme compression
        if (total_score >= 75 and 
            direction != 'NEUTRAL' and 
            direction_confidence == 'HIGH' and
            abs(direction_score) >= 0.5):
            return (SleeperState.READY.value, vc_extreme)
        
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
        
        # === BUILDING (Direction-Based Path) v4.1.2 ===
        # High score + clear direction = building position even without compression
        if (total_score >= 65 and 
            direction != 'NEUTRAL' and 
            direction_confidence in ['HIGH', 'MEDIUM'] and
            abs(direction_score) >= 0.4):
            return (SleeperState.BUILDING.value, vc_extreme)
        
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
