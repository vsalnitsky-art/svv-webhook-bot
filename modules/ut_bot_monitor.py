"""
UT Bot Monitor Module - Integration Hub

Інтегрує:
- Sleeper Scanner (Binance) → вибір найкращої монети
- Direction Engine v7 → визначення біасу (HH/LL)
- UT Bot Filter (Bybit) → генерація сигналів
- Paper Trading → тестові угоди

Автор: SVV Bot Team
Версія: 1.0 (2026-01-26)
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from modules.ut_bot_filter import get_ut_bot_filter, UTSignalType
from detection.direction_engine_v7 import get_direction_engine_v7, BiasDirection
from core.bybit_connector import get_connector
from storage.db_operations import get_db


@dataclass
class PotentialCoin:
    """Монета-кандидат для UT Bot"""
    symbol: str
    sleeper_score: float
    direction: str              # 'LONG' or 'SHORT'
    structure_type: str         # 'HH', 'HL', 'LH', 'LL'
    is_near_extreme: bool       # Near HH or LL (not middle)
    confidence: float
    added_at: datetime
    last_check: datetime = None
    priority: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'sleeper_score': round(self.sleeper_score, 1),
            'direction': self.direction,
            'structure_type': self.structure_type,
            'is_near_extreme': self.is_near_extreme,
            'confidence': round(self.confidence, 1),
            'priority': round(self.priority, 2),
            'added_at': self.added_at.isoformat(),
            'last_check': self.last_check.isoformat() if self.last_check else None
        }


@dataclass
class UTBotTrade:
    """Угода UT Bot (Paper Trading)"""
    id: int
    symbol: str
    direction: str              # 'LONG' or 'SHORT'
    status: str                 # 'OPEN', 'CLOSED', 'CANCELLED'
    
    entry_price: float
    exit_price: float = None
    current_price: float = None
    
    atr_stop: float = 0.0
    highest_price: float = None   # For LONG trailing
    lowest_price: float = None    # For SHORT trailing
    
    entry_signal: Dict = None
    exit_signal: Dict = None
    
    opened_at: datetime = None
    closed_at: datetime = None
    
    pnl_usdt: float = 0.0
    pnl_percent: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction,
            'status': self.status,
            'entry_price': round(self.entry_price, 8),
            'exit_price': round(self.exit_price, 8) if self.exit_price else None,
            'current_price': round(self.current_price, 8) if self.current_price else None,
            'atr_stop': round(self.atr_stop, 8),
            'pnl_usdt': round(self.pnl_usdt, 2),
            'pnl_percent': round(self.pnl_percent, 2),
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
        }


class UTBotMonitor:
    """
    UT Bot Monitor - Central Control Module
    
    Flow:
    1. Receive sleepers from Sleeper Scanner
    2. Filter: only HH/LL (not middle)
    3. Select TOP 1 coin by priority
    4. Monitor with UT Bot Filter (Bybit data)
    5. Generate Paper Trading signals
    """
    
    # Configuration
    DEFAULT_CONFIG = {
        'enabled': True,              # Module enabled by default
        'timeframe': '15m',           # UT Bot timeframe
        'atr_period': 10,
        'atr_multiplier': 1.0,
        'use_heikin_ashi': False,     # Heikin Ashi OFF by default
        'allow_first_entry': False,   # OFF by default - use pure crossover like TradingView
        
        # Filters - relaxed for more candidates
        'min_sleeper_score': 60,      # Lowered for more candidates
        'min_sleeper_hp': 4,          # Lowered for more candidates
        'require_structure': False,   # Allow all positions for testing
        
        # Trading
        'max_open_trades': 3,
        'position_size_usdt': 100,    # Paper trading size
        'min_signal_gap_minutes': 15, # Reduced cooldown
        'max_monitored_coins': 3,     # How many TOP coins to monitor for signals
        
        # Checks
        'check_interval_seconds': 60,
        'max_trade_duration_hours': 24,
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize UT Bot Monitor"""
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
        
        # Get DB first
        self.db = get_db()
        
        # Load config from DB
        self._load_config_from_db()
        
        self.ut_bot = get_ut_bot_filter({
            'key_value': self.config['atr_multiplier'],
            'atr_period': self.config['atr_period'],
            'use_heikin_ashi': self.config['use_heikin_ashi'],
            'timeframe': self.config['timeframe'],
        })
        
        self.direction_engine = get_direction_engine_v7()
        self.bybit = get_connector()
        
        # State
        self.potential_coins: Dict[str, PotentialCoin] = {}
        self.open_trades: Dict[str, UTBotTrade] = {}
        self.trade_history: List[UTBotTrade] = []
        self._trade_counter = 0
        
        # Cooldowns
        self._last_signal_time: Dict[str, datetime] = {}
        
        # Statistics
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'last_update': None
        }
        
        print(f"[UT BOT] Monitor initialized. Config: timeframe={self.config['timeframe']}, HA={self.config['use_heikin_ashi']}, enabled={self.config['enabled']}")
    
    def _load_config_from_db(self):
        """Load configuration from database"""
        try:
            # Load each setting
            tf = self.db.get_setting('ut_bot_timeframe', self.config['timeframe'])
            if tf:
                self.config['timeframe'] = str(tf)
            
            atr_p = self.db.get_setting('ut_bot_atr_period', self.config['atr_period'])
            if atr_p is not None:
                self.config['atr_period'] = int(atr_p)
            
            atr_m = self.db.get_setting('ut_bot_atr_mult', None)
            if atr_m is None:
                atr_m = self.db.get_setting('ut_bot_atr_multiplier', self.config['atr_multiplier'])
            if atr_m is not None:
                self.config['atr_multiplier'] = float(atr_m)
            
            # Load booleans - DB returns Python boolean from json.loads
            ha = self.db.get_setting('ut_bot_heikin_ashi', None)
            if ha is None:
                ha = self.db.get_setting('ut_bot_use_heikin_ashi', False)
            self.config['use_heikin_ashi'] = bool(ha)
            
            enabled = self.db.get_setting('module_ut_bot', None)
            if enabled is None:
                enabled = self.db.get_setting('ut_bot_enabled', True)
            self.config['enabled'] = bool(enabled)
            
            # Load max_monitored_coins
            max_coins = self.db.get_setting('ut_bot_max_coins', None)
            if max_coins is None:
                max_coins = self.db.get_setting('ut_bot_max_monitored_coins', self.config['max_monitored_coins'])
            if max_coins is not None:
                self.config['max_monitored_coins'] = int(max_coins)
            
            # Load position size
            pos_size = self.db.get_setting('ut_bot_position_size', self.config['position_size_usdt'])
            if pos_size is not None:
                self.config['position_size_usdt'] = float(pos_size)
            
            # Load max trades
            max_trades = self.db.get_setting('ut_bot_max_trades', self.config['max_open_trades'])
            if max_trades is not None:
                self.config['max_open_trades'] = int(max_trades)
            
            # Load FIRST ENTRY option (default: OFF for pure TradingView crossover logic)
            first_entry = self.db.get_setting('ut_bot_first_entry', False)
            self.config['allow_first_entry'] = bool(first_entry)
            
            # Update UT Bot filter with new settings
            self.ut_bot.set_config('key_value', self.config['atr_multiplier'])
            self.ut_bot.set_config('atr_period', self.config['atr_period'])
            self.ut_bot.set_config('use_heikin_ashi', self.config['use_heikin_ashi'])
            self.ut_bot.set_config('timeframe', self.config['timeframe'])
            
            print(f"[UT BOT] Loaded config: TF={self.config['timeframe']}, ATR={self.config['atr_period']}/{self.config['atr_multiplier']}, HA={self.config['use_heikin_ashi']}, enabled={self.config['enabled']}, max_coins={self.config['max_monitored_coins']}, first_entry={self.config['allow_first_entry']}")
        except Exception as e:
            print(f"[UT BOT] Config load error (using defaults): {e}")
            import traceback
            traceback.print_exc()
    
    def update_from_sleepers(self, sleepers: List[Dict]) -> int:
        """
        Update potential coins from Sleeper Scanner results
        
        STRATEGY v7.0.5:
        1. Монети з direction LONG/SHORT від Sleeper - приймаємо (score ≥ 55)
        2. Монети з NEUTRAL - визначаємо direction по структурі ціни:
           - distance_from_low < 20% → LONG (ціна біля підтримки)
           - distance_from_high < 20% → SHORT (ціна біля опору)
        3. Або по Market Structure Shift (MSS) якщо є
        
        Returns:
            Number of coins added/updated
        """
        if not self.config['enabled']:
            print("[UT BOT] Module disabled, skipping update")
            return 0
        
        updated = 0
        now = datetime.now()
        
        # Stats
        stats = {
            'total': len(sleepers),
            'low_score': 0,
            'low_hp': 0,
            'wrong_state': 0,
            'no_direction_found': 0,
            'not_on_bybit': 0,
            'passed_sleeper_dir': 0,
            'passed_structure_dir': 0,
            'passed_mss_dir': 0
        }
        
        for sleeper in sleepers:
            symbol = sleeper.get('symbol', '')
            if not symbol:
                continue
            
            score = sleeper.get('total_score', 0)
            hp = sleeper.get('hp', 0)
            state = sleeper.get('state', '')
            
            # State filter
            if state not in ['READY', 'BUILDING', 'TRIGGERED', 'WATCHING']:
                stats['wrong_state'] += 1
                continue
            
            # Score/HP thresholds (relaxed)
            min_score = 55
            min_hp = 3
            
            if score < min_score:
                stats['low_score'] += 1
                continue
            if hp < min_hp:
                stats['low_hp'] += 1
                continue
            
            # === DETERMINE DIRECTION ===
            sleeper_direction = sleeper.get('direction', 'NEUTRAL')
            direction_score = sleeper.get('direction_score', 0)
            direction = None
            confidence = 50
            source = None
            
            # 1. Use Sleeper direction if available
            if sleeper_direction in ['LONG', 'SHORT']:
                direction = sleeper_direction
                confidence = abs(direction_score) * 100 if direction_score else 60
                source = 'SLEEPER'
            
            # 2. Fallback: Use price structure (distance from high/low)
            else:
                dist_high = sleeper.get('distance_from_high', 50)
                dist_low = sleeper.get('distance_from_low', 50)
                
                # Near support (low) → LONG bias
                if dist_low < 20 and dist_low < dist_high:
                    direction = 'LONG'
                    confidence = max(50, 70 - dist_low)  # Closer = higher confidence
                    source = 'STRUCTURE_LOW'
                
                # Near resistance (high) → SHORT bias
                elif dist_high < 20 and dist_high < dist_low:
                    direction = 'SHORT'
                    confidence = max(50, 70 - dist_high)
                    source = 'STRUCTURE_HIGH'
                
                # 3. Fallback: Use MSS (Market Structure Shift)
                else:
                    mss_bias = sleeper.get('mss_bias', 0)
                    hl_count = sleeper.get('higher_lows_count', 0)
                    lh_count = sleeper.get('lower_highs_count', 0)
                    
                    if mss_bias > 0 or hl_count >= 2:
                        direction = 'LONG'
                        confidence = 55
                        source = 'MSS_BULLISH'
                    elif mss_bias < 0 or lh_count >= 2:
                        direction = 'SHORT'
                        confidence = 55
                        source = 'MSS_BEARISH'
            
            # No direction found
            if not direction:
                stats['no_direction_found'] += 1
                continue
            
            # Check Bybit availability
            if not self._is_symbol_on_bybit(symbol):
                stats['not_on_bybit'] += 1
                continue
            
            # Calculate priority
            priority = self._calculate_priority(score, hp, confidence, False, False)
            
            # Track source
            if source == 'SLEEPER':
                stats['passed_sleeper_dir'] += 1
            elif source and 'STRUCTURE' in source:
                stats['passed_structure_dir'] += 1
            elif source and 'MSS' in source:
                stats['passed_mss_dir'] += 1
            
            # Add coin to memory
            self.potential_coins[symbol] = PotentialCoin(
                symbol=symbol,
                sleeper_score=score,
                direction=direction,
                structure_type=f"{source}:{sleeper.get('market_phase', '?')}",
                is_near_extreme=sleeper.get('distance_from_low', 50) < 15 or sleeper.get('distance_from_high', 50) < 15,
                confidence=confidence,
                added_at=self.potential_coins.get(symbol, PotentialCoin(
                    symbol=symbol, sleeper_score=0, direction='', 
                    structure_type='', is_near_extreme=False,
                    confidence=0, added_at=now
                )).added_at,
                last_check=now,
                priority=priority
            )
            updated += 1
            print(f"[UT BOT] ✅ {symbol}: {direction} (src={source}, score={score}, conf={confidence:.0f}%)")
        
        # === SAVE ALL TO DATABASE ===
        try:
            from storage.db_models import UTBotPotentialCoin, get_session
            session = get_session()
            
            for symbol, coin in self.potential_coins.items():
                existing = session.query(UTBotPotentialCoin).filter_by(symbol=symbol).first()
                if existing:
                    # Update
                    existing.direction = coin.direction
                    existing.sleeper_score = coin.sleeper_score
                    existing.confidence = coin.confidence
                    existing.priority = coin.priority
                    existing.source = coin.structure_type.split(':')[0] if ':' in (coin.structure_type or '') else 'SLEEPER'
                    existing.structure_type = coin.structure_type
                    existing.is_near_extreme = coin.is_near_extreme
                    existing.last_check = now
                    existing.updated_at = now
                else:
                    # Create new
                    new_coin = UTBotPotentialCoin(
                        symbol=symbol,
                        direction=coin.direction,
                        sleeper_score=coin.sleeper_score,
                        confidence=coin.confidence,
                        priority=coin.priority,
                        source=coin.structure_type.split(':')[0] if ':' in (coin.structure_type or '') else 'SLEEPER',
                        structure_type=coin.structure_type,
                        is_near_extreme=coin.is_near_extreme,
                        added_at=coin.added_at,
                        last_check=now
                    )
                    session.add(new_coin)
            
            session.commit()
            db_count = session.query(UTBotPotentialCoin).count()
            session.close()
            print(f"[UT BOT] 💾 Saved {updated} coins to DB (total: {db_count})")
        except Exception as e:
            print(f"[UT BOT] Error saving to DB: {e}")
        
        # Summary
        print(f"[UT BOT] ═══════════════════════════════════")
        print(f"[UT BOT] Total: {stats['total']} | Rejected: score={stats['low_score']}, hp={stats['low_hp']}, state={stats['wrong_state']}, no_dir={stats['no_direction_found']}, bybit={stats['not_on_bybit']}")
        print(f"[UT BOT] ✅ PASSED: {updated} (sleeper={stats['passed_sleeper_dir']}, structure={stats['passed_structure_dir']}, mss={stats['passed_mss_dir']})")
        print(f"[UT BOT] Potential coins in memory: {len(self.potential_coins)}")
        print(f"[UT BOT] ═══════════════════════════════════")
        
        # Cleanup old (from both memory and DB)
        cutoff = now - timedelta(minutes=60)
        to_remove = [s for s, c in self.potential_coins.items() if c.last_check and c.last_check < cutoff]
        for s in to_remove:
            del self.potential_coins[s]
        
        # Cleanup old from DB
        try:
            from storage.db_models import UTBotPotentialCoin, get_session
            session = get_session()
            old_coins = session.query(UTBotPotentialCoin).filter(UTBotPotentialCoin.updated_at < cutoff).all()
            for c in old_coins:
                session.delete(c)
            if old_coins:
                session.commit()
                print(f"[UT BOT] 🗑️ Removed {len(old_coins)} stale coins from DB")
            session.close()
        except Exception as e:
            print(f"[UT BOT] Error cleaning DB: {e}")
        
        return updated
    
    def get_top_coin(self) -> Optional[PotentialCoin]:
        """Get the TOP 1 priority coin for trading (from DB)"""
        coins = self.get_top_coins(limit=1)
        return coins[0] if coins else None
    
    def get_top_coins(self, limit: int = None) -> List[PotentialCoin]:
        """
        Get the TOP N priority coins for trading (from DB)
        
        Args:
            limit: Number of coins to return. If None, uses config max_monitored_coins
        
        Selection criteria:
        1. Must have open position slot
        2. Highest priority score
        """
        if limit is None:
            limit = self.config.get('max_monitored_coins', 1)
        
        try:
            from storage.db_models import UTBotPotentialCoin, UTBotPaperTrade, get_session
            session = get_session()
            
            # Get all potential coins sorted by priority
            coins = session.query(UTBotPotentialCoin).order_by(UTBotPotentialCoin.priority.desc()).all()
            
            # Get symbols with open trades
            open_trades = session.query(UTBotPaperTrade).filter_by(status='OPEN').all()
            open_symbols = {t.symbol for t in open_trades}
            
            session.close()
            
            # Filter out coins with open trades
            available = [c for c in coins if c.symbol not in open_symbols]
            
            if not available:
                return []
            
            # Convert to PotentialCoin dataclass
            result = []
            for coin in available[:limit]:
                result.append(PotentialCoin(
                    symbol=coin.symbol,
                    sleeper_score=coin.sleeper_score or 0,
                    direction=coin.direction,
                    structure_type=coin.structure_type or 'UNKNOWN',
                    is_near_extreme=coin.is_near_extreme or False,
                    confidence=coin.confidence or 0,
                    added_at=coin.added_at or datetime.now(),
                    last_check=coin.last_check,
                    priority=coin.priority or 0
                ))
            
            return result
            
        except Exception as e:
            print(f"[UT BOT] Error getting top coins from DB: {e}")
            # Fallback to memory
            if not self.potential_coins:
                return []
            
            available = [c for s, c in self.potential_coins.items()
                         if s not in self.open_trades]
            
            if not available:
                return []
            
            available.sort(key=lambda x: x.priority, reverse=True)
            return available[:limit]
    
    def check_signals(self) -> List[Dict]:
        """
        Check UT Bot signals for potential coins
        
        Returns:
            List of signal events
        """
        # Reload config from DB to get latest settings
        self._load_config_from_db()
        
        if not self.config['enabled']:
            self._last_check_result = {'status': 'disabled'}
            return []
        
        events = []
        now = datetime.now()
        
        # Get how many coins to monitor
        max_monitored = self.config.get('max_monitored_coins', 1)
        
        # Initialize last check result
        self._last_check_result = {
            'timestamp': now.isoformat(),
            'status': 'checked',
            'top_coin': None,
            'monitored_coins': [],
            'signal': None,
            'action': 'NONE',
            'error': None
        }
        
        try:
            # =====================================================
            # 1. CHECK FOR OPEN SIGNALS ON TOP N COINS
            # =====================================================
            top_coins = self.get_top_coins(limit=max_monitored)
            
            if top_coins:
                self._last_check_result['top_coin'] = top_coins[0].symbol
                self._last_check_result['monitored_coins'] = [c.symbol for c in top_coins]
                
                # Log which coins we're checking
                coins_str = ", ".join([f"{c.symbol}({c.direction})" for c in top_coins])
                print(f"[UT BOT] Checking signals for TOP {len(top_coins)} coins: {coins_str}")
                
                for coin in top_coins:
                    # Check cooldown
                    last_signal = self._last_signal_time.get(coin.symbol)
                    skip_signal_check = False
                    
                    if last_signal:
                        elapsed = (now - last_signal).total_seconds()
                        if elapsed < self.config['min_signal_gap_minutes'] * 60:
                            # Don't log cooldown for every coin - too noisy
                            skip_signal_check = True
                    
                    if not skip_signal_check:
                        try:
                            # Check if we already have an open trade for this symbol (from DB!)
                            from storage.db_models import UTBotPaperTrade, get_session
                            session = get_session()
                            existing_trade = session.query(UTBotPaperTrade).filter_by(
                                symbol=coin.symbol,
                                status='OPEN'
                            ).first()
                            session.close()
                            has_open_trade = existing_trade is not None
                            
                            # Check UT Bot signal
                            # allow_first_entry based on:
                            # 1. Config setting (ut_bot_first_entry)
                            # 2. No existing open trade for this symbol
                            first_entry_enabled = self.config.get('allow_first_entry', True)
                            allow_first = first_entry_enabled and not has_open_trade
                            
                            signal_result = self.ut_bot.check_signal_with_bias(
                                coin.symbol,
                                coin.direction,
                                timeframe=self.config['timeframe'],
                                allow_first_entry=allow_first
                            )
                            
                            # Update last check result for first coin only
                            if coin == top_coins[0]:
                                self._last_check_result['signal'] = signal_result
                                self._last_check_result['action'] = signal_result.get('trade_action', 'HOLD')
                            
                            # Log signal result with details
                            price = signal_result.get('price', 0)
                            atr_stop = signal_result.get('atr_trailing_stop', 0)
                            pos = signal_result.get('position', 0)
                            prev_pos = signal_result.get('prev_position', 0)
                            bar_color = signal_result.get('bar_color', '?')
                            
                            # Calculate distance to stop
                            if price > 0 and atr_stop > 0:
                                distance_pct = abs(price - atr_stop) / price * 100
                                dist_dir = "above" if price > atr_stop else "below"
                            else:
                                distance_pct = 0
                                dist_dir = "?"
                            
                            print(f"[UT BOT] {coin.symbol}: pos={prev_pos}→{pos} | price={price:.6f} | stop={atr_stop:.6f} ({dist_dir} {distance_pct:.2f}%) | bar={bar_color}")
                            
                            # Process OPEN signal (aligned with bias)
                            trade_action = signal_result.get('trade_action', 'HOLD')
                            if signal_result.get('aligned') and trade_action.startswith('ENTER'):
                                # Open trade
                                trade = self._open_trade(coin, signal_result)
                                if trade:
                                    events.append({
                                        'type': 'TRADE_OPENED',
                                        'trade': trade.to_dict(),
                                        'signal': signal_result
                                    })
                                    self._last_signal_time[coin.symbol] = now
                                    print(f"[UT BOT] ✅ TRADE OPENED: {coin.symbol} {trade_action}")
                        
                        except Exception as e:
                            print(f"[UT BOT] Error checking signals for {coin.symbol}: {e}")
                            continue
            else:
                self._last_check_result['action'] = 'NO_COINS'
                print("[UT BOT] No potential coins available for signal check")
            
            # =====================================================
            # 2. MONITOR OPEN TRADES FOR EXIT SIGNALS
            # =====================================================
            # First, sync open_trades from database
            self._sync_open_trades_from_db()
            
            for symbol, trade in list(self.open_trades.items()):
                try:
                    exit_signal = self._check_exit_signal(trade)
                    if exit_signal:
                        closed_trade = self._close_trade(trade, exit_signal)
                        if closed_trade:
                            events.append({
                                'type': 'TRADE_CLOSED',
                                'trade': closed_trade.to_dict(),
                                'signal': exit_signal
                            })
                            print(f"[UT BOT] ✅ TRADE CLOSED: {symbol} - {exit_signal.get('reason')}")
                except Exception as e:
                    print(f"[UT BOT] Error checking exit for {symbol}: {e}")
                    continue
        
        except Exception as e:
            print(f"[UT BOT] Critical error in check_signals: {e}")
            self._last_check_result['status'] = 'error'
            self._last_check_result['error'] = str(e)
        
        return events
    
    def _open_trade(self, coin: PotentialCoin, signal: Dict) -> Optional[UTBotTrade]:
        """Open a paper trade and save to database"""
        import json
        from storage.db_models import UTBotPaperTrade, get_session
        
        # Check max trades from DB (not memory!)
        try:
            session = get_session()
            open_count = session.query(UTBotPaperTrade).filter_by(status='OPEN').count()
            session.close()
            
            if open_count >= self.config['max_open_trades']:
                print(f"[UT BOT] Max open trades ({self.config['max_open_trades']}) reached, skipping {coin.symbol}")
                return None
            
            # Check if already have trade for this symbol
            session = get_session()
            existing = session.query(UTBotPaperTrade).filter_by(
                symbol=coin.symbol, 
                status='OPEN'
            ).first()
            session.close()
            
            if existing:
                print(f"[UT BOT] Already have open trade for {coin.symbol}, skipping")
                return None
                
        except Exception as e:
            print(f"[UT BOT] Error checking open trades: {e}")
            return None
        
        # Create DB record
        try:
            session = get_session()
            
            # Convert numpy types to Python native types for PostgreSQL
            entry_price = float(signal.get('price', 0))
            atr_stop = float(signal.get('atr_trailing_stop', 0))
            
            db_trade = UTBotPaperTrade(
                symbol=coin.symbol,
                direction=coin.direction,
                status='OPEN',
                entry_price=entry_price,
                current_price=entry_price,
                atr_stop=atr_stop,
                highest_price=entry_price if coin.direction == 'LONG' else None,
                lowest_price=entry_price if coin.direction == 'SHORT' else None,
                entry_signal=json.dumps(signal, default=str) if signal else None,
                opened_at=datetime.now(),
                pnl_usdt=0.0,
                pnl_percent=0.0
            )
            
            session.add(db_trade)
            session.commit()
            trade_id = db_trade.id
            session.close()
            
            print(f"[UT BOT] 💾 Trade saved to DB: ID={trade_id}, {coin.symbol} {coin.direction}")
            
        except Exception as e:
            print(f"[UT BOT] ❌ Error saving trade to DB: {e}")
            session.rollback()
            session.close()
            return None
        
        # Also keep in memory for quick access
        self._trade_counter += 1
        
        trade = UTBotTrade(
            id=trade_id,
            symbol=coin.symbol,
            direction=coin.direction,
            status='OPEN',
            entry_price=entry_price,
            atr_stop=atr_stop,
            highest_price=entry_price if coin.direction == 'LONG' else None,
            lowest_price=entry_price if coin.direction == 'SHORT' else None,
            entry_signal=signal,
            opened_at=datetime.now()
        )
        
        self.open_trades[coin.symbol] = trade
        self.stats['total_trades'] += 1
        
        # Log event
        self.db.log_event(
            category='UT_BOT',
            symbol=coin.symbol,
            message=f"Paper trade opened: {coin.direction} @ {trade.entry_price}"
        )
        
        # Send Telegram notification
        try:
            from alerts.telegram_notifier import get_telegram_notifier
            telegram = get_telegram_notifier()
            if telegram and telegram.enabled:
                emoji = "🟢" if coin.direction == "LONG" else "🔴"
                reason = signal.get('reason', 'UT Bot signal')
                msg = (
                    f"{emoji} <b>UT Bot Paper Trade OPENED</b>\n\n"
                    f"<b>Symbol:</b> {coin.symbol}\n"
                    f"<b>Direction:</b> {coin.direction}\n"
                    f"<b>Entry Price:</b> {trade.entry_price:.6f}\n"
                    f"<b>ATR Stop:</b> {trade.atr_stop:.6f}\n"
                    f"<b>Reason:</b> {reason}\n"
                    f"<b>Position Size:</b> ${self.config['position_size_usdt']}"
                )
                telegram.send_message(msg, alert_type='ut_bot')
                print(f"[UT BOT] 📱 Telegram notification sent for {coin.symbol}")
        except Exception as e:
            print(f"[UT BOT] ⚠️ Telegram notification failed: {e}")
        
        return trade
    
    def _sync_open_trades_from_db(self):
        """
        Sync open_trades dict from database.
        This is needed because scheduler worker and HTTP workers have separate memory.
        """
        try:
            from storage.db_models import UTBotPaperTrade, get_session
            session = get_session()
            db_trades = session.query(UTBotPaperTrade).filter_by(status='OPEN').all()
            session.close()
            
            # Update memory from DB
            synced_symbols = set()
            for db_trade in db_trades:
                if db_trade.symbol not in self.open_trades:
                    # Create UTBotTrade from DB record
                    import json
                    entry_signal = {}
                    if db_trade.entry_signal:
                        try:
                            entry_signal = json.loads(db_trade.entry_signal)
                        except:
                            pass
                    
                    trade = UTBotTrade(
                        id=db_trade.id,
                        symbol=db_trade.symbol,
                        direction=db_trade.direction,
                        status=db_trade.status,
                        entry_price=db_trade.entry_price,
                        current_price=db_trade.current_price,
                        atr_stop=db_trade.atr_stop,
                        highest_price=db_trade.highest_price,
                        lowest_price=db_trade.lowest_price,
                        entry_signal=entry_signal,
                        opened_at=db_trade.opened_at
                    )
                    self.open_trades[db_trade.symbol] = trade
                synced_symbols.add(db_trade.symbol)
            
            # Remove from memory trades that are no longer in DB
            for symbol in list(self.open_trades.keys()):
                if symbol not in synced_symbols:
                    del self.open_trades[symbol]
                    
        except Exception as e:
            print(f"[UT BOT] Error syncing trades from DB: {e}")
    
    def _check_exit_signal(self, trade: UTBotTrade) -> Optional[Dict]:
        """Check if trade should be closed"""
        # Get current UT Bot signal
        signal = self.ut_bot.analyze(trade.symbol, timeframe=self.config['timeframe'])
        
        # Update current price
        trade.current_price = signal.price
        
        # Update trailing values
        if trade.direction == 'LONG' and signal.price > (trade.highest_price or 0):
            trade.highest_price = signal.price
        elif trade.direction == 'SHORT' and signal.price < (trade.lowest_price or float('inf')):
            trade.lowest_price = signal.price
        
        # Check exit conditions
        should_exit = False
        exit_reason = None
        
        # =====================================================
        # EXIT LOGIC (EXACT PINE SCRIPT)
        # =====================================================
        # Exit when UT Bot gives CLOSE signal for our direction
        # This happens when pos changes FROM our position TO neutral/opposite
        #
        # For LONG: pos was 1, now is 0 or -1 → CLOSE_LONG
        # For SHORT: pos was -1, now is 0 or 1 → CLOSE_SHORT
        # =====================================================
        
        # 1. UT Bot CLOSE signal matching our position direction
        if signal.signal_action == 'CLOSE':
            if trade.direction == 'LONG' and signal.direction == 'LONG':
                should_exit = True
                exit_reason = 'UT_CLOSE_LONG'
            elif trade.direction == 'SHORT' and signal.direction == 'SHORT':
                should_exit = True
                exit_reason = 'UT_CLOSE_SHORT'
        
        # 2. Opposite OPEN signal (alternative exit - more aggressive)
        # This covers the case when pos goes directly from 1 to -1 (or vice versa)
        elif signal.signal_action == 'OPEN':
            if trade.direction == 'LONG' and signal.direction == 'SHORT':
                should_exit = True
                exit_reason = 'UT_REVERSE_TO_SHORT'
            elif trade.direction == 'SHORT' and signal.direction == 'LONG':
                should_exit = True
                exit_reason = 'UT_REVERSE_TO_LONG'
        
        # 3. Max duration
        if trade.opened_at:
            duration = (datetime.now() - trade.opened_at).total_seconds() / 3600
            if duration > self.config['max_trade_duration_hours']:
                should_exit = True
                exit_reason = 'MAX_DURATION'
        
        if should_exit:
            return {
                'reason': exit_reason,
                'price': signal.price,
                'signal': signal.to_dict()
            }
        
        return None
    
    def _close_trade(self, trade: UTBotTrade, exit_signal: Dict) -> UTBotTrade:
        """Close a paper trade and update in database"""
        import json
        from storage.db_models import UTBotPaperTrade, get_session
        
        trade.status = 'CLOSED'
        trade.exit_price = float(exit_signal.get('price', trade.current_price or trade.entry_price))
        trade.exit_signal = exit_signal
        trade.closed_at = datetime.now()
        
        # Calculate PnL (ensure float types)
        entry = float(trade.entry_price)
        exit_p = float(trade.exit_price)
        
        if trade.direction == 'LONG':
            trade.pnl_percent = (exit_p - entry) / entry * 100
        else:
            trade.pnl_percent = (entry - exit_p) / entry * 100
        
        trade.pnl_usdt = float(self.config['position_size_usdt']) * trade.pnl_percent / 100
        
        # Update in database
        try:
            session = get_session()
            db_trade = session.query(UTBotPaperTrade).filter_by(
                symbol=trade.symbol,
                status='OPEN'
            ).first()
            
            if db_trade:
                db_trade.status = 'CLOSED'
                db_trade.exit_price = float(trade.exit_price)
                db_trade.exit_signal = json.dumps(exit_signal, default=str) if exit_signal else None
                db_trade.closed_at = trade.closed_at
                db_trade.pnl_usdt = float(trade.pnl_usdt)
                db_trade.pnl_percent = float(trade.pnl_percent)
                session.commit()
                print(f"[UT BOT] 💾 Trade closed in DB: ID={db_trade.id}, PnL=${trade.pnl_usdt:.2f}")
            else:
                print(f"[UT BOT] ⚠️ Trade not found in DB for {trade.symbol}")
            
            session.close()
        except Exception as e:
            print(f"[UT BOT] ❌ Error updating trade in DB: {e}")
            session.rollback()
            session.close()
        
        # Update stats
        self.stats['total_pnl'] += trade.pnl_usdt
        if trade.pnl_usdt > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        # Move to history (memory)
        self.trade_history.append(trade)
        if trade.symbol in self.open_trades:
            del self.open_trades[trade.symbol]
        
        # Log event
        self.db.log_event(
            category='UT_BOT',
            symbol=trade.symbol,
            message=f"Paper trade closed: PnL ${trade.pnl_usdt:.2f} ({trade.pnl_percent:.2f}%)"
        )
        
        # Send Telegram notification
        try:
            from alerts.telegram_notifier import get_telegram_notifier
            telegram = get_telegram_notifier()
            if telegram and telegram.enabled:
                emoji = "✅" if trade.pnl_usdt >= 0 else "❌"
                pnl_color = "🟢" if trade.pnl_usdt >= 0 else "🔴"
                reason = exit_signal.get('reason', 'Exit signal') if exit_signal else 'Unknown'
                msg = (
                    f"{emoji} <b>UT Bot Paper Trade CLOSED</b>\n\n"
                    f"<b>Symbol:</b> {trade.symbol}\n"
                    f"<b>Direction:</b> {trade.direction}\n"
                    f"<b>Entry:</b> {trade.entry_price:.6f}\n"
                    f"<b>Exit:</b> {trade.exit_price:.6f}\n"
                    f"{pnl_color} <b>PnL:</b> ${trade.pnl_usdt:.2f} ({trade.pnl_percent:.2f}%)\n"
                    f"<b>Reason:</b> {reason}"
                )
                telegram.send_message(msg, alert_type='ut_bot')
                print(f"[UT BOT] 📱 Telegram notification sent for {trade.symbol} close")
        except Exception as e:
            print(f"[UT BOT] ⚠️ Telegram notification failed: {e}")
        
        return trade
    
    def _analyze_direction(self, symbol: str) -> Optional[Dict]:
        """Analyze direction using Direction Engine v7"""
        try:
            # Get klines from Binance (sleeper source)
            from core.market_data import get_fetcher
            fetcher = get_fetcher()
            klines_4h = fetcher.get_klines(symbol, '4h', limit=100)
            klines_1h = fetcher.get_klines(symbol, '1h', limit=100)
            
            if not klines_4h:
                return None
            
            result = self.direction_engine.analyze(symbol, klines_4h, klines_1h)
            return result.to_dict()
        except Exception as e:
            print(f"[UT BOT] Direction analysis error for {symbol}: {e}")
            return None
    
    def _is_symbol_on_bybit(self, symbol: str) -> bool:
        """Check if symbol is available on Bybit (with caching)"""
        # Use cached list of Bybit symbols
        if not hasattr(self, '_bybit_symbols') or self._bybit_symbols is None:
            try:
                tickers = self.bybit.get_tickers()
                if tickers:
                    self._bybit_symbols = set(t.get('symbol', '') for t in tickers)
                    print(f"[UT BOT] Cached {len(self._bybit_symbols)} Bybit symbols")
                else:
                    self._bybit_symbols = set()
            except Exception as e:
                print(f"[UT BOT] Failed to get Bybit symbols: {e}")
                self._bybit_symbols = set()
        
        return symbol in self._bybit_symbols
    
    def _calculate_priority(self, score: float, hp: int, confidence: float,
                           is_near_high: bool, is_near_low: bool) -> float:
        """Calculate priority score for coin selection"""
        priority = 0.0
        
        # Base score (0-100 → 0-50)
        priority += score * 0.5
        
        # HP bonus (0-10 → 0-20)
        priority += hp * 2
        
        # Confidence bonus (0-100 → 0-15)
        priority += confidence * 0.15
        
        # Position bonus (near extreme = better entry)
        if is_near_high or is_near_low:
            priority += 20
        
        return priority
    
    def get_status(self) -> Dict:
        """Get current monitor status with detailed info"""
        max_monitored = self.config.get('max_monitored_coins', 1)
        top_coins = self.get_top_coins(limit=max_monitored)
        top_coin = top_coins[0] if top_coins else None
        
        # Get potential coins count from DB
        try:
            from storage.db_models import UTBotPotentialCoin, UTBotPaperTrade, get_session
            session = get_session()
            potential_count = session.query(UTBotPotentialCoin).count()
            open_trades_count = session.query(UTBotPaperTrade).filter_by(status='OPEN').count()
            session.close()
        except Exception as e:
            print(f"[UT BOT] Error getting counts from DB: {e}")
            potential_count = len(self.potential_coins)
            open_trades_count = len(self.open_trades)
        
        # Get last signal check info
        last_check_info = {}
        if top_coin and hasattr(self, '_last_check_result'):
            last_check_info = self._last_check_result
        
        return {
            'enabled': self.config['enabled'],
            'potential_coins': potential_count,
            'open_trades': open_trades_count,
            'top_coin': top_coin.to_dict() if top_coin else None,
            'top_coins': [c.to_dict() for c in top_coins],  # List of monitored coins
            'stats': self.stats,
            'config': {
                'timeframe': self.config['timeframe'],
                'atr_period': self.config['atr_period'],
                'atr_multiplier': self.config['atr_multiplier'],
                'use_heikin_ashi': self.config['use_heikin_ashi'],
                'max_monitored_coins': self.config.get('max_monitored_coins', 1),
            },
            # Additional status info
            'last_check': last_check_info,
            'bybit_symbols_cached': len(getattr(self, '_bybit_symbols', set())),
            'cooldowns_active': len(self._last_signal_time),
        }
    
    def get_potential_coins(self) -> List[Dict]:
        """Get all potential coins from database"""
        try:
            from storage.db_models import UTBotPotentialCoin, get_session
            session = get_session()
            coins = session.query(UTBotPotentialCoin).order_by(UTBotPotentialCoin.priority.desc()).all()
            result = [c.to_dict() for c in coins]
            session.close()
            return result
        except Exception as e:
            print(f"[UT BOT] Error getting potential coins from DB: {e}")
            # Fallback to memory
            coins = list(self.potential_coins.values())
            coins.sort(key=lambda x: x.priority, reverse=True)
            return [c.to_dict() for c in coins]
    
    def add_coin_manual(self, symbol: str, direction: str, score: float = 70.0) -> Dict:
        """
        Manually add a coin to potential coins (saves to DB)
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            direction: 'LONG' or 'SHORT'
            score: Priority score (default 70)
            
        Returns:
            Dict with result
        """
        symbol = symbol.upper()
        direction = direction.upper()
        
        if direction not in ['LONG', 'SHORT']:
            return {'success': False, 'error': 'Direction must be LONG or SHORT'}
        
        # Check if on Bybit
        if not self._is_symbol_on_bybit(symbol):
            return {'success': False, 'error': f'{symbol} not found on Bybit'}
        
        now = datetime.now()
        
        # Calculate priority
        priority = self._calculate_priority(score, 8, 70, False, False) + 20  # Boost for manual
        
        try:
            from storage.db_models import UTBotPotentialCoin, get_session
            session = get_session()
            
            # Check if exists
            existing = session.query(UTBotPotentialCoin).filter_by(symbol=symbol).first()
            if existing:
                # Update
                existing.direction = direction
                existing.sleeper_score = score
                existing.confidence = 70.0
                existing.priority = priority
                existing.source = 'MANUAL'
                existing.structure_type = 'MANUAL'
                existing.updated_at = now
            else:
                # Create new
                coin = UTBotPotentialCoin(
                    symbol=symbol,
                    direction=direction,
                    sleeper_score=score,
                    confidence=70.0,
                    priority=priority,
                    source='MANUAL',
                    structure_type='MANUAL',
                    is_near_extreme=False,
                    added_at=now
                )
                session.add(coin)
            
            session.commit()
            
            # Also add to memory for current worker
            self.potential_coins[symbol] = PotentialCoin(
                symbol=symbol,
                sleeper_score=score,
                direction=direction,
                structure_type='MANUAL',
                is_near_extreme=False,
                confidence=70.0,
                added_at=now,
                last_check=now,
                priority=priority
            )
            
            result = session.query(UTBotPotentialCoin).filter_by(symbol=symbol).first()
            coin_dict = result.to_dict() if result else self.potential_coins[symbol].to_dict()
            session.close()
            
            print(f"[UT BOT] ✅ Manually added to DB: {symbol} {direction} (score={score})")
            
            return {'success': True, 'coin': coin_dict}
            
        except Exception as e:
            print(f"[UT BOT] Error adding coin to DB: {e}")
            return {'success': False, 'error': str(e)}
    
    def remove_coin(self, symbol: str) -> Dict:
        """
        Remove a coin from potential coins (from DB)
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with result
        """
        symbol = symbol.upper()
        
        try:
            from storage.db_models import UTBotPotentialCoin, get_session
            session = get_session()
            
            coin = session.query(UTBotPotentialCoin).filter_by(symbol=symbol).first()
            if coin:
                session.delete(coin)
                session.commit()
                session.close()
                
                # Also remove from memory
                if symbol in self.potential_coins:
                    del self.potential_coins[symbol]
                
                print(f"[UT BOT] 🗑️ Removed from DB: {symbol}")
                return {'success': True, 'message': f'{symbol} removed'}
            else:
                session.close()
                return {'success': False, 'error': f'{symbol} not in potential coins'}
                
        except Exception as e:
            print(f"[UT BOT] Error removing coin from DB: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades from database"""
        try:
            from storage.db_models import UTBotPaperTrade, get_session
            session = get_session()
            trades = session.query(UTBotPaperTrade).filter_by(status='OPEN').all()
            result = [t.to_dict() for t in trades]
            session.close()
            return result
        except Exception as e:
            print(f"[UT BOT] Error getting open trades from DB: {e}")
            # Fallback to memory
            return [t.to_dict() for t in self.open_trades.values()]
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history from database"""
        try:
            from storage.db_models import UTBotPaperTrade, get_session
            session = get_session()
            trades = session.query(UTBotPaperTrade).filter_by(
                status='CLOSED'
            ).order_by(UTBotPaperTrade.closed_at.desc()).limit(limit).all()
            result = [t.to_dict() for t in trades]
            session.close()
            return result
        except Exception as e:
            print(f"[UT BOT] Error getting trade history from DB: {e}")
            # Fallback to memory
            return [t.to_dict() for t in self.trade_history[-limit:]]
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable monitor"""
        self.config['enabled'] = enabled
    
    def update_config(self, updates: Dict) -> None:
        """Update configuration"""
        for key, value in updates.items():
            if key in self.config:
                self.config[key] = value
        
        # Update UT Bot config
        self.ut_bot.set_config('key_value', self.config['atr_multiplier'])
        self.ut_bot.set_config('atr_period', self.config['atr_period'])
        self.ut_bot.set_config('use_heikin_ashi', self.config['use_heikin_ashi'])
        self.ut_bot.set_config('timeframe', self.config['timeframe'])


# Factory
_monitor_instance = None

def get_ut_bot_monitor(config: Dict = None) -> UTBotMonitor:
    """Get singleton instance of UT Bot Monitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = UTBotMonitor(config)
    return _monitor_instance
