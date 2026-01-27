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
        
        # Filters - relaxed for more candidates
        'min_sleeper_score': 60,      # Lowered for more candidates
        'min_sleeper_hp': 4,          # Lowered for more candidates
        'require_structure': False,   # Allow all positions for testing
        
        # Trading
        'max_open_trades': 3,
        'position_size_usdt': 100,    # Paper trading size
        'min_signal_gap_minutes': 15, # Reduced cooldown
        
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
            self.config['timeframe'] = tf
            
            atr_p = self.db.get_setting('ut_bot_atr_period', str(self.config['atr_period']))
            self.config['atr_period'] = int(atr_p) if atr_p else 10
            
            atr_m = self.db.get_setting('ut_bot_atr_multiplier', str(self.config['atr_multiplier']))
            self.config['atr_multiplier'] = float(atr_m) if atr_m else 1.0
            
            # Load booleans - support both '1'/'0' and 'true'/'false'
            ha = self.db.get_setting('ut_bot_use_heikin_ashi', '0')  # Default OFF
            self.config['use_heikin_ashi'] = ha in ('1', 'true', 'True', True)
            
            enabled = self.db.get_setting('ut_bot_enabled', '1')  # Default ON
            self.config['enabled'] = enabled in ('1', 'true', 'True', True)
            
            print(f"[UT BOT] Loaded config from DB: TF={self.config['timeframe']}, ATR={self.config['atr_period']}/{self.config['atr_multiplier']}, HA={self.config['use_heikin_ashi']}, enabled={self.config['enabled']}")
        except Exception as e:
            print(f"[UT BOT] Config load error (using defaults): {e}")
    
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
            
            # Add coin
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
        
        # Summary
        print(f"[UT BOT] ═══════════════════════════════════")
        print(f"[UT BOT] Total: {stats['total']} | Rejected: score={stats['low_score']}, hp={stats['low_hp']}, state={stats['wrong_state']}, no_dir={stats['no_direction_found']}, bybit={stats['not_on_bybit']}")
        print(f"[UT BOT] ✅ PASSED: {updated} (sleeper={stats['passed_sleeper_dir']}, structure={stats['passed_structure_dir']}, mss={stats['passed_mss_dir']})")
        print(f"[UT BOT] Potential coins in memory: {len(self.potential_coins)}")
        print(f"[UT BOT] ═══════════════════════════════════")
        
        # Cleanup old
        cutoff = now - timedelta(minutes=60)
        to_remove = [s for s, c in self.potential_coins.items() if c.last_check and c.last_check < cutoff]
        for s in to_remove:
            del self.potential_coins[s]
        
        return updated
    
    def get_top_coin(self) -> Optional[PotentialCoin]:
        """
        Get the TOP priority coin for trading
        
        Selection criteria:
        1. Must have open position slot
        2. Must be near HH or LL
        3. Highest priority score
        """
        if not self.potential_coins:
            return None
        
        # Filter out coins with open trades
        available = [c for s, c in self.potential_coins.items()
                     if s not in self.open_trades]
        
        if not available:
            return None
        
        # Sort by priority
        available.sort(key=lambda x: x.priority, reverse=True)
        return available[0]
    
    def check_signals(self) -> List[Dict]:
        """
        Check UT Bot signals for potential coins
        
        Returns:
            List of signal events
        """
        if not self.config['enabled']:
            self._last_check_result = {'status': 'disabled'}
            return []
        
        events = []
        now = datetime.now()
        
        # Initialize last check result
        self._last_check_result = {
            'timestamp': now.isoformat(),
            'status': 'checked',
            'top_coin': None,
            'signal': None,
            'action': 'NONE',
            'error': None
        }
        
        try:
            # =====================================================
            # 1. CHECK FOR OPEN SIGNALS ON TOP COIN
            # =====================================================
            top_coin = self.get_top_coin()
            
            if top_coin:
                self._last_check_result['top_coin'] = top_coin.symbol
                
                # Log which coin we're checking
                print(f"[UT BOT] Checking signals for TOP coin: {top_coin.symbol} ({top_coin.direction}, score={top_coin.sleeper_score})")
                
                # Check cooldown
                last_signal = self._last_signal_time.get(top_coin.symbol)
                skip_signal_check = False
                
                if last_signal:
                    elapsed = (now - last_signal).total_seconds()
                    if elapsed < self.config['min_signal_gap_minutes'] * 60:
                        print(f"[UT BOT] {top_coin.symbol} on cooldown ({elapsed:.0f}s < {self.config['min_signal_gap_minutes']*60}s)")
                        self._last_check_result['action'] = f'COOLDOWN ({int(elapsed)}s)'
                        skip_signal_check = True
                
                if not skip_signal_check:
                    try:
                        # Check UT Bot signal
                        signal_result = self.ut_bot.check_signal_with_bias(
                            top_coin.symbol,
                            top_coin.direction,
                            timeframe=self.config['timeframe']
                        )
                        
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
                        
                        print(f"[UT BOT] {top_coin.symbol}: pos={prev_pos}→{pos} | price={price:.6f} | stop={atr_stop:.6f} ({dist_dir} {distance_pct:.2f}%) | bar={bar_color}")
                        
                        # Process OPEN signal (aligned with bias)
                        trade_action = signal_result.get('trade_action', 'HOLD')
                        if signal_result.get('aligned') and trade_action.startswith('ENTER'):
                            # Open trade
                            trade = self._open_trade(top_coin, signal_result)
                            if trade:
                                events.append({
                                    'type': 'TRADE_OPENED',
                                    'trade': trade.to_dict(),
                                    'signal': signal_result
                                })
                                self._last_signal_time[top_coin.symbol] = now
                                print(f"[UT BOT] ✅ TRADE OPENED: {top_coin.symbol} {trade_action}")
                    
                    except Exception as e:
                        print(f"[UT BOT] Error checking signals for {top_coin.symbol}: {e}")
                        self._last_check_result['error'] = str(e)
                        self._last_check_result['action'] = f'ERROR'
            else:
                self._last_check_result['action'] = 'NO_COINS'
                print("[UT BOT] No potential coins available for signal check")
            
            # =====================================================
            # 2. MONITOR OPEN TRADES FOR EXIT SIGNALS
            # =====================================================
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
        """Open a paper trade"""
        if len(self.open_trades) >= self.config['max_open_trades']:
            return None
        
        self._trade_counter += 1
        
        trade = UTBotTrade(
            id=self._trade_counter,
            symbol=coin.symbol,
            direction=coin.direction,
            status='OPEN',
            entry_price=signal.get('price', 0),
            atr_stop=signal.get('atr_trailing_stop', 0),
            highest_price=signal.get('price', 0) if coin.direction == 'LONG' else None,
            lowest_price=signal.get('price', 0) if coin.direction == 'SHORT' else None,
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
        
        return trade
    
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
        """Close a paper trade"""
        trade.status = 'CLOSED'
        trade.exit_price = exit_signal.get('price', trade.current_price or trade.entry_price)
        trade.exit_signal = exit_signal
        trade.closed_at = datetime.now()
        
        # Calculate PnL
        if trade.direction == 'LONG':
            trade.pnl_percent = (trade.exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            trade.pnl_percent = (trade.entry_price - trade.exit_price) / trade.entry_price * 100
        
        trade.pnl_usdt = self.config['position_size_usdt'] * trade.pnl_percent / 100
        
        # Update stats
        self.stats['total_pnl'] += trade.pnl_usdt
        if trade.pnl_usdt > 0:
            self.stats['winning_trades'] += 1
        else:
            self.stats['losing_trades'] += 1
        
        # Move to history
        self.trade_history.append(trade)
        del self.open_trades[trade.symbol]
        
        # Log event
        self.db.log_event(
            category='UT_BOT',
            symbol=trade.symbol,
            message=f"Paper trade closed: PnL ${trade.pnl_usdt:.2f} ({trade.pnl_percent:.2f}%)"
        )
        
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
        top_coin = self.get_top_coin()
        
        # Get last signal check info
        last_check_info = {}
        if top_coin and hasattr(self, '_last_check_result'):
            last_check_info = self._last_check_result
        
        return {
            'enabled': self.config['enabled'],
            'potential_coins': len(self.potential_coins),
            'open_trades': len(self.open_trades),
            'top_coin': top_coin.to_dict() if top_coin else None,
            'stats': self.stats,
            'config': {
                'timeframe': self.config['timeframe'],
                'atr_period': self.config['atr_period'],
                'atr_multiplier': self.config['atr_multiplier'],
                'use_heikin_ashi': self.config['use_heikin_ashi'],
            },
            # Additional status info
            'last_check': last_check_info,
            'bybit_symbols_cached': len(getattr(self, '_bybit_symbols', set())),
            'cooldowns_active': len(self._last_signal_time),
        }
    
    def get_potential_coins(self) -> List[Dict]:
        """Get all potential coins"""
        coins = list(self.potential_coins.values())
        coins.sort(key=lambda x: x.priority, reverse=True)
        return [c.to_dict() for c in coins]
    
    def add_coin_manual(self, symbol: str, direction: str, score: float = 70.0) -> Dict:
        """
        Manually add a coin to potential coins
        
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
        priority = self._calculate_priority(score, 8, 70, False, False)
        
        # Add to potential coins with high priority for manual
        self.potential_coins[symbol] = PotentialCoin(
            symbol=symbol,
            sleeper_score=score,
            direction=direction,
            structure_type='MANUAL',
            is_near_extreme=False,
            confidence=70.0,
            added_at=now,
            last_check=now,
            priority=priority + 20  # Boost priority for manual adds
        )
        
        print(f"[UT BOT] ✅ Manually added: {symbol} {direction} (score={score})")
        
        return {
            'success': True,
            'coin': self.potential_coins[symbol].to_dict()
        }
    
    def remove_coin(self, symbol: str) -> Dict:
        """
        Remove a coin from potential coins
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with result
        """
        symbol = symbol.upper()
        
        if symbol in self.potential_coins:
            del self.potential_coins[symbol]
            print(f"[UT BOT] 🗑️ Removed: {symbol}")
            return {'success': True, 'message': f'{symbol} removed'}
        else:
            return {'success': False, 'error': f'{symbol} not in potential coins'}
    
    def get_open_trades(self) -> List[Dict]:
        """Get all open trades"""
        return [t.to_dict() for t in self.open_trades.values()]
    
    def get_trade_history(self, limit: int = 50) -> List[Dict]:
        """Get trade history"""
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
