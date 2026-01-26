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
        'enabled': True,
        'timeframe': '15m',           # UT Bot timeframe
        'atr_period': 10,
        'atr_multiplier': 1.0,
        'use_heikin_ashi': True,
        
        # Filters
        'min_sleeper_score': 65,
        'min_sleeper_hp': 6,
        'require_structure': True,     # Must be HH/LL not middle
        
        # Trading
        'max_open_trades': 3,
        'position_size_usdt': 100,     # Paper trading size
        'min_signal_gap_minutes': 30,
        
        # Checks
        'check_interval_seconds': 60,
        'max_trade_duration_hours': 24,
    }
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize UT Bot Monitor"""
        self.config = {**self.DEFAULT_CONFIG}
        if config:
            self.config.update(config)
        
        self.ut_bot = get_ut_bot_filter({
            'key_value': self.config['atr_multiplier'],
            'atr_period': self.config['atr_period'],
            'use_heikin_ashi': self.config['use_heikin_ashi'],
            'timeframe': self.config['timeframe'],
        })
        
        self.direction_engine = get_direction_engine_v7()
        self.bybit = get_connector()
        self.db = get_db()
        
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
    
    def update_from_sleepers(self, sleepers: List[Dict]) -> int:
        """
        Update potential coins from Sleeper Scanner results
        
        Args:
            sleepers: List of sleeper data from scanner
            
        Returns:
            Number of coins added/updated
        """
        if not self.config['enabled']:
            return 0
        
        updated = 0
        now = datetime.now()
        
        for sleeper in sleepers:
            symbol = sleeper.get('symbol', '')
            if not symbol:
                continue
            
            # Check minimum requirements
            score = sleeper.get('total_score', 0)
            hp = sleeper.get('hp', 0)
            state = sleeper.get('state', '')
            
            if score < self.config['min_sleeper_score']:
                continue
            if hp < self.config['min_sleeper_hp']:
                continue
            if state not in ['READY', 'BUILDING', 'TRIGGERED']:
                continue
            
            # Check direction and structure
            direction_data = sleeper.get('direction_data', {})
            if not direction_data:
                # Try to get direction from v7 if not provided
                direction_data = self._analyze_direction(symbol)
            
            if not direction_data:
                continue
            
            direction = direction_data.get('direction', 'NEUTRAL')
            if direction == 'NEUTRAL':
                continue
            
            structure = direction_data.get('structure', {})
            is_near_high = structure.get('is_near_high', False)
            is_near_low = structure.get('is_near_low', False)
            is_in_middle = structure.get('is_in_middle', True)
            
            # CRITICAL: Only accept coins near HH or LL (not in middle)
            if self.config['require_structure'] and is_in_middle:
                continue
            
            # Determine structure type
            structure_type = 'UNKNOWN'
            if structure.get('dominant', 'UNKNOWN') in ['HH', 'HL']:
                structure_type = 'HH' if is_near_high else 'HL'
            elif structure.get('dominant', 'UNKNOWN') in ['LH', 'LL']:
                structure_type = 'LL' if is_near_low else 'LH'
            
            # Check if on Bybit
            if not self._is_symbol_on_bybit(symbol):
                continue
            
            # Calculate priority
            priority = self._calculate_priority(
                score, hp, direction_data.get('confidence', 0),
                is_near_high, is_near_low
            )
            
            # Add or update
            self.potential_coins[symbol] = PotentialCoin(
                symbol=symbol,
                sleeper_score=score,
                direction=direction,
                structure_type=structure_type,
                is_near_extreme=is_near_high or is_near_low,
                confidence=direction_data.get('confidence', 0),
                added_at=self.potential_coins.get(symbol, PotentialCoin(
                    symbol=symbol, sleeper_score=0, direction='', 
                    structure_type='', is_near_extreme=False,
                    confidence=0, added_at=now
                )).added_at,
                last_check=now,
                priority=priority
            )
            updated += 1
        
        # Remove old coins (not updated in last 30 min)
        cutoff = now - timedelta(minutes=30)
        to_remove = [s for s, c in self.potential_coins.items() 
                     if c.last_check and c.last_check < cutoff]
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
            return []
        
        events = []
        now = datetime.now()
        
        # Get top coin to monitor
        top_coin = self.get_top_coin()
        
        if top_coin:
            # Check cooldown
            last_signal = self._last_signal_time.get(top_coin.symbol)
            if last_signal:
                elapsed = (now - last_signal).total_seconds()
                if elapsed < self.config['min_signal_gap_minutes'] * 60:
                    return events
            
            # Check UT Bot signal
            signal_result = self.ut_bot.check_signal_with_bias(
                top_coin.symbol,
                top_coin.direction,
                timeframe=self.config['timeframe']
            )
            
            # Process signal
            if signal_result.get('aligned') and signal_result.get('action', '').startswith('ENTER'):
                # Open trade
                trade = self._open_trade(top_coin, signal_result)
                if trade:
                    events.append({
                        'type': 'TRADE_OPENED',
                        'trade': trade.to_dict(),
                        'signal': signal_result
                    })
                    self._last_signal_time[top_coin.symbol] = now
        
        # Monitor open trades for exit
        for symbol, trade in list(self.open_trades.items()):
            exit_signal = self._check_exit_signal(trade)
            if exit_signal:
                closed_trade = self._close_trade(trade, exit_signal)
                if closed_trade:
                    events.append({
                        'type': 'TRADE_CLOSED',
                        'trade': closed_trade.to_dict(),
                        'signal': exit_signal
                    })
        
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
        
        # 1. Opposite signal
        if trade.direction == 'LONG' and signal.signal == UTSignalType.SELL:
            should_exit = True
            exit_reason = 'UT_SELL_SIGNAL'
        elif trade.direction == 'SHORT' and signal.signal == UTSignalType.BUY:
            should_exit = True
            exit_reason = 'UT_BUY_SIGNAL'
        
        # 2. Max duration
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
        """Check if symbol is available on Bybit"""
        try:
            # Simple check - try to get ticker
            ticker = self.bybit.get_ticker(symbol)
            return ticker is not None
        except:
            return False
    
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
        """Get current monitor status"""
        return {
            'enabled': self.config['enabled'],
            'potential_coins': len(self.potential_coins),
            'open_trades': len(self.open_trades),
            'top_coin': self.get_top_coin().to_dict() if self.get_top_coin() else None,
            'stats': self.stats,
            'config': {
                'timeframe': self.config['timeframe'],
                'atr_period': self.config['atr_period'],
                'atr_multiplier': self.config['atr_multiplier'],
                'use_heikin_ashi': self.config['use_heikin_ashi'],
            }
        }
    
    def get_potential_coins(self) -> List[Dict]:
        """Get all potential coins"""
        coins = list(self.potential_coins.values())
        coins.sort(key=lambda x: x.priority, reverse=True)
        return [c.to_dict() for c in coins]
    
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
