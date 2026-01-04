"""
Signal Merger - Integrates Sleeper and OB signals
Handles signal generation and execution modes
"""
from typing import Dict, List, Optional, Callable
from datetime import datetime
from config import ExecutionMode
from storage import get_db
from detection.sleeper_scanner import get_sleeper_scanner
from detection.ob_scanner import get_ob_scanner

class SignalMerger:
    """
    Signal Integration Module
    
    Combines Sleeper Detector and Order Block Detector signals
    Manages execution modes: AUTO, SEMI_AUTO, MANUAL
    """
    
    def __init__(self):
        self.db = get_db()
        self.sleeper_scanner = get_sleeper_scanner()
        self.ob_scanner = get_ob_scanner()
        self._signal_callback: Optional[Callable] = None
        self._pending_signals: List[Dict] = []
    
    def set_signal_callback(self, callback: Callable):
        """Set callback for new signals (used by Telegram notifier)"""
        self._signal_callback = callback
    
    def check_for_signals(self) -> List[Dict]:
        """
        Main signal generation loop
        Check READY sleepers for OB entry signals
        """
        signals = []
        
        # Get ready sleepers
        ready_sleepers = self.db.get_sleepers(state='READY')
        
        for sleeper in ready_sleepers:
            symbol = sleeper['symbol']
            direction = sleeper['direction']
            
            if direction == 'NEUTRAL':
                continue
            
            # Scan for order blocks
            obs = self.ob_scanner.scan_symbol(symbol)
            
            # Check if price is at any OB
            signal = self.ob_scanner.get_entry_signal(symbol, direction)
            
            if signal:
                # Enrich signal with sleeper data
                signal['sleeper_score'] = sleeper['total_score']
                signal['sleeper_hp'] = sleeper['hp']
                signal['signal_confidence'] = self._calculate_confidence(sleeper, signal)
                signal['timestamp'] = datetime.utcnow().isoformat()
                
                signals.append(signal)
                
                # Update sleeper state
                self.db.update_sleeper_state(symbol, 'TRIGGERED', direction)
                
                # Log event
                self.db.log_event(
                    f"🚀 SIGNAL: {symbol} {direction} @ {signal['entry_price']:.4f}",
                    level='SUCCESS', category='SIGNAL', symbol=symbol
                )
        
        # Process signals based on execution mode
        if signals:
            self._process_signals(signals)
        
        return signals
    
    def _calculate_confidence(self, sleeper: Dict, signal: Dict) -> float:
        """Calculate overall signal confidence (0-100)"""
        # Sleeper score (40%)
        sleeper_component = (sleeper['total_score'] / 100) * 40
        
        # OB quality (40%)
        ob_component = (signal['ob_quality'] / 100) * 40
        
        # HP bonus (10%)
        hp_component = (sleeper['hp'] / 10) * 10
        
        # Distance penalty (10%)
        distance = signal.get('distance_percent', 0)
        if distance < 0.5:
            distance_component = 10
        elif distance < 1:
            distance_component = 7
        elif distance < 2:
            distance_component = 4
        else:
            distance_component = 0
        
        return round(sleeper_component + ob_component + hp_component + distance_component, 2)
    
    def _process_signals(self, signals: List[Dict]):
        """Process signals based on execution mode"""
        execution_mode = self.db.get_setting('execution_mode', 'SEMI_AUTO')
        
        for signal in signals:
            signal['execution_mode'] = execution_mode
            
            if execution_mode == ExecutionMode.AUTO.value:
                # Execute immediately
                self._execute_signal(signal)
                
            elif execution_mode == ExecutionMode.SEMI_AUTO.value:
                # Send notification and wait for confirmation
                self._pending_signals.append(signal)
                if self._signal_callback:
                    self._signal_callback(signal, requires_confirmation=True)
                    
            else:  # MANUAL
                # Just send notification
                if self._signal_callback:
                    self._signal_callback(signal, requires_confirmation=False)
    
    def _execute_signal(self, signal: Dict):
        """Execute a trading signal"""
        # Import here to avoid circular import
        from trading.order_executor import get_executor
        
        executor = get_executor()
        result = executor.execute_signal(signal)
        
        if result['success']:
            self.db.log_event(
                f"✅ Executed {signal['direction']} on {signal['symbol']}",
                level='SUCCESS', category='TRADE', symbol=signal['symbol']
            )
        else:
            self.db.log_event(
                f"❌ Failed to execute {signal['symbol']}: {result.get('error')}",
                level='ERROR', category='TRADE', symbol=signal['symbol']
            )
    
    def confirm_signal(self, symbol: str) -> Dict:
        """Confirm a pending signal (for SEMI_AUTO mode)"""
        for i, signal in enumerate(self._pending_signals):
            if signal['symbol'] == symbol:
                self._execute_signal(signal)
                self._pending_signals.pop(i)
                return {'success': True, 'signal': signal}
        
        return {'success': False, 'error': 'Signal not found'}
    
    def reject_signal(self, symbol: str) -> Dict:
        """Reject a pending signal"""
        for i, signal in enumerate(self._pending_signals):
            if signal['symbol'] == symbol:
                self._pending_signals.pop(i)
                self.db.log_event(
                    f"Signal rejected for {symbol}",
                    level='WARN', category='SIGNAL', symbol=symbol
                )
                return {'success': True}
        
        return {'success': False, 'error': 'Signal not found'}
    
    def get_pending_signals(self) -> List[Dict]:
        """Get all pending signals"""
        return self._pending_signals.copy()
    
    def get_ready_for_entry(self) -> List[Dict]:
        """
        Get all ready-for-entry signals (Sleeper READY + OB touched)
        For dashboard display
        """
        ready = []
        ready_sleepers = self.db.get_sleepers(state='READY')
        
        for sleeper in ready_sleepers:
            symbol = sleeper['symbol']
            
            # Get active OBs for this symbol
            obs = self.db.get_orderblocks(symbol=symbol, status='ACTIVE')
            
            if not obs:
                continue
            
            # Find best matching OB
            best_ob = None
            for ob in obs:
                if (sleeper['direction'] == 'LONG' and ob['ob_type'] == 'BULLISH') or \
                   (sleeper['direction'] == 'SHORT' and ob['ob_type'] == 'BEARISH'):
                    if best_ob is None or ob['quality_score'] > best_ob['quality_score']:
                        best_ob = ob
            
            if best_ob:
                from core import get_fetcher
                current_price = get_fetcher().get_current_price(symbol) or 0
                
                ready.append({
                    'symbol': symbol,
                    'score': sleeper['total_score'],
                    'direction': sleeper['direction'],
                    'ob_low': best_ob['ob_low'],
                    'ob_high': best_ob['ob_high'],
                    'ob_quality': best_ob['quality_score'],
                    'current_price': current_price,
                    'distance_percent': abs(current_price - best_ob['ob_mid']) / best_ob['ob_mid'] * 100 if best_ob['ob_mid'] > 0 else 0,
                })
        
        return ready
    
    def run_full_scan(self) -> Dict:
        """Run full scan cycle: Sleepers -> OBs -> Signals"""
        results = {
            'sleepers_found': 0,
            'obs_found': 0,
            'signals_generated': 0,
        }
        
        # 1. Run sleeper scan
        max_symbols = self.db.get_setting('sleeper_max_symbols', 100)
        min_volume = self.db.get_setting('sleeper_min_volume', 20000000)
        sleepers = self.sleeper_scanner.scan(max_symbols, min_volume)
        results['sleepers_found'] = len(sleepers)
        
        # 2. Scan OBs for ready sleepers
        ready_sleepers = self.db.get_sleepers(state='READY')
        total_obs = 0
        
        for sleeper in ready_sleepers:
            obs = self.ob_scanner.scan_symbol(sleeper['symbol'])
            total_obs += len(obs)
        
        results['obs_found'] = total_obs
        
        # 3. Check for signals
        signals = self.check_for_signals()
        results['signals_generated'] = len(signals)
        
        # 4. Cleanup
        self.ob_scanner.cleanup_expired()
        
        return results


# Singleton instance
_merger_instance = None

def get_signal_merger() -> SignalMerger:
    """Get signal merger instance"""
    global _merger_instance
    if _merger_instance is None:
        _merger_instance = SignalMerger()
    return _merger_instance
