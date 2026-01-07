"""
Signal Merger - Integrates Sleeper and OB signals with Trend Context
Handles signal generation and execution modes with 4H trend filtering
"""
from typing import Dict, List, Optional, Callable
from datetime import datetime
from config import ExecutionMode
from storage import get_db
from detection.sleeper_scanner import get_sleeper_scanner
from detection.ob_scanner import get_ob_scanner
from detection.trend_analyzer import get_trend_analyzer, TrendRegime


class SignalMerger:
    """
    Signal Integration Module with Trend Context
    
    Combines:
    - Sleeper Detector (accumulation detection)
    - Order Block Detector (entry zones)
    - Trend Analyzer (4H context filter)
    
    Signal flow:
    1. Check 4H trend regime
    2. Filter sleepers by trend direction
    3. Find OB entry signals
    4. Execute based on mode
    """
    
    def __init__(self):
        self.db = get_db()
        self.sleeper_scanner = get_sleeper_scanner()
        self.ob_scanner = get_ob_scanner()
        self.trend_analyzer = get_trend_analyzer()
        self._signal_callback: Optional[Callable] = None
        self._pending_signals: List[Dict] = []
    
    def set_signal_callback(self, callback: Callable):
        """Set callback for new signals (used by Telegram notifier)"""
        self._signal_callback = callback
    
    def check_for_signals(self) -> List[Dict]:
        """
        Main signal generation loop with trend filtering
        
        Logic:
        1. Get ready sleepers
        2. Check 4H trend for each symbol
        3. Only allow signals aligned with trend
        4. Find OB entry signals
        """
        signals = []
        filtered_count = 0
        
        # Get trend filter settings
        use_trend_filter = self.db.get_setting('use_trend_filter', True)
        trend_timeframe = self.db.get_setting('trend_timeframe', '240')  # 4H default
        min_trend_score = self.db.get_setting('min_trend_score', 65)
        
        # Get ready sleepers
        ready_sleepers = self.db.get_sleepers(state='READY')
        
        for sleeper in ready_sleepers:
            symbol = sleeper['symbol']
            direction = sleeper['direction']
            
            if direction == 'NEUTRAL':
                continue
            
            # ========================================
            # TREND FILTER (4H Context)
            # ========================================
            if use_trend_filter:
                trend_result = self.trend_analyzer.analyze(symbol, trend_timeframe)
                
                if trend_result:
                    # Check if direction is allowed
                    regime = trend_result.regime
                    trend_score = trend_result.total_score
                    
                    # Log trend analysis
                    self.db.log_event(
                        f"[TREND] {symbol}: {regime.value} (score={trend_score:.1f})",
                        level='DEBUG', category='TREND', symbol=symbol
                    )
                    
                    # Filter by regime
                    if regime == TrendRegime.NO_TRADE:
                        filtered_count += 1
                        continue
                    
                    if regime == TrendRegime.BULLISH and direction != 'LONG':
                        filtered_count += 1
                        self.db.log_event(
                            f"[TREND] {symbol}: SHORT blocked (4H is BULLISH)",
                            level='DEBUG', category='TREND', symbol=symbol
                        )
                        continue
                    
                    if regime == TrendRegime.BEARISH and direction != 'SHORT':
                        filtered_count += 1
                        self.db.log_event(
                            f"[TREND] {symbol}: LONG blocked (4H is BEARISH)",
                            level='DEBUG', category='TREND', symbol=symbol
                        )
                        continue
                    
                    # Additional score check
                    if trend_score < min_trend_score:
                        filtered_count += 1
                        continue
                else:
                    # No trend data - skip or allow based on setting
                    allow_no_trend = self.db.get_setting('allow_signals_without_trend', False)
                    if not allow_no_trend:
                        filtered_count += 1
                        continue
            
            # ========================================
            # OB SIGNAL CHECK
            # ========================================
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
                
                # Add trend data if available
                if use_trend_filter and trend_result:
                    signal['trend_score'] = trend_result.total_score
                    signal['trend_regime'] = trend_result.regime.value
                    signal['trend_details'] = {
                        'structure': trend_result.structure_score,
                        'volatility': trend_result.volatility_score,
                        'acceptance': trend_result.acceptance_score,
                        'momentum': trend_result.momentum_score
                    }
                
                signals.append(signal)
                
                # Update sleeper state
                self.db.update_sleeper_state(symbol, 'TRIGGERED', direction)
                
                # Log event
                trend_info = f" [Trend: {trend_result.regime.value}]" if use_trend_filter and trend_result else ""
                self.db.log_event(
                    f"🚀 SIGNAL: {symbol} {direction} @ {signal['entry_price']:.4f}{trend_info}",
                    level='SUCCESS', category='SIGNAL', symbol=symbol
                )
        
        # Log filter stats
        if filtered_count > 0:
            self.db.log_event(
                f"[TREND] Filtered {filtered_count} signals by trend context",
                level='DEBUG', category='TREND'
            )
        
        # Process signals based on execution mode
        if signals:
            self._process_signals(signals)
        
        return signals
    
    def get_trend_for_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Get current trend analysis for a symbol.
        For dashboard display.
        """
        trend_timeframe = self.db.get_setting('trend_timeframe', '240')
        result = self.trend_analyzer.analyze(symbol, trend_timeframe)
        
        if result:
            return {
                'symbol': symbol,
                'timeframe': result.timeframe,
                'total_score': result.total_score,
                'regime': result.regime.value,
                'direction': result.overall_direction.value,
                'components': {
                    'structure': {
                        'score': result.structure_score,
                        'direction': result.structure_direction.value,
                        'weight': '30%'
                    },
                    'volatility': {
                        'score': result.volatility_score,
                        'weight': '25%'
                    },
                    'acceptance': {
                        'score': result.acceptance_score,
                        'weight': '25%'
                    },
                    'momentum': {
                        'score': result.momentum_score,
                        'weight': '20%'
                    }
                },
                'details': result.details,
                'calculated_at': result.calculated_at.isoformat()
            }
        return None
    
    def analyze_trends_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """Analyze trends for multiple symbols"""
        results = {}
        for symbol in symbols:
            trend = self.get_trend_for_symbol(symbol)
            if trend:
                results[symbol] = trend
        return results
    
    def _calculate_confidence(self, sleeper: Dict, signal: Dict) -> float:
        """Calculate overall signal confidence (0-100)"""
        # Sleeper score (40%)
        sleeper_component = (sleeper['total_score'] / 100) * 40
        
        # OB quality (40%)
        ob_component = (signal.get('ob_quality', 50) / 100) * 40
        
        # HP bonus (10%)
        hp_component = (sleeper.get('hp', 5) / 10) * 10
        
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
        For dashboard display. Includes trend info.
        """
        ready = []
        ready_sleepers = self.db.get_sleepers(state='READY')
        
        # Get trend filter setting
        use_trend_filter = self.db.get_setting('use_trend_filter', True)
        
        for sleeper in ready_sleepers:
            symbol = sleeper['symbol']
            
            # Get active OBs for this symbol
            obs = self.db.get_orderblocks(symbol=symbol, status='ACTIVE')
            
            if not obs:
                continue
            
            # Find best matching OB
            best_ob = None
            for ob in obs:
                if (sleeper['direction'] == 'LONG' and ob['ob_type'] == 'LONG') or \
                   (sleeper['direction'] == 'SHORT' and ob['ob_type'] == 'SHORT'):
                    if best_ob is None or ob['quality_score'] > best_ob['quality_score']:
                        best_ob = ob
            
            if best_ob:
                from core import get_fetcher
                current_price = get_fetcher().get_current_price(symbol) or 0
                
                entry = {
                    'symbol': symbol,
                    'score': sleeper['total_score'],
                    'direction': sleeper['direction'],
                    'ob_low': best_ob['ob_low'],
                    'ob_high': best_ob['ob_high'],
                    'ob_quality': best_ob['quality_score'],
                    'current_price': current_price,
                    'distance_percent': abs(current_price - best_ob['ob_mid']) / best_ob['ob_mid'] * 100 if best_ob['ob_mid'] > 0 else 0,
                }
                
                # Add trend info
                if use_trend_filter:
                    trend = self.get_trend_for_symbol(symbol)
                    if trend:
                        entry['trend_score'] = trend['total_score']
                        entry['trend_regime'] = trend['regime']
                        entry['trend_allowed'] = (
                            (trend['regime'] == 'BULLISH' and sleeper['direction'] == 'LONG') or
                            (trend['regime'] == 'BEARISH' and sleeper['direction'] == 'SHORT')
                        )
                    else:
                        entry['trend_allowed'] = False
                else:
                    entry['trend_allowed'] = True
                
                ready.append(entry)
        
        return ready
    
    def run_full_scan(self) -> Dict:
        """Run full scan cycle: Sleepers -> OBs -> Signals"""
        results = {
            'sleepers_found': 0,
            'obs_found': 0,
            'signals_generated': 0,
            'filtered_by_trend': 0,
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
        
        # 3. Check for signals (with trend filtering)
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
