"""
Order Executor - Execute trades (paper and live)
Торгівля на Bybit, сканування на Binance
"""
from typing import Dict, Optional
from datetime import datetime
from config import TRADING_CONSTANTS
from core import get_connector  # Bybit для торгівлі
from core.binance_connector import get_binance_connector  # Binance для цін (альтернатива)
from storage import get_db
from trading.risk_calculator import get_risk_calculator
from trading.position_tracker import get_position_tracker

class OrderExecutor:
    """
    Trade Execution Module
    
    Handles:
    - Paper trade execution
    - Live trade execution
    - Order validation
    - Position opening/closing
    
    NOTE: Торгівля виконується на Bybit
          Ціни беруться з Bybit для точності
    """
    
    def __init__(self):
        self.db = get_db()
        self.connector = get_connector()  # Bybit для торгівлі
        self.risk_calc = get_risk_calculator()
        self.position_tracker = get_position_tracker()
        self.constants = TRADING_CONSTANTS
    
    def execute_signal(self, signal: Dict) -> Dict:
        """
        Execute a trading signal
        
        Signal should contain:
        - symbol
        - direction (LONG/SHORT)
        - entry_price
        - stop_loss
        - take_profit (optional)
        - sleeper_score (optional)
        - ob_quality (optional)
        """
        symbol = signal['symbol']
        direction = signal['direction']
        # Отримуємо ціну з Bybit (де торгуємо)
        entry_price = signal.get('entry_price') or self.connector.get_price(symbol)
        stop_loss = signal['stop_loss']
        
        if not entry_price:
            return {'success': False, 'error': 'Could not get entry price'}
        
        # Calculate position size
        sizing = self.risk_calc.calculate_position_size(
            symbol=symbol,
            entry_price=entry_price,
            stop_loss=stop_loss,
            direction=direction
        )
        
        if not sizing['success']:
            return sizing
        
        position_size = sizing['position_size']
        leverage = sizing['leverage']
        
        # Calculate TP levels
        tp_levels = self.risk_calc.calculate_tp_levels(entry_price, stop_loss, direction)
        
        # Validate trade
        validation = self.risk_calc.validate_trade(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size
        )
        
        if not validation['valid']:
            return {
                'success': False,
                'error': '; '.join(validation['errors']),
                'warnings': validation['warnings']
            }
        
        is_paper = self.db.get_setting('paper_trading', True)
        
        if is_paper:
            result = self._execute_paper_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                tp_levels=tp_levels,
                signal=signal
            )
        else:
            result = self._execute_live_trade(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                position_size=position_size,
                leverage=leverage,
                stop_loss=stop_loss,
                tp_levels=tp_levels,
                signal=signal
            )
        
        return result
    
    def _execute_paper_trade(self, symbol: str, direction: str, entry_price: float,
                             position_size: float, leverage: int, stop_loss: float,
                             tp_levels: Dict, signal: Dict) -> Dict:
        """Execute paper trade"""
        
        # Deduct margin from paper balance
        balance = self.db.get_setting('paper_balance', 10000)
        margin = (position_size * entry_price) / leverage
        
        if margin > balance:
            return {'success': False, 'error': 'Insufficient paper balance'}
        
        # Estimate and deduct entry fees
        fees = self.risk_calc.estimate_fees(position_size * entry_price, entries=1, exits=0)
        
        new_balance = balance - margin - fees['entry_fee']
        self.db.set_setting('paper_balance', new_balance)
        
        # Create trade record
        trade_data = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'position_size': position_size,
            'position_value': position_size * entry_price,
            'leverage': leverage,
            'stop_loss': stop_loss,
            'take_profit_1': tp_levels['tp1'],
            'take_profit_2': tp_levels['tp2'],
            'take_profit_3': tp_levels['tp3'],
            'sleeper_score': signal.get('sleeper_score'),
            'ob_quality': signal.get('ob_quality'),
            'signal_confidence': signal.get('signal_confidence'),
            'is_paper': True,
            'execution_mode': signal.get('execution_mode', 'MANUAL'),
            'status': 'OPEN',
        }
        
        trade = self.db.add_trade(trade_data)
        
        if trade:
            self.db.log_event(
                f"📝 PAPER {direction} {symbol} @ {entry_price:.4f} | Size: {position_size:.4f}",
                level='SUCCESS', category='TRADE', symbol=symbol
            )
            
            return {
                'success': True,
                'trade_id': trade.id,
                'is_paper': True,
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'tp1': tp_levels['tp1'],
                'tp2': tp_levels['tp2'],
                'tp3': tp_levels['tp3'],
                'margin_used': margin,
                'fees_paid': fees['entry_fee'],
            }
        
        return {'success': False, 'error': 'Failed to create trade record'}
    
    def _execute_live_trade(self, symbol: str, direction: str, entry_price: float,
                            position_size: float, leverage: int, stop_loss: float,
                            tp_levels: Dict, signal: Dict) -> Dict:
        """Execute live trade on exchange"""
        
        try:
            # Set leverage
            self.connector.set_leverage(symbol, leverage)
            
            # Place market order
            side = 'Buy' if direction == 'LONG' else 'Sell'
            
            order_result = self.connector.place_order(
                symbol=symbol,
                side=side,
                qty=position_size,
                order_type='Market',
                stop_loss=stop_loss,
                take_profit=tp_levels['tp1']  # Initial TP
            )
            
            if not order_result:
                return {'success': False, 'error': 'Order placement failed'}
            
            # Get actual fill price
            fill_price = float(order_result.get('avgPrice', entry_price))
            
            # Create trade record
            trade_data = {
                'symbol': symbol,
                'direction': direction,
                'entry_price': fill_price,
                'position_size': position_size,
                'position_value': position_size * fill_price,
                'leverage': leverage,
                'stop_loss': stop_loss,
                'take_profit_1': tp_levels['tp1'],
                'take_profit_2': tp_levels['tp2'],
                'take_profit_3': tp_levels['tp3'],
                'sleeper_score': signal.get('sleeper_score'),
                'ob_quality': signal.get('ob_quality'),
                'signal_confidence': signal.get('signal_confidence'),
                'is_paper': False,
                'execution_mode': signal.get('execution_mode', 'MANUAL'),
                'status': 'OPEN',
            }
            
            trade = self.db.add_trade(trade_data)
            
            self.db.log_event(
                f"🔥 LIVE {direction} {symbol} @ {fill_price:.4f} | Size: {position_size:.4f}",
                level='SUCCESS', category='TRADE', symbol=symbol
            )
            
            return {
                'success': True,
                'trade_id': trade.id if trade else None,
                'is_paper': False,
                'symbol': symbol,
                'direction': direction,
                'entry_price': fill_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'tp1': tp_levels['tp1'],
                'order_id': order_result.get('orderId'),
            }
            
        except Exception as e:
            self.db.log_event(
                f"❌ Failed to execute {direction} {symbol}: {str(e)}",
                level='ERROR', category='TRADE', symbol=symbol
            )
            return {'success': False, 'error': str(e)}
    
    def manual_entry(self, symbol: str, direction: str, 
                     stop_loss: float = None) -> Dict:
        """
        Manual entry for a symbol
        Calculates SL if not provided
        """
        # Отримуємо ціну з Bybit (де торгуємо)
        entry_price = self.connector.get_price(symbol)
        if not entry_price:
            return {'success': False, 'error': 'Could not get price'}
        
        # Calculate SL from ATR if not provided
        if not stop_loss:
            stop_loss = self.risk_calc.calculate_sl_from_atr(
                symbol, entry_price, direction
            )
            
            if not stop_loss:
                return {'success': False, 'error': 'Could not calculate stop loss'}
        
        signal = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'execution_mode': 'MANUAL',
        }
        
        return self.execute_signal(signal)
    
    def close_position(self, trade_id: int, reason: str = 'MANUAL') -> Dict:
        """Close a position by trade ID"""
        return self.position_tracker.close_position(trade_id, reason)
    
    def close_all_positions(self, reason: str = 'MANUAL_CLOSE_ALL') -> Dict:
        """Close all open positions"""
        positions = self.position_tracker.get_all_positions()
        results = []
        
        for pos in positions:
            result = self.close_position(pos['id'], reason)
            results.append({
                'symbol': pos['symbol'],
                'success': result['success'],
                'pnl': result.get('pnl_usdt', 0)
            })
        
        total_pnl = sum(r.get('pnl', 0) for r in results)
        
        return {
            'success': True,
            'closed': len(results),
            'total_pnl': total_pnl,
            'details': results,
        }
    
    def modify_sl(self, trade_id: int, new_sl: float) -> Dict:
        """Modify stop loss for a position"""
        trades = self.db.get_trades(status='OPEN')
        trade = next((t for t in trades if t['id'] == trade_id), None)
        
        if not trade:
            return {'success': False, 'error': 'Trade not found'}
        
        # Validate new SL
        direction = trade['direction']
        entry = trade['entry_price']
        
        if direction == 'LONG' and new_sl >= entry:
            return {'success': False, 'error': 'SL must be below entry for LONG'}
        if direction == 'SHORT' and new_sl <= entry:
            return {'success': False, 'error': 'SL must be above entry for SHORT'}
        
        is_paper = trade['is_paper']
        
        if not is_paper:
            # Update on exchange
            try:
                self.connector.set_trading_stop(
                    symbol=trade['symbol'],
                    stop_loss=new_sl
                )
            except Exception as e:
                return {'success': False, 'error': str(e)}
        
        # Update in database (need to implement update method)
        self.db.log_event(
            f"Modified SL for {trade['symbol']}: {trade['stop_loss']:.4f} → {new_sl:.4f}",
            level='INFO', category='TRADE', symbol=trade['symbol']
        )
        
        return {
            'success': True,
            'old_sl': trade['stop_loss'],
            'new_sl': new_sl,
        }
    
    def move_to_breakeven(self, trade_id: int) -> Dict:
        """Move stop loss to breakeven (entry price)"""
        trades = self.db.get_trades(status='OPEN')
        trade = next((t for t in trades if t['id'] == trade_id), None)
        
        if not trade:
            return {'success': False, 'error': 'Trade not found'}
        
        entry = trade['entry_price']
        
        # Add small buffer
        if trade['direction'] == 'LONG':
            be_price = entry * 1.0001  # Tiny profit
        else:
            be_price = entry * 0.9999
        
        return self.modify_sl(trade_id, be_price)


# Singleton instance
_executor_instance = None

def get_executor() -> OrderExecutor:
    """Get order executor instance"""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = OrderExecutor()
    return _executor_instance
