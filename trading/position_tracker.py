"""
Position Tracker - Monitor and manage open positions
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from config import TRADING_CONSTANTS
from core import get_connector, get_fetcher
from storage import get_db

class PositionTracker:
    """
    Position Management Module
    
    Tracks:
    - Open positions (paper and real)
    - P&L calculations
    - TP/SL hit detection
    - Trailing stop management
    """
    
    def __init__(self):
        self.db = get_db()
        self.connector = get_connector()
        self.fetcher = get_fetcher()
        self.constants = TRADING_CONSTANTS
    
    def get_all_positions(self) -> List[Dict]:
        """Get all open positions with current P&L"""
        open_trades = self.db.get_trades(status='OPEN')
        
        if not open_trades:
            return []
        
        # Get current prices
        symbols = [t['symbol'] for t in open_trades]
        prices = self.fetcher.batch_get_prices(symbols)
        
        positions = []
        for trade in open_trades:
            current_price = prices.get(trade['symbol'], 0)
            
            if current_price > 0:
                pnl = self._calculate_pnl(trade, current_price)
                trade.update(pnl)
            
            trade['current_price'] = current_price
            trade['duration'] = self._calculate_duration(trade['entry_time'])
            
            positions.append(trade)
        
        return positions
    
    def _calculate_pnl(self, trade: Dict, current_price: float) -> Dict:
        """Calculate P&L for a trade"""
        entry_price = trade['entry_price']
        position_size = trade['position_size']
        direction = trade['direction']
        leverage = trade.get('leverage', 1)
        
        # Price change
        if direction == 'LONG':
            price_change = current_price - entry_price
        else:
            price_change = entry_price - current_price
        
        # P&L in USDT
        pnl_usdt = price_change * position_size
        
        # P&L percentage (on margin)
        margin = (position_size * entry_price) / leverage
        pnl_pct = (pnl_usdt / margin * 100) if margin > 0 else 0
        
        # R multiple
        stop_loss = trade.get('stop_loss', 0)
        if stop_loss > 0:
            if direction == 'LONG':
                risk = entry_price - stop_loss
            else:
                risk = stop_loss - entry_price
            r_multiple = price_change / risk if risk > 0 else 0
        else:
            r_multiple = 0
        
        return {
            'unrealized_pnl': round(pnl_usdt, 2),
            'pnl_percent': round(pnl_pct, 2),
            'r_multiple': round(r_multiple, 2),
            'price_change_pct': round((current_price - entry_price) / entry_price * 100, 2),
        }
    
    def _calculate_duration(self, entry_time) -> str:
        """Calculate trade duration as string"""
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))
        
        if not entry_time:
            return "Unknown"
        
        now = datetime.utcnow()
        if entry_time.tzinfo:
            entry_time = entry_time.replace(tzinfo=None)
        
        delta = now - entry_time
        
        if delta.days > 0:
            return f"{delta.days}d {delta.seconds // 3600}h"
        elif delta.seconds >= 3600:
            return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"
        else:
            return f"{delta.seconds // 60}m"
    
    def check_tp_sl(self) -> List[Dict]:
        """Check all positions for TP/SL hits"""
        positions = self.get_all_positions()
        events = []
        
        for pos in positions:
            event = self._check_position_levels(pos)
            if event:
                events.append(event)
        
        return events
    
    def _check_position_levels(self, position: Dict) -> Optional[Dict]:
        """Check if position hit TP or SL"""
        current_price = position.get('current_price', 0)
        if current_price <= 0:
            return None
        
        direction = position['direction']
        
        # Check Stop Loss
        stop_loss = position.get('stop_loss', 0)
        if stop_loss > 0:
            if (direction == 'LONG' and current_price <= stop_loss) or \
               (direction == 'SHORT' and current_price >= stop_loss):
                return {
                    'type': 'SL_HIT',
                    'trade_id': position['id'],
                    'symbol': position['symbol'],
                    'direction': direction,
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'stop_loss': stop_loss,
                }
        
        # Check Take Profits
        for tp_level in ['take_profit_1', 'take_profit_2', 'take_profit_3']:
            tp = position.get(tp_level, 0)
            if tp > 0:
                if (direction == 'LONG' and current_price >= tp) or \
                   (direction == 'SHORT' and current_price <= tp):
                    return {
                        'type': f'{tp_level.upper()}_HIT',
                        'trade_id': position['id'],
                        'symbol': position['symbol'],
                        'direction': direction,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'take_profit': tp,
                    }
        
        return None
    
    def update_trailing_stop(self, trade_id: int, current_price: float) -> Optional[float]:
        """Update trailing stop if price has moved favorably"""
        trades = self.db.get_trades(status='OPEN')
        trade = next((t for t in trades if t['id'] == trade_id), None)
        
        if not trade:
            return None
        
        direction = trade['direction']
        entry_price = trade['entry_price']
        stop_loss = trade.get('stop_loss', 0)
        
        if stop_loss <= 0:
            return None
        
        # Calculate risk
        if direction == 'LONG':
            risk = entry_price - stop_loss
            profit_r = (current_price - entry_price) / risk if risk > 0 else 0
        else:
            risk = stop_loss - entry_price
            profit_r = (entry_price - current_price) / risk if risk > 0 else 0
        
        # Only trail after reaching threshold
        if profit_r < self.constants['trailing_start_ratio']:
            return None
        
        # Calculate new trailing stop
        offset = current_price * (self.constants['trailing_offset_pct'] / 100)
        
        if direction == 'LONG':
            new_sl = current_price - offset
            # Only move SL up, never down
            if new_sl > stop_loss:
                return round(new_sl, 8)
        else:
            new_sl = current_price + offset
            # Only move SL down, never up
            if new_sl < stop_loss:
                return round(new_sl, 8)
        
        return None
    
    def update_all_trailing_stops(self) -> List[Dict]:
        """Update trailing stops for all open positions"""
        positions = self.get_all_positions()
        updates = []
        
        for pos in positions:
            if pos.get('current_price', 0) <= 0:
                continue
            
            new_sl = self.update_trailing_stop(pos['id'], pos['current_price'])
            if new_sl:
                # Update in database
                self.db.update_trade(pos['id'], {'stop_loss': new_sl})
                updates.append({
                    'trade_id': pos['id'],
                    'symbol': pos['symbol'],
                    'old_sl': pos.get('stop_loss'),
                    'new_sl': new_sl
                })
                
                self.db.log_event(
                    'INFO', 'TRADE',
                    f"Trailing stop updated: {pos.get('stop_loss'):.4f} → {new_sl:.4f}",
                    pos['symbol']
                )
        
        return updates
    
    def get_position_summary(self) -> Dict:
        """Get summary of all positions"""
        positions = self.get_all_positions()
        
        if not positions:
            return {
                'count': 0,
                'total_unrealized_pnl': 0,
                'total_margin': 0,
                'winning': 0,
                'losing': 0,
            }
        
        total_pnl = sum(p.get('unrealized_pnl', 0) for p in positions)
        total_margin = sum(
            (p['position_size'] * p['entry_price']) / p.get('leverage', 1) 
            for p in positions
        )
        
        winning = len([p for p in positions if p.get('unrealized_pnl', 0) > 0])
        losing = len([p for p in positions if p.get('unrealized_pnl', 0) <= 0])
        
        return {
            'count': len(positions),
            'total_unrealized_pnl': round(total_pnl, 2),
            'total_margin': round(total_margin, 2),
            'winning': winning,
            'losing': losing,
            'positions': positions,
        }
    
    def sync_with_exchange(self) -> Dict:
        """
        Sync paper positions with exchange (for live mode)
        """
        is_paper = self.db.get_setting('paper_trading', True)
        
        if is_paper:
            return {'synced': False, 'reason': 'Paper trading mode'}
        
        try:
            exchange_positions = self.connector.get_positions()
            db_trades = self.db.get_trades(status='OPEN')
            
            synced = []
            
            for ex_pos in exchange_positions:
                symbol = ex_pos['symbol']
                
                # Find matching DB trade
                db_trade = next(
                    (t for t in db_trades if t['symbol'] == symbol), 
                    None
                )
                
                if db_trade:
                    # Update with exchange data
                    synced.append({
                        'symbol': symbol,
                        'db_size': db_trade['position_size'],
                        'exchange_size': ex_pos['size'],
                    })
            
            return {
                'synced': True,
                'exchange_positions': len(exchange_positions),
                'db_positions': len(db_trades),
                'details': synced,
            }
            
        except Exception as e:
            return {'synced': False, 'error': str(e)}
    
    def close_position(self, trade_id: int, exit_reason: str = 'MANUAL') -> Dict:
        """Close a position"""
        trades = self.db.get_trades(status='OPEN')
        trade = next((t for t in trades if t['id'] == trade_id), None)
        
        if not trade:
            return {'success': False, 'error': 'Trade not found'}
        
        current_price = self.fetcher.get_current_price(trade['symbol'])
        if not current_price:
            return {'success': False, 'error': 'Could not get current price'}
        
        # Calculate final P&L
        pnl = self._calculate_pnl(trade, current_price)
        
        # Estimate fees
        from trading.risk_calculator import get_risk_calculator
        calc = get_risk_calculator()
        fees = calc.estimate_fees(trade['position_size'] * current_price)
        
        # Close in exchange if live
        is_paper = self.db.get_setting('paper_trading', True)
        
        if not is_paper:
            try:
                self.connector.close_position(
                    symbol=trade['symbol'],
                    side='Buy' if trade['direction'] == 'LONG' else 'Sell',
                    qty=trade['position_size']
                )
            except Exception as e:
                return {'success': False, 'error': f'Exchange error: {str(e)}'}
        
        # Update database
        self.db.close_trade(
            trade_id=trade_id,
            exit_price=current_price,
            exit_reason=exit_reason,
            pnl_usdt=pnl['unrealized_pnl'],
            pnl_percent=pnl['pnl_percent'],
            fees=fees['total_estimated']
        )
        
        # Update paper balance if paper trading
        if is_paper:
            balance = self.db.get_setting('paper_balance', 10000)
            new_balance = balance + pnl['unrealized_pnl'] - fees['total_estimated']
            self.db.set_setting('paper_balance', new_balance)
        
        self.db.log_event(
            f"Closed {trade['direction']} {trade['symbol']} @ {current_price:.4f} | P&L: ${pnl['unrealized_pnl']:.2f}",
            level='SUCCESS' if pnl['unrealized_pnl'] > 0 else 'WARN',
            category='TRADE',
            symbol=trade['symbol']
        )
        
        return {
            'success': True,
            'exit_price': current_price,
            'pnl_usdt': pnl['unrealized_pnl'],
            'pnl_percent': pnl['pnl_percent'],
            'fees': fees['total_estimated'],
        }


# Singleton instance
_tracker_instance = None

def get_position_tracker() -> PositionTracker:
    """Get position tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = PositionTracker()
    return _tracker_instance
