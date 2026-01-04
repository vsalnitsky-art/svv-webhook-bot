"""
Risk Calculator - Position sizing and risk management
"""
from typing import Dict, Optional, Tuple
from config import TRADING_CONSTANTS
from core import get_connector, get_fetcher
from storage import get_db

class RiskCalculator:
    """
    Risk Management Module
    
    Calculates:
    - Position size based on risk %
    - Stop loss levels
    - Take profit levels
    - Max position limits
    """
    
    def __init__(self):
        self.db = get_db()
        self.connector = get_connector()
        self.fetcher = get_fetcher()
        self.constants = TRADING_CONSTANTS
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                                stop_loss: float, direction: str) -> Dict:
        """
        Calculate position size based on risk parameters
        
        Returns:
        - position_size: Size in contracts
        - position_value: Value in USDT
        - risk_amount: Amount at risk in USDT
        """
        # Get settings
        risk_pct = self.db.get_setting('max_risk_per_trade', 1.0)
        max_positions = self.db.get_setting('max_open_positions', 3)
        leverage = self.db.get_setting('default_leverage', 5)
        
        # Check if we can open more positions
        open_trades = self.db.get_trades(status='OPEN')
        if len(open_trades) >= max_positions:
            return {
                'success': False,
                'error': f'Max positions ({max_positions}) reached',
                'position_size': 0
            }
        
        # Get account balance
        is_paper = self.db.get_setting('paper_trading', True)
        
        if is_paper:
            balance = self.db.get_setting('paper_balance', 10000)
        else:
            wallet = self.connector.get_wallet_balance()
            balance = wallet.get('available_balance', 0)
        
        if balance <= 0:
            return {
                'success': False,
                'error': 'Insufficient balance',
                'position_size': 0
            }
        
        # Calculate risk amount
        risk_amount = balance * (risk_pct / 100)
        
        # Calculate price risk
        if direction == 'LONG':
            price_risk = entry_price - stop_loss
        else:
            price_risk = stop_loss - entry_price
        
        price_risk_pct = price_risk / entry_price if entry_price > 0 else 0
        
        if price_risk_pct <= 0:
            return {
                'success': False,
                'error': 'Invalid stop loss',
                'position_size': 0
            }
        
        # Position value that risks the specified amount
        position_value = risk_amount / price_risk_pct
        
        # Apply leverage
        margin_required = position_value / leverage
        
        # Check max position size
        max_position_value = balance * (self.constants['max_position_size_pct'] / 100) * leverage
        position_value = min(position_value, max_position_value)
        
        # Calculate contracts
        position_size = position_value / entry_price
        
        # Get symbol info for rounding
        symbol_info = self.fetcher.get_symbol_info(symbol)
        if symbol_info:
            qty_step = symbol_info.get('qty_step', 0.001)
            min_qty = symbol_info.get('min_qty', 0.001)
            
            # Round to qty step
            position_size = round(position_size / qty_step) * qty_step
            position_size = max(position_size, min_qty)
        
        # Final checks
        final_value = position_size * entry_price
        if final_value < self.constants['min_position_size']:
            return {
                'success': False,
                'error': f'Position too small (min ${self.constants["min_position_size"]})',
                'position_size': 0
            }
        
        return {
            'success': True,
            'position_size': round(position_size, 6),
            'position_value': round(final_value, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_pct': risk_pct,
            'leverage': leverage,
            'margin_required': round(margin_required, 2),
            'balance': balance,
        }
    
    def calculate_tp_levels(self, entry_price: float, stop_loss: float, 
                            direction: str) -> Dict[str, float]:
        """
        Calculate take profit levels based on R:R ratios
        """
        if direction == 'LONG':
            risk = entry_price - stop_loss
            tp1 = entry_price + (risk * self.constants['tp1_ratio'])
            tp2 = entry_price + (risk * self.constants['tp2_ratio'])
            tp3 = entry_price + (risk * self.constants['tp3_ratio'])
        else:
            risk = stop_loss - entry_price
            tp1 = entry_price - (risk * self.constants['tp1_ratio'])
            tp2 = entry_price - (risk * self.constants['tp2_ratio'])
            tp3 = entry_price - (risk * self.constants['tp3_ratio'])
        
        return {
            'tp1': round(tp1, 8),
            'tp2': round(tp2, 8),
            'tp3': round(tp3, 8),
            'risk_per_r': round(risk, 8),
        }
    
    def calculate_sl_from_atr(self, symbol: str, entry_price: float, 
                              direction: str) -> Optional[float]:
        """
        Calculate stop loss using ATR multiplier
        """
        from core import get_indicators
        
        # Get recent klines
        klines = self.fetcher.get_klines(symbol, '15m', limit=20)
        if len(klines) < 15:
            return None
        
        # Calculate ATR
        indicators = get_indicators()
        highs = [k['high'] for k in klines]
        lows = [k['low'] for k in klines]
        closes = [k['close'] for k in klines]
        
        atr_values = indicators.atr(highs, lows, closes, period=14)
        atr = atr_values[-1] if atr_values else 0
        
        if atr == 0:
            return None
        
        atr_mult = self.db.get_setting('stop_loss_atr_mult', 1.5)
        sl_distance = atr * atr_mult
        
        if direction == 'LONG':
            return round(entry_price - sl_distance, 8)
        else:
            return round(entry_price + sl_distance, 8)
    
    def estimate_fees(self, position_value: float, entries: int = 1, 
                      exits: int = 1) -> Dict[str, float]:
        """
        Estimate trading fees
        """
        # Assume market orders (taker fees)
        entry_fee = position_value * self.constants['taker_fee'] * entries
        exit_fee = position_value * self.constants['taker_fee'] * exits
        
        # If using Smart TP (50/25/25 split)
        smart_tp_fee = (
            position_value * 0.5 * self.constants['taker_fee'] +
            position_value * 0.25 * self.constants['taker_fee'] +
            position_value * 0.25 * self.constants['taker_fee']
        )
        
        return {
            'entry_fee': round(entry_fee, 4),
            'exit_fee': round(exit_fee, 4),
            'total_estimated': round(entry_fee + exit_fee, 4),
            'smart_tp_total': round(entry_fee + smart_tp_fee, 4),
        }
    
    def validate_trade(self, symbol: str, direction: str, entry_price: float,
                       stop_loss: float, position_size: float) -> Dict:
        """
        Validate a trade before execution
        """
        errors = []
        warnings = []
        
        # 1. Check position size
        if position_size <= 0:
            errors.append('Invalid position size')
        
        # 2. Check SL direction
        if direction == 'LONG' and stop_loss >= entry_price:
            errors.append('Stop loss must be below entry for LONG')
        elif direction == 'SHORT' and stop_loss <= entry_price:
            errors.append('Stop loss must be above entry for SHORT')
        
        # 3. Check risk %
        risk_pct = abs(entry_price - stop_loss) / entry_price * 100
        if risk_pct > 5:
            warnings.append(f'High risk: {risk_pct:.2f}% per trade')
        if risk_pct > 10:
            errors.append(f'Risk too high: {risk_pct:.2f}%')
        
        # 4. Check balance
        position_value = position_size * entry_price
        is_paper = self.db.get_setting('paper_trading', True)
        
        if is_paper:
            balance = self.db.get_setting('paper_balance', 10000)
        else:
            wallet = self.connector.get_wallet_balance()
            balance = wallet.get('available_balance', 0)
        
        leverage = self.db.get_setting('default_leverage', 5)
        margin_required = position_value / leverage
        
        if margin_required > balance:
            errors.append(f'Insufficient margin: need ${margin_required:.2f}, have ${balance:.2f}')
        
        # 5. Check max positions
        max_positions = self.db.get_setting('max_open_positions', 3)
        open_trades = self.db.get_trades(status='OPEN')
        
        if len(open_trades) >= max_positions:
            errors.append(f'Max positions ({max_positions}) reached')
        
        # 6. Check for duplicate position
        for trade in open_trades:
            if trade['symbol'] == symbol:
                warnings.append(f'Already have open position on {symbol}')
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'risk_pct': round(risk_pct, 2),
            'margin_required': round(margin_required, 2),
            'available_balance': round(balance, 2),
        }


# Singleton instance
_calculator_instance = None

def get_risk_calculator() -> RiskCalculator:
    """Get risk calculator instance"""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = RiskCalculator()
    return _calculator_instance
