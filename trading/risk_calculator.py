"""
Risk Calculator v8.2 - OB Based Position Sizing
Торгівля на Bybit - використовуємо Bybit для всіх даних

v8.2: Розрахунок позиції на основі Order Block:
- Entry: Верхня межа OB (Long) / Нижня межа OB (Short)
- Stop Loss: За межею OB + buffer (0.2%)
- Take Profit: 1:3 R/R або до протилежного OB
"""
from typing import Dict, Optional, Tuple
from config import TRADING_CONSTANTS
from core import get_connector  # Bybit для торгівлі
from storage import get_db


class RiskCalculator:
    """
    Risk Management Module v8.2
    
    Calculates:
    - Position size based on risk % and OB distance
    - Stop loss from Order Block boundaries
    - Take profit with minimum 1:3 R/R
    - Max position limits
    
    NOTE: Використовує Bybit для всіх даних (торгівля на Bybit)
    """
    
    # Default settings
    DEFAULT_RISK_PCT = 1.0      # 1% ризику на угоду
    DEFAULT_LEVERAGE = 10       # Плече x10
    MIN_RR_RATIO = 3.0          # Мінімальний R/R
    MIN_STOP_PCT = 0.002        # Мін стоп 0.2% (захист від шуму)
    SL_BUFFER_PCT = 0.002       # Буфер за OB 0.2%
    
    def __init__(self):
        self.db = get_db()
        self.connector = get_connector()  # Bybit
        self.constants = TRADING_CONSTANTS
    
    def calculate_ob_position(self, 
                              symbol: str,
                              direction: str,
                              entry_price: float,
                              ob_high: float,
                              ob_low: float,
                              swing_target: float = None,
                              balance: float = None) -> Dict:
        """
        v8.2: Розрахунок позиції на основі Order Block
        
        Args:
            symbol: Торгова пара (BTCUSDT)
            direction: LONG або SHORT
            entry_price: Поточна ціна або ціна входу
            ob_high: Верхня межа Order Block
            ob_low: Нижня межа Order Block
            swing_target: Цільовий swing high/low для TP (опційно)
            balance: Баланс (якщо None - береться з налаштувань)
        
        Returns:
            Dict з усіма параметрами угоди
        """
        # Get settings
        risk_pct = self.db.get_setting('max_risk_per_trade', self.DEFAULT_RISK_PCT)
        leverage = self.db.get_setting('default_leverage', self.DEFAULT_LEVERAGE)
        min_rr = self.db.get_setting('min_rr_ratio', self.MIN_RR_RATIO)
        
        # Get balance
        if balance is None:
            is_paper = self.db.get_setting('paper_trading', True)
            if is_paper:
                balance = self.db.get_setting('paper_balance', 1000)
            else:
                wallet = self.connector.get_wallet_balance()
                balance = wallet.get('available_balance', 0)
        
        if balance <= 0:
            return self._error_result('Insufficient balance')
        
        # === CALCULATE ENTRY, SL, TP ===
        
        if direction == 'LONG':
            # LONG: Entry at OB high, SL below OB low
            final_entry = ob_high
            stop_loss = ob_low * (1 - self.SL_BUFFER_PCT)  # 0.2% buffer
            
            # TP: swing target or 1:3 R/R
            risk = final_entry - stop_loss
            if swing_target and swing_target > final_entry:
                take_profit = swing_target
            else:
                take_profit = final_entry + (risk * min_rr)
            
        else:  # SHORT
            # SHORT: Entry at OB low, SL above OB high
            final_entry = ob_low
            stop_loss = ob_high * (1 + self.SL_BUFFER_PCT)
            
            risk = stop_loss - final_entry
            if swing_target and swing_target < final_entry:
                take_profit = swing_target
            else:
                take_profit = final_entry - (risk * min_rr)
        
        # === CALCULATE RISK METRICS ===
        
        # Distance to stop loss
        if direction == 'LONG':
            stop_dist_pct = (final_entry - stop_loss) / final_entry
            reward_dist_pct = (take_profit - final_entry) / final_entry
        else:
            stop_dist_pct = (stop_loss - final_entry) / final_entry
            reward_dist_pct = (final_entry - take_profit) / final_entry
        
        # Enforce minimum stop
        if stop_dist_pct < self.MIN_STOP_PCT:
            stop_dist_pct = self.MIN_STOP_PCT
            if direction == 'LONG':
                stop_loss = final_entry * (1 - self.MIN_STOP_PCT)
            else:
                stop_loss = final_entry * (1 + self.MIN_STOP_PCT)
        
        # Actual R/R
        actual_rr = reward_dist_pct / stop_dist_pct if stop_dist_pct > 0 else 0
        
        # === POSITION SIZING ===
        
        # Risk amount in USDT
        risk_amount = balance * (risk_pct / 100)
        
        # Position size that risks exactly risk_amount
        position_value = risk_amount / stop_dist_pct
        
        # Apply leverage
        margin_required = position_value / leverage
        
        # Check max position
        max_pos_pct = self.constants.get('max_position_size_pct', 50)
        max_position_value = balance * (max_pos_pct / 100) * leverage
        position_value = min(position_value, max_position_value)
        
        # Position size in crypto
        position_size = position_value / final_entry
        
        # Round to symbol precision
        symbol_info = self.connector.get_instrument_info(symbol)
        if symbol_info:
            lot_filter = symbol_info.get('lotSizeFilter', {})
            qty_step = float(lot_filter.get('qtyStep', 0.001))
            min_qty = float(lot_filter.get('minOrderQty', 0.001))
            position_size = round(position_size / qty_step) * qty_step
            position_size = max(position_size, min_qty)
        
        # Recalculate final values
        final_value = position_size * final_entry
        
        return {
            'success': True,
            'symbol': symbol,
            'direction': direction,
            
            # Entry/SL/TP
            'entry_price': round(final_entry, 8),
            'stop_loss': round(stop_loss, 8),
            'take_profit': round(take_profit, 8),
            
            # Risk metrics
            'stop_pct': round(stop_dist_pct * 100, 2),
            'reward_pct': round(reward_dist_pct * 100, 2),
            'rr_ratio': round(actual_rr, 2),
            
            # Position sizing
            'position_size': round(position_size, 6),
            'position_value': round(final_value, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_pct': risk_pct,
            'leverage': leverage,
            'margin_required': round(margin_required, 2),
            'balance': balance,
            
            # OB info
            'ob_high': round(ob_high, 8),
            'ob_low': round(ob_low, 8),
        }
    
    def _error_result(self, error: str) -> Dict:
        """Return error result"""
        return {
            'success': False,
            'error': error,
            'position_size': 0,
            'position_value': 0,
            'risk_amount': 0,
        }
    
    def format_signal_message(self, calc_result: Dict) -> str:
        """
        Форматує результат розрахунку для Telegram
        
        Returns:
            Formatted Ukrainian message
        """
        if not calc_result.get('success'):
            return f"❌ Помилка: {calc_result.get('error', 'Unknown')}"
        
        direction_emoji = "🟢 LONG" if calc_result['direction'] == 'LONG' else "🔴 SHORT"
        rr_emoji = "🔥" if calc_result['rr_ratio'] >= 3 else "✅" if calc_result['rr_ratio'] >= 2 else "⚠️"
        
        msg = f"""
{direction_emoji} <b>{calc_result['symbol']}</b>

📊 <b>Параметри угоди:</b>
├ Вхід: <code>{calc_result['entry_price']:.6f}</code>
├ Стоп: <code>{calc_result['stop_loss']:.6f}</code> ({calc_result['stop_pct']:.1f}%)
├ Тейк: <code>{calc_result['take_profit']:.6f}</code> ({calc_result['reward_pct']:.1f}%)
└ R/R: {rr_emoji} <b>{calc_result['rr_ratio']:.1f}</b>

💰 <b>Розмір позиції:</b>
├ Об'єм: <b>{calc_result['position_value']:.0f}</b> USD
├ Маржа: {calc_result['margin_required']:.0f} USD (x{calc_result['leverage']})
└ Ризик: {calc_result['risk_amount']:.0f} USD ({calc_result['risk_pct']}% депо)
"""
        return msg.strip()
    
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
        
        # Get symbol info for rounding (from Bybit)
        symbol_info = self.connector.get_instrument_info(symbol)
        if symbol_info:
            price_filter = symbol_info.get('priceFilter', {})
            lot_filter = symbol_info.get('lotSizeFilter', {})
            qty_step = float(lot_filter.get('qtyStep', 0.001))
            min_qty = float(lot_filter.get('minOrderQty', 0.001))
            
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
        
        # Get recent klines from Bybit (де торгуємо)
        klines = self.connector.get_klines(symbol, '15', limit=20)
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
