import logging
import decimal
from datetime import datetime
from pybit.unified_trading import HTTP
from bot_config import config
from config import get_api_credentials
from statistics_service import stats_service
from tp_strategy_config import tp_config  # ⭐ НОВИЙ ІМПОРТ

logger = logging.getLogger(__name__)

class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)

    def normalize(self, s): return s.replace('.P', '')

    def get_bal(self):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            for acc in b['result']['list']:
                for c in acc['coin']:
                    if c['coin'] == "USDT": return float(c['walletBalance'])
            return 0.0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0.0

    def get_price(self, s):
        try: 
            return float(self.session.get_tickers(category="linear", symbol=self.normalize(s))['result']['list'][0]['lastPrice'])
        except Exception as e:
            logger.error(f"Error getting price for {s}: {e}")
            return 0.0

    def get_all_tickers(self):
        try:
            r = self.session.get_tickers(category="linear")
            if r['retCode'] == 0:
                return r['result']['list']
            else:
                logger.warning(f"Get all tickers API Error: {r}")
        except Exception as e:
             logger.error(f"Error getting all tickers: {e}")
        return []

    def get_instr(self, s):
        try:
            r = self.session.get_instruments_info(category="linear", symbol=self.normalize(s))
            return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except Exception as e:
            logger.error(f"Error getting instrument info for {s}: {e}")
            return None, None

    def round_val(self, val, step):
        import decimal
        try:
            d = abs(decimal.Decimal(str(step)).as_tuple().exponent)
            return round(val // step * step, d)
        except Exception as e:
            logger.error(f"Rounding error: {e}")
            return val

    def set_lev(self, s, l):
        try: 
            self.session.set_leverage(category="linear", symbol=self.normalize(s), buyLeverage=str(l), sellLeverage=str(l))
        except Exception as e:
            logger.info(f"Set leverage info for {s}: {e}")

    def place_order(self, data):
        try:
            action = data.get('action') # "Buy", "Sell" або "Close"
            symbol = data.get('symbol')
            norm = self.normalize(symbol)
            
            # ==========================================
            # ЛОГІКА ЗАКРИТТЯ ПОЗИЦІЇ (CLOSE)
            # ==========================================
            if action == "Close":
                direction = data.get('direction') # "Long" або "Short"
                reason = data.get('reason', 'Signal')
                
                logger.info(f"🛑 RECEIVED CLOSE SIGNAL: {symbol} | Direction: {direction} | Reason: {reason}")
                
                # 1. Отримуємо поточну позицію
                pos_data = self.session.get_positions(category="linear", symbol=norm)
                if pos_data['retCode'] != 0:
                    return {"status": "error", "reason": "API Error getting positions"}
                
                target_position = None
                for p in pos_data['result']['list']:
                    if float(p['size']) > 0:
                        target_position = p
                        break
                
                # 2. Перевірка: чи існує позиція?
                if not target_position:
                    logger.warning(f"⚠️ CLOSE IGNORED: No open position for {symbol}")
                    return {"status": "ignored", "reason": "No open position found"}
                
                # 3. Перевірка: чи співпадає напрямок?
                # Bybit повертає side як "Buy" (це Long) або "Sell" (це Short)
                current_side = target_position['side'] # "Buy" or "Sell"
                current_size = target_position['size']
                
                match_long = (direction == "Long" and current_side == "Buy")
                match_short = (direction == "Short" and current_side == "Sell")
                
                if not (match_long or match_short):
                    logger.warning(f"⚠️ CLOSE IGNORED: Direction mismatch. Signal: {direction}, Actual: {current_side}")
                    return {"status": "ignored", "reason": f"Direction mismatch. Holding {current_side}"}
                
                # 4. Виконання закриття
                # Щоб закрити Long (Buy), треба зробити Sell. Щоб закрити Short (Sell), треба зробити Buy.
                close_side = "Sell" if current_side == "Buy" else "Buy"
                
                logger.info(f"📉 CLOSING {current_side} {symbol} | Size: {current_size} | Reason: {reason}")
                
                self.session.place_order(
                    category="linear",
                    symbol=norm,
                    side=close_side,
                    orderType="Market",
                    qty=current_size,
                    reduceOnly=True, # Важливо! Тільки зменшує/закриває
                    timeInForce="IOC"
                )
                
                return {"status": "ok", "message": f"Closed {direction} position"}

            # ==========================================
            # ЛОГІКА ВІДКРИТТЯ ПОЗИЦІЇ (OPEN)
            # ==========================================
            
            # --- ПЕРЕВІРКА ВІДКРИТИХ ПОЗИЦІЙ (Щоб не дублювати) ---
            try:
                pos_data = self.session.get_positions(category="linear", symbol=norm)
                if pos_data['retCode'] == 0:
                    for p in pos_data['result']['list']:
                        if float(p['size']) > 0:
                            logger.warning(f"⚠️ IGNORING OPEN SIGNAL {action} {symbol}: Position already exists!")
                            return {
                                "status": "ignored", 
                                "reason": f"Position for {symbol} already exists. Size: {p['size']}"
                            }
            except Exception as e:
                logger.error(f"Error checking existing positions: {e}")
                return {"status": "error", "reason": "Failed to check existing positions"}
            
            # --- РОЗРАХУНОК ВХОДУ ---
            risk = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            lev = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            
            price = self.get_price(norm)
            lot, tick = self.get_instr(norm)
            if not price or not lot: 
                return {"status": "error", "reason": "Failed to get price/instrument info"}
            
            bal = self.get_bal()
            if bal < 5: 
                return {"status": "no_balance", "balance": bal}

            qty_step = float(lot['qtyStep'])
            min_qty = float(lot['minOrderQty'])
            tick_size = float(tick['tickSize'])
            
            raw_qty = (bal * (risk/100) * 0.98 * lev) / price
            qty = self.round_val(raw_qty, qty_step)
            if qty < min_qty: qty = min_qty
            
            self.set_lev(norm, lev)
            
            # 1. MARKET ORDER (ВХІД)
            logger.info(f"🚀 OPENING {action} {symbol} | Price: {price} | Qty: {qty}")
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            
            # 2. FIXED STOP LOSS (ТІЛЬКИ З JSON)
            sl_pct = float(data.get('stopLossPercent', 0.0))
            
            if sl_pct > 0:
                dist = price * (sl_pct / 100)
                dist = self.round_val(dist, tick_size)
                
                if action == "Buy":
                    sl_price = price - dist
                else: # Sell
                    sl_price = price + dist
                    
                sl_price = self.round_val(sl_price, tick_size)
                
                logger.info(f"🛡️ SETTING SL for {symbol} from JSON: {sl_price} ({sl_pct}%)")
                self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl_price), positionIdx=0)
                
                # 3. TAKE PROFIT (ТІЛЬКИ З JSON) ⭐ НОВИЙ v2.4
                logger.info(f"🎯 Checking for TP settings in JSON...")
                tp_result = self.apply_tp_from_json(data, symbol, price, action, qty)
                
                return {
                    "status": "ok", 
                    "sl_price": sl_price,
                    "tp_result": tp_result
                }
            else:
                logger.info(f"⚠️ NO SL in JSON for {symbol}. Checking for TP...")
                
                # Навіть без SL можна встановити TP
                tp_result = self.apply_tp_from_json(data, symbol, price, action, qty)
                
                return {
                    "status": "ok", 
                    "sl_price": None,
                    "tp_result": tp_result
                }

        except Exception as e:
            logger.error(f"Order Processing Critical Error: {e}")
            return {"status": "error", "reason": str(e)}

    def sync_history(self):
        try:
            import time
            end = int(time.time()*1000)
            start = end - (24*60*60*1000)
            r = self.session.get_closed_pnl(category="linear", startTime=start, endTime=end, limit=50)
            if r['retCode']==0:
                for t in r['result']['list']:
                    stats_service.save_trade({
                        'order_id': t['orderId'], 'symbol': t['symbol'],
                        'side': 'Long' if t['side']=='Sell' else 'Short',
                        'qty': float(t['qty']), 'entry_price': float(t['avgEntryPrice']),
                        'exit_price': float(t['avgExitPrice']), 'pnl': float(t['closedPnl']),
                        'exit_time': datetime.fromtimestamp(int(t['updatedTime'])/1000),
                        'exit_reason': 'Manual/SL/Signal'
                    })
        except Exception as e:
             logger.error(f"Sync history error: {e}")

    def sync_trades(self, days=7):
        try:
            import time
            end = int(time.time()*1000)
            start = end - (days*24*60*60*1000)
            r = self.session.get_closed_pnl(category="linear", startTime=start, endTime=end, limit=100)
            if r['retCode']==0:
                for t in r['result']['list']:
                    stats_service.save_trade({
                        'order_id': t['orderId'], 'symbol': t['symbol'],
                        'side': 'Long' if t['side']=='Sell' else 'Short',
                        'qty': float(t['qty']), 'entry_price': float(t['avgEntryPrice']),
                        'exit_price': float(t['avgExitPrice']), 'pnl': float(t['closedPnl']),
                        'exit_time': datetime.fromtimestamp(int(t['updatedTime'])/1000),
                        'exit_reason': 'Manual/SL/Signal'
                    })
        except Exception as e:
            logger.error(f"Sync trades error: {e}")

    def get_available_balance(self):
        return self.get_bal()
    
    # ⭐ НОВІ МЕТОДИ v2.4 - TAKE PROFIT STRATEGIES
    
    def set_take_profit_levels(self, symbol: str, entry_price: float, side: str, 
                               quantity: float, strategy_name: str = None):
        """
        Встановити рівні Take Profit за обраною стратегією
        
        Args:
            symbol: Символ монети (BTCUSDT)
            entry_price: Ціна входу
            side: 'Buy' (Long) або 'Sell' (Short)
            quantity: Загальна кількість позиції
            strategy_name: Назва стратегії (None = поточна)
        """
        try:
            norm = self.normalize(symbol)
            
            # Отримати стратегію
            strategy = tp_config.get_strategy(strategy_name)
            
            # Якщо стратегія = 'none', не встановлювати TP
            if not strategy['enabled'] or strategy_name == 'none':
                logger.info(f"ℹ️ TP Strategy 'none' selected - no TP will be set for {symbol}")
                return {'status': 'ok', 'tp_levels': 0, 'message': 'TP not set (strategy: none)'}
            
            # Розрахувати ціни TP
            tp_levels = tp_config.calculate_tp_prices(entry_price, side, strategy_name)
            
            if not tp_levels:
                logger.warning(f"⚠️ No TP levels calculated for {symbol}")
                return {'status': 'warning', 'message': 'No TP levels'}
            
            # Отримати інструмент для округлення
            lot, tick = self.get_instr(norm)
            if not lot or not tick:
                logger.error(f"❌ Failed to get instrument info for {symbol}")
                return {'status': 'error', 'message': 'Failed to get instrument info'}
            
            qty_step = float(lot['qtyStep'])
            tick_size = float(tick['tickSize'])
            
            logger.info(f"🎯 SETTING {len(tp_levels)} TP LEVELS for {symbol} ({strategy['name_ua']})")
            
            set_tps = []
            for i, level in enumerate(tp_levels):
                tp_price = self.round_val(level['price'], tick_size)
                tp_qty_pct = level['quantity_percent']
                tp_qty = self.round_val(quantity * (tp_qty_pct / 100), qty_step)
                
                # Мінімальна кількість
                min_qty = float(lot['minOrderQty'])
                if tp_qty < min_qty:
                    tp_qty = min_qty
                
                logger.info(f"   Level {i+1}: Price={tp_price} ({level['profit_percent']:.2f}%), "
                           f"Qty={tp_qty} ({tp_qty_pct}%)")
                
                # Встановити TP через limit order
                try:
                    # Для TP потрібен зворотний side
                    tp_side = 'Sell' if side == 'Buy' else 'Buy'
                    
                    result = self.session.place_order(
                        category="linear",
                        symbol=norm,
                        side=tp_side,
                        orderType="Limit",
                        qty=str(tp_qty),
                        price=str(tp_price),
                        reduceOnly=True,  # Важливо!
                        timeInForce="GTC"
                    )
                    
                    if result.get('retCode') == 0:
                        order_id = result['result']['orderId']
                        set_tps.append({
                            'level': i + 1,
                            'price': tp_price,
                            'quantity': tp_qty,
                            'profit_percent': level['profit_percent'],
                            'order_id': order_id
                        })
                        logger.info(f"   ✅ TP Level {i+1} set: Order ID {order_id}")
                    else:
                        logger.error(f"   ❌ Failed to set TP Level {i+1}: {result}")
                
                except Exception as e:
                    logger.error(f"   ❌ Error setting TP Level {i+1}: {e}")
            
            if set_tps:
                logger.info(f"✅ Successfully set {len(set_tps)}/{len(tp_levels)} TP levels for {symbol}")
                return {
                    'status': 'ok',
                    'tp_levels': len(set_tps),
                    'strategy': strategy['name_ua'],
                    'levels': set_tps
                }
            else:
                logger.warning(f"⚠️ No TP levels were set for {symbol}")
                return {'status': 'warning', 'message': 'Failed to set TP levels'}
        
        except Exception as e:
            logger.error(f"❌ Error in set_take_profit_levels: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def apply_tp_from_json(self, data: Dict, symbol: str, entry_price: float, 
                          side: str, quantity: float):
        """
        Застосувати TP з JSON повідомлення
        
        Підтримує різні формати:
        1. "tpStrategy": "balanced" - використати стратегію
        2. "takeProfitPercent": 2.0 - один рівень TP
        3. "tpLevels": [{percent: 1.0, quantity_percent: 50}, ...] - власні рівні
        """
        try:
            # Перевірка 1: Чи є tpStrategy?
            if 'tpStrategy' in data:
                strategy = data['tpStrategy']
                logger.info(f"📊 Using TP strategy from JSON: {strategy}")
                return self.set_take_profit_levels(symbol, entry_price, side, quantity, strategy)
            
            # Перевірка 2: Чи є tpLevels (custom)?
            elif 'tpLevels' in data and isinstance(data['tpLevels'], list):
                tp_levels = data['tpLevels']
                logger.info(f"📊 Using custom TP levels from JSON: {len(tp_levels)} levels")
                tp_config.set_custom_targets(tp_levels)
                return self.set_take_profit_levels(symbol, entry_price, side, quantity, 'custom')
            
            # Перевірка 3: Чи є takeProfitPercent (single)?
            elif 'takeProfitPercent' in data:
                tp_pct = float(data['takeProfitPercent'])
                logger.info(f"📊 Using single TP from JSON: {tp_pct}%")
                # Створити single strategy
                single_target = [{'percent': tp_pct, 'quantity_percent': 100}]
                tp_config.set_custom_targets(single_target)
                return self.set_take_profit_levels(symbol, entry_price, side, quantity, 'custom')
            
            # Якщо нічого немає - використати дефолтну стратегію
            else:
                logger.info(f"📊 Using default TP strategy: {tp_config.current_strategy}")
                return self.set_take_profit_levels(symbol, entry_price, side, quantity)
        
        except Exception as e:
            logger.error(f"❌ Error applying TP from JSON: {e}")
            return {'status': 'error', 'message': str(e)}

bot_instance = BybitTradingBot()
