import logging
import decimal
from datetime import datetime
from pybit.unified_trading import HTTP
from bot_config import config
from config import get_api_credentials
from statistics_service import stats_service

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
                return {"status": "ok", "sl_price": sl_price}
            else:
                logger.info(f"⚠️ NO SL in JSON for {symbol}. Position opened without SL.")
                return {"status": "ok", "sl_price": None}

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

bot_instance = BybitTradingBot()
