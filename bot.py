import logging
import decimal
import time
from datetime import datetime
from pybit.unified_trading import HTTP
from bot_config import config
from config import get_api_credentials
from settings_manager import settings
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
            
    def get_available_balance(self):
        return self.get_bal()

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
        except: pass

    # === СИНХРОНІЗАЦІЯ ІСТОРІЇ ===
    def sync_trades(self, days=7):
        """Завантажує історію закритих угод з Bybit у БД"""
        try:
            end_time = int(time.time() * 1000)
            start_time = end_time - (days * 24 * 60 * 60 * 1000)
            
            # ЗБІЛЬШЕНО ЛІМІТ ДО 100
            r = self.session.get_closed_pnl(category="linear", startTime=start_time, endTime=end_time, limit=100)
            
            if r['retCode'] == 0:
                for t in r['result']['list']:
                    trade_side = 'Long' if t['side'] == 'Sell' else 'Short'
                    
                    stats_service.save_trade({
                        'order_id': t['orderId'], 
                        'symbol': t['symbol'],
                        'side': trade_side,
                        'qty': float(t['qty']), 
                        'entry_price': float(t['avgEntryPrice']),
                        'exit_price': float(t['avgExitPrice']), 
                        'pnl': float(t['closedPnl']),
                        'exit_time': datetime.fromtimestamp(int(t['updatedTime'])/1000),
                        'exit_reason': 'Signal/TP/SL'
                    })
                logger.info(f"Synced trades for last {days} days")
            else:
                logger.warning(f"Sync API Error: {r}")
                
        except Exception as e:
            logger.error(f"Sync trades error: {e}")

    # === ОСНОВНА ФУНКЦІЯ ОРДЕРІВ ===
    def place_order(self, data):
        try:
            action = data.get('action') # Buy/Sell/Close
            symbol = data.get('symbol')
            norm = self.normalize(symbol)
            
            # --- CLOSE LOGIC ---
            if action == "Close":
                direction = data.get('direction') 
                logger.info(f"🛑 CLOSE SIGNAL: {symbol} | Dir: {direction}")
                
                pos_data = self.session.get_positions(category="linear", symbol=norm)
                target_position = None
                for p in pos_data['result']['list']:
                    if float(p['size']) > 0:
                        target_position = p
                        break
                
                if not target_position:
                    return {"status": "ignored", "reason": "No position"}
                
                current_side = target_position['side'] # Buy/Sell
                
                if (direction == "Long" and current_side == "Buy") or (direction == "Short" and current_side == "Sell"):
                    close_side = "Sell" if current_side == "Buy" else "Buy"
                    self.session.place_order(category="linear", symbol=norm, side=close_side, orderType="Market", qty=target_position['size'], reduceOnly=True)
                    try: self.session.cancel_all_orders(category="linear", symbol=norm)
                    except: pass
                    return {"status": "ok", "message": f"Closed {direction}"}
                else:
                    return {"status": "ignored", "reason": "Direction mismatch"}

            # --- OPEN LOGIC ---
            pos_data = self.session.get_positions(category="linear", symbol=norm)
            for p in pos_data['result']['list']:
                if float(p['size']) > 0:
                    return {"status": "ignored", "reason": "Position exists"}
            
            risk = float(data.get('riskPercent', settings.get('riskPercent')))
            lev = int(data.get('leverage', settings.get('leverage')))
            
            price = self.get_price(norm)
            lot, tick = self.get_instr(norm)
            if not price or not lot: return {"status": "error", "reason": "No price/info"}
            
            bal = self.get_bal()
            if bal < 5: return {"status": "no_balance", "balance": bal}

            qty_step = float(lot['qtyStep'])
            min_qty = float(lot['minOrderQty'])
            tick_size = float(tick['tickSize'])
            
            raw_qty = (bal * (risk/100) * 0.98 * lev) / price
            qty = self.round_val(raw_qty, qty_step)
            if qty < min_qty: qty = min_qty
            
            self.set_lev(norm, lev)
            
            logger.info(f"🚀 OPEN {action} {symbol} | Qty: {qty}")
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            
            sl_pct = float(data.get('stopLossPercent', settings.get('fixedSL')))
            if sl_pct > 0:
                sl_dist = price * (sl_pct / 100)
                sl_price = price - sl_dist if action == "Buy" else price + sl_dist
                sl_price = self.round_val(sl_price, tick_size)
                self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl_price), positionIdx=0)

            self._place_take_profits(norm, action, price, qty, data, tick_size, qty_step)

            return {"status": "ok", "qty": qty}

        except Exception as e:
            logger.error(f"Trading Error: {e}")
            return {"status": "error", "reason": str(e)}

    def _place_take_profits(self, symbol, side, entry_price, total_qty, data, tick_size, qty_step):
        try:
            tp_mode = settings.get("tp_mode")
            logger.info(f"Setting TP | Mode: {tp_mode}")

            if tp_mode == "None": return

            exit_side = "Sell" if side == "Buy" else "Buy"
            
            if tp_mode == "Fixed_1_50":
                percent = 0.01
                qty_to_close = self.round_val(total_qty * 0.5, qty_step)
                
                tp_price = entry_price * (1 + percent) if side == "Buy" else entry_price * (1 - percent)
                tp_price = self.round_val(tp_price, tick_size)
                
                if qty_to_close > 0:
                    self.session.place_order(category="linear", symbol=symbol, side=exit_side, orderType="Limit", qty=str(qty_to_close), price=str(tp_price), reduceOnly=True)

            elif tp_mode == "Ladder_3":
                total_tp_pct = float(data.get('takeProfitPercent', settings.get('fixedTP'))) / 100
                step_pct = total_tp_pct / 3
                
                q1 = self.round_val(total_qty * 0.33, qty_step)
                q2 = self.round_val(total_qty * 0.33, qty_step)
                q3 = self.round_val(total_qty - q1 - q2, qty_step) 
                
                targets = [(step_pct, q1), (step_pct * 2, q2), (total_tp_pct, q3)]
                
                for pct, q in targets:
                    if q <= 0: continue
                    tp_price = entry_price * (1 + pct) if side == "Buy" else entry_price * (1 - pct)
                    tp_price = self.round_val(tp_price, tick_size)
                    self.session.place_order(category="linear", symbol=symbol, side=exit_side, orderType="Limit", qty=str(q), price=str(tp_price), reduceOnly=True)

        except Exception as e:
            logger.error(f"TP Placement Error: {e}")

bot_instance = BybitTradingBot()