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
            action = data.get('action') # "Buy" or "Sell"
            symbol = data.get('symbol')
            norm = self.normalize(symbol)
            
            risk = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            lev = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            
            # Отримуємо ціну та параметри монети
            price = self.get_price(norm)
            lot, tick = self.get_instr(norm)
            if not price or not lot: 
                return {"status": "error", "reason": "Failed to get price/instrument info"}
            
            bal = self.get_bal()
            if bal < 5: 
                return {"status": "no_balance", "balance": bal}

            # Розрахунок кількості
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
            # Беремо SL, який прийшов в JSON. Якщо його немає або він 0 - SL не ставимо.
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
            logger.error(f"Order Placement Critical Error: {e}")
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
                        'exit_reason': 'Manual/SL'
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
                        'exit_reason': 'Manual/SL'
                    })
        except Exception as e:
            logger.error(f"Sync trades error: {e}")

    def get_available_balance(self):
        return self.get_bal()

bot_instance = BybitTradingBot()
