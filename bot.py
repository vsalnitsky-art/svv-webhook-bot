import logging
import decimal
from pybit.unified_trading import HTTP
from bot_config import config
from config import get_api_credentials
from statistics_service import stats_service

logger = logging.getLogger(__name__)

class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)
        self.position_tracker = {}

    def normalize(self, s): return s.replace('.P', '')

    def get_bal(self):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            for acc in b['result']['list']:
                for c in acc['coin']:
                    if c['coin'] == "USDT": return float(c['walletBalance'])
        except: return 0.0

    def get_price(self, s):
        try: return float(self.session.get_tickers(category="linear", symbol=self.normalize(s))['result']['list'][0]['lastPrice'])
        except: return 0.0

    def get_instr(self, s):
        try:
            r = self.session.get_instruments_info(category="linear", symbol=self.normalize(s))
            return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: return None, None

    def round_val(self, val, step):
        import decimal
        d = abs(decimal.Decimal(str(step)).as_tuple().exponent)
        return round(val // step * step, d)

    def set_lev(self, s, l):
        try: self.session.set_leverage(category="linear", symbol=self.normalize(s), buyLeverage=str(l), sellLeverage=str(l))
        except: pass

    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            norm = self.normalize(symbol)
            
            risk = float(data.get('riskPercent', 5.0))
            lev = int(data.get('leverage', 20))
            
            price = self.get_price(norm)
            lot, tick = self.get_instr(norm)
            if not price or not lot: return {"status": "error"}
            
            bal = self.get_bal()
            if bal < 5: return {"status": "no_balance"}

            qty_step = float(lot['qtyStep'])
            min_qty = float(lot['minOrderQty'])
            tick_size = float(tick['tickSize'])
            
            raw_qty = (bal * (risk/100) * 0.98 * lev) / price
            qty = self.round_val(raw_qty, qty_step)
            if qty < min_qty: qty = min_qty
            
            self.set_lev(norm, lev)
            
            # 1. MARKET ORDER
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            logger.info(f"✅ OPEN: {action} {symbol}")
            
            # 2. TRAILING STOP
            if symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]: tr_pct = 0.5
            elif any(x in symbol for x in ["SOL","XRP","ADA"]): tr_pct = 1.5
            else: tr_pct = 3.0
            
            dist = self.round_val(price * (tr_pct/100), tick_size)
            sl = price - dist if action == "Buy" else price + dist
            sl = self.round_val(sl, tick_size)
            
            self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl), trailingStop=str(dist), positionIdx=0)
            
            # 3. SPLIT TP
            tp_price = data.get('takeProfit')
            tp_pct = data.get('takeProfitPercent')
            target = 0.0
            direction = 1 if action == "Buy" else -1
            
            if tp_price: target = float(tp_price)
            elif tp_pct: target = price * (1 + (float(tp_pct)/100) * direction)
            else: target = price * (1 + 0.03 * direction)
            
            dist_tp = abs(target - price)
            tp1 = self.round_val(price + (dist_tp * 0.4 * direction), tick_size)
            tp2 = self.round_val(target, tick_size)
            
            q1 = self.round_val(qty * 0.5, qty_step)
            q2 = self.round_val(qty - q1, qty_step)
            
            side_exit = "Sell" if action == "Buy" else "Buy"
            
            if q1 >= min_qty:
                self.session.place_order(category="linear", symbol=norm, side=side_exit, orderType="Limit", qty=str(q1), price=str(tp1), reduceOnly=True)
            if q2 >= min_qty:
                self.session.place_order(category="linear", symbol=norm, side=side_exit, orderType="Limit", qty=str(q2), price=str(tp2), reduceOnly=True)

            return {"status": "ok"}
        except Exception as e:
            logger.error(f"Order Error: {e}")
            return {"status": "error"}

    def sync_history(self):
        # Проста синхронізація за 24 години для відображення
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
                        'exit_reason': 'Trailing/TP'
                    })
        except: pass

bot_instance = BybitTradingBot()