import logging, decimal, time
from datetime import datetime
from pybit.unified_trading import HTTP
from config import get_api_credentials; from settings_manager import settings; from statistics_service import stats_service
logger = logging.getLogger(__name__)
class BybitTradingBot:
    def __init__(self): k, s = get_api_credentials(); self.session = HTTP(testnet=False, api_key=k, api_secret=s)
    def normalize(self, s): return s.replace('.P', '')
    def get_bal(self):
        try: return float(next(c['walletBalance'] for a in self.session.get_wallet_balance(accountType="UNIFIED")['result']['list'] for c in a['coin'] if c['coin']=="USDT"))
        except: return 0.0
    def get_price(self, s): return float(self.session.get_tickers(category="linear", symbol=self.normalize(s))['result']['list'][0]['lastPrice'])
    def get_all_tickers(self): return self.session.get_tickers(category="linear")['result']['list']
    def get_instr(self, s):
        try: r = self.session.get_instruments_info(category="linear", symbol=self.normalize(s)); return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: return None, None
    def round_val(self, v, s): return round(v // s * s, abs(decimal.Decimal(str(s)).as_tuple().exponent))
    def sync_trades(self, days=7):
        try:
            now = int(time.time()*1000); start = now - (days*86400000); chunk = 7*86400000; cur_end = now
            while cur_end > start:
                cur_start = max(cur_end - chunk, start)
                r = self.session.get_closed_pnl(category="linear", startTime=int(cur_start), endTime=int(cur_end), limit=100)
                if r['retCode']==0:
                    for t in r['result']['list']:
                        stats_service.save_trade({'order_id':t['orderId'], 'symbol':t['symbol'], 'side':'Long' if t['side']=='Sell' else 'Short', 'qty':float(t['qty']), 'entry_price':float(t['avgEntryPrice']), 'exit_price':float(t['avgExitPrice']), 'pnl':float(t['closedPnl']), 'exit_time':datetime.fromtimestamp(int(t['updatedTime'])/1000)})
                cur_end = cur_start; time.sleep(0.2)
        except: pass
    def place_order(self, data):
        try:
            act, sym = data.get('action'), self.normalize(data.get('symbol'))
            if act == "Close":
                d = data.get('direction'); pos = self.session.get_positions(category="linear", symbol=sym)['result']['list']
                p = next((x for x in pos if float(x['size'])>0), None)
                if p and ((d=="Long" and p['side']=="Buy") or (d=="Short" and p['side']=="Sell")):
                    self.session.place_order(category="linear", symbol=sym, side="Sell" if p['side']=="Buy" else "Buy", orderType="Market", qty=p['size'], reduceOnly=True)
                    try: self.session.cancel_all_orders(category="linear", symbol=sym)
                    except: pass
                    return {"status": "ok"}
                return {"status": "ignored"}
            risk = float(data.get('riskPercent', settings.get('riskPercent'))); lev = int(data.get('leverage', settings.get('leverage')))
            price = self.get_price(sym); lot, tick = self.get_instr(sym)
            qty = self.round_val((self.get_bal() * (risk/100) * 0.98 * lev) / price, float(lot['qtyStep']))
            if qty < float(lot['minOrderQty']): return {"status": "skipped_min_qty"}
            self.session.set_leverage(category="linear", symbol=sym, buyLeverage=str(lev), sellLeverage=str(lev))
            self.session.place_order(category="linear", symbol=sym, side=act, orderType="Market", qty=str(qty))
            sl = float(data.get('sl_price', 0))
            if sl == 0: 
                sl_pct = float(settings.get('fixedSL', 1.5))
                sl = price * (1 - sl_pct/100) if act == "Buy" else price * (1 + sl_pct/100)
            self.session.set_trading_stop(category="linear", symbol=sym, stopLoss=str(self.round_val(sl, float(tick['tickSize']))), positionIdx=0)
            return {"status": "ok"}
        except Exception as e: return {"status": "error", "reason": str(e)}
bot_instance = BybitTradingBot()