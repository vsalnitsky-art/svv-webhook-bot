"""
Bot Module - Trading Logic & Managers 🤖
"""
import logging
import threading
import time
import decimal
from datetime import datetime
import requests
from pybit.unified_trading import HTTP

from bot_config import config
from config import get_api_credentials
from statistics_service import stats_service

logger = logging.getLogger(__name__)

# === HELPER: TELEGRAM ===
def send_telegram_message(text):
    clean = text.replace("<b>", "").replace("</b>", "").replace("\n", " | ")
    logger.info(f"\n🔔 [BOT]: {clean}")
    # Тут можна розкоментувати реальну відправку, коли налаштуєте
    # try:
    #     requests.post(f"https://api.telegram.org/bot{config.TG_BOT_TOKEN}/sendMessage", 
    #                   json={"chat_id": config.TG_CHAT_ID, "text": text, "parse_mode": "HTML"})
    # except: pass

class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)
        logger.info("✅ Bybit Connected (Bot Module)")

    def normalize_symbol(self, symbol): return symbol.replace('.P', '')

    def get_available_balance(self, currency="USDT"):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            if b.get('retCode') != 0: return None
            for acc in b.get('result', {}).get('list', []):
                for c in acc.get('coin', []):
                    if c.get('coin') == currency: return float(c.get('walletBalance', 0))
            return None
        except: return None

    def get_current_price(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            r = self.session.get_tickers(category="linear", symbol=norm)
            if r.get('retCode')==0: return float(r['result']['list'][0]['lastPrice'])
        except: return None

    def get_instrument_info(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            r = self.session.get_instruments_info(category="linear", symbol=norm)
            if r.get('retCode')==0: return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: return None, None

    def set_leverage(self, symbol, leverage):
        try:
            self.session.set_leverage(category="linear", symbol=self.normalize_symbol(symbol), buyLeverage=str(leverage), sellLeverage=str(leverage))
        except: pass

    def get_position_size(self, symbol):
        try:
            r = self.session.get_positions(category="linear", symbol=self.normalize_symbol(symbol))
            if r['retCode']==0: return float(r['result']['list'][0]['size'])
        except: return 0.0

    def round_qty(self, qty, step):
        if step <= 0: return qty
        import decimal
        d = abs(decimal.Decimal(str(step)).as_tuple().exponent)
        return round(qty // step * step, d)

    def round_price(self, price, tick):
        if tick <= 0: return price
        import decimal
        d = abs(decimal.Decimal(str(tick)).as_tuple().exponent)
        return round(price // tick * tick, d)

    # === ВХІД ===
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            norm = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm) > 0: return {"status": "ignored"}
            
            risk = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            lev = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            
            # TP / SL
            json_tp_price = data.get('takeProfit')
            json_tp_percent = data.get('takeProfitPercent')
            json_sl_price = data.get('stopLoss')
            json_sl_percent = data.get('stopLossPercent')
            
            price = self.get_current_price(norm)
            lot, tick = self.get_instrument_info(norm)
            if not price or not lot: return {"status": "error"}
            
            bal = self.get_available_balance()
            if not bal: return {"status": "error_balance"}
            
            qty_step = float(lot['qtyStep'])
            tick_size = float(tick['tickSize'])
            min_qty = float(lot['minOrderQty'])
            
            raw_qty = (bal * (risk/100) * 0.98 * lev) / price
            qty = self.round_qty(raw_qty, qty_step)
            
            if qty < min_qty:
                if bal > (min_qty*price/lev)*1.05: qty = min_qty
                else: return {"status": "error_min_qty"}
            
            self.set_leverage(norm, lev)
            
            # 1. Маркет Ордер
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            logger.info(f"✅ OPENED: {action} {qty} {norm} @ {price}")
            
            # 2. STOP LOSS
            sl_target = 0.0
            if json_sl_price: sl_target = float(json_sl_price)
            elif json_sl_percent:
                sl_pct = float(json_sl_percent)
                sl_target = price * (1 - sl_pct/100) if action == "Buy" else price * (1 + sl_pct/100)
            else:
                sl_pct = 1.5
                sl_target = price * (1 - sl_pct/100) if action == "Buy" else price * (1 + sl_pct/100)
            
            if sl_target > 0:
                sl_rounded = self.round_price(sl_target, tick_size)
                self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl_rounded), positionIdx=0)
                logger.info(f"🛡️ Fixed SL Set: {sl_rounded}")

            # 3. TAKE PROFIT SPLIT
            direction = 1 if action == "Buy" else -1
            final_tp_price = 0.0
            
            if json_tp_price and float(json_tp_price) > 0:
                final_tp_price = float(json_tp_price)
            elif json_tp_percent and float(json_tp_percent) > 0:
                tp_pct = float(json_tp_percent)
                final_tp_price = price * (1 + (tp_pct/100) * direction)
            else:
                final_tp_price = price * (1 + 0.03 * direction) # +3%

            total_dist = abs(final_tp_price - price)
            tp1_price = self.round_price(price + (total_dist * 0.40 * direction), tick_size)
            tp2_price = self.round_price(final_tp_price, tick_size)
            
            qty1 = self.round_qty(qty * 0.5, qty_step)
            qty2 = self.round_qty(qty - qty1, qty_step)
            
            tp_side = "Sell" if action=="Buy" else "Buy"
            
            if qty1 >= min_qty:
                self.session.place_order(category="linear", symbol=norm, side=tp_side, orderType="Limit", qty=str(qty1), price=str(tp1_price), reduceOnly=True)
                logger.info(f"🎯 TP1 (40%): {tp1_price}")
            
            if qty2 >= min_qty:
                self.session.place_order(category="linear", symbol=norm, side=tp_side, orderType="Limit", qty=str(qty2), price=str(tp2_price), reduceOnly=True)
                logger.info(f"🎯 TP2 (100%): {tp2_price}")

            return {"status": "success"}
        except Exception as e:
            logger.error(f"Order Error: {e}")
            return {"status": "error"}

    def sync_trades(self, days=30):
        try:
            now = datetime.now()
            for i in range(0, days, 7):
                end = now - timedelta(days=i)
                start = end - timedelta(days=min(7, days-i))
                r = self.session.get_closed_pnl(category="linear", startTime=int(start.timestamp()*1000), endTime=int(end.timestamp()*1000), limit=50)
                if r['retCode']==0:
                    for t in r['result']['list']:
                        stats_service.save_trade({
                            'order_id': t['orderId'], 'symbol': t['symbol'],
                            'side': 'Long' if t['side']=='Sell' else 'Short',
                            'qty': float(t['qty']), 'entry_price': float(t['avgEntryPrice']),
                            'exit_price': float(t['avgExitPrice']), 'pnl': float(t['closedPnl']),
                            'exit_time': datetime.fromtimestamp(int(t['updatedTime'])/1000),
                            'is_win': float(t['closedPnl'])>0,
                            'exit_reason': 'Manual/TP/SL'
                        })
        except: pass

# Створюємо екземпляр бота
bot_instance = BybitTradingBot()

# === МЕНЕДЖЕРИ ===
class SmartTradeManager:
    def __init__(self, bot, scanner):
        self.bot = bot
        self.scanner = scanner
        self.running = True
        self.last_log = 0
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.manage()
            except: pass
            time.sleep(5)
    def manage(self):
        r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']!=0: return
        
        should_log = (time.time() - self.last_log) > 15
        if should_log: self.last_log = time.time()

        for p in r['result']['list']:
            if float(p['size'])==0: continue
            sym = p['symbol']
            if should_log:
                stats_service.save_monitor_log({
                    'symbol': sym, 'price': float(p['avgPrice']), 'pnl': float(p['unrealisedPnl']),
                    'rsi': self.scanner.get_current_rsi(sym), 'pressure': self.scanner.get_market_pressure(sym)
                })

class PassiveManager:
    def __init__(self, bot, scanner):
        self.bot = bot
        self.scanner = scanner
        self.known = set()
        self.running = True
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.check()
            except: pass
            time.sleep(5)
    def check(self):
        r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']!=0: return
        curr = set(p['symbol'] for p in r['result']['list'] if float(p['size'])>0)
        closed = self.known - curr
        for sym in closed:
            try: stats_service.delete_coin_history(sym)
            except: pass
        self.known = curr
