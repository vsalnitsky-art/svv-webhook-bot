"""
Bot Module - Trading Logic 🤖
Відповідає за взаємодію з біржею: ордери, баланс, трейлінг.
"""
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
        logger.info("✅ Bybit Connected (Bot Module)")
        self.position_tracker = {} # Для майбутніх функцій

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

    # === ГОЛОВНА ФУНКЦІЯ ВХОДУ ===
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            norm = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm) > 0: return {"status": "ignored"}
            
            # Налаштування
            risk = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            lev = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            
            # Ринкові дані
            price = self.get_current_price(norm)
            lot, tick = self.get_instrument_info(norm)
            if not price or not lot: return {"status": "error_market_data"}
            
            bal = self.get_available_balance()
            if not bal: return {"status": "error_balance"}
            
            # Розрахунок
            qty = self.round_qty((bal * (risk/100) * 0.98 * lev) / price, float(lot['qtyStep']))
            min_qty = float(lot['minOrderQty'])
            
            if qty < min_qty:
                cost = (min_qty * price) / lev
                if bal > cost * 1.05: qty = min_qty
                else: return {"status": "error_min_qty"}
            
            self.set_leverage(norm, lev)
            
            # 1. Ордер
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            logger.info(f"✅ OPENED: {action} {qty} {norm}")
            
            # 2. Трейлінг Стоп (40% / 100% логіка тут реалізована як "Розумний Трейлінг")
            # Можна повернути лімітки TP, якщо потрібно, але ми домовились про чистий трейлінг
            if symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]: tr_pct = 0.5
            elif any(x in symbol for x in ["SOL","XRP","ADA","AVAX","DOGE"]): tr_pct = 1.5
            else: tr_pct = 3.0
            
            dist = self.round_price(price * (tr_pct/100), float(tick['tickSize']))
            sl_price = price - dist if action == "Buy" else price + dist
            sl = self.round_price(sl_price, float(tick['tickSize']))
            
            self.session.set_trading_stop(
                category="linear", symbol=norm, 
                stopLoss=str(sl), trailingStop=str(dist), positionIdx=0
            )
            logger.info(f"🛡️ Trailing Set: {tr_pct}%")
            
            # 3. Split TP (Опціонально, якщо є в JSON)
            self._place_split_tp(data, norm, price, qty, action, float(lot['qtyStep']), float(tick['tickSize']))

            return {"status": "success"}
        except Exception as e:
            logger.error(f"Order Error: {e}")
            return {"status": "error"}

    def _place_split_tp(self, data, symbol, entry_price, total_qty, action, qty_step, tick_size):
        """Приватний метод для розстановки TP"""
        tp_price = data.get('takeProfit')
        tp_pct = data.get('takeProfitPercent')
        
        target = 0.0
        direction = 1 if action == "Buy" else -1
        
        if tp_price: target = float(tp_price)
        elif tp_pct: target = entry_price * (1 + (float(tp_pct)/100) * direction)
        else: return # Немає TP - працюємо тільки по трейлінгу
        
        dist = abs(target - entry_price)
        tp1 = self.round_price(entry_price + (dist * 0.4 * direction), tick_size)
        tp2 = self.round_price(target, tick_size)
        
        q1 = self.round_qty(total_qty * 0.5, qty_step)
        q2 = self.round_qty(total_qty - q1, qty_step)
        
        side = "Sell" if action == "Buy" else "Buy"
        try:
            self.session.place_order(category="linear", symbol=symbol, side=side, orderType="Limit", qty=str(q1), price=str(tp1), reduceOnly=True)
            self.session.place_order(category="linear", symbol=symbol, side=side, orderType="Limit", qty=str(q2), price=str(tp2), reduceOnly=True)
            logger.info(f"🎯 Split TP Set: {tp1} / {tp2}")
        except: pass

    # Синхронізація для P&L
    def sync_trades(self, days=30):
        from datetime import datetime, timedelta
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
                            'exit_reason': 'Trailing/TP'
                        })
        except: pass

# Створюємо екземпляр тут, щоб імпортувати в інші файли
bot_instance = BybitTradingBot()
