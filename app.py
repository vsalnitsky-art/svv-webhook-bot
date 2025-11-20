from flask import Flask, request, jsonify
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests
import json
import re
from config import get_api_credentials

# Налаштування
PORT = 10000
MONITOR_INTERVAL = 5  # Інтервал перевірки позицій для безубитку (сек)

def keep_alive():
    """Функція для підтримки сервера активним (External URL)"""
    time.sleep(10)
    while True:
        try:
            requests.get('https://svv-webhook-bot.onrender.com/health', timeout=5)
            print("🔄 Keep-alive ping sent")
        except Exception as e:
            print(f"⚠️ Keep-alive ping failed: {e}")
        time.sleep(600)

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

keep_alive_thread = threading.Thread(target=keep_alive)
keep_alive_thread.daemon = True
keep_alive_thread.start()

class BybitTradingBot:
    def __init__(self):
        try:
            api_key, api_secret = get_api_credentials()
            self.session = HTTP(
                testnet=False,
                api_key=api_key,
                api_secret=api_secret,
            )
            logger.info("✅ Бот ініціалізований")
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації: {e}")
            raise

    def get_available_balance(self, currency="USDT"):
        try:
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            if balance_info.get('retCode') != 0: return None
            
            for account in balance_info.get('result', {}).get('list', []):
                for coin in account.get('coin', []):
                    if coin.get('coin') == currency:
                        return float(coin.get('walletBalance', 0))
            return None
        except Exception as e:
            logger.error(f"❌ Error balance: {e}")
            return None

    def normalize_symbol(self, symbol):
        return symbol.replace('.P', '')

    def get_current_price(self, symbol):
        try:
            norm_symbol = self.normalize_symbol(symbol)
            resp = self.session.get_tickers(category="linear", symbol=norm_symbol)
            if resp.get('retCode') == 0 and resp['result']['list']:
                return float(resp['result']['list'][0]['lastPrice'])
            return None
        except:
            return None

    def set_leverage(self, symbol, leverage):
        try:
            norm_symbol = self.normalize_symbol(symbol)
            self.session.set_leverage(
                category="linear", symbol=norm_symbol, 
                buyLeverage=str(leverage), sellLeverage=str(leverage)
            )
            return True
        except Exception as e:
            if "110043" in str(e): return True # Вже встановлено
            logger.error(f"❌ Leverage error: {e}")
            return False

    def get_instrument_info(self, symbol):
        """Отримання даних про крок ціни та лот"""
        try:
            norm_symbol = self.normalize_symbol(symbol)
            resp = self.session.get_instruments_info(category="linear", symbol=norm_symbol)
            if resp.get('retCode') == 0:
                return resp['result']['list'][0]['lotSizeFilter'], resp['result']['list'][0]['priceFilter']
            return None, None
        except:
            return None, None

    def round_qty(self, qty, step):
        if step <= 0: return qty
        import decimal
        step_decimals = abs(decimal.Decimal(str(step)).as_tuple().exponent)
        return round(qty // step * step, step_decimals)

    def round_price(self, price, tick_size):
        if tick_size <= 0: return price
        import decimal
        tick_decimals = abs(decimal.Decimal(str(tick_size)).as_tuple().exponent)
        return round(price // tick_size * tick_size, tick_decimals)

    def place_order(self, data):
        try:
            # 1. Валідація
            action = data.get('action')
            symbol = data.get('symbol')
            if not action or not symbol: return {"status": "error", "error": "Missing params"}

            riskPercent = float(data.get('riskPercent', 5.0))
            leverage = int(data.get('leverage', 20))
            tpPercent = float(data.get('takeProfitPercent', 0.0))
            slPercent = float(data.get('stopLossPercent', 0.0))

            norm_symbol = self.normalize_symbol(symbol)
            
            # 2. Ціна та Інфо
            cur_price = self.get_current_price(norm_symbol)
            if not cur_price: return {"status": "error", "error": "No price"}
            
            lot_filter, price_filter = self.get_instrument_info(norm_symbol)
            if not lot_filter: return {"status": "error", "error": "No instrument info"}

            qty_step = float(lot_filter['qtyStep'])
            min_qty = float(lot_filter['minOrderQty'])
            tick_size = float(price_filter['tickSize'])

            # 3. Баланс та Кількість
            balance = self.get_available_balance()
            if not balance: return {"status": "error", "error": "No balance"}

            # Динамічний розрахунок: Margin * Lev
            margin = balance * (riskPercent / 100)
            total_value = margin * leverage
            raw_qty = total_value / cur_price
            
            final_qty = self.round_qty(raw_qty, qty_step)
            if final_qty < min_qty: return {"status": "error", "error": "Qty too small"}

            # 4. Плече
            self.set_leverage(norm_symbol, leverage)

            # 5. Вхід в позицію (Market)
            logger.info(f"🚀 {action} {norm_symbol} | Qty: {final_qty} | Price: {cur_price}")
            entry_order = self.session.place_order(
                category="linear",
                symbol=norm_symbol,
                side=action,
                orderType="Market",
                qty=str(final_qty),
                timeInForce="GTC"
            )
            
            if entry_order['retCode'] != 0:
                return {"status": "error", "error": entry_order['retMsg']}

            # 6. Стоп Лосс (Один на всю позицію)
            sl_price = 0
            if slPercent > 0:
                if action == "Buy":
                    sl_price = cur_price * (1 - slPercent / 100)
                else:
                    sl_price = cur_price * (1 + slPercent / 100)
                
                sl_price = self.round_price(sl_price, tick_size)
                
                # Встановлюємо SL для позиції
                self.session.set_trading_stop(
                    category="linear",
                    symbol=norm_symbol,
                    stopLoss=str(sl_price),
                    positionIdx=0 
                )

            # 7. Розділення TP (50% / 50%)
            if tpPercent > 0:
                # TP1: 50% відстані, 50% об'єму
                tp1_dist_percent = tpPercent / 2
                # TP2: 100% відстані, решта об'єму
                tp2_dist_percent = tpPercent

                qty_tp1 = self.round_qty(final_qty / 2, qty_step)
                qty_tp2 = self.round_qty(final_qty - qty_tp1, qty_step) # Решта

                tp1_price = 0
                tp2_price = 0

                if action == "Buy":
                    tp1_price = cur_price * (1 + tp1_dist_percent / 100)
                    tp2_price = cur_price * (1 + tp2_dist_percent / 100)
                    tp_side = "Sell"
                else:
                    tp1_price = cur_price * (1 - tp1_dist_percent / 100)
                    tp2_price = cur_price * (1 - tp2_dist_percent / 100)
                    tp_side = "Buy"

                tp1_price = self.round_price(tp1_price, tick_size)
                tp2_price = self.round_price(tp2_price, tick_size)

                # Розміщуємо лімітні ордери (Reduce Only)
                # TP1
                if qty_tp1 >= min_qty:
                    self.session.place_order(
                        category="linear", symbol=norm_symbol, side=tp_side,
                        orderType="Limit", qty=str(qty_tp1), price=str(tp1_price),
                        reduceOnly=True, timeInForce="GTC"
                    )
                    logger.info(f"🎯 TP1 Placed: {tp1_price} (50%)")

                # TP2
                if qty_tp2 >= min_qty:
                    self.session.place_order(
                        category="linear", symbol=norm_symbol, side=tp_side,
                        orderType="Limit", qty=str(qty_tp2), price=str(tp2_price),
                        reduceOnly=True, timeInForce="GTC"
                    )
                    logger.info(f"🎯 TP2 Placed: {tp2_price} (100%)")

            return {
                "status": "success",
                "symbol": norm_symbol,
                "qty": final_qty,
                "tp1": tp1_price,
                "tp2": tp2_price,
                "sl": sl_price
            }

        except Exception as e:
            logger.error(f"💥 Order Error: {e}")
            return {"status": "error", "error": str(e)}

bot = BybitTradingBot()

# --- МОНІТОРИНГ БЕЗУБИТКУ (BREAKEVEN) ---
class BreakevenMonitor:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.running = True

    def start(self):
        thread = threading.Thread(target=self.loop)
        thread.daemon = True
        thread.start()

    def loop(self):
        logger.info("🛡️ Breakeven Monitor Started")
        while self.running:
            try:
                self.check_positions()
            except Exception as e:
                logger.error(f"⚠️ Monitor Error: {e}")
            time.sleep(MONITOR_INTERVAL)

    def check_positions(self):
        # Отримуємо всі відкриті позиції
        # ВИПРАВЛЕНО: settlementCoin -> settleCoin
        positions = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        
        if positions['retCode'] != 0:
            logger.warning(f"⚠️ API Error in Monitor: {positions['retMsg']}")
            return

        for pos in positions['result']['list']:
            size = float(pos['size'])
            if size == 0: continue # Пропускаємо закриті

            symbol = pos['symbol']
            side = pos['side'] # Buy or Sell
            entry_price = float(pos['avgPrice'])
            stop_loss = float(pos.get('stopLoss', 0))
            cur_price = float(pos['markPrice'])

            # Якщо SL вже біля входу (безубиток вже активований) - пропускаємо
            is_breakeven = False
            if side == "Buy" and stop_loss >= entry_price: is_breakeven = True
            if side == "Sell" and stop_loss > 0 and stop_loss <= entry_price: is_breakeven = True
            
            if is_breakeven: continue

            # Перевіряємо наявність активних ордерів TP
            orders = self.bot.session.get_open_orders(category="linear", symbol=symbol)
            tp_orders = []
            if orders['retCode'] == 0:
                for o in orders['result']['list']:
                    # Шукаємо лімітні ордери на закриття (TP)
                    if o['reduceOnly'] and o['orderType'] == 'Limit':
                        tp_orders.append(o)
            
            # ЛОГІКА:
            # Якщо залишився тільки 1 TP ордер (а ставили 2), значить TP1 виконався
            if len(tp_orders) == 1: 
                new_sl = entry_price
                try:
                    self.bot.session.set_trading_stop(
                        category="linear",
                        symbol=symbol,
                        stopLoss=str(new_sl),
                        positionIdx=0
                    )
                    logger.info(f"✅ {symbol} SL Moved to Breakeven: {new_sl}")
                except Exception as e:
                    logger.error(f"❌ Failed to move SL: {e}")

# Запускаємо монітор
monitor = BreakevenMonitor(bot)
monitor.start()

# --- WEBHOOK ---
def parse_tradingview_data(raw_data):
    try:
        if isinstance(raw_data, dict): return raw_data
        if isinstance(raw_data, str):
            return json.loads(raw_data)
    except:
        match = re.search(r'\{[^}]+\}', str(raw_data))
        if match: return json.loads(match.group())
    return {}

@app.route('/')
def home(): return "Trading Bot Active 🤖"

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json(silent=True)
        if not data:
            data = parse_tradingview_data(request.get_data(as_text=True))
        
        if not data: return jsonify({"error": "No data"}), 400
        
        threading.Thread(target=bot.place_order, args=(data,)).start()
        return jsonify({"status": "processing"})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
