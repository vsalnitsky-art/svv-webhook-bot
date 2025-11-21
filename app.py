from flask import Flask, request, jsonify
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests
import json
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from config import get_api_credentials

# --- НАЛАШТУВАННЯ EMAIL ---
# ⚠️ ЗАМІНІТЬ НА ВАШІ ДАНІ АБО ВИКОРИСТОВУЙТЕ ЗМІННІ СЕРЕДОВИЩА
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "vsalnitsky@gmail.com"
# Тут вставте 16-значний пароль додатка Google (НЕ звичайний пароль)
EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx" 
EMAIL_RECEIVER = "vsalnitsky@gmail.com"

# Налаштування
PORT = 10000
MONITOR_INTERVAL = 5 

def keep_alive():
    """Функція для підтримки сервера активним (External URL)"""
    time.sleep(10)
    while True:
        try:
            requests.get('https://svv-webhook-bot.onrender.com/health', timeout=5)
        except Exception as e:
            print(f"⚠️ Keep-alive ping failed: {e}")
        time.sleep(600)

app = Flask(__name__)
logging.basicConfig(level=logging.ERROR)
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
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації: {e}")
            raise

    # ... (Функції get_available_balance, normalize_symbol, get_current_price, set_leverage, get_instrument_info, round_qty, round_price, get_position_size залишаються без змін)
    
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
            if "110043" in str(e): return True
            logger.error(f"❌ Leverage error: {e}")
            return False

    def get_instrument_info(self, symbol):
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

    def get_position_size(self, symbol):
        try:
            norm_symbol = self.normalize_symbol(symbol)
            resp = self.session.get_positions(category="linear", symbol=norm_symbol)
            if resp['retCode'] == 0 and resp['result']['list']:
                return float(resp['result']['list'][0]['size'])
            return 0.0
        except Exception as e:
            logger.error(f"Error checking position: {e}")
            return 0.0

    # --- НОВИЙ ФУНКЦІОНАЛ: ОТРИМАННЯ СТАТИСТИКИ ---
    def get_pnl_stats(self, days=7):
        """Отримує статистику PnL за останні N днів"""
        try:
            end_time = int(time.time() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Запит до Bybit API (Closed PnL)
            resp = self.session.get_closed_pnl(category="linear", startTime=start_time, endTime=end_time, limit=100)
            
            if resp['retCode'] != 0:
                return None, f"API Error: {resp['retMsg']}"

            pnl_list = resp['result']['list']
            
            stats = {
                "total_trades": len(pnl_list),
                "total_pnl": 0.0,
                "win_trades": 0,
                "loss_trades": 0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "details": []
            }
            
            for trade in pnl_list:
                pnl = float(trade['closedPnl'])
                stats["total_pnl"] += pnl
                
                if pnl > 0: stats["win_trades"] += 1
                else: stats["loss_trades"] += 1
                
                if pnl > stats["best_trade"]: stats["best_trade"] = pnl
                if pnl < stats["worst_trade"]: stats["worst_trade"] = pnl
                
                # Деталі для звіту
                symbol = trade['symbol']
                side = trade['side'] # Buy/Sell (на момент закриття)
                fill_time = datetime.fromtimestamp(int(trade['updatedTime']) / 1000).strftime('%Y-%m-%d %H:%M')
                stats["details"].append(f"{fill_time} | {symbol} | PnL: {pnl:.2f} USDT")

            return stats, None

        except Exception as e:
            return None, str(e)

    def place_order(self, data):
        # ... (Ваша існуюча логіка place_order БЕЗ ЗМІН)
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            if not action or not symbol: return {"status": "error", "error": "Missing params"}
            
            norm_symbol = self.normalize_symbol(symbol)
            current_pos_size = self.get_position_size(norm_symbol)
            if current_pos_size > 0:
                return {"status": "ignored", "message": "Position already exists"}

            riskPercent = float(data.get('riskPercent', 5.0))
            leverage = int(data.get('leverage', 20))
            tpPercent = float(data.get('takeProfitPercent', 0.0))
            slPercent = float(data.get('stopLossPercent', 0.0))

            cur_price = self.get_current_price(norm_symbol)
            if not cur_price: return {"status": "error", "error": "No price"}
            
            lot_filter, price_filter = self.get_instrument_info(norm_symbol)
            if not lot_filter: return {"status": "error", "error": "No instrument info"}

            qty_step = float(lot_filter['qtyStep'])
            min_qty = float(lot_filter['minOrderQty'])
            tick_size = float(price_filter['tickSize'])

            balance = self.get_available_balance()
            if not balance: return {"status": "error", "error": "No balance"}

            margin = (balance * (riskPercent / 100)) * 0.98
            total_value = margin * leverage
            raw_qty = total_value / cur_price
            
            final_qty = self.round_qty(raw_qty, qty_step)
            if final_qty < min_qty: return {"status": "error", "error": "Qty too small"}

            self.set_leverage(norm_symbol, leverage)

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

            sl_price = 0
            if slPercent > 0:
                if action == "Buy": sl_price = cur_price * (1 - slPercent / 100)
                else: sl_price = cur_price * (1 + slPercent / 100)
                sl_price = self.round_price(sl_price, tick_size)
                self.session.set_trading_stop(category="linear", symbol=norm_symbol, stopLoss=str(sl_price), positionIdx=0)

            if tpPercent > 0:
                qty_tp1 = self.round_qty(final_qty * 0.50, qty_step)
                qty_tp2 = self.round_qty(final_qty * 0.35, qty_step)
                qty_tp3 = self.round_qty(final_qty - qty_tp1 - qty_tp2, qty_step)

                tp1_dist = tpPercent * 0.5
                tp2_dist = tpPercent * 1.0
                tp3_dist = tpPercent * 1.5

                if action == "Buy":
                    tp1_price = cur_price * (1 + tp1_dist / 100)
                    tp2_price = cur_price * (1 + tp2_dist / 100)
                    tp3_price = cur_price * (1 + tp3_dist / 100)
                    tp_side = "Sell"
                else:
                    tp1_price = cur_price * (1 - tp1_dist / 100)
                    tp2_price = cur_price * (1 - tp2_dist / 100)
                    tp3_price = cur_price * (1 - tp3_dist / 100)
                    tp_side = "Buy"

                tp1_price = self.round_price(tp1_price, tick_size)
                tp2_price = self.round_price(tp2_price, tick_size)
                tp3_price = self.round_price(tp3_price, tick_size)

                if qty_tp1 >= min_qty:
                    self.session.place_order(category="linear", symbol=norm_symbol, side=tp_side, orderType="Limit", qty=str(qty_tp1), price=str(tp1_price), reduceOnly=True, timeInForce="GTC")
                if qty_tp2 >= min_qty:
                    self.session.place_order(category="linear", symbol=norm_symbol, side=tp_side, orderType="Limit", qty=str(qty_tp2), price=str(tp2_price), reduceOnly=True, timeInForce="GTC")
                if qty_tp3 >= min_qty:
                    self.session.place_order(category="linear", symbol=norm_symbol, side=tp_side, orderType="Limit", qty=str(qty_tp3), price=str(tp3_price), reduceOnly=True, timeInForce="GTC")

            return {"status": "success", "symbol": norm_symbol}

        except Exception as e:
            logger.error(f"💥 Order Error: {e}")
            return {"status": "error", "error": str(e)}

bot = BybitTradingBot()

# --- МОНІТОРИНГ БЕЗУБИТКУ (БЕЗ ЗМІН) ---
class BreakevenMonitor:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.running = True

    def start(self):
        thread = threading.Thread(target=self.loop)
        thread.daemon = True
        thread.start()

    def loop(self):
        while self.running:
            try:
                self.check_positions()
            except Exception as e:
                logger.error(f"⚠️ Monitor Error: {e}")
            time.sleep(MONITOR_INTERVAL)

    def check_positions(self):
        positions = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if positions['retCode'] != 0: return

        for pos in positions['result']['list']:
            size = float(pos['size'])
            if size == 0: continue

            symbol = pos['symbol']
            side = pos['side']
            entry_price = float(pos['avgPrice'])
            stop_loss = float(pos.get('stopLoss', 0))
            
            is_breakeven = False
            if side == "Buy" and stop_loss >= entry_price: is_breakeven = True
            if side == "Sell" and stop_loss > 0 and stop_loss <= entry_price: is_breakeven = True
            if is_breakeven: continue

            orders = self.bot.session.get_open_orders(category="linear", symbol=symbol)
            tp_orders_count = 0
            if orders['retCode'] == 0:
                for o in orders['result']['list']:
                    if o['reduceOnly'] and o['orderType'] == 'Limit':
                        tp_orders_count += 1
            
            if tp_orders_count > 0 and tp_orders_count < 3:
                new_sl = entry_price
                try:
                    self.bot.session.set_trading_stop(category="linear", symbol=symbol, stopLoss=str(new_sl), positionIdx=0)
                except Exception as e:
                    logger.error(f"❌ Failed to move SL: {e}")

monitor = BreakevenMonitor(bot)
monitor.start()

# --- НОВІ ФУНКЦІЇ: EMAIL REPORT ---
def send_email(subject, body):
    """Відправка листа"""
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, text)
        server.quit()
        return True
    except Exception as e:
        logger.error(f"Email Error: {e}")
        return False

def generate_report(days):
    """Генерація тексту звіту"""
    stats, error = bot.get_pnl_stats(days=days)
    if error: return f"Error getting stats: {error}"
    
    balance = bot.get_available_balance()
    
    report = f"📊 TRADING REPORT ({days} days)\n"
    report += f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += f"💰 Current Balance: {balance:.2f} USDT\n"
    report += "-" * 30 + "\n"
    report += f"Total Trades: {stats['total_trades']}\n"
    report += f"Win Trades: {stats['win_trades']}\n"
    report += f"Loss Trades: {stats['loss_trades']}\n"
    report += f"💵 NET PnL: {stats['total_pnl']:.2f} USDT\n"
    report += "-" * 30 + "\n"
    report += f"Best Trade: {stats['best_trade']:.2f}\n"
    report += f"Worst Trade: {stats['worst_trade']:.2f}\n"
    report += "\nLast Trades:\n"
    
    # Останні 10 угод
    for det in stats['details'][:10]:
        report += det + "\n"
        
    return report

# --- WEBHOOK & ROUTES ---
def parse_tradingview_data(raw_data):
    try:
        if isinstance(raw_data, dict): return raw_data
        if isinstance(raw_data, str): return json.loads(raw_data)
    except:
        match = re.search(r'\{[^}]+\}', str(raw_data))
        if match: return json.loads(match.group())
    return {}

@app.route('/')
def home(): return "Bot Active"

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"})

# 🆕 НОВИЙ ЕНДПОІНТ ДЛЯ ЗВІТУ
# Викликати: https://ваш-урл.onrender.com/report?days=7
@app.route('/report', methods=['GET'])
def trigger_report():
    days = request.args.get('days', default=7, type=int)
    
    # Запускаємо в окремому потоці, щоб не блокувати сервер
    def process_report():
        text = generate_report(days)
        send_email(f"Trading Report ({days} days)", text)
        
    threading.Thread(target=process_report).start()
    return jsonify({"status": "Report sending initiated", "days": days})

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
