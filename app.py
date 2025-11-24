from flask import Flask, request, jsonify, render_template_string
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

# --- НАЛАШТУВАННЯ EMAIL (Залишаємо як опцію) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "vsalnitsky@gmail.com"
EMAIL_PASSWORD = "hovx cuvd cypv tmtx" 
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
            logger.warning(f"⚠️ Keep-alive ping failed: {e}")
        time.sleep(600)

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
            logger.info("✅ Bybit API session initialized successfully.")
        except Exception as e:
            logger.error(f"❌ Помилка ініціалізації: {e}")
            raise

    def get_available_balance(self, currency="USDT"):
        try:
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            if balance_info.get('retCode') != 0: 
                logger.error(f"❌ Failed to get wallet balance: {balance_info}")
                return None
            
            for account in balance_info.get('result', {}).get('list', []):
                for coin in account.get('coin', []):
                    if coin.get('coin') == currency:
                        balance = float(coin.get('walletBalance', 0))
                        return balance
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
        except Exception as e:
            logger.error(f"❌ Error getting price: {e}")
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
            if "110043" in str(e): 
                return True
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

    def get_pnl_stats(self, days=7):
        logger.info(f"📊 Fetching PnL stats for last {days} days...")
        try:
            now = datetime.now()
            all_pnl_list = []
            
            # Розбиваємо запит на чанки по 7 днів через обмеження API
            for i in range(0, days, 7):
                current_chunk_days = min(7, days - i)
                chunk_end_time = now - timedelta(days=i)
                chunk_start_time = chunk_end_time - timedelta(days=current_chunk_days)
                
                ts_end = int(chunk_end_time.timestamp() * 1000)
                ts_start = int(chunk_start_time.timestamp() * 1000)
                
                resp = self.session.get_closed_pnl(category="linear", startTime=ts_start, endTime=ts_end, limit=100)
                
                if resp['retCode'] != 0:
                    logger.error(f"❌ API Error getting PnL: {resp['retMsg']}")
                    return None, f"API Error: {resp['retMsg']}"

                all_pnl_list.extend(resp['result']['list'])
                time.sleep(0.1) # Anti-spam delay

            logger.info(f"✅ Found {len(all_pnl_list)} trades in total.")
            
            stats = {
                "total_trades": len(all_pnl_list),
                "total_pnl": 0.0,
                "win_trades": 0,
                "loss_trades": 0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "details": [] # Тепер тут буде список словників, а не рядків
            }
            
            for trade in all_pnl_list:
                pnl = float(trade['closedPnl'])
                stats["total_pnl"] += pnl
                
                if pnl > 0: stats["win_trades"] += 1
                else: stats["loss_trades"] += 1
                
                if pnl > stats["best_trade"]: stats["best_trade"] = pnl
                if pnl < stats["worst_trade"]: stats["worst_trade"] = pnl
                
                symbol = trade['symbol']
                fill_time = datetime.fromtimestamp(int(trade['updatedTime']) / 1000).strftime('%Y-%m-%d %H:%M')
                
                # Зберігаємо структуровані дані
                stats["details"].append({
                    "time": fill_time,
                    "symbol": symbol,
                    "pnl": pnl,
                    "side": trade['side'],
                    "price": trade['avgExitPrice'],
                    "qty": trade['qty']
                })

            # Сортуємо за часом (нові зверху)
            stats["details"].sort(key=lambda x: x['time'], reverse=True)
            
            return stats, None

        except Exception as e:
            logger.error(f"❌ Exception in get_pnl_stats: {e}")
            return None, str(e)

    def place_order(self, data):
        # (Ваш код place_order залишається без змін, скорочую для читабельності)
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            if not action or not symbol: return {"status": "error", "error": "Missing params"}
            norm_symbol = self.normalize_symbol(symbol)
            
            current_pos_size = self.get_position_size(norm_symbol)
            if current_pos_size > 0: return {"status": "ignored", "message": "Position already exists"}

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
                category="linear", symbol=norm_symbol, side=action,
                orderType="Market", qty=str(final_qty), timeInForce="GTC"
            )
            
            if entry_order['retCode'] != 0: return {"status": "error", "error": entry_order['retMsg']}
            
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
                    tp1_price, tp2_price, tp3_price = cur_price * (1 + tp1_dist/100), cur_price * (1 + tp2_dist/100), cur_price * (1 + tp3_dist/100)
                    tp_side = "Sell"
                else:
                    tp1_price, tp2_price, tp3_price = cur_price * (1 - tp1_dist/100), cur_price * (1 - tp2_dist/100), cur_price * (1 - tp3_dist/100)
                    tp_side = "Buy"

                for q, p in [(qty_tp1, tp1_price), (qty_tp2, tp2_price), (qty_tp3, tp3_price)]:
                    if q >= min_qty:
                        self.session.place_order(category="linear", symbol=norm_symbol, side=tp_side, orderType="Limit", 
                                                 qty=str(q), price=str(self.round_price(p, tick_size)), reduceOnly=True, timeInForce="GTC")

            return {"status": "success", "symbol": norm_symbol}
        except Exception as e:
            logger.error(f"Order Error: {e}")
            return {"status": "error", "error": str(e)}

bot = BybitTradingBot()

# --- МОНІТОРИНГ БЕЗУБИТКУ ---
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
                pass
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
                try:
                    self.bot.session.set_trading_stop(category="linear", symbol=symbol, stopLoss=str(entry_price), positionIdx=0)
                except: pass

monitor = BreakevenMonitor(bot)
monitor.start()

# --- ЗВІТ В HTML ФОРМАТІ ---
@app.route('/report', methods=['GET'])
def report_page():
    days = request.args.get('days', default=7, type=int)
    logger.info(f"🖥️ Generating HTML report for {days} days...")
    
    stats, error = bot.get_pnl_stats(days=days)
    balance = bot.get_available_balance()
    
    if error:
        return f"<h1>Error</h1><p>{error}</p>"
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Trading Report</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #f4f6f8; margin: 0; padding: 20px; color: #333; }
            .container { max_width: 900px; margin: 0 auto; }
            .header-card { background: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; }
            .balance-box { text-align: right; }
            .balance-val { font-size: 1.5em; font-weight: bold; color: #2c3e50; }
            h1 { margin: 0; font-size: 1.5em; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 15px; margin-bottom: 25px; }
            .stat-card { background: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.03); }
            .stat-val { font-size: 1.4em; font-weight: bold; margin-bottom: 5px; }
            .stat-label { font-size: 0.85em; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
            .green { color: #00b894; }
            .red { color: #d63031; }
            table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
            th { background: #f8f9fa; text-align: left; padding: 15px; font-size: 0.9em; color: #666; font-weight: 600; }
            td { padding: 15px; border-bottom: 1px solid #eee; font-size: 0.95em; }
            tr:last-child td { border-bottom: none; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
            .badge-buy { background: #e3fcef; color: #00b894; }
            .badge-sell { background: #ffecec; color: #d63031; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header-card">
                <div>
                    <h1>📊 Trading Report</h1>
                    <span style="color: #888; font-size: 0.9em;">Last {{ days }} days • {{ date }}</span>
                </div>
                <div class="balance-box">
                    <div class="balance-val">${{ "%.2f"|format(balance if balance else 0) }}</div>
                    <div style="color: #888; font-size: 0.8em;">Current Balance</div>
                </div>
            </div>

            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-val">{{ stats.total_trades }}</div>
                    <div class="stat-label">Total Trades</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val green">{{ stats.win_trades }}</div>
                    <div class="stat-label">Wins</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val red">{{ stats.loss_trades }}</div>
                    <div class="stat-label">Losses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val {{ 'green' if stats.total_pnl >= 0 else 'red' }}">
                        {{ "%.2f"|format(stats.total_pnl) }}
                    </div>
                    <div class="stat-label">Net PnL (USDT)</div>
                </div>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>PnL</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in stats.details %}
                    <tr>
                        <td>{{ trade.time }}</td>
                        <td><b>{{ trade.symbol }}</b></td>
                        <td><span class="badge {{ 'badge-buy' if trade.side == 'Buy' else 'badge-sell' }}">{{ trade.side }}</span></td>
                        <td class="{{ 'green' if trade.pnl >= 0 else 'red' }}">
                            {{ "+" if trade.pnl > 0 else "" }}{{ "%.2f"|format(trade.pnl) }}
                        </td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, stats=stats, balance=balance, days=days, date=datetime.now().strftime('%Y-%m-%d %H:%M'))

# Текстова версія для сумісності (якщо колись захочете повернути пошту)
def generate_report(days):
    stats, error = bot.get_pnl_stats(days=days)
    if error: return f"Error: {error}"
    report = f"📊 TRADING REPORT ({days} days)\nNet PnL: {stats['total_pnl']:.2f} USDT\n\n"
    for t in stats['details'][:20]:
        report += f"{t['time']} | {t['symbol']} | {t['pnl']:.2f}\n"
    return report

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json(silent=True)
        if not data: data = parse_tradingview_data(request.get_data(as_text=True))
        if not data: return jsonify({"error": "No data"}), 400
        threading.Thread(target=bot.place_order, args=(data,)).start()
        return jsonify({"status": "processing"})
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

# Допоміжні функції для вебхука
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
