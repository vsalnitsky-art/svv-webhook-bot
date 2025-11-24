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

# --- НАЛАШТУВАННЯ ---
PORT = 10000
MONITOR_INTERVAL = 5 

# --- EMAIL (Опціонально) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "vsalnitsky@gmail.com"
EMAIL_PASSWORD = "hovx cuvd cypv tmtx" 
EMAIL_RECEIVER = "vsalnitsky@gmail.com"

def keep_alive():
    """Функція для підтримки сервера активним"""
    time.sleep(10)
    while True:
        try:
            requests.get('https://svv-webhook-bot.onrender.com/health', timeout=5)
        except Exception as e:
            logger.warning(f"⚠️ Keep-alive ping failed: {e}")
        time.sleep(600)

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

keep_alive_thread = threading.Thread(target=keep_alive)
keep_alive_thread.daemon = True
keep_alive_thread.start()

class BybitTradingBot:
    def __init__(self):
        try:
            api_key, api_secret = get_api_credentials()
            self.session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
            logger.info("✅ Bybit API session initialized.")
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
            norm = self.normalize_symbol(symbol)
            resp = self.session.get_tickers(category="linear", symbol=norm)
            if resp.get('retCode') == 0 and resp['result']['list']:
                return float(resp['result']['list'][0]['lastPrice'])
            return None
        except: return None

    def set_leverage(self, symbol, leverage):
        try:
            norm = self.normalize_symbol(symbol)
            self.session.set_leverage(category="linear", symbol=norm, buyLeverage=str(leverage), sellLeverage=str(leverage))
            return True
        except Exception as e:
            if "110043" in str(e): return True
            return False

    def get_instrument_info(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            resp = self.session.get_instruments_info(category="linear", symbol=norm)
            if resp.get('retCode') == 0:
                return resp['result']['list'][0]['lotSizeFilter'], resp['result']['list'][0]['priceFilter']
            return None, None
        except: return None, None

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
            norm = self.normalize_symbol(symbol)
            resp = self.session.get_positions(category="linear", symbol=norm)
            if resp['retCode'] == 0 and resp['result']['list']:
                return float(resp['result']['list'][0]['size'])
            return 0.0
        except: return 0.0

    # --- ОСНОВНА ЛОГІКА ЗБОРУ СТАТИСТИКИ ---
    def get_pnl_stats(self, days=7):
        logger.info(f"📊 Fetching PnL stats for last {days} days...")
        try:
            now = datetime.now()
            all_trades = []
            
            # Завантаження частинами по 7 днів
            for i in range(0, days, 7):
                chunk_days = min(7, days - i)
                end_dt = now - timedelta(days=i)
                start_dt = end_dt - timedelta(days=chunk_days)
                
                ts_end = int(end_dt.timestamp() * 1000)
                ts_start = int(start_dt.timestamp() * 1000)
                
                resp = self.session.get_closed_pnl(category="linear", startTime=ts_start, endTime=ts_end, limit=100)
                if resp['retCode'] == 0:
                    all_trades.extend(resp['result']['list'])
                time.sleep(0.1)

            # Підготовка даних
            stats = {
                "total_trades": len(all_trades),
                "total_pnl": 0.0,
                "total_volume": 0.0,
                "win_trades": 0,
                "loss_trades": 0,
                "details": [],
                "chart_labels": [],
                "chart_data": [],
                "coin_performance": {}
            }

            # Сортуємо від старих до нових для графіка
            all_trades.sort(key=lambda x: int(x['updatedTime']))

            cumulative_pnl = 0.0
            
            for trade in all_trades:
                pnl = float(trade['closedPnl'])
                price = float(trade['avgExitPrice'])
                qty = float(trade['qty'])
                volume = price * qty
                
                stats["total_pnl"] += pnl
                stats["total_volume"] += volume
                cumulative_pnl += pnl
                
                if pnl > 0: stats["win_trades"] += 1
                else: stats["loss_trades"] += 1
                
                # Графік P&L (накопичувальний)
                fill_time = datetime.fromtimestamp(int(trade['updatedTime']) / 1000)
                stats["chart_labels"].append(fill_time.strftime('%m-%d'))
                stats["chart_data"].append(round(cumulative_pnl, 2))
                
                # P&L по монетам
                symbol = trade['symbol']
                if symbol not in stats["coin_performance"]:
                    stats["coin_performance"][symbol] = 0.0
                stats["coin_performance"][symbol] += pnl

                # Деталі для таблиці
                stats["details"].append({
                    "time": fill_time.strftime('%Y-%m-%d %H:%M'),
                    "symbol": symbol,
                    "side": trade['side'], # Buy/Sell
                    "qty": qty,
                    "entry_price": float(trade['avgEntryPrice']),
                    "exit_price": float(trade['avgExitPrice']),
                    "pnl": pnl,
                    "is_win": pnl > 0
                })

            # Сортуємо деталі для таблиці: нові зверху
            stats["details"].sort(key=lambda x: x['time'], reverse=True)
            
            # Сортуємо монети по PnL для топ-чарту
            sorted_coins = sorted(stats["coin_performance"].items(), key=lambda x: x[1], reverse=True)
            stats["top_coins_labels"] = [x[0] for x in sorted_coins[:5]] # Топ 5
            stats["top_coins_values"] = [round(x[1], 2) for x in sorted_coins[:5]]

            return stats, None

        except Exception as e:
            logger.error(f"❌ Stat Error: {e}")
            return None, str(e)

    def place_order(self, data):
        # (Логіка ордерів без змін)
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            if not action or not symbol: return {"status": "error"}
            norm_symbol = self.normalize_symbol(symbol)
            if self.get_position_size(norm_symbol) > 0: return {"status": "ignored"}

            riskPercent = float(data.get('riskPercent', 5.0))
            leverage = int(data.get('leverage', 20))
            tpPercent = float(data.get('takeProfitPercent', 0.0))
            slPercent = float(data.get('stopLossPercent', 0.0))

            cur_price = self.get_current_price(norm_symbol)
            if not cur_price: return {"status": "error"}
            
            lot_filter, price_filter = self.get_instrument_info(norm_symbol)
            if not lot_filter: return {"status": "error"}

            qty_step = float(lot_filter['qtyStep'])
            min_qty = float(lot_filter['minOrderQty'])
            tick_size = float(price_filter['tickSize'])
            balance = self.get_available_balance()
            
            if not balance: return {"status": "error"}

            margin = (balance * (riskPercent / 100)) * 0.98
            raw_qty = (margin * leverage) / cur_price
            final_qty = self.round_qty(raw_qty, qty_step)
            
            if final_qty < min_qty: return {"status": "error"}

            self.set_leverage(norm_symbol, leverage)
            self.session.place_order(category="linear", symbol=norm_symbol, side=action, orderType="Market", qty=str(final_qty), timeInForce="GTC")
            
            sl_price = 0
            if slPercent > 0:
                sl_price = cur_price * (1 - slPercent/100) if action == "Buy" else cur_price * (1 + slPercent/100)
                self.session.set_trading_stop(category="linear", symbol=norm_symbol, stopLoss=str(self.round_price(sl_price, tick_size)), positionIdx=0)

            if tpPercent > 0:
                qty_tp1 = self.round_qty(final_qty * 0.5, qty_step)
                qty_tp2 = self.round_qty(final_qty * 0.35, qty_step)
                qty_tp3 = self.round_qty(final_qty - qty_tp1 - qty_tp2, qty_step)
                
                tp1_dist = tpPercent * 0.5
                tp2_dist = tpPercent * 1.0
                tp3_dist = tpPercent * 1.5
                
                tp_side = "Sell" if action == "Buy" else "Buy"
                
                tps = [
                    (qty_tp1, cur_price * (1 + tp1_dist/100) if action == "Buy" else cur_price * (1 - tp1_dist/100)),
                    (qty_tp2, cur_price * (1 + tp2_dist/100) if action == "Buy" else cur_price * (1 - tp2_dist/100)),
                    (qty_tp3, cur_price * (1 + tp3_dist/100) if action == "Buy" else cur_price * (1 - tp3_dist/100))
                ]

                for q, p in tps:
                    if q >= min_qty:
                        self.session.place_order(category="linear", symbol=norm_symbol, side=tp_side, orderType="Limit", qty=str(q), price=str(self.round_price(p, tick_size)), reduceOnly=True)

            return {"status": "success"}
        except Exception as e:
            logger.error(f"Order Error: {e}")
            return {"status": "error"}

bot = BybitTradingBot()

class BreakevenMonitor:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.running = True
    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try:
                self.check_positions()
            except: pass
            time.sleep(MONITOR_INTERVAL)
    def check_positions(self):
        positions = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if positions['retCode'] != 0: return
        for pos in positions['result']['list']:
            if float(pos['size']) == 0: continue
            entry = float(pos['avgPrice'])
            sl = float(pos.get('stopLoss', 0))
            side = pos['side']
            if (side == "Buy" and sl >= entry) or (side == "Sell" and sl > 0 and sl <= entry): continue
            orders = self.bot.session.get_open_orders(category="linear", symbol=pos['symbol'])
            tps = sum(1 for o in orders.get('result', {}).get('list', []) if o['reduceOnly'])
            if tps > 0 and tps < 3:
                try: self.bot.session.set_trading_stop(category="linear", symbol=pos['symbol'], stopLoss=str(entry), positionIdx=0)
                except: pass

monitor = BreakevenMonitor(bot)
monitor.start()

# --- HTML REPORT BYBIT STYLE ---
@app.route('/report', methods=['GET'])
def report_page():
    days = request.args.get('days', default=7, type=int)
    stats, error = bot.get_pnl_stats(days=days)
    balance = bot.get_available_balance()
    
    if error: return f"<h1>Error</h1><p>{error}</p>"

    win_rate = 0
    if stats['total_trades'] > 0:
        win_rate = round((stats['win_trades'] / stats['total_trades']) * 100, 1)

    # Шаблон в стиле Bybit
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>P&L Analysis</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root {
                --bg-color: #f7f8fa;
                --card-bg: #ffffff;
                --text-primary: #121214;
                --text-secondary: #858e9c;
                --green: #20b26c;
                --red: #ef454a;
                --border: #eff2f5;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--bg-color);
                color: var(--text-primary);
                margin: 0;
                padding: 20px;
            }
            .container { max_width: 1200px; margin: 0 auto; }
            
            /* Cards */
            .card {
                background: var(--card-bg);
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            
            /* Header */
            .header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            h1 { font-size: 20px; font-weight: 600; margin: 0; }
            .balance { font-size: 24px; font-weight: 700; }
            .sub-text { font-size: 12px; color: var(--text-secondary); }

            /* KPI Grid */
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .kpi-val { font-size: 28px; font-weight: 700; margin-top: 5px; }
            .kpi-label { font-size: 13px; color: var(--text-secondary); }
            
            /* Colors */
            .text-green { color: var(--green); }
            .text-red { color: var(--red); }
            
            /* Charts Row */
            .charts-row {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            @media (max-width: 768px) { .charts-row { grid-template-columns: 1fr; } }
            .chart-container { position: relative; height: 300px; width: 100%; }

            /* Table */
            .table-container { overflow-x: auto; }
            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }
            th {
                text-align: left;
                color: var(--text-secondary);
                font-weight: 500;
                padding: 12px 16px;
                border-bottom: 1px solid var(--border);
            }
            td {
                padding: 14px 16px;
                border-bottom: 1px solid var(--border);
                vertical-align: middle;
            }
            .symbol-cell { display: flex; align-items: center; gap: 10px; font-weight: 600; }
            .coin-icon {
                width: 24px; height: 24px;
                border-radius: 50%;
                background: #e0e0e0;
                display: flex; align-items: center; justify-content: center;
                font-size: 10px; color: #555;
            }
            
            /* Badges */
            .badge {
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 11px;
                font-weight: 500;
            }
            .badge-win { background: rgba(32, 178, 108, 0.1); color: var(--green); }
            .badge-loss { background: rgba(239, 69, 74, 0.1); color: var(--red); }
            .type-long { color: var(--green); }
            .type-short { color: var(--red); }
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Summary Header -->
            <div class="card">
                <div class="header-row">
                    <div>
                        <h1>P&L Analysis <span style="font-size:14px; color:#999; font-weight:400; margin-left:10px;">Last {{ days }} days</span></h1>
                    </div>
                    <div style="text-align: right;">
                        <div class="sub-text">Wallet Balance</div>
                        <div class="balance">${{ "%.2f"|format(balance) }}</div>
                    </div>
                </div>

                <div class="kpi-grid">
                    <div>
                        <div class="kpi-label">Total Realized P&L</div>
                        <div class="kpi-val {{ 'text-green' if stats.total_pnl >= 0 else 'text-red' }}">
                            {{ "+" if stats.total_pnl > 0 }}{{ "%.2f"|format(stats.total_pnl) }} <span style="font-size:14px; color:#333;">USD</span>
                        </div>
                    </div>
                    <div>
                        <div class="kpi-label">Trading Volume</div>
                        <div class="kpi-val text-green">
                            {{ "%.2f"|format(stats.total_volume) }} <span style="font-size:14px; color:#333;">USD</span>
                        </div>
                    </div>
                    <div>
                        <div class="kpi-label">Total Trades</div>
                        <div class="kpi-val">{{ stats.total_trades }}</div>
                    </div>
                    <div>
                        <div class="kpi-label">Win Rate</div>
                        <div class="kpi-val">{{ win_rate }}%</div>
                    </div>
                </div>
            </div>

            <!-- Charts -->
            <div class="charts-row">
                <div class="card">
                    <div class="header-row">
                        <h3>Cumulative P&L ($)</h3>
                    </div>
                    <div class="chart-container">
                        <canvas id="pnlChart"></canvas>
                    </div>
                </div>
                <div class="card">
                    <div class="header-row">
                        <h3>Top Coins</h3>
                    </div>
                    <div class="chart-container">
                        <canvas id="coinChart"></canvas>
                    </div>
                </div>
            </div>

            <!-- Table -->
            <div class="card">
                <div class="header-row">
                    <h3>Closed Orders Details</h3>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Qty</th>
                                <th>Entry Price</th>
                                <th>Exit Price</th>
                                <th>Realized P&L</th>
                                <th>Result</th>
                                <th>Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for trade in stats.details %}
                            <tr>
                                <td>
                                    <div class="symbol-cell">
                                        <div class="coin-icon">{{ trade.symbol[0] }}</div>
                                        {{ trade.symbol }}
                                    </div>
                                </td>
                                <td class="{{ 'type-long' if trade.side == 'Buy' else 'type-short' }}">
                                    {{ "Long" if trade.side == "Buy" else "Short" }}
                                </td>
                                <td>{{ trade.qty }}</td>
                                <td>{{ trade.entry_price }}</td>
                                <td>{{ trade.exit_price }}</td>
                                <td class="{{ 'text-green' if trade.pnl > 0 else 'text-red' }}">
                                    {{ "+" if trade.pnl > 0 }}{{ "%.4f"|format(trade.pnl) }}
                                </td>
                                <td>
                                    <span class="badge {{ 'badge-win' if trade.is_win else 'badge-loss' }}">
                                        {{ "Win" if trade.is_win else "Loss" }}
                                    </span>
                                </td>
                                <td style="color: #888;">{{ trade.time }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // Config for Line Chart
            const ctxPnl = document.getElementById('pnlChart').getContext('2d');
            const gradient = ctxPnl.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, 'rgba(32, 178, 108, 0.2)');
            gradient.addColorStop(1, 'rgba(32, 178, 108, 0)');

            new Chart(ctxPnl, {
                type: 'line',
                data: {
                    labels: {{ stats.chart_labels | safe }},
                    datasets: [{
                        label: 'Cumulative P&L',
                        data: {{ stats.chart_data | safe }},
                        borderColor: '#20b26c',
                        backgroundColor: gradient,
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false } },
                        y: { grid: { color: '#eff2f5' } }
                    }
                }
            });

            // Config for Bar Chart
            new Chart(document.getElementById('coinChart'), {
                type: 'bar',
                data: {
                    labels: {{ stats.top_coins_labels | safe }},
                    datasets: [{
                        label: 'P&L by Coin',
                        data: {{ stats.top_coins_values | safe }},
                        backgroundColor: (ctx) => {
                            const val = ctx.raw;
                            return val >= 0 ? '#20b26c' : '#ef454a';
                        },
                        borderRadius: 4
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } }
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, stats=stats, balance=balance, days=days, win_rate=win_rate)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = request.get_json(silent=True)
        if not data: 
            match = re.search(r'\{[^}]+\}', str(request.get_data(as_text=True)))
            if match: data = json.loads(match.group())
        if not data: return jsonify({"error": "No data"}), 400
        threading.Thread(target=bot.place_order, args=(data,)).start()
        return jsonify({"status": "processing"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home(): return "Bot Active"
@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
