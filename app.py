from flask import Flask, request, jsonify, render_template_string
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests
import json
import re
import copy
from datetime import datetime, timedelta
from config import get_api_credentials

# --- НАЛАШТУВАННЯ ---
PORT = 10000
MONITOR_INTERVAL = 5 
SCANNER_INTERVAL = 60  # Сканування ринку раз на 60 секунд
VOLUME_SPIKE_THRESHOLD = 3.0 # Коефіцієнт аномалії (3.0 = об'єм у 3 рази вищий за середній)
MIN_24H_VOLUME = 1000000 # Ігнорувати монети з добовим об'ємом менше 1 млн$

# --- EMAIL (Опціонально) ---
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "vsalnitsky@gmail.com"
EMAIL_PASSWORD = "YOUR_APP_PASSWORD_HERE" # Вставте пароль сюди
EMAIL_RECEIVER = "vsalnitsky@gmail.com"

# --- FLASK APP ---
app = Flask(__name__)

# Налаштування логування для відображення в Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- KEEP ALIVE ---
def keep_alive():
    """Функція для підтримки сервера активним на Render"""
    time.sleep(10)
    while True:
        try:
            # Замініть URL на вашу реальну адресу на Render
            requests.get('https://svv-webhook-bot.onrender.com/health', timeout=5)
        except Exception as e:
            logger.warning(f"⚠️ Keep-alive ping failed: {e}")
        time.sleep(600)

keep_alive_thread = threading.Thread(target=keep_alive)
keep_alive_thread.daemon = True
keep_alive_thread.start()

# --- BOT CLASS ---
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
        # Видаляє .P, який іноді додає TradingView
        return symbol.replace('.P', '')

    def get_all_tickers(self):
        """Отримує дані по всім монетам відразу"""
        try:
            resp = self.session.get_tickers(category="linear")
            if resp['retCode'] == 0:
                return resp['result']['list']
            return []
        except Exception as e:
            logger.error(f"Ticker fetch error: {e}")
            return []

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
            if "110043" in str(e): return True # Leverage already set
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

    # --- P&L STATISTICS ---
    def get_pnl_stats(self, days=7):
        logger.info(f"📊 Fetching PnL stats for last {days} days...")
        try:
            now = datetime.now()
            all_trades = []
            
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

            stats = {
                "total_trades": len(all_trades),
                "total_pnl": 0.0,
                "total_volume": 0.0,
                "win_trades": 0, "loss_trades": 0,
                "long_trades": 0, "short_trades": 0,
                "details": [], "chart_labels": [], "chart_data": [],
                "coin_performance": {}
            }

            all_trades.sort(key=lambda x: int(x['updatedTime']))
            cumulative_pnl = 0.0
            
            for trade in all_trades:
                pnl = float(trade['closedPnl'])
                price = float(trade['avgExitPrice'])
                qty = float(trade['qty'])
                volume = price * qty
                side = trade['side']

                stats["total_pnl"] += pnl
                stats["total_volume"] += volume
                cumulative_pnl += pnl
                
                if pnl > 0: stats["win_trades"] += 1
                else: stats["loss_trades"] += 1

                if side == "Buy": stats["long_trades"] += 1
                else: stats["short_trades"] += 1
                
                fill_time = datetime.fromtimestamp(int(trade['updatedTime']) / 1000)
                stats["chart_labels"].append(fill_time.strftime('%m-%d'))
                stats["chart_data"].append(round(cumulative_pnl, 2))
                
                symbol = trade['symbol']
                if symbol not in stats["coin_performance"]: stats["coin_performance"][symbol] = 0.0
                stats["coin_performance"][symbol] += pnl

                stats["details"].append({
                    "time": fill_time.strftime('%Y-%m-%d %H:%M'),
                    "symbol": symbol, "side": side, "qty": qty,
                    "entry_price": float(trade['avgEntryPrice']),
                    "exit_price": float(trade['avgExitPrice']),
                    "pnl": pnl, "is_win": pnl > 0
                })

            stats["details"].sort(key=lambda x: x['time'], reverse=True)
            sorted_coins = sorted(stats["coin_performance"].items(), key=lambda x: x[1], reverse=True)
            stats["top_coins_labels"] = [x[0] for x in sorted_coins[:5]]
            stats["top_coins_values"] = [round(x[1], 2) for x in sorted_coins[:5]]

            return stats, None
        except Exception as e:
            logger.error(f"❌ Stat Error: {e}")
            return None, str(e)

    # --- ORDER PLACEMENT ---
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            
            logger.info(f"🤖 Placing Order: {symbol} {action}")
            
            if not action or not symbol: 
                logger.error("Missing action or symbol")
                return {"status": "error"}
                
            norm_symbol = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm_symbol) > 0: 
                logger.warning(f"Position already exists for {norm_symbol}. Ignored.")
                return {"status": "ignored"}

            riskPercent = float(data.get('riskPercent', 5.0))
            leverage = int(data.get('leverage', 20))
            tpPercent = float(data.get('takeProfitPercent', 0.0))
            slPercent = float(data.get('stopLossPercent', 0.0))

            cur_price = self.get_current_price(norm_symbol)
            if not cur_price: 
                logger.error(f"Could not get price for {norm_symbol}")
                return {"status": "error"}
            
            lot_filter, price_filter = self.get_instrument_info(norm_symbol)
            if not lot_filter: 
                logger.error(f"Could not get info for {norm_symbol}")
                return {"status": "error"}

            qty_step = float(lot_filter['qtyStep'])
            min_qty = float(lot_filter['minOrderQty'])
            tick_size = float(price_filter['tickSize'])
            balance = self.get_available_balance()
            
            if not balance: 
                logger.error("Balance is 0 or unavailable")
                return {"status": "error"}

            margin = (balance * (riskPercent / 100)) * 0.98
            raw_qty = (margin * leverage) / cur_price
            final_qty = self.round_qty(raw_qty, qty_step)
            
            if final_qty < min_qty: 
                logger.warning(f"Calculated Qty {final_qty} < Min Qty {min_qty}")
                return {"status": "error"}

            # 1. Set Leverage
            self.set_leverage(norm_symbol, leverage)
            
            # 2. Market Order
            self.session.place_order(category="linear", symbol=norm_symbol, side=action, orderType="Market", qty=str(final_qty), timeInForce="GTC")
            logger.info(f"✅ Market Order Placed: {final_qty} {norm_symbol}")
            
            # 3. Stop Loss
            if slPercent > 0:
                sl_price = cur_price * (1 - slPercent/100) if action == "Buy" else cur_price * (1 + slPercent/100)
                self.session.set_trading_stop(category="linear", symbol=norm_symbol, stopLoss=str(self.round_price(sl_price, tick_size)), positionIdx=0)
                logger.info("✅ SL Set")

            # 4. Take Profit (Ladder)
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
                logger.info("✅ TP Ladder Set")

            return {"status": "success"}
        except Exception as e:
            logger.error(f"🔥 Order Execution Error: {e}")
            return {"status": "error"}

bot = BybitTradingBot()

# --- MARKET SCANNER ---
class MarketScanner:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.running = True
        self.previous_snapshot = {}
        self.detected_pumps = [] # Зберігає знайдені пампи
        self.last_scan_time = None

    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        logger.info("🚀 Market Scanner Started")
        while self.running:
            try:
                self.scan_market()
            except Exception as e:
                logger.error(f"Scanner error: {e}")
            time.sleep(SCANNER_INTERVAL)

    def scan_market(self):
        tickers = self.bot.get_all_tickers()
        current_snapshot = {}
        
        # 1. Збираємо поточні дані
        for t in tickers:
            symbol = t['symbol']
            if "USDT" not in symbol: continue
            
            try:
                price = float(t['lastPrice'])
                turnover = float(t['turnover24h']) 
                
                if turnover < MIN_24H_VOLUME: continue 

                current_snapshot[symbol] = {
                    'price': price,
                    'turnover': turnover,
                    'timestamp': time.time(),
                    'change24h': float(t['price24hPcnt']) * 100
                }
            except: continue

        # 2. Якщо є попередні дані, порівнюємо
        if self.previous_snapshot:
            self.analyze_changes(current_snapshot)

        self.previous_snapshot = current_snapshot
        self.last_scan_time = datetime.now()

    def analyze_changes(self, current):
        detected = []
        now = datetime.now()
        
        for symbol, data in current.items():
            if symbol not in self.previous_snapshot: continue
            
            prev = self.previous_snapshot[symbol]
            
            # Різниця в часі в хвилинах
            time_delta = (data['timestamp'] - prev['timestamp']) / 60
            if time_delta == 0: continue

            vol_diff = data['turnover'] - prev['turnover']
            if vol_diff <= 0: continue 

            avg_vol_per_min = data['turnover'] / 1440
            expected_vol = avg_vol_per_min * time_delta
            
            spike_factor = vol_diff / expected_vol if expected_vol > 0 else 0
            
            price_change_interval = ((data['price'] - prev['price']) / prev['price']) * 100

            if spike_factor > VOLUME_SPIKE_THRESHOLD and price_change_interval > 0.5:
                detected.append({
                    "symbol": symbol,
                    "price": data['price'],
                    "spike_factor": round(spike_factor, 2),
                    "vol_inflow": round(vol_diff, 0),
                    "price_change_interval": round(price_change_interval, 2),
                    "time": now.strftime('%H:%M:%S'),
                    "timestamp_dt": now
                })

        if detected:
            logger.info(f"🚨 DETECTED {len(detected)} PUMPS!")
            self.detected_pumps = detected + self.detected_pumps

        # Фільтр: Залишаємо тільки пампи за останню 1 годину (60 хвилин)
        one_hour_ago = now - timedelta(hours=1)
        self.detected_pumps = [p for p in self.detected_pumps if p.get('timestamp_dt', now) > one_hour_ago]

scanner = MarketScanner(bot)
scanner.start()

# --- BREAKEVEN MONITOR ---
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
                try: 
                    self.bot.session.set_trading_stop(category="linear", symbol=pos['symbol'], stopLoss=str(entry), positionIdx=0)
                    logger.info(f"🛡️ Moved SL to Breakeven for {pos['symbol']}")
                except: pass

monitor = BreakevenMonitor(bot)
monitor.start()

# --- WEB ROUTES ---

@app.route('/scanner', methods=['GET'])
def scanner_page():
    """Сторінка зі звітом сканера та сортуванням"""
    last_update = scanner.last_scan_time.strftime('%H:%M:%S') if scanner.last_scan_time else "Запуск..."
    
    html_template = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Crypto Volume Scanner</title>
        <meta http-equiv="refresh" content="30">
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --green: #22c55e; --blue: #3b82f6; }
            body { font-family: sans-serif; background: var(--bg); color: var(--text); padding: 20px; margin: 0; }
            .container { max-width: 1000px; margin: 0 auto; }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .card { background: var(--card); padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h1 { margin: 0; font-size: 24px; }
            .badge { background: var(--blue); padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; }
            th { text-align: left; color: #94a3b8; padding: 10px; border-bottom: 1px solid #334155; cursor: pointer; user-select: none; }
            th:hover { color: #e2e8f0; }
            td { padding: 12px 10px; border-bottom: 1px solid #334155; }
            .pump-factor { color: var(--green); font-weight: bold; }
            .symbol { font-weight: bold; font-size: 1.1em; }
            .empty-msg { text-align: center; color: #64748b; padding: 40px; }
            .inflow { color: #818cf8; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📡 Market Scanner</h1>
                <span class="badge">Last Scan: {{ last_update }}</span>
            </div>
            
            <div class="card">
                <h3>🔥 Виявлені аномалії об'єму (Історія за 1 годину)</h3>
                <p style="color:#94a3b8; font-size: 14px;">Показує монети, де об'єм за хвилину в 3+ рази перевищує середній. Натисніть на заголовок для сортування.</p>
                
                {% if pumps %}
                <table id="pumpsTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Час ⇅</th>
                            <th onclick="sortTable(1)">Монета ⇅</th>
                            <th onclick="sortTable(2)">Ціна ⇅</th>
                            <th onclick="sortTable(3)">Зміна ціни (1хв) ⇅</th>
                            <th onclick="sortTable(4)">Аномалія (x разів) ⇅</th>
                            <th onclick="sortTable(5)">Вливання ($) ⇅</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for p in pumps %}
                        <tr>
                            <td>{{ p.time }}</td>
                            <td class="symbol">{{ p.symbol }}</td>
                            <td>{{ p.price }}</td>
                            <td style="color: {{ '#22c55e' if p.price_change_interval > 0 else '#ef4444' }}">
                                {{ "+" if p.price_change_interval > 0 }}{{ p.price_change_interval }}%
                            </td>
                            <td class="pump-factor">x{{ p.spike_factor }}</td>
                            <td class="inflow">${{ "{:,.0f}".format(p.vol_inflow) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                    <div class="empty-msg">Поки що тихо. Чекаємо на активність китів... 🐋</div>
                {% endif %}
            </div>
            
            <div style="text-align:center;">
                <a href="/report" style="color: #64748b; text-decoration: none;">Перейти до звіту P&L</a>
            </div>
        </div>

        <script>
        function sortTable(n) {
          var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
          table = document.getElementById("pumpsTable");
          switching = true;
          dir = "asc"; 
          while (switching) {
            switching = false;
            rows = table.rows;
            for (i = 1; i < (rows.length - 1); i++) {
              shouldSwitch = false;
              x = rows[i].getElementsByTagName("TD")[n];
              y = rows[i + 1].getElementsByTagName("TD")[n];
              let xVal = x.innerText.toLowerCase().replace(/[$,x%+,]/g, "");
              let yVal = y.innerText.toLowerCase().replace(/[$,x%+,]/g, "");
              if (!isNaN(parseFloat(xVal)) && isFinite(xVal)) { xVal = parseFloat(xVal); yVal = parseFloat(yVal); }
              if (dir == "asc") { if (xVal > yVal) { shouldSwitch = true; break; } } 
              else if (dir == "desc") { if (xVal < yVal) { shouldSwitch = true; break; } }
            }
            if (shouldSwitch) {
              rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
              switching = true;
              switchcount ++;      
            } else {
              if (switchcount == 0 && dir == "asc") { dir = "desc"; switching = true; }
            }
          }
        }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, pumps=scanner.detected_pumps, last_update=last_update)

@app.route('/report', methods=['GET'])
def report_page():
    days = request.args.get('days', default=7, type=int)
    stats, error = bot.get_pnl_stats(days=days)
    balance = bot.get_available_balance()
    
    if error: return f"<h1>Помилка</h1><p>{error}</p>"

    win_rate = 0
    if stats['total_trades'] > 0:
        win_rate = round((stats['win_trades'] / stats['total_trades']) * 100, 1)

    # Використовуємо той самий шаблон HTML, що був раніше (скорочено для економії місця, функціонал той самий)
    html_template = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Аналіз P&L</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root { --bg-color: #f7f8fa; --card-bg: #ffffff; --text-primary: #121214; --text-secondary: #858e9c; --green: #20b26c; --red: #ef454a; --border: #eff2f5; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-primary); margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: var(--card-bg); border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            h1 { font-size: 20px; font-weight: 600; margin: 0; }
            .balance { font-size: 24px; font-weight: 700; }
            .sub-text { font-size: 12px; color: var(--text-secondary); }
            .top-stats-grid { display: flex; gap: 40px; margin-bottom: 10px; flex-wrap: wrap; }
            .kpi-block { min-width: 150px; }
            .kpi-val { font-size: 24px; font-weight: 700; margin-top: 5px; }
            .kpi-label { font-size: 13px; color: var(--text-secondary); text-decoration: underline dotted; cursor: help; }
            .detail-stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            @media (max-width: 768px) { .detail-stats-grid { grid-template-columns: 1fr; } }
            .stat-box { background: white; border-radius: 8px; padding: 20px; display: flex; flex-direction: column; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .stat-box-header { font-size: 13px; color: var(--text-secondary); margin-bottom: 15px; }
            .stat-box-content { display: flex; align-items: center; gap: 20px; }
            .donut-ring, .gauge-ring { width: 50px; height: 50px; flex-shrink: 0; }
            .stat-big-num { font-size: 28px; font-weight: 700; line-height: 1.2; }
            .stat-sub-row { font-size: 13px; margin-top: 4px; display: flex; gap: 5px; }
            .text-green { color: var(--green); } .text-red { color: var(--red); } .text-gray { color: var(--text-secondary); }
            .charts-row { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; margin-bottom: 20px; }
            @media (max-width: 768px) { .charts-row { grid-template-columns: 1fr; } }
            .chart-container { position: relative; height: 300px; width: 100%; }
            .table-container { overflow-x: auto; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; }
            th { text-align: left; color: var(--text-secondary); font-weight: 500; padding: 12px 16px; border-bottom: 1px solid var(--border); }
            td { padding: 14px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }
            .symbol-cell { display: flex; align-items: center; gap: 10px; font-weight: 600; }
            .coin-icon { width: 24px; height: 24px; border-radius: 50%; background: #e0e0e0; display: flex; align-items: center; justify-content: center; font-size: 10px; color: #555; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
            .badge-win { background: rgba(32, 178, 108, 0.1); color: var(--green); }
            .badge-loss { background: rgba(239, 69, 74, 0.1); color: var(--red); }
            .type-long { color: var(--green); } .type-short { color: var(--red); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="header-row">
                    <h1>Аналіз P&L <span style="font-size:14px; color:#999; font-weight:400; margin-left:10px;">Останні {{ days }} днів</span></h1>
                    <div style="text-align: right;">
                        <div class="sub-text">Баланс Гаманця</div>
                        <div class="balance">${{ "%.2f"|format(balance) }}</div>
                    </div>
                </div>
                <div class="top-stats-grid">
                    <div class="kpi-block">
                        <div class="kpi-label">Загальний P&L</div>
                        <div class="kpi-val {{ 'text-green' if stats.total_pnl >= 0 else 'text-red' }}">
                            {{ "+" if stats.total_pnl > 0 }}{{ "%.2f"|format(stats.total_pnl) }} <span style="font-size:14px; color:#333;">USD</span>
                        </div>
                    </div>
                    <div class="kpi-block">
                        <div class="kpi-label">Торговий об'єм</div>
                        <div class="kpi-val text-green">
                            {{ "%.2f"|format(stats.total_volume) }} <span style="font-size:14px; color:#333;">USD</span>
                        </div>
                    </div>
                     <div class="kpi-block">
                        <div class="kpi-label">Інструменти</div>
                        <div class="kpi-val text-gray" style="font-size: 16px; margin-top: 10px;">
                            <a href="/scanner">📡 Перейти до Сканера</a>
                        </div>
                    </div>
                </div>
            </div>

            <div class="detail-stats-grid">
                <div class="stat-box">
                    <div class="stat-box-header">Загальна кількість закритих ордерів</div>
                    <div class="stat-box-content">
                        <div class="donut-ring">
                            <svg viewBox="0 0 36 36">
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#ef454a" stroke-width="4" />
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#20b26c" stroke-width="4" stroke-dasharray="{{ (stats.long_trades / stats.total_trades * 100) if stats.total_trades > 0 else 0 }}, 100" />
                            </svg>
                        </div>
                        <div>
                            <div class="stat-big-num">{{ stats.total_trades }}</div>
                            <div class="stat-sub-row">
                                <span class="text-green">{{ stats.long_trades }} Закрити лонг</span> / 
                                <span class="text-red">{{ stats.short_trades }} Закрити шорт</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="stat-box">
                    <div class="stat-box-header">Відсоток успішних угод</div>
                    <div class="stat-box-content">
                        <div class="gauge-ring">
                            <svg viewBox="0 0 36 36">
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#eff2f5" stroke-width="4" />
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#20b26c" stroke-width="4" stroke-dasharray="{{ win_rate }}, 100" />
                            </svg>
                        </div>
                        <div>
                            <div class="stat-big-num">{{ win_rate }} %</div>
                            <div class="stat-sub-row text-gray">
                                {{ stats.win_trades }} Успішні угоди / {{ stats.loss_trades }} Збитки
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="charts-row">
                <div class="card">
                    <div class="header-row"><h3>Кумулятивний P&L ($)</h3></div>
                    <div class="chart-container"><canvas id="pnlChart"></canvas></div>
                </div>
                <div class="card">
                    <div class="header-row"><h3>Топ монет</h3></div>
                    <div class="chart-container"><canvas id="coinChart"></canvas></div>
                </div>
            </div>

            <div class="card">
                <div class="header-row"><h3>Деталі закритих ордерів</h3></div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Тікер</th><th>Сторона</th><th>Кіл-сть</th><th>Ціна входу</th><th>Ціна виходу</th><th>Реаліз. P&L</th><th>Результат</th><th>Час</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for trade in stats.details %}
                            <tr>
                                <td><div class="symbol-cell"><div class="coin-icon">{{ trade.symbol[0] }}</div>{{ trade.symbol }}</div></td>
                                <td class="{{ 'type-long' if trade.side == 'Buy' else 'type-short' }}">{{ "Лонг" if trade.side == "Buy" else "Шорт" }}</td>
                                <td>{{ trade.qty }}</td>
                                <td>{{ trade.entry_price }}</td>
                                <td>{{ trade.exit_price }}</td>
                                <td class="{{ 'text-green' if trade.pnl > 0 else 'text-red' }}">{{ "+" if trade.pnl > 0 }}{{ "%.4f"|format(trade.pnl) }}</td>
                                <td><span class="badge {{ 'badge-win' if trade.is_win else 'badge-loss' }}">{{ "Прибуток" if trade.is_win else "Збиток" }}</span></td>
                                <td style="color: #888;">{{ trade.time }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <script>
            const ctxPnl = document.getElementById('pnlChart').getContext('2d');
            const gradient = ctxPnl.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, 'rgba(32, 178, 108, 0.2)');
            gradient.addColorStop(1, 'rgba(32, 178, 108, 0)');
            new Chart(ctxPnl, {
                type: 'line',
                data: {
                    labels: {{ stats.chart_labels | safe }},
                    datasets: [{
                        label: 'Кумулятивний P&L',
                        data: {{ stats.chart_data | safe }},
                        borderColor: '#20b26c', backgroundColor: gradient, borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { grid: { display: false } }, y: { grid: { color: '#eff2f5' } } } }
            });
            new Chart(document.getElementById('coinChart'), {
                type: 'bar',
                data: {
                    labels: {{ stats.top_coins_labels | safe }},
                    datasets: [{
                        label: 'P&L по монетам',
                        data: {{ stats.top_coins_values | safe }},
                        backgroundColor: (ctx) => { return ctx.raw >= 0 ? '#20b26c' : '#ef454a'; },
                        borderRadius: 4
                    }]
                },
                options: { indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } } }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, stats=stats, balance=balance, days=days, win_rate=win_rate)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обробка сигналів від TradingView (навіть із брудним JSON)"""
    try:
        # 1. Отримуємо сирі дані як текст
        raw_data = request.get_data(as_text=True)
        logger.info(f"📨 RAW WEBHOOK DATA: {raw_data}")

        data = None

        # 2. Спроба стандартного парсингу
        try:
            data = json.loads(raw_data)
        except:
            # 3. Спроба знайти JSON всередині тексту (Regex)
            # Шукає все між першою { та останньою }
            match = re.search(r'\{.*\}', raw_data, re.DOTALL)
            if match:
                try:
                    clean_json_str = match.group()
                    data = json.loads(clean_json_str)
                    logger.info("✅ JSON extracted from text successfully")
                except Exception as e:
                    logger.error(f"⚠️ JSON parse error after regex: {e}")

        if not data:
            logger.error("❌ Failed to find JSON in request")
            return jsonify({"error": "No valid JSON found"}), 400

        # 4. Запускаємо обробку ордера в окремому потоці
        logger.info(f"🚀 Processing Signal: {data.get('symbol')} {data.get('action')}")
        threading.Thread(target=bot.place_order, args=(data,)).start()
        
        return jsonify({"status": "processing"})

    except Exception as e:
        logger.error(f"🔥 Webhook Critical Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home(): return "Bot Active"

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
