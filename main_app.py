"""
Main Application - Ultimate Version
Включає:
1. Логіку торгівлі (Place Order, TP/SL) зі старого файлу.
2. Професійний інтерфейс (Bybit Style P&L, Dark Scanner).
3. Базу даних та ШІ.
"""

from flask import Flask, request, jsonify, render_template_string
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests
import json
import re
import os
import decimal
from datetime import datetime, timedelta
import ctypes # Для Windows Anti-Sleep

# Імпорт модулів проекту (Нова архітектура)
from bot_config import config
from models import db_manager
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from config import get_api_credentials

# === AI INIT ===
try:
    import ai_analyst
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI module not found")

# === WINDOWS ANTI-SLEEP (Тільки для Windows) ===
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

# === FLASK SETUP ===
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === TELEGRAM SENDER ===
def send_telegram_message(text):
    try:
        # 👇👇 ВСТАВТЕ ВАШІ ДАНІ НИЖЧЕ (або переконайтеся, що вони є в config.py) 👇👇
        tg_token = getattr(config, 'TG_BOT_TOKEN', "ВАШ_ТОКЕН")
        chat_id = getattr(config, 'TG_CHAT_ID', "ВАШ_CHAT_ID")
        
        if "ВАШ_" in tg_token: return 
        
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        logger.error(f"TG Error: {e}")

# === AI PROCESSOR ===
def process_signal_with_ai(data):
    symbol = data.get('symbol')
    action = data.get('action')
    ai_text = "AI Disabled"
    
    if AI_AVAILABLE:
        try: ai_text = ai_analyst.analyze_signal(symbol, action)
        except: ai_text = "Analysis Error"

    msg = f"🚀 <b>СИГНАЛ: {symbol}</b>\nДія: {action}\n\n🤖 <b>Gemini:</b> {ai_text}"
    send_telegram_message(msg)
    
    # Викликаємо логіку торгівлі
    bot.place_order(data)

# === SELF PING (KEEP ALIVE) ===
def keep_alive():
    time.sleep(10)
    # Використовуємо зовнішній URL для пінгу, щоб сервер Render не засинав
    external_url = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + '/health'
    while True:
        try: requests.get(external_url, timeout=10)
        except: pass
        time.sleep(300) # 5 хвилин

threading.Thread(target=keep_alive, daemon=True).start()

# === TRADING BOT CLASS (Об'єднаний) ===
class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)
        logger.info("✅ Bybit Connected")

    # --- Допоміжні методи зі старого файлу ---
    def normalize_symbol(self, symbol):
        return symbol.replace('.P', '')

    def get_available_balance(self, currency="USDT"):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            if b.get('retCode') != 0: return None
            for acc in b.get('result', {}).get('list', []):
                for c in acc.get('coin', []):
                    if c.get('coin') == currency: return float(c.get('walletBalance', 0))
            return None
        except: return None

    def get_all_tickers(self):
        try: return self.session.get_tickers(category="linear")['result']['list']
        except: return []

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
        step_decimals = abs(decimal.Decimal(str(step)).as_tuple().exponent)
        return round(qty // step * step, step_decimals)

    def round_price(self, price, tick_size):
        if tick_size <= 0: return price
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

    # --- ГОЛОВНА ЛОГІКА ТОРГІВЛІ (ЗІ СТАРОГО ФАЙЛУ) ---
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            
            logger.info(f"🤖 Placing Order: {symbol} {action}")
            
            if not action or not symbol: return {"status": "error"}
            norm_symbol = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm_symbol) > 0:
                logger.warning(f"Position exists for {norm_symbol}. Ignored.")
                return {"status": "ignored"}
            
            # Беремо налаштування з вебхука або дефолтні з конфігу
            riskPercent = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            leverage = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            tpPercent = float(data.get('takeProfitPercent', config.DEFAULT_TP_PERCENT))
            slPercent = float(data.get('stopLossPercent', config.DEFAULT_SL_PERCENT))
            
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
            
            # Логіка для малих позицій
            final_qty = self.round_qty(raw_qty, qty_step)
            if final_qty < min_qty:
                cost_of_min_qty = (min_qty * cur_price) / leverage
                if balance > cost_of_min_qty * 1.05:
                    final_qty = min_qty
                    logger.info(f"✅ Forced Min Qty: {final_qty}")
                else:
                    return {"status": "error_balance"}
            
            # 1. Плече
            self.set_leverage(norm_symbol, leverage)
            
            # 2. Маркет Ордер
            self.session.place_order(
                category="linear", symbol=norm_symbol, side=action, 
                orderType="Market", qty=str(final_qty), timeInForce="GTC"
            )
            logger.info(f"✅ Market Order Placed: {final_qty} {norm_symbol}")
            
            # 3. Stop Loss
            if slPercent > 0:
                sl_price = cur_price * (1 - slPercent/100) if action == "Buy" else cur_price * (1 + slPercent/100)
                self.session.set_trading_stop(
                    category="linear", symbol=norm_symbol, 
                    stopLoss=str(self.round_price(sl_price, tick_size)), positionIdx=0
                )
            
            # 4. Take Profit (Драбинка)
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
                        self.session.place_order(
                            category="linear", symbol=norm_symbol, side=tp_side, 
                            orderType="Limit", qty=str(q), 
                            price=str(self.round_price(p, tick_size)), reduceOnly=True
                        )
            return {"status": "success"}
        except Exception as e:
            logger.error(f"🔥 Order Error: {e}")
            return {"status": "error"}

    # --- P&L STATISTICS (ДЛЯ ЗВІТІВ) ---
    def get_pnl_stats(self, days=7):
        try:
            trades = stats_service.get_trades(days=days)
            if not trades:
                return {"total_pnl": 0.0, "total_trades": 0, "win_rate": 0, "chart_labels": [], "chart_data": [], "history": []}, None

            total_pnl = sum(t['pnl'] for t in trades)
            total_trades = len(trades)
            winners = sum(1 for t in trades if t['pnl'] > 0)
            losers = total_trades - winners
            win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
            total_vol = sum(t.get('qty', 0) * t.get('entry_price', 0) for t in trades)

            daily_pnl = {}
            for t in trades:
                if t['exit_time']:
                    date_str = t['exit_time'].split(' ')[0]
                    daily_pnl[date_str] = daily_pnl.get(date_str, 0) + t['pnl']
            
            sorted_dates = sorted(daily_pnl.keys())
            chart_labels = [d[5:] for d in sorted_dates]
            chart_data = []
            running_balance = 0
            for d in sorted_dates:
                running_balance += daily_pnl[d]
                chart_data.append(round(running_balance, 2))

            stats = {
                "total_pnl": round(total_pnl, 2), "total_trades": total_trades,
                "win_rate": round(win_rate, 1), "winners": winners, "losers": losers,
                "volume": round(total_vol, 2), "chart_labels": chart_labels,
                "chart_data": chart_data, "history": trades[:50]
            }
            return stats, None
        except Exception as e:
            return None, str(e)

# Ініціалізація
bot = BybitTradingBot()
scanner = EnhancedMarketScanner(bot, config.get_scanner_config())
scanner.start()

# --- BREAKEVEN MONITOR (ЗІ СТАРОГО ФАЙЛУ) ---
class BreakevenMonitor:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.running = True
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.check_positions()
            except: pass
            time.sleep(config.MONITOR_INTERVAL)
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

# === WEB ROUTES (UI) ===

@app.route('/scanner', methods=['GET'])
def scanner_page():
    """Професійний дашборд (UA)"""
    data = scanner.get_aggregated_data(hours=24)
    last_update = datetime.now().strftime('%H:%M:%S')
    
    html = """
    <!DOCTYPE html><html lang="uk" data-bs-theme="dark"><head><meta charset="UTF-8"><title>Whale Terminal Pro 🐋</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
    :root { --bg-dark: #0b0e11; --card-bg: #151a1f; --border: #2a3441; --green: #0ecb81; --red: #f6465d; }
    body { background-color: var(--bg-dark); font-family: 'Inter', sans-serif; color: #eaecef; font-size: 13px; }
    .card { background-color: var(--card-bg); border: 1px solid var(--border); margin-bottom: 15px; border-radius: 4px; }
    .card-header { background-color: #1b2129; border-bottom: 1px solid var(--border); font-weight: 600; font-size: 11px; color: #848e9c; padding: 8px 15px; }
    .table-dark { background: transparent; --bs-table-bg: transparent; }
    td, th { border-bottom: 1px solid var(--border) !important; vertical-align: middle; padding: 8px 10px; }
    .text-up { color: var(--green) !important; } .text-down { color: var(--red) !important; }
    .progress { height: 4px; background-color: #2b3139; margin-top: 4px; }
    .live-dot { height: 8px; width: 8px; background-color: var(--green); border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
    </style><meta http-equiv="refresh" content="60"></head><body>
    <nav class="navbar navbar-expand-lg navbar-dark" style="background: #181a20; border-bottom: 1px solid var(--border);">
    <div class="container-fluid"><a class="navbar-brand" href="#"><i class="fas fa-robot text-primary me-2"></i>Whale Terminal <span class="badge bg-primary" style="font-size: 9px;">PRO</span></a>
    <div class="d-flex align-items-center"><span class="text-muted me-3"><span class="live-dot me-1"></span> Live</span><span class="text-muted" style="font-size: 11px;">Оновлено: {{ last_update }}</span></div></div></nav>
    <div class="container-fluid mt-3"><div class="row">
    <div class="col-lg-6"><div class="card h-100"><div class="card-header text-up"><span><i class="fas fa-arrow-trend-up"></i> ТОП ПОКУПЦІВ (24Г)</span></div><div class="card-body p-0"><div class="table-responsive" style="max-height: 250px; overflow-y: auto;"><table class="table table-dark table-sm table-hover mb-0"><thead><tr><th>Актив</th><th class="text-end">Зміна</th><th class="text-end">Вхід</th></tr></thead><tbody>
    {% for coin in positive_coins[:15] %}<tr><td class="fw-bold">{{ coin.symbol }}</td><td class="text-end text-up">+{{ coin.avg_change }}%</td><td class="text-end">${{ "{:,.0f}".format(coin.inflow) }}<div class="progress"><div class="progress-bar bg-success" style="width: {{ coin.bar_pct }}%"></div></div></td></tr>{% endfor %}
    </tbody></table></div></div></div></div>
    <div class="col-lg-6"><div class="card h-100"><div class="card-header text-down"><span><i class="fas fa-arrow-trend-down"></i> ТОП ПРОДАВЦІВ (24Г)</span></div><div class="card-body p-0"><div class="table-responsive" style="max-height: 250px; overflow-y: auto;"><table class="table table-dark table-sm table-hover mb-0"><thead><tr><th>Актив</th><th class="text-end">Зміна</th><th class="text-end">Вихід</th></tr></thead><tbody>
    {% for coin in negative_coins[:15] %}<tr><td class="fw-bold">{{ coin.symbol }}</td><td class="text-end text-down">{{ coin.avg_change }}%</td><td class="text-end">${{ "{:,.0f}".format(coin.inflow) }}<div class="progress"><div class="progress-bar bg-danger" style="width: {{ coin.bar_pct }}%"></div></div></td></tr>{% endfor %}
    </tbody></table></div></div></div></div></div>
    <div class="row mt-3"><div class="col-12"><div class="card"><div class="card-header d-flex justify-content-between"><span><i class="fas fa-list-ul"></i> ЖУРНАЛ ОБ'ЄМІВ</span><a href="/report" class="btn btn-outline-secondary btn-sm" style="font-size: 10px; padding: 2px 8px;">ЗВІТ P&L</a></div><div class="card-body p-2"><table id="signalsTable" class="table table-dark table-hover w-100" style="font-size: 12px;"><thead class="text-muted"><tr><th>Час</th><th>Символ</th><th>Ціна</th><th>Зміна</th><th>Аномалія</th><th>Об'єм ($)</th><th>Дія</th></tr></thead><tbody>
    {% for p in pumps %}<tr style="{{ 'background: rgba(14, 203, 129, 0.08);' if p.vol_inflow > 1000000 else '' }}"><td class="text-muted">{{ p.time }}</td><td><span class="fw-bold text-primary">{{ p.symbol }}</span></td><td>{{ p.price }}</td><td class="{{ 'text-up' if p.price_change_interval > 0 else 'text-down' }}">{{ "+" if p.price_change_interval > 0 }}{{ p.price_change_interval }}%</td><td><span class="badge bg-warning text-dark">x{{ p.spike_factor }}</span></td><td class="fw-bold text-white">${{ "{:,.0f}".format(p.vol_inflow) }}</td><td><a href="https://www.bybit.com/trade/usdt/{{ p.symbol }}" target="_blank" class="text-muted hover-white"><i class="fas fa-external-link-alt"></i> Trade</a></td></tr>{% endfor %}
    </tbody></table></div></div></div></div></div>
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script><script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script><script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>
    <script>$(document).ready(function() {$('#signalsTable').DataTable({"order": [[ 0, "desc" ]], "pageLength": 50, "lengthMenu": [25, 50, 100], "dom": '<"d-flex justify-content-between mb-2"f>t<"d-flex justify-content-between mt-2"ip>'});});</script>
    </body></html>
    """
    return render_template_string(html, pumps=data['all_signals'], last_update=last_update, positive_coins=data['positive_coins'], negative_coins=data['negative_coins'])

@app.route('/report', methods=['GET'])
def report_page():
    """Сторінка P&L у стилі Bybit"""
    days = request.args.get('days', default=7, type=int)
    stats, error = bot.get_pnl_stats(days=days)
    if error or not stats: stats = {"total_pnl": 0, "chart_labels": [], "chart_data": [], "history": []}

    html = """
    <!DOCTYPE html><html lang="uk" data-bs-theme="dark"><head><meta charset="UTF-8"><title>P&L Аналіз</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>:root { --bg: #0b0e11; --card: #151a1f; --border: #2a3441; --green: #0ecb81; --red: #f6465d; } body { background: var(--bg); color: #eaecef; font-family: 'DIN Pro', sans-serif; } .card { background: var(--card); border: 1px solid var(--border); } .text-up { color: var(--green); } .text-down { color: var(--red); } .btn-p { background: #1e2329; border:none; color: #848e9c; font-size:12px; margin-right:5px; } .btn-p.active { background: #2b3139; color: #fff; } </style></head><body>
    <div class="container-fluid p-4"><div class="d-flex justify-content-between mb-4"><h3>P&L Аналіз</h3><div><a href="/report?days=7" class="btn btn-sm btn-p {{ 'active' if days==7 }}">7Д</a><a href="/report?days=30" class="btn btn-sm btn-p {{ 'active' if days==30 }}">30Д</a><a href="/scanner" class="btn btn-sm btn-outline-secondary ms-3">← Сканер</a></div></div>
    <div class="row mb-4"><div class="col-md-3"><div class="card"><div class="card-body"><small class="text-muted">P&L ({{days}}д)</small><h3 class="{{ 'text-up' if stats.total_pnl>=0 else 'text-down' }}">{{ '+' if stats.total_pnl>0 }}${{ "{:,.2f}".format(stats.total_pnl) }}</h3></div></div></div>
    <div class="col-md-3"><div class="card"><div class="card-body"><small class="text-muted">Win Rate</small><h3>{{ stats.win_rate }}%</h3></div></div></div>
    <div class="col-md-3"><div class="card"><div class="card-body"><small class="text-muted">Угод</small><h3>{{ stats.total_trades }}</h3></div></div></div>
    <div class="col-md-3"><div class="card"><div class="card-body"><small class="text-muted">Об'єм</small><h3>${{ "{:,.0f}".format(stats.volume) }}</h3></div></div></div></div>
    <div class="card mb-4"><div class="card-body"><canvas id="pnlChart" height="100"></canvas></div></div>
    <div class="card"><div class="card-header">Історія</div><div class="card-body p-0"><table class="table table-dark table-hover mb-0" style="font-size:13px;"><thead><tr><th>Час</th><th>Тікер</th><th>Сторона</th><th>Вхід</th><th>Вихід</th><th class="text-end">P&L</th></tr></thead><tbody>
    {% for t in stats.history %}<tr><td class="text-muted">{{ t.exit_time }}</td><td class="fw-bold">{{ t.symbol }}</td><td><span class="badge {{ 'bg-success' if t.side=='Long' else 'bg-danger' }} bg-opacity-10 {{ 'text-up' if t.side=='Long' else 'text-down' }}">{{ t.side }}</span></td><td>{{ t.entry_price }}</td><td>{{ t.exit_price }}</td><td class="text-end fw-bold {{ 'text-up' if t.pnl>0 else 'text-down' }}">{{ '+' if t.pnl>0 }}{{ t.pnl }}</td></tr>{% endfor %}
    </tbody></table></div></div></div>
    <script>
    const ctx = document.getElementById('pnlChart').getContext('2d');
    const grad = ctx.createLinearGradient(0,0,0,300); grad.addColorStop(0,'rgba(14,203,129,0.2)'); grad.addColorStop(1,'rgba(14,203,129,0)');
    new Chart(ctx, {type: 'line', data: {labels: {{ stats.chart_labels|tojson }}, datasets: [{label:'P&L', data: {{ stats.chart_data|tojson }}, borderColor:'#0ecb81', backgroundColor:grad, fill:true, tension:0.4}]}, options:{responsive:true, plugins:{legend:{display:false}}, scales:{x:{display:false}, y:{grid:{color:'#2a3441'}}}}});
    </script></body></html>
    """
    return render_template_string(html, stats=stats, days=days)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw = request.get_data(as_text=True)
        logger.info(f"📨 RAW WEBHOOK: {raw}")
        data = None
        try: data = json.loads(raw)
        except:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m: 
                try: 
                    data = json.loads(m.group())
                    logger.info("✅ JSON extracted via Regex")
                except: pass
        
        if data:
            logger.info(f"🚀 Processing: {data.get('symbol')} {data.get('action')}")
            # Використовуємо функцію з AI
            threading.Thread(target=process_signal_with_ai, args=(data,)).start()
            return jsonify({"status": "ok"})
        
        logger.error("❌ No valid JSON found")
        return jsonify({"error": "no json"}), 400
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    try: stats_service.cleanup_old_data(days=config.DATA_RETENTION_DAYS)
    except: pass
    app.run(host=config.HOST, port=config.PORT)
