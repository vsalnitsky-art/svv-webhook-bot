"""
Main Application - Professional Dashboard Version
Включает: Gemini AI, Telegram Alert, Windows Anti-Sleep, Self-Ping, Pro Scanner UI
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
from datetime import datetime, timedelta
import ctypes # Для Windows Anti-Sleep

# Импорт модулей проекта
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

# === WINDOWS ANTI-SLEEP ===
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

# === FLASK SETUP ===
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === TELEGRAM ===
def send_telegram_message(text):
    try:
        # 👇👇 ВСТАВЬТЕ ВАШИ ДАННЫЕ 👇👇
        tg_token = getattr(config, 'TG_BOT_TOKEN', "ВАШ_ТОКЕН")
        chat_id = getattr(config, 'TG_CHAT_ID', "ВАШ_CHAT_ID")
        
        if "ВАШ_" in tg_token: return 
        
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        logger.error(f"TG Error: {e}")

# === AI LOGIC ===
def process_signal_with_ai(data):
    symbol = data.get('symbol')
    action = data.get('action')
    ai_text = "AI Disabled"
    
    if AI_AVAILABLE:
        try: ai_text = ai_analyst.analyze_signal(symbol, action)
        except: ai_text = "Analysis Error"

    msg = f"🚀 <b>SIGNAL: {symbol}</b>\nAction: {action}\n\n🤖 <b>Gemini:</b> {ai_text}"
    send_telegram_message(msg)
    bot.place_order(data)

# === SELF PING ===
def keep_alive():
    time.sleep(10)
    external_url = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + '/health'
    while True:
        try: requests.get(external_url, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()

# === TRADING BOT ===
class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)
        logger.info("✅ Bybit Connected")

    def get_available_balance(self, currency="USDT"):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            for acc in b.get('result', {}).get('list', []):
                for c in acc.get('coin', []):
                    if c.get('coin') == currency: return float(c.get('walletBalance', 0))
            return None
        except: return None

    # ... (Остальные методы API оставлены без изменений для краткости, они работают отлично) ...
    # Важно: Здесь должны быть методы get_all_tickers, get_current_price, place_order и т.д.
    # Я использую "заглушку" ниже, чтобы код влез в сообщение, но вы ОСТАВЬТЕ свои методы из старого файла!
    
    def get_all_tickers(self):
        try: return self.session.get_tickers(category="linear")['result']['list']
        except: return []

    def place_order(self, data):
        # Вставьте сюда вашу полную логику place_order из старого файла
        # Она была написана правильно, нет смысла её менять.
        logger.info(f"Placing order for {data.get('symbol')}")
        pass
    
    def get_pnl_stats(self, days=7):
        return stats_service.get_trades(days=days), None

# Инициализация
bot = BybitTradingBot()
scanner = EnhancedMarketScanner(bot, config.get_scanner_config())
scanner.start()

# === WEB ROUTES (PROFESSIONAL UI) ===

@app.route('/scanner', methods=['GET'])
def scanner_page():
    """Профессиональный дашборд трейдера"""
    data = scanner.get_aggregated_data(hours=24)
    last_update = datetime.now().strftime('%H:%M:%S')
    
    # ПРОФЕССИОНАЛЬНЫЙ HTML ШАБЛОН
    html_template = """
    <!DOCTYPE html>
    <html lang="en" data-bs-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whale Terminal Pro</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            :root { --bg-dark: #0b0e11; --card-bg: #151a1f; --border: #2a3441; --green: #0ecb81; --red: #f6465d; }
            body { background-color: var(--bg-dark); font-family: 'Inter', sans-serif; color: #eaecef; font-size: 13px; }
            
            /* Карточки */
            .card { background-color: var(--card-bg); border: 1px solid var(--border); border-radius: 4px; margin-bottom: 15px; }
            .card-header { background-color: #1b2129; border-bottom: 1px solid var(--border); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 11px; color: #848e9c; padding: 8px 15px; }
            
            /* Таблицы */
            .table-dark { background-color: transparent; --bs-table-bg: transparent; }
            table.dataTable td, table.dataTable th { border-bottom: 1px solid var(--border) !important; vertical-align: middle; padding: 8px 10px; }
            .table-hover tbody tr:hover { background-color: #1e2329 !important; }
            
            /* Цвета текста */
            .text-up { color: var(--green) !important; }
            .text-down { color: var(--red) !important; }
            .text-muted { color: #848e9c !important; }
            
            /* Прогресс бары */
            .progress { height: 4px; background-color: #2b3139; margin-top: 4px; border-radius: 2px; }
            
            /* Бейджи */
            .badge-anomaly { background: rgba(240, 185, 11, 0.15); color: #f0b90b; border: 1px solid rgba(240, 185, 11, 0.3); }
            .live-dot { height: 8px; width: 8px; background-color: var(--green); border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
            @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.4; } 100% { opacity: 1; } }
            
            /* DataTables Customization */
            .dataTables_filter input { background: #1e2329; border: 1px solid var(--border); color: white; font-size: 12px; padding: 4px 8px; }
            .dataTables_length select { background: #1e2329; border: 1px solid var(--border); color: white; font-size: 12px; }
            .page-link { background: #1e2329; border-color: var(--border); color: #848e9c; font-size: 11px; }
            .page-item.active .page-link { background: #2b3139; border-color: #474d57; color: white; }
        </style>
        <meta http-equiv="refresh" content="60">
    </head>
    <body>
        <nav class="navbar navbar-expand-lg navbar-dark" style="background: #181a20; border-bottom: 1px solid var(--border);">
            <div class="container-fluid">
                <a class="navbar-brand" href="#"><i class="fas fa-robot text-primary me-2"></i>Whale Terminal <span class="badge bg-primary" style="font-size: 9px; vertical-align: top;">PRO</span></a>
                <div class="d-flex align-items-center">
                    <span class="text-muted me-3"><span class="live-dot me-1"></span> Live Scanner</span>
                    <span class="text-muted" style="font-size: 11px; border-left: 1px solid var(--border); padding-left: 10px;">Updated: {{ last_update }}</span>
                </div>
            </div>
        </nav>

        <div class="container-fluid mt-3">
            <div class="row">
                <div class="col-lg-6">
                    <div class="card h-100">
                        <div class="card-header text-up d-flex justify-content-between">
                            <span><i class="fas fa-arrow-trend-up me-1"></i> TOP BUYING PRESSURE (24H)</span>
                            <span>{{ positive_coins|length }} Assets</span>
                        </div>
                        <div class="card-body p-0">
                            <div class="table-responsive" style="max-height: 250px; overflow-y: auto;">
                                <table class="table table-dark table-sm table-hover mb-0">
                                    <thead class="text-muted" style="position: sticky; top: 0; background: #151a1f; z-index: 1;">
                                        <tr><th>Asset</th><th class="text-end">Change</th><th class="text-end">Net Inflow</th></tr>
                                    </thead>
                                    <tbody>
                                        {% for coin in positive_coins[:15] %}
                                        <tr>
                                            <td class="fw-bold">{{ coin.symbol }}</td>
                                            <td class="text-end text-up">+{{ coin.avg_change }}%</td>
                                            <td class="text-end">
                                                ${{ "{:,.0f}".format(coin.inflow) }}
                                                <div class="progress"><div class="progress-bar bg-success" style="width: {{ coin.bar_pct }}%"></div></div>
                                            </td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-6">
                    <div class="card h-100">
                        <div class="card-header text-down d-flex justify-content-between">
                            <span><i class="fas fa-arrow-trend-down me-1"></i> TOP SELLING PRESSURE (24H)</span>
                            <span>{{ negative_coins|length }} Assets</span>
                        </div>
                        <div class="card-body p-0">
                            <div class="table-responsive" style="max-height: 250px; overflow-y: auto;">
                                <table class="table table-dark table-sm table-hover mb-0">
                                    <thead class="text-muted" style="position: sticky; top: 0; background: #151a1f; z-index: 1;">
                                        <tr><th>Asset</th><th class="text-end">Change</th><th class="text-end">Net Outflow</th></tr>
                                    </thead>
                                    <tbody>
                                        {% for coin in negative_coins[:15] %}
                                        <tr>
                                            <td class="fw-bold">{{ coin.symbol }}</td>
                                            <td class="text-end text-down">{{ coin.avg_change }}%</td>
                                            <td class="text-end">
                                                ${{ "{:,.0f}".format(coin.inflow) }}
                                                <div class="progress"><div class="progress-bar bg-danger" style="width: {{ coin.bar_pct }}%"></div></div>
                                            </td>
                                        </tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <span><i class="fas fa-list-ul me-1"></i> INSTITUTIONAL VOLUME LOG</span>
                            <a href="/report" class="btn btn-outline-secondary btn-sm" style="font-size: 10px; padding: 2px 8px;">VIEW P&L REPORT</a>
                        </div>
                        <div class="card-body p-2">
                            <table id="signalsTable" class="table table-dark table-hover w-100" style="font-size: 12px;">
                                <thead class="text-muted">
                                    <tr>
                                        <th>Time</th>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>1m Change</th>
                                        <th>Anomaly Factor</th>
                                        <th>Est. Volume</th>
                                        <th>Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for p in pumps %}
                                    <tr style="{{ 'background: rgba(14, 203, 129, 0.08);' if p.vol_inflow > 1000000 else '' }}">
                                        <td class="text-muted">{{ p.time }}</td>
                                        <td>
                                            <span class="fw-bold text-primary">{{ p.symbol }}</span>
                                        </td>
                                        <td>{{ p.price }}</td>
                                        <td class="{{ 'text-up' if p.price_change_interval > 0 else 'text-down' }}">
                                            {{ "+" if p.price_change_interval > 0 }}{{ p.price_change_interval }}%
                                        </td>
                                        <td>
                                            <span class="badge badge-anomaly">x{{ p.spike_factor }}</span>
                                        </td>
                                        <td class="fw-bold text-white">
                                            ${{ "{:,.0f}".format(p.vol_inflow) }}
                                        </td>
                                        <td>
                                            <a href="https://www.bybit.com/trade/usdt/{{ p.symbol }}" target="_blank" class="text-decoration-none text-muted hover-white">
                                                <i class="fas fa-external-link-alt"></i> Trade
                                            </a>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script>
        <script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>
        <script>
            $(document).ready(function() {
                $('#signalsTable').DataTable({
                    "order": [[ 0, "desc" ]], // Сортировка по времени (свежие сверху)
                    "pageLength": 50,
                    "lengthMenu": [25, 50, 100],
                    "language": { "search": "", "searchPlaceholder": "Search Coin..." },
                    "dom": '<"d-flex justify-content-between mb-2"f>t<"d-flex justify-content-between mt-2"ip>'
                });
            });
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_template, 
                                  pumps=data['all_signals'], 
                                  last_update=last_update,
                                  positive_coins=data['positive_coins'], 
                                  negative_coins=data['negative_coins'])

@app.route('/report', methods=['GET'])
def report_page():
    days = request.args.get('days', default=7, type=int)
    stats, error = bot.get_pnl_stats(days=days)
    if error: return f"<h1>Error</h1><p>{error}</p>"
    
    # Простой HTML для отчета (можно тоже улучшить, но пока оставим минималистичным)
    return f"""
    <body style='background:#0b0e11; color:#fff; font-family:sans-serif; padding:20px;'>
        <h1>📊 P&L Report ({days} days)</h1>
        <div style='background:#151a1f; padding:20px; border-radius:8px; margin-bottom:20px;'>
            <h2>Total P&L: <span style='color:{'#0ecb81' if stats['total_pnl']>=0 else '#f6465d'}'>${stats['total_pnl']:.2f}</span></h2>
            <p>Trades: {stats['total_trades']} (Win Rate: {stats.get('win_trades',0)/stats['total_trades']*100 if stats['total_trades']>0 else 0:.1f}%)</p>
        </div>
        <a href='/scanner' style='color:#3b82f6'>← Back to Scanner</a>
    </body>
    """

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw = request.get_data(as_text=True)
        data = None
        try: data = json.loads(raw)
        except: 
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m: data = json.loads(m.group())
        
        if data:
            logger.info(f"🚀 Webhook: {data.get('symbol')} {data.get('action')}")
            threading.Thread(target=process_signal_with_ai, args=(data,)).start()
            return jsonify({"status": "ok"})
        return jsonify({"error": "no data"}), 400
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"

@app.route('/health')
def health(): return jsonify({"status": "ok"})

# Запуск очистки старых данных при старте
if __name__ == '__main__':
    try: stats_service.cleanup_old_data(days=config.DATA_RETENTION_DAYS)
    except: pass
    
    app.run(host=config.HOST, port=config.PORT)
