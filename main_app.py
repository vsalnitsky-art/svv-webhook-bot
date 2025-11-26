"""
Main Application - Modular Router 📡
Пов'язує всі модулі разом: Bot, Report, Scanner, Web.
"""
import logging
import threading
import time
import json
import re
import ctypes
import os
from datetime import datetime

from flask import Flask, request, jsonify, render_template_string
from bot_config import config
from bot import bot_instance  # Імпорт бота
from report import render_report_page # Імпорт репорту
from scanner import EnhancedMarketScanner
from statistics_service import stats_service

# AI
try:
    import ai_analyst
    AI_AVAILABLE = True
except: AI_AVAILABLE = False

try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_telegram_message(text):
    clean = text.replace("<b>", "").replace("</b>", "").replace("\n", " | ")
    logger.info(f"\n🔔 [BOT]: {clean}")

def process_signal_with_ai(data):
    symbol = data.get('symbol')
    action = data.get('action')
    ai_text = "AI Off"
    if AI_AVAILABLE:
        try: ai_text = ai_analyst.analyze_signal(symbol, action)
        except: pass
    
    msg = f"SIGNAL: {symbol} | {action} | {ai_text}"
    send_telegram_message(msg)
    
    # Викликаємо бота з модулю bot.py
    bot_instance.place_order(data)

def keep_alive():
    time.sleep(10)
    url = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + '/health'
    while True:
        try: requests.get(url, timeout=10)
        except: pass
        time.sleep(300)
threading.Thread(target=keep_alive, daemon=True).start()

# Запускаємо сканер
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

# === МОНІТОР (Тільки для бази) ===
def monitor_loop():
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                for p in r['result']['list']:
                    if float(p['size']) > 0:
                        stats_service.save_monitor_log({
                            'symbol': p['symbol'], 'price': float(p['avgPrice']), 'pnl': float(p['unrealisedPnl']),
                            'rsi': scanner.get_current_rsi(p['symbol']), 'pressure': scanner.get_market_pressure(p['symbol'])
                        })
        except: pass
        time.sleep(15)
threading.Thread(target=monitor_loop, daemon=True).start()

# === ROUTES ===
@app.route('/scanner', methods=['GET'])
def scanner_page():
    scan_data = scanner.get_aggregated_data(hours=24)
    last_update = datetime.now().strftime('%H:%M:%S')
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']==0:
            for p in r['result']['list']:
                if float(p['size'])>0:
                    sym = p['symbol']
                    rsi = scan_data['snapshots'].get(sym, {}).get('rsi', 50)
                    press = scanner.get_market_pressure(sym)
                    active.append({'symbol':sym, 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':rsi, 'pressure':round(press), 'rec':"Active", 'cls':"table-success", 'size':p['size'], 'entry':p['avgPrice']})
    except: pass

    logs = stats_service.get_monitor_logs(limit=30)
    history = stats_service.get_trades(days=1)

    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Scanner</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>body{background:#f7f9fc;font-family:'Roboto',sans-serif;font-size:13px;overflow:hidden}.navbar{background:#fff;border-bottom:1px solid #e0e0e0;height:50px}.top-block{height:30vh;overflow-y:auto;background:#fff;margin:10px;border-radius:4px;border:1px solid #ddd}.bottom-row{height:60vh;display:flex;padding:0 10px;gap:10px}.half{width:50%;background:#fff;border-radius:4px;border:1px solid #ddd;overflow-y:auto}.head{padding:10px;border-bottom:1px solid #eee;font-weight:bold;position:sticky;top:0;background:#fff}.text-up{color:#20b26c}.text-down{color:#ef454a}</style>
    <meta http-equiv="refresh" content="10"></head><body>
    <nav class="navbar px-3"><span>🐋 Scanner <small class="text-muted">{{ last_update }}</small></span><a href="/report" class="btn btn-sm btn-outline-secondary">P&L</a></nav>
    <div class="top-block"><div class="head">АКТИВНІ</div><table class="table table-hover mb-0"><thead><tr><th>Монета</th><th>Тип</th><th>Вхід</th><th>P&L</th><th>RSI</th><th>Тиск</th></tr></thead><tbody>{% for a in active %}<tr><td>{{a.symbol}}</td><td>{{a.side}}</td><td>{{a.entry}}</td><td class="{{ 'text-up' if a.pnl>0 else 'text-down' }}">{{a.pnl}}</td><td>{{a.rsi}}</td><td>{{a.pressure}}</td></tr>{% else %}<tr><td colspan="6" class="text-center p-3 text-muted">--</td></tr>{% endfor %}</tbody></table></div>
    <div class="bottom-row"><div class="half"><div class="head text-primary">МОНІТОРИНГ</div><table class="table table-striped mb-0"><thead><tr><th>Час</th><th>Монета</th><th>P&L</th><th>RSI</th></tr></thead><tbody>{% for l in logs %}<tr><td>{{l.time}}</td><td>{{l.symbol}}</td><td class="{{ 'text-up' if l.pnl>0 else 'text-down' }}">{{l.pnl}}</td><td>{{l.rsi}}</td></tr>{% endfor %}</tbody></table></div>
    <div class="half"><div class="head">ІСТОРІЯ</div><table class="table table-hover mb-0"><thead><tr><th>Час</th><th>Монета</th><th>Тип</th><th>P&L</th></tr></thead><tbody>{% for t in history %}<tr><td>{{t.exit_time}}</td><td>{{t.symbol}}</td><td>{{t.side}}</td><td class="{{ 'text-up' if t.pnl>0 else 'text-down' }}">{{t.pnl}}</td></tr>{% endfor %}</tbody></table></div></div>
    </body></html>
    """
    return render_template_string(html, last_update=last_update, active=active, logs=logs, history=history)

@app.route('/report', methods=['GET'])
def report_page():
    # Викликаємо функцію з окремого модуля
    return render_report_page(bot_instance, request)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        logger.info(f"🚀 Webhook: {data.get('symbol')}")
        threading.Thread(target=process_signal_with_ai, args=(data,)).start()
        return jsonify({"status": "ok"})
    except: 
        try: 
            data = json.loads(re.search(r'\{.*\}', request.get_data(as_text=True), re.DOTALL).group())
            threading.Thread(target=process_signal_with_ai, args=(data,)).start()
            return jsonify({"status": "ok"})
        except: return jsonify({"error": "error"}), 400

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"
@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)
