"""
Main Application - Modular Entry Point 🚀
"""
import os
import logging
import threading
import time
import json
import re
import ctypes
from flask import Flask, request, jsonify, render_template_string
import requests

# Імпорт модулів
from bot_config import config
from bot import bot_instance, send_telegram_message, SmartTradeManager, PassiveManager
from report import render_report_page
from scanner import EnhancedMarketScanner

try:
    import ai_analyst
    AI_AVAILABLE = True
except: AI_AVAILABLE = False

try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_signal_with_ai(data):
    symbol = data.get('symbol')
    action = data.get('action')
    ai_text = "AI OFF"
    if AI_AVAILABLE:
        try: ai_text = ai_analyst.analyze_signal(symbol, action)
        except: pass
    msg = f"ВХІД: {symbol} | {action} | {ai_text}"
    send_telegram_message(msg)
    bot_instance.place_order(data)

def keep_alive():
    time.sleep(10)
    url = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + '/health'
    while True:
        try: requests.get(url, timeout=10)
        except: pass
        time.sleep(300)
threading.Thread(target=keep_alive, daemon=True).start()

# Запуск сервісів
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

trade_manager = SmartTradeManager(bot_instance, scanner)
trade_manager.start()

pass_man = PassiveManager(bot_instance, scanner)
pass_man.start()

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
                    rec = "TP SPLIT"
                    cls = "table-success"
                    active.append({'symbol':sym, 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':rsi, 'pressure':round(press), 'rec':rec, 'cls':cls, 'size':p['size'], 'entry':p['avgPrice']})
    except: pass
    
    # HTML Сканера (короткий)
    from statistics_service import stats_service
    monitor_logs = stats_service.get_monitor_logs(limit=30)
    history = stats_service.get_trades(days=1)
    
    html = """<!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Scanner</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{background:#f7f9fc;font-family:'Roboto';font-size:13px} .card{border:none;box-shadow:0 2px 5px rgba(0,0,0,0.02)}</style><meta http-equiv="refresh" content="10"></head><body><nav class="navbar bg-white mb-3 border-bottom px-3"><span>🐋 Scanner</span><a href="/report" class="btn btn-sm btn-outline-secondary">P&L</a></nav><div class="container-fluid"><div class="card mb-3"><div class="card-header bg-white fw-bold">АКТИВНІ</div><table class="table table-hover mb-0"><thead><tr><th>Монета</th><th>Тип</th><th>Вхід</th><th>P&L</th><th>RSI</th><th>Тиск</th></tr></thead><tbody>{% for a in active %}<tr><td>{{a.symbol}}</td><td>{{a.side}}</td><td>{{a.entry}}</td><td>{{a.pnl}}</td><td>{{a.rsi}}</td><td>{{a.pressure}}</td></tr>{% else %}<tr><td colspan="6" class="text-center text-muted p-3">--</td></tr>{% endfor %}</tbody></table></div><div class="row"><div class="col-6"><div class="card"><div class="card-header bg-white">МОНІТОРИНГ</div><table class="table table-striped mb-0">{% for l in logs %}<tr><td>{{l.time}}</td><td>{{l.symbol}}</td><td>{{l.pnl}}</td></tr>{% endfor %}</table></div></div><div class="col-6"><div class="card"><div class="card-header bg-white">ІСТОРІЯ</div><table class="table table-hover mb-0">{% for t in history %}<tr><td>{{t.exit_time}}</td><td>{{t.symbol}}</td><td>{{t.pnl}}</td></tr>{% endfor %}</table></div></div></div></div></body></html>"""
    from datetime import datetime
    return render_template_string(html, last_update=last_update, active=active, logs=monitor_logs, history=history)

@app.route('/report', methods=['GET'])
def report_page_route():
    return render_report_page(bot_instance, request)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        threading.Thread(target=process_signal_with_ai, args=(data,)).start()
        return jsonify({"status": "ok"})
    except: return jsonify({"error": "error"}), 400

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"
@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)
