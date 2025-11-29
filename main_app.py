"""
Main App - Clean & Professional 
Updated with proper logging and self-ping mechanism
"""
import logging
import threading
import time
import json
import ctypes
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import requests

from bot_config import config
from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from report import render_report_page

# Запобігання сну у Windows (якщо запускається локально)
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Сканер для моніторингу
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

# Монітор для запису логів в базу
def monitor_active():
    """Фоновий потік для запису стану позицій в БД"""
    logger.info("Starting active position monitor...")
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                for p in r['result']['list']:
                    if float(p['size']) > 0:
                        stats_service.save_monitor_log({
                            'symbol': p['symbol'], 
                            'price': float(p['avgPrice']), 
                            'pnl': float(p['unrealisedPnl']), 
                            'rsi': scanner.get_current_rsi(p['symbol']), 
                            'pressure': scanner.get_market_pressure(p['symbol'])
                        })
            else:
                logger.warning(f"Monitor Warning: {r.get('retMsg')}")
        except Exception as e:
            logger.error(f"Error in monitor_active loop: {e}")
        
        time.sleep(10)

threading.Thread(target=monitor_active, daemon=True).start()

def keep_alive():
    """
    Механізм запобігання засипанню (Self-Ping).
    Пінгує сам себе кожні 5 хвилин.
    """
    time.sleep(5)
    
    external_url = os.environ.get('RENDER_EXTERNAL_URL')
    local_url = f'http://127.0.0.1:{config.PORT}/health'
    
    target_url = f"{external_url}/health" if external_url else local_url
    logger.info(f"💓 Keep-alive service started. Target: {target_url}")

    while True:
        try:
            response = requests.get(target_url, timeout=10)
            if response.status_code == 200:
                logger.info(f"💓 Self-Ping OK: {target_url}")
            else:
                logger.warning(f"⚠️ Self-Ping returned status: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Self-Ping Failed: {e}")
        
        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        # Логування дії (Close або Buy/Sell)
        logger.info(f"🔔 SIGNAL RECEIVED: {data.get('symbol')} {data.get('action')}")
        
        result = bot_instance.place_order(data)
        
        # Обробка статусів, що повернув bot.py
        if result.get("status") in ["ok", "ignored"]:
            return jsonify(result)
        else:
            logger.error(f"Order action failed: {result}")
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/scanner', methods=['GET'])
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    sym = p['symbol']
                    rsi = scanner.get_current_rsi(sym)
                    press = scanner.get_market_pressure(sym)
                    active.append({
                        'symbol': sym, 
                        'side': p['side'], 
                        'pnl': round(float(p['unrealisedPnl']), 2), 
                        'rsi': rsi, 
                        'pressure': round(press), 
                        'size': p['size'], 
                        'entry': p['avgPrice']
                    })
        else:
            logger.error(f"Scanner Page API Error: {r.get('retMsg')}")
    except Exception as e:
        logger.error(f"Error rendering scanner page: {e}")
    
    logs = stats_service.get_monitor_logs(30)
    history = stats_service.get_trades(1) 
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Scanner</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>body{background:#f7f9fc;font-size:13px} .card{border:none;box-shadow:0 2px 5px rgba(0,0,0,0.02)} .text-up{color:#20b26c;font-weight:bold} .text-down{color:#ef454a;font-weight:bold}</style>
    <meta http-equiv="refresh" content="5"></head><body>
    <nav class="navbar bg-white mb-3 px-3 border-bottom"><strong>🐋 Whale Scanner</strong><a href="/report" class="btn btn-sm btn-outline-secondary">Report</a></nav>
    <div class="container-fluid">
        <div class="card mb-3"><div class="card-header bg-white">АКТИВНІ</div><table class="table table-hover mb-0"><thead><tr><th>Монета</th><th>Тип</th><th>Вхід</th><th>P&L</th><th>RSI</th><th>Тиск</th></tr></thead><tbody>
        {% for a in active %}<tr><td>{{a.symbol}}</td><td><span class="badge {{ 'bg-success' if a.side=='Buy' else 'bg-danger' }}">{{a.side}}</span></td><td>{{a.entry}}</td><td class="{{ 'text-up' if a.pnl>0 else 'text-down' }}">{{a.pnl}}$</td><td>{{a.rsi}}</td><td>{{a.pressure}}</td></tr>{% else %}<tr><td colspan="6" class="text-center text-muted">--</td></tr>{% endfor %}
        </tbody></table></div>
        <div class="row"><div class="col-6"><div class="card"><div class="card-header bg-white">МОНІТОРИНГ</div><table class="table table-striped mb-0">{% for l in logs %}<tr><td>{{l.time}}</td><td>{{l.symbol}}</td><td>{{l.price}}</td><td class="{{ 'text-up' if l.pnl>0 else 'text-down' }}">{{l.pnl}}</td></tr>{% endfor %}</table></div></div>
        <div class="col-6"><div class="card"><div class="card-header bg-white">ІСТОРІЯ</div><table class="table table-hover mb-0">{% for t in history %}<tr><td>{{t.exit_time}}</td><td>{{t.symbol}}</td><td class="{{ 'text-up' if t.pnl>0 else 'text-down' }}">{{t.pnl}}</td></tr>{% endfor %}</table></div></div></div>
    </div></body></html>
    """
    return render_template_string(html, active=active, logs=logs, history=history)

@app.route('/report', methods=['GET'])
def report_route():
    from report import render_report_page
    return render_report_page(bot_instance, request)

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"
@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)
