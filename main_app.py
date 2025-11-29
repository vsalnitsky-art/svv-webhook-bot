"""
Main App - Clean & Professional
Updated: Scanner page now shows ONLY active trades.
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

# Сканер для моніторингу (тільки активні позиції)
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

# Монітор для запису логів в базу (фоновий запис історії для майбутнього аналізу)
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
        # Отримуємо дані тільки про відкриті позиції
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    sym = p['symbol']
                    # Отримуємо аналітику від сканера (RSI, Pressure)
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
    
    # Спрощений HTML - тільки активні угоди
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Active Trades</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f7f9fc;font-size:14px} 
        .card{border:none;box-shadow:0 2px 5px rgba(0,0,0,0.02)} 
        .text-up{color:#20b26c;font-weight:bold} 
        .text-down{color:#ef454a;font-weight:bold}
        .badge-buy{background-color:#20b26c}
        .badge-sell{background-color:#ef454a}
    </style>
    <meta http-equiv="refresh" content="5">
    </head><body>
    
    <nav class="navbar bg-white mb-4 px-3 border-bottom shadow-sm">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">🐋 Active Monitor</span>
            <a href="/report" class="btn btn-sm btn-outline-secondary">Звіт P&L</a>
        </div>
    </nav>

    <div class="container">
        <div class="card">
            <div class="card-header bg-white fw-bold py-3">
                ВІДКРИТІ ПОЗИЦІЇ
            </div>
            <div class="table-responsive">
                <table class="table table-hover align-middle mb-0">
                    <thead class="table-light">
                        <tr>
                            <th>Монета</th>
                            <th>Тип</th>
                            <th>Розмір</th>
                            <th>Вхід</th>
                            <th>RSI</th>
                            <th>Тиск</th>
                            <th>P&L (USDT)</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for a in active %}
                        <tr>
                            <td class="fw-bold">{{a.symbol}}</td>
                            <td><span class="badge {{ 'badge-buy' if a.side=='Buy' else 'badge-sell' }}">{{a.side}}</span></td>
                            <td>{{a.size}}</td>
                            <td>{{a.entry}}</td>
                            <td>
                                <span class="{{ 'text-danger' if a.rsi > 70 else 'text-success' if a.rsi < 30 else '' }}">
                                    {{a.rsi}}
                                </span>
                            </td>
                            <td>{{a.pressure}}</td>
                            <td class="{{ 'text-up' if a.pnl>0 else 'text-down' }}" style="font-size: 1.1em;">
                                {{ "+" if a.pnl > 0 }}{{a.pnl}}$
                            </td>
                        </tr>
                    {% else %}
                        <tr>
                            <td colspan="7" class="text-center text-muted py-5">
                                Немає активних угод 💤
                            </td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    </body></html>
    """
    return render_template_string(html, active=active)

@app.route('/report', methods=['GET'])
def report_route():
    # Сторінка зі звітами
    from report import render_report_page
    return render_report_page(bot_instance, request)

@app.route('/')
def home(): 
    # Редірект з головної на сканер
    return "<script>window.location.href='/scanner';</script>"

@app.route('/health')
def health(): 
    # Ендпоінт для Self-Ping
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)