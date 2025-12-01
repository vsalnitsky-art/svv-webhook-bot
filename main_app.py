import logging
import threading
import time
import json
import ctypes
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
import requests

from bot_config import config
from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from settings_manager import settings
from market_analyzer import market_analyzer

try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
app.secret_key = 'secret'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

def monitor_active():
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                for p in r['result']['list']:
                    if float(p['size']) > 0:
                        stats_service.save_monitor_log({
                            'symbol': p['symbol'], 'price': float(p['avgPrice']), 
                            'pnl': float(p['unrealisedPnl']), 'rsi': scanner.get_current_rsi(p['symbol']), 
                            'pressure': scanner.get_market_pressure(p['symbol'])
                        })
        except: pass
        time.sleep(10)

def keep_alive():
    time.sleep(5)
    target = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + "/health"
    while True:
        try: requests.get(target, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=monitor_active, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()

@app.route('/')
def home(): return render_template('index.html', time=datetime.utcnow().strftime('%H:%M:%S UTC'))

@app.route('/scanner')
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    ts = p.get('createdTime') or p.get('updatedTime', time.time()*1000)
                    active.append({'symbol':p['symbol'], 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':scanner.get_current_rsi(p['symbol']), 'size':p['size'], 'entry':p['avgPrice'], 'time':datetime.fromtimestamp(int(ts)/1000).strftime('%d.%m %H:%M')})
    except: pass
    return render_template('scanner.html', active=active)

@app.route('/analyzer')
def analyzer_page():
    return render_template('analyzer.html', results=market_analyzer.get_results(), conf=settings._cache, progress=market_analyzer.progress, status=market_analyzer.status_message, is_scanning=market_analyzer.is_scanning)

@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    if request.method == 'POST':
        f = request.form.to_dict()
        f['telegram_enabled'] = request.form.get('telegram_enabled') == 'on'
        settings.save_settings(f)
        return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)

@app.route('/ob_trend/settings', methods=['GET', 'POST'])
def ob_trend_settings_page():
    if request.method == 'POST':
        f = request.form.to_dict()
        for cb in ['obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter', 'obt_useBtcDominance', 'obt_useOBRetest']:
            f[cb] = request.form.get(cb) == 'on'
        settings.save_settings(f)
        return redirect(url_for('ob_trend_settings_page'))
    return render_template('strategy_ob_trend.html', conf=settings._cache)

@app.route('/analyzer/settings')
def analyzer_settings_redirect(): return redirect(url_for('ob_trend_settings_page'))

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        f = request.form.to_dict()
        # Спеціальна обробка для форми сканера
        if 'useOBRetest' not in f: f['obt_useOBRetest'] = False # Маппінг на новий ключ
        else: f['obt_useOBRetest'] = True
        
        # Маппінг фільтрів зі сканера на нові ключі obt_
        if 'useCloudFilter' in f: f['obt_useCloudFilter'] = True
        else: f['obt_useCloudFilter'] = False
        
        if 'useObvFilter' in f: f['obt_useObvFilter'] = True
        else: f['obt_useObvFilter'] = False
        
        if 'useRsiFilter' in f: f['obt_useRsiFilter'] = True
        else: f['obt_useRsiFilter'] = False

        settings.save_settings(f)
    market_analyzer.run_scan_thread()
    return jsonify({"status": "started"})

@app.route('/analyzer/status')
def get_scan_status(): return jsonify({"progress":market_analyzer.progress, "message":market_analyzer.status_message, "is_scanning":market_analyzer.is_scanning})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        bot_instance.place_order(data)
        return jsonify({"status": "ok"})
    except Exception as e: return jsonify({"error": str(e)}), 400

@app.route('/report')
def report_route():
    from report import render_report_page
    return render_report_page(bot_instance, request)

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)