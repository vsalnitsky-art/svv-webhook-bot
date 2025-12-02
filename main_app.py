import logging
import threading
import time
import json
import ctypes
import os
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response

from bot_config import config
from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from settings_manager import settings
from market_analyzer import market_analyzer

# Запобігання сну у Windows
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flask'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Запуск фонових процесів
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

# --- BACKGROUND TASKS ---

def monitor_active():
    """Фоновий потік для запису стану позицій в БД"""
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
        except: pass
        time.sleep(10)

def keep_alive():
    time.sleep(5)
    base_url = os.environ.get('RENDER_EXTERNAL_URL')
    if not base_url: base_url = f'http://127.0.0.1:{config.PORT}'
    target = f"{base_url}/health"
    logger.info(f"💓 Keep-alive target: {target}")
    while True:
        try: requests.get(target, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=monitor_active, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()

# --- ROUTES ---

@app.route('/')
def home():
    try:
        days_param = int(request.args.get('days', 7))
        if days_param not in [7, 30]: days_param = 7
    except: days_param = 7
    try: bot_instance.sync_trades(days=days_param)
    except Exception as e: logger.error(f"Sync error: {e}")

    balance = bot_instance.get_bal()
    active_count = len(scanner.get_active_symbols())
    trades = stats_service.get_trades(days=days_param)
    period_pnl = 0.0; wins = 0; longs = 0; shorts = 0
    
    for t in trades:
        period_pnl += t['pnl']
        if t['pnl'] > 0: wins += 1
        if t['side'] == 'Long': longs += 1
        elif t['side'] == 'Short': shorts += 1
            
    win_rate = int((wins / len(trades)) * 100) if len(trades) > 0 else 0
    current_date = datetime.utcnow().strftime('%d %b %Y')
    
    return render_template('index.html', date=current_date, balance=balance, active_count=active_count,
                           period_pnl=period_pnl, win_rate=win_rate, longs=longs, shorts=shorts,
                           days=days_param, trades=trades[:7])

@app.route('/scanner')
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    symbol = p['symbol']
                    active.append({
                        'symbol': symbol, 'side': p['side'], 'pnl': round(float(p['unrealisedPnl']), 2), 
                        'rsi': scanner.get_current_rsi(symbol), 'pressure': round(scanner.get_market_pressure(symbol)), 
                        'size': p['size'], 'entry': p['avgPrice'], 'time': datetime.now().strftime('%H:%M')
                    })
    except: pass
    return render_template('scanner.html', active=active)

@app.route('/analyzer')
def analyzer_page():
    return render_template('analyzer.html', results=market_analyzer.get_results(), conf=settings._cache, 
                           progress=market_analyzer.progress, status=market_analyzer.status_message, 
                           is_scanning=market_analyzer.is_scanning)

# === CENTRALIZED SETTINGS ROUTE ===
@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        
        # Обробка ВСІХ чекбоксів (Global + Strategy)
        checkboxes = [
            'telegram_enabled', 
            'obt_useCloudFilter', 
            'obt_useObvFilter', 
            'obt_useRsiFilter', 
            'obt_useOBRetest'
        ]
        
        for cb in checkboxes:
            form_data[cb] = request.form.get(cb) == 'on'
        
        settings.save_settings(form_data)
        return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        form_data = request.form.to_dict()
        if 'useOBRetest' in form_data and form_data['useOBRetest'] == 'on': form_data['obt_useOBRetest'] = True
        else: form_data['obt_useOBRetest'] = False
        filters_map = {'useCloudFilter': 'obt_useCloudFilter', 'useObvFilter': 'obt_useObvFilter', 'useRsiFilter': 'obt_useRsiFilter'}
        for short, long_k in filters_map.items():
            form_data[long_k] = (short in form_data and form_data[short] == 'on')
        settings.save_settings(form_data)
    market_analyzer.run_scan_thread()
    return jsonify({"status": "started"})

@app.route('/analyzer/status')
def get_scan_status():
    return jsonify({"progress": market_analyzer.progress, "message": market_analyzer.status_message, "is_scanning": market_analyzer.is_scanning})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        logger.info(f"🔔 SIGNAL: {data.get('symbol')} {data.get('action')}")
        result = bot_instance.place_order(data)
        return jsonify(result), (200 if result.get("status") in ["ok", "ignored"] else 400)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/settings/export')
def export_settings():
    return Response(json.dumps(settings.get_all(), indent=4), mimetype='application/json', headers={'Content-Disposition': 'attachment;filename=bot_settings.json'})

@app.route('/settings/import', methods=['POST'])
def import_settings():
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    try:
        if settings.import_settings(json.load(file)): return redirect(url_for('settings_general_page'))
        else: return "Error", 500
    except: return "Invalid JSON", 400

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)