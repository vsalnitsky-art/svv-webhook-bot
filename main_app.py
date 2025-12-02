"""
Main App - Stable & Fixed
Виправлено маппінг налаштувань зі сторінки Сканера.
"""
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
from strategy_ob_trend import ob_trend_strategy

# Запобігання сну у Windows
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
app.secret_key = 'secret_key_simple'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Запуск сканера активних позицій
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
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        time.sleep(10)

def keep_alive():
    """Self-Ping для Render"""
    time.sleep(5)
    base_url = os.environ.get('RENDER_EXTERNAL_URL')
    if not base_url:
        base_url = f'http://127.0.0.1:{config.PORT}'
    
    target = f"{base_url}/health"
    
    while True:
        try:
            requests.get(target, timeout=10)
        except:
            pass
        time.sleep(300)

threading.Thread(target=monitor_active, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()

# --- ROUTES ---

@app.route('/')
def home():
    return render_template('index.html', time=datetime.utcnow().strftime('%H:%M:%S UTC'))

@app.route('/scanner', methods=['GET'])
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    symbol = p['symbol']
                    
                    # Конвертація часу
                    c_time = p.get('createdTime')
                    if not c_time or c_time == '0':
                        c_time = p.get('updatedTime', time.time() * 1000)
                    
                    dt_obj = datetime.fromtimestamp(int(c_time) / 1000)
                    formatted_time = dt_obj.strftime('%d.%m %H:%M')
                    
                    active.append({
                        'symbol': symbol, 
                        'side': p['side'], 
                        'pnl': round(float(p['unrealisedPnl']), 2), 
                        'rsi': scanner.get_current_rsi(symbol), 
                        'pressure': round(scanner.get_market_pressure(symbol)), 
                        'size': p['size'], 
                        'entry': p['avgPrice'],
                        'time': formatted_time
                    })
    except Exception as e: logger.error(f"Scanner error: {e}")
    return render_template('scanner.html', active=active)

@app.route('/analyzer')
def analyzer_page():
    results = market_analyzer.get_results()
    conf = settings._cache
    return render_template('analyzer.html', 
                           results=results, 
                           conf=conf, 
                           progress=market_analyzer.progress,
                           status=market_analyzer.status_message,
                           is_scanning=market_analyzer.is_scanning)

@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['telegram_enabled'] = request.form.get('telegram_enabled') == 'on'
        settings.save_settings(form_data)
        return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)

@app.route('/ob_trend/settings', methods=['GET', 'POST'])
def ob_trend_settings_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        # Тут імена полів у формі вже відповідають базі (obt_...)
        for cb in ['obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter', 'obt_useBtcDominance', 'obt_useOBRetest']:
            form_data[cb] = request.form.get(cb) == 'on'
        
        settings.save_settings(form_data)
        return redirect(url_for('ob_trend_settings_page'))
    return render_template('strategy_ob_trend.html', conf=settings._cache)

@app.route('/analyzer/settings', methods=['GET', 'POST'])
def analyzer_settings_page():
    return redirect(url_for('ob_trend_settings_page'))

# --- SCANNER ACTION (ВИПРАВЛЕНО МАППІНГ) ---

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        form_data = request.form.to_dict()
        
        # 1. Маппінг чекбокса OB Retest
        # У формі HTML name="useOBRetest", в базі ключ "obt_useOBRetest"
        if 'useOBRetest' in form_data and form_data['useOBRetest'] == 'on':
            form_data['obt_useOBRetest'] = True
        else:
            form_data['obt_useOBRetest'] = False
            
        # 2. Маппінг фільтрів (Cloud, OBV, RSI)
        # У формі name="useCloudFilter", в базі "obt_useCloudFilter"
        filter_map = {
            'useCloudFilter': 'obt_useCloudFilter',
            'useObvFilter': 'obt_useObvFilter',
            'useRsiFilter': 'obt_useRsiFilter'
        }
        
        for ui_key, db_key in filter_map.items():
            if ui_key in form_data and form_data[ui_key] == 'on':
                form_data[db_key] = True
            else:
                form_data[db_key] = False
        
        # Зберігаємо в базу
        settings.save_settings(form_data)
    
    market_analyzer.run_scan_thread()
    return jsonify({"status": "started"})

@app.route('/analyzer/status')
def get_scan_status():
    return jsonify({
        "progress": market_analyzer.progress,
        "message": market_analyzer.status_message,
        "is_scanning": market_analyzer.is_scanning
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        logger.info(f"🔔 SIGNAL: {data.get('symbol')} {data.get('action')}")
        result = bot_instance.place_order(data)
        return jsonify(result), (200 if result.get("status") in ["ok", "ignored"] else 400)
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/report', methods=['GET'])
def report_route():
    from report import render_report_page
    return render_report_page(bot_instance, request)

@app.route('/health')
def health(): return jsonify({"status": "ok"})

@app.route('/settings/export')
def export_settings():
    data = settings.get_all()
    return Response(json.dumps(data, indent=4), mimetype='application/json', headers={'Content-Disposition': 'attachment;filename=bot_settings.json'})

@app.route('/settings/import', methods=['POST'])
def import_settings():
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No file", 400
    try:
        settings.import_settings(json.load(file))
        return redirect(url_for('settings_general_page'))
    except Exception as e: return f"Error: {e}", 400

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)