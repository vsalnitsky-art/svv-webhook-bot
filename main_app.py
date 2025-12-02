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
    """Self-Ping для Render"""
    time.sleep(5)
    base_url = os.environ.get('RENDER_EXTERNAL_URL')
    if not base_url:
        base_url = f'http://127.0.0.1:{config.PORT}'
    
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
    # 1. Обробка періоду (7 або 30 днів)
    try:
        days_param = int(request.args.get('days', 7))
        if days_param not in [7, 30]: days_param = 7
    except: days_param = 7
    
    # 2. Синхронізація (оновлення даних з біржі)
    try:
        bot_instance.sync_trades(days=days_param)
    except Exception as e:
        logger.error(f"Sync error on home load: {e}")

    # 3. Базові дані
    balance = bot_instance.get_bal()
    active_count = len(scanner.get_active_symbols())
    
    # 4. Статистика угод (P&L та L/S)
    trades = stats_service.get_trades(days=days_param)
    
    period_pnl = 0.0
    wins = 0
    longs = 0
    shorts = 0
    total_trades = len(trades)
    
    for t in trades:
        period_pnl += t['pnl']
        if t['pnl'] > 0: wins += 1
        
        # Підрахунок для кругової діаграми
        if t['side'] == 'Long': longs += 1
        elif t['side'] == 'Short': shorts += 1
            
    win_rate = int((wins / total_trades) * 100) if total_trades > 0 else 0
    
    # Дата замість часу (напр. 02 Dec 2025)
    current_date = datetime.utcnow().strftime('%d %b %Y')
    
    return render_template('index.html', 
                           date=current_date,
                           balance=balance,
                           active_count=active_count,
                           period_pnl=period_pnl,
                           win_rate=win_rate,
                           longs=longs,
                           shorts=shorts,
                           days=days_param, 
                           trades=trades[:7]) # Останні 7 угод для списку

@app.route('/scanner', methods=['GET'])
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    symbol = p['symbol']
                    active.append({
                        'symbol': symbol, 
                        'side': p['side'], 
                        'pnl': round(float(p['unrealisedPnl']), 2), 
                        'rsi': scanner.get_current_rsi(symbol), 
                        'pressure': round(scanner.get_market_pressure(symbol)), 
                        'size': p['size'], 
                        'entry': p['avgPrice'],
                        'time': datetime.now().strftime('%H:%M')
                    })
    except Exception as e:
        logger.error(f"Scanner page error: {e}")
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

# === СТРАТЕГІЯ OB + CLOUD (Fix 404) ===
@app.route('/ob_trend/settings', methods=['GET', 'POST'])
def ob_trend_settings_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        # Обробка чекбоксів для стратегії
        for cb in ['obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter']:
            form_data[cb] = request.form.get(cb) == 'on'
        
        settings.save_settings(form_data)
        return redirect(url_for('ob_trend_settings_page'))
    
    return render_template('strategy_ob_trend.html', conf=settings._cache)

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        form_data = request.form.to_dict()
        
        # Обробка чекбоксів
        if 'useOBRetest' in form_data and form_data['useOBRetest'] == 'on':
            form_data['obt_useOBRetest'] = True
        else:
            form_data['obt_useOBRetest'] = False
            
        filters_map = {
            'useCloudFilter': 'obt_useCloudFilter',
            'useObvFilter': 'obt_useObvFilter',
            'useRsiFilter': 'obt_useRsiFilter'
        }
        
        for short_key, long_key in filters_map.items():
            if short_key in form_data and form_data[short_key] == 'on':
                form_data[long_key] = True
            else:
                form_data[long_key] = False
        
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

@app.route('/settings/export')
def export_settings():
    data = settings.get_all()
    json_str = json.dumps(data, indent=4)
    return Response(json_str, mimetype='application/json', headers={'Content-Disposition': 'attachment;filename=bot_settings.json'})

@app.route('/settings/import', methods=['POST'])
def import_settings():
    if 'file' not in request.files: return "No file part", 400
    file = request.files['file']
    if file.filename == '': return "No selected file", 400
    try:
        data = json.load(file)
        if settings.import_settings(data): return redirect(url_for('settings_general_page'))
        else: return "Error importing settings", 500
    except Exception as e: return f"Invalid JSON: {e}", 400

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)