import logging
import threading
import time
import json
import ctypes
import os
import requests
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from sqlalchemy import desc

from bot_config import config
from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from settings_manager import settings
from models import db_manager, OrderBlock
from market_analyzer import market_analyzer

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
    while True:
        try: requests.get(target, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=monitor_active, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()

# --- API ENDPOINT FOR CHART DATA (FIXED) ---
@app.route('/api/chart_data/<symbol>')
def get_chart_data(symbol):
    try:
        # 1. Нормалізація символу (видалення .P для API)
        clean_symbol = symbol.replace('.P', '')
        # logger.info(f"📊 Chart Request: {symbol} -> {clean_symbol}")
        
        # 2. Get Candles (HTF)
        htf = settings.get("htfSelection")
        df = market_analyzer.fetch_candles(clean_symbol, htf, limit=200)
        
        candles = []
        if df is not None:
            for _, row in df.iterrows():
                candles.append({
                    'time': int(row['time'].timestamp()),
                    'open': row['open'], 'high': row['high'], 'low': row['low'], 'close': row['close']
                })
        else:
            logger.warning(f"⚠️ No candles found for {clean_symbol}")

        # 3. Get Order Blocks form DB
        session = db_manager.get_session()
        # Шукаємо блоки і для "BTCUSDT" і для "BTCUSDT.P" (на всяк випадок)
        db_blocks = session.query(OrderBlock).filter(OrderBlock.symbol.in_([symbol, clean_symbol])).all()
        
        blocks = []
        for b in db_blocks:
            blocks.append({
                'type': b.ob_type,
                'top': b.top,
                'bottom': b.bottom,
                'timeframe': b.timeframe
            })
        session.close()

        return jsonify({'candles': candles, 'blocks': blocks})
    except Exception as e:
        logger.error(f"❌ Chart Data Error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

# --- ROUTES ---
@app.route('/')
def home():
    try: days_param = int(request.args.get('days', 7))
    except: days_param = 7
    if days_param not in [7, 30, 90]: days_param = 7
    
    try: bot_instance.sync_trades(days=days_param)
    except: pass

    balance = bot_instance.get_bal()
    active_count = len(scanner.get_active_symbols())
    trades = stats_service.get_trades(days=days_param)
    
    period_pnl = sum(t['pnl'] for t in trades)
    longs = sum(1 for t in trades if t['side'] == 'Long')
    shorts = sum(1 for t in trades if t['side'] == 'Short')
    
    return render_template('index.html', 
                           date=datetime.utcnow().strftime('%d %b %Y'),
                           balance=balance, active_count=active_count,
                           period_pnl=period_pnl, longs=longs, shorts=shorts,
                           days=days_param, trades=trades[:10])

@app.route('/scanner', methods=['GET'])
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    symbol = p['symbol']
                    coin_data = scanner.get_coin_data(symbol)
                    active.append({
                        'symbol': symbol, 
                        'side': p['side'], 
                        'pnl': round(float(p['unrealisedPnl']), 2), 
                        'rsi': coin_data.get('rsi', 0),
                        'exit_status': coin_data.get('exit_status', 'Safe'),
                        'exit_details': coin_data.get('exit_details', '-'),
                        'pressure': round(scanner.get_market_pressure(symbol), 1), 
                        'size': p['size'], 
                        'entry': p['avgPrice'],
                        'time': datetime.now().strftime('%H:%M')
                    })
    except Exception as e:
        logger.error(f"Scanner page error: {e}")
    return render_template('scanner.html', active=active, conf=settings._cache)

@app.route('/analyzer')
def analyzer_page():
    results = market_analyzer.get_results()
    conf = settings._cache
    return render_template('analyzer.html', 
                           results=results, conf=conf, 
                           progress=market_analyzer.progress,
                           status=market_analyzer.status_message,
                           is_scanning=market_analyzer.is_scanning)

@app.route('/smart_money')
def smart_money_page():
    session = db_manager.get_session()
    try:
        blocks = session.query(OrderBlock).order_by(desc(OrderBlock.created_at)).limit(50).all()
        return render_template('smart_money.html', blocks=blocks)
    except Exception as e:
        return f"Database Error: {e}"
    finally:
        session.close()

@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['telegram_enabled'] = request.form.get('telegram_enabled') == 'on'
        form_data['exit_enableStrategy'] = request.form.get('exit_enableStrategy') == 'on'
        settings.save_settings(form_data)
        return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)

@app.route('/ob_trend/settings', methods=['GET', 'POST'])
def ob_trend_settings_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        filters = ['obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter', 'obt_useOBRetest']
        for cb in filters:
            form_data[cb] = request.form.get(cb) == 'on'
        settings.save_settings(form_data)
        return redirect(url_for('ob_trend_settings_page'))
    return render_template('strategy_ob_trend.html', conf=settings._cache)

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        form_data = request.form.to_dict()
        checkboxes = ['obt_useOBRetest', 'obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter']
        for cb in checkboxes:
            if cb in form_data and form_data[cb] == 'on': form_data[cb] = True
            else: form_data[cb] = False
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
        result = bot_instance.place_order(data)
        return jsonify(result), (200 if result.get("status") in ["ok", "ignored"] else 400)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/settings/export')
def export_settings():
    json_str = json.dumps(settings.get_all(), indent=4)
    return Response(json_str, mimetype='application/json', headers={'Content-Disposition': 'attachment;filename=bot_settings.json'})

@app.route('/settings/import', methods=['POST'])
def import_settings():
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    try:
        if settings.import_settings(json.load(file)): return redirect(url_for('settings_general_page'))
    except: pass
    return "Error", 400

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)