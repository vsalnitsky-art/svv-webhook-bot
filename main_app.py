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

# НОВИЙ МОДУЛЬ ТРЕКІНГУ
from smart_money_tracker import tracker
from models import db_manager, OrderBlock # Для відображення в UI

try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_flask'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Запуск фонових процесів
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

# ЗАПУСК ТРЕКЕРА
tracker.start()

# --- BACKGROUND TASKS ---
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
    t = (os.environ.get('RENDER_EXTERNAL_URL') or f'http://127.0.0.1:{config.PORT}') + "/health"
    while True:
        try: requests.get(t, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=monitor_active, daemon=True).start()
threading.Thread(target=keep_alive, daemon=True).start()

# --- ROUTES ---

@app.route('/')
def home():
    try: days = int(request.args.get('days', 7))
    except: days = 7
    try: bot_instance.sync_trades(days=days)
    except: pass
    
    balance = bot_instance.get_bal()
    active_count = len(scanner.get_active_symbols())
    trades = stats_service.get_trades(days=days)
    period_pnl = 0.0; wins = 0; longs = 0; shorts = 0
    for t in trades:
        period_pnl += t['pnl']
        if t['pnl'] > 0: wins += 1
        if t['side'] == 'Long': longs += 1
        elif t['side'] == 'Short': shorts += 1
    win = int((wins/len(trades))*100) if len(trades)>0 else 0
    
    return render_template('index.html', date=datetime.utcnow().strftime('%d.%m.%Y'), 
                           balance=balance, active_count=active_count, period_pnl=period_pnl, 
                           win_rate=win, longs=longs, shorts=shorts, days=days, trades=trades[:7])

# НОВИЙ МАРШРУТ: МОНІТОР НАКОПИЧЕНИХ БЛОКІВ
@app.route('/smart_money')
def smart_money_page():
    session = db_manager.get_session()
    try:
        # Показуємо тільки PENDING (активні)
        blocks = session.query(OrderBlock).filter_by(status='PENDING').order_by(OrderBlock.created_at.desc()).all()
        return render_template('smart_money.html', blocks=blocks)
    finally:
        session.close()

@app.route('/scanner')
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']==0:
            for p in r['result']['list']:
                if float(p['size'])>0:
                    s = p['symbol']
                    active.append({
                        'symbol':s, 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 
                        'rsi':scanner.get_current_rsi(s), 'pressure':round(scanner.get_market_pressure(s)), 
                        'size':p['size'], 'entry':p['avgPrice'], 'time':datetime.now().strftime('%H:%M')
                    })
    except: pass
    return render_template('scanner.html', active=active)

@app.route('/analyzer')
def analyzer_page():
    return render_template('analyzer.html', results=market_analyzer.get_results(), conf=settings._cache, 
                           progress=market_analyzer.progress, status=market_analyzer.status_message, 
                           is_scanning=market_analyzer.is_scanning)

@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    if request.method == 'POST':
        f = request.form.to_dict()
        for k in ['telegram_enabled', 'obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter', 'obt_useOBRetest']:
            f[k] = request.form.get(k) == 'on'
        settings.save_settings(f)
        return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        f = request.form.to_dict()
        if 'useOBRetest' in f: f['obt_useOBRetest'] = f['useOBRetest']=='on'
        filters = ['obt_useCloudFilter', 'obt_useObvFilter', 'obt_useRsiFilter']
        for k in filters: 
            short = k.replace('obt_', '')
            if short in f: f[k] = f[short]=='on'
        settings.save_settings(f)
    market_analyzer.run_scan_thread()
    return jsonify({"status": "started"})

@app.route('/analyzer/status')
def get_scan_status():
    return jsonify({"progress": market_analyzer.progress, "message": market_analyzer.status_message, "is_scanning": market_analyzer.is_scanning})

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        result = bot_instance.place_order(data)
        return jsonify(result), (200 if result.get("status") in ["ok","ignored"] else 400)
    except Exception as e: return jsonify({"error": str(e)}), 400

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)