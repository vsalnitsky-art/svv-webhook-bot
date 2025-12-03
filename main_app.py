import logging, threading, time, json, ctypes, os, requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from sqlalchemy import desc
from bot_config import config; from bot import bot_instance; from statistics_service import stats_service
from scanner import EnhancedMarketScanner; from settings_manager import settings; 
# Додано SmartMoneyTicker до імпортів для уникнення помилок
from models import db_manager, OrderBlock, SmartMoneyTicker 
from market_analyzer import market_analyzer
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002|0x00000001)
except: pass
app = Flask(__name__); app.secret_key='secret'; logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s'); logger = logging.getLogger(__name__)
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config()); scanner.start()
def monitor_active():
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode']==0:
                for p in r['result']['list']:
                    if float(p['size'])>0: stats_service.save_monitor_log({'symbol':p['symbol'], 'price':float(p['avgPrice']), 'pnl':float(p['unrealisedPnl']), 'rsi':scanner.get_current_rsi(p['symbol']), 'pressure':scanner.get_market_pressure(p['symbol'])})
        except: pass
        time.sleep(10)
def keep_alive():
    t = (os.environ.get('RENDER_EXTERNAL_URL') or f'http://127.0.0.1:{config.PORT}') + "/health"
    while True:
        try: requests.get(t, timeout=10)
        except: pass
        time.sleep(300)
threading.Thread(target=monitor_active, daemon=True).start(); threading.Thread(target=keep_alive, daemon=True).start()

# --- TABLE VIEW ROUTE ---
@app.route('/smart_money')
def smart_money_page(): 
    s=db_manager.get_session()
    # Показуємо АКТИВНІ (або очікуючі) зони, відсортовані за статусом
    blocks = s.query(OrderBlock).filter(OrderBlock.status != 'BROKEN').order_by(desc(OrderBlock.created_at)).all()
    s.close()
    return render_template('smart_money.html', blocks=blocks)

@app.route('/')
def home():
    d = int(request.args.get('days',7)); bot_instance.sync_trades(d)
    tr = stats_service.get_trades(d); pnl = sum(t['pnl'] for t in tr)
    return render_template('index.html', date=datetime.utcnow().strftime('%d %b %Y'), balance=bot_instance.get_bal(), active_count=len(scanner.get_active_symbols()), period_pnl=pnl, longs=sum(1 for t in tr if t['side']=='Long'), shorts=sum(1 for t in tr if t['side']=='Short'), days=d, trades=tr[:10])
@app.route('/scanner')
def scanner_page():
    act = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']==0:
            for p in r['result']['list']:
                if float(p['size'])>0:
                    d = scanner.get_coin_data(p['symbol'])
                    act.append({'symbol':p['symbol'], 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':d.get('rsi',0), 'exit_status':d.get('exit_status','Safe'), 'exit_details':d.get('exit_details','-'), 'size':p['size'], 'entry':p['avgPrice'], 'time':datetime.now().strftime('%H:%M')})
    except: pass
    return render_template('scanner.html', active=act, conf=settings._cache)
@app.route('/analyzer')
def analyzer_page(): return render_template('analyzer.html', results=market_analyzer.get_results(), conf=settings._cache, progress=market_analyzer.progress, status=market_analyzer.status_message, is_scanning=market_analyzer.is_scanning)
@app.route('/settings', methods=['GET','POST'])
def settings_general_page():
    if request.method=='POST': 
        f=request.form.to_dict(); f['telegram_enabled']=f.get('telegram_enabled')=='on'; f['exit_enableStrategy']=f.get('exit_enableStrategy')=='on'
        settings.save_settings(f); return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)
@app.route('/ob_trend/settings', methods=['GET','POST'])
def ob_trend_settings_page():
    if request.method=='POST':
        f=request.form.to_dict()
        for c in ['obt_useCloudFilter','obt_useObvFilter','obt_useRsiFilter','obt_useOBRetest']: f[c]=f.get(c)=='on'
        settings.save_settings(f); return redirect(url_for('ob_trend_settings_page'))
    return render_template('strategy_ob_trend.html', conf=settings._cache)
@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    f=request.form.to_dict()
    for c in ['obt_useOBRetest','obt_useCloudFilter','obt_useObvFilter','obt_useRsiFilter']: f[c]=f.get(c)=='on'
    settings.save_settings(f); market_analyzer.run_scan_thread(); return jsonify({"status":"started"})
@app.route('/analyzer/status')
def get_scan_status(): return jsonify({"progress":market_analyzer.progress, "message":market_analyzer.status_message, "is_scanning":market_analyzer.is_scanning})
@app.route('/webhook', methods=['POST'])
def webhook(): d=json.loads(request.get_data(as_text=True)); r=bot_instance.place_order(d); return jsonify(r), 200
@app.route('/settings/export')
def export_settings(): return Response(json.dumps(settings.get_all(), indent=4), mimetype='application/json', headers={'Content-Disposition':'attachment;filename=bot_settings.json'})
@app.route('/settings/import', methods=['POST'])
def import_settings():
    f = request.files['file']
    if f: settings.import_settings(json.load(f))
    return redirect(url_for('settings_general_page'))
@app.route('/health')
def health(): return jsonify({"status":"ok"})
if __name__ == '__main__': app.run(host=config.HOST, port=config.PORT)
