"""
Main App - Clean & Professional
Updated: Logs RAW payload before parsing to debug invalid JSON errors.
"""
import logging
import threading
import time
import json
import ctypes
import os
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import requests

from bot_config import config
from bot import bot_instance
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from report import render_report_page
from settings_manager import settings

# Запобігання сну у Windows
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Сканер
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

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
        except Exception as e:
            logger.error(f"Error in monitor_active loop: {e}")
        time.sleep(10)

threading.Thread(target=monitor_active, daemon=True).start()

def keep_alive():
    time.sleep(5)
    external_url = os.environ.get('RENDER_EXTERNAL_URL')
    local_url = f'http://127.0.0.1:{config.PORT}/health'
    target_url = f"{external_url}/health" if external_url else local_url
    logger.info(f"💓 Keep-alive service started. Target: {target_url}")

    while True:
        try:
            requests.get(target_url, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()

@app.route('/webhook', methods=['POST'])
def webhook():
    raw_data = ""
    try:
        # 1. Отримуємо сирий текст запиту
        raw_data = request.get_data(as_text=True)
        
        # 2. Логуємо його ВІДРАЗУ (щоб бачити навіть помилковий JSON)
        logger.info(f"📥 RAW PAYLOAD: {raw_data}")
        
        # 3. Тільки тепер пробуємо парсити
        data = json.loads(raw_data)
        
        logger.info(f"🔔 SIGNAL PARSED: {data.get('symbol')} {data.get('action')}")
        result = bot_instance.place_order(data)
        
        if result.get("status") in ["ok", "ignored"]:
            return jsonify(result)
        else:
            logger.error(f"Order action failed: {result}")
            return jsonify(result), 400
            
    except Exception as e:
        # Якщо впало - показуємо помилку і ще раз текст, який її викликав
        logger.error(f"❌ Webhook Error: {e} | Payload was: {raw_data}")
        return jsonify({"error": str(e)}), 400

@app.route('/settings', methods=['GET', 'POST'])
def settings_page():
    if request.method == 'POST':
        settings.save_settings(request.form)
        return redirect(url_for('settings_page'))
    
    conf = settings._cache
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Bot Settings</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f7f9fc; color:#333; font-size:14px;}
        .card{border:none; box-shadow:0 2px 10px rgba(0,0,0,0.05); margin-bottom: 20px;}
        .card-header{background:white; border-bottom:1px solid #eee; font-weight:bold; color:#20b26c;}
        .form-label{font-weight:500; font-size: 0.9rem;}
        .input-group-text{font-size: 0.85rem;}
        .section-title { font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 15px; margin-top: 10px; font-weight: bold; }
        .btn-save { background-color: #20b26c; color: white; border: none; padding: 10px 30px; }
        .btn-save:hover { background-color: #1a965a; color: white; }
    </style>
    </head><body>
    
    <nav class="navbar bg-white mb-4 px-3 border-bottom shadow-sm">
        <div class="container">
            <span class="navbar-brand mb-0 h1">⚙️ Bot Configuration</span>
            <div>
                <a href="/scanner" class="btn btn-sm btn-outline-secondary me-2">Scanner</a>
                <a href="/report" class="btn btn-sm btn-outline-secondary">Report</a>
            </div>
        </div>
    </nav>

    <div class="container" style="max-width: 900px;">
        <form method="POST">
            
            <div class="card">
                <div class="card-header">🎛️ Логіка та Фільтри</div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useCloudFilter" id="cloud" {{ 'checked' if conf.get('useCloudFilter') }}><label class="form-check-label" for="cloud">Cloud Filter</label></div></div>
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useObvFilter" id="obv" {{ 'checked' if conf.get('useObvFilter') }}><label class="form-check-label" for="obv">OBV Filter</label></div></div>
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useRsiFilter" id="rsi" {{ 'checked' if conf.get('useRsiFilter') }}><label class="form-check-label" for="rsi">RSI Filter</label></div></div>
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useMfiFilter" id="mfi" {{ 'checked' if conf.get('useMfiFilter') }}><label class="form-check-label" for="mfi">MFI Filter</label></div></div>
                    </div>
                    
                    <div class="row mt-2">
                        <div class="col-md-4">
                            <label class="form-label">Глобальний TF (хв)</label>
                            <select class="form-select" name="htfSelection">
                                <option value="60" {{ 'selected' if str(conf.get('htfSelection')) == '60' }}>1 Година</option>
                                <option value="240" {{ 'selected' if str(conf.get('htfSelection')) == '240' }}>4 Години</option>
                                <option value="D" {{ 'selected' if str(conf.get('htfSelection')) == 'D' }}>1 День</option>
                            </select>
                        </div>
                        <div class="col-md-4"><label class="form-label">Cloud Fast EMA</label><input type="number" class="form-control" name="cloudFastLen" value="{{ conf.get('cloudFastLen') }}"></div>
                        <div class="col-md-4"><label class="form-label">Cloud Slow EMA</label><input type="number" class="form-control" name="cloudSlowLen" value="{{ conf.get('cloudSlowLen') }}"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header" style="color: #6f42c1;">📊 RSI Indicator Settings</div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-md-12"><div class="section-title">Основні параметри</div></div>
                        <div class="col-md-4"><label class="form-label">RSI Length</label><input type="number" class="form-control" name="rsiLength" value="{{ conf.get('rsiLength') }}"></div>
                        <div class="col-md-4"><label class="form-label">MFI Length</label><input type="number" class="form-control" name="mfiLength" value="{{ conf.get('mfiLength') }}"></div>

                        <div class="col-md-12"><div class="section-title">Рівні Входу (Signal Entry)</div></div>
                        <div class="col-md-6"><label class="form-label fw-bold text-success">Buy Level (Oversold)</label><div class="input-group"><span class="input-group-text"><=</span><input type="number" class="form-control" name="entryRsiOversold" value="{{ conf.get('entryRsiOversold') }}"></div></div>
                        <div class="col-md-6"><label class="form-label fw-bold text-danger">Sell Level (Overbought)</label><div class="input-group"><span class="input-group-text">>=</span><input type="number" class="form-control" name="entryRsiOverbought" value="{{ conf.get('entryRsiOverbought') }}"></div></div>

                        <div class="col-md-12"><div class="section-title">Рівні Виходу (Exit Signal)</div></div>
                        <div class="col-md-6"><label class="form-label text-muted">Close Short Level</label><input type="number" class="form-control" name="exitRsiOversold" value="{{ conf.get('exitRsiOversold') }}"></div>
                        <div class="col-md-6"><label class="form-label text-muted">Close Long Level</label><input type="number" class="form-control" name="exitRsiOverbought" value="{{ conf.get('exitRsiOverbought') }}"></div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6"><div class="card h-100"><div class="card-header">💰 Ризик Менеджмент</div><div class="card-body">
                    <div class="mb-3"><label class="form-label">Ризик на угоду (%)</label><input type="number" step="0.1" class="form-control" name="riskPercent" value="{{ conf.get('riskPercent') }}"></div>
                    <div class="mb-3"><label class="form-label">Кредитне плече (x)</label><input type="number" class="form-control" name="leverage" value="{{ conf.get('leverage') }}"></div>
                    <div class="row"><div class="col-6"><label class="form-label">ATR SL Mult</label><input type="number" step="0.1" class="form-control" name="atrMultiplierSL" value="{{ conf.get('atrMultiplierSL') }}"></div><div class="col-6"><label class="form-label">ATR TP Mult</label><input type="number" step="0.1" class="form-control" name="atrMultiplierTP" value="{{ conf.get('atrMultiplierTP') }}"></div></div>
                </div></div></div>
                <div class="col-md-6"><div class="card h-100"><div class="card-header">🌊 Інші Налаштування</div><div class="card-body">
                    <div class="mb-3"><label class="form-label">OBV Length</label><input type="number" class="form-control" name="obvEntryLen" value="{{ conf.get('obvEntryLen') }}"></div>
                    <div class="mb-3"><label class="form-label">Swing Length (Order Blocks)</label><input type="number" class="form-control" name="swingLength" value="{{ conf.get('swingLength') }}"></div>
                    <div class="mb-3"><label class="form-label">Volume Spike Threshold</label><input type="number" step="0.1" class="form-control" name="volumeSpikeThreshold" value="{{ conf.get('volumeSpikeThreshold') }}"></div>
                </div></div></div>
            </div>

            <div class="text-center my-4">
                <button type="submit" class="btn btn-save btn-lg shadow">Зберегти налаштування</button>
            </div>
            
        </form>
    </div>
    </body></html>
    """
    return render_template_string(html, conf=conf)

@app.route('/scanner', methods=['GET'])
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    sym = p['symbol']
                    active.append({
                        'symbol': sym, 
                        'side': p['side'], 
                        'pnl': round(float(p['unrealisedPnl']), 2), 
                        'rsi': scanner.get_current_rsi(sym), 
                        'pressure': round(scanner.get_market_pressure(sym)), 
                        'size': p['size'], 
                        'entry': p['avgPrice']
                    })
    except Exception as e:
        logger.error(f"Error rendering scanner page: {e}")
    
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
            <div>
                <a href="/settings" class="btn btn-sm btn-outline-primary me-2">⚙️ Налаштування</a>
                <a href="/report" class="btn btn-sm btn-outline-secondary">Звіт P&L</a>
            </div>
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
    from report import render_report_page
    return render_report_page(bot_instance, request)

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)