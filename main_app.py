"""
Main App - Modular Architecture
Corrected: Analyzer settings are fully visible (no accordion).
Added 'OB Retest' toggle clearly.
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
from market_analyzer import market_analyzer

try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config())
scanner.start()

def monitor_active():
    logger.info("Starting active position monitor...")
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode'] == 0:
                for p in r['result']['list']:
                    if float(p['size']) > 0:
                        stats_service.save_monitor_log({
                            'symbol': p['symbol'], 'price': float(p['avgPrice']), 'pnl': float(p['unrealisedPnl']), 
                            'rsi': scanner.get_current_rsi(p['symbol']), 'pressure': scanner.get_market_pressure(p['symbol'])
                        })
        except Exception as e: logger.error(f"Error in monitor: {e}")
        time.sleep(10)

threading.Thread(target=monitor_active, daemon=True).start()

def keep_alive():
    time.sleep(5)
    external_url = os.environ.get('RENDER_EXTERNAL_URL')
    local_url = f'http://127.0.0.1:{config.PORT}/health'
    target_url = f"{external_url}/health" if external_url else local_url
    while True:
        try: requests.get(target_url, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw_data = request.get_data(as_text=True)
        logger.info(f"📥 RAW PAYLOAD: {raw_data}")
        data = json.loads(raw_data)
        logger.info(f"🔔 SIGNAL PARSED: {data.get('symbol')} {data.get('action')}")
        result = bot_instance.place_order(data)
        if result.get("status") in ["ok", "ignored"]: return jsonify(result)
        else:
            logger.error(f"Order failed: {result}")
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"❌ Webhook Error: {e}")
        return jsonify({"error": str(e)}), 400

@app.route('/')
def home():
    html = """<!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Dashboard</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{background-color:#f8f9fa;font-family:'Segoe UI',sans-serif}.hero{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);color:white;padding:40px 0;margin-bottom:30px;border-radius:0 0 20px 20px}.card{transition:0.3s;border:none;border-radius:15px;box-shadow:0 2px 10px rgba(0,0,0,0.05);cursor:pointer;text-decoration:none;color:inherit;height:100%}.card:hover{transform:translateY(-5px);box-shadow:0 8px 20px rgba(0,0,0,0.15)}.icon{font-size:3.5rem;margin-bottom:15px}a{text-decoration:none}</style></head><body><div class="hero text-center"><div class="container"><h1>🤖 Trading Bot Hub</h1></div></div><div class="container mb-5"><div class="row g-4"><div class="col-md-6 col-lg-3"><a href="/scanner"><div class="card p-4 text-center"><div class="card-body"><div class="icon">🐋</div><h4>Монітор</h4></div></div></a></div><div class="col-md-6 col-lg-3"><a href="/analyzer"><div class="card p-4 text-center"><div class="card-body"><div class="icon">🚀</div><h4>Сканер</h4></div></div></a></div><div class="col-md-6 col-lg-3"><a href="/settings"><div class="card p-4 text-center"><div class="card-body"><div class="icon">⚙️</div><h4>Налаштування</h4></div></div></a></div><div class="col-md-6 col-lg-3"><a href="/report"><div class="card p-4 text-center"><div class="card-body"><div class="icon">📊</div><h4>Звітність</h4></div></div></a></div></div></div></body></html>"""
    return render_template_string(html)

@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['telegram_enabled'] = request.form.get('telegram_enabled') == 'on'
        settings.save_settings(form_data)
        return redirect(url_for('settings_general_page'))
    conf = settings._cache
    return render_template_string("""<!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>General Settings</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"></head><body><nav class="navbar px-3 border-bottom"><a href="/" class="btn btn-sm btn-outline-dark">🏠</a><span class="navbar-brand h1">⚙️ General</span><div><a href="/analyzer/settings" class="btn btn-sm btn-outline-secondary">Стратегія →</a></div></nav><div class="container mt-4" style="max-width:700px"><form method="POST"><div class="card mb-3"><div class="card-header fw-bold">Telegram</div><div class="card-body"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="telegram_enabled" {{ 'checked' if conf.get('telegram_enabled') }}><label>Enabled</label></div><input type="text" class="form-control mt-2" name="telegram_bot_token" value="{{ conf.get('telegram_bot_token','') }}" placeholder="Token"><input type="text" class="form-control mt-2" name="telegram_chat_id" value="{{ conf.get('telegram_chat_id','') }}" placeholder="Chat ID"></div></div><div class="card mb-3"><div class="card-header fw-bold">Scan & Risk</div><div class="card-body"><label>Currency</label><select class="form-select" name="scanner_quote_coin"><option value="USDT" {{ 'selected' if conf.get('scanner_quote_coin')=='USDT' }}>USDT</option><option value="USDC" {{ 'selected' if conf.get('scanner_quote_coin')=='USDC' }}>USDC</option></select><div class="row mt-2"><div class="col-6"><label>Risk %</label><input type="number" step="0.1" class="form-control" name="riskPercent" value="{{ conf.get('riskPercent') }}"></div><div class="col-6"><label>Leverage</label><input type="number" class="form-control" name="leverage" value="{{ conf.get('leverage') }}"></div></div></div></div><button type="submit" class="btn btn-primary w-100">Save</button></form></div></body></html>""", conf=conf)

@app.route('/analyzer/settings', methods=['GET', 'POST'])
def analyzer_settings_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        for cb in ['useCloudFilter', 'useObvFilter', 'useRsiFilter', 'useMfiFilter', 'useOBRetest']:
            form_data[cb] = request.form.get(cb) == 'on'
        settings.save_settings(form_data)
        return redirect(url_for('analyzer_settings_page'))
    
    conf = settings._cache
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Strategy</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"></head><body>
    <nav class="navbar px-3 border-bottom"><a href="/settings" class="btn btn-sm btn-outline-secondary">← Back</a><span class="navbar-brand h1">📊 Strategy</span></nav>
    <div class="container mt-4" style="max-width:800px"><form method="POST">
    <div class="card mb-3"><div class="card-header fw-bold text-success">Filters & Logic</div><div class="card-body"><div class="row">
    <div class="col-3"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="useCloudFilter" {{ 'checked' if conf.get('useCloudFilter') }}><label>Cloud</label></div></div>
    <div class="col-3"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="useObvFilter" {{ 'checked' if conf.get('useObvFilter') }}><label>OBV</label></div></div>
    <div class="col-3"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="useRsiFilter" {{ 'checked' if conf.get('useRsiFilter') }}><label>RSI</label></div></div>
    <div class="col-3"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="useOBRetest" {{ 'checked' if conf.get('useOBRetest') }}><label class="fw-bold text-primary">OB Retest</label></div></div>
    </div></div></div>
    <div class="card mb-3"><div class="card-header fw-bold">Indicators</div><div class="card-body"><div class="row g-3">
    <div class="col-4"><label>RSI Len</label><input type="number" class="form-control" name="rsiLength" value="{{ conf.get('rsiLength') }}"></div>
    <div class="col-4"><label>Cloud Fast</label><input type="number" class="form-control" name="cloudFastLen" value="{{ conf.get('cloudFastLen') }}"></div>
    <div class="col-4"><label>OB Swing</label><input type="number" class="form-control" name="swingLength" value="{{ conf.get('swingLength') }}"></div>
    </div></div></div>
    <button type="submit" class="btn btn-success w-100">Save Strategy</button></form></div></body></html>
    """
    return render_template_string(html, conf=conf)

@app.route('/analyzer')
def analyzer_page():
    results = market_analyzer.get_results()
    conf = settings._cache
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Market Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>body{background:#f0f2f5;font-size:14px} .badge-buy{background:#20b26c} .badge-sell{background:#ef454a} .progress{height:8px}</style>
    </head><body>
    
    <nav class="navbar bg-white mb-4 px-3 border-bottom"><div class="container-fluid"><span class="navbar-brand fw-bold">🚀 Analyzer</span><div><a href="/analyzer/settings" class="btn btn-sm btn-outline-primary">Strategy</a><a href="/" class="btn btn-sm btn-outline-secondary ms-2">Menu</a></div></div></nav>
    <div class="container">
        
        <div class="card mb-4 border-0 shadow-sm">
            <div class="card-body bg-white">
                <form id="scan-form">
                    <div class="row mb-3 align-items-end">
                        <div class="col-md-2">
                            <label class="small text-muted">Currency</label>
                            <select class="form-select form-select-sm" name="scanner_quote_coin"><option value="USDT" {{ 'selected' if conf.get('scanner_quote_coin')=='USDT' }}>USDT</option><option value="USDC" {{ 'selected' if conf.get('scanner_quote_coin')=='USDC' }}>USDC</option></select>
                        </div>
                        <div class="col-md-2">
                            <label class="small text-muted">Depth</label>
                            <input type="number" class="form-control form-control-sm" name="scan_limit" value="{{ conf.get('scan_limit', 100) }}">
                        </div>
                        <div class="col-md-2">
                            <label class="small text-muted">HTF</label>
                            <select class="form-select form-select-sm" name="htfSelection"><option value="60" {{ 'selected' if conf.get('htfSelection')|string=='60' }}>1H</option><option value="240" {{ 'selected' if conf.get('htfSelection')|string=='240' }}>4H</option></select>
                        </div>
                        <div class="col-md-2">
                            <label class="small text-muted">LTF</label>
                            <select class="form-select form-select-sm" name="ltfSelection"><option value="5" {{ 'selected' if conf.get('ltfSelection')|string=='5' }}>5m</option><option value="15" {{ 'selected' if conf.get('ltfSelection')|string=='15' }}>15m</option></select>
                        </div>
                        <div class="col-md-2">
                            <div class="form-check form-switch mb-1">
                                <input class="form-check-input" type="checkbox" name="useOBRetest" {{ 'checked' if conf.get('useOBRetest') }}>
                                <label class="form-check-label small fw-bold text-primary">OB Retest</label>
                            </div>
                        </div>
                        <div class="col-md-2">
                            <button type="button" id="btn-scan" class="btn btn-primary btn-sm w-100" onclick="startScan()" {{ 'disabled' if is_scanning }}>
                                {{ 'SCANNING...' if is_scanning else 'START SCAN 🚀' }}
                            </button>
                        </div>
                    </div>
                    
                    <div class="row g-2">
                        <div class="col-auto"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="useCloudFilter" {{ 'checked' if conf.get('useCloudFilter') }}><label class="small text-muted">Cloud</label></div></div>
                        <div class="col-auto"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="useObvFilter" {{ 'checked' if conf.get('useObvFilter') }}><label class="small text-muted">OBV</label></div></div>
                        <div class="col-auto"><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="useRsiFilter" {{ 'checked' if conf.get('useRsiFilter') }}><label class="small text-muted">RSI</label></div></div>
                    </div>
                </form>
            </div>
        </div>

        <div class="card mb-3 border-0 shadow-sm" style="display: {{ 'block' if is_scanning else 'none' }}" id="status-card">
            <div class="card-body py-3">
                <div class="d-flex justify-content-between mb-1">
                    <span id="status-text" class="small text-muted">{{ status }}</span>
                    <span id="status-percent" class="small fw-bold text-primary">{{ progress }}%</span>
                </div>
                <div class="progress" style="height: 6px;"><div id="progress-bar" class="progress-bar bg-primary" style="width: {{ progress }}%"></div></div>
            </div>
        </div>
        
        <div class="card border-0 shadow-sm"><div class="card-header bg-white fw-bold">Results</div><div class="table-responsive"><table class="table table-hover align-middle mb-0"><thead class="table-light"><tr><th>Pair</th><th>Price</th><th>Signal</th><th>Score</th><th>RSI(H)</th><th>RSI(L)</th><th>Details</th></tr></thead><tbody>
        {% for r in results %}<tr><td class="fw-bold">{{ r.symbol }}</td><td>{{ r.price }}</td><td><span class="badge {{ 'badge-buy' if r.signal=='Buy' else 'badge-sell' }}">{{ r.signal }}</span></td><td>{{ r.score }}</td><td>{{ r.rsi_htf }}</td><td>{{ r.rsi_ltf }}</td><td class="small text-muted">{{ r.details }}</td></tr>{% else %}<tr><td colspan="7" class="text-center py-5 text-muted">No signals found.</td></tr>{% endfor %}
        </tbody></table></div></div>
    </div>
    <script>
    function startScan() {
        $('#btn-scan').prop('disabled', true).text('Working...');
        $('#status-card').show();
        
        var formData = {};
        $('#scan-form').serializeArray().forEach(function(item) { formData[item.name] = item.value; });
        $('#scan-form input[type=checkbox]').each(function() { formData[this.name] = this.checked ? 'on' : 'off'; });
        
        $.post('/analyzer/scan', formData, function(data) { 
            pollStatus(); 
        });
    }
    function pollStatus() {
        let interval = setInterval(function() {
            $.get('/analyzer/status', function(data) {
                $('#progress-bar').css('width', data.progress + '%');
                $('#status-percent').text(data.progress + '%');
                $('#status-text').text(data.message);
                
                if (data.is_scanning) {
                    $('#btn-scan').prop('disabled', true).text('Scanning...');
                    $('#status-card').show();
                } else {
                    clearInterval(interval);
                    location.reload(); 
                }
            });
        }, 1000);
    }
    {% if is_scanning %} pollStatus(); {% endif %}
    </script></body></html>
    """
    return render_template_string(html, results=results, progress=market_analyzer.progress, status=market_analyzer.status_message, is_scanning=market_analyzer.is_scanning, conf=conf)

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        form_data = request.form.to_dict()
        if 'useOBRetest' not in form_data: form_data['useOBRetest'] = 'off'
        settings.save_settings(form_data)
    market_analyzer.run_scan_thread()
    return jsonify({"status": "started"})

@app.route('/analyzer/status')
def get_scan_status():
    return jsonify({"progress": market_analyzer.progress, "message": market_analyzer.status_message, "is_scanning": market_analyzer.is_scanning})

@app.route('/scanner', methods=['GET'])
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode'] == 0:
            for p in r['result']['list']:
                if float(p['size']) > 0:
                    active.append({'symbol': p['symbol'], 'side': p['side'], 'pnl': round(float(p['unrealisedPnl']), 2), 'rsi': scanner.get_current_rsi(p['symbol']), 'pressure': round(scanner.get_market_pressure(p['symbol'])), 'size': p['size'], 'entry': p['avgPrice']})
    except: pass
    html = """<!DOCTYPE html><html lang="uk"><head><title>Active</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"></head><body><nav class="navbar bg-white px-3 border-bottom"><a href="/" class="btn btn-sm btn-outline-dark">🏠</a><span class="navbar-brand ms-2 h1">Monitor</span></nav><div class="container mt-4"><div class="card"><div class="card-header fw-bold">Active Trades</div><table class="table"><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>RSI</th><th>PnL</th></tr></thead><tbody>{% for a in active %}<tr><td>{{a.symbol}}</td><td>{{a.side}}</td><td>{{a.entry}}</td><td>{{a.rsi}}</td><td>{{a.pnl}}</td></tr>{% else %}<tr><td colspan="5" class="text-center text-muted">No active trades</td></tr>{% endfor %}</tbody></table></div></div></body></html>"""
    return render_template_string(html, active=active)

@app.route('/report', methods=['GET'])
def report_route(): return render_report_page(bot_instance, request)

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)