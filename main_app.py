"""
Main App - Modular Architecture
Updated: Added Dashboard Home Page with navigation buttons.
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
from models import AnalysisResult # Додано імпорт для відображення статистики на головній (опціонально)

# Запобігання сну у Windows
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Сканер активних позицій
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

# --- WEBHOOK ---
@app.route('/webhook', methods=['POST'])
def webhook():
    raw_data = ""
    try:
        raw_data = request.get_data(as_text=True)
        logger.info(f"📥 RAW PAYLOAD: {raw_data}")
        data = json.loads(raw_data)
        logger.info(f"🔔 SIGNAL PARSED: {data.get('symbol')} {data.get('action')}")
        result = bot_instance.place_order(data)
        if result.get("status") in ["ok", "ignored"]:
            return jsonify(result)
        else:
            logger.error(f"Order action failed: {result}")
            return jsonify(result), 400
    except Exception as e:
        logger.error(f"❌ Webhook Error: {e} | Payload was: {raw_data}")
        return jsonify({"error": str(e)}), 400

# --- HOME PAGE (DASHBOARD) ---
@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Trading Bot Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            .hero-section { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 0; margin-bottom: 30px; border-radius: 0 0 20px 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
            .dashboard-card { transition: all 0.3s ease; border: none; border-radius: 15px; height: 100%; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.05); cursor: pointer; text-decoration: none; color: inherit; }
            .dashboard-card:hover { transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0,0,0,0.15); }
            .card-icon { font-size: 3.5rem; margin-bottom: 15px; }
            .card-title { font-weight: 700; color: #2d3748; }
            .card-text { color: #718096; font-size: 0.9rem; }
            .status-badge { position: absolute; top: 15px; right: 15px; background: #48bb78; color: white; padding: 5px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; }
            a { text-decoration: none; }
        </style>
    </head>
    <body>
        
        <div class="hero-section text-center">
            <div class="container">
                <h1 class="display-5 fw-bold">🤖 Trading Bot Hub</h1>
                <p class="lead">Центр керування вашою торговою системою</p>
            </div>
        </div>

        <div class="container mb-5">
            <div class="row g-4">
                
                <div class="col-md-6 col-lg-3">
                    <a href="/scanner">
                        <div class="card dashboard-card p-4 text-center">
                            <div class="status-badge">Live</div>
                            <div class="card-body">
                                <div class="card-icon">🐋</div>
                                <h4 class="card-title">Монітор</h4>
                                <p class="card-text">Відслідковування активних угод, P&L та RSI в реальному часі.</p>
                            </div>
                        </div>
                    </a>
                </div>

                <div class="col-md-6 col-lg-3">
                    <a href="/analyzer">
                        <div class="card dashboard-card p-4 text-center">
                            <div class="card-body">
                                <div class="card-icon">🚀</div>
                                <h4 class="card-title">Сканер Ринку</h4>
                                <p class="card-text">Пошук нових сигналів (Order Blocks) та аналіз топ-монет.</p>
                            </div>
                        </div>
                    </a>
                </div>

                <div class="col-md-6 col-lg-3">
                    <a href="/settings">
                        <div class="card dashboard-card p-4 text-center">
                            <div class="card-body">
                                <div class="card-icon">⚙️</div>
                                <h4 class="card-title">Налаштування</h4>
                                <p class="card-text">Конфігурація стратегії, індикаторів та ризик-менеджменту.</p>
                            </div>
                        </div>
                    </a>
                </div>

                <div class="col-md-6 col-lg-3">
                    <a href="/report">
                        <div class="card dashboard-card p-4 text-center">
                            <div class="card-body">
                                <div class="card-icon">📊</div>
                                <h4 class="card-title">Звітність</h4>
                                <p class="card-text">Історія торгів, аналіз прибутковості та статистика.</p>
                            </div>
                        </div>
                    </a>
                </div>

            </div>
            
            <div class="text-center mt-5 text-muted small">
                <p>System Status: <strong>Operational</strong> | Server Time: {{ time }}</p>
            </div>
        </div>

    </body>
    </html>
    """
    return render_template_string(html, time=datetime.utcnow().strftime('%H:%M:%S UTC'))

# --- GENERAL SETTINGS ---
@app.route('/settings', methods=['GET', 'POST'])
def settings_general_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        form_data['telegram_enabled'] = request.form.get('telegram_enabled') == 'on'
        settings.save_settings(form_data)
        return redirect(url_for('settings_general_page'))
    
    conf = settings._cache
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>General Settings</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>body{background:#f7f9fc;font-size:14px} .card{margin-bottom:20px; border:none; box-shadow:0 2px 10px rgba(0,0,0,0.05)} .navbar{background:white;}</style>
    </head><body>
    <nav class="navbar mb-4 px-3 border-bottom shadow-sm">
        <div class="container-fluid"><a href="/" class="btn btn-sm btn-outline-dark me-2">🏠 Головна</a><span class="navbar-brand mb-0 h1">⚙️ Загальні Налаштування</span>
        <div><a href="/scanner" class="btn btn-sm btn-outline-secondary">Монітор</a><a href="/analyzer" class="btn btn-sm btn-outline-primary">Аналізатор</a></div></div>
    </nav>
    <div class="container" style="max-width: 700px;">
        <form method="POST">
            <div class="card">
                <div class="card-header bg-white fw-bold text-primary">🌍 Параметри Аналізатора</div>
                <div class="card-body">
                    <div class="mb-3">
                        <label class="form-label">Валюта торгівлі (Quote Coin)</label>
                        <select class="form-select" name="scanner_quote_coin">
                            <option value="USDT" {{ 'selected' if conf.get('scanner_quote_coin') == 'USDT' }}>USDT</option>
                            <option value="USDC" {{ 'selected' if conf.get('scanner_quote_coin') == 'USDC' }}>USDC</option>
                        </select>
                        <div class="form-text">Сканер буде шукати тільки пари з цією валютою.</div>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Режим Сканера</label>
                        <select class="form-select" name="scanner_mode">
                            <option value="Manual" {{ 'selected' if conf.get('scanner_mode') == 'Manual' }}>Manual (Тільки кнопка)</option>
                            <option value="Auto" {{ 'selected' if conf.get('scanner_mode') == 'Auto' }}>Auto (Кожні 15 хв)</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Глибина сканування (Top N)</label>
                        <input type="number" class="form-control" name="scan_limit" value="{{ conf.get('scan_limit', 100) }}">
                        <div class="form-text">Макс. 200 монет за раз (щоб не перевищити ліміти API).</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header bg-white fw-bold text-info">✈️ Telegram Інтеграція</div>
                <div class="card-body">
                    <div class="form-check form-switch mb-3">
                        <input class="form-check-input" type="checkbox" name="telegram_enabled" id="tg_on" {{ 'checked' if conf.get('telegram_enabled') }}>
                        <label class="form-check-label" for="tg_on">Увімкнути сповіщення</label>
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Bot Token</label>
                        <input type="text" class="form-control" name="telegram_bot_token" value="{{ conf.get('telegram_bot_token', '') }}" placeholder="123456:ABC...">
                    </div>
                    <div class="mb-3">
                        <label class="form-label">Chat ID</label>
                        <input type="text" class="form-control" name="telegram_chat_id" value="{{ conf.get('telegram_chat_id', '') }}">
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header bg-white fw-bold text-danger">💰 Ризик Менеджмент (Глобальний)</div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-6"><label class="form-label">Ризик на угоду (%)</label><input type="number" step="0.1" class="form-control" name="riskPercent" value="{{ conf.get('riskPercent') }}"></div>
                        <div class="col-6"><label class="form-label">Плече (x)</label><input type="number" class="form-control" name="leverage" value="{{ conf.get('leverage') }}"></div>
                    </div>
                </div>
            </div>
            
            <div class="text-center pb-5">
                <button type="submit" class="btn btn-primary btn-lg shadow">Зберегти Загальні</button>
                <a href="/analyzer/settings" class="btn btn-outline-secondary btn-lg ms-2">Налаштування Стратегії →</a>
            </div>
        </form>
    </div></body></html>
    """
    return render_template_string(html, conf=conf)

# --- ANALYZER SETTINGS ---
@app.route('/analyzer/settings', methods=['GET', 'POST'])
def analyzer_settings_page():
    if request.method == 'POST':
        form_data = request.form.to_dict()
        checkboxes = ['useCloudFilter', 'useObvFilter', 'useRsiFilter', 'useMfiFilter']
        for cb in checkboxes:
            form_data[cb] = request.form.get(cb) == 'on'
        settings.save_settings(form_data)
        return redirect(url_for('analyzer_settings_page'))
    
    conf = settings._cache
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Strategy Settings</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>body{background:#f7f9fc;font-size:14px} .card{margin-bottom:20px; border:none; box-shadow:0 2px 10px rgba(0,0,0,0.05)}</style>
    </head><body>
    <nav class="navbar bg-white mb-4 px-3 border-bottom shadow-sm">
        <div class="container-fluid"><a href="/" class="btn btn-sm btn-outline-dark me-2">🏠</a><span class="navbar-brand mb-0 h1">📊 Налаштування Стратегії</span>
        <div><a href="/settings" class="btn btn-sm btn-outline-secondary">← Назад</a></div></div>
    </nav>
    <div class="container" style="max-width: 900px;">
        <form method="POST">
            <div class="card">
                <div class="card-header fw-bold text-success">🎛️ Логіка та Фільтри</div>
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useCloudFilter" {{ 'checked' if conf.get('useCloudFilter') }}><label class="form-check-label">Cloud Filter</label></div></div>
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useObvFilter" {{ 'checked' if conf.get('useObvFilter') }}><label class="form-check-label">OBV Filter</label></div></div>
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useRsiFilter" {{ 'checked' if conf.get('useRsiFilter') }}><label class="form-check-label">RSI Filter</label></div></div>
                        <div class="col-md-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="useMfiFilter" {{ 'checked' if conf.get('useMfiFilter') }}><label class="form-check-label">MFI Filter</label></div></div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-6"><label class="form-label">Глобальний TF (HTF)</label><select class="form-select" name="htfSelection">
                            <option value="60" {{ 'selected' if conf.get('htfSelection')|string == '60' }}>1 Година</option>
                            <option value="240" {{ 'selected' if conf.get('htfSelection')|string == '240' }}>4 Години</option>
                            <option value="D" {{ 'selected' if conf.get('htfSelection')|string == 'D' }}>1 День</option>
                        </select></div>
                        <div class="col-md-6"><label class="form-label">Вхід TF (LTF)</label><select class="form-select" name="ltfSelection">
                            <option value="5" {{ 'selected' if conf.get('ltfSelection')|string == '5' }}>5 Хвилин</option>
                            <option value="15" {{ 'selected' if conf.get('ltfSelection')|string == '15' }}>15 Хвилин</option>
                            <option value="60" {{ 'selected' if conf.get('ltfSelection')|string == '60' }}>1 Година</option>
                        </select></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-header fw-bold" style="color: #6f42c1;">📈 Параметри Індикаторів</div>
                <div class="card-body">
                    <div class="row g-3">
                        <div class="col-md-4"><label class="form-label">RSI Length</label><input type="number" class="form-control" name="rsiLength" value="{{ conf.get('rsiLength') }}"></div>
                        <div class="col-md-4"><label class="form-label">RSI Buy (<=)</label><input type="number" class="form-control" name="entryRsiOversold" value="{{ conf.get('entryRsiOversold') }}"></div>
                        <div class="col-md-4"><label class="form-label">RSI Sell (>=)</label><input type="number" class="form-control" name="entryRsiOverbought" value="{{ conf.get('entryRsiOverbought') }}"></div>
                        
                        <div class="col-md-4"><label class="form-label">Cloud Fast</label><input type="number" class="form-control" name="cloudFastLen" value="{{ conf.get('cloudFastLen') }}"></div>
                        <div class="col-md-4"><label class="form-label">Cloud Slow</label><input type="number" class="form-control" name="cloudSlowLen" value="{{ conf.get('cloudSlowLen') }}"></div>
                        <div class="col-md-4"><label class="form-label">Swing Length (OB)</label><input type="number" class="form-control" name="swingLength" value="{{ conf.get('swingLength') }}"></div>
                    </div>
                </div>
            </div>
            
            <div class="text-center pb-5"><button type="submit" class="btn btn-success btn-lg shadow">Зберегти Стратегію</button></div>
        </form>
    </div></body></html>
    """
    return render_template_string(html, conf=conf)

# --- MARKET ANALYZER DASHBOARD ---
@app.route('/analyzer')
def analyzer_page():
    results = market_analyzer.get_results()
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Market Analyzer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body{background:#f0f2f5; font-size:14px;}
        .score-high{color:#20b26c;font-weight:bold}
        .badge-buy{background:#20b26c} .badge-sell{background:#ef454a}
        .progress{height: 8px; border-radius: 4px;}
    </style>
    </head><body>
    
    <nav class="navbar bg-white mb-4 px-3 border-bottom">
        <div class="container-fluid"><a href="/" class="btn btn-sm btn-outline-dark me-2">🏠</a><span class="navbar-brand fw-bold">🚀 Market Analyzer</span>
        <div><a href="/settings" class="btn btn-sm btn-outline-secondary">Налаштування</a></div></div>
    </nav>

    <div class="container">
        <div class="card mb-4 border-0 shadow-sm">
            <div class="card-body text-center py-4">
                <h5 id="status-text" class="text-muted mb-3">{{ status }}</h5>
                <div class="progress mb-4 w-75 mx-auto">
                    <div id="progress-bar" class="progress-bar bg-primary" style="width: {{ progress }}%"></div>
                </div>
                <button id="btn-scan" class="btn btn-primary btn-lg px-5 shadow-sm" onclick="startScan()" {{ 'disabled' if is_scanning }}>
                    {{ 'SCANNING...' if is_scanning else 'START SCAN 🔎' }}
                </button>
            </div>
        </div>

        <div class="card border-0 shadow-sm">
            <div class="card-header bg-white fw-bold py-3">ЗНАЙДЕНІ МОЖЛИВОСТІ</div>
            <div class="table-responsive">
                <table class="table table-hover align-middle mb-0">
                    <thead class="table-light">
                        <tr><th>Pair</th><th>Price</th><th>Signal</th><th>Score</th><th>RSI (4H)</th><th>RSI (15m)</th><th>Time</th><th>Details</th></tr>
                    </thead>
                    <tbody>
                    {% for r in results %}
                        <tr>
                            <td class="fw-bold">{{ r.symbol }}</td>
                            <td>{{ r.price }}</td>
                            <td><span class="badge {{ 'badge-buy' if r.signal=='Buy' else 'badge-sell' }}">{{ r.signal }}</span></td>
                            <td class="score-high">{{ r.score }}</td>
                            <td>{{ r.rsi_htf }}</td>
                            <td>{{ r.rsi_ltf }}</td>
                            <td class="text-muted">{{ r.time }}</td>
                            <td class="small text-muted">{{ r.details }}</td>
                        </tr>
                    {% else %}
                        <tr><td colspan="8" class="text-center py-5 text-muted">Сигналів не знайдено. Натисніть Start Scan!</td></tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        function startScan() {
            $('#btn-scan').prop('disabled', true).text('Starting...');
            $.post('/analyzer/scan', function(data) {
                pollStatus();
            });
        }

        function pollStatus() {
            let interval = setInterval(function() {
                $.get('/analyzer/status', function(data) {
                    $('#progress-bar').css('width', data.progress + '%');
                    $('#status-text').text(data.message);
                    if (data.is_scanning) {
                        $('#btn-scan').text('Scanning (' + data.progress + '%)...').prop('disabled', true);
                    } else {
                        clearInterval(interval);
                        location.reload(); 
                    }
                });
            }, 1000);
        }
        
        {% if is_scanning %} pollStatus(); {% endif %}
    </script>
    </body></html>
    """
    return render_template_string(html, 
                                  results=results, 
                                  progress=market_analyzer.progress, 
                                  status=market_analyzer.status_message,
                                  is_scanning=market_analyzer.is_scanning)

@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    market_analyzer.run_scan_thread()
    return jsonify({"status": "started"})

@app.route('/analyzer/status')
def get_scan_status():
    return jsonify({
        "progress": market_analyzer.progress,
        "message": market_analyzer.status_message,
        "is_scanning": market_analyzer.is_scanning
    })

# --- ACTIVE SCANNER PAGE (Main) ---
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
        logger.error(f"Scanner Page Error: {e}")
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Active Trades</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>body{background:#f7f9fc;font-size:14px} .text-up{color:#20b26c} .text-down{color:#ef454a} .badge-buy{background-color:#20b26c} .badge-sell{background-color:#ef454a}</style>
    <meta http-equiv="refresh" content="5"></head><body>
    <nav class="navbar bg-white mb-4 px-3 border-bottom shadow-sm">
        <div class="container-fluid"><a href="/" class="btn btn-sm btn-outline-dark me-2">🏠</a><span class="navbar-brand mb-0 h1">🐋 Active Monitor</span>
        <div><a href="/settings" class="btn btn-sm btn-outline-primary me-2">Налаштування</a><a href="/report" class="btn btn-sm btn-outline-secondary">Звіт</a></div></div>
    </nav>
    <div class="container"><div class="card"><div class="card-header bg-white fw-bold py-3">ВІДКРИТІ ПОЗИЦІЇ</div>
            <div class="table-responsive"><table class="table table-hover align-middle mb-0"><thead class="table-light"><tr><th>Монета</th><th>Тип</th><th>Розмір</th><th>Вхід</th><th>RSI</th><th>Тиск</th><th>P&L</th></tr></thead><tbody>
    {% for a in active %}<tr><td class="fw-bold">{{a.symbol}}</td><td><span class="badge {{ 'badge-buy' if a.side=='Buy' else 'badge-sell' }}">{{a.side}}</span></td><td>{{a.size}}</td><td>{{a.entry}}</td><td><span class="{{ 'text-danger' if a.rsi > 70 else 'text-success' if a.rsi < 30 else '' }}">{{a.rsi}}</span></td><td>{{a.pressure}}</td><td class="{{ 'text-up' if a.pnl>0 else 'text-down' }}">{{ "+" if a.pnl > 0 }}{{a.pnl}}$</td></tr>{% else %}<tr><td colspan="7" class="text-center text-muted py-5">Немає активних угод 💤</td></tr>{% endfor %}
    </tbody></table></div></div></div></body></html>
    """
    return render_template_string(html, active=active)

@app.route('/report', methods=['GET'])
def report_route():
    from report import render_report_page
    return render_report_page(bot_instance, request)

@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)