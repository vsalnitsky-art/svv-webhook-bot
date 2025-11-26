"""
Main Application - Full V3 Layout 📊
- Верх: Активні
- Ліво: Монітор угоди (Лог з бази)
- Право: Живий сканер
"""

import os
import logging
import threading
import time
import json
import re
import decimal
import ctypes
from datetime import datetime, timedelta

# Видалення старої бази не потрібне, бо в models.py нова назва trading_bot_v3.db

from flask import Flask, request, jsonify, render_template_string
from pybit.unified_trading import HTTP
import requests

from bot_config import config
from models import db_manager
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from config import get_api_credentials

try:
    import ai_analyst
    AI_AVAILABLE = True
except: AI_AVAILABLE = False

try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def send_telegram_message(text):
    clean = text.replace("<b>", "").replace("</b>", "").replace("\n", " | ")
    logger.info(f"\n🔔 [BOT]: {clean}")

def process_signal_with_ai(data):
    symbol = data.get('symbol')
    action = data.get('action')
    ai_text = "AI OFF"
    if AI_AVAILABLE:
        try: ai_text = ai_analyst.analyze_signal(symbol, action)
        except: pass
    msg = f"ВХІД: {symbol} | {action} | {ai_text}"
    send_telegram_message(msg)
    bot.place_order(data)

def keep_alive():
    time.sleep(10)
    url = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + '/health'
    while True:
        try: requests.get(url, timeout=10)
        except: pass
        time.sleep(300)
threading.Thread(target=keep_alive, daemon=True).start()

class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)
        logger.info("✅ Bybit Connected")

    def normalize_symbol(self, symbol): return symbol.replace('.P', '')

    def get_available_balance(self, currency="USDT"):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            if b.get('retCode') != 0: return None
            for acc in b.get('result', {}).get('list', []):
                for c in acc.get('coin', []):
                    if c.get('coin') == currency: return float(c.get('walletBalance', 0))
            return None
        except: return None

    def get_all_tickers(self):
        try: return self.session.get_tickers(category="linear")['result']['list']
        except: return []

    def get_current_price(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            r = self.session.get_tickers(category="linear", symbol=norm)
            if r.get('retCode')==0: return float(r['result']['list'][0]['lastPrice'])
        except: return None

    def set_leverage(self, symbol, leverage):
        try:
            norm = self.normalize_symbol(symbol)
            self.session.set_leverage(category="linear", symbol=norm, buyLeverage=str(leverage), sellLeverage=str(leverage))
        except: pass

    def get_instrument_info(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            r = self.session.get_instruments_info(category="linear", symbol=norm)
            if r.get('retCode')==0: return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: return None, None

    def round_qty(self, qty, step):
        if step <= 0: return qty
        import decimal
        d = abs(decimal.Decimal(str(step)).as_tuple().exponent)
        return round(qty // step * step, d)

    def round_price(self, price, tick):
        if tick <= 0: return price
        import decimal
        d = abs(decimal.Decimal(str(tick)).as_tuple().exponent)
        return round(price // tick * tick, d)

    def get_position_size(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            r = self.session.get_positions(category="linear", symbol=norm)
            if r['retCode']==0: return float(r['result']['list'][0]['size'])
        except: return 0.0

    # === ВХІД ===
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            norm = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm) > 0: return {"status": "ignored"}
            
            risk = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            lev = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            
            price = self.get_current_price(norm)
            lot, tick = self.get_instrument_info(norm)
            if not price or not lot: return {"status": "error"}
            
            bal = self.get_available_balance()
            if not bal: return {"status": "error_balance"}
            
            qty = self.round_qty((bal * (risk/100) * 0.98 * lev) / price, float(lot['qtyStep']))
            min_qty = float(lot['minOrderQty'])
            if qty < min_qty:
                if bal > (min_qty*price/lev)*1.05: qty = min_qty
                else: return {"status": "error_min_qty"}
            
            self.set_leverage(norm, lev)
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            
            # Trailing
            if symbol in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]: tr_pct = 0.8
            elif any(x in symbol for x in ["SOL","XRP","ADA"]): tr_pct = 2.0
            else: tr_pct = 4.0
            
            dist = self.round_price(price * (tr_pct/100), float(tick['tickSize']))
            sl = price - dist if action == "Buy" else price + dist
            sl = self.round_price(sl, float(tick['tickSize']))
            
            self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl), trailingStop=str(dist), positionIdx=0)
            logger.info(f"✅ Opened {symbol} with Trailing {tr_pct}%")
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Order Error: {e}")
            return {"status": "error"}

    def sync_trades_from_bybit(self, days=30):
        try:
            now = datetime.now()
            for i in range(0, days, 7):
                end = now - timedelta(days=i)
                start = end - timedelta(days=min(7, days-i))
                r = self.session.get_closed_pnl(category="linear", startTime=int(start.timestamp()*1000), endTime=int(end.timestamp()*1000), limit=50)
                if r['retCode']==0:
                    for t in r['result']['list']:
                        stats_service.save_trade({
                            'order_id': t['orderId'], 'symbol': t['symbol'],
                            'side': 'Long' if t['side']=='Sell' else 'Short',
                            'qty': float(t['qty']), 'entry_price': float(t['avgEntryPrice']),
                            'exit_price': float(t['avgExitPrice']), 'pnl': float(t['closedPnl']),
                            'exit_time': datetime.fromtimestamp(int(t['updatedTime'])/1000),
                            'is_win': float(t['closedPnl'])>0,
                            'exit_reason': 'Trailing/TP'
                        })
        except: pass

    def get_pnl_stats(self, days=None, start_date=None, end_date=None):
        self.sync_trades_from_bybit(30)
        try:
            trades = stats_service.get_trades(90)
            if not trades: return None, None
            filtered = []
            s_dt, e_dt = None, None
            if start_date and end_date:
                s_dt = datetime.strptime(start_date, '%Y-%m-%d')
                e_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            elif days:
                e_dt = datetime.now()
                s_dt = e_dt - timedelta(days=days)
            for t in trades:
                if not t['exit_time']: continue
                et = datetime.strptime(t['exit_time'], '%d.%m %H:%M') if isinstance(t['exit_time'], str) else t['exit_time']
                et = et.replace(year=datetime.now().year)
                if s_dt and e_dt:
                    if s_dt <= et <= e_dt: filtered.append(t)
                else: filtered.append(t)
            stats = {"total_trades": len(filtered), "total_pnl": 0.0, "total_volume": 0.0, "win_trades": 0, "loss_trades": 0, "long_trades": 0, "short_trades": 0, "long_pnl": 0, "short_pnl": 0, "details": [], "chart_labels": [], "chart_data": [], "coin_performance": {}}
            filtered.sort(key=lambda x: x['exit_time'], reverse=False)
            run_bal = 0
            daily = {}
            for t in filtered:
                stats["total_pnl"] += t['pnl']
                run_bal += t['pnl']
                stats["total_volume"] += t.get('qty',0)*t.get('exit_price',0)
                if t['pnl']>0: stats["win_trades"]+=1
                else: stats["loss_trades"]+=1
                if t['side'] == 'Long': stats['long_trades'] += 1; stats['long_pnl'] += t['pnl']
                else: stats['short_trades'] += 1; stats['short_pnl'] += t['pnl']
                sym = t['symbol']
                if sym not in stats['coin_performance']: stats['coin_performance'][sym] = 0.0
                stats['coin_performance'][sym] += t['pnl']
                d_str = t['exit_time'].split(' ')[0]
                daily[d_str] = daily.get(d_str, 0) + t['pnl']
                stats["details"].append(t)
            rb = 0
            for d in sorted(daily.keys()):
                rb += daily[d]
                stats["chart_labels"].append(d)
                stats["chart_data"].append(round(rb, 2))
            top = sorted(stats['coin_performance'].items(), key=lambda x: x[1], reverse=True)
            stats['top_coins_labels'] = [x[0] for x in top[:5]]
            stats['top_coins_values'] = [round(x[1], 2) for x in top[:5]]
            stats["details"].sort(key=lambda x: x['exit_time'], reverse=True)
            if stats["total_trades"]>0: stats["win_rate"] = round((stats["win_trades"]/stats["total_trades"])*100,1)
            return stats, None
        except Exception as e: return None, str(e)

bot = BybitTradingBot()
scanner = EnhancedMarketScanner(bot, config.get_scanner_config())
scanner.start()

# === SMART MANAGER (Моніторинг в базу) ===
class SmartTradeManager:
    def __init__(self, bot, scanner):
        self.bot = bot
        self.scanner = scanner
        self.running = True
        self.last_log_time = 0
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.manage()
            except: pass
            time.sleep(5)
    def manage(self):
        r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']!=0: return
        
        # Записуємо статистику в базу раз на 15 сек, щоб не спамити
        should_log = (time.time() - self.last_log_time) > 15
        if should_log: self.last_log_time = time.time()

        for p in r['result']['list']:
            if float(p['size'])==0: continue
            sym = p['symbol']
            pnl = float(p['unrealisedPnl'])
            rsi = self.scanner.get_current_rsi(sym)
            press = self.scanner.get_market_pressure(sym)
            
            if should_log:
                stats_service.save_monitor_log({'symbol':sym, 'price':float(p['avgPrice']), 'pnl':pnl, 'rsi':rsi, 'pressure':press})

            # Smart Exit логіка тут (видалена, бо лише Трейлінг)
            
trade_manager = SmartTradeManager(bot, scanner)
trade_manager.start()

# === ROUTES ===
@app.route('/scanner', methods=['GET'])
def scanner_page():
    scan_data = scanner.get_aggregated_data(hours=24)
    last_update = datetime.now().strftime('%H:%M:%S')
    
    active = []
    try:
        r = bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']==0:
            for p in r['result']['list']:
                if float(p['size'])>0:
                    sym = p['symbol']
                    rsi = scan_data['snapshots'].get(sym, {}).get('rsi', 50)
                    press = scanner.get_market_pressure(sym)
                    rec, cls = "Трейлінг Активний", "table-success"
                    active.append({'symbol':sym, 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':rsi, 'pressure':round(press), 'rec':rec, 'cls':cls, 'size':p['size'], 'entry':p['avgPrice']})
    except: pass

    monitor_logs = stats_service.get_monitor_logs(limit=30)
    live_signals = scan_data['all_signals'][:30]

    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Whale Scanner</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f7f9fc;color:#333;font-family:'Roboto',sans-serif;font-size:13px; overflow-y:hidden;}
        .navbar{background:#fff;border-bottom:1px solid #e0e0e0;height:50px;}
        .container-fluid{height:calc(100vh - 60px); padding:10px; display:flex; flex-direction:column;}
        
        .top-block { height: 30%; margin-bottom:10px; overflow-y: auto; background:#fff; border:1px solid #e0e0e0; border-radius:4px;}
        .bottom-row { height: 70%; display:flex; gap:10px; }
        .half-block { width: 50%; height: 100%; overflow-y: auto; background:#fff; border:1px solid #e0e0e0; border-radius:4px; }
        
        .block-header { position:sticky; top:0; background:#fff; padding:10px; border-bottom:1px solid #eee; font-weight:700; z-index:10; display:flex; justify-content:space-between;}
        .table { margin:0; font-size:12px; }
        .table th { font-weight:500; color:#888; position:sticky; top:40px; background:#f9f9f9; }
        
        .text-up{color:#20b26c;font-weight:600} .text-down{color:#ef454a;font-weight:600}
        .badge-long{background:#e6fffa;color:#20b26c} .badge-short{background:#fff5f5;color:#ef454a}
        
        ::-webkit-scrollbar {width: 6px; height: 6px;}
        ::-webkit-scrollbar-track {background: #f1f1f1;}
        ::-webkit-scrollbar-thumb {background: #ccc; border-radius: 3px;}
    </style>
    <meta http-equiv="refresh" content="10">
    </head><body>
    
    <nav class="navbar navbar-light px-3">
        <span class="navbar-brand h6 m-0">🐋 Whale Terminal <small class="text-muted">{{ last_update }}</small></span>
        <a href="/report" class="btn btn-sm btn-outline-secondary">Звіт P&L</a>
    </nav>

    <div class="container-fluid">
        <div class="top-block">
            <div class="block-header"><span>АКТИВНІ УГОДИ</span></div>
            <table class="table table-hover">
                <thead><tr><th>Монета</th><th>Тип</th><th>Розмір</th><th>Вхід</th><th>P&L</th><th>RSI</th><th>Тиск</th></tr></thead>
                <tbody>
                {% for a in active %}
                <tr>
                    <td class="fw-bold">{{a.symbol}}</td>
                    <td><span class="badge {{ 'badge-long' if a.side=='Buy' else 'badge-short' }}">{{a.side}}</span></td>
                    <td>{{a.size}}</td><td>{{a.entry}}</td>
                    <td class="{{ 'text-up' if a.pnl>0 else 'text-down' }}">{{a.pnl}}$</td>
                    <td>{{a.rsi}}</td><td class="{{ 'text-up' if a.pressure>0 else 'text-down' }}">{{ "{:,.0f}".format(a.pressure) }}</td>
                </tr>
                {% else %}
                <tr><td colspan="7" class="text-center text-muted p-3">Немає активних угод</td></tr>
                {% endfor %}
                </tbody>
            </table>
        </div>

        <div class="bottom-row">
            <div class="half-block">
                <div class="block-header"><span class="text-primary">📊 МОНІТОРИНГ УГОДИ (БАЗА)</span></div>
                <table class="table table-striped">
                    <thead><tr><th>Час</th><th>Монета</th><th>Ціна</th><th>P&L</th><th>RSI</th><th>Тиск</th></tr></thead>
                    <tbody>
                    {% for log in logs %}
                    <tr>
                        <td class="text-muted">{{log.time}}</td><td class="fw-bold">{{log.symbol}}</td>
                        <td>{{log.price}}</td><td class="{{ 'text-up' if log.pnl>0 else 'text-down' }}">{{log.pnl}}</td>
                        <td>{{log.rsi}}</td><td>{{ "{:,.0f}".format(log.pressure) }}</td>
                    </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>

            <div class="half-block">
                <div class="block-header"><span>📡 ЖИВИЙ СКАНЕР (ВХІД)</span></div>
                <table class="table table-hover">
                    <thead><tr><th>Час</th><th>Монета</th><th>Ціна</th><th>Зміна</th><th>Об'єм</th></tr></thead>
                    <tbody>
                    {% for s in signals %}
                    <tr>
                        <td class="text-muted">{{s.time}}</td><td class="fw-bold">{{s.symbol}}</td>
                        <td>{{s.price}}</td><td class="{{ 'text-up' if s.price_change_interval>0 else 'text-down' }}">{{s.price_change_interval}}%</td>
                        <td class="fw-bold">{{ "{:,.0f}".format(s.vol_inflow) }}</td>
                    </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    </body></html>
    """
    return render_template_string(html, last_update=last_update, active=active, logs=monitor_logs, signals=live_signals)

@app.route('/report', methods=['GET'])
def report_page():
    days = int(request.args.get('days', 7))
    stats, err = bot.get_pnl_stats(days)
    bal = bot.get_available_balance() or 0.0
    if err or not stats: stats = {"total_pnl":0, "win_rate":0, "total_trades":0, "volume":0, "chart_labels":[], "chart_data":[], "details":[], "long_trades":0, "short_trades":0, "win_trades":0, "loss_trades":0, "top_coins_labels":[], "top_coins_values":[], "long_pnl":0, "short_pnl":0}

    # BYBIT STYLE REPORT (Повернуто повний код)
    html = """
    <!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><title>P&L</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root { --bg-color: #ffffff; --text-primary: #121214; --text-secondary: #858e9c; --green: #20b26c; --red: #ef454a; --btn-active-bg: #fff8d9; --btn-active-text: #cf9e04; --border: #f4f4f4; }
        body { font-family: 'Roboto', sans-serif; background-color: var(--bg-color); color: var(--text-primary); margin: 0; padding: 20px; }
        .container { max-width: 1280px; margin: 0 auto; }
        .header { display: flex; align-items: center; margin-bottom: 30px; }
        .title { font-size: 20px; font-weight: 700; margin-right: 20px; }
        .btn-group { display: flex; gap: 10px; }
        .btn { border: none; background: none; padding: 6px 12px; border-radius: 4px; font-size: 13px; cursor: pointer; color: var(--text-primary); font-weight: 500; text-decoration:none; }
        .btn:hover { background: #f5f5f5; }
        .btn.active { background-color: var(--btn-active-bg); color: var(--btn-active-text); }
        .summary-grid { display: flex; gap: 60px; margin-bottom: 30px; }
        .stat-item { display: flex; flex-direction: column; }
        .stat-label { font-size: 12px; color: var(--text-secondary); margin-bottom: 5px; text-decoration: underline dotted; cursor: help; }
        .stat-value { font-size: 28px; font-weight: 700; }
        .text-green { color: var(--green); } .text-red { color: var(--red); }
        .charts-container { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; margin-bottom: 40px; }
        .bottom-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }
        .b-stat-box { padding: 15px 0; }
        .b-stat-header { font-size: 12px; color: var(--text-secondary); margin-bottom: 10px; }
        .b-stat-val { font-size: 24px; font-weight: 700; }
        .custom-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        .custom-table th { text-align: left; color: var(--text-secondary); font-weight: 400; padding: 10px 0; border-bottom: 1px solid var(--border); }
        .custom-table td { padding: 14px 0; border-bottom: 1px solid var(--border); vertical-align: middle; }
        .badge { padding: 2px 6px; border-radius: 2px; font-size: 11px; }
        .badge-success { background: #fff8ec; color: #cf9e04; } .badge-loss { background: #f5f5f5; color: #858e9c; }
    </style>
    </head><body>
    <div class="container"><div class="header"><div class="title">P&L</div><div class="btn-group"><a href="/report?days=7" class="btn {{ 'active' if days==7 }}">7 дн.</a><a href="/report?days=30" class="btn {{ 'active' if days==30 }}">30 дн.</a><a href="/scanner" class="btn">← Сканер</a></div></div>
    <div class="summary-grid"><div class="stat-item"><div class="stat-label">Общий P&L</div><div class="stat-value {{ 'text-green' if stats.total_pnl >= 0 else 'text-red' }}">{{ "+" if stats.total_pnl > 0 }}{{ "%.2f"|format(stats.total_pnl) }} USD</div></div><div class="stat-item"><div class="stat-label">Объем</div><div class="stat-value text-green">{{ "{:,.0f}".format(stats.total_volume) }} USD</div></div></div>
    <div class="charts-container"><div class="chart-box"><div style="height: 300px;"><canvas id="pnlChart"></canvas></div></div><div class="chart-box"><div style="height: 300px;"><canvas id="rankChart"></canvas></div></div></div>
    <div class="bottom-stats"><div class="b-stat-box"><div class="b-stat-header">Всего ордеров</div><div class="b-stat-val">{{ stats.total_trades }}</div></div><div class="b-stat-box"><div class="b-stat-header">Успешных</div><div class="b-stat-val">{{ stats.win_rate }} %</div></div></div>
    <table class="custom-table"><thead><tr><th>Контракт</th><th>Тип</th><th>P&L</th><th>Результат</th><th>Время</th></tr></thead><tbody>
    {% for t in stats.details %}<tr><td style="font-weight: 500;">{{ t.symbol }}</td><td class="{{ 'text-green' if t.side == 'Long' else 'text-red' }}">{{ t.side }}</td><td class="{{ 'text-red' if t.pnl < 0 else 'text-green' }}">{{ "+" if t.pnl > 0 }}{{ "%.4f"|format(t.pnl) }}</td><td><span class="badge {{ 'badge-success' if t.pnl > 0 else 'badge-loss' }}">{{ "Успех" if t.pnl > 0 else "Убыток" }}</span></td><td style="color: var(--text-secondary);">{{ t.exit_time }}</td></tr>{% endfor %}
    </tbody></table></div>
    <script>
    const ctx = document.getElementById('pnlChart').getContext('2d'); const gradient = ctx.createLinearGradient(0, 0, 0, 300); gradient.addColorStop(0, 'rgba(239, 69, 74, 0.2)'); gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    new Chart(ctx, {type: 'line', data: {labels: {{ stats.chart_labels|tojson }}, datasets: [{data: {{ stats.chart_data|tojson }}, borderColor: '#ef454a', backgroundColor: gradient, borderWidth: 2, pointRadius: 0, fill: true}]}, options: {responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {grid: {display: false}}, y: {grid: {color: '#f4f4f4'}}}}});
    const ctxBar = document.getElementById('rankChart').getContext('2d'); new Chart(ctxBar, {type: 'bar', data: {labels: {{ stats.top_coins_labels|tojson }}, datasets: [{data: {{ stats.top_coins_values|tojson }}, backgroundColor: (ctx) => ctx.raw >= 0 ? '#20b26c' : '#ef454a', borderRadius: 2}]}, options: {indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: {legend: {display: false}}, scales: {x: {display: false}, y: {grid: {display: false}}}}});
    </script></body></html>
    """
    return render_template_string(html, stats=stats, bal=bal, days=days)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        logger.info(f"🚀 Webhook: {data.get('symbol')}")
        threading.Thread(target=process_signal_with_ai, args=(data,)).start()
        return jsonify({"status": "ok"})
    except: 
        try: 
            data = json.loads(re.search(r'\{.*\}', request.get_data(as_text=True), re.DOTALL).group())
            threading.Thread(target=process_signal_with_ai, args=(data,)).start()
            return jsonify({"status": "ok"})
        except: return jsonify({"error": "error"}), 400

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"
@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host=config.HOST, port=config.PORT)
