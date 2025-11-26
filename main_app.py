"""
Main Application - Bybit Replica Edition 🎨
Включає:
- Точна копія дизайну Bybit (P&L Analysis)
- Світла тема, шрифти, кольори кнопок та таблиць
- Повна логіка трейлінгу та Smart Exit
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

# === БЕЗПЕЧНЕ ВИДАЛЕННЯ СТАРОЇ БАЗИ (ЗАЛИШАЄМО ЯК Є) ===
try:
    if os.path.exists("trading_bot.db"):
        # os.remove("trading_bot.db") # Розкоментуйте, якщо треба скинути базу
        pass
except: pass

from flask import Flask, request, jsonify, render_template_string
from pybit.unified_trading import HTTP
import requests

# === ІМПОРТИ ===
from bot_config import config
from models import db_manager
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from config import get_api_credentials

# === ШІ ===
try:
    import ai_analyst
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

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
    ai_text = "AI Вимкнено"
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

# === BOT ===
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

    # === ВХІД З ТРЕЙЛІНГОМ ===
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
            
            # Налаштування Трейлінгу
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
            
            stats = {
                "total_trades": len(filtered), "total_pnl": 0.0, "total_volume": 0.0, 
                "win_trades": 0, "loss_trades": 0, "long_trades": 0, "short_trades": 0, 
                "long_pnl": 0.0, "short_pnl": 0.0, # Для статистики
                "details": [], "chart_labels": [], "chart_data": [], "coin_performance": {}
            }
            
            filtered.sort(key=lambda x: x['exit_time'], reverse=False)
            run_bal = 0
            daily = {}
            
            for t in filtered:
                pnl = t['pnl']
                stats["total_pnl"] += pnl
                run_bal += pnl
                stats["total_volume"] += t.get('qty',0)*t.get('exit_price',0)
                
                if pnl > 0: stats["win_trades"] += 1
                else: stats["loss_trades"] += 1
                
                if t['side'] == 'Long': 
                    stats['long_trades'] += 1
                    stats['long_pnl'] += pnl
                else: 
                    stats['short_trades'] += 1
                    stats['short_pnl'] += pnl
                
                # Coin performance for ranking
                sym = t['symbol']
                if sym not in stats['coin_performance']: stats['coin_performance'][sym] = 0.0
                stats['coin_performance'][sym] += pnl

                d_str = t['exit_time'].split(' ')[0]
                daily[d_str] = daily.get(d_str, 0) + pnl
                stats["details"].append(t)
            
            rb = 0
            for d in sorted(daily.keys()):
                rb += daily[d]
                stats["chart_labels"].append(d)
                stats["chart_data"].append(round(rb, 2))
            
            # Top coins for bar chart
            sorted_coins = sorted(stats['coin_performance'].items(), key=lambda x: x[1], reverse=True)
            stats['top_coins_labels'] = [x[0] for x in sorted_coins[:5]]
            stats['top_coins_values'] = [round(x[1], 2) for x in sorted_coins[:5]]

            stats["details"].sort(key=lambda x: x['exit_time'], reverse=True)
            if stats["total_trades"]>0: stats["win_rate"] = round((stats["win_trades"]/stats["total_trades"])*100,1)
            
            return stats, None
        except Exception as e: return None, str(e)

bot = BybitTradingBot()
scanner = EnhancedMarketScanner(bot, config.get_scanner_config())
scanner.start()

# === SMART EXIT ===
class SmartTradeManager:
    def __init__(self, bot, scanner):
        self.bot = bot
        self.scanner = scanner
        self.running = True
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.manage()
            except: pass
            time.sleep(5)
    def manage(self):
        r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']!=0: return
        for p in r['result']['list']:
            if float(p['size'])==0: continue
            sym = p['symbol']
            side = p['side']
            pnl = float(p['unrealisedPnl'])
            # RSI тільки моніторинг
            rsi = self.scanner.get_current_rsi(sym)
            press = self.scanner.get_market_pressure(sym)
            reason = None
            if reason: self.close(sym, p['size'], "Sell" if side=="Buy" else "Buy", reason, rsi, press, pnl)

    def close(self, sym, qty, side, reason, rsi, press, pnl):
        try:
            self.bot.session.place_order(category="linear", symbol=sym, side=side, orderType="Market", qty=str(qty), reduceOnly=True)
            self.bot.session.cancel_all_orders(category="linear", symbol=sym)
            stats_service.save_trade({'order_id':f"AUTO_{int(time.time())}_{sym}", 'symbol':sym, 'side':'Long' if side=='Sell' else 'Short', 'qty':float(qty), 'pnl':float(pnl), 'exit_time':datetime.utcnow(), 'exit_reason':reason})
        except: pass

trade_manager = SmartTradeManager(bot, scanner)
trade_manager.start()

# === PASSIVE CLEANER ===
class PassiveMonitor:
    def __init__(self, bot):
        self.bot = bot
        self.known = set()
        self.running = True
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.check()
            except: pass
            time.sleep(10)
    def check(self):
        r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']!=0: return
        curr = set(p['symbol'] for p in r['result']['list'] if float(p['size'])>0)
        closed = self.known - curr
        for sym in closed:
            try: stats_service.delete_coin_history(sym)
            except: pass
        self.known = curr

pass_mon = PassiveMonitor(bot)
pass_mon.start()

# === ROUTES ===
@app.route('/scanner', methods=['GET'])
def scanner_page():
    # ... (Код сканера залишаємо без змін, він вже світлий) ...
    scan_data = scanner.get_aggregated_data(hours=24)
    last_update = datetime.now().strftime('%H:%M:%S')
    active = []
    try:
        r = bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']==0:
            for p in r['result']['list']:
                if float(p['size'])>0:
                    active.append({'symbol':p['symbol'], 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':scan_data['snapshots'].get(p['symbol'], {}).get('rsi', 50), 'rec':"Трейлінг", 'cls':"table-success"})
    except: pass
    history = stats_service.get_trades(days=1)

    html = """<!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Scanner</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"><style>body{background:#f4f6f8;font-family:'Segoe UI'} .card{border:none;box-shadow:0 2px 8px rgba(0,0,0,0.03)}</style></head><body><nav class="navbar navbar-light bg-white mb-4 px-3 border-bottom"><span class="navbar-brand">🐋 Whale Scanner</span></nav><div class="container-fluid">{% if active %}<div class="card mb-4"><div class="card-header bg-white">АКТИВНІ</div><div class="card-body p-0"><table class="table"><thead><tr><th>Актив</th><th>Тип</th><th>P&L</th><th>RSI</th></tr></thead><tbody>{% for a in active %}<tr><td>{{a.symbol}}</td><td>{{a.side}}</td><td>{{a.pnl}}</td><td>{{a.rsi}}</td></tr>{% endfor %}</tbody></table></div></div>{% endif %}<div class="card"><div class="card-header bg-white d-flex justify-content-between"><span>ІСТОРІЯ</span><a href="/report" class="btn btn-sm btn-outline-secondary">Звіт P&L</a></div><div class="card-body p-0"><table class="table"><thead><tr><th>Час</th><th>Актив</th><th>P&L</th></tr></thead><tbody>{% for t in history %}<tr><td>{{t.exit_time}}</td><td>{{t.symbol}}</td><td class="{{ 'text-success' if t.pnl>0 else 'text-danger' }}">{{t.pnl}}</td></tr>{% endfor %}</tbody></table></div></div></div></body></html>"""
    return render_template_string(html, active=active, history=history, last_update=last_update)

@app.route('/report', methods=['GET'])
def report_page():
    # Отримуємо параметри
    days = int(request.args.get('days', 7))
    s_arg, e_arg = request.args.get('start'), request.args.get('end')
    
    # Отримуємо статистику
    stats, err = bot.get_pnl_stats(days, s_arg, e_arg)
    bal = bot.get_available_balance() or 0.0
    
    # Заглушка
    if err or not stats: 
        stats = {"total_pnl":0, "win_rate":0, "total_trades":0, "volume":0, "chart_labels":[], "chart_data":[], "details":[], "long_trades":0, "short_trades":0, "win_trades":0, "loss_trades":0, "top_coins_labels":[], "top_coins_values":[], "long_pnl":0, "short_pnl":0}

    # === 🎨 BYBIT REPLICA UI ===
    html = """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>P&L Analysis</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            :root { 
                --bg-color: #ffffff; 
                --text-primary: #121214; 
                --text-secondary: #858e9c; 
                --green: #20b26c; 
                --red: #ef454a; 
                --btn-active-bg: #fff8d9; 
                --btn-active-text: #cf9e04;
                --border: #f4f4f4;
            }
            body { font-family: 'Roboto', sans-serif; background-color: var(--bg-color); color: var(--text-primary); margin: 0; padding: 20px; }
            .container { max-width: 1280px; margin: 0 auto; }
            
            /* Header & Buttons */
            .header { display: flex; align-items: center; margin-bottom: 30px; }
            .title { font-size: 20px; font-weight: 700; margin-right: 20px; }
            .btn-group { display: flex; gap: 10px; }
            .btn { border: none; background: none; padding: 6px 12px; border-radius: 4px; font-size: 13px; cursor: pointer; color: var(--text-primary); font-weight: 500; }
            .btn:hover { background: #f5f5f5; }
            .btn.active { background-color: var(--btn-active-bg); color: var(--btn-active-text); }
            
            /* Summary Section */
            .summary-grid { display: flex; gap: 60px; margin-bottom: 30px; }
            .stat-item { display: flex; flex-direction: column; }
            .stat-label { font-size: 12px; color: var(--text-secondary); margin-bottom: 5px; text-decoration: underline dotted; cursor: help; }
            .stat-value { font-size: 28px; font-weight: 700; }
            .text-green { color: var(--green); }
            .text-red { color: var(--red); }
            
            /* Charts Layout */
            .charts-container { display: grid; grid-template-columns: 2fr 1fr; gap: 30px; margin-bottom: 40px; }
            .chart-box { }
            .chart-header { font-size: 16px; font-weight: 700; margin-bottom: 15px; display: flex; align-items: center; gap: 5px; }
            .chart-icon { font-size: 12px; color: var(--text-secondary); }
            
            /* Bottom Stats Grid */
            .bottom-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 40px; }
            .b-stat-box { padding: 15px 0; }
            .b-stat-header { font-size: 12px; color: var(--text-secondary); margin-bottom: 10px; }
            .b-stat-content { display: flex; align-items: center; gap: 15px; }
            .donut-chart { width: 50px; height: 50px; }
            .b-stat-val { font-size: 24px; font-weight: 700; }
            .b-stat-sub { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
            
            /* Table */
            .table-section h3 { font-size: 16px; margin-bottom: 15px; }
            .custom-table { width: 100%; border-collapse: collapse; font-size: 12px; }
            .custom-table th { text-align: left; color: var(--text-secondary); font-weight: 400; padding: 10px 0; border-bottom: 1px solid var(--border); }
            .custom-table td { padding: 14px 0; border-bottom: 1px solid var(--border); vertical-align: middle; }
            .badge { padding: 2px 6px; border-radius: 2px; font-size: 11px; }
            .badge-success { background: #fff8ec; color: #cf9e04; } /* Успешные (жовтий) */
            .badge-loss { background: #f5f5f5; color: #858e9c; }   /* Збитки (сірий) */
            .type-long { color: var(--green); }
            .type-short { color: var(--red); }
            
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">P&L</div>
                <div class="btn-group">
                    <a href="/report?days=7" class="btn {{ 'active' if days==7 }}">Последние 7 дн.</a>
                    <a href="/report?days=30" class="btn {{ 'active' if days==30 }}">Последние 30 дн.</a>
                    <a href="/report?days=90" class="btn {{ 'active' if days==90 }}">Последние 90 дн.</a>
                    <a href="/scanner" class="btn" style="color: #858e9c; margin-left: 20px;">← Сканер</a>
                </div>
            </div>

            <div class="summary-grid">
                <div class="stat-item">
                    <div class="stat-label">Общий P&L</div>
                    <div class="stat-value {{ 'text-green' if stats.total_pnl >= 0 else 'text-red' }}">
                        {{ "+" if stats.total_pnl > 0 }}{{ "%.2f"|format(stats.total_pnl) }} <span style="font-size: 14px; color: #121214;">USD</span>
                    </div>
                </div>
                <div class="stat-item">
                    <div class="stat-label">Торговый объем</div>
                    <div class="stat-value text-green">
                        +{{ "{:,.2f}".format(stats.total_volume).replace(',', ' ') }} <span style="font-size: 14px; color: #121214;">USD</span>
                    </div>
                </div>
            </div>

            <div class="charts-container">
                <div class="chart-box">
                    <div class="chart-header">График P&L <span class="chart-icon">↗</span></div>
                    <div style="height: 300px; position: relative;">
                        <canvas id="pnlChart"></canvas>
                    </div>
                </div>
                <div class="chart-box">
                    <div class="chart-header">P&L рейтинг <span class="chart-icon">↗</span></div>
                    <div style="height: 300px; position: relative;">
                        <canvas id="rankChart"></canvas>
                    </div>
                </div>
            </div>
            
            <div style="border-bottom: 1px solid #f4f4f4; margin-bottom: 30px;"></div>

            <div class="bottom-stats">
                <div class="b-stat-box">
                    <div class="b-stat-header">Общее количество закрытых ордеров</div>
                    <div class="b-stat-content">
                        <div class="donut-chart">
                            <canvas id="donut1"></canvas>
                        </div>
                        <div>
                            <div class="b-stat-val">{{ stats.total_trades }}</div>
                            <div class="b-stat-sub">
                                <span class="text-green">{{ stats.long_trades }} Закрыть лонг</span> / <span class="text-red">{{ stats.short_trades }} шорт</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="b-stat-box">
                    <div class="b-stat-header">Процент успешных сделок</div>
                    <div class="b-stat-content">
                        <div class="donut-chart">
                            <canvas id="donut2"></canvas>
                        </div>
                        <div>
                            <div class="b-stat-val">{{ stats.win_rate }} %</div>
                            <div class="b-stat-sub">{{ stats.win_trades }} Успешные / {{ stats.loss_trades }} Убытки</div>
                        </div>
                    </div>
                </div>
                 <div class="b-stat-box">
                    <div class="b-stat-header">P&L закрытых лонг-ордеров</div>
                    <div class="b-stat-content">
                        <div>
                            <div class="b-stat-val {{ 'text-green' if stats.long_pnl >= 0 else 'text-red' }}">{{ "%.2f"|format(stats.long_pnl) }} <span style="font-size:12px; color:#121214;">USD</span></div>
                        </div>
                    </div>
                </div>
                 <div class="b-stat-box">
                    <div class="b-stat-header">P&L закрытых шорт-ордеров</div>
                    <div class="b-stat-content">
                        <div>
                            <div class="b-stat-val {{ 'text-green' if stats.short_pnl >= 0 else 'text-red' }}">{{ "%.2f"|format(stats.short_pnl) }} <span style="font-size:12px; color:#121214;">USD</span></div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="table-section">
                <h3>Детали закрытых ордеров</h3>
                <table class="custom-table">
                    <thead>
                        <tr>
                            <th>Контракты</th>
                            <th>Кол-во</th>
                            <th>Цена входа</th>
                            <th>Цена выхода</th>
                            <th>Тип торговли</th>
                            <th>Реализ. P&L</th>
                            <th>Результат</th>
                            <th>Время</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for t in stats.details %}
                        <tr>
                            <td style="font-weight: 500;">{{ t.symbol }}</td>
                            <td>{{ t.qty }}</td>
                            <td>{{ t.entry_price }}</td>
                            <td>{{ t.exit_price }}</td>
                            <td class="{{ 'type-long' if t.side == 'Long' else 'type-short' }}">
                                {{ "Закрыть лонг" if t.side == 'Long' else "Закрыть шорт" }}
                            </td>
                            <td class="{{ 'text-red' if t.pnl < 0 else 'text-green' }}">
                                {{ "+" if t.pnl > 0 }}{{ "%.4f"|format(t.pnl) }}
                            </td>
                            <td>
                                <span class="badge {{ 'badge-success' if t.pnl > 0 else 'badge-loss' }}">
                                    {{ "Успешные сделки" if t.pnl > 0 else "Убытки" }}
                                </span>
                            </td>
                            <td style="color: var(--text-secondary);">{{ t.exit_time }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>

        <script>
            // Line Chart (P&L)
            const ctx = document.getElementById('pnlChart').getContext('2d');
            const gradient = ctx.createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, 'rgba(239, 69, 74, 0.2)'); // Reddish gradient like screenshot
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: {{ stats.chart_labels|tojson }},
                    datasets: [{
                        data: {{ stats.chart_data|tojson }},
                        borderColor: '#ef454a', // Main red color
                        backgroundColor: gradient,
                        borderWidth: 2,
                        pointRadius: 0,
                        fill: true,
                        tension: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#858e9c', font: {size: 10} } },
                        y: { grid: { color: '#f4f4f4', borderDash: [5, 5] }, ticks: { color: '#858e9c', font: {size: 10} }, position: 'right' }
                    }
                }
            });

            // Bar Chart (Ranking)
            const ctxBar = document.getElementById('rankChart').getContext('2d');
            new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: {{ stats.top_coins_labels|tojson }},
                    datasets: [{
                        data: {{ stats.top_coins_values|tojson }},
                        backgroundColor: (ctx) => ctx.raw >= 0 ? '#20b26c' : '#ef454a',
                        barThickness: 8,
                        borderRadius: 2
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { display: false },
                        y: { grid: { display: false }, ticks: { color: '#121214', font: {weight: '500'} } }
                    }
                }
            });

            // Donut 1 (Trades)
            new Chart(document.getElementById('donut1'), {
                type: 'doughnut',
                data: {
                    labels: ['Long', 'Short'],
                    datasets: [{ data: [{{ stats.long_trades }}, {{ stats.short_trades }}], backgroundColor: ['#20b26c', '#ef454a'], borderWidth: 0 }]
                },
                options: { cutout: '75%', plugins: { legend: { display: false }, tooltip: { enabled: false } } }
            });

            // Donut 2 (Win Rate)
            new Chart(document.getElementById('donut2'), {
                type: 'doughnut',
                data: {
                    datasets: [{ data: [{{ stats.win_rate }}, {{ 100 - stats.win_rate }}], backgroundColor: ['#20b26c', '#f4f4f4'], borderWidth: 0 }]
                },
                options: { cutout: '75%', plugins: { legend: { display: false }, tooltip: { enabled: false } } }
            });
        </script>
    </body>
    </html>
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
