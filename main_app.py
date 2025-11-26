"""
Main Application - Clean Monitor Edition 🖥️
Включає:
- UI Сканера: Тільки активні угоди на весь екран + Час відкриття
- UI Звіту: Повний Bybit Style
- Логіка: Вхід + Split TP + Trailing Stop
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

# === БАЗА ДАНИХ ===
try:
    if os.path.exists("trading_bot.db"):
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

    # === ВХІД ===
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            norm = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm) > 0: return {"status": "ignored"}
            
            risk = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            lev = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            
            json_tp_price = data.get('takeProfit')
            json_tp_percent = data.get('takeProfitPercent')
            json_sl_price = data.get('stopLoss')
            json_sl_percent = data.get('stopLossPercent')
            
            price = self.get_current_price(norm)
            lot, tick = self.get_instrument_info(norm)
            if not price or not lot: return {"status": "error"}
            
            bal = self.get_available_balance()
            if not bal: return {"status": "error_balance"}
            
            qty_step = float(lot['qtyStep'])
            tick_size = float(tick['tickSize'])
            min_qty = float(lot['minOrderQty'])
            
            raw_qty = (bal * (risk/100) * 0.98 * lev) / price
            qty = self.round_qty(raw_qty, qty_step)
            
            if qty < min_qty:
                if bal > (min_qty*price/lev)*1.05: qty = min_qty
                else: return {"status": "error_min_qty"}
            
            self.set_leverage(norm, lev)
            
            # 1. ВІДКРИТТЯ
            self.session.place_order(category="linear", symbol=norm, side=action, orderType="Market", qty=str(qty))
            logger.info(f"✅ OPENED: {action} {qty} {norm} @ {price}")
            
            # 2. STOP LOSS
            sl_target = 0.0
            if json_sl_price:
                sl_target = float(json_sl_price)
            elif json_sl_percent:
                sl_pct = float(json_sl_percent)
                sl_target = price * (1 - sl_pct/100) if action == "Buy" else price * (1 + sl_pct/100)
            else:
                sl_pct = 1.5
                sl_target = price * (1 - sl_pct/100) if action == "Buy" else price * (1 + sl_pct/100)
            
            if sl_target > 0:
                sl_rounded = self.round_price(sl_target, tick_size)
                self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(sl_rounded), positionIdx=0)
                logger.info(f"🛡️ Fixed SL Set: {sl_rounded}")

            # 3. TAKE PROFIT SPLIT
            direction = 1 if action == "Buy" else -1
            final_tp_price = 0.0
            
            if json_tp_price and float(json_tp_price) > 0:
                final_tp_price = float(json_tp_price)
            elif json_tp_percent and float(json_tp_percent) > 0:
                tp_pct = float(json_tp_percent)
                final_tp_price = price * (1 + (tp_pct/100) * direction)
            else:
                final_tp_price = price * (1 + 0.03 * direction)

            total_dist = abs(final_tp_price - price)
            
            tp1_price = self.round_price(price + (total_dist * 0.40 * direction), tick_size)
            tp2_price = self.round_price(final_tp_price, tick_size)
            
            qty1 = self.round_qty(qty * 0.5, qty_step)
            qty2 = self.round_qty(qty - qty1, qty_step)
            
            tp_side = "Sell" if action=="Buy" else "Buy"
            
            if qty1 >= min_qty:
                self.session.place_order(category="linear", symbol=norm, side=tp_side, orderType="Limit", qty=str(qty1), price=str(tp1_price), reduceOnly=True)
                logger.info(f"🎯 TP1 (40%): {tp1_price}")
            
            if qty2 >= min_qty:
                self.session.place_order(category="linear", symbol=norm, side=tp_side, orderType="Limit", qty=str(qty2), price=str(tp2_price), reduceOnly=True)
                logger.info(f"🎯 TP2 (100%): {tp2_price}")

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
                            'exit_reason': 'Manual/TP/SL'
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

# === MONITOR (Тільки логування) ===
class PassiveManager:
    def __init__(self, bot, scanner):
        self.bot = bot
        self.scanner = scanner
        self.known = set()
        self.running = True
        self.last_log = 0
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.check()
            except: pass
            time.sleep(5)
    def check(self):
        r = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']!=0: return
        curr = set()
        should_log = (time.time() - self.last_log) > 15
        if should_log: self.last_log = time.time()
        for p in r['result']['list']:
            if float(p['size'])==0: continue
            sym = p['symbol']
            curr.add(sym)
            if should_log:
                stats_service.save_monitor_log({
                    'symbol': sym, 'price': float(p['avgPrice']), 'pnl': float(p['unrealisedPnl']),
                    'rsi': self.scanner.get_current_rsi(sym), 'pressure': self.scanner.get_market_pressure(sym)
                })
        closed = self.known - curr
        for sym in closed:
            try: stats_service.delete_coin_history(sym)
            except: pass
        self.known = curr

pass_man = PassiveManager(bot, scanner)
pass_man.start()

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
                    # Додаємо час відкриття
                    open_time_str = "-"
                    try:
                        created_time = int(p.get('createdTime', 0))
                        if created_time > 0:
                            open_time_str = datetime.fromtimestamp(created_time / 1000).strftime('%d.%m %H:%M:%S')
                    except: pass
                    
                    rec, cls = "TP ACTIVE", "table-success"
                    active.append({'symbol':sym, 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':rsi, 'pressure':round(press), 'rec':rec, 'cls':cls, 'size':p['size'], 'entry':p['avgPrice'], 'open_time': open_time_str})
    except: pass

    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Active Trades</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f7f9fc;color:#333;font-family:'Roboto',sans-serif;font-size:14px;}
        .navbar{background:#fff;border-bottom:1px solid #e0e0e0;height:60px;}
        .container-fluid{padding:20px;}
        .card{background:#fff;border:1px solid #e0e0e0;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.03); height: 100%;}
        .card-header{background:#fff;border-bottom:1px solid #f0f0f0;font-weight:700;color:#555;padding:15px; display:flex; justify-content:space-between; align-items:center;}
        .table th {font-weight:500; color:#888; border-bottom:2px solid #f0f0f0;}
        .text-up{color:#20b26c;font-weight:600} .text-down{color:#ef454a;font-weight:600}
        .badge-long{background:#e6fffa;color:#20b26c} .badge-short{background:#fff5f5;color:#ef454a}
    </style>
    <meta http-equiv="refresh" content="10">
    </head><body>
    <nav class="navbar navbar-light px-4">
        <span class="navbar-brand h5 m-0">🐋 Whale Monitor <small class="text-muted ms-2">{{ last_update }}</small></span>
        <a href="/report" class="btn btn-primary btn-sm px-4">📊 Звіт P&L</a>
    </nav>
    <div class="container-fluid">
        <div class="card border-0">
            <div class="card-header">
                <span class="text-primary fs-5">📡 АКТИВНІ УГОДИ</span>
            </div>
            <div class="card-body p-0">
                <div class="table-responsive">
                    <table class="table table-hover align-middle mb-0">
                        <thead class="table-light">
                            <tr>
                                <th>Дата відкриття</th>
                                <th>Монета</th>
                                <th>Тип</th>
                                <th>Розмір</th>
                                <th>Вхід</th>
                                <th>P&L ($)</th>
                                <th>RSI</th>
                                <th>Тиск</th>
                                <th>Статус</th>
                            </tr>
                        </thead>
                        <tbody>
                        {% for a in active %}
                        <tr class="{{a.cls}}">
                            <td class="text-muted">{{ a.open_time }}</td>
                            <td class="fw-bold">{{a.symbol}}</td>
                            <td><span class="badge {{ 'badge-long' if a.side=='Buy' else 'badge-short' }}">{{a.side}}</span></td>
                            <td>{{a.size}}</td>
                            <td>{{a.entry}}</td>
                            <td class="{{ 'text-up' if a.pnl>0 else 'text-down' }} fs-6">{{a.pnl}} $</td>
                            <td>{{a.rsi}}</td>
                            <td class="{{ 'text-up' if a.pressure>0 else 'text-down' }}">{{ "{:,.0f}".format(a.pressure) }}</td>
                            <td>{{a.rec}}</td>
                        </tr>
                        {% else %}
                        <tr><td colspan="9" class="text-center text-muted py-5 fs-5">Немає активних угод 💤</td></tr>
                        {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div></body></html>
    """
    return render_template_string(html, last_update=last_update, active=active)

@app.route('/report', methods=['GET'])
def report_page():
    days = int(request.args.get('days', 7))
    stats, err = bot.get_pnl_stats(days)
    bal = bot.get_available_balance() or 0.0
    if err or not stats: stats = {"total_pnl":0, "win_rate":0, "total_trades":0, "volume":0, "chart_labels":[], "chart_data":[], "details":[], "long_trades":0, "short_trades":0, "win_trades":0, "loss_trades":0, "top_coins_labels":[], "top_coins_values":[], "long_pnl":0, "short_pnl":0}

    # ПОВНИЙ BYBIT STYLE REPORT
    html = """
    <!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8"><title>P&L Analysis</title><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet"><style>:root{--bg-color:#ffffff;--text-primary:#121214;--text-secondary:#858e9c;--green:#20b26c;--red:#ef454a;--btn-active-bg:#fff8d9;--btn-active-text:#cf9e04;--border:#f4f4f4}body{font-family:'Roboto',sans-serif;background-color:var(--bg-color);color:var(--text-primary);margin:0;padding:20px}.container{max-width:1280px;margin:0 auto}.header{display:flex;align-items:center;margin-bottom:30px}.title{font-size:20px;font-weight:700;margin-right:20px}.btn-group{display:flex;gap:10px}.btn{border:none;background:none;padding:6px 12px;border-radius:4px;font-size:13px;cursor:pointer;color:var(--text-primary);font-weight:500;text-decoration:none}.btn:hover{background:#f5f5f5}.btn.active{background-color:var(--btn-active-bg);color:var(--btn-active-text)}.summary-grid{display:flex;gap:60px;margin-bottom:30px}.stat-item{display:flex;flex-direction:column}.stat-label{font-size:12px;color:var(--text-secondary);margin-bottom:5px;text-decoration:underline dotted;cursor:help}.stat-value{font-size:28px;font-weight:700}.text-green{color:var(--green)}.text-red{color:var(--red)}.charts-container{display:grid;grid-template-columns:2fr 1fr;gap:30px;margin-bottom:40px}.bottom-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:40px}.b-stat-box{padding:15px 0}.b-stat-header{font-size:12px;color:var(--text-secondary);margin-bottom:10px}.b-stat-val{font-size:24px;font-weight:700}.custom-table{width:100%;border-collapse:collapse;font-size:12px}.custom-table th{text-align:left;color:var(--text-secondary);font-weight:400;padding:10px 0;border-bottom:1px solid var(--border)}.custom-table td{padding:14px 0;border-bottom:1px solid var(--border);vertical-align:middle}.badge{padding:2px 6px;border-radius:2px;font-size:11px}.badge-success{background:#fff8ec;color:#cf9e04}.badge-loss{background:#f5f5f5;color:#858e9c}.type-long{color:var(--green)}.type-short{color:var(--red)}</style></head><body><div class="container"><div class="header"><div class="title">P&L</div><div class="btn-group"><a href="/report?days=7" class="btn {{ 'active' if days==7 }}">7 дн.</a><a href="/report?days=30" class="btn {{ 'active' if days==30 }}">30 дн.</a><a href="/scanner" class="btn">← Сканер</a></div></div><div class="summary-grid"><div class="stat-item"><div class="stat-label">Общий P&L</div><div class="stat-value {{ 'text-green' if stats.total_pnl >= 0 else 'text-red' }}">{{ "+" if stats.total_pnl > 0 }}{{ "%.2f"|format(stats.total_pnl) }} USD</div></div><div class="stat-item"><div class="stat-label">Объем</div><div class="stat-value text-green">{{ "{:,.0f}".format(stats.total_volume) }} USD</div></div></div><div class="charts-container"><div class="chart-box"><div style="height: 300px;"><canvas id="pnlChart"></canvas></div></div><div class="chart-box"><div style="height: 300px;"><canvas id="rankChart"></canvas></div></div></div><div class="bottom-stats"><div class="b-stat-box"><div class="b-stat-header">Всего ордеров</div><div class="b-stat-val">{{ stats.total_trades }}</div></div><div class="b-stat-box"><div class="b-stat-header">Успешных</div><div class="b-stat-val">{{ stats.win_rate }} %</div></div></div><table class="custom-table"><thead><tr><th>Контракт</th><th>Тип</th><th>P&L</th><th>Результат</th><th>Время</th></tr></thead><tbody>{% for t in stats.details %}<tr><td style="font-weight: 500;">{{ t.symbol }}</td><td class="{{ 'type-long' if t.side == 'Long' else 'type-short' }}">{{ "Лонг" if t.side == 'Long' else "Шорт" }}</td><td class="{{ 'text-red' if t.pnl < 0 else 'text-green' }}">{{ "+" if t.pnl > 0 }}{{ "%.4f"|format(t.pnl) }}</td><td><span class="badge {{ 'badge-success' if t.pnl > 0 else 'badge-loss' }}">{{ "Успех" if t.pnl > 0 else "Убыток" }}</span></td><td style="color: var(--text-secondary);">{{ t.exit_time }}</td></tr>{% endfor %}</tbody></table></div><script>const ctx=document.getElementById('pnlChart').getContext('2d');const gradient=ctx.createLinearGradient(0,0,0,300);gradient.addColorStop(0,'rgba(239,69,74,0.2)');gradient.addColorStop(1,'rgba(255,255,255,0)');new Chart(ctx,{type:'line',data:{labels:{{ stats.chart_labels|tojson }},datasets:[{data:{{ stats.chart_data|tojson }},borderColor:'#ef454a',backgroundColor:gradient,borderWidth:2,pointRadius:0,fill:true}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false}},y:{grid:{color:'#f4f4f4'}}}}});const ctxBar=document.getElementById('rankChart').getContext('2d');new Chart(ctxBar,{type:'bar',data:{labels:{{ stats.top_coins_labels|tojson }},datasets:[{data:{{ stats.top_coins_values|tojson }},backgroundColor:(ctx)=>ctx.raw>=0?'#20b26c':'#ef454a',borderRadius:2}]},options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{grid:{display:false}}}}});</script></body></html>
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
