"""
Main Application - Clean UI Edition ✨
Включає:
- Трейлінг-стоп при вході
- Smart Exit (RSI + Pressure)
- Інтерфейс: Тільки Активні Угоди та Історія
- Світла тема
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

# === БЕЗПЕЧНЕ ВИДАЛЕННЯ СТАРОЇ БАЗИ (Щоб оновити структуру) ===
try:
    if os.path.exists("trading_bot.db"):
        # Перевіряємо розмір, якщо 0 або помилка структури - видаляємо
        # Тут просто лишаємо видалення для чистоти експерименту, як ви просили раніше
        os.remove("trading_bot.db")
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
    logger.info(f"\n{'='*60}\n🔔 [BOT]: {clean}\n{'='*60}")

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
                            'is_win': float(t['closedPnl'])>0
                        })
        except: pass

    def get_pnl_stats(self, days=None, start_date=None, end_date=None):
        self.sync_trades_from_bybit(30)
        try:
            trades = stats_service.get_trades(90)
            if not trades: return {"total_pnl":0, "chart_labels":[], "chart_data":[], "history":[]}, None
            
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
                # Приблизна перевірка року (бо у форматі %d.%m немає року, припускаємо поточний для сортування)
                et = et.replace(year=datetime.now().year)
                
                if s_dt and e_dt:
                    if s_dt <= et <= e_dt: filtered.append(t)
                else: filtered.append(t)
            
            stats = {"total_trades": len(filtered), "total_pnl": 0.0, "total_volume": 0.0, "win_trades":0, "loss_trades":0, "details": [], "chart_labels":[], "chart_data":[]}
            
            filtered.sort(key=lambda x: x['exit_time'], reverse=False)
            run_bal = 0
            daily = {}
            
            for t in filtered:
                stats["total_pnl"] += t['pnl']
                run_bal += t['pnl']
                stats["total_volume"] += t.get('qty',0)*t.get('exit_price',0)
                if t['pnl']>0: stats["win_trades"]+=1
                else: stats["loss_trades"]+=1
                
                d_str = t['exit_time'].split(' ')[0]
                daily[d_str] = daily.get(d_str, 0) + t['pnl']
                stats["details"].append(t)
            
            rb = 0
            for d in sorted(daily.keys()):
                rb += daily[d]
                stats["chart_labels"].append(d)
                stats["chart_data"].append(round(rb, 2))
                
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
            
            # Cooldown
            if time.time() - int(p['updatedTime'])/1000 < 60: continue
            
            rsi = self.scanner.get_current_rsi(sym)
            press = self.scanner.get_market_pressure(sym)
            
            reason = None
            if side=="Buy":
                if rsi>=80 and pnl>0: reason = f"RSI High ({rsi})"
                elif press < -100000: reason = "Volume Dump"
            elif side=="Sell":
                if rsi<=20 and pnl>0: reason = f"RSI Low ({rsi})"
                elif press > 100000: reason = "Volume Pump"
            
            if reason: self.close(sym, p['size'], "Sell" if side=="Buy" else "Buy", reason, rsi, press, pnl)

    def close(self, sym, qty, side, reason, rsi, press, pnl):
        try:
            self.bot.session.place_order(category="linear", symbol=sym, side=side, orderType="Market", qty=str(qty), reduceOnly=True)
            self.bot.session.cancel_all_orders(category="linear", symbol=sym)
            send_telegram_message(f"AUTO-CLOSE: {sym} | {reason}")
            stats_service.save_trade({
                'order_id': f"AUTO_{int(time.time())}_{sym}", 'symbol': sym,
                'side': 'Long' if side=='Sell' else 'Short', 'qty': float(qty),
                'pnl': float(pnl), 'exit_time': datetime.utcnow(),
                'exit_reason': reason, 'exit_rsi': rsi, 'exit_pressure': press
            })
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
            try: stats_service.delete_coin_history(sym) # Очищаємо старі сигнали сканера
            except: pass
        self.known = curr

pass_mon = PassiveMonitor(bot)
pass_mon.start()

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
                    rec, cls = "ТРИМАТИ", ""
                    if p['side']=="Buy":
                        if rsi>75: rec, cls = "RSI HIGH", "table-danger"
                    elif p['side']=="Sell":
                        if rsi<25: rec, cls = "RSI LOW", "table-danger"
                    active.append({'symbol':sym, 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':rsi, 'pressure':round(press), 'rec':rec, 'cls':cls})
    except: pass

    history = stats_service.get_trades(days=1)

    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>Whale Terminal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body{background:#f4f6f8;color:#333;font-family:'Segoe UI',sans-serif;font-size:14px}
        .navbar{background:#fff;border-bottom:1px solid #e1e4e8}
        .card{background:#fff;border:1px solid #e1e4e8;box-shadow:0 2px 5px rgba(0,0,0,0.02);margin-bottom:20px;border-radius:8px}
        .card-header{background:#fff;border-bottom:1px solid #f0f0f0;font-weight:700;color:#555;padding:15px}
        .table{margin-bottom:0} .text-up{color:#00b894;font-weight:600} .text-down{color:#d63031;font-weight:600}
    </style><meta http-equiv="refresh" content="30"></head><body>
    <nav class="navbar navbar-light mb-4 px-3"><span class="navbar-brand h1">🐋 Whale Scanner <span class="badge bg-light text-dark border">CLEAN</span></span><span class="text-muted small">{{ last_update }}</span></nav>
    <div class="container-fluid">
        {% if active %}
        <div class="card border-primary"><div class="card-header text-primary bg-light">АКТИВНІ УГОДИ</div><div class="card-body p-0"><table class="table table-hover"><thead><tr><th>Монета</th><th>Тип</th><th>P&L</th><th>RSI</th><th>Тиск</th><th>Статус</th></tr></thead><tbody>
        {% for a in active %}<tr class="{{a.cls}}"><td class="fw-bold">{{a.symbol}}</td><td>{{a.side}}</td><td class="{{ 'text-up' if a.pnl>0 else 'text-down' }}">{{a.pnl}}$</td><td>{{a.rsi}}</td><td>{{a.pressure}}</td><td>{{a.rec}}</td></tr>{% endfor %}
        </tbody></table></div></div>{% endif %}
        
        <div class="card"><div class="card-header d-flex justify-content-between"><span>ІСТОРІЯ ЗАКРИТИХ УГОД (24Г)</span><a href="/report" class="btn btn-sm btn-outline-secondary">Звіт</a></div><div class="card-body p-0"><table class="table table-hover"><thead><tr><th>Час</th><th>Монета</th><th>Тип</th><th>P&L</th><th>Причина</th><th>RSI</th></tr></thead><tbody>
        {% for t in history %}<tr><td class="text-muted">{{t.exit_time}}</td><td class="fw-bold">{{t.symbol}}</td><td>{{t.side}}</td><td class="{{ 'text-up' if t.pnl>0 else 'text-down' }}">{{t.pnl}}$</td><td>{{t.exit_reason or 'Trailing/TP'}}</td><td>{{t.exit_rsi or '-'}}</td></tr>{% endfor %}
        {% if not history %}<tr><td colspan="6" class="text-center text-muted p-3">Історія порожня</td></tr>{% endif %}
        </tbody></table></div></div>
    </div></body></html>
    """
    return render_template_string(html, last_update=last_update, active=active, history=history)

@app.route('/report', methods=['GET'])
def report_page():
    days = int(request.args.get('days', 7))
    s_arg, e_arg = request.args.get('start'), request.args.get('end')
    stats, err = bot.get_pnl_stats(days, s_arg, e_arg)
    bal = bot.get_available_balance() or 0.0
    if err or not stats: stats = {"total_pnl":0, "win_rate":0, "total_trades":0, "volume":0, "chart_labels":[], "chart_data":[], "history":[]}
    
    html = """
    <!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><title>P&L</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>body{background:#f7f9fc;color:#333;padding:20px}.card{background:#fff;border:1px solid #e0e0e0;box-shadow:0 2px 5px rgba(0,0,0,0.03)}.text-up{color:#00b894}.text-down{color:#d63031}</style></head><body>
    <div class="container"><div class="d-flex justify-content-between mb-4"><h3>Аналіз P&L</h3><div><span class="h5 me-3">Баланс: ${{ "%.2f"|format(bal) }}</span><a href="/scanner" class="btn btn-outline-secondary">← Сканер</a></div></div>
    <div class="mb-3"><a href="/report?days=1" class="btn btn-sm btn-outline-primary">1Д</a><a href="/report?days=7" class="btn btn-sm btn-outline-primary">7Д</a><a href="/report?days=30" class="btn btn-sm btn-outline-primary">30Д</a></div>
    <div class="row mb-3"><div class="col-md-3"><div class="card p-3"><h6>P&L</h6><h3 class="{{ 'text-up' if stats.total_pnl>=0 else 'text-down' }}">${{ "{:,.2f}".format(stats.total_pnl) }}</h3></div></div>
    <div class="col-md-3"><div class="card p-3"><h6>Win Rate</h6><h3>{{ stats.win_rate }}%</h3></div></div><div class="col-md-3"><div class="card p-3"><h6>Угоди</h6><h3>{{ stats.total_trades }}</h3></div></div></div>
    <div class="card p-3 mb-3"><canvas id="c" height="80"></canvas></div>
    <div class="card p-0"><table class="table table-hover mb-0"><thead><tr><th>Час</th><th>Монета</th><th>Сторона</th><th>Вхід</th><th>Вихід</th><th>P&L</th></tr></thead><tbody>
    {% for t in stats.details %}<tr><td>{{t.exit_time}}</td><td>{{t.symbol}}</td><td>{{t.side}}</td><td>{{t.entry_price}}</td><td>{{t.exit_price}}</td><td class="{{ 'text-up' if t.pnl>0 else 'text-down' }}">{{t.pnl}}</td></tr>{% endfor %}
    </tbody></table></div></div>
    <script>new Chart(document.getElementById('c'),{type:'line',data:{labels:{{ stats.chart_labels|tojson }},datasets:[{label:'P&L',data:{{ stats.chart_data|tojson }},borderColor:'#00b894',backgroundColor:'rgba(0,184,148,0.1)',fill:true}]}});</script></body></html>
    """
    return render_template_string(html, stats=stats, bal=bal)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.get_data(as_text=True))
        logger.info(f"🚀 Webhook: {data.get('symbol')}")
        threading.Thread(target=process_signal_with_ai, args=(data,)).start()
        return jsonify({"status": "ok"})
    except: 
        # Спроба regex якщо чистий json не пройшов
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
