"""
Main Application - Ultimate Edition 🚀
Включає:
- Стару перевірену логіку відкриття ордерів
- Новий Smart Exit (вихід по RSI)
- Професійний UI (Сканер + P&L) українською
- Базу даних та ШІ
"""

from flask import Flask, request, jsonify, render_template_string
from pybit.unified_trading import HTTP
import logging
import threading
import time
import requests
import json
import re
import os
import decimal
from datetime import datetime, timedelta
import ctypes

# === ІМПОРТИ МОДУЛІВ ===
from bot_config import config
from models import db_manager
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from config import get_api_credentials

# === ШІ МОДУЛЬ ===
try:
    import ai_analyst
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI module not found")

# === WINDOWS ANTI-SLEEP ===
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

# === FLASK SETUP ===
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === TELEGRAM ===
def send_telegram_message(text):
    try:
        # 👇👇 ВСТАВТЕ ВАШІ ДАНІ НИЖЧЕ (ЯКЩО ВОНИ НЕ В CONFIG) 👇👇
        tg_token = getattr(config, 'TG_BOT_TOKEN', "ВАШ_ТОКЕН")
        chat_id = getattr(config, 'TG_CHAT_ID', "ВАШ_CHAT_ID")
        
        if "ВАШ_" in tg_token: return 
        
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        logger.error(f"TG Error: {e}")

# === ОБРОБКА СИГНАЛУ (ШІ + ТОРГІВЛЯ) ===
def process_signal_with_ai(data):
    symbol = data.get('symbol')
    action = data.get('action')
    ai_text = "AI Вимкнено"
    
    if AI_AVAILABLE:
        try: ai_text = ai_analyst.analyze_signal(symbol, action)
        except: ai_text = "Помилка аналізу"

    msg = f"🚀 <b>СИГНАЛ: {symbol}</b>\nДія: {action}\n\n🤖 <b>Gemini:</b> {ai_text}"
    send_telegram_message(msg)
    
    # Запуск торгівлі
    bot.place_order(data)

# === SELF PING (KEEP ALIVE) ===
def keep_alive():
    time.sleep(10)
    external_url = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + '/health'
    while True:
        try: requests.get(external_url, timeout=10)
        except: pass
        time.sleep(300) # 5 хвилин

threading.Thread(target=keep_alive, daemon=True).start()

# === ТОРГОВИЙ БОТ (BYBIT) ===
class BybitTradingBot:
    def __init__(self):
        k, s = get_api_credentials()
        self.session = HTTP(testnet=False, api_key=k, api_secret=s)
        logger.info("✅ Bybit Connected")

    # --- Допоміжні методи ---
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
            resp = self.session.get_tickers(category="linear", symbol=norm)
            if resp.get('retCode') == 0: return float(resp['result']['list'][0]['lastPrice'])
            return None
        except: return None

    def set_leverage(self, symbol, leverage):
        try:
            norm = self.normalize_symbol(symbol)
            self.session.set_leverage(category="linear", symbol=norm, buyLeverage=str(leverage), sellLeverage=str(leverage))
            return True
        except Exception as e:
            if "110043" in str(e): return True
            return False

    def get_instrument_info(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            resp = self.session.get_instruments_info(category="linear", symbol=norm)
            if resp.get('retCode') == 0:
                return resp['result']['list'][0]['lotSizeFilter'], resp['result']['list'][0]['priceFilter']
            return None, None
        except: return None, None

    def round_qty(self, qty, step):
        if step <= 0: return qty
        import decimal
        step_decimals = abs(decimal.Decimal(str(step)).as_tuple().exponent)
        return round(qty // step * step, step_decimals)

    def round_price(self, price, tick_size):
        if tick_size <= 0: return price
        import decimal
        tick_decimals = abs(decimal.Decimal(str(tick_size)).as_tuple().exponent)
        return round(price // tick_size * tick_size, tick_decimals)

    def get_position_size(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            resp = self.session.get_positions(category="linear", symbol=norm)
            if resp['retCode'] == 0 and resp['result']['list']:
                return float(resp['result']['list'][0]['size'])
            return 0.0
        except: return 0.0

    # --- ГОЛОВНА ЛОГІКА ВХОДУ (З ВАШОГО СТАРОГО ФАЙЛУ) ---
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            logger.info(f"🤖 Placing Order: {symbol} {action}")
            
            if not action or not symbol: return {"status": "error"}
            norm_symbol = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm_symbol) > 0:
                logger.warning(f"Position exists for {norm_symbol}. Ignored.")
                return {"status": "ignored"}
            
            riskPercent = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            leverage = int(data.get('leverage', config.DEFAULT_LEVERAGE))
            tpPercent = float(data.get('takeProfitPercent', config.DEFAULT_TP_PERCENT))
            slPercent = float(data.get('stopLossPercent', config.DEFAULT_SL_PERCENT))
            
            cur_price = self.get_current_price(norm_symbol)
            if not cur_price: return {"status": "error"}
            
            lot_filter, price_filter = self.get_instrument_info(norm_symbol)
            if not lot_filter: return {"status": "error"}
            
            qty_step = float(lot_filter['qtyStep'])
            min_qty = float(lot_filter['minOrderQty'])
            tick_size = float(price_filter['tickSize'])
            balance = self.get_available_balance()
            if not balance: return {"status": "error"}
            
            margin = (balance * (riskPercent / 100)) * 0.98
            raw_qty = (margin * leverage) / cur_price
            final_qty = self.round_qty(raw_qty, qty_step)
            
            if final_qty < min_qty:
                cost_of_min_qty = (min_qty * cur_price) / leverage
                if balance > cost_of_min_qty * 1.05: final_qty = min_qty
                else: return {"status": "error_balance"}
            
            self.set_leverage(norm_symbol, leverage)
            self.session.place_order(category="linear", symbol=norm_symbol, side=action, orderType="Market", qty=str(final_qty), timeInForce="GTC")
            logger.info(f"✅ Market Order Placed: {final_qty} {norm_symbol}")
            
            if slPercent > 0:
                sl_price = cur_price * (1 - slPercent/100) if action == "Buy" else cur_price * (1 + slPercent/100)
                self.session.set_trading_stop(category="linear", symbol=norm_symbol, stopLoss=str(self.round_price(sl_price, tick_size)), positionIdx=0)
            
            if tpPercent > 0:
                qty_tp1 = self.round_qty(final_qty * 0.5, qty_step)
                qty_tp2 = self.round_qty(final_qty * 0.35, qty_step)
                qty_tp3 = self.round_qty(final_qty - qty_tp1 - qty_tp2, qty_step)
                tp1_dist = tpPercent * 0.5
                tp2_dist = tpPercent * 1.0
                tp3_dist = tpPercent * 1.5
                tp_side = "Sell" if action == "Buy" else "Buy"
                tps = [(qty_tp1, cur_price * (1 + tp1_dist/100) if action == "Buy" else cur_price * (1 - tp1_dist/100)),
                       (qty_tp2, cur_price * (1 + tp2_dist/100) if action == "Buy" else cur_price * (1 - tp2_dist/100)),
                       (qty_tp3, cur_price * (1 + tp3_dist/100) if action == "Buy" else cur_price * (1 - tp3_dist/100))]
                for q, p in tps:
                    if q >= min_qty:
                        self.session.place_order(category="linear", symbol=norm_symbol, side=tp_side, orderType="Limit", qty=str(q), price=str(self.round_price(p, tick_size)), reduceOnly=True)
            return {"status": "success"}
        except Exception as e:
            logger.error(f"🔥 Order Error: {e}")
            return {"status": "error"}

    # --- P&L СТАТИСТИКА (ВИПРАВЛЕНА) ---
    def get_pnl_stats(self, days=7):
        try:
            trades = stats_service.get_trades(days=days)
            if not trades:
                return {"total_pnl": 0.0, "total_trades": 0, "win_rate": 0, "chart_labels": [], "chart_data": [], "history": []}, None

            total_pnl = sum(t['pnl'] for t in trades)
            total_trades = len(trades)
            winners = sum(1 for t in trades if t['pnl'] > 0)
            losers = total_trades - winners
            win_rate = (winners / total_trades * 100) if total_trades > 0 else 0
            total_vol = sum(t.get('qty', 0) * t.get('entry_price', 0) for t in trades)

            daily_pnl = {}
            for t in trades:
                if t['exit_time']:
                    date_str = t['exit_time'].split(' ')[0]
                    daily_pnl[date_str] = daily_pnl.get(date_str, 0) + t['pnl']
            
            sorted_dates = sorted(daily_pnl.keys())
            chart_labels = [d[5:] for d in sorted_dates]
            chart_data = []
            running_balance = 0
            for d in sorted_dates:
                running_balance += daily_pnl[d]
                chart_data.append(round(running_balance, 2))

            stats = {
                "total_pnl": round(total_pnl, 2), "total_trades": total_trades,
                "win_rate": round(win_rate, 1), "winners": winners, "losers": losers,
                "volume": round(total_vol, 2), "chart_labels": chart_labels,
                "chart_data": chart_data, "history": trades[:50]
            }
            return stats, None
        except Exception as e: return None, str(e)

# Ініціалізація
bot = BybitTradingBot()
scanner = EnhancedMarketScanner(bot, config.get_scanner_config())
scanner.start()

# === 🔥 SMART TRADE MANAGER (АВТОМАТИЧНИЙ ВИХІД) ===
class SmartTradeManager:
    def __init__(self, bot_instance, scanner_instance):
        self.bot = bot_instance
        self.scanner = scanner_instance
        self.running = True
        
    def start(self):
        threading.Thread(target=self.loop, daemon=True).start()
        logger.info("🛡️ Smart Trade Manager Active")
        
    def loop(self):
        while self.running:
            try: self.manage_positions()
            except Exception as e: logger.error(f"Manager Error: {e}")
            time.sleep(5)
            
    def manage_positions(self):
        resp = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if resp['retCode'] != 0: return
        
        for pos in resp['result']['list']:
            size = float(pos['size'])
            if size == 0: continue
            
            symbol = pos['symbol']
            side = pos['side']
            entry_price = float(pos['avgPrice'])
            unrealized_pnl = float(pos['unrealisedPnl'])
            
            # Отримуємо RSI
            rsi = self.scanner.get_current_rsi(symbol)
            
            # 1. Вихід по RSI (Піковий)
            if side == "Buy" and rsi >= 78 and unrealized_pnl > 0:
                self.close_position(symbol, size, "Sell", f"RSI Overbought ({rsi})")
                continue
            if side == "Sell" and rsi <= 22 and unrealized_pnl > 0:
                self.close_position(symbol, size, "Buy", f"RSI Oversold ({rsi})")
                continue

            # 2. Трейлінг Стоп
            pnl_pct = (unrealized_pnl / (entry_price * size / float(pos['leverage']))) * 100
            current_sl = float(pos.get('stopLoss', 0))
            
            if side == "Buy":
                if pnl_pct > 1.5: # Якщо прибуток > 1.5%, ставимо БУ+
                    new_sl = entry_price * 1.005
                    if current_sl < new_sl: self.update_sl(symbol, new_sl)
                elif pnl_pct > 3.0: # Якщо прибуток > 3%, підтягуємо
                    new_sl = entry_price * 1.02
                    if current_sl < new_sl: self.update_sl(symbol, new_sl)
            elif side == "Sell":
                if pnl_pct > 1.5:
                    new_sl = entry_price * 0.995
                    if current_sl == 0 or current_sl > new_sl: self.update_sl(symbol, new_sl)

    def close_position(self, symbol, qty, side, reason):
        try:
            self.bot.session.place_order(category="linear", symbol=symbol, side=side, orderType="Market", qty=str(qty), reduceOnly=True)
            send_telegram_message(f"💰 <b>AUTO-CLOSE: {symbol}</b>\nПричина: {reason}\nP&L фіксується.")
            logger.info(f"✅ Auto-closed {symbol}: {reason}")
        except Exception as e: logger.error(f"Failed to close {symbol}: {e}")

    def update_sl(self, symbol, price):
        try:
            price_str = "{:.4f}".format(price)
            self.bot.session.set_trading_stop(category="linear", symbol=symbol, stopLoss=price_str, positionIdx=0)
            logger.info(f"🛡️ Updated SL for {symbol} to {price_str}")
        except: pass

trade_manager = SmartTradeManager(bot, scanner)
trade_manager.start()

# === WEB ROUTES (UI) ===

@app.route('/scanner', methods=['GET'])
def scanner_page():
    scan_data = scanner.get_aggregated_data(hours=24)
    last_update = datetime.now().strftime('%H:%M:%S')
    
    # Отримуємо активні позиції для відображення
    active_positions = []
    try:
        pos_data = bot.session.get_positions(category="linear", settleCoin="USDT")
        if pos_data['retCode'] == 0:
            for p in pos_data['result']['list']:
                if float(p['size']) > 0:
                    symbol = p['symbol']
                    market_data = scan_data['snapshots'].get(symbol, {})
                    current_rsi = market_data.get('rsi', 50)
                    pnl = float(p['unrealisedPnl'])
                    
                    rec = "ТРИМАТИ 🟢"
                    row_class = ""
                    if p['side'] == "Buy" and current_rsi > 75: rec = "ВИХІД (RSI) 🔴"; row_class = "table-danger"
                    elif p['side'] == "Sell" and current_rsi < 25: rec = "ВИХІД (RSI) 🔴"; row_class = "table-danger"

                    active_positions.append({
                        'symbol': symbol, 'side': p['side'], 'entry': p['avgPrice'], 'pnl': round(pnl, 2),
                        'rsi': current_rsi, 'recommendation': rec, 'row_class': row_class
                    })
    except: pass

    # Сортування
    coin_stats = {}
    for p in scan_data['all_signals']:
        sym = p['symbol']
        if sym not in coin_stats: coin_stats[sym] = {'inflow': 0, 'change': 0, 'count': 0}
        coin_stats[sym]['inflow'] += p['vol_inflow']
        coin_stats[sym]['change'] += p['price_change_interval']
        coin_stats[sym]['count'] += 1
    
    positive_coins = [{'symbol':k, 'inflow':v['inflow'], 'avg_change':round(v['change']/v['count'],2), 'bar_pct': 50} for k,v in coin_stats.items() if v['change']>=0]
    negative_coins = [{'symbol':k, 'inflow':v['inflow'], 'avg_change':round(v['change']/v['count'],2), 'bar_pct': 50} for k,v in coin_stats.items() if v['change']<0]
    positive_coins.sort(key=lambda x: x['inflow'], reverse=True)
    negative_coins.sort(key=lambda x: x['inflow'], reverse=True)

    html = """
    <!DOCTYPE html><html lang="uk" data-bs-theme="dark"><head><meta charset="UTF-8"><title>Whale Terminal Pro</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.7/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <style>
        body{background:#0b0e11;color:#eaecef;font-family:'Segoe UI',sans-serif;font-size:13px}
        .card{background:#151a1f;border:1px solid #2a3441;margin-bottom:15px}
        .table-dark{--bs-table-bg:transparent} td{vertical-align:middle}
        .text-up{color:#0ecb81!important} .text-down{color:#f6465d!important}
        .badge-rsi-high{background:rgba(246,70,93,0.2);color:#f6465d;border:1px solid #f6465d}
    </style>
    <meta http-equiv="refresh" content="30"></head><body>
    <nav class="navbar navbar-dark bg-dark border-bottom border-secondary mb-3 p-2"><span class="navbar-brand">🐋 Whale Terminal PRO</span><span class="text-muted">{{ last_update }}</span></nav>
    <div class="container-fluid">
        {% if positions %}
        <div class="card border-primary mb-3"><div class="card-header bg-primary bg-opacity-10">МОЇ ПОЗИЦІЇ</div><div class="card-body p-0"><table class="table table-dark table-hover mb-0"><thead><tr><th>Монета</th><th>Тип</th><th>P&L</th><th>RSI</th><th>Статус</th></tr></thead><tbody>
        {% for pos in positions %}<tr class="{{ pos.row_class }}">
            <td class="fw-bold">{{ pos.symbol }}</td><td>{{ pos.side }}</td><td class="{{ 'text-up' if pos.pnl>0 else 'text-down' }}">{{ pos.pnl }}$</td>
            <td><span class="badge {{ 'badge-rsi-high' if pos.rsi>70 else 'bg-secondary' }}">{{ pos.rsi }}</span></td><td>{{ pos.recommendation }}</td>
        </tr>{% endfor %}</tbody></table></div></div>
        {% endif %}
        
        <div class="row"><div class="col-md-6"><div class="card"><div class="card-header text-up">ТОП ПОКУПЦІВ</div><div class="card-body p-0"><table class="table table-dark table-sm mb-0"><thead><tr><th>Актив</th><th>Вхід</th></tr></thead><tbody>{% for c in positive_coins[:5] %}<tr><td>{{c.symbol}}</td><td class="text-up">${{"{:,.0f}".format(c.inflow)}}</td></tr>{% endfor %}</tbody></table></div></div></div>
        <div class="col-md-6"><div class="card"><div class="card-header text-down">ТОП ПРОДАВЦІВ</div><div class="card-body p-0"><table class="table table-dark table-sm mb-0"><thead><tr><th>Актив</th><th>Вхід</th></tr></thead><tbody>{% for c in negative_coins[:5] %}<tr><td>{{c.symbol}}</td><td class="text-down">${{"{:,.0f}".format(c.inflow)}}</td></tr>{% endfor %}</tbody></table></div></div></div></div>

        <div class="card"><div class="card-header d-flex justify-content-between"><span>ЖУРНАЛ СИГНАЛІВ</span><a href="/report" class="btn btn-sm btn-outline-light">Звіт P&L</a></div>
        <div class="card-body p-0"><table id="signalsTable" class="table table-dark table-hover w-100"><thead><tr><th>Час</th><th>Символ</th><th>Ціна</th><th>Зміна</th><th>Аномалія</th><th>RSI</th><th>Об'єм</th></tr></thead><tbody>
        {% for p in pumps %}<tr><td class="text-muted">{{ p.time }}</td><td class="fw-bold text-primary">{{ p.symbol }}</td><td>{{ p.price }}</td><td class="{{ 'text-up' if p.price_change_interval>0 else 'text-down' }}">{{ p.price_change_interval }}%</td><td>x{{ p.spike_factor }}</td><td>{{ p.get('rsi','-') }}</td><td>${{ "{:,.0f}".format(p.vol_inflow) }}</td></tr>{% endfor %}
        </tbody></table></div></div>
    </div>
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script><script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script><script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script>
    <script>$(document).ready(function(){$('#signalsTable').DataTable({"order":[[0,"desc"]],"pageLength":25});});</script></body></html>
    """
    return render_template_string(html, pumps=scan_data['all_signals'], last_update=last_update, positive_coins=positive_coins, negative_coins=negative_coins, positions=active_positions)

@app.route('/report', methods=['GET'])
def report_page():
    days = request.args.get('days', default=7, type=int)
    stats, error = bot.get_pnl_stats(days=days)
    if error or not stats: stats = {"total_pnl": 0, "win_rate": 0, "total_trades": 0, "volume": 0, "chart_labels": [], "chart_data": [], "history": []}
    balance = bot.get_available_balance() or 0.0
    
    html = """
    <!DOCTYPE html><html lang="uk" data-bs-theme="dark"><head><title>P&L</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"><script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>body{background:#0b0e11;color:#eaecef;padding:20px}.card{background:#151a1f;border:1px solid #2a3441}.text-up{color:#0ecb81}.text-down{color:#f6465d}</style></head><body>
    <div class="container"><div class="d-flex justify-content-between mb-4"><h3>P&L Аналіз</h3><a href="/scanner" class="btn btn-outline-light">← Сканер</a></div>
    <div class="row mb-3"><div class="col-md-3"><div class="card p-3"><h6>P&L</h6><h3 class="{{ 'text-up' if stats.total_pnl>=0 else 'text-down' }}">${{ "{:,.2f}".format(stats.total_pnl) }}</h3></div></div>
    <div class="col-md-3"><div class="card p-3"><h6>Win Rate</h6><h3>{{ stats.win_rate }}%</h3></div></div><div class="col-md-3"><div class="card p-3"><h6>Баланс</h6><h3>${{ "%.2f"|format(balance) }}</h3></div></div></div>
    <div class="card p-3 mb-3"><canvas id="chart" height="80"></canvas></div>
    <div class="card p-0"><table class="table table-dark table-hover mb-0"><thead><tr><th>Час</th><th>Монета</th><th>Сторона</th><th>Вхід</th><th>Вихід</th><th>P&L</th></tr></thead><tbody>
    {% for t in stats.history %}<tr><td>{{ t.exit_time }}</td><td>{{ t.symbol }}</td><td>{{ t.side }}</td><td>{{ t.entry_price }}</td><td>{{ t.exit_price }}</td><td class="{{ 'text-up' if t.pnl>0 else 'text-down' }}">{{ t.pnl }}</td></tr>{% endfor %}
    </tbody></table></div></div>
    <script>new Chart(document.getElementById('chart'),{type:'line',data:{labels:{{ stats.chart_labels|tojson }},datasets:[{label:'P&L',data:{{ stats.chart_data|tojson }},borderColor:'#0ecb81',backgroundColor:'rgba(14,203,129,0.2)',fill:true}]}});</script></body></html>
    """
    return render_template_string(html, stats=stats, balance=balance)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw = request.get_data(as_text=True)
        data = None
        try: data = json.loads(raw)
        except: 
            m = re.search(r'\{.*\}', raw, re.DOTALL); 
            if m: data = json.loads(m.group())
        if data:
            logger.info(f"🚀 Webhook: {data.get('symbol')}")
            threading.Thread(target=process_signal_with_ai, args=(data,)).start()
            return jsonify({"status": "ok"})
        return jsonify({"error": "no data"}), 400
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/')
def home(): return "<script>window.location.href='/scanner';</script>"
@app.route('/health')
def health(): return jsonify({"status": "ok"})

if __name__ == '__main__':
    try: stats_service.cleanup_old_data(days=config.DATA_RETENTION_DAYS)
    except: pass
    app.run(host=config.HOST, port=config.PORT)
