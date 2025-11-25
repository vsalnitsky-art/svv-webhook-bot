"""
Main Application - Custom Period Analytics 📅
Включає:
- Фільтр статистики: 1 день, 7 днів, 30 днів, Довільний період
- Професійний UI
- Smart Exit та Торгівля
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
from datetime import datetime, timedelta, date
import ctypes

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

# === WINDOWS ANTI-SLEEP ===
try:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
except: pass

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === TELEGRAM ===
def send_telegram_message(text):
    try:
        tg_token = getattr(config, 'TG_BOT_TOKEN', "ВАШ_ТОКЕН")
        chat_id = getattr(config, 'TG_CHAT_ID', "ВАШ_CHAT_ID")
        if "ВАШ_" in tg_token: return 
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        logger.error(f"TG Error: {e}")

# === ОБРОБКА СИГНАЛУ ===
def process_signal_with_ai(data):
    symbol = data.get('symbol')
    action = data.get('action')
    ai_text = "AI Вимкнено"
    
    if AI_AVAILABLE:
        try: ai_text = ai_analyst.analyze_signal(symbol, action)
        except: ai_text = "Помилка аналізу"

    msg = f"🚀 <b>ВХІД: {symbol}</b>\nНапрямок: {action}\n\n🤖 <b>Gemini:</b> {ai_text}"
    send_telegram_message(msg)
    bot.place_order(data)

# === SELF PING ===
def keep_alive():
    time.sleep(10)
    external_url = os.environ.get('RENDER_EXTERNAL_URL', f'http://127.0.0.1:{config.PORT}') + '/health'
    while True:
        try: requests.get(external_url, timeout=10)
        except: pass
        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()

# === ТОРГОВИЙ БОТ ===
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

    # === ВХІД В УГОДУ ===
    def place_order(self, data):
        try:
            action = data.get('action')
            symbol = data.get('symbol')
            logger.info(f"🤖 Processing Order: {symbol} {action}")
            
            if not action or not symbol: return {"status": "error"}
            norm_symbol = self.normalize_symbol(symbol)
            
            if self.get_position_size(norm_symbol) > 0:
                logger.warning(f"Position exists for {norm_symbol}. Ignored.")
                return {"status": "ignored"}
            
            riskPercent = float(data.get('riskPercent', config.DEFAULT_RISK_PERCENT))
            leverage = int(data.get('leverage', config.DEFAULT_LEVERAGE))
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
                if balance > cost_of_min_qty * 1.05:
                    final_qty = min_qty
                    logger.info(f"✅ Forced Min Qty: {final_qty}")
                else:
                    return {"status": "error_balance"}
            
            self.set_leverage(norm_symbol, leverage)
            self.session.place_order(category="linear", symbol=norm_symbol, side=action, orderType="Market", qty=str(final_qty), timeInForce="GTC")
            logger.info(f"✅ Opened: {action} {final_qty} {norm_symbol}")
            
            if slPercent > 0:
                buffer_percent = 0.2
                if action == "Buy":
                    raw_sl = cur_price * (1 - slPercent/100)
                    buffered_sl = raw_sl * (1 - buffer_percent/100)
                else:
                    raw_sl = cur_price * (1 + slPercent/100)
                    buffered_sl = raw_sl * (1 + buffer_percent/100)
                
                final_sl = self.round_price(buffered_sl, tick_size)
                self.session.set_trading_stop(category="linear", symbol=norm_symbol, stopLoss=str(final_sl), positionIdx=0)
                logger.info(f"🛡️ SL Set with Buffer: {final_sl}")

            return {"status": "success"}
        except Exception as e:
            logger.error(f"🔥 Order Error: {e}")
            return {"status": "error"}

    # === СИНХРОНІЗАЦІЯ ===
    def sync_trades_from_bybit(self, days=30):
        """Вантажимо трохи більше даних для гнучкості"""
        logger.info(f"🔄 Syncing P&L...")
        try:
            now = datetime.now()
            for i in range(0, days, 7):
                chunk_days = min(7, days - i)
                end_dt = now - timedelta(days=i)
                start_dt = end_dt - timedelta(days=chunk_days)
                ts_end = int(end_dt.timestamp() * 1000)
                ts_start = int(start_dt.timestamp() * 1000)
                resp = self.session.get_closed_pnl(category="linear", startTime=ts_start, endTime=ts_end, limit=50)
                if resp['retCode'] == 0:
                    trades = resp['result']['list']
                    for t in trades:
                        api_side = t['side'] 
                        trade_data = {
                            'order_id': t['orderId'], 'symbol': t['symbol'],
                            'side': 'Long' if api_side == 'Sell' else 'Short', 
                            'qty': float(t['qty']), 'entry_price': float(t['avgEntryPrice']),
                            'exit_price': float(t['avgExitPrice']), 'pnl': float(t['closedPnl']),
                            'exit_time': datetime.fromtimestamp(int(t['updatedTime'])/1000),
                            'is_win': float(t['closedPnl']) > 0
                        }
                        stats_service.save_trade(trade_data)
        except: pass

    # === P&L ЛОГІКА З ФІЛЬТРАМИ ДАТ ===
    def get_pnl_stats(self, days=None, start_date=None, end_date=None):
        # 1. Синхронізуємо дані за максимальний період (про всяк випадок)
        self.sync_trades_from_bybit(days=30) 
        
        try:
            # 2. Отримуємо всі угоди з бази (за 90 днів для запасу)
            all_trades = stats_service.get_trades(days=90)
            if not all_trades:
                return {"total_pnl": 0.0, "total_trades": 0, "win_rate": 0, "chart_labels": [], "chart_data": [], "history": []}, None

            # 3. Фільтруємо угоди по датах
            filtered_trades = []
            
            # Визначаємо межі часу
            filter_start = None
            filter_end = None

            if start_date and end_date:
                # Якщо вибрано календар
                filter_start = datetime.strptime(start_date, '%Y-%m-%d')
                filter_end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1) # Включаючи кінець дня
            elif days:
                # Якщо вибрано кнопку (1, 7, 30 днів)
                filter_end = datetime.now()
                filter_start = filter_end - timedelta(days=days)

            for t in all_trades:
                if not t['exit_time']: continue
                # Конвертуємо рядок дати з БД в об'єкт datetime, якщо треба
                exit_dt = t['exit_time'] 
                if isinstance(exit_dt, str):
                    try: exit_dt = datetime.strptime(exit_dt, '%Y-%m-%d %H:%M')
                    except: continue
                
                # Перевірка діапазону
                if filter_start and filter_end:
                    if filter_start <= exit_dt <= filter_end:
                        filtered_trades.append(t)
                else:
                    filtered_trades.append(t)

            # 4. Рахуємо статистику по відфільтрованим даним
            stats = {
                "total_trades": len(filtered_trades),
                "total_pnl": 0.0, "total_volume": 0.0,
                "win_trades": 0, "loss_trades": 0,
                "long_trades": 0, "short_trades": 0,
                "details": [], "chart_labels": [], "chart_data": []
            }

            cumulative_pnl = 0.0
            # Сортуємо від старих до нових для графіка
            filtered_trades.sort(key=lambda x: x['exit_time'] if x['exit_time'] else '', reverse=False)

            daily_pnl = {} # Для агрегації по днях на графіку

            for trade in filtered_trades:
                pnl = trade['pnl']
                stats["total_pnl"] += pnl
                cumulative_pnl += pnl
                
                qty = trade.get('qty', 0)
                price = trade.get('exit_price', 0)
                stats["total_volume"] += qty * price

                if pnl > 0: stats["win_trades"] += 1
                else: stats["loss_trades"] += 1

                if trade['side'] == "Long": stats["long_trades"] += 1
                else: stats["short_trades"] += 1

                # Графік
                if trade['exit_time']:
                    if isinstance(trade['exit_time'], str): d_str = trade['exit_time'].split(' ')[0]
                    else: d_str = trade['exit_time'].strftime('%Y-%m-%d')
                    
                    daily_pnl[d_str] = daily_pnl.get(d_str, 0) + pnl

                stats["details"].append(trade)

            # Будуємо красивий графік по днях
            running_balance = 0
            for d in sorted(daily_pnl.keys()):
                running_balance += daily_pnl[d]
                stats["chart_labels"].append(d[5:]) # MM-DD
                stats["chart_data"].append(round(running_balance, 2))

            # Сортуємо деталі: нові зверху для таблиці
            stats["details"].sort(key=lambda x: x['exit_time'] if x['exit_time'] else '', reverse=True)
            
            if stats["total_trades"] > 0:
                stats["win_rate"] = round((stats["win_trades"] / stats["total_trades"]) * 100, 1)
            
            return stats, None
        except Exception as e: return None, str(e)

# Ініціалізація
bot = BybitTradingBot()
scanner = EnhancedMarketScanner(bot, config.get_scanner_config())
scanner.start()

# === SMART EXIT MANAGER ===
class SmartTradeManager:
    def __init__(self, bot_instance, scanner_instance):
        self.bot = bot_instance
        self.scanner = scanner_instance
        self.running = True
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.manage_positions()
            except: pass
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
            rsi = self.scanner.get_current_rsi(symbol)
            
            if side == "Buy" and rsi >= 78 and unrealized_pnl > 0:
                self.close_position(symbol, size, "Sell", f"RSI High ({rsi})")
                continue
            if side == "Sell" and rsi <= 22 and unrealized_pnl > 0:
                self.close_position(symbol, size, "Buy", f"RSI Low ({rsi})")
                continue

            pnl_pct = (unrealized_pnl / (entry_price * size / float(pos['leverage']))) * 100
            current_sl = float(pos.get('stopLoss', 0))
            if side == "Buy":
                if pnl_pct > 1.5:
                    new_sl = entry_price * 1.005
                    if current_sl < new_sl: self.update_sl(symbol, new_sl)
                elif pnl_pct > 3.0:
                    new_sl = entry_price * 1.02
                    if current_sl < new_sl: self.update_sl(symbol, new_sl)
            elif side == "Sell":
                if pnl_pct > 1.5:
                    new_sl = entry_price * 0.995
                    if current_sl == 0 or current_sl > new_sl: self.update_sl(symbol, new_sl)

    def close_position(self, symbol, qty, side, reason):
        try:
            self.bot.session.place_order(category="linear", symbol=symbol, side=side, orderType="Market", qty=str(qty), reduceOnly=True)
            self.bot.session.cancel_all_orders(category="linear", symbol=symbol)
            send_telegram_message(f"💰 <b>AUTO-CLOSE: {symbol}</b>\nПричина: {reason}\nP&L фіксується.")
        except: pass

    def update_sl(self, symbol, price):
        try:
            price_str = "{:.4f}".format(price)
            self.bot.session.set_trading_stop(category="linear", symbol=symbol, stopLoss=price_str, positionIdx=0)
        except: pass

trade_manager = SmartTradeManager(bot, scanner)
trade_manager.start()

# === WEB ROUTES ===

@app.route('/scanner', methods=['GET'])
def scanner_page():
    scan_data = scanner.get_aggregated_data(hours=24)
    last_update = datetime.now().strftime('%H:%M:%S')
    active_positions = []
    try:
        pos_data = bot.session.get_positions(category="linear", settleCoin="USDT")
        if pos_data['retCode'] == 0:
            for p in pos_data['result']['list']:
                if float(p['size']) > 0:
                    symbol = p['symbol']
                    current_rsi = scan_data['snapshots'].get(symbol, {}).get('rsi', 50)
                    pnl = float(p['unrealisedPnl'])
                    rec, row_class = "ТРИМАТИ 🟢", ""
                    if p['side'] == "Buy" and current_rsi > 75: rec = "ВИХІД (RSI) 🔴"; row_class = "table-danger"
                    elif p['side'] == "Sell" and current_rsi < 25: rec = "ВИХІД (RSI) 🔴"; row_class = "table-danger"
                    active_positions.append({'symbol': symbol, 'side': p['side'], 'entry': p['avgPrice'], 'pnl': round(pnl, 2), 'rsi': current_rsi, 'recommendation': rec, 'row_class': row_class})
    except: pass

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
    <style>body{background:#0b0e11;color:#eaecef;font-family:'Segoe UI',sans-serif;font-size:13px}.card{background:#151a1f;border:1px solid #2a3441;margin-bottom:15px}.table-dark{--bs-table-bg:transparent} td{vertical-align:middle}.text-up{color:#0ecb81!important}.text-down{color:#f6465d!important}.badge-rsi-high{background:rgba(246,70,93,0.2);color:#f6465d;border:1px solid #f6465d}</style>
    <meta http-equiv="refresh" content="30"></head><body>
    <nav class="navbar navbar-dark bg-dark border-bottom border-secondary mb-3 p-2"><span class="navbar-brand">🐋 Whale Terminal PRO</span><span class="text-muted">{{ last_update }}</span></nav>
    <div class="container-fluid">
        {% if positions %}<div class="card border-primary mb-3"><div class="card-header bg-primary bg-opacity-10">МОЇ ПОЗИЦІЇ</div><div class="card-body p-0"><table class="table table-dark table-hover mb-0"><thead><tr><th>Монета</th><th>Тип</th><th>P&L</th><th>RSI</th><th>Статус</th></tr></thead><tbody>{% for pos in positions %}<tr class="{{ pos.row_class }}"><td class="fw-bold">{{ pos.symbol }}</td><td>{{ pos.side }}</td><td class="{{ 'text-up' if pos.pnl>0 else 'text-down' }}">{{ pos.pnl }}$</td><td><span class="badge {{ 'badge-rsi-high' if pos.rsi>70 else 'bg-secondary' }}">{{ pos.rsi }}</span></td><td>{{ pos.recommendation }}</td></tr>{% endfor %}</tbody></table></div></div>{% endif %}
        <div class="row"><div class="col-md-6"><div class="card"><div class="card-header text-up">ТОП ПОКУПЦІВ</div><div class="card-body p-0"><table class="table table-dark table-sm mb-0"><thead><tr><th>Актив</th><th>Вхід</th></tr></thead><tbody>{% for c in positive_coins[:5] %}<tr><td>{{c.symbol}}</td><td class="text-up">${{"{:,.0f}".format(c.inflow)}}</td></tr>{% endfor %}</tbody></table></div></div></div><div class="col-md-6"><div class="card"><div class="card-header text-down">ТОП ПРОДАВЦІВ</div><div class="card-body p-0"><table class="table table-dark table-sm mb-0"><thead><tr><th>Актив</th><th>Вхід</th></tr></thead><tbody>{% for c in negative_coins[:5] %}<tr><td>{{c.symbol}}</td><td class="text-down">${{"{:,.0f}".format(c.inflow)}}</td></tr>{% endfor %}</tbody></table></div></div></div></div>
        <div class="card"><div class="card-header d-flex justify-content-between"><span>ЖУРНАЛ СИГНАЛІВ</span><a href="/report" class="btn btn-sm btn-outline-light">Звіт P&L</a></div><div class="card-body p-0"><table id="signalsTable" class="table table-dark table-hover w-100"><thead><tr><th>Час</th><th>Символ</th><th>Ціна</th><th>Зміна</th><th>Аномалія</th><th>RSI</th><th>Об'єм</th></tr></thead><tbody>{% for p in pumps %}<tr><td class="text-muted">{{ p.time }}</td><td class="fw-bold text-primary">{{ p.symbol }}</td><td>{{ p.price }}</td><td class="{{ 'text-up' if p.price_change_interval>0 else 'text-down' }}">{{ p.price_change_interval }}%</td><td>x{{ p.spike_factor }}</td><td>{{ p.get('rsi','-') }}</td><td>${{ "{:,.0f}".format(p.vol_inflow) }}</td></tr>{% endfor %}</tbody></table></div></div>
    </div>
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script><script src="https://cdn.datatables.net/1.13.7/js/jquery.dataTables.min.js"></script><script src="https://cdn.datatables.net/1.13.7/js/dataTables.bootstrap5.min.js"></script><script>$(document).ready(function(){$('#signalsTable').DataTable({"order":[[0,"desc"]],"pageLength":25});});</script></body></html>
    """
    return render_template_string(html, pumps=scan_data['all_signals'], last_update=last_update, positive_coins=positive_coins, negative_coins=negative_coins, positions=active_positions)

@app.route('/report', methods=['GET'])
def report_page():
    # Отримуємо параметри фільтрації
    days_arg = request.args.get('days')
    start_arg = request.args.get('start')
    end_arg = request.args.get('end')
    
    days = int(days_arg) if days_arg else None
    
    # Якщо параметрів немає, за замовчуванням 7 днів
    if not days and not start_arg: days = 7

    stats, error = bot.get_pnl_stats(days=days, start_date=start_arg, end_date=end_arg)
    balance = bot.get_available_balance() or 0.0
    if error or not stats: stats = {"total_pnl": 0, "win_rate": 0, "total_trades": 0, "volume": 0, "chart_labels": [], "chart_data": [], "details": [], "long_trades":0, "short_trades":0, "win_trades":0, "loss_trades":0}

    # Відображення періоду у заголовку
    if start_arg and end_arg: period_label = f"{start_arg} — {end_arg}"
    else: period_label = f"Останні {days} днів"

    html_template = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <title>Аналіз P&L</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            :root { --bg-color: #f7f8fa; --card-bg: #ffffff; --text-primary: #121214; --text-secondary: #858e9c; --green: #20b26c; --red: #ef454a; --border: #eff2f5; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: var(--bg-color); color: var(--text-primary); margin: 0; padding: 20px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: var(--card-bg); border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .header-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .balance { font-size: 24px; font-weight: 700; }
            .top-stats-grid { display: flex; gap: 40px; margin-bottom: 10px; flex-wrap: wrap; }
            .kpi-block { min-width: 150px; }
            .kpi-val { font-size: 24px; font-weight: 700; margin-top: 5px; }
            .kpi-label { font-size: 13px; color: var(--text-secondary); text-decoration: underline dotted; cursor: help; }
            .detail-stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            .stat-box { background: white; border-radius: 8px; padding: 20px; display: flex; flex-direction: column; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
            .stat-box-content { display: flex; align-items: center; gap: 20px; }
            .donut-ring, .gauge-ring { width: 50px; height: 50px; flex-shrink: 0; }
            .stat-big-num { font-size: 28px; font-weight: 700; line-height: 1.2; }
            .text-green { color: var(--green); } .text-red { color: var(--red); } .text-gray { color: var(--text-secondary); }
            .chart-container { position: relative; height: 300px; width: 100%; }
            table { width: 100%; border-collapse: collapse; font-size: 13px; }
            th { text-align: left; color: var(--text-secondary); font-weight: 500; padding: 12px 16px; border-bottom: 1px solid var(--border); }
            td { padding: 14px 16px; border-bottom: 1px solid var(--border); vertical-align: middle; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }
            .badge-win { background: rgba(32, 178, 108, 0.1); color: var(--green); }
            .badge-loss { background: rgba(239, 69, 74, 0.1); color: var(--red); }
            .filter-bar { display: flex; gap: 10px; align-items: center; background: #fff; padding: 10px; border-radius: 8px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="header-row">
                    <h1>Аналіз P&L <span style="font-size:14px; color:#999; font-weight:400;">{{ period_label }}</span></h1>
                    <div style="text-align: right;">
                        <div style="font-size: 12px; color: #858e9c;">Баланс Гаманця</div>
                        <div class="balance">${{ "%.2f"|format(balance) }}</div>
                    </div>
                </div>
                
                <div class="filter-bar">
                    <a href="/report?days=1" class="btn btn-sm btn-outline-primary">Сьогодні</a>
                    <a href="/report?days=7" class="btn btn-sm btn-outline-primary">7 Днів</a>
                    <a href="/report?days=30" class="btn btn-sm btn-outline-primary">30 Днів</a>
                    <div style="border-left: 1px solid #eee; height: 20px; margin: 0 10px;"></div>
                    <form action="/report" method="get" class="d-flex gap-2 align-items-center m-0">
                        <input type="date" name="start" class="form-control form-control-sm" required>
                        <span>-</span>
                        <input type="date" name="end" class="form-control form-control-sm" required>
                        <button type="submit" class="btn btn-sm btn-primary">OK</button>
                    </form>
                    <a href="/scanner" class="btn btn-sm btn-secondary ms-auto">← Сканер</a>
                </div>

                <div class="top-stats-grid">
                    <div class="kpi-block"><div class="kpi-label">Загальний P&L</div><div class="kpi-val {{ 'text-green' if stats.total_pnl >= 0 else 'text-red' }}">{{ "+" if stats.total_pnl > 0 }}{{ "%.2f"|format(stats.total_pnl) }} USD</div></div>
                    <div class="kpi-block"><div class="kpi-label">Торговий об'єм</div><div class="kpi-val text-green">{{ "%.2f"|format(stats.total_volume) }} USD</div></div>
                    <div class="kpi-block"><div class="kpi-label">Угоди</div><div class="kpi-val">{{ stats.total_trades }}</div></div>
                </div>
            </div>

            <div class="detail-stats-grid">
                <div class="stat-box">
                    <div style="margin-bottom:15px; color:#858e9c; font-size:13px;">Закриті ордери</div>
                    <div class="stat-box-content">
                        <div class="donut-ring">
                            <svg viewBox="0 0 36 36">
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#ef454a" stroke-width="4" />
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#20b26c" stroke-width="4" stroke-dasharray="{{ (stats.long_trades / stats.total_trades * 100) if stats.total_trades > 0 else 0 }}, 100" />
                            </svg>
                        </div>
                        <div>
                            <div class="stat-big-num">{{ stats.total_trades }}</div>
                            <div style="font-size:13px;"><span class="text-green">{{ stats.long_trades }} Лонг</span> / <span class="text-red">{{ stats.short_trades }} Шорт</span></div>
                        </div>
                    </div>
                </div>
                <div class="stat-box">
                    <div style="margin-bottom:15px; color:#858e9c; font-size:13px;">Win Rate</div>
                    <div class="stat-box-content">
                        <div class="gauge-ring">
                            <svg viewBox="0 0 36 36">
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#eff2f5" stroke-width="4" />
                                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#20b26c" stroke-width="4" stroke-dasharray="{{ stats.win_rate }}, 100" />
                            </svg>
                        </div>
                        <div>
                            <div class="stat-big-num">{{ stats.win_rate }}%</div>
                            <div style="font-size:13px; color:#858e9c;">{{ stats.win_trades }} Win / {{ stats.loss_trades }} Loss</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3>Кумулятивний P&L</h3>
                <div class="chart-container"><canvas id="pnlChart"></canvas></div>
            </div>

            <div class="card">
                <h3>Історія Угод</h3>
                <div style="overflow-x: auto;">
                    <table>
                        <thead><tr><th>Тікер</th><th>Сторона</th><th>Кіл-сть</th><th>Вхід</th><th>Вихід</th><th>P&L</th><th>Результат</th><th>Час</th></tr></thead>
                        <tbody>
                            {% for trade in stats.details %}
                            <tr>
                                <td style="font-weight:600;">{{ trade.symbol }}</td>
                                <td class="{{ 'text-green' if trade.side == 'Long' else 'text-red' }}">{{ trade.side }}</td>
                                <td>{{ trade.qty }}</td>
                                <td>{{ trade.entry_price }}</td>
                                <td>{{ trade.exit_price }}</td>
                                <td class="{{ 'text-green' if trade.pnl > 0 else 'text-red' }}">{{ "+" if trade.pnl > 0 }}{{ "%.4f"|format(trade.pnl) }}</td>
                                <td><span class="badge {{ 'badge-win' if trade.is_win else 'badge-loss' }}">{{ "WIN" if trade.is_win else "LOSS" }}</span></td>
                                <td style="color:#888;">{{ trade.exit_time }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        <script>
            const ctx = document.getElementById('pnlChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: {{ stats.chart_labels | tojson }},
                    datasets: [{
                        label: 'P&L ($)',
                        data: {{ stats.chart_data | tojson }},
                        borderColor: '#20b26c',
                        backgroundColor: 'rgba(32, 178, 108, 0.1)',
                        fill: true, tension: 0.4
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, stats=stats, balance=balance, days=days, start_arg=start_arg, end_arg=end_arg, period_label=period_label)

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
