"""
Main Application - Full Version
Включает: Gemini AI, Telegram Alert, Windows Anti-Sleep, Self-Ping (5 min)
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
from datetime import datetime, timedelta
# Импортируйте ваши модули
from bot_config import config
from models import db_manager
from statistics_service import stats_service
from scanner import EnhancedMarketScanner
from config import get_api_credentials

# === НОВЫЙ ИМПОРТ (AI) ===
try:
    import ai_analyst
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ ai_analyst.py не найден. AI функции отключены.")

# === WINDOWS NO-SLEEP (БЛОКИРОВКА СПЯЩЕГО РЕЖИМА) ===
# Только для Windows - на Linux/Mac не нужно
try:
    import platform
    if platform.system() == 'Windows':
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002 | 0x00000001)
        print("☕ Режим 'Без сна' для Windows активирован успешно.")
    else:
        print("ℹ️ Linux/Mac обнаружен - режим без сна не требуется")
except Exception as e:
    print(f"ℹ️ Не удалось активировать режим без сна: {e}")

# === FLASK APP ===
app = Flask(__name__)

# Логирование
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === TELEGRAM SENDER ===
def send_telegram_message(text):
    """Отправка уведомления в Telegram"""
    try:
        # 👇👇 ВСТАВЬТЕ СЮДА ВАШИ ДАННЫЕ 👇👇
        tg_token = "ВАШ_ТОКЕН_БОТА"  # Пример: "123456:ABC-DEF..."
        chat_id = "ВАШ_CHAT_ID"      # Пример: "987654321"
        
        # Если данные в конфиге, пробуем взять оттуда
        if hasattr(config, 'TG_BOT_TOKEN') and config.TG_BOT_TOKEN:
            tg_token = config.TG_BOT_TOKEN
        if hasattr(config, 'TG_CHAT_ID') and config.TG_CHAT_ID:
            chat_id = config.TG_CHAT_ID

        if "ВАШ_" in tg_token or "ВАШ_" in str(chat_id):
            logger.warning("⚠️ Telegram не настроен! Впишите токен в main_app.py")
            return

        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    except Exception as e:
        logger.error(f"❌ Telegram Error: {e}")

# === AI PROCESSOR ===
def process_signal_with_ai(data):
    """Цепочка: Анализ -> Телеграм -> Торговля"""
    symbol = data.get('symbol')
    action = data.get('action')
    
    # 1. Спрашиваем Gemini
    ai_text = "AI модуль недоступен."
    if AI_AVAILABLE:
        try:
            logger.info(f"🧠 Gemini Analyzing {symbol}...")
            ai_text = ai_analyst.analyze_signal(symbol, action)
        except Exception as e:
            logger.error(f"AI Error: {e}")
            ai_text = "Ошибка при анализе."

    # 2. Шлем отчет
    msg = (
        f"🚀 <b>СИГНАЛ: {symbol}</b>\n"
        f"Действие: <b>{action}</b>\n"
        f"-------------------\n"
        f"🤖 <b>Мнение Gemini:</b>\n"
        f"{ai_text}"
    )
    send_telegram_message(msg)

    # 3. Торгуем
    bot.place_order(data)

# === KEEP ALIVE (САМО-ПИНГ 5 МИНУТ) ===
def keep_alive():
    """Бот пингует сам себя каждые 5 минут чтобы Render не усыплял"""
    logger.info("⏰ Система само-пинга запущена")
    time.sleep(10) # Даем серверу время на полный старт
    
    # ✅ ВНЕШНИЙ URL - Render считает это активностью!
    # Замените на ваш реальный домен
    external_url = os.environ.get('RENDER_EXTERNAL_URL', 'https://svv-webhook-bot.onrender.com') + '/health'
    
    while True:
        try:
            # Пингуем ВНЕШНИЙ URL, не localhost!
            response = requests.get(external_url, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"💓 PING (5min): Бот в норме. Статус: 200 OK")
            else:
                logger.warning(f"⚠️ PING STRANGE: Код ответа {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ PING ERROR: Сервер не отвечает: {e}")
            
        # Ждем ровно 5 минут (300 секунд) перед следующей проверкой
        time.sleep(300)

keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()

# === BOT CLASS ===
class BybitTradingBot:
    """Основной класс бота (без изменений логики)"""
    
    def __init__(self):
        try:
            api_key, api_secret = get_api_credentials()
            self.session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
            logger.info("✅ Bybit API session initialized.")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации API: {e}")
            raise
    
    def get_available_balance(self, currency="USDT"):
        try:
            balance_info = self.session.get_wallet_balance(accountType="UNIFIED")
            if balance_info.get('retCode') != 0:
                return None
            for account in balance_info.get('result', {}).get('list', []):
                for coin in account.get('coin', []):
                    if coin.get('coin') == currency:
                        return float(coin.get('walletBalance', 0))
            return None
        except Exception as e:
            logger.error(f"❌ Error balance: {e}")
            return None
    
    def normalize_symbol(self, symbol):
        return symbol.replace('.P', '')
    
    def get_all_tickers(self):
        try:
            resp = self.session.get_tickers(category="linear")
            if resp['retCode'] == 0:
                return resp['result']['list']
            return []
        except Exception as e:
            logger.error(f"Ticker fetch error: {e}")
            return []
    
    def get_current_price(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            resp = self.session.get_tickers(category="linear", symbol=norm)
            if resp.get('retCode') == 0 and resp['result']['list']:
                return float(resp['result']['list'][0]['lastPrice'])
            return None
        except:
            return None
    
    def set_leverage(self, symbol, leverage):
        try:
            norm = self.normalize_symbol(symbol)
            self.session.set_leverage(
                category="linear",
                symbol=norm,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage)
            )
            return True
        except Exception as e:
            if "110043" in str(e):
                return True
            return False
    
    def get_instrument_info(self, symbol):
        try:
            norm = self.normalize_symbol(symbol)
            resp = self.session.get_instruments_info(category="linear", symbol=norm)
            if resp.get('retCode') == 0:
                return (
                    resp['result']['list'][0]['lotSizeFilter'],
                    resp['result']['list'][0]['priceFilter']
                )
            return None, None
        except:
            return None, None
    
    def round_qty(self, qty, step):
        if step <= 0:
            return qty
        import decimal
        step_decimals = abs(decimal.Decimal(str(step)).as_tuple().exponent)
        return round(qty // step * step, step_decimals)
    
    def round_price(self, price, tick_size):
        if tick_size <= 0:
            return price
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
        except:
            return 0.0
    
    def get_pnl_stats(self, days=7):
        logger.info(f"📊 Fetching PnL stats for last {days} days...")
        try:
            trades = stats_service.get_trades(days=days)
            if not trades:
                self.sync_trades_from_bybit(days)
                trades = stats_service.get_trades(days=days)
            
            stats = {
                "total_trades": len(trades),
                "total_pnl": 0.0,
                "total_volume": 0.0,
                "win_trades": 0,
                "loss_trades": 0,
                "long_trades": 0,
                "short_trades": 0,
                "details": [],
                "chart_labels": [],
                "chart_data": [],
                "coin_performance": {}
            }
            
            cumulative_pnl = 0.0
            for trade in trades:
                pnl = trade['pnl']
                stats["total_pnl"] += pnl
                cumulative_pnl += pnl
                if pnl > 0: stats["win_trades"] += 1
                else: stats["loss_trades"] += 1
                if trade['side'] == "Long": stats["long_trades"] += 1
                else: stats["short_trades"] += 1
                
                if trade['exit_time']:
                    date_str = datetime.strptime(trade['exit_time'], '%Y-%m-%d %H:%M').strftime('%m-%d')
                    stats["chart_labels"].append(date_str)
                    stats["chart_data"].append(round(cumulative_pnl, 2))
                
                symbol = trade['symbol']
                if symbol not in stats["coin_performance"]:
                    stats["coin_performance"][symbol] = 0.0
                stats["coin_performance"][symbol] += pnl
                stats["details"].append(trade)
            
            sorted_coins = sorted(stats["coin_performance"].items(), key=lambda x: x[1], reverse=True)
            stats["top_coins_labels"] = [x[0] for x in sorted_coins[:5]]
            stats["top_coins_values"] = [round(x[1], 2) for x in sorted_coins[:5]]
            return stats, None
        except Exception as e:
            logger.error(f"❌ Stat Error: {e}")
            return None, str(e)
    
    def sync_trades_from_bybit(self, days=7):
        logger.info("🔄 Syncing trades from Bybit...")
        try:
            now = datetime.now()
            all_trades = []
            for i in range(0, days, 7):
                chunk_days = min(7, days - i)
                end_dt = now - timedelta(days=i)
                start_dt = end_dt - timedelta(days=chunk_days)
                ts_end = int(end_dt.timestamp() * 1000)
                ts_start = int(start_dt.timestamp() * 1000)
                
                resp = self.session.get_closed_pnl(category="linear", startTime=ts_start, endTime=ts_end, limit=100)
                if resp['retCode'] == 0:
                    all_trades.extend(resp['result']['list'])
                time.sleep(0.1)
            
            for trade in all_trades:
                api_side = trade['side']
                real_side = "Long" if api_side == "Sell" else "Short"
                entry_time = datetime.fromtimestamp(int(trade['createdTime']) / 1000)
                exit_time = datetime.fromtimestamp(int(trade['updatedTime']) / 1000)
                duration = (exit_time - entry_time).total_seconds() / 60
                
                trade_data = {
                    'order_id': trade['orderId'], 'symbol': trade['symbol'], 'side': real_side,
                    'qty': float(trade['qty']), 'entry_price': float(trade['avgEntryPrice']),
                    'exit_price': float(trade['avgExitPrice']), 'pnl': float(trade['closedPnl']),
                    'entry_time': entry_time, 'exit_time': exit_time, 'duration_minutes': int(duration)
                }
                stats_service.save_trade(trade_data)
            logger.info(f"✅ Synced {len(all_trades)} trades to database")
        except Exception as e:
            logger.error(f"❌ Sync error: {e}")
    
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
            logger.error(f"🔥 Order Execution Error: {e}")
            return {"status": "error"}

# Инициализация
bot = BybitTradingBot()
scanner = EnhancedMarketScanner(bot, config.get_scanner_config())
scanner.start()

# === BREAKEVEN MONITOR ===
class BreakevenMonitor:
    def __init__(self, bot_instance):
        self.bot = bot_instance
        self.running = True
    def start(self): threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while self.running:
            try: self.check_positions()
            except: pass
            time.sleep(config.MONITOR_INTERVAL)
    def check_positions(self):
        positions = self.bot.session.get_positions(category="linear", settleCoin="USDT")
        if positions['retCode'] != 0: return
        for pos in positions['result']['list']:
            if float(pos['size']) == 0: continue
            entry = float(pos['avgPrice'])
            sl = float(pos.get('stopLoss', 0))
            side = pos['side']
            if (side == "Buy" and sl >= entry) or (side == "Sell" and sl > 0 and sl <= entry): continue
            orders = self.bot.session.get_open_orders(category="linear", symbol=pos['symbol'])
            tps = sum(1 for o in orders.get('result', {}).get('list', []) if o['reduceOnly'])
            if tps > 0 and tps < 3:
                try:
                    self.bot.session.set_trading_stop(category="linear", symbol=pos['symbol'], stopLoss=str(entry), positionIdx=0)
                    logger.info(f"🛡️ Moved SL to Breakeven for {pos['symbol']}")
                except: pass

monitor = BreakevenMonitor(bot)
monitor.start()

# === WEB ROUTES ===
@app.route('/scanner', methods=['GET'])
def scanner_page():
    data = scanner.get_aggregated_data(hours=24)
    last_update = data['last_scan'].strftime('%H:%M:%S') if data['last_scan'] else "Запуск..."
    
    html_template = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whale Scanner 🐋</title>
        <meta http-equiv="refresh" content="30">
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --green: #22c55e; --red: #ef4444; --blue: #3b82f6; }
            body { font-family: sans-serif; background: var(--bg); color: var(--text); padding: 20px; margin: 0; }
            .container { max-width: 1200px; margin: 0 auto; }
            .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
            .card { background: var(--card); padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h1 { margin: 0; font-size: 24px; }
            .badge { background: var(--blue); padding: 5px 10px; border-radius: 20px; font-size: 12px; font-weight: bold; }
            
            .split-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
            @media (max-width: 768px) { .split-container { grid-template-columns: 1fr; } }
            
            .scroll-box { max-height: 400px; overflow-y: auto; padding-right: 5px; }
            .scroll-box::-webkit-scrollbar { width: 6px; }
            
            .mini-table { width: 100%; border-collapse: collapse; }
            .mini-table th { text-align: left; color: #94a3b8; padding: 10px; position: sticky; top: 0; background: #1e293b; z-index: 10; border-bottom: 2px solid #334155; }
            .mini-table td { padding: 12px 5px; border-bottom: 1px solid #334155; vertical-align: middle; }
            
            .coin-row-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
            .coin-name { font-weight: bold; font-size: 1.1em; }
            .coin-change { font-weight: bold; }
            
            .bar-container { height: 6px; background: #334155; border-radius: 3px; width: 100%; overflow: hidden; }
            .bar-fill { height: 100%; border-radius: 3px; transition: width 0.5s ease; }
            .bar-green { background: #22c55e; }
            .bar-red { background: #ef4444; }
            
            .inflow-val { font-size: 0.9em; color: #cbd5e1; margin-top: 2px; }
            
            #pumpsTable { width: 100%; border-collapse: collapse; }
            #pumpsTable th { text-align: left; color: #94a3b8; padding: 12px 10px; border-bottom: 1px solid #334155; cursor: pointer; }
            #pumpsTable td { padding: 14px 10px; border-bottom: 1px solid #334155; }
            tr.huge-move { background: rgba(34, 197, 94, 0.05); }
            .inflow { font-weight: bold; color: #facc15; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div>
                    <h1>🐋 Whale Scanner (>$1M/min)</h1>
                    <span class="badge" style="margin-top: 5px; display: inline-block;">Last Scan: {{ last_update }}</span>
                </div>
                <a href="/report" style="background: #334155; padding: 10px 20px; border-radius: 8px; color: #fff; text-decoration: none; font-weight: bold;">📊 Звіт P&L</a>
            </div>
            
            <div class="split-container">
                <div class="card">
                    <h3 style="color: #22c55e; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-top:0;">
                        📈 Зелена Зона ({{ positive_coins|length }})
                    </h3>
                    <div class="scroll-box">
                        <table class="mini-table">
                            <thead>
                                <tr><th>Монета</th><th>Вливання (24г)</th></tr>
                            </thead>
                            <tbody>
                                {% for coin in positive_coins %}
                                <tr>
                                    <td width="40%">
                                        <div class="coin-name">{{ coin.symbol }}</div>
                                        <div class="coin-change" style="color: #22c55e;">+{{ coin.avg_change }}%</div>
                                    </td>
                                    <td>
                                        <div class="inflow-val">${{ "{:,.0f}".format(coin.inflow) }}</div>
                                        <div class="bar-container"><div class="bar-fill bar-green" style="width: {{ coin.bar_pct }}%;"></div></div>
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>

                <div class="card">
                    <h3 style="color: #ef4444; border-bottom: 1px solid #334155; padding-bottom: 10px; margin-top:0;">
                        📉 Червона Зона ({{ negative_coins|length }})
                    </h3>
                    <div class="scroll-box">
                        <table class="mini-table">
                            <thead>
                                <tr><th>Монета</th><th>Вливання (24г)</th></tr>
                            </thead>
                            <tbody>
                                {% for coin in negative_coins %}
                                <tr>
                                    <td width="40%">
                                        <div class="coin-name">{{ coin.symbol }}</div>
                                        <div class="coin-change" style="color: #ef4444;">{{ coin.avg_change }}%</div>
                                    </td>
                                    <td>
                                        <div class="inflow-val">${{ "{:,.0f}".format(coin.inflow) }}</div>
                                        <div class="bar-container"><div class="bar-fill bar-red" style="width: {{ coin.bar_pct }}%;"></div></div>
                                    </td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>🔥 Лог подій (Всі сигнали)</h3>
                {% if pumps %}
                <table id="pumpsTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Час ⇅</th>
                            <th onclick="sortTable(1)">Монета ⇅</th>
                            <th onclick="sortTable(2)">Ціна ⇅</th>
                            <th onclick="sortTable(3)">Зміна (1хв) ⇅</th>
                            <th onclick="sortTable(4)">Аномалія (x) ⇅</th>
                            <th onclick="sortTable(5)">Вливання ($) ⇅</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for p in pumps %}
                        <tr class="{{ 'huge-move' if p.vol_inflow > 5000000 else '' }}">
                            <td>{{ p.time }}</td>
                            <td class="symbol">{{ p.symbol }}</td>
                            <td>{{ p.price }}</td>
                            <td style="color: {{ '#22c55e' if p.price_change_interval >= 0 else '#ef4444' }}">
                                {{ "+" if p.price_change_interval > 0 }}{{ p.price_change_interval }}%
                            </td>
                            <td class="pump-factor">x{{ p.spike_factor }}</td>
                            <td class="inflow">${{ "{:,.0f}".format(p.vol_inflow) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                    <div class="empty-msg">🐋 Великі гравці поки що сплять.</div>
                {% endif %}
            </div>
        </div>
        <script>
        function sortTable(n) {
          var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
          table = document.getElementById("pumpsTable");
          switching = true;
          dir = "desc"; 
          while (switching) {
            switching = false;
            rows = table.rows;
            for (i = 1; i < (rows.length - 1); i++) {
              shouldSwitch = false;
              x = rows[i].getElementsByTagName("TD")[n];
              y = rows[i + 1].getElementsByTagName("TD")[n];
              let xVal = x.innerText.toLowerCase().replace(/[$,x%+,]/g, "").trim();
              let yVal = y.innerText.toLowerCase().replace(/[$,x%+,]/g, "").trim();
              if (n === 0) {} else if (!isNaN(parseFloat(xVal)) && isFinite(xVal)) { xVal = parseFloat(xVal); yVal = parseFloat(yVal); }
              if (dir == "asc") { if (xVal > yVal) { shouldSwitch = true; break; } } 
              else if (dir == "desc") { if (xVal < yVal) { shouldSwitch = true; break; } }
            }
            if (shouldSwitch) {
              rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
              switching = true;
              switchcount ++;      
            } else {
              if (switchcount == 0 && dir == "desc") { dir = "asc"; switching = true; }
            }
          }
        }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, pumps=data['all_signals'], last_update=last_update,
                                  positive_coins=data['positive_coins'], negative_coins=data['negative_coins'])

@app.route('/report', methods=['GET'])
def report_page():
    """P&L Report страница"""
    days = request.args.get('days', default=7, type=int)
    stats, error = bot.get_pnl_stats(days=days)
    
    if error:
        return f"<h1>Ошибка</h1><p>{error}</p>"
    
    # HTML шаблон отчёта
    html_template = """
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>P&L Report</title>
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --green: #22c55e; --red: #ef4444; --blue: #3b82f6; }
            body { font-family: sans-serif; background: var(--bg); color: var(--text); padding: 20px; margin: 0; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: var(--card); padding: 20px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
            h1 { margin: 0 0 20px 0; font-size: 24px; }
            .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
            .stat-box { background: #334155; padding: 15px; border-radius: 8px; text-align: center; }
            .stat-value { font-size: 28px; font-weight: bold; margin: 5px 0; }
            .stat-label { font-size: 14px; color: #94a3b8; }
            .positive { color: var(--green); }
            .negative { color: var(--red); }
            table { width: 100%; border-collapse: collapse; margin-top: 15px; }
            th { text-align: left; color: #94a3b8; padding: 12px; border-bottom: 2px solid #334155; }
            td { padding: 12px; border-bottom: 1px solid #334155; }
            .btn { display: inline-block; padding: 8px 16px; margin: 5px; background: #3b82f6; color: white; text-decoration: none; border-radius: 6px; }
            .btn:hover { background: #2563eb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <h1>📊 P&L Report ({{ days }} днів)</h1>
                
                <div style="margin-bottom: 20px;">
                    <a href="/report?days=1" class="btn">1 день</a>
                    <a href="/report?days=7" class="btn">7 днів</a>
                    <a href="/report?days=30" class="btn">30 днів</a>
                    <a href="/scanner" class="btn" style="background: #334155;">← Сканер</a>
                </div>
                
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-label">Загальний P&L</div>
                        <div class="stat-value {{ 'positive' if stats.total_pnl >= 0 else 'negative' }}">
                            {{ '+' if stats.total_pnl >= 0 else '' }}${{ "{:,.2f}".format(stats.total_pnl) }}
                        </div>
                    </div>
                    
                    <div class="stat-box">
                        <div class="stat-label">Загальні угоди</div>
                        <div class="stat-value">{{ stats.total_trades }}</div>
                    </div>
                    
                    <div class="stat-box">
                        <div class="stat-label">Win Rate</div>
                        <div class="stat-value {{ 'positive' if stats.win_rate >= 50 else 'negative' }}">
                            {{ "{:.1f}".format(stats.win_rate) }}%
                        </div>
                    </div>
                    
                    <div class="stat-box">
                        <div class="stat-label">Прибуткові</div>
                        <div class="stat-value positive">{{ stats.profitable_trades }}</div>
                    </div>
                    
                    <div class="stat-box">
                        <div class="stat-label">Збиткові</div>
                        <div class="stat-value negative">{{ stats.losing_trades }}</div>
                    </div>
                    
                    <div class="stat-box">
                        <div class="stat-label">Avg P&L</div>
                        <div class="stat-value">
                            ${{ "{:,.2f}".format(stats.avg_pnl) }}
                        </div>
                    </div>
                </div>
            </div>
            
            {% if stats.top_performers %}
            <div class="card">
                <h2>🏆 Топ монет</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Монета</th>
                            <th>Угод</th>
                            <th>Win Rate</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for coin in stats.top_performers %}
                        <tr>
                            <td><strong>{{ coin.symbol }}</strong></td>
                            <td>{{ coin.trades }}</td>
                            <td class="{{ 'positive' if coin.win_rate >= 50 else 'negative' }}">
                                {{ "{:.1f}".format(coin.win_rate) }}%
                            </td>
                            <td class="{{ 'positive' if coin.pnl >= 0 else 'negative' }}">
                                {{ '+' if coin.pnl >= 0 else '' }}${{ "{:,.2f}".format(coin.pnl) }}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
            
            {% if stats.recent_trades %}
            <div class="card">
                <h2>📜 Останні угоди</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Час</th>
                            <th>Монета</th>
                            <th>Side</th>
                            <th>Ціна входу</th>
                            <th>Ціна виходу</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for trade in stats.recent_trades %}
                        <tr>
                            <td>{{ trade.closed_at.strftime('%d.%m %H:%M') }}</td>
                            <td><strong>{{ trade.symbol }}</strong></td>
                            <td>{{ trade.side }}</td>
                            <td>${{ "{:,.2f}".format(trade.entry_price) }}</td>
                            <td>${{ "{:,.2f}".format(trade.exit_price) }}</td>
                            <td class="{{ 'positive' if trade.pnl >= 0 else 'negative' }}">
                                {{ '+' if trade.pnl >= 0 else '' }}${{ "{:,.2f}".format(trade.pnl) }}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """
    
    return render_template_string(html_template, stats=stats, days=days)

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        raw_data = request.get_data(as_text=True)
        logger.info(f"📨 RAW WEBHOOK DATA: {raw_data}")
        data = None
        try: data = json.loads(raw_data)
        except:
            match = re.search(r'\{.*\}', raw_data, re.DOTALL)
            if match: data = json.loads(match.group())
        
        if not data:
            logger.error("❌ Failed to find JSON")
            return jsonify({"error": "No valid JSON"}), 400
        
        logger.info(f"🚀 Processing: {data.get('symbol')} {data.get('action')}")
        
        # ЗАПУСК ЧЕРЕЗ AI ФУНКЦИЮ
        threading.Thread(target=process_signal_with_ai, args=(data,)).start()
        
        return jsonify({"status": "processing"})
    except Exception as e:
        logger.error(f"🔥 Webhook Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home(): return "Bot Active with Gemini & DB"

@app.route('/health', methods=['GET'])
def health(): return jsonify({"status": "ok", "database": "connected", "ai": AI_AVAILABLE})

@app.route('/api/stats/top-coins', methods=['GET'])
def api_top_coins():
    coins = stats_service.get_top_coins(period_hours=request.args.get('hours', 24, int), limit=20)
    return jsonify(coins)

@app.route('/api/stats/coin-performance', methods=['GET'])
def api_coin_performance():
    return jsonify(stats_service.get_coin_performance())

if __name__ == '__main__':
    stats_service.cleanup_old_data(days=config.DATA_RETENTION_DAYS)
    app.run(host=config.HOST, port=config.PORT)
