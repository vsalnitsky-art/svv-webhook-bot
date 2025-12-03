import threading
import time
import pandas as pd
import logging
from datetime import datetime, timedelta
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult, OrderBlock, SmartMoneyTicker, PaperTrade
from strategy_ob_trend import ob_trend_strategy as strategy_engine

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"
        # Запускаємо фоновий симулятор
        threading.Thread(target=self._monitor_smart_money, daemon=True).start()

    def get_top_tickers(self, limit=100):
        try:
            q = settings.get("scanner_quote_coin")
            return sorted([t for t in bot_instance.get_all_tickers() if t['symbol'].endswith(q)], key=lambda x: float(x.get('turnover24h', 0)), reverse=True)[:int(limit)]
        except: return []

    def fetch_candles(self, symbol, timeframe, limit=300):
        try:
            m = {'5':'5','15':'15','30':'30','45':'45','60':'60','240':'240','D':'D'}
            # Bybit API mapping check
            req_tf = m.get(str(timeframe), '240')
            if req_tf == '45': req_tf = '15' # Fallback if specific api issues, but usually fine. Let's try direct.
            
            r = bot_instance.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=limit)
            if r['retCode'] == 0 and r['result']['list']:
                df = pd.DataFrame(r['result']['list'], columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)
                return df.sort_values('time').reset_index(drop=True)
        except: pass
        return None

    # === СКАНЕР ДЛЯ РУЧНОГО ЗАПУСКУ ===
    def run_scan_thread(self):
        if not self.is_scanning: threading.Thread(target=self._scan_process, daemon=True).start()

    def _scan_process(self):
        self.is_scanning = True; self.progress = 0; self.status_message = "Starting..."
        session = db_manager.get_session()
        try:
            session.query(AnalysisResult).delete(); session.commit()
            limit = settings.get("scan_limit"); tickers = self.get_top_tickers(limit)
            htf, ltf = settings.get("htfSelection"), settings.get("ltfSelection")
            
            total = len(tickers)
            for i, t in enumerate(tickers):
                if not self.is_scanning: break
                sym = t['symbol']
                self.status_message = f"Scanning {sym} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                
                try:
                    df_h = self.fetch_candles(sym, htf); time.sleep(0.05)
                    if df_h is None: continue
                    df_l = self.fetch_candles(sym, ltf)
                    if df_l is None: continue
                    
                    sigs = strategy_engine.analyze(df_l, df_h)
                    for sg in sigs:
                        res = AnalysisResult(symbol=sym, signal_type=sg['action'], status="New", score=85, price=sg['price'], htf_rsi=0, ltf_rsi=sg['rsi'], details=f"{sg['reason']} | SL: {round(sg['sl_price'],4)}")
                        session.add(res)
                        
                        # Авто-додавання в Watchlist
                        if not session.query(SmartMoneyTicker).filter_by(symbol=sym).first():
                            session.add(SmartMoneyTicker(symbol=sym))
                        
                        session.commit()
                except: pass
                time.sleep(0.1)
            self.progress = 100; self.status_message = "Completed"
        finally: self.is_scanning = False; session.close()

    # === ГОЛОВНИЙ ЦИКЛ СИМУЛЯЦІЇ ===
    def _monitor_smart_money(self):
        """
        1. Оновлює Paper Trades (перевіряє SL/TP).
        2. Шукає нові входи для монет з Watchlist.
        """
        logger.info("🧠 Smart Money Simulator Started")
        while True:
            try:
                session = db_manager.get_session()
                
                # --- ЧАСТИНА 1: МЕНЕДЖМЕНТ ПОЗИЦІЙ ---
                active_trades = session.query(PaperTrade).filter(PaperTrade.status.in_(['OPEN', 'PENDING'])).all()
                
                for trade in active_trades:
                    current_price = bot_instance.get_price(trade.symbol)
                    if current_price == 0: continue

                    # 1.1 Обробка PENDING (Limit Mode)
                    if trade.status == 'PENDING':
                        triggered = False
                        # Якщо ціна торкнулася входу
                        if trade.direction == 'Long' and current_price <= trade.entry_price: triggered = True
                        if trade.direction == 'Short' and current_price >= trade.entry_price: triggered = True
                        
                        # Якщо ціна пішла проти нас і пробила SL до входу - скасовуємо
                        sl_hit = False
                        if trade.direction == 'Long' and current_price <= trade.sl_price: sl_hit = True
                        if trade.direction == 'Short' and current_price >= trade.sl_price: sl_hit = True

                        if sl_hit:
                            trade.status = 'CANCELED'
                            trade.closed_at = datetime.utcnow()
                            trade.details = "SL hit before Entry"
                        elif triggered:
                            trade.status = 'OPEN'
                            trade.created_at = datetime.utcnow() # Reset time to entry
                            trade.details = "Limit Triggered"

                    # 1.2 Обробка OPEN
                    elif trade.status == 'OPEN':
                        # Розрахунок PnL
                        if trade.direction == 'Long':
                            pnl = (current_price - trade.entry_price) / trade.entry_price * 100
                        else:
                            pnl = (trade.entry_price - current_price) / trade.entry_price * 100
                        
                        trade.pnl_percent = pnl
                        trade.pnl = pnl # Спрощено, в %

                        # Перевірка SL
                        is_sl = False
                        if trade.direction == 'Long' and current_price <= trade.sl_price: is_sl = True
                        if trade.direction == 'Short' and current_price >= trade.sl_price: is_sl = True
                        
                        # Перевірка TP
                        is_tp = False
                        if trade.tp_price:
                            if trade.direction == 'Long' and current_price >= trade.tp_price: is_tp = True
                            if trade.direction == 'Short' and current_price <= trade.tp_price: is_tp = True
                        
                        if is_sl:
                            trade.status = 'CLOSED_LOSS'
                            trade.exit_price = trade.sl_price # Slippage ігноруємо
                            trade.closed_at = datetime.utcnow()
                            trade.details = "Stop Loss"
                        elif is_tp:
                            trade.status = 'CLOSED_WIN'
                            trade.exit_price = trade.tp_price
                            trade.closed_at = datetime.utcnow()
                            trade.details = "Take Profit"

                session.commit()

                # --- ЧАСТИНА 2: ПОШУК НОВИХ ВХОДІВ ---
                watchlist = session.query(SmartMoneyTicker).all()
                if not watchlist:
                    session.close()
                    time.sleep(10)
                    continue

                htf = settings.get("htfSelection") 
                ltf = settings.get("ltfSelection")
                
                # Налаштування входу
                entry_mode = settings.get("sm_entry_mode", "Limit")
                sl_buffer_pct = float(settings.get("sm_sl_buffer", 0.2)) / 100
                tp_mode = settings.get("sm_tp_mode", "None")
                tp_val = float(settings.get("sm_tp_value", 3.0))

                for item in watchlist:
                    sym = item.symbol
                    
                    # Перевірка: чи є вже активна угода по цій монеті?
                    existing = session.query(PaperTrade).filter(
                        PaperTrade.symbol == sym, 
                        PaperTrade.status.in_(['OPEN', 'PENDING'])
                    ).first()
                    
                    if existing: continue # Один символ - одна угода

                    try:
                        # 1. Отримуємо дані
                        df_h = self.fetch_candles(sym, htf, limit=100)
                        if df_h is None: continue
                        
                        # 2. Фільтри HTF (Тільки тренд)
                        # Розраховуємо індикатори
                        df_h = strategy_engine.calculate_indicators(df_h)
                        last_h = df_h.iloc[-1]
                        
                        is_bull = True; is_bear = True
                        # Cloud
                        if settings.get('obt_useCloudFilter'):
                            if last_h['hma_fast'] <= last_h['hma_slow']: is_bull = False
                            if last_h['hma_fast'] >= last_h['hma_slow']: is_bear = False
                        # RSI
                        if settings.get('obt_useRsiFilter'):
                            if last_h['rsi'] > 55: is_bull = False # Входимо в лонг тільки якщо є простір
                            if last_h['rsi'] < 45: is_bear = False
                        
                        if not is_bull and not is_bear: continue

                        # 3. Пошук блоків на LTF
                        df_l = self.fetch_candles(sym, ltf, limit=100)
                        if df_l is None: continue
                        
                        obs = strategy_engine.find_order_blocks(df_l)
                        
                        # Чи є СВІЖИЙ блок? (останні 3 свічки)
                        last_candle_time = df_l.iloc[-1]['time']
                        
                        trade_signal = None
                        
                        if is_bull and obs['buy']:
                            best_ob = obs['buy'][-1] # Останній
                            # Перевіряємо свіжість (блок створений недавно)
                            if (last_candle_time - best_ob['created_at']) < timedelta(minutes=int(ltf)*5):
                                trade_signal = {
                                    'dir': 'Long', 'ob': best_ob, 
                                    'sl': best_ob['bottom'] * (1 - sl_buffer_pct)
                                }
                        
                        elif is_bear and obs['sell']:
                            best_ob = obs['sell'][-1]
                            if (last_candle_time - best_ob['created_at']) < timedelta(minutes=int(ltf)*5):
                                trade_signal = {
                                    'dir': 'Short', 'ob': best_ob,
                                    'sl': best_ob['top'] * (1 + sl_buffer_pct)
                                }

                        # 4. Створення угоди
                        if trade_signal:
                            current_price = df_l.iloc[-1]['close']
                            
                            # Ціна входу
                            entry_price = current_price
                            if entry_mode == "Limit":
                                # Для Long вхід від верху блоку, для Short від низу
                                if trade_signal['dir'] == 'Long': entry_price = trade_signal['ob']['top']
                                else: entry_price = trade_signal['ob']['bottom']
                            
                            # Тейк профіт
                            tp_price = None
                            dist_to_sl = abs(entry_price - trade_signal['sl'])
                            
                            if tp_mode == "Fixed":
                                pct = tp_val / 100
                                if trade_signal['dir'] == 'Long': tp_price = entry_price * (1 + pct)
                                else: tp_price = entry_price * (1 - pct)
                            elif tp_mode == "RR":
                                if trade_signal['dir'] == 'Long': tp_price = entry_price + (dist_to_sl * tp_val)
                                else: tp_price = entry_price - (dist_to_sl * tp_val)

                            # Запис
                            new_trade = PaperTrade(
                                symbol=sym,
                                direction=trade_signal['dir'],
                                entry_mode=entry_mode,
                                status='PENDING' if entry_mode == 'Limit' else 'OPEN',
                                entry_price=entry_price,
                                sl_price=trade_signal['sl'],
                                tp_price=tp_price,
                                details=f"Found on {ltf}m"
                            )
                            session.add(new_trade)
                            session.commit()
                            logger.info(f"✨ New Paper Trade: {sym} {trade_signal['dir']}")

                    except Exception as e:
                        # logger.error(f"Loop error {sym}: {e}")
                        pass
                    
                    time.sleep(0.5) # Anti-spam API
                
                session.close()
                time.sleep(30) # Пауза циклу
                
            except Exception as e:
                logger.error(f"SM Monitor Global Error: {e}")
                time.sleep(30)

    def get_results(self):
        s = db_manager.get_session()
        try: return [{'symbol':r.symbol,'signal':r.signal_type,'score':r.score,'price':r.price,'rsi_ltf':round(r.ltf_rsi,1),'time':r.found_at.strftime('%H:%M'),'details':r.details} for r in s.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()]
        finally: s.close()

market_analyzer = MarketAnalyzer()