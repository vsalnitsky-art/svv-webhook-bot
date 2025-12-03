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
        # Запускаємо фоновий симулятор Smart Money
        threading.Thread(target=self._monitor_smart_money, daemon=True).start()

    def get_top_tickers(self, limit=100):
        """
        Отримує список тікерів з урахуванням фільтру об'єму (Min Vol).
        Захищено від помилок конвертації даних.
        """
        try:
            q = settings.get("scanner_quote_coin")
            # Безпечне отримання налаштувань (bool)
            use_vol_filter = settings.get("scan_use_min_volume")
            if isinstance(use_vol_filter, str):
                use_vol_filter = use_vol_filter.lower() in ['true', 'on', '1']
            elif use_vol_filter is None:
                use_vol_filter = True # Default

            # Отримуємо всі тікери з біржі
            all_tickers = bot_instance.get_all_tickers()
            if not all_tickers:
                logger.warning("⚠️ API returned empty ticker list")
                return []

            # 1. Фільтр по Quote Coin (наприклад, тільки USDT пари)
            usdt_tickers = [t for t in all_tickers if t['symbol'].endswith(q)]
            
            # 2. Фільтр за об'ємом (Volume Filter)
            valid_tickers = []
            
            if use_vol_filter:
                try:
                    min_vol_mln = float(settings.get("scan_min_volume", 10))
                except:
                    min_vol_mln = 10.0 # Fallback default
                
                min_vol_raw = min_vol_mln * 1_000_000
                
                for t in usdt_tickers:
                    try:
                        # Безпечна конвертація Turnover
                        vol_str = t.get('turnover24h', 0)
                        if vol_str is None or vol_str == "":
                            vol = 0.0
                        else:
                            vol = float(vol_str)
                        
                        if vol >= min_vol_raw:
                            valid_tickers.append(t)
                    except Exception:
                        continue # Пропускаємо "биту" монету, а не крашимо весь список
            else:
                valid_tickers = usdt_tickers

            # 3. Сортування за об'ємом (від найбільшого) та ліміт кількості
            sorted_tickers = sorted(valid_tickers, key=lambda x: float(x.get('turnover24h', 0) or 0), reverse=True)
            return sorted_tickers[:int(limit)]

        except Exception as e:
            logger.error(f"❌ Critical Error in get_top_tickers: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=300):
        try:
            m = {'5':'5','15':'15','30':'30','45':'45','60':'60','240':'240','D':'D'}
            req_tf = m.get(str(timeframe), '240')
            if req_tf == '45': req_tf = '15' # Bybit не має 45m, беремо 15m
            
            r = bot_instance.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=limit)
            if r['retCode'] == 0 and r['result']['list']:
                df = pd.DataFrame(r['result']['list'], columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)
                return df.sort_values('time').reset_index(drop=True)
        except: pass
        return None

    # === MANUAL SCANNER (РУЧНИЙ СКАНЕР - DIAGNOSTIC MODE) ===
    def run_scan_thread(self):
        if not self.is_scanning: threading.Thread(target=self._scan_process, daemon=True).start()

    def _scan_process(self):
        self.is_scanning = True
        self.progress = 0
        self.status_message = "🚀 Starting Scan..."
        session = db_manager.get_session()
        try:
            # === ПРИМУСОВЕ ПЕРЕСТВОРЕННЯ ТАБЛИЦІ (Для нової колонки volume) ===
            db_manager.recreate_analysis_table()
            
            limit = settings.get("scan_limit")
            tickers = self.get_top_tickers(limit)
            
            # --- DEBUG LOG ---
            print(f"🔎 DEBUG: Found {len(tickers)} tickers to scan.")
            # -----------------

            htf, ltf = settings.get("htfSelection"), settings.get("ltfSelection")
            
            total = len(tickers)
            for i, t in enumerate(tickers):
                if not self.is_scanning: break
                sym = t['symbol']
                
                # Безпечне отримання об'єму для запису в БД
                try:
                    vol_24h = float(t.get('turnover24h', 0))
                except:
                    vol_24h = 0.0
                
                self.status_message = f"🔍 Analyzing {sym} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                
                try:
                    df_h = self.fetch_candles(sym, htf); time.sleep(0.05)
                    if df_h is None: 
                        print(f"❌ {sym}: No HTF candles")
                        continue
                    
                    df_l = self.fetch_candles(sym, ltf)
                    if df_l is None: 
                        print(f"❌ {sym}: No LTF candles")
                        continue
                    
                    sigs = strategy_engine.analyze(df_l, df_h)
                    
                    # --- DEBUG LOG ---
                    if sigs:
                        print(f"👉 {sym}: Found {len(sigs)} signals.")
                    # -----------------

                    for sg in sigs:
                        res = AnalysisResult(
                            symbol=sym, 
                            signal_type=sg['action'], 
                            status="New", 
                            score=85, 
                            price=sg['price'], 
                            htf_rsi=0, 
                            ltf_rsi=sg['rsi'], 
                            volume_24h=vol_24h, # Зберігаємо об'єм
                            details=f"{sg['reason']} | SL: {round(sg['sl_price'],4)}"
                        )
                        session.add(res)
                        
                        # Додаємо в Watchlist (Ліміт 20)
                        if not session.query(SmartMoneyTicker).filter_by(symbol=sym).first():
                            count = session.query(SmartMoneyTicker).count()
                            if count >= 20:
                                oldest = session.query(SmartMoneyTicker).order_by(SmartMoneyTicker.added_at.asc()).first()
                                if oldest: session.delete(oldest)
                            session.add(SmartMoneyTicker(symbol=sym))
                        
                        session.commit()
                except Exception as e:
                    print(f"❌ Error scanning {sym}: {e}")
                    pass
                
                time.sleep(0.1)
            
            self.progress = 100
            self.status_message = "✅ Scan Completed"
            
        except Exception as e:
            logger.error(f"Global Scan Error: {e}")
            self.status_message = "❌ Error during scan"
        finally: 
            self.is_scanning = False
            session.close()

    # === BACKGROUND SIMULATOR (SMART MONEY ENGINE) ===
    def _monitor_smart_money(self):
        logger.info("🧠 Smart Money Simulator Started")
        while True:
            try:
                session = db_manager.get_session()
                
                # --- PHASE 1: TRADE MANAGEMENT (СУПРОВІД) ---
                active_trades = session.query(PaperTrade).filter(PaperTrade.status.in_(['OPEN', 'PENDING'])).all()
                for trade in active_trades:
                    current_price = bot_instance.get_price(trade.symbol)
                    if current_price == 0: continue
                    
                    self.status_message = f"⚡ Monitoring Trade: {trade.symbol}"

                    # 1. PENDING (LIMIT) LOGIC
                    if trade.status == 'PENDING':
                        triggered = False
                        if trade.direction == 'Long' and current_price <= trade.entry_price: triggered = True
                        if trade.direction == 'Short' and current_price >= trade.entry_price: triggered = True
                        
                        sl_hit = False
                        if trade.direction == 'Long' and current_price <= trade.sl_price: sl_hit = True
                        if trade.direction == 'Short' and current_price >= trade.sl_price: sl_hit = True
                        
                        if sl_hit:
                            trade.status = 'CANCELED'
                            trade.closed_at = datetime.utcnow()
                            trade.details = "SL hit before Entry"
                        elif triggered:
                            trade.status = 'OPEN'
                            trade.created_at = datetime.utcnow()
                            trade.details = "Limit Triggered"

                    # 2. OPEN TRADE LOGIC
                    elif trade.status == 'OPEN':
                        pnl = 0.0
                        if trade.entry_price > 0:
                            if trade.direction == 'Long':
                                pnl = (current_price - trade.entry_price) / trade.entry_price * 100
                            else:
                                pnl = (trade.entry_price - current_price) / trade.entry_price * 100
                        
                        trade.pnl_percent = pnl
                        trade.pnl = pnl

                        # Check SL
                        is_sl = False
                        if trade.direction == 'Long' and current_price <= trade.sl_price: is_sl = True
                        if trade.direction == 'Short' and current_price >= trade.sl_price: is_sl = True
                        
                        # Check TP
                        is_tp = False
                        if trade.tp_price:
                            if trade.direction == 'Long' and current_price >= trade.tp_price: is_tp = True
                            if trade.direction == 'Short' and current_price <= trade.tp_price: is_tp = True
                        
                        if is_sl:
                            trade.status = 'CLOSED_LOSS'
                            trade.exit_price = trade.sl_price
                            trade.closed_at = datetime.utcnow()
                            trade.details = "Stop Loss"
                        elif is_tp:
                            trade.status = 'CLOSED_WIN'
                            trade.exit_price = trade.tp_price
                            trade.closed_at = datetime.utcnow()
                            trade.details = "Take Profit"
                
                session.commit()

                # --- PHASE 2: SEEKING NEW ENTRIES (ПОШУК ВХОДУ) ---
                if not self.is_scanning:
                    watchlist = session.query(SmartMoneyTicker).all()
                    if watchlist:
                        for item in watchlist:
                            sym = item.symbol
                            self.status_message = f"👀 Checking {sym} for Setup..."
                            
                            existing = session.query(PaperTrade).filter(
                                PaperTrade.symbol == sym, 
                                PaperTrade.status.in_(['OPEN', 'PENDING'])
                            ).first()
                            
                            if existing: continue

                            try:
                                htf = settings.get("htfSelection")
                                ltf = settings.get("ltfSelection")
                                df_h = self.fetch_candles(sym, htf, limit=100)
                                if df_h is None: continue
                                
                                # HTF Check (Global Trend)
                                df_h = strategy_engine.calculate_indicators(df_h)
                                last_h = df_h.iloc[-1]
                                
                                is_bull = True; is_bear = True
                                if settings.get('obt_useCloudFilter'):
                                    if last_h['hma_fast'] <= last_h['hma_slow']: is_bull = False
                                    if last_h['hma_fast'] >= last_h['hma_slow']: is_bear = False
                                if settings.get('obt_useRsiFilter'):
                                    if last_h['rsi'] > 55: is_bull = False
                                    if last_h['rsi'] < 45: is_bear = False
                                
                                if not is_bull and not is_bear: 
                                    self.status_message = f"❌ {sym}: Filter Rejected"
                                    time.sleep(0.2)
                                    continue

                                # LTF Check (Order Block)
                                df_l = self.fetch_candles(sym, ltf, limit=100)
                                if df_l is None: continue
                                obs = strategy_engine.find_order_blocks(df_l)
                                last_candle_time = df_l.iloc[-1]['time']
                                
                                trade_signal = None
                                sl_buffer_pct = float(settings.get("sm_sl_buffer", 0.2)) / 100
                                
                                if is_bull and obs['buy']:
                                    best_ob = obs['buy'][-1]
                                    if (last_candle_time - best_ob['created_at']) < timedelta(minutes=int(ltf)*5):
                                        trade_signal = {'dir': 'Long', 'ob': best_ob, 'sl': best_ob['bottom'] * (1 - sl_buffer_pct)}
                                elif is_bear and obs['sell']:
                                    best_ob = obs['sell'][-1]
                                    if (last_candle_time - best_ob['created_at']) < timedelta(minutes=int(ltf)*5):
                                        trade_signal = {'dir': 'Short', 'ob': best_ob, 'sl': best_ob['top'] * (1 + sl_buffer_pct)}

                                if trade_signal:
                                    self.status_message = f"💎 ENTRY FOUND: {sym} {trade_signal['dir']}"
                                    
                                    entry_mode = settings.get("sm_entry_mode", "Market")
                                    current_price = df_l.iloc[-1]['close']
                                    
                                    entry_price = current_price
                                    if entry_mode == 'Limit':
                                        entry_price = trade_signal['ob']['top'] if trade_signal['dir'] == 'Long' else trade_signal['ob']['bottom']
                                    
                                    tp_mode = settings.get("sm_tp_mode", "None")
                                    tp_val = float(settings.get("sm_tp_value", 3.0))
                                    tp_price = None
                                    dist_to_sl = abs(entry_price - trade_signal['sl'])
                                    
                                    if tp_mode == "Fixed":
                                        tp_price = entry_price * (1 + tp_val/100) if trade_signal['dir'] == 'Long' else entry_price * (1 - tp_val/100)
                                    elif tp_mode == "RR":
                                        tp_price = entry_price + (dist_to_sl * tp_val) if trade_signal['dir'] == 'Long' else entry_price - (dist_to_sl * tp_val)

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
                                    
                                    # ВИДАЛЕННЯ З WATCHLIST
                                    ticker_remove = session.query(SmartMoneyTicker).filter_by(symbol=sym).first()
                                    if ticker_remove:
                                        session.delete(ticker_remove)

                                    session.commit()
                                    time.sleep(1)

                            except Exception as e:
                                pass
                            
                            time.sleep(0.5)
                        
                        self.status_message = "💤 Waiting for next cycle..."
                    else:
                        self.status_message = "⚠️ Watchlist Empty. Run Scanner."

                session.close()
                time.sleep(5 if self.is_scanning else 30)
                
            except Exception as e:
                logger.error(f"SM Monitor Global Error: {e}")
                time.sleep(30)

    def get_results(self):
        s = db_manager.get_session()
        try: 
            return [
                {
                    'symbol': r.symbol,
                    'signal': r.signal_type,
                    'score': r.score,
                    'price': r.price,
                    'rsi_ltf': round(r.ltf_rsi, 1),
                    'volume': r.volume_24h, # Повертаємо об'єм для шаблону
                    'time': r.found_at.strftime('%H:%M'),
                    'details': r.details
                } 
                for r in s.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()
            ]
        finally: 
            s.close()

market_analyzer = MarketAnalyzer()