import threading
import time
import pandas as pd
import logging
from datetime import datetime
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult, OrderBlock

# Smart Money Strategy
from strategy_ob_trend import ob_trend_strategy as strategy_engine

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"

    def get_top_tickers(self, limit=100):
        try:
            quote_coin = settings.get("scanner_quote_coin")
            all_tickers = bot_instance.get_all_tickers()
            filtered = [t for t in all_tickers if t['symbol'].endswith(quote_coin)]
            sorted_tickers = sorted(filtered, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            return sorted_tickers[:limit]
        except Exception as e:
            logger.error(f"Ticker fetch error: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=300):
        try:
            if str(timeframe) == "45":
                req_tf = "15"
                req_limit = limit * 3 
            else:
                req_tf = str(timeframe)
                req_limit = limit

            tf_map = {'5':'5', '15':'15', '30':'30', '60':'60', '240':'240', 'D':'D'}
            bybit_tf = tf_map.get(req_tf, '240')
            
            resp = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=bybit_tf, limit=req_limit
            )
            
            if resp['retCode'] == 0 and resp['result']['list']:
                data = resp['result']['list']
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                return df.sort_values('time').reset_index(drop=True)
            return None
        except: return None

    def run_scan_thread(self):
        if self.is_scanning: return
        threading.Thread(target=self._scan_process, daemon=True).start()

    def _scan_process(self):
        self.is_scanning = True
        self.progress = 0
        self.status_message = "Scan Started..."
        session = db_manager.get_session()
        
        try:
            # Очищаємо лог аналізу (але НЕ чіпаємо таблицю order_blocks, бо там історія!)
            session.query(AnalysisResult).delete()
            session.commit()
            
            limit = settings.get("scan_limit")
            tickers = self.get_top_tickers(limit)
            total = len(tickers)
            
            htf = settings.get("htfSelection")
            ltf = settings.get("ltfSelection")
            
            # Режим Auto тепер означає "Додавати в Трекер"
            is_auto_mode = settings.get("scanner_mode") == "Auto"

            for i, ticker in enumerate(tickers):
                symbol = ticker['symbol']
                self.status_message = f"Analysing {symbol} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                
                try:
                    df_htf = self.fetch_candles(symbol, htf)
                    if df_htf is None: 
                        time.sleep(0.1); continue
                    
                    time.sleep(0.1)
                    
                    df_ltf = self.fetch_candles(symbol, ltf)
                    if df_ltf is None: continue
                    
                    # === АНАЛІЗ ===
                    signal_data = strategy_engine.analyze(df_ltf, df_htf)
                    
                    if signal_data['action']:
                        sl_price = signal_data.get('sl_price', 0)
                        
                        # 1. Запис в лог сканера (для UI)
                        res = AnalysisResult(
                            symbol=symbol, 
                            signal_type=signal_data['action'], 
                            status="Smart Money Found", 
                            score=85, 
                            price=signal_data.get('price', 0), 
                            htf_rsi=0, ltf_rsi=0, 
                            details=f"Pending: {signal_data['reason']}"
                        )
                        session.add(res)
                        
                        # 2. Якщо режим AUTO -> Додаємо в базу для Трекера
                        if is_auto_mode:
                            # Перевірка, чи вже є такий активний блок, щоб не дублювати
                            existing = session.query(OrderBlock).filter_by(
                                symbol=symbol, 
                                status='PENDING',
                                ob_type=signal_data['action']
                            ).first()
                            
                            if not existing:
                                # Отримуємо межі блоку (приблизно)
                                # У реальній стратегії ми могли б повертати точні top/bottom
                                # Тут беремо SL як орієнтир: 
                                # Для Buy: SL < Entry. Top ~ Entry. Bottom = SL / (1-buff)
                                # Спрощено зберігаємо поточні дані як орієнтир
                                
                                new_ob = OrderBlock(
                                    symbol=symbol,
                                    timeframe=ltf,
                                    ob_type=signal_data['action'],
                                    entry_price=signal_data.get('price'), # Ціна на момент сигналу
                                    sl_price=sl_price,
                                    status='PENDING',
                                    # Умовно: зона десь поруч з ціною
                                    top=signal_data.get('price') * 1.01,
                                    bottom=signal_data.get('price') * 0.99
                                )
                                session.add(new_ob)
                                logger.info(f"🧱 NEW OB SAVED: {symbol} {signal_data['action']}")

                        session.commit()
                            
                except Exception as e:
                    logger.error(f"Scan error {symbol}: {e}")
                
                time.sleep(0.2)

            self.progress = 100
            self.status_message = "Scan Completed"
            
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            logger.error(f"Scan fatal error: {e}")
        finally:
            self.is_scanning = False
            session.close()

    def get_results(self):
        session = db_manager.get_session()
        try:
            res = session.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()
            return [{
                'symbol': r.symbol, 'signal': r.signal_type, 'status': r.status, 'score': r.score, 
                'price': r.price, 'rsi_htf': r.htf_rsi, 'rsi_ltf': r.ltf_rsi, 
                'time': r.found_at.strftime('%H:%M'), 'details': r.details
            } for r in res]
        finally:
            session.close()

market_analyzer = MarketAnalyzer()