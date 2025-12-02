import threading
import time
import pandas as pd
import logging
from datetime import datetime
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult, OrderBlock

# Імпортуємо правильний клас стратегії з правильного файлу
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
            # Сортуємо за об'ємом за 24г
            sorted_tickers = sorted(filtered, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            return sorted_tickers[:int(limit)]
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=300):
        try:
            # Мапінг таймфреймів для Bybit API
            tf_map = {'5': '5', '15': '15', '30': '30', '45': '15', '60': '60', '240': '240', 'D': 'D'}
            req_tf = tf_map.get(str(timeframe), '240')
            
            # Якщо таймфрейм "45", беремо "15" і множимо ліміт, щоб потім зібрати (або просто аналізуємо на 15)
            # Для простоти тут беремо базовий API запит
            
            resp = bot_instance.session.get_kline(
                category="linear", symbol=symbol, interval=req_tf, limit=limit
            )
            
            if resp['retCode'] == 0 and resp['result']['list']:
                data = resp['result']['list']
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Сортуємо: старі -> нові
                return df.sort_values('time').reset_index(drop=True)
            return None
        except: return None

    def run_scan_thread(self):
        if self.is_scanning: return
        threading.Thread(target=self._scan_process, daemon=True).start()

    def _scan_process(self):
        self.is_scanning = True
        self.progress = 0
        self.status_message = "Starting..."
        session = db_manager.get_session()
        
        try:
            # Очищуємо старі результати
            session.query(AnalysisResult).delete()
            session.commit()
            
            limit = settings.get("scan_limit")
            tickers = self.get_top_tickers(limit)
            total = len(tickers)
            
            htf = settings.get("htfSelection")
            ltf = settings.get("ltfSelection")

            for i, ticker in enumerate(tickers):
                if not self.is_scanning: break # Можливість зупинки (якщо реалізовано)
                
                symbol = ticker['symbol']
                self.status_message = f"Scanning {symbol} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                
                try:
                    # 1. Завантаження даних HTF
                    df_htf = self.fetch_candles(symbol, htf)
                    if df_htf is None: 
                        time.sleep(0.1); continue
                    
                    time.sleep(0.1)
                    
                    # 2. Завантаження даних LTF
                    df_ltf = self.fetch_candles(symbol, ltf)
                    if df_ltf is None: continue
                    
                    # 3. Аналіз стратегією
                    signals = strategy_engine.analyze(df_ltf, df_htf)
                    
                    # 4. Збереження результатів
                    for sig in signals:
                        score = 80 # Базовий скор
                        
                        # Формування запису в БД
                        res = AnalysisResult(
                            symbol=symbol, 
                            signal_type=sig.get('action'), 
                            status="New", 
                            score=score, 
                            price=sig.get('price'), 
                            htf_rsi=0.0, # Можна дістати з df_htf якщо треба
                            ltf_rsi=sig.get('rsi', 0), 
                            details=f"{sig.get('reason')} | SL: {round(sig.get('sl_price',0),4)}"
                        )
                        session.add(res)
                        
                        # Якщо використовуємо Smart Money Tracker, можна зберігати блоки в таблицю OrderBlock
                        # (Це опціонально, але корисно для розділу Smart Money)
                        if 'OB' in sig.get('reason', ''):
                            ob_type = sig.get('action') # Buy or Sell
                            new_ob = OrderBlock(
                                symbol=symbol,
                                timeframe=str(ltf),
                                ob_type=ob_type,
                                top=sig.get('price') * 1.01, # Приблизно, бо стратегія повертає сигнал, а не сирий блок тут
                                bottom=sig.get('price') * 0.99,
                                entry_price=sig.get('price'),
                                sl_price=sig.get('sl_price', 0),
                                status='PENDING'
                            )
                            session.add(new_ob)

                        session.commit()
                        logger.info(f"🚀 FOUND: {symbol} {sig.get('action')}")
                
                except Exception as e:
                    # logger.error(f"Scan error {symbol}: {e}")
                    pass
                
                # Пауза, щоб не перевищити ліміти API
                time.sleep(0.2)

            self.progress = 100
            self.status_message = "Scan Completed"
            
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            logger.error(f"Scan failed: {e}")
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