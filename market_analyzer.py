import threading
import time
import pandas as pd
import logging
from datetime import datetime
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult

# !!! ВАЖЛИВО: Використовуємо єдину стратегію !!!
from strategy import strategy_engine 

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"

    def get_top_tickers(self, limit=100):
        """Отримує список найліквідніших пар"""
        try:
            quote_coin = settings.get("scanner_quote_coin")
            all_tickers = bot_instance.get_all_tickers()
            # Фільтруємо тільки потрібні пари (наприклад, USDT)
            filtered = [t for t in all_tickers if t['symbol'].endswith(quote_coin)]
            # Сортуємо за об'ємом торгів (turnover24h)
            sorted_tickers = sorted(filtered, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            return sorted_tickers[:limit]
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=300):
        """Завантажує історичні дані"""
        try:
            # Мапінг таймфреймів для Bybit API
            tf_map = {'5':'5', '15':'15', '30':'30', '60':'60', '240':'240', 'D':'D'}
            bybit_tf = tf_map.get(str(timeframe), '240')
            
            resp = bot_instance.session.get_kline(
                category="linear", 
                symbol=symbol, 
                interval=bybit_tf, 
                limit=limit
            )
            
            if resp['retCode'] == 0 and resp['result']['list']:
                data = resp['result']['list']
                columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
                df = pd.DataFrame(data, columns=columns)
                
                # Конвертація типів
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Сортування: старі -> нові
                return df.sort_values('time').reset_index(drop=True)
            return None
        except: return None

    def run_scan_thread(self):
        """Запуск сканування в окремому потоці"""
        if self.is_scanning: return
        threading.Thread(target=self._scan_process, daemon=True).start()

    def _scan_process(self):
        self.is_scanning = True
        self.progress = 0
        self.status_message = "Starting..."
        
        session = db_manager.get_session()
        try:
            # Очищення попередніх результатів
            session.query(AnalysisResult).delete()
            session.commit()
            
            # Отримання налаштувань
            limit = settings.get("scan_limit")
            tickers = self.get_top_tickers(limit)
            total = len(tickers)
            
            htf = settings.get("htfSelection")
            ltf = settings.get("ltfSelection")

            for i, ticker in enumerate(tickers):
                symbol = ticker['symbol']
                
                # Оновлення статусу для UI
                self.status_message = f"Scanning {symbol} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                
                try:
                    # 1. Завантаження HTF (старший ТФ)
                    df_htf = self.fetch_candles(symbol, htf)
                    if df_htf is None: 
                        time.sleep(0.1); continue
                    
                    time.sleep(0.1) # Rate limit protection
                    
                    # 2. Завантаження LTF (молодший ТФ)
                    df_ltf = self.fetch_candles(symbol, ltf)
                    if df_ltf is None: continue
                    
                    # 3. АНАЛІЗ ЧЕРЕЗ СТРАТЕГІЮ
                    # Strategy повертає словник: {'action': 'Buy', 'reason': '...', 'sl_price': 123.45}
                    signal_data = strategy_engine.get_signal(df_ltf, df_htf)
                    
                    if signal_data['action']:
                        score = 80 # Базовий рейтинг сигналу
                        sl_info = f" | SL: {round(signal_data.get('sl_price', 0), 4)}" if signal_data.get('sl_price') else ""
                        
                        res = AnalysisResult(
                            symbol=symbol, 
                            signal_type=signal_data['action'], 
                            status="Signal", 
                            score=score, 
                            price=signal_data.get('price', 0), 
                            htf_rsi=0, # Можна додати, якщо стратегія поверне це
                            ltf_rsi=0, 
                            details=f"{signal_data['reason']}{sl_info}"
                        )
                        session.add(res)
                        session.commit()
                        logger.info(f"🚀 FOUND: {symbol} {signal_data['action']}")
                
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                
                time.sleep(0.2) # Пауза між монетами

            self.progress = 100
            self.status_message = "Scan Completed"
            
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
            logger.error(f"Scan failed: {e}")
        finally:
            self.is_scanning = False
            session.close()

    def get_results(self):
        """Отримати результати для відображення в UI"""
        session = db_manager.get_session()
        try:
            res = session.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()
            return [{
                'symbol': r.symbol, 
                'signal': r.signal_type, 
                'status': r.status, 
                'score': r.score, 
                'price': r.price, 
                'rsi_htf': round(r.htf_rsi, 1), 
                'rsi_ltf': round(r.ltf_rsi, 1), 
                'time': r.found_at.strftime('%H:%M'), 
                'details': r.details
            } for r in res]
        finally:
            session.close()

market_analyzer = MarketAnalyzer()