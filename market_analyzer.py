import threading
import time
import pandas as pd
import logging
from datetime import datetime
from bot import bot_instance
from settings_manager import settings
from strategy import strategy_engine
from models import db_manager, AnalysisResult

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"
        self.last_scan_time = None
        self._stop_event = threading.Event()

    def get_top_tickers(self, limit=100):
        """Отримує список пар, відфільтрованих по валюті та відсортованих по об'єму"""
        try:
            quote_coin = settings.get("scanner_quote_coin")
            all_tickers = bot_instance.get_all_tickers()
            
            # Фільтр по USDT/USDC
            filtered = [t for t in all_tickers if t['symbol'].endswith(quote_coin)]
            
            # Сортування по обороту (turnover24h)
            sorted_tickers = sorted(filtered, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            
            return sorted_tickers[:limit]
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=200):
        """Завантажує свічки з Bybit"""
        try:
            tf_map = {'15': '15', '60': '60', '240': '240', 'D': 'D'}
            bybit_tf = tf_map.get(str(timeframe), '240')
            
            resp = bot_instance.session.get_kline(
                category="linear",
                symbol=symbol,
                interval=bybit_tf,
                limit=limit
            )
            
            if resp['retCode'] == 0 and resp['result']['list']:
                # Bybit: [time, open, high, low, close, volume, turnover]
                data = resp['result']['list']
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['time'] = pd.to_numeric(df['time'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = df[col].astype(float)
                
                # Сортуємо від старого до нового
                df = df.sort_values('time').reset_index(drop=True)
                return df
            return None
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return None

    def run_scan_thread(self):
        if self.is_scanning: return
        threading.Thread(target=self._scan_process, daemon=True).start()

    def _scan_process(self):
        self.is_scanning = True
        self.progress = 0
        self.status_message = "Starting..."
        
        session = db_manager.get_session()
        try:
            # 1. Очистка старих результатів
            session.query(AnalysisResult).delete()
            session.commit()
            
            # 2. Отримання списку
            limit = settings.get("scan_limit")
            tickers = self.get_top_tickers(limit)
            total = len(tickers)
            
            self.status_message = f"Found {total} pairs. Scanning..."
            logger.info(f"Starting scan for {total} pairs")
            
            htf = settings.get("htfSelection")
            ltf = settings.get("ltfSelection")

            for i, ticker in enumerate(tickers):
                symbol = ticker['symbol']
                self.status_message = f"Scanning {symbol} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                
                try:
                    # А. Старший ТФ (фільтри)
                    df_htf = self.fetch_candles(symbol, htf)
                    if df_htf is None: 
                        time.sleep(0.1)
                        continue
                    
                    df_htf = strategy_engine.calculate_indicators(df_htf)
                    filters = strategy_engine.check_htf_filters(df_htf.iloc[-1])
                    
                    if not (filters['bull'] or filters['bear']):
                        # === ЗБІЛЬШЕНА ПАУЗА ===
                        time.sleep(0.3) 
                        continue 
                    
                    # Б. Молодший ТФ (вхід)
                    # === ПАУЗА МІЖ ЗАПИТАМИ ===
                    time.sleep(0.2) 
                    df_ltf = self.fetch_candles(symbol, ltf)
                    if df_ltf is None: continue
                    
                    # В. Аналіз
                    signal = strategy_engine.get_signal(df_ltf, df_htf)
                    
                    if signal['action']:
                        # Знайдено сигнал
                        score_base = 50
                        if filters['bull'] and signal['action'] == 'Buy': score_base += 20
                        if filters['bear'] and signal['action'] == 'Sell': score_base += 20
                        
                        res = AnalysisResult(
                            symbol=symbol,
                            signal_type=signal['action'],
                            status="Zone Retest", 
                            score=score_base, 
                            price=float(df_ltf['close'].iloc[-1]),
                            htf_rsi=float(filters['details'].get('rsi', 0)),
                            ltf_rsi=float(df_ltf['rsi'].iloc[-1] if 'rsi' in df_ltf else 0),
                            details=signal['reason']
                        )
                        session.add(res)
                        session.commit()
                        logger.info(f"🚀 FOUND: {symbol} {signal['action']}")
                
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                
                # === ГЛАВНА ПАУЗА: Збільшено до 0.5 сек, щоб уникнути бану ===
                time.sleep(0.5) 

            self.progress = 100
            self.status_message = "Scan Completed"
            self.last_scan_time = datetime.now()
            
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