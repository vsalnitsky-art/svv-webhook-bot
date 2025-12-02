import threading
import time
import pandas as pd
import logging
from datetime import datetime
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult

# !!! ВИКОРИСТОВУЄМО НОВУ СТРАТЕГІЮ !!!
from strategy_ob_trend import ob_trend_strategy as strategy_engine

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"
        self._stop_event = threading.Event()

    def get_top_tickers(self, limit=100):
        try:
            quote_coin = settings.get("scanner_quote_coin")
            all_tickers = bot_instance.get_all_tickers()
            filtered = [t for t in all_tickers if t['symbol'].endswith(quote_coin)]
            sorted_tickers = sorted(filtered, key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
            return sorted_tickers[:limit]
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=200):
        try:
            tf_map = {'15': '15', '60': '60', '240': '240', 'D': 'D'}
            bybit_tf = tf_map.get(str(timeframe), '240')
            resp = bot_instance.session.get_kline(category="linear", symbol=symbol, interval=bybit_tf, limit=limit)
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
        self.status_message = "Starting..."
        session = db_manager.get_session()
        try:
            session.query(AnalysisResult).delete()
            session.commit()
            limit = settings.get("scan_limit")
            tickers = self.get_top_tickers(limit)
            total = len(tickers)
            htf = settings.get("htfSelection")
            ltf = settings.get("ltfSelection")

            for i, ticker in enumerate(tickers):
                symbol = ticker['symbol']
                self.status_message = f"Scanning {symbol} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                try:
                    df_htf = self.fetch_candles(symbol, htf)
                    if df_htf is None: 
                        time.sleep(0.1); continue
                    
                    time.sleep(0.2) 
                    df_ltf = self.fetch_candles(symbol, ltf)
                    if df_ltf is None: continue
                    
                    # ВИКЛИК НОВОЇ СТРАТЕГІЇ
                    signals = strategy_engine.analyze(df_ltf, df_htf)
                    
                    for sig in signals:
                        score = 70 
                        res = AnalysisResult(
                            symbol=symbol, 
                            signal_type=sig.get('type'), 
                            status="Signal", 
                            score=score, 
                            price=sig.get('price'), 
                            htf_rsi=sig.get('rsi', 0),
                            ltf_rsi=sig.get('rsi', 0), 
                            details=sig.get('details', 'OB Trend Signal')
                        )
                        session.add(res); session.commit()
                        logger.info(f"🚀 FOUND: {symbol} {sig.get('type')}")
                except: pass
                time.sleep(0.5) 

            self.progress = 100
            self.status_message = "Scan Completed"
        except Exception as e:
            self.status_message = f"Error: {str(e)}"
        finally:
            self.is_scanning = False
            session.close()

    def get_results(self):
        session = db_manager.get_session()
        try:
            res = session.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()
            return [{
                'symbol': r.symbol, 'signal': r.signal_type, 'status': r.status, 'score': r.score, 'price': r.price, 
                'rsi_htf': round(r.htf_rsi, 1), 'rsi_ltf': round(r.ltf_rsi, 1), 
                'time': r.found_at.strftime('%H:%M'), 'details': r.details
            } for r in res]
        finally:
            session.close()

market_analyzer = MarketAnalyzer()