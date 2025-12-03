import threading
import time
import pandas as pd
import logging
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult, OrderBlock, SmartMoneyTicker # Додано імпорт

from strategy_ob_trend import ob_trend_strategy as strategy_engine

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"

    def get_top_tickers(self, limit=100):
        try:
            q = settings.get("scanner_quote_coin")
            return sorted([t for t in bot_instance.get_all_tickers() if t['symbol'].endswith(q)], key=lambda x: float(x.get('turnover24h', 0)), reverse=True)[:int(limit)]
        except Exception as e:
            logger.error(f"Error fetching tickers: {e}")
            return []

    def fetch_candles(self, symbol, timeframe, limit=300):
        try:
            m = {'5':'5','15':'15','30':'30','45':'15','60':'60','240':'240','D':'D'}
            r = bot_instance.session.get_kline(category="linear", symbol=symbol, interval=m.get(str(timeframe),'240'), limit=limit)
            if r['retCode']==0: 
                df = pd.DataFrame(r['result']['list'], columns=['time','open','high','low','close','vol','to'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
                return df.sort_values('time').reset_index(drop=True)
        except: pass
        return None

    def run_scan_thread(self):
        if not self.is_scanning: threading.Thread(target=self._scan_process, daemon=True).start()

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
                if not self.is_scanning: break 
                
                symbol = ticker['symbol']
                self.status_message = f"Scanning {symbol} ({i+1}/{total})"
                self.progress = int((i / total) * 100)
                
                try:
                    df_htf = self.fetch_candles(symbol, htf)
                    if df_htf is None: 
                        time.sleep(0.1); continue
                    
                    time.sleep(0.1)
                    df_ltf = self.fetch_candles(symbol, ltf)
                    if df_ltf is None: continue
                    
                    signals = strategy_engine.analyze(df_ltf, df_htf)
                    
                    for sig in signals:
                        # 1. Запис результату для Сканера
                        res = AnalysisResult(
                            symbol=symbol, 
                            signal_type=sig.get('action'), 
                            status="New", 
                            score=85, 
                            price=sig.get('price'), 
                            htf_rsi=0.0, 
                            ltf_rsi=sig.get('rsi', 0), 
                            details=f"{sig.get('reason')} | SL: {round(sig.get('sl_price',0),4)}"
                        )
                        session.add(res)
                        
                        # 2. WATCHLIST LOGIC: Перевіряємо та додаємо в список Smart Money
                        existing_ticker = session.query(SmartMoneyTicker).filter_by(symbol=symbol).first()
                        
                        if not existing_ticker:
                            new_ticker = SmartMoneyTicker(symbol=symbol)
                            session.add(new_ticker)
                            logger.info(f"🆕 Added to Watchlist: {symbol}")
                        
                        # (Опціонально) Залишаємо запис в OrderBlock для сумісності, якщо треба
                        if 'OB' in sig.get('reason', ''):
                             session.add(OrderBlock(
                                symbol=symbol, timeframe=str(ltf), ob_type=sig.get('action'),
                                top=sig.get('price')*1.01, bottom=sig.get('price')*0.99,
                                entry_price=sig.get('price'), sl_price=sig.get('sl_price', 0)
                            ))

                        session.commit()
                
                except Exception as e:
                    pass
                
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