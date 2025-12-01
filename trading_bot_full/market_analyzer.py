import threading, time, pandas as pd, logging
from datetime import datetime
from bot import bot_instance; from settings_manager import settings; from strategy import strategy_engine; from models import db_manager, AnalysisResult
logger = logging.getLogger(__name__)
class MarketAnalyzer:
    def __init__(self): self.is_scanning = False; self.progress = 0; self.status_message = "Ready"
    def get_top_tickers(self, limit=100):
        try:
            q = settings.get("scanner_quote_coin")
            return sorted([t for t in bot_instance.get_all_tickers() if t['symbol'].endswith(q)], key=lambda x: float(x.get('turnover24h', 0)), reverse=True)[:limit]
        except: return []
    def fetch_candles(self, s, tf, l=200):
        try:
            bybit_tf = {'15':'15','60':'60','240':'240','D':'D'}.get(str(tf), '240')
            r = bot_instance.session.get_kline(category="linear", symbol=s, interval=bybit_tf, limit=l)
            if r['retCode']==0: 
                df = pd.DataFrame(r['result']['list'], columns=['time','open','high','low','close','vol','to'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for c in ['open','high','low','close','vol']: df[c] = df[c].astype(float)
                return df.sort_values('time').reset_index(drop=True)
        except: pass
        return None
    def run_scan_thread(self):
        if not self.is_scanning: threading.Thread(target=self._scan, daemon=True).start()
    def _scan(self):
        self.is_scanning = True; self.progress = 0; self.status_message = "Starting..."
        session = db_manager.get_session()
        try:
            session.query(AnalysisResult).delete(); session.commit()
            limit = settings.get("scan_limit"); tickers = self.get_top_tickers(limit)
            htf, ltf = settings.get("htfSelection"), settings.get("ltfSelection")
            for i, t in enumerate(tickers):
                self.status_message = f"Scanning {t['symbol']} ({i+1}/{len(tickers)})"
                self.progress = int((i/len(tickers))*100)
                try:
                    df_htf = self.fetch_candles(t['symbol'], htf)
                    if df_htf is None: time.sleep(0.1); continue
                    filters = strategy_engine.check_htf_filters(strategy_engine.calculate_indicators(df_htf).iloc[-1])
                    if not (filters['bull'] or filters['bear']): time.sleep(0.3); continue
                    time.sleep(0.2)
                    df_ltf = self.fetch_candles(t['symbol'], ltf)
                    if df_ltf is None: continue
                    sig = strategy_engine.get_signal(df_ltf, df_htf)
                    if sig['action']:
                        sc = 50 + (20 if filters['bull'] else 0)
                        res = AnalysisResult(symbol=t['symbol'], signal_type=sig['action'], status="Retest", score=sc, price=float(df_ltf['close'].iloc[-1]), htf_rsi=float(filters['details'].get('rsi',0)), ltf_rsi=float(df_ltf['rsi'].iloc[-1] if 'rsi' in df_ltf else 0), details=sig['reason'])
                        session.add(res); session.commit()
                except: pass
                time.sleep(0.5)
            self.progress = 100; self.status_message = "Completed"
        finally: self.is_scanning = False; session.close()
    def get_results(self):
        s = db_manager.get_session()
        try: return [{'symbol': r.symbol, 'signal': r.signal_type, 'score': r.score, 'price': r.price, 'rsi_htf': round(r.htf_rsi,1), 'rsi_ltf': round(r.ltf_rsi,1), 'time': r.found_at.strftime('%H:%M'), 'details': r.details} for r in s.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()]
        finally: s.close()
market_analyzer = MarketAnalyzer()
