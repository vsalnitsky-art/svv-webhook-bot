import threading, time, pandas as pd, logging
from bot import bot_instance; from settings_manager import settings
from models import db_manager, AnalysisResult, OrderBlock, SmartMoneyTicker 
from strategy_ob_trend import ob_trend_strategy as strategy_engine

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False; self.progress = 0; self.status_message = "Ready"
        # Фоновий потік оновлення статусів зон
        threading.Thread(target=self._monitor_smart_money, daemon=True).start()

    def get_top_tickers(self, limit=100):
        try:
            q = settings.get("scanner_quote_coin")
            return sorted([t for t in bot_instance.get_all_tickers() if t['symbol'].endswith(q)], key=lambda x: float(x.get('turnover24h', 0)), reverse=True)[:int(limit)]
        except: return []

    def fetch_candles(self, s, tf, limit=300):
        try:
            m = {'5':'5','15':'15','30':'30','45':'15','60':'60','240':'240','D':'D'}
            r = bot_instance.session.get_kline(category="linear", symbol=s, interval=m.get(str(tf),'240'), limit=limit)
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
        s = db_manager.get_session()
        try:
            s.query(AnalysisResult).delete(); s.commit()
            limit = settings.get("scan_limit"); tickers = self.get_top_tickers(limit)
            htf, ltf = settings.get("htfSelection"), settings.get("ltfSelection")
            for i, t in enumerate(tickers):
                if not self.is_scanning: break
                sym = t['symbol']; self.status_message = f"Scanning {sym} ({i+1}/{len(tickers)})"; self.progress = int((i/len(tickers))*100)
                try:
                    df_h = self.fetch_candles(sym, htf); time.sleep(0.1)
                    if df_h is None: continue
                    df_l = self.fetch_candles(sym, ltf)
                    if df_l is None: continue
                    sigs = strategy_engine.analyze(df_l, df_h)
                    for sg in sigs:
                        # 1. Результат
                        res = AnalysisResult(symbol=sym, signal_type=sg['action'], status="New", score=85, price=sg['price'], htf_rsi=0, ltf_rsi=sg['rsi'], details=f"{sg['reason']} | SL: {round(sg['sl_price'],4)}")
                        s.add(res)
                        # 2. Watchlist Add
                        if not s.query(SmartMoneyTicker).filter_by(symbol=sym).first():
                            s.add(SmartMoneyTicker(symbol=sym))
                            logger.info(f"🆕 Added {sym} to Watchlist")
                        s.commit()
                except: pass
                time.sleep(0.2)
            self.progress = 100; self.status_message = "Completed"
        finally: self.is_scanning = False; s.close()

    # === UPDATE ZONES STATUSES ===
    def _monitor_smart_money(self):
        while True:
            try:
                session = db_manager.get_session()
                watchlist = session.query(SmartMoneyTicker).all()
                if not watchlist:
                    session.close(); time.sleep(10); continue
                
                htf = settings.get("htfSelection")
                for item in watchlist:
                    sym = item.symbol
                    try:
                        df = self.fetch_candles(sym, htf, limit=300)
                        if df is not None:
                            curr_price = df['close'].iloc[-1]
                            obs = strategy_engine.find_order_blocks(df)
                            
                            # Refresh blocks
                            session.query(OrderBlock).filter_by(symbol=sym).delete()
                            
                            # Helper to add
                            def add_ob(o_list, type_str):
                                for b in o_list:
                                    status = 'WAITING'
                                    # Status Logic
                                    entry = b['top'] if type_str=='Buy' else b['bottom']
                                    sl = b['bottom'] if type_str=='Buy' else b['top']
                                    
                                    # Broken Check
                                    if type_str=='Buy' and curr_price < sl: status = 'BROKEN'
                                    elif type_str=='Sell' and curr_price > sl: status = 'BROKEN'
                                    # Inside Check
                                    elif type_str=='Buy' and b['bottom'] <= curr_price <= b['top']: status = '🔥 INSIDE'
                                    elif type_str=='Sell' and b['bottom'] <= curr_price <= b['top']: status = '🔥 INSIDE'
                                    # Near Check (0.5%)
                                    elif abs(curr_price - entry)/entry < 0.005: status = '⚠️ NEAR'
                                    
                                    session.add(OrderBlock(symbol=sym, timeframe=str(htf), ob_type=type_str, top=b['top'], bottom=b['bottom'], entry_price=entry, sl_price=sl, created_at=b['created_at'], status=status))

                            add_ob(obs['buy'], 'Buy')
                            add_ob(obs['sell'], 'Sell')
                            session.commit()
                    except: pass
                    time.sleep(1.0)
                session.close()
                time.sleep(15)
            except: time.sleep(30)

    def get_results(self):
        s = db_manager.get_session()
        try: return [{'symbol':r.symbol,'signal':r.signal_type,'score':r.score,'price':r.price,'rsi_ltf':round(r.ltf_rsi,1),'time':r.found_at.strftime('%H:%M'),'details':r.details} for r in s.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()]
        finally: s.close()
market_analyzer = MarketAnalyzer()
