import threading
import time
import pandas as pd
import logging
from bot import bot_instance
from settings_manager import settings
from models import db_manager, AnalysisResult, OrderBlock, SmartMoneyTicker
from strategy_ob_trend import ob_trend_strategy as strategy_engine

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    def __init__(self):
        self.is_scanning = False
        self.progress = 0
        self.status_message = "Ready"
        # Запускаємо фоновий моніторинг Smart Money окремо від сканера
        threading.Thread(target=self._monitor_smart_money, daemon=True).start()

    def get_top_tickers(self, limit=100):
        try:
            q = settings.get("scanner_quote_coin")
            return sorted([t for t in bot_instance.get_all_tickers() if t['symbol'].endswith(q)], key=lambda x: float(x.get('turnover24h', 0)), reverse=True)[:int(limit)]
        except: return []

    def fetch_candles(self, symbol, timeframe, limit=300):
        try:
            m = {'5':'5','15':'15','30':'30','45':'15','60':'60','240':'240','D':'D'}
            req_tf = m.get(str(timeframe), '240')
            # Для 45m беремо 15m і покладаємось на стратегію або просто беремо 15m (спрощено)
            # Тут використовуємо прямий мапінг для стабільності
            
            r = bot_instance.session.get_kline(category="linear", symbol=symbol, interval=req_tf, limit=limit)
            if r['retCode'] == 0 and r['result']['list']:
                df = pd.DataFrame(r['result']['list'], columns=['time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['time'] = pd.to_datetime(pd.to_numeric(df['time']), unit='ms')
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)
                return df.sort_values('time').reset_index(drop=True)
        except: pass
        return None

    # === ОСНОВНИЙ СКАНЕР (ПОШУК НОВИХ) ===
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
                    df_h = self.fetch_candles(sym, htf); time.sleep(0.1)
                    if df_h is None: continue
                    df_l = self.fetch_candles(sym, ltf)
                    if df_l is None: continue
                    
                    sigs = strategy_engine.analyze(df_l, df_h)
                    for sg in sigs:
                        # 1. Результат для сканера
                        res = AnalysisResult(symbol=sym, signal_type=sg['action'], status="New", score=85, price=sg['price'], htf_rsi=0, ltf_rsi=sg['rsi'], details=f"{sg['reason']} | SL: {round(sg['sl_price'],4)}")
                        session.add(res)
                        
                        # 2. Додаємо в Watchlist, якщо ще немає
                        if not session.query(SmartMoneyTicker).filter_by(symbol=sym).first():
                            session.add(SmartMoneyTicker(symbol=sym))
                            logger.info(f"🆕 Watchlist Add: {sym}")
                        
                        session.commit()
                except: pass
                time.sleep(0.2)
            self.progress = 100; self.status_message = "Completed"
        finally: self.is_scanning = False; session.close()

    # === SMART MONEY MONITOR (ФОНОВЕ ОНОВЛЕННЯ ЗОН) ===
    def _monitor_smart_money(self):
        """Постійно оновлює зони Order Blocks для монет з Watchlist"""
        logger.info("🕵️ Smart Money Monitor Started")
        while True:
            try:
                session = db_manager.get_session()
                watchlist = session.query(SmartMoneyTicker).all()
                
                if not watchlist:
                    session.close()
                    time.sleep(10)
                    continue

                htf = settings.get("htfSelection") # Шукаємо блоки на Глобальному ТФ (або можна на LTF)
                # Для точності візьмемо той самий ТФ, що в налаштуваннях
                
                for item in watchlist:
                    sym = item.symbol
                    try:
                        df = self.fetch_candles(sym, htf, limit=300)
                        if df is not None:
                            # Шукаємо блоки
                            obs = strategy_engine.find_order_blocks(df)
                            
                            # Очищаємо старі блоки для цієї монети (щоб не було дублікатів і пробитих)
                            # Це найпростіший спосіб тримати базу актуальною
                            session.query(OrderBlock).filter_by(symbol=sym).delete()
                            
                            # Записуємо актуальні Buy блоки
                            for b in obs['buy']:
                                session.add(OrderBlock(
                                    symbol=sym, timeframe=str(htf), ob_type="Buy",
                                    top=b['top'], bottom=b['bottom'], 
                                    entry_price=b['top'], sl_price=b['bottom'], # Entry зверху блоку
                                    created_at=b['created_at']
                                ))
                            
                            # Записуємо актуальні Sell блоки
                            for b in obs['sell']:
                                session.add(OrderBlock(
                                    symbol=sym, timeframe=str(htf), ob_type="Sell",
                                    top=b['top'], bottom=b['bottom'],
                                    entry_price=b['bottom'], sl_price=b['top'], # Entry знизу блоку
                                    created_at=b['created_at']
                                ))
                            
                            session.commit()
                    except Exception as e:
                        # logger.error(f"SM Monitor error {sym}: {e}")
                        pass
                    
                    time.sleep(1.5) # Пауза між монетами, щоб не спамити API
                
                session.close()
                time.sleep(30) # Пауза після повного кола
                
            except Exception as e:
                logger.error(f"SM Loop Error: {e}")
                time.sleep(30)

    def get_results(self):
        s = db_manager.get_session()
        try: return [{'symbol':r.symbol,'signal':r.signal_type,'score':r.score,'price':r.price,'rsi_ltf':round(r.ltf_rsi,1),'time':r.found_at.strftime('%H:%M'),'details':r.details} for r in s.query(AnalysisResult).order_by(AnalysisResult.score.desc()).all()]
        finally: s.close()

market_analyzer = MarketAnalyzer()