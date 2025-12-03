import os

# Назва кореневої папки
PROJECT_ROOT = "trading_bot_final"

# Створення папок
if not os.path.exists(PROJECT_ROOT):
    os.makedirs(PROJECT_ROOT)

templates_dir = os.path.join(PROJECT_ROOT, "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

# === ВМІСТ ФАЙЛІВ (ОНОВЛЕНО 03.12.2025 - Table View) ===
files = {
    # ---------------------------------------------------------
    # 1. КОНФІГУРАЦІЯ
    # ---------------------------------------------------------
    "requirements.txt": """flask
requests
pybit
sqlalchemy
pandas
pandas_ta
cryptography
gunicorn
""",

    "bot_config.py": """import os
class Config:
    PORT = int(os.environ.get("PORT", 10000))
    HOST = "0.0.0.0"
    DEFAULT_RISK_PERCENT = 5.0
    DEFAULT_LEVERAGE = 20
    DEFAULT_TP_PERCENT = 0.0
    DEFAULT_SL_PERCENT = 0.0
    SCANNER_INTERVAL = 5
    DATA_RETENTION_DAYS = 30
    @classmethod
    def get_scanner_config(cls): return {'SCANNER_INTERVAL': cls.SCANNER_INTERVAL}
config = Config()
""",

    "config.py": """import os
from cryptography.fernet import Fernet
def get_api_credentials():
    encryption_key = os.environ.get('ENCRYPTION_KEY')
    if encryption_key:
        try:
            cipher = Fernet(encryption_key.encode())
            k = os.environ.get('BYBIT_API_KEY_ENCRYPTED')
            s = os.environ.get('BYBIT_API_SECRET_ENCRYPTED')
            if k and s: return cipher.decrypt(k.encode()).decode(), cipher.decrypt(s.encode()).decode()
        except: pass
    return os.environ.get('BYBIT_API_KEY', ''), os.environ.get('BYBIT_API_SECRET', '')
API_KEY, API_SECRET = get_api_credentials()
""",

    # ---------------------------------------------------------
    # 2. БАЗА ДАНИХ
    # ---------------------------------------------------------
    "models.py": """from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10)) 
    qty = Column(Float); entry_price = Column(Float); exit_price = Column(Float); pnl = Column(Float)
    is_win = Column(Boolean); exit_time = Column(DateTime, default=datetime.utcnow); exit_reason = Column(String(100))

class TradeMonitorLog(Base):
    __tablename__ = 'trade_monitor_logs'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow); current_price = Column(Float); current_pnl = Column(Float)
    rsi = Column(Float); pressure = Column(Float)

class BotSetting(Base):
    __tablename__ = 'bot_settings'
    key = Column(String(50), primary_key=True); value = Column(String(255))

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), index=True)
    signal_type = Column(String(10)); status = Column(String(50)); score = Column(Integer)
    price = Column(Float); htf_rsi = Column(Float); ltf_rsi = Column(Float)
    found_at = Column(DateTime, default=datetime.utcnow); details = Column(Text)

class OrderBlock(Base):
    __tablename__ = 'order_blocks'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), index=True)
    timeframe = Column(String(10)); ob_type = Column(String(10))
    top = Column(Float); bottom = Column(Float); entry_price = Column(Float); sl_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    # Status: WAITING, NEAR, INSIDE, BROKEN
    status = Column(String(20), default='WAITING') 
    volume_score = Column(Float, default=0.0)

class SmartMoneyTicker(Base):
    __tablename__ = 'smart_money_tickers'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)

class DatabaseManager:
    def __init__(self, db_filename='trading_bot_final.db'):
        db_url = os.environ.get('DATABASE_URL')
        if db_url and db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_folder = os.path.join(current_dir, 'BASE')
            try: os.makedirs(base_folder, exist_ok=True); db_path = os.path.join(base_folder, db_filename)
            except OSError: db_path = os.path.join(current_dir, db_filename)
            db_url = f'sqlite:///{db_path}'
        self.engine = create_engine(db_url, echo=False)
        try: Base.metadata.create_all(self.engine)
        except: pass
        self.Session = sessionmaker(bind=self.engine)
    def get_session(self): return self.Session()
db_manager = DatabaseManager()
""",

    "settings_manager.py": """import logging
from models import db_manager, BotSetting
logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "scanner_quote_coin": "USDT", "scanner_mode": "Manual", "scan_limit": 100,
    "telegram_enabled": False, "telegram_bot_token": "", "telegram_chat_id": "",
    "obt_useCloudFilter": True, "obt_useObvFilter": True, "obt_useRsiFilter": True, 
    "obt_useOBRetest": False, 
    "htfSelection": "240", "ltfSelection": "45",
    "obt_cloudFastLen": 10, "obt_cloudSlowLen": 40, "obt_rsiLength": 14,
    "obt_entryRsiOversold": 45, "obt_entryRsiOverbought": 55, "obt_obvEntryLen": 20, "obt_swingLength": 5,
    "exit_enableStrategy": True, "exit_rsiOverbought": 70, "exit_rsiOversold": 30, "exit_obvLength": 10,
    "riskPercent": 2.0, "leverage": 20, "tp_mode": "Fixed_1_50", "fixedTP": 3.0,
    "sl_mode": "OB_Extremity", "fixedSL": 1.5, "obBufferPercent": 0.2,
}

class SettingsManager:
    def __init__(self):
        self.db = db_manager; self._cache = {}; self.reload_settings()
    def _cast_value(self, key, value_str):
        if key not in DEFAULT_SETTINGS: return value_str
        default = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default, bool): return str(value_str).lower() in ['true', 'on', '1']
            elif isinstance(default, int): return int(float(value_str))
            elif isinstance(default, float): return float(value_str)
            else: return str(value_str)
        except: return default
    def reload_settings(self):
        session = self.db.get_session()
        try:
            db_s = session.query(BotSetting).all()
            if not db_s:
                self._cache = DEFAULT_SETTINGS.copy()
                for k, v in DEFAULT_SETTINGS.items():
                    val = "true" if v is True else "false" if v is False else str(v)
                    session.add(BotSetting(key=k, value=val))
                session.commit()
            else:
                loaded = {}
                db_keys = set()
                for s in db_s: loaded[s.key] = self._cast_value(s.key, s.value); db_keys.add(s.key)
                missing = set(DEFAULT_SETTINGS.keys()) - db_keys
                if missing:
                    for k in missing:
                        v = DEFAULT_SETTINGS[k]; val = "true" if v is True else "false" if v is False else str(v)
                        session.add(BotSetting(key=k, value=val)); loaded[k] = v
                    session.commit()
                self._cache = DEFAULT_SETTINGS.copy(); self._cache.update(loaded)
        except: self._cache = DEFAULT_SETTINGS.copy()
        finally: session.close()
    def save_settings(self, new_settings):
        session = self.db.get_session()
        try:
            for k, v in new_settings.items():
                val = str(v)
                if k in DEFAULT_SETTINGS:
                    if isinstance(DEFAULT_SETTINGS[k], bool):
                        val = "true" if (v=='on' or v is True) else "false"
                        self._cache[k] = (val == "true")
                    else:
                        self._cache[k] = self._cast_value(k, v); val = str(v)
                else: self._cache[k] = v
                ex = session.query(BotSetting).filter_by(key=k).first()
                if ex: ex.value = val
                else: session.add(BotSetting(key=k, value=val))
            session.commit()
        except: session.rollback()
        finally: session.close()
    def get_all(self): return self._cache.copy()
    def get(self, key, default=None): return self._cache.get(key, default if default is not None else DEFAULT_SETTINGS.get(key))
    def import_settings(self, json_data):
        session = self.db.get_session()
        try:
            for k, v in json_data.items():
                val = str(v).lower() if isinstance(v, bool) else str(v)
                self._cache[k] = v
                ex = session.query(BotSetting).filter_by(key=k).first()
                if ex: ex.value = val
                else: session.add(BotSetting(key=k, value=val))
            session.commit(); return True
        except: return False
        finally: session.close()
settings = SettingsManager()
""",

    "statistics_service.py": """from models import db_manager, Trade, TradeMonitorLog
from datetime import datetime, timedelta
from sqlalchemy import desc
class StatisticsService:
    def __init__(self): self.db = db_manager
    def save_trade(self, d):
        s = self.db.get_session()
        try:
            if d.get('order_id') and s.query(Trade).filter_by(order_id=d['order_id']).first(): return
            t = Trade(order_id=d.get('order_id'), symbol=d['symbol'], side=d['side'], qty=d.get('qty',0), entry_price=d.get('entry_price',0), exit_price=d.get('exit_price',0), pnl=d.get('pnl',0), is_win=d.get('pnl',0)>0, exit_time=d.get('exit_time'), exit_reason=d.get('exit_reason'))
            s.add(t); s.commit()
        except: pass
        finally: s.close()
    def save_monitor_log(self, data):
        s = self.db.get_session()
        try:
            l = TradeMonitorLog(symbol=data['symbol'], timestamp=datetime.utcnow(), current_price=data['price'], current_pnl=data['pnl'], rsi=data['rsi'], pressure=data['pressure'])
            s.add(l); s.commit()
        except: pass
        finally: s.close()
    def get_trades(self, days=7):
        s = self.db.get_session()
        try:
            d = datetime.utcnow() - timedelta(days=days)
            return [t.__dict__ for t in s.query(Trade).filter(Trade.exit_time >= d).order_by(desc(Trade.exit_time)).all()]
        finally: s.close()
stats_service = StatisticsService()
""",

    # ---------------------------------------------------------
    # 3. ЛОГІКА: СТРАТЕГІЯ, СКАНЕР, БОТ
    # ---------------------------------------------------------
    "strategy_ob_trend.py": """import pandas_ta as ta
import pandas as pd
import numpy as np
from settings_manager import settings

class OBTrendStrategy:
    def __init__(self): pass
    def _get_param(self, key, default=None):
        val = settings.get(key); return val if val is not None else default
    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            fast_len = int(self._get_param('obt_cloudFastLen', 10))
            slow_len = int(self._get_param('obt_cloudSlowLen', 40))
            df['hma_fast'] = ta.hma(df['close'], length=fast_len)
            df['hma_slow'] = ta.hma(df['close'], length=slow_len)
            rsi_len = int(self._get_param('obt_rsiLength', 14))
            df['rsi'] = ta.rsi(df['close'], length=rsi_len)
            df['obv'] = ta.obv(df['close'], df['volume'])
            obv_len = int(self._get_param('obt_obvEntryLen', 20))
            obv_exit_len = int(self._get_param('exit_obvLength', 10))
            if 'obv' in df:
                df['obv_ma'] = ta.sma(df['obv'], length=obv_len)
                df['obv_exit_ma'] = ta.ema(df['obv'], length=obv_exit_len)
        except: pass
        return df
    def find_order_blocks(self, df):
        obs = {'buy': [], 'sell': []}
        if df is None or len(df) < 100: return obs
        swing = int(self._get_param('obt_swingLength', 5))
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing, len(subset) - swing):
            current_low = subset['low'].iloc[i]; current_high = subset['high'].iloc[i]
            is_swing_low = True
            for j in range(1, swing + 1):
                if subset['low'].iloc[i-j] <= current_low or subset['low'].iloc[i+j] <= current_low: is_swing_low = False; break
            if is_swing_low and subset['close'].iloc[i+1] > subset['high'].iloc[i]:
                obs['buy'].append({'top': subset['high'].iloc[i], 'bottom': subset['low'].iloc[i], 'created_at': subset['time'].iloc[i]})
            is_swing_high = True
            for j in range(1, swing + 1):
                if subset['high'].iloc[i-j] >= current_high or subset['high'].iloc[i+j] >= current_high: is_swing_high = False; break
            if is_swing_high and subset['close'].iloc[i+1] < subset['low'].iloc[i]:
                obs['sell'].append({'top': subset['high'].iloc[i], 'bottom': subset['low'].iloc[i], 'created_at': subset['time'].iloc[i]})
        return {'buy': obs['buy'][-3:], 'sell': obs['sell'][-3:]}
    def check_exit_signal(self, df_htf, position_side):
        result = {'close': False, 'reason': '', 'details': {}}
        if df_htf is None or len(df_htf) < 50: return result
        df = self.calculate_indicators(df_htf); last_row = df.iloc[-1]
        exit_rsi_overbought = float(self._get_param('exit_rsiOverbought', 70))
        exit_rsi_oversold = float(self._get_param('exit_rsiOversold', 30))
        curr_rsi = last_row.get('rsi', 50); curr_obv = last_row.get('obv', 0); curr_obv_ma = last_row.get('obv_exit_ma', 0)
        result['details'] = {'rsi': round(curr_rsi, 1), 'obv_cross': 'UP' if curr_obv > curr_obv_ma else 'DOWN'}
        if position_side == 'Buy':
            rsi_exit = curr_rsi >= exit_rsi_overbought
            confluence_exit = (curr_rsi >= exit_rsi_overbought) and (curr_obv < curr_obv_ma)
            if rsi_exit or confluence_exit: result['close'] = True; result['reason'] = "Confluence Exit" if confluence_exit else "RSI Exhaustion"
        elif position_side == 'Sell':
            rsi_exit = curr_rsi <= exit_rsi_oversold
            confluence_exit = (curr_rsi <= exit_rsi_oversold) and (curr_obv > curr_obv_ma)
            if rsi_exit or confluence_exit: result['close'] = True; result['reason'] = "Confluence Exit" if confluence_exit else "RSI Exhaustion"
        return result
    def analyze(self, df_ltf, df_htf):
        signals = []
        df_htf = self.calculate_indicators(df_htf); df_ltf = self.calculate_indicators(df_ltf)
        if df_htf is None or df_ltf is None or df_ltf.empty: return []
        curr_price = df_ltf['close'].iloc[-1]; htf_row = df_htf.iloc[-1]
        use_cloud = self._get_param('obt_useCloudFilter', True); use_obv = self._get_param('obt_useObvFilter', True)
        is_bull = True; is_bear = True
        if use_cloud:
            if htf_row['hma_fast'] <= htf_row['hma_slow']: is_bull = False
            if htf_row['hma_fast'] >= htf_row['hma_slow']: is_bear = False
        if use_obv and 'obv_ma' in htf_row:
            if htf_row['obv'] <= htf_row['obv_ma']: is_bull = False
            if htf_row['obv'] >= htf_row['obv_ma']: is_bear = False
        if not is_bull and not is_bear: return []
        ltf_row = df_ltf.iloc[-1]; use_retest = self._get_param('obt_useOBRetest', False)
        signal = None; details = []; sl_price = 0.0
        if is_bull:
            rsi_ok = ltf_row['rsi'] <= float(self._get_param('obt_entryRsiOversold', 45))
            if use_retest:
                obs = self.find_order_blocks(df_ltf); in_zone = False; active_ob = None
                for ob in obs['buy']:
                    if ob['bottom'] <= curr_price <= (ob['top'] * 1.003): in_zone = True; active_ob = ob; break
                if in_zone and rsi_ok: signal = 'Buy'; details.append("Trend+OB Retest"); sl_price = active_ob['bottom'] * 0.995 
            elif rsi_ok: signal = 'Buy'; details.append("Trend+RSI"); sl_price = curr_price * (1 - float(self._get_param('fixedSL', 1.5))/100)
        elif is_bear:
            rsi_ok = ltf_row['rsi'] >= float(self._get_param('obt_entryRsiOverbought', 55))
            if use_retest:
                obs = self.find_order_blocks(df_ltf); in_zone = False; active_ob = None
                for ob in obs['sell']:
                    if (ob['bottom'] * 0.997) <= curr_price <= ob['top']: in_zone = True; active_ob = ob; break
                if in_zone and rsi_ok: signal = 'Sell'; details.append("Trend+OB Retest"); sl_price = active_ob['top'] * 1.005
            elif rsi_ok: signal = 'Sell'; details.append("Trend+RSI"); sl_price = curr_price * (1 + float(self._get_param('fixedSL', 1.5))/100)
        if signal: signals.append({'action': signal, 'price': curr_price, 'rsi': ltf_row['rsi'], 'reason': ", ".join(details), 'sl_price': sl_price})
        return signals
ob_trend_strategy = OBTrendStrategy()
""",

    "market_analyzer.py": """import threading, time, pandas as pd, logging
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
""",

    "bot.py": """import logging, decimal, time
from datetime import datetime
from pybit.unified_trading import HTTP
from config import get_api_credentials; from settings_manager import settings; from statistics_service import stats_service
logger = logging.getLogger(__name__)
class BybitTradingBot:
    def __init__(self): k, s = get_api_credentials(); self.session = HTTP(testnet=False, api_key=k, api_secret=s)
    def normalize(self, s): return s.replace('.P', '')
    def get_bal(self):
        try: return float(next(c['walletBalance'] for a in self.session.get_wallet_balance(accountType="UNIFIED")['result']['list'] for c in a['coin'] if c['coin']=="USDT"))
        except: return 0.0
    def get_price(self, s): return float(self.session.get_tickers(category="linear", symbol=self.normalize(s))['result']['list'][0]['lastPrice'])
    def get_all_tickers(self): return self.session.get_tickers(category="linear")['result']['list']
    def get_instr(self, s):
        try: r = self.session.get_instruments_info(category="linear", symbol=self.normalize(s)); return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: return None, None
    def round_val(self, v, s): return round(v // s * s, abs(decimal.Decimal(str(s)).as_tuple().exponent))
    def sync_trades(self, days=7):
        try:
            now = int(time.time()*1000); start = now - (days*86400000); chunk = 7*86400000; cur_end = now
            while cur_end > start:
                cur_start = max(cur_end - chunk, start)
                r = self.session.get_closed_pnl(category="linear", startTime=int(cur_start), endTime=int(cur_end), limit=100)
                if r['retCode']==0:
                    for t in r['result']['list']:
                        stats_service.save_trade({'order_id':t['orderId'], 'symbol':t['symbol'], 'side':'Long' if t['side']=='Sell' else 'Short', 'qty':float(t['qty']), 'entry_price':float(t['avgEntryPrice']), 'exit_price':float(t['avgExitPrice']), 'pnl':float(t['closedPnl']), 'exit_time':datetime.fromtimestamp(int(t['updatedTime'])/1000)})
                cur_end = cur_start; time.sleep(0.2)
        except: pass
    def place_order(self, data):
        try:
            act, sym = data.get('action'), self.normalize(data.get('symbol'))
            if act == "Close":
                d = data.get('direction'); pos = self.session.get_positions(category="linear", symbol=sym)['result']['list']
                p = next((x for x in pos if float(x['size'])>0), None)
                if p and ((d=="Long" and p['side']=="Buy") or (d=="Short" and p['side']=="Sell")):
                    self.session.place_order(category="linear", symbol=sym, side="Sell" if p['side']=="Buy" else "Buy", orderType="Market", qty=p['size'], reduceOnly=True)
                    try: self.session.cancel_all_orders(category="linear", symbol=sym)
                    except: pass
                    return {"status": "ok"}
                return {"status": "ignored"}
            risk = float(data.get('riskPercent', settings.get('riskPercent'))); lev = int(data.get('leverage', settings.get('leverage')))
            price = self.get_price(sym); lot, tick = self.get_instr(sym)
            qty = self.round_val((self.get_bal() * (risk/100) * 0.98 * lev) / price, float(lot['qtyStep']))
            if qty < float(lot['minOrderQty']): return {"status": "skipped_min_qty"}
            self.session.set_leverage(category="linear", symbol=sym, buyLeverage=str(lev), sellLeverage=str(lev))
            self.session.place_order(category="linear", symbol=sym, side=act, orderType="Market", qty=str(qty))
            sl = float(data.get('sl_price', 0))
            if sl == 0: 
                sl_pct = float(settings.get('fixedSL', 1.5))
                sl = price * (1 - sl_pct/100) if act == "Buy" else price * (1 + sl_pct/100)
            self.session.set_trading_stop(category="linear", symbol=sym, stopLoss=str(self.round_val(sl, float(tick['tickSize']))), positionIdx=0)
            return {"status": "ok"}
        except Exception as e: return {"status": "error", "reason": str(e)}
bot_instance = BybitTradingBot()
""",

    "main_app.py": """import logging, threading, time, json, ctypes, os, requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from sqlalchemy import desc
from bot_config import config; from bot import bot_instance; from statistics_service import stats_service
from scanner import EnhancedMarketScanner; from settings_manager import settings; 
# Додано SmartMoneyTicker до імпортів для уникнення помилок
from models import db_manager, OrderBlock, SmartMoneyTicker 
from market_analyzer import market_analyzer
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002|0x00000001)
except: pass
app = Flask(__name__); app.secret_key='secret'; logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s'); logger = logging.getLogger(__name__)
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config()); scanner.start()
def monitor_active():
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode']==0:
                for p in r['result']['list']:
                    if float(p['size'])>0: stats_service.save_monitor_log({'symbol':p['symbol'], 'price':float(p['avgPrice']), 'pnl':float(p['unrealisedPnl']), 'rsi':scanner.get_current_rsi(p['symbol']), 'pressure':scanner.get_market_pressure(p['symbol'])})
        except: pass
        time.sleep(10)
def keep_alive():
    t = (os.environ.get('RENDER_EXTERNAL_URL') or f'http://127.0.0.1:{config.PORT}') + "/health"
    while True:
        try: requests.get(t, timeout=10)
        except: pass
        time.sleep(300)
threading.Thread(target=monitor_active, daemon=True).start(); threading.Thread(target=keep_alive, daemon=True).start()

# --- TABLE VIEW ROUTE ---
@app.route('/smart_money')
def smart_money_page(): 
    s=db_manager.get_session()
    # Показуємо АКТИВНІ (або очікуючі) зони, відсортовані за статусом
    blocks = s.query(OrderBlock).filter(OrderBlock.status != 'BROKEN').order_by(desc(OrderBlock.created_at)).all()
    s.close()
    return render_template('smart_money.html', blocks=blocks)

@app.route('/')
def home():
    d = int(request.args.get('days',7)); bot_instance.sync_trades(d)
    tr = stats_service.get_trades(d); pnl = sum(t['pnl'] for t in tr)
    return render_template('index.html', date=datetime.utcnow().strftime('%d %b %Y'), balance=bot_instance.get_bal(), active_count=len(scanner.get_active_symbols()), period_pnl=pnl, longs=sum(1 for t in tr if t['side']=='Long'), shorts=sum(1 for t in tr if t['side']=='Short'), days=d, trades=tr[:10])
@app.route('/scanner')
def scanner_page():
    act = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']==0:
            for p in r['result']['list']:
                if float(p['size'])>0:
                    d = scanner.get_coin_data(p['symbol'])
                    act.append({'symbol':p['symbol'], 'side':p['side'], 'pnl':round(float(p['unrealisedPnl']),2), 'rsi':d.get('rsi',0), 'exit_status':d.get('exit_status','Safe'), 'exit_details':d.get('exit_details','-'), 'size':p['size'], 'entry':p['avgPrice'], 'time':datetime.now().strftime('%H:%M')})
    except: pass
    return render_template('scanner.html', active=act, conf=settings._cache)
@app.route('/analyzer')
def analyzer_page(): return render_template('analyzer.html', results=market_analyzer.get_results(), conf=settings._cache, progress=market_analyzer.progress, status=market_analyzer.status_message, is_scanning=market_analyzer.is_scanning)
@app.route('/settings', methods=['GET','POST'])
def settings_general_page():
    if request.method=='POST': 
        f=request.form.to_dict(); f['telegram_enabled']=f.get('telegram_enabled')=='on'; f['exit_enableStrategy']=f.get('exit_enableStrategy')=='on'
        settings.save_settings(f); return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)
@app.route('/ob_trend/settings', methods=['GET','POST'])
def ob_trend_settings_page():
    if request.method=='POST':
        f=request.form.to_dict()
        for c in ['obt_useCloudFilter','obt_useObvFilter','obt_useRsiFilter','obt_useOBRetest']: f[c]=f.get(c)=='on'
        settings.save_settings(f); return redirect(url_for('ob_trend_settings_page'))
    return render_template('strategy_ob_trend.html', conf=settings._cache)
@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    f=request.form.to_dict()
    for c in ['obt_useOBRetest','obt_useCloudFilter','obt_useObvFilter','obt_useRsiFilter']: f[c]=f.get(c)=='on'
    settings.save_settings(f); market_analyzer.run_scan_thread(); return jsonify({"status":"started"})
@app.route('/analyzer/status')
def get_scan_status(): return jsonify({"progress":market_analyzer.progress, "message":market_analyzer.status_message, "is_scanning":market_analyzer.is_scanning})
@app.route('/webhook', methods=['POST'])
def webhook(): d=json.loads(request.get_data(as_text=True)); r=bot_instance.place_order(d); return jsonify(r), 200
@app.route('/settings/export')
def export_settings(): return Response(json.dumps(settings.get_all(), indent=4), mimetype='application/json', headers={'Content-Disposition':'attachment;filename=bot_settings.json'})
@app.route('/settings/import', methods=['POST'])
def import_settings():
    f = request.files['file']
    if f: settings.import_settings(json.load(f))
    return redirect(url_for('settings_general_page'))
@app.route('/health')
def health(): return jsonify({"status":"ok"})
if __name__ == '__main__': app.run(host=config.HOST, port=config.PORT)
""",

    # ---------------------------------------------------------
    # 4. ШАБЛОНИ (ОНОВЛЕНІ)
    # ---------------------------------------------------------
    "templates/base.html": """<!DOCTYPE html><html lang="uk" data-bs-theme="light"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{% block title %}AlgoBot{% endblock %}</title><link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"><link href="https://cdn.jsdelivr.net/npm/remixicon@3.5.0/fonts/remixicon.css" rel="stylesheet"><script src="https://cdn.jsdelivr.net/npm/chart.js"></script><style>:root{--bg-body:#f8fafc;--bg-sidebar:#ffffff;--bg-card:#ffffff;--accent:#4f46e5;--success:#059669;--danger:#dc2626;--sidebar-width:260px}body{background:var(--bg-body);font-family:'Inter',sans-serif;font-size:0.875rem}#sidebar{width:var(--sidebar-width);background:var(--bg-sidebar);border-right:1px solid #e2e8f0;position:fixed;top:0;left:0;height:100vh;z-index:1000}#content{margin-left:var(--sidebar-width);padding:2rem}.nav-link{color:#64748b;padding:12px 20px;display:flex;align-items:center;font-weight:500}.nav-link:hover,.nav-link.active{color:var(--accent);background:#eef2ff}.card-custom{background:var(--bg-card);border:1px solid #e2e8f0;border-radius:8px;box-shadow:0 1px 2px rgba(0,0,0,0.05)}.font-mono{font-family:'JetBrains Mono',monospace}.text-up{color:var(--success)}.text-down{color:var(--danger)}</style>{% block head %}{% endblock %}</head><body><nav id="sidebar"><div class="d-flex align-items-center px-4 border-bottom" style="height:70px"><i class="ri-code-box-fill text-accent fs-3 me-3"></i><span class="fw-bold">ALGOBOT</span></div><div class="py-4"><ul class="list-unstyled"><li><a href="/" class="nav-link"><i class="ri-layout-grid-line me-3"></i>Огляд</a></li><li><a href="/scanner" class="nav-link"><i class="ri-pulse-line me-3"></i>Монітор</a></li><li><a href="/analyzer" class="nav-link"><i class="ri-radar-line me-3"></i>Сканер</a></li><li><a href="/smart_money" class="nav-link"><i class="ri-eye-2-line me-3"></i>Smart Money</a></li><li><a href="/settings" class="nav-link"><i class="ri-settings-line me-3"></i>Налаштування</a></li></ul></div></nav><main id="content">{% block content %}{% endblock %}</main><script src="https://code.jquery.com/jquery-3.6.0.min.js"></script><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script></body></html>""",

    "templates/settings.html": """{% extends "base.html" %}{% block title %}Config{% endblock %}{% block content %}
<div class="container-fluid" style="max-width:900px"><h4 class="mb-4">⚙️ Налаштування</h4><form method="POST">
<div class="card-custom mb-4 p-4"><h6 class="text-primary mb-3">⏳ Таймфрейми & Індикатори (Adaptive)</h6>
<div class="row g-3"><div class="col-md-6"><label class="form-label fw-bold">Глобальний Тренд (HTF)</label>
<select class="form-select" name="htfSelection" id="htfSelector" onchange="updateIndicators()">
<option value="15" {{'selected' if conf.get('htfSelection')|string=='15'}}>15m (Scalp)</option>
<option value="30" {{'selected' if conf.get('htfSelection')|string=='30'}}>30m</option>
<option value="45" {{'selected' if conf.get('htfSelection')|string=='45'}}>45m</option>
<option value="60" {{'selected' if conf.get('htfSelection')|string=='60'}}>1H (Intraday)</option>
<option value="240" {{'selected' if conf.get('htfSelection')|string=='240'}}>4H (Swing)</option>
<option value="D" {{'selected' if conf.get('htfSelection')|string=='D'}}>1D</option></select></div>
<div class="col-md-6"><label class="form-label fw-bold">Вхід (LTF)</label><select class="form-select" name="ltfSelection"><option value="5" {{'selected' if conf.get('ltfSelection')|string=='5'}}>5m</option><option value="15" {{'selected' if conf.get('ltfSelection')|string=='15'}}>15m</option><option value="45" {{'selected' if conf.get('ltfSelection')|string=='45'}}>45m</option></select></div>
<div class="col-12"><hr></div>
<div class="col-md-3"><label>Cloud Fast</label><input type="number" class="form-control" id="cloudFast" name="obt_cloudFastLen" value="{{conf.get('obt_cloudFastLen')}}"></div>
<div class="col-md-3"><label>Cloud Slow</label><input type="number" class="form-control" id="cloudSlow" name="obt_cloudSlowLen" value="{{conf.get('obt_cloudSlowLen')}}"></div>
<div class="col-md-3"><label>OBV Len</label><input type="number" class="form-control" id="obvLen" name="obt_obvEntryLen" value="{{conf.get('obt_obvEntryLen')}}"></div>
<div class="col-md-3"><label>Swing</label><input type="number" class="form-control" id="swingLen" name="obt_swingLength" value="{{conf.get('obt_swingLength')}}"></div>
<div class="col-md-3"><label>RSI Buy</label><input type="number" class="form-control" id="entryRsiBuy" name="obt_entryRsiOversold" value="{{conf.get('obt_entryRsiOversold')}}"></div>
<div class="col-md-3"><label>RSI Sell</label><input type="number" class="form-control" id="entryRsiSell" name="obt_entryRsiOverbought" value="{{conf.get('obt_entryRsiOverbought')}}"></div></div></div>
<div class="card-custom mb-4 p-4"><div class="d-flex justify-content-between mb-3"><h6 class="text-danger m-0">🔴 Smart Exit Strategy</h6><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="exit_enableStrategy" id="exitSwitch" {{'checked' if conf.get('exit_enableStrategy')}}><label for="exitSwitch">Active</label></div></div>
<div class="row g-3"><div class="col-md-4"><label>RSI Exit Short (<)</label><input type="number" class="form-control" id="exitRsiShort" name="exit_rsiOversold" value="{{conf.get('exit_rsiOversold')}}"></div>
<div class="col-md-4"><label>RSI Exit Long (>)</label><input type="number" class="form-control" id="exitRsiLong" name="exit_rsiOverbought" value="{{conf.get('exit_rsiOverbought')}}"></div>
<div class="col-md-4"><label>OBV Exit Len</label><input type="number" class="form-control" id="exitObvLen" name="exit_obvLength" value="{{conf.get('exit_obvLength')}}"></div></div></div>
<button type="submit" class="btn btn-primary w-100">Зберегти</button></form></div>
<script>
const pM={'15':{cf:20,cs:60,ob:50,sw:10,rb:40,rs:60,exS:25,exL:75,exO:25},'30':{cf:15,cs:50,ob:40,sw:8,rb:40,rs:60,exS:25,exL:75,exO:20},'45':{cf:12,cs:45,ob:30,sw:7,rb:45,rs:55,exS:30,exL:70,exO:15},'60':{cf:10,cs:40,ob:20,sw:5,rb:45,rs:55,exS:30,exL:70,exO:12},'240':{cf:10,cs:40,ob:20,sw:5,rb:45,rs:55,exS:30,exL:70,exO:10},'D':{cf:10,cs:40,ob:20,sw:5,rb:45,rs:55,exS:30,exL:70,exO:10}};
function updateIndicators(){const t=document.getElementById('htfSelector').value;const c=pM[t];if(c){ff('cloudFast',c.cf);ff('cloudSlow',c.cs);ff('obvLen',c.ob);ff('swingLen',c.sw);ff('entryRsiBuy',c.rb);ff('entryRsiSell',c.rs);ff('exitRsiShort',c.exS);ff('exitRsiLong',c.exL);ff('exitObvLen',c.exO);}}
function ff(id,v){const e=document.getElementById(id);if(e){e.style.backgroundColor='#d1fae5';e.value=v;setTimeout(()=>e.style.backgroundColor='white',300);}}
</script>{% endblock %}""",

    "templates/scanner.html": """{% extends "base.html" %}{% block title %}Monitor{% endblock %}{% block content %}
<div class="container-fluid"><div class="card-custom mb-4 p-3"><div class="row align-items-center">
<div class="col-md-4 border-end">Smart Exit: {% if conf.get('exit_enableStrategy') %}<span class="text-success fw-bold">ON 🟢</span>{% else %}<span class="text-danger fw-bold">OFF 🔴</span>{% endif %}</div>
<div class="col-md-4 border-end text-center">Work TF: <span class="badge bg-light text-dark border">{% set h=conf.get('htfSelection')|string %}{% if h=='60'%}1H{% elif h=='240'%}4H{% elif h=='D'%}1D{% else %}{{h}}m{% endif %}</span></div>
<div class="col-md-4 text-end">RSI Limits: <span class="text-danger">>{{conf.get('exit_rsiOverbought')}}</span> / <span class="text-success"><{{conf.get('exit_rsiOversold')}}</span></div></div></div>
<div class="card-custom"><table class="table mb-0"><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>RSI (HTF)</th><th>Status</th><th>P&L</th><th>Time</th></tr></thead><tbody>
{% for a in active %}<tr><td class="fw-bold">{{a.symbol}}</td><td><span class="badge {{'bg-success' if a.side=='Buy' else 'bg-danger'}}">{{a.side}}</span></td><td>{{a.entry}}</td>
<td><span class="{{'text-danger' if a.rsi>conf.get('exit_rsiOverbought')|float else 'text-success' if a.rsi<conf.get('exit_rsiOversold')|float else 'text-muted'}} fw-bold">{{a.rsi}}</span></td>
<td>{% if 'EXIT' in a.exit_status %}<span class="badge bg-danger blink">EXIT</span>{% elif 'Warn' in a.exit_status %}<span class="badge bg-warning text-dark">WARN</span>{% else %}<span class="badge bg-light text-muted border">SAFE</span>{% endif %}<div class="small text-muted">{{a.exit_details}}</div></td>
<td class="{{'text-up' if a.pnl>0 else 'text-down'}} fw-bold">{{a.pnl}}</td><td class="text-muted small">{{a.time}}</td></tr>{% else %}<tr><td colspan="7" class="text-center py-4">No positions</td></tr>{% endfor %}</tbody></table></div></div>
<style>.blink{animation:b 1s infinite}@keyframes b{50%{opacity:0}}</style>{% endblock %}""",

    "templates/analyzer.html": """{% extends "base.html" %}{% block title %}Scanner{% endblock %}{% block content %}
<div class="container-fluid"><div class="card-custom mb-4 p-3"><form id="scan-form">
<div class="d-flex gap-3 align-items-end mb-3"><div class="flex-grow-1"><label>Depth</label><input type="number" name="scan_limit" class="form-control" value="{{conf.get('scan_limit',100)}}"></div>
<div class="d-flex gap-2 align-items-end"><span class="badge bg-white text-dark border p-2">HTF: {% set h=conf.get('htfSelection')|string %}{% if h=='60'%}1H{% elif h=='240'%}4H{% elif h=='D'%}1D{% else %}{{h}}m{% endif %}</span><span class="badge bg-white text-dark border p-2">LTF: {% set l=conf.get('ltfSelection')|string %}{% if l=='60'%}1H{% else %}{{l}}m{% endif %}</span></div>
<div><button type="button" onclick="go()" id="btn" class="btn btn-primary px-5">SCAN 🚀</button></div></div>
<div class="row g-2">
<div class="col-3"><input type="checkbox" class="btn-check" id="c1" name="obt_useCloudFilter" {{'checked' if conf.get('obt_useCloudFilter')}}><label class="btn btn-outline-secondary w-100 d-flex justify-content-between" for="c1"><span>Cloud <small>({{conf.get('obt_cloudFastLen')}}/{{conf.get('obt_cloudSlowLen')}})</small></span><i class="ri-checkbox-circle-line"></i></label></div>
<div class="col-3"><input type="checkbox" class="btn-check" id="c2" name="obt_useObvFilter" {{'checked' if conf.get('obt_useObvFilter')}}><label class="btn btn-outline-secondary w-100 d-flex justify-content-between" for="c2"><span>OBV <small>({{conf.get('obt_obvEntryLen')}})</small></span><i class="ri-bar-chart-line"></i></label></div>
<div class="col-3"><input type="checkbox" class="btn-check" id="c3" name="obt_useRsiFilter" {{'checked' if conf.get('obt_useRsiFilter')}}><label class="btn btn-outline-secondary w-100 d-flex justify-content-between" for="c3"><span>RSI <small>({{conf.get('obt_rsiLength')}})</small></span><i class="ri-pulse-line"></i></label></div>
<div class="col-3"><input type="checkbox" class="btn-check" id="c4" name="obt_useOBRetest" {{'checked' if conf.get('obt_useOBRetest')}}><label class="btn btn-outline-secondary w-100 d-flex justify-content-between" for="c4"><span>Retest <small>(Sw {{conf.get('obt_swingLength')}})</small></span><i class="ri-arrow-go-back-line"></i></label></div>
</div></form></div>
<div id="prog" class="card-custom mb-3 p-3 bg-light" style="display:none"><div class="progress"><div id="bar" class="progress-bar progress-bar-striped progress-bar-animated" style="width:0%"></div></div><div class="text-center mt-2 fw-bold" id="txt">...</div></div>
<div class="card-custom"><table class="table"><thead><tr><th>Sym</th><th>Price</th><th>Signal</th><th>Score</th><th>Det</th></tr></thead><tbody>{% for r in results %}<tr><td class="fw-bold">{{r.symbol}}</td><td>{{r.price}}</td><td><span class="badge {{'bg-success' if r.signal=='Buy' else 'bg-danger'}}">{{r.signal}}</span></td><td>{{r.score}}</td><td class="small text-muted">{{r.details}}</td></tr>{% endfor %}</tbody></table></div></div>
<script>function go(){$('#btn').prop('disabled',1);$('#prog').show();$.post('/analyzer/scan',$('#scan-form').serialize(),function(){p()})}
function p(){let i=setInterval(function(){$.get('/analyzer/status',function(d){$('#bar').css('width',d.progress+'%');$('#txt').text(d.message);if(!d.is_scanning){clearInterval(i);location.reload()}})},1000)}</script>{% endblock %}""",

    "templates/index.html": """{% extends "base.html" %}{% block title %}Dash{% endblock %}{% block content %}<div class="container-fluid"><h4 class="mb-4">Dashboard</h4><div class="row g-3 mb-4"><div class="col-md-3"><div class="card-custom p-3"><h6>Balance</h6><h3>{{balance}}</h3></div></div><div class="col-md-3"><div class="card-custom p-3"><h6>Active</h6><h3>{{active_count}}</h3></div></div><div class="col-md-3"><div class="card-custom p-3"><h6>P&L ({{days}}d)</h6><h3 class="{{'text-up' if period_pnl>0 else 'text-down'}}">{{period_pnl}}</h3></div></div></div><div class="card-custom"><div class="card-header bg-white">Recent Trades</div><table class="table mb-0">{% for t in trades %}<tr><td>{{t.symbol}}</td><td><span class="badge {{'bg-success' if t.side=='Long' else 'bg-danger'}}">{{t.side}}</span></td><td class="{{'text-up' if t.pnl>0 else 'text-down'}}">{{t.pnl}}</td><td>{{t.exit_time}}</td></tr>{% endfor %}</table></div></div>{% endblock %}""",
    
    # ---------------------------------------------------------
    # 5. SMART MONEY TABLE (TABLE VIEW INSTEAD OF CHART)
    # ---------------------------------------------------------
    "templates/smart_money.html": """{% extends "base.html" %}{% block content %}
<div class="container-fluid"><div class="d-flex justify-content-between mb-3"><h4 class="fw-bold">Smart Money Zones</h4><a href="/analyzer" class="btn btn-sm btn-outline-secondary">Scanner</a></div>
<div class="card-custom"><table class="table align-middle"><thead><tr><th>Symbol</th><th>Zone</th><th>Range (Entry)</th><th>SL</th><th>Status</th><th>Time</th><th>Action</th></tr></thead><tbody>
{% for b in blocks %}<tr><td class="fw-bold">{{b.symbol}}</td><td><span class="badge {{'bg-success' if b.ob_type=='Buy' else 'bg-danger'}}">{{b.ob_type}}</span></td>
<td class="font-mono">{{b.bottom}} - {{b.top}}</td><td class="font-mono text-muted">{{b.sl_price}}</td>
<td>{% if 'INSIDE' in b.status %}<span class="badge bg-success blink">INSIDE</span>{% elif 'NEAR' in b.status %}<span class="badge bg-warning text-dark">NEAR</span>{% elif 'BROKEN' in b.status %}<span class="badge bg-secondary text-decoration-line-through">BROKEN</span>{% else %}<span class="badge bg-light text-muted border">WAITING</span>{% endif %}</td>
<td class="small">{{b.created_at.strftime('%H:%M')}}</td><td><a href="https://www.bybit.com/trade/usdt/{{b.symbol}}" target="_blank" class="btn btn-xs btn-primary">Bybit ↗</a></td></tr>
{% else %}<tr><td colspan="7" class="text-center py-5">No active zones found.</td></tr>{% endfor %}</tbody></table></div></div>
<style>.blink{animation:b 1s infinite}@keyframes b{50%{opacity:0}}</style>{% endblock %}""",
    
    "templates/strategy_ob_trend.html": """{% extends "base.html" %}{% block content %}<div class="container-fluid" style="max-width:600px"><h4>Filters</h4><form method="POST"><div class="card-custom p-3 mb-3"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="obt_useCloudFilter" {{'checked' if conf.get('obt_useCloudFilter')}}><label>Cloud</label></div><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="obt_useObvFilter" {{'checked' if conf.get('obt_useObvFilter')}}><label>OBV</label></div><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="obt_useRsiFilter" {{'checked' if conf.get('obt_useRsiFilter')}}><label>RSI</label></div><div class="form-check form-switch"><input class="form-check-input" type="checkbox" name="obt_useOBRetest" {{'checked' if conf.get('obt_useOBRetest')}}><label>Retest</label></div></div><button class="btn btn-primary w-100">Save</button></form></div>{% endblock %}"""
}

import os

for filename, content in files.items():
    path = os.path.join(PROJECT_ROOT, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Updated: {filename}")

print(f"\n🎉 Project UPDATED to Table View Mode")