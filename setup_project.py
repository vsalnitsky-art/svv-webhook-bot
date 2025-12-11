#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

# Назва кореневої папки проекту
PROJECT_ROOT = "trading_bot_full"

# Вміст усіх файлів (Остання стабільна версія)
files = {
    # --- PYTHON FILES ---
    "requirements.txt": """flask
requests
pybit
sqlalchemy
pandas
cryptography
gunicorn
""",

    "bot_config.py": """import os
class Config:
    PORT = int(os.environ.get("PORT", 10000))
    HOST = "0.0.0.0"
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

class DatabaseManager:
    def __init__(self, db_filename='trading_bot_final.db'):
        db_url = os.environ.get('DATABASE_URL')
        if db_url and db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        else:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_folder = os.path.join(current_dir, 'BASE')
            try:
                os.makedirs(base_folder, exist_ok=True)
                db_path = os.path.join(base_folder, db_filename)
            except OSError:
                db_path = os.path.join(current_dir, db_filename)
            db_url = f'sqlite:///{db_path}'
        self.engine = create_engine(db_url, echo=False)
        # try: Base.metada# ta.create_all(self.engine)
        except: pass
        self.Session = sessionmaker(bind=self.engine)
    def get_session(self): return self.Session()
db_manager = DatabaseManager()
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
stats_service = StatisticsService()
""",

    "settings_manager.py": """import logging
from models import db_manager, BotSetting
logger = logging.getLogger(__name__)
DEFAULT_SETTINGS = {
    "scanner_quote_coin": "USDT", "scanner_mode": "Manual", "scan_limit": 100,
    "telegram_enabled": False, "telegram_bot_token": "", "telegram_chat_id": "",
    "useCloudFilter": True, "useObvFilter": True, "useRsiFilter": True, "useMfiFilter": False, "useOBRetest": False,
    "htfSelection": "240", "ltfSelection": "15", "cloudFastLen": 10, "cloudSlowLen": 40,
    "entryRsiOversold": 45, "entryRsiOverbought": 55, "rsiLength": 14,
    "exitRsiOversold": 30, "exitRsiOverbought": 70, "mfiLength": 20, "obvEntryLen": 20, "obvExitLen": 20,
    "riskPercent": 2.0, "leverage": 20, "fixedTP": 3.0, "fixedSL": 1.5,
    "atrMultiplierSL": 1.5, "atrMultiplierTP": 3.0, "swingLength": 5, "volumeSpikeThreshold": 1.8, "tp_mode": "None"
}
class SettingsManager:
    def __init__(self):
        self.db = db_manager; self._cache = {}; self.reload_settings()
    def _cast_value(self, key, value_str):
        if key not in DEFAULT_SETTINGS: return value_str
        default = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default, bool): return str(value_str).lower() == 'true'
            elif isinstance(default, int): return int(value_str)
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
                for s in db_s: loaded[s.key] = self._cast_value(s.key, s.value)
                self._cache = DEFAULT_SETTINGS.copy(); self._cache.update(loaded)
        except: self._cache = DEFAULT_SETTINGS.copy()
        finally: session.close()
    def save_settings(self, new_settings):
        session = self.db.get_session()
        try:
            for k, v in new_settings.items():
                if k in DEFAULT_SETTINGS:
                    val = v
                    if isinstance(DEFAULT_SETTINGS[k], bool): val = "true" if (v=='on' or v is True) else "false"
                    else: val = str(v)
                    self._cache[k] = self._cast_value(k, v)
                    ex = session.query(BotSetting).filter_by(key=k).first()
                    if ex: ex.value = val
                    else: session.add(BotSetting(key=k, value=val))
            session.commit()
        except: session.rollback()
        finally: session.close()
    def get(self, key): return self._cache.get(key, DEFAULT_SETTINGS.get(key))
settings = SettingsManager()
""",

    "strategy.py": """import pandas as pd
import logging
from settings_manager import settings
logger = logging.getLogger(__name__)
class StrategyEngine:
    def __init__(self): pass
    def get_param(self, key): return settings.get(key)
    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            # df['hma_fast'] = # ta.hma(df['close'], length=self.get_param('cloudFastLen'))
            # df['hma_slow'] = # ta.hma(df['close'], length=self.get_param('cloudSlowLen'))
            # df['rsi'] = # ta.rsi(df['close'], length=self.get_param('rsiLength'))
            # df['obv'] = # ta.obv(df['close'], df['volume'])
            # if 'obv' in df: df['obv_ma'] = # ta.sma(df['obv'], length=self.get_param('obvEntryLen'))
            # df['atr'] = # ta.atr(df['high'], df['low'], df['close'], length=14)
        except: pass
        return df
    def check_htf_filters(self, htf_row):
        if htf_row is None or 'hma_fast' not in htf_row.index: return {'bull': False, 'bear': False}
        use_cloud, use_rsi, use_obv = self.get_param('useCloudFilter'), self.get_param('useRsiFilter'), self.get_param('useObvFilter')
        cloud_bull = (htf_row['hma_fast'] > htf_row['hma_slow']) if use_cloud else True
        cloud_bear = (htf_row['hma_fast'] < htf_row['hma_slow']) if use_cloud else True
        rsi_bull = (htf_row['rsi'] <= self.get_param('entryRsiOversold')) if use_rsi else True
        rsi_bear = (htf_row['rsi'] >= self.get_param('entryRsiOverbought')) if use_rsi else True
        obv_bull = (htf_row['obv'] > htf_row['obv_ma']) if use_obv else True
        obv_bear = (htf_row['obv'] < htf_row['obv_ma']) if use_obv else True
        return {'bull': bool(cloud_bull and obv_bull and rsi_bull), 'bear': bool(cloud_bear and obv_bear and rsi_bear), 'details': {'rsi': htf_row['rsi']}}
    def detect_order_blocks(self, df):
        if df is None or len(df) < 50: return [], []
        bull, bear, swing = [], [], self.get_param('swingLength')
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing, len(subset) - swing):
            cur = i
            is_h = all(subset['high'][cur] > subset['high'][cur+j] and subset['high'][cur] > subset['high'][cur-j] for j in range(1, swing+1))
            is_l = all(subset['low'][cur] < subset['low'][cur+j] and subset['low'][cur] < subset['low'][cur-j] for j in range(1, swing+1))
            if is_h:
                top = subset['high'][cur]
                for k in range(cur+1, len(subset)):
                    if subset['close'][k] > top: 
                        wave = subset.iloc[cur:k]; bull.append({'top': wave['high'].max(), 'bottom': wave['low'].min()}); break
            if is_l:
                btm = subset['low'][cur]
                for k in range(cur+1, len(subset)):
                    if subset['close'][k] < btm: 
                        wave = subset.iloc[cur:k]; bear.append({'top': wave['high'].max(), 'bottom': wave['low'].min()}); break
        curr = subset['close'].iloc[-1]
        return [b for b in bull if curr > b['bottom']][-5:], [b for b in bear if curr < b['top']][-5:]
    def get_signal(self, df_ltf, df_htf):
        if df_ltf is None or 'hma_fast' not in df_htf.columns: return {'action': None, 'reason': ''}
        filters = self.check_htf_filters(df_htf.iloc[-1])
        use_retest = self.get_param('useOBRetest')
        cur = df_ltf['close'].iloc[-1]
        sig, reason = None, ""
        if filters['bull']:
            trig = False
            if use_retest:
                bulls, _ = self.detect_order_blocks(df_ltf)
                for b in bulls: 
                    if b['bottom'] <= cur <= b['top']: trig = True; reason = "Bull Retest"; break
            else: trig = True; reason = "Bull Trend"
            if trig: sig = "Buy"
        elif filters['bear']:
            trig = False
            if use_retest:
                _, bears = self.detect_order_blocks(df_ltf)
                for b in bears:
                    if b['bottom'] <= cur <= b['top']: trig = True; reason = "Bear Retest"; break
            else: trig = True; reason = "Bear Trend"
            if trig: sig = "Sell"
        return {'action': sig, 'reason': reason}
strategy_engine = StrategyEngine()
""",

    "scanner.py": """import threading, time, logging, pandas as pd, pandas_ta as ta
from settings_manager import settings
logger = logging.getLogger(__name__)
class EnhancedMarketScanner:
    def __init__(self, bot, cfg): self.bot = bot; self.data = {}; threading.Thread(target=self.loop, daemon=True).start()
    def loop(self):
        while True:
            try:
                pos = self.bot.session.get_positions(category="linear", settleCoin="USDT")['result']['list']
                actives = [p['symbol'] for p in pos if safe_float(p.get('size'), 0) > 0]
                if not actives: self.data = {}; time.sleep(5); continue
                tickers = self.bot.get_all_tickers()
                for t in tickers:
                    sym = t['symbol']
                    if sym not in actives: continue
                    if sym not in self.data: self.data[sym] = {'rsi': 50, 'pressure': 0}
                    self.data[sym]['rsi'] = self.fetch_rsi(sym)
                    price, turn = float(t['lastPrice']), float(t['turnover24h'])
                    diff = turn - self.data[sym].get('prev_turn', turn)
                    if diff > 0:
                        d = 1 if price >= self.data[sym].get('prev_price', price) else -1
                        self.data[sym]['pressure'] = self.data[sym]['pressure']*0.9 + (diff*d)
                    self.data[sym]['prev_turn'] = turn; self.data[sym]['prev_price'] = price
                    time.sleep(0.2)
            except: pass
            time.sleep(5)
    def fetch_rsi(self, sym):
        try:
            tf = settings.get("ltfSelection") or "15"
            r = self.bot.session.get_kline(category="linear", symbol=sym, interval=str(tf), limit=30)
            df = pd.DataFrame(r['result']['list'], columns=['t','o','h','l','c','v','to'])
            # return round(# ta.rsi(df.iloc[::-1]['c'].astype(float), length=14).iloc[-1], 1)
        except: return 50
    # def get_current_rsi(self, s): return self.da# ta.get(s, {}).get('rsi', 50)
    # def get_market_pressure(self, s): return self.da# ta.get(s, {}).get('pressure', 0)
""",

    "market_analyzer.py": """import threading, time, pandas as pd, logging
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
""",

    "bot.py": """import logging, decimal, time
from datetime import datetime
from pybit.unified_trading import HTTP
from bot_config import config; from config import get_api_credentials; from settings_manager import settings; from statistics_service import stats_service
logger = logging.getLogger(__name__)
class BybitTradingBot:
    def __init__(self): k, s = get_api_credentials(); self.session = HTTP(testnet=False, api_key=k, api_secret=s)
    def normalize(self, s): return s.replace('.P', '')
    def get_bal(self):
        try:
            b = self.session.get_wallet_balance(accountType="UNIFIED")
            for a in b['result']['list']:
                for c in a['coin']:
                    if c['coin']=="USDT": return float(c['walletBalance'])
        except: pass
        return 0.0
    def get_available_balance(self): return self.get_bal()
    def get_price(self, s):
        try: return float(self.session.get_tickers(category="linear", symbol=self.normalize(s))['result']['list'][0]['lastPrice'])
        except: return 0.0
    def get_all_tickers(self):
        try: return self.session.get_tickers(category="linear")['result']['list']
        except: return []
    def get_instr(self, s):
        try: r = self.session.get_instruments_info(category="linear", symbol=self.normalize(s)); return r['result']['list'][0]['lotSizeFilter'], r['result']['list'][0]['priceFilter']
        except: return None, None
    def round_val(self, v, s):
        try: return round(v // s * s, abs(decimal.Decimal(str(s)).as_tuple().exponent))
        except: return v
    def set_lev(self, s, l):
        try: self.session.set_leverage(category="linear", symbol=self.normalize(s), buyLeverage=str(l), sellLeverage=str(l))
        except: pass
    def sync_trades(self, days=7):
        try:
            now = int(time.time()*1000); start = now - (days*86400000); chunk = 7*86400000; cur_end = now
            while cur_end > start:
                cur_start = max(cur_end - chunk, start)
                r = self.session.get_closed_pnl(category="linear", startTime=int(cur_start), endTime=int(cur_end), limit=100)
                if r['retCode']==0:
                    for t in r['result']['list']:
                        stats_service.save_trade({'order_id':t['orderId'], 'symbol':t['symbol'], 'side':'Long' if t['side']=='Sell' else 'Short', 'qty':float(t['qty']), 'entry_price':float(t['avgEntryPrice']), 'exit_price':float(t['avgExitPrice']), 'pnl':float(t['closedPnl']), 'exit_time':datetime.fromtimestamp(int(t['updatedTime'])/1000), 'exit_reason':'Signal/TP/SL'})
                cur_end = cur_start; time.sleep(0.2)
        except: pass
    def place_order(self, data):
        try:
            # act, sym = da# ta.get('action'), da# ta.get('symbol'); norm = self.normalize(sym)
            if act == "Close":
                # d = da# ta.get('direction')
                pos = self.session.get_positions(category="linear", symbol=norm)['result']['list']
                p = next((x for x in pos if float(x['size'])>0), None)
                if not p: return {"status": "ignored"}
                if (d=="Long" and p['side']=="Buy") or (d=="Short" and p['side']=="Sell"):
                    self.session.place_order(category="linear", symbol=norm, side="Sell" if p['side']=="Buy" else "Buy", orderType="Market", qty=p['size'], reduceOnly=True)
                    try: self.session.cancel_all_orders(category="linear", symbol=norm)
                    except: pass
                    return {"status": "ok"}
                return {"status": "ignored"}
            
            pos = self.session.get_positions(category="linear", symbol=norm)['result']['list']
            if any(safe_float(p.get('size'), 0)>0 for p in pos): return {"status": "ignored"}
            
            # risk = float(da# ta.get('riskPercent', settings.get('riskPercent'))); lev = int(da# ta.get('leverage', settings.get('leverage')))
            price = self.get_price(norm); lot, tick = self.get_instr(norm)
            qty = self.round_val((self.get_bal() * (risk/100) * 0.98 * lev) / price, float(lot['qtyStep']))
            if qty < float(lot['minOrderQty']): qty = float(lot['minOrderQty'])
            
            self.set_lev(norm, lev)
            self.session.place_order(category="linear", symbol=norm, side=act, orderType="Market", qty=str(qty))
            
            # sl = float(da# ta.get('stopLossPercent', settings.get('fixedSL')))
            if sl > 0:
                slp = self.round_val(price * (1 - sl/100) if act == "Buy" else price * (1 + sl/100), float(tick['tickSize']))
                self.session.set_trading_stop(category="linear", symbol=norm, stopLoss=str(slp), positionIdx=0)
            
            self._tp(norm, act, price, qty, data, float(tick['tickSize']), float(lot['qtyStep']))
            return {"status": "ok"}
        except Exception as e: return {"status": "error", "reason": str(e)}

    def _tp(self, s, side, ep, qty, d, tick, step):
        try:
            mode = settings.get("tp_mode"); exit_side = "Sell" if side=="Buy" else "Buy"
            if mode == "Fixed_1_50":
                q = self.round_val(qty*0.5, step); p = self.round_val(ep*1.01 if side=="Buy" else ep*0.99, tick)
                if q>0: self.session.place_order(category="linear", symbol=s, side=exit_side, orderType="Limit", qty=str(q), price=str(p), reduceOnly=True)
            elif mode == "Ladder_3":
                tp = float(d.get('takeProfitPercent', settings.get('fixedTP')))/100; q_step = self.round_val(qty*0.33, step)
                for i, mult in enumerate([1/3, 2/3, 1]):
                    pct = tp*mult; p = self.round_val(ep*(1+pct) if side=="Buy" else ep*(1-pct), tick)
                    q = self.round_val(qty - q_step*2, step) if i==2 else q_step
                    if q>0: self.session.place_order(category="linear", symbol=s, side=exit_side, orderType="Limit", qty=str(q), price=str(p), reduceOnly=True)
        except: pass
bot_instance = BybitTradingBot()
""",

    "main_app.py": """import logging, threading, time, json, ctypes, os, requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for
from bot_config import config; from bot import bot_instance; from statistics_service import stats_service; from scanner import EnhancedMarketScanner; from settings_manager import settings; from market_analyzer import market_analyzer
try: ctypes.windll.kernel32.SetThreadExecutionState(0x80000002|0x00000001)
except: pass
app = Flask(__name__); logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s'); logger = logging.getLogger(__name__)
scanner = EnhancedMarketScanner(bot_instance, config.get_scanner_config()); scanner.start()
def monitor_active():
    while True:
        try:
            r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
            if r['retCode']==0:
                for p in r['result']['list']:
                    if safe_float(p.get('size'), 0)>0: stats_service.save_monitor_log({'symbol':p['symbol'], 'price':safe_float(p.get('avgPrice'), 0), 'pnl':safe_float(p.get('unrealisedPnl'), 0), 'rsi':scanner.get_current_rsi(p['symbol']), 'pressure':scanner.get_market_pressure(p['symbol'])})
        except: pass
        time.sleep(10)
def keep_alive():
    t = (os.environ.get('RENDER_EXTERNAL_URL') or f'http://127.0.0.1:{config.PORT}') + "/health"
    while True:
        try: requests.get(t, timeout=10)
        except: pass
        time.sleep(300)
threading.Thread(target=monitor_active, daemon=True).start(); threading.Thread(target=keep_alive, daemon=True).start()
@app.route('/')
def home(): return render_template('index.html', time=datetime.utcnow().strftime('%H:%M:%S UTC'))
@app.route('/scanner')
def scanner_page():
    active = []
    try:
        r = bot_instance.session.get_positions(category="linear", settleCoin="USDT")
        if r['retCode']==0:
            for p in r['result']['list']:
                if safe_float(p.get('size'), 0)>0:
                    ts = p.get('createdTime') or p.get('updatedTime', time.time()*1000)
                    active.append({'symbol':p['symbol'], 'side':p['side'], 'pnl':round(safe_float(p.get('unrealisedPnl'), 0),2), 'rsi':scanner.get_current_rsi(p['symbol']), 'pressure':round(scanner.get_market_pressure(p['symbol'])), 'size':p['size'], 'entry':p['avgPrice'], 'time':datetime.fromtimestamp(int(ts)/1000).strftime('%d.%m %H:%M')})
    except: pass
    return render_template('scanner.html', active=active)
@app.route('/analyzer')
def analyzer_page(): return render_template('analyzer.html', results=market_analyzer.get_results(), conf=settings._cache, progress=market_analyzer.progress, status=market_analyzer.status_message, is_scanning=market_analyzer.is_scanning)
@app.route('/settings', methods=['GET','POST'])
def settings_general_page():
    if request.method=='POST': f=request.form.to_dict(); f['telegram_enabled']=request.form.get('telegram_enabled')=='on'; settings.save_settings(f); return redirect(url_for('settings_general_page'))
    return render_template('settings.html', conf=settings._cache)
@app.route('/analyzer/settings', methods=['GET','POST'])
def analyzer_settings_page():
    if request.method=='POST':
        f=request.form.to_dict()
        for c in ['useCloudFilter','useObvFilter','useRsiFilter','useMfiFilter','useOBRetest']: f[c]=request.form.get(c)=='on'
        settings.save_settings(f); return redirect(url_for('analyzer_settings_page'))
    return render_template('strategy.html', conf=settings._cache)
@app.route('/analyzer/scan', methods=['POST'])
def run_scan():
    if request.form:
        f=request.form.to_dict()
        if 'useOBRetest' not in f: f['useOBRetest']='off'
        for c in ['useCloudFilter','useObvFilter','useRsiFilter']: 
            if c not in f: f[c]='off'
        settings.save_settings(f)
    market_analyzer.run_scan_thread(); return jsonify({"status":"started"})
@app.route('/analyzer/status')
def get_scan_status(): return jsonify({"progress":market_analyzer.progress, "message":market_analyzer.status_message, "is_scanning":market_analyzer.is_scanning})
@app.route('/webhook', methods=['POST'])
def webhook():
    try: d=json.loads(request.get_data(as_text=True)); r=bot_instance.place_order(d); return jsonify(r), 200 if r.get("status") in ["ok","ignored"] else 400
    except Exception as e: return jsonify({"error":str(e)}), 400
@app.route('/report')
def report_route(): from report import render_report_page; return render_report_page(bot_instance, request)
@app.route('/health')
def health(): return jsonify({"status":"ok"})
@app.route('/settings/export')
def export_settings(): from flask import Response; return Response(json.dumps(settings.get_all(), indent=4), mimetype='application/json', headers={'Content-Disposition':'attachment;filename=bot_settings.json'})
@app.route('/settings/import', methods=['POST'])
def import_settings():
    f = request.files['file']
    if f and f.filename: settings.import_settings(json.load(f)); return redirect(url_for('settings_general_page'))
    return "Error", 400
if __name__ == '__main__': app.run(host=config.HOST, port=config.PORT)
""",

    # --- TEMPLATES ---
    "templates/base.html": """<!DOCTYPE html><html lang="uk"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{% block title %}Trading Bot{% endblock %}</title><link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet"><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css"><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet"><script src="https://code.jquery.com/jquery-3.6.0.min.js"></script><script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script><style>:root{--primary:#4f46e5;--primary-hover:#4338ca;--bg-body:#f3f4f6;--bg-card:#ffffff;--text-main:#111827;--text-muted:#6b7280;--success:#10b981;--danger:#ef4444}body{background-color:var(--bg-body);font-family:'Inter',sans-serif;color:var(--text-main);font-size:14px}.navbar{background:var(--bg-card);box-shadow:0 1px 3px 0 rgba(0,0,0,0.1);padding:0.8rem 1rem}.navbar-brand{font-weight:700;color:var(--primary)!important;font-size:1.25rem}.nav-link-custom{color:var(--text-muted);font-weight:500;padding:0.5rem 1rem;border-radius:8px;text-decoration:none;transition:all 0.2s;border:1px solid transparent;margin-left:5px}.nav-link-custom:hover{background-color:#f9fafb;color:var(--primary);border-color:#e5e7eb}.text-up{color:var(--success);font-weight:700}.text-down{color:var(--danger);font-weight:700}.badge-buy{background-color:#d1fae5;color:#065f46;border:1px solid #a7f3d0}.badge-sell{background-color:#fee2e2;color:#991b1b;border:1px solid #fecaca}.card{border:none;border-radius:12px;box-shadow:0 2px 4px rgba(0,0,0,0.05)}.help-icon{font-size:0.9rem;color:#9ca3af;cursor:help;margin-left:5px;transition:color 0.2s}.help-icon:hover{color:var(--primary)}</style>{% block head %}{% endblock %}</head><body><nav class="navbar navbar-expand-lg navbar-light sticky-top"><div class="container"><a class="navbar-brand d-flex align-items-center" href="/"><span style="font-size:1.5rem;margin-right:8px">🤖</span> AlgoBot Pro</a><button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav"><span class="navbar-toggler-icon"></span></button><div class="collapse navbar-collapse" id="navbarNav"><ul class="navbar-nav ms-auto align-items-center"><li class="nav-item"><a class="nav-link-custom" href="/scanner">🐋 Монітор</a></li><li class="nav-item"><a class="nav-link-custom" href="/analyzer">🚀 Сканер</a></li><li class="nav-item"><a class="nav-link-custom" href="/settings">⚙️ Налаштування</a></li><li class="nav-item"><a class="nav-link-custom" href="/report" style="background:var(--primary);color:white">📊 Звіт</a></li></ul></div></div></nav>{% block content %}{% endblock %}<script>var tooltipTriggerList=[].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));var tooltipList=tooltipTriggerList.map(function(t){return new bootstrap.Tooltip(t)})</script></body></html>""",

    "templates/index.html": """{% extends "base.html" %}{% block title %}Dashboard{% endblock %}{% block head %}<style>.hero-section{background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);color:white;padding:3rem 0 4rem;border-radius:0 0 24px 24px;margin-bottom:-2rem}.dashboard-card{height:100%;border:1px solid rgba(255,255,255,0.5);cursor:pointer;text-decoration:none;color:inherit;display:block;background:white;border-radius:16px;box-shadow:0 4px 6px -1px rgba(0,0,0,0.1);transition:transform 0.2s}.dashboard-card:hover{transform:translateY(-5px);box-shadow:0 12px 20px -8px rgba(0,0,0,0.15)}.icon-box{width:64px;height:64px;background:#eff6ff;color:var(--primary);border-radius:16px;display:flex;align-items:center;justify-content:center;font-size:2rem;margin:0 auto 1.5rem}.card-title{font-weight:700;margin-bottom:0.5rem;color:#111827}.card-desc{color:#6b7280;font-size:0.9rem;line-height:1.5}</style>{% endblock %}{% block content %}<div class="hero-section text-center"><div class="container"><h1 class="display-5 fw-bold mb-2">🤖 Command Center</h1><p class="opacity-75">Панель керування торговою системою</p></div></div><div class="container pb-5"><div class="row g-4 justify-content-center"><div class="col-md-6 col-lg-3"><a href="/scanner" class="dashboard-card p-4 text-center"><div class="card-body"><div class="icon-box">🐋</div><h5 class="card-title">Монітор Угод</h5><p class="card-desc">Контроль активних позицій.</p></div></a></div><div class="col-md-6 col-lg-3"><a href="/analyzer" class="dashboard-card p-4 text-center"><div class="card-body"><div class="icon-box" style="background:#fdf2f8;color:#db2777">🚀</div><h5 class="card-title">Аналізатор</h5><p class="card-desc">Пошук сигналів та сканування.</p></div></a></div><div class="col-md-6 col-lg-3"><a href="/settings" class="dashboard-card p-4 text-center"><div class="card-body"><div class="icon-box" style="background:#f0fdf4;color:#16a34a">⚙️</div><h5 class="card-title">Налаштування</h5><p class="card-desc">Конфігурація бота.</p></div></a></div><div class="col-md-6 col-lg-3"><a href="/report" class="dashboard-card p-4 text-center"><div class="card-body"><div class="icon-box" style="background:#fff7ed;color:#ea580c">📊</div><h5 class="card-title">Звітність</h5><p class="card-desc">Статистика торгів.</p></div></a></div></div><div class="text-center mt-5 text-muted" style="font-size:0.8rem">Server Time: {{ time }} | Status: <span class="text-success fw-bold">Active</span></div></div>{% endblock %}""",

    "templates/scanner.html": """{% extends "base.html" %}{% block title %}Active Monitor{% endblock %}{% block content %}<div class="container mt-4"><div class="d-flex justify-content-between align-items-center mb-3"><h4 class="fw-bold mb-0">🐋 Монітор Активних Угод</h4><span class="badge bg-primary fs-6">{{ active|length }} Позицій</span></div><div class="card shadow-sm border-0"><div class="table-responsive"><table class="table table-hover align-middle mb-0"><thead class="table-light"><tr><th class="py-3">Час</th><th class="py-3">Монета</th><th class="py-3">Тип</th><th class="py-3">Розмір</th><th class="py-3">Вхід</th><th class="py-3">RSI</th><th class="py-3">P&L (USDT)</th></tr></thead><tbody>{% for a in active %}<tr><td class="text-muted small fw-bold">{{ a.time }}</td><td class="fw-bold text-primary">{{ a.symbol }}</td><td><span class="badge {{ 'badge-buy' if a.side=='Buy' else 'badge-sell' }}">{{ a.side }}</span></td><td>{{ a.size }}</td><td>{{ a.entry }}</td><td><span class="{{ 'text-danger fw-bold' if a.rsi > 70 else 'text-success fw-bold' if a.rsi < 30 else 'text-muted' }}">{{ a.rsi }}</span></td><td class="{{ 'text-up' if a.pnl > 0 else 'text-down' }}" style="font-size:1.1em">{{ "+" if a.pnl > 0 }}{{ a.pnl }}$</td></tr>{% else %}<tr><td colspan="7" class="text-center py-5 text-muted"><div style="font-size:2.5rem;margin-bottom:1rem">💤</div><h5>Немає активних угод</h5></td></tr>{% endfor %}</tbody></table></div></div></div>{% endblock %}""",

    "templates/analyzer.html": """{% extends "base.html" %}{% block title %}Аналізатор{% endblock %}{% block content %}<div class="container mt-4"><div class="card mb-4 shadow-sm border-0"><div class="card-header bg-white py-3"><div class="d-flex align-items-center"><span style="font-size:1.5rem;margin-right:10px">🚀</span><h5 class="fw-bold mb-0">Параметри Сканування</h5></div></div><div class="card-body bg-light"><form id="scan-form"><div class="row g-3 mb-4"><div class="col-md-3"><label class="form-label fw-bold small text-muted">Таймфрейм Тренду</label><select class="form-select" name="htfSelection"><option value="60" {{ 'selected' if conf.get('htfSelection')|string=='60' }}>1H</option><option value="240" {{ 'selected' if conf.get('htfSelection')|string=='240' }}>4H</option><option value="D" {{ 'selected' if conf.get('htfSelection')|string=='D' }}>1D</option></select></div><div class="col-md-3"><label class="form-label fw-bold small text-muted">Таймфрейм Входу</label><select class="form-select" name="ltfSelection"><option value="5" {{ 'selected' if conf.get('ltfSelection')|string=='5' }}>5m</option><option value="15" {{ 'selected' if conf.get('ltfSelection')|string=='15' }}>15m</option><option value="60" {{ 'selected' if conf.get('ltfSelection')|string=='60' }}>1H</option></select></div><div class="col-md-3"><label class="form-label fw-bold small text-muted">Глибина</label><div class="input-group"><span class="input-group-text bg-white">#</span><input type="number" class="form-control" name="scan_limit" value="{{ conf.get('scan_limit', 100) }}"></div></div><div class="col-md-3"><label class="form-label fw-bold small text-muted">Режим Входу</label><div class="form-check form-switch p-2 bg-white border rounded d-flex align-items-center justify-content-between px-3 shadow-sm" style="min-height:38px"><label class="form-check-label fw-bold text-primary m-0">OB Retest</label><input class="form-check-input m-0" type="checkbox" name="useOBRetest" {{ 'checked' if conf.get('useOBRetest') }}></div></div></div><div class="border-top my-4"></div><div class="row align-items-end g-3"><div class="col-md-9"><label class="form-label fw-bold small text-muted text-uppercase mb-2">Активні Фільтри</label><div class="row g-3"><div class="col-md-4"><div class="form-check form-switch p-2 bg-white border rounded d-flex align-items-center justify-content-between px-3 shadow-sm"><label class="form-check-label fw-bold text-dark m-0">Cloud HMA</label><input class="form-check-input m-0" type="checkbox" name="useCloudFilter" {{ 'checked' if conf.get('useCloudFilter') }}></div></div><div class="col-md-4"><div class="form-check form-switch p-2 bg-white border rounded d-flex align-items-center justify-content-between px-3 shadow-sm"><label class="form-check-label fw-bold text-dark m-0">OBV Trend</label><input class="form-check-input m-0" type="checkbox" name="useObvFilter" {{ 'checked' if conf.get('useObvFilter') }}></div></div><div class="col-md-4"><div class="form-check form-switch p-2 bg-white border rounded d-flex align-items-center justify-content-between px-3 shadow-sm"><label class="form-check-label fw-bold text-dark m-0">RSI Logic</label><input class="form-check-input m-0" type="checkbox" name="useRsiFilter" {{ 'checked' if conf.get('useRsiFilter') }}></div></div></div></div><div class="col-md-3"><button type="button" id="btn-scan" class="btn btn-primary btn-lg w-100 shadow fw-bold" onclick="startScan()" {{ 'disabled' if is_scanning }}>{{ '⏳ СКАНУВАННЯ...' if is_scanning else 'ЗБЕРЕГТИ ТА СКАНУВАТИ 🚀' }}</button></div></div></form></div></div><div class="card mb-4 border-0 shadow-sm" style="display:{{ 'block' if is_scanning else 'none' }}" id="status-card"><div class="card-body py-3" style="background:#e0e7ff"><div class="d-flex justify-content-between mb-2"><span id="status-text" class="fw-bold text-primary">{{ status }}</span><span id="status-percent" class="fw-bold text-dark">{{ progress }}%</span></div><div class="progress" style="height:12px;border-radius:6px;background:white"><div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-primary" style="width:{{ progress }}%"></div></div></div></div><div class="card shadow-sm border-0"><div class="card-header bg-white fw-bold py-3 d-flex justify-content-between align-items-center"><span>📋 РЕЗУЛЬТАТИ</span><span class="badge bg-secondary rounded-pill">{{ results|length }} Знахідок</span></div><div class="table-responsive"><table class="table table-hover align-middle mb-0"><thead class="table-light"><tr><th class="py-3">Монета</th><th class="py-3">Ціна</th><th class="py-3">Сигнал</th><th class="py-3">Score</th><th class="py-3">RSI (HTF)</th><th class="py-3">RSI (LTF)</th><th class="py-3">Час</th><th class="py-3">Деталі</th></tr></thead><tbody>{% for r in results %}<tr><td class="fw-bold text-primary">{{ r.symbol }}</td><td>{{ r.price }}</td><td><span class="badge {{ 'badge-buy' if r.signal=='Buy' else 'badge-sell' }}">{{ r.signal }}</span></td><td><div class="d-flex align-items-center"><span class="fw-bold me-2">{{ r.score }}</span><div class="progress flex-grow-1" style="height:6px;width:60px;border-radius:3px"><div class="progress-bar {{ 'bg-success' if r.score >= 70 else 'bg-warning' }}" style="width:{{ r.score }}%"></div></div></div></td><td>{{ r.rsi_htf }}</td><td>{{ r.rsi_ltf }}</td><td class="text-muted small">{{ r.time }}</td><td class="small text-muted" title="{{ r.details }}" style="max-width:250px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{{ r.details }}</td></tr>{% else %}<tr><td colspan="8" class="text-center py-5"><div class="text-muted mb-2" style="font-size:2.5rem">🔍</div><h5 class="text-muted">Сигналів поки не знайдено</h5></td></tr>{% endfor %}</tbody></table></div></div></div><script>function startScan(){$('#btn-scan').prop('disabled',true).html('<span class="spinner-border spinner-border-sm me-2"></span> Збереження...');$('#status-card').slideDown();var formData={};$('#scan-form').serializeArray().forEach(function(item){formData[item.name]=item.value});$('#scan-form input[type=checkbox]').each(function(){formData[this.name]=this.checked?'on':'off'});$.post('/analyzer/scan',formData,function(data){pollStatus()}).fail(function(){alert('Помилка.');$('#btn-scan').prop('disabled',false).text('ЗБЕРЕГТИ ТА СКАНУВАТИ 🚀');$('#status-card').hide()})}function pollStatus(){let interval=setInterval(function(){$.get('/analyzer/status',function(data){$('#progress-bar').css('width',da# ta.progress+'%');$('#status-percent').text(da# ta.progress+'%');$('#status-text').text(da# ta.message);if(da# ta.is_scanning){$('#btn-scan').prop('disabled',true).html('Сканування...')}else{clearInterval(interval);location.reload()}})},1000)}{% if is_scanning %}pollStatus(){% endif %}</script>{% endblock %}""",

    "templates/settings.html": """{% extends "base.html" %}{% block title %}Settings{% endblock %}{% block content %}<div class="container mt-4" style="max-width:700px"><h4 class="fw-bold mb-4">⚙️ Загальні Налаштування</h4><div class="card mb-4 border-primary"><div class="card-header bg-primary text-white fw-bold d-flex justify-content-between align-items-center"><span>💾 Резервне копіювання</span><i class="bi bi-hdd-network"></i></div><div class="card-body"><div class="row align-items-center"><div class="col-md-6 mb-3 mb-md-0 border-end"><h6 class="fw-bold text-muted mb-2">Експорт</h6><a href="/settings/export" class="btn btn-outline-primary btn-sm w-100"><i class="bi bi-download me-2"></i>Завантажити файл</a></div><div class="col-md-6 ps-md-4"><h6 class="fw-bold text-muted mb-2">Імпорт</h6><form action="/settings/import" method="POST" enctype="multipart/form-data" class="d-flex gap-2"><input class="form-control form-control-sm" type="file" name="file" accept=".json" required><button type="submit" class="btn btn-primary btn-sm"><i class="bi bi-upload"></i></button></form></div></div></div></div><form method="POST"><div class="card mb-4"><div class="card-header text-primary fw-bold">🌍 Глобальні Параметри</div><div class="card-body"><div class="row g-3"><div class="col-md-6"><label class="form-label">Валюта Торгівлі</label><select class="form-select" name="scanner_quote_coin"><option value="USDT" {{ 'selected' if conf.get('scanner_quote_coin')=='USDT' }}>USDT</option><option value="USDC" {{ 'selected' if conf.get('scanner_quote_coin')=='USDC' }}>USDC</option></select></div><div class="col-md-6"><label class="form-label">Режим Сканера</label><select class="form-select" name="scanner_mode"><option value="Manual" {{ 'selected' if conf.get('scanner_mode')=='Manual' }}>Manual</option><option value="Auto" {{ 'selected' if conf.get('scanner_mode')=='Auto' }}>Auto</option></select></div></div></div></div><div class="card mb-4"><div class="card-header text-danger fw-bold">💰 Ризик Менеджмент</div><div class="card-body"><div class="row g-3"><div class="col-md-6"><label class="form-label">Ризик (%)</label><input type="number" step="0.1" class="form-control" name="riskPercent" value="{{ conf.get('riskPercent') }}"></div><div class="col-md-6"><label class="form-label">Плече (x)</label><input type="number" class="form-control" name="leverage" value="{{ conf.get('leverage') }}"></div><div class="col-md-12"><label class="form-label fw-bold">TP Strategy</label><select class="form-select" name="tp_mode"><option value="None" {{ 'selected' if conf.get('tp_mode')=='None' }}>None</option><option value="Fixed_1_50" {{ 'selected' if conf.get('tp_mode')=='Fixed_1_50' }}>Fixed 1% (50%)</option><option value="Ladder_3" {{ 'selected' if conf.get('tp_mode')=='Ladder_3' }}>Ladder (3 steps)</option></select></div></div></div></div><div class="card mb-4"><div class="card-header fw-bold" style="color:#0891b2">✈️ Telegram</div><div class="card-body"><div class="form-check form-switch mb-3"><input class="form-check-input" type="checkbox" name="telegram_enabled" {{ 'checked' if conf.get('telegram_enabled') }}><label class="form-check-label fw-bold">Увімкнути</label></div><div class="mb-3"><label class="form-label">Token</label><input type="text" class="form-control" name="telegram_bot_token" value="{{ conf.get('telegram_bot_token','') }}"></div><div class="mb-3"><label class="form-label">Chat ID</label><input type="text" class="form-control" name="telegram_chat_id" value="{{ conf.get('telegram_chat_id','') }}"></div></div></div><div class="d-grid gap-2 mb-5"><button type="submit" class="btn btn-primary btn-lg shadow-sm">💾 Зберегти Зміни</button></div></form></div>{% endblock %}""",

    "templates/strategy.html": """{% extends "base.html" %}{% block title %}Strategy{% endblock %}{% block content %}<div class="container mt-4" style="max-width:800px"><div class="d-flex justify-content-between align-items-center mb-4"><h4 class="fw-bold mb-0">📊 Стратегія</h4><a href="/settings" class="btn btn-outline-secondary btn-sm">← Загальні</a></div><form method="POST"><div class="card"><div class="card-header text-success">🎛️ Логіка</div><div class="card-body"><div class="row mb-4"><div class="col-md-3"><div class="form-check form-switch p-3 bg-light rounded h-100"><input class="form-check-input" type="checkbox" name="useCloudFilter" {{ 'checked' if conf.get('useCloudFilter') }}><label class="form-check-label fw-bold">Cloud</label></div></div><div class="col-md-3"><div class="form-check form-switch p-3 bg-light rounded h-100"><input class="form-check-input" type="checkbox" name="useObvFilter" {{ 'checked' if conf.get('useObvFilter') }}><label class="form-check-label fw-bold">OBV</label></div></div><div class="col-md-3"><div class="form-check form-switch p-3 bg-light rounded h-100"><input class="form-check-input" type="checkbox" name="useRsiFilter" {{ 'checked' if conf.get('useRsiFilter') }}><label class="form-check-label fw-bold">RSI</label></div></div><div class="col-md-3"><div class="form-check form-switch p-3 rounded h-100" style="background:#e0e7ff;border:1px solid #4f46e5"><input class="form-check-input" type="checkbox" name="useOBRetest" {{ 'checked' if conf.get('useOBRetest') }}><label class="form-check-label fw-bold text-primary">OB Retest</label></div></div></div><div class="row"><div class="col-md-6"><label class="form-label">HTF</label><select class="form-select" name="htfSelection"><option value="60" {{ 'selected' if conf.get('htfSelection')|string=='60' }}>1H</option><option value="240" {{ 'selected' if conf.get('htfSelection')|string=='240' }}>4H</option></select></div><div class="col-md-6"><label class="form-label">LTF</label><select class="form-select" name="ltfSelection"><option value="5" {{ 'selected' if conf.get('ltfSelection')|string=='5' }}>5m</option><option value="15" {{ 'selected' if conf.get('ltfSelection')|string=='15' }}>15m</option></select></div></div></div></div><div class="card mt-4"><div class="card-header" style="color:#7c3aed">📈 Індикатори</div><div class="card-body"><div class="row g-3"><div class="col-md-4"><label class="form-label">RSI Len</label><input type="number" class="form-control" name="rsiLength" value="{{ conf.get('rsiLength') }}"></div><div class="col-md-4"><label class="form-label text-success">Buy <=</label><input type="number" class="form-control" name="entryRsiOversold" value="{{ conf.get('entryRsiOversold') }}"></div><div class="col-md-4"><label class="form-label text-danger">Sell >=</label><input type="number" class="form-control" name="entryRsiOverbought" value="{{ conf.get('entryRsiOverbought') }}"></div><div class="col-md-4"><label class="form-label">Cloud Fast</label><input type="number" class="form-control" name="cloudFastLen" value="{{ conf.get('cloudFastLen') }}"></div><div class="col-md-4"><label class="form-label">Cloud Slow</label><input type="number" class="form-control" name="cloudSlowLen" value="{{ conf.get('cloudSlowLen') }}"></div><div class="col-md-4"><label class="form-label fw-bold">OB Swing</label><input type="number" class="form-control" name="swingLength" value="{{ conf.get('swingLength') }}"></div></div></div></div><div class="d-grid gap-2 mb-5 mt-4"><button type="submit" class="btn btn-success btn-lg">💾 Зберегти</button></div></form></div>{% endblock %}""",

    "templates/report.html": """{% extends "base.html" %}{% block title %}Report{% endblock %}{% block head %}<script src="https://cdn.jsdelivr.net/npm/chart.js"></script><style>.kpi-card{background:white;border-radius:12px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,0.04);height:100%;display:flex;flex-direction:column;justify-content:center}.kpi-value{font-size:1.5rem;font-weight:700;color:#1e2329;margin-bottom:5px}.filter-btn{border:1px solid #eaecef;color:#1e2329;padding:6px 16px;border-radius:20px;text-decoration:none;font-size:0.9rem;margin-left:5px;transition:0.2s}.filter-btn:hover{background-color:#f3f4f6}.filter-btn.active{background:#fcd535;border-color:#fcd535;font-weight:600;box-shadow:0 2px 4px rgba(252,213,53,0.2)}.type-long{color:#20b26c;background:rgba(32,178,108,0.1);padding:4px 8px;border-radius:4px;font-size:12px;font-weight:600}.type-short{color:#ef454a;background:rgba(239,69,74,0.1);padding:4px 8px;border-radius:4px;font-size:12px;font-weight:600}.ls-badge{font-size:0.85rem;font-weight:600;padding:3px 10px;border-radius:6px}.ls-long{color:#20b26c;background:#e6f8f0;border:1px solid #bdf0db}.ls-short{color:#ef454a;background:#feeced;border:1px solid #fad1d3}</style>{% endblock %}{% block content %}<div class="container mt-4 mb-5"><div class="d-flex justify-content-between align-items-center mb-4"><div><h4 class="fw-bold mb-1">P&L Аналіз</h4><p class="text-muted small mb-0">Статистика</p></div><div><a href="/report?days=7" class="filter-btn {{ 'active' if days==7 }}">7 Днів</a><a href="/report?days=30" class="filter-btn {{ 'active' if days==30 }}">30 Днів</a><a href="/report?days=90" class="filter-btn {{ 'active' if days==90 }}">90 Днів</a></div></div><div class="row g-3 mb-4"><div class="col-md-3"><div class="kpi-card"><div class="text-muted small mb-1">P&L (USDT)</div><div class="kpi-value {{ 'text-success' if stats.total_pnl >= 0 else 'text-danger' }}">{{ "+" if stats.total_pnl > 0 }}{{ stats.total_pnl|round(2) }}</div><div class="small text-muted">Vol: {{ stats.volume|round(0) }}</div></div></div><div class="col-md-3"><div class="kpi-card"><div class="text-muted small mb-1">Win Rate</div><div class="kpi-value">{{ stats.win_rate }}%</div><div class="progress mt-2" style="height:6px"><div class="progress-bar bg-success" style="width:{{ stats.win_rate }}%"></div></div></div></div><div class="col-md-3"><div class="kpi-card"><div class="text-muted small mb-1">Profit Factor</div><div class="kpi-value">{{ stats.profit_factor }}</div><div class="small text-muted">Avg: {{ stats.avg_profit|round(2) }}$</div></div></div><div class="col-md-3"><div class="kpi-card"><div class="text-muted small mb-1">Всього Угод</div><div class="kpi-value mb-2">{{ stats.total_trades }}</div><div class="d-flex gap-2"><span class="ls-badge ls-long">L: {{ stats.longs }}</span><span class="ls-badge ls-short">S: {{ stats.shorts }}</span></div></div></div></div><div class="row g-4 mb-4"><div class="col-lg-8"><div class="card shadow-sm border-0 h-100"><div class="card-header bg-white fw-bold border-bottom-0 pt-3">📈 Equity</div><div class="card-body"><canvas id="equityChart" height="250"></canvas></div></div></div><div class="col-lg-4"><div class="card shadow-sm border-0 h-100"><div class="card-header bg-white fw-bold border-bottom-0 pt-3">📅 Daily P&L</div><div class="card-body"><canvas id="dailyChart" height="250"></canvas></div></div></div></div><div class="card shadow-sm border-0"><div class="card-header bg-white py-3 fw-bold border-bottom">📜 Історія</div><div class="table-responsive"><table class="table table-hover align-middle mb-0" style="font-family:'Inter',sans-serif"><thead class="table-light"><tr><th class="ps-4">Час</th><th>Пара</th><th>Тип</th><th>Вхід</th><th>Вихід</th><th>К-сть</th><th>P&L</th><th>Статус</th></tr></thead><tbody>{% for t in trades %}<tr><td class="ps-4 text-muted small">{{ t.exit_time.strftime('%d.%m %H:%M') }}</td><td class="fw-bold text-primary">{{ t.symbol }}</td><td><span class="{{ 'type-long' if t.side=='Long' else 'type-short' }}">{{ t.side }}</span></td><td>{{ t.entry_price|round(4) }}</td><td>{{ t.exit_price|round(4) }}</td><td>{{ t.qty }}</td><td class="fw-bold {{ 'text-success' if t.pnl>0 else 'text-danger' }}">{{ "+" if t.pnl>0 }}{{ t.pnl|round(2) }}$</td><td>{% if t.is_win %}<span class="badge bg-success bg-opacity-10 text-success">WIN</span>{% else %}<span class="badge bg-danger bg-opacity-10 text-danger">LOSS</span>{% endif %}</td></tr>{% else %}<tr><td colspan="8" class="text-center py-5 text-muted">Історія порожня</td></tr>{% endfor %}</tbody></table></div></div></div><script>const ctxEquity=document.getElementById('equityChart').getContext('2d');let gradient=ctxEquity.createLinearGradient(0,0,0,300);gradient.addColorStop(0,'rgba(32,178,108,0.2)');gradient.addColorStop(1,'rgba(32,178,108,0.0)');new Chart(ctxEquity,{type:'line',data:{labels:{{ chart_labels|safe }},datasets:[{label:'Cumulative P&L',data:{{ chart_equity|safe }},borderColor:'#20b26c',backgroundColor:gradient,borderWidth:2,pointRadius:0,pointHoverRadius:4,fill:true,tension:0.3}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false}},y:{grid:{color:'#f7f8fa'}}},interaction:{mode:'index',intersect:false}}});const ctxDaily=document.getElementById('dailyChart').getContext('2d');const dailyData={{ daily_values|safe }};new Chart(ctxDaily,{type:'bar',data:{labels:{{ daily_labels|safe }},datasets:[{data:dailyData,backgroundColor:dailyDa# ta.map(v=>v>=0?'#20b26c':'#ef454a'),borderRadius:4}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{display:false},y:{display:false}}}});</script>{% endblock %}"""
}

import os

# Створення структури проекту
if not os.path.exists(PROJECT_ROOT):
    os.makedirs(PROJECT_ROOT)

# Створення папки templates
templates_dir = os.path.join(PROJECT_ROOT, "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)

# Запис файлів
for filename, content in files.items():
    path = os.path.join(PROJECT_ROOT, filename)
    
    # Якщо це файл шаблону, він має бути в папці templates
    if filename.startswith("templates/"):
        path = os.path.join(PROJECT_ROOT, filename) # вже містить 'templates/' у ключі
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Created: {filename}")

print("\n🎉 Project setup complete in folder: " + PROJECT_ROOT)
print("To run locally:")
print(f"cd {PROJECT_ROOT}")
print("pip install -r requirements.txt")
print("python main_app.py")
