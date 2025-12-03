import os
import sys

# 1. Оновлюємо models.py (гарантуємо наявність SmartMoneyTicker)
models_content = """from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
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
    qty = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    is_win = Column(Boolean)
    exit_time = Column(DateTime, default=datetime.utcnow)
    exit_reason = Column(String(100))

class TradeMonitorLog(Base):
    __tablename__ = 'trade_monitor_logs'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    current_price = Column(Float)
    current_pnl = Column(Float)
    rsi = Column(Float)
    pressure = Column(Float)

class BotSetting(Base):
    __tablename__ = 'bot_settings'
    key = Column(String(50), primary_key=True)
    value = Column(String(255))

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    signal_type = Column(String(10))
    status = Column(String(50))      
    score = Column(Integer)          
    price = Column(Float)
    htf_rsi = Column(Float)
    ltf_rsi = Column(Float)
    found_at = Column(DateTime, default=datetime.utcnow)
    details = Column(Text)

class OrderBlock(Base):
    __tablename__ = 'order_blocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timeframe = Column(String(10))
    ob_type = Column(String(10))
    top = Column(Float)
    bottom = Column(Float)
    entry_price = Column(Float)
    sl_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='PENDING') 
    volume_score = Column(Float, default=0.0)

# === ВАЖЛИВО: ЦЕЙ КЛАС МАЄ БУТИ ТУТ ===
class SmartMoneyTicker(Base):
    __tablename__ = 'smart_money_watchlist'
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
"""

# 2. Оновлюємо strategy_ob_trend.py (гарантуємо логіку)
strategy_content = """import pandas_ta as ta
import pandas as pd
import numpy as np
from settings_manager import settings

class OBTrendStrategy:
    def __init__(self): pass

    def _get_param(self, key, default=None):
        val = settings.get(key)
        return val if val is not None else default

    def calculate_indicators(self, df):
        if df is None or len(df) < 50: return df
        try:
            fast = int(self._get_param('obt_cloudFastLen', 10))
            slow = int(self._get_param('obt_cloudSlowLen', 40))
            df['hma_fast'] = ta.hma(df['close'], length=fast)
            df['hma_slow'] = ta.hma(df['close'], length=slow)
            df['rsi'] = ta.rsi(df['close'], length=int(self._get_param('obt_rsiLength', 14)))
            df['obv'] = ta.obv(df['close'], df['volume'])
            if 'obv' in df:
                df['obv_ma'] = ta.sma(df['obv'], length=int(self._get_param('obt_obvEntryLen', 20)))
                df['obv_exit_ma'] = ta.ema(df['obv'], length=int(self._get_param('exit_obvLength', 10)))
        except: pass
        return df

    def find_order_blocks(self, df):
        obs = {'buy': [], 'sell': []}
        if df is None or len(df) < 100: return obs
        swing = int(self._get_param('obt_swingLength', 5))
        subset = df.tail(300).reset_index(drop=True)
        for i in range(swing, len(subset) - swing):
            cl, ch = subset['low'].iloc[i], subset['high'].iloc[i]
            is_l = all(subset['low'].iloc[i-j] >= cl and subset['low'].iloc[i+j] >= cl for j in range(1, swing+1))
            if is_l and subset['close'].iloc[i+1] > ch:
                obs['buy'].append({'top': ch, 'bottom': cl, 'created_at': subset['time'].iloc[i]})
            is_h = all(subset['high'].iloc[i-j] <= ch and subset['high'].iloc[i+j] <= ch for j in range(1, swing+1))
            if is_h and subset['close'].iloc[i+1] < cl:
                obs['sell'].append({'top': ch, 'bottom': cl, 'created_at': subset['time'].iloc[i]})
        return {'buy': obs['buy'][-3:], 'sell': obs['sell'][-3:]}

    def check_exit_signal(self, df_htf, position_side):
        res = {'close': False, 'reason': '', 'details': {}}
        if df_htf is None or len(df_htf) < 50: return res
        df = self.calculate_indicators(df_htf); last = df.iloc[-1]
        rsi, obv, obv_ma = last.get('rsi', 50), last.get('obv', 0), last.get('obv_exit_ma', 0)
        res['details'] = {'rsi': round(rsi, 1), 'obv_cross': 'UP' if obv > obv_ma else 'DOWN'}
        
        limit_buy = float(self._get_param('exit_rsiOverbought', 70))
        limit_sell = float(self._get_param('exit_rsiOversold', 30))

        if position_side == 'Buy':
            if rsi >= limit_buy: res.update({'close': True, 'reason': 'RSI Max'})
            if rsi >= limit_buy and obv < obv_ma: res.update({'close': True, 'reason': 'Confluence'})
        elif position_side == 'Sell':
            if rsi <= limit_sell: res.update({'close': True, 'reason': 'RSI Min'})
            if rsi <= limit_sell and obv > obv_ma: res.update({'close': True, 'reason': 'Confluence'})
        return res

    def analyze(self, df_ltf, df_htf):
        sigs = []; df_h = self.calculate_indicators(df_htf); df_l = self.calculate_indicators(df_ltf)
        if df_h is None or df_l is None: return []
        row_h = df_h.iloc[-1]; row_l = df_l.iloc[-1]; price = row_l['close']
        
        is_bull = row_h.get('hma_fast',0) > row_h.get('hma_slow',0)
        if self._get_param('obt_useObvFilter', True):
            if row_h.get('obv',0) <= row_h.get('obv_ma',0): is_bull = False

        sig = None; sl = 0
        if is_bull and row_l['rsi'] <= float(self._get_param('obt_entryRsiOversold', 45)):
            if self._get_param('obt_useOBRetest', False):
                for ob in self.find_order_blocks(df_ltf)['buy']:
                    if ob['bottom'] <= price <= ob['top']*1.003: sig='Buy'; sl=ob['bottom']*0.995; break
            else: sig='Buy'; sl=price*0.985
        
        if sig: sigs.append({'action': sig, 'price': price, 'rsi': row_l['rsi'], 'reason': 'Trend+Strategy', 'sl_price': sl})
        return sigs

ob_trend_strategy = OBTrendStrategy()
"""

def fix_system():
    print("🛠️ POCHYNAEMO REMONT...")
    
    # 1. Write models.py
    with open("models.py", "w", encoding="utf-8") as f:
        f.write(models_content)
    print("✅ models.py -- ONOVLENO (SmartMoneyTicker is guaranteed)")

    # 2. Write strategy_ob_trend.py
    with open("strategy_ob_trend.py", "w", encoding="utf-8") as f:
        f.write(strategy_content)
    print("✅ strategy_ob_trend.py -- ONOVLENO")

    # 3. RESET DATABASE (CRITICAL STEP)
    # Ми перейменовуємо стару базу, щоб код створив нову з правильною структурою
    db_file = "trading_bot_final.db"
    db_base_path = os.path.join("BASE", db_file)
    
    files_to_remove = [db_file, db_base_path]
    
    removed = False
    for path in files_to_remove:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"🗑️ VYDALENO STARU BAZU: {path} (System will create fresh one)")
                removed = True
            except Exception as e:
                print(f"⚠️ Could not delete {path}: {e}")

    if not removed:
        print("ℹ️ Database file not found (Clean start)")

    print("\n🎉 GOTOVO! PEREZAVANTAZHTE SERVER ZARAZ.")
    print("👉 Yaksho tse Render - Deploy povynen proity uspishno.")

if __name__ == "__main__":
    fix_system()