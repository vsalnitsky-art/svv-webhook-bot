from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)
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
    # === НОВА КОЛОНКА ОБ'ЄМУ ===
    volume_24h = Column(Float, default=0.0)

class OrderBlock(Base):
    __tablename__ = 'order_blocks'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), index=True)
    timeframe = Column(String(10)); ob_type = Column(String(10))
    top = Column(Float); bottom = Column(Float); entry_price = Column(Float); sl_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow); status = Column(String(20), default='PENDING'); volume_score = Column(Float, default=0.0)

class SmartMoneyTicker(Base):
    __tablename__ = 'smart_money_watchlist'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), unique=True, index=True); added_at = Column(DateTime, default=datetime.utcnow)

class PaperTrade(Base):
    __tablename__ = 'paper_trades'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), index=True)
    direction = Column(String(10)); entry_mode = Column(String(20)); status = Column(String(20))
    entry_price = Column(Float); sl_price = Column(Float); tp_price = Column(Float, nullable=True); exit_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0); pnl_percent = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow); closed_at = Column(DateTime, nullable=True); details = Column(String(255))

class DatabaseManager:
    def __init__(self, db_filename='trading_bot_final.db'):
        db_url = os.environ.get('DATABASE_URL')
        
        # === ДІАГНОСТИКА ПІДКЛЮЧЕННЯ ===
        if db_url:
            print(f"\n🔌 DATABASE: FOUND DATABASE_URL. CONNECTING TO POSTGRESQL...")
            if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
        else:
            print(f"\n⚠️ WARNING: DATABASE_URL NOT FOUND. USING LOCAL SQLITE (DATA WILL BE LOST ON RESTART).")
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
    
    # === ФУНКЦІЯ ПРИМУСОВОГО ПЕРЕСТВОРЕННЯ ТАБЛИЦІ ===
    def recreate_analysis_table(self):
        try:
            AnalysisResult.__table__.drop(self.engine, checkfirst=True)
            AnalysisResult.__table__.create(self.engine, checkfirst=True)
            # print("✅ Table 'analysis_results' recreated successfully.")
        except Exception as e:
            print(f"❌ Error recreating table: {e}")

db_manager = DatabaseManager()