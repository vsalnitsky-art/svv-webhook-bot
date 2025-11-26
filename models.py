"""
Database Models - Full V3 (With Monitor Log)
"""
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class WhaleSignal(Base):
    __tablename__ = 'whale_signals'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    price = Column(Float)
    volume_inflow = Column(Float)
    spike_factor = Column(Float)
    price_change_1min = Column(Float)
    turnover_24h = Column(Float)

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
    exit_rsi = Column(Float)
    exit_pressure = Column(Float)

# ✅ ЦЯ ТАБЛИЦЯ ПОТРІБНА ДЛЯ ЛІВОГО БЛОКУ
class TradeMonitorLog(Base):
    __tablename__ = 'trade_monitor_logs'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    current_price = Column(Float)
    current_pnl = Column(Float)
    rsi = Column(Float)
    pressure = Column(Float)
    session_id = Column(String(50), index=True) 

class CoinStatistics(Base):
    __tablename__ = 'coin_statistics'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True)
    total_signals = Column(Integer, default=0)
    total_inflow_24h = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)

class CoinPerformance(Base):
    __tablename__ = 'coin_performance'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)

class DatabaseManager:
    # Використовуємо v3, щоб створити чисту базу з новою структурою
    def __init__(self, db_path='trading_bot_v3.db'):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self): return self.Session()
    def close(self): self.engine.dispose()

db_manager = DatabaseManager()
