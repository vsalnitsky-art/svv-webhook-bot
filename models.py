"""
Database Models - Updated (V2)
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

class CoinStatistics(Base):
    __tablename__ = 'coin_statistics'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True)
    total_signals = Column(Integer, default=0)
    total_inflow_24h = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow)

class Trade(Base):
    """Закриті угоди з деталями виходу"""
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
    
    # Нові колонки (тепер точно будуть створені)
    exit_reason = Column(String(100))
    exit_rsi = Column(Float)
    exit_pressure = Column(Float)

class CoinPerformance(Base):
    __tablename__ = 'coin_performance'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)

class DatabaseManager:
    # 🔥 ЗМІНЕНО НАЗВУ ФАЙЛУ ТУТ 👇
    def __init__(self, db_path='trading_bot_v2.db'):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self): return self.Session()
    def close(self): self.engine.dispose()

db_manager = DatabaseManager()
