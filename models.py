"""
Database Models для статистики торгівлі
Використовує SQLite для простоти та SQLAlchemy для ORM
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class WhaleSignal(Base):
    """Зберігає всі сигнали від whale scanner"""
    __tablename__ = 'whale_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Дані про вливання
    price = Column(Float, nullable=False)
    volume_inflow = Column(Float, nullable=False)  # $ за період
    spike_factor = Column(Float, nullable=False)    # Множник аномалії
    price_change_1min = Column(Float, nullable=False)  # % зміни ціни
    turnover_24h = Column(Float)  # Загальний об'єм за добу
    
    # Мета-дані
    scan_interval = Column(Integer, default=60)  # Секунд між сканами
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class CoinStatistics(Base):
    """Агрегована статистика по кожній монеті"""
    __tablename__ = 'coin_statistics'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    
    # Накопичені дані
    total_signals = Column(Integer, default=0)
    total_inflow_24h = Column(Float, default=0.0)
    total_inflow_7d = Column(Float, default=0.0)
    total_inflow_30d = Column(Float, default=0.0)
    
    avg_spike_factor = Column(Float, default=0.0)
    max_spike_factor = Column(Float, default=0.0)
    
    # Цінова динаміка
    avg_price_change = Column(Float, default=0.0)
    positive_signals = Column(Integer, default=0)
    negative_signals = Column(Integer, default=0)
    
    # Часові мітки
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Trade(Base):
    """Закриті угоди (дублює дані з Bybit для швидкого доступу)"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Деталі угоди
    side = Column(String(10), nullable=False)  # Long/Short
    qty = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    
    # Результат
    pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float)
    is_win = Column(Boolean, nullable=False)
    
    # Параметри
    leverage = Column(Integer)
    volume_usd = Column(Float)
    
    # Часові мітки
    entry_time = Column(DateTime)
    exit_time = Column(DateTime, index=True)
    duration_minutes = Column(Integer)
    
    __table_args__ = (
        Index('idx_symbol_exit_time', 'symbol', 'exit_time'),
    )

class TradingSession(Base):
    """Денні торгові сесії для аналітики"""
    __tablename__ = 'trading_sessions'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, index=True)
    
    # Фінансові показники
    starting_balance = Column(Float)
    ending_balance = Column(Float)
    daily_pnl = Column(Float)
    daily_pnl_percent = Column(Float)
    
    # Статистика угод
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    
    total_volume = Column(Float, default=0.0)
    
    # Ризик-менеджмент
    max_drawdown = Column(Float)
    largest_win = Column(Float)
    largest_loss = Column(Float)

class CoinPerformance(Base):
    """Історична продуктивність кожної монети"""
    __tablename__ = 'coin_performance'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Торгова статистика
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    
    # P&L
    total_pnl = Column(Float, default=0.0)
    avg_pnl_per_trade = Column(Float)
    best_trade = Column(Float)
    worst_trade = Column(Float)
    
    # Об'єми
    total_volume = Column(Float, default=0.0)
    avg_volume_per_trade = Column(Float)
    
    # Сторони
    long_trades = Column(Integer, default=0)
    short_trades = Column(Integer, default=0)
    long_win_rate = Column(Float)
    short_win_rate = Column(Float)
    
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_coin_performance', 'symbol', 'total_pnl'),
    )

# === DATABASE MANAGER ===

class DatabaseManager:
    def __init__(self, db_path='trading_bot.db'):
        """Ініціалізація бази даних"""
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Отримати нову сесію"""
        return self.Session()
    
    def close(self):
        """Закрити з'єднання"""
        self.engine.dispose()

# Глобальний інстанс
db_manager = DatabaseManager()
