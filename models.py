from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

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
    exit_reason = Column(String(100)) # Причина виходу

class TradeMonitorLog(Base):
    __tablename__ = 'trade_monitor_logs'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    current_price = Column(Float)
    current_pnl = Column(Float)
    rsi = Column(Float)
    pressure = Column(Float)

# ==================== НОВЫЕ ТАБЛИЦЫ ДЛЯ РАСШИРЕННОЙ АНАЛИТИКИ ====================

class PositionAnalytics(Base):
    """Детальная аналитика по закрытым позициям"""
    __tablename__ = 'position_analytics'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(String(10))  # Long/Short
    open_time = Column(DateTime)
    close_time = Column(DateTime, default=datetime.utcnow)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float)
    pnl_percent = Column(Float)
    max_pnl = Column(Float)  # Максимальный P&L
    min_pnl = Column(Float)  # Минимальный P&L (просадка)
    avg_rsi = Column(Float)
    rsi_range = Column(String(20))  # "30-75"
    avg_mfi = Column(Float)
    mfi_trend = Column(String(20))  # Bullish/Bearish/Neutral
    signal_count = Column(Integer)  # Количество сигналов
    hold_duration = Column(Integer)  # Секунды
    close_reason = Column(String(100))
    close_method = Column(String(50))  # auto/manual

class PositionSnapshot(Base):
    """Снимки состояния активных позиций (каждую минуту)"""
    __tablename__ = 'position_snapshots'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    price = Column(Float)
    pnl = Column(Float)
    rsi = Column(Float)
    mfi = Column(Float)
    mfi_trend = Column(String(20))  # Bullish/Bearish/Neutral
    current_signal = Column(String(50))
    momentum = Column(String(20))  # Растет/Падает

class MarketCandidate(Base):
    """Кандидаты для входа из сканирования рынка"""
    __tablename__ = 'market_candidates'
    id = Column(Integer, primary_key=True)
    scan_id = Column(Integer, index=True)  # Связь со сканированием
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    direction = Column(String(10))  # Long/Short
    signal_strength = Column(String(20))  # Strong/Regular
    rsi = Column(Float)
    mfi = Column(Float)
    mfi_trend = Column(String(20))
    price = Column(Float)
    volume_24h = Column(Float)
    change_24h = Column(Float)
    rating = Column(Float)  # 0-100
    reason = Column(Text)  # Причина рекомендации

class ScanHistory(Base):
    """История сканирования рынка"""
    __tablename__ = 'scan_history'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    total_coins = Column(Integer)
    filtered_coins = Column(Integer)
    candidates_found = Column(Integer)
    scan_duration = Column(Float)  # Секунды
    top_candidate = Column(String(20))
    top_rating = Column(Float)

class AutoCloseDecision(Base):
    """Решения системы автозакрытия"""
    __tablename__ = 'auto_close_decisions'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    decision = Column(String(20))  # close/hold
    reason = Column(Text)
    rsi = Column(Float)
    mfi = Column(Float)
    signal = Column(String(50))
    pnl_at_decision = Column(Float)
    executed = Column(Boolean)

class ScannerConfig(Base):
    """Конфигурация сканера (сохранение настроек)"""
    __tablename__ = 'scanner_config'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    trading_style = Column(String(20))
    aggressiveness = Column(String(20))
    automation_mode = Column(String(20))
    params_json = Column(Text)  # JSON со всеми параметрами
    is_active = Column(Boolean, default=True)

# Заглушки для сумісності (якщо старий код їх викличе)
class WhaleSignal(Base):
    __tablename__ = 'whale_signals'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
class CoinStatistics(Base):
    __tablename__ = 'coin_statistics'
    id = Column(Integer, primary_key=True)
class CoinPerformance(Base):
    __tablename__ = 'coin_performance'
    id = Column(Integer, primary_key=True)

class DatabaseManager:
    def __init__(self, db_path='trading_bot_final.db'):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self): return self.Session()

db_manager = DatabaseManager()