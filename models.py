from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# === ТОРГОВА ІСТОРІЯ ===
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

# === МОНІТОРИНГ АКТИВНИХ ===
class TradeMonitorLog(Base):
    __tablename__ = 'trade_monitor_logs'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    current_price = Column(Float)
    current_pnl = Column(Float)
    rsi = Column(Float)
    pressure = Column(Float)

# === НАЛАШТУВАННЯ ===
class BotSetting(Base):
    __tablename__ = 'bot_settings'
    key = Column(String(50), primary_key=True)
    value = Column(String(255))

# === РЕЗУЛЬТАТИ СКАНЕРА ===
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

# === SMART MONEY ORDER BLOCKS (Накопичення) ===
class OrderBlock(Base):
    __tablename__ = 'order_blocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    timeframe = Column(String(10))      # "15", "60", "240"
    ob_type = Column(String(10))        # "Buy" або "Sell"
    
    # Координати зони
    top = Column(Float)
    bottom = Column(Float)
    
    # Параметри
    entry_price = Column(Float)
    sl_price = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='PENDING') 
    volume_score = Column(Float, default=0.0)

# === DATABASE MANAGER ===
class DatabaseManager:
    def __init__(self, db_filename='trading_bot_final.db'):
        # Підтримка PostgreSQL для Render
        db_url = os.environ.get('DATABASE_URL')
        if db_url and db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        else:
            # Локальна SQLite
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_folder = os.path.join(current_dir, 'BASE')
            try:
                os.makedirs(base_folder, exist_ok=True)
                db_path = os.path.join(base_folder, db_filename)
            except OSError:
                db_path = os.path.join(current_dir, db_filename)
            db_url = f'sqlite:///{db_path}'

        self.engine = create_engine(db_url, echo=False)
        
        # Створюємо таблиці, якщо їх немає
        try:
            Base.metadata.create_all(self.engine)
        except Exception as e:
            print(f"DB Init Error (ignore if tables exist): {e}")
            
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self): return self.Session()

db_manager = DatabaseManager()