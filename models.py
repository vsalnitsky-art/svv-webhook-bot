from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# === МОДЕЛІ ТАБЛИЦЬ ===

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

# Заглушки для сумісності
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

# === МЕНЕДЖЕР БАЗИ ДАНИХ ===

class DatabaseManager:
    def __init__(self, db_filename='trading_bot_final.db', data_folder='BASE'):
        """
        Ініціалізація бази даних.
        Якщо використовується SQLite, файл буде створено у папці data_folder (за замовчуванням 'BASE').
        """
        
        # 1. Перевірка на наявність зовнішньої бази (PostgreSQL на Render)
        db_url = os.environ.get('DATABASE_URL')
        if db_url and db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        
        # 2. Якщо зовнішньої бази немає — використовуємо локальну SQLite у папці
        if not db_url:
            # Отримуємо абсолютний шлях до папки, де лежить цей скрипт (models.py)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Формуємо шлях до папки з даними (наприклад: /app/BASE)
            target_folder = os.path.join(base_dir, data_folder)
            
            # Створюємо папку, якщо її немає
            if not os.path.exists(target_folder):
                try:
                    os.makedirs(target_folder)
                    print(f"📁 Created database folder: {target_folder}")
                except Exception as e:
                    print(f"⚠️ Error creating folder: {e}")
            
            # Формуємо повний шлях до файлу
            db_path = os.path.join(target_folder, db_filename)
            db_url = f'sqlite:///{db_path}'
            print(f"💾 Using SQLite database at: {db_path}")

        # 3. Підключення
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self): return self.Session()

# Ініціалізація (створить папку BASE автоматично при імпорті)
db_manager = DatabaseManager()