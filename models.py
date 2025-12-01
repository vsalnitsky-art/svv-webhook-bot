from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Index, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

# === МОДЕЛІ ДАНИХ ===

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
    def __init__(self, db_filename='trading_bot_final.db'):
        
        # 1. Спроба підключення до PostgreSQL (Render Env)
        db_url = os.environ.get('DATABASE_URL')
        if db_url and db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
            print("🔵 Using PostgreSQL connection.")
        
        # 2. SQLite (Локальна файлова система)
        else:
            # --- ВАШ ЗАПИТ: Конкретний шлях ---
            preferred_path = '/workspaces/svv-webhook-bot/BASE'
            
            # Резервний шлях (поруч зі скриптом, для Render/інших серверів)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fallback_path = os.path.join(current_dir, 'BASE')
            
            target_folder = fallback_path # Початок з резервного
            
            # Спроба використати бажаний шлях
            try:
                # Перевіряємо, чи можемо ми писати в /workspaces/...
                os.makedirs(preferred_path, exist_ok=True)
                # Перевірка запису (іноді папка є, а прав немає)
                test_file = os.path.join(preferred_path, '.test_write')
                with open(test_file, 'w') as f: f.write('test')
                os.remove(test_file)
                
                # Якщо все ок - використовуємо його
                target_folder = preferred_path
                print(f"✅ Using custom path: {target_folder}")
            except OSError:
                # Якщо не вийшло - використовуємо локальну папку
                print(f"⚠️ Custom path unavailable. Using local: {fallback_path}")
                os.makedirs(fallback_path, exist_ok=True)
                target_folder = fallback_path

            # Формуємо фінальний шлях
            db_abs_path = os.path.join(target_folder, db_filename)
            db_url = f'sqlite:///{db_abs_path}'
            print(f"💾 Database initialized at: {db_abs_path}")

        # 3. Підключення
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self): return self.Session()

db_manager = DatabaseManager()