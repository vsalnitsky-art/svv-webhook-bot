#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    # ✨ НОВІ ПОЛЯ для комісій
    opening_fee = Column(Float, default=0.0); closing_fee = Column(Float, default=0.0); funding_fee = Column(Float, default=0.0)

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
    volume_24h = Column(Float, default=0.0)

class OrderBlock(Base):
    __tablename__ = 'order_blocks'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), index=True)
    timeframe = Column(String(10)); ob_type = Column(String(10))
    top = Column(Float); bottom = Column(Float); entry_price = Column(Float); sl_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow); status = Column(String(20), default='PENDING'); volume_score = Column(Float, default=0.0)

class SmartMoneyTicker(Base):
    """Watchlist для Smart Money сканера"""
    __tablename__ = 'smart_money_watchlist'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, index=True)
    direction = Column(String(10))  # BUY або SELL
    source = Column(String(20), default='Manual')  # Manual або Screener
    added_at = Column(DateTime, default=datetime.utcnow)


class DetectedOrderBlock(Base):
    """Знайдені Order Blocks (для тестування перед реальними угодами)"""
    __tablename__ = 'detected_order_blocks'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    direction = Column(String(10))  # BUY або SELL
    ob_type = Column(String(10))    # Bull або Bear
    ob_top = Column(Float)
    ob_bottom = Column(Float)
    entry_price = Column(Float)
    sl_price = Column(Float)
    current_price = Column(Float)
    atr = Column(Float)
    status = Column(String(20), default='Valid')  # Valid, Waiting Retest, Triggered, Executed
    timeframe = Column(String(10))
    # 🆕 Додаткові OB поля
    ob_start_time = Column(DateTime, nullable=True)  # Коли OB утворився
    ob_midline = Column(Float, nullable=True)  # Середня лінія OB
    ob_size_percent = Column(Float, nullable=True)  # Розмір OB в %
    # Timestamps
    detected_at = Column(DateTime, default=datetime.utcnow)
    executed_at = Column(DateTime, nullable=True)
    trade_result = Column(String(50), nullable=True)  # Результат угоди якщо виконано

class PaperTrade(Base):
    __tablename__ = 'paper_trades'
    id = Column(Integer, primary_key=True); symbol = Column(String(20), index=True)
    direction = Column(String(10)); entry_mode = Column(String(20)); status = Column(String(20))
    entry_price = Column(Float); sl_price = Column(Float); tp_price = Column(Float, nullable=True); exit_price = Column(Float, nullable=True)
    pnl = Column(Float, default=0.0); pnl_percent = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow); closed_at = Column(DateTime, nullable=True); details = Column(String(255))

# === НОВА МОДЕЛЬ: WHALE STRATEGY ===
class WhaleSignal(Base):
    __tablename__ = 'whale_signals'
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), index=True)
    price = Column(Float)
    score = Column(Integer)
    squeeze_val = Column(Float)
    obv_slope = Column(Float)
    rsi = Column(Float, nullable=True, default=0)  # RSI значення (nullable для міграції)
    details = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='NEW')


# === SMART MONEY EXECUTION LOG ===
class SmartMoneyExecutionLog(Base):
    """Лог виконаних угод Smart Money системи"""
    __tablename__ = 'smart_money_execution_log'
    id = Column(Integer, primary_key=True)
    
    # Основна інформація
    symbol = Column(String(20), index=True)
    direction = Column(String(10))  # LONG або SHORT
    
    # Entry інформація
    entry_price = Column(Float)
    entry_time = Column(DateTime)
    sl_price = Column(Float)
    ob_top = Column(Float)
    ob_bottom = Column(Float)
    ob_timeframe = Column(String(10))
    entry_mode = Column(String(20))  # Immediate або Retest
    
    # 🆕 Розширена інформація про Order Block
    ob_type = Column(String(10), nullable=True)  # Bullish або Bearish
    ob_start_time = Column(DateTime, nullable=True)  # Коли OB утворився
    ob_midline = Column(Float, nullable=True)  # Середня лінія OB
    ob_size_percent = Column(Float, nullable=True)  # Розмір OB в %
    
    # Exit інформація
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(50), nullable=True)  # Opposite OB, SL Hit, Manual, Timeout
    exit_ob_top = Column(Float, nullable=True)  # Exit OB зона (якщо закрито по OB)
    exit_ob_bottom = Column(Float, nullable=True)
    
    # Результат
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float, default=0.0)
    is_win = Column(Boolean, nullable=True)
    
    # Статус
    status = Column(String(20), default='OPEN')  # OPEN, CLOSED, CANCELLED, SKIPPED, FAILED
    paper_trade = Column(Boolean, default=True)  # True = paper, False = real
    
    # Bybit order info (для реальних угод)
    order_id = Column(String(50), nullable=True)
    qty = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DatabaseManager:
    def __init__(self, db_filename='trading_bot_final.db'):
        db_url = os.environ.get('DATABASE_URL')
        
        if db_url:
            print(f"\n🔌 DATABASE: FOUND DATABASE_URL.")
            if db_url.startswith("postgres://"): db_url = db_url.replace("postgres://", "postgresql://", 1)
        else:
            print(f"\n⚠️ WARNING: DATABASE_URL NOT FOUND. USING LOCAL SQLITE.")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_folder = os.path.join(current_dir, 'BASE')
            try: os.makedirs(base_folder, exist_ok=True); db_path = os.path.join(base_folder, db_filename)
            except OSError: db_path = os.path.join(current_dir, db_filename)
            db_url = f'sqlite:///{db_path}'
            
        try:
            self.engine = create_engine(db_url, echo=False)
            with self.engine.connect() as conn:
                logger.info(f"✅ Database connected: {db_url[:50]}...")
            
            # Створюємо таблиці
            Base.metadata.create_all(self.engine)
            logger.info("✅ Database tables created/verified")
            
            # ✨ МІГРАЦІЯ: Додавання нових колонок для комісій
            self._migrate_add_fee_columns()
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Database error: {e}")
            raise  # Передаємо помилку далі
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self): return self.Session()
    
    # ✨ МІГРАЦІЯ: Додавання колонок для комісій
    def _migrate_add_fee_columns(self):
        """Додає колонки opening_fee, closing_fee, funding_fee до trades якщо вони ще не існують"""
        try:
            from sqlalchemy import inspect, text
            
            inspector = inspect(self.engine)
            
            # === TRADES TABLE ===
            columns = {col['name'] for col in inspector.get_columns('trades')}
            
            # Колонки які потрібно додати
            needed_columns = {
                'opening_fee': 'REAL DEFAULT 0.0',
                'closing_fee': 'REAL DEFAULT 0.0',
                'funding_fee': 'REAL DEFAULT 0.0'
            }
            
            with self.engine.connect() as conn:
                for col_name, col_def in needed_columns.items():
                    if col_name not in columns:
                        try:
                            conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col_name} {col_def}"))
                            conn.commit()
                            logger.info(f"✅ Migration: Added column '{col_name}' to trades")
                        except Exception as e:
                            logger.warning(f"⚠️ Migration: Column '{col_name}' already exists or error: {e}")
            
            # === SMART_MONEY_WATCHLIST TABLE ===
            try:
                sm_columns = {col['name'] for col in inspector.get_columns('smart_money_watchlist')}
                sm_needed = {
                    'direction': "VARCHAR(10) DEFAULT 'BUY'",
                    'source': "VARCHAR(20) DEFAULT 'Manual'"
                }
                
                with self.engine.connect() as conn:
                    for col_name, col_def in sm_needed.items():
                        if col_name not in sm_columns:
                            try:
                                conn.execute(text(f"ALTER TABLE smart_money_watchlist ADD COLUMN {col_name} {col_def}"))
                                conn.commit()
                                logger.info(f"✅ Migration: Added column '{col_name}' to smart_money_watchlist")
                            except Exception as e:
                                pass  # Column might exist
            except Exception as e:
                pass  # Table might not exist yet
            
            # === SMART_MONEY_EXECUTION_LOG TABLE (OB columns) ===
            try:
                exec_columns = {col['name'] for col in inspector.get_columns('smart_money_execution_log')}
                exec_needed = {
                    'ob_type': "VARCHAR(10)",
                    'ob_start_time': "DATETIME",
                    'ob_midline': "REAL",
                    'ob_size_percent': "REAL"
                }
                
                with self.engine.connect() as conn:
                    for col_name, col_def in exec_needed.items():
                        if col_name not in exec_columns:
                            try:
                                conn.execute(text(f"ALTER TABLE smart_money_execution_log ADD COLUMN {col_name} {col_def}"))
                                conn.commit()
                                logger.info(f"✅ Migration: Added column '{col_name}' to smart_money_execution_log")
                            except Exception as e:
                                pass
            except Exception as e:
                pass  # Table might not exist yet
            
            # === DETECTED_ORDER_BLOCKS TABLE (OB columns) ===
            try:
                ob_columns = {col['name'] for col in inspector.get_columns('detected_order_blocks')}
                ob_needed = {
                    'ob_start_time': "DATETIME",
                    'ob_midline': "REAL",
                    'ob_size_percent': "REAL"
                }
                
                with self.engine.connect() as conn:
                    for col_name, col_def in ob_needed.items():
                        if col_name not in ob_columns:
                            try:
                                conn.execute(text(f"ALTER TABLE detected_order_blocks ADD COLUMN {col_name} {col_def}"))
                                conn.commit()
                                logger.info(f"✅ Migration: Added column '{col_name}' to detected_order_blocks")
                            except Exception as e:
                                pass
            except Exception as e:
                pass  # Table might not exist yet
                
        except Exception as e:
            logger.warning(f"⚠️ Migration failed: {e}")
    
    # === ФУНКЦІЯ ПРИМУСОВОГО ПЕРЕСТВОРЕННЯ ТАБЛИЦІ ===
    def recreate_analysis_table(self):
        try:
            AnalysisResult.__table__.drop(self.engine, checkfirst=True)
            AnalysisResult.__table__.create(self.engine, checkfirst=True)
        except Exception as e:
            print(f"❌ Error recreating table: {e}")

db_manager = DatabaseManager()
