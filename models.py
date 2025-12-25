#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     SQUEEZE DETECTOR v1.0 - MODELS                           ║
║                                                                              ║
║  SQLAlchemy моделі для зберігання:                                           ║
║  • market_snapshots - історичні дані ринку                                   ║
║  • squeeze_signals - знайдені сигнали                                        ║
║  • squeeze_watchlist - активні спостереження                                 ║
║                                                                              ║
║  Автор: SVV Webhook Bot                                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import (
    Column, Integer, Float, String, Boolean, DateTime, Text,
    Index, UniqueConstraint, create_engine, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

logger = logging.getLogger(__name__)

# Base для моделей
Base = declarative_base()


class MarketSnapshot(Base):
    """
    Знімок ринкових даних для монети.
    Записується кожні 1-5 хвилин для топ-100 монет.
    
    Це основа для розрахунку K = ΔOI / ΔPrice
    """
    __tablename__ = 'market_snapshots'
    __table_args__ = (
        # Композитний індекс для швидкого пошуку по symbol + timestamp
        Index('idx_snapshots_symbol_time', 'symbol', 'timestamp'),
        # Індекс для очистки старих даних
        Index('idx_snapshots_timestamp', 'timestamp'),
        # Унікальність: одна монета - один timestamp
        UniqueConstraint('symbol', 'timestamp', name='uq_snapshot_symbol_time'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Ідентифікація
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Open Interest - КЛЮЧОВА метрика
    open_interest = Column(Float, nullable=True)       # В USD
    open_interest_qty = Column(Float, nullable=True)   # В монетах
    
    # Ціни
    mark_price = Column(Float, nullable=True)
    index_price = Column(Float, nullable=True)
    last_price = Column(Float, nullable=True)
    
    # Funding - для визначення напрямку squeeze
    funding_rate = Column(Float, nullable=True)        # Поточний funding
    next_funding_time = Column(DateTime, nullable=True)
    
    # Об'єми
    volume_24h = Column(Float, nullable=True)          # Volume в монетах
    turnover_24h = Column(Float, nullable=True)        # Turnover в USD
    
    # Order book (для Order Book Imbalance)
    bid1_price = Column(Float, nullable=True)
    bid1_size = Column(Float, nullable=True)
    ask1_price = Column(Float, nullable=True)
    ask1_size = Column(Float, nullable=True)
    
    # Spread
    spread_percent = Column(Float, nullable=True)
    
    # 24h статистика
    high_24h = Column(Float, nullable=True)
    low_24h = Column(Float, nullable=True)
    price_change_24h = Column(Float, nullable=True)    # В %
    
    def __repr__(self):
        return f"<MarketSnapshot {self.symbol} @ {self.timestamp}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертує в словник для JSON"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'open_interest': self.open_interest,
            'open_interest_qty': self.open_interest_qty,
            'mark_price': self.mark_price,
            'last_price': self.last_price,
            'funding_rate': self.funding_rate,
            'volume_24h': self.volume_24h,
            'turnover_24h': self.turnover_24h,
            'bid1_price': self.bid1_price,
            'ask1_price': self.ask1_price,
            'spread_percent': self.spread_percent,
            'price_change_24h': self.price_change_24h,
        }


class SqueezeSignal(Base):
    """
    Знайдений сигнал накопичення/squeeze.
    
    Генерується коли:
    - K = ΔOI / ΔPrice > threshold
    - Ціна у флеті (< 2% за 4h)
    - OI росте (> 5% за 4h)
    """
    __tablename__ = 'squeeze_signals'
    __table_args__ = (
        Index('idx_signals_symbol_created', 'symbol', 'created_at'),
        Index('idx_signals_type', 'signal_type'),
        Index('idx_signals_direction', 'direction'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Ідентифікація
    symbol = Column(String(20), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Тип сигналу
    signal_type = Column(String(30), nullable=False)
    # Можливі значення:
    # - ACCUMULATION_START: Початок накопичення
    # - ACCUMULATION_CONTINUE: Продовження накопичення
    # - SQUEEZE_READY: Пружина стиснута, готовність до вибуху
    # - BREAKOUT_UP: Пробій вгору
    # - BREAKOUT_DOWN: Пробій вниз
    
    # Напрямок (на основі funding та інших факторів)
    direction = Column(String(10), nullable=False, default='UNKNOWN')
    # LONG - очікується памп (Short Squeeze)
    # SHORT - очікується дамп (Long Squeeze)
    # UNKNOWN - напрямок невизначений
    
    # Ключові метрики
    k_coefficient = Column(Float, nullable=True)       # K = ΔOI / ΔPrice
    
    # Зміни за період аналізу
    price_change_pct = Column(Float, nullable=True)    # % зміни ціни
    oi_change_pct = Column(Float, nullable=True)       # % зміни OI
    volume_change_pct = Column(Float, nullable=True)   # % зміни об'єму
    
    # Funding інформація
    funding_rate = Column(Float, nullable=True)
    funding_bias = Column(String(10), nullable=True)   # LONG, SHORT, NEUTRAL
    
    # Ціни на момент сигналу
    price_at_signal = Column(Float, nullable=True)
    oi_at_signal = Column(Float, nullable=True)
    
    # Lookback період (в годинах)
    lookback_hours = Column(Integer, nullable=True, default=4)
    
    # Confidence score (0-100)
    confidence = Column(Integer, nullable=True, default=50)
    
    # Статус
    notified = Column(Boolean, nullable=False, default=False)   # Відправлено в Telegram
    executed = Column(Boolean, nullable=False, default=False)   # Відкрито позицію
    
    # Результат (заповнюється пізніше)
    result_price = Column(Float, nullable=True)        # Ціна після X часу
    result_pnl_pct = Column(Float, nullable=True)      # P&L в %
    result_status = Column(String(20), nullable=True)  # WIN, LOSS, PENDING
    
    # Додаткові дані
    notes = Column(Text, nullable=True)
    raw_data = Column(Text, nullable=True)             # JSON з додатковими даними
    
    def __repr__(self):
        return f"<SqueezeSignal {self.symbol} {self.signal_type} K={self.k_coefficient}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертує в словник для JSON"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'signal_type': self.signal_type,
            'direction': self.direction,
            'k_coefficient': round(self.k_coefficient, 2) if self.k_coefficient else None,
            'price_change_pct': round(self.price_change_pct, 2) if self.price_change_pct else None,
            'oi_change_pct': round(self.oi_change_pct, 2) if self.oi_change_pct else None,
            'funding_rate': self.funding_rate,
            'funding_bias': self.funding_bias,
            'price_at_signal': self.price_at_signal,
            'lookback_hours': self.lookback_hours,
            'confidence': self.confidence,
            'notified': self.notified,
            'executed': self.executed,
            'result_status': self.result_status,
        }


class SqueezeWatchlist(Base):
    """
    Активний watchlist монет під спостереженням.
    
    Монета додається коли виявлено початок накопичення.
    Оновлюється кожен скан.
    Видаляється після breakout або timeout.
    """
    __tablename__ = 'squeeze_watchlist'
    __table_args__ = (
        Index('idx_watchlist_symbol', 'symbol'),
        Index('idx_watchlist_phase', 'phase'),
        UniqueConstraint('symbol', name='uq_watchlist_symbol'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Ідентифікація
    symbol = Column(String(20), nullable=False, unique=True, index=True)
    
    # Часові мітки
    added_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_update = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Фаза спостереження
    phase = Column(String(20), nullable=False, default='WATCHING')
    # Можливі значення:
    # - WATCHING: Початкове спостереження
    # - ACCUMULATING: Підтверджене накопичення (K > threshold кілька разів)
    # - SQUEEZE_READY: Пружина стиснута, очікуємо breakout
    # - TRIGGERED: Breakout стався
    # - EXPIRED: Timeout без breakout
    
    # Лічильники
    consecutive_signals = Column(Integer, nullable=False, default=0)  # Скільки сканів підряд K > threshold
    total_signals = Column(Integer, nullable=False, default=0)        # Загальна кількість сигналів
    
    # Метрики при додаванні
    entry_price = Column(Float, nullable=True)         # Ціна при додаванні
    entry_oi = Column(Float, nullable=True)            # OI при додаванні
    
    # Поточні метрики
    current_price = Column(Float, nullable=True)
    current_oi = Column(Float, nullable=True)
    current_k = Column(Float, nullable=True)           # Поточний K coefficient
    
    # Накопичені зміни
    total_oi_change_pct = Column(Float, nullable=True) # Загальна зміна OI з моменту додавання
    total_price_change_pct = Column(Float, nullable=True)
    
    # Напрямок
    direction = Column(String(10), nullable=True)      # LONG, SHORT, UNKNOWN
    direction_confidence = Column(Integer, nullable=True)  # 0-100
    
    # Funding історія
    avg_funding_rate = Column(Float, nullable=True)
    funding_bias = Column(String(10), nullable=True)
    
    # Торгові параметри (для auto-trade)
    target_price = Column(Float, nullable=True)        # TP
    stop_loss = Column(Float, nullable=True)           # SL
    position_size_usdt = Column(Float, nullable=True)
    leverage = Column(Integer, nullable=True)
    
    # Auto-trade
    auto_trade_enabled = Column(Boolean, nullable=False, default=False)
    position_opened = Column(Boolean, nullable=False, default=False)
    position_side = Column(String(10), nullable=True)  # LONG, SHORT
    
    # Результат
    result_status = Column(String(20), nullable=True)  # WIN, LOSS, PENDING, EXPIRED
    result_pnl_pct = Column(Float, nullable=True)
    
    # Примітки
    notes = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<SqueezeWatchlist {self.symbol} phase={self.phase} K={self.current_k}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертує в словник для JSON"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'phase': self.phase,
            'consecutive_signals': self.consecutive_signals,
            'total_signals': self.total_signals,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'current_k': round(self.current_k, 2) if self.current_k else None,
            'total_oi_change_pct': round(self.total_oi_change_pct, 2) if self.total_oi_change_pct else None,
            'total_price_change_pct': round(self.total_price_change_pct, 2) if self.total_price_change_pct else None,
            'direction': self.direction,
            'direction_confidence': self.direction_confidence,
            'funding_bias': self.funding_bias,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'auto_trade_enabled': self.auto_trade_enabled,
            'position_opened': self.position_opened,
            'result_status': self.result_status,
        }
    
    def update_phase(self, new_phase: str):
        """Оновлює фазу з валідацією"""
        valid_phases = ['WATCHING', 'ACCUMULATING', 'SQUEEZE_READY', 'TRIGGERED', 'EXPIRED']
        if new_phase not in valid_phases:
            raise ValueError(f"Invalid phase: {new_phase}. Must be one of {valid_phases}")
        self.phase = new_phase
        self.last_update = datetime.utcnow()


class SqueezeConfig(Base):
    """
    Конфігурація Squeeze Detector.
    Зберігається в БД для персистентності.
    """
    __tablename__ = 'squeeze_config'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(50), nullable=False, unique=True, index=True)
    value = Column(String(500), nullable=True)
    description = Column(String(200), nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SqueezeConfig {self.key}={self.value}>"


# ============================================================================
#                              HELPER FUNCTIONS
# ============================================================================

def create_squeeze_tables(engine):
    """
    Створює всі таблиці Squeeze Detector.
    Викликається при ініціалізації.
    """
    try:
        Base.metadata.create_all(engine, checkfirst=True)
        logger.info("✅ Squeeze Detector tables created/verified")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create Squeeze Detector tables: {e}")
        return False


def run_migrations(engine):
    """
    Виконує міграції для додавання нових колонок.
    Безпечно ігнорує якщо колонка вже існує.
    """
    from sqlalchemy import inspect, text
    
    try:
        inspector = inspect(engine)
        
        migrations = [
            # (table_name, column_name, column_type)
            ('market_snapshots', 'spread_percent', 'FLOAT'),
            ('market_snapshots', 'bid1_size', 'FLOAT'),
            ('market_snapshots', 'ask1_size', 'FLOAT'),
            ('squeeze_signals', 'volume_change_pct', 'FLOAT'),
            ('squeeze_signals', 'raw_data', 'TEXT'),
            ('squeeze_watchlist', 'total_signals', 'INTEGER DEFAULT 0'),
            ('squeeze_watchlist', 'direction_confidence', 'INTEGER'),
        ]
        
        with engine.connect() as conn:
            for table_name, col_name, col_type in migrations:
                # Перевіряємо чи таблиця існує
                if table_name not in inspector.get_table_names():
                    continue
                
                # Перевіряємо чи колонка існує
                existing_columns = {col['name'] for col in inspector.get_columns(table_name)}
                if col_name not in existing_columns:
                    try:
                        conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
                        logger.info(f"📦 Migration: added {table_name}.{col_name}")
                    except Exception as e:
                        # Ігноруємо якщо колонка вже існує (race condition)
                        logger.debug(f"Migration skip {table_name}.{col_name}: {e}")
            
            conn.commit()
        
        logger.info("✅ Squeeze Detector migrations complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration error: {e}")
        return False


def cleanup_old_snapshots(session: Session, days: int = 7) -> int:
    """
    Видаляє старі snapshots старші за N днів.
    Повертає кількість видалених записів.
    """
    from datetime import timedelta
    
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        deleted = session.query(MarketSnapshot).filter(
            MarketSnapshot.timestamp < cutoff
        ).delete(synchronize_session=False)
        
        session.commit()
        
        if deleted > 0:
            logger.info(f"🗑️ Cleaned up {deleted} old snapshots (older than {days} days)")
        
        return deleted
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        session.rollback()
        return 0


def get_snapshot_count(session: Session, symbol: str = None) -> int:
    """Повертає кількість snapshots"""
    try:
        query = session.query(MarketSnapshot)
        if symbol:
            query = query.filter(MarketSnapshot.symbol == symbol)
        return query.count()
    except Exception as e:
        logger.error(f"Count error: {e}")
        return 0


def get_latest_snapshot(session: Session, symbol: str) -> Optional[MarketSnapshot]:
    """Повертає останній snapshot для монети"""
    try:
        return session.query(MarketSnapshot).filter(
            MarketSnapshot.symbol == symbol
        ).order_by(MarketSnapshot.timestamp.desc()).first()
    except Exception as e:
        logger.error(f"Get latest error: {e}")
        return None


def get_snapshots_for_period(
    session: Session,
    symbol: str,
    hours: int = 4
) -> List[MarketSnapshot]:
    """Повертає snapshots за останні N годин"""
    from datetime import timedelta
    
    try:
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        return session.query(MarketSnapshot).filter(
            MarketSnapshot.symbol == symbol,
            MarketSnapshot.timestamp >= cutoff
        ).order_by(MarketSnapshot.timestamp.asc()).all()
        
    except Exception as e:
        logger.error(f"Get snapshots error: {e}")
        return []
