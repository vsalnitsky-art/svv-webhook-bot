"""
Database Models - SQLAlchemy models for Sleeper OB Bot
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL

Base = declarative_base()

class SleeperCandidate(Base):
    """Sleeper detector candidates"""
    __tablename__ = 'sleeper_candidates'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    
    # Score breakdown
    total_score = Column(Float, default=0)
    fuel_score = Column(Float, default=0)
    volatility_score = Column(Float, default=0)
    price_score = Column(Float, default=0)
    liquidity_score = Column(Float, default=0)
    
    # State
    state = Column(String(20), default='IDLE', index=True)  # IDLE/WATCHING/BUILDING/READY/TRIGGERED
    hp = Column(Integer, default=5)
    direction = Column(String(10), default='NEUTRAL')  # LONG/SHORT/NEUTRAL
    
    # Metrics
    funding_rate = Column(Float)
    oi_change_4h = Column(Float)
    bb_width = Column(Float)
    bb_width_change = Column(Float)
    volume_24h = Column(Float)
    volume_ratio = Column(Float)
    price_range_pct = Column(Float)
    rsi = Column(Float)
    
    # Tracking
    added_at = Column(DateTime, default=datetime.utcnow)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    checks_count = Column(Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'total_score': self.total_score,
            'fuel_score': self.fuel_score,
            'volatility_score': self.volatility_score,
            'price_score': self.price_score,
            'liquidity_score': self.liquidity_score,
            'state': self.state,
            'hp': self.hp,
            'direction': self.direction,
            'funding_rate': self.funding_rate,
            'oi_change_4h': self.oi_change_4h,
            'bb_width': self.bb_width,
            'volume_24h': self.volume_24h,
            'rsi': self.rsi,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'checks_count': self.checks_count,
        }


class OrderBlock(Base):
    """Detected order blocks"""
    __tablename__ = 'order_blocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False)  # 15m/5m/1m
    
    # OB Parameters
    ob_type = Column(String(10), nullable=False)  # BULLISH/BEARISH
    ob_high = Column(Float, nullable=False)
    ob_low = Column(Float, nullable=False)
    ob_mid = Column(Float, nullable=False)
    
    # Quality
    quality_score = Column(Float, default=0)
    volume_ratio = Column(Float, default=1.0)
    impulse_pct = Column(Float, default=0)
    
    # Status
    status = Column(String(20), default='ACTIVE', index=True)  # ACTIVE/TOUCHED/MITIGATED/EXPIRED
    touch_count = Column(Integer, default=0)
    
    # Related sleeper
    sleeper_symbol = Column(String(20))
    sleeper_score = Column(Float)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    touched_at = Column(DateTime)
    expires_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_ob_symbol_status', 'symbol', 'status'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'ob_type': self.ob_type,
            'ob_high': self.ob_high,
            'ob_low': self.ob_low,
            'ob_mid': self.ob_mid,
            'quality_score': self.quality_score,
            'volume_ratio': self.volume_ratio,
            'impulse_pct': self.impulse_pct,
            'status': self.status,
            'touch_count': self.touch_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }


class Trade(Base):
    """Trade records"""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Entry
    direction = Column(String(10), nullable=False)  # LONG/SHORT
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, default=datetime.utcnow)
    
    # Position
    position_size = Column(Float, nullable=False)
    position_value = Column(Float)
    leverage = Column(Integer, default=1)
    
    # Levels
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float)
    take_profit_2 = Column(Float)
    take_profit_3 = Column(Float)
    
    # Exit
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_reason = Column(String(50))  # TP1/TP2/TP3/SL/MANUAL/TRAILING
    
    # P&L
    pnl_usdt = Column(Float, default=0)
    pnl_percent = Column(Float, default=0)
    fees_paid = Column(Float, default=0)
    
    # Meta
    sleeper_score = Column(Float)
    ob_quality = Column(Float)
    signal_confidence = Column(Float)
    
    # Mode
    is_paper = Column(Boolean, default=True)
    execution_mode = Column(String(20))  # AUTO/SEMI_AUTO/MANUAL
    
    # Status
    status = Column(String(20), default='OPEN', index=True)  # OPEN/CLOSED/CANCELLED
    
    # Notes
    notes = Column(Text)
    
    __table_args__ = (
        Index('idx_trade_symbol_status', 'symbol', 'status'),
        Index('idx_trade_entry_time', 'entry_time'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'position_size': self.position_size,
            'position_value': self.position_value,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'pnl_usdt': self.pnl_usdt,
            'pnl_percent': self.pnl_percent,
            'fees_paid': self.fees_paid,
            'sleeper_score': self.sleeper_score,
            'ob_quality': self.ob_quality,
            'is_paper': self.is_paper,
            'execution_mode': self.execution_mode,
            'status': self.status,
        }


class PerformanceStats(Base):
    """Daily performance statistics"""
    __tablename__ = 'performance_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, unique=True, nullable=False, index=True)
    
    # Counts
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # P&L
    total_pnl_usdt = Column(Float, default=0)
    total_pnl_percent = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    
    # Rates
    win_rate = Column(Float, default=0)
    avg_win = Column(Float, default=0)
    avg_loss = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    
    # Sleeper stats
    sleeper_signals = Column(Integer, default=0)
    ob_signals = Column(Integer, default=0)
    
    def to_dict(self):
        return {
            'date': self.date.isoformat() if self.date else None,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl_usdt': self.total_pnl_usdt,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
        }


class BotSetting(Base):
    """Bot settings storage"""
    __tablename__ = 'bot_settings'
    
    key = Column(String(50), primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EventLog(Base):
    """Event log for dashboard"""
    __tablename__ = 'event_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(10), default='INFO')  # INFO/WARN/ERROR/SUCCESS
    category = Column(String(20))  # SLEEPER/OB/TRADE/SYSTEM
    message = Column(Text)
    symbol = Column(String(20))
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'level': self.level,
            'category': self.category,
            'message': self.message,
            'symbol': self.symbol,
        }


# Engine and session factory
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def get_session():
    """Get database session"""
    return SessionLocal()
