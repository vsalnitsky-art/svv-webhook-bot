"""
Database Models - SQLAlchemy models for Sleeper OB Bot
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL
from config.bot_settings import OBStatus  # Re-export for convenience

Base = declarative_base()

# Table prefix to avoid conflicts with other bots
TABLE_PREFIX = 'sob_'  # sleeper_ob_bot

class SleeperCandidate(Base):
    """Sleeper detector candidates - 5-Day Strategy v3.0"""
    __tablename__ = f'{TABLE_PREFIX}sleeper_candidates'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    
    # === NEW: 5-Day Strategy Score Breakdown ===
    total_score = Column(Float, default=0)
    
    # Primary metrics (weights: 40/25/20/15)
    volatility_compression = Column(Float, default=0)  # 40% - BB squeeze over 5 days
    volume_suppression = Column(Float, default=0)      # 25% - volume decline ratio
    oi_growth = Column(Float, default=0)               # 20% - OI accumulation
    order_book_imbalance = Column(Float, default=0)    # 15% - bid/ask imbalance
    
    # Legacy scores (keep for backward compatibility)
    fuel_score = Column(Float, default=0)
    volatility_score = Column(Float, default=0)
    price_score = Column(Float, default=0)
    liquidity_score = Column(Float, default=0)
    
    # State
    state = Column(String(20), default='IDLE', index=True)  # IDLE/WATCHING/BUILDING/READY/TRIGGERED
    hp = Column(Integer, default=5)
    direction = Column(String(10), default='NEUTRAL')  # LONG/SHORT/NEUTRAL
    
    # === NEW: 5-Day Metrics ===
    # Volatility compression data
    bb_width_5d_start = Column(Float)    # BB width 5 days ago
    bb_width_current = Column(Float)     # Current BB width
    bb_compression_pct = Column(Float)   # Compression percentage
    
    # Volume suppression data
    volume_5d_avg = Column(Float)        # 5-day average volume
    volume_current = Column(Float)       # Current volume
    volume_ratio = Column(Float)         # current/average ratio
    
    # OI growth data
    oi_5d_start = Column(Float)          # OI 5 days ago
    oi_current = Column(Float)           # Current OI
    oi_growth_pct = Column(Float)        # Growth percentage
    
    # Order book data
    bid_volume = Column(Float)           # Total bid volume
    ask_volume = Column(Float)           # Total ask volume
    ob_imbalance_pct = Column(Float)     # Imbalance percentage
    
    # === Trigger flags ===
    volume_spike_detected = Column(Boolean, default=False)  # Volume > 200% of avg
    oi_jump_detected = Column(Boolean, default=False)       # OI jump > 15%
    breakout_detected = Column(Boolean, default=False)      # Price breakout
    
    # Legacy metrics (keep for compatibility)
    funding_rate = Column(Float)
    oi_change_4h = Column(Float)
    bb_width = Column(Float)
    bb_width_change = Column(Float)
    volume_24h = Column(Float)
    price_range_pct = Column(Float)
    rsi = Column(Float)
    
    # Tracking
    added_at = Column(DateTime, default=datetime.utcnow)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    checks_count = Column(Integer, default=0)
    
    # State transition tracking
    watching_since = Column(DateTime)
    building_since = Column(DateTime)
    ready_since = Column(DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            # New 5-day scores
            'total_score': self.total_score,
            'volatility_compression': self.volatility_compression,
            'volume_suppression': self.volume_suppression,
            'oi_growth': self.oi_growth,
            'order_book_imbalance': self.order_book_imbalance,
            # Legacy scores
            'fuel_score': self.fuel_score,
            'volatility_score': self.volatility_score,
            'price_score': self.price_score,
            'liquidity_score': self.liquidity_score,
            # State
            'state': self.state,
            'hp': self.hp,
            'direction': self.direction,
            # 5-day metrics
            'bb_compression_pct': self.bb_compression_pct,
            'volume_ratio': self.volume_ratio,
            'oi_growth_pct': self.oi_growth_pct,
            'ob_imbalance_pct': self.ob_imbalance_pct,
            # Trigger flags
            'volume_spike_detected': self.volume_spike_detected,
            'oi_jump_detected': self.oi_jump_detected,
            'breakout_detected': self.breakout_detected,
            # Legacy metrics
            'funding_rate': self.funding_rate,
            'oi_change_4h': self.oi_change_4h,
            'bb_width': self.bb_width or self.bb_width_current,
            'volume_24h': self.volume_24h,
            'rsi': self.rsi,
            # Timestamps
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'checks_count': self.checks_count,
        }


class OrderBlock(Base):
    """Detected order blocks"""
    __tablename__ = f'{TABLE_PREFIX}order_blocks'
    
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
            # Aliases for OB scanner compatibility
            'top': self.ob_high,
            'bottom': self.ob_low,
            'quality': self.quality_score,
            # Original fields
            'quality_score': self.quality_score,
            'volume_ratio': self.volume_ratio,
            'impulse_pct': self.impulse_pct,
            'status': self.status,
            'touch_count': self.touch_count,
            # Keep as datetime for template compatibility
            'created_at': self.created_at,
            'expires_at': self.expires_at,
        }


class Trade(Base):
    """Trade records"""
    __tablename__ = f'{TABLE_PREFIX}trades'
    
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
    __tablename__ = f'{TABLE_PREFIX}performance_stats'
    
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
    __tablename__ = f'{TABLE_PREFIX}bot_settings'
    
    key = Column(String(50), primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EventLog(Base):
    """Event log for dashboard"""
    __tablename__ = f'{TABLE_PREFIX}event_logs'
    
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


# Engine and session factory with connection pool settings
engine = create_engine(
    DATABASE_URL, 
    echo=False,
    pool_pre_ping=True,      # Verify connection before use
    pool_recycle=300,        # Recycle connections every 5 min
    pool_size=5,             # Number of connections to keep
    max_overflow=10,         # Allow up to 10 extra connections
)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    """Initialize database tables (creates only if not exist)"""
    print("[DB] Creating tables if not exist...")
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables ready")

def get_session():
    """Get database session"""
    return SessionLocal()
