"""
Database Models - SQLAlchemy models for Sleeper OB Bot
v8.3.0 - Added CTR Scanner models
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, Index, LargeBinary
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
    
    # === Direction Engine v4.1 ===
    direction_score = Column(Float, default=0)           # -1.0 to +1.0
    direction_confidence = Column(String(10), default='LOW')  # HIGH/MEDIUM/LOW
    direction_htf_bias = Column(Float, default=0)        # HTF layer result (deprecated in v5)
    direction_ltf_bias = Column(Float, default=0)        # LTF layer result (deprecated in v5)
    direction_deriv_bias = Column(Float, default=0)      # Derivatives layer result (deprecated in v5)
    
    # === Direction Engine v5 - Phase & Exhaustion ===
    market_phase = Column(String(20), default='UNKNOWN')       # ACCUMULATION/MARKUP/DISTRIBUTION/MARKDOWN
    phase_maturity = Column(String(20), default='MIDDLE')      # EARLY/MIDDLE/LATE/EXHAUSTED
    is_reversal_setup = Column(Boolean, default=False)         # Reversal detected
    exhaustion_score = Column(Float, default=0)                # 0-1 exhaustion level
    direction_reason = Column(String(200))                     # Primary reason for direction
    
    # === v5: Price Structure ===
    price_change_5d = Column(Float)          # 5-day price change %
    price_change_20d = Column(Float)         # 20-day price change %
    distance_from_high = Column(Float)       # % distance from recent high
    distance_from_low = Column(Float)        # % distance from recent low
    support_level = Column(Float)            # Calculated support price
    resistance_level = Column(Float)         # Calculated resistance price
    
    # === v5: Exhaustion Signals ===
    rsi_divergence = Column(String(10))      # bullish/bearish/none
    at_support = Column(Boolean, default=False)     # Price at support
    at_resistance = Column(Boolean, default=False)  # Price at resistance
    
    # === v4 ADX data ===
    adx_value = Column(Float)                # Current ADX (0-100)
    adx_trendless = Column(Boolean, default=False)  # ADX < 20
    adx_bonus = Column(Integer, default=0)   # Bonus points from ADX
    
    # === v4 POC data ===
    poc_price = Column(Float)                # Point of Control price
    poc_distance_pct = Column(Float)         # Distance from POC %
    poc_strength = Column(Float)             # POC strength %
    price_at_poc = Column(Boolean, default=False)  # Is price at POC
    poc_bonus = Column(Integer, default=0)   # Bonus points from POC
    
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
    vc_extreme_detected = Column(Boolean, default=False)    # VC > 95% + VOL < 1.2x
    
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
            # v4.1: Direction Engine
            'direction_score': self.direction_score,
            'direction_confidence': self.direction_confidence,
            'direction_htf_bias': self.direction_htf_bias,
            'direction_ltf_bias': self.direction_ltf_bias,
            'direction_deriv_bias': self.direction_deriv_bias,
            # v5: Phase & Exhaustion
            'market_phase': self.market_phase,
            'phase_maturity': self.phase_maturity,
            'is_reversal_setup': self.is_reversal_setup,
            'exhaustion_score': self.exhaustion_score,
            'direction_reason': self.direction_reason,
            # v5: Price Structure
            'price_change_5d': self.price_change_5d,
            'price_change_20d': self.price_change_20d,
            'distance_from_high': self.distance_from_high,
            'distance_from_low': self.distance_from_low,
            'support_level': self.support_level,
            'resistance_level': self.resistance_level,
            # v5: Exhaustion Signals
            'rsi_divergence': self.rsi_divergence,
            'at_support': self.at_support,
            'at_resistance': self.at_resistance,
            # v4: ADX data
            'adx_value': self.adx_value,
            'adx_trendless': self.adx_trendless,
            'adx_bonus': self.adx_bonus,
            # v4: POC data
            'poc_price': self.poc_price,
            'poc_distance_pct': self.poc_distance_pct,
            'poc_strength': self.poc_strength,
            'price_at_poc': self.price_at_poc,
            'poc_bonus': self.poc_bonus,
            # 5-day metrics
            'bb_compression_pct': self.bb_compression_pct,
            'volume_ratio': self.volume_ratio,
            'oi_growth_pct': self.oi_growth_pct,
            'ob_imbalance_pct': self.ob_imbalance_pct,
            # Trigger flags
            'volume_spike_detected': self.volume_spike_detected,
            'oi_jump_detected': self.oi_jump_detected,
            'breakout_detected': self.breakout_detected,
            'vc_extreme_detected': self.vc_extreme_detected,
            # Tracking
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }


class OrderBlock(Base):
    """Order Block entries"""
    __tablename__ = f'{TABLE_PREFIX}order_blocks'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), default='15m')
    
    # OB Data
    ob_type = Column(String(10))  # BULLISH/BEARISH
    ob_high = Column(Float)
    ob_low = Column(Float)
    ob_volume = Column(Float)
    ob_strength = Column(Float, default=0)  # 0-100
    
    # Status
    status = Column(String(20), default='ACTIVE', index=True)  # ACTIVE/MITIGATED/EXPIRED
    entry_price = Column(Float)
    sl_price = Column(Float)
    tp_price = Column(Float)
    
    # Tracking
    detected_at = Column(DateTime, default=datetime.utcnow)
    mitigated_at = Column(DateTime)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'ob_type': self.ob_type,
            'ob_high': self.ob_high,
            'ob_low': self.ob_low,
            'ob_strength': self.ob_strength,
            'status': self.status,
            'entry_price': self.entry_price,
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'detected_at': self.detected_at.isoformat() if self.detected_at else None,
        }


class Trade(Base):
    """Trade history"""
    __tablename__ = f'{TABLE_PREFIX}trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    # Trade data
    side = Column(String(10))  # LONG/SHORT
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Float)
    leverage = Column(Integer, default=1)
    
    # PnL
    realized_pnl = Column(Float)
    fee = Column(Float, default=0)
    
    # Status
    status = Column(String(20), default='OPEN', index=True)  # OPEN/CLOSED/CANCELLED
    exit_reason = Column(String(50))  # TP/SL/MANUAL/SIGNAL
    
    # Timestamps
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    
    # Metadata
    order_block_id = Column(Integer)
    sleeper_symbol = Column(String(20))
    strategy = Column(String(50))
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'leverage': self.leverage,
            'realized_pnl': self.realized_pnl,
            'fee': self.fee,
            'status': self.status,
            'exit_reason': self.exit_reason,
            'opened_at': self.opened_at.isoformat() if self.opened_at else None,
            'closed_at': self.closed_at.isoformat() if self.closed_at else None,
            'strategy': self.strategy,
        }


class PerformanceStats(Base):
    """Daily performance statistics"""
    __tablename__ = f'{TABLE_PREFIX}performance_stats'
    
    id = Column(Integer, primary_key=True)
    date = Column(String(10), unique=True, index=True)  # YYYY-MM-DD
    
    # Trade stats
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    
    # PnL
    gross_pnl = Column(Float, default=0)
    fees = Column(Float, default=0)
    net_pnl = Column(Float, default=0)
    
    # Best/Worst
    best_trade_pnl = Column(Float)
    worst_trade_pnl = Column(Float)
    
    def to_dict(self):
        return {
            'date': self.date,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'gross_pnl': self.gross_pnl,
            'fees': self.fees,
            'net_pnl': self.net_pnl,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
        }


class BotSetting(Base):
    """Bot settings storage"""
    __tablename__ = f'{TABLE_PREFIX}settings'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EventLog(Base):
    """Event log for debugging and monitoring"""
    __tablename__ = f'{TABLE_PREFIX}events'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    level = Column(String(10), index=True)  # INFO/WARNING/ERROR
    category = Column(String(50), index=True)  # TRADE/SIGNAL/SYSTEM/etc
    message = Column(Text)
    data = Column(Text)  # JSON extra data
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'level': self.level,
            'category': self.category,
            'message': self.message,
        }


class SymbolBlacklist(Base):
    """Blacklisted symbols - excluded from scanning"""
    __tablename__ = f'{TABLE_PREFIX}symbol_blacklist'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    reason = Column(String(50))  # LOW_VOLATILITY / STABLECOIN / MANUAL / DELISTED
    volatility_24h = Column(Float)  # Volatility when added
    added_at = Column(DateTime, default=datetime.utcnow)
    note = Column(String(200))  # Optional note
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'reason': self.reason,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'volatility_24h': self.volatility_24h,
            'note': self.note,
        }


# ============================================
# CTR SCANNER MODELS (v8.3.0)
# ============================================

class CTRWatchlistItem(Base):
    """CTR Scanner watchlist items stored in database"""
    __tablename__ = f'{TABLE_PREFIX}ctr_watchlist'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'is_active': self.is_active,
        }


class CTRSignal(Base):
    """CTR Scanner executed signals"""
    __tablename__ = f'{TABLE_PREFIX}ctr_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    signal_type = Column(String(10), nullable=False)  # BUY/SELL
    price = Column(Float, nullable=False)
    stc = Column(Float)
    timeframe = Column(String(10))
    smc_filtered = Column(Boolean, default=False)
    smc_trend = Column(String(20))  # BULLISH/BEARISH/NEUTRAL
    zone = Column(String(20))  # PREMIUM/DISCOUNT/EQUILIBRIUM
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    notified = Column(Boolean, default=True)  # Was notification sent?
    
    # Index for fast lookups
    __table_args__ = (
        Index(f'ix_{TABLE_PREFIX}ctr_signals_symbol_time', 'symbol', 'timestamp'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'type': self.signal_type,
            'price': self.price,
            'stc': self.stc,
            'timeframe': self.timeframe,
            'smc_filtered': self.smc_filtered,
            'smc_trend': self.smc_trend,
            'zone': self.zone,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'notified': self.notified,
        }


class CTRKlineCache(Base):
    """CTR Scanner kline cache - persisted between restarts"""
    __tablename__ = f'{TABLE_PREFIX}ctr_kline_cache'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    klines_data = Column(Text)  # JSON serialized klines
    candles_count = Column(Integer, default=0)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Composite unique constraint
    __table_args__ = (
        Index(f'ix_{TABLE_PREFIX}ctr_kline_cache_symbol_tf', 'symbol', 'timeframe', unique=True),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'candles_count': self.candles_count,
            'last_update': self.last_update.isoformat() if self.last_update else None,
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


def cleanup_ut_bot_data():
    """Remove UT Bot module data from database (one-time cleanup v8.0)"""
    from sqlalchemy import text
    
    with engine.connect() as conn:
        # Check if cleanup already done
        check_sql = text(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{TABLE_PREFIX}ut_potential_coins'
            );
        """)
        result = conn.execute(check_sql)
        if not result.scalar():
            return  # Already cleaned up
        
        print("[DB CLEANUP] Removing UT Bot module data...")
        
        try:
            # Drop UT Bot tables
            conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_PREFIX}ut_potential_coins CASCADE;"))
            conn.execute(text(f"DROP TABLE IF EXISTS {TABLE_PREFIX}ut_paper_trades CASCADE;"))
            conn.commit()
            print("[DB CLEANUP] ✓ Dropped UT Bot tables")
        except Exception as e:
            print(f"[DB CLEANUP] Error dropping tables: {e}")
            conn.rollback()
        
        try:
            # Remove UT Bot settings
            conn.execute(text(f"DELETE FROM {TABLE_PREFIX}settings WHERE key LIKE 'ut_bot_%';"))
            conn.execute(text(f"DELETE FROM {TABLE_PREFIX}settings WHERE key = 'module_ut_bot';"))
            conn.execute(text(f"DELETE FROM {TABLE_PREFIX}settings WHERE key = 'alert_ut_bot';"))
            conn.commit()
            print("[DB CLEANUP] ✓ Removed UT Bot settings")
        except Exception as e:
            print(f"[DB CLEANUP] Error removing settings: {e}")
            conn.rollback()
        
        print("[DB CLEANUP] UT Bot cleanup complete")


def init_db():
    """Initialize database tables (creates only if not exist)"""
    print("[DB] Creating tables if not exist...")
    Base.metadata.create_all(bind=engine)
    print("[DB] Tables ready")
    
    # Cleanup old UT Bot data (one-time migration)
    cleanup_ut_bot_data()
    
    # Run migrations for new columns
    migrate_sleeper_candidates_v3()
    
    # Run trades table migration
    migrate_trades_table()
    
    # Run CTR migrations
    migrate_ctr_tables()


def migrate_sleeper_candidates_v3():
    """Add new columns for 5-Day Strategy v3.0 and Direction Engine v4.1"""
    from sqlalchemy import text
    
    new_columns = [
        # 5-Day Strategy scores
        ("volatility_compression", "FLOAT DEFAULT 0"),
        ("volume_suppression", "FLOAT DEFAULT 0"),
        ("oi_growth", "FLOAT DEFAULT 0"),
        ("order_book_imbalance", "FLOAT DEFAULT 0"),
        # BB compression data
        ("bb_width_5d_start", "FLOAT"),
        ("bb_width_current", "FLOAT"),
        ("bb_compression_pct", "FLOAT"),
        # Volume suppression data
        ("volume_5d_avg", "FLOAT"),
        ("volume_current", "FLOAT"),
        # OI growth data
        ("oi_5d_start", "FLOAT"),
        ("oi_current", "FLOAT"),
        ("oi_growth_pct", "FLOAT"),
        # Order book data
        ("bid_volume", "FLOAT"),
        ("ask_volume", "FLOAT"),
        ("ob_imbalance_pct", "FLOAT"),
        # Trigger flags
        ("volume_spike_detected", "BOOLEAN DEFAULT FALSE"),
        ("oi_jump_detected", "BOOLEAN DEFAULT FALSE"),
        ("breakout_detected", "BOOLEAN DEFAULT FALSE"),
        ("vc_extreme_detected", "BOOLEAN DEFAULT FALSE"),
        # State transition tracking
        ("watching_since", "TIMESTAMP"),
        ("building_since", "TIMESTAMP"),
        ("ready_since", "TIMESTAMP"),
        # Updated at alias
        ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        
        # === v4.1 Direction Engine ===
        ("direction_score", "FLOAT DEFAULT 0"),
        ("direction_confidence", "VARCHAR(10) DEFAULT 'LOW'"),
        ("direction_htf_bias", "FLOAT DEFAULT 0"),
        ("direction_ltf_bias", "FLOAT DEFAULT 0"),
        ("direction_deriv_bias", "FLOAT DEFAULT 0"),
        
        # === v4.1 ADX data ===
        ("adx_value", "FLOAT"),
        ("adx_trendless", "BOOLEAN DEFAULT FALSE"),
        ("adx_bonus", "INTEGER DEFAULT 0"),
        
        # === v4.1 POC data ===
        ("poc_price", "FLOAT"),
        ("poc_distance_pct", "FLOAT"),
        ("poc_strength", "FLOAT"),
        ("price_at_poc", "BOOLEAN DEFAULT FALSE"),
        ("poc_bonus", "INTEGER DEFAULT 0"),
        
        # === v5.0 Direction Engine - Phase & Exhaustion ===
        ("market_phase", "VARCHAR(20) DEFAULT 'UNKNOWN'"),
        ("phase_maturity", "VARCHAR(20) DEFAULT 'MIDDLE'"),
        ("is_reversal_setup", "BOOLEAN DEFAULT FALSE"),
        ("exhaustion_score", "FLOAT DEFAULT 0"),
        ("direction_reason", "VARCHAR(200)"),
        
        # === v5.0 Price Structure ===
        ("price_change_5d", "FLOAT"),
        ("price_change_20d", "FLOAT"),
        ("distance_from_high", "FLOAT"),
        ("distance_from_low", "FLOAT"),
        ("support_level", "FLOAT"),
        ("resistance_level", "FLOAT"),
        
        # === v5.0 Exhaustion Signals ===
        ("rsi_divergence", "VARCHAR(10)"),
        ("at_support", "BOOLEAN DEFAULT FALSE"),
        ("at_resistance", "BOOLEAN DEFAULT FALSE"),
    ]
    
    table_name = f"{TABLE_PREFIX}sleeper_candidates"
    
    with engine.connect() as conn:
        for col_name, col_type in new_columns:
            try:
                sql = text(f"""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = '{table_name}' AND column_name = '{col_name}'
                        ) THEN 
                            ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type};
                            RAISE NOTICE 'Added column: {col_name}';
                        END IF;
                    END $$;
                """)
                conn.execute(sql)
                conn.commit()
            except Exception as e:
                error_str = str(e)
                if 'already exists' not in error_str.lower():
                    print(f"[DB MIGRATE] Error adding {col_name}: {e}")
                conn.rollback()
    
    print("[DB MIGRATE] Sleeper migration complete")


def migrate_ctr_tables():
    """Migrate CTR tables - add new columns v8.3.0"""
    from sqlalchemy import text
    
    # CTR Signals new columns
    signals_table = f"{TABLE_PREFIX}ctr_signals"
    new_columns = [
        ("smc_trend", "VARCHAR(20)"),
        ("zone", "VARCHAR(20)"),
        ("notified", "BOOLEAN DEFAULT TRUE"),
    ]
    
    with engine.connect() as conn:
        for col_name, col_type in new_columns:
            try:
                sql = text(f"""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = '{signals_table}' AND column_name = '{col_name}'
                        ) THEN 
                            ALTER TABLE {signals_table} ADD COLUMN {col_name} {col_type};
                        END IF;
                    END $$;
                """)
                conn.execute(sql)
                conn.commit()
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    print(f"[DB MIGRATE CTR] Error adding {col_name}: {e}")
                conn.rollback()
    
    print("[DB MIGRATE] CTR migration complete")


def migrate_trades_table():
    """Migrate trades table - add missing columns v8.3.0"""
    from sqlalchemy import text
    
    trades_table = f"{TABLE_PREFIX}trades"
    
    # Columns that should exist in trades table
    new_columns = [
        ("side", "VARCHAR(10)"),
        ("entry_price", "FLOAT"),
        ("exit_price", "FLOAT"),
        ("quantity", "FLOAT"),
        ("leverage", "INTEGER DEFAULT 1"),
        ("realized_pnl", "FLOAT"),
        ("fee", "FLOAT DEFAULT 0"),
        ("status", "VARCHAR(20) DEFAULT 'OPEN'"),
        ("exit_reason", "VARCHAR(50)"),
        ("opened_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
        ("closed_at", "TIMESTAMP"),
        ("order_block_id", "INTEGER"),
        ("sleeper_symbol", "VARCHAR(20)"),
        ("strategy", "VARCHAR(50)"),
    ]
    
    with engine.connect() as conn:
        for col_name, col_type in new_columns:
            try:
                sql = text(f"""
                    DO $$ 
                    BEGIN 
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns 
                            WHERE table_name = '{trades_table}' AND column_name = '{col_name}'
                        ) THEN 
                            ALTER TABLE {trades_table} ADD COLUMN {col_name} {col_type};
                            RAISE NOTICE 'Added column: {col_name}';
                        END IF;
                    END $$;
                """)
                conn.execute(sql)
                conn.commit()
            except Exception as e:
                if 'already exists' not in str(e).lower():
                    print(f"[DB MIGRATE TRADES] Error adding {col_name}: {e}")
                conn.rollback()
    
    print("[DB MIGRATE] Trades migration complete")


def get_session():
    """Get database session"""
    return SessionLocal()
