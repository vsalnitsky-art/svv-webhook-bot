"""
Database Models - SQLAlchemy models for Sleeper OB Bot
"""
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, Boolean, DateTime, Text, Index
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
            'vc_extreme_detected': self.vc_extreme_detected,  # v3.1
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


class TradeArchive(Base):
    """Permanent, append-only archive of every closed trade.

    Unlike the rolling `tm_closed_trades` setting (capped for the UI), this
    table is NEVER auto-trimmed — it is the dataset for backtesting entry
    quality. Each row carries a full pre-trade snapshot captured
    at OPEN time (entry decision, move-potential, hold score, ATR/runway/
    exhaustion) so we can later test which signals actually predicted good
    vs bad trades. One row per trade; scales to thousands without bloating
    a single JSON blob.
    """
    __tablename__ = f'{TABLE_PREFIX}trade_archive'

    id = Column(Integer, primary_key=True, autoincrement=True)
    is_paper = Column(Boolean, default=False, index=True)   # real vs Test Mode
    symbol = Column(String(20), index=True)
    side = Column(String(8))                                # LONG / SHORT
    entry_price = Column(Float)
    exit_price = Column(Float)
    qty = Column(Float, default=0)
    pnl_pct = Column(Float, index=True)
    pnl_usd = Column(Float, default=0)
    reason = Column(String(40), index=True)                 # close reason code
    reason_detail = Column(Text)                            # enriched reason
    opened_by = Column(String(40))
    opened_at = Column(Float)                               # unix ts
    closed_at = Column(Float, index=True)                   # unix ts
    duration_secs = Column(Float, default=0)
    # Pre-trade snapshot (captured at OPEN) — the backtest feature set.
    # Stored as JSON text so the schema can evolve without migrations.
    entry_snapshot = Column(Text)                           # JSON: decision+move+hold at open
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index(f'ix_{TABLE_PREFIX}arch_sym_closed', 'symbol', 'closed_at'),
    )


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


class SymbolBlacklist(Base):
    """
    Blacklist - v8.2: Монети виключені з аналізу
    
    Причини для blacklist:
    - LOW_VOLATILITY: Рухається < 3% на день
    - STABLECOIN: USDC, BUSD, etc.
    - MANUAL: Вручну додано користувачем
    - DELISTED: Монета видалена з біржі
    """
    __tablename__ = f'{TABLE_PREFIX}symbol_blacklist'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    reason = Column(String(50), default='MANUAL')  # LOW_VOLATILITY/STABLECOIN/MANUAL/DELISTED
    added_at = Column(DateTime, default=datetime.utcnow)
    volatility_24h = Column(Float, default=0)  # % руху за добу коли було додано
    note = Column(String(200))
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'reason': self.reason,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'volatility_24h': self.volatility_24h,
            'note': self.note,
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
        ("vc_extreme_detected", "BOOLEAN DEFAULT FALSE"),  # v3.1: VC > 95% + VOL < 1.2x
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
                # PostgreSQL-compatible: use DO block to add column if not exists
                # This avoids race conditions and works reliably
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
                # Log error but continue with other columns
                error_str = str(e)
                if 'already exists' not in error_str.lower():
                    print(f"[DB MIGRATE] Error adding {col_name}: {e}")
                conn.rollback()
    
    print("[DB MIGRATE] Migration complete")
    
    # ==========================================================
    # Migrate SMCOBState — bar_time_ms / created_at_t Integer→BigInteger
    # ==========================================================
    # The first deploy of SMCOBState used Integer (32-bit) for ms-epoch
    # columns. PostgreSQL Integer max is ~2.1B; ms timestamps are ~1.78e12
    # and overflow on every insert. This migration alters those columns
    # in-place. Idempotent — checks current column type before altering.
    # SQLite ignores this entirely (it stores as INTEGER 64-bit always).
    smc_ob_table = f"{TABLE_PREFIX}smc_ob_state"
    smc_ob_bigint_cols = ['bar_time_ms', 'created_at_t']
    from sqlalchemy import text
    with engine.connect() as conn:
        for col_name in smc_ob_bigint_cols:
            try:
                # PostgreSQL only — SQLite has dynamic typing
                # Check if column exists AND is currently INTEGER
                check_sql = text(f"""
                    SELECT data_type FROM information_schema.columns
                    WHERE table_name = '{smc_ob_table}'
                      AND column_name = '{col_name}'
                """)
                result = conn.execute(check_sql).fetchone()
                if result and result[0] == 'integer':
                    alter_sql = text(
                        f"ALTER TABLE {smc_ob_table} "
                        f"ALTER COLUMN {col_name} TYPE BIGINT "
                        f"USING {col_name}::BIGINT"
                    )
                    conn.execute(alter_sql)
                    conn.commit()
                    print(f"[DB MIGRATE] Altered {smc_ob_table}.{col_name} → BIGINT")
                # If result is None: table doesn't exist yet (will be created
                # with correct types by Base.metadata.create_all). If type is
                # already 'bigint': nothing to do.
            except Exception as e:
                # Don't crash boot — just log. SQLite raises here because
                # information_schema doesn't exist; that's expected and fine.
                error_str = str(e)
                if 'no such table: information_schema' not in error_str.lower() \
                   and 'no such column' not in error_str.lower():
                    print(f"[DB MIGRATE] OB state migration warning ({col_name}): {e}")
                conn.rollback()
        
        # Add `created_by_tag` column on tables that pre-date its introduction.
        # Same approach: check information_schema first, ALTER if missing.
        # On SQLite this is a no-op via the same swallow-errors path below.
        try:
            check_sql = text(f"""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = '{smc_ob_table}'
                  AND column_name = 'created_by_tag'
            """)
            result = conn.execute(check_sql).fetchone()
            if result is None:
                # Column missing — add it. NULL is fine for existing rows;
                # next scan tick will populate with the live value.
                # But we also need to verify the table itself exists first
                # (check_sql returns None whether table missing OR column missing).
                table_check = conn.execute(text(f"""
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = '{smc_ob_table}'
                """)).fetchone()
                if table_check is not None:
                    alter_sql = text(
                        f"ALTER TABLE {smc_ob_table} "
                        f"ADD COLUMN created_by_tag VARCHAR(10)"
                    )
                    conn.execute(alter_sql)
                    conn.commit()
                    print(f"[DB MIGRATE] Added {smc_ob_table}.created_by_tag")
        except Exception as e:
            error_str = str(e)
            if 'no such table: information_schema' not in error_str.lower() \
               and 'duplicate column' not in error_str.lower():
                print(f"[DB MIGRATE] OB state created_by_tag warning: {e}")
            conn.rollback()
        
        # Add `zone` and `zone_correct` columns on Top100OBSnapshot. Pre-
        # existing tables won't have them; new schema does. Idempotent —
        # checks information_schema before each ALTER.
        top100_table = f"{TABLE_PREFIX}top100_ob_snapshots"
        top100_new_cols = [
            ('zone', 'VARCHAR(15)'),
            ('zone_correct', 'BOOLEAN'),
            # zone_pct: position % of OB midpoint within the latest swing
            # range. Added in the threshold-based-filter rework (1H Top-100
            # default). FLOAT lets us store fractions (e.g. 12.3%) and
            # values outside [0,100] when OB is beyond range extremes.
            ('zone_pct', 'FLOAT'),
        ]
        try:
            table_check = conn.execute(text(f"""
                SELECT 1 FROM information_schema.tables
                WHERE table_name = '{top100_table}'
            """)).fetchone()
            if table_check is not None:
                for col_name, col_type in top100_new_cols:
                    col_check = conn.execute(text(f"""
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = '{top100_table}'
                          AND column_name = '{col_name}'
                    """)).fetchone()
                    if col_check is None:
                        conn.execute(text(
                            f"ALTER TABLE {top100_table} "
                            f"ADD COLUMN {col_name} {col_type}"
                        ))
                        conn.commit()
                        print(f"[DB MIGRATE] Added {top100_table}.{col_name}")
        except Exception as e:
            error_str = str(e)
            if 'no such table: information_schema' not in error_str.lower() \
               and 'duplicate column' not in error_str.lower():
                print(f"[DB MIGRATE] Top100 zone migration warning: {e}")
            conn.rollback()


def get_session():
    """Get database session"""
    return SessionLocal()


# ============================================================
# SMC Order Block State (Pine SMC_PRO_BOT__47_ algorithm)
# ============================================================
# One row per symbol holds the LAST VALID (unmitigated) Order Block
# computed at the user-selected OB timeframe. Updated on every scan tick
# for every symbol in the watchlist. Used by the OB Filter to gate
# signal opening: a LONG signal only fires when this row's bias is
# BULLISH; SHORT only when BEARISH; missing/mitigated → block.
#
# This is INTENTIONALLY a separate table from `OrderBlock` (which is for
# the other "Volumized Order Blocks" scanner with different semantics).
# Naming clarifies which OB scanner owns the row.
#
# Storage strategy: one row per (symbol, timeframe). When timeframe
# changes via settings, old rows become stale but harmless — the writer
# always writes to the symbol+TF currently configured, and the reader
# always queries the symbol+TF currently configured. We intentionally
# don't garbage-collect old TF rows on settings change since the user
# may flip back; they'll just get overwritten when scanned.

class SMCOBState(Base):
    """Last valid Pine SMC OB per symbol+timeframe."""
    __tablename__ = f'{TABLE_PREFIX}smc_ob_state'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(5), nullable=False)  # 15m, 30m, 1h, 4h
    
    # OB data — None when no valid OB exists for the (symbol, tf)
    bias = Column(String(10))         # 'BULLISH' | 'BEARISH' | NULL
    bar_high = Column(Float)
    bar_low = Column(Float)
    # IMPORTANT: ms-epoch timestamps need BigInteger (64-bit). Postgres
    # Integer is signed 32-bit (max ~2.1B) and 13-digit ms epochs overflow.
    # Without BigInteger every upsert fails with NumericValueOutOfRange and
    # the entire OB Filter pipeline silently writes nothing to DB.
    bar_time_ms = Column(BigInteger)  # ms epoch — Pine barTime
    bar_idx = Column(Integer)
    created_at_idx = Column(Integer)  # bar where the BOS/CHoCH triggered storage
    created_at_t = Column(BigInteger) # ms epoch of the triggering event
    # Which event tag created this OB — 'CHoCH' or 'BOS'. Pine fires
    # storeOrdeBlock on both equally; we keep the tag so the OB Filter
    # can require CHoCH-only (fresh trend reversal) and reject continuation
    # OBs created by BOS-after-CHoCH.
    created_by_tag = Column(String(10))
    
    # When this row was last refreshed by the scanner
    computed_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_smc_ob_symbol_tf', 'symbol', 'timeframe', unique=True),
    )
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'bias': self.bias,
            'bar_high': self.bar_high,
            'bar_low': self.bar_low,
            'bar_time': self.bar_time_ms,
            'bar_idx': self.bar_idx,
            'created_at_idx': self.created_at_idx,
            'created_at_t': self.created_at_t,
            'created_by_tag': self.created_by_tag,
            'computed_at': self.computed_at,
        }


# ============================================================
# TOP-100 4H OB Radar (Variant B: scheduled scan)
# ============================================================
# Two tables for the TOP-100 OB scanner module.
#
# Top100OBSnapshot — current state: one row per symbol with the latest
#   detected OB on the 4H timeframe (or NULL if no valid OB exists).
#   Overwritten on every scan. Plus tracking metadata: when we first saw
#   this OB (`discovered_at`) and when we last confirmed it
#   (`last_seen_at`). A "fresh" OB is one whose discovered_at == this
#   scan's timestamp — i.e. didn't exist on the previous scan.
#
# Top100OBHistory — append-only audit log. Each row records an event:
#   when an OB was created, mitigated, or replaced by a newer OB on the
#   same symbol. This is what the "Recent Discoveries" UI feed reads.
#
# This is INTENTIONALLY separate from SMCOBState (the per-watchlist gate
# table). Different scanner, different cadence, different semantics. The
# TOP-100 scanner does NOT gate signals — purely informational.

class Top100OBSnapshot(Base):
    """Current state of the latest 4H OB for each TOP-100 symbol."""
    __tablename__ = f'{TABLE_PREFIX}top100_ob_snapshots'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(30), nullable=False, unique=True, index=True)
    
    # 24h market context — captured at scan time so UI doesn't have to
    # re-fetch tickers when rendering
    quote_volume_24h = Column(Float)   # USD turnover, used for min vol filter
    last_price = Column(Float)
    price_change_24h = Column(Float)   # % change
    
    # OB data — NULL if no valid OB exists
    bias = Column(String(10))         # 'BULLISH' | 'BEARISH' | NULL
    bar_high = Column(Float)
    bar_low = Column(Float)
    bar_time_ms = Column(BigInteger)  # ms epoch of OB pivot bar
    created_at_t = Column(BigInteger) # ms epoch when OB was created
    created_by_tag = Column(String(10))  # 'CHoCH' or 'BOS'
    
    # Premium/Discount/Equilibrium zone classification.
    # Computed at scan time using the latest swing high/low pivots:
    #   range = swing_high.level - swing_low.level
    #   ob_mid = (bar_high + bar_low) / 2
    #   pos_pct = (ob_mid - swing_low) / range × 100
    # Classification:
    #   pos_pct < 38.2  → 'Discount'  (lower third — favorable for LONG)
    #   pos_pct > 61.8  → 'Premium'   (upper third — favorable for SHORT)
    #   otherwise       → 'Equilibrium'
    # zone_correct: True when zone aligns with OB direction:
    #   BULLISH OB in Discount  → True
    #   BEARISH OB in Premium   → True
    #   anything else           → False
    # NULL when range cannot be determined (no swing pivots yet).
    zone = Column(String(15))         # 'Discount' | 'Mid' | 'Premium' | NULL
    zone_correct = Column(Boolean)    # True if zone aligns with OB bias
    # Position percent within the latest swing range. Range [0, 100] for
    # OBs inside the trading range; can exceed when OB is beyond extremes.
    # Stored separately from `zone` so UI can show the exact percent
    # alongside the zone label (e.g. "Discount 12%" vs "Premium 87%").
    zone_pct = Column(Float)
    
    # Lifecycle tracking — separates "first time seeing this OB" from
    # "OB still here from previous scan". `discovered_at` is set when the
    # OB first appears (created_at_t differs from prior scan's value).
    # `last_seen_at` updates every scan that the same OB is still valid.
    discovered_at = Column(DateTime)   # First scan when this exact OB appeared
    last_seen_at = Column(DateTime)    # Last scan that confirmed this OB
    
    # Snapshot metadata
    scanned_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_top100_snapshot_symbol', 'symbol', unique=True),
    )
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'quote_volume_24h': self.quote_volume_24h,
            'last_price': self.last_price,
            'price_change_24h': self.price_change_24h,
            'bias': self.bias,
            'bar_high': self.bar_high,
            'bar_low': self.bar_low,
            'bar_time_ms': self.bar_time_ms,
            'created_at_t': self.created_at_t,
            'created_by_tag': self.created_by_tag,
            'zone': self.zone,
            'zone_correct': self.zone_correct,
            'zone_pct': self.zone_pct,
            'discovered_at': self.discovered_at.isoformat() if self.discovered_at else None,
            'last_seen_at': self.last_seen_at.isoformat() if self.last_seen_at else None,
            'scanned_at': self.scanned_at.isoformat() if self.scanned_at else None,
        }


class Top100OBHistory(Base):
    """Audit log of OB lifecycle events for TOP-100 4H scanner."""
    __tablename__ = f'{TABLE_PREFIX}top100_ob_history'
    
    # BIGINT id — this table grows continuously (one row per event ~5-30
    # rows per scan × 6 scans/day = up to 180 rows/day = ~65k/year).
    # Standard Integer would last ~33k years anyway, but BigInteger
    # signals "this is meant to grow" and matches existing conventions.
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String(30), nullable=False, index=True)
    
    # 'created' = OB first appeared (snapshot didn't have it last scan)
    # 'mitigated' = OB was active, now no longer valid
    # 'replaced' = newer OB took the same direction (BOS continuation usually)
    event_type = Column(String(20), nullable=False)
    
    # Snapshot of the OB at event time (denormalized for audit clarity)
    bias = Column(String(10))
    bar_high = Column(Float)
    bar_low = Column(Float)
    bar_time_ms = Column(BigInteger)
    created_by_tag = Column(String(10))
    
    # Market context at event time
    price_at_event = Column(Float)
    quote_volume_24h = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index('idx_top100_history_symbol_time', 'symbol', 'created_at'),
        Index('idx_top100_history_event_time', 'event_type', 'created_at'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'event_type': self.event_type,
            'bias': self.bias,
            'bar_high': self.bar_high,
            'bar_low': self.bar_low,
            'bar_time_ms': self.bar_time_ms,
            'created_by_tag': self.created_by_tag,
            'price_at_event': self.price_at_event,
            'quote_volume_24h': self.quote_volume_24h,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


# ============================================================
# Volumized OB Radar — three tables
# ============================================================
# 1. VolumizedRadarMetadata — per-symbol tracking while item is "owned" by
#    the radar. Row exists only between auto-add and removal (auto/manual/
#    signal-fired). Indexed on `expires_at` for fast cleanup-daemon scan.
# 2. VolumizedRadarStat — lifetime counters per symbol. Survives metadata
#    row removal. Used for analytics ("how often does BTC give radar
#    signals that actually fire?") and for cooldown enforcement.
# 3. VolumizedRadarSnapshot — per-scan audit log. Records every scan
#    decision for inspection. Indexed on scan_time for time-range queries.
# ============================================================

class VolumizedRadarMetadata(Base):
    """Tracking metadata for symbols currently watched by Volumized OB Radar.
    
    Lifecycle: row inserted on auto-add (qualifying Volumized OB found in
    P/D zone). Deleted when (a) SMC signal fires within 24h → row +
    stats.times_signal_fired++; (b) 24h passes without signal → row +
    stats.times_auto_removed++ + cooldown set; (c) user manually removes
    from watchlist → row + stats.times_manual_removed++ + cooldown.
    
    The radar never reads this for "is symbol in watchlist?" — that's the
    smc_watchlist JSON list's job. This table is solely for radar's own
    bookkeeping: who's on the clock, until when, with what OB context.
    """
    __tablename__ = 'volumized_radar_metadata'
    
    symbol = Column(String(20), primary_key=True)
    added_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    # 24h after added_at by default. Stored explicitly to make cleanup
    # query trivial (`WHERE expires_at < now`) without datetime arithmetic.
    expires_at = Column(DateTime, nullable=False)
    
    # OB context when added — for UI tooltip and audit
    ob_direction = Column(String(10))     # 'BULL' | 'BEAR'
    ob_top = Column(Float)
    ob_bottom = Column(Float)
    ob_volume = Column(Float)              # OB's volume (for Vol indicator)
    pd_zone_pct = Column(Float)            # 0.0-100.0; ≤38.2 = Discount, ≥61.8 = Premium
    scan_tf = Column(String(10))           # '1h' | '4h' etc. — Volumized scan TF
    
    # Latched the moment _send_alert fires for this symbol. Used by cleanup
    # daemon to remove the row WITHOUT also removing from watchlist (user
    # said: "сигнал спрацював — символ переходить у normal flow як всі").
    signal_fired_at = Column(DateTime, nullable=True)
    
    __table_args__ = (
        # Cleanup daemon scans this index — fastest possible TTL-expiry query
        Index('idx_volradar_expires', 'expires_at'),
    )
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'ob_direction': self.ob_direction,
            'ob_top': self.ob_top,
            'ob_bottom': self.ob_bottom,
            'ob_volume': self.ob_volume,
            'pd_zone_pct': self.pd_zone_pct,
            'scan_tf': self.scan_tf,
            'signal_fired_at': self.signal_fired_at.isoformat() if self.signal_fired_at else None,
        }


class VolumizedRadarStat(Base):
    """Lifetime counters per symbol — survives metadata row removal.
    
    Powers the radar analytics view: which coins does the radar pick up most
    often, what's their signal-fire conversion rate, how often were they
    manually rejected. Also stores `last_cooldown_until` to enforce the
    re-add cooldown (default 6h after any removal — prevents the radar
    from re-adding a symbol the user just dismissed).
    """
    __tablename__ = 'volumized_radar_stats'
    
    symbol = Column(String(20), primary_key=True)
    times_added = Column(Integer, default=0, nullable=False)
    times_signal_fired = Column(Integer, default=0, nullable=False)
    times_auto_removed = Column(Integer, default=0, nullable=False)
    times_manual_removed = Column(Integer, default=0, nullable=False)
    last_added_at = Column(DateTime, nullable=True)
    last_cooldown_until = Column(DateTime, nullable=True)
    
    def to_dict(self):
        total_resolved = (self.times_signal_fired + self.times_auto_removed
                         + self.times_manual_removed)
        conv_rate = (self.times_signal_fired / total_resolved
                    if total_resolved > 0 else 0.0)
        return {
            'symbol': self.symbol,
            'times_added': self.times_added,
            'times_signal_fired': self.times_signal_fired,
            'times_auto_removed': self.times_auto_removed,
            'times_manual_removed': self.times_manual_removed,
            'conversion_rate': round(conv_rate, 3),
            'last_added_at': self.last_added_at.isoformat() if self.last_added_at else None,
            'last_cooldown_until': self.last_cooldown_until.isoformat() if self.last_cooldown_until else None,
        }


class VolumizedRadarSnapshot(Base):
    """Per-scan audit log — one row per (scan_time, symbol) tuple.
    
    Records EVERY scan decision (not only successful adds). Lets users
    answer "why didn't this symbol get added?" by inspecting the snapshot:
    qualifies/doesn't, action taken, P/D zone pct, etc.
    
    Auto-pruned by db_operations.prune_volradar_snapshots() — keeps last
    7 days by default.
    """
    __tablename__ = 'volumized_radar_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    scan_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    
    # OB context (nullable when no OB found at all)
    ob_direction = Column(String(10), nullable=True)
    ob_top = Column(Float, nullable=True)
    ob_bottom = Column(Float, nullable=True)
    pd_zone_pct = Column(Float, nullable=True)
    
    qualified = Column(Boolean, default=False, nullable=False)
    # 'added' | 'skipped_already_in_watchlist' | 'skipped_cooldown'
    # | 'skipped_not_in_zone' | 'skipped_no_swings' | 'skipped_no_ob'
    # | 'skipped_capacity_full' | 'error'
    action = Column(String(40), nullable=False)
    error_msg = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_volradar_snap_time', 'scan_time'),
        Index('idx_volradar_snap_symbol_time', 'symbol', 'scan_time'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'scan_time': self.scan_time.isoformat() if self.scan_time else None,
            'symbol': self.symbol,
            'ob_direction': self.ob_direction,
            'ob_top': self.ob_top,
            'ob_bottom': self.ob_bottom,
            'pd_zone_pct': self.pd_zone_pct,
            'qualified': self.qualified,
            'action': self.action,
            'error_msg': self.error_msg,
        }


# ============================================================================
# Liquidation Map module — estimated liquidation level clusters per symbol
# ============================================================================

class LiquidationBucket(Base):
    """One estimated liquidation cluster at a specific (symbol, price, side,
    leverage). Cumulative — each OI-delta tick adds to existing bucket if one
    matches, otherwise creates new. Mitigated when price wicks through it.
    
    Phase 1 storage layout: one row per (symbol, bucket_price, side, leverage,
    source). At ~$25 bucket size on BTC ±10% from price, that's ~80 buckets ×
    2 sides × 3 leverage tiers × 2 sources = ~960 rows per symbol active at
    once. With ~2 background symbols + a few on-demand it's well under 10K
    active rows — cheap to index and query.
    """
    __tablename__ = f'{TABLE_PREFIX}liquidation_buckets'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    
    # Bucket midpoint price. We bucket-quantize the raw liquidation price so
    # multiple individual position estimates collapse onto the same row.
    bucket_price = Column(Float, nullable=False)
    side = Column(String(10), nullable=False)        # 'long' | 'short'
    leverage = Column(Integer, nullable=False)       # 25 | 50 | 100
    
    # Accumulated notional USD attributed to this bucket since first_seen
    cumulative_usd = Column(Float, nullable=False, default=0.0)
    # How many distinct OI-delta events fed into this bucket
    contribution_count = Column(Integer, nullable=False, default=0)
    
    first_seen_ts = Column(BigInteger, nullable=False)   # epoch seconds
    last_updated_ts = Column(BigInteger, nullable=False)
    
    # When price wick'd through the bucket and (presumably) liquidated the
    # positions there. NULL = still active. Mitigated buckets are kept for
    # post-mortem analysis (UI toggle "show mitigated"); retention job drops
    # them after 30 days regardless.
    mitigated_at_ts = Column(BigInteger, nullable=True)
    
    # Provenance — lets us trust some buckets more than others in UI:
    # 'estimation' = derived from aggregated OI deltas + assumed lev weights
    # 'hyperliquid' = derived from actual on-chain position data (subset of
    #                 the market but accurate)
    source = Column(String(20), nullable=False, default='estimation')
    
    __table_args__ = (
        Index('idx_liqmap_active', 'symbol', 'mitigated_at_ts'),
        Index('idx_liqmap_time', 'symbol', 'last_updated_ts'),
        # Lookups for "do we already have a bucket at this price?" hot-path
        Index('idx_liqmap_lookup', 'symbol', 'bucket_price', 'side',
              'leverage', 'source', 'mitigated_at_ts'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'bucket_price': self.bucket_price,
            'side': self.side,
            'leverage': self.leverage,
            'cumulative_usd': self.cumulative_usd,
            'contribution_count': self.contribution_count,
            'first_seen_ts': self.first_seen_ts,
            'last_updated_ts': self.last_updated_ts,
            'mitigated_at_ts': self.mitigated_at_ts,
            'source': self.source,
        }


class LiquidationOISnapshot(Base):
    """One OI snapshot per (symbol, exchange) per scan tick. Used by the
    daemon to compute OI deltas between consecutive ticks. Older snapshots
    are pruned aggressively — we only need t-1 to compute the latest delta.
    """
    __tablename__ = f'{TABLE_PREFIX}liquidation_oi_snapshots'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    exchange = Column(String(20), nullable=False)
    ts = Column(BigInteger, nullable=False)              # epoch seconds
    
    open_interest_usd = Column(Float, nullable=False)
    mark_price = Column(Float, nullable=False)
    # Optional — Binance topLongShortPositionRatio when available
    long_ratio = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_liqoi_latest', 'symbol', 'exchange', 'ts'),
    )


# ============================================================================
# Event-based Liquidation Map storage (new in Phase 3)
# ============================================================================
# Replaces the cumulative-bucket flow of LiquidationBucket with per-event
# rows: every OI tick that adds new positions writes a separate row with
# its own timestamp. This is what gives the Hyblock-style "many short
# dashes scattered across time" rendering — each dash = one event.
#
# Each event represents one (price, side, leverage, source) contribution
# in time. Frontend renders each as a short horizontal dash at (event.ts,
# event.bucket_price). Aggregation back to clusters/heatmap density is
# done on-read.
#
# Volume math: at ~60s tick × 2 providers × ~6 contributions per tick =
# ~720 events/hour/symbol → ~17K/day → ~500K/month/symbol. With 2 BG
# symbols + ad-hoc that's ~1.5M rows after 30d. Indexed by (symbol, ts)
# this scales fine; queries are bounded by lookback + LIMIT.
class LiquidationEvent(Base):
    __tablename__ = f'{TABLE_PREFIX}liquidation_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    
    # When the event happened (epoch seconds). Events are immutable —
    # once written, never updated except for mitigated_at_ts.
    ts = Column(BigInteger, nullable=False)
    
    bucket_price = Column(Float, nullable=False)
    side = Column(String(10), nullable=False)        # 'long' | 'short'
    leverage = Column(Integer, nullable=False)        # 25 | 50 | 100
    
    # USD value of THIS individual contribution. The cumulative volume
    # at a price level is the SUM of usd_added across all events with
    # the same (symbol, bucket_price, side, leverage) — computed on-read.
    usd_added = Column(Float, nullable=False)
    
    # 'estimation' = derived from aggregated OI deltas
    # 'hyperliquid' = derived from on-chain position data
    source = Column(String(20), nullable=False, default='estimation')
    
    # When price wick'd through this event's bucket_price.
    # NULL = still active.
    mitigated_at_ts = Column(BigInteger, nullable=True)
    
    __table_args__ = (
        # Time-bound queries (the hot path — fetching last N hours)
        Index('idx_liqev_time', 'symbol', 'ts'),
        # Mitigation pass — find unmitigated events in a price band
        Index('idx_liqev_mitigation', 'symbol', 'mitigated_at_ts', 'bucket_price'),
        # Active-only time queries (UI default — exclude mitigated)
        Index('idx_liqev_active', 'symbol', 'mitigated_at_ts', 'ts'),
    )
    
    def to_dict(self):
        return {
            'id': self.id,
            'ts': self.ts,
            'bucket_price': self.bucket_price,
            'side': self.side,
            'leverage': self.leverage,
            'usd_added': self.usd_added,
            'source': self.source,
            'mitigated_at_ts': self.mitigated_at_ts,
        }


# ============================================================================
# Liquidity Heatmap profiles — depth snapshots over time per symbol
# ============================================================================
# Stores full depth profile (bid + ask clusters within ±3% of mid-price)
# every ~60s per tracked symbol. The frontend Heatmap tab aggregates
# these into a time × price grid for the CoinGlass-style visualization.
#
# Replaces the previous setting-blob storage (db.set_setting('liq_heatmap_profiles', list))
# which rewrote the full JSON on every snapshot — unworkable at scale.
#
# Sizing: 3 symbols × 60s × 24h × 7d = 30240 rows. Each row ~1-2 KB of JSON
# in bid/ask. Total ~50 MB worst case, bounded by cleanup_liq_heatmap_profiles.
class LiqHeatmapProfile(Base):
    __tablename__ = f'{TABLE_PREFIX}liq_heatmap_profiles'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    ts = Column(DateTime, nullable=False, default=datetime.utcnow)
    mid_price = Column(Float, nullable=False)
    # Compact JSON arrays: [[price_int, volume_thousands_int], ...]
    bid_data = Column(Text, nullable=False, default='[]')
    ask_data = Column(Text, nullable=False, default='[]')

    __table_args__ = (
        # Hot path: fetch last N hours for a symbol
        Index('idx_liqhm_symbol_ts', 'symbol', 'ts'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'ts': self.ts.isoformat() if self.ts else None,
            'mid_price': self.mid_price,
            'bid_data': self.bid_data,
            'ask_data': self.ask_data,
        }
