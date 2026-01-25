"""
Bot Settings - Main configuration for Sleeper OB Bot
"""
import os
from enum import Enum

class ExecutionMode(Enum):
    MANUAL = "MANUAL"
    SEMI_AUTO = "SEMI_AUTO"
    AUTO = "AUTO"

class TradingMode(Enum):
    """Trading style mode"""
    SCALPING = "SCALPING"   # Quick trades, tight stops, frequent
    SWING = "SWING"         # Longer holds, wider stops, patient

class SleeperState(Enum):
    IDLE = "IDLE"
    WATCHING = "WATCHING"
    BUILDING = "BUILDING"
    READY = "READY"
    TRIGGERED = "TRIGGERED"

class OBStatus(Enum):
    ACTIVE = "ACTIVE"
    TOUCHED = "TOUCHED"
    MITIGATED = "MITIGATED"
    EXPIRED = "EXPIRED"

class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"

# === ENVIRONMENT VARIABLES ===
# Binance - для сканування та аналізу
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET', '')

# Bybit - для торгівлі
BYBIT_API_KEY = os.environ.get('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.environ.get('BYBIT_API_SECRET', '')

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# === DATABASE ===
# Render uses postgres:// but SQLAlchemy 2.0 requires postgresql://
_db_url = os.environ.get('DATABASE_URL', 'sqlite:///sleeper_ob_bot.db')
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
DATABASE_URL = _db_url

# === DEFAULT SETTINGS ===
DEFAULT_SETTINGS = {
    # === EXECUTION ===
    'execution_mode': ExecutionMode.SEMI_AUTO.value,
    'trading_mode': TradingMode.SWING.value,  # SCALPING or SWING
    'paper_trading': True,
    'paper_balance': 10000.0,
    
    # === TREND ANALYZER (4H Context) ===
    'use_trend_filter': True,              # Enable 4H trend filtering
    'trend_timeframe': '240',              # 4H = 240 minutes
    'min_trend_score': 65,                 # Minimum score to allow signals
    'allow_signals_without_trend': False,  # Block if trend data unavailable
    
    # Trend component weights (must sum to 1.0)
    'trend_weight_structure': 0.30,    # Market structure (swing HH/HL)
    'trend_weight_volatility': 0.25,   # Volatility expansion
    'trend_weight_acceptance': 0.25,   # VWAP acceptance
    'trend_weight_momentum': 0.20,     # Momentum asymmetry
    
    # === SLEEPER SCANNER ===
    'sleeper_scan_interval': 240,
    'sleeper_min_score': 40,           # Minimum to track
    'sleeper_building_score': 55,      # BUILDING state threshold
    'sleeper_ready_score': 60,         # READY state threshold
    'sleeper_max_symbols': 200,
    'sleeper_min_volume': 20000000,
    
    # Sleeper component weights
    'sleeper_weight_flatness': 0.30,
    'sleeper_weight_volume': 0.25,
    'sleeper_weight_pressure': 0.25,
    'sleeper_weight_liquidity': 0.20,
    
    # === ORDER BLOCK SCANNER ===
    'ob_scan_interval': 1,
    'ob_min_quality': 65,
    'ob_max_age_minutes': 60,
    'ob_timeframes': '15,5,1',         # Minutes
    'ob_swing_length': 5,
    'ob_max_atr_mult': 3.0,
    'ob_zone_count': 5,
    'ob_end_method': 'Wick',
    'ob_max_count': 10,
    
    # === RISK MANAGEMENT ===
    'max_risk_per_trade': 1.0,         # % of balance
    'max_open_positions': 3,
    'default_leverage': 5,
    
    # Scalping mode risk
    'scalping_stop_loss_atr': 1.0,     # Tight stop
    'scalping_take_profit_rr': 1.5,    # Lower RR
    'scalping_max_hold_minutes': 60,   # Max 1 hour
    
    # Swing mode risk
    'swing_stop_loss_atr': 2.0,        # Wider stop
    'swing_take_profit_rr': 3.0,       # Higher RR
    'swing_trailing_start_rr': 1.5,    # Start trailing after 1.5R
    
    # Legacy (used if mode not set)
    'stop_loss_atr_mult': 1.5,
    'take_profit_rr': 2.0,
    
    # === TELEGRAM ===
    'telegram_enabled': True,
    'telegram_chat_id': TELEGRAM_CHAT_ID,
    'alert_on_signal': True,
    'alert_on_entry': True,
    'alert_on_exit': True,
}

# === TRADING MODE PRESETS ===
SCALPING_PRESET = {
    'setup_timeframe': '15',       # 15m for setup
    'entry_timeframe': '5',        # 5m or 1m for entry
    'stop_loss_atr': 1.0,
    'take_profit_rr': 1.5,
    'max_hold_minutes': 60,
    'target_winrate': 0.60,        # 60%+ expected
    'min_trend_score': 70,         # Stricter in scalping
}

SWING_PRESET = {
    'setup_timeframe': '60',       # 1H for setup
    'entry_timeframe': '15',       # 15m for entry
    'stop_loss_atr': 2.0,
    'take_profit_rr': 3.0,
    'trailing_start_rr': 1.5,
    'target_winrate': 0.40,        # 40%+ expected
    'min_trend_score': 65,         # Standard threshold
}

BYBIT_CONFIG = {
    'testnet': False,
    'recv_window': 5000,
    'timeout': 30,
}

BINANCE_CONFIG = {
    'testnet': False,
    'timeout': 30,
    # Rate limiting
    'requests_per_minute': 1200,  # Binance limit
    'safe_margin': 0.8,  # Use 80% of limit
}

WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 5000)),
    'debug': os.environ.get('DEBUG', 'False').lower() == 'true',
}

# === DIRECTION ENGINE v1.0 ===
# Professional 3-layer direction model
DIRECTION_ENGINE = {
    # Thresholds for direction decision
    'long_threshold': 0.5,      # score >= 0.5 → LONG
    'short_threshold': -0.5,    # score <= -0.5 → SHORT
    # Between -0.5 and 0.5 → NEUTRAL (don't trade)
    
    # Layer weights (must sum to 1.0)
    'weights': {
        'htf': 0.5,         # HTF Structural Bias (1D structure + 4H EMA)
        'ltf': 0.3,         # LTF Momentum Shift (RSI divergence + BB)
        'derivatives': 0.2   # Derivatives Positioning (OI + Funding)
    },
    
    # HTF thresholds
    'htf_ema_period': 50,       # EMA period for 1D structure
    'htf_slope_period': 5,      # Candles for slope calculation
    'htf_price_threshold': 1,   # % distance from EMA for bias
    
    # Derivatives thresholds
    'funding_high': 0.0005,     # 0.05% = crowded
    'funding_low': -0.0003,     # -0.03% = squeeze potential
    'oi_significant': 5,        # 5% OI change = significant
    'price_move_threshold': 1,  # 1% price move = significant
}
