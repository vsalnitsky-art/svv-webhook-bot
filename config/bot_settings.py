"""
Bot Settings - Main configuration for Sleeper OB Bot
"""
import os
from enum import Enum

class ExecutionMode(Enum):
    MANUAL = "MANUAL"
    SEMI_AUTO = "SEMI_AUTO"
    AUTO = "AUTO"

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
BYBIT_API_KEY = os.environ.get('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.environ.get('BYBIT_API_SECRET', '')
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
    'execution_mode': ExecutionMode.SEMI_AUTO.value,
    'paper_trading': True,
    'paper_balance': 10000.0,
    
    'sleeper_scan_interval': 240,
    'sleeper_min_score': 60,
    'sleeper_ready_score': 80,
    'sleeper_max_symbols': 100,
    'sleeper_min_volume': 20000000,
    
    'ob_scan_interval': 1,
    'ob_min_quality': 65,
    'ob_max_age_minutes': 60,
    'ob_timeframes': ['15m', '5m', '1m'],
    
    'max_risk_per_trade': 1.0,
    'max_open_positions': 3,
    'default_leverage': 5,
    'stop_loss_atr_mult': 1.5,
    'take_profit_rr': 2.0,
    
    'telegram_enabled': True,
    'telegram_chat_id': TELEGRAM_CHAT_ID,
    'alert_on_signal': True,
    'alert_on_entry': True,
    'alert_on_exit': True,
}

BYBIT_CONFIG = {
    'testnet': False,
    'recv_window': 5000,
    'timeout': 30,
}

WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.environ.get('PORT', 5000)),
    'debug': os.environ.get('DEBUG', 'False').lower() == 'true',
}
