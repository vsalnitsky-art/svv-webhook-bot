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
    READY = "READY"           # CHoCH detected, готовий до полювання
    STALKING = "STALKING"     # v8.1: Чекаємо відкат до OB
    ENTRY_FOUND = "ENTRY_FOUND"  # v8.1: Ціна в зоні OB - час входити!
    TRIGGERED = "TRIGGERED"
    POSITION = "POSITION"     # v8.1: Позиція відкрита

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

# === Bybit API Keys — three-tier resolver ===
# Priority (highest → lowest):
#   1. DB encrypted blob (set via UI → "Save (encrypted)" button)
#   2. ENV plain (`BYBIT_API_KEY` / `BYBIT_API_SECRET`)
#   3. ENV encrypted (`BYBIT_API_KEY_ENCRYPTED` / `BYBIT_API_SECRET_ENCRYPTED`)
#
# The DB-first priority fixes the original bug that caused us to remove
# the encryption layer: stale ENV `_ENCRYPTED` values silently shadowing
# plain keys when the user updated plain after `_ENCRYPTED` expired. Now
# the most recent action (UI save → DB) always wins, ENV is fallback only.
#
# Master encryption key (`ENCRYPTION_KEY` env var) is REQUIRED only if
# you actually use encrypted storage (DB or ENV). Pure plain ENV works
# without it. Generate with:
#   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# The UI exposes a "Generate" button that produces this for you.

def _decrypt_fernet(encrypted_value: str, encryption_key: str) -> str:
    """Decrypt a Fernet-encrypted blob with the given master key.
    
    Returns the decrypted plaintext, or empty string on failure (with
    a printed warning — never raises). Empty input → empty output, no
    error log (treated as "nothing to decrypt", not a failure).
    """
    if not encrypted_value or not encryption_key:
        return ''
    try:
        from cryptography.fernet import Fernet
        f = Fernet(encryption_key.encode() if isinstance(encryption_key, str)
                   else encryption_key)
        decrypted = f.decrypt(encrypted_value.encode() if isinstance(encrypted_value, str)
                              else encrypted_value)
        return decrypted.decode()
    except Exception as e:
        print(f"[CONFIG] ⚠️ Fernet decrypt failed: {e}")
        return ''


def _encrypt_fernet(plaintext: str, encryption_key: str) -> str:
    """Encrypt plaintext with Fernet (master key from ENCRYPTION_KEY env).
    
    Returned blob is the standard URL-safe-base64 Fernet token (starts
    with `gAAAA...`). Stored in DB and/or shown to user for copy-paste
    into ENV vars. Returns empty string on failure.
    """
    if not plaintext or not encryption_key:
        return ''
    try:
        from cryptography.fernet import Fernet
        f = Fernet(encryption_key.encode() if isinstance(encryption_key, str)
                   else encryption_key)
        token = f.encrypt(plaintext.encode() if isinstance(plaintext, str)
                          else plaintext)
        return token.decode()
    except Exception as e:
        print(f"[CONFIG] ⚠️ Fernet encrypt failed: {e}")
        return ''


def _resolve_bybit_keys_from_db():
    """Try to load encrypted keys from DB. Returns (key, secret, source_tag) or None.
    
    Lazy DB import — bot_settings is imported very early in the boot
    chain, before storage.db_operations is fully wired. The import here
    handles both cases: if DB is up, we use it; if not, we silently fall
    through to env fallbacks.
    """
    enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
    if not enc_key:
        return None  # No master key → can't decrypt anything
    try:
        from storage.db_operations import get_db
        db = get_db()
        if db is None:
            return None
        # Use get_setting which returns the stored value or default ''
        db_key_enc = db.get_setting('bybit_api_key_encrypted', '') or ''
        db_secret_enc = db.get_setting('bybit_api_secret_encrypted', '') or ''
        if not (db_key_enc and db_secret_enc):
            return None
        k = _decrypt_fernet(db_key_enc, enc_key)
        s = _decrypt_fernet(db_secret_enc, enc_key)
        if k and s:
            return k, s, 'db_encrypted'
        return None
    except Exception as e:
        # DB not ready or any other failure — silent fall-through.
        # We don't print here because this gets called at module import
        # time when DB might not be initialised yet (expected race).
        return None


def _resolve_bybit_keys():
    """Three-tier resolver — see module-level docstring for priority order.
    
    Returns (api_key, api_secret). Both empty when nothing configured
    (bot still runs in public-API mode without trading).
    
    Logs which source won at INFO level so users can verify in startup
    logs which key set is active. The log includes char count (NOT the
    key itself — never log credentials) for quick visual sanity check.
    """
    enc_key = os.environ.get('ENCRYPTION_KEY', '').strip()
    
    # 1) DB encrypted — highest priority (most recent user action wins)
    db_result = _resolve_bybit_keys_from_db()
    if db_result:
        k, s, _ = db_result
        print(f"[CONFIG] ✅ Bybit keys: DB encrypted ({len(k)} chars)")
        return k, s
    
    # 2) ENV plain — simple Render setup, no encryption needed
    env_plain_key = os.environ.get('BYBIT_API_KEY', '').strip()
    env_plain_secret = os.environ.get('BYBIT_API_SECRET', '').strip()
    if env_plain_key and env_plain_secret:
        print(f"[CONFIG] ✅ Bybit keys: ENV plain ({len(env_plain_key)} chars)")
        return env_plain_key, env_plain_secret
    
    # 3) ENV encrypted — legacy / explicit fallback
    env_enc_key = os.environ.get('BYBIT_API_KEY_ENCRYPTED', '').strip()
    env_enc_secret = os.environ.get('BYBIT_API_SECRET_ENCRYPTED', '').strip()
    if env_enc_key and env_enc_secret and enc_key:
        k = _decrypt_fernet(env_enc_key, enc_key)
        s = _decrypt_fernet(env_enc_secret, enc_key)
        if k and s:
            print(f"[CONFIG] ✅ Bybit keys: ENV encrypted ({len(k)} chars)")
            return k, s
        else:
            # Decrypt failed even though both vars are set — surface this
            # loudly because it usually means a wrong ENCRYPTION_KEY.
            print(f"[CONFIG] ❌ ENV encrypted keys present but DECRYPT FAILED — "
                  f"check ENCRYPTION_KEY matches the one used for encryption")
    
    # Diagnose what was missing for the user's benefit
    if (env_plain_key and not env_plain_secret) or (env_plain_secret and not env_plain_key):
        print(f"[CONFIG] ❌ Bybit ENV plain INCOMPLETE — both BYBIT_API_KEY and "
              f"BYBIT_API_SECRET must be set together")
    elif (env_enc_key or env_enc_secret) and not enc_key:
        print(f"[CONFIG] ❌ Bybit ENV encrypted set but ENCRYPTION_KEY missing — "
              f"can't decrypt")
    else:
        print(f"[CONFIG] ⚠️ Bybit keys not configured — public API only "
              f"(set via UI Credentials panel, or BYBIT_API_KEY env)")
    return '', ''


# Module-level constants — read once at startup (before DB is fully ready,
# so this typically picks up ENV-based keys). If keys are stored in DB,
# the call to _resolve_bybit_keys_from_db() returns None here (DB not
# initialised yet) — but reload_bybit_keys() is called later from
# main_bot startup AFTER db is ready, picking up the DB values.
BYBIT_API_KEY, BYBIT_API_SECRET = _resolve_bybit_keys()


def reload_bybit_keys():
    """Re-read Bybit keys from all sources. Called after DB init AND after
    UI save to pick up new keys without a full restart.
    
    Mutates module-level BYBIT_API_KEY and BYBIT_API_SECRET. Code that
    imports those at module load time still has stale references — to
    use the fresh values on each access, call this module's
    `get_bybit_keys()` instead.
    """
    global BYBIT_API_KEY, BYBIT_API_SECRET
    BYBIT_API_KEY, BYBIT_API_SECRET = _resolve_bybit_keys()
    return BYBIT_API_KEY, BYBIT_API_SECRET


def get_bybit_keys():
    """Fresh getter — always resolves at call time. Use this in places
    that need up-to-date keys (e.g., after a UI save). For startup-time
    config, the module constants are fine.
    """
    return _resolve_bybit_keys()

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
    
    # v8.2: Volatility Filter
    'min_volatility_pct': 3.0,         # Мінімальна волатильність (%) за 24h
    'auto_blacklist_enabled': False,   # Автоматично додавати "важкі" монети в blacklist
    
    # v8.2: Multi-timeframe settings
    'htf_timeframe': '4h',             # HTF for global bias (4H)
    'ltf_timeframe': '15m',            # LTF for entry signals (15M for fast markets)
    
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
    # Thresholds for direction decision (lowered for more signals)
    'long_threshold': 0.3,      # score >= 0.3 → LONG (was 0.5)
    'short_threshold': -0.3,    # score <= -0.3 → SHORT (was -0.5)
    # Between -0.3 and 0.3 → NEUTRAL (don't trade)
    
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
