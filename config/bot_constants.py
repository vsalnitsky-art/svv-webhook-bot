"""
Bot Constants - Thresholds and fixed values for Sleeper OB Bot
"""

# === SLEEPER DETECTOR THRESHOLDS ===
SLEEPER_THRESHOLDS = {
    # Fuel Score components
    'funding_rate_extreme': 0.0005,      # 0.05% = strong directional bias
    'funding_rate_moderate': 0.0002,     # 0.02% = moderate bias
    
    # OI Change thresholds
    'oi_change_high': 10.0,              # 10% = significant accumulation
    'oi_change_moderate': 5.0,           # 5% = moderate accumulation
    
    # Bollinger Band Width change
    'bb_squeeze_ratio': 0.7,             # < 0.7 = squeeze forming
    'bb_expansion_ratio': 1.3,           # > 1.3 = breakout
    
    # Volume analysis
    'volume_spike_ratio': 2.0,           # 2x average = spike
    'volume_decline_ratio': 0.5,         # < 0.5 = quiet accumulation
    
    # Price range
    'price_range_tight': 0.02,           # 2% range = consolidation
    'price_range_wide': 0.05,            # 5% range = volatile
    
    # HP (Health Points) system
    'hp_initial': 5,                     # Starting HP
    'hp_max': 10,                        # Maximum HP
    'hp_min': 0,                         # Removal threshold
    
    # Score weights
    'weight_fuel': 0.30,                 # 30%
    'weight_volatility': 0.25,           # 25%
    'weight_price': 0.25,                # 25%
    'weight_liquidity': 0.20,            # 20%
}

# === ORDER BLOCK THRESHOLDS ===
OB_THRESHOLDS = {
    # Volume requirements
    'ob_volume_ratio_min': 1.5,          # 1.5x average for valid OB
    'ob_volume_ratio_strong': 2.5,       # 2.5x = strong OB
    
    # Price movement after OB
    'ob_impulse_min': 0.005,             # 0.5% minimum impulse
    'ob_impulse_strong': 0.015,          # 1.5% = strong impulse
    
    # OB zone tolerances
    'ob_touch_tolerance': 0.001,         # 0.1% tolerance for touch
    'ob_mitigation_depth': 0.5,          # 50% into zone = mitigated
    
    # Quality score weights
    'quality_weight_volume': 0.35,       # 35%
    'quality_weight_impulse': 0.30,      # 30%
    'quality_weight_fresh': 0.20,        # 20%
    'quality_weight_structure': 0.15,    # 15%
    
    # Multi-timeframe confirmation
    'mtf_confirmation_bonus': 10,        # +10 to quality if MTF confirmed
}

# === TRADING CONSTANTS ===
TRADING_CONSTANTS = {
    # Position sizing
    'min_position_size': 10,             # $10 minimum
    'max_position_size_pct': 20,         # 20% of balance max
    
    # Leverage limits
    'min_leverage': 1,
    'max_leverage': 20,
    
    # Take profit levels (R:R ratios)
    'tp1_ratio': 1.0,                    # 1R
    'tp2_ratio': 2.0,                    # 2R
    'tp3_ratio': 3.0,                    # 3R
    
    # Position splits at TP levels
    'tp1_close_pct': 50,                 # Close 50% at TP1
    'tp2_close_pct': 25,                 # Close 25% at TP2
    'tp3_close_pct': 25,                 # Close 25% at TP3
    
    # Trailing stop
    'trailing_start_ratio': 1.5,         # Start trailing after 1.5R
    'trailing_offset_pct': 0.5,          # 0.5% trailing offset
    
    # Fees (approximate)
    'maker_fee': 0.0002,                 # 0.02%
    'taker_fee': 0.0006,                 # 0.06%
}

# === TIMEFRAME MAPPINGS ===
TIMEFRAME_MAP = {
    '1m': 1,
    '3m': 3,
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '2h': 120,
    '4h': 240,
    '1d': 1440,
}

# === API LIMITS ===
API_LIMITS = {
    'kline_limit': 200,                  # Max candles per request
    'symbols_per_batch': 20,             # Process 20 symbols at a time
    'rate_limit_delay': 0.1,             # 100ms between requests
    'max_retries': 3,                    # Retry failed requests
}
