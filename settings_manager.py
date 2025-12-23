#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from models import db_manager, BotSetting

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    # === GENERAL ===
    "scanner_quote_coin": "USDT",
    "scanner_mode": "Manual",
    "scan_limit": 100,
    "scan_min_volume": 10,
    "scan_use_min_volume": True,
    
    # === TELEGRAM ===
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    
    # === RSI/MFI SCREENER ===
    # Timeframes
    "screener_main_tf": "60",           # 1h
    "screener_htf": "240",              # 4h (auto-linked)
    
    # Volume
    "screener_min_volume": 10000000,    # $10M
    
    # RSI Settings
    "screener_rsi_length": 14,
    "screener_oversold": 30,
    "screener_overbought": 70,
    
    # MFI Settings
    "screener_mfi_length": 20,
    "screener_fast_mfi_ema": 5,
    "screener_slow_mfi_ema": 13,
    
    # HMA Settings
    "screener_hma_fast": 10,
    "screener_hma_slow": 30,
    
    # Signal Settings
    "screener_min_peak_strength": 2,
    "screener_require_volume": False,
    "screener_trend_confirmation": False,
    
    # Filter Levels
    "screener_rsi_filter_overbought": 60,  # Long: RSI ≤ 60
    "screener_rsi_filter_oversold": 40,    # Short: RSI ≥ 40
    
    # Filters ON/OFF (all ON by default)
    "screener_use_rsi_filter": True,
    "screener_use_mfi_filter": True,
    "screener_use_momentum_filter": True,
    "screener_use_cloud_filter": True,
    "screener_use_htf_signal_filter": True,
    "screener_use_last_signal_filter": True,
    
    # Auto Scan Settings
    "screener_auto_scan": False,            # Автоматичне сканування
    "screener_scan_interval": 15,           # Інтервал в хвилинах (5, 10, 15, 30, 60)

    # === STRATEGY FILTERS (Для сумісності, якщо знадобиться) ===
    "obt_useCloudFilter": True,
    "obt_useObvFilter": True,
    "obt_useRsiFilter": True,
    "obt_useOBRetest": False, 

    # === TIMEFRAMES ===
    "htfSelection": "240",
    "ltfSelection": "45",
    
    # === INDICATORS ===
    "obt_cloudFastLen": 10,
    "obt_cloudSlowLen": 40,
    "obt_rsiLength": 14,
    "obt_entryRsiOversold": 45,
    "obt_entryRsiOverbought": 55,
    "obt_obvEntryLen": 20,
    "obt_swingLength": 5,

    # === SMART EXIT & TRAILING ===
    "exit_enableStrategy": False,  # Світч для RSI-based exit стратегії
    "exit_ltf": "60",              # LTF для розрахунків виходу (60хв = стандарт Bybit)
    
    # ✨ TRAILING тепер активується АВТОМАТИЧНО після TP2!
    "trailing_enabled": False,      # Manual trailing (вимкнено для Smart TP)
    "trailing_rsi_activation": 65,  # Не використовується в Smart TP
    "trailing_atr_length": 14,      # Період ATR для trailing
    "trailing_atr_multiplier": 2.5, # Множник ATR (SL = Price ± ATR × mult)
    "trailing_activation_delay": 5, # Не використовується в Smart TP

    "exit_rsiOverbought": 70,
    "exit_rsiOversold": 30,
    "exit_obvLength": 10,

    # === RISK ===
    "riskPercent": 2.0,
    "leverage": 20,
    "use_tp": True,  # 🎯 Take Profit увімкнено за замовчуванням
    "tp_mode": "Smart_TP",  # ✨ НОВИЙ РЕЖИМ: 50/25/25 з auto-BE та Trailing
    "fixedTP": 3.0,
    "sl_mode": "OB_Extremity",
    "fixedSL": 1.5,
    "obBufferPercent": 0.2,

    # === SMART MONEY SIMULATOR ===
    "sm_entry_mode": "Market",
    "sm_sl_buffer": 0.2,
    "sm_tp_mode": "None",
    "sm_tp_value": 3.0,

    # === WHALE STRATEGY RSI FILTER ===
    "whale_rsi_filter_enabled": False,  # Вкл/Викл RSI фільтр
    "whale_rsi_min": 30,                # Шукати RSI <= цього (перепроданість)
    "whale_rsi_max": 70,                # Шукати RSI >= цього (перекупленість)
    
    # === ORDER BLOCK SCANNER (Smart Money) ===
    # Detection Settings
    "ob_source_tf": "15",               # Таймфрейм для пошуку OB
    "ob_swing_length": 3,               # Довжина свінга (min: 2)
    "ob_zone_count": "High",            # One(1), Low(3), Medium(5), High(10)
    "ob_max_atr_mult": 3.5,             # Макс. розмір OB в ATR
    "ob_invalidation_method": "Wick",   # Wick або Close
    "ob_combine_obs": True,             # Комбінувати перекриваючі OB
    
    # Entry Settings
    "ob_entry_mode": "Immediate",       # Immediate або Retest
    "ob_selection": "Newest",           # Newest або Closest
    "ob_persistence_check": False,      # Чекати 1 бар підтвердження
    "ob_sl_atr_mult": 0.3,              # ATR множник для SL
    
    # Automation
    "ob_auto_scan": False,              # Автоматичне сканування
    "ob_auto_add_from_screener": False, # Автоматично додавати з RSI/MFI Screener
    "ob_execute_trades": False,         # Відкривати угоди (за замовчуванням вимкнено!)
    "ob_scan_interval": 60,             # Інтервал сканування OB в секундах
    "ob_watchlist_timeout": "24h",      # No, 12h, 24h, 48h, 72h
    "ob_watchlist_limit": 50,           # Макс. монет в watchlist
    
    # === WHALE HUNTER PRO ===
    "whp_enabled": True,
    "whp_auto_mode": False,
    "whp_auto_interval": 60,
    "whp_min_score": 50,
    "whp_min_volume": 5000000,
    "whp_scan_limit": 50,
    "whp_main_tf": "60",
    "whp_htf": "240",
    
    # WHP Scoring Weights
    "whp_use_rsi": True,
    "whp_rsi_weight": 20,
    "whp_rsi_oversold": 40,
    "whp_rsi_overbought": 60,
    
    "whp_use_mfi": True,
    "whp_mfi_weight": 15,
    
    "whp_use_rvol": True,
    "whp_rvol_weight": 15,
    "whp_rvol_threshold": 1.5,
    
    "whp_use_ob": True,
    "whp_ob_weight": 20,
    "whp_ob_distance": 3.0,
    
    "whp_use_btc": True,                # BTC Trend Filter - за замовчуванням увімкнено
    "whp_btc_weight": 15,
    
    "whp_use_adx": True,
    "whp_adx_weight": 10,
    "whp_adx_threshold": 25,
    
    "whp_use_squeeze": True,
    "whp_squeeze_weight": 5,
    
    "whp_use_divergence": True,
    "whp_divergence_weight": 10,
    
    "whp_add_to_watchlist": True,
    
    # === CONFLUENCE SCALPER ===
    "cs_enabled": True,
    "cs_timeframe": "15",
    "cs_auto_preset": True,
    
    # Confluence Weights
    "cs_min_confluence": 72,
    "cs_weight_whale": 30,
    "cs_weight_ob": 30,
    "cs_weight_volume": 20,
    "cs_weight_trend": 20,
    
    # Filters
    "cs_use_btc_filter": True,
    "cs_use_volume_filter": True,
    "cs_use_volatility_filter": True,
    "cs_use_time_filter": False,
    "cs_use_correlation_filter": True,
    
    # Order Block
    "cs_ob_distance_max": 1.2,
    "cs_ob_swing_length": 3,
    "cs_entry_mode": "Retest",
    
    # Take Profit
    "cs_tp1_percent": 0.5,
    "cs_tp2_percent": 1.0,
    "cs_use_trailing": True,
    "cs_trailing_offset": 0.3,
    
    # Stop Loss
    "cs_sl_mode": "OB_Edge",
    "cs_sl_fixed_percent": 0.5,
    "cs_sl_atr_mult": 0.4,
    "cs_sl_buffer": 0.1,
    
    # Risk Management
    "cs_max_daily_trades": 3,
    "cs_max_open_positions": 2,
    "cs_max_same_direction": 2,
    "cs_position_size_percent": 5.0,
    "cs_max_daily_loss": 3.0,
    
    # Timing
    "cs_signal_expiry": 10,
    "cs_max_hold_time": 60,
    "cs_scan_interval": 30,
    
    # Execution
    "cs_paper_trading": True,
    "cs_auto_execute": False,
    "cs_telegram_signals": False,
    "cs_telegram_trades": False,
    
    # Analytics
    "cs_use_analytics": True,
    "cs_avoid_problem_symbols": True,
    "cs_adjust_on_losses": True,
    "cs_leverage": 10,
    
    # === RSI SNIPER PRO ===
    # RSI Settings
    "rsp_rsi_length": 14,
    "rsp_oversold": 30,
    "rsp_overbought": 70,
    "rsp_min_peak_strength": 2,
    
    # MFI Cloud Settings
    "rsp_mfi_length": 20,
    "rsp_fast_mfi_ema": 5,
    "rsp_slow_mfi_ema": 13,
    
    # Bollinger Bands
    "rsp_use_bb": True,
    "rsp_bb_length": 20,
    "rsp_bb_mult": 2.0,
    
    # Structure & Divergence
    "rsp_show_structure": True,
    "rsp_show_divergence": True,
    "rsp_pivot_left": 5,
    "rsp_pivot_right": 2,
    
    # Signal Types
    "rsp_enable_royal": True,
    "rsp_enable_sniper": True,
    "rsp_enable_divergence": True,
    "rsp_enable_flow": True,
    
    # Filters
    "rsp_require_volume": False,
    "rsp_trend_confirmation": False,
    "rsp_min_volume_24h": 10000000,
    "rsp_scan_limit": 50,
    
    # Timeframes
    "rsp_main_tf": "15",
    "rsp_htf": "60",
    "rsp_htf_auto": True,
    
    # Risk Management
    "rsp_max_daily_trades": 5,
    "rsp_max_open_positions": 2,
    "rsp_position_size_percent": 5.0,
    "rsp_leverage": 10,
    "rsp_max_daily_loss": 3.0,
    
    # Execution
    "rsp_paper_trading": True,
    "rsp_auto_execute": True,
    "rsp_close_on_opposite": True,
    "rsp_telegram_signals": False,
    
    # Auto Mode
    "rsp_auto_mode": True,
    "rsp_scan_interval": 1,
}

class SettingsManager:
    def __init__(self):
        self.db = db_manager
        self._cache = {}
        self.reload_settings()

    def _cast_value(self, key, value_str):
        # Обробка порожніх рядків - повертаємо default або None
        if value_str is None or value_str == '':
            return DEFAULT_SETTINGS.get(key, None)
        
        # Якщо ключ в DEFAULT_SETTINGS - конвертуємо по типу default
        if key in DEFAULT_SETTINGS:
            default_val = DEFAULT_SETTINGS[key]
            try:
                if isinstance(default_val, bool): 
                    return str(value_str).lower() in ['true', 'on', '1']
                elif isinstance(default_val, int): 
                    return int(float(value_str))
                elif isinstance(default_val, float): 
                    return float(value_str)
                else: 
                    return str(value_str)
            except: 
                return default_val
        
        # Для ключів НЕ в DEFAULT_SETTINGS - намагаємось вгадати тип
        # Конвертуємо "true"/"false" строки в boolean
        if isinstance(value_str, str):
            if value_str.lower() == 'true':
                return True
            elif value_str.lower() == 'false':
                return False
            # Спробувати конвертувати в число
            try:
                if '.' in value_str:
                    return float(value_str)
                return int(value_str)
            except ValueError:
                pass
        
        return value_str

    def reload_settings(self):
        session = self.db.get_session()
        try:
            db_settings = session.query(BotSetting).all()
            if not db_settings:
                logger.info("📋 No settings in DB, using defaults")
                self._cache = DEFAULT_SETTINGS.copy()
                for k, v in DEFAULT_SETTINGS.items():
                    val_str = "true" if v is True else "false" if v is False else str(v)
                    session.add(BotSetting(key=k, value=val_str))
                session.commit()
            else:
                loaded = {}
                db_keys = set()
                for s in db_settings: 
                    casted = self._cast_value(s.key, s.value)
                    loaded[s.key] = casted
                    db_keys.add(s.key)
                    # Логуємо whp_use_btc
                    if s.key == 'whp_use_btc':
                        logger.info(f"🔧 reload_settings: whp_use_btc db='{s.value}' -> cache={casted} (type={type(casted).__name__})")
                
                # Log important settings
                auto_scan = loaded.get('ob_auto_scan', False)
                auto_add = loaded.get('ob_auto_add_from_screener', False)
                logger.info(f"📋 Loaded from DB: auto_scan={auto_scan}, auto_add={auto_add}")
                
                missing_keys = set(DEFAULT_SETTINGS.keys()) - db_keys
                if missing_keys:
                    for k in missing_keys:
                        v = DEFAULT_SETTINGS[k]
                        val_str = "true" if v is True else "false" if v is False else str(v)
                        session.add(BotSetting(key=k, value=val_str))
                        loaded[k] = v
                    session.commit()

                merged = DEFAULT_SETTINGS.copy()
                merged.update(loaded)
                self._cache = merged
        except Exception as e:
            logger.error(f"Settings load error: {e}")
            self._cache = DEFAULT_SETTINGS.copy()
        finally: 
            session.close()

    def save_settings(self, new_settings_dict):
        session = self.db.get_session()
        try:
            logger.info(f"💾 Saving settings: {list(new_settings_dict.keys())}")
            
            for k, v in new_settings_dict.items():
                val_to_store = str(v)
                
                # Визначаємо чи це boolean значення
                is_bool_value = False
                
                # Перевіряємо по DEFAULT_SETTINGS
                if k in DEFAULT_SETTINGS:
                    default_type = type(DEFAULT_SETTINGS[k])
                    if default_type == bool:
                        is_bool_value = True
                
                # Також перевіряємо якщо значення вже boolean (з JSON)
                if isinstance(v, bool):
                    is_bool_value = True
                
                # Конвертуємо boolean
                if is_bool_value:
                    is_true = (v == 'on' or v == 'true' or v is True or str(v).lower() == 'true')
                    val_to_store = "true" if is_true else "false"
                    self._cache[k] = is_true
                    # Логуємо whp_use_btc для діагностики
                    if k == 'whp_use_btc':
                        logger.info(f"  🔧 {k}: input={v} (type={type(v).__name__}) -> cache={is_true}, db={val_to_store}")
                elif k in DEFAULT_SETTINGS:
                    self._cache[k] = self._cast_value(k, v)
                    val_to_store = str(v)
                else:
                    self._cache[k] = v
                
                existing = session.query(BotSetting).filter_by(key=k).first()
                if existing: 
                    existing.value = val_to_store
                else: 
                    session.add(BotSetting(key=k, value=val_to_store))
            
            session.commit()
            logger.info(f"✅ Settings saved successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"❌ Settings save error: {e}")
        finally: 
            session.close()

    def get_all(self): return self._cache.copy()
    def get(self, key, default=None): 
        return self._cache.get(key, default if default is not None else DEFAULT_SETTINGS.get(key))
    
    def import_settings(self, json_data):
        session = self.db.get_session()
        try:
            for k, v in json_data.items():
                val = str(v).lower() if isinstance(v, bool) else str(v)
                self._cache[k] = v
                ex = session.query(BotSetting).filter_by(key=k).first()
                if ex: ex.value = val
                else: session.add(BotSetting(key=k, value=val))
            session.commit(); return True
        except: return False
        finally: session.close()

settings = SettingsManager()
