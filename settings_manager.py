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
    "ob_watchlist_limit": 50            # Макс. монет в watchlist
}

class SettingsManager:
    def __init__(self):
        self.db = db_manager
        self._cache = {}
        self.reload_settings()

    def _cast_value(self, key, value_str):
        if key not in DEFAULT_SETTINGS: 
            return value_str
        default_val = DEFAULT_SETTINGS[key]
        
        # Обробка порожніх рядків - повертаємо default
        if value_str is None or value_str == '':
            return default_val
            
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

    def reload_settings(self):
        session = self.db.get_session()
        try:
            db_settings = session.query(BotSetting).all()
            if not db_settings:
                self._cache = DEFAULT_SETTINGS.copy()
                for k, v in DEFAULT_SETTINGS.items():
                    val_str = "true" if v is True else "false" if v is False else str(v)
                    session.add(BotSetting(key=k, value=val_str))
                session.commit()
            else:
                loaded = {}
                db_keys = set()
                for s in db_settings: 
                    loaded[s.key] = self._cast_value(s.key, s.value)
                    db_keys.add(s.key)
                
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
            for k, v in new_settings_dict.items():
                val_to_store = str(v)
                if k in DEFAULT_SETTINGS:
                    default_type = type(DEFAULT_SETTINGS[k])
                    if default_type == bool:
                        is_true = (v == 'on' or v == 'true' or v is True)
                        val_to_store = "true" if is_true else "false"
                        self._cache[k] = is_true
                    else:
                        self._cache[k] = self._cast_value(k, v)
                        val_to_store = str(v)
                else:
                    self._cache[k] = v
                
                existing = session.query(BotSetting).filter_by(key=k).first()
                if existing: existing.value = val_to_store
                else: session.add(BotSetting(key=k, value=val_to_store))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Settings save error: {e}")
        finally: session.close()

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
