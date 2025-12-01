import logging
from models import db_manager, BotSetting

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    # === GENERAL SETTINGS (Глобальні) ===
    "scanner_quote_coin": "USDT",
    "scanner_mode": "Manual",
    "scan_limit": 100,
    
    # === TELEGRAM ===
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",

    # === OLD STRATEGY SETTINGS (Залишаємо для сумісності зі старим сканером) ===
    "useCloudFilter": True, "useObvFilter": True, "useRsiFilter": True, "useMfiFilter": False, "useOBRetest": False, 
    "htfSelection": "240", "ltfSelection": "15",
    "cloudFastLen": 10, "cloudSlowLen": 40,
    "entryRsiOversold": 45, "entryRsiOverbought": 55, "rsiLength": 14,
    "exitRsiOversold": 30, "exitRsiOverbought": 70,
    "mfiLength": 20, "obvEntryLen": 20, "obvExitLen": 20,
    "riskPercent": 2.0, "leverage": 20, "fixedTP": 3.0, "fixedSL": 1.5,
    "atrMultiplierSL": 1.5, "atrMultiplierTP": 3.0,
    "swingLength": 5, "volumeSpikeThreshold": 1.8, "tp_mode": "None",

    # ======================================================
    # === NEW: OB + TREND AUTONOMOUS MODULE SETTINGS ===
    # ======================================================
    
    # 1. Фільтри
    "obt_useCloudFilter": True,
    "obt_useObvFilter": True,
    "obt_useRsiFilter": True,
    "obt_useBtcDominance": True,
    
    # 2. Технічні параметри
    "obt_cloudFastLen": 10,
    "obt_cloudSlowLen": 40,
    "obt_rsiLength": 14,
    "obt_entryRsiOversold": 40,
    "obt_entryRsiOverbought": 60,
    "obt_obvEntryLen": 20,
    "obt_obvEntryType": "SMA",
    "obt_swingLength": 5,
    "obt_maxATRMult": 3.0,
    "obt_obBufferPercent": 0.1,

    # 3. Ризик та Виконання (Окремо для цього модуля)
    "obt_riskPercent": 2.0,      # Власний ризик
    "obt_leverage": 20,          # Власне плече
    "obt_fixedTP": 3.0,          # Власний TP
    "obt_fixedSL": 1.5,          # Власний SL
    "obt_tp_mode": "Fixed",      # Режим TP: "None", "Fixed", "Ladder"
    "obt_sl_mode": "OB_Level"    # Режим SL: "Fixed", "OB_Level" (під блоком)
}

class SettingsManager:
    def __init__(self):
        self.db = db_manager
        self._cache = {}
        self.reload_settings()

    def _cast_value(self, key, value_str):
        if key not in DEFAULT_SETTINGS: return value_str
        default_val = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default_val, bool): return str(value_str).lower() == 'true'
            elif isinstance(default_val, int): return int(value_str)
            elif isinstance(default_val, float): return float(value_str)
            else: return str(value_str)
        except: return default_val

    def reload_settings(self):
        session = self.db.get_session()
        try:
            db_settings = session.query(BotSetting).all()
            if not db_settings:
                logger.info("Seeding default settings...")
                self._cache = DEFAULT_SETTINGS.copy()
                for k, v in DEFAULT_SETTINGS.items():
                    val_str = "true" if v is True else "false" if v is False else str(v)
                    session.add(BotSetting(key=k, value=val_str))
                session.commit()
            else:
                loaded = {}
                for s in db_settings: loaded[s.key] = self._cast_value(s.key, s.value)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(loaded)
                self._cache = merged
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            self._cache = DEFAULT_SETTINGS.copy()
        finally: session.close()

    def save_settings(self, new_settings_dict):
        session = self.db.get_session()
        try:
            for k, v in new_settings_dict.items():
                if k in DEFAULT_SETTINGS:
                    val_to_store = v
                    if isinstance(DEFAULT_SETTINGS[k], bool):
                        val_to_store = "true" if v == 'on' or v is True else "false"
                        self._cache[k] = (val_to_store == "true")
                    else:
                        val_to_store = str(v)
                        self._cache[k] = self._cast_value(k, v)
                    
                    existing = session.query(BotSetting).filter_by(key=k).first()
                    if existing: existing.value = val_to_store
                    else: session.add(BotSetting(key=k, value=val_to_store))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Save error: {e}")
        finally: session.close()
        
    def import_settings(self, json_data):
        session = self.db.get_session()
        try:
            for k, v in json_data.items():
                if k in DEFAULT_SETTINGS:
                    val_to_store = str(v).lower() if isinstance(v, bool) else str(v)
                    self._cache[k] = v
                    existing = session.query(BotSetting).filter_by(key=k).first()
                    if existing: existing.value = val_to_store
                    else: session.add(BotSetting(key=k, value=val_to_store))
            session.commit()
            return True
        except: return False
        finally: session.close()

    def get_all(self): return self._cache.copy()
    def get(self, key): return self._cache.get(key, DEFAULT_SETTINGS.get(key))

settings = SettingsManager()