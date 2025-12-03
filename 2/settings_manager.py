import logging
from models import db_manager, BotSetting

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    # === GENERAL ===
    "scanner_quote_coin": "USDT",
    "scanner_mode": "Manual",
    "scan_limit": 100,
    
    # === TELEGRAM ===
    "telegram_enabled": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",

    # === STRATEGY FILTERS ===
    "obt_useCloudFilter": True,
    "obt_useObvFilter": True,
    "obt_useRsiFilter": True,
    "obt_useOBRetest": False, # <--- ЗМІНЕНО НА FALSE (Вимкнено за замовчуванням)

    # === TIMEFRAMES ===
    "htfSelection": "240", # 4H (Default HTF)
    "ltfSelection": "45",  # 45m (Default LTF)
    
    # === INDICATORS (DEFAULT FOR 4H) ===
    "obt_cloudFastLen": 10,
    "obt_cloudSlowLen": 40,
    "obt_rsiLength": 14,
    "obt_entryRsiOversold": 45,
    "obt_entryRsiOverbought": 55,
    "obt_obvEntryLen": 20,
    "obt_swingLength": 5,

    # === SMART EXIT (ENABLED BY DEFAULT) ===
    "exit_enableStrategy": True,
    "exit_rsiOverbought": 70,
    "exit_rsiOversold": 30,
    "exit_obvLength": 10,

    # === RISK ===
    "riskPercent": 2.0,
    "leverage": 20,
    "tp_mode": "Fixed_1_50", 
    "fixedTP": 3.0,
    "sl_mode": "OB_Extremity",
    "fixedSL": 1.5,
    "obBufferPercent": 0.2,
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
