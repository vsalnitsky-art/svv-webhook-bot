import logging
from models import db_manager, BotSetting
logger = logging.getLogger(__name__)

DEFAULT_SETTINGS = {
    "scanner_quote_coin": "USDT", "scanner_mode": "Manual", "scan_limit": 100,
    "telegram_enabled": False, "telegram_bot_token": "", "telegram_chat_id": "",
    "obt_useCloudFilter": True, "obt_useObvFilter": True, "obt_useRsiFilter": True, 
    "obt_useOBRetest": False, 
    "htfSelection": "240", "ltfSelection": "45",
    "obt_cloudFastLen": 10, "obt_cloudSlowLen": 40, "obt_rsiLength": 14,
    "obt_entryRsiOversold": 45, "obt_entryRsiOverbought": 55, "obt_obvEntryLen": 20, "obt_swingLength": 5,
    "exit_enableStrategy": True, "exit_rsiOverbought": 70, "exit_rsiOversold": 30, "exit_obvLength": 10,
    "riskPercent": 2.0, "leverage": 20, "tp_mode": "Fixed_1_50", "fixedTP": 3.0,
    "sl_mode": "OB_Extremity", "fixedSL": 1.5, "obBufferPercent": 0.2,
}

class SettingsManager:
    def __init__(self):
        self.db = db_manager; self._cache = {}; self.reload_settings()
    def _cast_value(self, key, value_str):
        if key not in DEFAULT_SETTINGS: return value_str
        default = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default, bool): return str(value_str).lower() in ['true', 'on', '1']
            elif isinstance(default, int): return int(float(value_str))
            elif isinstance(default, float): return float(value_str)
            else: return str(value_str)
        except: return default
    def reload_settings(self):
        session = self.db.get_session()
        try:
            db_s = session.query(BotSetting).all()
            if not db_s:
                self._cache = DEFAULT_SETTINGS.copy()
                for k, v in DEFAULT_SETTINGS.items():
                    val = "true" if v is True else "false" if v is False else str(v)
                    session.add(BotSetting(key=k, value=val))
                session.commit()
            else:
                loaded = {}
                db_keys = set()
                for s in db_s: loaded[s.key] = self._cast_value(s.key, s.value); db_keys.add(s.key)
                missing = set(DEFAULT_SETTINGS.keys()) - db_keys
                if missing:
                    for k in missing:
                        v = DEFAULT_SETTINGS[k]; val = "true" if v is True else "false" if v is False else str(v)
                        session.add(BotSetting(key=k, value=val)); loaded[k] = v
                    session.commit()
                self._cache = DEFAULT_SETTINGS.copy(); self._cache.update(loaded)
        except: self._cache = DEFAULT_SETTINGS.copy()
        finally: session.close()
    def save_settings(self, new_settings):
        session = self.db.get_session()
        try:
            for k, v in new_settings.items():
                val = str(v)
                if k in DEFAULT_SETTINGS:
                    if isinstance(DEFAULT_SETTINGS[k], bool):
                        val = "true" if (v=='on' or v is True) else "false"
                        self._cache[k] = (val == "true")
                    else:
                        self._cache[k] = self._cast_value(k, v); val = str(v)
                else: self._cache[k] = v
                ex = session.query(BotSetting).filter_by(key=k).first()
                if ex: ex.value = val
                else: session.add(BotSetting(key=k, value=val))
            session.commit()
        except: session.rollback()
        finally: session.close()
    def get_all(self): return self._cache.copy()
    def get(self, key, default=None): return self._cache.get(key, default if default is not None else DEFAULT_SETTINGS.get(key))
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
