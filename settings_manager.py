import logging
from models import db_manager, BotSetting
logger = logging.getLogger(__name__)
DEFAULT_SETTINGS = {
    "scanner_quote_coin": "USDT", "scanner_mode": "Manual", "scan_limit": 100,
    "telegram_enabled": False, "telegram_bot_token": "", "telegram_chat_id": "",
    "useCloudFilter": True, "useObvFilter": True, "useRsiFilter": True, "useMfiFilter": False, "useOBRetest": False,
    "htfSelection": "240", "ltfSelection": "15", "cloudFastLen": 10, "cloudSlowLen": 40,
    "entryRsiOversold": 45, "entryRsiOverbought": 55, "rsiLength": 14,
    "exitRsiOversold": 30, "exitRsiOverbought": 70, "mfiLength": 20, "obvEntryLen": 20, "obvExitLen": 20,
    "riskPercent": 2.0, "leverage": 20, "fixedTP": 3.0, "fixedSL": 1.5,
    "atrMultiplierSL": 1.5, "atrMultiplierTP": 3.0, "swingLength": 5, "volumeSpikeThreshold": 1.8, "tp_mode": "None"
}
class SettingsManager:
    def __init__(self):
        self.db = db_manager; self._cache = {}; self.reload_settings()
    def _cast_value(self, key, value_str):
        if key not in DEFAULT_SETTINGS: return value_str
        default = DEFAULT_SETTINGS[key]
        try:
            if isinstance(default, bool): return str(value_str).lower() == 'true'
            elif isinstance(default, int): return int(value_str)
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
                for s in db_s: loaded[s.key] = self._cast_value(s.key, s.value)
                self._cache = DEFAULT_SETTINGS.copy(); self._cache.update(loaded)
        except: self._cache = DEFAULT_SETTINGS.copy()
        finally: session.close()
    def save_settings(self, new_settings):
        session = self.db.get_session()
        try:
            for k, v in new_settings.items():
                if k in DEFAULT_SETTINGS:
                    val = v
                    if isinstance(DEFAULT_SETTINGS[k], bool): val = "true" if (v=='on' or v is True) else "false"
                    else: val = str(v)
                    self._cache[k] = self._cast_value(k, v)
                    ex = session.query(BotSetting).filter_by(key=k).first()
                    if ex: ex.value = val
                    else: session.add(BotSetting(key=k, value=val))
            session.commit()
        except: session.rollback()
        finally: session.close()
    def get(self, key): return self._cache.get(key, DEFAULT_SETTINGS.get(key))
settings = SettingsManager()
