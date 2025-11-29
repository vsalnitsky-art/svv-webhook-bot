import logging
from models import db_manager, BotSetting

logger = logging.getLogger(__name__)

# Повні налаштування за замовчуванням (використовуються як еталон типів)
DEFAULT_SETTINGS = {
    # --- CONTROL PANEL ---
    "useCloudFilter": True,
    "useObvFilter": True,
    "useRsiFilter": True,
    "useMfiFilter": False,
    
    # --- GLOBAL ---
    "htfSelection": "240",
    "cloudFastLen": 10,
    "cloudSlowLen": 40,
    
    # --- RSI SETTINGS ---
    "entryRsiOversold": 45,
    "entryRsiOverbought": 55,
    "rsiLength": 14,
    "exitRsiOversold": 30,
    "exitRsiOverbought": 70,
    
    # --- MFI SETTINGS ---
    "mfiLength": 20,
    
    # --- OBV SETTINGS ---
    "obvEntryLen": 20,
    "obvExitLen": 20,
    
    # --- RISK MANAGEMENT ---
    "riskPercent": 2.0,
    "leverage": 20,
    "fixedTP": 3.0,
    "fixedSL": 1.5,
    "atrMultiplierSL": 1.5,
    "atrMultiplierTP": 3.0,
    
    # --- ORDER BLOCKS ---
    "swingLength": 5,
    "volumeSpikeThreshold": 1.8
}

class SettingsManager:
    def __init__(self):
        self.db = db_manager
        # Кеш налаштувань, щоб не дьоргати базу при кожному тику
        self._cache = {}
        self.reload_settings()

    def _cast_value(self, key, value_str):
        """Конвертує рядок з БД у правильний тип на основі DEFAULT_SETTINGS"""
        if key not in DEFAULT_SETTINGS:
            return value_str # Невідомий параметр, повертаємо як є
            
        default_val = DEFAULT_SETTINGS[key]
        
        try:
            if isinstance(default_val, bool):
                return str(value_str).lower() == 'true'
            elif isinstance(default_val, int):
                return int(value_str)
            elif isinstance(default_val, float):
                return float(value_str)
            else:
                return str(value_str)
        except:
            return default_val

    def reload_settings(self):
        """Завантажує налаштування з БД у кеш. Якщо БД порожня - ініціалізує її."""
        session = self.db.get_session()
        try:
            db_settings = session.query(BotSetting).all()
            
            # Якщо база порожня - записуємо дефолтні
            if not db_settings:
                logger.info("Settings DB is empty. Seeding defaults...")
                self._cache = DEFAULT_SETTINGS.copy()
                for k, v in DEFAULT_SETTINGS.items():
                    val_str = str(v).lower() if isinstance(v, bool) else str(v)
                    session.add(BotSetting(key=k, value=val_str))
                session.commit()
            else:
                # Завантажуємо з бази
                loaded = {}
                for s in db_settings:
                    loaded[s.key] = self._cast_value(s.key, s.value)
                
                # Додаємо нові ключі з DEFAULT, якщо їх немає в базі (міграція на льоту)
                self._cache = DEFAULT_SETTINGS.copy()
                self._cache.update(loaded)
                
        except Exception as e:
            logger.error(f"Error loading settings from DB: {e}")
            self._cache = DEFAULT_SETTINGS.copy()
        finally:
            session.close()

    def save_settings(self, new_settings_dict):
        """Зберігає нові налаштування в БД та оновлює кеш"""
        session = self.db.get_session()
        try:
            # 1. Підготовка даних (обробка чекбоксів та типів)
            to_save = {}
            for key, default_val in DEFAULT_SETTINGS.items():
                if key in new_settings_dict:
                    raw_val = new_settings_dict[key]
                    
                    # Конвертація для збереження
                    if isinstance(default_val, bool):
                        # Чекбокси HTML: якщо є в dict - то True, якщо 'on' - то True
                        is_true = (raw_val == 'on' or raw_val is True or str(raw_val).lower() == 'true')
                        val_to_store = "true" if is_true else "false"
                        self._cache[key] = is_true
                    else:
                        val_to_store = str(raw_val)
                        # Оновлюємо кеш типізованим значенням
                        self._cache[key] = self._cast_value(key, raw_val)
                    
                    to_save[key] = val_to_store
                
                elif isinstance(default_val, bool):
                    # Якщо чекбокс не прийшов у формі, значить він вимкнений (False)
                    self._cache[key] = False
                    to_save[key] = "false"

            # 2. Upsert в базу (оновлення або вставка)
            for k, v in to_save.items():
                existing = session.query(BotSetting).filter_by(key=k).first()
                if existing:
                    existing.value = v
                else:
                    session.add(BotSetting(key=k, value=v))
            
            session.commit()
            logger.info("Settings saved to DB successfully")
            
        except Exception as e:
            logger.error(f"Error saving settings to DB: {e}")
            session.rollback()
        finally:
            session.close()

    def get(self, key):
        """Отримати значення параметра (з кешу)"""
        return self._cache.get(key, DEFAULT_SETTINGS.get(key))

# Створюємо глобальний екземпляр
settings = SettingsManager()