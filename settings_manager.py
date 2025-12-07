#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from models import db_manager, BotSetting
from datetime import datetime
from typing import Any

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

    # === STRATEGY FILTERS (Для сумісності, якщо знадобиться) ===
    "obt_useCloudFilter": True,
    "obt_useObvFilter": True,
    "obt_useRsiFilter": True,
    "obt_useOBRetest": False, 

    # === TIMEFRAMES ===
    "htfSelection": "60", # ЗМІНЕНО: 1 Година (1H)
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
    "exit_enableStrategy": False,  # Світч для RSI-виходу
    "exit_rsiLength": 10,
    "exit_rsiOverbought": 80,
    "exit_rsiOversold": 20,
    "exit_enableTrailing": True,
    "exit_trailingMode": "ATR",  # ATR | Percent
    "exit_trailingAtrPeriod": 10,
    "exit_trailingAtrMultiplier": 1.5,
    "exit_trailingPercent": 0.5,
    
    # === WHALE CORE ===
    "whale_tf": "60",
    "whale_limit": 50,
    "whale_min_vol": 5, # в мільйонах
    "whale_bb_max": 0.15,
    "whale_ema_period": 200,
    
    # === PAPER TRADING ===
    "paper_mode_enabled": False,
    "paper_balance": 10000.0
}

# Мапінг для коректного приведення типів
TYPE_MAP = {
    "scan_limit": int, "scan_min_volume": float, "whale_limit": int,
    "obt_cloudFastLen": int, "obt_cloudSlowLen": int, "obt_rsiLength": int,
    "obt_entryRsiOversold": float, "obt_entryRsiOverbought": float,
    "obt_obvEntryLen": int, "obt_swingLength": int, "exit_rsiLength": int,
    "exit_rsiOverbought": float, "exit_rsiOversold": float,
    "exit_trailingAtrPeriod": int, "exit_trailingAtrMultiplier": float,
    "exit_trailingPercent": float, "whale_min_vol": float,
    "whale_bb_max": float, "whale_ema_period": int, "paper_balance": float
}


# SQLAlchemy Модель
# (Має бути імпортована з models, тут лише використання)
# class BotSetting(Base): ...

class SettingsManager:
    def __init__(self):
        self.db = db_manager
        self.db.setup() # Переконатися, що база даних готова
        self._cache = {}
        self.load_settings()
        self.load_missing_defaults()
        
    def _cast_value(self, key: str, value: Any):
        """Приводить значення до коректного типу згідно TYPE_MAP або булевого"""
        if isinstance(value, bool):
            return value
            
        type_func = TYPE_MAP.get(key)
        
        # Перевірка на булеві значення, які можуть бути рядками "true" / "false"
        if key.startswith(('telegram_enabled', 'scan_use_min_volume', 'obt_use', 'exit_enable', 'paper_mode_enabled')):
            v = str(value).lower()
            return v == 'true' or v == 'on' or v == '1'
            
        if type_func:
            try:
                return type_func(value)
            except (ValueError, TypeError):
                # Повертаємо дефолтне значення, якщо конвертація не вдалася
                default_val = DEFAULT_SETTINGS.get(key)
                if default_val is not None:
                    try: return type_func(default_val)
                    except: return default_val
                return value
        return value

    def load_settings(self):
        """Завантажує налаштування з БД в кеш."""
        session = self.db.get_session()
        try:
            db_settings = session.query(BotSetting).all()
            for setting in db_settings:
                self._cache[setting.key] = self._cast_value(setting.key, setting.value)
            logger.info("✅ Settings loaded from DB")
        except Exception as e:
            logger.error(f"Settings load error: {e}")
        finally: session.close()

    def load_missing_defaults(self):
        """Додає дефолтні значення для відсутніх ключів у кеш."""
        changes = False
        for key, default_value in DEFAULT_SETTINGS.items():
            if key not in self._cache:
                self._cache[key] = default_value
                changes = True
        if changes:
             logger.info("⚙️ Default settings loaded for missing keys")
             # Не зберігаємо в БД, оскільки вони будуть збережені при першій зміні
             
    def save_settings(self, new_settings: dict):
        """Зберігає оновлені налаштування в БД і кеш."""
        session = self.db.get_session()
        try:
            for k, v in new_settings.items():
                if k not in DEFAULT_SETTINGS: continue # Ігноруємо невідомі ключі
                
                val_to_store = str(v)
                
                # Обробка булевих значень
                if k.startswith(('telegram_enabled', 'scan_use_min_volume', 'obt_use', 'exit_enable', 'paper_mode_enabled')):
                    is_true = (str(v).lower() == 'on' or str(v).lower() == 'true' or v is True)
                    val_to_store = "true" if is_true else "false"
                    self._cache[k] = is_true
                else:
                    self._cache[k] = self._cast_value(k, v)
                    val_to_store = str(v)
                
                existing = session.query(BotSetting).filter_by(key=k).first()
                if existing: existing.value = val_to_store
                else: session.add(BotSetting(key=k, value=val_to_store))
            session.commit()
            logger.info("✅ Settings saved to DB", keys=list(new_settings.keys()))
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
                # Перевірка на існування ключа в DEFAULT_SETTINGS
                if k not in DEFAULT_SETTINGS:
                    logger.warning(f"Skipping unknown setting key: {k}")
                    continue
                    
                # Приведення булевих значень
                if k.startswith(('telegram_enabled', 'scan_use_min_volume', 'obt_use', 'exit_enable', 'paper_mode_enabled')):
                    val = "true" if v else "false"
                else:
                    val = str(v)

                self._cache[k] = self._cast_value(k, v) # Оновлення кешу
                
                ex = session.query(BotSetting).filter_by(key=k).first()
                if ex: ex.value = val
                else: session.add(BotSetting(key=k, value=val))
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Import settings error: {e}")
            return False
        finally:
            session.close()

settings = SettingsManager()
