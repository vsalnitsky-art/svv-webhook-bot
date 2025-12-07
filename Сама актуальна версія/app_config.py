#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Configuration Module
Сцентралізована конфігурація з валідацією Pydantic
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from cryptography.fernet import Fernet
import logging

logger = logging.getLogger(__name__)

class AppConfig(BaseSettings):
    """Основна конфігурація додатку"""
    
    # === FLASK & DEPLOYMENT ===
    PORT: int = Field(default=10000, env='PORT')
    HOST: str = Field(default='0.0.0.0')
    ENV: str = Field(default='production', env='ENV')
    DEBUG: bool = Field(default=False)
    
    # 🔐 FLASK SECRET KEY (КРИТИЧНЕ - з ENV, ніколи не в коді!)
    FLASK_SECRET_KEY: str = Field(env='FLASK_SECRET_KEY', default='')
    
    # === BYBIT API ===
    BYBIT_API_KEY: str = Field(env='BYBIT_API_KEY', default='')
    BYBIT_API_SECRET: str = Field(env='BYBIT_API_SECRET', default='')
    BYBIT_TESTNET: bool = Field(default=False, env='BYBIT_TESTNET')
    
    # === ENCRYPTION ===
    ENCRYPTION_KEY: Optional[str] = Field(default=None, env='ENCRYPTION_KEY')
    BYBIT_API_KEY_ENCRYPTED: Optional[str] = Field(default=None, env='BYBIT_API_KEY_ENCRYPTED')
    BYBIT_API_SECRET_ENCRYPTED: Optional[str] = Field(default=None, env='BYBIT_API_SECRET_ENCRYPTED')
    
    # === DATABASE ===
    DATABASE_URL: Optional[str] = Field(default=None, env='DATABASE_URL')
    
    # === TRADING ===
    DEFAULT_RISK_PERCENT: float = Field(default=2.0)
    DEFAULT_LEVERAGE: int = Field(default=20)
    MIN_BALANCE: float = Field(default=5.0)
    
    # === API RATE LIMITS ===
    API_TIMEOUT: int = Field(default=10)
    MAX_RETRIES: int = Field(default=3)
    RETRY_BACKOFF_FACTOR: float = Field(default=1.5)
    
    # === MONITORING ===
    SCANNER_INTERVAL: int = Field(default=5)  # сек
    TRADES_SYNC_INTERVAL: int = Field(default=1800)  # 30 хвилин
    DATA_RETENTION_DAYS: int = Field(default=30)
    
    # === LOGGING ===
    LOG_LEVEL: str = Field(default='INFO')
    LOG_FORMAT: str = Field(default='json')  # 'json' або 'text'
    
    class Config:
        env_file = '.env'
        case_sensitive = True
        extra = 'allow'
    
    @validator('FLASK_SECRET_KEY', pre=True, always=True)
    def validate_secret_key(cls, v):
        """Перевіряє, що secret key встановлений і достатньо довгий"""
        if not v:
            # ⚠️ РОЗРОБКА: генеруємо тимчасовий
            if os.environ.get('ENV') == 'development':
                logger.warning("⚠️ DEVELOPMENT: Using temporary secret key. Set FLASK_SECRET_KEY in .env!")
                return 'dev-temp-key-change-me'
            else:
                # ПРОДАКШН: помилка!
                raise ValueError(
                    "❌ FLASK_SECRET_KEY not set! This is REQUIRED for security.\n"
                    "Set it in environment variables or .env file:\n"
                    "FLASK_SECRET_KEY=<generate with: python -c 'import secrets; print(secrets.token_hex(32))'>"
                )
        if len(v) < 32:
            logger.warning(f"⚠️ FLASK_SECRET_KEY too short ({len(v)} chars). Recommend 32+ chars")
        return v
    
    @validator('DATABASE_URL', pre=True, always=True)
    def normalize_db_url(cls, v):
        """Нормалізує URL БД для PostgreSQL"""
        if v and v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql://", 1)
        return v
    
    def get_api_credentials(self) -> tuple[str, str]:
        """Отримує API ключі (зашифровані або ні)"""
        
        # Спроба 1: Зашифровані ключі
        if self.ENCRYPTION_KEY and self.BYBIT_API_KEY_ENCRYPTED and self.BYBIT_API_SECRET_ENCRYPTED:
            try:
                cipher = Fernet(self.ENCRYPTION_KEY.encode())
                api_key = cipher.decrypt(self.BYBIT_API_KEY_ENCRYPTED.encode()).decode()
                api_secret = cipher.decrypt(self.BYBIT_API_SECRET_ENCRYPTED.encode()).decode()
                logger.info("✅ Loaded encrypted API credentials")
                return api_key, api_secret
            except Exception as e:
                logger.warning(f"⚠️ Failed to decrypt: {e}. Trying plain credentials...")
        
        # Спроба 2: Звичайні ключі
        if self.BYBIT_API_KEY and self.BYBIT_API_SECRET:
            logger.info("✅ Loaded plain API credentials")
            return self.BYBIT_API_KEY, self.BYBIT_API_SECRET
        
        # Помилка
        raise ValueError(
            "❌ API credentials not found!\n"
            "Please set BYBIT_API_KEY and BYBIT_API_SECRET in environment or .env"
        )
    
    @staticmethod
    def generate_encryption_key() -> str:
        """Генерує новий Fernet ключ для шифрування"""
        key = Fernet.generate_key()
        key_str = key.decode()
        print(f"🔑 New encryption key: {key_str}")
        print("⚠️ Save this to ENCRYPTION_KEY environment variable")
        return key_str

# Глобальний інстанс конфігу
config = AppConfig()

# === ПЕРЕНЕСЕННЯ СТАРИХ ФУНКЦІЙ ДЛЯ СУМІСНОСТІ ===
def get_api_credentials():
    """Заради сумісності з старим кодом"""
    return config.get_api_credentials()

API_KEY, API_SECRET = get_api_credentials()
