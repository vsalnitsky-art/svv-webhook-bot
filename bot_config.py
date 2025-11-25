"""
Configuration - Всі налаштування бота
"""

import os

class Config:
    """Централізована конфігурація"""
    
    # === SERVER ===
    PORT = int(os.environ.get("PORT", 10000))
    HOST = "0.0.0.0"
    DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
    
    # === DATABASE ===
    DATABASE_PATH = os.environ.get("DATABASE_PATH", "trading_bot.db")
    
    # === MONITORING ===
    MONITOR_INTERVAL = 5  # Секунд між перевірками позицій
    KEEP_ALIVE_INTERVAL = 600  # 10 хвилин
    
    # === SCANNER SETTINGS ===
    SCANNER_INTERVAL = 60  # Сканування ринку кожні 60 сек
    VOLUME_SPIKE_THRESHOLD = 2.0  # Об'єм має бути в 2x більше середнього
    MIN_INFLOW_VALUE = 1000000  # Мінімальне вливання: $1,000,000
    MIN_24H_VOLUME = 10000000  # Мінімальний добовий об'єм: $10,000,000
    
    # === TRADING ===
    DEFAULT_RISK_PERCENT = 5.0
    DEFAULT_LEVERAGE = 20
    DEFAULT_TP_PERCENT = 0.0
    DEFAULT_SL_PERCENT = 0.0
    
    # === EMAIL (Optional) ===
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
    EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")
    EMAIL_ENABLED = bool(EMAIL_SENDER and EMAIL_PASSWORD)
    
    # === CLEANUP ===
    DATA_RETENTION_DAYS = 30  # Зберігати дані за 30 днів
    
    @classmethod
    def get_scanner_config(cls):
        """Отримати конфіг для сканера"""
        return {
            'SCANNER_INTERVAL': cls.SCANNER_INTERVAL,
            'VOLUME_SPIKE_THRESHOLD': cls.VOLUME_SPIKE_THRESHOLD,
            'MIN_INFLOW_VALUE': cls.MIN_INFLOW_VALUE,
            'MIN_24H_VOLUME': cls.MIN_24H_VOLUME
        }
    
    @classmethod
    def get_trading_config(cls):
        """Отримати конфіг для торгівлі"""
        return {
            'DEFAULT_RISK_PERCENT': cls.DEFAULT_RISK_PERCENT,
            'DEFAULT_LEVERAGE': cls.DEFAULT_LEVERAGE,
            'DEFAULT_TP_PERCENT': cls.DEFAULT_TP_PERCENT,
            'DEFAULT_SL_PERCENT': cls.DEFAULT_SL_PERCENT
        }

# Експорт для зручності
config = Config()
