import os

class Config:
    # Сервер
    PORT = int(os.environ.get("PORT", 10000))
    HOST = "0.0.0.0"
    
    # Торгівля
    DEFAULT_RISK_PERCENT = 5.0
    DEFAULT_LEVERAGE = 20
    DEFAULT_TP_PERCENT = 0.0
    DEFAULT_SL_PERCENT = 0.0
    
    # Сканер (тільки для моніторингу активних)
    SCANNER_INTERVAL = 5
    
    # Очищення
    DATA_RETENTION_DAYS = 30

    @classmethod
    def get_scanner_config(cls):
        return {'SCANNER_INTERVAL': cls.SCANNER_INTERVAL}

config = Config()