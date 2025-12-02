import os
class Config:
    PORT = int(os.environ.get("PORT", 10000))
    HOST = "0.0.0.0"
    SCANNER_INTERVAL = 5
    DATA_RETENTION_DAYS = 30
    @classmethod
    def get_scanner_config(cls): return {'SCANNER_INTERVAL': cls.SCANNER_INTERVAL}
config = Config()
