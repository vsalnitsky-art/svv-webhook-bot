import os

# Безопасное получение ключей из переменных окружения
API_KEY = os.environ.get('BYBIT_API_KEY')
API_SECRET = os.environ.get('BYBIT_API_SECRET')

if not API_KEY or not API_SECRET:
    raise ValueError("API keys not found in environment variables")

# Настройки торговли
DEFAULT_ACCOUNT_BALANCE = 1000
DEFAULT_LEVERAGE = 10
DEFAULT_RISK_PERCENT = 2.0
TESTNET_MODE = False  # Используем Mainnet
