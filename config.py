import os
from cryptography.fernet import Fernet

# ==================== НАСТРОЙКИ БЕЗОПАСНОСТИ ====================

# 🔐 ВАШ КЛЮЧ ШИФРОВАНИЯ (вставьте ключ из шага 1)
ENCRYPTION_KEY = "4r7S7ml99enLCvwCZHVrExMnHpCWg6yAOVHBbwVOtXo="

# 🔒 ЗАШИФРОВАННЫЕ API КЛЮЧИ (вставьте зашифрованные ключи из шага 2)
API_KEY_ENCRYPTED = "gAAAAABpGg-awgI-pWHhwD0h3heGJ88HwbJcRh_PAJSDOAcLzOKqtigXC12ofeNkaNtMMI9deMOE2fD7JGaYXHa_DyulzAbce_SkRV-fr8qBr-lkKnscK3Y="
SECRET_ENCRYPTED = "gAAAAABpGg-a1cAYymalpGeRw4szXZVGJ5yKphIVgR_T12o01R0thECVGJSrcDFUCZqyo38wwhQISaBUvVD7cCC_Z3eTvXpOS5Va4iuE-IUJRbjah8ZDpyD8APOfjMtmt97jHy6EiQdO"

# ==================== ФУНКЦИИ ШИФРОВАНИЯ ====================

def decrypt_data(encrypted_data):
    """Расшифровка данных"""
    try:
        fernet = Fernet(ENCRYPTION_KEY)
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        raise ValueError(f"Ошибка расшифровки: {e}")

def get_api_credentials():
    """Безопасное получение API ключей"""
    try:
        api_key = decrypt_data(API_KEY_ENCRYPTED)
        api_secret = decrypt_data(SECRET_ENCRYPTED)
        return api_key, api_secret
    except Exception as e:
        raise ValueError(f"Не удалось получить API ключи: {e}")

# ==================== НАСТРОЙКИ ТОРГОВЛИ ====================

# Настройки по умолчанию
DEFAULT_LEVERAGE = 5
DEFAULT_RISK_PERCENT = 1.0

# ==================== ПРОВЕРКА НАСТРОЕК ====================

def check_config():
    """Проверка корректности настроек"""
    try:
        api_key, api_secret = get_api_credentials()
        print("✅ Конфигурация корректна")
        return True
    except Exception as e:
        print(f"❌ Ошибка конфигурации: {e}")
        return False

# Проверяем при импорте
if __name__ == "__main__":
    check_config()
