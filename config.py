"""
Config with Fernet Encryption for Render Deployment
Підтримує шифрування API ключів через Environment Variables
"""

import os
from cryptography.fernet import Fernet

def get_api_credentials():
    """
    Отримати API credentials з зашифрованих Environment Variables
    """
    
    # Отримати ключ шифрування
    encryption_key = os.environ.get('ENCRYPTION_KEY')
    
    # Спроба 1: Використати зашифровані ключі
    if encryption_key:
        try:
            cipher = Fernet(encryption_key.encode())
            
            encrypted_api_key = os.environ.get('BYBIT_API_KEY_ENCRYPTED')
            encrypted_api_secret = os.environ.get('BYBIT_API_SECRET_ENCRYPTED')
            
            if encrypted_api_key and encrypted_api_secret:
                api_key = cipher.decrypt(encrypted_api_key.encode()).decode()
                api_secret = cipher.decrypt(encrypted_api_secret.encode()).decode()
                return api_key, api_secret
        except Exception as e:
            print(f"⚠️ Decryption failed: {e}")
            print("Trying plain text credentials...")
    
    # Спроба 2: Використати незашифровані ключі (fallback)
    api_key = os.environ.get('BYBIT_API_KEY')
    api_secret = os.environ.get('BYBIT_API_SECRET')
    
    if api_key and api_secret:
        return api_key, api_secret
    
    # Спроба 3: Локальна розробка (hardcoded) - ТІЛЬКИ ДЛЯ ТЕСТУВАННЯ!
    if not os.environ.get('RENDER'):
        print("⚠️ Running in local mode with hardcoded credentials")
        # ⚠️ ЗАМІНІТЬ НА ВАШІ РЕАЛЬНІ КЛЮЧІ ДЛЯ ЛОКАЛЬНОГО ТЕСТУВАННЯ
        api_key = "YOUR_API_KEY_HERE"
        api_secret = "YOUR_API_SECRET_HERE"
        return api_key, api_secret
    
    # Якщо нічого не спрацювало на продакшні - помилка
    raise ValueError(
        "❌ API credentials not found!\n\n"
        "Please set BYBIT_API_KEY and BYBIT_API_SECRET in Render Environment Variables."
    )

# Допоміжні функції для генерації ключів (можна викликати локально)
def generate_encryption_key():
    key = Fernet.generate_key()
    print(key.decode())
    return key

# Для зворотної сумісності
API_KEY, API_SECRET = get_api_credentials()
