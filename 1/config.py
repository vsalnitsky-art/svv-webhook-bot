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
    
    # Спроба 2: Використати незашифровані ключи (fallback)
    api_key = os.environ.get('BYBIT_API_KEY')
    api_secret = os.environ.get('BYBIT_API_SECRET')
    
    if api_key and api_secret:
        return api_key, api_secret
    
    # Спроба 3: Локальна розробка через .env файл
    if not os.environ.get('RENDER'):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.environ.get('BYBIT_API_KEY')
            api_secret = os.environ.get('BYBIT_API_SECRET')
            
            if api_key and api_secret:
                print("✅ Loaded credentials from .env file")
                return api_key, api_secret
        except:
            pass
        
        raise ValueError(
            "❌ LOCAL MODE: API credentials not found!\n"
            "Please create .env file with:\n"
            "BYBIT_API_KEY=your_test_key\n"
            "BYBIT_API_SECRET=your_test_secret"
        )
    
    # Якщо нічого не спрацювало на продакшні - помилка
    raise ValueError(
        "❌ API credentials not found!\n\n"
        "Please set BYBIT_API_KEY and BYBIT_API_SECRET in Render Environment Variables."
    )

# Допоміжні функції для генерації ключів (можна викликати локально)
def generate_encryption_key():
    """Генерує новий ключ Fernet для шифрування"""
    key = Fernet.generate_key()
    print(f"🔑 New encryption key: {key.decode()}")
    print("⚠️ Save this key securely in ENCRYPTION_KEY environment variable")
    return key

# Для зворотної сумісності
API_KEY, API_SECRET = get_api_credentials()
