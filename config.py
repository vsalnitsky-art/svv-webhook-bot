"""
Config with Fernet Encryption for Render Deployment
Підтримує шифрування API ключів через Environment Variables
"""

import os
from cryptography.fernet import Fernet

def get_api_credentials():
    """
    Отримати API credentials з зашифрованих Environment Variables
    
    На Render встановіть в Settings -> Environment:
    1. ENCRYPTION_KEY - ваш Fernet ключ (32 байти base64)
    2. BYBIT_API_KEY_ENCRYPTED - зашифрований API key
    3. BYBIT_API_SECRET_ENCRYPTED - зашифрований API secret
    
    АБО використайте незашифровані (для швидкого тесту):
    BYBIT_API_KEY - ваш API key (plain text)
    BYBIT_API_SECRET - ваш API secret (plain text)
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
        "For Render deployment, set Environment Variables:\n\n"
        "Option 1 (Encrypted - Recommended):\n"
        "  ENCRYPTION_KEY=your_fernet_key\n"
        "  BYBIT_API_KEY_ENCRYPTED=encrypted_key\n"
        "  BYBIT_API_SECRET_ENCRYPTED=encrypted_secret\n\n"
        "Option 2 (Plain text - Quick setup):\n"
        "  BYBIT_API_KEY=your_api_key\n"
        "  BYBIT_API_SECRET=your_api_secret\n\n"
        "Go to: Render Dashboard -> Environment -> Add Variables"
    )


# ===== HELPER FUNCTIONS =====

def generate_encryption_key():
    """
    Згенерувати новий ключ шифрування (один раз)
    Запустіть локально: python -c "from config import generate_encryption_key; generate_encryption_key()"
    """
    key = Fernet.generate_key()
    print("=" * 60)
    print("🔐 NEW ENCRYPTION KEY (Base64):")
    print(key.decode())
    print("=" * 60)
    print("\nSet this as ENCRYPTION_KEY in Render Environment Variables")
    return key


def encrypt_credentials(api_key, api_secret, encryption_key):
    """
    Зашифрувати API credentials
    
    Використання:
    python -c "from config import encrypt_credentials; encrypt_credentials('your_key', 'your_secret', 'your_encryption_key')"
    """
    cipher = Fernet(encryption_key.encode() if isinstance(encryption_key, str) else encryption_key)
    
    encrypted_key = cipher.encrypt(api_key.encode()).decode()
    encrypted_secret = cipher.encrypt(api_secret.encode()).decode()
    
    print("=" * 60)
    print("🔐 ENCRYPTED CREDENTIALS:")
    print("=" * 60)
    print(f"\nBYBIT_API_KEY_ENCRYPTED={encrypted_key}")
    print(f"\nBYBIT_API_SECRET_ENCRYPTED={encrypted_secret}")
    print("\n" + "=" * 60)
    print("Copy these to Render Environment Variables")
    
    return encrypted_key, encrypted_secret


# Для зворотної сумісності зі старим кодом
API_KEY, API_SECRET = get_api_credentials()


# ===== QUICK SETUP GUIDE =====
if __name__ == "__main__":
    print("""
    🔐 ENCRYPTION SETUP GUIDE
    ═══════════════════════════════════════════════
    
    Step 1: Generate encryption key (run once)
    ───────────────────────────────────────────────
    python -c "from config import generate_encryption_key; generate_encryption_key()"
    
    Step 2: Encrypt your API credentials
    ───────────────────────────────────────────────
    python -c "from config import encrypt_credentials; encrypt_credentials('YOUR_API_KEY', 'YOUR_API_SECRET', 'YOUR_ENCRYPTION_KEY')"
    
    Step 3: Add to Render Environment Variables
    ───────────────────────────────────────────────
    ENCRYPTION_KEY=<from step 1>
    BYBIT_API_KEY_ENCRYPTED=<from step 2>
    BYBIT_API_SECRET_ENCRYPTED=<from step 2>
    
    ═══════════════════════════════════════════════
    
    OR use plain text (quick setup):
    ───────────────────────────────────────────────
    BYBIT_API_KEY=your_real_key
    BYBIT_API_SECRET=your_real_secret
    """)
