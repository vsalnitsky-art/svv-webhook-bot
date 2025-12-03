import os
from cryptography.fernet import Fernet
def get_api_credentials():
    encryption_key = os.environ.get('ENCRYPTION_KEY')
    if encryption_key:
        try:
            cipher = Fernet(encryption_key.encode())
            k = os.environ.get('BYBIT_API_KEY_ENCRYPTED')
            s = os.environ.get('BYBIT_API_SECRET_ENCRYPTED')
            if k and s: return cipher.decrypt(k.encode()).decode(), cipher.decrypt(s.encode()).decode()
        except: pass
    return os.environ.get('BYBIT_API_KEY', ''), os.environ.get('BYBIT_API_SECRET', '')
API_KEY, API_SECRET = get_api_credentials()