"""🔐 Ключі біржі (Bybit) — ЄДИНА реалізація статусу / збереження / видалення /
перевірки (вимога 24.11: «ключі адміністратора бота перенеси в особистий
кабінет»).

Кличуть ОБИДВА входи:
  • особистий кабінет адміністратора (`/api/me/keys*`, web/user_trading.py);
  • старі маршрути бота `/api/credentials/*` (сумісність, налаштування бота).

Зберігання не змінилось: Fernet (`ENCRYPTION_KEY` з env) → БД
`bybit_api_key_encrypted` / `bybit_api_secret_encrypted` — саме ці ключі
читає резолвер бота (`config.bot_settings.get_bybit_keys`), тож ключі,
внесені в кабінеті, ОДРАЗУ стають ключами торгівлі бота. Відкритий текст
ніколи не повертається клієнту й не пишеться в лог.
"""
import os

KEY_DB = 'bybit_api_key_encrypted'
SECRET_DB = 'bybit_api_secret_encrypted'


def mask(secret: str) -> str:
    """'••••••••AB12' (8 крапок + 4 останні символи); порожньо → 'none'."""
    if not secret:
        return 'none'
    if len(secret) <= 4:
        return '•' * len(secret)
    return '•' * 8 + secret[-4:]


def _enc_key() -> str:
    return os.environ.get('ENCRYPTION_KEY', '').strip()


def _reload():
    from config.bot_settings import reload_bybit_keys
    reload_bybit_keys()
    try:
        from core.bybit_connector import get_connector
        c = get_connector()
        if c:
            c.reload_keys()
    except Exception as e:                       # БД записано — підхопить рестарт
        print(f'[CREDS] Connector reload failed: {e}')


def status() -> dict:
    """Звідки взято ключі, маски, чи задано ENCRYPTION_KEY, чи є запис у БД."""
    from config.bot_settings import get_bybit_keys, _resolve_bybit_keys_from_db
    k, s = get_bybit_keys()
    source = 'none'
    if k and s:
        db_res = _resolve_bybit_keys_from_db()
        if db_res and db_res[0] == k:
            source = 'db_encrypted'
        elif os.environ.get('BYBIT_API_KEY', '').strip() == k:
            source = 'env_plain'
        else:
            source = 'env_encrypted'
    db_stored = False
    try:
        from storage.db_operations import get_db
        db = get_db()
        db_stored = bool(db.get_setting(KEY_DB, '') and db.get_setting(SECRET_DB, ''))
    except Exception:
        pass
    return {'ok': True, 'exchange': 'bybit', 'source': source,
            'has_keys': bool(k and s), 'key_preview': mask(k), 'secret_preview': mask(s),
            'encryption_key_set': bool(_enc_key()), 'db_has_stored': db_stored,
            'env_plain_set': bool(os.environ.get('BYBIT_API_KEY', '').strip()
                                  and os.environ.get('BYBIT_API_SECRET', '').strip()),
            'env_encrypted_set': bool(os.environ.get('BYBIT_API_KEY_ENCRYPTED', '').strip()
                                      and os.environ.get('BYBIT_API_SECRET_ENCRYPTED', '').strip())}


def validate(api_key: str, api_secret: str) -> str:
    """ЧИСТА: '' або причина відмови (порожньо / закоротко / пробіли)."""
    k, s = (api_key or '').strip(), (api_secret or '').strip()
    if not k or not s:
        return 'Потрібні обидва поля: API Key і API Secret'
    if any(c.isspace() for c in k + s):
        return 'Ключ або секрет містить пробіли — скопіюйте ще раз'
    if len(k) < 8 or len(s) < 16:
        return (f'Ключі закороткі (key={len(k)}, secret={len(s)} символів) — '
                'перевірте, чи скопійовано повністю')
    return ''


def save(api_key: str, api_secret: str) -> dict:
    """Зашифрувати (Fernet) і зберегти в БД → бот одразу торгує з ними."""
    from config.bot_settings import _encrypt_fernet, get_bybit_keys
    ek = _enc_key()
    if not ek:
        return {'ok': False, 'code': 400,
                'error': 'На сервері не задано ENCRYPTION_KEY — без нього ключі не зашифрувати. '
                         'Додайте змінну середовища ENCRYPTION_KEY і перезапустіть бота.'}
    err = validate(api_key, api_secret)
    if err:
        return {'ok': False, 'code': 400, 'error': err}
    k_enc = _encrypt_fernet(api_key.strip(), ek)
    s_enc = _encrypt_fernet(api_secret.strip(), ek)
    if not (k_enc and s_enc):
        return {'ok': False, 'code': 500,
                'error': 'Шифрування не вдалось — перевірте формат ENCRYPTION_KEY (Fernet base64)'}
    try:
        from storage.db_operations import get_db
        db = get_db()
        db.set_setting(KEY_DB, k_enc)
        db.set_setting(SECRET_DB, s_enc)
    except Exception as e:
        return {'ok': False, 'code': 500, 'error': f'Запис у БД не вдався: {e}'}
    _reload()
    k, _ = get_bybit_keys()
    print(f'[CREDS] ✅ Bybit keys updated (now {len(k)} chars, source=DB encrypted)')
    return {'ok': True, 'source': 'db_encrypted', 'key_preview': mask(k),
            'message': 'Ключі зашифровано і збережено. Діють одразу.'}


def clear() -> dict:
    """Прибрати ключі з БД → бот бере ENV (якщо задано), інакше без торгівлі."""
    try:
        from storage.db_operations import get_db
        db = get_db()
        db.set_setting(KEY_DB, '')
        db.set_setting(SECRET_DB, '')
    except Exception as e:
        return {'ok': False, 'code': 500, 'error': f'Запис у БД не вдався: {e}'}
    _reload()
    print('[CREDS] ✅ DB-stored Bybit keys cleared; falling back to ENV')
    return {'ok': True, 'message': 'Ключі прибрано з БД. Бот бере ENV, якщо він заданий.'}


FRIENDLY = {10003: 'Невірний API Key',
            10004: 'Помилка підпису — невірний API Secret',
            10005: 'Немає прав — ключу потрібні дозволи Read + Trade (Unified Trading)',
            10006: 'Перевищено ліміт запитів — спробуйте за кілька секунд',
            10010: 'IP сервера не в білому списку ключа на Bybit',
            33004: 'Строк дії ключа минув — створіть нові ключі на Bybit'}


def test() -> dict:
    """Перевірити чинні ключі приватним запитом (баланс гаманця; суми НЕ
    повертаємо)."""
    from core.bybit_connector import get_connector
    c = get_connector()
    if c is None:
        return {'ok': False, 'code': 500, 'error': 'Конектор біржі не запущено'}
    if not (getattr(c, 'api_key', None) and getattr(c, 'api_secret', None)):
        return {'ok': False, 'code': 400, 'error': 'Ключі не задано'}
    try:
        r = c.session.get_wallet_balance(accountType='UNIFIED')
    except Exception as e:
        return {'ok': False, 'auth_works': False, 'error': str(e)[:300]}
    code = r.get('retCode', -1)
    if code == 0:
        return {'ok': True, 'auth_works': True, 'key_preview': mask(c.api_key),
                'message': 'Авторизація успішна — ключі робочі'}
    return {'ok': False, 'auth_works': False, 'ret_code': code, 'ret_msg': r.get('retMsg', ''),
            'friendly': FRIENDLY.get(code, r.get('retMsg') or 'невідома відповідь біржі')}
