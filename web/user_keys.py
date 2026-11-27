"""🔐 КЛЮЧІ БІРЖІ КОРИСТУВАЧА — окремі від ключів бота (вимога 24.11).

Дослівно: «Ключі адміністратора бота — то для роботи і торгівлі бота, а ключі
користувачів — це зовсім окремі ключі і призначені чітко для конкретного
користувача і працюють лише на цього і для цього користувача (тобто власника
ключів). Це дуже важливе правило.»

• Ключі БОТА — лише `web.exchange_keys` (адміністратор; БД
  `bybit_api_*_encrypted`, їх читає резолвер бота).
• Ключі КОРИСТУВАЧА — лише тут: окремий запис БД на кожного
  (`user_exkeys_v1_<uid>`), поза `prefs` (ні клієнт, ні список користувачів
  адмінки їх не бачать), Fernet (`ENCRYPTION_KEY`). Запис несе `uid` власника і
  читається лише для нього (`load` звіряє). Використовує їх ЛИШЕ особистий
  Trade Manager власника (`detection/user_tm.py`) — бот їх не читає ніколи.
• Ключі бота як ключі користувача зареєструвати НЕ можна (відмова) — два
  власники одного рахунку ламають розділення.
• Відкритий текст не повертається клієнту й не пишеться в лог.
"""
import os
import time

KEY_FMT = 'user_exkeys_v1_{uid}'


def _db():
    from storage.db_operations import get_db
    return get_db()


def _enc_key() -> str:
    return os.environ.get('ENCRYPTION_KEY', '').strip()


def db_key(uid) -> str:
    return KEY_FMT.format(uid=int(uid))


def mask(v: str) -> str:
    from web.exchange_keys import mask as _m
    return _m(v)


def _record(uid):
    try:
        rec = _db().get_setting(db_key(uid), None)
    except Exception:
        return None
    if not isinstance(rec, dict) or not rec.get('k') or not rec.get('s'):
        return None
    try:
        if int(rec.get('uid')) != int(uid):          # запис мусить належати власнику
            return None
    except (TypeError, ValueError):
        return None
    return rec


def load(uid):
    """(key, secret) ЛИШЕ для власника `uid` або (None, None). Внутрішнє —
    ніколи не віддавати клієнту."""
    rec = _record(uid)
    ek = _enc_key()
    if rec is None or not ek:
        return None, None
    from config.bot_settings import _decrypt_fernet
    k = _decrypt_fernet(rec['k'], ek)
    s = _decrypt_fernet(rec['s'], ek)
    return (k, s) if (k and s) else (None, None)


def is_bot_key(api_key: str) -> bool:
    """ЧИСТА над резолвером бота: цей ключ — ключ БОТА?"""
    try:
        from config.bot_settings import get_bybit_keys
        bk, _ = get_bybit_keys()
    except Exception:
        return False
    return bool(bk) and (api_key or '').strip() == bk


def status(uid) -> dict:
    rec = _record(uid)
    k, s = load(uid) if rec else (None, None)
    return {'ok': True, 'owner': 'user', 'exchange': 'bybit',
            'has_keys': bool(k and s), 'stored': rec is not None,
            'key_preview': mask(k or ''), 'secret_preview': mask(s or ''),
            'saved_at': (rec or {}).get('at'),
            'encryption_key_set': bool(_enc_key())}


def _changed(uid):
    try:
        from detection import user_tm
        user_tm.on_keys_changed(uid)
    except Exception as e:
        print(f'[UKEYS] user TM reload u{uid}: {e}')


def save(uid, api_key: str, api_secret: str) -> dict:
    from web.exchange_keys import validate
    from config.bot_settings import _encrypt_fernet
    ek = _enc_key()
    if not ek:
        return {'ok': False, 'code': 400,
                'error': 'На сервері не задано ENCRYPTION_KEY — ключі не зашифрувати.'}
    err = validate(api_key, api_secret)
    if err:
        return {'ok': False, 'code': 400, 'error': err}
    if is_bot_key(api_key):
        return {'ok': False, 'code': 400,
                'error': 'Це ключі БОТА. Ваш Trade Manager працює лише з ВАШИМИ '
                         'власними ключами — створіть окремі ключі на своєму акаунті Bybit.'}
    k_enc = _encrypt_fernet(api_key.strip(), ek)
    s_enc = _encrypt_fernet(api_secret.strip(), ek)
    if not (k_enc and s_enc):
        return {'ok': False, 'code': 500, 'error': 'Шифрування не вдалось (формат ENCRYPTION_KEY)'}
    try:
        _db().set_setting(db_key(uid), {'k': k_enc, 's': s_enc, 'at': time.time(),
                                        'uid': int(uid)})
    except Exception as e:
        return {'ok': False, 'code': 500, 'error': f'Запис у БД не вдався: {e}'}
    _changed(uid)
    return {'ok': True, 'owner': 'user', 'key_preview': mask(api_key.strip()),
            'message': 'Ваші ключі зашифровано і збережено. Ними торгує ЛИШЕ ваш Trade Manager.'}


def clear(uid) -> dict:
    try:
        _db().set_setting(db_key(uid), {})
    except Exception as e:
        return {'ok': False, 'code': 500, 'error': f'Запис у БД не вдався: {e}'}
    _changed(uid)
    return {'ok': True, 'owner': 'user', 'message': 'Ваші ключі видалено.'}


def purge(uid) -> None:
    """Видалення користувача → його ключі теж."""
    try:
        _db().set_setting(db_key(uid), {})
    except Exception:
        pass
    _changed(uid)


def test(uid) -> dict:
    k, s = load(uid)
    if not (k and s):
        return {'ok': False, 'code': 400, 'error': 'Ваші ключі не задано'}
    try:
        from core.bybit_connector import BybitConnector
        c = BybitConnector.for_keys(k, s)
        r = c.session.get_wallet_balance(accountType='UNIFIED')
    except Exception as e:
        return {'ok': False, 'auth_works': False, 'error': str(e)[:300]}
    from web.exchange_keys import FRIENDLY
    code = r.get('retCode', -1)
    if code == 0:
        return {'ok': True, 'auth_works': True, 'owner': 'user', 'key_preview': mask(k),
                'message': 'Авторизація успішна — ваші ключі робочі'}
    return {'ok': False, 'auth_works': False, 'ret_code': code,
            'friendly': FRIENDLY.get(code, r.get('retMsg') or 'невідома відповідь біржі')}
