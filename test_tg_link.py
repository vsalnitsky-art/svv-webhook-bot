"""🔗 Привʼязка Telegram до ІСНУЮЧОГО акаунта (вимога 08.11).

Акаунт, створений адміністратором (або сам адмін), раніше не мав способу
отримати Telegram: /start бота шле лише посилання на НОВУ реєстрацію.
Тепер кабінет видає одноразовий код → `t.me/<бот>?start=lk_<код>` → бот
привʼязує chat_id саме до цього акаунта."""
import os
import sys
import tempfile
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_DB = os.path.join(tempfile.mkdtemp(), 'tglink.db')
os.environ['DATABASE_URL'] = 'sqlite:///' + _DB

from storage.db_models import User, engine   # noqa: E402
User.__table__.create(engine, checkfirst=True)

_spec = importlib.util.spec_from_file_location('auth_t', os.path.join(_HERE, 'web', 'auth.py'))
auth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auth)


def _check(c, m):
    if not c:
        raise AssertionError(m)


def test_link_existing_account_by_code():
    uid = auth.create_user('admin', 'secret123', is_admin=True, approved=True)
    _check(uid, 'акаунт створено')
    code, until = auth.tg_link_issue(uid)
    _check(auth.tg_link_code_ok(code) and until > 0, 'код валідний')
    ok, msg, got = auth.tg_link_consume(code, '777', 'adm', 'Admin')
    _check(ok and got == uid and msg == 'admin', (ok, msg, got))
    u = auth.get_user_by_id(uid)
    _check(u.telegram_chat_id == '777' and u.telegram_username == 'adm', 'chat_id привʼязано')
    _check('tg_link' not in auth._load_prefs(u), 'код одноразовий — прибрано')
    ok2, _m, _u = auth.tg_link_consume(code, '777')
    _check(not ok2, 'повторно той самий код не працює')
    print('✓ код привʼязує Telegram до існуючого акаунта, одноразово')


def test_expired_and_foreign_chat_are_refused():
    a = auth.create_user('user_a', 'secret123', approved=True, telegram_chat_id='555')
    b = auth.create_user('user_b', 'secret123', approved=True)
    code, _ = auth.tg_link_issue(b, now=1000)
    ok, msg, _u = auth.tg_link_consume(code, '999', now=1000 + auth.TG_LINK_TTL_SEC + 1)
    _check(not ok and 'простроч' in msg, ('прострочений код', msg))
    code, _ = auth.tg_link_issue(b)
    ok, msg, _u = auth.tg_link_consume(code, '555')
    _check(not ok and 'user_a' in msg, ('чужий чат не перепривʼязується', msg))
    _check(auth.get_user_by_id(b).telegram_chat_id is None, 'акаунт b без змін')
    _check(not auth.tg_link_code_ok('bad.code!') and not auth.tg_link_consume('x', '1')[0],
           'сміттєвий код відкидається')
    _ = a
    print('✓ прострочений код і чужий Telegram — відмова з причиною')


def test_bot_and_pages_wire_the_flow():
    tg = open(os.path.join(_HERE, 'web', 'tg_bot.py'), encoding='utf-8').read()
    _check(tg.count('if _link_begin(cid, _arg, uname, fname):') == 2,
           '/start lk_… обробляється і в чаті адміна, і в користувача')
    _check('Привʼязати Telegram' in tg, 'новому /start підказано про привʼязку')
    au = open(os.path.join(_HERE, 'web', 'auth.py'), encoding='utf-8').read()
    _check("@app.route('/api/me/tg-link'" in au and 'tglink()' in au, 'маршрут + кнопка в кабінеті бота')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('data-tglink' in js and '/api/me/tg-link' in js, 'кнопка на інфосайті')
    print('✓ бот, кабінет і інфосайт повʼязані')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
