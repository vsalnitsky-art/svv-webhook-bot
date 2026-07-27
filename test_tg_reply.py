"""Telegram support-reply routing tests.

Guards the fix that stops an admin's support reply from being broadcast to ALL
subscribers: (1) swipe-Reply routes to the ONE user — via the persisted map OR a
chat_id parsed from the forwarded header; (2) a plain admin message or a failed
reply is a HINT, never a broadcast; (3) mass sends require an explicit /announce;
(4) the reply-map survives a restart (DB persist/restore).
"""

import sys, os, types, importlib.util
sys.path.insert(0, '.')

# Load web/tg_bot.py directly, stubbing the `web` package (avoid heavy __init__).
if 'web' not in sys.modules:
    _wpkg = types.ModuleType('web'); _wpkg.__path__ = ['./web']
    sys.modules['web'] = _wpkg
_spec = importlib.util.spec_from_file_location('web.tg_bot', 'web/tg_bot.py')
tg = importlib.util.module_from_spec(_spec)
sys.modules['web.tg_bot'] = tg
_spec.loader.exec_module(tg)

# Fake DB for persist/restore, injected as storage.db_operations.get_db.
class _DB:
    store = {}
    def get_setting(self, k, d=None): return _DB.store.get(k, d)
    def set_setting(self, k, v): _DB.store[k] = v
_stg = types.ModuleType('storage'); _stg.__path__ = []
_dbmod = types.ModuleType('storage.db_operations')
_dbmod.get_db = lambda: _DB()
sys.modules['storage'] = _stg
sys.modules['storage.db_operations'] = _dbmod

os.environ['TELEGRAM_CHAT_ID'] = '999'   # admin chat


def _reset_stubs():
    tg._reply_map.clear()
    _DB.store.clear()
    tg.sent = []
    tg.announced = []
    tg.tg_send = lambda cid, text, buttons=None: (tg.sent.append((str(cid), text)) or True)
    tg._copy_message = lambda to, frm, mid: 111
    tg._admin_broadcast = lambda m, cid, body_override=None: tg.announced.append(body_override or '<media>')
    tg._handle_start = lambda *a, **k: None
    tg._has_media = lambda m: False


def _admin_msg(text, reply_to=None):
    m = {'chat': {'id': 999, 'type': 'private'}, 'from': {'id': 999}, 'text': text,
         'message_id': 5}
    if reply_to is not None:
        m['reply_to_message'] = reply_to
    return m


def test_chat_id_parser():
    assert tg._chat_id_from_text('chat_id: <code>7659029832</code>') == '7659029832'
    assert tg._chat_id_from_text('chat_id:7659029832') == '7659029832'
    assert tg._chat_id_from_text('немає айді тут') is None
    print('✓ chat_id парситься із шапки')


def test_plain_admin_message_is_hint_not_broadcast():
    _reset_stubs()
    tg._handle_message(_admin_msg('This project is under development…'))
    assert tg.announced == [], 'plain message must NOT broadcast'
    assert tg.sent and 'Кому це надіслати' in tg.sent[-1][1], tg.sent
    print('✓ звичайне повідомлення адміна → підказка, НЕ розсилка')


def test_swipe_reply_routes_via_map():
    _reset_stubs()
    tg._remember(42, '7659029832')          # forwarded header msg_id=42 → user
    rt = {'message_id': 42, 'text': 'anything'}
    tg._handle_message(_admin_msg('Ось відповідь', reply_to=rt))
    to_user = [t for c, t in tg.sent if c == '7659029832']
    assert to_user and to_user[0].startswith('💬'), tg.sent
    assert tg.announced == [], 'reply must not broadcast'
    print('✓ свайп-Reply → лише конкретному юзеру (через мапу)')


def test_swipe_reply_falls_back_to_header_chatid():
    _reset_stubs()   # map EMPTY (simulates restart)
    rt = {'message_id': 42, 'text': 'chat_id: <code>7659029832</code>'}
    tg._handle_message(_admin_msg('Ось відповідь', reply_to=rt))
    assert any(c == '7659029832' and t.startswith('💬') for c, t in tg.sent), tg.sent
    assert tg.announced == [], 'reply must not broadcast even without the map'
    print('✓ мапа порожня (рестарт) → chat_id береться з шапки, без розсилки')


def test_unlinkable_reply_is_hint_not_broadcast():
    _reset_stubs()
    rt = {'message_id': 42, 'text': 'без айді'}   # map empty + no chat_id
    tg._handle_message(_admin_msg('щось', reply_to=rt))
    assert tg.announced == [], 'unlinkable reply must NOT broadcast'
    assert any('Не вдалося визначити' in t for _, t in tg.sent), tg.sent
    print('✓ reply без звʼязку → підказка /reply, НЕ розсилка')


def test_announce_broadcasts():
    _reset_stubs()
    tg._handle_message(_admin_msg('/announce Знижка сьогодні!'))
    assert tg.announced == ['Знижка сьогодні!'], tg.announced
    print('✓ /announce → розсилка всім (явна команда)')


def test_reply_map_persists_across_restart():
    _reset_stubs()
    tg._remember(77, '12345')
    tg._reply_map.clear()             # simulate process restart
    tg._load_reply_map()              # restore from DB
    assert tg._reply_map.get(77) == '12345', tg._reply_map
    print('✓ reply-map відновлюється з БД після рестарту')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll tg-reply tests passed ✓ ({len(tests)} tests)')
