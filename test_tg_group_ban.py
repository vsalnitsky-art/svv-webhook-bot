"""🚫 БЛОК У БОТІ = БАН У TELEGRAM-ГРУПІ, І НАВПАКИ (вимога 23.11).

Дослівно: «заблокованого користувача в боті разом автоматично блокувати в
телеграм каналі… повністю блокувати, а потім при нагоді автоматично знімати
цей блок. І навпаки.»
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))


def _check(c, msg=''):
    if not c:
        raise AssertionError(msg)


def _load():
    sp = importlib.util.spec_from_file_location('tg_bot_gb', os.path.join(_HERE, 'web', 'tg_bot.py'))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


ENV = {'TELEGRAM_FORUM_CHAT': '-1001', 'TELEGRAM_GROUP_CHAT': '-1001',
       'TELEGRAM_CHAT_ID': '555', 'TELEGRAM_CHAT_TRADES': '-1002',
       'TELEGRAM_CHAT_REGISTER': '-1009', 'TELEGRAM_CHAT_BTC': '777',
       'TELEGRAM_CHAT_SIGNALS': '@vsv_signals'}


def test_ban_chats_every_group_not_private():
    tg = _load()
    ch = tg.ban_chats_of(ENV)
    _check(ch[0] == '-1001', ('головна група першою', ch))
    _check('-1002' in ch and '@vsv_signals' in ch, ('публічні чати категорій теж', ch))
    _check(ch.count('-1001') == 1, 'без дублів')
    _check('555' not in ch and '777' not in ch, 'особисті чати / адмін — ні')
    _check('-1009' not in ch, 'адмін-тема реєстрації — ні')
    _check(tg.ban_chats_of({}) == [], 'немає налаштувань → нічого')
    print('✓ бан у кожному чаті бота')


def test_group_action_full_ban():
    tg = _load()
    _check(tg.group_action(True, False, None) == 'ban')
    _check(tg.group_action(True, False, 'restricted') == 'ban', 'стара «обмежена» → бан')
    _check(tg.group_action(True, False, 'banned') is None)
    _check(tg.group_action(False, True, 'banned') == 'restore')
    _check(tg.group_action(False, False, None) is None, 'очікує схвалення — не чіпаємо')
    print('✓ заблоковано → бан, активний → розбан')


def test_member_event_and_own_actions_ignored():
    tg = _load()
    _check(tg.tg_member_event('member', 'kicked') == 'banned')
    _check(tg.tg_member_event('restricted', 'kicked') == 'banned')
    _check(tg.tg_member_event('kicked', 'left') == 'unbanned')
    _check(tg.tg_member_event('kicked', 'member') == 'unbanned')
    _check(tg.tg_member_event('member', 'left') is None, 'вийшов сам — не блок')
    _check(tg.tg_member_event('kicked', 'kicked') is None)
    calls = []
    fake = types.ModuleType('web.auth')
    fake.tg_group_member_event = lambda tid, ev: calls.append((tid, ev)) or ev
    sys.modules['web.auth'] = fake
    old_env = dict(os.environ)
    try:
        os.environ.update({'TELEGRAM_FORUM_CHAT': '-1001', 'TELEGRAM_BOT_TOKEN': '4242:abc'})

        def upd(actor, old, new, chat=-1001, uid=77, bot=False):
            return {'chat': {'id': chat}, 'from': {'id': actor},
                    'old_chat_member': {'status': old, 'user': {'id': uid}},
                    'new_chat_member': {'status': new, 'user': {'id': uid, 'is_bot': bot}}}
        _check(tg._handle_chat_member(upd(4242, 'member', 'kicked')) is None, 'власний бан бота — пропуск')
        _check(tg._handle_chat_member(upd(1, 'member', 'kicked', chat=-999)) is None, 'чужий чат — пропуск')
        _check(tg._handle_chat_member(upd(1, 'member', 'kicked', bot=True)) is None, 'боти — пропуск')
        _check(tg._handle_chat_member(upd(1, 'member', 'kicked')) == 'banned', 'бан адміном групи')
        _check(tg._handle_chat_member(upd(1, 'kicked', 'left')) == 'unbanned', 'розбан адміном групи')
        _check(calls == [('77', 'banned'), ('77', 'unbanned')], calls)
    finally:
        os.environ.clear()
        os.environ.update(old_env)
        sys.modules.pop('web.auth', None)
    print('✓ навпаки: бан / розбан у групі → акаунт у боті')


def test_wiring():
    src = open(os.path.join(_HERE, 'web', 'tg_bot.py'), encoding='utf-8').read()
    _check("'chat_member']" in src and "_handle_chat_member(upd['chat_member'])" in src,
           'поллер отримує chat_member')
    sync = src.split('def group_access_sync')[1].split('\ndef ')[0]
    _check('group_ban(chat)' in sync and "'banned'" in sync and 'group_restrict(' not in sync,
           'синхронізація банить, а не обмежує')
    _check('invite_user_to_group(chat, back=True)' in sync, 'розбан → запрошення назад')
    auth = open(os.path.join(_HERE, 'web', 'auth.py'), encoding='utf-8').read()
    ev = auth.split('def tg_group_member_event')[1].split('\ndef ')[0]
    _check('disabled=True' in ev and 'disabled=False' in ev and 'u.is_admin' in ev,
           'бан → вимкнено, розбан → увімкнено, адміна не чіпаємо')
    _check("set_user_pref(uid, 'grp_state', None)" in ev, 'розбан при неактивному доступі → бан повертається')
    _check("action == 'group_sync'" in auth and "'grp_state':" in auth, 'адмінка: стан і синхронізація')
    adm = open(os.path.join(_HERE, 'web', 'admin_pages.py'), encoding='utf-8').read()
    _check('📨 Telegram-група' in adm and "'group_sync'" in adm and 'grpBadge' in adm, 'показ в адмінці')
    print('✓ звʼязки')


if __name__ == '__main__':
    for _n, _f in list(globals().items()):
        if _n.startswith('test_') and callable(_f):
            _f()
    print('OK')
