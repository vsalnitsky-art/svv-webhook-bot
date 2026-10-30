"""💬 Коментарі в темах (вимога 30.10): перевірка й увімкнення через Bot API.

  • права учасників ПЕРЕНОСЯТЬСЯ — `setChatPermissions` не забирає медіа тощо;
  • закрита тема відкривається, TOPIC_NOT_MODIFIED — не помилка;
  • канал без групи обговорень → чесна причина, жодних викликів зміни;
  • приватний чат адміна → коментарів там немає (не чіпаємо).
"""
import os, sys, types, importlib.util
sys.path.insert(0, '.')
for n in ('storage', 'storage.db_operations'):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['storage.db_operations'].get_db = lambda: None
_w = types.ModuleType('web'); _w.__path__ = ['./web']; sys.modules['web'] = _w
_sp = importlib.util.spec_from_file_location('web.tg_bot', 'web/tg_bot.py')
T = importlib.util.module_from_spec(_sp); sys.modules['web.tg_bot'] = T; _sp.loader.exec_module(T)


def _run(chat_info, routes, enable, api_answers=None):
    calls = []
    def api(m, p=None, timeout=35):
        calls.append((m, p))
        if m == 'getChat':
            return {'ok': True, 'result': dict(chat_info)}
        return (api_answers or {}).get(m, {'ok': True, 'result': True})
    T._api = api
    T._cat_chat = lambda c: routes.get(c, (None, None))
    T._admin_chat = lambda: '42'
    return T.comments_status(enable=enable), calls


def test_merge_keeps_other_rights():
    m = T._merge_send_permission({'can_send_photos': True, 'can_send_polls': False,
                                  'can_send_messages': False})
    assert m == {'can_send_photos': True, 'can_send_polls': False, 'can_send_messages': True}


def test_enable_in_forum_group():
    info = {'type': 'supergroup', 'is_forum': True,
            'permissions': {'can_send_messages': False, 'can_send_photos': True}}
    routes = {c: ('-100', str(i + 1)) for i, c in enumerate(T._COMMENT_CATS)}
    rows, calls = _run(info, routes, enable=True)
    perm = [p for m, p in calls if m == 'setChatPermissions']
    assert len(perm) == 1, 'права групи міняються ОДИН раз на чат'
    assert perm[0]['permissions'] == {'can_send_messages': True, 'can_send_photos': True}
    assert len([1 for m, _ in calls if m == 'reopenForumTopic']) == len(T._COMMENT_CATS)
    assert all(r['can_comment'] and r['fixed'] for r in rows), rows
    assert len([1 for m, _ in calls if m == 'getChat']) == 1


def test_check_only_changes_nothing():
    info = {'type': 'supergroup', 'permissions': {'can_send_messages': False}}
    rows, calls = _run(info, {'sbr': ('-100', '7')}, enable=False)
    assert not [m for m, _ in calls if m in ('setChatPermissions', 'reopenForumTopic')]
    sbr = [r for r in rows if r['category'] == 'sbr'][0]
    assert not sbr['can_comment'] and 'заборонено' in sbr['reason']


def test_topic_not_modified_is_success():
    info = {'type': 'supergroup', 'permissions': {'can_send_messages': True}}
    rows, _ = _run(info, {'sbr': ('-100', '7')}, enable=True, api_answers={
        'reopenForumTopic': {'ok': False, 'description': 'Bad Request: TOPIC_NOT_MODIFIED'}})
    assert [r for r in rows if r['category'] == 'sbr'][0]['can_comment']


def test_channel_without_discussion_and_admin_private():
    rows, calls = _run({'type': 'channel'}, {'btc': ('-200', None), 'trades': ('42', None)},
                       enable=True)
    by = {r['category']: r for r in rows}
    assert not by['btc']['can_comment'] and 'Обговорення' in by['btc']['reason']
    assert 'ПРИВАТНИЙ' in by['trades']['reason']
    assert not [m for m, _ in calls if m == 'setChatPermissions']


def test_route_and_ui_wired():
    app = open('web/flask_app.py', encoding='utf-8').read()
    html = open('templates/smart_money.html', encoding='utf-8').read()
    assert "'/api/tg/comments'" in app and 'comments_status' in app
    assert "fetch('/api/tg/comments'" in html and 'onclick="tgComments(true)"' in html


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('✓', k); n += 1
    print(f'OK — {n} тестів')
