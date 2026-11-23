"""🛡 Права користувача на інфосайті (вимога 24.11).

Дослівно: «В боті "🛡 Керування користувачами" мають бути по кожному
користувачу всі варіанти що можна робити а що не можна, що відображати а не
відображати. У адміністратора бота має бути доступ до всього, а користувачам
все налаштовує адміністратор з бота» + «стосовно інфосайту, які дані
показувати користувачу а які ні».

python3 test_user_perms.py
"""
import importlib.util
import json
import os
import re
import types

ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


P = _load('user_perms_t', 'web/user_perms.py')


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


def test_admin_gets_everything():
    e = P.effective(True, {'sec_terminal': False, 'trade_real': False})
    assert all(e.values()) and set(e) == set(P.KEYS)


def test_user_defaults_and_dependency():
    e = P.effective(False, None)
    assert e['sec_terminal'] and e['ly_ob'] and e['ntf_sbr']
    assert not e['trade_paper'] and not e['trade_real']        # торгівлю дає лише адмін
    e = P.effective(False, {'sec_terminal': False, 'trade_paper': True, 'ly_ob': True})
    # без Термінала немає ні шарів, ні сповіщень, ні торгівлі
    assert not any(e[k] for k in P.KEYS if k.startswith(('ly_', 'ntf_', 'trade_')))
    assert e['sec_signals'] and e['sec_paper']


def test_norm_apply_books():
    assert P.norm({'ly_ob': 0, 'bogus': True, 'trade_real': 'yes'}) == {'ly_ob': False, 'trade_real': True}
    cur = P.apply({'ly_ob': False}, {'key': 'trade_paper', 'on': True})
    assert cur['ly_ob'] is False and cur['trade_paper'] is True and set(cur) == set(P.KEYS)
    cur = P.apply(cur, {'perms': {k: False for k in P.KEYS}})
    assert not any(cur.values())
    assert P.apply(cur, {'key': 'nope', 'on': True}) == cur
    assert P.books({'trade_paper': True, 'trade_real': True}) == ('paper', 'real')
    assert P.books({'trade_real': True}) == ('real',) and P.books({}) == ()


def test_strip_coin_hides_forbidden_layers_without_touching_cache():
    pay = {'ok': True, 'ohlc': [1], 'ob': [{'a': 1}], 'vob': [{'b': 1}], 'imb': [1], 'events': [1],
           'sbr': [1], 'sbr_last': {'1h': {'t': 1}}, 'hl': [1], 'poc': {'poc': 1}, 'shl': {'h': 1},
           'pd': {'pct': 10}, 'vsplit': {'m': []}, 'sig': 'x'}
    pm = P.effective(False, {'ly_ob': False, 'ly_pd': False, 'ly_sbr': False})
    out = P.strip_coin(pay, pm)
    assert out['ob'] == [] and out['pd'] is None and out['sbr'] == [] and out['sbr_last'] == {}
    assert out['vob'] == [{'b': 1}] and out['ohlc'] == [1]
    assert set(out['perm_off']) == {'ly_ob', 'ly_pd', 'ly_sbr'}
    assert pay['ob'] == [{'a': 1}] and pay['pd'] == {'pct': 10}            # спільний кеш цілий
    assert P.strip_coin(pay, P.effective(True, None)) is pay
    assert P.sig(pm) != P.sig(P.effective(False, None))


def test_of_user_reads_prefs():
    u = types.SimpleNamespace(is_admin=False, prefs=json.dumps({'perms': {'sec_paper': False}}))
    assert P.of_user(u)['sec_paper'] is False and P.of_user(u)['sec_signals'] is True
    assert not any(P.of_user(None).values())


def test_server_enforces_every_section():
    ip = _src('web/info_portal.py')
    for route, key in (('/api/info/rows', 'sec_terminal'), ('/api/info/coin', 'sec_terminal'),
                       ('/api/info/universe', 'sec_terminal'), ('/api/info/liq', 'ly_liq'),
                       ('/api/me/liq-calc', 'ly_liq'), ('/api/info/paper', 'sec_paper'),
                       ('/api/info/paper/days', 'sec_paper'), ('/api/info/recommended', 'sec_signals')):
        i = ip.index(f"@app.route('{route}'")
        seg = ip[i:ip.index('@app.route', i + 10)]
        assert f"_no('{key}')" in seg, route
    assert 'user_perms.strip_coin(p, pm)' in ip and "('ntf_sbr'" in ip.replace("('sbr', 'ntf_sbr')", "('ntf_sbr'")
    auth = _src('web/auth.py')
    for fn, key in (('info_sbr_subscribers', 'ntf_sbr'), ('info_anom_subscribers', 'ntf_anom')):
        body = auth[auth.index('def ' + fn):]
        body = body[:body.index('\ndef ', 5)]
        assert f".get('{key}')" in body, fn


def test_user_cannot_overwrite_server_prefs():
    auth = _src('web/auth.py')
    i = auth.index("@app.route('/api/me/prefs'")
    seg = auth[i:auth.index('@app.route', i + 10)]
    assert "prefs['ui'] = " in seg and 'prefs=json.dumps(d)' not in seg


def test_admin_panel_has_perms():
    auth = _src('web/auth.py')
    assert "action == 'set_perms'" in auth and "'perm_defs': _up().defs()" in auth
    assert "'perms': _perms_raw(u)" in auth and 'function permsHTML(u)' in auth
    assert 'адміністратор — доступ до всього' in auth
    # кабінет бота: торгівля + ключі (адмін)
    assert 'id="keys"' in auth and "fetch('/api/me/keys')" in auth and 'id="trade"' in auth


def test_infosite_hides_by_perms():
    js = _src('infosite/js/app.js')
    html = _src('infosite/index.html')
    css = _src('infosite/css/style.css')
    keys = re.search(r'var PERM_KEYS = \[([^\]]+)\]', js).group(1)
    assert sorted(re.findall(r'"([a-z_]+)"', keys)) == sorted(P.KEYS)      # дзеркало сервера
    for k in ('applyPerms(d.perms)', 'applyPerms(me.perms)', 'pOk("ly_pd")', 'pOk("sec_paper")',
              'pOk("sec_signals")', 'pOk("ntf_sbr")', 'perm-no-trade'):
        assert k in js, k
    for k in ('pm-sec_terminal', 'pm-sec_signals', 'pm-sec_paper', 'pm-ly_ob', 'pm-ly_vob',
              'pm-ly_poc', 'pm-ly_liq', 'pm-ly_vol', 'pm-trade', 'id="perm-note"'):
        assert k in html, k
        if k.startswith('pm-'):
            assert 'body.perm-no-' + k[3:] + ' .' + k in css, k


if __name__ == '__main__':
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_') and callable(fn):
            fn()
            n += 1
    print(f'OK {n}')
