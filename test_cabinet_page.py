"""👤 Особистий кабінет — повна сторінка, вхід лише зі сторінки авторизованого
користувача (вимога 24.11).

Дослівно: «Особистий кабінет користувача зроби повною сторінкою і з доступом
лише із головної сторінки авторизованого користувача.»

python3 test_cabinet_page.py
"""
import ast
import html
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


def _pure(rel, names):
    """Чисті функції з файлу без його імпортів (auth.py тягне flask)."""
    src = _src(rel)
    tree = ast.parse(src)
    ns = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.Module([n], []), rel, 'exec'), ns)
    return ns


def _admin_pages():
    if 'markupsafe' not in sys.modules:
        ms = types.ModuleType('markupsafe')
        ms.escape = lambda v: html.escape(str(v), quote=True)
        sys.modules['markupsafe'] = ms
    sp = importlib.util.spec_from_file_location('admin_pages_t', os.path.join(ROOT, 'web/admin_pages.py'))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


A = _pure('web/auth.py', {'cabinet_entry_ok', 'cabinet_home'})


def test_entry_only_from_signed_in_pages():
    ok = A['cabinet_entry_ok']
    host, info = 'bot.example.com', 'https://info.example.com'
    assert ok('https://bot.example.com/', '', host, info)                    # головна бота
    assert ok('https://info.example.com/#trade-card', 'cross-site', host, info)   # інфосайт
    assert ok('', 'same-origin', host, info)                                 # навігація в межах бота
    assert not ok('', 'none', host, info)                                    # набраний / закладка
    assert not ok('', '', host, info)
    assert not ok('https://evil.example.org/x', 'cross-site', host, info)    # посилання ззовні
    assert not ok('https://evil.example.org/x', 'cross-site', host, '')
    assert not ok('not a url', '', host, info)


def test_home_is_the_users_main_page():
    h = A['cabinet_home']
    assert h(True, False, 'https://info.x') == '/'
    assert h(True, True, 'https://info.x') == 'https://info.x/'
    assert h(False, False, 'https://info.x') == 'https://info.x/'
    assert h(False, False, '') == '/info/'


def test_route_redirects_direct_entry_and_renders_full_page():
    src = _src('web/auth.py')
    i = src.index("@app.route('/cabinet')")
    seg = src[i:src.index('@app.route', i + 10)]
    assert 'cabinet_entry_ok(' in seg and 'return redirect(home)' in seg
    assert seg.index('cabinet_entry_ok(') < seg.index('render_cabinet(')
    assert "_page('Кабінет'" not in src and 'width=500' not in seg          # старе вікно прибрано
    assert "'/cabinet'" in src[src.index('_USER_NO_BOT_ALLOWED'):][:200]   # лише для авторизованих


def test_full_page_in_infosite_style():
    AP = _admin_pages()
    base = {'email': 'u@x', 'role': 'Користувач', 'tg': '—', 'tg_linked': False, 'bot_ok': False,
            'bot': 'ні', 'access': 'безлімітний', 'trade': True, 'trade_href': '/info/#trade-card',
            'home': '/info/', 'home_lbl': '← На головну'}
    p = AP.render_cabinet(dict(base, email='<b>x</b>@y'), '/info/')
    assert p.startswith('<!doctype html>') and 'class="topbar"' and 'class="grid2"' in p
    assert '&lt;b&gt;x&lt;/b&gt;@y' in p and '<b>x</b>@y' not in p        # екранування
    assert 'id="keys"' not in p and 'Дашборд' not in p and '/admin/users' not in p
    assert 'id="trade"' in p and 'Мій Trade Manager' in p and 'paper' not in p.split('<script>')[0]
    a = AP.render_cabinet(dict(base, is_admin=True, bot_ok=True, role='Адміністратор',
                               home='/', home_lbl='← На дашборд'), '/info/')
    assert 'id="keys"' in a and 'id="nbtc"' in a and '/admin/users' in a and 'VSV<small>ADMIN' in a
    assert 'onclick="tglink()"' in p and 'onclick="chpw()"' in p


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
        print('✓', f.__name__)
    print(f'OK {len(fns)}')
