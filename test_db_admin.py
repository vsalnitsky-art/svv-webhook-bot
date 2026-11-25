"""🗄️ Database Administration: одна класифікація таблиць, очистка за білим
списком, 🔬 дослідження — лише вручну, ⚠ «Очистити дані» аномалій (17.11).

Запуск: python3 test_db_admin.py
"""
import ast
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_APP = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
_ADM = open(os.path.join(_ROOT, 'templates', 'database_admin.html'), encoding='utf-8').read()
_BASE = open(os.path.join(_ROOT, 'templates', 'base.html'), encoding='utf-8').read()
_ADMIN = open(os.path.join(_ROOT, 'web', 'admin_pages.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
_DBO = open(os.path.join(_ROOT, 'storage', 'db_operations.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _ns():
    """Модульні словники й чисті функції flask_app — без імпорту Flask."""
    tree = ast.parse(_APP)
    keep = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') in (
                '_SERVICE_TABLES_TIME', '_RESEARCH_TABLES_TIME', '_DB_TABLE_LABELS') for t in n.targets):
            keep.append(n)
        if isinstance(n, ast.FunctionDef) and n.name in ('_db_table_kind', '_db_prune_sql', '_db_prune_table'):
            keep.append(n)
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'flask_app', 'exec'), ns)
    return ns


def test_single_classification():
    ns = _ns()
    k = ns['_db_table_kind']
    for t in ns['_SERVICE_TABLES_TIME']:
        _check(k(t) == 'service', f'{t} — службова')
    for t in ('sob_vol_anomaly', 'sob_trade_research', 'sob_ai_episode'):
        _check(k(t) == 'research', f'{t} — дослідження')
        _check(t not in ns['_SERVICE_TABLES_TIME'], f'{t} не чиститься автоочисткою')
    for t in ('sob_trades', 'sob_trade_archive', 'sob_bot_settings', 'sob_users'):
        _check(k(t) == 'data', f'{t} — дані стратегії')
    _check(k('sob_order_blocks') == 'service' and k('sob_readiness_log') == 'service',
           'сторінка і очистка класифікують однаково')
    _check(_APP.count('_db_table_kind(') >= 3, 'сторінка бере ту саму функцію')


def test_prune_sql_and_whitelist():
    ns = _ns()
    sql = ns['_db_prune_sql']
    _check("INTERVAL '30 days'" in sql('t', 'c', 'dt', 30), 'dt')
    _check('86400' in sql('t', 'c', 'sec', 7) and '*1000' in sql('t', 'c', 'ms', 7), 'sec / ms')
    _check(sql('t', 'c', 'dt', 7, wipe_all=True) == 'DELETE FROM t', 'усі')
    _check(sql('t', 'c', '??', 7) is None, 'невідомий тип → None')
    try:
        ns['_db_prune_table']('sob_trades', 1)
        raise AssertionError('таблиця поза білим списком мала відмовити')
    except ValueError:
        pass


def test_each_table_own_transaction_and_routes():
    _check('for tbl in _SERVICE_TABLES_TIME:' in _APP and 'engine.begin()' in _APP,
           'кожна таблиця — у своїй транзакції; автоочистка лише службові')
    _check("action in ('table_old', 'table_all')" in _APP, 'очистка однієї таблиці')
    _check("@app.route('/api/vol-anomaly/clear', methods=['POST'])" in _APP, 'маршрут очистки аномалій')
    _check('def clear_vol_anomalies' in _DBO, 'операція БД')
    _check("cleanupTable('sob_vol_anomaly', 'all')" in _ADM and 'db-chip research' in _ADM,
           'картка аномалій і чип «Дослідження» на сторінці адміністрування')
    _check('onclick="vanomClear();return false"' in _HTML and 'id="vanom-keep"' in _HTML,
           '«🗑 Очистити дані» і «Тримати N днів» у панелі аномалій')


def test_database_page_is_reachable_from_both_admin_navigations():
    _check("@app.route('/database')" in _APP and "render_template('database_admin.html')" in _APP,
           'маршрут і шаблон адміністрування БД')
    _check('{% extends "base.html" %}' in _ADM and '{% block content %}' in _ADM,
           'сторінка БД використовує спільний дизайн бота')
    _check("url_for('database_admin')" in _BASE, 'посилання DB у головній навігації')
    _check("('/database', '🗄 База даних', 'db')" in _ADMIN,
           'посилання на БД у навігації керування користувачами')


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)):
        try:
            fn()
            print('✓', name)
        except Exception as e:
            fails += 1
            print('✗', name, '—', repr(e))
    print('OK' if not fails else f'FAIL {fails}')
    sys.exit(1 if fails else 0)
