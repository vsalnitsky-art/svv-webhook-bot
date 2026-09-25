"""📨 «▶️ ВІДКРИТО» більше не пише «SL: null» (кейс DASHUSDT 25.09).

Угоди з черг / ✋ групового відкриття отримують стоп від Fuel Filter ПІСЛЯ
відкриття (1-3 с), тож повідомлення тепер ЧЕКАЄ стопа і несе його таймфрейм.
"""
import ast, os, sys, threading
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import test_autosl_chain as _base
TM = _base.TM
_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _tm():
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t.sent = []
    t._notify = lambda msg, is_test=False, category=None: t.sent.append((msg, is_test))
    t._fmt_price = lambda v: f'${float(v):.4f}'
    t._tp_lines = lambda pos: ''
    t._dir_dot = lambda side: '🟢' if side == 'LONG' else '🔴'
    return t


def _pos(**kw):
    p = {'symbol': 'DASHUSDT', 'side': 'LONG', 'entry_price': 63.93}
    p.update(kw)
    return p


def test_no_stop_yet_means_the_notice_waits():
    t = _tm(); p = _pos()
    _check(t._notify_open(p, is_test=True, now=1000.0) is False, 'без стопа — чекаємо')
    _check(t.sent == [] and p.get('_open_notice_at') == 1000.0, f'{t.sent} {p}')
    print('✓ без стопа повідомлення не йде з null, а чекає')


def test_the_stop_from_fuel_filter_arrives_with_its_timeframe():
    t = _tm(); p = _pos()
    t._notify_open(p, is_test=True, now=1000.0)
    _check(t._flush_open_notice(p, True, now=1001.0) is False, 'стопа ще нема — тримаємо')
    p['manual_sl'] = 61.888; p['manual_sl_tf'] = '1H'      # як ставить Черга-4
    _check(t._flush_open_notice(p, True, now=1002.0) is True, 'стоп зʼявився — шлемо')
    msg, test = t.sent[0]
    _check('SL: <b>$61.8880 · 1H</b>' in msg and '🧪 ТЕСТ' in msg and test, msg)
    _check('_open_notice_at' not in p, 'позначка знята')
    _check(t._flush_open_notice(p, True, now=1003.0) is False and len(t.sent) == 1,
           'рівно ОДНЕ повідомлення')
    print('✓ стоп від Черги-4 доїжджає в TG разом із TF')


def test_timeout_says_not_set_instead_of_null():
    t = _tm(); p = _pos()
    t._notify_open(p, now=1000.0)
    t._flush_open_notice(p, False, now=1000.0 + TM.OPEN_NOTICE_WAIT_SEC)
    msg = t.sent[0][0]
    _check('не виставлено' in msg and 'null' not in msg, msg)
    print('✓ стопа так і нема → «не виставлено», а не null')


def test_stop_already_set_goes_immediately_and_percent_is_named():
    t = _tm(); p = _pos(manual_sl=62.65, manual_sl_tf='%')
    _check(t._notify_open(p, now=1.0) is True, 'стоп є — одразу')
    _check('% від входу' in t.sent[0][0], t.sent[0][0])
    t2 = _tm(); p2 = _pos(sl_price=60.0)
    t2._notify_open(p2)
    _check('SL: <b>$60.0000</b>' in t2.sent[0][0], 'стратегічний SL — без TF')
    print('✓ готовий стоп — одразу; % і стратегічний SL підписані чесно')


def test_both_monitors_flush_and_both_books_use_one_text():
    for fn in ('_monitor_position', '_monitor_shadow_position'):
        tree = ast.parse(_SRC)
        body = next(ast.get_source_segment(_SRC, n) for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == fn)
        _check('_flush_open_notice' in body, f'{fn} мусить досилати відкладене')
    _check(_SRC.count('f"▶️ ВІДКРИТО') == 1, 'текст відкриття — в ОДНОМУ місці')
    print('✓ обидва монітори досилають, текст один на обидві книги')


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for f in fns:
        f()
    print(f'\n{len(fns)}/{len(fns)} passed')
