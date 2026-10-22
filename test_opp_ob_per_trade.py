"""🧱 Пер-угодний Opposite OB Exit + авто-вмикання сигналом 🚨 МММ-Decision (22.10).

Дослівно: «Додай ще колонку в таблиці відкритих угод з тумблером (такий як
"Manual 🎯 БЗ") він буде автоматично активувати алгоритм [Opposite OB Exit] по
монеті незалежно увімкнений він в налаштуваннях чи ні. Автоматичне ввімкнення
відбуватиметься в момент, коли сигнальне поле перейде в статус, який
відповідає, що угода має тенденцію до закриття. А якщо статус сигнального поля
покращився то тумблер відповідно автоматично вимикаємо.»
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_exit_rules as _er   # noqa: E402  (ізольований TM + стаби OB/логу)

TM = _er.TM
HOUR_MS = _er.HOUR_MS
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
_FF = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _tm(**kw):
    t = _er._reset(**kw)
    t._persist_positions = lambda: None
    t._persist_shadow_positions = lambda: None
    t._manual_locked = lambda pos: bool(pos.get('manual_mode'))
    return t


# ═══════════ 1. Тумблер угоди діє НЕЗАЛЕЖНО від глобального ═══════════════
def test_per_trade_flag_works_with_global_off():
    t = _tm(side='SHORT', use_opposite_ob_exit=False)
    _er._OB_ROWS['15m'] = _er._ob('BULLISH', 1_500_000_000)
    t.on_main_ob_update('MNTUSDT')
    _er._OB_ROWS['15m'] = _er._ob('BULLISH', 1_500_000_000 + HOUR_MS)
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [], 'глобально вимкнено і тумблера немає → не закриває')
    r = t.update_opp_ob_exit('MNTUSDT', True)
    _check(r['ok'] and t._positions['MNTUSDT']['opp_ob_exit_by'] == 'user', r)
    t.on_main_ob_update('MNTUSDT')                 # база = блок на момент ВМИКАННЯ
    _check(t.closed == [], 'блок, що вже висів до вмикання, — не подія')
    _er._OB_ROWS['15m'] = _er._ob('BULLISH', 1_500_000_000 + 2 * HOUR_MS)
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [('real', 'opposite_ob_exit')], t.closed)
    _check(any('тумблер угоди' in x for x in _er._LOG), _er._LOG)
    print('✓ тумблер угоди закриває на НОВОМУ протилежному OB при вимкненому глобальному')


def test_each_book_is_checked_apart():
    t = _tm(side='SHORT', use_opposite_ob_exit=False)
    paper = dict(t._positions['MNTUSDT'], side='LONG', opened_at=2000.0)
    t._shadow_positions['MNTUSDT'] = paper
    t.update_opp_ob_exit('MNTUSDT', True, is_shadow=True)
    _er._OB_ROWS['15m'] = _er._ob('BULLISH', 1_500_000_000)
    t.on_main_ob_update('MNTUSDT')
    _er._OB_ROWS['15m'] = _er._ob('BEARISH', 1_500_000_000 + HOUR_MS)
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [('paper', 'opposite_ob_exit')],
           f'лише паперова книга з тумблером: {t.closed}')
    _check('MNTUSDT|S' in t._opp_ob_base and 'MNTUSDT|R' not in t._opp_ob_base,
           t._opp_ob_base)
    print('✓ real і paper — окремі книги, окремі бази')


# ═══════════ 2. Автомат: перехід у «тенденцію до закриття» ════════════════
def test_auto_action_pure_rule():
    A = TM.opp_ob_auto_action
    _check(A(None, 'weaker', False, '') is True, 'перший показ уже поганий → увімкнути')
    _check(A('ok', 'against', False, '') is True, 'ok → проти → увімкнути')
    _check(A('ok', 'neutral', False, '') is True, 'ok → нейтраль → увімкнути')
    _check(A('weaker', 'against', True, 'auto') is None, 'вже увімкнено')
    _check(A('weaker', 'against', False, '') is None,
           'людина вимкнула посеред поганого стану — автомат не перебиває')
    _check(A('against', 'ok', True, 'auto') is False, 'покращення → вимкнути')
    _check(A('against', 'ok', True, 'user') is None, 'увімкнене людиною не вимикаємо')
    _check(A(None, 'ok', False, '') is None and A('ok', 'ok', True, 'auto') is None, 'ок → нічого')
    print('✓ правило автомата: лише переходи, рішення людини сильніше')


def test_apply_alert_toggles_and_logs():
    t = _tm(side='LONG', use_opposite_ob_exit=False)
    _check(t.apply_mmd_alert('MNTUSDT', False, {'level': 'ok'}) is None, 'ok')
    _check(t.apply_mmd_alert('MNTUSDT', False, {'level': 'weaker', 'text': 'слабшає'}) is True, '')
    p = t._positions['MNTUSDT']
    _check(p['opp_ob_exit'] and p['opp_ob_exit_by'] == 'auto' and p['mmd_alert_lvl'] == 'weaker', p)
    _check(any('🧠 МММ-Decision' in x and 'УВІМКНЕНО' in x for x in _er._LOG), _er._LOG)
    _check(t.apply_mmd_alert('MNTUSDT', False, {'level': 'against'}) is None, 'вже увімкнено')
    _check(t.apply_mmd_alert('MNTUSDT', False, {'level': 'ok'}) is False, 'покращення')
    _check('opp_ob_exit' not in p and any('ВИМКНЕНО' in x for x in _er._LOG), p)
    print('✓ 🚨 погіршення вмикає 🧱, покращення до ✓ вимикає (і це в лозі)')


def test_manual_lock_blocks_the_automat():
    t = _tm(side='LONG')
    t._positions['MNTUSDT']['manual_mode'] = True
    _check(t.apply_mmd_alert('MNTUSDT', False, {'level': 'against'}) is None, '📌')
    _check('opp_ob_exit' not in t._positions['MNTUSDT'], '📌 угоду автомат не чіпає')
    print('✓ 📌 Manual-угоду автомат не чіпає')


# ═══════════ 3. Проводка ═══════════════════════════════════════════════
def test_engine_feeds_the_same_alert_as_the_column():
    a = _FF.index('def _mmd_apply_trade_alerts(')
    fn = _FF[a:_FF.index('\n    def ', a + 10)]
    _check('dec_trade_for(' in fn and 'apply_mmd_alert(' in fn, 'одне джерело з колонкою 🚨')
    _check(_FF.count('self._mmd_apply_trade_alerts()') == 2,
           'автомат мусить працювати і при вимкненому блоці 🧠')
    print('✓ двигун подає той самий сигнал, що малює колонку 🚨')


def test_route_and_column():
    _check("@app.route('/api/tm/positions/opp-ob-exit'" in _FLASK and
           "origin='user'" in _FLASK, 'маршрут тумблера')
    _check(_HTML.count('${oppObExitCellHTML(p, false)}') == 1 and
           _HTML.count('${oppObExitCellHTML(p, true)}') == 1, 'колонка в обох таблицях')
    _check(_HTML.count('>🧱 OB</th>') == 2, 'заголовок в обох таблицях')
    _check("'/api/tm/positions/opp-ob-exit'" in _HTML, 'фронт шле на той самий маршрут')
    print('✓ маршрут + колонка «🧱 OB» у real і paper')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
