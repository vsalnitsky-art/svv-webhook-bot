"""✋ Manual SL — РУЧНЕ КЕРУВАННЯ + 🏦 рівні реальних угод ЛИШЕ в боті (12.10).

Вимога дослівно: «1. Я не можу видалити Manual SL. Має бути ручне керування,
його немає. 2. Для реальних угод Manual SL і Manual TP-1 Manual TP-2 значення
мають бути в боті, на біржі має бути лише відкрита угода».

Корінь п.1: очищене поле SL через ~4с заповнювала гарантія «🛡 Стоп ЗАВЖДИ»
(`_auto_ob_manual_sl`) або трейл автопілота — видалити рівень було неможливо.
"""
import ast
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _base

TM = _base.TM
_StoreTM = _base._StoreTM
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_BB_SRC = open(os.path.join(_HERE, 'core', 'bybit_connector.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _fn_src(src, name):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise AssertionError(f'функцію {name} не знайдено')


# ═══════════════════════ ✋ ЗНЯТТЯ Manual SL ════════════════════════════════
def test_user_clear_sets_the_lock_and_user_set_removes_it():
    t = _StoreTM('SHORT', 100.0)
    t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0, origin='auto')
    t.update_manual_sl_tp('BTCUSDT', manual_sl=0)            # руками зняли
    p = t._positions['BTCUSDT']
    _check('manual_sl' not in p, 'рівень мусить зникнути')
    _check(TM._sl_cleared_by_user(p), 'зняття оператором ставить замок')
    t.update_manual_sl_tp('BTCUSDT', manual_sl=108.0)        # руками вписали
    _check(p.get('manual_sl') == 108.0, 'ручний рівень мусить стати')
    _check(not p.get('sl_cleared_by_user'), 'ручний рівень знімає замок')
    print('✓ ручне зняття ставить замок, ручне значення його знімає')


def test_bot_cannot_put_the_cleared_sl_back():
    t = _StoreTM('SHORT', 100.0)
    t.update_manual_sl_tp('BTCUSDT', manual_sl=0)
    r = t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0, origin='auto',
                              origin_label='Автопілот · структура')
    p = t._positions['BTCUSDT']
    _check(not r.get('ok') and r.get('sl_user_cleared'), f'бот мав отримати відмову: {r}')
    _check('manual_sl' not in p, f'поле мусить лишитись порожнім: {p}')
    print('✓ бот не повертає знятий оператором SL')


def test_bot_tp_still_applies_when_sl_is_locked():
    t = _StoreTM('SHORT', 100.0)
    t.update_manual_sl_tp('BTCUSDT', manual_sl=0)
    t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0, manual_tp=90.0, origin='auto')
    p = t._positions['BTCUSDT']
    _check('manual_sl' not in p, 'SL лишається порожнім')
    _check(p.get('manual_tp') == 90.0, 'TP того самого виклику застосовується')
    print('✓ замок SL не блокує TP того самого виклику')


def test_auto_sl_guarantee_respects_the_lock():
    _base._reset()
    _base._OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    p = _base._pos()
    p['sl_cleared_by_user'] = True
    _base._tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check('manual_sl' not in p, f'гарантія стопа не має повертати рівень: {p}')
    # Без замка той самий виклик стоп СТАВИТЬ — тест справжній.
    p2 = _base._pos()
    _base._tm()._auto_ob_manual_sl('MNTUSDT', p2, 0.51430)
    _check(p2.get('manual_sl'), 'контроль: без замка стоп ставиться')
    print('✓ «🛡 Стоп ЗАВЖДИ» не перекриває ручне зняття')


def test_autopilot_trail_checks_the_lock_before_writing():
    src = _fn_src(_TM_SRC, '_pilot_tick')
    i = src.index('self._sl_cleared_by_user(pos)')
    j = src.index("origin_label='Автопілот · структура'")
    _check(i < j, 'перевірка замка мусить стояти ДО запису трейлу')
    print('✓ трейл автопілота перевіряє замок до запису')


# ═══════════════════════ 🏦 БІРЖА ТРИМАЄ ЛИШЕ ПОЗИЦІЮ ═══════════════════════
def test_real_open_sends_no_sl_tp_to_exchange():
    # Ордер і запис позиції живуть у `_open_position_placed` (з 30.10 —
    # `_open_position` лише ставить позначку «відкриваємо» навколо них).
    src = _fn_src(_TM_SRC, '_open_position_placed')
    k = src.index('self.bybit.place_order(')
    call = src[k:src.index(')', src.index("order_type='Market'", k))]
    _check('stop_loss' not in call and 'take_profit' not in call,
           f'place_order не має нести SL/TP: {call}')
    _check("position['exchange_sltp'] = False" in src, 'нова позиція позначена')
    print('✓ реальне відкриття не шле SL/TP на біржу')


def test_exchange_sl_update_is_a_noop():
    called = []

    class _BB:
        def set_trading_stop(self, **kw):
            called.append(kw)
            return True
    t = TM.__new__(TM)
    t.bybit = _BB()
    t._update_exchange_sl('BTCUSDT', 101.0)
    _check(not called, f'біржу не чіпаємо: {called}')
    print('✓ _update_exchange_sl більше нічого не шле')


def _strip_tm(ok=True):
    calls = []

    class _BB:
        def set_trading_stop(self, **kw):
            calls.append(kw)
            return ok
    t = TM.__new__(TM)
    t.bybit = _BB()
    t._persist_positions = lambda: None
    return t, calls


def test_legacy_position_gets_its_exchange_levels_stripped_once():
    _base._LOG.clear()
    t, calls = _strip_tm(True)
    pos = {'side': 'LONG', 'sl_price': 95.0, 'tp_price': 110.0, 'opened_by': 'choch'}
    t._strip_exchange_sltp('BTCUSDT', pos)
    t._strip_exchange_sltp('BTCUSDT', pos)
    _check(calls == [{'symbol': 'BTCUSDT', 'stop_loss': 0, 'take_profit': 0}],
           f'рівно один виклик скасування: {calls}')
    _check(pos['exchange_sltp'] is False, 'позначка після успіху')
    print('✓ легасі SL/TP знімаються з біржі один раз')


def test_strip_skips_new_and_external_positions():
    t, calls = _strip_tm(True)
    t._strip_exchange_sltp('A', {'sl_price': 1, 'exchange_sltp': False})
    t._strip_exchange_sltp('B', {'sl_price': 1, 'opened_by': 'external'})
    _check(not calls, f'нові й зовнішні не чіпаємо: {calls}')
    print('✓ нові та зовнішні позиції біржею не чіпаються')


def test_strip_failure_is_logged_and_retried_later():
    _base._LOG.clear()
    t, calls = _strip_tm(False)
    pos = {'side': 'LONG', 'sl_price': 95.0, 'opened_by': 'choch'}
    t._strip_exchange_sltp('BTCUSDT', pos)
    t._strip_exchange_sltp('BTCUSDT', pos)        # тротл 5 хв
    _check(len(calls) == 1, f'повтор не частіше ніж раз на 5 хв: {calls}')
    _check(pos.get('exchange_sltp') is not False, 'невдача не ставить позначку')
    _check('НЕ вдалось зняти SL/TP' in _base._text(), f'невдача в лозі: {_base._text()}')
    print('✓ відмова біржі видна в лозі й повторюється пізніше')


def test_connector_can_cancel_a_stop_with_zero():
    src = _fn_src(_BB_SRC, 'set_trading_stop')
    _check('if stop_loss is not None:' in src, 'нуль мусить доходити до біржі')
    print('✓ set_trading_stop пропускає stop_loss=0 (скасування)')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
