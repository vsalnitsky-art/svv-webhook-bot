"""📌 Manual на угоді = ЖОДЕН автоматичний важіль її не чіпає (вимога 24.09).

Кейс зі скріна: CCUSDT і STXUSDT стояли з 📌, а їх закрили «🧮 МММ LiQ ⚖» і
«🧮 МММ LiQ ПРОТИ». Причина: правила виходу за вердиктом і автопілот стояли в
моніторах ДО гейта `manual_mode`, тож 📌 їх не зупиняв.

Лишаються ЛИШЕ дії людини: ручні Manual SL / TP-1 / TP-2, озброєний вручну
🟰 беззбиток і кнопка «Закрити».
"""
import ast
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _base   # той самий ізольований завантажувач TM

TM = _base.TM
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_FF_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _fn_src(src, name):
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(src, n)
    raise AssertionError(f'функцію {name} не знайдено')


def _monitor_tm(pos, is_shadow):
    """TM, у якого КОЖЕН важіль лише записує, що його викликали."""
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t._positions, t._shadow_positions = {}, {}
    t._pos_state, t._shadow_pos_state = {}, {}
    (t._shadow_positions if is_shadow else t._positions)['XUSDT'] = pos
    calls = []
    t._get_current_price = lambda s: 100.0
    t._ctr_snapshot = lambda s: {'x': 1}
    t.compute_decision = lambda *a, **k: {'x': 1}
    t._auto_ob_manual_sl = lambda *a, **k: calls.append('autosl')
    t._check_signal_exits = lambda *a, **k: calls.append('signal_exits') or False
    t._pilot_tick = lambda *a, **k: calls.append('pilot') or False
    t._check_manual_tp1 = lambda *a, **k: calls.append('tp1')
    t._check_manual_sl_tp = lambda *a, **k: calls.append('manual_sltp') or None
    t._check_breakeven_close = lambda *a, **k: calls.append('be_close') or False
    t._bot_soft_exit = lambda *a, **k: calls.append('soft') or None
    t._ctr_reversal_eval = lambda *a, **k: (0, False, 0)
    t._rev_persisted = lambda *a, **k: calls.append('ctr_rev') or False
    t._settings = {}
    return t, calls


def _pos(manual):
    return {'symbol': 'XUSDT', 'side': 'LONG', 'entry_price': 100.0,
            'opened_at': 0, 'manual_mode': manual}


# ═════════════════════════ ПОВЕДІНКА МОНІТОРІВ ══════════════════════════════
def test_manual_trade_skips_every_automatic_lever_in_both_books():
    for is_shadow in (False, True):
        t, calls = _monitor_tm(_pos(True), is_shadow)
        (t._monitor_shadow_position if is_shadow else t._monitor_position)('XUSDT')
        for lever in ('autosl', 'signal_exits', 'pilot', 'soft', 'ctr_rev'):
            _check(lever not in calls,
                   f'{"paper" if is_shadow else "real"}: 📌 угоду мав пропустити «{lever}»: {calls}')
        for human in ('tp1', 'manual_sltp', 'be_close'):
            _check(human in calls,
                   f'{"paper" if is_shadow else "real"}: ручне «{human}» мусить працювати: {calls}')
    print('✓ 📌 Manual: автоматика пропущена, ручні рівні працюють (real + paper)')


def test_normal_trade_still_gets_every_lever():
    for is_shadow in (False, True):
        t, calls = _monitor_tm(_pos(False), is_shadow)
        (t._monitor_shadow_position if is_shadow else t._monitor_position)('XUSDT')
        for lever in ('autosl', 'signal_exits', 'pilot', 'tp1', 'manual_sltp'):
            _check(lever in calls, f'без 📌 «{lever}» мусить працювати як раніше: {calls}')
    print('✓ без 📌 поведінка не змінилась')


# ═════════════════════ РІВНІ: БОТ НЕ ЗМІНЮЄ, ЛЮДИНА — ТАК ═══════════════════
class _StoreTM(TM):
    def __init__(self, manual):
        self._lock = threading.RLock()
        self._positions = {'BTCUSDT': {'symbol': 'BTCUSDT', 'side': 'SHORT',
                                       'entry_price': 100.0, 'manual_mode': manual}}
        self._shadow_positions = {}
    def _get_current_price(self, symbol): return 100.0
    def _persist_positions(self): pass
    def _persist_shadow_positions(self): pass


def test_bot_cannot_move_levels_of_a_manual_trade():
    t = _StoreTM(True)
    r = t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0, origin='auto',
                              origin_label='Автопілот · структура')
    _check(not r.get('ok') and r.get('manual_lock'), f'бот не мав змінити рівень: {r}')
    _check('manual_sl' not in t._positions['BTCUSDT'], 'рівень не мав зʼявитись')
    r = t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0)        # людина
    _check(r.get('ok'), f'людина свій рівень ставить завжди: {r}')
    print('✓ 📌 Manual: бот рівні не змінює, людина — так')


def test_bot_levels_on_a_normal_trade_are_untouched():
    t = _StoreTM(False)
    r = t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0, origin='auto')
    _check(r.get('ok'), f'без 📌 бот ставить рівні як раніше: {r}')
    print('✓ без 📌 бот ставить рівні як раніше')


def test_public_reader_matches_the_rule():
    t = _StoreTM(True)
    _check(t.is_manual_locked('BTCUSDT') is True, 'real 📌')
    _check(t.is_manual_locked('BTCUSDT', is_shadow=True) is False, 'в paper угоди немає')
    _check(TM._manual_locked(None) is False and TM._manual_locked({}) is False, 'порожнє')
    print('✓ is_manual_locked — одне правило')


# ═══════════════════════ ЗАМКИ НА КОЖЕН ШЛЯХ ═════════════════════════════════
def test_opposite_ob_exit_respects_manual():
    _check('_manual_locked(pos)' in _fn_src(_TM_SRC, 'on_main_ob_update'),
           'Opposite OB exit мусить пропускати 📌 угоди')
    print('✓ Opposite OB — 📌 пропускається')


def test_every_fuel_filter_auto_close_respects_manual():
    _check('_tm_manual_locked(symbol, is_real)' in _fn_src(_FF_SRC, '_close'),
           'FF _close (fuel_flipped/faded/potential…) мусить пропускати 📌')
    _check('_tm_manual_locked' in _fn_src(_FF_SRC, '_reverse_close_opposite'),
           'реверс Черги-2 мусить пропускати 📌')
    _check("p.get('manual_mode')" in _FF_SRC.split('def _close_book')[1][:900],
           'фліп ₿-сеансу мусить пропускати 📌')
    _check('is_manual_locked' in _fn_src(_FF_SRC, '_tm_manual_locked'),
           'FF читає ПУБЛІЧНЕ правило TM, а не свою копію')
    print('✓ Fuel Filter: усі автоматичні закриття пропускають 📌')


def test_gate_stands_before_the_levers_in_both_monitors():
    for fn in ('_monitor_position', '_monitor_shadow_position'):
        src = _fn_src(_TM_SRC, fn)
        g = src.index('_manual_locked(pos)')
        for lever in ('_auto_ob_manual_sl(', '_check_signal_exits(', '_pilot_tick('):
            _check(g < src.index(lever), f'{fn}: гейт 📌 мусить стояти ПЕРЕД {lever}')
    print('✓ гейт стоїть перед усіма автоматичними важелями')


if __name__ == '__main__':
    _fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
