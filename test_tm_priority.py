"""🏦 Trade Manager має ПРІОРИТЕТ над Test Mode (вимога 27.10).

Дослівно: «при увімкненому тумблері Trade Manager не звертати увагу на
відкриті угоди Test Mode, якщо позволяє 📊 Max Open Positions: то відкриваємо
паралельно реальні угоди» + «Trade Manager — завжди має пріоритет».
"""
import os
import sys
import threading
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _tmb  # noqa: E402

TM = _tmb.TM
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_FF_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _tm(real=(), paper=(), enabled=True, trd=True, max_pos=10):
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t._positions = {s: {'side': 'LONG'} for s in real}
    t._shadow_positions = {s: {'side': 'LONG'} for s in paper}
    t._settings = {'max_open_positions': max_pos, 'enabled': enabled}
    t.is_enabled = lambda: enabled
    t._is_tradeable = lambda sym: trd
    return t


def test_paper_does_not_block_real_when_real_book_is_available():
    t = _tm(paper=['XUSDT'])
    _check(TM.real_open_possible(t, 'XUSDT'), 'реальна книга доступна')
    _check(not TM.blocks_new_open(t, 'XUSDT'), 'paper не блокує реальну')
    print('✓ paper-угода не блокує реальну, коли TM ✓ · TRD ✓ · є ліміт')


def test_paper_blocks_when_real_book_is_unavailable():
    for kw in ({'enabled': False}, {'trd': False}, {'max_pos': 1, 'real': ['AUSDT']}):
        t = _tm(paper=['XUSDT'], **kw)
        _check(TM.blocks_new_open(t, 'XUSDT'), f'реальна недоступна ({kw}) → paper блокує')
    print('✓ реальна недоступна (TM / TRD / ліміт) → paper блокує як і раніше')


def test_real_always_blocks():
    t = _tm(real=['XUSDT'])
    _check(TM.blocks_new_open(t, 'XUSDT'), 'реальна позиція блокує завжди')
    _check(not TM.blocks_new_open(t, 'YUSDT'), 'чужа монета не блокує')
    print('✓ реальна позиція по монеті блокує завжди')


def test_manual_open_blocks_only_within_the_same_book():
    i = _TM_SRC.index('    def manual_open(')
    body = _TM_SRC[i:_TM_SRC.index('\n    def ', i + 10)]
    _check("book == 'real' and symbol in self._positions" in body, 'real блокує лише real')
    _check("book == 'paper' and symbol in self._shadow_positions" in body, 'paper блокує лише paper')
    print('✓ manual_open: позиція блокує лише свою книгу')


def test_on_signal_ignores_paper_when_real_possible():
    i = _TM_SRC.index('    def on_signal(')
    body = _TM_SRC[i:_TM_SRC.index('\n    def ', i + 10)]
    _check('self.real_open_possible(symbol)' in body, 'on_signal питає доступність реальної книги')
    _check('existing_real or existing_shadow_for_route' in body, 'маршрут без paper при TM-пріоритеті')
    print('✓ on_signal: paper не маршрутизує сигнал, коли реальна доступна')


def test_fuel_filter_uses_single_rule_everywhere():
    _check('def _tm_blocks_open' in _FF_SRC and 'blocks_new_open' in _FF_SRC, 'хелпер FF')
    # Подвійна перевірка «real АБО paper» у воротах відкриття більше не лишилась.
    for name in ('_engine_tick_queue4', '_engine_tick_readiness', 'force_open_queue4', 'intercept'):
        i = _FF_SRC.index(f'    def {name}(')
        body = _FF_SRC[i:_FF_SRC.index('\n    def ', i + 10)]
        _check('_tm_blocks_open(' in body, f'{name} мусить питати _tm_blocks_open')
    print('✓ Fuel Filter: одне правило в intercept / Q3 / Q4 / ✋ Черга-4')


def test_ff_helper_honours_tm_decision():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'fuel_filter_prio_test', os.path.join(_HERE, 'detection', 'fuel_filter.py'))
    ff_mod = importlib.util.module_from_spec(spec)
    sys.modules['fuel_filter_prio_test'] = ff_mod
    spec.loader.exec_module(ff_mod)
    FF = ff_mod.FuelFilterDaemon
    ff = FF.__new__(FF)
    t = _tm(paper=['XUSDT'])
    ff._get_tm = lambda: t
    ff._fuel_managed = {'XUSDT': {'mode': 'paper'}}
    _check(not FF._tm_blocks_open(ff, 'XUSDT'), 'FF-позначка paper не блокує реальну')
    t2 = _tm(paper=['XUSDT'], trd=False)
    ff._get_tm = lambda: t2
    _check(FF._tm_blocks_open(ff, 'XUSDT'), 'TRD вимкнено → блокує')
    ff._get_tm = lambda: types.SimpleNamespace(_positions={}, _shadow_positions={'XUSDT': {}})
    _check(FF._tm_blocks_open(ff, 'XUSDT'), 'старіший TM без методу → стара поведінка')
    print('✓ хелпер FF іде за рішенням TM, фолбек — стара поведінка')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
