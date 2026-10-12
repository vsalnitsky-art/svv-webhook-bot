"""✋ РУЧНЕ ВІДКРИТТЯ БЕЗ ВОРІТ (вимога 12.10).

Дослівно: «Щойно хотів відкрити дві угоди в ручному режимі. Було відхилення.
Зроби при ручному відкритті щоб не було таких перевірок, це є ручне, тому не
має бути таких блокувань.»

Лог: ✋ групове відкриття NEARUSDT/ZECUSDT LONG → «відхилено на рівні `_open`»
без причини. Ручний шлях (`by_hand=True`) усе ще проходив повторну перевірку
Forecast/Decision, WAIT і головні кнопки напрямку (у FF і в TM).
"""
import os
import sys
import threading
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований імпорт fuel_filter)

FF = _mon.FF
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _TM:
    def __init__(self):
        self._positions, self._shadow_positions = {}, {}
        self.calls = []

    def get_settings(self):
        return {'enabled': False, 'test_mode': True}

    def _open_shadow(self, symbol, side, entry, tag, bypass_gates=False, by_hand=False):
        self.calls.append({'symbol': symbol, 'by_hand': by_hand})
        if not by_hand:   # як справжній TM при вимкненій кнопці
            return {'ok': False, 'reason': f'{side} entries are disabled'}
        self._shadow_positions[symbol] = {'side': side}
        return {'ok': True}


class _Scanner:
    _settings = {'forecast_1h_filter_enabled': True, 'decision_filter_enabled': True}

    def _forecast_filter_allows(self, *a):
        return False

    def _forecast_strength_allows(self, *a):
        return False

    def _decision_filter_allows(self, *a):
        return False


def _install(direction_on=False):
    dg = types.ModuleType('detection.direction_gate')
    dg.live_settings = lambda: {'x': 1}
    dg.allows = lambda s, side: direction_on
    dg.reason = lambda s, side: f'🚦 Кнопка {side} вимкнена'
    sys.modules['detection.direction_gate'] = dg
    _mon._pkg.direction_gate = dg
    sc = types.ModuleType('detection.smc_scanner')
    sc.get_smc_scanner = lambda: _Scanner()
    sys.modules['detection.smc_scanner'] = sc
    _mon._pkg.smc_scanner = sc
    _mon._install_log()


def _ff():
    ff = FF.__new__(FF)
    ff._lock = threading.RLock()
    ff._engine_skip = {}
    ff._fuel_managed = {}
    ff._persist_state = lambda: None
    ff._new_situation_logged = None
    tm = _TM()
    ff._get_tm = lambda: tm
    ff._exhaustion = lambda *a: 99.0
    ff._is_wait_verdict = lambda *a: True
    ff.manual_close_block = lambda *a: (True, 30.0)
    ff._soft_safeguard = lambda *a, **k: (False, 'слабкий МММ')
    return ff, tm


_S = {'max_exhaustion_pct': 75, 'skip_wait_coins': True, 'safeguard_on': True}


def test_by_hand_opens_despite_every_gate():
    _install(direction_on=False)
    ff, tm = _ff()
    ok = ff._open('NEARUSDT', 'LONG', {'mark_price': 2.5}, dict(_S), by_hand=True)
    _check(ok, f'ручне відкриття мусило пройти: {ff._engine_skip}')
    _check(tm.calls and tm.calls[0]['by_hand'] is True, f'TM отримав by_hand: {tm.calls}')
    print('✓ ✋ ручне відкриття проходить попри кнопку, Forecast, WAIT, виснаженість')


def test_automatic_open_is_still_gated():
    _install(direction_on=False)
    ff, tm = _ff()
    ok = ff._open('NEARUSDT', 'LONG', {'mark_price': 2.5}, dict(_S))
    _check(not ok and not tm.calls, 'автоматичне відкриття ворота й далі тримають')
    _check('Кнопка LONG' in ff._engine_skip.get('NEARUSDT', ''), ff._engine_skip)
    _install(direction_on=True)
    ff, tm = _ff()
    _check(not ff._open('NEARUSDT', 'LONG', {'mark_price': 2.5}, dict(_S)),
           'авто: виснаженість/Forecast теж ріжуть')
    print('✓ автоматичні відкриття — без змін (ворота діють)')


def test_failure_reason_is_named_not_guessed():
    _install(direction_on=True)
    ff, tm = _ff()
    ff._get_tm = lambda: None
    _check(not ff._open('ZECUSDT', 'LONG', {'mark_price': 300}, dict(_S), by_hand=True), '')
    _check(ff._engine_skip.get('ZECUSDT') == 'Trade Manager недоступний', ff._engine_skip)
    src = _mon._SRC
    _check('напрямок вимкнено кнопкою / ціна / розмір' not in src,
           'загадкова «кнопка / ціна / розмір» замість реальної причини')
    print('✓ відмова ручного відкриття називає СПРАВЖНЮ причину')


def test_tm_master_switch_skipped_only_by_hand():
    _check(_TM_SRC.count('if not by_hand and not self._side_allowed(side):') == 2,
           'real + paper: кнопки знімає ЛИШЕ by_hand')
    _check('if bypass_gates and not by_hand and self._reentry_blocked(symbol):' in _TM_SRC,
           'кулдаун повторного входу — не для ручного')
    print('✓ TM: головні кнопки обходить лише ✋ by_hand (не bypass_gates черг)')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
