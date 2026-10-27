"""✋ ЛЮБЕ РУЧНЕ ВІДКРИТТЯ НЕ БЛОКУЄТЬСЯ (вимога 25.10) + чому хедж ще ні.

Дослівно: «Якщо відкриття угоди відбувається в ручному режимі, нічого не має
блокувати відкриття цієї угоди. Взагалі, любе ручне відкриття не має
блокуватись.»

Що блокувало ручне відкриття до правки:
  • кнопки ⬆LONG/⬇SHORT банера Decision кликали `manual_open` БЕЗ `by_hand` →
    🚦 кнопки напрямку, а при увімкненому FF — монета йшла в ЧЕРГУ замість угоди;
  • стеля `max_open_positions` (ручне падало в paper або відмовляло);
  • ❤️ «Fuel-confirm» у paper-книзі;
  • майстер-тумблер ❤️ Fuel Auto-Filter у ✋ груповому / ✋ Черга-4 / ✋ таймер;
  • кнопки ⬆/⬇ гасились при вимкненому TM, хоча Test Mode готовий прийняти paper.
Лишаються ЛИШЕ технічні умови: ціна, розмір, біржа, наявність книги — і
друга угода по ТІЙ САМІЙ монеті (хедж): книги TM ключовані символом, а Bybit
у one-way режимі закрив би наявну позицію зустрічним ордером.
"""
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _tmb  # noqa: E402  (ізольований trade_manager)

TM = _tmb.TM
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_FF_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'), encoding='utf-8').read()
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _fn(src, name, nxt='\n    def '):
    i = src.index(f'def {name}(')
    j = src.find(nxt, i + 10)
    return src[i:j if j > 0 else None]


class _Bybit:
    api_key = 'k'


def _tm(at_limit=True, enabled=True, test_mode=True):
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t._positions, t._shadow_positions = {}, {}
    t._settings = {'test_mode': test_mode, 'max_open_positions': 1}
    t.bybit = _Bybit()
    t.calls = []
    t._at_max_positions = lambda: at_limit
    t._reentry_blocked = lambda s: True
    t.is_enabled = lambda: enabled
    t._get_current_price = lambda s: 100.0
    t._is_tradeable = lambda s: True

    def _open_position(symbol, side, price, opened_by=None, bypass_gates=False,
                       manual_sl=None, manual_tp=None, by_hand=False):
        t.calls.append(('real', symbol, side, by_hand))
        t._positions[symbol] = {'side': side}
        return {'ok': True}

    def _open_shadow(symbol, side, price, opened_by=None, bypass_gates=False,
                     manual_sl=None, manual_tp=None, by_hand=False):
        t.calls.append(('paper', symbol, side, by_hand))
        t._shadow_positions[symbol] = {'side': side}
        return {'ok': True}
    t._open_position, t._open_shadow = _open_position, _open_shadow
    return t


def test_manual_open_ignores_the_position_cap():
    # ⚠️ ПЕРЕПИСАНО 27.10 (вимога змінилась): «real = TRD + TM + стеля дозволяє,
    # інакше Test Mode» — для ✋ ручних ТЕЖ. Ручне не відмовляє, а йде в paper.
    t = _tm(at_limit=True)
    r = TM.manual_open(t, 'BTCUSDT', 'LONG', by_hand=True)
    _check(r.get('ok') and r.get('mode') == 'test', r)
    _check(t.calls == [('paper', 'BTCUSDT', 'LONG', True)], t.calls)
    a = _tm(at_limit=False)
    r = TM.manual_open(a, 'BTCUSDT', 'LONG', by_hand=True)
    _check(r.get('mode') == 'real', f'стеля дозволяє → реальна: {r}')
    print('✓ стеля max_open_positions: ручне на стелі → paper (правило 27.10)')


def test_manual_open_never_goes_to_a_queue():
    body = _fn(_TM_SRC, 'manual_open')
    _check('if not bypass_gates and not by_hand:' in body,
           'ручне відкриття не має ставати в чергу FF')
    _check('if bypass_gates and not by_hand and self._reentry_blocked' in body,
           'кулдаун повторного входу — лише для авто')
    print('✓ ручне відкриття не стає в чергу і не чекає кулдауну')


def test_decision_buttons_are_manual():
    i = _FLASK.index('def api_tm_open_position(')
    body = _FLASK[i:_FLASK.index('@app.route', i)]
    _check('by_hand=True' in body, '⬆LONG/⬇SHORT мусять іти як ручне відкриття')
    print('✓ кнопки ⬆LONG/⬇SHORT — ручне відкриття')


def test_paper_fuel_confirm_does_not_block_hands():
    body = _fn(_TM_SRC, '_open_shadow')
    _check('not bypass_gates and not by_hand' in body, 'Fuel-confirm різав ручне paper')
    print('✓ ❤️ Fuel-confirm не ріже ручне paper-відкриття')


def test_ff_master_switch_does_not_block_manual_paths():
    for name in ('group_open', 'force_open_timer', 'force_open_queue4'):
        body = _fn(_FF_SRC, name)
        _check("Fuel Auto-Filter вимкнено'" not in body
               and 'Fuel Auto-Filter is disabled' not in body,
               f'{name}: майстер-тумблер FF блокує ручне')
    print('✓ майстер-тумблер ❤️ FF не блокує ✋ групове / Черга-4 / таймер')


def test_buttons_stay_live_in_test_mode():
    i = _HTML.index('function updateManualEntryButtons(')
    body = _HTML[i:_HTML.index('\nfunction ', i + 10)]
    _check('if (!tmEnabled && !testMode)' in body,
           'кнопки гасимо лише коли немає ЖОДНОЇ книги')
    _check("window._tmSettings.test_mode" in body, 'Test Mode не читається')
    print('✓ при вимкненому TM кнопки ⬆/⬇ відкривають paper')


def test_same_symbol_second_trade_is_refused_with_an_honest_reason():
    t = _tm(at_limit=False)
    t._positions['BTCUSDT'] = {'side': 'LONG'}
    r = TM.manual_open(t, 'BTCUSDT', 'SHORT', by_hand=True)
    _check(not r.get('ok') and 'Хедж' in r.get('reason', ''), r)
    _check(t.calls == [], 'зустрічний ордер у one-way режимі закрив би позицію')
    i = _FF_SRC.index('if by_hand and (self._tm_has_position(symbol, True)')
    _check('return False' in _FF_SRC[i:i + 600],
           'FF не має мовчки «приймати в облік» наявну позицію як ручне відкриття')
    print('✓ друга угода по тій самій монеті — чесна відмова (хедж ще не підтримується)')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
