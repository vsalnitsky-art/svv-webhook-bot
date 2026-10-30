"""🔒 Угоду, яку бот САМ щойно відкрив, звірка з біржею не «адоптує» як
зовнішню (кейс ORDIUSDT 29.10: 1:50 «Adopted external position», 1:51
«ВІДКРИТО» — між ордером і записом у книгу ідуть Decision/POC/знімок)."""
import sys
sys.path.insert(0, '.')
import test_bybit_reconcile as base

LIVE = [{'symbol': 'ORDIUSDT', 'side': 'Sell', 'size': 27.24,
         'entry_price': 3.666, 'stop_loss': 0, 'take_profit': 0}]


def _tm():
    tm, bybit = base._make_tm([])
    adopted = []
    tm._adopt_external_position = lambda p: adopted.append(p['symbol']) or True
    bybit.get_positions_checked = lambda: (list(LIVE), True)
    return tm, adopted


def test_symbol_being_opened_is_not_adopted():
    tm, adopted = _tm()
    tm._opening['ORDIUSDT'] = base.tm_mod.time.time()
    tm._reconcile_with_bybit()
    assert adopted == [], adopted
    print('✓ монета, яку бот саме відкриває, не стає «зовнішньою»')


def test_really_external_position_is_still_adopted():
    tm, adopted = _tm()
    tm._reconcile_with_bybit()
    assert adopted == ['ORDIUSDT']
    print('✓ справді зовнішня позиція і далі береться під облік')


def test_stale_marker_does_not_hide_forever():
    tm, adopted = _tm()
    tm._opening['ORDIUSDT'] = base.tm_mod.time.time() - tm.OPENING_GUARD_SEC - 1
    tm._reconcile_with_bybit()
    assert adopted == ['ORDIUSDT']
    print('✓ зависла позначка не ховає позицію назавжди')


def test_marker_is_set_around_the_order_and_cleared():
    src = open('detection/trade_manager.py', encoding='utf-8').read()
    i = src.index('def _open_position(')
    body = src[i:src.index('def _open_position_placed(', i)]
    assert body.index('self._opening[symbol]') < body.index('_open_position_placed(')
    assert 'finally:' in body and 'self._opening.pop(symbol' in body
    print('✓ позначка ставиться ДО ордера і знімається у finally')


if __name__ == '__main__':
    for k, f in list(globals().items()):
        if k.startswith('test_'):
            f()
