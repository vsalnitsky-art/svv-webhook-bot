"""🛡 Масове закриття на біржі (кейс 27.10): «закрив усі реальні угоди на
Bybit одночасно — бот далі відслідковує неіснуючі».

Корінь: запобіжник «≥5 і ≥50% книги зникли за раз = поганий фетч» пропускав
такий тік НАЗАВЖДИ — кожен наступний чистий фетч давав ту саму картину.
Тепер зникнення закривається після MASS_CLOSE_CONFIRM_FETCHES чистих фетчів
поспіль і MASS_CLOSE_CONFIRM_SEC секунд; разовий поганий фетч не доживає.
"""
import sys
sys.path.insert(0, '.')
import test_bybit_reconcile as base  # стаби модулів + TradeManager

TM = base.TradeManager


def _tm_with(n):
    tm, bybit = base._make_tm([])
    closed = []
    tm._close_externally = lambda s, p, reason='': (closed.append(s),
                                                     tm._positions.pop(s, None))
    tm._get_current_price = lambda s: 100.0
    tm._adopt_external_position = lambda p: False
    for i in range(n):
        tm._positions[f'C{i}USDT'] = {'symbol': f'C{i}USDT', 'side': 'LONG',
                                       'entry_price': 100.0, 'qty': 1}
    bybit.get_positions_checked = lambda: ([], True)
    return tm, closed


def test_pure_rule():
    N, T = TM.MASS_CLOSE_CONFIRM_FETCHES, TM.MASS_CLOSE_CONFIRM_SEC
    assert not TM.mass_close_confirmed([], 100)
    assert not TM.mass_close_confirmed([None], 100)
    assert not TM.mass_close_confirmed([(N - 1, 0)], 1000)      # мало фетчів
    assert not TM.mass_close_confirmed([(N, 990)], 1000)        # мало часу
    assert TM.mass_close_confirmed([(N, 1000 - T)], 1000)
    assert not TM.mass_close_confirmed([(N, 0), (1, 0)], 1000)  # один не дозрів
    print('✓ чисте правило підтвердження')


def test_flattening_everything_is_eventually_closed():
    tm, closed = _tm_with(8)
    clock = [1000.0]
    real_time = base.tm_mod.time.time
    base.tm_mod.time.time = lambda: clock[0]
    try:
        tm._reconcile_with_bybit()
        assert closed == [], 'перший масовий фетч — ще не закриваємо'
        clock[0] += 10; tm._reconcile_with_bybit()
        assert closed == []
        clock[0] += 15; tm._reconcile_with_bybit()   # 3-й фетч, 25с
        assert len(closed) == 8 and not tm._positions, closed
    finally:
        base.tm_mod.time.time = real_time
    print('✓ закриті на біржі всі угоди знімаються з обліку після підтвердження')


def test_one_bad_fetch_does_not_close():
    tm, closed = _tm_with(8)
    syms = list(tm._positions)
    clock = [1000.0]
    real_time = base.tm_mod.time.time
    base.tm_mod.time.time = lambda: clock[0]
    live = [{'symbol': s, 'side': 'Buy', 'size': 1} for s in syms]
    try:
        tm.bybit.get_positions_checked = lambda: ([], True)
        tm._reconcile_with_bybit()                    # глюк: порожньо
        tm.bybit.get_positions_checked = lambda: (live, True)
        for _ in range(4):
            clock[0] += 15; tm._reconcile_with_bybit()  # позиції знову є
        assert closed == [], closed
        assert not tm._missing_streak, 'серія скинута, коли позиції повернулись'
    finally:
        base.tm_mod.time.time = real_time
    print('✓ разовий поганий фетч нічого не закриває')


def test_small_close_is_still_immediate():
    tm, closed = _tm_with(8)
    live = [{'symbol': s, 'side': 'Buy', 'size': 1}
            for s in list(tm._positions)[1:]]
    tm.bybit.get_positions_checked = lambda: (live, True)
    tm._reconcile_with_bybit()
    assert closed == ['C0USDT'], closed
    print('✓ одиничне закриття — як і раніше, одразу')


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f(); n += 1
    print(f'OK — {n} тестів')
