"""👁 Монету прибрано з WATCHLIST → її стан не показується (вимога 26.10).

Дослівно: «Монети видалені із WATCHLIST, а аналіз по ним ніби то як
відбувається і далі… монет яких вже немає, а вона наприклад є в угоді, то по
ним не має бути даних, які відображають в якому стані знаходиться монета
(наприклад OB 4H або 🧠 МММ-Decision 🚨).»
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_trades_mmd as _t   # noqa: E402  (ізольований fuel_filter + стаби TM)
import test_trades_mm_ob4h as _o   # noqa: E402

_m, _mon = _t._m, _t._mon
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _ff_with_wl(wl, **kw):
    ff, tm = _t._ff(**kw)
    ff._mm_watched = lambda: set(wl)
    return ff, tm


def test_trade_coin_outside_watchlist_has_no_decision():
    pos = {'BTCUSDT': {'side': 'LONG', 'opened_at': 1000.0},
           'DOGEUSDT': {'side': 'SHORT', 'opened_at': 1000.0}}
    ff, tm = _ff_with_wl({'BTCUSDT'}, data={'BTCUSDT': _t._dec('LONG', 0.85),
                                           'DOGEUSDT': _t._dec('SHORT', 0.2)},
                         positions=pos, dec_on=True)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3, DOGEUSDT=-0.3), now=2000.0)
    _check('DOGEUSDT' not in tm.calls, f'вердикт для прибраної монети не рахується: {tm.calls}')
    r = ff.dec_trade_for([('BTCUSDT', 'LONG', 1000.0), ('DOGEUSDT', 'SHORT', 1000.0)])
    _check(_m.mmd_trade_key('BTCUSDT', 'LONG', 1000.0) in r, r)
    _check(_m.mmd_trade_key('DOGEUSDT', 'SHORT', 1000.0) not in r,
           'угода по прибраній монеті — 🧠/🚨 порожні')
    rows = {x['symbol'] for x in ff.mm_monitor_state(book='dec')['rows']}
    _check(rows == {'BTCUSDT'}, rows)
    print('✓ угода поза watchlist: 🧠/🚨 порожні, у таблиці 🧠 її немає')


def test_old_book_keeps_exit_data_but_hides_the_row():
    pos = {'DOGEUSDT': {'side': 'SHORT', 'opened_at': 1000.0}}
    ff, _ = _ff_with_wl({'BTCUSDT'}, data={}, positions=pos)
    ff._legacy = {'BTCUSDT': 0.3, 'DOGEUSDT': -0.3, 'XRPUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3, DOGEUSDT=-0.3, XRPUSDT=0.3), now=2000.0)
    snap = ff._mm_snapshot
    _check('XRPUSDT' not in snap, 'прибрана монета без угоди — геть зі знімка')
    _check(snap.get('DOGEUSDT', {}).get('unwatched'),
           'прибрана монета в угоді лишається лише для правила виходу')
    rows = {x['symbol'] for x in ff.mm_monitor_state()['rows']}
    _check(rows == {'BTCUSDT'}, f'у таблиці 🧮 лише watchlist: {rows}')
    for book in ('new',):
        rows = {x['symbol'] for x in ff.mm_monitor_state(book=book)['rows']}
        _check('DOGEUSDT' not in rows and 'XRPUSDT' not in rows, (book, rows))
    print('✓ 🧮 знімок тримає угоду поза списком лише для виходу; рядків і важеля немає')


def test_ob4h_mark_only_for_watchlist():
    sc = _o._sc.SMCScanner.__new__(_o._sc.SMCScanner)
    sc._ob_mark = {'BTCUSDT': {'bias': 'BULLISH'}, 'DOGEUSDT': {'bias': 'BEARISH'}}
    sc._ob_mark_restored = set()
    sc._watchlist = ['BTCUSDT']
    r = sc.ob_mark_for(['BTCUSDT', 'DOGEUSDT'])
    _check(set(r) == {'BTCUSDT'}, f'OB 4H лише для watchlist: {r}')
    print('✓ OB 4H біля Side / у таблицях МММ — лише для монет watchlist')


def test_remove_symbol_drops_displayed_state():
    a = _SC_SRC.index('def remove_symbol(')
    fn = _SC_SRC[a:_SC_SRC.index('\n    def ', a + 10)]
    for k in ('_ob_mark.pop', '_ob_zones.pop', '_vob_diag.pop', '_ob_htf_cache.pop'):
        _check(k in fn, f'remove_symbol чистить {k}')
    print('✓ видалення монети чистить її OB 4H / бокси / бейджі')


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); n += 1
    print(f'OK — {n} тестів')
