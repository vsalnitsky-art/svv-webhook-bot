"""🏦 Закрита реальна угода = ФАКТ БІРЖІ (вимога 27.10).

Дослівно: «бот має відобразити всі дані які взяті саме із біржі. Лише загальні
підрахунки може робити бот, а вся інформація має повністю братись з біржі.»
"""
import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_autosl_chain as _tmb  # noqa: E402

TM = _tmb.TM
tm_mod = sys.modules[TM.__module__]
summ = tm_mod.exchange_close_summary
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _rec(ts, qty, entry, exit_p, pnl, side='Sell', of=0.05, cf=0.06, lev=10):
    return {'side': side, 'qty': qty, 'avg_entry_price': entry, 'avg_exit_price': exit_p,
            'closed_pnl': pnl, 'cum_entry_value': entry * qty, 'cum_exit_value': exit_p * qty,
            'open_fee': of, 'close_fee': cf, 'leverage': lev,
            'updated_time': int(ts * 1000), 'order_id': f'o{ts}'}


def test_partials_and_final_are_summed():
    recs = [_rec(2000, 5, 100, 110, 49.9),       # фінальне
            _rec(1500, 5, 100, 105, 24.9),       # TP-1 часткове
            _rec(900, 7, 100, 90, -70)]          # ПОПЕРЕДНЯ угода (до відкриття)
    s = summ(recs, 'LONG', 1000, 2000)
    _check(s['parts'] == 2 and s['qty'] == 10, s)
    _check(abs(s['pnl_usd'] - 74.8) < 1e-9, 'PnL = сума записів біржі')
    _check(abs(s['exit_price'] - 107.5) < 1e-9, 'середній вихід = вартість / кількість')
    _check(abs(s['fees_usd'] - 0.22) < 1e-9 and abs(s['gross_usd'] - 75.02) < 1e-9, s)
    _check(s['leverage'] == 10 and abs(s['roi_pct'] - 74.8 / 100 * 100) < 1e-9, s)
    print('✓ часткові + фінальне сумуються, чужа угода не підмішується')


def test_pending_until_final_record_exists():
    _check(summ([_rec(1500, 5, 100, 105, 24.9)], 'LONG', 1000, 2000) is None,
           'без фінального запису — None (не вигадуємо)')
    _check(summ([_rec(2000, 5, 100, 105, 1, side='Buy')], 'LONG', 1000, 2000) is None,
           'Buy — закриття SHORT, не LONG')
    _check(summ([_rec(2500, 5, 100, 105, 1)], 'LONG', 1000, 2000) is None,
           'запис НАСТУПНОЇ угоди не береться')
    print('✓ немає запису біржі → None')


def test_external_close_detected_late_still_resolves():
    # Кейс 27.10: закрито на біржі о 2000, бот помітив о 2300 (підтвердження
    # масового закриття) — запис біржі на 300с старший за closed_at.
    recs = [_rec(2000, 10, 0.0767, 0.07556, 1.11, side='Buy')]
    _check(summ(recs, 'SHORT', 1000, 2300) is None,
           'без кількості угоди фінал за часом не видно (стара поведінка)')
    s = summ(recs, 'SHORT', 1000, 2300, expected_qty=10)
    _check(s and s['pnl_usd'] == 1.11 and s['qty'] == 10, s)
    _check(summ(recs, 'SHORT', 1000, 2300, expected_qty=20) is None,
           'покрито лише половину кількості — ще не фінал')
    print('✓ закриття поза ботом, помічене пізно, звіряється за кількістю')


def test_missing_fees_stay_unknown():
    r = _rec(2000, 1, 100, 101, 0.9)
    r['open_fee'] = None
    s = summ([r], 'LONG', 1000, 2000)
    _check(s['fees_usd'] is None and s['gross_usd'] is None, 'невідомо ≠ нуль')
    print('✓ комісій немає у відповіді → None, а не 0')


class _B:
    def __init__(self, recs):
        self.recs, self.calls = recs, 0

    def get_closed_pnl(self, symbol, limit=5):
        self.calls += 1
        return self.recs


def _tm(recs):
    t = TM.__new__(TM)
    t._lock = threading.RLock()
    t._closed_trades = []
    t.bybit = _B(recs)
    t._persist_closed_trades = lambda: None
    return t


def test_apply_truth_and_pending_resync():
    c = {'symbol': 'XUSDT', 'side': 'LONG', 'opened_at': 1000, 'closed_at': 2000,
         'pnl_usd': 1.0, 'pnl_pct': 1.0, 'entry_price': 99, 'exit_price': 104}
    t = _tm([])
    _check(TM._apply_exchange_close_truth(t, 'XUSDT', c, 'LONG', attempts=1) is False, '')
    _check(c['pnl_source'] == 'pending', 'немає запису → pending, а не «готово»')
    t.bybit.recs = [_rec(2000, 5, 100, 110, 49.9)]
    _check(TM._apply_exchange_close_truth(t, 'XUSDT', c, 'LONG', attempts=1), '')
    _check(c['pnl_source'] == 'exchange' and c['entry_price'] == 100
           and c['exit_price'] == 110 and c['pnl_usd'] == 49.9, c)
    _check(c['exchange']['fees_usd'] is not None, 'розклад біржі на угоді')
    print('✓ pending → факт біржі (ціни, PnL, комісії)')


def test_resync_is_wired_and_page_shows_source():
    src = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    _check('self._resync_exchange_closes_bg()' in src, 'ресинк не підключено до монітора')
    _check('def update_archive_trade' in open(os.path.join(_HERE, 'storage', 'db_operations.py'),
                                              encoding='utf-8').read(), 'архів не оновлюється')
    _check('${closedRealPnlCellHTML(c)}' in _HTML, 'комірка закритої угоди')
    _check("(c.pnl_source || '') + c.pnl_usd" in _HTML,
           'без джерела в сигнатурі рядок не перемалюється після звірки')
    _check('не звірено з біржею' in _HTML, 'підсумок має казати про незвірені')
    print('✓ ресинк підключено; сторінка показує ⇄ біржа / ⏳ очікує')


def test_resync_picks_records_without_source():
    # Кейс 27.10: стара версія ставила 'approx' лише в памʼяті й не персистила →
    # після рестарту поле зникало, UI показував ⏳, а ресинк угоду не брав.
    import time as _t
    now = _t.time()
    t = _tm([_rec(now - 100, 10, 0.0767, 0.07556, 1.11, side='Buy')])
    t._archive_update = lambda c: None
    c = {'symbol': 'XUSDT', 'side': 'SHORT', 'opened_at': now - 1000,
         'closed_at': now - 50, 'qty': 10, 'pnl_usd': 0.5, 'pnl_pct': 0.5}
    old = {'symbol': 'YUSDT', 'side': 'LONG', 'opened_at': 1, 'closed_at': now - 5 * 86400}
    t._closed_trades = [old, c]
    started = []
    real_thread = threading.Thread

    class _Sync:
        def __init__(self, target=None, **kw):
            self.t = target
        def start(self):
            started.append(1)
            self.t()
    threading.Thread = _Sync
    try:
        TM._resync_exchange_closes_bg(t)
    finally:
        threading.Thread = real_thread
    _check(started, 'ресинк мусить запуститись для угоди без pnl_source')
    _check(c.get('pnl_source') == 'exchange' and c['pnl_usd'] == 1.11, c)
    _check('pnl_source' not in old, 'старше 3 діб — не чіпаємо')
    print('✓ угода без pnl_source (після рестарту) звіряється з біржею')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
