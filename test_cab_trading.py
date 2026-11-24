"""🤖 Торгівля з особистого кабінету за сигналом ⚠ «Аномалії обсягу» (24.11).

Дослівно: «Організуй алгоритм відкриття угод за цими сигналами… дозволено
лише адміністратору… контроль куди ідуть сигнали реал чи пепер… сигнал у тому
ж напрямку — якщо угода відкрита, нічого не робимо; протилежний — закриваємо
поточну угоду і відкриваємо нову… апі ключі біржі із шифруванням (ключі
адміністратора бота перенеси в особистий кабінет)».

python3 test_cab_trading.py
"""
import importlib.util
import os
import re
import sys
import threading
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import test_anom_alarm as T      # noqa: E402  (ai_signals + стаби)

A = T.A
NOW = T.NOW
for _n in ['detection.market_data', 'detection.smc_scanner', 'detection.smc_structure',
           'detection.position_evaluator', 'detection.ob_detector',
           'core.bybit_connector', 'core.telegram_notifier', 'core']:
    sys.modules.setdefault(_n, types.ModuleType(_n))


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


TMM = _load('tm_cab_t', 'detection/trade_manager.py')
UT = _load('user_trading_t', 'web/user_trading.py')
TM = TMM.TradeManager


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


# ── правило користувача ──────────────────────────────────────────────────
def test_action_rule():
    assert TM.cab_trade_action(None, 'LONG') == 'open'
    assert TM.cab_trade_action('LONG', 'LONG') == 'hold'
    assert TM.cab_trade_action('LONG', 'SHORT') == 'reverse'
    assert TM.cab_trade_action('short', 'LONG') == 'reverse'


class _FakeTM:
    """Мінімальна книга: те, що чіпає cab_signal_trade."""
    cab_trade_action = staticmethod(TM.cab_trade_action)
    cab_avg_check = staticmethod(TM.cab_avg_check)
    _fmt_price = staticmethod(lambda v: f'{v}')

    def __init__(self, flat_ok=True):
        self._lock = threading.RLock()
        self._positions, self._shadow_positions = {}, {}
        self.calls, self.flat_ok = [], flat_ok

    def _get_current_price(self, s):
        return 100.0

    def _close_shadow(self, s, px, reason):
        self.calls.append(('close_paper', s, reason))
        self._shadow_positions.pop(s, None)

    def _flatten_on_exchange(self, s):
        self.calls.append(('flatten', s))
        return (True, '') if self.flat_ok else (False, 'біржа не підтвердила')

    def _close_position(self, s, px, reason, exchange_done=False):
        self.calls.append(('close_real', s, reason, exchange_done))
        self._positions.pop(s, None)

    def manual_open(self, s, side, **kw):
        self.calls.append(('open', s, side, kw.get('book'), kw.get('by_hand'), kw.get('opened_by')))
        (self._shadow_positions if kw.get('book') == 'paper' else self._positions)[s] = {'side': side}
        return {'ok': True, 'mode': 'test' if kw.get('book') == 'paper' else 'real'}

    def _notify(self, *a, **kw):
        self.calls.append(('notify',))

    def _cab_average(self, s, side, is_shadow, px):
        self.calls.append(('average', s, side, is_shadow, px))
        book = self._shadow_positions if is_shadow else self._positions
        book[s] = dict(book[s], cab_avg_n=int(book[s].get('cab_avg_n') or 0) + 1, entry_price=px)
        return {'ok': True, 'pos': book[s]}


def _trade(tm, *a, **kw):
    return TM.cab_signal_trade(tm, *a, **kw)


def test_open_hold_reverse_paper():
    tm = _FakeTM()
    r = _trade(tm, 'BTCUSDT', 'LONG', 'paper')
    assert r['status'] == 'opened' and tm.calls[-1] == ('open', 'BTCUSDT', 'LONG', 'paper', True, 'anom'), tm.calls
    n = len(tm.calls)
    r = _trade(tm, 'BTCUSDT', 'LONG', 'paper')
    assert r['status'] == 'hold' and len(tm.calls) == n                 # той самий бік — нічого
    r = _trade(tm, 'BTCUSDT', 'SHORT', 'paper')
    assert r['status'] == 'reversed', r
    assert tm.calls[n:] == [('close_paper', 'BTCUSDT', 'reverse_signal'),
                            ('open', 'BTCUSDT', 'SHORT', 'paper', True, 'anom')], tm.calls
    assert not any(c[0] == 'flatten' for c in tm.calls)                 # paper — біржі немає


def test_real_reverse_closes_on_exchange_first():
    tm = _FakeTM()
    tm._positions['ETHUSDT'] = {'side': 'LONG'}
    tm._shadow_positions['ETHUSDT'] = {'side': 'SHORT'}                 # інша книга не заважає
    r = _trade(tm, 'ETHUSDT', 'SHORT', 'real')
    assert r['status'] == 'reversed', r
    kinds = [c[0] for c in tm.calls]
    assert kinds == ['flatten', 'close_real', 'open'], tm.calls
    assert tm.calls[1][3] is True                                       # ордер не дублюється
    assert tm._shadow_positions['ETHUSDT']['side'] == 'SHORT'


def test_real_reverse_stops_when_exchange_not_flat():
    tm = _FakeTM(flat_ok=False)
    tm._positions['ETHUSDT'] = {'side': 'LONG'}
    r = _trade(tm, 'ETHUSDT', 'SHORT', 'real')
    assert r['status'] == 'rejected' and 'не закрилась' in r['reason'], r
    assert not any(c[0] in ('close_real', 'open') for c in tm.calls), tm.calls
    assert tm._positions['ETHUSDT']['side'] == 'LONG'                   # стара лишається


def test_bad_book_is_rejected():
    tm = _FakeTM()
    assert _trade(tm, 'BTCUSDT', 'LONG', 'auto')['status'] == 'rejected'
    assert not tm.calls


# ── ➕ усереднення (вимога 24.11) ──────────────────────────────────────────
def _anom_pos(side='LONG', e=100.0, **kw):
    return dict({'side': side, 'entry_price': e, 'opened_by': 'anom · LONG 70%',
                 'opened_at': 1.0, 'qty': 2.0, 'remaining_qty': 2.0}, **kw)


def test_avg_rule_is_pure_and_strict():
    assert TM.cab_trade_action('LONG', 'LONG', avg_ok=True) == 'average'
    assert TM.cab_trade_action('LONG', 'SHORT', avg_ok=True) == 'reverse'
    assert TM.cab_trade_action(None, 'LONG', avg_ok=True) == 'open'
    ok, _ = TM.cab_avg_check(_anom_pos(), 'LONG', 99.5)               # −0.5% — краща
    assert ok
    ok, why = TM.cab_avg_check(_anom_pos(), 'LONG', 99.9)             # −0.1% < порога
    assert not ok and 'не краща' in why, why
    ok, _ = TM.cab_avg_check(_anom_pos(), 'LONG', 101.0)              # гірша
    assert not ok
    ok, _ = TM.cab_avg_check(_anom_pos('SHORT'), 'SHORT', 100.6)      # SHORT — вища краща
    assert ok
    # краща за КОЖНУ попередню точку, а не лише за середню
    p = _anom_pos(e=99.0, cab_avg_n=1, cab_tranches=[{'px': 100.0}, {'px': 98.0}])
    assert TM.cab_best_entry(p) == 98.0
    assert not TM.cab_avg_check(p, 'LONG', 98.5)[0]
    assert TM.cab_avg_check(p, 'LONG', 97.0)[0]
    for bad, word in ((dict(opened_by='choch_bos → Q4'), 'не ⚠'), (dict(ai_owned=True), 'ШІ'),
                      (dict(manual_mode=True), 'Manual'),
                      (dict(cab_avg_n=TM.CAB_AVG_MAX_ADDS), 'стеля')):
        ok, why = TM.cab_avg_check(_anom_pos(**bad), 'LONG', 90.0)
        assert not ok and word in why, (bad, why)


def test_avg_merge_real_weighted_by_remaining():
    p = _anom_pos(qty=3.0, remaining_qty=2.0)                         # 1 монету вже частково закрито
    n = TM.cab_avg_merge_real(p, 2.0, 90.0, 50.0)
    assert abs(n['entry_price'] - 95.0) < 1e-9, n                      # (2·100 + 2·90) / 4
    assert n['remaining_qty'] == 4.0 and n['qty'] == 5.0 and n['cab_avg_n'] == 1
    assert [t['px'] for t in n['cab_tranches']] == [100.0, 90.0]
    assert p['entry_price'] == 100.0                                    # ЧИСТА — вхід не змінено


def test_avg_merge_paper_keeps_pnl_exact():
    sz = {'qty': 1.0, 'notional': 100.0, 'margin': 10.0, 'fee_open_usd': 0.055,
          'lev': 10, 'fee_pct': 0.055}
    szn = dict(sz, qty=1.2, notional=108.0, margin=10.8, fee_open_usd=0.0594)
    # частково закрито 40% по 104 — брутто й комісія вже у partial-полях
    p = {'side': 'LONG', 'entry_price': 100.0, 'remaining_qty': 0.6,
         'paper_part_usd': 0.4 * 4.0, 'paper_part_fees': 0.4 * 104 * 0.00055, 'opened_at': 1}
    n = TM.cab_avg_merge_paper(p, sz, szn, 90.0, 9.0)
    got = TM._paper_pnl(n['sz'], n['entry_price'], 95.0, 'LONG', remaining=n['remaining_qty'],
                        part_usd=n['paper_part_usd'], part_fees=n['paper_part_fees'])
    f = 0.00055
    gross = 1.6 + 0.6 * (95 - 100) + 1.2 * (95 - 90)
    fees = 1.0 * 100 * f + 1.2 * 90 * f + 0.4 * 104 * f + (0.6 + 1.2) * 95 * f
    assert abs(got['net_usd'] - round(gross - fees, 4)) < 1e-3, (got, gross - fees)
    assert n['sz']['qty'] == 2.2 and abs(n['sz']['margin'] - 20.8) < 1e-9
    assert abs(n['remaining_qty'] - 1.8 / 2.2) < 1e-9


def test_signal_averages_better_entry_both_books():
    for book in ('paper', 'real'):
        tm = _FakeTM()
        (tm._shadow_positions if book == 'paper' else tm._positions)['BTCUSDT'] = _anom_pos(e=101.0)
        r = _trade(tm, 'BTCUSDT', 'LONG', book)                        # ціна 100 < 101 −0.3%
        assert r['status'] == 'averaged' and r['avg_n'] == 1, r
        assert tm.calls[0] == ('average', 'BTCUSDT', 'LONG', book == 'paper', 100.0), tm.calls
        r = _trade(tm, 'BTCUSDT', 'LONG', book)                        # уже 100 — не краща
        assert r['status'] == 'hold', r


def test_averaged_real_closes_everything_on_exchange():
    src = _src('detection/trade_manager.py')
    body = src[src.index('    def _close_position('):src.index('        # Calculate PnL')]
    assert "pos.get('cab_avg_n')" in body and '_flatten_on_exchange(symbol)' in body
    for need in ("'cab_avg_n': int(pos.get('cab_avg_n') or 0) or None",):
        assert src.count(need) == 3, need                               # real · external · paper
    page = _src('templates/smart_money.html')
    for fn in ('function histSymOpen', 'function histSymCell'):
        i = page.index(fn)
        assert 'cabAvgMarkHTML(' in page[i:i + 1400], fn
    app = _src('infosite/js/app.js')
    assert app.count('ppAvgMark(') >= 3 and 'cab_avg_n' in _src('web/info_portal.py')


# ── налаштування кабінету ────────────────────────────────────────────────
def test_trade_norm_and_apply():
    wl = ['BTCUSDT', 'ETHUSDT']
    t = UT.trade_norm({'enabled': 1, 'coins': {'btc': 'real', 'SOLUSDT': 'paper', 'eth': 'x'}}, wl)
    assert t == {'enabled': True, 'coins': {'BTCUSDT': 'real', 'ETHUSDT': 'paper'}}, t
    t = UT.trade_apply(t, {'coin': {'symbol': 'ETHUSDT', 'on': True, 'book': 'real'}}, wl)
    assert t['coins']['ETHUSDT'] == 'real'
    t = UT.trade_apply(t, {'coin': {'symbol': 'BTCUSDT', 'on': False}, 'enabled': False}, wl)
    assert t == {'enabled': False, 'coins': {'ETHUSDT': 'real'}}, t
    t = UT.trade_apply(t, {'coin': {'symbol': 'SOLUSDT', 'on': True}}, wl)
    assert 'SOLUSDT' not in t['coins']                                  # лише монети WATCHLIST
    t = UT.trade_apply({}, {'coin': {'symbol': 'BTCUSDT', 'on': True}}, wl)
    assert t['coins'] == {'BTCUSDT': 'paper'}                           # книга за замовчуванням


def test_execute_one_action_per_book():
    subs = [(1, 'a@x', {'enabled': True, 'coins': {'BTCUSDT': 'real'}}),
            (2, 'b@x', {'enabled': True, 'coins': {'BTCUSDT': 'real', 'ETHUSDT': 'paper'}}),
            (3, 'c@x', {'enabled': False, 'coins': {'BTCUSDT': 'paper'}})]
    assert UT.trade_plan(subs, 'btc') == {'real': ['a@x', 'b@x']}
    assert UT.trade_symbols(subs) == ['BTCUSDT', 'ETHUSDT']
    calls = []

    class _T:
        def cab_signal_trade(self, s, side, book, opened_by='', note=''):
            calls.append((s, side, book, opened_by))
            return {'ok': True, 'status': 'opened', 'mode': book}
    out = UT.trade_execute('BTCUSDT', 'LONG', note='⚠ x', subs=subs, tm=_T())
    assert calls == [('BTCUSDT', 'LONG', 'real', 'anom')], calls          # дві людини — одна угода
    assert out[0]['by'] == ['a@x', 'b@x'] and UT.trade_log()[0]['symbol'] == 'BTCUSDT'
    assert UT.trade_execute('SOLUSDT', 'LONG', subs=subs, tm=_T()) == []


def test_admin_only():
    """🛡 24.11: торгівля — адмін завжди, користувач — лише з правом від адміна
    (trade_paper / trade_real); 🔐 ключі — лише адмін."""
    import json as _j
    adm = types.SimpleNamespace(is_admin=True, prefs='{}')
    usr = types.SimpleNamespace(is_admin=False, prefs='{}')
    usr_p = types.SimpleNamespace(is_admin=False, prefs=_j.dumps({'perms': {'trade_paper': True}}))
    assert UT.trade_allowed(adm) and not UT.trade_allowed(usr) and not UT.trade_allowed(None)
    assert UT.trade_allowed(usr_p) and UT.user_books(usr_p) == ('paper',)
    assert UT.keys_allowed(adm) and not UT.keys_allowed(usr_p)
    auth = _src('web/auth.py')
    body = auth[auth.index('def info_trade_subscribers'):auth.index('def is_blocked')]
    assert '_is_active(u)' in body and '.books(' in body and 'effective(u.is_admin' in body
    ut = _src('web/user_trading.py')
    for r in ('/api/me/keys', '/api/me/keys/clear', '/api/me/keys/test'):
        i = ut.index(f"'{r}'")
        assert 'if not keys_allowed(u):' in ut[i:i + 400], r


# ── підключення до сигналу ⚠ ─────────────────────────────────────────────
def test_signal_triggers_one_trade_per_new_scenario():
    trades = []
    e = A.AiEngine(db=None, scanner=T.L._Sc(True), tm=T.L._Tm(), clock=lambda: NOW,
                   anom_subs=lambda: ['BTCUSDT'], anom_send=lambda S, t: (0, 0),
                   anom_alarm=lambda *a: None,
                   anom_trade=lambda S, side, sc, px, now: trades.append((S, side)))
    sc = {'BTCUSDT': (T._r('LONG'), T._r('SHORT', False), {'price': 100.0}, None)}
    e._anom_personal(['BTCUSDT'], sc, {}, {}, NOW)
    e._anom_personal(['BTCUSDT'], sc, {}, {}, NOW + 60)
    assert trades == [('BTCUSDT', 'LONG')], trades                       # той самий сценарій — раз
    sc2 = {'BTCUSDT': (T._r('LONG', False), T._r('SHORT', last=NOW + 900), {}, None)}
    e._anom_personal(['BTCUSDT'], sc2, {}, {}, NOW + 900)
    assert trades[-1] == ('BTCUSDT', 'SHORT'), trades                   # протилежний — дія (реверс у TM)
    ai = _src('detection/ai_signals.py')
    assert "sys.modules.get('web.user_trading')" in ai and 'ut.trade_symbols()' in ai


# ── 🔐 ключі: одна реалізація, шифрування ────────────────────────────────
def test_keys_encrypted_roundtrip():
    try:
        from cryptography.fernet import Fernet
    except BaseException as e:            # зламаний системний cryptography (немає cffi)
        print(f'  ⏭ шифрування не перевірено: cryptography недоступний ({type(e).__name__})')
        return
    store = {}
    db = types.SimpleNamespace(get_setting=lambda k, d=None: store.get(k, d),
                               set_setting=lambda k, v: store.__setitem__(k, v))
    sm = types.ModuleType('storage.db_operations')
    sm.get_db = lambda: db
    old = {k: sys.modules.get(k) for k in ('storage', 'storage.db_operations')}
    sys.modules['storage'] = types.ModuleType('storage')
    sys.modules['storage'].__path__ = []
    sys.modules['storage.db_operations'] = sm
    os.environ['ENCRYPTION_KEY'] = Fernet.generate_key().decode()
    try:
        EK = _load('exchange_keys_t', 'web/exchange_keys.py')
        assert EK.validate('', 'x') and EK.validate('a b c d e f g h', 'x' * 20)
        r = EK.save('ABCDEFGHIJKL', 'S' * 30)
        assert r['ok'], r
        assert store[EK.KEY_DB].startswith('gAAAA') and 'ABCDEFGHIJKL' not in str(store)
        st = EK.status()
        assert st['source'] == 'db_encrypted' and st['key_preview'].endswith('IJKL'), st
        assert 'ABCDEFGHIJKL' not in str(st) and 'S' * 30 not in str(st)
        assert EK.clear()['ok'] and store[EK.KEY_DB] == ''
        del os.environ['ENCRYPTION_KEY']
        assert EK.save('ABCDEFGHIJKL', 'S' * 30)['ok'] is False           # без майстер-ключа — ні
    finally:
        os.environ.pop('ENCRYPTION_KEY', None)
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_bot_routes_and_page_use_the_same_keys_module():
    app = _src('web/flask_app.py')
    for fn in ('status()', 'save(', 'clear()', 'test()'):
        assert 'exchange_keys.' + fn in app, fn
    assert 'register_user_trading(app)' in app
    st = _src('templates/settings.html')
    assert 'creds-moved' in st and 'creds-plain-key' not in st and '/cabinet#keys' in st


def test_cabinet_ui():
    js = _src('infosite/js/app.js')
    html = _src('infosite/index.html')
    assert 'id="trade-card"' in html and 'href="#trade-card"' in html
    for k in ('data-belltrade', 'data-belltradebook', '/api/me/trading', '/api/me/keys',
              '/api/me/keys/test', '/api/me/keys/clear', 'trConfirmReal', 'type="password"',
              'TR.keys_allowed', 'trDefBook'):
        assert k in js, k
    assert re.search(r'"anom": "⚠ Аномалії обсягу"', js)


if __name__ == '__main__':
    n = 0
    for k, fn in list(globals().items()):
        if k.startswith('test_') and callable(fn):
            fn()
            n += 1
    print(f'OK {n}')
