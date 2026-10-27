"""Test manual_open — user-initiated position opens from the Decision
Center panel. Backend wraps the existing _open_position auto-trade path
with validation: TM must be enabled, no existing position, valid side.
"""

import sys, types, importlib.util
sys.path.insert(0, '.')


class _FakeBybit:
    def __init__(self):
        self.api_key = 'TEST_KEY'   # required for manual_open validation
        self.orders_placed = []
        self.leverage_set = []
    def get_positions(self, symbol=None): return []
    def get_wallet_balance(self): return 1000
    def get_price(self, s): return None
    def set_leverage(self, sym, lev): self.leverage_set.append((sym, lev))
    def place_order(self, **kw):
        self.orders_placed.append(kw)
        return {'order_id': 'STUB_' + kw['symbol']}
    def close_position(self, **kw): return {'order_id': 'CLOSE'}


class _StubDB:
    def __init__(self): self.store = {}
    def get_setting(self, k, default=None): return self.store.get(k, default)
    def set_setting(self, k, v): self.store[k] = v


storage_pkg = types.ModuleType('storage'); storage_pkg.__path__ = ['./storage']
sys.modules['storage'] = storage_pkg
db_mod = types.ModuleType('storage.db_operations')
db_mod.get_db = lambda: _StubDB()
sys.modules['storage.db_operations'] = db_mod

detection_pkg = types.ModuleType('detection'); detection_pkg.__path__ = ['./detection']
sys.modules['detection'] = detection_pkg

for mod_name in ['detection.market_data', 'detection.smc_scanner',
                  'detection.smc_structure', 'detection.position_evaluator',
                  'detection.ob_detector', 'core.bybit_connector',
                  'core.telegram_notifier', 'core', 'web']:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)

spec = importlib.util.spec_from_file_location(
    'detection.trade_manager', 'detection/trade_manager.py')
tm_mod = importlib.util.module_from_spec(spec)
sys.modules['detection.trade_manager'] = tm_mod
spec.loader.exec_module(tm_mod)
TradeManager = tm_mod.TradeManager


def _make_tm(enabled=True):
    """TM ready to manual_open with stubbed price + sizing."""
    bybit = _FakeBybit()
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=bybit)
    # Enabled state controls is_enabled() — set via settings + master toggle
    tm._settings['enabled'] = enabled
    tm._settings['test_mode'] = False
    # Stub current price lookup — manual_open needs it
    tm._get_current_price = lambda s: 100.0
    from types import SimpleNamespace as _SN
    tm.scanner = _SN(get_tradeable_symbols=lambda: ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'])
    # Use default fixed_usd sizing — $100 per trade
    tm._settings['sizing_mode'] = 'fixed_usd'
    tm._settings['fixed_usd_amount'] = 100.0
    tm._settings['leverage'] = 10
    tm._settings['use_sl'] = True
    tm._settings['use_tp'] = True
    tm._settings['sl_pct'] = 2.0
    tm._settings['tp_pct'] = 4.0
    return tm, bybit


# ============================================================================
# Happy path
# ============================================================================

def test_manual_open_long_places_buy_order():
    """Click LONG → backend places Buy order on Bybit with correct params
    from TM settings. Position lands in self._positions."""
    tm, bybit = _make_tm()
    res = tm.manual_open('BTCUSDT', 'LONG')
    assert res['ok'] is True
    assert 'BTCUSDT' in tm._positions
    pos = tm._positions['BTCUSDT']
    assert pos['side'] == 'LONG'
    assert pos['entry_price'] == 100.0
    assert pos['opened_by'].startswith('manual_ui')
    # Bybit was called with Buy side
    assert len(bybit.orders_placed) == 1
    assert bybit.orders_placed[0]['side'] == 'Buy'
    assert bybit.orders_placed[0]['symbol'] == 'BTCUSDT'
    print('✓ manual_open LONG places Buy order, position recorded')


def test_manual_open_short_places_sell_order():
    tm, bybit = _make_tm()
    res = tm.manual_open('ETHUSDT', 'SHORT')
    assert res['ok'] is True
    pos = tm._positions['ETHUSDT']
    assert pos['side'] == 'SHORT'
    assert bybit.orders_placed[0]['side'] == 'Sell'
    print('✓ manual_open SHORT places Sell order')


def test_manual_open_uses_sizing_settings():
    """Verify Position Sizing values flow through to _calculate_qty path."""
    tm, bybit = _make_tm()
    tm._settings['fixed_usd_amount'] = 500.0  # bigger trade
    tm.manual_open('BTCUSDT', 'LONG')
    pos = tm._positions['BTCUSDT']
    # qty = 500 USD / 100 USD-per-coin = 5 coins
    assert pos['qty'] == 5.0, f"expected qty=5.0, got {pos['qty']}"
    print('✓ manual_open uses fixed_usd_amount from settings')


def test_manual_open_applies_sl_tp_from_settings():
    """When use_sl + use_tp ON, manual_open computes prices from %."""
    tm, bybit = _make_tm()
    tm._settings['sl_pct'] = 5.0
    tm._settings['tp_pct'] = 10.0
    tm.manual_open('BTCUSDT', 'LONG')
    pos = tm._positions['BTCUSDT']
    # Entry 100, LONG: SL = 100 * 0.95 = 95, TP = 100 * 1.10 = 110
    assert abs(pos['sl_price'] - 95.0) < 0.01
    assert abs(pos['tp_price'] - 110.0) < 0.01
    print('✓ manual_open applies SL/TP from settings')


# ============================================================================
# Validation — rejected cases
# ============================================================================

def test_manual_open_rejected_when_tm_disabled():
    """Master toggle OFF → no order placed, clear error reason."""
    tm, bybit = _make_tm(enabled=False)
    res = tm.manual_open('BTCUSDT', 'LONG')
    assert res['ok'] is False
    assert 'trade manager вимкнено' in res['reason'].lower(), res
    assert bybit.orders_placed == []
    print('✓ manual_open rejected when TM disabled (no order)')


def test_manual_open_rejected_when_bybit_no_api_key():
    """No Bybit API key → reject, no panic."""
    tm, bybit = _make_tm()
    bybit.api_key = None
    res = tm.manual_open('BTCUSDT', 'LONG')
    assert res['ok'] is False
    assert 'bybit' in res['reason'].lower() or 'api key' in res['reason'].lower()
    print('✓ manual_open rejected when no Bybit API key')


def test_manual_open_rejected_when_position_exists():
    """If TM already has a real position for this symbol, reject. Avoids
    confusing state (averaging? doubling? — undefined)."""
    tm, bybit = _make_tm()
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG',
        'entry_price': 100.0, 'qty': 1.0, 'opened_at': 0,
        'opened_by': 'choch',
    }
    res = tm.manual_open('BTCUSDT', 'SHORT')
    assert res['ok'] is False
    assert 'вже є' in res['reason'].lower()
    assert bybit.orders_placed == []
    print('✓ manual_open rejected when real position already exists')


def test_manual_open_rejected_when_shadow_position_exists():
    """🏦 СКАСОВАНО 27.10 — «Trade Manager завжди має пріоритет»: paper-угода
    більше НЕ блокує реальну. Блокує лише позиція В ТІЙ САМІЙ книзі: коли
    реальна книга недоступна (тут — вимкнений TM), друга paper неможлива."""
    tm, bybit = _make_tm()
    tm._shadow_positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG',
        'entry_price': 100.0, 'qty': 1.0, 'opened_at': 0,
        'opened_by': 'choch',
    }
    tm._settings['enabled'] = False
    tm._settings['test_mode'] = True
    res = tm.manual_open('BTCUSDT', 'LONG')
    assert res['ok'] is False
    assert 'paper' in res['reason'].lower() or 'shadow' in res['reason'].lower()
    assert tm.blocks_new_open('BTCUSDT') is True
    print('✓ реальна книга недоступна → paper по монеті блокує другу paper')


def test_real_opens_in_parallel_with_a_paper_trade():
    """🏦 Trade Manager має ПРІОРИТЕТ (вимога 27.10): paper-угода по монеті
    не заважає реальній, коли TM ✓ · TRD ✓ · ліміт дозволяє."""
    tm, bybit = _make_tm()
    tm._shadow_positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG',
        'entry_price': 100.0, 'qty': 1.0, 'opened_at': 0,
        'opened_by': 'choch',
    }
    assert tm.blocks_new_open('BTCUSDT') is False
    res = tm.manual_open('BTCUSDT', 'LONG')
    assert res['ok'] is True and res.get('mode') == 'real', res
    assert len(bybit.orders_placed) == 1
    assert 'BTCUSDT' in tm._positions and 'BTCUSDT' in tm._shadow_positions
    print('✓ реальна угода відкривається паралельно з paper')


def test_manual_open_rejected_on_bad_side():
    tm, bybit = _make_tm()
    res = tm.manual_open('BTCUSDT', 'BUYY')
    assert res['ok'] is False
    assert 'side' in res['reason'].lower()
    print('✓ manual_open rejected on invalid side')


def test_manual_open_rejected_on_empty_symbol():
    tm, _ = _make_tm()
    res = tm.manual_open('', 'LONG')
    assert res['ok'] is False
    assert 'symbol' in res['reason'].lower()
    print('✓ manual_open rejected on empty symbol')


def test_manual_open_rejected_when_no_price():
    """Market data returns None → can't compute size → reject before
    placing order. Avoids garbage-in scenarios."""
    tm, bybit = _make_tm()
    tm._get_current_price = lambda s: None
    res = tm.manual_open('BTCUSDT', 'LONG')
    assert res['ok'] is False
    assert 'price' in res['reason'].lower()
    assert bybit.orders_placed == []
    print('✓ manual_open rejected when no current price available')


def test_manual_open_normalizes_symbol_case():
    """User might send 'btcusdt' from URL or external — must work."""
    tm, bybit = _make_tm()
    res = tm.manual_open('btcusdt', 'long')
    assert res['ok'] is True
    assert 'BTCUSDT' in tm._positions
    assert tm._positions['BTCUSDT']['side'] == 'LONG'
    print('✓ manual_open normalizes lowercase symbol/side')


def test_trd_off_blocks_real_for_queue_opens_only():
    """📒 27.10: real = TRD ✓ + TM ✓ + стеля дозволяє; інакше paper (Test Mode),
    інакше відмова. Діє для ВСІХ відкриттів — черги і ✋ ручні."""
    from types import SimpleNamespace
    # TRD вимкнено + Test Mode → paper (і авто, і ручне)
    for kw in ({'bypass_gates': True}, {'by_hand': True}):
        tm, bybit = _make_tm()
        tm.scanner = SimpleNamespace(get_tradeable_symbols=lambda: [])
        tm._settings['test_mode'] = True
        r = tm.manual_open('BTCUSDT', 'LONG', **kw)
        assert r.get('mode') == 'test' and 'BTCUSDT' not in tm._positions, (kw, r)
        assert not bybit.orders_placed and 'TRD' in r.get('reason', ''), r
    # TRD вимкнено, Test Mode вимкнено → відмова з причиною
    tm2, bybit2 = _make_tm()
    tm2.scanner = SimpleNamespace(get_tradeable_symbols=lambda: [])
    r2 = tm2.manual_open('ETHUSDT', 'LONG', by_hand=True)
    assert r2['ok'] is False and 'TRD' in r2['reason'] and not bybit2.orders_placed
    # TM вимкнено → paper
    tm3, _ = _make_tm(enabled=False)
    tm3._settings['test_mode'] = True
    r3 = tm3.manual_open('SOLUSDT', 'LONG', by_hand=True)
    assert r3.get('mode') == 'test' and 'Trade Manager' in r3['reason'], r3
    # усе ✓ → реальна
    tm4, _ = _make_tm()
    r4 = tm4.manual_open('XRPUSDT', 'LONG', bypass_gates=True)
    assert r4['ok'] is True and 'XRPUSDT' in tm4._positions, r4
    # чисте правило
    RB = tm4.route_book
    assert RB(True, True, False, False)[0] == 'real'
    assert RB(True, False, False, True)[0] == 'paper'
    assert RB(True, True, True, True)[0] == 'paper'
    assert RB(False, True, False, False)[0] is None
    print('✓ real = TRD + TM + стеля; інакше Test Mode — для всіх відкриттів')

if __name__ == '__main__':
    test_trd_off_blocks_real_for_queue_opens_only()
    test_manual_open_long_places_buy_order()
    test_manual_open_short_places_sell_order()
    test_manual_open_uses_sizing_settings()
    test_manual_open_applies_sl_tp_from_settings()
    test_manual_open_rejected_when_tm_disabled()
    test_manual_open_rejected_when_bybit_no_api_key()
    test_manual_open_rejected_when_position_exists()
    test_manual_open_rejected_when_shadow_position_exists()
    test_real_opens_in_parallel_with_a_paper_trade()
    test_manual_open_rejected_on_bad_side()
    test_manual_open_rejected_on_empty_symbol()
    test_manual_open_rejected_when_no_price()
    test_manual_open_normalizes_symbol_case()
    print()
    print('All manual_open tests passed ✓')
