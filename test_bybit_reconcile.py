"""Test Bybit position reconciliation in TradeManager.

When TM is enabled, it must:
  1. Adopt positions present on Bybit but missing from TM (manual opens,
     leftover from previous deploys, etc.)
  2. Detect closes for positions in TM but missing from Bybit (manual
     close, Bybit-side SL/TP fill, liquidation)
  3. Apply its full algorithm (HTF flip, Reverse SMC, BOS partials, etc.)
     to adopted positions — they look the same as TM-opened positions to
     the monitor loop.
"""

import sys, types, importlib.util
sys.path.insert(0, '.')


class _FakeBybit:
    """Bybit stub with controllable get_positions() return value."""
    def __init__(self, positions=None):
        self._positions = positions or []
        self.close_calls = []
    
    def get_positions(self, symbol=None):
        if symbol:
            return [p for p in self._positions if p['symbol'] == symbol]
        return list(self._positions)
    
    def close_position(self, symbol, side, qty):
        self.close_calls.append((symbol, side, qty))
        return {'order_id': 'CLOSE'}
    
    def get_wallet_balance(self): return 1000
    def get_price(self, s): return None
    def set_leverage(self, *a, **kw): pass
    def place_order(self, **kw): return {'order_id': 'STUB'}


class _StubDB:
    def __init__(self):
        self.store = {}
    def get_setting(self, k, default=None):
        return self.store.get(k, default)
    def set_setting(self, k, v):
        self.store[k] = v


# === Stub external imports trade_manager depends on ===
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
        m = types.ModuleType(mod_name)
        sys.modules[mod_name] = m

spec = importlib.util.spec_from_file_location(
    'detection.trade_manager', 'detection/trade_manager.py')
tm_mod = importlib.util.module_from_spec(spec)
sys.modules['detection.trade_manager'] = tm_mod
spec.loader.exec_module(tm_mod)
TradeManager = tm_mod.TradeManager


def _make_tm(bybit_positions=None, enabled=True):
    """Build a TM with stubs. bybit_positions = list of dicts (Bybit format)."""
    bybit = _FakeBybit(bybit_positions or [])
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=bybit)
    if enabled:
        # Override settings to mark TM as enabled
        tm._settings['enabled'] = True
        tm._settings['test_mode'] = False
    # Stub _get_current_price so external-close path works
    tm._get_current_price = lambda s: 100.0
    return tm, bybit


def _bybit_pos(symbol, side, entry_price=100.0, size=1.0,
                stop_loss=0.0, take_profit=0.0):
    """Make a Bybit-shape position dict."""
    return {
        'symbol': symbol, 'side': side, 'size': size,
        'entry_price': entry_price, 'mark_price': entry_price,
        'unrealized_pnl': 0.0, 'leverage': 10.0,
        'position_value': entry_price * size, 'liq_price': 0.0,
        'stop_loss': stop_loss, 'take_profit': take_profit,
    }


# ============================================================================
# Adoption — positions on Bybit not in TM
# ============================================================================

def test_adopt_long_position_with_no_sl_tp():
    """User opened BTCUSDT LONG manually on Bybit. TM reconciles and
    adopts it. Position appears in TM with opened_by='external'."""
    tm, _ = _make_tm([_bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0)])
    result = tm._reconcile_with_bybit()
    assert 'BTCUSDT' in result['adopted']
    pos = tm._positions['BTCUSDT']
    assert pos['side'] == 'LONG'
    assert pos['entry_price'] == 100.0
    assert pos['qty'] == 1.0
    assert pos['opened_by'] == 'external'
    assert pos['external'] is True
    assert pos['sl_price'] is None
    assert pos['tp_price'] is None
    print('✓ Adopt LONG with no SL/TP: opened_by=external, fields correct')


def test_adopt_short_position_with_sl_tp():
    """Bybit has SHORT with SL/TP set. TM preserves those levels."""
    tm, _ = _make_tm([
        _bybit_pos('ETHUSDT', 'Sell', 200.0, 2.5,
                    stop_loss=220.0, take_profit=180.0),
    ])
    tm._reconcile_with_bybit()
    pos = tm._positions['ETHUSDT']
    assert pos['side'] == 'SHORT'
    assert pos['entry_price'] == 200.0
    assert pos['qty'] == 2.5
    assert pos['sl_price'] == 220.0
    assert pos['tp_price'] == 180.0
    print('✓ Adopt SHORT with Bybit-set SL/TP: levels preserved')


def test_adopt_skips_position_with_zero_sl_tp():
    """Bybit returns 0.0 for unset SL/TP — must normalize to None, not store 0."""
    tm, _ = _make_tm([
        _bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0,
                    stop_loss=0.0, take_profit=0.0),
    ])
    tm._reconcile_with_bybit()
    pos = tm._positions['BTCUSDT']
    assert pos['sl_price'] is None, f"expected None, got {pos['sl_price']}"
    assert pos['tp_price'] is None
    print('✓ Bybit SL/TP=0 normalized to None on adoption')


def test_adopt_rejects_invalid_side():
    """Bybit returns weird 'side' value — adoption rejected gracefully."""
    tm, _ = _make_tm([
        {'symbol': 'BTCUSDT', 'side': '', 'size': 1.0,
         'entry_price': 100.0, 'stop_loss': 0.0, 'take_profit': 0.0},
    ])
    tm._reconcile_with_bybit()
    assert 'BTCUSDT' not in tm._positions
    print('✓ Adopt rejected for invalid side')


def test_adopt_rejects_zero_entry_or_qty():
    """Defensive: zero qty or zero entry price → reject."""
    tm, _ = _make_tm([
        _bybit_pos('BTCUSDT', 'Buy', 0.0, 1.0),
        _bybit_pos('ETHUSDT', 'Buy', 100.0, 0.0),
    ])
    tm._reconcile_with_bybit()
    assert 'BTCUSDT' not in tm._positions
    assert 'ETHUSDT' not in tm._positions
    print('✓ Adopt rejected for invalid entry/qty')


def test_already_managed_position_not_re_adopted():
    """If position is in both TM and Bybit, reconcile leaves it alone."""
    tm, _ = _make_tm([_bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0)])
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG',
        'entry_price': 99.5, 'qty': 1.0,  # different entry — TM's original
        'opened_at': 0, 'opened_by': 'choch',
    }
    result = tm._reconcile_with_bybit()
    assert result['adopted'] == []
    # Entry not overwritten
    assert tm._positions['BTCUSDT']['entry_price'] == 99.5
    assert tm._positions['BTCUSDT']['opened_by'] == 'choch'
    print('✓ Existing position in TM is not re-adopted')


# ============================================================================
# External close detection — positions in TM but vanished from Bybit
# ============================================================================

def test_external_close_detected_for_vanished_position():
    """Position is in TM but Bybit returns empty list — closed externally.
    TM moves it to closed_trades with reason='external_close'."""
    tm, bybit = _make_tm(bybit_positions=[])
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG',
        'entry_price': 90.0, 'qty': 1.0, 'remaining_qty': 1.0,
        'opened_at': 0, 'opened_by': 'choch',
    }
    result = tm._reconcile_with_bybit()
    assert 'BTCUSDT' in result['closed_externally']
    assert 'BTCUSDT' not in tm._positions
    closed = tm._closed_trades[-1]
    assert closed['symbol'] == 'BTCUSDT'
    assert closed['reason'] == 'external_close'
    # PnL was current_price (100) vs entry (90) for LONG = +11.11%
    assert closed['pnl_pct'] > 10
    print('✓ External close detected → moved to closed_trades with PnL')


def test_external_close_does_not_call_bybit_close():
    """Once a position is gone from Bybit, calling close_position would
    error. _close_externally must skip that API call."""
    tm, bybit = _make_tm(bybit_positions=[])
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG',
        'entry_price': 90.0, 'qty': 1.0, 'remaining_qty': 1.0,
        'opened_at': 0, 'opened_by': 'choch',
    }
    tm._reconcile_with_bybit()
    assert bybit.close_calls == [], f"unexpected Bybit close: {bybit.close_calls}"
    print('✓ External close does NOT call bybit.close_position')


# ============================================================================
# Disabled TM should not reconcile
# ============================================================================

def test_reconcile_skipped_when_tm_disabled():
    """When TM is disabled (master toggle off), reconcile must not adopt."""
    tm, _ = _make_tm([_bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0)],
                       enabled=False)
    result = tm._reconcile_with_bybit()
    assert result.get('skipped') is True
    assert 'BTCUSDT' not in tm._positions
    print('✓ Reconcile correctly skipped when TM disabled')


def test_reconcile_skipped_when_no_bybit():
    """Reconcile fails-safe when there's no Bybit client configured."""
    tm, _ = _make_tm([])
    tm.bybit = None
    result = tm._reconcile_with_bybit()
    assert result.get('skipped') is True
    print('✓ Reconcile skipped when no bybit client')


# ============================================================================
# Combined scenario — both adoption and external close
# ============================================================================

def test_combined_adopt_and_external_close():
    """TM has ETHUSDT; Bybit has BTCUSDT (new). Both events in one cycle."""
    tm, _ = _make_tm([_bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0)])
    tm._positions['ETHUSDT'] = {
        'symbol': 'ETHUSDT', 'side': 'SHORT',
        'entry_price': 200.0, 'qty': 1.0, 'remaining_qty': 1.0,
        'opened_at': 0, 'opened_by': 'choch',
    }
    result = tm._reconcile_with_bybit()
    assert 'BTCUSDT' in result['adopted']
    assert 'ETHUSDT' in result['closed_externally']
    assert 'BTCUSDT' in tm._positions
    assert 'ETHUSDT' not in tm._positions
    print('✓ Combined adopt + external-close in single reconcile cycle')


# ============================================================================
# Adopted positions go through normal monitor logic
# ============================================================================

def test_adopted_position_appears_in_get_state():
    """Adopted positions show up in get_state() output the same way as
    TM-opened ones. UI sees them, can apply manual SL/TP, force close."""
    tm, _ = _make_tm([_bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0)])
    tm._reconcile_with_bybit()
    state = tm.get_state()
    syms = [p['symbol'] for p in state['positions']]
    assert 'BTCUSDT' in syms
    btc = next(p for p in state['positions'] if p['symbol'] == 'BTCUSDT')
    assert btc.get('external') is True
    assert btc.get('opened_by') == 'external'
    print('✓ Adopted position visible in get_state() with external flag')


def test_adopted_position_supports_manual_mode():
    """Manual mode toggle works on adopted positions."""
    tm, _ = _make_tm([_bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0)])
    tm._reconcile_with_bybit()
    res = tm.update_manual_mode('BTCUSDT', enabled=True, is_shadow=False)
    assert res['ok']
    assert tm._positions['BTCUSDT']['manual_mode'] is True
    print('✓ Adopted position supports manual_mode toggle')


def test_adopted_position_supports_manual_sl_tp():
    """Manual SL/TP per-position works on adopted positions."""
    tm, _ = _make_tm([_bybit_pos('BTCUSDT', 'Buy', 100.0, 1.0)])
    tm._reconcile_with_bybit()
    res = tm.update_manual_sl_tp('BTCUSDT', manual_sl=95.0, manual_tp=110.0,
                                    is_shadow=False)
    assert res['ok']
    assert tm._positions['BTCUSDT']['manual_sl'] == 95.0
    assert tm._positions['BTCUSDT']['manual_tp'] == 110.0
    print('✓ Adopted position supports manual SL/TP')


if __name__ == '__main__':
    test_adopt_long_position_with_no_sl_tp()
    test_adopt_short_position_with_sl_tp()
    test_adopt_skips_position_with_zero_sl_tp()
    test_adopt_rejects_invalid_side()
    test_adopt_rejects_zero_entry_or_qty()
    test_already_managed_position_not_re_adopted()
    test_external_close_detected_for_vanished_position()
    test_external_close_does_not_call_bybit_close()
    test_reconcile_skipped_when_tm_disabled()
    test_reconcile_skipped_when_no_bybit()
    test_combined_adopt_and_external_close()
    test_adopted_position_appears_in_get_state()
    test_adopted_position_supports_manual_mode()
    test_adopted_position_supports_manual_sl_tp()
    print()
    print('All Bybit Reconciliation tests passed ✓')
