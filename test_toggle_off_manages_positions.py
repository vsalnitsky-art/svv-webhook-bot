"""Test that exit/management rules work through OPEN positions even when the
master toggles are OFF.

Requirement: the 💼 Trade Manager (real) and 🧪 Test Mode (paper) toggles
gate NEW entries only. An already-open position must still be worked through
by its exit rules regardless of toggle state:
  - reverse_smc / forecast_1h_confluence / opposite_ob (CHoCH-driven)
  - BOS-N partial closes (BOS-driven)
  - tick-driven SL/TP/trailing/BE (already toggle-independent in _monitor_*)

New entries (on_signal opening) remain blocked when the relevant toggle is OFF.
"""

import sys, types, importlib.util
sys.path.insert(0, '.')


class _StubBybit:
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
    def get_smc_ob_state(self, symbol, tf):
        return None


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


def _make_tm(enabled=False, test_mode=False):
    """TM with both toggles OFF by default, holding one real + one shadow pos."""
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=_StubBybit())
    tm._settings['enabled'] = enabled
    tm._settings['test_mode'] = test_mode
    # Exit rules under test
    tm._settings['use_reverse_smc'] = True
    tm._settings['use_bos_partials'] = True
    tm._settings['bos_2_close_pct'] = 50
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG',
        'entry_price': 100.0, 'qty': 1.0,
        'sl_price': 90.0, 'tp_price': 120.0,
        'opened_at': 0, 'opened_by': 'choch',
    }
    tm._shadow_positions['ETHUSDT'] = {
        'symbol': 'ETHUSDT', 'side': 'SHORT',
        'entry_price': 200.0, 'qty': 1.0,
        'sl_price': 220.0, 'tp_price': 180.0,
        'opened_at': 0, 'opened_by': 'choch',
    }
    tm._get_current_price = lambda s: {'BTCUSDT': 100.0, 'ETHUSDT': 200.0}.get(s)
    return tm


# ============================================================================
# REAL position — exit rules fire with TM master toggle OFF
# ============================================================================

def test_reverse_smc_closes_real_when_tm_disabled():
    tm = _make_tm(enabled=False)
    closed = []
    tm._close_position = lambda sym, price, reason=None: closed.append((sym, reason))
    # Opposite CHoCH (bear) against LONG real position
    tm.on_choch_event('BTCUSDT', direction='bear', level=99.0, bar_t=1000)
    assert closed == [('BTCUSDT', 'reverse_smc')], closed
    print('✓ reverse_smc closes REAL position when TM toggle is OFF')


def test_bos_partial_fires_real_when_tm_disabled():
    tm = _make_tm(enabled=False)
    called = []
    tm._process_bos_real = lambda *a, **kw: called.append(a)
    # Same-direction BOS (bull) on LONG real position → partial close path
    tm.on_bos_event('BTCUSDT', direction='bull', level=101.0, bar_t=1000)
    assert len(called) == 1, called
    print('✓ BOS-N partial close fires for REAL position when TM toggle is OFF')


# ============================================================================
# SHADOW position — exit rules fire with Test Mode toggle OFF
# ============================================================================

def test_reverse_smc_closes_shadow_when_test_mode_off():
    tm = _make_tm(test_mode=False)
    closed = []
    tm._close_shadow = lambda sym, price, reason=None: closed.append((sym, reason))
    # Opposite CHoCH (bull) against SHORT shadow position
    tm.on_choch_event('ETHUSDT', direction='bull', level=201.0, bar_t=1000)
    assert closed == [('ETHUSDT', 'reverse_smc')], closed
    print('✓ reverse_smc closes SHADOW position when Test Mode is OFF')


def test_bos_partial_fires_shadow_when_test_mode_off():
    tm = _make_tm(test_mode=False)
    called = []
    tm._process_bos_shadow = lambda *a, **kw: called.append(a)
    # Same-direction BOS (bear) on SHORT shadow position → partial close path
    tm.on_bos_event('ETHUSDT', direction='bear', level=199.0, bar_t=1000)
    assert len(called) == 1, called
    print('✓ BOS-N partial close fires for SHADOW position when Test Mode is OFF')


# ============================================================================
# New entries STILL blocked when toggles OFF
# ============================================================================

def test_new_real_entry_blocked_when_tm_disabled():
    tm = _make_tm(enabled=False, test_mode=False)
    opened = []
    tm._open_position = lambda *a, **kw: opened.append(a)
    # Fresh symbol, directional signal — must NOT open a real position
    tm.on_signal('SOLUSDT', side='LONG', entry_price=50.0, opened_by='choch')
    assert opened == [], opened
    print('✓ New REAL entry blocked when TM toggle is OFF')


def test_new_shadow_entry_blocked_when_test_mode_off():
    tm = _make_tm(enabled=False, test_mode=False)
    opened = []
    tm._open_shadow = lambda *a, **kw: opened.append(a)
    tm.on_signal('SOLUSDT', side='LONG', entry_price=50.0, opened_by='choch')
    assert opened == [], opened
    print('✓ New SHADOW entry blocked when Test Mode is OFF')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll toggle-off management tests passed ✓ ({len(tests)} tests)')
