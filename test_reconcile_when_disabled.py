"""Test that reconcile loads and tracks existing positions even when TM is disabled.

The issue: user has TM toggle OFF but open positions on Bybit → bot ignores them.

Root cause: _reconcile_with_bybit had early return `if not is_enabled()`, so
existing positions were never loaded into _positions at startup.

Fix: reconcile always runs (loads existing + detects external closes), but
ADOPTION of new external positions is gated by is_enabled().
"""

import sys, types, importlib.util
sys.path.insert(0, '.')


class _StubBybit:
    def __init__(self):
        self.live_positions = []

    def get_wallet_balance(self):
        return 1000

    def get_price(self, s):
        return None

    def set_leverage(self, *a, **kw):
        pass

    def place_order(self, **kw):
        return {'order_id': 'STUB'}

    def get_positions(self):
        return self.live_positions

    def get_positions_checked(self):
        return self.live_positions, True


class _StubDB:
    def __init__(self):
        self.store = {}

    def get_setting(self, k, default=None):
        return self.store.get(k, default)

    def set_setting(self, k, v):
        self.store[k] = v

    def get_smc_ob_state(self, symbol, tf):
        return None


# === Stub external imports ===
storage_pkg = types.ModuleType('storage')
storage_pkg.__path__ = ['./storage']
sys.modules['storage'] = storage_pkg
db_mod = types.ModuleType('storage.db_operations')
db_mod.get_db = lambda: _StubDB()
sys.modules['storage.db_operations'] = db_mod

detection_pkg = types.ModuleType('detection')
detection_pkg.__path__ = ['./detection']
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


def test_reconcile_loads_existing_when_disabled():
    """When TM is disabled but user already has positions on Bybit,
    reconcile should load them (update existing) but NOT adopt new ones."""

    bybit = _StubBybit()
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=bybit)

    # TM disabled
    tm._settings['enabled'] = False
    tm._settings['tradeable_symbols'] = ['BTCUSDT', 'ETHUSDT']

    # Pre-inject one position into TM (as if it was opened before TM was disabled)
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'entry_price': 100.0,
        'qty': 1.0,
        'sl_price': 90.0,
        'tp_price': 120.0,
        'opened_at': 0,
        'opened_by': 'choch',
    }

    # Bybit has two positions: BTCUSDT (already managed) + ETHUSDT (new)
    bybit.live_positions = [
        {
            'symbol': 'BTCUSDT',
            'side': 'Buy',
            'entry_price': 101.0,  # Different from TM (external edit)
            'size': 1.5,  # Bybit uses 'size', not 'qty'
            'unrealised_pnl': 10.0,
            'mark_price': 102.0,
        },
        {
            'symbol': 'ETHUSDT',
            'side': 'Sell',
            'entry_price': 200.0,
            'size': 2.0,  # Bybit uses 'size', not 'qty'
            'unrealised_pnl': -5.0,
            'mark_price': 201.0,
        },
    ]

    # Run reconcile
    result = tm._reconcile_with_bybit()

    # Assertions:
    # 1. reconcile did NOT skip (no early return)
    assert 'skipped' not in result or not result['skipped'], f"reconcile skipped: {result}"

    # 2. BTCUSDT (already managed) was updated with Bybit truth
    assert 'BTCUSDT' in tm._positions
    pos = tm._positions['BTCUSDT']
    assert pos['entry_price'] == 101.0, f"entry not synced: {pos['entry_price']}"
    assert pos['qty'] == 1.5, f"qty not synced: {pos['qty']}"
    print('✓ Reconcile updated existing position BTCUSDT with Bybit truth (TM disabled)')

    # 3. ETHUSDT (new external position) was NOT adopted (TM disabled = no new exposure)
    assert 'ETHUSDT' not in tm._positions, "ETHUSDT was adopted despite TM disabled"
    print('✓ Reconcile skipped adoption of new position ETHUSDT (TM disabled)')


def test_reconcile_detects_external_close_when_disabled():
    """When TM is disabled, reconcile should still detect external closes."""

    bybit = _StubBybit()
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=bybit)

    tm._settings['enabled'] = False

    # TM has one position
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT',
        'side': 'LONG',
        'entry_price': 100.0,
        'qty': 1.0,
        'opened_at': 0,
        'opened_by': 'choch',
    }

    # Bybit has NO positions (user closed externally)
    bybit.live_positions = []

    closed_externally = []
    tm._close_externally = lambda sym, *a, **kw: closed_externally.append(sym)

    tm._reconcile_with_bybit()

    # reconcile should detect that BTCUSDT was closed externally
    assert 'BTCUSDT' in closed_externally, f"external close not detected: {closed_externally}"
    print('✓ Reconcile detected external close of BTCUSDT (TM disabled)')


def test_reconcile_adopts_when_enabled():
    """When TM is enabled, reconcile should adopt new external positions."""

    bybit = _StubBybit()
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=bybit)

    # TM enabled
    tm._settings['enabled'] = True
    tm._settings['tradeable_symbols'] = ['BTCUSDT']

    # Bybit has one position that TM doesn't know about
    bybit.live_positions = [
        {
            'symbol': 'BTCUSDT',
            'side': 'Buy',
            'entry_price': 100.0,
            'size': 1.0,  # Bybit uses 'size', not 'qty'
            'unrealised_pnl': 5.0,
            'mark_price': 101.0,
        },
    ]

    tm._reconcile_with_bybit()

    # reconcile should adopt BTCUSDT (TM enabled)
    assert 'BTCUSDT' in tm._positions, "BTCUSDT not adopted despite TM enabled"
    assert tm._positions['BTCUSDT']['entry_price'] == 100.0
    print('✓ Reconcile adopted new position BTCUSDT (TM enabled)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll reconcile-when-disabled tests passed ✓ ({len(tests)} tests)')
