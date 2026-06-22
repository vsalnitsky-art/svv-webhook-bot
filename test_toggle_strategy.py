"""Toggle strategy test:

  - Toggle OFF + open positions → stop accepting NEW trades, but fully manage
    existing ones (load/sync via reconcile, exit rules, BOS partials).
  - Toggle ON → work as usual.

The toggle gates NEW entries only — nothing else.
"""

import sys, types, importlib.util
sys.path.insert(0, '.')


class _StubBybit:
    def __init__(self):
        self.live_positions = []
    def get_wallet_balance(self): return 1000
    def get_price(self, s): return None
    def set_leverage(self, *a, **kw): pass
    def place_order(self, **kw): return {'order_id': 'STUB'}
    def get_positions(self): return self.live_positions
    def get_positions_checked(self): return self.live_positions, True


class _StubDB:
    def __init__(self): self.store = {}
    def get_setting(self, k, default=None): return self.store.get(k, default)
    def set_setting(self, k, v): self.store[k] = v
    def get_smc_ob_state(self, symbol, tf): return None


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


def _tm(enabled, test_mode=False):
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=_StubBybit())
    tm._settings['enabled'] = enabled
    tm._settings['test_mode'] = test_mode
    tm._settings['use_reverse_smc'] = True
    tm._settings['use_bos_partials'] = True
    tm._positions['BTCUSDT'] = {
        'symbol': 'BTCUSDT', 'side': 'LONG', 'entry_price': 100.0, 'qty': 1.0,
        'sl_price': 90.0, 'tp_price': 120.0, 'opened_at': 0, 'opened_by': 'choch',
    }
    tm._shadow_positions['ETHUSDT'] = {
        'symbol': 'ETHUSDT', 'side': 'SHORT', 'entry_price': 200.0, 'qty': 1.0,
        'sl_price': 220.0, 'tp_price': 180.0, 'opened_at': 0, 'opened_by': 'choch',
    }
    tm._get_current_price = lambda s: {'BTCUSDT': 100.0, 'ETHUSDT': 200.0}.get(s)
    return tm


# --- Toggle OFF: manage existing ---

def test_off_reverse_smc_closes_real():
    tm = _tm(enabled=False)
    closed = []
    tm._close_position = lambda sym, price, reason=None: closed.append((sym, reason))
    tm.on_choch_event('BTCUSDT', direction='bear', level=99.0, bar_t=1000)
    assert closed == [('BTCUSDT', 'reverse_smc')], closed
    print('✓ OFF: reverse_smc closes existing REAL position')


def test_off_bos_partial_real():
    tm = _tm(enabled=False)
    called = []
    tm._process_bos_real = lambda *a, **kw: called.append(a)
    tm.on_bos_event('BTCUSDT', direction='bull', level=101.0, bar_t=1000)
    assert len(called) == 1, called
    print('✓ OFF: BOS partial manages existing REAL position')


def test_off_reconcile_runs():
    tm = _tm(enabled=False)
    res = tm._reconcile_with_bybit()
    assert not res.get('skipped'), f"reconcile skipped when disabled: {res}"
    print('✓ OFF: reconcile runs (loads/syncs existing positions)')


# --- Toggle OFF: block new ---

def test_off_blocks_new_real_entry():
    tm = _tm(enabled=False)
    opened = []
    tm._open_position = lambda *a, **kw: opened.append(a)
    tm.on_signal('SOLUSDT', side='LONG', entry_price=50.0, opened_by='choch')
    assert opened == [], opened
    print('✓ OFF: new REAL entry blocked')


# --- Toggle ON: work as usual ---

def test_on_allows_new_real_entry():
    tm = _tm(enabled=True)
    tm._is_tradeable = lambda sym: True
    opened = []
    tm._open_position = lambda *a, **kw: opened.append(a)
    tm.on_signal('SOLUSDT', side='LONG', entry_price=50.0, opened_by='choch')
    assert len(opened) == 1, opened
    print('✓ ON: new REAL entry allowed (works as usual)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll toggle-strategy tests passed ✓ ({len(tests)} tests)')
