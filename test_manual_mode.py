"""Test the per-position Manual Mode toggle in trade_manager.

When manual_mode is ON for a position:
  - on_signal ignores new SMC signals for that symbol
  - on_choch_event skips CHoCH-driven exits
  - _monitor_position bypasses automatic exits BUT still runs manual SL/TP
  - Force close (manual_close / manual_close_shadow) still works
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


def _make_tm():
    """Build a TradeManager with stubs and place a fake REAL + SHADOW position."""
    tm = TradeManager(db=_StubDB(), notifier=None, bybit=_StubBybit())
    # Inject a real position so the manual mode codepaths have something to act on
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
    return tm


# ============================================================================
# update_manual_mode — core API
# ============================================================================

def test_enable_manual_mode_on_real_position():
    tm = _make_tm()
    res = tm.update_manual_mode('BTCUSDT', enabled=True, is_shadow=False)
    assert res['ok'] is True
    assert res['manual_mode'] is True
    assert tm._positions['BTCUSDT']['manual_mode'] is True
    print('✓ Enable manual mode on REAL position sets the flag')


def test_enable_manual_mode_on_shadow_position():
    tm = _make_tm()
    res = tm.update_manual_mode('ETHUSDT', enabled=True, is_shadow=True)
    assert res['ok'] is True
    assert tm._shadow_positions['ETHUSDT']['manual_mode'] is True
    print('✓ Enable manual mode on SHADOW position sets the flag')


def test_disable_manual_mode():
    tm = _make_tm()
    tm.update_manual_mode('BTCUSDT', enabled=True, is_shadow=False)
    res = tm.update_manual_mode('BTCUSDT', enabled=False, is_shadow=False)
    assert res['ok'] is True
    assert res['manual_mode'] is False
    assert tm._positions['BTCUSDT']['manual_mode'] is False
    print('✓ Disable manual mode resets the flag to False')


def test_update_unknown_symbol_returns_error():
    tm = _make_tm()
    res = tm.update_manual_mode('NONEXISTENT', enabled=True, is_shadow=False)
    assert res['ok'] is False
    assert 'No open' in res['reason']
    print('✓ Toggling unknown symbol returns error')


def test_real_and_shadow_are_independent():
    """Toggling REAL manual_mode does not affect SHADOW and vice versa."""
    tm = _make_tm()
    # Add same symbol to both stores
    tm._positions['XRPUSDT'] = {
        'symbol': 'XRPUSDT', 'side': 'LONG',
        'entry_price': 1.0, 'qty': 100, 'sl_price': 0.9,
        'tp_price': 1.2, 'opened_at': 0, 'opened_by': 'choch',
    }
    tm._shadow_positions['XRPUSDT'] = {
        'symbol': 'XRPUSDT', 'side': 'LONG',
        'entry_price': 1.0, 'qty': 100, 'sl_price': 0.9,
        'tp_price': 1.2, 'opened_at': 0, 'opened_by': 'choch',
    }
    tm.update_manual_mode('XRPUSDT', enabled=True, is_shadow=False)
    assert tm._positions['XRPUSDT']['manual_mode'] is True
    assert tm._shadow_positions['XRPUSDT'].get('manual_mode') in (False, None)
    print('✓ Toggling REAL leaves SHADOW unchanged (and vice versa)')


# ============================================================================
# on_signal — gating new entries
# ============================================================================

def test_new_signal_blocked_on_manual_real_position():
    """A new SMC signal on a symbol whose REAL position is in manual mode
    is ignored — no reverse, no _open_position."""
    tm = _make_tm()
    tm._positions['BTCUSDT']['manual_mode'] = True
    # Track whether _open_position was called
    open_calls = []
    tm._open_position = lambda *a, **kw: open_calls.append(a)
    # Also track _close_position so we can prove reverse logic is bypassed
    close_calls = []
    tm._close_position = lambda *a, **kw: close_calls.append(a)
    # Fire opposite-direction signal — would normally trigger reverse
    tm.on_signal('BTCUSDT', side='SHORT', entry_price=99.0, opened_by='choch')
    assert open_calls == []
    assert close_calls == []
    print('✓ on_signal ignored when REAL position is in manual mode')


def test_new_signal_blocked_on_manual_shadow_position():
    tm = _make_tm()
    tm._shadow_positions['ETHUSDT']['manual_mode'] = True
    open_calls = []
    tm._open_shadow = lambda *a, **kw: open_calls.append(a)
    close_calls = []
    tm._close_shadow = lambda *a, **kw: close_calls.append(a)
    tm.on_signal('ETHUSDT', side='LONG', entry_price=201.0, opened_by='choch')
    assert open_calls == []
    assert close_calls == []
    print('✓ on_signal ignored when SHADOW position is in manual mode')


def test_normal_signal_still_works_when_no_manual_mode():
    """Sanity check: with manual_mode=False (default), the on_signal path
    is unaffected — same-direction is no-op, no error."""
    tm = _make_tm()
    # Same-direction signal — should silently no-op (existing logic)
    open_calls = []
    tm._open_position = lambda *a, **kw: open_calls.append(a)
    tm.on_signal('BTCUSDT', side='LONG', entry_price=101.0, opened_by='choch')
    assert open_calls == []  # same direction, no open
    print('✓ Normal (non-manual) signal handling unchanged')


# ============================================================================
# on_choch_event — gating CHoCH-driven exits
# ============================================================================

def test_choch_exit_blocked_in_manual_mode():
    """When manual_mode=True, opposite-CHoCH does not trigger Reverse SMC
    or Forecast 1H Confluence closes."""
    tm = _make_tm()
    tm._positions['BTCUSDT']['manual_mode'] = True
    close_calls = []
    tm._close_position = lambda *a, **kw: close_calls.append(a)
    tm._close_shadow = lambda *a, **kw: close_calls.append(('shadow', a))
    # Opposite CHoCH for LONG position → bear
    tm.on_choch_event('BTCUSDT', direction='bear', level=99.0, bar_t=1000)
    assert close_calls == []
    print('✓ on_choch_event skipped in manual mode')


# ============================================================================
# Manual SL/TP STILL fires in manual mode (the only auto-exit path retained)
# ============================================================================

def test_manual_sl_still_fires_in_manual_mode():
    """The whole point of manual mode is the user can still set a manual
    SL via the UI and have it enforced. Verify _check_manual_sl_tp works
    independently of manual_mode flag."""
    tm = _make_tm()
    pos = tm._positions['BTCUSDT']
    pos['manual_mode'] = True
    pos['manual_sl'] = 95.0  # LONG entry 100, stop at 95
    reason = tm._check_manual_sl_tp(pos, current_price=94.5)
    assert reason == 'manual_sl'
    print('✓ Manual SL fires even in manual mode (intended)')


def test_manual_tp_still_fires_in_manual_mode():
    tm = _make_tm()
    pos = tm._positions['BTCUSDT']
    pos['manual_mode'] = True
    pos['manual_tp'] = 110.0  # LONG entry 100, take profit at 110
    reason = tm._check_manual_sl_tp(pos, current_price=110.5)
    assert reason == 'manual_tp'
    print('✓ Manual TP fires even in manual mode (intended)')


# ============================================================================
# Persistence
# ============================================================================

def test_manual_mode_persists_to_db():
    """Toggle should call _persist_positions so flag survives restart."""
    tm = _make_tm()
    persist_calls = []
    orig = tm._persist_positions
    tm._persist_positions = lambda: persist_calls.append('real') or orig()
    tm.update_manual_mode('BTCUSDT', enabled=True, is_shadow=False)
    assert persist_calls == ['real']
    print('✓ Real manual_mode toggle triggers persist')


def test_shadow_manual_mode_persists_separately():
    tm = _make_tm()
    persist_calls = []
    orig = tm._persist_shadow_positions
    tm._persist_shadow_positions = lambda: persist_calls.append('shadow') or orig()
    tm.update_manual_mode('ETHUSDT', enabled=True, is_shadow=True)
    assert persist_calls == ['shadow']
    print('✓ Shadow manual_mode toggle triggers shadow persist')


if __name__ == '__main__':
    test_enable_manual_mode_on_real_position()
    test_enable_manual_mode_on_shadow_position()
    test_disable_manual_mode()
    test_update_unknown_symbol_returns_error()
    test_real_and_shadow_are_independent()
    test_new_signal_blocked_on_manual_real_position()
    test_new_signal_blocked_on_manual_shadow_position()
    test_normal_signal_still_works_when_no_manual_mode()
    test_choch_exit_blocked_in_manual_mode()
    test_manual_sl_still_fires_in_manual_mode()
    test_manual_tp_still_fires_in_manual_mode()
    test_manual_mode_persists_to_db()
    test_shadow_manual_mode_persists_separately()
    print()
    print('All Manual Mode tests passed ✓')
