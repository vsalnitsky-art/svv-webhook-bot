"""«Готовність» (Readiness) strategy engine tests.

The readiness engine opens a queued coin the moment its SCORE = STRONG HOLD in
the queued direction — no ₿ START / session gating — and logs EVERY decision to
the readiness_log table + the event log. These tests pin that decision logic
and the logging contract with light stubs (no DB / exchange).
"""

import sys, types, importlib.util
sys.path.insert(0, '.')

# Load fuel_filter.py directly, bypassing detection/__init__ (which pulls heavy
# exchange deps not needed here). Mirror the pattern in test_toggle_strategy.py.
if 'detection' not in sys.modules:
    detection_pkg = types.ModuleType('detection')
    detection_pkg.__path__ = ['./detection']
    sys.modules['detection'] = detection_pkg

_spec = importlib.util.spec_from_file_location(
    'detection.fuel_filter', 'detection/fuel_filter.py')
_ff_mod = importlib.util.module_from_spec(_spec)
sys.modules['detection.fuel_filter'] = _ff_mod
_spec.loader.exec_module(_ff_mod)
FuelFilterDaemon = _ff_mod.FuelFilterDaemon


class _StubDB:
    def __init__(self):
        self.store = {}
        self.readiness_rows = []
        self.events = []

    def get_setting(self, k, default=None):
        return self.store.get(k, default)

    def set_setting(self, k, v):
        self.store[k] = v

    def log_readiness(self, **fields):
        self.readiness_rows.append(fields)

    def log_event(self, message, level='INFO', category='SYSTEM', symbol=None):
        self.events.append({'message': message, 'level': level,
                            'category': category, 'symbol': symbol})


def _ff(strategy='readiness'):
    db = _StubDB()
    ff = FuelFilterDaemon(db=db, get_trade_manager=lambda: None,
                          get_watchlist=lambda: [])
    # Active strategy + FF on.
    db.set_setting('fuel_filter_settings', {
        'enabled': True,
        'active_strategy': strategy,
        'readiness_log_enabled': True,
        'max_exhaustion_pct': 75,
    })
    # Common stubs — both buttons ON, coin has price, no existing position.
    ff._entry_gates = lambda: (True, True)
    ff._tm_has_position = lambda sym, real: False
    ff._fuel_dir_smoothed = lambda sym: {'status': 'LONG', 'dir': 0.5,
                                         'mark_price': 100.0}
    ff._exhaustion = lambda sym, d: 10.0
    ff.opened = []
    ff._open = lambda sym, d, fuel, s, opened_by=None: (ff.opened.append((sym, d, opened_by)) or True)
    return ff, db


def _score(label, d='LONG', score=80):
    return {'score': score, 'label': label, 'dir': d, 'exh': 10.0,
            'conflict': False, 'fuel_strength': 50,
            'components': {'room': 0.9, 'hold': 0.0, 'fuel': 0.8, 'mom': 1.0}}


# --- OPEN path ---

def test_strong_hold_opens_and_logs():
    ff, db = _ff()
    ff._pending = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._timer_score_for = lambda *a, **k: _score('STRONG HOLD')
    ff._engine_tick()
    assert ff.opened and ff.opened[0][0] == 'SOLUSDT', ff.opened
    assert 'SOLUSDT' not in ff._pending, 'opened coin must leave the queue'
    opened_rows = [r for r in db.readiness_rows if r['outcome'] == 'opened']
    assert opened_rows, db.readiness_rows
    assert any(e['category'] == 'READINESS' for e in db.events)
    print('✓ STRONG HOLD → opens, leaves queue, logs opened (DB + event)')


# --- HOLD path (wrong label) ---

def test_weak_holds_no_open():
    ff, db = _ff()
    ff._pending = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._timer_score_for = lambda *a, **k: _score('WEAK')
    ff._engine_tick()
    assert not ff.opened, 'WEAK must not open'
    assert 'SOLUSDT' in ff._pending, 'held coin stays in queue'
    assert any(r['outcome'] == 'hold' for r in db.readiness_rows), db.readiness_rows
    print('✓ WEAK → holds, stays queued, logs hold')


# --- HOLD path (label ok, direction mismatch) ---

def test_dir_mismatch_holds():
    ff, db = _ff()
    ff._pending = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._timer_score_for = lambda *a, **k: _score('STRONG HOLD', d='SHORT')
    ff._engine_tick()
    assert not ff.opened, 'SCORE dir SHORT ≠ queue LONG must not open'
    print('✓ STRONG HOLD but dir mismatch → holds')


# --- Exhaustion gate ---

def test_exhaustion_gate_skips():
    ff, db = _ff()
    ff._exhaustion = lambda sym, d: 90.0     # > max_exhaustion_pct (75)
    ff._pending = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._timer_score_for = lambda *a, **k: _score('STRONG HOLD')
    ff._engine_tick()
    assert not ff.opened, 'too-exhausted coin must be skipped'
    assert any(r['outcome'] == 'skipped' for r in db.readiness_rows), db.readiness_rows
    print('✓ exhaustion > max → skipped, logged')


# --- Strategy isolation: fuel strategy must NOT use the readiness path ---

def test_fuel_strategy_ignores_readiness_engine():
    ff, db = _ff(strategy='fuel')
    ff._pending = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._timer_score_for = lambda *a, **k: _score('STRONG HOLD')
    # Fuel strategy with both engine modes OFF → engine idle, no readiness log.
    ff._engine_tick()
    assert not db.readiness_rows, 'fuel strategy must not write readiness log'
    print('✓ fuel strategy does not touch the readiness engine/log')


# --- Log throttle: unchanged hold not re-logged within the gap ---

def test_hold_log_throttled():
    ff, db = _ff()
    ff._pending = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._timer_score_for = lambda *a, **k: _score('WEAK')
    ff._engine_tick()
    ff._engine_tick()   # same decision immediately → throttled
    holds = [r for r in db.readiness_rows if r['outcome'] == 'hold']
    assert len(holds) == 1, f'unchanged hold should log once, got {len(holds)}'
    print('✓ unchanged hold is throttled (logged once)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll readiness-strategy tests passed ✓ ({len(tests)} tests)')
