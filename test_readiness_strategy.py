"""🎯 «Готовність» (Queue-3) strategy engine tests.

The readiness engine (_engine_tick_readiness) opens a queued coin the moment its
SMC setup-grade (grade_setup, cached in _setup_cache) is HOT in the queued
direction — no ₿ START / session gating — and logs EVERY decision. These tests
pin that decision logic + logging with light stubs (no DB / exchange).
"""

import sys, types, importlib.util
sys.path.insert(0, '.')

# Load fuel_filter.py directly, bypassing detection/__init__ (heavy deps).
if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection')
    _pkg.__path__ = ['./detection']
    sys.modules['detection'] = _pkg
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

    def get_setting(self, k, default=None):
        return self.store.get(k, default)

    def set_setting(self, k, v):
        self.store[k] = v

    def log_readiness(self, **fields):
        self.readiness_rows.append(fields)


def _ff(queue3=True):
    db = _StubDB()
    ff = FuelFilterDaemon(db=db, get_trade_manager=lambda: None,
                          get_watchlist=lambda: [])
    db.set_setting('fuel_filter_settings', {
        'enabled': True,
        'queue1_enabled': False,
        'queue2_enabled': False,
        'queue3_enabled': queue3,
        'readiness_log_enabled': True,
        'max_exhaustion_pct': 75,
    })
    ff._entry_gates = lambda: (True, True)
    ff._tm_has_position = lambda sym, real: False
    ff._fuel_dir_smoothed = lambda sym: {'status': 'LONG', 'dir': 0.5,
                                         'mark_price': 100.0}
    ff.opened = []
    ff._open = lambda sym, d, fuel, s, opened_by=None: (
        ff.opened.append((sym, d, opened_by)) or True)
    return ff, db


def _grade(hot, d='LONG', score=75, grade='ВІДМІННИЙ'):
    return {'ok': True, 'dir': d, 'score': score, 'grade': grade, 'hot': hot,
            'blocks': {'structure': 0.8, 'poi': 0.6, 'zone': 0.7,
                       'liquidity': 0.6, 'mm': 0.5, 'timing': 0.6,
                       'context': 0.4},
            'vetoes': []}


# --- OPEN path ---
def test_hot_opens_and_logs():
    ff, db = _ff()
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(True)}
    ff._engine_tick_readiness()
    assert ff.opened and ff.opened[0][0] == 'SOLUSDT', ff.opened
    assert 'SOLUSDT' not in ff._pending3, 'opened coin must leave Queue 3'
    assert any(r['outcome'] == 'opened' for r in db.readiness_rows), db.readiness_rows
    print('✓ HOT → opens, leaves Queue 3, logs opened')


# --- HOLD (not hot) ---
def test_not_hot_holds():
    ff, db = _ff()
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(False, score=55, grade='ХОРОШИЙ')}
    ff._engine_tick_readiness()
    assert not ff.opened, 'not-HOT must not open'
    assert 'SOLUSDT' in ff._pending3, 'held coin stays in Queue 3'
    assert any(r['outcome'] == 'hold' for r in db.readiness_rows), db.readiness_rows
    print('✓ not HOT → holds, stays queued, logs hold')


# --- HOLD (hot but direction mismatch) ---
def test_dir_mismatch_holds():
    ff, db = _ff()
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(True, d='SHORT')}
    ff._engine_tick_readiness()
    assert not ff.opened, 'HOT but grade dir SHORT ≠ queue LONG must not open'
    print('✓ HOT but dir mismatch → holds')


# --- HOLD (setup not computed yet) ---
def test_setup_missing_holds():
    ff, db = _ff()
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {}   # not graded yet
    ff._engine_tick_readiness()
    assert not ff.opened
    assert any(r['outcome'] == 'hold' for r in db.readiness_rows), db.readiness_rows
    print('✓ setup not computed → holds')


# --- Strategy isolation: queue3 OFF → engine no-op ---
def test_queue3_off_noop():
    ff, db = _ff(queue3=False)
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(True)}
    ff._engine_tick_readiness()
    assert not ff.opened, 'queue3 OFF must not open'
    assert not db.readiness_rows, 'queue3 OFF must not log'
    print('✓ queue3 OFF → engine no-op')


# --- Log throttle: unchanged hold logged once ---
def test_hold_log_throttled():
    ff, db = _ff()
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(False, score=55, grade='ХОРОШИЙ')}
    ff._engine_tick_readiness()
    ff._engine_tick_readiness()   # same decision immediately → throttled
    holds = [r for r in db.readiness_rows if r['outcome'] == 'hold']
    assert len(holds) == 1, f'unchanged hold should log once, got {len(holds)}'
    print('✓ unchanged hold is throttled (logged once)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll «Готовність» strategy tests passed ✓ ({len(tests)} tests)')
