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


# --- HOLD (not hot AND below score threshold) ---
def test_below_threshold_holds():
    ff, db = _ff()   # default queue3_open_min_score = 53
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(False, score=40, grade='СЕРЕДНІЙ')}
    ff._engine_tick_readiness()
    assert not ff.opened, 'not-HOT & score<threshold must not open'
    assert 'SOLUSDT' in ff._pending3, 'held coin stays in Queue 3'
    assert any(r['outcome'] == 'hold' for r in db.readiness_rows), db.readiness_rows
    print('✓ not HOT & SCORE<53 → holds, stays queued, logs hold')


# --- OPEN via score threshold (not hot, but SCORE >= threshold) ---
def test_score_threshold_opens():
    ff, db = _ff()   # default queue3_open_min_score = 53
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(False, score=60, grade='ХОРОШИЙ')}
    ff._engine_tick_readiness()
    assert ff.opened and ff.opened[0][0] == 'SOLUSDT', 'SCORE 60 ≥ 53 must open'
    assert any(r['outcome'] == 'opened' for r in db.readiness_rows), db.readiness_rows
    print('✓ not HOT but SCORE 60 ≥ 53 → opens')


# --- score threshold = 0 → only HOT opens ---
def test_threshold_zero_only_hot():
    ff, db = _ff()
    ff._db.set_setting('fuel_filter_settings', dict(
        ff._db.get_setting('fuel_filter_settings'), queue3_open_min_score=0))
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 0}}
    ff._setup_cache = {'SOLUSDT': _grade(False, score=66, grade='ХОРОШИЙ')}
    ff._engine_tick_readiness()
    assert not ff.opened, 'threshold=0 → non-HOT must not open even at 66'
    print('✓ threshold 0 → лише HOT (SCORE 66 не відкриває)')


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
    ff._setup_cache = {'SOLUSDT': _grade(False, score=40, grade='СЕРЕДНІЙ')}
    ff._engine_tick_readiness()
    ff._engine_tick_readiness()   # same decision immediately → throttled
    holds = [r for r in db.readiness_rows if r['outcome'] == 'hold']
    assert len(holds) == 1, f'unchanged hold should log once, got {len(holds)}'
    print('✓ unchanged hold is throttled (logged once)')


# --- ⚡ funding scalper «Готовність» cache ---
def test_scalp_off_clears_cache():
    ff, db = _ff()   # funding_setup_scalp_on default False
    ff._setup_scalp_cache = {'X': {'ok': True}}
    ff._setup_scalp_at = {'X': 1.0}
    ff._refresh_setup_scalp_cache(ff.get_settings())
    assert ff._setup_scalp_cache == {} and ff._setup_scalp_at == {}, 'must clear when off'
    print('✓ scalp OFF → cache cleared')


def test_scalp_on_grades_funding_on_tf():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        funding_setup_scalp_on=True, funding_setup_tf='5m', funding_setup_htf='15m'))
    ff._anomalies = {'ENAUSDT': {'dir': 'LONG'}, 'SUIUSDT': {'dir': 'SHORT'}}
    calls = []
    ff._compute_setup = lambda sym, d, s, base_tf='1h', htf_tf='4h', write_exit=True: (
        calls.append((sym, d, base_tf, htf_tf, write_exit)) or {'ok': True, 'score': 60, 'dir': d})
    ff._refresh_setup_scalp_cache(ff.get_settings())
    assert set(ff._setup_scalp_cache) == {'ENAUSDT', 'SUIUSDT'}, ff._setup_scalp_cache
    assert all(c[2] == '5m' and c[3] == '15m' and c[4] is False for c in calls), calls
    print('✓ scalp ON → grades funding coins on 5m/15m, write_exit=False (не чіпає exit)')


def test_scalp_drops_left_coins():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'), funding_setup_scalp_on=True))
    ff._setup_scalp_cache = {'GONEUSDT': {'ok': True}}
    ff._setup_scalp_at = {'GONEUSDT': 1.0}
    ff._anomalies = {'ENAUSDT': {'dir': 'LONG'}}
    ff._compute_setup = lambda *a, **k: {'ok': True, 'score': 60, 'dir': 'LONG'}
    ff._refresh_setup_scalp_cache(ff.get_settings())
    assert 'GONEUSDT' not in ff._setup_scalp_cache, 'coin gone from funding table must drop'
    assert 'ENAUSDT' in ff._setup_scalp_cache
    print('✓ scalp → drops coins that left the funding table')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll «Готовність» strategy tests passed ✓ ({len(tests)} tests)')
