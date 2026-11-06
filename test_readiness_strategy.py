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
    # ⚠️ Стаб мусить приймати ВСІ kwargs справжнього `_open` — інакше двигун
    # ловить TypeError, мовчки нічого не відкриває, і тест падає «на рівному
    # місці» (саме так він і відстав після появи signal_at/origin_trace).
    ff._open = lambda sym, d, fuel, s, opened_by=None, skip_ctr_safeguard=False, \
        skip_exhaustion=False, by_hand=False, signal_at=None, origin_trace=None, **_kw: (
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


# --- Queue-3 TTL ejects stale coins ---
def test_queue3_ttl_ejects_stale():
    ff, db = _ff()   # queue3_ttl_hours default 6
    ff._pending3 = {'SOLUSDT': {'dir': 'LONG', 'added_at': 1.0}}   # ancient
    ff._setup_cache = {'SOLUSDT': _grade(True)}
    ff._engine_tick_readiness()
    assert 'SOLUSDT' not in ff._pending3, 'stale coin must be ejected by TTL'
    assert not ff.opened, 'ejected by TTL, not opened'
    assert any(r['outcome'] == 'skipped' and 'протерміновано' in (r.get('reason') or '')
               for r in db.readiness_rows), db.readiness_rows
    print('✓ Черга-3 TTL: протерміновує застарілу монету (не тримає годинами)')


# --- CTR safeguard skip for Q3 (reversal) opens ---
def test_safeguard_skips_ctr_for_q3():
    ff, db = _ff()
    s = ff.get_settings()   # safeguard_on + safeguard_ctr default True
    ff._fuel_str = {'ENAUSDT': 50}            # MMM ok (≥30)
    ff._exhaustion = lambda sym, side: 10.0   # not exhausted
    ff._ctr_state = lambda sym, band: ('SHORT', 80, 0)   # CTR AGAINST the LONG
    ok, reason = ff._soft_safeguard('ENAUSDT', 'LONG', s, skip_ctr=True)
    assert ok, f'skip_ctr must bypass the CTR safeguard, got: {reason}'
    ok2, reason2 = ff._soft_safeguard('ENAUSDT', 'LONG', s, skip_ctr=False)
    assert not ok2 and 'CTR' in reason2, (ok2, reason2)
    print('✓ safeguard: skip_ctr bypasses CTR (Q3 reversals), normal still blocks')


# --- 🎯 Черга-3: авто-відкриття за VOB + 5 шарів + SL з блоку OB ---
class _FakeTM:
    def __init__(self):
        self._positions = {}
        self._shadow_positions = {}
        self.sl_calls = []

    def update_manual_sl_tp(self, symbol, manual_sl=None, manual_tp=None, is_shadow=False):
        self.sl_calls.append((symbol, manual_sl, is_shadow))
        return {'ok': True}


# --- 🛡 Запобіжник: виснаженість (skip) + МММ price-override + OB-match ---
def test_soft_safeguard_skip_exhaustion():
    ff, db = _ff()
    s = ff.get_settings()
    sym = 'NEARUSDT'
    ff._fuel_str = {sym: 50}                       # МММ ok (≥30)
    ff._exhaustion = lambda sym, side: 88.0        # виснажено > 80
    ok, reason = ff._soft_safeguard(sym, 'SHORT', s, skip_ctr=True, skip_exhaustion=False)
    assert not ok and 'виснажено' in reason, (ok, reason)
    ok2, _ = ff._soft_safeguard(sym, 'SHORT', s, skip_ctr=True, skip_exhaustion=True)
    assert ok2, 'skip_exhaustion → виснаженість не ріже'
    print('✓ safeguard: skip_exhaustion знімає жорстке вето виснаженості (Черга-3)')


def test_soft_safeguard_mm_price_override():
    ff, db = _ff()
    s = dict(ff.get_settings(), safeguard_mm_min=30, safeguard_mm_price_override=True)
    sym = 'NEARUSDT'
    ff._fuel_str = {sym: 5}                         # МММ слабкий (< 30)
    ff._exhaustion = lambda sym, side: 0.0
    ff._candle_momentum = lambda sym, tf: ('SHORT', 1.0)   # ціна чітко в бік SHORT
    ok, reason = ff._soft_safeguard(sym, 'SHORT', s, skip_ctr=True)
    assert ok, ('override має пропустити слабкий МММ, коли ціна в бік', reason)
    ff._candle_momentum = lambda sym, tf: ('LONG', 1.0)    # ціна НЕ в бік
    ok2, reason2 = ff._soft_safeguard(sym, 'SHORT', s, skip_ctr=True)
    assert not ok2 and 'МММ слабкий' in reason2, (ok2, reason2)
    s_off = dict(s, safeguard_mm_price_override=False)      # override вимкнено
    ff._candle_momentum = lambda sym, tf: ('SHORT', 1.0)
    ok3, _ = ff._soft_safeguard(sym, 'SHORT', s_off, skip_ctr=True)
    assert not ok3, 'override off → слабкий МММ ріже навіть коли ціна в бік'
    print('✓ safeguard: МММ-override пропускає слабкий МММ лише коли ціна чітко в бік')


def test_ob_match_gate():
    import types as _t, sys as _s
    ff, db = _ff()
    s = ff.get_settings()
    sym = 'ENAUSDT'
    # 1) queue3_require_ob_match=False → завжди ok
    assert ff._ob_match_ok(sym, 'LONG', dict(s, queue3_require_ob_match=False))[0]
    # 2) tm/scanner з увімкненим OB-фільтром + fake get_smc_ob_state
    class _Scan:
        _settings = {'ob_filter_enabled': True, 'ob_filter_timeframe': '1h'}
    class _TM:
        scanner = _Scan()
    ff._get_tm = lambda: _TM()
    _bias = {'v': 'BEARISH'}
    fake = _t.ModuleType('storage.db_operations')
    class _DB:
        def get_smc_ob_state(self, sym, tf): return {'bias': _bias['v']}
    fake.get_db = lambda: _DB()
    _saved_pkg = _s.modules.get('storage')
    _saved_mod = _s.modules.get('storage.db_operations')
    _s.modules.setdefault('storage', _t.ModuleType('storage'))
    _s.modules['storage.db_operations'] = fake
    try:
        ok_ag, r = ff._ob_match_ok(sym, 'LONG', s)      # OB BEARISH проти LONG → блок
        assert not ok_ag and 'проти' in r, (ok_ag, r)
        _bias['v'] = 'BULLISH'
        assert ff._ob_match_ok(sym, 'LONG', s)[0]         # OB у бік → ok
        _bias['v'] = None
        assert ff._ob_match_ok(sym, 'LONG', s)[0]         # немає OB → не блокуємо
    finally:
        if _saved_mod is not None: _s.modules['storage.db_operations'] = _saved_mod
        else: _s.modules.pop('storage.db_operations', None)
        if _saved_pkg is not None: _s.modules['storage'] = _saved_pkg
    print('✓ OB-match: блок лише на явний контр-OB; у бік/немає/вимк → ok')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll «Готовність» strategy tests passed ✓ ({len(tests)} tests)')
