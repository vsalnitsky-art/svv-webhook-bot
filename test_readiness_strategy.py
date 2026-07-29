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
    ff._open = lambda sym, d, fuel, s, opened_by=None, skip_ctr_safeguard=False: (
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


# --- ⚡ scalper «good signal» Telegram alert (edge-trigger + cooldown) ---
def _scalp_ff():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        funding_setup_scalp_on=True, scalp_tg_on=True,
        scalp_tg_min_score=60, scalp_tg_cooldown_min=30, scalp_tg_dir='any'))
    ff.sent = []
    ff._send_scalp_alert = lambda sym, a, su, s: ff.sent.append(sym)
    ff._anomalies = {'ENAUSDT': {'dir': 'LONG', 'rate': -1.0}}
    return ff, db


def test_scalp_alert_edge_fires_once():
    ff, db = _scalp_ff()
    ff._setup_scalp_cache = {'ENAUSDT': {'ok': True, 'dir': 'LONG', 'hot': True, 'score': 75}}
    ff._scalp_setup_alert(ff.get_settings(), 2_000_000.0)
    ff._scalp_setup_alert(ff.get_settings(), 2_000_001.0)   # still good → edge already fired
    assert ff.sent == ['ENAUSDT'], f'good signal must alert ONCE on rising edge, got {ff.sent}'
    print('✓ scalp TG → fires once on rising edge (no re-spam while good)')


def test_scalp_alert_below_threshold_silent():
    ff, db = _scalp_ff()
    ff._setup_scalp_cache = {'ENAUSDT': {'ok': True, 'dir': 'LONG', 'hot': False, 'score': 45}}
    ff._scalp_setup_alert(ff.get_settings(), 2_000_000.0)
    assert ff.sent == [], 'score 45 < 60 and not HOT → no alert'
    print('✓ scalp TG → silent below threshold & not HOT')


def test_scalp_alert_off_no_send():
    ff, db = _scalp_ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'), scalp_tg_on=False))
    ff._setup_scalp_cache = {'ENAUSDT': {'ok': True, 'dir': 'LONG', 'hot': True, 'score': 90}}
    ff._scalp_setup_alert(ff.get_settings(), 2_000_000.0)
    assert ff.sent == [], 'scalp_tg_on=False → no alert'
    print('✓ scalp TG OFF → no send')


def test_scalp_alert_dir_filter():
    ff, db = _scalp_ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'), scalp_tg_dir='SHORT'))
    ff._setup_scalp_cache = {'ENAUSDT': {'ok': True, 'dir': 'LONG', 'hot': True, 'score': 90}}
    ff._scalp_setup_alert(ff.get_settings(), 2_000_000.0)
    assert ff.sent == [], 'dir filter SHORT → LONG signal suppressed'
    print('✓ scalp TG → direction filter works')


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


# --- 🎯 funding 5-layer confluence ---
def test_funding_layers_all_five():
    ff, db = _ff()
    sym = 'ENAUSDT'
    ff._fuel_dir_smoothed = lambda s: {'status': 'LONG', 'mark_price': 1.0}
    ff._fuel_str = {sym: 20}                                   # 1) МММ легкий у бік
    ff._fuel_str_prev = {sym: 10}                              #    + сила РОСТЕ (↑)
    ff._score_cache = {sym: {'dir': 'LONG', 'score': 45, 'label': 'СЕРЕДНІЙ'}}  # 2)
    ff._setup_cache = {sym: {'ok': True, 'dir': 'LONG', 'score': 40, 'grade': 'СЕРЕДНІЙ'}}  # 3)
    ff._setup_scalp_cache = {sym: {'ok': True, 'dir': 'LONG', 'score': 50, 'grade': 'ХОРОШИЙ'}}  # 4)
    lay = ff._funding_layers(sym, {'dir': 'LONG'})
    vob = next(l for l in lay['layers'] if l['key'] == 'vob')
    # 5-й (VOB) — одноразовий тригер: у колонці ЗАВЖДИ off. base4=4, count=4.
    assert lay['base4'] == 4 and not vob['ok'] and lay['count'] == 4, \
        [(l['key'], l['ok']) for l in lay['layers']]
    print('✓ funding layers: 1-4 засвічені, 5-й (VOB) завжди off (одноразовий тригер)')


def test_funding_layers_direction_and_thresholds():
    ff, db = _ff()
    sym = 'SUIUSDT'
    ff._fuel_dir_smoothed = lambda s: {'status': 'LONG', 'mark_price': 1.0}  # МММ ПРОТИ SHORT → fail
    ff._fuel_str = {sym: 50}
    ff._score_cache = {sym: {'dir': 'SHORT', 'score': 45}}     # у бік + ≥40 → ok
    ff._setup_cache = {sym: {'ok': True, 'dir': 'SHORT', 'score': 20}}  # <38 → fail
    ff._setup_scalp_cache = {}                                 # немає → fail
    ff._funding_trends = {sym: 0.3}                            # не поглиблюється → fail
    lay = ff._funding_layers(sym, {'dir': 'SHORT'})
    assert lay['count'] == 1, [(l['key'], l['ok']) for l in lay['layers']]
    print('✓ funding layers: рахує лише збіги в бік напрямку + пороги')


def test_layer_alert_vob_confirms_and_dedups():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        layer_tg_on=True, layer_tg_min=5, layer_tg_cooldown_min=0))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG', 'rate': -1.0}}
    ff._funding_layers = lambda s, a: {'base4': 4, 'count': 5, 'layers': []}
    ff._funding_vob = lambda sym, d: {'formation_time': 111, 'top': 1.2, 'bottom': 1.1, 'breaker': False}
    ff.sent = []
    ff._send_layer_alert = lambda sym, a, lay, ob, s: ff.sent.append((sym, ob['formation_time']))
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    ff._layer_signal_alert(ff.get_settings(), 2_000_010.0)   # same OB → no re-send
    assert ff.sent == [(sym, 111)], ff.sent
    ff._funding_vob = lambda sym, d: {'formation_time': 222, 'top': 1.3, 'bottom': 1.2, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_020.0)   # NEW OB → new alert
    assert ff.sent == [(sym, 111), (sym, 222)], ff.sent
    print('✓ layer TG: спрацьовує на НОВИЙ Volumized OB (1m), не дублює той самий')


def test_layer_alert_needs_base4():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        layer_tg_on=True, layer_tg_min=5, layer_tg_cooldown_min=0))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG'}}
    ff._funding_layers = lambda s, a: {'base4': 3, 'count': 3, 'layers': []}   # 1-4 не всі
    ff.sent = []
    ff._send_layer_alert = lambda *a, **k: ff.sent.append(1)
    ff._funding_vob = lambda sym, d: {'formation_time': 111, 'top': 1, 'bottom': 1, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    assert ff.sent == [], 'без усіх базових шарів 1-4 — VOB не перевіряється, сигналу нема'
    print('✓ layer TG: 5-й (новий VOB) лише ПІСЛЯ зходження шарів 1-4')


# --- ✦ Золотий funding: чисті рівні (цілі 1..4 + виняток 0.5) ---
def test_gold_step_levels():
    ff, db = _ff()
    g = ff._gold_funding_step
    # Чисті цілі → золото
    assert g({'rate': -2.000}, 0.005) == 2.0
    assert g({'rate': 3.000}, 0.005) == 3.0
    assert g({'rate': -1.000}, 0.005) == 1.0
    assert g({'rate': -4.000}, 0.005) == 4.0
    # Чисті половинки (крок 0.5) → золото
    assert g({'rate': -0.500}, 0.005) == 0.5
    assert g({'rate': -1.500}, 0.005) == 1.5
    assert g({'rate': -2.500}, 0.005) == 2.5
    assert g({'rate': -3.500}, 0.005) == 3.5
    # Не чисте / поза діапазоном
    assert g({'rate': -1.007}, 0.005) is None
    assert g({'rate': -0.520}, 0.005) is None   # поза допуском 0.5
    assert g({'rate': -1.480}, 0.005) is None   # поза допуском 1.5
    assert g({'rate': -5.000}, 0.005) is None   # поза діапазоном
    assert g({'rate': -0.250}, 0.005) is None   # чверть — не крок 0.5
    assert g({'rate': -0.502}, 0.005) == 0.5    # у межах допуску → чистий 0.5
    assert g({'rate': -2.498}, 0.005) == 2.5    # у межах допуску → чистий 2.5
    print('✓ gold levels: крок 0.5 у 0.5..4.0; проміжні й поза діапазоном — ні')


# --- ✦ Золотий funding: відразу на появі + повтор раз на кулдаун ---
def test_gold_funding_immediate_then_repeat():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        funding_gold_tg=True, funding_gold_cooldown_min=60,
        spike_tg=False, spike_auto_open=False, opportunity_auto_open=False))
    sym = 'DEXEUSDT'
    ff._anomalies = {sym: {'dir': 'SHORT', 'rate': -2.0}}
    ff._opportunity_for = lambda sym, a, s: (0, False, [])
    ff._gold_funding_step = lambda a, tol: 2.0
    ff.gold_sent = []
    ff._send_gold_alert = lambda sym, a, step, held: ff.gold_sent.append((sym, held))
    s = ff.get_settings()
    ff._opportunity_alert(s, 1_000_000.0)               # зʼявився → ВІДРАЗУ
    ff._opportunity_alert(s, 1_000_030.0)               # +30с (< 60хв) → без повтору
    assert len(ff.gold_sent) == 1, ff.gold_sent
    ff._opportunity_alert(s, 1_000_000.0 + 61 * 60)     # +61хв → повтор
    assert len(ff.gold_sent) == 2, ff.gold_sent
    print('✓ gold funding: відразу на появі + повтор раз на кулдаун (60хв)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll «Готовність» strategy tests passed ✓ ({len(tests)} tests)')
