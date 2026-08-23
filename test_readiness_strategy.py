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
    ff._open = lambda sym, d, fuel, s, opened_by=None, skip_ctr_safeguard=False, skip_exhaustion=False: (
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
    ff._funding_price = {sym: {'dir': 'up', 'chg': 1.4}}       # 5) ЦІНА росте (LONG)
    lay = ff._funding_layers(sym, {'dir': 'LONG'})
    price = next(l for l in lay['layers'] if l['key'] == 'price')
    # 5-й тепер = ЦІНА (постійний шар). Усі 5 світяться → base=count=5.
    assert lay['base'] == 5 and lay['count'] == 5 and price['ok'], \
        [(l['key'], l['ok']) for l in lay['layers']]
    # Кожен засвічений шар несе dir=='LONG' → у колонці усі зелені.
    assert all(l['dir'] == 'LONG' for l in lay['layers'] if l['ok']), \
        [(l['key'], l['dir']) for l in lay['layers']]
    assert lay['base4'] == 4          # сумісність: базові 1-4
    print('✓ funding layers: 5 шарів (5-й=ЦІНА росте), усі dir=LONG (зелені)')


def test_funding_layers_price_direction():
    ff, db = _ff()
    sym = 'WIFUSDT'
    ff._fuel_dir_smoothed = lambda s: {'status': 'SHORT', 'mark_price': 1.0}
    ff._fuel_str = {sym: 20}; ff._fuel_str_prev = {sym: 10}
    ff._score_cache = {sym: {'dir': 'SHORT', 'score': 45}}
    ff._setup_cache = {sym: {'ok': True, 'dir': 'SHORT', 'score': 40}}
    ff._setup_scalp_cache = {sym: {'ok': True, 'dir': 'SHORT', 'score': 40}}
    # SHORT: ЦІНА має СПАДАТИ (down) → шар світиться; up → гасне.
    ff._funding_price = {sym: {'dir': 'down', 'chg': -1.2}}
    lay = ff._funding_layers(sym, {'dir': 'SHORT'})
    price = next(l for l in lay['layers'] if l['key'] == 'price')
    assert price['ok'] and price['dir'] == 'SHORT' and lay['base'] == 5
    ff._funding_price = {sym: {'dir': 'up', 'chg': 1.2}}       # росте проти SHORT
    lay2 = ff._funding_layers(sym, {'dir': 'SHORT'})
    price2 = next(l for l in lay2['layers'] if l['key'] == 'price')
    assert not price2['ok'] and lay2['base'] == 4
    print('✓ funding layers: 5-й ЦІНА — LONG=росте / SHORT=спадає')


def test_funding_layers_mm_trend_asymmetry():
    """Шар МММ: SHORT світиться на ↑ або → (не слабшає); LONG — лише на ↑."""
    ff, db = _ff()
    sym = 'ASYMUSDT'
    # Плато сили (→): prev≈now → ані rising, ані falling.
    ff._fuel_str = {sym: 40}
    ff._fuel_str_prev = {sym: 40}
    # SHORT + плато (→) → МММ СВІТИТЬСЯ.
    ff._fuel_dir_smoothed = lambda s: {'status': 'SHORT', 'mark_price': 1.0}
    ls = next(l for l in ff._funding_layers(sym, {'dir': 'SHORT'})['layers'] if l['key'] == 'mm')
    assert ls['ok'], 'SHORT: плато (→) має світити шар МММ'
    # LONG + плато (→) → МММ ГАСНЕ (потрібне строге ↑).
    ff._fuel_dir_smoothed = lambda s: {'status': 'LONG', 'mark_price': 1.0}
    ll = next(l for l in ff._funding_layers(sym, {'dir': 'LONG'})['layers'] if l['key'] == 'mm')
    assert not ll['ok'], 'LONG: плато (→) НЕ світить шар МММ (як було раніше)'
    # LONG + росте (↑) → МММ світиться.
    ff._fuel_str_prev = {sym: 20}
    ll2 = next(l for l in ff._funding_layers(sym, {'dir': 'LONG'})['layers'] if l['key'] == 'mm')
    assert ll2['ok'], 'LONG: сила РОСТЕ (↑) → шар МММ світиться'
    # SHORT + слабшає (↓) → МММ гасне (тиск згасає).
    ff._fuel_dir_smoothed = lambda s: {'status': 'SHORT', 'mark_price': 1.0}
    ff._fuel_str = {sym: 20}; ff._fuel_str_prev = {sym: 40}
    ls2 = next(l for l in ff._funding_layers(sym, {'dir': 'SHORT'})['layers'] if l['key'] == 'mm')
    assert not ls2['ok'], 'SHORT: сила СЛАБШАЄ (↓) → шар МММ гасне'
    print('✓ funding layers: шар МММ — SHORT ↑/→, LONG лише ↑')


def test_funding_layers_direction_and_thresholds():
    ff, db = _ff()
    sym = 'SUIUSDT'
    ff._fuel_dir_smoothed = lambda s: {'status': 'LONG', 'mark_price': 1.0}  # МММ ПРОТИ SHORT → fail
    ff._fuel_str = {sym: 50}
    ff._score_cache = {sym: {'dir': 'SHORT', 'score': 45}}     # у бік + ≥40 → ok
    ff._setup_cache = {sym: {'ok': True, 'dir': 'SHORT', 'score': 20}}  # <38 → fail
    ff._setup_scalp_cache = {}                                 # немає → fail
    ff._funding_trends = {sym: 0.3}                            # не поглиблюється → fail
    ff._funding_price = {sym: {'dir': 'up', 'chg': 0.8}}       # росте (проти SHORT) → fail
    lay = ff._funding_layers(sym, {'dir': 'SHORT'})
    assert lay['count'] == 1, [(l['key'], l['ok']) for l in lay['layers']]
    print('✓ funding layers: рахує лише збіги в бік напрямку + пороги')


def test_layer_alert_vob_confirms_and_dedups():
    # Чиста «Рекомендація бота» (без авто-відкриття): queue3_vob_open=False.
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        layer_tg_on=True, funding_route_q4=False, layer_tg_min=5, layer_tg_cooldown_min=0,
        queue3_vob_open=False))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG', 'rate': -1.0}}
    ff._funding_layers = lambda s, a: {'base': 5, 'base4': 4, 'count': 5, 'layers': []}
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 1.2, 'bottom': 1.1, 'breaker': False}
    ff.sent = []
    ff._send_layer_alert = lambda sym, a, d, sl=None, mode='signal': ff.sent.append((sym, mode))
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    ff._layer_signal_alert(ff.get_settings(), 2_000_010.0)   # same OB → no re-send
    assert len(ff.sent) == 1, ff.sent
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 222, 'top': 1.3, 'bottom': 1.2, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_020.0)   # NEW OB → new alert
    assert len(ff.sent) == 2, ff.sent
    print('✓ layer TG: спрацьовує на НОВИЙ Volumized OB (1m), не дублює той самий')


def test_layer_alert_needs_all_layers():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        layer_tg_on=True, funding_route_q4=False, layer_tg_min=5, layer_tg_cooldown_min=0,
        queue3_vob_open=False))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG'}}
    ff._funding_layers = lambda s, a: {'base': 4, 'base4': 4, 'count': 4, 'layers': []}  # 5 з 5 не всі
    ff.sent = []
    ff._send_layer_alert = lambda *a, **k: ff.sent.append(1)
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 1, 'bottom': 1, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    assert ff.sent == [], 'без усіх 5 шарів — новий VOB не дає сигналу'
    print('✓ layer TG: новий VOB дає сигнал лише ПІСЛЯ зходження всіх 5 шарів')


# --- 🎯 Черга-3: авто-відкриття за VOB + 5 шарів + SL з блоку OB ---
class _FakeTM:
    def __init__(self):
        self._positions = {}
        self._shadow_positions = {}
        self.sl_calls = []

    def update_manual_sl_tp(self, symbol, manual_sl=None, manual_tp=None, is_shadow=False):
        self.sl_calls.append((symbol, manual_sl, is_shadow))
        return {'ok': True}


def _wire_vob(ff, tm, direction='LONG'):
    """Спільна проводка: fake TM + справжній _tm_has_position + опен, що «створює»
    реальну позицію в fake TM."""
    ff._get_tm = lambda: tm
    ff._tm_has_position = lambda s, real: (s in (tm._positions if real else tm._shadow_positions))
    ff._fmt_price = lambda p: (f"{p:.4f}" if isinstance(p, (int, float)) else '—')

    def _open(sym, d, fuel, s, opened_by=None, skip_ctr_safeguard=False, skip_exhaustion=False):
        tm._positions[sym] = {'side': d}
        ff.opened.append((sym, d, opened_by))
        return True
    ff._open = _open


def test_vob_open_opens_and_sets_sl():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        queue3_vob_open=True, queue3_vob_sl_buffer_pct=0.10, layer_tg_on=False))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG', 'rate': -1.0, 'last_price': 100.0}}
    tm = _FakeTM(); _wire_vob(ff, tm)
    ff._funding_layers = lambda s, a: {'base': 5, 'base4': 4, 'count': 5, 'layers': []}
    # bullish OB: bottom=98, top=99 → LONG SL = 98 * (1 - 0.001) = 97.902
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 99.0, 'bottom': 98.0, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    assert ff.opened == [(sym, 'LONG', 'Q3-VOB(funding)')], ff.opened
    assert tm.sl_calls and abs(tm.sl_calls[-1][1] - 97.902) < 1e-6, tm.sl_calls
    assert tm.sl_calls[-1][2] is False                     # real book
    assert ff._vob_trade.get(sym, {}).get('side') == 'LONG'
    print('✓ VOB-open: 5 шарів + новий OB → відкрито LONG, SL під низом блоку + буфер')


def test_vob_open_short_sl_above_block():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        queue3_vob_open=True, queue3_vob_sl_buffer_pct=0.10, layer_tg_on=False))
    sym = 'WIFUSDT'
    ff._anomalies = {sym: {'dir': 'SHORT', 'rate': -1.0, 'last_price': 50.0}}
    tm = _FakeTM(); _wire_vob(ff, tm)
    ff._funding_layers = lambda s, a: {'base': 5, 'base4': 4, 'count': 5, 'layers': []}
    # bearish OB: top=51, bottom=50 → SHORT SL = 51 * (1 + 0.001) = 51.051
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 51.0, 'bottom': 50.0, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    assert ff.opened == [(sym, 'SHORT', 'Q3-VOB(funding)')], ff.opened
    assert abs(tm.sl_calls[-1][1] - 51.051) < 1e-6, tm.sl_calls
    print('✓ VOB-open: SHORT → SL над верхом блоку + буфер')


def test_vob_open_retrigger_moves_sl_no_reopen():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        queue3_vob_open=True, queue3_vob_sl_buffer_pct=0.10, layer_tg_on=False))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG', 'rate': -1.0, 'last_price': 100.0}}
    tm = _FakeTM(); _wire_vob(ff, tm)
    ff._funding_layers = lambda s, a: {'base': 5, 'base4': 4, 'count': 5, 'layers': []}
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 99.0, 'bottom': 98.0, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)     # 1) відкрили
    ff.opened = []
    # НОВИЙ OB (інший ft) по вже відкритій монеті → лише пересунути SL.
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 222, 'top': 99.5, 'bottom': 98.5, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_050.0)
    assert ff.opened == [], 'повторний VOB НЕ відкриває нову угоду'
    assert ff._vob_trade[sym]['ftime'] == 222
    assert abs(tm.sl_calls[-1][1] - (98.5 * 0.999)) < 1e-6, tm.sl_calls  # SL пересунуто
    print('✓ VOB-open: повторний OB → лише пересув SL, без нової угоди')


def test_vob_open_disabled():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        queue3_vob_open=False, layer_tg_on=False))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG', 'rate': -1.0, 'last_price': 100.0}}
    tm = _FakeTM(); _wire_vob(ff, tm)
    ff._funding_layers = lambda s, a: {'base': 5, 'base4': 4, 'count': 5, 'layers': []}
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 99.0, 'bottom': 98.0, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    assert ff.opened == [] and not tm.sl_calls, 'вимкнено → нічого не відкриваємо'
    print('✓ VOB-open: вимкнено (queue3_vob_open=False) → жодних дій')


def test_vob_open_needs_all_five_layers():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        queue3_vob_open=True, layer_tg_on=False))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG', 'rate': -1.0, 'last_price': 100.0}}
    tm = _FakeTM(); _wire_vob(ff, tm)
    ff._funding_layers = lambda s, a: {'base': 4, 'base4': 4, 'count': 4, 'layers': []}  # не всі 5
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 99.0, 'bottom': 98.0, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    assert ff.opened == [], 'без усіх 5 шарів VOB-угоду не відкриваємо'
    print('✓ VOB-open: нова угода лише коли всі 5 шарів зійшлись')


def test_vob_open_blocked_against_overall_trend():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        queue3_vob_open=True, layer_tg_on=False,
        queue3_vob_block_against_trend=True))
    sym = 'ENAUSDT'
    ff._anomalies = {sym: {'dir': 'LONG', 'rate': -1.0, 'last_price': 100.0}}
    tm = _FakeTM(); _wire_vob(ff, tm)
    ff._funding_layers = lambda s, a: {'base': 5, 'base4': 4, 'count': 5, 'layers': []}
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 111, 'top': 99.0, 'bottom': 98.0, 'breaker': False}
    # ЗАГАЛЬНИЙ тренд ВНИЗ → LONG проти нього → НЕ відкриваємо.
    ff._funding_price = {sym: {'dir': 'up', 'chg': 1.0, 'dir_overall': 'down', 'chg_overall': -5.0}}
    ff._layer_signal_alert(ff.get_settings(), 2_000_000.0)
    assert ff.opened == [], 'LONG проти загального тренду ↓ — не відкриваємо'
    # Загальний тренд ВГОРУ → LONG у бік → відкриваємо.
    ff._funding_price = {sym: {'dir': 'up', 'chg': 1.0, 'dir_overall': 'up', 'chg_overall': 5.0}}
    ff._funding_vob = lambda sym, d, tf=None: {'formation_time': 222, 'top': 99.0, 'bottom': 98.0, 'breaker': False}
    ff._layer_signal_alert(ff.get_settings(), 2_000_100.0)
    assert ff.opened == [(sym, 'LONG', 'Q3-VOB(funding)')], ff.opened
    print('✓ VOB-open: ворота «проти загального тренду» блокують контртренд, пропускають у бік')


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
def test_gold_funding_confirm_then_repeat():
    ff, db = _ff()
    db.set_setting('fuel_filter_settings', dict(
        db.get_setting('fuel_filter_settings'),
        funding_gold_tg=True, funding_gold_cooldown_min=60,
        funding_gold_confirm_sec=30,
        spike_tg=False, spike_auto_open=False, opportunity_auto_open=False))
    sym = 'DEXEUSDT'
    ff._anomalies = {sym: {'dir': 'SHORT', 'rate': -2.0}}
    ff._opportunity_for = lambda sym, a, s: (0, False, [])
    ff._gold_funding_step = lambda a, tol: 2.0
    ff.gold_sent = []
    ff._send_gold_alert = lambda sym, a, step: ff.gold_sent.append((sym, step))
    s = ff.get_settings()
    t0 = 1_000_000.0
    ff._opportunity_alert(s, t0)                    # зʼявився → _since=t0, ще НЕ підтверджено
    assert ff.gold_sent == [], 'до підтвердження (30с) не шлемо'
    ff._opportunity_alert(s, t0 + 30)              # +30с → підтверджено → перше
    assert len(ff.gold_sent) == 1, ff.gold_sent
    ff._opportunity_alert(s, t0 + 60)             # <60хв від першого → без повтору
    assert len(ff.gold_sent) == 1, ff.gold_sent
    ff._opportunity_alert(s, t0 + 30 + 61 * 60)   # +61хв → повтор
    assert len(ff.gold_sent) == 2, ff.gold_sent
    assert ff.gold_sent[0][1] == 2.0, 'у повідомленні — снепнутий чистий крок'
    print('✓ gold funding: затримка-підтвердження + повтор раз на кулдаун; чистий крок')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll «Готовність» strategy tests passed ✓ ({len(tests)} tests)')
