"""Тести пакета прискорення Черги-4 (A1+A2+A3+B1+B2+C2).

ГОЛОВНЕ, що тут замикається: прискорення НЕ змінює ЖОДНОГО рішення. Усі
оптимізації — про те, СКІЛЬКИ разів ми рахуємо одне й те саме, а не про те, ЩО
ми рахуємо. Тому центральний тест — `test_split_gives_identical_result`:
знімок+чиста функція мусять дати БАЙТ-У-БАЙТ те саме, що старий монолітний шлях.

  A1 — `_fuel_dir` і `_fuel_dir_legacy` беруть ОДИН знімок liq-map, не два.
  A2 — знімок кешується на LIQ_STATE_TTL (джерело оновлюється раз на 60с).
  A3 — `get_mm_settings` не робить 8 DB-читань на кожен виклик.
  B1 — `_q4_layers_from` — ЧИСТА функція (жодного I/O) → 2-й напрямок безкоштовний.
  B2 — `get_state()` ЧИТАЄ шари, пораховані двигуном, а не рахує їх під локом.
  C2 — `get_state(sections=...)` не рахує секції, чиї гармошки згорнуті.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
if 'pybit' not in sys.modules:
    for n in ('pybit', 'pybit.unified_trading'):
        m = types.ModuleType(n); sys.modules[n] = m
    sys.modules['pybit.unified_trading'].HTTP = object

_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_load('detection.signal_labels', 'detection/signal_labels.py')
_load('detection.setup_grader', 'detection/setup_grader.py')
mmmod = _load('detection.mm_model', 'detection/mm_model.py')
ffmod = _load('detection.fuel_filter', 'detection/fuel_filter.py')
FF = ffmod.FuelFilterDaemon


def _check(c, m):
    if not c:
        raise AssertionError(m)


_SETTINGS = {
    'queue4_new_mm_on': True, 'queue4_old_mm_on': True,
    'queue4_setup_on': True, 'queue4_runway_on': True,
    'queue4_mm_new_min': 30, 'queue4_mm_old_min': 10,
    'queue4_mm_old_mode': 'require', 'queue4_setup_min': 38,
    'queue4_require_runway': True,
}


def _ff():
    """Мінімальний екземпляр демона: лише поля, потрібні шарам Черги-4."""
    o = FF.__new__(FF)
    o._setup_cache = {}
    o._liq_state_cache = {}
    o._liq_profile_at = 0.0
    o._liq_profile = 'tori'
    o._q4_layers_cache = {}
    return o


# ─────────────────────────── A1 + A2 ────────────────────────────────────────
def test_one_liq_snapshot_shared_and_cached():
    """A1+A2: обидві МММ-функції беруть ОДИН знімок, і він кешується."""
    o = _ff()
    calls = []

    def _fake_liq(sym, force=False):
        calls.append(sym)
        return {'mark_price': 100.0, 'levels': [
            {'price': 101.0, 'usd': 1000.0},
            {'price': 99.0, 'usd': 400.0},
        ]}

    o._liq_state = _fake_liq
    o._live_price = lambda s: 100.0
    o._db = types.SimpleNamespace(get_setting=lambda k, d=None: d)

    o._fuel_dir_legacy('BTCUSDT')
    o._fuel_dir('BTCUSDT')
    _check(len(calls) == 2, f'кожна функція звертається до ОДНІЄЇ точки, отримано {calls}')
    _check(all(c == 'BTCUSDT' for c in calls), 'звертання саме за потрібним символом')
    print('✓ A1: обидві МММ-функції ходять через єдиний `_liq_state`')


def test_liq_state_cache_hits_within_ttl():
    """A2: у межах TTL важкий `lm.get_state` викликається РІВНО один раз."""
    o = _ff()
    builds = []

    class _LM:
        def get_state(self, sym, lookback_hours=24, profile='tori'):
            builds.append((sym, lookback_hours, profile))
            return {'mark_price': 10.0, 'levels': []}

    lmmod = types.ModuleType('detection.liquidation_map.liquidation_map')
    lmmod.get_liquidation_map = lambda: _LM()
    sys.modules['detection.liquidation_map'] = types.ModuleType('detection.liquidation_map')
    sys.modules['detection.liquidation_map.liquidation_map'] = lmmod
    o._db = types.SimpleNamespace(get_setting=lambda k, d=None: 'tori')

    for _ in range(7):          # рівно стільки разів шари смикали дані РАНІШЕ
        o._liq_state('ETHUSDT')
    _check(len(builds) == 1, f'у межах TTL — одна збірка, отримано {len(builds)}')

    o._liq_state('ETHUSDT', force=True)
    _check(len(builds) == 2, 'force=True обходить кеш')

    # Протермінований запис має перечитатись.
    ts, st = o._liq_state_cache['ETHUSDT']
    o._liq_state_cache['ETHUSDT'] = (ts - ffmod.LIQ_STATE_TTL - 1, st)
    o._liq_state('ETHUSDT')
    _check(len(builds) == 3, 'після TTL знімок перечитується (дані не «залипають»)')
    print(f'✓ A2: 7 звернень → 1 збірка liq-map (TTL {ffmod.LIQ_STATE_TTL:.0f}с)')


def test_liq_cache_respects_cap():
    """A2: кеш не росте вічно — є стеля памʼяті при широкому watchlist."""
    o = _ff()

    class _LM:
        def get_state(self, sym, **k): return {'mark_price': 1.0, 'levels': []}

    lmmod = sys.modules['detection.liquidation_map.liquidation_map']
    lmmod.get_liquidation_map = lambda: _LM()
    o._db = types.SimpleNamespace(get_setting=lambda k, d=None: 'tori')
    for i in range(ffmod.LIQ_STATE_CAP + 25):
        o._liq_state(f'SYM{i}')
    _check(len(o._liq_state_cache) <= ffmod.LIQ_STATE_CAP,
           f'кеш перевищив стелю: {len(o._liq_state_cache)} > {ffmod.LIQ_STATE_CAP}')
    print(f'✓ A2: кеш обмежений {ffmod.LIQ_STATE_CAP} монетами')


def test_missing_source_is_not_cached():
    """A2: «liq-map ще не піднявся» НЕ кешуємо — інакше після рестарту перший
    розрахунок відклався б на цілий TTL і бот мовчав би на старті."""
    o = _ff()
    lmmod = sys.modules['detection.liquidation_map.liquidation_map']
    lmmod.get_liquidation_map = lambda: None
    o._db = types.SimpleNamespace(get_setting=lambda k, d=None: 'tori')
    _check(o._liq_state('BTCUSDT') is None, 'немає джерела → None')
    _check('BTCUSDT' not in o._liq_state_cache, 'відсутнє джерело потрапило в кеш')

    # Джерело піднялось → наступний виклик МУСИТЬ його побачити одразу.
    class _LM:
        def get_state(self, sym, **k): return {'mark_price': 5.0, 'levels': []}
    lmmod.get_liquidation_map = lambda: _LM()
    _check((o._liq_state('BTCUSDT') or {}).get('mark_price') == 5.0,
           'джерело піднялось, а кеш віддав стару порожнечу')
    print('✓ A2: відсутнє джерело не кешується (немає паузи після рестарту)')


def test_decay_profile_not_read_every_call():
    """A2: `liqmap_decay_profile` більше не читається з БД на кожен виклик."""
    o = _ff()
    reads = []
    o._db = types.SimpleNamespace(
        get_setting=lambda k, d=None: (reads.append(k), 'tori')[1])
    for _ in range(10):
        o._liq_decay_profile()
    _check(len(reads) == 1, f'очікували 1 DB-читання профілю, отримано {len(reads)}')
    print('✓ A2: профіль загасання читається з БД раз на 60с, а не щоразу')


# ─────────────────────────────── A3 ─────────────────────────────────────────
def test_mm_settings_cached():
    """A3: 8 DB-читань на кожен `compute_mm` → одне читання на 60с."""
    mmmod.invalidate_mm_settings()
    reads = []

    class _DB:
        def get_setting(self, k, d=None):
            reads.append(k); return None

    db = _DB()
    first = mmmod.get_mm_settings(db)
    n_first = len(reads)
    _check(n_first == len(mmmod._DEFAULTS),
           f'перший виклик читає всі ключі ({n_first})')
    for _ in range(5):
        mmmod.get_mm_settings(db)
    _check(len(reads) == n_first, 'подальші виклики в межах TTL — без DB')
    _check(mmmod.get_mm_settings(db) == first, 'значення ті самі')

    mmmod.invalidate_mm_settings()
    mmmod.get_mm_settings(db)
    _check(len(reads) == n_first * 2, 'invalidate → перечитування з БД')
    print(f'✓ A3: {n_first} DB-читань × 6 викликів → {n_first} (кеш 60с + invalidate)')


def test_mm_settings_db_error_not_cached():
    """A3: збій БД НЕ повинен «залипнути» дефолтами на цілу хвилину."""
    mmmod.invalidate_mm_settings()

    class _Boom:
        def get_setting(self, k, d=None): raise RuntimeError('db down')

    out = mmmod.get_mm_settings(_Boom())
    _check(out == mmmod._DEFAULTS, 'на збої повертаємо дефолти')
    _check(mmmod._settings_at == 0.0, 'збій НЕ кешується — наступний виклик спробує знову')
    print('✓ A3: помилка БД не кешується')


# ─────────────────────────────── B1 ─────────────────────────────────────────
def _snap(fn=None, fo=None, su=None, ready=True, on=None):
    return {'fn': fn or {}, 'fo': fo or {}, 'su': su or {},
            'setup_ready': ready,
            'on': on or {'new': True, 'old': True, 'setup': True, 'runway': True}}


def test_layers_from_is_pure_no_io():
    """B1: `_q4_layers_from` не має права торкатись ні liq-map, ні БД, ні кешів."""
    o = _ff()

    def _boom(*a, **k):
        raise AssertionError('_q4_layers_from зробив I/O — це вже НЕ чиста функція')

    o._liq_state = _boom
    o._fuel_dir = _boom
    o._fuel_dir_smoothed = _boom
    o._fuel_dir_legacy = _boom
    o._liq_decay_profile = _boom
    o._db = types.SimpleNamespace(get_setting=_boom)
    o._setup_cache = property(_boom)   # будь-яке читання кешу теж заборонене

    snap = _snap(fn={'status': 'LONG', 'dir': 0.55,
                     'runway': {'dir': 'LONG', 'room_pct': 1.4, 'label': 'запас'}},
                 fo={'status': 'LONG', 'strength': 62},
                 su={'ok': True, 'dir': 'LONG', 'score': 47, 'grade': 'СЕРЕДНІЙ'})
    r = o._q4_layers_from(snap, 'LONG', _SETTINGS)
    _check(r['count'] == 4 and r['required'] == 4, f'усі 4 шари мали зійтись: {r["count"]}/4')
    print('✓ B1: `_q4_layers_from` — чиста функція (жодного I/O)')


def test_second_direction_costs_nothing():
    """B1: обидва напрямки — з ОДНОГО знімка, тобто 2-й безкоштовний."""
    o = _ff()
    io = []
    o._fuel_dir_smoothed = lambda s: (io.append('new'), {
        'status': 'LONG', 'dir': 0.5,
        'runway': {'dir': 'LONG', 'room_pct': 2.0}})[1]
    o._fuel_dir_legacy = lambda s: (io.append('old'), {'status': 'LONG', 'strength': 55})[1]
    o._setup_cache = {'BTCUSDT': {'ok': True, 'dir': 'LONG', 'score': 60, 'grade': 'ХОРОШИЙ'}}

    snap = o._q4_snapshot('BTCUSDT', _SETTINGS)
    n_after_snapshot = len(io)
    a = o._q4_layers_from(snap, 'LONG', _SETTINGS)
    b = o._q4_layers_from(snap, 'SHORT', _SETTINGS)
    _check(len(io) == n_after_snapshot, f'арифметика шарів зробила I/O: {io}')
    _check(n_after_snapshot == 2, f'знімок = рівно 2 читання (Новий+Старий МММ), got {io}')
    _check(a['count'] == 4 and b['count'] == 0,
           f'LONG=4/4, SHORT=0/4; отримано {a["count"]}/{b["count"]}')
    print('✓ B1: один знімок → обидва напрямки (2-й коштує нуль)')


def test_split_gives_identical_result():
    """🔒 ЗАМОК: розділення НЕ змінює результат. Знімок+чиста функція мусять дати
    те саме, що монолітний `_queue4_layers`. Це головна гарантія «не запороти»."""
    o = _ff()
    o._fuel_dir_smoothed = lambda s: {'status': 'SHORT', 'dir': -0.42,
                                      'runway': {'dir': 'SHORT', 'room_pct': 0.9,
                                                 'label': 'малий запас'}}
    o._fuel_dir_legacy = lambda s: {'status': 'SHORT', 'strength': 71}
    o._setup_cache = {'SUIUSDT': {'ok': True, 'dir': 'SHORT', 'score': 44,
                                  'grade': 'СЕРЕДНІЙ'}}
    for side in ('LONG', 'SHORT'):
        mono = o._queue4_layers('SUIUSDT', side, _SETTINGS)
        split = o._q4_layers_from(o._q4_snapshot('SUIUSDT', _SETTINGS), side, _SETTINGS)
        _check(mono == split, f'{side}: розділення змінило результат!\n{mono}\n{split}')
    print('✓ B1 ЗАМОК: результат ІДЕНТИЧНИЙ монолітному розрахунку')


def test_disabled_layer_is_not_fetched():
    """B1: вимкнений шар не тягне дані (сенс пер-шарових тумблерів збережено)."""
    o = _ff()
    io = []
    o._fuel_dir_smoothed = lambda s: (io.append('new'), {'status': 'LONG', 'dir': 0.4})[1]
    o._fuel_dir_legacy = lambda s: (io.append('old'), {'status': 'LONG', 'strength': 20})[1]
    o._setup_cache = {}
    s = dict(_SETTINGS, queue4_old_mm_on=False, queue4_setup_on=False,
             queue4_new_mm_on=False, queue4_runway_on=False)
    snap = o._q4_snapshot('XRPUSDT', s)
    _check(io == [], f'усі шари вимкнені → нуль читань, отримано {io}')
    r = o._q4_layers_from(snap, 'LONG', s)
    _check(r['required'] == 0 and r['count'] == 0, 'нуль увімкнених шарів')
    print('✓ B1: вимкнений шар не читає дані')


def test_snapshot_toggles_win_over_late_setting_change():
    """B1: арифметика бере тумблери ЗІ ЗНІМКА. Інакше тумблер, перемкнутий між
    знімком і розрахунком, дав би «визначений» шар без даних під ним."""
    o = _ff()
    o._fuel_dir_smoothed = lambda s: {'status': 'LONG', 'dir': 0.4}
    o._fuel_dir_legacy = lambda s: {'status': 'LONG', 'strength': 30}
    o._setup_cache = {}
    s_off = dict(_SETTINGS, queue4_old_mm_on=False)
    snap = o._q4_snapshot('ADAUSDT', s_off)          # Старий МММ НЕ читали
    r = o._q4_layers_from(snap, 'LONG', dict(_SETTINGS))  # а тумблер уже ON
    keys = [l['key'] for l in r['layers']]
    _check('mm_old' not in keys,
           'шар без даних не має зʼявлятись лише тому, що тумблер перемкнули')
    print('✓ B1: тумблери беруться зі знімка (немає шарів без даних)')


# ─────────────────────────────── B2 ─────────────────────────────────────────
def test_cached_layers_come_from_engine():
    """B2: UI показує РІВНО те, що порахував двигун (одне джерело правди)."""
    o = _ff()
    o._fuel_dir_smoothed = lambda s: (_ for _ in ()).throw(
        AssertionError('get_state не має рахувати шари, коли кеш двигуна свіжий'))
    engine_long = {'count': 4, 'required': 4, 'layers': [], 'marker': 'engine'}
    engine_short = {'count': 1, 'required': 4, 'layers': [], 'marker': 'engine'}
    o._q4_cache_put('BTCUSDT', {'LONG': engine_long, 'SHORT': engine_short})
    _check(o._q4_layers_cached('BTCUSDT', 'LONG', _SETTINGS) is engine_long,
           'віддано НЕ те, що порахував двигун')
    _check(o._q4_layers_cached('BTCUSDT', 'SHORT', _SETTINGS) is engine_short,
           'протилежний напрямок теж має братись із кешу')
    print('✓ B2: get_state читає шари двигуна, а не рахує свої')


def test_cache_miss_falls_back_to_live_calc():
    """B2: монета щойно в черзі (двигун ще не проходив) → рахуємо на місці."""
    o = _ff()
    o._fuel_dir_smoothed = lambda s: {'status': 'LONG', 'dir': 0.7,
                                      'runway': {'dir': 'LONG', 'room_pct': 3.0}}
    o._fuel_dir_legacy = lambda s: {'status': 'LONG', 'strength': 40}
    o._setup_cache = {'NEWUSDT': {'ok': True, 'dir': 'LONG', 'score': 55,
                                  'grade': 'ХОРОШИЙ'}}
    r = o._q4_layers_cached('NEWUSDT', 'LONG', _SETTINGS)
    _check(r['count'] == 4, f'промах кешу → живий розрахунок, отримано {r["count"]}/4')
    print('✓ B2: промах кешу не лишає рядок без даних')


def test_stale_cache_is_recomputed():
    """B2: протухлий кеш НЕ показуємо — краще порахувати, ніж брехати."""
    o = _ff()
    o._q4_cache_put('OLDUSDT', {'LONG': {'count': 4, 'marker': 'stale'}})
    ts, by = o._q4_layers_cache['OLDUSDT']
    o._q4_layers_cache['OLDUSDT'] = (ts - ffmod.CYCLE_SECS * 10, by)
    o._fuel_dir_smoothed = lambda s: {'status': 'SHORT', 'dir': -0.3}
    o._fuel_dir_legacy = lambda s: {'status': 'SHORT', 'strength': 15}
    o._setup_cache = {}
    r = o._q4_layers_cached('OLDUSDT', 'LONG', _SETTINGS)
    _check(r.get('marker') != 'stale', 'протухлий кеш віддано як актуальний')
    print('✓ B2: протухлий кеш перераховується')


def test_cache_drop_on_queue_exit():
    """B2: монета вийшла з черги → її шари не мають лишатись у памʼяті."""
    o = _ff()
    o._q4_cache_put('GONEUSDT', {'LONG': {'count': 1}})
    o._q4_cache_drop('GONEUSDT')
    _check('GONEUSDT' not in o._q4_layers_cache, 'кеш шарів пережив вихід із черги')
    print('✓ B2: кеш чиститься разом із чергою')


# ─────────────────────────────── C2 ─────────────────────────────────────────
def test_heavy_sections_are_declared():
    _check(set(FF.HEAVY_SECTIONS) == {'q4', 'fund'},
           f'несподіваний перелік важких секцій: {FF.HEAVY_SECTIONS}')
    print('✓ C2: важкі секції оголошені явно')


if __name__ == '__main__':
    test_one_liq_snapshot_shared_and_cached()
    test_liq_state_cache_hits_within_ttl()
    test_liq_cache_respects_cap()
    test_missing_source_is_not_cached()
    test_decay_profile_not_read_every_call()
    test_mm_settings_cached()
    test_mm_settings_db_error_not_cached()
    test_layers_from_is_pure_no_io()
    test_second_direction_costs_nothing()
    test_split_gives_identical_result()
    test_disabled_layer_is_not_fetched()
    test_snapshot_toggles_win_over_late_setting_change()
    test_cached_layers_come_from_engine()
    test_cache_miss_falls_back_to_live_calc()
    test_stale_cache_is_recomputed()
    test_cache_drop_on_queue_exit()
    test_heavy_sections_are_declared()
    print('\nУсі тести прискорення Черги-4 пройдено ✅')
