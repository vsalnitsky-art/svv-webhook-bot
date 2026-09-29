"""🧮 СИГНАЛИ ЧЕРЕЗ МММ-МОНІТОР (вимога 29.09).

Дослівно: «Давай ось такі сигнали [рядок ALERTS: CHoCH / CHoCH+BOS, Volumized
OB…] буде проганяти через МММ-монітор, якщо він увімкнений. Тобто сигнал
пропускаємо, якщо він відповідає напрямку МММ-монітор і є у відповідній
МММ-монітор таблиці за напрямком.»

Реалізовано у СПІЛЬНИХ воротах `SMCScanner._signal_allowed`, тож умова діє на
ВСІ сигнали сканера разом, а не на кожен шлях окремо.
"""
import ast
import importlib.util
import os
import sys
import threading
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, 'detection', fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sc = _load('smc_scanner_mmgate_test', 'smc_scanner.py')
ffm = _load('fuel_filter_mmgate_test', 'fuel_filter.py')
S = sc.SMCScanner
FF = ffm.FuelFilterDaemon
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


def _v(on=True, d='LONG', coin='LONG'):
    return {'on': on, 'dir': d, 'coin': coin}


# ── 1. чисте правило ───────────────────────────────────────────────────────
def test_passes_only_when_banner_and_coin_tab_match_the_signal():
    app, ok, chip, why = sc.mm_gate_decide(_v(), 'LONG')
    _check(app and ok and '✓' in chip and not why, (app, ok, chip, why))
    for view, side, word in ((_v(d=None), 'LONG', 'без напрямку'),
                             (_v(d='SHORT', coin='SHORT'), 'LONG', 'банер SHORT'),
                             (_v(coin='FLAT'), 'LONG', 'Рівновага'),
                             (_v(coin='SHORT'), 'LONG', 'вкладці'),
                             (_v(coin=None), 'LONG', 'немає в таблиці')):
        app, ok, chip, why = sc.mm_gate_decide(view, side)
        _check(app and not ok and word in why and '✗' in chip, (view, why))
    print('✓ пропуск лише: банер = бік сигналу і монета у вкладці того ж боку')


def test_short_mirrors_long():
    _check(sc.mm_gate_decide(_v(d='SHORT', coin='SHORT'), 'SHORT')[1], 'SHORT ok')
    _check(not sc.mm_gate_decide(_v(d='SHORT', coin='LONG'), 'SHORT')[1], 'чужа вкладка')
    print('✓ SHORT дзеркальний')


def test_monitor_off_or_missing_does_not_apply():
    for view in ({}, None, _v(on=False, d=None, coin=None)):
        app, ok, chip, why = sc.mm_gate_decide(view, 'LONG')
        _check(not app and ok and not chip, (view, app, ok, chip))
    print('✓ монітор вимкнено / FF недоступний — умова не діє')


# ── 2. читання монітора ─────────────────────────────────────────────────────
def _ff(bias=None, snap=None, on=True):
    f = FF.__new__(FF)
    f._lock = threading.RLock()
    f._mm_bias = bias or {}
    f._mm_snapshot = snap or {}
    f._mm_mon_on = on
    return f


def test_view_reads_the_same_snapshot_that_sorts_the_tabs():
    f = _ff({'dir': 'LONG'}, {'AAAUSDT': {'status': 'LONG', 'strength': 40},
                              'BBBUSDT': {'status': None, 'strength': 5}})
    _check(f.mm_gate_view('aaausdt') == {'on': True, 'dir': 'LONG', 'coin': 'LONG'},
           f.mm_gate_view('aaausdt'))
    _check(f.mm_gate_view('BBBUSDT')['coin'] == 'FLAT', 'без напрямку = ⚖')
    _check(f.mm_gate_view('CCCUSDT')['coin'] is None, 'немає рядка = None')
    _check(_ff({'dir': None}).mm_gate_view('X')['dir'] is None, 'банер ⚖')
    _check(_ff(on=False).mm_gate_view('X')['on'] is False, 'монітор вимкнено')
    print('✓ mm_gate_view читає готовий знімок монітора')


def test_capture_records_the_monitor_switch():
    src = ast.get_source_segment(open(ffm.__file__, encoding='utf-8').read(),
                                 next(n for n in ast.walk(ast.parse(open(ffm.__file__, encoding='utf-8').read()))
                                      if isinstance(n, ast.FunctionDef) and n.name == '_mm_capture'))
    _check('self._mm_mon_on = _mon' in src, 'такт пише стан тумблера')
    print('✓ такт двигуна запамʼятовує тумблер монітора (без походу в БД у воротах)')


# ── 3. спільні ворота ──────────────────────────────────────────────────────
def _ns(gate=True):
    ns = types.SimpleNamespace()
    ns._settings = {'mm_gate_enabled': gate}
    return ns


def test_gate_blocks_signal_in_shared_gate():
    old = sc._mm_gate_view
    try:
        sc._mm_gate_view = lambda sym: _v(d='SHORT', coin='SHORT')
        ok, why, detail = S._signal_allowed(_ns(), 'AAAUSDT', 'LONG')
        _check(ok is False and 'банер SHORT' in why and '🧮МММ' in detail,
               (ok, why, detail))
        sc._mm_gate_view = lambda sym: _v(coin='FLAT')
        ok, why, _ = S._signal_allowed(_ns(), 'AAAUSDT', 'LONG')
        _check(ok is False and 'Рівновага' in why, why)
    finally:
        sc._mm_gate_view = old
    print('✓ спільні ворота відхиляють сигнал проти монітора')


def test_toggle_off_skips_the_gate():
    old = sc._mm_gate_view
    called = []
    try:
        sc._mm_gate_view = lambda sym: called.append(sym) or _v(d='SHORT')
        try:
            S._signal_allowed(_ns(False), 'AAAUSDT', 'LONG')
        except Exception:
            pass          # решта фільтрів у голому стабі — не предмет тесту
        _check(not called, 'вимкнений тумблер не питає монітор узагалі')
    finally:
        sc._mm_gate_view = old
    print('✓ тумблер «🧮 Через МММ-монітор» вимикає умову')


def test_gate_sits_right_after_direction_buttons_before_other_filters():
    body = _SC_SRC.split('def _signal_allowed')[1]
    i_dg = body.index('_dg.allows(')
    i_mm = body.index('mm_gate_decide(')
    i_ob = body.index("self._settings.get('ob_filter_enabled'")
    _check(i_dg < i_mm < i_ob, 'порядок: 🚦 кнопки → 🧮 МММ → решта фільтрів')
    print('✓ 🧮 МММ стоїть одразу за головними кнопками')


def test_setting_default_whitelist_and_ui():
    _check(sc.DEFAULT_SETTINGS.get('mm_gate_enabled') is True, 'дефолт УВІМК')
    _check("'mm_gate_enabled'," in _SC_SRC, 'у білому списку')
    _check('id="sm-mm-gate"' in _HTML and 'mm_gate_enabled: enabled' in _HTML,
           'чекбокс у рядку ALERTS шле налаштування')
    _check("s.mm_gate_enabled !== false" in _HTML, 'стан підтягується при завантаженні')
    print('✓ тумблер: дефолт УВІМК, білий список, UI')


# ── 4. корекція (вимога 29.09): сигнал ігнорується повністю ────────────────
def test_correction_rejects_signal_in_shared_gate():
    old_v, old_c = sc._mm_gate_view, sc._corr_gate
    try:
        sc._mm_gate_view = lambda sym: {}
        sc._corr_gate = lambda: (True, '🔻 КОРЕКЦІЯ проти банера LONG')
        ok, why, detail = S._signal_allowed(_ns(), 'AAAUSDT', 'LONG')
        _check(ok is False and 'проігноровано' in why and '🔻Корекція:✗' in detail,
               (ok, why, detail))
        called = []
        sc._corr_gate = lambda: called.append(1) or (False, '')
        try:
            S._signal_allowed(_ns(), 'AAAUSDT', 'LONG')
        except Exception:
            pass          # решта фільтрів у голому стабі — не предмет тесту
        _check(called, 'поза корекцією ворота питаються і пропускають далі')
    finally:
        sc._mm_gate_view, sc._corr_gate = old_v, old_c
    print('✓ підтверджена корекція відкидає сигнал у спільних воротах')


def test_correction_gate_reads_the_single_verdict():
    class _F:
        def __init__(self, r): self.r = r
        def correction_blocks_open(self): return self.r
    import types as _t
    fake = _t.ModuleType('detection.fuel_filter')
    old = sys.modules.get('detection.fuel_filter')
    try:
        fake.get_fuel_filter = lambda: _F((True, 'X'))
        sys.modules['detection.fuel_filter'] = fake
        _check(sc._corr_gate() == (True, 'X'), 'бере вердикт correction_blocks_open')
        fake.get_fuel_filter = lambda: _F((False, ''))   # ⏸ пауза / немає корекції
        _check(sc._corr_gate() == (False, ''), 'пауза знімає ворота')
        fake.get_fuel_filter = lambda: None
        _check(sc._corr_gate() == (False, ''), 'немає FF → fail-open')
    finally:
        if old is not None:
            sys.modules['detection.fuel_filter'] = old
        else:
            sys.modules.pop('detection.fuel_filter', None)
    print('✓ ворота корекції — той самий вердикт, що в _open/on_signal')


def test_correction_gate_sits_after_mm_before_other_filters():
    body = _SC_SRC.split('def _signal_allowed')[1]
    i_mm = body.index('mm_gate_decide(')
    i_c = body.index('_corr_gate()')
    i_ob = body.index("self._settings.get('ob_filter_enabled'")
    _check(i_mm < i_c < i_ob, 'порядок: 🧮 МММ → 🔻 корекція → решта')
    print('✓ 🔻 корекція стоїть одразу за 🧮 МММ')


if __name__ == '__main__':
    _fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
