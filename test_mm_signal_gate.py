"""🧮 СИГНАЛИ ЧЕРЕЗ МММ-NEW (вимога 01.10; до того — через МММ-монітор, 29.09).

01.10 дослівно: «Кожен сигнал має перевірятись у таблиці "Список монет" банера
"МММ-NEW" за відповідним напрямком. Якщо сигнал LONG і є ця монета в таблиці в
закладці LONG — пропускаємо далі по алгоритму. Інакше ігноруємо з відповідним
записом в лог бота. І додай ще тумблер (за замовчуванням вимкнений), пропускати
сигнали лише за напрямком банера "МММ-NEW".»

Історія (29.09):

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
def test_passes_when_the_coin_sits_in_the_signal_tab():
    """Базове правило 01.10: дивимось ЛИШЕ вкладку монети; банер не важить."""
    app, ok, chip, why = sc.mm_gate_decide(_v(), 'LONG')
    _check(app and ok and '✓' in chip and not why, (app, ok, chip, why))
    for d in (None, 'SHORT'):
        _check(sc.mm_gate_decide(_v(d=d, coin='LONG'), 'LONG')[1],
               f'банер {d} без тумблера не ріже')
    for view, word in ((_v(coin='FLAT'), 'Рівновага'),
                       (_v(coin='SHORT'), 'вкладці'),
                       (_v(coin=None), 'немає у «Списку монет»')):
        app, ok, chip, why = sc.mm_gate_decide(view, 'LONG')
        _check(app and not ok and word in why and '✗' in chip, (view, why))
    print('✓ пропуск лише коли монета у вкладці того ж напрямку МММ-new')


def test_banner_toggle_adds_the_banner_requirement():
    for view, word in ((_v(d=None), 'без напрямку'),
                       (_v(d='SHORT', coin='LONG'), 'банер SHORT')):
        app, ok, chip, why = sc.mm_gate_decide(view, 'LONG', True)
        _check(app and not ok and word in why, (view, why))
    app, ok, chip, _ = sc.mm_gate_decide(_v(), 'LONG', True)
    _check(ok and 'банер LONG' in chip, chip)
    _check(not sc.mm_gate_decide(_v(coin='SHORT'), 'LONG', True)[1],
           'банер збігся, але вкладка чужа — відмова')
    print('✓ тумблер «за банером МММ-NEW» додає вимогу напрямку банера')


def test_short_mirrors_long():
    _check(sc.mm_gate_decide(_v(d='SHORT', coin='SHORT'), 'SHORT', True)[1], 'SHORT ok')
    _check(not sc.mm_gate_decide(_v(d='SHORT', coin='LONG'), 'SHORT')[1], 'чужа вкладка')
    print('✓ SHORT дзеркальний')


def test_monitor_off_or_missing_does_not_apply():
    for view in ({}, None, _v(on=False, d=None, coin=None)):
        app, ok, chip, why = sc.mm_gate_decide(view, 'LONG')
        _check(not app and ok and not chip, (view, app, ok, chip))
    print('✓ блок МММ-new вимкнено / FF недоступний — умова не діє')


# ── 2. читання монітора ─────────────────────────────────────────────────────
def _ff(bias=None, snap=None, on=True):
    f = FF.__new__(FF)
    f._lock = threading.RLock()
    f._mm_bias_new = bias or {}
    f._mmn_snapshot = snap or {}
    f._mmn_on = on
    # МММ LiQ свідомо подаємо ПРОТИЛЕЖНИМ — ворота мусять читати саме МММ-new.
    f._mm_bias = {'dir': 'SHORT'}
    f._mm_snapshot = {k: {'status': 'SHORT'} for k in (snap or {})}
    f._mm_mon_on = True
    return f


def test_view_reads_the_same_snapshot_that_sorts_the_tabs():
    f = _ff({'dir': 'LONG'}, {'AAAUSDT': {'status': 'LONG', 'strength': 40},
                              'BBBUSDT': {'status': None, 'strength': 5}})
    _check(f.mm_gate_view('aaausdt') == {'on': True, 'dir': 'LONG', 'coin': 'LONG'},
           f.mm_gate_view('aaausdt'))
    _check(f.mm_gate_view('BBBUSDT')['coin'] == 'FLAT', 'без напрямку = ⚖')
    _check(f.mm_gate_view('CCCUSDT')['coin'] is None, 'немає рядка = None')
    _check(_ff({'dir': None}).mm_gate_view('X')['dir'] is None, 'банер ⚖')
    _check(_ff(on=False).mm_gate_view('X')['on'] is False, 'МММ-new вимкнено')
    print('✓ mm_gate_view читає готовий знімок МММ-new (не МММ LiQ)')


def test_capture_records_the_monitor_switch():
    src = ast.get_source_segment(open(ffm.__file__, encoding='utf-8').read(),
                                 next(n for n in ast.walk(ast.parse(open(ffm.__file__, encoding='utf-8').read()))
                                      if isinstance(n, ast.FunctionDef) and n.name == '_mm_capture'))
    _check('self._mmn_on = _new' in src, 'такт пише стан тумблера МММ-new')
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
        _check(ok is False and 'вкладці' in why and '🆕МММ-new' in detail,
               (ok, why, detail))
        sc._mm_gate_view = lambda sym: _v(d='SHORT', coin='LONG')
        ns = _ns(); ns._settings['mm_gate_banner'] = True
        ok, why, _ = S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        _check(ok is False and 'банер SHORT' in why, why)
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
    _check(sc.DEFAULT_SETTINGS.get('mm_gate_banner') is False, 'банер — дефолт ВИМК')
    _check("'mm_gate_enabled', 'mm_gate_banner'," in _SC_SRC, 'у білому списку')
    _check('id="sm-mm-gate-banner"' in _HTML and 'mm_gate_banner: banner' in _HTML
           and '!!s.mm_gate_banner' in _HTML, 'тумблер банера в UI')
    _check('🧮 Через МММ-NEW' in _HTML and '🧮 Через МММ-монітор' not in _HTML,
           'підпис перейменовано')
    _check('id="sm-mm-gate"' in _HTML and 'mm_gate_enabled: enabled' in _HTML,
           'чекбокс у рядку ALERTS шле налаштування')
    _check("s.mm_gate_enabled !== false" in _HTML, 'стан підтягується при завантаженні')
    print('✓ тумблер: дефолт УВІМК, білий список, UI')


# ── 3б. 💧 ОКРЕМА ПЕРЕВІРКА СКАНЕРОМ ЛІКВІДНОСТІ (вимога 01.10) ─────────────
def test_liq_gate_rule():
    _check(sc.liq_gate_decide(False, None, 'LONG')[:2] == (False, True),
           'вимкнений Сканер — умова не діє')
    app, ok, chip, why = sc.liq_gate_decide(True, None, 'LONG')
    _check(app and not ok and 'немає' in why, why)
    app, ok, chip, why = sc.liq_gate_decide(True, {'side': 'SHORT'}, 'LONG')
    _check(app and not ok and 'SHORT' in why, why)
    app, ok, chip, why = sc.liq_gate_decide(True, {'side': 'LONG', 'mass_pct': 70.1}, 'LONG')
    _check(app and ok and '✓' in chip and '70.1' in chip, chip)
    print('✓ 💧 правило: монета в таблиці Сканера під бік сигналу')


def test_liq_gate_is_separate_and_off_by_default():
    _check(sc.DEFAULT_SETTINGS.get('liq_gate_enabled') is False, 'дефолт ВИМК')
    _check("'liq_gate_enabled'," in _SC_SRC, 'у білому списку')
    old_r, old_m = sc._liq_gate_row, sc._mm_gate_view
    calls = []
    try:
        sc._mm_gate_view = lambda sym: {}            # МММ-NEW не діє
        sc._liq_gate_row = lambda sym: calls.append(sym) or (True, None)
        ns = _ns(); ns._settings['liq_gate_enabled'] = True
        ok, why, detail = S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        _check(ok is False and '💧 Сканер' in why and '💧Сканер' in detail, why)
        calls.clear(); ns._settings['liq_gate_enabled'] = False
        try:
            S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        except Exception:
            pass
        _check(not calls, 'вимкнений тумблер Сканер не питає')
    finally:
        sc._liq_gate_row, sc._mm_gate_view = old_r, old_m
    body = _SC_SRC.split('def _signal_allowed')[1]
    _check(body.index('mm_gate_decide(') < body.index('liq_gate_decide(')
           < body.index("self._settings.get('ob_filter_enabled'"), 'порядок')
    _check('id="sm-liq-gate"' in _HTML and 'liq_gate_enabled:' in _HTML
           and '!!s.liq_gate_enabled' in _HTML, 'UI-тумблер')
    print('✓ 💧 «Через Сканер ліквідності» — окремий тумблер, деф. ВИМК')


# ── 4. 🗑 корекцію ВИДАЛЕНО (вимога 01.10) — у воротах її більше немає ─────
def test_correction_gate_is_gone():
    """«Банер "Корекція" і весь алгоритм дій з ним — коректно видалити».
    Сигнали корекцією більше не відхиляються: у спільних воротах немає ні
    `_corr_gate`, ні сегмента 🔻 у розкладі."""
    _check(not hasattr(sc, '_corr_gate'), '_corr_gate лишився в сканері')
    body = _SC_SRC.split('def _signal_allowed')[1].split('\n    def ')[0]
    _check('_corr_gate' not in body and '🔻Корекція' not in body,
           'ворота досі питають корекцію')
    _check('correction_blocks_open' not in _SC_SRC, 'сканер досі читає вердикт корекції')
    print('✓ 🔻 корекції у спільних воротах більше немає')


if __name__ == '__main__':
    _fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
