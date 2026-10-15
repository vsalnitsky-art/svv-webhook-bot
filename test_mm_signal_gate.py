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
    # ⚑ Пороги «Сила ≥» вкладок (вимога 12.10) — кеш, який пише такт двигуна.
    f._mmn_str_mins_c = {'all': 0, 'LONG': 0, 'SHORT': 0, 'flat': 0}
    f._mm_str_mins_c = {'all': 0, 'LONG': 0, 'SHORT': 0, 'flat': 0}
    return f


def test_view_reads_the_same_snapshot_that_sorts_the_tabs():
    f = _ff({'dir': 'LONG'}, {'AAAUSDT': {'status': 'LONG', 'strength': 40},
                              'BBBUSDT': {'status': None, 'strength': 5}})
    _v1 = f.mm_gate_view('aaausdt')
    _check({k: _v1[k] for k in ('on', 'dir', 'coin')}
           == {'on': True, 'dir': 'LONG', 'coin': 'LONG'}, _v1)
    _check(_v1['strength'] == 40 and _v1['str_min'] == 0, _v1)
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
        sc._mm_gate_view = lambda sym, book='new': _v(d='SHORT', coin='SHORT')
        ok, why, detail = S._signal_allowed(_ns(), 'AAAUSDT', 'LONG')
        _check(ok is False and 'вкладці' in why and '🆕МММ-new' in detail,
               (ok, why, detail))
        sc._mm_gate_view = lambda sym, book='new': _v(d='SHORT', coin='LONG')
        ns = _ns(); ns._settings['mm_gate_banner'] = True
        ok, why, _ = S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        _check(ok is False and 'банер SHORT' in why, why)
        sc._mm_gate_view = lambda sym, book='new': _v(coin='FLAT')
        ok, why, _ = S._signal_allowed(_ns(), 'AAAUSDT', 'LONG')
        _check(ok is False and 'Рівновага' in why, why)
    finally:
        sc._mm_gate_view = old
    print('✓ спільні ворота відхиляють сигнал проти монітора')


def test_toggle_off_skips_the_gate():
    old = sc._mm_gate_view
    called = []
    try:
        sc._mm_gate_view = lambda sym, book='new': called.append(sym) or _v(d='SHORT')
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
    """09.10: «загальні ворота … для всіх сигналів (за замовченням вимкнені)»."""
    for k in ('mm_gate_enabled', 'mm_gate_banner', 'mmo_gate_enabled', 'mmo_gate_banner'):
        _check(sc.DEFAULT_SETTINGS.get(k) is False, f'{k}: дефолт ВИМК')
    _check("'mmo_gate_enabled', 'mmo_gate_banner'," in _SC_SRC
           and "'mm_gate_enabled', 'mm_gate_banner'," in _SC_SRC, 'у білому списку')
    for cid in ('sm-mm-gate', 'sm-mm-gate-banner', 'sm-mmo-gate', 'sm-mmo-gate-banner'):
        _check(f'id="{cid}"' in _HTML, cid)
    _check('Через МММ-NEW' in _HTML and 'Через МММ-МОНІТОР' in _HTML
           and 'за банером МММ-МОНІТОР' in _HTML, 'підписи')
    _check("['sm-mmo-gate', 'sm-mmo-gate-banner', 'mmo_gate_enabled', 'mmo_gate_banner']" in _HTML,
           'updateMmGate шле обидві пари')
    _check('id="sm-mm-gate" checked' not in _HTML, 'чекбокс не увімкнений за замовчуванням')
    print('✓ чотири тумблери: дефолт ВИМК, білий список, UI')


# ── 3а. 🧮 ВОРОТА «ЧЕРЕЗ МММ-МОНІТОР» (вимога 09.10) ────────────────────────
def test_view_old_book_reads_the_monitor_snapshot():
    f = _ff({'dir': 'LONG'}, {'AAAUSDT': {'status': 'LONG'}})
    v = f.mm_gate_view('AAAUSDT', book='old')
    _check({k: v[k] for k in ('on', 'dir', 'coin')} == {'on': True, 'dir': 'SHORT', 'coin': 'SHORT'}, v)
    f._mm_mon_on = False
    _check(f.mm_gate_view('AAAUSDT', book='old')['on'] is False,
           'вимкнений монітор → ворота не діють')
    print('✓ mm_gate_view(book=old) читає знімок і банер 🧮 МММ-монітора')


def test_old_book_gate_uses_its_own_labels():
    app, ok, chip, why = sc.mm_gate_decide(_v(coin='SHORT'), 'LONG', False, 'old')
    _check(app and not ok and '🧮МММ-монітор' in chip and 'МММ-МОНІТОР' in why, (chip, why))
    app, ok, chip, _ = sc.mm_gate_decide(_v(d='SHORT'), 'LONG', True, 'old')
    _check(not ok and 'банер SHORT' in chip, chip)
    print('✓ ворота МММ-монітора — те саме правило, свої підписи')


def test_both_gates_apply_independently():
    seen = []
    old = sc._mm_gate_view
    try:
        def fake(sym, book='new'):
            seen.append(book)
            return _v(coin='LONG') if book == 'new' else _v(coin='SHORT')
        sc._mm_gate_view = fake
        ns = _ns(False); ns._settings['mmo_gate_enabled'] = True
        ok, why, detail = S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        _check(ok is False and 'МММ-МОНІТОР' in why and seen == ['old'], (why, seen))
        seen.clear(); ns._settings['mm_gate_enabled'] = True
        ok, why, detail = S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        _check(ok is False and seen == ['new', 'old'] and '🆕МММ-new[вкладка LONG]:✓' in detail,
               (seen, detail))
    finally:
        sc._mm_gate_view = old
    print('✓ увімкнені обидві пари — сигнал мусить пройти обидві')


def test_gates_apply_to_every_origin_including_new_ob():
    """«для всіх сигналів»: окремого винятку для 🆕 Новий OB більше немає."""
    _check(sc.mm_gates_active({'mm_gate_enabled': True}) == [('new', False)], 'new')
    _check(sc.mm_gates_active({'mmo_gate_enabled': True, 'mmo_gate_banner': True})
           == [('old', True)], 'old')
    _check(sc.mm_gates_active({}) == [], 'усі вимкнені за замовчуванням')
    _check('ob_alert_mm_gate' not in _SC_SRC and 'mm_gate_applies' not in _SC_SRC,
           'виняток для Нового OB прибрано')
    _check('sm-ob-alert-mm' not in _HTML, 'чекбокс у групі Нового OB прибрано')
    print('✓ ворота діють на всі сигнали, включно з 🆕 Новим OB')


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
        sc._mm_gate_view = lambda sym, book='new': {}            # МММ-NEW не діє
        sc._liq_gate_row = lambda sym, book='new': calls.append(sym) or (True, None)
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


# ── ⚑ ПОРІГ «Сила ≥» ВКЛАДКИ У ВОРОТАХ (вимога 12.10) ─────────────────────
def test_strength_gate_needs_the_coin_visible_in_its_tab():
    v = dict(_v(), strength=12, str_min=20)
    app, ok, chip, why = sc.mm_gate_decide(v, 'LONG', need_strength=True)
    _check(app and not ok and '12%' in why and '20%' in why, (ok, chip, why))
    v['strength'] = 25
    app, ok, chip, _ = sc.mm_gate_decide(v, 'LONG', need_strength=True)
    _check(ok and '25%≥20%' in chip, chip)
    print('✓ ⚑ сила нижче порогу вкладки → сигнал відхилено; вище → пропущено')


def test_strength_gate_threshold_zero_or_toggle_off_does_nothing():
    v = dict(_v(), strength=5, str_min=0)
    _check(sc.mm_gate_decide(v, 'LONG', need_strength=True)[1], 'поріг 0 = фільтр не задіяно')
    v['str_min'] = 50
    _check(sc.mm_gate_decide(v, 'LONG')[1], 'тумблер вимкнено → поріг не діє')
    print('✓ ⚑ поріг 0 або вимкнений тумблер — умова не діє')


def test_view_carries_the_threshold_of_the_coin_tab_per_book():
    f = _ff({'dir': 'LONG'}, {'AAAUSDT': {'status': 'LONG', 'strength': 30},
                              'BBBUSDT': {'status': None, 'strength': 7}})
    f._mmn_str_mins_c = {'all': 5, 'LONG': 40, 'SHORT': 60, 'flat': 3}
    f._mm_str_mins_c = {'all': 0, 'LONG': 0, 'SHORT': 25, 'flat': 0}
    _check(f.mm_gate_view('AAAUSDT')['str_min'] == 40, 'вкладка LONG МММ-new')
    _check(f.mm_gate_view('BBBUSDT')['str_min'] == 3, 'вкладка ⚖')
    _check(f.mm_gate_view('AAAUSDT', 'old')['str_min'] == 25, '🧮: SHORT-вкладка свого блоку')
    print('✓ ⚑ view несе поріг САМЕ вкладки монети і САМЕ свого блоку')


def test_strength_toggle_reaches_the_shared_gate_and_ui():
    old = sc._mm_gate_view
    try:
        sc._mm_gate_view = lambda sym, book='new': dict(_v(), strength=10, str_min=30)
        ns = _ns(); ns._settings['mm_gate_strength'] = True
        ok, why, detail = S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        _check(ok is False and 'поріг вкладки' in why, (ok, why))
    finally:
        sc._mm_gate_view = old
    _check(sc.SMCScanner.__init__ and "'mm_gate_strength': False" in _SC_SRC, 'дефолт ВИМК')
    for need in ('id="sm-mm-gate-str"', "['sm-mm-gate-str', 'mm_gate_strength']",
                 'body.mm_gate_strength', 'class="mm-dir-thr"', '⚑≥'):
        _check(need in _HTML, f'на сторінці немає «{need}»')
    print('✓ ⚑ тумблер доходить до воріт; UI має чекбокс і позначку вкладки')


# ── 3в. ⚡ ВОРОТА «ЧЕРЕЗ МММ-CTR» (вимога 15.10) ─────────────────────────────
def test_view_ctr_book_reads_the_ctr_snapshot():
    f = _ff({'dir': 'LONG'}, {'AAAUSDT': {'status': 'LONG'}})
    f._mm_bias_ctr = {'dir': 'SHORT'}
    f._mmc_snapshot = {'AAAUSDT': {'status': None, 'strength': 4}}
    f._mmc_on = True
    f._mmc_str_mins_c = {'all': 0, 'LONG': 0, 'SHORT': 0, 'flat': 7}
    v = f.mm_gate_view('AAAUSDT', book='ctr')
    _check({k: v[k] for k in ('on', 'dir', 'coin', 'str_min')}
           == {'on': True, 'dir': 'SHORT', 'coin': 'FLAT', 'str_min': 7}, v)
    f._mmc_on = False
    _check(f.mm_gate_view('AAAUSDT', book='ctr')['on'] is False, 'вимкнений CTR → не діє')
    print('✓ mm_gate_view(book=ctr) читає знімок, банер і пороги ⚡ МММ-CTR')


def test_ctr_gate_rule_labels_and_activation():
    app, ok, chip, why = sc.mm_gate_decide(_v(coin='SHORT'), 'LONG', False, 'ctr')
    _check(app and not ok and '⚡МММ-CTR' in chip and '⚡ МММ-CTR' in why, (chip, why))
    _check(sc.mm_gates_active({'mmc_gate_enabled': True, 'mmc_gate_banner': True})
           == [('ctr', True)], 'ctr у переліку воріт')
    seen = []
    old = sc._mm_gate_view
    try:
        def fake(sym, book='new'):
            seen.append(book)
            return _v(coin='SHORT')
        sc._mm_gate_view = fake
        ns = _ns(False); ns._settings['mmc_gate_enabled'] = True
        ok, why, detail = S._signal_allowed(ns, 'AAAUSDT', 'LONG')
        _check(ok is False and seen == ['ctr'] and '⚡ МММ-CTR' in why
               and '⚡МММ-CTR[вкладка SHORT]:✗' in detail, (seen, why, detail))
    finally:
        sc._mm_gate_view = old
    for k in ('mmc_gate_enabled', 'mmc_gate_banner'):
        _check(f"'{k}': False" in _SC_SRC, f'дефолт ВИМК: {k}')
    _check("'mmc_gate_enabled', 'mmc_gate_banner'," in _SC_SRC, 'білий список')
    for need in ('id="sm-mmc-gate"', 'id="sm-mmc-gate-banner"',
                 "['sm-mmc-gate', 'sm-mmc-gate-banner', 'mmc_gate_enabled', 'mmc_gate_banner']",
                 "'⚡ МММ-CTR']].forEach"):
        _check(need in _HTML, f'на сторінці немає «{need}»')
    print('✓ «Через МММ-CTR» — те саме правило, свої підписи, деф. ВИМК, UI')


if __name__ == '__main__':
    _fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
