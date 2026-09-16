"""🧮 МММ-МОНІТОР: живий напрямок МММ по монетах + групове ✋ відкриття.

**Вимога користувача (15.09), дослівно:** «Я хочу відслідковувати в реальному
часі, який на даний момент по монетах МММ стан. І сортувати їх окремо на LONG
чи SHORT… Додай до кожної монети можливість в таблиці її помітити для групового
відкриття угод.»

**Чому ОКРЕМА секція, а не «Черга на вхід» (розбір варіанта користувача).**
`_pending` (Черга-1) наповнюється ВИКЛЮЧНО з `intercept`, тобто монетами, які
ДАЛИ СИГНАЛ і пройшли ворота. Це черга подій, а не список спостереження: усього
watchlist там не буде НІКОЛИ (на скріні — 0 рядків). Плюс рядок черги живе під
двигуном (TTL, виселення, авто-відкриття), тож галочка на ньому означала б
вибір на записі, який може зникнути або відкритись сам.

Що стережуть ці тести:
  • монітор ЧИТАЄ знімок двигуна і нічого не рахує сам (одне джерело правди);
  • групове відкриття бере напрямок КОЖНОЇ монети з ТОГО САМОГО знімка;
  • монета в угоді не потрапляє у вибір, «рівновага» не відкривається;
  • часткова невдача звітується ПО КОЖНІЙ монеті, а не «все або нічого»;
  • режим економії (`mmm_limited_mode`) НАЗВАНИЙ вголос, а не мовчить;
  • черги і ворота входу лишились недоторканими.
"""
import ast
import importlib.util
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

# Порожній пакет `detection` з правильним `__path__`: справжній `__init__.py`
# тягне `sleeper_scanner → core → pybit`, тобто півпроєкту.
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg
for _n in ('pybit', 'pybit.unified_trading'):
    if _n not in sys.modules:
        sys.modules[_n] = types.ModuleType(_n)
sys.modules['pybit.unified_trading'].HTTP = object

_spec = importlib.util.spec_from_file_location(
    'fuel_filter_mm_test', os.path.join(_HERE, 'detection', 'fuel_filter.py'))
_m = importlib.util.module_from_spec(_spec)
sys.modules['fuel_filter_mm_test'] = _m
_spec.loader.exec_module(_m)
FF = _m.FuelFilterDaemon

_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'),
            encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
             encoding='utf-8').read()


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


_LOGGED = []


def _install_log():
    _LOGGED.clear()
    mod = types.ModuleType('detection.activity_log')

    def _log(symbol, event, detail='', side=None, source='', extra=None):
        _LOGGED.append({'symbol': symbol, 'event': event, 'detail': detail,
                        'side': side, 'source': source})
    mod.log_activity = _log
    sys.modules['detection.activity_log'] = mod
    _pkg.activity_log = mod


def _mk(limited=False, enabled=True, mon=True):
    """Мінімальний демон: лише те, чого торкається монітор.

    ⚠️ `_fuel_dir_legacy` підмінено стабом: монітор показує САМЕ СТАРИЙ МММ
    (вимога 15.09), і тести мають ганяти той самий шлях, що прод."""
    import threading
    ff = FF.__new__(FF)
    ff._lock = threading.RLock()
    ff._mm_snapshot = {}
    ff._mm_str_hist = {}
    ff._mm_vob_seen = {}
    ff._mm_vob_at = {}
    ff._mm_price_hist = {}
    ff._mm_decision = {}
    ff._clock = [10_000.0]
    ff._mm_grow_since = {}
    ff._mm_snapshot_ts = 0.0
    ff._fuel_managed = {}
    ff._pending, ff._pending2, ff._pending3, ff._pending4 = {}, {}, {}, {}
    ff._timers = {}
    ff._get_tm = None
    ff._settings = {'enabled': enabled, 'mmm_limited_mode': limited,
                    'mm_monitor_enabled': mon}
    ff.get_settings = lambda: dict(ff._settings)
    # Старий МММ: той самий поріг ±0.1 і та сама сила |dir|×100, що в коді.
    ff._legacy = {}
    ff._legacy_calls = []

    def _leg(sym):
        ff._legacy_calls.append(sym)
        d = ff._legacy.get(str(sym).upper())
        if d is None:
            return None
        st = 'LONG' if d > 0.1 else ('SHORT' if d < -0.1 else None)
        return {'dir': round(d, 3), 'status': st,
                'strength': int(round(abs(d) * 100))}
    ff._fuel_dir_legacy = _leg
    ff._tm_has_position = lambda s, shadow: False
    ff._q4_set_vob_sl = lambda *a, **k: None
    ff.opened = []

    def _open(sym, side, fuel, s, **kw):
        ff.opened.append({'symbol': sym, 'side': side, 'kw': kw,
                          'mark': (fuel or {}).get('mark_price')})
        return True
    ff._open = _open
    return ff


def _fuels(**pairs):
    """{'BTCUSDT': 0.42} → форма, яку віддає `_fuel_dir_smoothed` (НОВИЙ МММ)."""
    out = {}
    for sym, d in pairs.items():
        st = 'LONG' if d > 0.15 else ('SHORT' if d < -0.15 else None)
        out[sym] = {'dir': d, 'status': st, 'mark_price': 100.0 + abs(d)}
    return out


def _cap(ff, **pairs):
    """Один такт двигуна: `pairs` — це значення СТАРОГО МММ.

    ⚠️ Новий МММ навмисно подаємо ПРОТИЛЕЖНИМ (`-v`) — тож КОЖЕН тест заразом
    доводить, що монітор бере саме СТАРИЙ показник, а не той, що лежить поруч
    у `fuels`.

    ⚠️ Кожен виклик СУНЕ ВІРТУАЛЬНИЙ ГОДИННИК на `CYCLE_SECS` — приріст сили
    тепер міряється за ВІКНОМ ЧАСУ, а не «попереднім тактом», тож без руху
    годинника тести перевіряли б неіснуючу поведінку (усі такти в одну мить)."""
    ff._legacy = {str(k).upper(): v for k, v in pairs.items()}
    ff._mm_capture(_fuels(**{k: -v for k, v in pairs.items()}), now=ff._clock[0])
    ff._clock[0] += _m.CYCLE_SECS


def _caps(ff, n=3, **pairs):
    """`n` тактів поспіль із тими самими значеннями — щоб набралось вікно
    (`MM_GROW_MIN_SPAN_SEC`) і приріст став ВИМІРЯНИМ, а не вигаданим."""
    for _ in range(n):
        _cap(ff, **pairs)


# ═══════════ 1. ДЖЕРЕЛО ЧИСЕЛ — ЗНІМОК ДВИГУНА ═══════════════════════════
def test_monitor_reads_the_engine_snapshot_and_computes_nothing():
    """⚠️ ГОЛОВНЕ. Монітор не має права рахувати МММ сам: інакше в таблиці
    стояло б одне число, а рішення двигун ухвалював би за іншим (той самий
    урок, що з шарами Черги-4 — «двигун рахує, get_state читає»)."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.42, ETHUSDT=-0.31, XRPUSDT=0.02)
    st = ff.mm_monitor_state()
    by = {r['symbol']: r for r in st['rows']}
    _check(by['BTCUSDT']['mm'] == 'LONG', by['BTCUSDT'])
    _check(by['ETHUSDT']['mm'] == 'SHORT', by['ETHUSDT'])
    _check(by['XRPUSDT']['mm'] is None, 'слабкий тиск = рівновага, не напрямок')
    # Сила — ТА САМА конвенція, що в банері ₿: |fuel_dir| × 100.
    _check(by['BTCUSDT']['strength'] == 42, by['BTCUSDT'])
    _check(by['ETHUSDT']['strength'] == 31, by['ETHUSDT'])
    # У читача НЕ має бути жодного походу в liq-map / EMA.
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == 'mm_monitor_state')
    body = ast.dump(fn)
    for bad in ('_fuel_dir', '_liq_state', '_fuel_dir_smoothed', 'get_liquidation_map'):
        _check(bad not in body, f'монітор рахує сам через «{bad}» — це регресія')
    print('✓ монітор читає готовий знімок двигуна, сам нічого не рахує')


def test_strength_trend_comes_from_the_window_baseline():
    """Стрілку ↑/↓ малює спільний віджет `ffFuelCell`, тож віддаємо БАЗОВУ
    силу, а не власний висновок «up/down» — друге правило тренду розійшлося б
    із першим. База — найстаріше значення у вікні, а не попередній такт."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.20)
    _cap(ff, BTCUSDT=0.45)
    _cap(ff, BTCUSDT=0.45)
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['strength'] == 45 and r['strength_prev'] == 20, r)
    _check('trend' not in r, 'готового «up/down» у рядку бути не має')
    print('✓ тренд сили: віддаємо базове число, стрілку малює спільний віджет')


def test_rows_are_sorted_by_strength_then_symbol():
    """Найвиразніший напрямок зверху; тайбрейк за символом — щоб рядки не
    «стрибали» під курсором між поллами."""
    ff = _mk()
    _cap(ff, AAAUSDT=0.30, BBBUSDT=-0.90, CCCUSDT=0.30)
    got = [r['symbol'] for r in ff.mm_monitor_state()['rows']]
    _check(got == ['BBBUSDT', 'AAAUSDT', 'CCCUSDT'], got)
    print('✓ сортування: сила ↓, далі символ (стабільний порядок)')


def test_backend_no_longer_ships_tab_counts():
    """⚠️ Лічильники вкладок рахує ФРОНТ — разом із фільтром «Сила ≥», який
    теж живе там. Серверне число поруч із відфільтрованою таблицею означало б,
    що «(36)» на вкладці і кількість рядків під нею — різні речі (скарга
    15.09). Два джерела одного числа не тримаємо."""
    ff = _mk()
    _cap(ff, A=0.5, B=0.4, C=-0.6, D=0.01)
    st = ff.mm_monitor_state()
    _check('counts' not in st, f'сервер усе ще шле власні лічильники: {st.keys()}')
    _check(len(st['rows']) == 4, st['rows'])
    print('✓ лічильників на бекенді немає — їх рахує фронт із фільтром')


# ═══════════ 2. ПРИДАТНІСТЬ ДО ВИБОРУ ════════════════════════════════════
def test_a_coin_already_in_a_trade_is_not_shown_at_all():
    """Вимога 15.09: «монета, яка в угоді, не потрібно відображати у таблиці».
    ⚠️ Але зі ЗНІМКА вона НЕ зникає — знімок живить колонку «🧮 Старий МММ»
    у таблицях угод."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.5, ETHUSDT=0.5)
    ff._fuel_managed = {'BTCUSDT': {}}
    by = {r['symbol']: r for r in ff.mm_monitor_state()['rows']}
    _check('BTCUSDT' not in by, f'монета в угоді лишилась у таблиці: {list(by)}')
    _check(by['ETHUSDT']['selectable'], by['ETHUSDT'])
    _check(ff.mm_snapshot_for(['BTCUSDT']).get('BTCUSDT'),
           'колонка в таблиці угод втратила число — знімок чіпати не можна')
    print('✓ монета в угоді: у таблиці немає, у знімку для угод — є')


def test_balanced_mm_is_not_selectable():
    """⚖ рівновага — напрямку немає, відкривати нічого."""
    ff = _mk()
    _cap(ff, XRPUSDT=0.02)
    r = ff.mm_monitor_state()['rows'][0]
    _check(not r['selectable'], r)
    print('✓ ⚖ рівновага не обирається (немає чого відкривати)')


# ═══════════ 3. ГРУПОВЕ ВІДКРИТТЯ ════════════════════════════════════════
def test_group_open_uses_each_coin_own_mm_direction():
    """Напрямок КОЖНОЇ монети — її власний МММ зі знімка, тобто РІВНО те, що
    людина бачила в рядку."""
    _install_log()
    ff = _mk()
    _cap(ff, BTCUSDT=0.5, ETHUSDT=-0.5)
    res = ff.group_open(['BTCUSDT', 'ETHUSDT'])
    _check(res['opened'] == 2 and res['failed'] == 0, res)
    got = {o['symbol']: o['side'] for o in ff.opened}
    _check(got == {'BTCUSDT': 'LONG', 'ETHUSDT': 'SHORT'}, got)
    print('✓ групове відкриття: напрямок кожної монети = її МММ')


def test_group_open_goes_through_the_same_manual_path_as_queue4():
    """Мітка ✋ Ручний + пропуск воріт — як у ✋ відкритті з Черги-4, щоб
    походження угоди не губилось і поведінка не розходилась."""
    _install_log()
    ff = _mk()
    _cap(ff, BTCUSDT=0.5)
    ff.group_open(['BTCUSDT'])
    kw = ff.opened[0]['kw']
    _check(kw.get('by_hand') is True, 'рішення людини → by_hand')
    _check(kw.get('opened_by') == 'manual → MMM', kw.get('opened_by'))
    _check(kw.get('skip_safeguard') and kw.get('skip_ctr_safeguard'), kw)
    print(f'✓ шлях ✋ ручного відкриття, мітка {kw.get("opened_by")!r}')


def test_partial_failure_is_reported_per_coin():
    """Часткова невдача НЕ має виглядати як загальний успіх (і навпаки)."""
    _install_log()
    ff = _mk()
    _cap(ff, BTCUSDT=0.5, ETHUSDT=0.02, LTCUSDT=0.4)
    ff._fuel_managed = {'LTCUSDT': {}}
    res = ff.group_open(['BTCUSDT', 'ETHUSDT', 'LTCUSDT'])
    _check(res['opened'] == 1 and res['failed'] == 2, res)
    by = {r['symbol']: r for r in res['results']}
    _check(by['BTCUSDT']['ok'], by['BTCUSDT'])
    _check(not by['ETHUSDT']['ok'] and 'рівновага' in by['ETHUSDT']['reason'],
           by['ETHUSDT'])
    _check(not by['LTCUSDT']['ok'] and 'угоді' in by['LTCUSDT']['reason'],
           by['LTCUSDT'])
    print('✓ результат по КОЖНІЙ монеті окремо (1 відкрито, 2 — з причинами)')


def test_rejected_open_is_not_counted_as_opened():
    """`_open` відхилив (кнопка напрямку / ціна / розмір) → це НЕ відкриття,
    і в лозі стоїть чесна причина."""
    _install_log()
    ff = _mk()
    _cap(ff, BTCUSDT=0.5)
    ff._open = lambda *a, **k: False
    res = ff.group_open(['BTCUSDT'])
    _check(res['opened'] == 0 and res['failed'] == 1, res)
    _check(not ff._timers, 'таймер не мав стартувати')
    _check(any(e['event'] == 'skipped' and 'МММ-монітор' in e['detail']
               for e in _LOGGED), _LOGGED)
    print('✓ відмова `_open` не рахується відкриттям')


def test_duplicates_and_empty_input_are_safe():
    _install_log()
    ff = _mk()
    _cap(ff, BTCUSDT=0.5)
    res = ff.group_open(['BTCUSDT', 'btcusdt', ' BTCUSDT '])
    _check(len(ff.opened) == 1, f'той самий символ двічі: {ff.opened}')
    _check(res['opened'] == 1, res)
    for bad in ([], None, ['   ']):
        r = ff.group_open(bad)
        _check(not r['ok'] and r['opened'] == 0, f'{bad!r} → {r}')
    print('✓ дублікати згортаються, порожній вибір нічого не відкриває')


def test_group_open_respects_the_master_switch():
    ff = _mk(enabled=False)
    _cap(ff, BTCUSDT=0.5)
    r = ff.group_open(['BTCUSDT'])
    _check(not r['ok'] and not ff.opened, r)
    print('✓ Fuel Auto-Filter вимкнено → групове відкриття не працює')


def test_direction_buttons_are_not_bypassed():
    """⚠️ 🚦 Головні кнопки напрямку ✋ ручне НЕ обходить (задокументоване
    рішення користувача). Тут це забезпечує САМ `_open` — тож перевіряємо, що
    груповий шлях не має власного обходу воріт напрямку."""
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == '_mm_open_one')
    body = ast.dump(fn)
    for bad in ('allow_long_entries', 'allow_short_entries', 'bypass_gates',
                'direction_gate', '_entry_gates'):
        _check(bad not in body,
               f'груповий шлях чіпає ворота напрямку («{bad}») — так не можна')
    _check('_open' in body, 'відкриття мусить іти через спільний `_open`')
    print('✓ ворота напрямку лишаються за `_open` — власного обходу немає')


# ═══════════ 4. РЕЖИМ ЕКОНОМІЇ НАЗВАНИЙ ВГОЛОС ═══════════════════════════
def test_limited_mode_is_said_out_loud():
    """⚠️ При `mmm_limited_mode` МММ рахується лише по монетах «у роботі».
    Мовчазно коротка таблиця читалась би як «бот сліпий» — той самий принцип,
    що з мовчазним clamp-ом у скані ліквідності."""
    _check(_mk(limited=True).mm_monitor_state()['limited'] is True, 'обмежений')
    _check(_mk(limited=False).mm_monitor_state()['limited'] is False, 'повний')
    _check('mm-limited-hint' in _HTML, 'на сторінці немає попередження')
    _check('Режим економії' in _HTML, 'попередження мусить пояснювати ПРИЧИНУ')
    print('✓ режим економії: прапорець у відповіді + пояснення на сторінці')


def test_empty_snapshot_is_not_an_error():
    st = _mk().mm_monitor_state()
    _check(st['rows'] == [] and st['ts'] == 0.0, st)
    _check(st['enabled'] is True and 'limited' in st, st)
    print('✓ порожній знімок (бот щойно піднявся) — коректний стан')


# ═══════════ 5. ЧЕРГИ І ВОРОТА — НЕДОТОРКАНІ ═════════════════════════════
def test_the_monitor_is_not_a_queue():
    """Монітор НЕ пише в жодну чергу: рядок не «чекає», двигун його не
    відкриває, TTL не виселяє. Саме тому це окрема секція, а не Черга-1."""
    for name in ('mm_monitor_state', '_mm_capture', '_mm_open_one'):
        fn = next(n for n in ast.walk(ast.parse(_SRC))
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        for node in ast.walk(fn):
            # Запис у чергу = присвоєння в `self._pendingN[...]`.
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Subscript) and \
                            str(getattr(t.value, 'attr', '')).startswith('_pending'):
                        raise AssertionError(f'{name} пише в чергу — це не монітор')
    print('✓ монітор у черги не пише (це спостереження, а не чергa)')


def test_queue1_intercept_is_untouched():
    """Черга-1 і далі наповнюється ВИКЛЮЧНО з `intercept` — ми її не чіпали."""
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == 'intercept')
    _check('_mm_snapshot' not in ast.dump(fn), 'intercept знає про монітор')
    _check("self._pending[sym] = {'dir': side" in _SRC,
           'наповнення Черги-1 змінилось — перевірити')
    print('✓ Черга-1 (intercept) лишилась як була')


# ═══════════ 6. UI ═══════════════════════════════════════════════════════
def test_ui_section_exists_with_selection_and_group_open():
    for need in ('id="mm-body"', 'id="mm-tbody"', 'id="mm-check-all"',
                 'id="mm-open-btn"', 'mmOpenSelected()', 'mmToggleAll('):
        _check(need in _HTML, f'у розмітці немає «{need}»')
    _check("'mm'" in _HTML.split('const _PANEL_IDS')[1].split(']')[0],
           'секція не зареєстрована в _PANEL_IDS (стан гармошки не памʼятається)')
    print('✓ UI: секція, галочки, «Відкрити обрані»')


def test_selection_survives_the_poll():
    """⚠️ Таблиця перемальовується кожні ~10с. Якби галочки трималися лише в
    DOM, їх стирало б просто під час вибору — групове відкриття стало б
    неможливим. Тому вибір живе в `_mmSel` ПОЗА розміткою."""
    _check('let _mmSel = new Set()' in _HTML, 'вибір не винесений із DOM')
    # ⚠️ Зріз — ДО КІНЦЯ функції, а не «перші N символів»: `mmRender` росте, і
    # фіксоване вікно вже врізало рядок із відновленням галочок (той самий
    # капкан, що вже ловили на `<th>` і на віджеті МММ).
    i = _HTML.index('function mmRender()')
    fn = _HTML[i:_HTML.index('function mmApplyState(', i)]
    _check('_mmSel.has(r.symbol)' in fn, 'рендер не відновлює галочки з набору')
    _check("tb.dataset.sig" in fn, 'немає сигнатури — DOM перебудовується даремно')
    print('✓ вибір переживає полл (живе поза DOM) + сигнатура таблиці')


def test_ui_reads_served_flag_before_drawing():
    """Без прапорця `served.mm` порожня секція (її просто не просили, бо
    гармошка згорнута) стерла б таблицю — той самий урок, що з Чергою-4."""
    _check('_served.mm && d.mm_monitor' in _HTML, 'served.mm не перевіряється')
    i = _HTML.index('function _ffWantedSections()')
    _check("want.push('mm')" in _HTML[i:i + 800], 'секція не запитується')
    print('✓ фронт просить секцію лише коли розгорнуто і читає served.mm')


def test_ui_uses_the_shared_mm_widget():
    """Та сама метрика у двох місцях — ОДИН вигляд: монітор малює МММ тим самим
    `ffFuelCell`, що й колонка «МММ» черг."""
    # Вікно беремо ДО кінця функції, а не «перші N символів»: `mmRender` росте,
    # і фіксований зріз почав би врізати рядок із викликом віджета.
    i = _HTML.index('function mmRender()')
    fn = _HTML[i:_HTML.index('function mmApplyState(', i)]
    _check('ffFuelCell(r.mm, r.strength, r.strength_prev)' in fn,
           'монітор малює МММ власним віджетом — вигляд розійдеться')
    print('✓ МММ малює спільний віджет ffFuelCell')


def test_engine_label_is_mirrored_everywhere():
    """🏷 Мітка двигуна — ЄДИНЕ джерело + ОБИДВА обовʼязкові JS-дзеркала."""
    labels = importlib.import_module('detection.signal_labels')
    _check(labels.ENGINE_BADGES.get('MMM') == '🧮 МММ-монітор',
           labels.ENGINE_BADGES.get('MMM'))
    _check(labels.pretty_opened_by('manual → MMM') == '✋ Ручний → 🧮 МММ-монітор',
           labels.pretty_opened_by('manual → MMM'))
    _check("'MMM': '🧮 МММ-монітор'" in _HTML and "'MMM': '🧮'" in _HTML,
           'дзеркало у smart_money.html не синхронне')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('"MMM": "🧮 МММ-монітор"' in js and '"MMM": "🧮"' in js,
           'дзеркало в infosite/app.js не синхронне')
    print('✓ мітка 🧮 МММ-монітор: бекенд + обидва JS-дзеркала')


def _run_js(body):
    """Виконати РЕАЛЬНИЙ JS монітора зі сторінки на мінімальному фейк-DOM.

    Замки на рядках коду ловлять «прибрали механізм», але не ловлять «механізм
    є, а поводиться не так». Тут беремо ТОЙ САМИЙ блок зі `smart_money.html` і
    ганяємо його node-ом."""
    import subprocess, tempfile
    i = _HTML.index('// 🧮 МММ-МОНІТОР — жива таблиця')
    j = _HTML.index('// ⚡ C2: просимо в сервера ЛИШЕ ті важкі секції')
    src = _HTML[i:j]
    pre = r'''
const _els = {};
function _el(id) {
  if (!_els[id]) _els[id] = {id, dataset:{}, style:{}, innerHTML:'', textContent:'',
                             checked:false, disabled:false, value:'0',
                             querySelector:()=>({textContent:''})};
  return _els[id];
}
const _tabSpans = {};
const _tabs = ['all','LONG','SHORT','flat'].map(d => ({
  dataset:{mmdir:d}, style:{},
  // ⚠️ Стабільний span: якби фейк віддавав НОВИЙ обʼєкт на кожен виклик,
  // прочитати назад те, що записав рендер, було б неможливо.
  querySelector:()=>(_tabSpans[d] = _tabSpans[d] || {textContent:''})}));
// Заголовки таблиці — ОКРЕМІ стаби з `getAttribute`: `_mmHeaderArrows` шукає
// саме `th[data-mmsort]`, і якби фейк повертав на будь-який селектор вкладки,
// тест падав би «на рівному місці» (так і сталось).
const _arrows = {};
const _ths = ['symbol','strength','delta','grow','pchg'].map(c => ({
  _c:c, getAttribute:()=>c,
  querySelector:()=>(_arrows[c] = _arrows[c] || {textContent:''})}));
const document = {
  getElementById: id => (['mm-tbody','mm-check-all','mm-min-str','mm-sel-count',
                          'mm-open-btn','mm-updated','mm-limited-hint',
                          'mm-str-out'].includes(id)
                         ? _el(id) : null),
  querySelectorAll: sel => (String(sel).includes('data-mmsort') ? _ths : _tabs),
  // Вкладки шукають і поштучно (лічильник «у фільтрі / поза фільтром»).
  querySelector: sel => {
    const m = String(sel).match(/data-mmdir="([^"]+)"/);
    return m ? (_tabs.find(t => t.dataset.mmdir === m[1]) || null) : null;
  },
};
// Памʼять фільтрів (localStorage) — сторінка без неї не запускається.
const _LS = {};
const localStorage = {
  getItem: k => (k in _LS ? _LS[k] : null),
  setItem: (k, v) => { _LS[k] = String(v); },
  removeItem: k => { delete _LS[k]; },
};
const fetch = async () => ({ok:true, json: async () => ({})});
const confirm = () => false;
const loadFuelFilterStatus = () => {};
const saveFuelFilterSettings = () => {};
const alert = () => {};
const ffFuelCell = () => '<mm/>';
const fmtPriceJS = p => String(p);
// Спільний лінк на TradingView (той самий, що в watchlist) — у фейк-DOM
// підміняємо простим текстом, щоб перевіряти РОЗТАШУВАННЯ, а не розмітку.
const tvSym = (s, label) => String(label != null ? label : s);
'''
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False,
                                     encoding='utf-8') as f:
        f.write(pre + src + '\n' + body)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    if r.returncode != 0:
        raise AssertionError(f'node: {r.stderr[:900]}')
    return r.stdout.strip()


def test_js_selection_survives_a_redraw_with_new_numbers():
    """⚠️ КЛЮЧОВА ПОВЕДІНКА. Полл приходить кожні ~10с і сила МММ у ньому
    інша → сигнатура змінюється → таблиця перемальовується. Галочки МУСЯТЬ
    лишитись, інакше обрати кілька монет фізично неможливо."""
    out = _run_js(r'''
mmApplyState({rows:[
  {symbol:'AAAUSDT', mm:'LONG', strength:50, strength_prev:40, price:1, in_trade:false, in_queue:false, selectable:true},
  {symbol:'BBBUSDT', mm:'SHORT', strength:30, strength_prev:30, price:2, in_trade:false, in_queue:false, selectable:true}],
  limited:false, ts:1, counts:{LONG:1,SHORT:1,flat:0}});
mmRowToggle('AAAUSDT', true);
// НОВИЙ полл: та сама монета, ІНША сила → DOM перебудується
mmApplyState({rows:[
  {symbol:'AAAUSDT', mm:'LONG', strength:77, strength_prev:50, price:1, in_trade:false, in_queue:false, selectable:true},
  {symbol:'BBBUSDT', mm:'SHORT', strength:31, strength_prev:30, price:2, in_trade:false, in_queue:false, selectable:true}],
  limited:false, ts:2, counts:{LONG:1,SHORT:1,flat:0}});
// ⚠️ Рахуємо САМЕ атрибут `checked` на полі, а не всі входження слова:
// `this.checked` в обробнику кожного рядка теж містить його.
const _h = document.getElementById('mm-tbody').innerHTML;
console.log(JSON.stringify({sel:[..._mmSel],
  checked:(_h.match(/type="checkbox" checked/g)||[]).length,
  aaaRow:/AAAUSDT/.test(_h)}));
''')
    import json
    d = json.loads(out)
    _check(d['sel'] == ['AAAUSDT'], f'вибір загубився під час полла: {d}')
    _check(d['aaaRow'], f'рядок зник із таблиці: {d}')
    _check(d['checked'] == 1,
           f'галочка не відновилась у перемальованій розмітці: {d}')
    print('✓ JS: галочка переживає перемальовку з новими числами')


def test_js_selection_drops_a_coin_that_left_the_snapshot():
    """Монета відкрилась і зникла зі знімка → знімати її з вибору мусить сам
    рендер, інакше кнопка обіцяла б відкрити те, чого в таблиці вже немає."""
    out = _run_js(r'''
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:50, strength_prev:null, price:1, in_trade:false, in_queue:false, selectable:true}],
  limited:false, ts:1, counts:{LONG:1,SHORT:0,flat:0}});
mmRowToggle('AAAUSDT', true);
mmApplyState({rows:[{symbol:'BBBUSDT', mm:'LONG', strength:50, strength_prev:null, price:1, in_trade:false, in_queue:false, selectable:true}],
  limited:false, ts:2, counts:{LONG:1,SHORT:0,flat:0}});
console.log(JSON.stringify({sel:[..._mmSel]}));
''')
    import json
    _check(json.loads(out)['sel'] == [], 'зникла монета лишилась обраною')
    print('✓ JS: монета, що зникла зі знімка, знімається з вибору')


def test_js_select_all_touches_only_visible_and_selectable_rows():
    """«Обрати всі» під фільтром SHORT не має тихо обрати LONG-монети, а
    монету в угоді — не має обирати взагалі."""
    out = _run_js(r'''
mmApplyState({rows:[
  {symbol:'LONGA', mm:'LONG', strength:50, strength_prev:null, price:1, in_trade:false, in_queue:false, selectable:true},
  {symbol:'SHRTA', mm:'SHORT', strength:40, strength_prev:null, price:1, in_trade:false, in_queue:false, selectable:true},
  {symbol:'SHRTB', mm:'SHORT', strength:35, strength_prev:null, price:1, in_trade:true,  in_queue:false, selectable:false}],
  limited:false, ts:1, counts:{LONG:1,SHORT:2,flat:0}});
mmSetDir('SHORT');
mmToggleAll(true);
console.log(JSON.stringify({sel:[..._mmSel].sort()}));
''')
    import json
    _check(json.loads(out)['sel'] == ['SHRTA'],
           f"«обрати всі» зачепило зайве: {out}")
    print('✓ JS: «обрати всі» — лише видимі й придатні рядки')


def test_js_strength_filter_narrows_the_view():
    out = _run_js(r'''
mmApplyState({rows:[
  {symbol:'STRONG', mm:'LONG', strength:60, strength_prev:null, price:1, in_trade:false, in_queue:false, selectable:true},
  {symbol:'WEAKER', mm:'LONG', strength:12, strength_prev:null, price:1, in_trade:false, in_queue:false, selectable:true}],
  limited:false, ts:1, counts:{LONG:2,SHORT:0,flat:0}});
document.getElementById('mm-min-str').value = '50';
mmRender();
const h = document.getElementById('mm-tbody').innerHTML;
console.log(JSON.stringify({strong:h.includes('STRONG'), weak:h.includes('WEAKER')}));
''')
    import json
    d = json.loads(out)
    _check(d['strong'] and not d['weak'], f'фільтр сили не працює: {d}')
    print('✓ JS: фільтр «Сила ≥» звужує показ')


def test_route_exists_for_the_button():
    """URL фронта мусить збігатися з маршрутом Flask (урок `submitManualTp1`:
    вигадана адреса давала мовчазний 404)."""
    fa = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    _check("@app.route('/api/fuel-filter/mm-monitor/open'" in fa, 'немає маршруту')
    _check("fetch('/api/fuel-filter/mm-monitor/open'" in _HTML, 'фронт не кличе')
    _check('if (!r.ok)' in _HTML[_HTML.index('async function mmOpenSelected'):
                                 _HTML.index('async function mmOpenSelected') + 2200],
           'HTTP-статус не перевіряється — 404 виглядав би як «відкрито»')
    print('✓ маршрут існує, фронт його кличе і перевіряє статус')


# ═══════════ 7. ТУМБЛЕР СЕКЦІЇ (вимога 15.09 #1) ═════════════════════════
def test_toggle_default_is_on_and_is_a_real_setting():
    """«Додай тумблер, так як у всіх Черг» — власний ключ налаштувань,
    зведений до bool (UI шле і рядки, і None), дефолт УВІМК: монітор нічого
    не відкриває сам, тож увімкненим він потік угод не розширює."""
    _check(_m.DEFAULT_SETTINGS.get('mm_monitor_enabled') is True,
           _m.DEFAULT_SETTINGS.get('mm_monitor_enabled'))
    _check("s['mm_monitor_enabled'] = bool(" in _SRC,
           'ключ не зводиться до bool у нормалізації налаштувань')
    print('✓ тумблер: окремий ключ, дефолт УВІМК, bool-нормалізація')


def test_toggle_off_stops_the_work_not_just_the_view():
    """⚠️ ГОЛОВНЕ ПРО ТУМБЛЕР: вимкнено → знімок НЕ будується взагалі.
    Це і є економія — старий МММ по 200+ монетах щотакту не рахується.
    Якби ми лише ховали таблицю, робота лишилась би, а тумблер брехав би."""
    ff = _mk(mon=False)
    _cap(ff, BTCUSDT=0.5, ETHUSDT=-0.5)
    _check(not ff._legacy_calls,
           f'вимкнений монітор усе одно рахував МММ: {ff._legacy_calls}')
    st = ff.mm_monitor_state()
    _check(st['enabled'] is False and st['rows'] == [], st)
    print('✓ тумблер OFF → нічого не рахуємо і нічого не віддаємо')


def test_toggle_off_clears_the_old_snapshot():
    """Інакше таблиця показувала б ЗАМОРОЖЕНІ числа, які вже ніхто не оновлює
    — а це гірше за порожню таблицю: виглядає як живі дані."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.5)
    _check(ff.mm_monitor_state()['rows'], 'знімок мав зʼявитись')
    ff._settings['mm_monitor_enabled'] = False
    _cap(ff, BTCUSDT=0.5)
    _check(ff.mm_monitor_state()['rows'] == [], 'старий знімок не прибрано')
    print('✓ вимкнення чистить старий знімок (без «замороженої» таблиці)')


def test_group_open_refuses_clearly_when_the_monitor_is_off():
    """⚠️ Без окремої перевірки кожна монета дістала б причину «МММ без
    напрямку» — тобто тумблер виглядав би як «ринок у рівновазі»."""
    _install_log()
    ff = _mk(mon=False)
    r = ff.group_open(['BTCUSDT'])
    _check(not r['ok'] and not ff.opened, r)
    _check('вимкнено' in (r.get('reason') or ''), r)
    print(f'✓ групове відкриття при вимкненому моніторі: {r["reason"]!r}')


def test_ui_has_the_section_toggle_like_the_queues():
    i = _HTML.index('togglePanel(\'mm\')')
    head = _HTML[i - 400:i + 2200]
    _check('id="ff-mm-monitor-enabled"' in head, 'тумблера немає в шапці секції')
    _check('tm-toggle-switch' in head, 'тумблер має бути таким самим, як у Черг')
    _check('saveFuelFilterSettings()' in head, 'тумблер нічого не зберігає')
    # ⚠️ Клік по тумблеру НЕ має згортати гармошку (шапка — клікабельна).
    _check('event.stopPropagation()' in head,
           'клік по тумблеру згортатиме секцію')
    _check("mm_monitor_enabled: _c('ff-mm-monitor-enabled')" in _HTML,
           'ключ не йде в збереження налаштувань')
    _check("setIf('ff-mm-monitor-enabled'" in _HTML,
           'стан тумблера не відновлюється із налаштувань')
    print('✓ UI: тумблер секції — як у Черг (зберігає і відновлюється)')


def test_js_says_why_the_table_is_empty_when_switched_off():
    """Три РІЗНІ причини порожньої таблиці не мають зливатись в одну."""
    out = _run_js(r'''
const h = () => document.getElementById('mm-tbody').innerHTML;
mmApplyState({rows:[], enabled:false, limited:false, ts:0, counts:{}});
const off = h();
mmApplyState({rows:[], enabled:true, limited:false, ts:1, counts:{}});
const nodata = h();
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:50, strength_prev:null, price:1, in_trade:false, in_queue:false, selectable:true}],
  enabled:true, limited:false, ts:2, counts:{LONG:1,SHORT:0,flat:0}});
mmSetDir('SHORT');
const filtered = h();
console.log(JSON.stringify({off, nodata, filtered}));
''')
    import json
    d = json.loads(out)
    _check('вимкнено' in d['off'], f'не сказано про тумблер: {d["off"]}')
    _check('Немає даних' in d['nodata'], d['nodata'])
    _check('фільтр' in d['filtered'], d['filtered'])
    _check(len({d['off'], d['nodata'], d['filtered']}) == 3,
           'три різні стани дають однаковий текст')
    print('✓ JS: порожня таблиця називає СВОЮ причину')


# ═══════════ 8. САМЕ СТАРИЙ МММ (вимога 15.09 #2) ════════════════════════
def test_monitor_shows_the_LEGACY_mm_not_the_new_one():
    """Дослівно: «Мені потрібно МММ саме старого зразка показник».
    Старий = `_fuel_dir_legacy` (сирий (fa−fb)/den по кластерах liq-map) —
    ТОЙ САМИЙ, що живить шар «Старий МММ» у Черзі-4."""
    ff = _mk()
    # Старий каже SHORT −0.40, новий у той самий момент — LONG +0.40.
    ff._legacy = {'BTCUSDT': -0.40}
    ff._mm_capture(_fuels(BTCUSDT=0.40))
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['mm'] == 'SHORT', f'показано НОВИЙ МММ замість старого: {r}')
    _check(r['dir'] == -0.4 and r['strength'] == 40, r)
    print('✓ у таблиці СТАРИЙ МММ (новий поруч — і його НЕ показуємо)')


def test_capture_uses_the_shared_legacy_function_and_no_extra_network():
    """ЗАМОК У КОДІ. Legacy читає `_liq_state` — той самий кешований знімок,
    який щойно взяв новий МММ, тож зайвих запитів немає. Своєї копії формули
    в моніторі бути НЕ повинно (інакше два «однакових» числа розійдуться)."""
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == '_mm_capture')
    body = ast.dump(fn)
    _check('_fuel_dir_legacy' in body, 'знімок не бере СТАРИЙ МММ')
    for bad in ('get_liquidation_map', 'fetch_klines', 'get_ticker'):
        _check(bad not in body, f'знімок ходить у мережу («{bad}»)')
    print('✓ знімок бере старий МММ спільною функцією, без зайвої мережі')


def test_price_still_comes_from_the_snapshot_because_legacy_has_none():
    """`_fuel_dir_legacy` не повертає `mark_price` взагалі — ціну беремо з
    того самого зрізу (новий МММ), бо це ОДНЕ Й ТЕ САМЕ число."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.5)
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['price'] == 100.5, f'ціна загубилась: {r}')
    print('✓ ціна зі знімка (legacy її не віддає)')


def test_a_coin_without_legacy_data_is_skipped():
    """Немає старого МММ по монеті → рядка немає. Вигадувати «рівновагу» не
    можна: це різні речі («даних нема» ≠ «тиску нема»)."""
    ff = _mk()
    ff._legacy = {'BTCUSDT': 0.5}          # ETH свідомо без legacy
    ff._mm_capture(_fuels(BTCUSDT=-0.5, ETHUSDT=-0.5))
    got = [r['symbol'] for r in ff.mm_monitor_state()['rows']]
    _check(got == ['BTCUSDT'], got)
    print('✓ монета без старого МММ у таблицю не потрапляє')


def test_ui_labels_the_column_as_the_old_mm():
    i = _HTML.index('id="mm-table"')
    head = _HTML[i:i + 1600]
    _check('Старий МММ' in head, 'колонка не підписана як СТАРИЙ МММ')
    _check('Черзі-4' in head, 'у підказці не сказано, що це той самий показник')
    print('✓ UI: колонка підписана «Старий МММ»')


# ═══════════ 9. КОЛОНКА «СИЛА» ПРИБРАНА (вимога 15.09 #3) ════════════════
def test_duplicate_strength_column_is_gone_from_the_table():
    """«Прибери колонку Сила — вона все одно дублює показник МММ»: віджет
    `ffFuelCell` уже малює і напрямок, і 0-100%, і стрілку тренду.

    ⚠️ НЕ ПЛУТАТИ з колонкою «Сила **росте**» (додана пізніше): та показує
    ЗМІНУ сили в пунктах + таймер росту — інша метрика, не дублікат."""
    i = _HTML.index('id="mm-table"')
    tbl = _HTML[i:i + 2600]
    _check('>Сила</th>' not in tbl and '>Сила<span' not in tbl,
           'колонка-дублікат «Сила» повернулась у заголовок')
    _check('Сила росте' in tbl, 'колонка ПРИРОСТУ має лишатись')
    print('✓ UI: дублікат «Сила» прибрано (а «Сила росте» — інша метрика)')


def test_row_cells_match_the_header_and_percent_lives_in_the_widget():
    """Кількість `<td>` мусить збігатися з кількістю `<th>` — інакше таблиця
    «зʼїжджає». Саме число % нікуди не зникло: воно всередині віджета МММ."""
    out = _run_js(r'''
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:57, strength_prev:40,
  delta:17, delta_rel:42.5, grow_since:null, price:1.5, in_trade:false,
  in_queue:false, selectable:true}],
  enabled:true, limited:false, ts:1, counts:{LONG:1,SHORT:0,flat:0}});
const h = document.getElementById('mm-tbody').innerHTML;
console.log(JSON.stringify({cells:(h.match(/<td/g)||[]).length,
                            widget:h.includes('<mm/>')}));
''')
    import json, re as _re
    d = json.loads(out)
    # ⚠️ Ріжемо рівно по `</thead>`, а не «перші N символів»: підказки в
    # заголовках ростуть, і фіксоване вікно вже одного разу обрізало таблицю
    # посередині — тест падав «на рівному місці».
    i = _HTML.index('id="mm-table"')
    _th = len(_re.findall(r'<th[\s>]', _HTML[i:_HTML.index('</thead>', i)]))
    _check(d['cells'] == _th, f'{d["cells"]} комірок проти {_th} заголовків')
    _check(d['widget'], 'віджет МММ (де і стоїть %) зник із рядка')
    # Сила лишається у ДАНИХ — вона живить сортування і фільтр «Сила ≥».
    _check('_mmVisibleRows' in _HTML and 'mm-min-str' in _HTML,
           'фільтр/сортування за силою прибирати НЕ просили')
    print(f'✓ рядок: {d["cells"]} комірок = {_th} заголовків, % — у віджеті')


# ═══════════ 10. 📈 «СИЛА РОСТЕ» ЧИСЛОМ + ⏱ ТАЙМЕР (вимога 15.09) ════════
def test_growth_is_measured_in_points_not_relative_percent():
    """⚠️ ГОЛОВНЕ РІШЕННЯ. Сила — це вже ЧАСТКА у відсотках, тож її зміна
    міряється в ПУНКТАХ (45% → 57% = +12 п.п.). Відносний % тут пастка: 1% → 5%
    дало б «+400%», і сортування за зростанням підняло б нагору ШУМ замість
    монет із реальним тиском. Відносне число віддаємо ОКРЕМО — для підказки."""
    ff = _mk()
    _cap(ff, AAAUSDT=0.45, BBBUSDT=0.01)
    _cap(ff, AAAUSDT=0.57, BBBUSDT=0.05)
    _cap(ff, AAAUSDT=0.57, BBBUSDT=0.05)      # добираємо вікно спостереження
    by = {r['symbol']: r for r in ff.mm_monitor_state()['rows']}
    _check(by['AAAUSDT']['delta'] == 12, by['AAAUSDT'])      # 45 → 57 п.п.
    _check(by['BBBUSDT']['delta'] == 4, by['BBBUSDT'])       # 1 → 5 п.п.
    # Відносний % теж є — але саме як ДОВІДКА поруч, не як критерій.
    _check(by['BBBUSDT']['delta_rel'] == 400.0, by['BBBUSDT'])
    _check(by['AAAUSDT']['delta'] > by['BBBUSDT']['delta'],
           'за пунктами реальний тиск має бути вище за шум')
    print('✓ приріст у ПУНКТАХ (відносний % — лише в підказці)')


def test_no_previous_tick_means_no_number_invented():
    ff = _mk()
    _cap(ff, BTCUSDT=0.40)
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['delta'] is None and r['delta_rel'] is None, r)
    print('✓ немає попереднього такту → приріст не вигадуємо')


def _since(ff, sym='BTCUSDT'):
    r = [x for x in ff.mm_monitor_state()['rows'] if x['symbol'] == sym]
    return r[0]['grow_since'] if r else None


def test_grow_timer_starts_on_growth_and_holds_through_a_single_flat_tick():
    """Вимога 15.09: «включай таймер при кожному старті росту і обнуляй, коли
    перестає рости». Уточнення 15.09 («щоб не було частого мерехтіння»): ОДИН
    рівний такт — це ще НЕ «перестала рости», інакше таймер гас щохвилини на
    дрібному тремтінні цілого числа сили."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.20)
    _check(_since(ff) is None, 'ще не росла')
    _cap(ff, BTCUSDT=0.40)
    _cap(ff, BTCUSDT=0.55)                       # вікно набралось → СТАРТ
    t1 = _since(ff)
    _check(t1, 'таймер не стартував на рості')
    _cap(ff, BTCUSDT=0.60)                       # росте далі → той самий старт
    _check(_since(ff) == t1, 'таймер перезапустився посеред росту')
    _cap(ff, BTCUSDT=0.60)                       # ОДИН рівний такт
    _check(_since(ff) == t1,
           'один рівний такт НЕ має збивати таймер — це і є мерехтіння')
    print('✓ таймер: старт на рості, один рівний такт його не збиває')


def test_grow_timer_stops_when_the_rise_ages_out_of_the_window():
    """Друга половина «золотої середини»: показник не має ЗАСТРЯГАТИ. Сила
    стоїть на місці → щойно старий низький рівень випав із вікна, приросту
    більше немає і таймер гасне сам."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.20)
    _cap(ff, BTCUSDT=0.40)
    _cap(ff, BTCUSDT=0.60)
    _check(_since(ff), 'таймер мав стартувати')
    for _ in range(int(_m.MM_GROW_WINDOW_SEC / _m.CYCLE_SECS) + 1):
        _cap(ff, BTCUSDT=0.60)                   # рівне плато
    _check(_since(ff) is None,
           'плато довше за вікно мусить погасити таймер')
    _cap(ff, BTCUSDT=0.90)
    _cap(ff, BTCUSDT=0.90)
    t3 = _since(ff)
    _check(t3, f'новий ріст мусить дати НОВИЙ старт: {t3}')
    print('✓ таймер гасне, коли ріст випав із вікна, і стартує на новому')


def test_giveback_from_the_peak_stops_the_timer_at_once():
    """⚠️ Без цього вікно тягнуло б «росте» ще кілька хвилин після розвороту —
    та сама «застарілість», лише з іншого боку. Відкат від піку вікна більший
    за поріг = ріст скінчився, і це видно ОДРАЗУ."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.20)
    _cap(ff, BTCUSDT=0.40)
    _cap(ff, BTCUSDT=0.60)
    _check(_since(ff), 'таймер мав стартувати')
    _cap(ff, BTCUSDT=0.46)                       # −14 п.п. від піку
    r = [x for x in ff.mm_monitor_state()['rows'] if x['symbol'] == 'BTCUSDT'][0]
    _check(r['grow_since'] is None,
           'відкат від піку мусить гасити таймер негайно')
    _check(r['delta'] is not None and r['delta'] > 0,
           f'приріст за вікном ще додатний — саме тому й потрібне окреме '
           f'правило відкату: {r["delta"]}')
    print('✓ відкат від піку гасить таймер одразу, не чекаючи вікна')


def test_falling_strength_also_clears_the_timer():
    ff = _mk()
    _cap(ff, BTCUSDT=0.20)
    _cap(ff, BTCUSDT=0.40)
    _cap(ff, BTCUSDT=0.60)
    _check(ff.mm_monitor_state()['rows'][0]['grow_since'], 'мав стартувати')
    _cap(ff, BTCUSDT=0.30)                       # падіння
    _check(ff.mm_monitor_state()['rows'][0]['grow_since'] is None,
           'падіння — це теж «перестала рости»')
    print('✓ падіння сили обнуляє таймер так само, як плато')


def test_growth_threshold_matches_the_shared_arrow_widget():
    """⚠️ ОДНЕ ВИЗНАЧЕННЯ «РОСТЕ». Стрілку ↑ малює спільний `ffFuelCell` за
    умовою `now > prev + 1`. Якби таймер мав ІНШИЙ поріг, у рядку стрілка
    казала б «→», а таймер біг — суперечність прямо на екрані."""
    _check(_m.MM_GROW_MIN_DELTA == 1, _m.MM_GROW_MIN_DELTA)
    ff = _mk()
    _cap(ff, BTCUSDT=0.20)
    _cap(ff, BTCUSDT=0.21)                       # +1 п.п. — для стрілки це «→»
    _check(ff.mm_monitor_state()['rows'][0]['grow_since'] is None,
           '+1 п.п. стрілка показує як «→» — таймер не має стартувати')
    _cap(ff, BTCUSDT=0.23)                       # +2 п.п. — стрілка вже ↑
    _check(ff.mm_monitor_state()['rows'][0]['grow_since'], '+2 п.п. → ↑ і таймер')
    # І сам віджет на сторінці мусить лишатись із тим самим порогом.
    i = _HTML.index('function ffFuelCell(')
    _check('now > prev + 1' in _HTML[i:i + 2500],
           'поріг стрілки у ffFuelCell змінився — таймер розійдеться з нею')
    print('✓ «росте» визначено ОДИН раз (поріг таймера = поріг стрілки)')


def test_timer_is_cleared_when_the_monitor_is_switched_off():
    """Інакше після вмикання таймер показував би час, протягом якого монітор
    узагалі нічого не рахував."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.20)
    _cap(ff, BTCUSDT=0.40)
    _cap(ff, BTCUSDT=0.60)
    _check(ff._mm_grow_since, 'таймер мав бути')
    ff._settings['mm_monitor_enabled'] = False
    _cap(ff, BTCUSDT=0.90)
    _check(not ff._mm_grow_since, 'вимкнений монітор не має тримати таймери')
    _check(not ff._mm_str_hist, 'історія сили теж мусить піти')
    print('✓ вимкнення монітора чистить і таймери росту')


def test_one_missing_tick_does_not_wipe_the_indicator():
    """⚠️ ЦЕ Й БУВ КОРІНЬ МЕРЕХТІННЯ. liq-map мить не віддала стан → монета
    випадала зі знімка, її база стиралась, і показник гас на ДВА такти, хоча
    дані вже повернулись. Тепер база лежить в історії й переживає пропуск."""
    ff = _mk()
    _cap(ff, AAAUSDT=0.40)
    _cap(ff, AAAUSDT=0.44)
    _cap(ff, AAAUSDT=0.47)
    _check(ff.mm_monitor_state()['rows'][0]['delta'] == 7, 'база не набралась')
    _cap(ff)                                     # ТАКТ БЕЗ ДАНИХ по монеті
    _check(ff.mm_monitor_state()['rows'] == [], 'рядка не мало бути')
    _cap(ff, AAAUSDT=0.50)                       # дані повернулись
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['delta'] == 10,
           f'разовий пропуск стер базу — показник знову мерехтить: {r}')
    _check(r['grow_since'], 'і таймер росту теж мусив пережити пропуск')
    print('✓ разовий пропуск даних більше не стирає приріст і таймер')


def test_history_of_a_long_gone_coin_is_forgotten():
    """Памʼять обмежена ТИМ САМИМ вікном: монета, якої не було довше за нього,
    починає з чистого аркуша — її стара база непорівнянна з теперішнім станом."""
    ff = _mk()
    _cap(ff, AAAUSDT=0.20, BBBUSDT=0.20)
    _cap(ff, AAAUSDT=0.60, BBBUSDT=0.60)
    _cap(ff, AAAUSDT=0.60, BBBUSDT=0.60)
    _check(len(ff._mm_grow_since) == 2, ff._mm_grow_since)
    for _ in range(int(_m.MM_GROW_WINDOW_SEC / _m.CYCLE_SECS) + 2):
        _cap(ff, AAAUSDT=0.90)                   # BBB немає ДОВШЕ за вікно
    _check('BBBUSDT' not in ff._mm_grow_since and 'BBBUSDT' not in ff._mm_str_hist,
           f'стан зниклої монети лишився: {ff._mm_grow_since} / {ff._mm_str_hist}')
    print('✓ монета зникла надовго → історія і таймер прибрані')


# ═══════════ 11. UI: TradingView · колонка приросту · сортування ══════════
def test_symbol_opens_tradingview_exactly_like_the_watchlist():
    """«Зроби щоб при натисканні на монету відкривався TradingView, так як і в
    WATCHLIST» — беремо ТУ САМУ функцію `tvSym`, а не свій лінк."""
    i = _HTML.index('function mmRender()')
    fn = _HTML[i:i + 6000]
    _check('tvSym(r.symbol)' in fn, 'назва монети не веде у TradingView')
    # Та сама функція, що й у watchlist-рядку → та сама вкладка і той самий
    # формат символу (BYBIT:<SYM>.P). Другого лінка в проєкті не заводимо.
    _check('tvSym(sym, short)' in _HTML, 'watchlist мусить лишитись на tvSym')
    j = _HTML.index('function tvSym(')
    tv = _HTML[j:j + 900]
    _check("window.open(this.href,'tvchart')" in tv, 'одна вкладка tvchart')
    _check('event.stopPropagation()' in tv,
           'без stopPropagation клік по назві смикав би галочку рядка')
    print('✓ назва монети → TradingView через спільний tvSym (як у WATCHLIST)')


def test_ui_has_growth_column_with_timer_and_sorting():
    i = _HTML.index('id="mm-table"')
    tbl = _HTML[i:_HTML.index('</table>', i)]
    import re as _re
    _check('Сила росте' in tbl, 'немає колонки приросту')
    # ⏱ Таймер — ОКРЕМА колонка (вимога 15.09), а не хвіст комірки приросту.
    _check('⏱ Росте' in tbl, 'таймер не винесено в окрему колонку')
    _n = len(_re.findall(r'<th[\s>]', tbl))
    _check(_n == 8, f'очікував 8 колонок: {_n}')
    _check(f'colspan="{_n}"' in tbl,
           f'colspan порожнього рядка не дорівнює числу колонок ({_n})')
    for col in ('symbol', 'strength', 'delta', 'grow', 'pchg'):
        _check(f'data-mmsort="{col}"' in tbl, f'колонка {col} не сортується')
    _check("mmSort('delta')" in tbl, 'сортування за приростом не підключене')
    _check("mmSort('grow')" in tbl, 'сортування за таймером не підключене')
    _check('п.п.' in tbl, 'у підказці не сказано, що це ПУНКТИ, а не відносний %')
    print(f'✓ UI: «Сила росте» + окрема ⏱ колонка, {_n} колонок із сортуванням')


def test_timer_uses_the_shared_one_second_ticker():
    """⚠️ Секунди НЕ мають перебудовувати таблицю: той самий прийом, що з
    `held_sec` у Черзі-4 — `.ff-timer[data-since]` + ОДИН глобальний тікер."""
    # ⏱ Таймер живе у ВЛАСНІЙ функції комірки (окрема колонка), і саме її
    # перевіряємо — раніше зріз від `_mmGrowCell` випадково накривав сусідню
    # функцію, тобто тест «проходив» ні за що.
    i = _HTML.index('function _mmTimerCell(')
    fn = _HTML[i:_HTML.index('\n}', i)]
    _check('class="ff-timer" data-since=' in fn, 'таймер не на спільному класі')
    _check('fmtTimer' not in fn, 'секунди не треба малювати тут — їх веде тікер')
    _check('function _mmTimerCell(' in _HTML and '_mmTimerCell(r)' in _HTML,
           'комірка таймера не підключена до рядка таблиці')
    # Сам тікер уже існує і бере ВСІ такі елементи сторінки.
    _check(".ff-timer[data-since]" in _HTML and 'window._ffTimerTick' in _HTML,
           'глобальний 1с-тікер зник — таймер стоятиме')
    # `grow_since` у сигнатурі є (щоб старт/зупинка перемалювались), а секунд там
    # немає (інакше DOM перебудовувався б щосекунди).
    j = _HTML.index('const sig = _mmDir')
    sig = _HTML[j:j + 600]
    _check('r.grow_since' in sig, 'старт таймера не входить у сигнатуру')
    _check('Date.now' not in sig, 'у сигнатурі не має бути живого часу')
    print('✓ таймер: спільний 1с-тікер, таблиця не перебудовується щосекунди')


def test_js_sorting_by_growth_puts_the_fastest_first():
    out = _run_js(r'''
const R = (s, st, d, g) => ({symbol:s, mm:'LONG', strength:st, strength_prev:st-(d||0),
  delta:d, delta_rel:null, grow_since:g, price:1, in_trade:false, in_queue:false, selectable:true});
mmApplyState({rows:[R('AAA',50,3,null), R('BBB',80,null,null), R('CCC',40,12,null)],
  enabled:true, limited:false, ts:1, counts:{LONG:3,SHORT:0,flat:0}});
mmSort('delta');                     // перший клік по новій колонці → спадання
const h = document.getElementById('mm-tbody').innerHTML;
const order = ['AAA','BBB','CCC'].map(s => h.indexOf(s));
console.log(JSON.stringify({order}));
''')
    import json
    o = json.loads(out)['order']
    a, b, c = o[0], o[1], o[2]
    _check(c < a, 'монета з більшим приростом мусить бути вище')
    # ⚠️ Рядок БЕЗ приросту (null) — у кінець, а не «як нуль».
    _check(b > a and b > c, f'рядок без даних мусить бути внизу: {o}')
    print('✓ JS: сортування за приростом, «немає даних» — у кінець')


def test_js_growth_cell_agrees_with_the_arrow_deadzone():
    """Зелене число і стрілка ↑ мусять зʼявлятись ОДНОЧАСНО: +1 — це ще «→»."""
    out = _run_js(r'''
const R = (s, d) => ({symbol:s, mm:'LONG', strength:50, strength_prev:50-d, delta:d,
  delta_rel:null, grow_since:null, price:1, in_trade:false, in_queue:false, selectable:true});
mmApplyState({rows:[R('AAA',1), R('BBB',5), R('CCC',-5)], enabled:true,
  limited:false, ts:1, counts:{LONG:3,SHORT:0,flat:0}});
// ⚠️ Ріжемо саме на РЯДКИ: символ трапляється двічі в одному рядку (в
// обробнику галочки й у комірці назви), тож split по символу дав би шматок
// БЕЗ клітинки приросту — саме на цьому перша версія тесту й помилилась.
const rows = document.getElementById('mm-tbody').innerHTML.split('</tr>');
const cell = s => rows.filter(x => x.includes(s))[0] || '';
console.log(JSON.stringify({one:cell('AAA').includes('→'),
  up:cell('BBB').includes('↑'), down:cell('CCC').includes('↓')}));
''')
    import json
    d = json.loads(out)
    _check(d['one'], '+1 п.п. мусить показуватись як «→» (як і стрілка)')
    _check(d['up'] and d['down'], f'напрямок приросту не показано: {d}')
    print('✓ JS: поріг клітинки приросту збігається зі стрілкою')



# ═══════════ 11. ⏱ ТАЙМЕР — ОКРЕМА КОЛОНКА (вимога 15.09) ════════════════
def test_js_timer_lives_in_its_own_cell_not_in_the_growth_cell():
    """Вимога дослівно: «Таймер зроби окремою колонкою». Раніше він тулився
    хвостом у комірку приросту й розтягував її; тепер це власний стовпчик."""
    out = _run_js(r'''
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:57, strength_prev:40,
  delta:17, delta_rel:42.5, grow_since:1700000000, price:1.5,
  in_trade:false, in_queue:false, selectable:true}],
  enabled:true, limited:false, ts:1, counts:{LONG:1,SHORT:0,flat:0}});
const h = document.getElementById('mm-tbody').innerHTML;
// Розбираємо рядок на комірки: у ЯКІЙ саме стоїть таймер?
const tds = h.split('<td').slice(1);
const iTimer = tds.findIndex(t => t.includes('ff-timer'));
const iGrow = tds.findIndex(t => t.includes('+17'));
console.log(JSON.stringify({iTimer, iGrow, n:tds.length}));
''')
    import json
    d = json.loads(out)
    _check(d['iTimer'] >= 0, 'таймера немає в рядку взагалі')
    _check(d['iGrow'] >= 0, 'приросту немає в рядку')
    _check(d['iTimer'] != d['iGrow'],
           f'таймер і приріст в ОДНІЙ комірці — колонку не виділено: {d}')
    _check(d['iTimer'] == d['iGrow'] + 1,
           f'таймер мусить стояти одразу за приростом: {d}')
    print('✓ JS: ⏱ таймер — окрема комірка одразу за «Сила росте»')


def test_js_timer_cell_is_empty_when_strength_is_not_growing():
    """«Обнуляється, коли перестає рости» має бути ВИДНО: порожній таймер —
    це стан «не росте», а не «дані загубились»."""
    out = _run_js(r'''
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:57, strength_prev:57,
  delta:0, delta_rel:0, grow_since:null, price:1.5,
  in_trade:false, in_queue:false, selectable:true}],
  enabled:true, limited:false, ts:1, counts:{LONG:1,SHORT:0,flat:0}});
const h = document.getElementById('mm-tbody').innerHTML;
console.log(JSON.stringify({timer:h.includes('ff-timer'),
  says:/не росте/.test(h)}));
''')
    import json
    d = json.loads(out)
    _check(not d['timer'], 'таймер малюється, хоча сила не росте')
    _check(d['says'], 'порожній таймер нічого не пояснює')
    print('✓ JS: не росте → таймера немає, і причина названа')


# ═══════════ 12. 💹 НАПРЯМОК ЦІНИ (вимога 15.09) ═════════════════════════
def test_price_move_is_a_pure_function_with_a_deadzone():
    """⚠️ Мертва зона обовʼязкова: без неї «росте/падає» мигало б на
    кожній сотій відсотка. Правило те саме, що в колонці Price у 💰 Funding."""
    mv = _m.mm_price_move
    now = 10_000.0
    up = mv([(now - 600, 100.0), (now, 101.0)], now)
    down = mv([(now - 600, 100.0), (now, 99.0)], now)
    flat = mv([(now - 600, 100.0), (now, 100.05)], now)
    _check(up['dir'] == 'up' and up['chg'] == 1.0, up)
    _check(down['dir'] == 'down' and down['chg'] == -1.0, down)
    _check(flat['dir'] == 'flat', f'0.05% — це шум, а не напрямок: {flat}')
    # Рівно на межі мертвої зони — ще НЕ напрямок (строге «>»).
    edge = mv([(now - 600, 100.0), (now, 100.10)], now)
    _check(edge['dir'] == 'flat', f'на межі мертвої зони напрямку немає: {edge}')
    print('✓ рух ціни: чиста функція + мертва зона 0.10%')


def test_price_move_reports_how_much_history_it_actually_had():
    """⚠️ «+0.4% за 40 секунд» і «+0.4% за 15 хвилин» — різні за вагою
    твердження. Без `span` таблиця видавала б перше за друге."""
    mv = _m.mm_price_move
    now = 10_000.0
    fresh = mv([(now - 40, 100.0), (now, 100.4)], now)
    _check(fresh['span'] == 40.0, fresh)
    # Точки ПОЗА вікном участі не беруть.
    old = mv([(now - 5000, 50.0), (now - 60, 100.0), (now, 101.0)], now)
    _check(old['points'] == 2 and old['chg'] == 1.0,
           f'точка поза вікном потрапила в розрахунок: {old}')
    # Менше двох точок → чесне «немає напрямку», а не вигаданий нуль-рух.
    _check(mv([(now, 100.0)], now)['span'] == 0.0, 'одна точка — це не рух')
    _check(mv([], now)['dir'] == 'flat', 'порожня історія')
    print('✓ рух ціни: вікно і кількість точок віддаються чесно')


def test_price_window_matches_the_funding_column():
    """⚠️ ЗАМОК МІЖ ФАЙЛАМИ. Дві колонки на одній сторінці не мають називати
    «росте» різні речі: вікно і мертва зона мусять збігатися з
    `funding_monitor.PRICE_WINDOW` / `PRICE_DEADZONE`."""
    import re as _re
    src = open(os.path.join(_HERE, 'detection', 'funding_monitor.py'),
               encoding='utf-8').read()
    w = int(_re.search(r'^PRICE_WINDOW\s*=\s*(\d+)', src, _re.M).group(1))
    dz = float(_re.search(r'^PRICE_DEADZONE\s*=\s*([\d.]+)', src, _re.M).group(1))
    _check(_m.MM_PRICE_WINDOW_SEC == w * 60,
           f'вікно розійшлось: монітор {_m.MM_PRICE_WINDOW_SEC}с, фандинг {w}хв')
    _check(_m.MM_PRICE_DEADZONE == dz,
           f'мертва зона розійшлась: {_m.MM_PRICE_DEADZONE} проти {dz}')
    print(f'✓ вікно/мертва зона ціни збігаються з 💰 Funding ({w}хв · {dz}%)')


def test_snapshot_carries_the_price_direction():
    """Напрямок рахує ДВИГУН і кладе у знімок; `mm_monitor_state` лише читає
    (той самий поділ, що з шарами Черги-4)."""
    ff = _mk()
    ff._mm_price_hist = {'BTCUSDT': [(time.time() - 600, 100.0)]}
    _cap(ff, BTCUSDT=0.5)
    row = ff.mm_monitor_state()['rows'][0]
    _check(row['price_dir'] in ('up', 'down', 'flat'), row)
    _check('price_chg' in row and 'price_span' in row, row)
    # Ціна знімка — `mark_price` із нового зрізу (legacy її не віддає).
    _check(row['price'] is not None, row)
    print(f"✓ знімок несе напрямок ціни: {row['price_dir']} {row['price_chg']}%")


def test_price_history_is_trimmed_and_forgets_dead_symbols():
    """Інакше памʼять росла б, а монета, що повернулась у знімок, рахувала б
    «рух» від ціни годинної давності."""
    ff = _mk()
    # ⚠️ Годинник ВІРТУАЛЬНИЙ (`ff._clock`) — `time.time()` тут дав би точку
    # з майбутнього і тест перевіряв би не те.
    old = ff._clock[0] - (_m.MM_PRICE_WINDOW_SEC + 10 * _m.CYCLE_SECS)
    ff._mm_price_hist = {'BTCUSDT': [(old, 1.0)], 'ZZZUSDT': [(old, 2.0)]}
    _cap(ff, BTCUSDT=0.5)
    _check('ZZZUSDT' not in ff._mm_price_hist,
           'історія монети, якої немає у знімку, не прибрана')
    _check(all(t > old for t, _ in ff._mm_price_hist['BTCUSDT']),
           f'точка поза вікном лишилась: {ff._mm_price_hist}')
    print('✓ історія цін: обрізається вікном, сироти прибираються')


def test_js_price_cell_shows_direction_and_the_real_window():
    """⚠️ Вікно НЕ зашите у фронті: показуємо `price_span`, що прийшов із
    бекенда. Інакше одразу після рестарту «+0.4%» за 40с читалось би як
    «+0.4% за 15 хвилин»."""
    out = _run_js(r'''
const R = (s, dir, chg, span) => ({symbol:s, mm:'LONG', strength:50, strength_prev:50,
  delta:0, grow_since:null, price:1.5, price_dir:dir, price_chg:chg, price_span:span,
  in_trade:false, in_queue:false, selectable:true});
mmApplyState({rows:[R('AAAUSDT','up',0.42,900), R('BBBUSDT','down',-1.1,40),
  R('CCCUSDT','flat',0.02,900), R('DDDUSDT',null,0,0)],
  enabled:true, limited:false, ts:1, counts:{LONG:4,SHORT:0,flat:0}});
const rows = document.getElementById('mm-tbody').innerHTML.split('</tr>');
const cell = s => rows.filter(x => x.includes(s))[0] || '';
console.log(JSON.stringify({
  up:cell('AAA').includes('▲') && cell('AAA').includes('+0.42%'),
  win:cell('AAA').includes('15хв'), fresh:cell('BBB').includes('40с'),
  down:cell('BBB').includes('▼'), flat:cell('CCC').includes('▬'),
  none:!/[▲▼▬]/.test(cell('DDD'))}));
''')
    import json
    d = json.loads(out)
    _check(d['up'] and d['down'] and d['flat'], f'стрілки напрямку ціни: {d}')
    _check(d['win'] and d['fresh'],
           f'показане вікно не збігається з тим, що віддав бекенд: {d}')
    _check(d['none'], 'без історії напрямок вигадувати не можна')
    print('✓ JS: ціна ▲/▼/▬ + чесне вікно спостереження')


def test_js_price_sorting_uses_the_move_not_the_price():
    """Колонку додали, щоб бачити, ЯКА монета зараз іде вгору, — тож клік по
    заголовку сортує за РУХОМ. Рядок без історії — у кінець."""
    out = _run_js(r'''
const R = (s, px, dir, chg) => ({symbol:s, mm:'LONG', strength:50, strength_prev:50,
  delta:0, grow_since:null, price:px, price_dir:dir, price_chg:chg, price_span:900,
  in_trade:false, in_queue:false, selectable:true});
mmApplyState({rows:[R('AAAUSDT',900,'up',0.2), R('BBBUSDT',1,'up',5.0),
  R('CCCUSDT',50,null,0)], enabled:true, limited:false, ts:1,
  counts:{LONG:3,SHORT:0,flat:0}});
mmSort('pchg');
const h = document.getElementById('mm-tbody').innerHTML;
console.log(JSON.stringify({order:['AAA','BBB','CCC'].map(s => h.indexOf(s))}));
''')
    import json
    a, b, c = json.loads(out)['order']
    _check(b < a, 'монета з більшим РУХОМ мусить бути вище за дорожчу')
    _check(c > a and c > b, 'рядок без напрямку — у кінець')
    print('✓ JS: сортування колонки «Ціна» — за рухом, а не за ціною')


# ═══════════ 13. 🔮 ПРОГНОЗ + 🧠 РІШЕННЯ (скріни 1 і 2) ══════════════════
def _fn_code(name):
    """Тіло методу БЕЗ докстрінга — щоб пояснення в коментарях («ensure_fresh
    тут заборонений») не видавалось за виклик. Замок має дивитись на КОД."""
    src = _SRC[_SRC.index(f'def {name}('):]
    src = src[:src.index('\n    def ')]
    if '"""' in src:
        src = src[src.index('"""', src.index('"""') + 3) + 3:]
    return src


class _FE:
    """Фейковий кеш прогнозу — рівно тієї форми, що `forecast_engine.get`."""

    def __init__(self, data):
        self.data = data
        self.calls = []

    def get(self, sym):
        self.calls.append(sym)
        return self.data.get(sym)


def test_forecast_is_read_from_the_same_cache_as_the_chart_badge():
    """⚠️ Бейдж «🔮 1H» над графіком і колонка монітора мусять показувати ОДНЕ
    число — тому джерело одне: `forecast_engine.get`. Нічого не рахуємо і не
    довантажуємо (`ensure_fresh` ходить по свічки — це 200+ монет щотакту)."""
    fe = _FE({'BTCUSDT': {'forecast_1h': {'side': 1, 'pct': 100, 'confidence': 90},
                          'forecast_4h': {'side': 1, 'pct': 70, 'confidence': 75}}})
    out = FF._mm_forecast(fe, 'BTCUSDT')
    _check(out['f1'] == {'side': 1, 'pct': 100.0, 'conf': 90}, out)
    _check(out['f4'] == {'side': 1, 'pct': 70.0, 'conf': 75}, out)
    _check(FF._mm_forecast(None, 'X') == {'f1': None, 'f4': None},
           'без двигуна прогнозу — порожньо, а не виняток')
    _check(FF._mm_forecast(_FE({}), 'X') == {'f1': None, 'f4': None},
           'немає в кеші — порожньо')
    # ⚠️ `ensure_fresh` у цьому шляху заборонений — він ходить по свічки.
    _check('ensure_fresh' not in _fn_code('_mm_forecast'),
           'монітор не має права довантажувати прогноз (200+ монет щотакту)')
    print('✓ прогноз: той самий кеш, що в бейджа; нічого не рахуємо')


def test_neutral_forecast_is_not_the_same_as_no_forecast():
    """«ней» — це змістовна відповідь двигуна; порожній кеш — ні. У таблиці
    вони мусять читатись по-різному."""
    fe = _FE({'AAA': {'forecast_1h': {'side': 0, 'pct': 0, 'confidence': 12},
                      'forecast_4h': {}}})
    out = FF._mm_forecast(fe, 'AAA')
    _check(out['f1'] == {'side': 0, 'pct': 0.0, 'conf': 12}, out)
    _check(out['f4'] is None, f'порожній прогноз мусить бути None: {out}')
    print('✓ прогноз: нейтраль ≠ «немає даних»')


def test_snapshot_carries_the_forecast_pair():
    ff = _mk()
    fe = _FE({'BTCUSDT': {'forecast_1h': {'side': -1, 'pct': -40, 'confidence': 60},
                          'forecast_4h': {'side': 1, 'pct': 12, 'confidence': 30}}})
    ff._mm_forecast_engine = lambda: fe
    _cap(ff, BTCUSDT=0.5)
    row = ff.mm_monitor_state()['rows'][0]
    _check(row['f1']['side'] == -1 and row['f4']['side'] == 1, row)
    # Двигун прогнозу беремо ОДИН раз на такт, а не на монету.
    _check(fe.calls == ['BTCUSDT'], fe.calls)
    print('✓ знімок несе пару прогнозів 1H/4H')


def test_js_forecast_is_two_columns_with_direction_only():
    """Вимога 15.09: дві ОКРЕМІ колонки і в кожній лише «1H: 🔴 SHORT».
    Числа (рух і впевненість) нікуди не зникли — вони в підказці комірки."""
    out = _run_js(r'''
const R = (s, f1, f4) => ({symbol:s, mm:'LONG', strength:50, strength_prev:50, delta:0,
  grow_since:null, price:1, price_dir:'flat', price_chg:0, price_span:900,
  delta_span:180, f1:f1, f4:f4, selectable:true});
mmApplyState({rows:[
  R('AAAUSDT', {side:1,pct:100,conf:90}, {side:-1,pct:-70,conf:75}),
  R('BBBUSDT', {side:0,pct:0,conf:12}, null)],
  enabled:true, limited:false, ts:1});
const rows = document.getElementById('mm-tbody').innerHTML.split('</tr>');
const cell = s => rows.filter(x => x.includes(s))[0] || '';
const a = cell('AAA'), b = cell('BBB');
console.log(JSON.stringify({
  a1:a.includes('1H: 🟢 LONG'), a4:a.includes('4H: 🔴 SHORT'),
  // ДВІ окремі комірки, а не одна з <br>
  twoCells:(a.split('<td').length - 1) === 8 && !/1H.*<br>.*4H/.test(a),
  // ⚠️ Числа МУСЯТЬ бути в підказці, тож шукаємо їх лише у ВИДИМОМУ тексті
  // (title вирізаємо) — інакше тест забороняв би те, що ми свідомо лишили.
  noPct:!a.replace(/title="[^"]*"/g, '').includes('+100%'),
  tip:a.includes('очікуваний рух'),
  neutral:b.includes('1H: ⚪ ней'), none:b.includes('4H: —')}));
''')
    import json
    d = json.loads(out)
    _check(d['a1'] and d['a4'], f'напрямок прогнозу не показано: {d}')
    _check(d['twoCells'], f'прогноз має бути у ДВОХ окремих комірках: {d}')
    _check(d['noPct'], f'у комірці лишились відсотки — просили лише напрямок: {d}')
    _check(d['tip'], f'числа мусять лишитись у підказці: {d}')
    _check(d['neutral'] and d['none'],
           f'«ней» і «немає прогнозу» мусять читатись по-різному: {d}')
    print('✓ JS: 🔮 дві колонки, лише напрямок, числа — у підказці')

def test_snapshot_for_reads_and_computes_nothing():
    """⚠️ `/api/tm/state` опитується кожні 5с під ОДНИМ gunicorn-воркером.
    Саме через розрахунки на цьому ендпоінті колонку «МММ» колись і прибрали —
    тож тепер вона бере ГОТОВИЙ знімок двигуна."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.5, ETHUSDT=-0.42)
    ff._legacy_calls.clear()
    got = ff.mm_snapshot_for(['btcusdt', 'ZZZUSDT'])
    _check(not ff._legacy_calls, f'читач порахував МММ: {ff._legacy_calls}')
    _check(set(got) == {'BTCUSDT'}, f'символ не нормалізовано / зайве: {got}')
    _check(got['BTCUSDT']['mm'] == 'LONG' and got['BTCUSDT']['strength'] == 50, got)
    _check(ff.mm_snapshot_for([]) == {}, 'порожній запит — порожня відповідь')
    print('✓ mm_snapshot_for: лише читання знімка')


def test_trades_column_shows_the_same_number_as_the_monitor():
    """Та сама метрика у двох місцях = ОДНЕ значення (правило проєкту)."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.30)
    _cap(ff, BTCUSDT=0.55)
    row = [r for r in ff.mm_monitor_state()['rows'] if r['symbol'] == 'BTCUSDT'][0]
    cell = ff.mm_snapshot_for(['BTCUSDT'])['BTCUSDT']
    for k in ('mm', 'strength', 'strength_prev', 'delta', 'grow_since'):
        _check(row[k] == cell[k],
               f'{k}: монітор {row[k]!r} ≠ таблиця угод {cell[k]!r}')
    print('✓ колонка угод і монітор показують ОДНЕ число')


def test_open_trades_keep_their_mm_even_when_the_monitor_is_off():
    """⚠️ Колонка в таблиці УГОД до монітора стосунку не має. Прив'язати її до
    чужого тумблера означало б «вимкнув монітор — зникли числа в угодах».
    Але сама ТАБЛИЦЯ МОНІТОРА при цьому лишається порожньою."""
    ff = _mk(mon=False)
    ff._fuel_managed = {'BTCUSDT': {}}
    _cap(ff, BTCUSDT=0.5, ETHUSDT=-0.5)
    _check(ff._legacy_calls == ['BTCUSDT'],
           f'при вимкненому моніторі рахуємо ЛИШЕ монети в угоді: {ff._legacy_calls}')
    _check(ff.mm_snapshot_for(['BTCUSDT']).get('BTCUSDT'),
           'колонка в таблиці угод лишилась без числа')
    st = ff.mm_monitor_state()
    _check(st['enabled'] is False and st['rows'] == [],
           f'вимкнений монітор показує рядки: {st}')
    print('✓ тумблер монітора не гасить колонку в таблицях угод')


def test_state_route_enriches_positions_from_the_snapshot():
    src = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    i = src.index("@app.route('/api/tm/state')")
    blk = src[i:i + 4000]
    _check('mm_snapshot_for' in blk, 'маршрут не читає знімок МММ')
    _check("pos['old_mm']" in blk, 'позиція не отримує поле old_mm')
    # ⚠️ Жодних розрахунків на цьому гарячому ендпоінті.
    for bad in ('_fuel_dir_legacy', 'compute_mm', '_liq_state'):
        _check(bad not in blk, f'на /api/tm/state зʼявився розрахунок: {bad}')
    print('✓ /api/tm/state: колонка живиться знімком, без розрахунків')


def test_ui_trades_tables_got_the_column_and_the_right_colspan():
    import re as _re
    for tid, empty in (('tm-open-table', 'No open positions'),
                       ('tm-shadow-open-table', 'No paper positions')):
        i = _HTML.index(f'id="{tid}"')
        tbl = _HTML[i:_HTML.index('</table>', i)]
        _check('Старий МММ' in tbl, f'{tid}: немає заголовка колонки')
        n = len(_re.findall(r'<th[\s>]', tbl))
        m = _re.search(r'colspan="(\d+)"', tbl)
        _check(m and int(m.group(1)) == n,
               f'{tid}: colspan {m and m.group(1)} ≠ {n} заголовків')
    # Комірку малює ТОЙ САМИЙ спільний віджет, що й у моніторі та чергах.
    i = _HTML.index('function oldMmCellHTML(')
    fn = _HTML[i:_HTML.index('\n}', i)]
    _check('ffFuelCell(' in fn, 'колонка малює МММ своїм способом')
    _check(_HTML.count('${oldMmCellHTML(p)}') == 2,
           'комірка підключена не в обидві таблиці (real + paper)')
    print('✓ UI: колонка «🧮 Старий МММ» у real+paper, спільний віджет')


# ═══ 15. 🩹 БОРОТЬБА З МЕРЕХТІННЯМ: вікно замість «попереднього такту» ════
def test_window_change_is_one_shared_pure_function():
    """⚠️ Дві «однакові» функції зміни-за-вікном розійшлись би, і «за 3 хв» у
    сусідніх колонках означало б різні речі. Тому ядро ОДНЕ."""
    f = _m.mm_window_change
    now = 10_000.0
    h = [(now - 120, 40), (now - 60, 55), (now, 50)]
    w = f(h, now, 180)
    _check(w['first'] == 40 and w['last'] == 50 and w['abs'] == 10, w)
    _check(w['peak'] == 55 and w['low'] == 40, f'пік/дно потрібні для відкату: {w}')
    _check(w['span'] == 120.0 and w['points'] == 3, w)
    # Менше двох точок у вікні → чесно порожньо, а не вигаданий нуль.
    _check(f([(now, 5)], now, 180)['abs'] is None, 'одна точка — це не зміна')
    _check(f([], now, 180)['span'] == 0.0, 'порожня історія')
    # Точки поза вікном участі не беруть.
    _check(f([(now - 5000, 1), (now - 10, 7), (now, 9)], now, 180)['first'] == 7,
           'стара точка потрапила у вікно')
    # І ЦЕ САМЕ ЯДРО живить рух ціни — окремої реалізації немає.
    src = _SRC[_SRC.index('def mm_price_move('):]
    src = src[:src.index('\n\nclass ')]
    _check('mm_window_change(' in src, 'рух ціни рахує щось своє')
    print('✓ зміна-за-вікном: одна чиста функція на ціну і на силу')


def test_growth_needs_a_real_window_before_it_is_shown():
    """⚠️ Половина вимоги «не застарілий»: у межах ОДНОГО оновлення джерела
    рівні liq-map ідентичні, тож «приріст» там був би шумом ціни, а не ростом.
    Поки історії менше — чесне «—» плюс `delta_span`, щоб було видно, скільки
    вже набралось."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.40)
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['delta'] is None and r['delta_span'] == 0.0, r)
    _cap(ff, BTCUSDT=0.50)
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['delta'] is None, f'30с історії — ще не вимір: {r}')
    _check(r['delta_span'] == float(_m.CYCLE_SECS),
           f'набране вікно мусить бути видно: {r}')
    _cap(ff, BTCUSDT=0.50)
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['delta'] == 10 and r['delta_span'] >= _m.MM_GROW_MIN_SPAN_SEC, r)
    print('✓ приріст показуємо лише з реальним вікном, і вікно назване')


def test_tiny_drift_no_longer_blanks_the_indicator():
    """Сила — ЦІЛЕ число (|dir|×100): рух 0.520 → 0.525 дає ті самі «52».
    Раніше це означало приріст 0 і обнулений таймер КОЖЕН другий такт."""
    ff = _mk()
    _cap(ff, BTCUSDT=0.40)
    _cap(ff, BTCUSDT=0.45)
    _cap(ff, BTCUSDT=0.455)                 # те саме ціле 45
    _cap(ff, BTCUSDT=0.452)                 # і знову 45
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['delta'] == 5, f'дрібний дрейф зʼїв приріст: {r}')
    _check(r['grow_since'], 'дрібний дрейф не має гасити таймер')
    print('✓ тремтіння цілого числа сили більше не гасить показник')


def test_window_constants_are_consistent_with_the_source():
    """Вікно мусить накривати щонайменше два оновлення джерела, інакше
    повертається та сама вада, яку й лікуємо."""
    _check(_m.MM_GROW_MIN_SPAN_SEC >= 60,
           'мінімальне вікно менше за оновлення liq-map (60с) — це знову шум')
    _check(_m.MM_GROW_WINDOW_SEC >= 2 * _m.MM_GROW_MIN_SPAN_SEC,
           'вікно завузьке, щоб пережити разовий пропуск такту')
    _check(_m.MM_GROW_WINDOW_SEC > _m.CYCLE_SECS * 2,
           'вікно мусить накривати більше за два такти двигуна')
    print(f'✓ константи: вікно {_m.MM_GROW_WINDOW_SEC}с, мінімум '
          f'{_m.MM_GROW_MIN_SPAN_SEC}с, такт {_m.CYCLE_SECS}с')


def test_js_growth_cell_tells_the_two_empty_states_apart():
    """«Ще набираємо історію» і «історії немає» — РІЗНІ речі, і комірка мусить
    називати їх по-різному, а вікно брати з бекенда, а не зашивати своє."""
    out = _run_js(r'''
const R = (s, d, span) => ({symbol:s, mm:'LONG', strength:50, strength_prev:(d==null?null:45),
  delta:d, delta_rel:null, delta_span:span, grow_since:null, price:1,
  price_dir:'flat', price_chg:0, price_span:900,
  in_trade:false, in_queue:false, selectable:true});
mmApplyState({rows:[R('AAAUSDT',null,0), R('BBBUSDT',null,30), R('CCCUSDT',5,180)],
  enabled:true, limited:false, ts:1, counts:{LONG:3,SHORT:0,flat:0}});
const rows = document.getElementById('mm-tbody').innerHTML.split('</tr>');
const cell = s => rows.filter(x => x.includes(s))[0] || '';
console.log(JSON.stringify({
  none:/Історії ще немає/.test(cell('AAA')),
  warming:/Набираємо історію \(30с\)/.test(cell('BBB')),
  value:cell('CCC').includes('+5'), win:/за 3хв/.test(cell('CCC'))}));
''')
    import json
    d = json.loads(out)
    _check(d['none'] and d['warming'],
           f'два порожні стани не розрізняються: {d}')
    _check(d['value'] and d['win'],
           f'у підказці немає РЕАЛЬНОГО вікна, за яке виміряно приріст: {d}')
    print('✓ JS: «немає історії» ≠ «набираємо», вікно — з бекенда')


def test_delta_span_is_in_the_table_signature():
    """Інакше перехід «—» → число (вікно нарешті набралось) не перемалював би
    таблицю: самі `delta`/`strength` у цей момент могли не змінитись."""
    i = _HTML.index('const sig = _mmDir')
    sig = _HTML[i:i + 900]
    _check('r.delta_span' in sig, 'delta_span не входить у сигнатуру таблиці')
    print('✓ delta_span у сигнатурі — поява показника перемальовує рядок')



# ═══ 16. 🎛 ФІЛЬТРИ: ПАМʼЯТЬ + ЧЕСНІ ЛІЧИЛЬНИКИ (скарга 15.09) ════════════
def test_js_tab_counters_respect_the_strength_filter():
    """🐞 СКАРГА ЗІ СКРІНА: стояло «Сила ≥ 50», на вкладках світилось
    «🟢 LONG (36) · 🔴 SHORT (6)», а в таблиці — лічені рядки. Лічильник мусить
    рахувати ТЕ САМЕ, що видно під ним.
    ⚠️ Фільтр НАПРЯМКУ на лічильники НЕ впливає — інакше кожна необрана
    вкладка завжди показувала б (0) і перемкнутись було б неможливо."""
    out = _run_js(r'''
const R = (s, mm, st) => ({symbol:s, mm:mm, strength:st, strength_prev:st, delta:0,
  grow_since:null, price:1, price_dir:'flat', price_chg:0, price_span:900,
  delta_span:180, f1:null, f4:null, selectable:mm !== null});
const rows = [R('A','LONG',80), R('B','LONG',20), R('C','SHORT',70),
              R('D','SHORT',10), R('E',null,5)];
const N = () => _tabs.map(t => t.querySelector().textContent);
mmApplyState({rows:rows, enabled:true, limited:false, ts:1});
const before = N();
document.getElementById('mm-min-str').value = '50';
mmRender();
const after = N();
const shown = (document.getElementById('mm-tbody').innerHTML.match(/<tr/g)||[]).length;
console.log(JSON.stringify({before, after, shown}));
''')
    import json
    d = json.loads(out)
    _check(d['before'] == ['(5)', '(2)', '(2)', '(1)'],
           f'без фільтра лічильники мусять бути повними: {d["before"]}')
    _check(d['after'] == ['(2)', '(1)', '(1)', '(0)'],
           f'із «Сила ≥ 50» лічильники не звузились: {d["after"]}')
    _check(d['shown'] == 2,
           f'кількість рядків мусить збігатися з лічильником «Всі»: {d}')
    print('✓ JS: лічильники вкладок = те, що реально видно під ними')


def test_js_filters_survive_a_page_reload():
    """«Сила ≥ не запамʼятовується» — запамʼятовуємо ВСІ три UI-фільтри
    (поріг сили, вкладка напрямку, сортування) у localStorage."""
    out = _run_js(r'''
mmApplyState({rows:[], enabled:true, limited:false, ts:1});
document.getElementById('mm-min-str').value = '35';
mmMinStrChanged();
mmSetDir('SHORT');
mmSort('price');
const saved = JSON.parse(_LS[_MM_UI_KEY] || 'null');
// Імітуємо перезавантаження: скидаємо стан у дефолти і читаємо збережене.
_mmDir = 'all'; _mmSort = {col:'strength', dir:-1};
document.getElementById('mm-min-str').value = '0';
_mmLoadUi();
console.log(JSON.stringify({saved, dir:_mmDir, sort:_mmSort,
  minStr:document.getElementById('mm-min-str').value}));
''')
    import json
    d = json.loads(out)
    _check(d['saved'] and d['saved']['minStr'] == 35,
           f'поріг сили не збережено: {d["saved"]}')
    _check(str(d['minStr']) == '35', f'поріг не відновлено: {d}')
    _check(d['dir'] == 'SHORT', f'вкладку напрямку не відновлено: {d}')
    _check(d['sort']['col'] == 'price', f'сортування не відновлено: {d}')
    print('✓ JS: поріг сили, вкладка і сортування переживають перезавантаження')


def test_ui_table_is_striped_so_rows_do_not_blend():
    i = _HTML.index('id="mm-table"')
    blk = _HTML[max(0, i - 900):i]
    _check('#mm-table tbody tr:nth-child(even)' in blk,
           'немає зебри — рядки зливаються (скарга 15.09)')
    _check('#mm-table tbody tr:hover' in blk, 'немає підсвітки рядка під курсором')
    print('✓ UI: таблиця монітора — зебра + підсвітка рядка')


def test_js_price_cell_is_a_single_line():
    """Вимога 15.09: «Колонку Ціна пиши в один рядок»."""
    out = _run_js(r'''
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:50, strength_prev:50,
  delta:0, grow_since:null, price:1.5, price_dir:'up', price_chg:0.42,
  price_span:900, delta_span:180, f1:null, f4:null, selectable:true}],
  enabled:true, limited:false, ts:1});
const h = document.getElementById('mm-tbody').innerHTML;
const cell = h.split('<td')[6] || '';
console.log(JSON.stringify({br:cell.includes('<br>'), px:cell.includes('1.5'),
  arrow:cell.includes('▲'), pct:cell.includes('+0.42%'),
  win:cell.replace(/title="[^"]*"/g, '').includes('15хв')}));
''')
    import json
    d = json.loads(out)
    _check(not d['br'], f'ціна досі у два рядки: {d}')
    _check(d['px'] and d['arrow'] and d['pct'], f'ціна/напрямок/рух: {d}')
    _check(not d['win'], 'вікно спостереження мусить піти в підказку, не в рядок')
    print('✓ JS: колонка «Ціна» — один рядок (вікно — у підказці)')


# ═══ 17. 🟪 НОВИЙ VOB → АВТО-ВІДКРИТТЯ (вимога 15.09) ════════════════════
def _vob(ff, warm=False, **per_symbol):
    """Підмінити СКАНЕР: {'AAAUSDT': {'LONG': ft}} — який блок і в який бік
    віддає скан «📦 Volumized OB Trend». Заразом рахуємо виклики.

    ⚠️ Стабимо саме сканер, бо монітор БІЛЬШЕ не має власного детектора: блок
    мусить бути той самий, що на графіку. `warm` = чи є свіжий знімок скану
    (тоді читання безкоштовне й бюджет порції не витрачається).
    """
    ff.vob_calls = []

    ff.signals = []
    ff.gate = getattr(ff, 'gate', (True, '', 'розклад ✓'))

    class _SC:
        def volumized_tf(self):
            return '5m'

        def has_fresh_vob(self, sym):
            return warm

        def volumized_ob_side(self, sym, side, allow_fetch=True):
            ff.vob_calls.append((sym, side, '5m'))
            ft = (per_symbol.get(sym) or {}).get(side)
            return None if ft is None else {'formation_time': ft,
                                            'breaker': False,
                                            'top': 1.05, 'bottom': 0.95}

        def _signal_allowed(self, sym, side, at_intake=False, **kw):
            return ff.gate

    class _TM:
        def on_signal(self, symbol, side, entry_price, opened_by):
            ff.signals.append({'symbol': symbol, 'side': side,
                               'opened_by': opened_by, 'price': entry_price})
            return {'status': 'queued'}

    ff._mm_vob_scanner = lambda: _SC()
    ff._get_tm = lambda: _TM()


def test_new_vob_in_the_mm_direction_opens_a_trade():
    """Вимога дослівно: «автоматичне відкриття угоди, коли зʼявляється саме
    НОВИЙ VOB по монеті… VOB має співпадати з напрямком монети»."""
    _install_log()
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True, 'mm_vob_tf': '5m'})
    _vob(ff, AAAUSDT={'LONG': 1000})
    _cap(ff, AAAUSDT=0.40)                   # перший показ — ТИХА БАЗА
    _check(not ff.signals, f'перший показ не має давати сигнал: {ff.signals}')
    _cap(ff, AAAUSDT=0.45)
    _cap(ff, AAAUSDT=0.50)
    _vob(ff, AAAUSDT={'LONG': 2000})         # НОВИЙ блок
    _cap(ff, AAAUSDT=0.55)
    _check(len(ff.signals) == 1 and ff.signals[0]['side'] == 'LONG', ff.signals)
    _check(ff.vob_calls and ff.vob_calls[0][2] == '5m',
           f'TF мусить братись із налаштування: {ff.vob_calls}')
    _check(any('Volumized OB' in (x['detail'] or '') for x in _LOGGED),
           f'подія не пояснена в 🧾 Лозі: {_LOGGED}')
    print('✓ 🟪 новий VOD у бік МММ → угода відкрита')


def test_first_sight_is_a_silent_baseline():
    """⚠️ Після рестарту (а `botupdate` роблять часто) перший знайдений блок
    виглядав би «новим». Відкрити по ньому означало б торгувати блоком, якому
    може бути півдня — той самий урок, що у VOB-алерті."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'LONG': 999})
    _cap(ff, AAAUSDT=0.5)
    _cap(ff, AAAUSDT=0.5)                    # той самий блок — не новий
    _check(not ff.signals, f'той самий блок дав сигнал: {ff.signals}')
    print('✓ 🟪 перший показ — тиха база, той самий блок не відкриває')


def test_vob_against_the_mm_direction_is_ignored():
    """«VOB має співпадати з напрямком» — блок протилежного боку не береться
    ВЗАГАЛІ: детектор питається саме про бік МММ."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'SHORT': 1000})        # є лише ВЕДМЕЖИЙ блок
    _cap(ff, AAAUSDT=0.5)                    # а МММ каже LONG
    _vob(ff, AAAUSDT={'SHORT': 2000})
    _cap(ff, AAAUSDT=0.5)
    _check(not ff.signals, f'сигнал проти напрямку МММ: {ff.signals}')
    _check(all(c[1] == 'LONG' for c in ff.vob_calls),
           f'детектор питали не в бік МММ: {ff.vob_calls}')
    print('✓ 🟪 блок проти напрямку МММ ігнорується')


def test_flat_and_in_trade_coins_are_not_candidates():
    """Кандидати — РІВНО ті монети, що видно в таблиці: ⚖ рівновага відкривати
    нічого не може, а монета в угоді в таблиці й не показується."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    ff._fuel_managed = {'BBBUSDT': {}}
    _vob(ff, AAAUSDT={'LONG': 1}, BBBUSDT={'LONG': 1}, CCCUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.5, BBBUSDT=0.5, CCCUSDT=0.02)   # CCC — рівновага
    asked = {c[0] for c in ff.vob_calls}
    _check(asked == {'AAAUSDT'}, f'кандидати відібрані невірно: {asked}')
    print('✓ 🟪 кандидати = рядки таблиці (без ⚖ і без монет в угоді)')


def test_cold_coins_are_capped_per_tick():
    """⚠️ Бюджет порції витрачають ЛИШЕ монети, по яких доводиться РАХУВАТИ
    самим (3000 барів). Без цього повний watchlist дав би сотні запитів."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    n = _m.MM_VOB_MAX_PER_TICK
    pairs = {f'C{i:03d}USDT': 0.5 for i in range(n * 2)}
    _vob(ff, **{k: {'LONG': 1} for k in pairs})
    _cap(ff, **pairs)
    _check(len(ff.vob_calls) == n, f'порція не обмежена: {len(ff.vob_calls)}')
    first = {c[0] for c in ff.vob_calls}
    _vob(ff, **{k: {'LONG': 1} for k in pairs})
    _cap(ff, **pairs)
    _check(not ({c[0] for c in ff.vob_calls} & first),
           'другий такт перевіряє ТІ САМІ монети — черга не рухається')
    print(f'✓ 🟪 не більше {n} ХОЛОДНИХ монет за такт, черга рухається')


def test_warm_coins_cost_nothing_and_are_all_served():
    """Скан уже поклав знімок → читання безкоштовне, тож обмежувати нема чого.
    Раніше бюджет їли ВСІ підряд і повне коло по 230 монетах тривало ~8 хв."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    n = _m.MM_VOB_MAX_PER_TICK
    pairs = {f'C{i:03d}USDT': 0.5 for i in range(n * 3)}
    _vob(ff, warm=True, **{k: {'LONG': 1} for k in pairs})
    _cap(ff, **pairs)
    _check(len(ff.vob_calls) == len(pairs),
           f'теплі монети мусять оброблятись усі за такт: {len(ff.vob_calls)}')
    print(f'✓ 🟪 теплі монети (знімок скану) — усі {len(pairs)} за ОДИН такт')


def test_toggle_off_means_no_network_at_all():
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': False})
    _vob(ff, AAAUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.5)
    _check(not ff.vob_calls, f'вимкнений тумблер усе одно ходив по свічки: {ff.vob_calls}')
    _check(_m.DEFAULT_SETTINGS['mm_vob_open'] is True, 'дефолт мав бути УВІМК')
    _check('mm_vob_tf' not in _m.DEFAULT_SETTINGS,
           'власний TF монітора мав зникнути — його диктує блок Volumized')
    print('✓ 🟪 тумблер OFF → жодного запиту; власного TF немає')


def test_monitor_toggle_off_stops_vob_opening_too():
    """Вимкнений монітор = таблиці немає. Відкривати по невидимому списку не
    можна, хоча знімок для колонки в угодах лишається."""
    ff = _mk(mon=False)
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    ff._fuel_managed = {'AAAUSDT': {}}
    _vob(ff, AAAUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.5)
    _check(not ff.vob_calls and not ff.signals,
           f'вимкнений монітор усе одно сигналив: {ff.vob_calls}')
    print('✓ 🟪 вимкнений монітор не відкриває по VOB')


def test_vob_goes_out_as_a_signal_through_the_shared_gate():
    """Вимога 16.09: «не відкриває угоду, а передає монету у вигляді сигналу
    далі по алгоритму». Тобто СПІЛЬНІ ВОРОТА сканера → `tm.on_signal`, а вже
    черги/двигун вирішують, куди монета потрапить."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.50)
    _vob(ff, AAAUSDT={'LONG': 2})
    _cap(ff, AAAUSDT=0.50)
    _check(not ff.opened, f'монітор НЕ має відкривати сам: {ff.opened}')
    _check(len(ff.signals) == 1, f'сигнал не пішов: {ff.signals}')
    _check(ff.signals[0]['opened_by'] == 'mm_vob',
           f'мітка сигналу мусить називати монітор: {ff.signals[0]}')
    _check(ff.signals[0]['price'] > 0, 'сигнал без ціни входу')
    print(f"✓ 🧮 VOB → сигнал {ff.signals[0]['opened_by']!r} (не пряме відкриття)")


def test_blocked_by_the_scanner_filters_means_no_signal():
    """⚠️ Ворота сканера обходити НЕ можна (урок ASTERUSDT). Не пропустили —
    сигналу немає, а причина названа в 🧾 Лозі."""
    _install_log()
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    ff.gate = (False, 'OB-фільтр заблокував', 'OB(1h BEARISH):✗')
    _vob(ff, AAAUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.50)
    _vob(ff, AAAUSDT={'LONG': 2})
    _cap(ff, AAAUSDT=0.50)
    _check(not ff.signals and not ff.opened, f'сигнал пішов повз ворота: {ff.signals}')
    _check(any(x['event'] == 'rejected' and 'OB-фільтр' in (x['detail'] or '')
               for x in _LOGGED), f'причину відмови не названо: {_LOGGED}')
    print('✓ 🧮 фільтри сканера ріжуть сигнал монітора, як будь-який інший')


def test_every_signal_is_written_to_the_log():
    """«Де сигнали?» — шлях більше НЕ мовчить: і сам сигнал, і розклад фільтрів
    ідуть у 🧾 Лог із джерелом MMM."""
    _install_log()
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.50)
    _vob(ff, AAAUSDT={'LONG': 2})
    _cap(ff, AAAUSDT=0.50)
    _sig = [x for x in _LOGGED if x['event'] == 'signal' and x['source'] == 'MMM']
    _check(len(_sig) == 1, f'сигнал не потрапив у лог: {_LOGGED}')
    _check('розклад' in (_sig[0]['detail'] or ''),
           f'розклад фільтрів не доїхав у рядок: {_sig[0]}')
    print('✓ 🧮 кожен сигнал монітора видно в 🧾 Лозі (з розкладом фільтрів)')


def test_baseline_survives_a_one_tick_dropout():
    """⚠️ КОРІНЬ «сигналів немає»: монета, що випала зі знімка на ОДИН такт,
    поверталась із чистою базою і ковтала наступний РЕАЛЬНО новий блок."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'LONG': 1}, BBBUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.50, BBBUSDT=0.50)
    _cap(ff, BBBUSDT=0.50)                    # AAA зникла на такт
    _check('AAAUSDT' in ff._mm_vob_seen, 'базу стерто на першому ж пропуску')
    _vob(ff, AAAUSDT={'LONG': 2}, BBBUSDT={'LONG': 1})
    _cap(ff, AAAUSDT=0.50, BBBUSDT=0.50)
    _check(any(x['symbol'] == 'AAAUSDT' for x in ff.signals),
           f'новий блок після пропуску не дав сигналу: {ff.signals}')
    print('✓ 🧮 разовий пропуск монети більше не ковтає наступний блок')


# ── ⚑ «СКІЛЬКИ У ФІЛЬТРІ, А СКІЛЬКИ ПОЗА» (вимога 16.09) ─────────────────
# «Коли увімкнений фільтр "Сила ≥" — незрозуміло що з іншою кількістю монет».
# «Всі (6)» без другого числа читається як «у боті всього 6 монет».

def test_js_says_how_many_the_strength_filter_left_out():
    out = _run_js(r'''
const R = (s, str) => ({symbol:s, mm:'LONG', strength:str, strength_prev:str,
  delta:0, grow_since:null, price:1, price_dir:'flat', price_chg:0,
  price_span:900, delta_span:180, f1:null, f4:null, selectable:true});
const rows = [];
for (let i = 0; i < 6; i++) rows.push(R('A' + i + 'USDT', 70));   // у фільтрі
for (let i = 0; i < 17; i++) rows.push(R('B' + i + 'USDT', 10));  // поза
document.getElementById('mm-min-str').value = '50';
mmApplyState({rows: rows, enabled:true, limited:false, ts:1});
const badge = document.getElementById('mm-str-out');
const tab = document.querySelector('#mm-body [data-mmdir="all"]');
console.log(JSON.stringify({
  txt: badge.textContent, tip: badge.title || '',
  tabN: (tab.querySelector('.mm-dir-n') || {}).textContent,
  tabTip: tab.title || ''}));
''')
    import json
    d = json.loads(out)
    _check('17' in d['txt'] and '23' in d['txt'],
           f'не видно, скільки монет поза фільтром: {d["txt"]!r}')
    _check(d['tabN'] == '(6)', f'вкладка мусить рахувати ВИДИМІ: {d["tabN"]!r}')
    _check('поза фільтром 17' in d['tabTip'],
           f'у підказці вкладки немає розкладу: {d["tabTip"]!r}')
    print(f'✓ ⚑ «{d["txt"]}» + розклад у підказці вкладки')


def test_js_badge_is_silent_when_nothing_is_filtered_out():
    """Нічого не відсіяно → напису НЕМАЄ: зайвий «поза фільтром 0» — шум."""
    out = _run_js(r'''
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:70, strength_prev:70,
  delta:0, grow_since:null, price:1, price_dir:'flat', price_chg:0,
  price_span:900, delta_span:180, f1:null, f4:null, selectable:true}],
  enabled:true, limited:false, ts:1});
const b = document.getElementById('mm-str-out');
console.log(JSON.stringify({txt:b.textContent, tip:b.title || ''}));
''')
    import json
    d = json.loads(out)
    _check(not d['txt'] and not d['tip'], f'напис лишився без потреби: {d}')
    print('✓ ⚑ фільтр нічого не відсік → напису немає')


# ── 🟦 ДЖЕРЕЛО БЛОКУ = СКАН «📦 Volumized OB Trend» (вимога 16.09) ────────
# «Використовуй VOB алгоритм, що на скріні… має бути задіяний цей скан VOB».
# Монітор мусить брати РІВНО той блок, що намальовано на графіку, — тобто з
# параметрами користувача (TF · Swing · Zone Invalidation · ATR · Zone Count ·
# Combine), а не з власних констант funding-стратегії.

def _src(path, fn):
    """Тіло функції `fn` із файлу `path` — БЕЗ докстрінга (для тест-замків).

    ⚠️ Докстрінг ріжемо обовʼязково: він ПОЯСНЮЄ, чому власного детектора тут
    більше немає, і згадує `_funding_vob` — інакше замок падав би на власному
    коментарі (та сама пастка, що вже ловила `ensure_fresh`)."""
    code = open(os.path.join(_HERE, path), encoding='utf-8').read()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            body = node.body[1:] if ast.get_docstring(node) else node.body
            return '\n'.join(ast.get_source_segment(code, n) or '' for n in body)
    raise AssertionError(f'{fn} не знайдено у {path}')


def test_monitor_has_no_detector_of_its_own():
    body = _src('detection/fuel_filter.py', '_mm_vob_tick')
    _check('volumized_ob_side' in body,
           'монітор не питає блок у скану «Volumized OB Trend»')
    for bad in ('_funding_vob', 'detect_volumized_obs', 'fetch_klines'):
        _check(bad not in body,
               f'у монітор повернувся власний детектор ({bad}) — блок розійдеться '
               'із тим, що на графіку')
    print('✓ 🟦 монітор лише ЧИТАЄ блок скану, свого детектора не має')


def test_scanner_side_lookup_uses_the_user_settings_and_full_window():
    body = _src('detection/smc_scanner.py', 'volumized_ob_side')
    for key in ('volumized_timeframe', 'volumized_swing_length',
                'volumized_ob_end_method', 'volumized_max_atr_mult',
                'volumized_zone_count', 'volumized_combine_obs'):
        _check(key in body or key == 'volumized_timeframe',
               f'{key} не береться з налаштувань користувача')
    _check('VOB_KLINES_LIMIT' in body and 'limit=200' not in body,
           'глибина барів мусить бути та сама, що у графіка (3000), а не 200')
    _check('use_volumized_ob' in body,
           'вимкнений Enable мусить означати «блоку немає»')
    print('✓ 🟦 сканер віддає блок за налаштуваннями користувача на 3000 барах')


def test_scan_warms_the_side_cache_for_free():
    """Знімок кладе САМ скан із уже порахованого `vol_result` — тож у робочому
    стані монітор не робить ЖОДНОГО запиту."""
    code = open(os.path.join(_HERE, 'detection/smc_scanner.py'), encoding='utf-8').read()
    i = code.index('vol_result = get_latest_ob_trend')
    _check('_vob_sides_put(symbol, vol_result' in code[i:i + 2500],
           'скан не наповнює кеш блоків за напрямком')
    print('✓ 🟦 кеш наповнює скан — читання для монітора безкоштовне')


# ═══ 18. 📈 ОЧІКУВАННЯ РОСТУ СИЛИ — СКАСОВАНО КОРИСТУВАЧЕМ (16.09) ══════
# Був гейт «новий VOB є, але сила МММ НЕ росте — відкриття ВІДКЛАДЕНО» (черга
# `_mm_vob_pending`, поле `vob_wait`, ⏳ у таблиці). Користувач СКАСУВАВ його
# дослівно: «відміни цю перевірку». Тести нижче — ЗАМКИ, щоб гейт не повернувся
# тихо: ріст сили лишається ПОКАЗНИКОМ, а не умовою входу.


def test_new_vob_opens_even_when_strength_is_not_growing():
    """Плато сили — і все одно відкриваємо: це і є скасування перевірки."""
    _install_log()
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'LONG': 1000})
    for _ in range(3):
        _cap(ff, AAAUSDT=0.50)               # база + плато → росту НЕМАЄ
    _check(not ff._mm_grow_since.get('AAAUSDT'), 'фікстура: росту бути не мало')
    _vob(ff, AAAUSDT={'LONG': 2000})         # НОВИЙ блок
    _cap(ff, AAAUSDT=0.50)
    _check(len(ff.signals) == 1 and ff.signals[0]['side'] == 'LONG',
           f'сигналу немає, хоча перевірку скасовано: {ff.signals}')
    _check(not any('НЕ росте' in (x['detail'] or '') for x in _LOGGED),
           f'у 🧾 Лозі лишилось «відкладено через ріст»: {_LOGGED}')
    print('✓ 📈 перевірку скасовано: плато сили більше не відкладає вхід')


def test_falling_strength_does_not_delay_the_entry_either():
    """Не лише плато: навіть спад сили більше нічого не відкладає."""
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'LONG': 1000})
    _cap(ff, AAAUSDT=0.70)
    _cap(ff, AAAUSDT=0.60)
    _cap(ff, AAAUSDT=0.50)                   # сила ПАДАЄ
    _vob(ff, AAAUSDT={'LONG': 2000})
    _cap(ff, AAAUSDT=0.40)
    _check(len(ff.signals) == 1, f'спад сили відклав сигнал: {ff.signals}')
    print('✓ 📈 спад сили теж не блокує — умова входу знята повністю')


def test_no_waiting_state_is_produced_anywhere():
    """Разом із гейтом пішли і його сліди: черга очікування, поле рядка і
    згадка `_mm_grow_since` у самому шляху відкриття."""
    import inspect
    ff = _mk()
    ff._settings.update({'enabled': True, 'mm_vob_open': True})
    _vob(ff, AAAUSDT={'LONG': 1000})
    for _ in range(3):
        _cap(ff, AAAUSDT=0.50)
    _vob(ff, AAAUSDT={'LONG': 2000})
    _cap(ff, AAAUSDT=0.50)
    _check(not hasattr(ff, '_mm_vob_pending'),
           'черга очікування `_mm_vob_pending` лишилась у стані')
    rows = ff.mm_monitor_state()['rows']
    _check(all('vob_wait' not in r for r in rows),
           f'поле `vob_wait` ще їде в рядок: {rows}')
    src = inspect.getsource(_m.FuelFilterDaemon._mm_vob_tick)
    _check('_mm_grow_since' not in src,
           'шлях відкриття знову дивиться на таймер росту')
    print('✓ 📈 слідів гейта не лишилось (черга · поле рядка · код)')


def test_js_no_longer_draws_the_waiting_badge():
    """⏳ біля назви монети більше немає — затримки, яку він пояснював, теж."""
    _check('_mmWaitBadge' not in _HTML, 'функція ⏳-бейджа лишилась у сторінці')
    _check('vob_wait' not in _HTML, 'сторінка ще читає `vob_wait`')
    out = _run_js(r'''
mmApplyState({rows:[{symbol:'AAAUSDT', mm:'LONG', strength:50, strength_prev:50,
  delta:0, grow_since:null, price:1, price_dir:'flat', price_chg:0,
  price_span:900, delta_span:180, f1:null, f4:null, selectable:true}],
  enabled:true, limited:false, ts:1});
const h = document.getElementById('mm-tbody').innerHTML;
console.log(JSON.stringify({rows:h.includes('AAAUSDT'), hourglass:h.includes('⏳')}));
''')
    import json
    d = json.loads(out)
    _check(d['rows'], 'рядок монітора взагалі не намалювався')
    _check(not d['hourglass'], 'у рядку досі малюється ⏳')
    print('✓ JS: ⏳ прибрано, рядок малюється без нього')


def test_ui_has_the_vob_controls_wired_both_ways():
    _check('id="ff-mm-vob-open"' in _HTML, 'немає тумблера VOB→угода')
    _check('mm_vob_open:' in _HTML, 'mm_vob_open не йде у збереження налаштувань')
    _check('s.mm_vob_open' in _HTML, 'mm_vob_open не відновлюється з налаштувань')
    # ⚠️ ВЛАСНОГО TF у монітора БІЛЬШЕ НЕМАЄ — його диктує блок «📦 Volumized
    # OB Trend». Мертвий контрол тут гірший за його відсутність: він означав би
    # другий набір параметрів і другий «останній блок».
    _check('ff-mm-vob-tf' not in _HTML, 'мертва випадайка TF лишилась у сторінці')
    _check('mm_vob_tf' not in _HTML, 'сторінка досі шле/читає mm_vob_tf')
    # …але звідки беруться параметри, мусить бути СКАЗАНО прямо.
    _check('ff-mm-vob-src' in _HTML and 'Volumized OB Trend' in _HTML,
           'не видно, що параметри беруться зі скану Volumized OB Trend')
    print('✓ UI: 🟪 тумблер + підпис «параметри з 📦 Volumized OB Trend»')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as e:
            bad += 1
            print(f'  FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - bad}/{len(fns)} passed')
    sys.exit(1 if bad else 0)
