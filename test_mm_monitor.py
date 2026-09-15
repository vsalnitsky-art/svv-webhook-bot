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


def _mk(limited=False, enabled=True):
    """Мінімальний демон: лише те, чого торкається монітор."""
    import threading
    ff = FF.__new__(FF)
    ff._lock = threading.RLock()
    ff._mm_snapshot = {}
    ff._mm_prev = {}
    ff._mm_snapshot_ts = 0.0
    ff._fuel_managed = {}
    ff._pending, ff._pending2, ff._pending3, ff._pending4 = {}, {}, {}, {}
    ff._timers = {}
    ff._get_tm = None
    ff._settings = {'enabled': enabled, 'mmm_limited_mode': limited}
    ff.get_settings = lambda: dict(ff._settings)
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
    """{'BTCUSDT': 0.42} → форма, яку віддає `_fuel_dir_smoothed`."""
    out = {}
    for sym, d in pairs.items():
        st = 'LONG' if d > 0.15 else ('SHORT' if d < -0.15 else None)
        out[sym] = {'dir': d, 'status': st, 'mark_price': 100.0 + abs(d)}
    return out


# ═══════════ 1. ДЖЕРЕЛО ЧИСЕЛ — ЗНІМОК ДВИГУНА ═══════════════════════════
def test_monitor_reads_the_engine_snapshot_and_computes_nothing():
    """⚠️ ГОЛОВНЕ. Монітор не має права рахувати МММ сам: інакше в таблиці
    стояло б одне число, а рішення двигун ухвалював би за іншим (той самий
    урок, що з шарами Черги-4 — «двигун рахує, get_state читає»)."""
    ff = _mk()
    ff._mm_capture(_fuels(BTCUSDT=0.42, ETHUSDT=-0.31, XRPUSDT=0.02))
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


def test_strength_trend_comes_from_the_previous_tick():
    """Стрілку ↑/↓ малює спільний віджет `ffFuelCell`, тож віддаємо ПОПЕРЕДНЮ
    силу, а не власний висновок «up/down» — друге правило тренду розійшлося б
    із першим."""
    ff = _mk()
    ff._mm_capture(_fuels(BTCUSDT=0.20))
    ff._mm_capture(_fuels(BTCUSDT=0.45))
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['strength'] == 45 and r['strength_prev'] == 20, r)
    _check('trend' not in r, 'готового «up/down» у рядку бути не має')
    print('✓ тренд сили: віддаємо попереднє число, стрілку малює спільний віджет')


def test_rows_are_sorted_by_strength_then_symbol():
    """Найвиразніший напрямок зверху; тайбрейк за символом — щоб рядки не
    «стрибали» під курсором між поллами."""
    ff = _mk()
    ff._mm_capture(_fuels(AAAUSDT=0.30, BBBUSDT=-0.90, CCCUSDT=0.30))
    got = [r['symbol'] for r in ff.mm_monitor_state()['rows']]
    _check(got == ['BBBUSDT', 'AAAUSDT', 'CCCUSDT'], got)
    print('✓ сортування: сила ↓, далі символ (стабільний порядок)')


def test_groups_are_countable_for_the_long_short_tabs():
    """«Сортувати окремо на LONG чи SHORT» — вкладки рахують ПОВНИЙ знімок."""
    ff = _mk()
    ff._mm_capture(_fuels(A=0.5, B=0.4, C=-0.6, D=0.01))
    c = ff.mm_monitor_state()['counts']
    _check(c == {'LONG': 2, 'SHORT': 1, 'flat': 1}, c)
    print('✓ лічильники вкладок: LONG / SHORT / ⚖ рівновага')


# ═══════════ 2. ПРИДАТНІСТЬ ДО ВИБОРУ ════════════════════════════════════
def test_a_coin_already_in_a_trade_cannot_be_selected():
    ff = _mk()
    ff._mm_capture(_fuels(BTCUSDT=0.5, ETHUSDT=0.5))
    ff._fuel_managed = {'BTCUSDT': {}}
    by = {r['symbol']: r for r in ff.mm_monitor_state()['rows']}
    _check(by['BTCUSDT']['in_trade'] and not by['BTCUSDT']['selectable'],
           by['BTCUSDT'])
    _check(by['ETHUSDT']['selectable'], by['ETHUSDT'])
    print('✓ монета в угоді: помічена і НЕ обирається')


def test_balanced_mm_is_not_selectable():
    """⚖ рівновага — напрямку немає, відкривати нічого."""
    ff = _mk()
    ff._mm_capture(_fuels(XRPUSDT=0.02))
    r = ff.mm_monitor_state()['rows'][0]
    _check(not r['selectable'], r)
    print('✓ ⚖ рівновага не обирається (немає чого відкривати)')


def test_queue_membership_is_shown_but_does_not_block():
    """Монета може стояти в черзі — це ІНФОРМАЦІЯ, а не заборона: ✋ ручне
    відкриття свідомо проходить повз ворота черг."""
    ff = _mk()
    ff._mm_capture(_fuels(BTCUSDT=0.5))
    ff._pending4 = {'BTCUSDT': {'dir': 'LONG'}}
    r = ff.mm_monitor_state()['rows'][0]
    _check(r['in_queue'] and r['selectable'], r)
    print('✓ «у черзі» показано, але вибору не блокує')


# ═══════════ 3. ГРУПОВЕ ВІДКРИТТЯ ════════════════════════════════════════
def test_group_open_uses_each_coin_own_mm_direction():
    """Напрямок КОЖНОЇ монети — її власний МММ зі знімка, тобто РІВНО те, що
    людина бачила в рядку."""
    _install_log()
    ff = _mk()
    ff._mm_capture(_fuels(BTCUSDT=0.5, ETHUSDT=-0.5))
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
    ff._mm_capture(_fuels(BTCUSDT=0.5))
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
    ff._mm_capture(_fuels(BTCUSDT=0.5, ETHUSDT=0.02, LTCUSDT=0.4))
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
    ff._mm_capture(_fuels(BTCUSDT=0.5))
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
    ff._mm_capture(_fuels(BTCUSDT=0.5))
    res = ff.group_open(['BTCUSDT', 'btcusdt', ' BTCUSDT '])
    _check(len(ff.opened) == 1, f'той самий символ двічі: {ff.opened}')
    _check(res['opened'] == 1, res)
    for bad in ([], None, ['   ']):
        r = ff.group_open(bad)
        _check(not r['ok'] and r['opened'] == 0, f'{bad!r} → {r}')
    print('✓ дублікати згортаються, порожній вибір нічого не відкриває')


def test_group_open_respects_the_master_switch():
    ff = _mk(enabled=False)
    ff._mm_capture(_fuels(BTCUSDT=0.5))
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
    _check(st['counts'] == {'LONG': 0, 'SHORT': 0, 'flat': 0}, st)
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
    i = _HTML.index('function mmRender()')
    fn = _HTML[i:i + 4000]
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
    i = _HTML.index('function mmRender()')
    _check('ffFuelCell(r.mm, r.strength, r.strength_prev)' in _HTML[i:i + 4000],
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
const _tabs = ['all','LONG','SHORT','flat'].map(d => ({
  dataset:{mmdir:d}, style:{}, querySelector:()=>({textContent:''})}));
const document = {
  getElementById: id => (['mm-tbody','mm-check-all','mm-min-str','mm-sel-count',
                          'mm-open-btn','mm-updated','mm-limited-hint'].includes(id)
                         ? _el(id) : null),
  querySelectorAll: () => _tabs,
};
const fetch = async () => ({ok:true, json: async () => ({})});
const confirm = () => false;
const loadFuelFilterStatus = () => {};
const ffFuelCell = () => '<mm/>';
const fmtPriceJS = p => String(p);
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
