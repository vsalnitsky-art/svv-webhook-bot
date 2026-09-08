"""🚦 ГОЛОВНІ КНОПКИ НАПРЯМКУ (Trade Direction) — ЗАМКИ.

Вимога користувача дослівно: «Це головні кнопки бота. Тобто якщо вони вимкнені,
це має означати, що бот взагалі не сканує і не відправляє сигнали, відповідно,
якщо увімкнені обидві кнопки, бот працює у всіх напрямках LONG і SHORT. Бо
працює в тому напрямку, в якому є ввімкненою кнопка.»

**Аудит 08.09 показав, що кнопки не діяли НІДЕ на робочій установці:**
  • сканер про них не знав узагалі (жодної згадки ключів у файлі);
  • `intercept` пускав у чергу БУДЬ-ЯКИЙ напрямок;
  • двигун Черги-4 свідомо їх не дивився («приймаємо всі сигнали»);
  • у TM перевірка стояла під `if not bypass_gates`, а FF відкриває саме з
    `bypass_gates=True` — тобто КОЖНА угода з черги проходила повз вимикач.

Ці тести стережуть УСІ чотири вузли. Найважливіші два — `test_bypass_gates_
no_longer_defeats_the_master_switch` (корінь дефекту) і
`test_single_source_of_keys` (щоб ключі знову не розповзлись по файлах).
"""
import ast
import os
import sys
import types
import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection')
    _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
    sys.modules['detection'] = _pkg

DG = _load('detection.direction_gate', 'detection/direction_gate.py')
sys.modules['detection'].direction_gate = DG


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _src(rel):
    with open(os.path.join(_ROOT, rel), encoding='utf-8') as f:
        return f.read()


BOTH = {'allow_long_entries': True, 'allow_short_entries': True}
ONLY_SHORT = {'allow_long_entries': False, 'allow_short_entries': True}
ONLY_LONG = {'allow_long_entries': True, 'allow_short_entries': False}
OFF = {'allow_long_entries': False, 'allow_short_entries': False}


# ═════════ 1. ЧИСТА ЛОГІКА ВОРІТ ═══════════════════════════════════════════
def test_works_in_the_direction_whose_button_is_on():
    """Дослівне правило користувача: «працює в тому напрямку, в якому є
    ввімкненою кнопка»."""
    _check(DG.allows(BOTH, 'LONG') and DG.allows(BOTH, 'SHORT'),
           'обидві увімкнені → працюють ОБА напрямки')
    _check(DG.allows(ONLY_SHORT, 'SHORT') and not DG.allows(ONLY_SHORT, 'LONG'),
           'лише SHORT → LONG заблоковано')
    _check(DG.allows(ONLY_LONG, 'LONG') and not DG.allows(ONLY_LONG, 'SHORT'),
           'лише LONG → SHORT заблоковано')
    _check(not DG.allows(OFF, 'LONG') and not DG.allows(OFF, 'SHORT'),
           'обидві вимкнені → НІЧОГО не працює')
    print('✓ бот працює рівно в тому напрямку, чия кнопка увімкнена')


def test_both_off_is_a_full_stop():
    _check(DG.both_off(OFF), 'обидві вимкнені = ПАУЗА')
    for s in (BOTH, ONLY_LONG, ONLY_SHORT):
        _check(not DG.both_off(s), f'не пауза: {s}')
    _check('жодних сигналів' in DG.reason(OFF, 'LONG'),
           f'причина мусить казати про повну паузу: {DG.reason(OFF, "LONG")}')
    _check(DG.state_label(OFF).startswith('ПАУЗА'), DG.state_label(OFF))
    print('✓ обидві вимкнені = повна пауза, і причина це прямо каже')


def test_missing_keys_default_to_open():
    """Стара БД без цих ключів НЕ має мовчки зупинити бота."""
    _check(DG.allows({}, 'LONG') and DG.allows({}, 'SHORT'), 'дефолт — дозволено')
    _check(DG.allows(None, 'SHORT'), 'None → дозволено')
    print('✓ немає ключів → дозволено (стара БД не зупиняє бота)')


def test_unknown_side_is_not_blocked():
    """Ворота — про НАПРЯМОК. Вигадувати відмову для чогось іншого не можна."""
    _check(DG.allows(OFF, ''), 'порожній бік — не наша справа')
    _check(DG.allows(OFF, 'FLAT'), 'невідомий бік — не наша справа')
    _check(DG.reason(OFF, 'FLAT') == '', 'і причини бути не має')
    print('✓ невідомий бік воротами не блокується')


def test_chip_is_silent_when_the_switch_does_nothing():
    """Розклад у 🧾 Лозі не має рости на рівному місці: коли обидві кнопки
    увімкнені, головний вимикач нічого не робить → сегмента немає."""
    _check(DG.chip(BOTH, 'LONG') == '', f'мусить бути порожньо: {DG.chip(BOTH, "LONG")!r}')
    _check(DG.chip(ONLY_SHORT, 'LONG') == 'Напрямок[LONG]:✗', DG.chip(ONLY_SHORT, 'LONG'))
    _check(DG.chip(ONLY_SHORT, 'SHORT') == 'Напрямок[SHORT]:✓', DG.chip(ONLY_SHORT, 'SHORT'))
    print('✓ сегмент розкладу з\'являється, лише коли вимикач реально діє')


def test_fail_open_when_state_is_unknown():
    """Ворота вміють зупинити ТОРГІВЛЮ, тож збій читання не сміє стати тихою
    зупинкою — краще пропустити, ніж мовчки стати."""
    _orig = DG.live_settings
    try:
        DG.live_settings = lambda: None
        _check(DG.live_gates() == (True, True), 'немає TM → (True, True)')
        _check(DG.live_allows('LONG') and DG.live_allows('SHORT'), 'fail-open')
        _check(DG.live_reason('LONG') == '', 'без стану — без вигаданої причини')
    finally:
        DG.live_settings = _orig
    print('✓ стан невідомий → fail-open, без тихої зупинки')


# ═════════ 2. КОРІНЬ ДЕФЕКТУ: bypass_gates ═════════════════════════════════
def test_bypass_gates_no_longer_defeats_the_master_switch():
    """ГОЛОВНИЙ ЗАМОК. У `_open_position` і `_open_shadow` перевірка кнопок
    стояла під `if not bypass_gates`, а Fuel Filter відкриває саме з
    `bypass_gates=True` — тобто кнопки не діяли на ЖОДНУ угоду з черг."""
    src = _src('detection/trade_manager.py')
    bad = 'not bypass_gates and not self._side_allowed'
    _check(bad not in src,
           f'`{bad}` повертає баг: FF відкриває з bypass_gates=True '
           'і кнопки знову стануть декоративними')
    _check(src.count('if not self._side_allowed(side):') == 2,
           'перевірка мусить стояти в ОБОХ шляхах: _open_position і _open_shadow')
    print('✓ bypass_gates більше не знімає головні кнопки (real + paper)')


def test_bypass_gates_still_governs_the_other_gates():
    """⚠️ Прибрати `bypass_gates` ЦІЛКОМ було б іншим багом: він законно знімає
    ІНШІ ворота (напр. FF-підтвердження). Знято лише з головних кнопок."""
    src = _src('detection/trade_manager.py')
    _check("not bypass_gates and self._settings.get('require_fuel_confirm'" in src,
           'FF-підтвердження мусить і далі поважати bypass_gates')
    print('✓ bypass_gates лишився чинним для решти воріт')


def test_ff_opens_with_bypass_gates_so_the_gate_must_live_in_open():
    """Фіксуємо ФАКТ, через який ворота потрібні і в `fuel_filter._open`:
    FF свідомо кличе TM з bypass_gates=True."""
    src = _src('detection/fuel_filter.py')
    _check('bypass_gates=True' in src,
           'якщо FF перестане обходити ворота — цей тест переглянути')
    print('✓ FF і далі кличе TM з bypass_gates=True — ворота в _open потрібні')


# ═════════ 3. ВОРОТА СТОЯТЬ НА ВСІХ ЧОТИРЬОХ ВУЗЛАХ ════════════════════════
def test_scanner_gate_is_first_in_the_shared_chain():
    """Сканер про кнопки не знав ВЗАГАЛІ. Тепер вони — найперша перевірка в
    `_signal_allowed`, тобто ДО OB/PD/Forecast/Decision/POC/Ліквідності."""
    src = _src('detection/smc_scanner.py')
    i = src.index('def _signal_allowed')
    body = src[i:src.index('\n    def ', i + 10)]
    # Докстрінг перелічує назви фільтрів — шукаємо в КОДІ, не в тексті.
    body = body[body.index('parts = []'):]
    _check('_dg_mod()' in body, 'сканер мусить питати ворота напрямку')
    gate = body.index('_dg.allows(')
    for nxt in ("ob_filter_enabled", "use_pd_zone_filter", "poc_filter_enabled",
                "liq_filter_enabled"):
        if nxt in body:
            _check(gate < body.index(nxt),
                   f'ворота напрямку мусять стояти ДО фільтра {nxt}')
    print('✓ сканер: ворота напрямку — найперші у спільному ланцюгу')


def test_scanner_does_not_drag_the_whole_package_into_the_hot_path():
    """⚠️ Урок, на якому впали два ЧУЖІ тести. Будь-який імпорт підмодуля
    виконує `detection/__init__.py`, а той тягне `sleeper_scanner → core →
    pybit`. Сканер навмисно вміє вантажитись САМОСТІЙНО (так його беруть
    ізольовані тести), тож пакетний імпорт у `_signal_allowed` зламав
    `test_ob_choch_only.py` і `test_signal_gate_unified.py` на рівному місці."""
    src = _src('detection/smc_scanner.py')
    i = src.index('def _signal_allowed')
    body = src[i:src.index('\n    def ', i + 10)]
    _check('from detection import direction_gate' not in body,
           'у гарячому шляху — лише `_dg_mod()`, без пакетного імпорту')
    _check('def _dg_mod()' in src, 'потрібен завантажувач із фолбеком на файл')
    j = src.index('def _dg_mod()')
    loader = src[j:j + 1200]
    _check('spec_from_file_location' in loader,
           'фолбек мусить вантажити сусідній файл напряму')
    _check('_DG_CACHE' in loader, 'модуль треба кешувати, а не шукати щоразу')
    print('✓ сканер не тягне важкий пакет у гарячий шлях')


def test_blocked_direction_short_circuits_the_filters():
    """На вимкненому напрямку решту фільтрів рахувати НІ ДО ЧОГО — вони ходять
    у БД/кеші, а результат нікому не потрібен."""
    src = _src('detection/smc_scanner.py')
    i = src.index('def _signal_allowed')
    body = src[i:src.index('\n    def ', i + 10)]
    # Докстрінг перелічує назви фільтрів — шукаємо в КОДІ, не в тексті.
    body = body[body.index('parts = []'):]
    j = body.index('_dg.allows(')
    tail = body[j:j + 500]
    _check('return (False,' in tail,
           'заблокований напрямок мусить виходити ОДРАЗУ, а не рахувати фільтри')
    print('✓ вимкнений напрямок виходить одразу, без зайвих розрахунків')


def test_intercept_refuses_a_disabled_direction_before_any_queue():
    """Питання користувача було саме про це: сигнал вимкненого напрямку НЕ
    сідає в чергу."""
    src = _src('detection/fuel_filter.py')
    i = src.index('def intercept')
    body = src[i:i + 4000]
    g = body.index('_dg.allows(')
    _check("blocked_dir" in body[g:g + 400],
           'intercept мусить віддати ОКРЕМУ диспозицію')
    for q in ("q1 = bool(", "q2 = bool(", "q3 = bool(", "q4 = bool("):
        _check(g < body.index(q), f'ворота мусять стояти ДО розбору черг ({q})')
    print('✓ intercept: вимкнений напрямок у чергу не потрапляє')


def test_blocked_dir_is_not_empty_string():
    """⚠️ Порожній рядок з `intercept` означає «жодна черга не взяла — відкривай
    НАПРЯМУ». Віддати '' на заблокованому напрямку = пустити сигнал повз кнопки
    прямісінько у відкриття."""
    src = _src('detection/fuel_filter.py')
    i = src.index('def intercept')
    body = src[i:i + 4000]
    g = body.index('_dg.allows(')
    seg = body[g:g + 400]
    _check("return 'blocked_dir'" in seg, seg)
    _check("return ''" not in seg, 'НЕ можна повертати порожній рядок')
    print('✓ intercept віддає blocked_dir, а не «відкривай напряму»')


def test_trade_manager_handles_blocked_dir_in_both_call_sites():
    """⚠️ Без явної гілки значення провалилось би повз обидва `if` прямо у
    ПРЯМЕ ВІДКРИТТЯ (перевірки на '' там немає)."""
    src = _src('detection/trade_manager.py')
    _check(src.count("_disp == 'blocked_dir'") == 2,
           'обидва місця виклику intercept (on_signal + manual_open) мусять '
           'обробити blocked_dir')
    print('✓ TM обробляє blocked_dir в обох місцях виклику intercept')


def test_ff_open_is_the_last_lock_and_by_hand_does_not_bypass_it():
    """`_open` — ЄДИНИЙ вузол усіх черг. Запис міг лягти в чергу ДО того, як
    кнопку вимкнули, тож ворота потрібні і тут. ✋ їх НЕ обходить — це рішення
    користувача («головні кнопки») і те саме, що обіцяє підказка кнопки."""
    src = _src('detection/fuel_filter.py')
    i = src.index('    def _open(self, symbol: str, side: str')
    body = src[i:src.index('\n    def ', i + 10)]
    g = body.index('_dg.allows(')
    _check('return False' in body[g:g + 600], 'заблокований напрямок → відмова')
    # Ворота мусять стояти ДО перевірок, які `by_hand` СВІДОМО обходить
    # (блокування після ручного закриття, «нова ситуація»).
    _check(g < body.index('if not by_hand:'),
           'ворота напрямку мусять стояти ДО by_hand-винятків')
    _check('by_hand' not in body[g:g + 400],
           'сама перевірка воріт НЕ сміє мати винятку by_hand')
    print('✓ _open: останній замок, ✋ його не обходить')


def test_queue4_engine_now_respects_the_buttons():
    """Двигун Черги-4 свідомо їх не дивився — саме тому на робочій установці
    кнопки не діяли."""
    src = _src('detection/fuel_filter.py')
    i = src.index('def _engine_tick_queue4')
    body = src[i:i + 6000]
    _check('НЕ фільтрує кнопками' not in body,
           'застарілий докстрінг суперечить новій поведінці — прибрати')
    _check('_allow_long, _allow_short = self._entry_gates()' in body,
           'двигун мусить читати ворота')
    _check("d == 'LONG' and not _allow_long" in body,
           'і пропускати запис вимкненого напрямку')
    print('✓ Черга-4 більше не ігнорує головні кнопки')


def test_queue4_keeps_the_record_when_a_button_is_off():
    """⚠️ Кнопку можна увімкнути назад — на відміну від «відпрацьованого»
    запису, цей ще актуальний. Виселяти його НЕ можна (інакше сигнал зникне
    назавжди, як у кейсі ARBUSDT)."""
    src = _src('detection/fuel_filter.py')
    i = src.index('def _engine_tick_queue4')
    body = src[i:i + 6000]
    j = body.index("d == 'LONG' and not _allow_long")
    seg = body[j:j + 300]
    _check('continue' in seg, 'мусить бути пропуск')
    _check('_pending4.pop' not in seg, 'запис НЕ виселяємо — кнопку можуть увімкнути')
    print('✓ вимкнена кнопка ПРОПУСКАЄ запис, а не викидає його з черги')


# ═════════ 4. ЄДИНЕ ДЖЕРЕЛО КЛЮЧІВ ═════════════════════════════════════════
def test_single_source_of_keys():
    """Корінь усього дефекту — ті самі два ключі, набрані руками в кількох
    файлах, кожен зі своїм читанням. Літерали дозволені ЛИШЕ у
    `direction_gate.py` (визначення), `trade_manager.py` (DEFAULT_SETTINGS +
    список полів) і `auto_gate.py` (він їх ПИШЕ)."""
    allowed = {'detection/direction_gate.py', 'detection/trade_manager.py',
               'detection/auto_gate.py'}
    offenders = []
    for root, dirs, files in os.walk(_ROOT):
        dirs[:] = [d for d in dirs
                   if d not in ('.git', '__pycache__', 'node_modules', 'venv')]
        for fn in files:
            if not fn.endswith('.py') or fn.startswith('test_'):
                continue
            rel = os.path.relpath(os.path.join(root, fn), _ROOT).replace(os.sep, '/')
            if rel in allowed:
                continue
            body = _src(rel)
            if 'allow_long_entries' in body or 'allow_short_entries' in body:
                offenders.append(rel)
    _check(not offenders,
           f'ключі мусять читатись через direction_gate, а не руками: {offenders}')
    print('✓ ключі кнопок живуть в одному місці')


def test_direction_gate_has_no_import_cycle():
    """`direction_gate` не сміє тягнути `trade_manager` на рівні модуля —
    інакше імпорт заклинить (TM сам його імпортує)."""
    tree = ast.parse(_src('detection/direction_gate.py'))
    for node in tree.body:      # ТІЛЬКИ верхній рівень
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            mod = getattr(node, 'module', '') or ''
            _check('trade_manager' not in mod and
                   not any('trade_manager' in n for n in names),
                   f'циклічний імпорт на верхньому рівні: {mod} {names}')
    print('✓ direction_gate не має циклічного імпорту')


# ═════════ 5. UI КАЖЕ ПРАВДУ ═══════════════════════════════════════════════
def test_ui_tooltip_matches_the_new_behaviour():
    html = _src('templates/smart_money.html')
    _check(html.count('ГОЛОВНИЙ ВИМИКАЧ напрямку') == 2,
           'обидві кнопки мусять пояснювати, що це вимикач, а не фільтр')
    _check(html.count('Обидві кнопки вимкнені = бот не приймає жодних сигналів') == 2,
           'правило «обидві вимкнені = стоп» має бути в підказці')
    _check('sm-side-gates-state' in html, 'потрібен видимий стан вимикача')
    print('✓ UI: підказки й видимий стан відповідають поведінці')


def test_ui_state_badge_is_hidden_when_both_are_on():
    """Обидві увімкнені — вимикач нічого не робить, підпис був би шумом."""
    html = _src('templates/smart_money.html')
    i = html.index("const badge = document.getElementById('sm-side-gates-state')")
    seg = html[i:i + 900]
    _check("badge.style.display = 'none'" in seg, 'мусить ховатись')
    _check('longOn && shortOn' in seg, 'умова — саме «обидві увімкнені»')
    print('✓ підпис стану ховається, коли вимикач нейтральний')


if __name__ == '__main__':
    _tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    _order = [
        test_works_in_the_direction_whose_button_is_on,
        test_both_off_is_a_full_stop,
        test_missing_keys_default_to_open,
        test_unknown_side_is_not_blocked,
        test_chip_is_silent_when_the_switch_does_nothing,
        test_fail_open_when_state_is_unknown,
        test_bypass_gates_no_longer_defeats_the_master_switch,
        test_bypass_gates_still_governs_the_other_gates,
        test_ff_opens_with_bypass_gates_so_the_gate_must_live_in_open,
        test_scanner_gate_is_first_in_the_shared_chain,
        test_scanner_does_not_drag_the_whole_package_into_the_hot_path,
        test_blocked_direction_short_circuits_the_filters,
        test_intercept_refuses_a_disabled_direction_before_any_queue,
        test_blocked_dir_is_not_empty_string,
        test_trade_manager_handles_blocked_dir_in_both_call_sites,
        test_ff_open_is_the_last_lock_and_by_hand_does_not_bypass_it,
        test_queue4_engine_now_respects_the_buttons,
        test_queue4_keeps_the_record_when_a_button_is_off,
        test_single_source_of_keys,
        test_direction_gate_has_no_import_cycle,
        test_ui_tooltip_matches_the_new_behaviour,
        test_ui_state_badge_is_hidden_when_both_are_on,
    ]
    _check(len(_order) == len(_tests), 'усі тести мусять бути в списку запуску')
    for t in _order:
        t()
    print(f'\nУсі тести «головні кнопки напрямку» пройдено ✅ ({len(_order)})')
