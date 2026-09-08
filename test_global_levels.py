"""🎯 РІВНІ УГОДИ (SL · TP-1 · TP-2) — ГЛОБАЛЬНІ, ПОЗА ЧЕРГАМИ.

**Вимога користувача (08.09), дослівно:** «Винеси за межі Черга-4 автоматичний
SL і TP, тобто "🛑 SL з", "📐 Мін. R на відкриття ≥" … виділи ці параметри, щоб
їх було видно. Це має спрацьовувати навіть, коли всі черги вимкнені. Сигнал
відразу летів на відкриття угоди і присвоювались йому автоматично SL і TP.»
Плюс: «Які параметри застосовуються для визначення Manual TP-2, винеси їх
також для налаштування.»

**Що було не так (аудит):**
1. `🛑 SL з` (`queue4_sl_source`) застосовувався в `trade_manager.
   _auto_ob_manual_sl` ЛИШЕ коли `opened_by` містив 'Q4'. На прямому відкритті
   (усі черги вимкнені) налаштування мовчки не діяло — стоп брався з
   `q2_auto_ob_sl_tf` (деф. 15m). Те саме поле давало РІЗНИЙ стоп залежно від
   того, хто відкрив угоду.
2. `📐 Мін. R` і фіксація ЦІЛІ (= Manual TP-2) жили ТІЛЬКИ в двигуні Черги-4.
   При вимкнених чергах сигнал ішов повз них прямо у відкриття: R ніхто не
   рахував, 🧲 магніт ціллю не ставав.
3. `pilot_tp2_from_magnet` — головний перемикач TP-2 — не мав UI ВЗАГАЛІ.
4. Вікно пошуку TP-1 (`tp1_min_path_pct`/`tp1_max_path_pct`) і похідний фолбек
   (`tp1_fallback_path_pct`) існували в алгоритмі, але не мали ключів у
   налаштуваннях, тож `cfg` їх не бачив: 30/75/50 було замуровано в коді.
"""
import ast
import os
import re
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _src(rel):
    with open(os.path.join(_ROOT, rel), encoding='utf-8') as f:
        return f.read()


def _method(rel, name):
    """Тіло методу від `def name` до наступного `def` того ж рівня."""
    src = _src(rel)
    i = src.index(f'def {name}(')
    j = src.find('\n    def ', i + 10)
    return src[i:(j if j > 0 else len(src))]


def _strip_comments(py: str) -> str:
    """Прибрати коментарі й докстрінги — щоб перевіряти КОД, а не пояснення."""
    out = []
    for ln in py.splitlines():
        s = ln.split('#', 1)[0]
        out.append(s)
    return '\n'.join(out)


# ═════════ 1. 🛑 ДЖЕРЕЛО SL — ГЛОБАЛЬНЕ ════════════════════════════════════
def test_sl_source_is_no_longer_queue4_only():
    """ГОЛОВНИЙ ЗАМОК №1. Вибір «🛑 SL з» мусить іти ПЕРШИМ для БУДЬ-ЯКОЇ
    угоди, а не лише для тих, чий `opened_by` містить 'Q4'."""
    body = _strip_comments(_method('detection/trade_manager.py', '_auto_ob_manual_sl'))
    _check("'Q4' in str(pos.get('opened_by')" not in body,
           'умова «лише для Q4» повертає баг: на прямому відкритті вибір '
           'користувача мовчки не діяв би')
    _check('_from_q4' not in body, 'прапорець Q4-походження більше не потрібен')
    _check("queue4_sl_source" in body, 'джерело мусить читатись із налаштування')
    print('✓ джерело SL застосовується до КОЖНОЇ угоди, не лише до Q4')


def test_chosen_source_is_first_in_the_chain():
    """Обране джерело — ПЕРШЕ; решта лишається фолбеком (гарантія стопа)."""
    body = _strip_comments(_method('detection/trade_manager.py', '_auto_ob_manual_sl'))
    i = body.index('_sl_src')
    tail = body[i:]
    first = tail.index("if _sl_src == '1h':")
    # Фолбеки мусять іти ПІСЛЯ гілки вибору.
    for later in ('_add_ob(ob_tf', 'sources.append(_from_pct)'):
        _check(first < tail.index(later), f'{later} мусить бути ПІСЛЯ обраного джерела')
    _check('sources.append(_from_pct)' in tail,
           '🛡 гарантія «% від входу» мусить лишитись останнім кроком')
    print('✓ обране джерело перше, гарантія стопа лишилась останньою')


# ═════════ 2. 📐 R + ЦІЛЬ — ПРАЦЮЮТЬ БЕЗ ЧЕРГ ══════════════════════════════
def test_fuel_filter_exposes_a_public_open_plan():
    """Розрахунок мусить мати ПУБЛІЧНІ двері — інакше прямий шлях або не
    порахує нічого, або заведе ДРУГУ реалізацію (і числа розійдуться)."""
    src = _src('detection/fuel_filter.py')
    _check('def open_plan(self, symbol: str, side: str' in src,
           'потрібні публічні двері до розрахунку рівнів')
    _check('def min_open_r(self' in src, 'потрібен публічний доступ до порога R')
    body = _method('detection/fuel_filter.py', 'open_plan')
    _check('self._q4_expected_r(' in body,
           'open_plan мусить ДЕЛЕГУВАТИ, а не рахувати по-своєму — '
           'дві реалізації рано чи пізно розійдуться')
    _check('_q4_rr_objective.pop(' in body,
           'ціль передається одноразово (pop), а не кешується')
    print('✓ FF має публічні двері до ТОГО САМОГО розрахунку')


def test_open_plan_does_not_depend_on_queues_being_enabled():
    """Ворота рівнів — не про черги. `open_plan`/`min_open_r` не сміють
    перевіряти `enabled` чи `queue*_enabled`."""
    for name in ('open_plan', 'min_open_r'):
        body = _strip_comments(_method('detection/fuel_filter.py', name))
        for bad in ("get('enabled')", 'queue4_enabled', 'queue1_enabled', 'is_enabled()'):
            _check(bad not in body, f'{name} не сміє залежати від {bad}')
    print('✓ рівні рахуються незалежно від того, які черги увімкнені')


def test_direct_open_path_runs_the_r_gate_and_fixes_the_objective():
    """ГОЛОВНИЙ ЗАМОК №2. Прямий шлях (усі черги вимкнені) мусить пройти той
    самий гейт за R і зафіксувати ціль угоди."""
    body = _method('detection/trade_manager.py', 'on_signal')
    _check('open_plan(symbol, side)' in body, 'прямий шлях мусить питати план входу')
    _check('min_open_r()' in body, 'і застосовувати поріг R')
    _check('set_pending_objective(symbol, _robj)' in body,
           'ціль мусить лягти на угоду — інакше R рішення ≠ R угоди (VETUSDT)')
    # Гейт стоїть ДО обох шляхів відкриття.
    g = body.index('open_plan(symbol, side)')
    _check(g < body.index('=== Real-money track'),
           'гейт мусить стояти ДО реального треку')
    _check(g < body.index('=== Paper (shadow) track'),
           'і ДО паперового — інакше paper відкривався б повз нього')
    print('✓ прямий шлях проходить гейт R і фіксує ціль (обидва треки)')


def test_r_gate_never_blocks_on_its_own_error():
    """Збій розрахунку не сміє стати мовчазною відмовою відкривати."""
    body = _method('detection/trade_manager.py', 'on_signal')
    i = body.index('open_plan(symbol, side)')
    seg = body[i:i + 1400]
    _check('except Exception' in seg, 'потрібен захист від винятку')
    j = seg.index('except Exception')
    _check('return' not in seg[j:j + 320],
           'у except НЕ можна повертати відмову — це вигадана причина')
    print('✓ помилка гейта не блокує відкриття (fail-open)')


def test_gate_skips_work_when_position_already_held():
    """Магніт — це запит до біржі, а `_pilot_context` важкий. На дублікаті
    (та сама монета, той самий бік) рахувати нема чого."""
    body = _method('detection/trade_manager.py', 'on_signal')
    _check('_same_side' in body, 'потрібна перевірка «вже в угоді тим самим боком»')
    i = body.index('_same_side = ')
    _check(body.index('open_plan(symbol, side)') > i,
           'перевірка мусить стояти ДО розрахунку')
    print('✓ на дублікаті важкий розрахунок не запускається')


# ═════════ 3. TP-2 / TP-1 — ПАРАМЕТРИ ВИНЕСЕНІ В НАЛАШТУВАННЯ ══════════════
def test_tp1_window_keys_exist_so_cfg_can_see_them():
    """`cfg` збирається як `{k[6:]: v for k in s if k.startswith('pilot_')
    and k[6:] in trade_pilot.DEFAULTS}` — без ключів у налаштуваннях вікно
    TP-1 було замуроване на 30/75/50."""
    tm = _src('detection/trade_manager.py')
    tp = _src('detection/trade_pilot.py')
    for k in ('tp1_min_path_pct', 'tp1_max_path_pct', 'tp1_fallback_path_pct'):
        _check(f"'{k}'" in tp, f'{k} мусить бути в trade_pilot.DEFAULTS')
        _check(f"'pilot_{k}'" in tm,
               f'pilot_{k} мусить бути в налаштуваннях TM — інакше cfg його не візьме')
    print('✓ вікно TP-1 і фолбек тепер налаштовні (cfg їх бачить)')


def test_tp1_window_is_validated_and_cannot_invert():
    """max < min = порожнє вікно → TP-1 не знайдеться НІКОЛИ, і виглядало б
    як «алгоритм не працює». Зводимо до коректного порядку."""
    src = _src('detection/trade_manager.py')
    _check("pilot_tp1_max_path_pct'] < self._settings['pilot_tp1_min_path_pct']" in src,
           'потрібна перевірка порядку меж вікна')
    print('✓ вікно TP-1 не може вивернутись навиворіт')


def test_zero_fallback_survives_the_save():
    """0 = «не ставити TP-1» — ЗМІСТОВНЕ значення. `parseFloat(v) || d` мовчки
    перетворив би його на 50, і вимкнути фолбек було б неможливо."""
    html = _src('templates/smart_money.html')
    _check("pilot_tp1_fallback_path_pct: _numOr(" in html,
           'фолбек мусить зберігатись через _numOr, а не через `|| дефолт`')
    _check('function _numOr(' in html, 'потрібен помічник, що поважає нуль')
    i = html.index('function _numOr(')
    seg = html[i:i + 400]
    _check('isFinite(v) ? v : d' in seg, 'нуль мусить проходити як значення')
    print('✓ «фолбек = 0» зберігається, а не підміняється дефолтом')


def test_tp2_magnet_toggle_finally_has_ui():
    """`pilot_tp2_from_magnet` — головний перемикач TP-2 — не мав UI ВЗАГАЛІ:
    змінити його можна було лише через API."""
    html = _src('templates/smart_money.html')
    _check('tm-pilot-tp2-magnet' in html, 'потрібен видимий тумблер магніту')
    _check('pilot_tp2_from_magnet: !!(document.getElementById' in html,
           'тумблер мусить зберігатись')
    _check("s.pilot_tp2_from_magnet !== false" in html,
           'дефолт ON мусить читатись як ON, а не як «не задано → OFF»')
    print('✓ 🧲 TP-2 з магніту тепер видно й можна вимкнути')


def test_magnet_source_is_one_key_shown_in_two_places():
    """Біржа й глибина магніту — ТІ САМІ ключі сканера, що у 💧 фільтра.
    Другого джерела НЕ заводимо: поріг фільтра і ціль угоди мусять рахуватись
    з одних даних (урок PD-зони)."""
    html = _src('templates/smart_money.html')
    _check('function updateMagnetSource(' in html, 'потрібен сейв магніт-джерела')
    body = html[html.index('async function updateMagnetSource('):][:900]
    _check('liq_filter_exchange' in body and 'liq_filter_bars' in body,
           'мусять писатись САМЕ ключі сканера, без власних дублів')
    for bad in ('magnet_exchange', 'magnet_bars', 'tp2_exchange'):
        _check(bad not in html, f'окремий ключ {bad} розколов би джерело правди')
    _check('function _syncMagnetControls(' in html,
           'обидва набори контролів мусять показувати однакове значення')
    print('✓ магніт-джерело — один ключ, два місця показу')


# ═════════ 4. UI: ПАРАМЕТРИ ВИДНО, І ВОНИ НЕ В ЧЕРЗІ ═══════════════════════
def test_controls_left_the_queue4_accordion():
    """ГОЛОВНИЙ ЗАМОК №3. «🛑 SL з» і «📐 Мін. R» більше не всередині
    гармошки Черги-4 — інакше при вимкненій черзі їх не знайти."""
    html = _src('templates/smart_money.html')
    q4 = html[html.index('id="ff-q4-settings"'):html.index('id="ff-timers4-table"')]
    for cid in ('ff-queue4-sl-source', 'ff-queue4-min-rr'):
        _check(cid not in q4, f'{cid} мусить піти з налаштувань Черги-4')
        _check(cid in html, f'{cid} мусить лишитись на сторінці (перенесено)')
    print('✓ контроли пішли з гармошки Черги-4')


def test_controls_are_in_the_global_accordion():
    html = _src('templates/smart_money.html')
    g = html[html.index('<div id="autosl-body">'):html.index('❤️ FF base')]
    for cid in ('ff-queue4-sl-source', 'ff-queue4-min-rr', 'tm-pilot-tp2-magnet',
                'ff-mag-exchange', 'ff-mag-bars', 'tm-pilot-tp1-min-path',
                'tm-pilot-tp1-max-path', 'tm-pilot-tp1-fallback'):
        _check(cid in g, f'{cid} мусить бути у ГЛОБАЛЬНІЙ гармошці')
    _check('🎯 Рівні угоди' in g, 'блок мусить бути підписаний і помітний')
    _check('коли ВСІ черги вимкнені' in g,
           'підпис мусить прямо казати, що це працює без черг')
    print('✓ усі параметри рівнів — в одному видимому глобальному блоці')


def test_keys_were_not_renamed():
    """⚠️ `queue4_sl_source` / `queue4_min_rr` лишились як були: перейменування
    зламало б УЖЕ ЗБЕРЕЖЕНІ налаштування в БД. Історична назва ключа ≠ область
    дії — той самий прецедент, що з `q2_auto_ob_sl*`."""
    html = _src('templates/smart_money.html')
    _check("queue4_sl_source: _v('ff-queue4-sl-source'" in html,
           'ключ SL-джерела мусить лишитись незмінним')
    _check('queue4_min_rr' in html, 'ключ порога R мусить лишитись незмінним')
    print('✓ ключі не перейменовані — збережені налаштування живі')


if __name__ == '__main__':
    _order = [
        test_sl_source_is_no_longer_queue4_only,
        test_chosen_source_is_first_in_the_chain,
        test_fuel_filter_exposes_a_public_open_plan,
        test_open_plan_does_not_depend_on_queues_being_enabled,
        test_direct_open_path_runs_the_r_gate_and_fixes_the_objective,
        test_r_gate_never_blocks_on_its_own_error,
        test_gate_skips_work_when_position_already_held,
        test_tp1_window_keys_exist_so_cfg_can_see_them,
        test_tp1_window_is_validated_and_cannot_invert,
        test_zero_fallback_survives_the_save,
        test_tp2_magnet_toggle_finally_has_ui,
        test_magnet_source_is_one_key_shown_in_two_places,
        test_controls_left_the_queue4_accordion,
        test_controls_are_in_the_global_accordion,
        test_keys_were_not_renamed,
    ]
    _all = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    _check(len(_order) == len(_all), 'усі тести мусять бути у списку запуску')
    for t in _order:
        t()
    print(f'\nУсі тести «глобальні рівні угоди» пройдено ✅ ({len(_order)})')
