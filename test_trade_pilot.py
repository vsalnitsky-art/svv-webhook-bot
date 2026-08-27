"""Тести 🎯 АВТОПІЛОТА УГОДИ — супроводу позиції по обʼєктах ГРАФІКА.

Вимога користувача: «навчи бота аналізувати графік, щоб максимально вигідно
тримати угоду і для чіткого виходу при максимальному результаті».

Дві протилежні задачі одночасно:
  • не вийти зарано, поки структура працює на нас;
  • не віддати назад набране і зафіксувати там, де рух реально має спинитись.

Рішення будуються на РЕАЛЬНИХ обʼєктах, які бот уже рахує: пули ліквідації
(runway з liq-map), екстремуми дилінг-діапазону (той самий, що дає PD-зону й
лінії на графіку) та POC. Тут — чисті функції, тож усе перевіряється точно.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


tp = _load('detection.trade_pilot', 'detection/trade_pilot.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


SWING_LONG = {'high': {'price': 110.0, 'label': 'Weak High'},
              'low': {'price': 96.0, 'label': 'Strong Low'}}
RUNWAY_UP = {'dir': 'LONG',
             'next': {'price': 103.0, 'usd': 2e6},
             'main': {'price': 108.0, 'usd': 9e6}}


# ═══════════════════ 🎯 Цілі: що бот бачить попереду ═══════════════════════
def test_targets_are_real_chart_objects_ahead_only():
    t = tp.collect_targets('LONG', 100.0, swing=SWING_LONG, runway=RUNWAY_UP, poc=105.0)
    kinds = [x['kind'] for x in t]
    _check(set(kinds) == {'liq_next', 'liq_main', 'swing', 'poc'},
           f'мають зібратись усі 4 типи обʼєктів: {kinds}')
    _check(all(x['price'] > 100.0 for x in t),
           'для LONG беруться ЛИШЕ рівні ВИЩЕ ціни')
    _check(t == sorted(t, key=lambda x: x['dist_pct']),
           'список має йти від найближчого до дальшого')
    print(f'✓ цілі попереду руху зібрані й відсортовані: '
          f'{[(x["kind"], x["price"]) for x in t]}')


def test_short_takes_levels_below_price():
    sw = {'high': {'price': 104.0, 'label': 'Strong High'},
          'low': {'price': 90.0, 'label': 'Weak Low'}}
    rw = {'dir': 'SHORT', 'next': {'price': 97.0}, 'main': {'price': 92.0}}
    t = tp.collect_targets('SHORT', 100.0, swing=sw, runway=rw, poc=95.0)
    _check(all(x['price'] < 100.0 for x in t), 'для SHORT — лише рівні НИЖЧЕ ціни')
    _check('swing' in [x['kind'] for x in t], 'swing-low має потрапити в цілі')
    print('✓ SHORT дзеркально бере рівні знизу')


def test_noise_and_too_far_levels_are_ignored():
    t = tp.collect_targets('LONG', 100.0,
                           swing={'high': {'price': 100.05}},      # 0.05% — шум
                           runway={'dir': 'LONG', 'main': {'price': 400.0}},  # +300%
                           poc=None)
    _check(t == [], f'шум і надто далекі рівні не є цілями: {t}')
    print('✓ шум (<0.3%) і задалеке (>15%) відсіюються')


def test_objective_is_the_furthest_meaningful_target():
    """«Максимальний результат» = найдальший змістовний обʼєкт, а не найближчий."""
    t = tp.collect_targets('LONG', 100.0, swing=SWING_LONG, runway=RUNWAY_UP, poc=105.0)
    obj = tp.pick_objective(t)
    _check(obj['price'] == 110.0,
           f'головна ціль — найдальша (swing 110), отримано {obj}')
    _check(t[0]['price'] == 103.0, 'найближча перепона лишається окремо (103)')
    print('✓ головна ціль = найдальша; найближча — окремо як перепона')


# ═══════════════════ 🛡 Структурний стоп ═══════════════════════════════════
def test_structure_stop_sits_behind_the_swing():
    st = tp.structure_stop('LONG', 100.0, SWING_LONG, {'stop_buf_pct': 0.1})
    _check(abs(st['price'] - 96.0 * 0.999) < 1e-9,
           f'LONG → під swing-low + буфер, отримано {st}')
    _check('Strong Low' in st['label'], f'підпис має називати рівень: {st}')
    st2 = tp.structure_stop('SHORT', 100.0,
                            {'high': {'price': 104.0, 'label': 'Strong High'}},
                            {'stop_buf_pct': 0.1})
    _check(abs(st2['price'] - 104.0 * 1.001) < 1e-9, f'SHORT → над swing-high: {st2}')
    print('✓ структурний стоп стоїть ЗА екстремумом діапазону + буфер')


def test_broken_structure_yields_no_stop():
    """Рівень з неправильного боку від ціни = структура вже зламана. Стоп там
    спрацював би миттєво, тож вигадувати його не можна."""
    st = tp.structure_stop('LONG', 95.0, SWING_LONG)   # ціна ВЖЕ під swing-low
    _check(st is None, f'стоп вище ціни для LONG неприпустимий: {st}')
    print('✓ зламана структура → стоп не вигадуємо')


def test_ratchet_never_loosens_the_stop():
    _check(tp.better_stop('LONG', 98.0, 96.0) is True, 'LONG: вище — краще')
    _check(tp.better_stop('LONG', 95.0, 96.0) is False, 'LONG: нижче — послаблення')
    _check(tp.better_stop('SHORT', 102.0, 104.0) is True, 'SHORT: нижче — краще')
    _check(tp.better_stop('SHORT', 106.0, 104.0) is False, 'SHORT: вище — послаблення')
    _check(tp.better_stop('LONG', 98.0, None) is True, 'стопа ще не було → ставимо')
    _check(tp.better_stop('LONG', None, 96.0) is False, 'без нового рівня — нічого')
    print('✓ ратчет: стоп рухається ЛИШЕ в бік прибутку')


# ═══════════════════ 🧭 План: тримати / підтягнути / фіксувати ═════════════
def test_plan_holds_while_target_is_ahead():
    # +0.3% від входу при ризику 1% = 0.3R — беззбиток ще не вмикається,
    # а структурний стоп (95.9) гірший за наявний 99.0 → нічого не міняємо.
    r = tp.plan('LONG', 100.0, 100.3, swing=SWING_LONG, runway=RUNWAY_UP,
                poc=105.0, prev_stop=99.0)
    _check(r['action'] == 'hold', f'ціль ще попереду → тримаємо: {r}')
    _check(r['objective']['price'] == 110.0, 'ціль показана')
    _check(any('тримаємо до цілі' in x for x in r['reasons']),
           f'причина має бути людською: {r["reasons"]}')
    print(f'✓ тримаємо: {r["reasons"][0]}')


def test_plan_trails_stop_when_structure_improves():
    r = tp.plan('LONG', 100.0, 104.0, swing=SWING_LONG, runway=RUNWAY_UP,
                poc=None, prev_stop=93.0)
    _check(r['action'] == 'trail', f'структурний стоп кращий → підтягуємо: {r}')
    _check(abs(r['stop'] - 96.0 * 0.999) < 1e-9, f'рівень: {r["stop"]}')
    _check(any('стоп підтягнуто' in x for x in r['reasons']), f'{r["reasons"]}')
    print(f'✓ трейл: {r["reasons"][0]}')


def test_plan_takes_profit_at_the_objective():
    """Ціль ФІКСУЄТЬСЯ один раз і не «тікає». Без замка вона зникала б зі
    списку «попереду» рівно тоді, коли ціна до неї дійшла, найдальшою ставала
    наступна — і угода НІКОЛИ не доходила б до фіксації (рухомі ворота)."""
    # 1) Перший тік — ціль обрано (110), ще далеко.
    r1 = tp.plan('LONG', 100.0, 101.0, swing=SWING_LONG, runway=RUNWAY_UP,
                 poc=105.0, prev_stop=95.0)
    lock = r1['objective']
    _check(lock and lock['price'] == 110.0, f'ціль мала зафіксуватись на 110: {lock}')
    # 2) Ціна дійшла до неї — фіксуємо, хоч 110 уже НЕ «попереду».
    r2 = tp.plan('LONG', 100.0, 110.0, swing=SWING_LONG, runway=RUNWAY_UP,
                 poc=105.0, objective_lock=lock)
    _check(r2['action'] == 'take', f'ціль досягнуто → фіксуємо: {r2}')
    _check(any('ціль досягнуто' in x for x in r2['reasons']), f'{r2["reasons"]}')
    print(f'✓ фіксація: {r2["reasons"][0]}')


def test_objective_does_not_drift_between_ticks():
    """Поки ціль не досягнута, вона лишається ТА САМА, навіть коли попереду
    зʼявились нові рівні."""
    lock = {'price': 108.0, 'label': 'головний пул ліквідності', 'kind': 'liq_main'}
    r = tp.plan('LONG', 100.0, 104.0, swing={'high': {'price': 130.0},
                                             'low': {'price': 96.0}},
                runway=RUNWAY_UP, poc=None, prev_stop=95.0, objective_lock=lock)
    _check(r['objective']['price'] == 108.0,
           f'ціль не має перестрибувати на дальший рівень: {r["objective"]}')
    print('✓ ціль не «тікає» між тіками (немає рухомих воріт)')


def test_plan_does_not_loosen_an_existing_stop():
    r = tp.plan('LONG', 100.0, 100.3, swing=SWING_LONG, runway=RUNWAY_UP,
                poc=None, prev_stop=99.0)   # 99 вже КРАЩЕ за структурні 95.9
    _check(r['action'] == 'hold', 'кращий стоп уже стоїть — не чіпаємо')
    _check(any('не кращий за поточний' in x for x in r['reasons']),
           f'це має бути сказано прямо: {r["reasons"]}')
    print('✓ наявний кращий стоп не послаблюється')


def test_breakeven_kicks_in_after_r_multiple():
    """Пройшли ≥1R → стоп не має бути гіршим за беззбиток."""
    # ризик = |100 − 98| = 2%; ціна 102.5 → пройдено 2.5% = 1.25R
    r = tp.plan('LONG', 100.0, 102.5, swing={'low': {'price': 90.0}},
                runway={'dir': 'LONG', 'main': {'price': 120.0}},
                prev_stop=98.0, cfg={'be_at_r': 1.0, 'be_lock_pct': 0.05})
    _check(r['action'] == 'trail', f'мав спрацювати беззбиток: {r}')
    _check(r['stop'] > 100.0, f'стоп має піти В ПЛЮС, отримано {r["stop"]}')
    _check(any('беззбиток' in x for x in r['reasons']), f'{r["reasons"]}')
    print(f'✓ беззбиток після 1R: стоп {r["stop"]:.4f} · {r["reasons"][0]}')


def test_no_targets_still_holds_with_stop():
    r = tp.plan('LONG', 100.0, 100.2, swing=SWING_LONG, runway=None, poc=None,
                prev_stop=99.0)
    _check(r['action'] == 'hold', 'без цілей тримаємо за стопом')
    _check(r['objective'] is not None, 'swing-high усе одно є ціллю')
    print('✓ навіть без liq-даних працює по свінгу')


def test_completely_blind_context_is_safe():
    r = tp.plan('LONG', 100.0, 100.0, swing=None, runway=None, poc=None)
    _check(r['action'] == 'hold' and r['stop'] is None,
           f'без жодних даних нічого не робимо: {r}')
    _check(any('не видно' in x for x in r['reasons']), f'{r["reasons"]}')
    print('✓ порожній контекст безпечний (нічого не вигадуємо)')


def test_garbage_input_never_raises():
    for bad in (None, 'abc', float('nan'), float('inf'), -5):
        r = tp.plan('LONG', 100.0, bad, swing=SWING_LONG)
        _check(isinstance(r, dict) and 'action' in r, f'сміття {bad!r} зламало план')
    r2 = tp.plan('WHAT', 100.0, 100.0, swing=SWING_LONG)
    _check(r2['action'] == 'hold', 'невідомий напрямок → тримаємо')
    print('✓ сміттєві дані не ламають автопілот')


def test_defaults_are_conservative_and_off():
    import importlib.util as _iu
    spec = _iu.spec_from_file_location('_tm_probe',
                                       os.path.join(_ROOT, 'detection/trade_manager.py'))
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    _check("'pilot_enabled': False" in src,
           'автопілот має бути ВИМКНЕНИЙ за замовчуванням')
    _check(tp.DEFAULTS['min_target_dist_pct'] > 0, 'поріг шуму має бути ненульовий')
    _check(tp.DEFAULTS['stop_buf_pct'] > 0, 'буфер стопа має бути ненульовий')
    print('✓ дефолти консервативні, автопілот вимкнено')


# ═════════ 📊 ВІЗУАЛІЗАЦІЯ: прогрес до цілі для таблиці угод ═══════════════
def test_progress_measures_path_from_entry_to_objective():
    """Смужка в таблиці = скільки шляху ВІД ВХОДУ до цілі вже пройдено."""
    obj = {'price': 110.0, 'label': 'Weak High'}
    p0 = tp.progress('LONG', 100.0, 100.0, obj)
    _check(p0['pct'] == 0.0, f'на вході — 0%: {p0}')
    p50 = tp.progress('LONG', 100.0, 105.0, obj)
    _check(p50['pct'] == 50.0, f'півдороги — 50%: {p50}')
    p100 = tp.progress('LONG', 100.0, 110.0, obj)
    _check(p100['pct'] == 100.0, f'на цілі — 100%: {p100}')
    _check(abs(p50['left_pct'] - 4.762) < 0.01, f'залишок у % ціни: {p50}')
    print(f'✓ прогрес до цілі: 0% → {p50["pct"]}% → {p100["pct"]}%')


def test_progress_for_short_mirrors():
    obj = {'price': 90.0, 'label': 'Weak Low'}
    r = tp.progress('SHORT', 100.0, 95.0, obj)
    _check(r['pct'] == 50.0, f'SHORT: рух ВНИЗ — це прогрес: {r}')
    _check(r['done_pct'] == 5.0, f'пройдено +5% у бік угоди: {r}')
    print('✓ SHORT рахується дзеркально')


def test_progress_is_clamped_but_keeps_raw_move():
    """Смужка не має «вилітати», але СИРИЙ рух треба бачити — зокрема мінусовий."""
    r = tp.progress('LONG', 100.0, 98.0, {'price': 110.0})
    _check(r['pct'] == 0.0, f'ціна пішла проти → смужка 0%: {r}')
    _check(r['done_pct'] == -2.0, f'але сирий рух показує −2%: {r}')
    r2 = tp.progress('LONG', 100.0, 120.0, {'price': 110.0})
    _check(r2['pct'] == 100.0, f'перелетіли ціль → смужка 100%: {r2}')
    print('✓ смужка обрізана [0..100], сирий рух зберігається')


def test_progress_safe_without_data():
    for args in ((None, 100.0, {'price': 110.0}), (100.0, None, {'price': 110.0}),
                 (100.0, 100.0, None), (100.0, 100.0, {}),
                 (100.0, 100.0, {'price': 100.0})):
        _check(tp.progress('LONG', *args) is None,
               f'без достатніх даних прогрес не вигадуємо: {args}')
    print('✓ немає даних → прогрес None (нічого не вигадуємо)')


# ═════════ 🎯 TP-1 / TP-2 — аналітика поділу фіксації ══════════════════════
def _tg(price=100.0):
    return tp.collect_targets('LONG', price, swing=SWING_LONG,
                              runway=RUNWAY_UP, poc=105.0)


def test_tp2_is_the_main_objective_tp1_is_the_nearest_obstacle():
    """TP-2 — головна ціль (найдальша), TP-1 — найближча змістовна перепона
    МІЖ входом і нею. Саме там рух статистично найчастіше сповільнюється."""
    r = tp.plan_targets('LONG', 100.0, 100.0, _tg(), stop=98.0)
    _check(r['tp2']['price'] == 110.0, f"TP-2 = головна ціль: {r['tp2']}")
    _check(r['tp1']['price'] == 103.0, f"TP-1 = найближча перепона: {r['tp1']}")
    _check(r['tp1']['from_entry_pct'] == 3.0, f"відстань від входу: {r['tp1']}")
    _check(r['tp1']['r'] == 1.5 and r['tp2']['r'] == 5.0,
           f"R-кратність має рахуватись від стопа: {r['tp1']['r']} / {r['tp2']['r']}")
    print(f"✓ TP-1 {r['tp1']['price']} ({r['tp1']['r']}R) → "
          f"TP-2 {r['tp2']['price']} ({r['tp2']['r']}R)")


def test_tp_levels_are_strictly_ordered():
    r = tp.plan_targets('LONG', 100.0, 100.0, _tg())
    _check(100.0 < r['tp1']['price'] < r['tp2']['price'],
           f"порядок вхід < TP-1 < TP-2 обовʼязковий: {r}")
    rs = tp.plan_targets('SHORT', 100.0, 100.0,
                         tp.collect_targets('SHORT', 100.0,
                                            swing={'low': {'price': 90.0}},
                                            runway={'dir': 'SHORT',
                                                    'next': {'price': 97.0},
                                                    'main': {'price': 93.0}}))
    _check(100.0 > rs['tp1']['price'] > rs['tp2']['price'],
           f"для SHORT порядок дзеркальний: {rs}")
    print('✓ рівні строго впорядковані (і для SHORT дзеркально)')


def test_no_intermediate_target_means_no_tp1():
    """Проміжної цілі немає → TP-1 НЕ вигадуємо, працює лише TP-2."""
    only_far = [{'price': 110.0, 'kind': 'swing', 'label': 'Weak High'}]
    r = tp.plan_targets('LONG', 100.0, 100.0, only_far)
    _check(r['tp2'] and r['tp1'] is None, f'має бути лише TP-2: {r}')
    _check(any('проміжної цілі немає' in x for x in r['reasons']), f"{r['reasons']}")
    print('✓ немає проміжної цілі → лише TP-2, нічого не вигадуємо')


def test_levels_too_close_are_not_split():
    """Два TP впритул — поділ без сенсу: комісія зʼїсть частковий вихід."""
    tight = [{'price': 100.2, 'kind': 'poc', 'label': 'POC'},
             {'price': 100.5, 'kind': 'swing', 'label': 'Weak High'}]
    r = tp.plan_targets('LONG', 100.0, 100.0, tight, cfg={'tp_min_gap_pct': 0.4})
    _check(r['tp1'] is None, f'TP-1 не має зʼявитись впритул до TP-2: {r}')
    print('✓ рівні впритул не діляться (мін. зазор поважається)')


def test_target_closer_than_gap_gives_nothing():
    near = [{'price': 100.1, 'kind': 'poc', 'label': 'POC'}]
    r = tp.plan_targets('LONG', 100.0, 100.0, near, cfg={'tp_min_gap_pct': 0.4})
    _check(r['tp1'] is None and r['tp2'] is None,
           f'ціль ближче за зазор — це шум, TP не виставляємо: {r}')
    print('✓ ціль ближче за мін. зазор → TP не виставляються')


def test_objective_lock_is_respected_for_tp2():
    """TP-2 має відповідати ЗАФІКСОВАНІЙ цілі угоди, а не перераховуватись."""
    lock = {'price': 108.0, 'kind': 'liq_main', 'label': 'головний пул'}
    r = tp.plan_targets('LONG', 100.0, 104.0, _tg(104.0), objective=lock)
    _check(r['tp2']['price'] == 108.0, f'TP-2 = зафіксована ціль: {r["tp2"]}')
    print('✓ TP-2 бере зафіксовану ціль угоди')


def test_r_is_absent_without_a_stop():
    r = tp.plan_targets('LONG', 100.0, 100.0, _tg(), stop=None)
    _check(r['tp2']['r'] is None and r['tp1']['r'] is None,
           'без стопа R порахувати нема з чого — не вигадуємо')
    print('✓ без стопа R = None (не вигадуємо кратність)')


def test_plan_targets_safe_on_garbage():
    for bad_entry in (None, 0, -1, 'abc'):
        r = tp.plan_targets('LONG', bad_entry, 100.0, _tg())
        _check(r['tp1'] is None and r['tp2'] is None, f'сміття {bad_entry!r}')
    r2 = tp.plan_targets('WHAT', 100.0, 100.0, _tg())
    _check(r2['tp2'] is None, 'невідомий напрямок → нічого')
    r3 = tp.plan_targets('LONG', 100.0, 100.0, [])
    _check(r3['tp2'] is None, 'порожній список цілей → нічого')
    print('✓ сміттєві дані не ламають розкладку TP')


def test_tp_defaults_are_off():
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    _check("'pilot_autofill_tp': False" in src,
           'автозаповнення TP має бути ВИМКНЕНЕ за замовчуванням')
    _check("'pilot_tp1_close_pct': 50" in src, 'TP-1 закриває 50% за замовчуванням')
    _check(tp.DEFAULTS['tp_min_gap_pct'] > 0, 'мін. зазор має бути ненульовий')
    print('✓ дефолти: автозаповнення OFF, TP-1 = 50%, зазор > 0')


if __name__ == '__main__':
    test_targets_are_real_chart_objects_ahead_only()
    test_short_takes_levels_below_price()
    test_noise_and_too_far_levels_are_ignored()
    test_objective_is_the_furthest_meaningful_target()
    test_structure_stop_sits_behind_the_swing()
    test_broken_structure_yields_no_stop()
    test_ratchet_never_loosens_the_stop()
    test_plan_holds_while_target_is_ahead()
    test_plan_trails_stop_when_structure_improves()
    test_plan_takes_profit_at_the_objective()
    test_objective_does_not_drift_between_ticks()
    test_plan_does_not_loosen_an_existing_stop()
    test_breakeven_kicks_in_after_r_multiple()
    test_no_targets_still_holds_with_stop()
    test_completely_blind_context_is_safe()
    test_garbage_input_never_raises()
    test_defaults_are_conservative_and_off()
    test_progress_measures_path_from_entry_to_objective()
    test_progress_for_short_mirrors()
    test_progress_is_clamped_but_keeps_raw_move()
    test_progress_safe_without_data()
    test_tp2_is_the_main_objective_tp1_is_the_nearest_obstacle()
    test_tp_levels_are_strictly_ordered()
    test_no_intermediate_target_means_no_tp1()
    test_levels_too_close_are_not_split()
    test_target_closer_than_gap_gives_nothing()
    test_objective_lock_is_respected_for_tp2()
    test_r_is_absent_without_a_stop()
    test_plan_targets_safe_on_garbage()
    test_tp_defaults_are_off()
    print('\nУсі тести автопілота угоди пройдено ✅')
