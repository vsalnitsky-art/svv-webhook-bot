"""Тести 💧 ДРАБИНИ ЛІКВІДНОСТІ — рівні ліквідації у ВІДСОТКАХ.

Запит користувача: бот має збирати той самий зріз, що публікують у каналах
(«на $82k — стільки-то стопів, на $78k — стільки-то»), і показувати його
біля банера настрою.

⚠️ КЛЮЧОВЕ РІШЕННЯ: показуємо ЧАСТКИ, а не «кількість стоп-наказів».
Реальних стопів трейдерів не публікує жодна централізована біржа, тож будь-яке
таке число — чужа оцінка (долари ÷ припущений середній розмір позиції). Частка
від усієї ліквідності рахується з НАШИХ даних і нічого не вигадує.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg
_sub = types.ModuleType('detection.liquidation_map')
_sub.__path__ = [os.path.join(_ROOT, 'detection', 'liquidation_map')]
sys.modules['detection.liquidation_map'] = _sub

_spec = importlib.util.spec_from_file_location(
    'detection.liquidation_map.ladder',
    os.path.join(_ROOT, 'detection/liquidation_map/ladder.py'))
L = importlib.util.module_from_spec(_spec)
sys.modules['detection.liquidation_map.ladder'] = L
_spec.loader.exec_module(L)


def _check(c, m):
    if not c:
        raise AssertionError(m)


MARK = 80_000.0
# Кластери як на скріні: великий під ціною ($78k) і менший над нею.
LEVELS = [
    {'price': 78_200.0, 'usd': 6_000_000, 'side': 'long',  'age_min': 12},
    {'price': 78_600.0, 'usd': 2_000_000, 'side': 'long',  'age_min': 30},
    {'price': 82_100.0, 'usd': 1_500_000, 'side': 'short', 'age_min': 8},
    {'price': 84_400.0, 'usd': 500_000,   'side': 'short', 'age_min': 90},
]


def test_levels_collapse_into_steps():
    """Два рівні в межах однієї $1000-сходинки мають скластися в один рядок."""
    r = L.build_ladder(LEVELS, MARK, step_usd=1000)
    _check(r['ok'], r)
    step78 = [x for x in r['rows'] if x['price'] == 78_000.0]
    _check(len(step78) == 1, f'78 200 і 78 600 — та сама сходинка: {r["rows"]}')
    _check(step78[0]['usd'] == 8_000_000, f'суми мають додатись: {step78[0]}')
    _check(step78[0]['price_hi'] == 79_000.0, f'верхня межа: {step78[0]}')
    print(f"✓ рівні згортаються у кроки по $1000: {step78[0]['pct']}% на $78 000")


def test_percentages_are_shares_of_total():
    r = L.build_ladder(LEVELS, MARK, step_usd=1000)
    _check(abs(r['total_usd'] - 10_000_000) < 1, f'сума: {r["total_usd"]}')
    by = {x['price']: x['pct'] for x in r['rows']}
    _check(by[78_000.0] == 80.0, f'8M з 10M = 80%: {by}')
    _check(by[82_000.0] == 15.0, f'1.5M з 10M = 15%: {by}')
    print('✓ відсотки — це частки від УСІЄЇ ліквідності у вікні')


def test_split_above_below_and_pull():
    """Маса під ціною тягне ціну ВНИЗ — це головний висновок банера."""
    r = L.build_ladder(LEVELS, MARK, step_usd=1000)
    _check(r['below']['pct'] == 80.0 and r['above']['pct'] == 20.0, r)
    _check(r['pull'] == 'down', f'80% знизу → тягне вниз: {r["pull"]}')
    _check(r['pull_pct'] == 60.0, f'перевага 60 п.п.: {r["pull_pct"]}')
    print(f"✓ ⬆{r['above']['pct']}% / ⬇{r['below']['pct']}% → тягне {r['pull']}")


def test_balanced_market_is_not_called_a_direction():
    """⚠️ Перевага в межах 10 п.п. — це НЕ напрямок. Видавати 52/48 за «тягне
    вгору» означало б вигадувати сигнал там, де його немає."""
    even = [{'price': 81_000.0, 'usd': 1_040_000, 'side': 'short'},
            {'price': 79_000.0, 'usd': 960_000, 'side': 'long'}]
    r = L.build_ladder(even, MARK, step_usd=1000)
    _check(r['pull'] == 'flat', f'52/48 → рівновага: {r}')
    print('✓ невелика перевага не видається за напрямок')


def test_distance_is_measured_from_the_step_middle():
    r = L.build_ladder(LEVELS, MARK, step_usd=1000)
    row = [x for x in r['rows'] if x['price'] == 82_000.0][0]
    # середина сходинки 82 500 → (82500-80000)/80000 = 3.125%
    _check(abs(row['dist_pct'] - 3.125) < 0.02, f'відстань: {row}')
    _check(row['dir'] == 'up', f'над ціною: {row}')
    print(f"✓ відстань міряється від середини сходинки: {row['dist_pct']}%")


def test_noise_rows_are_dropped():
    noisy = LEVELS + [{'price': 90_000.0, 'usd': 10_000, 'side': 'short'}]
    r = L.build_ladder(noisy, MARK, step_usd=1000, min_row_pct=1.0)
    _check(not any(x['price'] == 90_000.0 for x in r['rows']),
           f'0.1% — це шум, у драбині йому не місце: {r["rows"]}')
    print('✓ дрібні рівні не забивають список')


def test_rows_sorted_by_share_and_capped():
    many = [{'price': 70_000.0 + i * 1000, 'usd': (i + 1) * 100_000,
             'side': 'long' if i < 10 else 'short'} for i in range(20)]
    r = L.build_ladder(many, MARK, step_usd=1000, top_n=5)
    _check(len(r['rows']) == 5, f'top_n має обмежувати: {len(r["rows"])}')
    pcts = [x['pct'] for x in r['rows']]
    _check(pcts == sorted(pcts, reverse=True), f'найбільша частка перша: {pcts}')
    print('✓ сортування за часткою + обмеження top_n')


def test_auto_step_scales_with_price():
    """На BTC крок ~$1000, на дешевій монеті — пропорційно менший, інакше вся
    драбина злилась би в одну сходинку."""
    _check(L.step_for(80_000.0) == 1000.0, L.step_for(80_000.0))
    _check(L.step_for(3_000.0) == 50.0, L.step_for(3_000.0))
    _check(L.step_for(0.5) <= 1.0, L.step_for(0.5))
    _check(L.step_for(80_000.0, 500) == 500.0, 'явний крок виграє')
    print('✓ авто-крок масштабується від ціни, явний крок має пріоритет')


def test_no_data_is_reported_not_faked():
    for bad in (None, [], [{'price': 0, 'usd': 5}], [{'price': 1, 'usd': 0}]):
        r = L.build_ladder(bad, MARK)
        _check(r['ok'] is False and r['rows'] == [],
               f'нема даних → чесно кажемо, а не малюємо нулі: {r}')
    r2 = L.build_ladder(LEVELS, 0)
    _check(r2['ok'] is False, 'немає ціни → нічого не рахуємо')
    r3 = L.build_ladder(LEVELS, None)
    _check(r3['ok'] is False, 'None-ціна не має ламати виклик')
    print('✓ відсутність даних повідомляється, а не імітується')


def test_garbage_levels_never_raise():
    junk = [{'price': 'abc', 'usd': 1}, {'usd': 5}, {'price': 78_000.0},
            {'price': 78_000.0, 'usd': 1e6, 'side': 'wat'}, None, 42]
    r = L.build_ladder(junk + LEVELS, MARK, step_usd=1000)
    _check(r['ok'], 'сміття серед валідних рівнів не має ламати драбину')
    _check(abs(r['total_usd'] - 10_000_000) < 1,
           f'у суму йдуть ЛИШЕ валідні рівні: {r["total_usd"]}')
    print('✓ сміттєві рівні ігноруються, підсумок лишається чесним')


def test_module_does_not_promise_stop_counts():
    """Замок від спокуси: у драбині НЕ має зʼявитись поле «кількість стопів».
    Це чужа оцінка з двох припущень, і видавати її за наші дані не можна."""
    r = L.build_ladder(LEVELS, MARK, step_usd=1000)
    banned = {'stops', 'stop_count', 'orders', 'count', 'traders'}
    for row in r['rows']:
        _check(not (banned & set(row.keys())),
               f'у рядку зʼявилось поле-вигадка: {row}')
    src = open(os.path.join(_ROOT, 'detection/liquidation_map/ladder.py')).read()
    _check('не публікує' in src,
           'у модулі має лишатись пояснення, ЧОМУ ми не рахуємо стопи')
    print('✓ драбина не вигадує «кількість стоп-наказів»')


if __name__ == '__main__':
    test_levels_collapse_into_steps()
    test_percentages_are_shares_of_total()
    test_split_above_below_and_pull()
    test_balanced_market_is_not_called_a_direction()
    test_distance_is_measured_from_the_step_middle()
    test_noise_rows_are_dropped()
    test_rows_sorted_by_share_and_capped()
    test_auto_step_scales_with_price()
    test_no_data_is_reported_not_faked()
    test_garbage_levels_never_raise()
    test_module_does_not_promise_stop_counts()
    print('\nУсі тести драбини ліквідності пройдено ✅')
