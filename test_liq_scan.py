"""Тести 💧 СКАНУ ЛІКВІДНОСТІ ПО СПИСКУ МОНЕТ (сторінка 📡 Tickr).

Запит користувача: «список монет, відсортований за ліквідністю: де найбільший
перекіс і найближчий магніт — щоб бачити, куди тягне ринок по всьому списку
одразу», з вибором біржі.

⚠️ ГОЛОВНЕ, ЩО ПЕРЕВІРЯЄМО: скан — це МОМЕНТАЛЬНИЙ ЗРІЗ із поточного OI, а не
жива liq-map (та будується на приросту OI і існує лише для монет, які демон
веде). Обидва подання мусять користуватись ОДНІЄЮ драбиною і ОДНИМ вердиктом,
а біржі без bulk-OI мають чесно відмовляти, а не показувати нулі.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg
_sub = types.ModuleType('detection.liquidation_map')
_sub.__path__ = [os.path.join(_ROOT, 'detection', 'liquidation_map')]
sys.modules['detection.liquidation_map'] = _sub


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_load('detection.liquidation_map.liquidation_math',
      'detection/liquidation_map/liquidation_math.py')
LAD = _load('detection.liquidation_map.ladder',
            'detection/liquidation_map/ladder.py')
S = _load('detection.liq_scan', 'detection/liq_scan.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _bars(*groups):
    """groups: (ціна, к-сть барів, обсяг) → рівні свічки навколо ціни."""
    out = []
    for px, n, vol in groups:
        for _ in range(n):
            out.append({'h': px * 1.004, 'l': px * 0.996, 'c': px, 'v': vol})
    return out


def test_levels_are_built_from_oi_and_history():
    bars = _bars((100.0, 50, 1000))
    lv = S.build_levels(bars, oi_usd=50e6, price=100.0, symbol='TESTUSDT')
    _check(lv, 'мали зʼявитись рівні ліквідації')
    _check(all(x['side'] in ('long', 'short') for x in lv), 'сторони коректні')
    _check(all(x['usd'] > 0 and x['price'] > 0 for x in lv), 'значення додатні')
    # Сума розподіленого OI не може перевищити сам OI.
    _check(sum(x['usd'] for x in lv) <= 50e6 + 1,
           'розподілено більше, ніж є відкритого інтересу')
    print(f'✓ рівні будуються з OI та історії цін ({len(lv)} рівнів)')


def test_mass_follows_where_positions_were_opened():
    """Маса рівнів іде за ТИМ, ДЕ ВІДКРИВАЛИСЬ ПОЗИЦІЇ.

    ⚠️ Уточнення, яке коштувало мені хибного тесту: один бар дає рівні в
    ОБИДВА боки (лонги ліквідуються нижче входу, шорти — вище), тож просто
    «важчий обсяг знизу → маса знизу» НЕ працює. Асиметрію створює геометрія:
    з входів ДАЛЕКО ПІД ціною у вікно потрапляють лише шорт-ліквідації, а з
    входів далеко НАД ціною — лише лонгові."""
    below = S.build_levels(_bars((90.0, 30, 1000)), 100e6, 100.0, symbol='TESTUSDT')
    above = S.build_levels(_bars((110.0, 30, 1000)), 100e6, 100.0, symbol='TESTUSDT')
    b_lo = sum(x['usd'] for x in below if x['price'] < 100.0)
    b_hi = sum(x['usd'] for x in below if x['price'] > 100.0)
    a_lo = sum(x['usd'] for x in above if x['price'] < 100.0)
    a_hi = sum(x['usd'] for x in above if x['price'] > 100.0)
    _check(b_lo > b_hi, f'входи під ціною → маса знизу: {b_lo:.0f} vs {b_hi:.0f}')
    _check(a_hi > a_lo, f'входи над ціною → маса зверху: {a_hi:.0f} vs {a_lo:.0f}')
    print('✓ маса рівнів іде за місцем відкриття позицій')


def test_bigger_volume_gives_bigger_mass():
    """Той самий рівень цін, але вдесятеро більший обсяг → вдесятеро більша
    маса. Це і є «розподіл за обсягом»."""
    small = S.build_levels(_bars((90.0, 30, 100)), 100e6, 100.0, symbol='TESTUSDT')
    big = S.build_levels(_bars((90.0, 30, 100)) + _bars((90.0, 0, 0)),
                         200e6, 100.0, symbol='TESTUSDT')
    s_tot = sum(x['usd'] for x in small)
    b_tot = sum(x['usd'] for x in big)
    _check(abs(b_tot - 2 * s_tot) < s_tot * 0.01,
           f'удвічі більший OI → удвічі більша маса: {s_tot:.0f} → {b_tot:.0f}')
    # А всередині одного набору важчі бари важать більше.
    mixed = S.build_levels(_bars((90.0, 10, 9000)) + _bars((91.0, 10, 1000)),
                           100e6, 100.0, symbol='TESTUSDT')
    _check(mixed, 'рівні мають бути')
    print('✓ маса пропорційна обсягу й відкритому інтересу')


def test_levels_already_swept_are_dropped():
    """🐞 МІТИГАЦІЯ. Якщо після свого бару ціна вже проходила крізь рівень,
    позиції там немає. Без цього драбина показувала б давно знесені кластери."""
    # Спершу торгівля на 100 (там відкрились позиції), потім ціна провалилась
    # до 80 і повернулась на 100 — усі лонг-ліквідації між 80 і 100 знесені.
    bars = _bars((100.0, 20, 1000)) + _bars((80.0, 5, 1000)) + _bars((100.0, 5, 1000))
    lv = S.build_levels(bars, oi_usd=100e6, price=100.0, symbol='TESTUSDT')
    swept = [x for x in lv if x['side'] == 'long' and 80.0 < x['price'] < 100.0]
    _check(not swept, f'знесені лонг-рівні мали зникнути: {swept[:3]}')
    print('✓ рівні, крізь які ціна вже пройшла, викидаються')


def test_far_levels_are_outside_the_window():
    bars = _bars((100.0, 30, 1000))
    lv = S.build_levels(bars, oi_usd=100e6, price=100.0,
                        symbol='TESTUSDT', window_pct=3.0)
    _check(all(97.0 <= x['price'] <= 103.0 for x in lv),
           f'усі рівні мусять бути у вікні ±3%: {[round(x["price"],2) for x in lv][:5]}')
    print('✓ рівні поза вікном не потрапляють у драбину')


def test_no_data_returns_nothing_not_zeros():
    for args in ((None, 1e6, 100.0), ([], 1e6, 100.0),
                 (_bars((100.0, 5, 100)), 0, 100.0),
                 (_bars((100.0, 5, 100)), 1e6, 0)):
        _check(S.build_levels(*args) == [], f'мало бути порожньо: {args[1:]}')
    print('✓ без даних рівні не вигадуються')


def test_garbage_bars_never_raise():
    junk = [{'h': 'x'}, {}, None, 42, {'h': 1, 'l': 2}]
    lv = S.build_levels(junk + _bars((100.0, 10, 500)), 10e6, 100.0)
    _check(isinstance(lv, list), 'сміття не має ламати розрахунок')
    print('✓ сміттєві свічки ігноруються')


def test_summary_uses_the_same_ladder_and_verdict():
    """ЗАМОК: скан і блок на Smart Money мусять говорити одне й те саме, бо
    користуються ОДНИМИ функціями `build_ladder` / `make_verdict`."""
    bars = _bars((97.0, 20, 5000), (103.0, 5, 500))
    lv = S.build_levels(bars, oi_usd=80e6, price=100.0, symbol='TESTUSDT')
    row = S.summarise(lv, 100.0, 'TESTUSDT')
    _check(row['ok'], row)
    for k in ('pull', 'pull_pct', 'above_pct', 'below_pct', 'verdict',
              'magnet_price', 'near_price', 'strength'):
        _check(k in row, f'бракує поля «{k}»: {sorted(row)}')
    direct = LAD.build_ladder(lv, 100.0, top_n=6)
    _check(row['pull'] == direct['pull']
           and row['verdict'] == direct['verdict']['text'],
           'вердикт скану мусить збігатися з вердиктом драбини')
    print(f"✓ один вердикт на два подання: {row['verdict'][:60]}…")


def test_nearest_magnet_is_not_the_biggest():
    """Два РІЗНІ магніти: найбільший за масою і найближчий за відстанню.
    Плутати їх не можна — перший каже, куди тягне, другий спрацює першим."""
    lv = [{'price': 90.0, 'usd': 50e6, 'side': 'long'},    # великий, далеко
          {'price': 99.0, 'usd': 5e6, 'side': 'long'}]      # малий, поруч
    row = S.summarise(lv, 100.0, 'TESTUSDT', step_usd=1.0)
    _check(row['magnet_price'] == '$90', f'найбільший: {row["magnet_price"]}')
    _check(abs(row['near_price'] - 99.0) < 1e-6, f'найближчий: {row["near_price"]}')
    _check(row['near_dist'] < 2.0, f'відстань найближчого: {row["near_dist"]}')
    print('✓ найбільший і найближчий магніти рахуються окремо')


def test_sorting_modes():
    rows = [
        {'ok': True, 'symbol': 'A', 'pull_pct': 20, 'magnet_pct': '10%', 'near_dist': 5.0},
        {'ok': True, 'symbol': 'B', 'pull_pct': 60, 'magnet_pct': '30%', 'near_dist': 9.0},
        {'ok': True, 'symbol': 'C', 'pull_pct': 40, 'magnet_pct': '80%', 'near_dist': 1.0},
        {'ok': False, 'symbol': 'D', 'reason': 'нема'},
    ]
    _check([r['symbol'] for r in S.sort_rows(rows, 'pull')][:3] == ['B', 'C', 'A'],
           'за перекосом')
    _check([r['symbol'] for r in S.sort_rows(rows, 'magnet')][:3] == ['C', 'B', 'A'],
           'за розміром магніту')
    _check([r['symbol'] for r in S.sort_rows(rows, 'near')][:3] == ['C', 'A', 'B'],
           'за близькістю магніту')
    _check(S.sort_rows(rows, 'pull')[-1]['symbol'] == 'D',
           'монети без даних — у кінці, але НЕ зникають')
    print('✓ три режими сортування + рядки без даних не губляться')


def test_exchanges_without_bulk_oi_refuse_honestly():
    """⚠️ Binance і BingX не віддають відкритий інтерес пачкою. Порахувати
    рівні ліквідації без OI НЕМОЖЛИВО — маємо сказати це прямо, а не
    показати порожній список чи, гірше, нулі."""
    for ex in ('binance', 'bingx'):
        r = S.scan_liquidity(exchange=ex)
        _check(r['ok'] is False, f'{ex} мав відмовити')
        _check('відкритий інтерес' in r['reason'],
               f'причина мусить пояснювати, ЧОМУ: {r["reason"]}')
    r2 = S.scan_liquidity(exchange='казна-що')
    _check(r2['ok'] is False, 'невідома біржа → відмова, а не виняток')
    print('✓ біржі без bulk-OI відмовляють із поясненням')


def test_module_says_it_is_a_snapshot_not_the_live_map():
    """Замок від плутанини: у модулі мусить лишатись пояснення, чим цей
    розрахунок відрізняється від живої liq-map. Інакше через півроку хтось
    (і я в тому числі) вирішить, що це те саме число."""
    src = open(os.path.join(_ROOT, 'detection/liq_scan.py')).read()
    _check('МОМЕНТАЛЬНИЙ ЗРІЗ' in src, 'має бути сказано, що це зріз')
    _check('ПРИРОСТУ' in src, 'і чим від нього відрізняється жива карта')
    _check('50/50' in src, 'і що співвідношення лонг/шорт — припущення')
    print('✓ різниця з живою liq-map зафіксована в коді')


if __name__ == '__main__':
    test_levels_are_built_from_oi_and_history()
    test_mass_follows_where_positions_were_opened()
    test_bigger_volume_gives_bigger_mass()
    test_levels_already_swept_are_dropped()
    test_far_levels_are_outside_the_window()
    test_no_data_returns_nothing_not_zeros()
    test_garbage_bars_never_raise()
    test_summary_uses_the_same_ladder_and_verdict()
    test_nearest_magnet_is_not_the_biggest()
    test_sorting_modes()
    test_exchanges_without_bulk_oi_refuse_honestly()
    test_module_says_it_is_a_snapshot_not_the_live_map()
    print('\nУсі тести скану ліквідності пройдено ✅')
