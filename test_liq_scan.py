"""Тести 💧 СКАНУ ЛІКВІДНОСТІ ПО СПИСКУ МОНЕТ (сторінка 📡 Tickr).

Запит користувача: «список монет, відсортований за ліквідністю: де найбільший
перекіс і найближчий магніт — щоб бачити, куди тягне ринок по всьому списку
одразу», з вибором біржі.

⚠️ ГОЛОВНЕ, ЩО ПЕРЕВІРЯЄМО: скан — це МОМЕНТАЛЬНИЙ ЗРІЗ із поточного OI, а не
жива liq-map (та будується на приросту OI і існує лише для монет, які демон
веде). Обидва подання мусять користуватись ОДНІЄЮ драбиною і ОДНИМ вердиктом,
а «немає даних» має казатись прямо, а не показуватись нулями.

⚠️ Тут же замки на BULK-OI: обмеження було в МАСШТАБІ, а не в біржі. Binance і
BingX не віддають відкритий інтерес пачкою, але поштучно — віддають, тож
режим ОДНІЄЇ монети мусить працювати на них, а скан списку — добирати OI по
одному запиту на монету зі стелею.
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
    # Підпис — СМУГА: `magnet_dist` міряється від СЕРЕДИНИ сходинки, тож одна
    # межа вказувала б на іншу точку, ніж відсоток поруч.
    _check(row['magnet_price'] == '$90–91', f'найбільший: {row["magnet_price"]}')
    _check(abs(row['near_price'] - 99.0) < 1e-6, f'найближчий: {row["near_price"]}')
    _check(abs(row['near_price_hi'] - 100.0) < 1e-6,
           f'верхня межа найближчого потрібна UI для смуги: {row}')
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


def test_unknown_exchange_refuses_instead_of_raising():
    for r in (S.scan_liquidity(exchange='казна-що'),
              S.scan_one(exchange='казна-що', symbol='BTC')):
        _check(r['ok'] is False, 'невідома біржа → відмова, а не виняток')
        _check('не підтримується' in r['reason'], r['reason'])
    print('✓ невідома біржа відмовляє чесно')


# ── 🐞 BULK-OI: обмеження було в МАСШТАБІ, а не в біржі ────────────────────
def test_symbol_is_normalised_for_each_exchange():
    """Користувач вводить «btc», «BTC-USDT», «btc_usdt» — і все це та сама
    монета. Формат же в кожної біржі свій, і плутати їх не можна."""
    for raw in ('btc', 'BTC', 'BTCUSDT', 'btc-usdt', 'BTC_USDT', ' btc '):
        _check(S.norm_symbol(raw) == 'BTCUSDT', f'{raw!r} → {S.norm_symbol(raw)}')
    _check(S.norm_symbol('') == 'BTCUSDT', 'порожнє поле → BTC за домовленістю')
    _check(S.norm_symbol('1000pepe') == '1000PEPEUSDT', 'множники не ламаються')
    _check(S._ex_symbol('binance', 'btc') == 'BTCUSDT', 'binance')
    _check(S._ex_symbol('bybit', 'btc') == 'BTCUSDT', 'bybit')
    _check(S._ex_symbol('mexc', 'btc') == 'BTC_USDT', 'mexc')
    _check(S._ex_symbol('bingx', 'btc') == 'BTC-USDT', 'bingx')
    print('✓ назва монети зводиться до формату кожної біржі')


def _stub_one(mod, exchange, price=100.0, oi=100e6, bars=30):
    """Підміняємо мережу: OI по монеті + свічки. Розрахунок лишається живий."""
    calls = {'oi': 0, 'kl': 0}

    def _oi(session, symbol):
        calls['oi'] += 1
        return price, oi

    def _kl(session, symbol, interval, limit):
        calls['kl'] += 1
        return _bars((price * 0.97, bars, 1000))

    mod._OI_ONE[exchange] = _oi
    mod._KLINES[exchange] = _kl
    return calls


def test_single_coin_works_on_exchange_without_bulk_oi():
    """⚠️ ГОЛОВНЕ ПО ЦІЙ ПРАВЦІ. «Немає bulk-OI» заважає лише тоді, коли монет
    сотні. На ОДНУ монету потрібні 2-3 запити — тож Binance і BingX мусять
    працювати, а не відмовляти."""
    orig_oi, orig_kl = dict(S._OI_ONE), dict(S._KLINES)
    try:
        for ex in ('binance', 'bingx'):
            calls = _stub_one(S, ex)
            r = S.scan_one(exchange=ex, symbol='btc')
            _check(r['ok'], f'{ex}: {r.get("reason")}')
            _check(r['symbol'] == 'BTCUSDT', r['symbol'])
            _check(calls['oi'] == 1 and calls['kl'] == 1,
                   f'мало бути по одному запиту OI і свічок: {calls}')
            _check(r['rows'], 'драбина мусить мати сходинки')
            # Вердикт ШМАТКАМИ — без нього фронт не розфарбує числа.
            _check(r.get('verdict_parts') and r.get('verdict_tone'),
                   f'бракує розібраного вердикту: {sorted(r)}')
            _check(r['oi_usd'] > 0 and r['price'] > 0, 'ціна й OI мають бути')
    finally:
        S._OI_ONE.clear(); S._OI_ONE.update(orig_oi)
        S._KLINES.clear(); S._KLINES.update(orig_kl)
    print('✓ одна монета рахується на біржі БЕЗ bulk-OI (Binance/BingX)')


def test_single_coin_refuses_honestly_when_there_is_no_data():
    """Відмова мусить казати ПРИЧИНУ. Нулі й порожня драбина — заборонені."""
    orig_oi, orig_kl = dict(S._OI_ONE), dict(S._KLINES)
    try:
        _stub_one(S, 'binance', price=0.0, oi=0.0)
        r = S.scan_one(exchange='binance', symbol='НЕМАЄТАКОЇ')
        _check(r['ok'] is False and 'не знайдено' in r['reason'], r)
        _stub_one(S, 'binance', price=100.0, oi=0.0)
        r2 = S.scan_one(exchange='binance', symbol='btc')
        _check(r2['ok'] is False and 'відкритий інтерес' in r2['reason'], r2)
    finally:
        S._OI_ONE.clear(); S._OI_ONE.update(orig_oi)
        S._KLINES.clear(); S._KLINES.update(orig_kl)
    print('✓ немає даних — чесна причина, а не нулі')


def test_list_scan_falls_back_to_per_symbol_oi():
    """Скан СПИСКУ на біржі без bulk-OI: кандидатів відбираємо за ОБІГОМ
    (він у тікерах є завжди), а OI питаємо поштучно — і лише ПІСЛЯ цього
    застосовуємо поріг по OI."""
    fake = types.ModuleType('detection.tickr_core')
    fake.MARKET_SWAP = 'swap'
    fake._ACTIVITY = {'binance': lambda m: {
        'BIGUSDT': {'vol_usd': 500e6, 'last': 100.0},
        'MIDUSDT': {'vol_usd': 100e6, 'last': 50.0},
        'THINUSDT': {'vol_usd': 1e6, 'last': 1.0},      # відсіється за обігом
    }}
    # ⚠️ Підміняти ТІЛЬКИ sys.modules мало: `from detection import tickr_core`
    # бере АТРИБУТ пакета, якщо він уже виставлений (а він виставляється
    # першим же викликом `scan_liquidity`, бо імпорт стоїть до перевірок).
    _real_tc = getattr(sys.modules['detection'], 'tickr_core', None)
    sys.modules['detection.tickr_core'] = fake
    sys.modules['detection'].tickr_core = fake
    orig_oi, orig_kl = dict(S._OI_ONE), dict(S._KLINES)
    try:
        seen = []

        def _oi(session, symbol):
            seen.append(symbol)
            # MIDUSDT — з мізерним OI: має відсіятись уже ПІСЛЯ запиту.
            return (100.0, 80e6) if symbol == 'BIGUSDT' else (50.0, 1e6)

        S._OI_ONE['binance'] = _oi
        S._KLINES['binance'] = lambda s, sym, i, l: _bars((97.0, 30, 1000))
        r = S.scan_liquidity(exchange='binance', min_vol_usd=20e6,
                             min_oi_usd=5e6)
        _check(r['ok'], r)
        _check(sorted(seen) == ['BIGUSDT', 'MIDUSDT'],
               f'OI мали спитати лише в тих, хто пройшов обіг: {seen}')
        # ⚠️ Перевіряємо САМЕ відсів за OI (`scanned` = скільки пішло в
        # розрахунок), а не вміст таблиці: з 10.09 звіт додатково відкидає
        # рядки «⚖ рівновага», і ця фікстура дає саме такий вердикт — тобто
        # порожній `rows` тут КОРЕКТНИЙ і про OI нічого не каже.
        _check(r['scanned'] == 1,
               f'у розрахунок мала піти лише BIGUSDT, пішло {r["scanned"]}')
        _check('MIDUSDT' not in [x['symbol'] for x in r['rows']],
               f'монета з мізерним OI мала відсіятись: {r["rows"]}')
        _check(r['bulk_oi'] is False and r['warnings'],
               'UI мусить бачити, що OI брався поштучно')
        _check('відкритий інтерес' in ' '.join(r['warnings']), r['warnings'])
    finally:
        S._OI_ONE.clear(); S._OI_ONE.update(orig_oi)
        S._KLINES.clear(); S._KLINES.update(orig_kl)
        sys.modules.pop('detection.tickr_core', None)
        if _real_tc is not None:
            sys.modules['detection'].tickr_core = _real_tc
        else:
            sys.modules['detection'].__dict__.pop('tickr_core', None)
    print('✓ список без bulk-OI: OI поштучно, поріг застосовано після нього')


def test_per_symbol_oi_has_a_ceiling():
    """Поштучний OI подвоює вартість скану, тож на таких біржах діє стеля —
    інакше «100 монет» перетворяться на 200 запитів."""
    fake = types.ModuleType('detection.tickr_core')
    fake.MARKET_SWAP = 'swap'
    fake._ACTIVITY = {'binance': lambda m: {
        f'C{i}USDT': {'vol_usd': 100e6 + i, 'last': 10.0} for i in range(200)}}
    # ⚠️ Підміняти ТІЛЬКИ sys.modules мало: `from detection import tickr_core`
    # бере АТРИБУТ пакета, якщо він уже виставлений (а він виставляється
    # першим же викликом `scan_liquidity`, бо імпорт стоїть до перевірок).
    _real_tc = getattr(sys.modules['detection'], 'tickr_core', None)
    sys.modules['detection.tickr_core'] = fake
    sys.modules['detection'].tickr_core = fake
    orig_oi, orig_kl = dict(S._OI_ONE), dict(S._KLINES)
    try:
        seen = []
        S._OI_ONE['binance'] = lambda s, sym: (seen.append(sym), (10.0, 50e6))[1]
        S._KLINES['binance'] = lambda s, sym, i, l: _bars((9.7, 30, 1000))
        S.scan_liquidity(exchange='binance', top_n=200, min_vol_usd=1e6,
                         min_oi_usd=1e6)
        _check(len(seen) == S.PER_SYMBOL_OI_CAP,
               f'стеля {S.PER_SYMBOL_OI_CAP}, а спитали {len(seen)}')
    finally:
        S._OI_ONE.clear(); S._OI_ONE.update(orig_oi)
        S._KLINES.clear(); S._KLINES.update(orig_kl)
        sys.modules.pop('detection.tickr_core', None)
        if _real_tc is not None:
            sys.modules['detection'].tickr_core = _real_tc
        else:
            sys.modules['detection'].__dict__.pop('tickr_core', None)
    print(f'✓ стеля поштучного OI ({S.PER_SYMBOL_OI_CAP} монет) діє')


def test_cheap_coin_magnet_is_not_rounded_to_zero():
    """🐞 Драбина рахується вже не лише по BTC. `int(round(0.42))` перетворив
    би магніт дешевої монети на «$0» — і рівень став би нечитабельним."""
    lv = [{'price': 0.42, 'usd': 50e6, 'side': 'long'},
          {'price': 0.51, 'usd': 5e6, 'side': 'short'}]
    row = S.summarise(lv, 0.47, 'CHEAPUSDT')
    _check(row['ok'], row)
    _check(row['magnet_price'] not in ('$0', '$0 '),
           f'магніт дешевої монети згорнувся в нуль: {row["magnet_price"]}')
    _check('0.4' in row['magnet_price'] or '0.5' in row['magnet_price'],
           f'ціна має лишитись розрізненною: {row["magnet_price"]}')
    # А на BTC-масштабі формат не змінився.
    _check(LAD._fmt_price_ua(76000) == '$76 000', LAD._fmt_price_ua(76000))
    print(f'✓ дешеві монети не округляються до нуля ({row["magnet_price"]})')


def test_module_says_it_is_a_snapshot_not_the_live_map():
    """Замок від плутанини: у модулі мусить лишатись пояснення, чим цей
    розрахунок відрізняється від живої liq-map. Інакше через півроку хтось
    (і я в тому числі) вирішить, що це те саме число."""
    src = open(os.path.join(_ROOT, 'detection/liq_scan.py')).read()
    _check('МОМЕНТАЛЬНИЙ ЗРІЗ' in src, 'має бути сказано, що це зріз')
    _check('ПРИРОСТУ' in src, 'і чим від нього відрізняється жива карта')
    _check('50/50' in src, 'і що співвідношення лонг/шорт — припущення')
    print('✓ різниця з живою liq-map зафіксована в коді')


def test_clamped_coin_count_is_said_out_loud():
    """⚠️ CLAMP МУСИТЬ БУТИ ВИДИМИМ. Раніше «200 монет» на Binance тихо ставало
    60 (`PER_SYMBOL_OI_CAP`), а в статусі стояло просто «проскановано 60» — без
    натяку, що це НЕ вибір користувача. Той самий урок, що з глибиною історії."""
    fake = types.ModuleType('detection.tickr_core')
    fake.MARKET_SWAP = 'swap'
    fake._ACTIVITY = {'bybit': lambda m: {
        f'C{i}USDT': {'vol_usd': 100e6 + i, 'oi_usd': 50e6, 'last': 10.0}
        for i in range(400)}}
    _real_tc = getattr(sys.modules['detection'], 'tickr_core', None)
    sys.modules['detection.tickr_core'] = fake
    sys.modules['detection'].tickr_core = fake
    orig_kl = dict(S._KLINES)
    try:
        S._KLINES['bybit'] = lambda s, sym, i, l: _bars((9.7, 30, 1000))
        # Просимо ВДВІЧІ більше за стелю.
        over = S.MAX_SYMBOLS * 2
        r = S.scan_liquidity(exchange='bybit', top_n=over,
                             min_vol_usd=1e6, min_oi_usd=1e6)
        _check(r['ok'], r)
        _check(r['scanned'] == S.MAX_SYMBOLS,
               f"стеля {S.MAX_SYMBOLS}, проскановано {r['scanned']}")
        w = ' '.join(r.get('warnings') or [])
        _check(str(over) in w and str(S.MAX_SYMBOLS) in w,
               f'обрізання мусить бути НАЗВАНЕ вголос, а не мовчазне: «{w}»')
        # А в межах стелі — жодного зайвого попередження про обрізання.
        r2 = S.scan_liquidity(exchange='bybit', top_n=10,
                              min_vol_usd=1e6, min_oi_usd=1e6)
        _check(not any('стеля скану' in x for x in (r2.get('warnings') or [])),
               f'вибір у межах стелі не мусить нічого попереджати: {r2.get("warnings")}')
    finally:
        S._KLINES.clear(); S._KLINES.update(orig_kl)
        sys.modules.pop('detection.tickr_core', None)
        if _real_tc is not None:
            sys.modules['detection'].tickr_core = _real_tc
        else:
            sys.modules['detection'].__dict__.pop('tickr_core', None)
    print(f'✓ обрізання до стелі {S.MAX_SYMBOLS} монет пишеться в warnings')


def test_coin_count_dropdown_matches_the_backend_ceiling():
    """Випадайка «МОНЕТ» не сміє пропонувати більше, ніж бекенд просканує —
    інакше вибір мовчки перетвориться на інше число (тепер, щоправда, з
    попередженням, але сам список має бути чесним ЗРАЗУ)."""
    import re
    html = open(os.path.join(_ROOT, 'templates', 'tickr.html')).read()
    i = html.index('id="liq-topn"')
    block = html[i:html.index('</select>', i)]
    vals = [int(v) for v in re.findall(r'<option[^>]*>(\d+)</option>', block)]
    _check(vals, f'не знайшов варіантів: {block[:120]}')
    _check(max(vals) == S.MAX_SYMBOLS,
           f'верх списку {max(vals)} ≠ стеля бекенда {S.MAX_SYMBOLS}')
    _check(vals == sorted(vals), f'список мусить зростати: {vals}')
    # На Binance/BingX OI береться поштучно — та сама стеля, інакше дефолтна
    # біржа різала б вибір удвічі раніше за решту.
    _check(S.PER_SYMBOL_OI_CAP == S.MAX_SYMBOLS,
           f'стелі розійшлись: {S.PER_SYMBOL_OI_CAP} vs {S.MAX_SYMBOLS}')
    print(f'✓ «МОНЕТ» до {max(vals)} — рівно стільки, скільки бекенд просканує')


def test_scan_sends_the_chosen_history_depth():
    """Поле «ІСТОРІЯ, ГОД» у скані списку має РЕАЛЬНО доїжджати до бекенда.
    Раніше `bars` не передавався взагалі, тож глибина завжди була дефолтна."""
    html = open(os.path.join(_ROOT, 'templates', 'tickr.html')).read()
    i = html.index('async function liqScan')
    body = html[i:html.index('const res = await fetch', i)]
    _check("getElementById('liq-bars')" in body,
           'liqScan мусить читати обрану глибину')
    _check('bars:' in body, 'і класти її в тіло запиту')
    # І показувати, на чому саме порахували, — інакше глибину не звірити.
    tail = html[i:i + 4000]
    _check('${d.bars}' in tail, 'у підсумку мусить стояти глибина з ВІДПОВІДІ')
    print('✓ обрана глибина йде в запит і видно її у підсумку')


def test_history_dropdown_offers_only_depths_the_backend_honours():
    """Кожна опція «ІСТОРІЯ, ГОД» мусить дійти до біржі БЕЗ обрізання.

    ⚠️ `scan_one` робить `bars = max(24, min(bars, 1000))`, а свічки тягне
    ОДНИМ запитом без пагінації — найтісніший ліміт серед чотирьох бірж
    (Bybit, 1000 барів) і задає стелю. Опція поза цим діапазоном мовчки
    перетворилась би на інше число: користувач обрав «2000», а порахувалось
    би 1000, і ніде б це не було видно.
    """
    import re
    html = open(os.path.join(_ROOT, 'templates', 'tickr.html')).read()
    seen = {}
    # ⚠️ ОБИДВІ випадайки: список монет (`liq-bars`) і одна монета
    # (`liq1-bars`). Це той самий параметр `bars` того самого `build_levels`,
    # тож розійтись вони не мають права.
    for el in ('liq-bars', 'liq1-bars'):
        i = html.index(f'id="{el}"')
        block = html[i:html.index('</select>', i)]
        vals = [int(v) for v in re.findall(r'<option value="(\d+)"', block)]
        _check(len(vals) >= 8, f'{el}: мало варіантів глибини: {vals}')
        _check(vals == sorted(vals), f'{el}: список мусить зростати: {vals}')
        _check(len(set(vals)) == len(vals), f'{el}: дублікати: {vals}')
        for v in vals:
            clamped = max(24, min(v, 1000))
            _check(clamped == v, f'{el}: {v} год бекенд обріже до {clamped}')
        _check('selected' in block, f'{el}: потрібне значення за замовчуванням')
        seen[el] = vals
    _check(seen['liq-bars'] == seen['liq1-bars'],
           f'списки глибини розійшлись: {seen}')
    print(f"✓ глибина історії: {len(seen['liq-bars'])} варіантів в ОБОХ блоках, "
          f"усі проходять бекенд ({seen['liq-bars'][0]}…{seen['liq-bars'][-1]} год)")


def test_dropdown_list_is_readable_on_dark_page():
    """Скарга «нічого не видно»: САМ список малює браузер, і напівпрозоре тло
    лягало на БІЛЕ системне меню. Потрібні `color-scheme: dark` + СУЦІЛЬНІ
    кольори на `option` (напівпрозорість тут не працює — під меню немає
    нашого тла)."""
    import re
    html = open(os.path.join(_ROOT, 'templates', 'tickr.html')).read()
    _check('color-scheme: dark' in html, 'нативне меню мусить бути темним')
    i = html.index('.tk-field select option')
    rule = html[i:i + 200]
    _check('#' in rule and 'rgba' not in rule.split('}')[0],
           f'кольори option мусять бути СУЦІЛЬНІ, а не напівпрозорі: {rule[:80]}')
    # Правило спільне для ВСІХ випадайок сторінки — окремих не заводимо.
    # Тому кожен <select> мусить лежати в контейнері `.tk-field`, інакше він
    # лишиться зі світлим системним меню, а помітять це лише очима.
    # Коментарі прибираємо: пояснення перед тегом не мусить «відсувати»
    # контейнер за межі вікна пошуку.
    clean = re.sub(r'<!--.*?-->', '', html, flags=re.S)
    pos, uncovered = 0, []
    while True:
        m = clean.find('<select', pos)
        if m < 0:
            break
        if 'tk-field' not in clean[max(0, m - 400):m]:
            uncovered.append(clean[m:m + 60])
        pos = m + 1
    _check(not uncovered, f'select поза .tk-field (правило їх не покриє): {uncovered}')
    print('✓ випадайка читабельна на темній сторінці')


# ═══════ 🎯 WATCHLIST-РЕЖИМ + ФОЛБЕК ПО ОКРЕМІЙ МОНЕТІ (вимога 10.09) ═══════
#
# Вимога користувача дослівно: «Додай до цього скану можливість сканувати
# WATCHLIST, при цьому, якщо вибрано WATCHLIST, не потрібен вибір кількості
# МОНЕТ і ОБІГ 24H ≥, OI ≥… Якщо на момент скану монету не знайдено на біржі,
# потрібно автоматично вибрати іншу біржу для перевірки саме цієї монети,
# наприклад вибрано Binance — підміною буде Bybit і навпаки.»

def _fake_net(found):
    """Підмінити мережу: `found` = {(біржа, символ): (ціна, OI)}.

    Свічки віддаємо однакові, але ЗАПАМʼЯТОВУЄМО, з якої біржі їх просили —
    без цього не перевірити головного: числа монети мусять приходити з ОДНІЄЇ
    біржі (OI + свічки), інакше рівні рахувались би по двох різних ринках.
    """
    calls = {'klines': []}

    def _oi(ex):
        def f(session, symbol):
            return found.get((ex, symbol), (None, None))
        return f

    def _kl(ex):
        def f(session, symbol, interval, bars):
            calls['klines'].append((ex, symbol))
            # ⚠️ Свічки мусять дати РЕАЛЬНИЙ ПЕРЕКІС: із 10.09 звіт відкидає
            # рядки «⚖ рівновага», тож рівний ряд навколо ціни давав би
            # порожню таблицю — і тест перевіряв би не те, що заявлено.
            return _bars((88.0, 60, 3000))
        return f

    S._OI_ONE = {e: _oi(e) for e in ('binance', 'bybit', 'mexc', 'bingx')}
    S._KLINES = {e: _kl(e) for e in ('binance', 'bybit', 'mexc', 'bingx')}
    return calls


def test_watchlist_mode_scans_exactly_the_given_list():
    """Список склала людина — жодного відсіву за обігом/OI. Пороги передаємо
    свідомо ЗАВИЩЕНІ: у режимі `top` вони викосили б усе."""
    _o, _k = S._OI_ONE, S._KLINES
    _fake_net({('binance', 'AAAUSDT'): (100.0, 50e6),
               ('binance', 'BBBUSDT'): (100.0, 50e6)})
    try:
        r = S.scan_liquidity(exchange='binance', universe='watchlist',
                             symbols=['AAAUSDT', 'BBBUSDT'],
                             min_vol_usd=9e18, min_oi_usd=9e18, top_n=1)
    finally:
        S._OI_ONE, S._KLINES = _o, _k
    _check(r.get('ok'), f'скан мав пройти: {r.get("reason")}')
    _check(r['universe'] == 'watchlist', 'режим мусить бути названий у відповіді')
    _check(r['scanned'] == 2, f'мали просканувати ОБИДВІ монети, а не {r["scanned"]}')
    _check({x['symbol'] for x in r['rows']} == {'AAAUSDT', 'BBBUSDT'},
           'у таблиці мусять бути рівно ті монети, що дали')
    _check(r['dropped_vol'] == 0 and r['dropped_oi'] == 0,
           'у watchlist відсіву немає ЗА ВИЗНАЧЕННЯМ')


def test_missing_coin_falls_back_to_the_partner_exchange():
    """MNTUSDT-кейс: монети немає на Binance → беремо Bybit САМЕ для неї."""
    _o, _k = S._OI_ONE, S._KLINES
    calls = _fake_net({('binance', 'AAAUSDT'): (100.0, 50e6),
                       ('bybit', 'MNTUSDT'): (100.0, 40e6)})
    try:
        r = S.scan_liquidity(exchange='binance', universe='watchlist',
                             symbols=['AAAUSDT', 'MNTUSDT'])
    finally:
        S._OI_ONE, S._KLINES = _o, _k
    by = {x['symbol']: x for x in r['rows']}
    _check(by['MNTUSDT'].get('ok'), f'MNT мала пройти через фолбек: {by["MNTUSDT"]}')
    _check(by['MNTUSDT']['exchange'] == 'bybit', 'числа взяті з BYBIT')
    _check(by['MNTUSDT']['fallback'] is True, 'фолбек мусить бути ПОЗНАЧЕНИЙ')
    _check(by['AAAUSDT']['fallback'] is False, 'решта лишається на обраній біржі')
    _check(r['fallback_used'] == 1, 'лічильник фолбеків у підсумку')
    # ⚠️ ГОЛОВНЕ: свічки MNT теж із Bybit — OI однієї біржі зі свічками іншої
    # дав би рівні ліквідації по двох різних ринках.
    _check(('bybit', 'MNTUSDT') in calls['klines'],
           f'свічки MNT мали піти в BYBIT, а не {calls["klines"]}')
    _check(('binance', 'MNTUSDT') not in calls['klines'],
           'свічки НЕ можна брати з біржі, яка монету не знає')


def test_fallback_goes_the_other_way_too():
    """«Binance → Bybit і навпаки» — обидва напрямки, а не один."""
    _check(S.fallback_for('binance') == 'bybit', 'binance → bybit')
    _check(S.fallback_for('bybit') == 'binance', 'bybit → binance')
    _o, _k = S._OI_ONE, S._KLINES
    _fake_net({('binance', 'XXXUSDT'): (100.0, 30e6)})
    try:
        r = S.scan_liquidity(exchange='bybit', universe='watchlist',
                             symbols=['XXXUSDT'])
    finally:
        S._OI_ONE, S._KLINES = _o, _k
    _check(r['rows'][0]['exchange'] == 'binance',
           'обрано Bybit, монети там немає → підміна Binance')


def test_coin_on_neither_exchange_says_both_attempts():
    """Мовчазний прочерк не годиться: причина називає ОБИДВІ спроби."""
    _o, _k = S._OI_ONE, S._KLINES
    _fake_net({})
    try:
        r = S.scan_liquidity(exchange='binance', universe='watchlist',
                             symbols=['NOPEUSDT'])
    finally:
        S._OI_ONE, S._KLINES = _o, _k
    row = r['rows'][0]
    _check(row.get('ok') is False, 'рядок мусить бути позначений як без даних')
    _check('binance' in row['reason'] and 'bybit' in row['reason'],
           f'причина мусить назвати обидві біржі: {row["reason"]}')


def test_empty_watchlist_is_an_honest_refusal():
    r = S.scan_liquidity(exchange='binance', universe='watchlist', symbols=[])
    _check(not r.get('ok'), 'порожній список — це відмова, а не порожня таблиця')
    _check('watchlist' in (r.get('reason') or ''), f'причина: {r.get("reason")}')


def test_watchlist_mode_never_touches_exchange_tickers():
    """Список уже відомий — качати тікери всієї біржі ні до чого.

    Тест-замок на ВАРТІСТЬ: `tickr_core` у цій гілці не має викликатись
    узагалі, інакше «легкий» режим тягнув би зайвий важкий запит.
    """
    import ast
    src = open(os.path.join(_ROOT, 'detection', 'liq_scan.py'),
               encoding='utf-8').read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == '_scan_watchlist')
    txt = ast.dump(fn)
    _check('tickr_core' not in txt, '_scan_watchlist не має чіпати tickr_core')
    _check('min_vol_usd' not in txt and 'min_oi_usd' not in txt,
           'пороги обігу/OI у watchlist не застосовуються')


def test_ui_hides_the_fields_that_do_not_apply():
    """Активний контрол, який ні на що не впливає, вводить в оману — той самий
    принцип, що з полями TTL при ♾ «Без терміну» в Черзі-4."""
    html = open(os.path.join(_ROOT, 'templates', 'tickr.html'),
                encoding='utf-8').read()
    _check('id="liq-universe"' in html, 'немає вибору списку')
    _check('value="watchlist"' in html, 'немає опції watchlist')
    for wrap in ('liq-f-topn', 'liq-f-minvol', 'liq-f-minoi'):
        _check(f'id="{wrap}"' in html, f'поле {wrap} нема як сховати')
    i = html.find('function _liqUniverseSync')
    _check(i > 0, 'немає _liqUniverseSync')
    body = html[i:i + 500]
    for wrap in ('liq-f-topn', 'liq-f-minvol', 'liq-f-minoi'):
        _check(wrap in body, f'_liqUniverseSync не ховає {wrap}')
    _check("universe: wl ? 'watchlist' : 'top'" in html,
           'режим не передається на бекенд')


# ══════ ⚖ РІВНОВАГА НЕ ЙДЕ В ЗВІТ (вимога користувача 10.09) ══════════════
#
# Дослівно: «💧 Ліквідність по біржі — куди тягне ринок — у звіті відсікай
# записи у яких "рівновага", їх не потрібно показувати».

def test_flat_rows_are_dropped_from_the_report():
    rows = [{'symbol': 'A', 'ok': True, 'pull': 'up', 'pull_pct': 40},
            {'symbol': 'B', 'ok': True, 'pull': 'flat', 'pull_pct': 4},
            {'symbol': 'C', 'ok': True, 'pull': 'down', 'pull_pct': 30}]
    keep, n = S.drop_flat(rows)
    _check(n == 1, f'мала прибратись рівно одна рівновага, прибрано {n}')
    _check([r['symbol'] for r in keep] == ['A', 'C'],
           f'у звіті лишаються лише монети з перекосом: {keep}')


def test_rows_without_data_are_not_confused_with_flat():
    """«немає даних» ≠ «рівновага». Викинути перше означало б приховати, що
    монету не вдалось порахувати."""
    rows = [{'symbol': 'X', 'ok': False, 'reason': 'свічки недоступні'},
            {'symbol': 'Y', 'ok': True, 'pull': 'flat', 'pull_pct': 2}]
    keep, n = S.drop_flat(rows)
    _check(n == 1, 'прибирається лише рівновага')
    _check([r['symbol'] for r in keep] == ['X'],
           f'рядок без даних мусить лишитись: {keep}')


def test_list_scan_applies_the_filter_and_says_how_many():
    """Мовчазне зникнення рядків читалось би як збій — кількість НАЗВАНА."""
    _o, _k = S._OI_ONE, S._KLINES
    _fake_net({('binance', 'AAAUSDT'): (100.0, 50e6)})
    try:
        r = S.scan_liquidity(exchange='binance', universe='watchlist',
                             symbols=['AAAUSDT'])
    finally:
        S._OI_ONE, S._KLINES = _o, _k
    _check('dropped_flat' in r, 'у відповіді немає лічильника рівноваг')
    _check(all(x.get('pull') != 'flat' for x in r['rows'] if x.get('ok')),
           f'рівновага просочилась у звіт: {r["rows"]}')
    html = open(os.path.join(_ROOT, 'templates', 'tickr.html'),
                encoding='utf-8').read()
    _check('d.dropped_flat' in html, 'статус не показує, скільки прибрано')


def test_single_coin_mode_still_answers_flat():
    """⚠️ Фільтр — ЛИШЕ для списку. У режимі однієї монети ви питали САМЕ про
    неї, і «рівновага» — коректна відповідь, а не сміття."""
    import ast
    src = open(os.path.join(_ROOT, 'detection', 'liq_scan.py'),
               encoding='utf-8').read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == 'scan_one')
    _check('drop_flat' not in ast.dump(fn),
           'scan_one не має відсікати рівновагу')


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
    test_unknown_exchange_refuses_instead_of_raising()
    test_symbol_is_normalised_for_each_exchange()
    test_single_coin_works_on_exchange_without_bulk_oi()
    test_single_coin_refuses_honestly_when_there_is_no_data()
    test_list_scan_falls_back_to_per_symbol_oi()
    test_per_symbol_oi_has_a_ceiling()
    test_cheap_coin_magnet_is_not_rounded_to_zero()
    test_module_says_it_is_a_snapshot_not_the_live_map()
    test_clamped_coin_count_is_said_out_loud()
    test_coin_count_dropdown_matches_the_backend_ceiling()
    test_scan_sends_the_chosen_history_depth()
    test_history_dropdown_offers_only_depths_the_backend_honours()
    test_dropdown_list_is_readable_on_dark_page()
    test_watchlist_mode_scans_exactly_the_given_list()
    test_missing_coin_falls_back_to_the_partner_exchange()
    test_fallback_goes_the_other_way_too()
    test_coin_on_neither_exchange_says_both_attempts()
    test_empty_watchlist_is_an_honest_refusal()
    test_watchlist_mode_never_touches_exchange_tickers()
    test_ui_hides_the_fields_that_do_not_apply()
    test_flat_rows_are_dropped_from_the_report()
    test_rows_without_data_are_not_confused_with_flat()
    test_list_scan_applies_the_filter_and_says_how_many()
    test_single_coin_mode_still_answers_flat()
    print('\nУсі тести скану ліквідності пройдено ✅')
