"""🖼 БЛОКИ НА ГРАФІКУ — ЯК У TRADINGVIEW (кейс BTCUSDT 07.09).

**Скарга користувача дослівно:** «В TradingView на графіку вже давно є на 1Н OB
SHORT, чому тут ще досі 1Н LONG?» і далі «чому бот не малює на графіку SHORT
OB 1Н, бо на графіку TradingView він вже є».

**Знайдено ДВІ причини, обидві — втрата даних, а не помилка розрахунку:**

1. `detect_last_order_block` віддавав РІВНО ОДИН блок — `internalOrderBlocks[0]`.
   Живий блок ПРОТИЛЕЖНОГО боку рахувався в тому ж проході й викидався на
   `return`. LuxAlgo ж малює ВСІ живі блоки одночасно.

2. `_update_smc_ob` рахував `detect_smc_structure(...)` з ОБОМА структурами —
   і брав лише `result['internal']`. Тобто **Swing Order Blocks** (`swing_size`,
   у LuxAlgo окрема сім'я з власним тумблером) бот не мав У ПРИНЦИПІ, хоча
   свінг-півоти й події вже були пораховані та лежали поруч.

⚠️ ВОРОТА ВХОДУ НЕ ЧІПАЛИ. `Require OB Match` / «1H OB лише з CHoCH» / такт
`vob_one_per_ob` і далі читають ОДИН поточний internal-блок із
`sob_smc_ob_state`. Зміна суто показова — і ці тести це стережуть.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


OB = _load('detection.ob_detector', 'detection/ob_detector.py')
ST = _load('detection.smc_structure', 'detection/smc_structure.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _bars(seq, t0=1_700_000_000_000, step=3_600_000):
    """seq: список (high, low, close) → бари у форматі детектора."""
    return [{'h': float(h), 'l': float(l), 'p': float(c), 't': t0 + i * step}
            for i, (h, l, c) in enumerate(seq)]


def _leg(px, target, steps, seq, amp=0.35):
    """Плавна нога від `px` до `target` за `steps` барів (повертає нову ціну).

    ⚠️ Пилка тут ОБОВʼЯЗКОВА, а не для краси. Монотонний ряд НЕ дає ЖОДНОГО
    півота: `detect_smc_structure` шукає бар, чий high вищий за максимум
    наступних `size` барів — у рівному підйомі такого не буває, тож подій 0 і
    блоків 0. Саме на цьому впала перша версія фікстури.
    """
    d = (target - px) / steps
    for _ in range(steps):
        px += d
        seq.append((px + amp, px - amp, px))
    return px


def _market():
    """Ринок, де ЖИВІ блоки є з ОБОХ боків одночасно — і в internal, і в swing.

    Спершу довга пилка (прогрів ATR(200) + купа internal-півотів), далі ноги
    ДОВШІ за `swing_size=50`, щоб зʼявились і свінг-півоти: підйом → відкат →
    пробій свінг-максимуму (BOS) → розворот униз (CHoCH). Бичачі блоки
    лишаються живими (їхній low не пробито), ведмежий — теж: рівно та
    ситуація, яку користувач бачить у LuxAlgo.
    """
    seq = []
    px = 100.0
    for _ in range(22):                     # 220 барів пилки — прогрів ATR
        px = _leg(px, px + 2.0, 5, seq)
        px = _leg(px, px - 1.6, 5, seq)
    px = _leg(px, px + 30, 60, seq)         # велика нога вгору → swing high
    px = _leg(px, px - 12, 55, seq)         # відкат            → swing low
    px = _leg(px, px + 26, 60, seq)         # пробій свінг-максимуму → BOS
    px = _leg(px, px - 34, 70, seq)         # розворот униз          → CHoCH
    px = _leg(px, px + 6, 12, seq)
    px = _leg(px, px - 10, 15, seq)
    return _bars(seq)


def _detect(klines, which='internal'):
    r = ST.detect_smc_structure(klines, internal_size=5, swing_size=50)
    part = r.get(which) or {}
    return part.get('pivots', []), part.get('events', [])


# ═════════ 1. ДЕТЕКТОР ВІДДАЄ ВСІ ЖИВІ БЛОКИ ═══════════════════════════════
def test_full_list_contains_what_the_single_block_hides():
    """ГОЛОВНИЙ ЗАМОК. `detect_order_blocks` мусить віддати БІЛЬШЕ, ніж один
    блок, який показував `detect_last_order_block` — інакше графік бота ніколи
    не збіжиться з TradingView."""
    kl = _market()
    piv, ev = _detect(kl)
    one = OB.detect_last_order_block(klines=kl, pivots=piv, events=ev)
    many = OB.detect_order_blocks(klines=kl, pivots=piv, events=ev)
    _check(one, f'сценарій мусить дати хоч один блок (подій: {len(ev)})')
    _check(len(many) >= 1, many)
    # Перший у списку — ТОЙ САМИЙ, що віддає стара функція (ворота не поїхали).
    _check(many[0]['bar_time'] == one['bar_time'] and many[0]['bias'] == one['bias'],
           f'список мусить починатись із поточного блоку: {many[0]} vs {one}')
    print(f'✓ живих блоків {len(many)}, перший = поточний ({one["bias"]})')


def test_both_sides_are_returned_when_both_are_alive():
    """Саме те, чого бракувало: коли живі блоки є з ОБОХ боків, у списку
    мусять бути обидва. Раніше протилежний зникав на `return`."""
    kl = _market()
    piv, ev = _detect(kl)
    many = OB.detect_order_blocks(klines=kl, pivots=piv, events=ev, limit=20)
    sides = {o['bias'] for o in many}
    _check(len(many) >= 2, f'мало блоків для перевірки обох боків: {many}')
    _check(sides == {'BULLISH', 'BEARISH'},
           f'мусять бути ОБИДВА боки, отримано {sides}: {[(o["bias"], o["bar_time"]) for o in many]}')
    print(f'✓ обидва боки в списку: {sorted(sides)}')


def test_gate_function_is_untouched():
    """`detect_last_order_block` — ворота входу. Її результат мусить лишитись
    БАЙТ-У-БАЙТ таким, як був: один блок = `obs[0]`."""
    kl = _market()
    piv, ev = _detect(kl)
    one = OB.detect_last_order_block(klines=kl, pivots=piv, events=ev)
    many = OB.detect_order_blocks(klines=kl, pivots=piv, events=ev, limit=99)
    _check(isinstance(one, dict), 'ворота мусять віддавати ОДИН dict, а не список')
    _check(one == many[0], f'той самий блок: {one} vs {many[0]}')
    for k in ('bias', 'bar_high', 'bar_low', 'bar_time', 'created_by_tag'):
        _check(k in one, f'форма відповіді воріт не сміє змінитись — бракує {k}')
    print('✓ функція воріт віддає той самий один блок (форма не змінилась)')


def test_limit_caps_the_display_list():
    kl = _market()
    piv, ev = _detect(kl)
    _check(len(OB.detect_order_blocks(klines=kl, pivots=piv, events=ev, limit=1)) <= 1,
           'ліміт показу мусить діяти')
    print('✓ ліміт показу діє')


def test_swing_family_has_its_own_blocks():
    """Свінг-сімʼя — це НЕ ті самі блоки під іншою назвою. Її події/півоти
    інші (`swing_size=50` проти 5), і блоки вона дає СВОЇ. Поки бот її не
    малював, частину боксів TradingView відтворити було НІЧИМ."""
    kl = _market()
    ip, ie = _detect(kl, 'internal')
    sp, se = _detect(kl, 'swing')
    _check(len(se) >= 1, f'сценарій мусить дати свінг-події, отримано {len(se)}')
    sw = OB.detect_order_blocks(klines=kl, pivots=sp, events=se, limit=20)
    inr = OB.detect_order_blocks(klines=kl, pivots=ip, events=ie, limit=20)
    _check(sw, 'свінг-сімʼя мусить дати хоч один живий блок')
    _check({o['bar_time'] for o in sw} - {o['bar_time'] for o in inr} or
           len(sw) != len(inr),
           f'свінг-блоки мусять відрізнятись від internal: {sw} vs {inr}')
    print(f'✓ свінг-сімʼя дає власні блоки: {len(sw)} (internal {len(inr)})')


def test_no_history_is_empty_list_not_none():
    """Мало барів → порожній СПИСОК. `None` тут зламав би `.forEach` на фронті."""
    r = OB.detect_order_blocks(klines=_bars([(1, 1, 1)] * 10), pivots=[], events=[])
    _check(r == [], f'мусить бути [], отримано {r!r}')
    print('✓ без історії — порожній список, а не None')


# ═════════ 2. СКАНЕР РАХУЄ ОБИДВІ СІМ'Ї ════════════════════════════════════
def test_scanner_computes_swing_family_too():
    """🐞 Друга причина: `result['swing']` рахувався і ВИКИДАВСЯ. У LuxAlgo це
    окрема сім'я блоків (Swing Order Blocks, size=50), і без неї картинка не
    збіжиться в принципі."""
    src = open(os.path.join(_ROOT, 'detection/smc_scanner.py')).read()
    i = src.index('def _update_smc_ob')
    body = src[i:src.index('def _current_ob_bartime', i)]
    _check("result.get('swing'" in body,
           'свінг-структура вже рахується — її мусять і використати')
    _check(body.count('detect_order_blocks(') == 2,
           'потрібні ДВІ сім\'ї: internal + swing')
    _check("self._ob_zones[symbol]" in body, 'зони мусять зберігатись для показу')
    # І ворота лишились на ОДНОМУ блоці.
    _check('upsert_smc_ob_state(symbol, ob_tf, ob)' in body,
           'у БД (для воріт) і далі пише РІВНО поточний блок')
    print('✓ сканер рахує обидві сім\'ї, а в БД для воріт — той самий один блок')


def test_chart_payload_carries_zones_without_computing_them():
    """`get_chart_data` мусить ВІДДАВАТИ готовий знімок, а не рахувати OB на
    місці. Інлайн-розрахунок тут уже давав «бейдж каже 1H, а числа з чарт-TF»."""
    src = open(os.path.join(_ROOT, 'detection/smc_scanner.py')).read()
    _check("'ob_zones': dict(self._ob_zones.get(symbol) or {})" in src,
           'payload мусить нести ob_zones зі знімка')
    i = src.index("'ob_zones':")
    j = src.rindex('def get_chart_data', 0, i)
    _check('detect_order_blocks(' not in src[j:i],
           'у get_chart_data НЕ можна рахувати блоки інлайн')
    print('✓ payload несе готові зони, без інлайн-розрахунку')


# ═════════ 3. ФРОНТ МАЛЮЄ ВСІ ══════════════════════════════════════════════
def test_primitive_draws_a_list_not_one_box():
    html = open(os.path.join(_ROOT, 'templates/smart_money.html')).read()
    i = html.index('class VolOBPrimitive')
    body = html[i:html.index('function _snapSecToChart', i)]
    _check('setOBs(' in body, 'примітив мусить приймати СПИСОК')
    _check('this._obs' in body, 'і тримати список, а не один бокс')
    _check('setOB(ob) { this.setOBs(' in body,
           'старий setOB лишити як сумісний фасад — його кличе Volumized-бокс')
    _check('for (let _i = self._obs.length - 1; _i >= 0; _i--)' in body,
           'малювати задом наперед, щоб найновіший блок лежав ЗВЕРХУ')
    print('✓ примітив малює список; найновіший — зверху')


def test_chart_draws_both_families_from_zones():
    html = open(os.path.join(_ROOT, 'templates/smart_money.html')).read()
    i = html.index('// 🖼 ЯК У TRADINGVIEW')
    body = html[i:html.index('=== Trend badge', i)]
    _check("d.ob_zones" in body, 'малюємо зі знімка зон')
    _check("zones.internal" in body and "zones.swing" in body,
           'мусять малюватись ОБИДВІ сім\'ї')
    _check('obMatchPrim.setOBs(' in body, 'віддаємо примітиву список')
    _check('seen.has(key)' in body,
           'той самий блок може бути в обох сім\'ях — дублікат не малюємо двічі')
    _check('push(d.last_ob' in body,
           'потрібен фолбек на last_ob, поки скан не заповнив зони')
    # ⚠️ ПОКАЗ НЕ ЗАЛЕЖИТЬ ВІД ВОРІТ. Раніше бокси малювались лише при
    # увімкненому `Require OB Match` — тобто при вимкненому фільтрі графік
    # мовчав про структуру, яка на ньому Є. У LuxAlgo блоки малюються завжди.
    code = '\n'.join(l for l in body.splitlines() if not l.strip().startswith('//'))
    _check('if (obMatchPrim) {' in code and 'ob_filter_enabled' not in code,
           'малювання боксів НЕ сміє залежати від тумблера воріт')
    print('✓ графік малює обидві сім\'ї, без дублів, із фолбеком і без прив\'язки до воріт')


def test_badge_still_shows_the_single_gate_block():
    """Бейдж — це СТАН ВОРІТ, він мусить лишитись на одному поточному блоці.
    Інакше «намалювали обидва боки» перетворилось би на «бейдж каже одне,
    ворота роблять інше»."""
    html = open(os.path.join(_ROOT, 'templates/smart_money.html')).read()
    i = html.index("const obBadge = document.getElementById('sm-ob-badge');", 
                   html.index('=== Require OB Match box'))
    body = html[i:i + 2500]
    _check('const ob = d.last_ob;' in body, 'бейдж читає ОДИН поточний блок')
    _check('Живих блоків на' in html,
           'у тултипі має бути видно, скільки блоків бачить бот')
    print('✓ бейдж лишився станом воріт; кількість живих блоків — у тултипі')


if __name__ == '__main__':
    test_full_list_contains_what_the_single_block_hides()
    test_both_sides_are_returned_when_both_are_alive()
    test_gate_function_is_untouched()
    test_limit_caps_the_display_list()
    test_swing_family_has_its_own_blocks()
    test_no_history_is_empty_list_not_none()
    test_scanner_computes_swing_family_too()
    test_chart_payload_carries_zones_without_computing_them()
    test_primitive_draws_a_list_not_one_box()
    test_chart_draws_both_families_from_zones()
    test_badge_still_shows_the_single_gate_block()
    print('\nУсі тести «блоки як у TradingView» пройдено ✅')
