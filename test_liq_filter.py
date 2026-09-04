"""💧 ФІЛЬТР ЛІКВІДНОСТІ ЗА НАПРЯМКОМ — новий незалежний фільтр воріт входу.

Вимога користувача: «Додай до цих налаштувань ще один фільтр і задій його вже
з існуючими фільтрами. А саме: Біржа (за замовчуванням Binance), ІСТОРІЯ ГОД
(за замовчуванням 14 днів) і поле "Ліквідність" — за замовчуванням 75. Тобто
фільтр має перевіряти чи є по монеті ліквідність за напрямком не меншою за
вказаним параметром. Якщо фільтр увімкнено, перевіряй доступність вибраної
біржі (зроби якусь індикацію).»

⚠️ ЩО САМЕ ФІКСУЄМО:
  • «ліквідність ЗА НАПРЯМКОМ» = частка маси, що ТЯГНЕ ціну в бік угоди:
    LONG → `above_pct`, SHORT → `below_pct` (та сама конвенція, що фарбує
    драбину: вище ціни = вгору = зелений);
  • число бере ТА САМА `liq_scan.scan_one`, що живить 📡 Tickr — інакше та
    сама монета показувала б різні відсотки в різних місцях (урок PD-зони);
  • **FAIL-OPEN на недоступності біржі** — свідоме рішення, не недогляд:
    джерело ЗОВНІШНЄ (Binance регулярно блокує IP дата-центрів), і блок на
    відмові тихо зупинив би торгівлю цілком. Натомість стан видно в
    індикаторі й у розкладі 🧾 логу;
  • **КЕШ ОБОВʼЯЗКОВИЙ**: ворота кличуться і на сигнал, і на КОЖЕН тік
    Q4-recheck по кожній монеті черги. Без кешу це сотні HTTP за хвилину.
"""
import importlib.util
import inspect
import os
import sys
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg

_spec = importlib.util.spec_from_file_location(
    'smc_scanner_liq_test', os.path.join(_ROOT, 'detection', 'smc_scanner.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
S = _m.SMCScanner

_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html')).read()
_FLASK = open(os.path.join(_ROOT, 'web', 'flask_app.py')).read()
_SRC = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py')).read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ── стенд: підміняємо ЛИШЕ мережевий `scan_one` ────────────────────────────
_CALLS = []
_RESULT = {}


_BY_EX = {}


def _fake_scan_one(exchange='binance', symbol='BTCUSDT', bars=336, rows=12):
    _CALLS.append((exchange, symbol, bars))
    if _BY_EX:
        return dict(_BY_EX.get(exchange) or
                    {'ok': False, 'reason': f'{symbol} не знайдено на {exchange}'})
    return dict(_RESULT)


def _install_liq(result):
    """⚠️ Пастка, на якій уже наступали з `tickr_core`: `from detection import
    liq_scan` бере АТРИБУТ пакета, якщо він виставлений, тож підміни лише
    sys.modules замало — інакше тест мовчки піде в мережу."""
    global _RESULT
    _RESULT = dict(result)
    _BY_EX.clear()
    _CALLS.clear()
    fake = types.ModuleType('detection.liq_scan')
    fake.scan_one = _fake_scan_one
    fake._OI_ONE = {'binance': lambda s, sym: (100.0, 5e8),
                    'bybit': lambda s, sym: (100.0, 4e8),
                    'mexc': lambda s, sym: (100.0, 3e8),
                    'bingx': lambda s, sym: (0.0, 0.0)}
    sys.modules['detection.liq_scan'] = fake
    _pkg.liq_scan = fake
    return fake


def _sc(**over):
    o = S.__new__(S)
    o._settings = {'liq_filter_enabled': True, 'liq_filter_exchange': 'binance',
                   'liq_filter_bars': 336, 'liq_filter_min_pct': 75.0,
                   'liq_filter_ttl_sec': 300}
    o._settings.update(over)
    o._liq_filter_cache = {}
    o._liq_exchange_state = {}
    return o


def _lad(above, below, ok=True, reason=''):
    return {'ok': ok, 'above_pct': above, 'below_pct': below,
            'price': 100.0, 'reason': reason}


# ═══════════ 1. ДЕФОЛТИ (дослівно за вимогою) ══════════════════════════════
def test_defaults_match_the_request():
    d = _m.DEFAULT_SETTINGS
    _check(d.get('liq_filter_exchange') == 'binance',
           f"біржа за замовчуванням Binance: {d.get('liq_filter_exchange')}")
    _check(d.get('liq_filter_bars') == 336,
           f"історія 14 діб = 336 год: {d.get('liq_filter_bars')}")
    _check(d.get('liq_filter_min_pct') == 75.0,
           f"поріг 75: {d.get('liq_filter_min_pct')}")
    # Сам тумблер — ВИМКНЕНО, як і кожен інший фільтр цього блоку: новий
    # фільтр не має мовчки звузити потік на вже працюючій установці.
    _check(d.get('liq_filter_enabled') is False,
           'новий фільтр мусить бути вимкнений за замовчуванням')
    print('✓ дефолти: Binance · 336 год (14 діб) · 75% · тумблер OFF')


def test_key_is_whitelisted_and_validated():
    for k in ('liq_filter_enabled', 'liq_filter_exchange', 'liq_filter_bars',
              'liq_filter_min_pct'):
        _check(f"'{k}'," in _SRC, f'ключа {k} немає у білому списку update_settings')
    _check("self._settings['liq_filter_bars'] = max(24, min(_lb, 1000))" in _SRC,
           'глибина історії мусить обмежуватись')
    _check("self._settings['liq_filter_min_pct'] = max(0.0, min(_lp, 100.0))" in _SRC,
           'поріг мусить обмежуватись 0..100')
    _check("self._liq_filter_cache.clear()" in _SRC,
           'зміна параметрів мусить скидати кеш — інакше фільтр ще TTL '
           'відповідав би за СТАРОЮ біржею')
    print('✓ ключі у білому списку, значення валідуються, кеш скидається')


# ═══════════ 2. НАПРЯМОК — головна семантика ═══════════════════════════════
def test_direction_uses_the_side_that_pulls_our_way():
    """ГОЛОВНИЙ ЗАМОК. Для LONG рахується маса ВИЩЕ ціни, для SHORT — НИЖЧЕ.
    Переплутати боки = фільтр пускатиме рівно протилежне."""
    _install_liq(_lad(above=80.0, below=20.0))
    o = _sc()
    _check(o._liq_filter_allows('X', 'LONG') is True,
           '80% маси вгору при порозі 75 — LONG мусить пройти')
    _check(o._liq_filter_allows('X', 'SHORT') is False,
           'той самий розклад для SHORT — 20% вниз, мусить блокувати')

    _install_liq(_lad(above=10.0, below=90.0))
    o = _sc()
    _check(o._liq_filter_allows('X', 'SHORT') is True, '90% вниз → SHORT проходить')
    _check(o._liq_filter_allows('X', 'LONG') is False, '10% вгору → LONG блок')
    print('✓ LONG дивиться вгору, SHORT — вниз')


def test_threshold_is_inclusive_and_configurable():
    _install_liq(_lad(above=75.0, below=25.0))
    _check(_sc()._liq_filter_allows('X', 'LONG') is True,
           'рівно поріг мусить проходити (≥, а не >)')
    _install_liq(_lad(above=74.9, below=25.1))
    _check(_sc()._liq_filter_allows('X', 'LONG') is False, '74.9 < 75 → блок')
    # Нижчий поріг — та сама монета проходить.
    _install_liq(_lad(above=60.0, below=40.0))
    _check(_sc(liq_filter_min_pct=50.0)._liq_filter_allows('X', 'LONG') is True,
           'поріг 50 → 60% проходить')
    _check(_sc(liq_filter_min_pct=90.0)._liq_filter_allows('X', 'LONG') is False,
           'поріг 90 → 60% блок')
    print('✓ поріг включний і керується налаштуванням')


def test_no_data_never_blocks_trading():
    """⚠️ FAIL-OPEN — свідоме рішення. Джерело ЗОВНІШНЄ: Binance періодично
    блокує IP дата-центрів (це вже задокументовано для банера настрою). Якби
    недоступність блокувала, одна відмова біржі тихо зупинила б торгівлю."""
    for bad in (_lad(None, None, ok=False, reason='biржа не відповіла'),
                _lad(None, None, ok=True),          # ok, але без рівнів
                {'ok': False, 'reason': 'timeout'}):
        _install_liq(bad)
        o = _sc()
        _check(o._liq_filter_allows('X', 'LONG') is True,
               f'немає даних → НЕ блокуємо: {bad}')
        _check(o._liq_filter_allows('X', 'SHORT') is True, f'те саме для SHORT: {bad}')
    print('✓ немає даних — фільтр пропускає (торгівля не зупиняється)')


def test_state_label_shows_the_number():
    """Голе «Ліквідність:✗» нічого не пояснює — у розкладі мусить бути число."""
    _install_liq(_lad(above=31.6, below=68.4))
    o = _sc()
    _check(o._liq_state_label('X', 'SHORT') == '68.4% вниз', o._liq_state_label('X', 'SHORT'))
    _check(o._liq_state_label('X', 'LONG') == '31.6% вгору', o._liq_state_label('X', 'LONG'))
    _install_liq({'ok': False, 'reason': 'HTTP 451'})
    _check('451' in _sc()._liq_state_label('X', 'LONG'),
           'причина недоступності мусить бути видна')
    print('✓ у розкладі видно число, а при збої — причину')


# ═══════════ 3. КЕШ (без нього фільтр забʼє біржу) ═════════════════════════
def test_result_is_cached_within_ttl():
    _install_liq(_lad(above=80.0, below=20.0))
    o = _sc()
    for _ in range(10):
        o._liq_filter_allows('X', 'LONG')
        o._liq_state_label('X', 'LONG')
    _check(len(_CALLS) == 1,
           f'10 перевірок мусять коштувати ОДИН запит, а не {len(_CALLS)}')
    # Інша монета — свій запит.
    o._liq_filter_allows('Y', 'LONG')
    _check(len(_CALLS) == 2, f'нова монета → новий запит: {len(_CALLS)}')
    print('✓ зріз кешується (10 перевірок = 1 запит)')


def test_failure_is_cached_too():
    """Найбільший тиск на біржу настає САМЕ тоді, коли вона лежить. Невдачу
    теж кешуємо, інакше кожен тік Q4-recheck ломився б у мертвий ендпоінт."""
    _install_liq({'ok': False, 'reason': 'timeout'})
    o = _sc()
    for _ in range(8):
        o._liq_filter_allows('X', 'LONG')
    # ⚠️ Невдача коштує ДВІ спроби (обрана біржа + фолбек на Bybit) — і саме
    # тому кеш тут ще важливіший: без нього 8 перевірок дали б 16 запитів.
    _check(len(_CALLS) == 2,
           f'обрана + фолбек ОДИН раз, далі кеш: {len(_CALLS)} запитів')
    print('✓ невдалий зріз кешується (не добиваємо биржу, що лежить)')


def test_cache_key_includes_exchange_and_depth():
    _install_liq(_lad(above=80.0, below=20.0))
    o = _sc()
    o._liq_filter_allows('X', 'LONG')
    o._settings['liq_filter_exchange'] = 'bybit'
    o._liq_filter_allows('X', 'LONG')
    o._settings['liq_filter_bars'] = 168
    o._liq_filter_allows('X', 'LONG')
    _check(len(_CALLS) == 3, f'біржа і глибина — частина ключа: {_CALLS}')
    _check(_CALLS[1][0] == 'bybit' and _CALLS[2][2] == 168, _CALLS)
    print('✓ ключ кешу враховує біржу і глибину історії')


def test_settings_reach_scan_one():
    _install_liq(_lad(above=80.0, below=20.0))
    _sc(liq_filter_exchange='mexc', liq_filter_bars=720)._liq_filter_allows('X', 'LONG')
    _check(_CALLS == [('mexc', 'X', 720)], f'параметри мусять доїхати: {_CALLS}')
    print('✓ біржа й історія доїжджають до розрахунку')


# ═══════════ 4. ВБУДОВАНО У ВОРОТА ═════════════════════════════════════════
def _gate_ns(decision_ok=True, **over):
    o = _sc(**over)
    o._settings.setdefault('use_pd_zone_filter', False)
    o._forecast_pair = lambda sym: ('—', '—')
    o.get_pd_pct = lambda sym: None
    o._decision_gate = lambda sym, side, at_intake=False: (
        decision_ok, 'LONG 80%' if decision_ok else 'NEUTRAL')
    return o


def test_gate_blocks_and_explains():
    # 24% вгору / 76% вниз: LONG не дотягує до 75, SHORT дотягує.
    _install_liq(_lad(above=24.0, below=76.0))
    ok, reason, detail = S._signal_allowed(_gate_ns(), 'X', 'LONG')
    _check(ok is False, 'ліквідність проти → сигнал не проходить')
    _check('24.0% вгору' in reason and '75' in reason, f'причина: {reason}')
    _check('Ліквідність[BINANCE≥75.0%](24.0% вгору):✗' in detail, f'розклад: {detail}')

    ok2, _, detail2 = S._signal_allowed(_gate_ns(), 'X', 'SHORT')
    _check(ok2 is True, 'той самий розклад для SHORT мусить пройти')
    _check('76.0% вниз):✓' in detail2, f'розклад: {detail2}')
    print('✓ ворота блокують і пояснюють числом')


def test_gate_ignores_the_filter_when_off():
    _install_liq(_lad(above=1.0, below=99.0))
    ok, reason, detail = S._signal_allowed(_gate_ns(liq_filter_enabled=False), 'X', 'LONG')
    _check(ok is True and reason == '', f'{reason}')
    _check('Ліквідність' not in detail,
           f'вимкнений фільтр не має зʼявлятись у розкладі: {detail}')
    _check(_CALLS == [], 'вимкнений фільтр НЕ сміє смикати біржу')
    print('✓ вимкнений фільтр не рахується і не показується')


def test_gate_is_independent_of_other_filters():
    """Кожен фільтр цього блоку — НЕЗАЛЕЖНИЙ. Ліквідність не повинна залежати
    від того, увімкнені інші чи ні."""
    _install_liq(_lad(above=90.0, below=10.0))
    ok, _, detail = S._signal_allowed(_gate_ns(), 'X', 'LONG')
    _check(ok is True and 'Ліквідність' in detail, detail)
    _check('OB(' not in detail and 'PD(' not in detail,
           f'інші фільтри вимкнені — їх у розкладі бути не має: {detail}')
    print('✓ фільтр працює самостійно, поряд із рештою ланцюга')


# ═══════════ 5. ДОСТУПНІСТЬ БІРЖІ + ІНДИКАЦІЯ ══════════════════════════════
def test_exchange_probe_reports_availability():
    _install_liq(_lad(above=80.0, below=20.0))
    o = _sc()
    r = o.check_liq_exchange('binance')
    _check(r['ok'] is True and r['exchange'] == 'binance', r)
    _check('took_ms' in r, 'проба мусить казати, скільки тривала')
    # BingX у стенді віддає нулі → «відповіла, але без OI».
    r2 = o.check_liq_exchange('bingx')
    _check(r2['ok'] is False and 'відкрит' in r2['reason'], r2)
    r3 = o.check_liq_exchange('казна-що')
    _check(r3['ok'] is False and 'не підтримується' in r3['reason'], r3)
    # Стан зберігається для індикатора без повторного запиту.
    _check(o.get_liq_exchange_state().get('exchange') == 'казна-що',
           o.get_liq_exchange_state())
    print('✓ проба біржі: доступна / без OI / не підтримується')


def test_filter_work_updates_the_indicator():
    """Індикатор мусить оживати і від САМОЇ РОБОТИ фільтра, а не лише від
    ручної проби — інакше «біржа впала» видно тільки після кліку."""
    _install_liq({'ok': False, 'reason': 'HTTP 451'})
    o = _sc()
    o._liq_filter_allows('X', 'LONG')
    st = o.get_liq_exchange_state()
    _check(st.get('ok') is False and '451' in st.get('reason', ''), st)
    _install_liq(_lad(above=80.0, below=20.0))
    o2 = _sc()
    o2._liq_filter_allows('X', 'LONG')
    _check(o2.get_liq_exchange_state().get('ok') is True, o2.get_liq_exchange_state())
    print('✓ робота фільтра сама оновлює індикатор доступності')


def test_api_endpoint_exists():
    _check("@app.route('/api/smc/liq-exchange-check')" in _FLASK,
           'потрібен ендпоінт перевірки біржі')
    _check("cached" in _FLASK.split('liq-exchange-check')[1][:900],
           'мусить бути режим «віддай останній стан» — автополл не має бити біржу')
    print('✓ ендпоінт перевірки біржі на місці')


# ═══════════ 6. UI ═════════════════════════════════════════════════════════
def test_ui_is_wired_end_to_end():
    for needle in ('id="sm-liq-filter-enabled"', 'id="sm-liq-filter-exchange"',
                   'id="sm-liq-filter-bars"', 'id="sm-liq-filter-min"',
                   'id="sm-liq-ex-badge"',
                   'async function updateLiqFilter()',
                   'liq_filter_enabled: en', 'liq_filter_min_pct: mn',
                   's.liq_filter_exchange || \'binance\'',
                   'async function checkLiqExchange('):
        _check(needle in _HTML, f'бракує у smart_money.html: {needle}')
    # Дефолти мусять стояти й у РОЗМІТЦІ — інакше до першого завантаження
    # стану користувач бачив би не те, що працює.
    i = _HTML.index('id="sm-liq-filter-exchange"')
    _check('value="binance" selected' in _HTML[i:i + 400], 'Binance має бути обраний')
    j = _HTML.index('id="sm-liq-filter-bars"')
    _check('value="336" selected' in _HTML[j:j + 400], '336 год має бути обрано')
    k = _HTML.index('id="sm-liq-filter-min"')
    _check('value="75"' in _HTML[k:k + 200], 'поріг 75 має стояти в розмітці')
    print('✓ UI: поля, дефолти, обробник і індикатор на місці')


def test_badge_hidden_while_filter_is_off():
    _check('_liqSyncBadge' in _HTML, 'потрібна синхронізація видимості індикатора')
    i = _HTML.index('function _liqSyncBadge')
    body = _HTML[i:i + 400]
    _check("b.style.display = enabled ? '' : 'none'" in body,
           'вимкнений фільтр не має повідомляти про стан того, чим не користується')
    print('✓ індикатор показується лише при увімкненому фільтрі')


# ═══════════ 7. ОСТАННІЙ У ЛАНЦЮГУ (економія звернень до біржі) ════════════
def test_liquidity_is_skipped_when_an_earlier_filter_already_blocked():
    """⚠️ Вимога користувача: «щоб не навантажувати біржу лишніми зверненнями».
    Це ЄДИНИЙ фільтр, що ходить у зовнішню біржу, тож коли сигнал уже зарізано,
    питати її немає сенсу — відповідь нічого не змінить.

    ⚠️ Мало ПЕРЕСУНУТИ блок у кінець: раніше КОЖЕН фільтр рахувався незалежно
    від `allowed`. Економію дає саме перевірка «чи ще дозволено»."""
    _install_liq(_lad(above=90.0, below=10.0))   # ліквідність ЗА нас
    ns = _gate_ns(decision_ok=False, decision_filter_enabled=True)
    ok, reason, detail = S._signal_allowed(ns, 'X', 'LONG')
    _check(ok is False, 'ріже Decision')
    _check('Decision' in reason, f'причина мусить лишитись від Decision: {reason}')
    _check(_CALLS == [],
           f'біржу НЕ смикаємо, коли сигнал уже зарізано: {_CALLS}')
    _check('не перевірялась' in detail,
           f'у розкладі мусить бути чесно сказано, що фільтр не виконувався: {detail}')
    print('✓ сигнал уже зарізано → до біржі не звертаємось узагалі')


def test_liquidity_runs_when_everything_else_passed():
    """Зворотний бік того самого правила: якщо решта пропустила — фільтр
    ОБОВʼЯЗКОВО відпрацьовує (інакше він став би декоративним)."""
    _install_liq(_lad(above=90.0, below=10.0))
    ok, _, detail = S._signal_allowed(
        _gate_ns(decision_ok=True, decision_filter_enabled=True), 'X', 'LONG')
    _check(ok is True, 'усі пропустили')
    _check(len(_CALLS) == 1, f'фільтр мусив спитати біржу рівно раз: {_CALLS}')
    _check('90.0% вгору' in detail, detail)
    print('✓ решта пропустила → ліквідність рахується')


def test_liquidity_is_the_last_part_of_the_breakdown():
    """Порядок у розкладі = порядок перевірок. Ліквідність мусить стояти
    ОСТАННЬОЮ — і коли пройшла, і коли її пропустили."""
    _install_liq(_lad(above=90.0, below=10.0))
    _, _, d1 = S._signal_allowed(
        _gate_ns(decision_ok=True, decision_filter_enabled=True), 'X', 'LONG')
    _check(d1.split(' · ')[-1].startswith('Ліквідність'), f'пройшла: {d1}')
    _, _, d2 = S._signal_allowed(
        _gate_ns(decision_ok=False, decision_filter_enabled=True), 'X', 'LONG')
    _check(d2.split(' · ')[-1].startswith('Ліквідність'), f'пропущено: {d2}')
    # І порядок у САМОМУ КОДІ воріт теж зафіксуємо. ⚠️ Шукаємо В ТІЛІ
    # `_signal_allowed`, а не по всьому файлу: `liq_filter_enabled`
    # зустрічається ще й у валідації налаштувань набагато вище, і глобальний
    # `.index()` порівнював би не ті входження.
    body = inspect.getsource(S._signal_allowed)
    _check(body.index("liq_filter_enabled") > body.index("decision_filter_enabled"),
           'у ланцюгу блок ліквідності мусить стояти ПІСЛЯ Decision')
    _check('if not allowed:' in body,
           'мусить бути пропуск виклику, а не лише перестановка блоку')
    print('✓ ліквідність — останній сегмент розкладу і останній блок у коді')


# ═══════════ 8. 🔁 ФОЛБЕК НА BYBIT (кейс MNTUSDT) ══════════════════════════
def _install_by_exchange(mapping):
    """Різні відповіді для різних бірж — саме так виглядає реальність, коли
    монета є на Bybit, але не лістована на Binance Futures."""
    _install_liq({})
    _BY_EX.update(mapping)
    _CALLS.clear()


def test_coin_missing_on_binance_falls_back_to_bybit():
    """🐞 КЕЙС ЗІ СКРІНА (04.09): `Ліквідність[BINANCE≥70.0%](немає даних:
    MNTUSDT не знайдено на binance (перевір назву)):✓` — монета просто НЕ
    лістована на Binance Futures, і фільтр проходив «порожняком».

    ⚠️ Bybit — біржа, на якій бот ТОРГУЄ, отже кожна монета watchlist там є
    ЗА ВИЗНАЧЕННЯМ. Тож фолбек не «вгадує», а бере єдине надійне джерело."""
    _install_by_exchange({
        'binance': {'ok': False, 'reason': 'MNTUSDT не знайдено на binance'},
        'bybit': _lad(above=82.0, below=18.0),
    })
    o = _sc(liq_filter_min_pct=70.0)
    _check(o._liq_filter_allows('MNTUSDT', 'LONG') is True,
           '82% вгору з Bybit ≥ 70 → LONG проходить ПО ДАНИХ, а не fail-open')
    _check(o._liq_filter_allows('MNTUSDT', 'SHORT') is False,
           'і, головне, тепер фільтр реально РІЖЕ протилежний бік')
    snap = o._liq_snapshot('MNTUSDT')
    _check(snap['ok'] and snap['exchange'] == 'bybit', snap)
    _check(snap['requested_exchange'] == 'binance', snap)
    _check(snap['fallback'] is True, snap)
    print('✓ монети немає на Binance → числа беруться з Bybit')


def test_fallback_is_visible_in_the_log_line():
    """Фолбек НЕ мовчазний: біля числа мусить стояти, з якої біржі воно.
    Інакше в лозі був би поріг «BINANCE≥70%» поруч із числом Bybit."""
    _install_by_exchange({
        'binance': {'ok': False, 'reason': 'не знайдено'},
        'bybit': _lad(above=82.0, below=18.0),
    })
    lbl = _sc()._liq_state_label('MNTUSDT', 'LONG')
    _check('82.0% вгору' in lbl and 'BYBIT' in lbl and 'фолбек' in lbl, lbl)
    _, _, detail = S._signal_allowed(_gate_ns(liq_filter_min_pct=70.0),
                                     'MNTUSDT', 'LONG')
    _check('через BYBIT (фолбек)' in detail, detail)
    print(f'✓ у розкладі видно джерело: {lbl}')


def test_magnet_for_tp2_uses_the_fallback_too():
    """Той самий зріз живить 🧲 магніт → Manual TP-2. Одна правка мусить
    покрити і сигнал, і відкриття (авто й ручне) — бо джерело ОДНЕ."""
    _install_by_exchange({
        'binance': {'ok': False, 'reason': 'не знайдено'},
        'bybit': dict(_lad(above=82.0, below=18.0), magnet_pct='41.0%',
                      magnet_row={'price': 110.0, 'price_hi': 112.0,
                                  'pct': 41.0, 'dist_pct': 4.0, 'dir': 'up'}),
    })
    mg = _sc().get_liq_magnet('MNTUSDT')
    _check(mg['ok'] and mg['exchange'] == 'bybit', mg)
    _check((mg['row'] or {}).get('price') == 110.0, mg)
    print('✓ магніт для TP-2 теж бере фолбек-біржу')


def test_both_exchanges_down_stays_fail_open_and_says_both():
    _install_by_exchange({
        'binance': {'ok': False, 'reason': 'HTTP 451'},
        'bybit': {'ok': False, 'reason': 'timeout'},
    })
    o = _sc()
    _check(o._liq_filter_allows('X', 'LONG') is True,
           'обидві мовчать → fail-open (торгівля не зупиняється)')
    r = o._liq_snapshot('X')['reason']
    _check('451' in r and 'bybit' in r and 'timeout' in r,
           f'причина мусить назвати ОБИДВІ спроби: {r}')
    print('✓ обидві біржі мовчать → fail-open + причина по обох')


def test_no_double_request_when_bybit_is_the_choice():
    """Фолбек на самого себе — марна робота."""
    _install_by_exchange({'bybit': {'ok': False, 'reason': 'timeout'}})
    _sc(liq_filter_exchange='bybit')._liq_filter_allows('X', 'LONG')
    _check(len(_CALLS) == 1, f'обрано bybit → рівно один запит: {_CALLS}')
    print('✓ обрано Bybit → фолбеку на себе немає')


def test_fallback_result_is_cached():
    """Інакше монета, якої немає на Binance, коштувала б ДВА запити на кожен
    тік — тобто фолбек зробив би гірше, ніж було."""
    _install_by_exchange({
        'binance': {'ok': False, 'reason': 'не знайдено'},
        'bybit': _lad(above=82.0, below=18.0),
    })
    o = _sc()
    for _ in range(6):
        o._liq_filter_allows('MNTUSDT', 'LONG')
    _check(len(_CALLS) == 2,
           f'дві спроби ОДИН раз, далі кеш: {len(_CALLS)} запитів')
    print('✓ фолбек-зріз кешується (2 запити на 6 перевірок)')


def test_tickr_page_does_not_silently_switch_exchange():
    """⚠️ На 📡 Tickr користувач ОБИРАЄ біржу свідомо — там мовчазна підміна
    була б брехнею. Фолбек живе САМЕ у воротах/магніті, не в ендпоінті."""
    i = _FLASK.index('liquidity-one')
    body = _FLASK[i:i + 1200]
    _check('LIQ_FALLBACK' not in body and 'fallback' not in body.lower(),
           'ендпоінт однієї монети не має підмінювати біржу')
    print('✓ сторінка Tickr біржу не підміняє (там вибір користувача)')


def test_badge_has_the_fallback_state():
    _check('fallback_ok' in _HTML, 'бейдж мусить знати про фолбек')
    _check('sm-liq-ex-badge.warn' in _HTML, 'потрібен окремий (жовтий) стан')
    i = _HTML.index('d.fallback_ok')
    _check('мовчить →' in _HTML[i:i + 400],
           'бейдж мусить казати, що працюємо через іншу біржу')
    print('✓ індикатор розрізняє «недоступна» і «працюємо через фолбек»')


if __name__ == '__main__':
    test_defaults_match_the_request()
    test_key_is_whitelisted_and_validated()
    test_direction_uses_the_side_that_pulls_our_way()
    test_threshold_is_inclusive_and_configurable()
    test_no_data_never_blocks_trading()
    test_state_label_shows_the_number()
    test_result_is_cached_within_ttl()
    test_failure_is_cached_too()
    test_cache_key_includes_exchange_and_depth()
    test_settings_reach_scan_one()
    test_gate_blocks_and_explains()
    test_gate_ignores_the_filter_when_off()
    test_gate_is_independent_of_other_filters()
    test_exchange_probe_reports_availability()
    test_filter_work_updates_the_indicator()
    test_api_endpoint_exists()
    test_ui_is_wired_end_to_end()
    test_badge_hidden_while_filter_is_off()
    test_liquidity_is_skipped_when_an_earlier_filter_already_blocked()
    test_liquidity_runs_when_everything_else_passed()
    test_liquidity_is_the_last_part_of_the_breakdown()
    test_coin_missing_on_binance_falls_back_to_bybit()
    test_fallback_is_visible_in_the_log_line()
    test_magnet_for_tp2_uses_the_fallback_too()
    test_both_exchanges_down_stays_fail_open_and_says_both()
    test_no_double_request_when_bybit_is_the_choice()
    test_fallback_result_is_cached()
    test_tickr_page_does_not_silently_switch_exchange()
    test_badge_has_the_fallback_state()
    print('\nУсі тести фільтра ліквідності пройдено ✅')
