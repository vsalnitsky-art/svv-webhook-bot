"""💧 СКАНЕР ЛІКВІДНОСТІ + ⏳ СПІЛЬНА ЧЕРГА СКАНІВ БІРЖІ (вимоги 18.09).

**Вимоги користувача, дослівно:**
  2) «організуємо сканер із своїм вікном і основним тумблером і гармошкою
     налаштувань, під вікном "МММ-монітор", який буде відбирати монети за
     певним фільтром і додавати в свою таблицю… періодичність сканування (за
     замовчуванням 15хв) і фільтр саме які монети беремо (за замовчуванням
     65%) — "Маса ліквідності НИЖЧЕ ціни 65% · тягне ВНИЗ" за напрямком, який
     вказує на даний момент банер "МММ-монітор"… Дані при кожному скані
     оновлюються, але потрібно [перевірити] чи немає вже конкретної монети в
     роботі (відкрита угода або черга)… Відображай дані по монеті відсоток
     "Маса ліквідності" і Найсильніший магніт — напрямок.»
  3) «При зміні напрямку банера "МММ-монітор" — перескановуй.»
  4) «Організуй всі скани біржі по графіку, щоб не навантажувати біржу, кожен
     скан в свою чергу.»

Що стережуть ці тести:
  • бік відбору — САМЕ той, що показує банер, і за ТІЄЮ САМОЮ конвенцією, що в
    💧 фільтрі входу (LONG → маса вище ціни, SHORT → нижче);
  • монета «в роботі» (угода або будь-яка черга) повторно не додається;
  • ⚖ банер без напрямку → біржу НЕ чіпаємо взагалі;
  • фліп банера пересканує, але не частіше за мінімальний проміжок;
  • кожен скан іде спільною чергою, і два скани ніколи не працюють разом;
  • дефолти збігаються з блоком 💧 на 📡 Tickr (звідки взято алгоритм).
"""
import ast
import importlib.util
import os
import sys
import threading
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

# Порожній пакет `detection`: справжній `__init__.py` тягне півпроєкту.
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, 'detection', fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_sq = _load('detection.scan_queue', 'scan_queue.py')
_pkg.scan_queue = _sq
_lh = _load('detection.liq_hunter', 'liq_hunter.py')
_pkg.liq_hunter = _lh

# ⏳ Пауза МІЖ сканами у ТЕСТАХ не потрібна: вона про навантаження на біржу, і
# її перевіряє окремий тест (`test_the_queue_runs_one_scan_at_a_time` ставить
# своє значення). Без цього кожен скан у файлі чекав би 5 секунд.
_sq.MIN_GAP_SEC = 0.0

_SRC = open(os.path.join(_HERE, 'detection', 'liq_hunter.py'),
            encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
             encoding='utf-8').read()
_TICKR = open(os.path.join(_HERE, 'templates', 'tickr.html'),
              encoding='utf-8').read()
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'),
              encoding='utf-8').read()


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


class _DB:
    def __init__(self):
        self.d = {}

    def get_setting(self, k, dflt=None):
        return self.d.get(k, dflt)

    def set_setting(self, k, v):
        self.d[k] = v


class _FF:
    """Заглушка Fuel Filter: лише те, що читає сканер."""

    def __init__(self, direction='SHORT', work=()):
        self.direction = direction
        self.work = set(work)

    def mm_bias(self):
        return {'dir': self.direction}

    def symbols_in_work(self):
        return set(self.work)


def _row(sym, above, below, ok=True, **kw):
    """Рядок у формі, яку віддає `liq_scan.summarise`."""
    r = {'symbol': sym, 'ok': ok, 'above_pct': above, 'below_pct': below,
         'pull': 'up' if above > below else 'down',
         'pull_pct': abs(above - below), 'price': 1.0,
         'magnet_price': '$0.64000', 'magnet_pct': 14.2,
         'magnet_dist': '↓5.58%', 'magnet_dir': 'down'}
    r.update(kw)
    return r


def _mk(rows=None, direction='SHORT', work=(), **settings):
    calls = []

    def _scan(s):
        calls.append(dict(s))
        return {'ok': True, 'rows': list(rows or []), 'took_sec': 1.0}

    h = _lh.LiqHunterDaemon(_DB(), get_watchlist=lambda: ['AAAUSDT'],
                            scan_fn=_scan,
                            get_fuel_filter=lambda: _FF(direction, work))
    # ⚠️ ФОНОВИЙ ПОТІК У ТЕСТАХ НЕ ПІДНІМАЄМО. `update_settings({'enabled':
    # True})` у проді стартує цикл — і в тестах десяток таких циклів прокидався
    # б посеред інших тестів, подаючи у СПІЛЬНУ чергу сканів завдання з тим
    # самим імʼям. Дедуп (правильно) віддавав би чужий результат, і тест падав
    # би «на рівному місці». Тік ми й так кличемо руками.
    h.start = lambda: None
    base = {'enabled': True}
    base.update(settings)
    h.update_settings(base)
    h.calls = calls
    return h


# ═══════════ 1. БІК ВІДБОРУ — ЗА БАНЕРОМ, І ТІЄЮ САМОЮ КОНВЕНЦІЄЮ ════════
def test_direction_uses_the_side_that_pulls_our_way():
    """⚠️ ГОЛОВНИЙ ЗАМОК. LONG → `above_pct`, SHORT → `below_pct` — та сама
    конвенція, що в 💧 фільтрі входу. Переплутати боки = відбирати рівно
    протилежні монети, і помітити це можна було б лише по збитках."""
    r = _row('AAAUSDT', above=80.0, below=20.0)
    _check(_lh.mass_pct(r, 'LONG') == 80.0, 'LONG мусить брати масу ВИЩЕ ціни')
    _check(_lh.mass_pct(r, 'SHORT') == 20.0, 'SHORT мусить брати масу НИЖЧЕ ціни')
    print('✓ бік: LONG → вище ціни, SHORT → нижче ціни')


def test_the_banner_direction_decides_what_is_taken():
    down = _row('DOWNUSDT', above=30.0, below=70.0)
    up = _row('UPUSDT', above=72.0, below=28.0)
    h = _mk([down, up], direction='SHORT', min_mass_pct=65)
    h.scan('test')
    got = [r['symbol'] for r in h.get_state()['rows']]
    _check(got == ['DOWNUSDT'], f'банер SHORT → лише «тягне ВНИЗ»: {got}')
    h = _mk([down, up], direction='LONG', min_mass_pct=65)
    h.scan('test')
    got = [r['symbol'] for r in h.get_state()['rows']]
    _check(got == ['UPUSDT'], f'банер LONG → лише «тягне ВГОРУ»: {got}')
    print('✓ відбір іде САМЕ в бік банера МММ-монітора')


def test_the_threshold_is_the_setting_not_a_magic_number():
    rows = [_row('AUSDT', 30.0, 70.0), _row('BUSDT', 40.0, 60.0)]
    h = _mk(rows, direction='SHORT', min_mass_pct=65)
    h.scan('t')
    _check([r['symbol'] for r in h.get_state()['rows']] == ['AUSDT'],
           'поріг 65% мусить відсіяти 60%')
    h = _mk(rows, direction='SHORT', min_mass_pct=55)
    h.scan('t')
    _check(len(h.get_state()['rows']) == 2, 'поріг 55% мусить пропустити обидві')
    print('✓ поріг маси — налаштування, а не зашите число')


def test_default_threshold_and_interval_match_the_requirement():
    _check(_lh.DEFAULTS['min_mass_pct'] == 65.0, 'дефолт порога — 65%')
    _check(_lh.DEFAULTS['interval_min'] == 15, 'дефолт періодичності — 15 хв')
    _check(_lh.DEFAULTS['enabled'] is False,
           'новий рушій не має вмикатись мовчки')
    print('✓ дефолти: 65% і 15 хв, тумблер OFF')


# ═══════════ 2. МОНЕТА «В РОБОТІ» ПОВТОРНО НЕ ДОДАЄТЬСЯ ══════════════════
def test_a_coin_already_in_work_is_not_added_again():
    """Дослівна вимога: «потрібно [перевірити] чи немає вже конкретної монети
    в роботі (відкрита угода або черга) — в такому випадку додавати повторно
    монету не потрібно»."""
    rows = [_row('AUSDT', 20.0, 80.0), _row('BUSDT', 25.0, 75.0)]
    h = _mk(rows, direction='SHORT', work={'BUSDT'}, min_mass_pct=65)
    res = h.scan('t')
    got = [r['symbol'] for r in h.get_state()['rows']]
    _check(got == ['AUSDT'], f'монета в роботі мусить бути пропущена: {got}')
    _check(res['in_work'] == 1, f'пропуск мусить бути НАЗВАНИЙ у розкладі: {res}')
    _check('у роботі 1' in h.get_state()['status'],
           'розклад у статусі мусить казати, скільки пропущено')
    print('✓ монета в роботі (угода або черга) повторно не додається')


def test_in_work_comes_from_the_single_source():
    """Набір «в роботі» береться `ff.symbols_in_work()`, а не власним обходом
    приватних черг — інакше з часом це були б ДВА різні списки."""
    i = _SRC.index('def _in_work(')
    body = _SRC[i:_SRC.index('\n    def ', i + 10)]
    _check('symbols_in_work' in body, 'потрібен єдиний публічний читач')
    for _bad in ('_pending', '_fuel_managed', '_positions'):
        _check(_bad not in body, f'сканер не має лазити у {_bad} сам')
    print('✓ «в роботі» — з одного джерела')


# ═══════════ 3. ⚖ БАНЕР БЕЗ НАПРЯМКУ — БІРЖУ НЕ ЧІПАЄМО ══════════════════
def test_a_flat_banner_never_touches_the_exchange():
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction=None)
    res = h.scan('t')
    _check(res.get('reason') == 'flat', f'мусить бути явна причина: {res}')
    _check(not h.calls, 'при ⚖ рівновазі скан біржі не має запускатись ВЗАГАЛІ')
    _check('⚖' in h.get_state()['status'], 'причина мусить бути в статусі')
    print('✓ ⚖ банер без напрямку → нуль запитів до біржі')


def test_a_flip_to_flat_clears_the_old_table():
    """Рядки відбирались під напрямок, якого вже немає — лишати їх означало б
    показувати список, що СУПЕРЕЧИТЬ банеру над ним."""
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction='SHORT', min_mass_pct=65)
    h.scan('t')
    _check(len(h.get_state()['rows']) == 1, 'перший скан мусить дати рядок')
    h._get_ff = lambda: _FF(None)
    h.scan('t')
    st = h.get_state()
    _check(st['rows'] == [] and st['dir'] is None, 'таблицю мусить почистити')
    print('✓ фліп у ⚖ чистить застарілу таблицю')


# ═══════════ 4. ПЕРЕСКАН НА ФЛІПІ + ПЕРІОДИЧНІСТЬ ════════════════════════
def test_a_banner_flip_triggers_a_rescan():
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction='SHORT', min_mass_pct=65)
    h._tick()
    _check(len(h.calls) == 1, 'перший тік мусить сканувати')
    h._tick()
    _check(len(h.calls) == 1, 'без фліпу і до строку — не сканувати')
    # Банер розвернувся → пересканувати ОДРАЗУ (з поправкою на мін. проміжок).
    h._get_ff = lambda: _FF('LONG')
    h._ts = time.time() - _lh.MIN_GAP_SEC - 1
    h._tick()
    _check(len(h.calls) == 2, 'фліп банера мусить пересканувати')
    print('✓ фліп банера пересканує')


def test_a_flip_right_after_a_scan_is_delayed_not_dropped():
    """Антиспам: перескан не частіше за `MIN_GAP_SEC`, але подію не губимо —
    вона просто зсувається на кінець проміжку."""
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction='SHORT', min_mass_pct=65)
    h._tick()
    h._get_ff = lambda: _FF('LONG')
    h._tick()
    _check(len(h.calls) == 1, 'одразу після скану фліп не має бити по біржі')
    h._ts = time.time() - _lh.MIN_GAP_SEC - 1
    h._next_at = 0
    h._tick()
    _check(len(h.calls) == 2, 'після проміжку скан мусить відбутись')
    print('✓ фліп впритул до скану — відкладається, а не губиться')


def test_the_schedule_uses_the_interval_setting():
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction='SHORT', interval_min=15)
    h._tick()
    nxt = h.get_state()['next_at']
    _check(abs((nxt - time.time()) - 15 * 60) < 5,
           f'наступний скан мусить бути через 15 хв: {nxt - time.time():.0f}с')
    h._tick()
    _check(len(h.calls) == 1, 'до строку другий скан не запускається')
    print('✓ періодичність береться з налаштування')


def test_a_disabled_scanner_does_nothing_and_clears_the_table():
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction='SHORT', min_mass_pct=65)
    h.scan('t')
    h.update_settings({'enabled': False})
    _check(h.get_state()['rows'] == [], 'вимкнений сканер чистить таблицю')
    h._tick()
    _check(len(h.calls) == 1, 'вимкнений сканер не ходить на біржу')
    print('✓ вимкнений сканер мовчить і не тримає «заморожених» рядків')


# ═══════════ 5. ЩО САМЕ ПОКАЗУЄМО ════════════════════════════════════════
def test_every_row_carries_the_mass_and_the_magnet():
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction='SHORT', min_mass_pct=65)
    h.scan('t')
    r = h.get_state()['rows'][0]
    _check(r['mass_pct'] == 80.0 and r['mass_dir'] == 'down',
           f'маса мусить бути в рядку: {r}')
    for _k in ('magnet_price', 'magnet_pct', 'magnet_dist', 'magnet_dir'):
        _check(r.get(_k) is not None, f'магніт мусить нести {_k}')
    print('✓ у рядку — маса ліквідності і найсильніший магніт із напрямком')


def test_a_coin_that_stays_keeps_its_timer():
    """«Дані при кожному скані оновлюються» — але монета, що тримається у
    списку, не «зʼявляється» заново, інакше таймер обнулявся б щоскану."""
    rows = [_row('AUSDT', 20.0, 80.0)]
    h = _mk(rows, direction='SHORT', min_mass_pct=65)
    h.scan('t')
    since = h.get_state()['rows'][0]['since']
    time.sleep(0.01)
    h.scan('t')
    _check(h.get_state()['rows'][0]['since'] == since,
           'таймер монети, що лишилась у списку, не має обнулятись')
    print('✓ монета, що тримається у списку, зберігає свій таймер')


def test_the_row_count_is_explained_not_guessed():
    rows = [_row('AUSDT', 20.0, 80.0), _row('BUSDT', 45.0, 55.0),
            _row('CUSDT', 10.0, 90.0), _row('DUSDT', 0, 0, ok=False)]
    h = _mk(rows, direction='SHORT', work={'CUSDT'}, min_mass_pct=65)
    res = h.scan('t')
    _check(res['scanned'] == 4 and res['weak'] == 1
           and res['in_work'] == 1 and res['nodata'] == 1,
           f'розклад мусить сходитись: {res}')
    print('✓ «чому монет мало» — розклад, а не здогадка')


# ═══════════ 6. ⏳ СПІЛЬНА ЧЕРГА СКАНІВ ══════════════════════════════════
def test_the_queue_runs_one_scan_at_a_time():
    q = _sq.ScanQueue()
    q_gap = _sq.MIN_GAP_SEC
    _sq.MIN_GAP_SEC = 0.0
    try:
        live, peak = [0], [0]
        lock = threading.Lock()

        def _job(n):
            with lock:
                live[0] += 1
                peak[0] = max(peak[0], live[0])
            time.sleep(0.05)
            with lock:
                live[0] -= 1
            return {'ok': True, 'n': n}

        jobs = [q.submit(f'j{i}', (lambda n=i: _job(n))) for i in range(4)]
        for j in jobs:
            _check(j.wait(10), 'завдання мусить завершитись')
        _check(peak[0] == 1, f'одночасно працював не один скан: {peak[0]}')
    finally:
        _sq.MIN_GAP_SEC = q_gap
    print('✓ черга: за раз працює РІВНО один скан')


def test_the_same_scan_asked_twice_is_not_doubled():
    """Подвійний клік по «Сканувати» не має коштувати подвійного залпу."""
    q = _sq.ScanQueue()
    runs = []
    hold = threading.Event()

    def _slow():
        hold.wait(5)
        runs.append(1)
        return {'ok': True}

    a = q.submit('same', _slow)
    # ⚠️ Другий і третій запити приходять, коли перший УЖЕ ПРАЦЮЄ (саме так і
    # буває при подвійному кліку) — дедуп мусить накривати і цей випадок, а не
    # лише «ще стоїть у черзі».
    for _ in range(50):
        if (q.state().get('running') or {}).get('name') == 'same':
            break
        time.sleep(0.02)
    b = q.submit('same', _slow)
    c = q.submit('same', _slow)
    _check(b is a and c is a, 'той самий скан мусить чекати на ТОЙ САМИЙ job')
    hold.set()
    _check(a.wait(10), 'скан мусить завершитись')
    _check(len(runs) == 1, f'скан виконався {len(runs)} разів замість одного')
    print('✓ черга: дедуп за іменем — один залп замість трьох')


def test_a_failing_scan_does_not_kill_the_queue():
    q = _sq.ScanQueue()
    bad = q.submit('bad', lambda: (_ for _ in ()).throw(RuntimeError('біржа')))
    _check(bad.wait(10) and bad.error, 'виняток мусить дійти до того, хто чекав')
    ok = q.run('good', lambda: {'ok': True}, timeout=10)
    _check(ok.get('ok'), f'наступний скан мусить відпрацювати: {ok}')
    print('✓ черга: збій одного скану не ламає решту')


def test_the_hunter_scans_through_the_shared_queue():
    i = _SRC.index('def scan(self')
    body = _SRC[i:_SRC.index('\n    def ', i + 10)]
    _check('scan_queue.run' in body and _lh.SCAN_JOB in _SRC,
           'скан сканера мусить іти спільною чергою')
    print('✓ сканер ходить на біржу лише через спільну чергу')


def test_the_tickr_scans_go_through_the_queue_too():
    """Інакше ручний скан зі сторінки 📡 Tickr і періодичний скан 💧 могли б
    стартувати одночасно — рівно те, що вимога 4 і забороняє."""
    for _route in ('api_tickr_liquidity_scan', 'api_tickr_liquidity_one'):
        i = _FLASK.index(f'def {_route}(')
        body = _FLASK[i:_FLASK.index('\n    @app.route', i)]
        _check('scan_queue.run' in body,
               f'{_route} мусить іти спільною чергою сканів')
    print('✓ скани 📡 Tickr теж стоять у спільній черзі')


def test_the_periodic_opportunity_scan_is_queued_too():
    """Другий ПЕРІОДИЧНИЙ скан біржі в боті — теж у черзі, інакше він міг би
    стартувати рівно тоді, коли працює скан ліквідності."""
    src = open(os.path.join(_HERE, 'detection',
                            'tickr_opportunity_daemon.py'), encoding='utf-8').read()
    i = src.index('def _run_scan(self')
    body = src[i:src.index('\n    def ', i + 10)]
    _check('scan_queue.run' in body, 'скан можливостей мусить іти спільною чергою')
    print('✓ періодичний скан можливостей теж у спільній черзі')


# ═══════════ 7. UI ═══════════════════════════════════════════════════════
def test_the_panel_sits_under_the_monitor_and_above_poc():
    _check(_HTML.index('id="mm-monitor-panel"')
           < _HTML.index('id="liq-hunter-panel"')
           < _HTML.index('id="poc-setup-panel"'),
           'вікно сканера мусить стояти ПІД МММ-монітором')
    print('✓ UI: вікно сканера — під МММ-монітором')


def test_the_panel_has_a_master_toggle_and_a_settings_accordion():
    i = _HTML.index('id="liq-hunter-panel"')
    sec = _HTML[i:_HTML.index('id="poc-setup-panel"')]
    _check('id="lh-enabled"' in sec, 'потрібен основний тумблер')
    _check("togglePanel('lh')" in sec and 'id="lh-body"' in sec,
           'вікно мусить бути гармошкою')
    _check('<details' in sec and 'Налаштування' in sec,
           'потрібна гармошка налаштувань')
    _check('lhScanNow()' in sec and 'Сканувати' in sec,
           'кнопка ручного скану мусить лишитись')
    for _id in ('lh-exchange', 'lh-universe', 'lh-topn', 'lh-bars', 'lh-sort',
                'lh-minvol', 'lh-minoi', 'lh-interval', 'lh-mass',
                'lh-rescan-flip'):
        _check(f'id="{_id}"' in sec, f'у налаштуваннях немає {_id}')
    print('✓ UI: тумблер, гармошка налаштувань, кнопка «Сканувати»')


def test_the_table_shows_mass_and_magnet():
    i = _HTML.index('id="lh-table"')
    head = _HTML[i:_HTML.index('</thead>', i)]
    _check('Маса ліквідності' in head and 'магніт' in head.lower(),
           'колонки «Маса ліквідності» і «Найсильніший магніт» обовʼязкові')
    # ⚠️ рахуємо саме «<th » — інакше в лічильник потрапляє ще й «<thead>».
    _check(head.count('<th ') == 4, f'очікуємо 4 колонки: {head.count("<th ")}')
    body = _HTML[_HTML.index('function lhRender()'):]
    body = body[:body.index('\n// ')]
    _check("colspan=\"4\"" in body, 'colspan порожнього рядка мусить збігатись')
    print('✓ UI: у таблиці — маса і магніт')


def test_the_page_calls_routes_that_really_exist():
    """Той самий замок, що для `/api/tm/positions/*`: вигадати URL не можна."""
    import re
    used = set(re.findall(r"fetch\('(/api/liq-hunter/[^']+)'", _HTML))
    _check(used, 'сторінка мусить кликати маршрути сканера')
    for u in used:
        _check(f"@app.route('{u}'" in _FLASK, f'маршруту {u} немає у flask_app')
    print(f'✓ UI: всі {len(used)} маршрути сканера існують')


def test_defaults_mirror_the_tickr_block_we_took_the_algorithm_from():
    """Алгоритм узято з блоку 💧 на 📡 Tickr, і дефолти мусять бути ТІ САМІ —
    інакше «як на скріні» означало б інші числа."""
    _check(_lh.DEFAULTS['exchange'] == 'binance'
           and '<option value="binance" selected>' in _TICKR,
           'біржа за замовчуванням мусить збігатись із 📡 Tickr')
    _check(_lh.DEFAULTS['universe'] == 'top'
           and '<option value="top" selected>' in _TICKR, 'список: топ біржі')
    _check(_lh.DEFAULTS['top_n'] == 40 and '<option selected>40</option>' in _TICKR,
           'кількість монет: 40')
    _check(_lh.DEFAULTS['bars'] == 168 and 'value="168" selected' in _TICKR,
           'глибина історії: 168 год')
    _check(_lh.DEFAULTS['sort_by'] == 'pull'
           and '<option value="pull" selected>' in _TICKR, 'сортування: перекіс')
    _check(_lh.DEFAULTS['min_vol_usd'] == 20_000_000
           and 'id="liq-minvol" value="20"' in _TICKR, 'обіг: $20M')
    _check(_lh.DEFAULTS['min_oi_usd'] == 5_000_000
           and 'id="liq-minoi" value="5"' in _TICKR, 'OI: $5M')
    print('✓ дефолти 1-в-1 з блоком 💧 на 📡 Tickr')


def test_the_scanner_never_opens_trades_itself():
    """Це спостереження, а не торгівля: жодного `_open` / `on_signal`."""
    tree = ast.parse(_SRC)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ('_open', 'on_signal', 'intercept', 'manual_open', 'group_open'):
        _check(bad not in names, f'сканер не має кликати {bad}')
    print('✓ сканер нічого не відкриває і не сигналить')



# ═══════════ 8. 🧲 ФІЛЬТР «МАГНІТ НЕ ЗАБЛИЗЬКО» + РІВНА КОЛОНКА (18.09) ════
# Дослівно: «додай в параметри фільтр обмежуючий малий відсоток (за
# замовчуванням 3%), тобто менше цього відсотка від ціни, монети не додавати
# в таблицю» + «відсотки вирівняй в рівну колонку і виділи кольором».
def test_the_magnet_distance_is_read_raw_not_parsed_from_the_label():
    """⚠️ `magnet_dist` — ФОРМАТОВАНИЙ рядок («↓5.58%»), і парсити його заради
    числа не можна (задокументована пастка). Беремо `magnet_row['dist_pct']`."""
    r = _row('AUSDT', 20.0, 80.0, magnet_row={'dist_pct': 5.58})
    _check(_lh.magnet_dist_pct(r) == 5.58, 'відстань мусить братись сирою')
    _check(_lh.magnet_dist_pct(_row('B', 20.0, 80.0)) is None,
           'немає сходинки → None, а НЕ нуль')
    i = _SRC.index('def magnet_dist_pct(')
    body = _SRC[i:_SRC.index('\ndef ', i + 10)]
    _check("'magnet_dist'" not in body.replace("'magnet_dist_pct'", ''),
           'форматований підпис парсити не можна')
    print('✓ 🧲 відстань береться сирим числом, а не з підпису')


def test_a_magnet_too_close_to_price_is_not_added():
    near = _row('NEARUSDT', 20.0, 80.0, magnet_row={'dist_pct': -1.2})
    far = _row('FARUSDT', 20.0, 80.0, magnet_row={'dist_pct': -7.4})
    h = _mk([near, far], direction='SHORT', min_mass_pct=65,
            min_magnet_dist_pct=3)
    res = h.scan('t')
    got = [r['symbol'] for r in h.get_state()['rows']]
    _check(got == ['FARUSDT'], f'магніт за 1.2% мусить бути відсіяний: {got}')
    _check(res['near'] == 1, f'відсів мусить бути НАЗВАНИЙ у розкладі: {res}')
    _check('магніт ближче' in h.get_state()['status'],
           f'статус мовчить про відсів: {h.get_state()["status"]}')
    print('✓ 🧲 монета з близьким магнітом у таблицю не додається')


def test_the_magnet_filter_default_is_three_percent_and_zero_disables_it():
    _check(_lh.DEFAULTS['min_magnet_dist_pct'] == 3.0, 'дефолт — 3%')
    near = _row('NEARUSDT', 20.0, 80.0, magnet_row={'dist_pct': -1.2})
    h = _mk([near], direction='SHORT', min_mass_pct=65, min_magnet_dist_pct=0)
    h.scan('t')
    _check(len(h.get_state()['rows']) == 1, '0 = фільтр вимкнено')
    print('✓ 🧲 дефолт 3%, «0» вимикає фільтр')


def test_an_unknown_distance_is_not_treated_as_close():
    """«Невідомо» ≠ «близько»: вигаданої відмови не даємо (той самий принцип,
    що fail-open у 💧 фільтра входу)."""
    h = _mk([_row('AUSDT', 20.0, 80.0)], direction='SHORT', min_mass_pct=65,
            min_magnet_dist_pct=3)
    res = h.scan('t')
    _check(len(h.get_state()['rows']) == 1 and res['near'] == 0,
           f'рядок без сходинки не має різатись: {res}')
    print('✓ 🧲 магніт без відстані фільтром не ріжеться')


def test_the_row_carries_the_raw_distance_for_the_page():
    h = _mk([_row('AUSDT', 20.0, 80.0, magnet_row={'dist_pct': -7.4})],
            direction='SHORT', min_mass_pct=65)
    h.scan('t')
    r = h.get_state()['rows'][0]
    _check(r['magnet_dist_pct'] == 7.4,
           f'сире число мусить доїхати до сторінки: {r}')
    print('✓ 🧲 сире число відстані їде в рядок (колонка й поріг — одне число)')


def test_ui_has_the_magnet_filter_field_and_an_even_column():
    i = _HTML.index('id="liq-hunter-panel"')
    sec = _HTML[i:_HTML.index('id="poc-setup-panel"')]
    _check('id="lh-magdist"' in sec, 'немає поля «🧲 Магніт ≥, %»')
    _check('min_magnet_dist_pct' in _HTML,
           'ключ мусить і зберігатись, і підставлятись у поле')
    body = _HTML[_HTML.index('function _lhMagnetCell(r) {'):
                 _HTML.index('function lhRender()')]
    # Рівну колонку тримають САМЕ фіксовані ширини сегментів (той самий прийом,
    # що в колонці «🎯 Автопілот»), а не пробіли в тексті.
    _check('width:${w}px' in body and 'text-align' in body,
           'відсотки мусять стояти у сегментах фіксованої ширини')
    _check('#4ade80' in body and '#f87171' in body, 'відсотки мусять мати колір')
    _check('magnet_dist_pct' in body,
           'малюємо з СИРОГО числа — того самого, що живить поріг')
    print('✓ UI: поле «🧲 Магніт ≥ %» + рівна кольорова колонка')


def test_the_settings_summary_has_both_sides_so_it_does_not_drift_right():
    """Скарга 18.09 «поля ховаються, переглянь візуальні відступи»:
    `.sm-settings-summary` — це flex зі `space-between`, і при ОДНОМУ дочірньому
    вузлі підпис відлітав до правого краю. Потрібні ДВА вузли."""
    i = _HTML.index('id="liq-hunter-panel"')
    sec = _HTML[i:_HTML.index('id="poc-setup-panel"')]
    j = sec.index('<summary class="sm-settings-summary"')
    head = sec[j:sec.index('</summary>', j)]
    _check(head.count('<span') >= 2,
           'у підписі гармошки мусить бути і лівий текст, і правий розклад')
    _check('sm-settings-summary-right' in head,
           'правий блок мусить бути ТИМ САМИМ класом, що в решти гармошок')
    _check('id="lh-sum"' in head and 'lh-sum' in _HTML.split('function lhApply')[1],
           'розклад параметрів мусить оновлюватись живими даними')
    print('✓ UI: підпис гармошки не відлітає праворуч і несе розклад')


def test_the_fields_are_one_grid_with_equal_gaps():
    i = _HTML.index('id="liq-hunter-panel"')
    sec = _HTML[i:_HTML.index('id="poc-setup-panel"')]
    _check('.lh-grid' in sec and 'class="lh-grid"' in sec,
           'поля мусять лежати в одній сітці з однаковим gap')
    _check(sec.count('class="lh-f"') >= 9,
           'кожне поле — той самий клас (однакова висота й відступи)')
    # Чекбокс переїхав у ВЛАСНИЙ рядок за роздільником — на скріні він тиснувся
    # збоку до вузького поля.
    _check('lh-flip' in sec and 'flex-basis:100%' in sec,
           'тумблер переcкану мусить стояти окремим рядком')
    print('✓ UI: одна сітка полів + тумблер окремим рядком')


# ═══════════ 9. 📏 ОДИН РЯДОК + 🦓 ЗЕБРА В ТАБЛИЦІ СКАНЕРА (18.09) ════════
# Дослівно: «Навіщо в дві стрічки робити записи, розмісти все в одну стрічку і
# зроби зебру в таблиці, щоб не зливались.» На скріні смуга магніту
# «$0.22000–0.22500» переносилась, і кожен запис займав два рядки.
def test_every_row_is_one_line_and_the_table_is_striped():
    i = _HTML.index('id="liq-hunter-panel"')
    sec = _HTML[i:_HTML.index('id="poc-setup-panel"')]
    _check('#lh-table th, #lh-table td { white-space:nowrap' in sec,
           'комірки мусять лишатись в ОДНУ стрічку')
    _check('#lh-table tbody tr:nth-child(even)' in sec, 'немає зебри')
    _check('#lh-table tbody tr:hover' in sec, 'немає підсвітки рядка')
    body = _HTML[_HTML.index('function _lhMagnetCell(r) {'):
                 _HTML.index('function lhRender()')]
    _check('white-space:nowrap' in sec.split('.lh-seg')[1][:160],
           'сегменти магніту теж не мають переноситись')
    # Смуга ціни довга («$0.22000–0.22500»), тож її сегмент мусить бути ширшим
    # за решту — інакше саме він і переносив рядок.
    _check('seg(152,' in body, 'сегмент ціни магніту завузький для смуги')
    print('✓ 📏 рядок сканера — в одну стрічку, таблиця зі смугами')


def test_the_magnet_share_is_not_printed_with_two_percent_signs():
    """`ladder.make_verdict` віддає `pct` УЖЕ з «%» — на екрані було «23.0%%»."""
    body = _HTML[_HTML.index('function _lhMagnetCell(r) {'):
                 _HTML.index('function lhRender()')]
    _check('${r.magnet_pct}%' not in body,
           'до готового «23.0%» додається ще один знак відсотка')
    _check('${r.magnet_pct}' in body, 'частку магніту все одно треба показати')
    print('✓ 🧲 частка магніту друкується без подвійного «%%»')

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
