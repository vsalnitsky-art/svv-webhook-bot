"""🆕 АЛЕРТ «НОВИЙ OB НА ГРАФІКУ» — реакція, час появи, ціна, 4H, кольори.

**Вимога користувача (09.09), дослівно:** «Мені потрібно моментальна реакція на
появу на графіку нового OB 1H і моментальна відправка повідомлення в "Лог роботи
бота" про появу нового OB в який саме час і поточна ціна монети на графіку. І ще
один додатковий нюанс — потрібно вияснити який на даний момент останній і
актуальний OB 4H і також записати цю інформацію до інформації про OB 1H.»
Плюс: «Розмалюй цю інформацію кольорами, щоб не зливався текст.»

Що саме стережуть ці тести:
  • «зʼявився» = `created_at_t + тривалість бару` (момент, коли блок став
    ВИДИМИЙ), а НЕ `bar_time` самої свічки блоку;
  • «новий» = ЩЕ НЕ ОПРАЦЬОВАНИЙ (пастка breaker: старіший bar_time теж новий);
  • старий блок → ТИХА база, у лог НЕ пишемо (це СТАН, а не подія);
  • 4H рахується ЛИШЕ в момент події і НЕ смикає біржу, коли TF той самий;
  • швидка смуга працює РІВНО на новому барі і НЕ дублює роботу;
  • алерт НІЧОГО не дозволяє і не блокує (ворота входу недоторкані);
  • рядок у логу РОЗФАРБОВАНИЙ зі структурних полів, а не розбором тексту.
"""
import importlib.util
import os
import re
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

# Пакет `detection` підставляємо ЗАРАНЕ: справжній `detection/__init__.py`
# тягне `sleeper_scanner → core → pybit`, тобто півпроєкту. Порожній модуль із
# правильним `__path__` дає імпортувати СУСІДНІ файли напряму.
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg

_spec = importlib.util.spec_from_file_location(
    "smc_scanner_obalert_test", os.path.join(_HERE, "detection", "smc_scanner.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
S = _m.SMCScanner

oba = importlib.import_module('detection.ob_alert')

HOUR = 3600


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


# ── фейкові сусідні модули ────────────────────────────────────────────────
_LOGGED = []


def _install_log():
    """Підміна `detection.activity_log.log_activity` — збираємо записи."""
    _LOGGED.clear()
    mod = types.ModuleType('detection.activity_log')

    def _log(symbol, event, detail='', side=None, source='', extra=None):
        _LOGGED.append({'symbol': symbol, 'event': event, 'detail': detail,
                        'side': side, 'source': source, 'extra': extra or {}})
    mod.log_activity = _log
    sys.modules['detection.activity_log'] = mod
    _pkg.activity_log = mod


def _install_detectors(ob4=None, fail=False):
    """Підміна детекторів для шляху старшого TF (без 700-барових фікстур)."""
    d1 = types.ModuleType('detection.ob_detector')
    d2 = types.ModuleType('detection.smc_structure')

    def _dlob(klines=None, pivots=None, events=None):
        if fail:
            raise RuntimeError('boom')
        return ob4
    d1.detect_last_order_block = _dlob
    d1.detect_order_blocks = lambda **k: []
    d2.detect_smc_structure = lambda *a, **k: {'internal': {'pivots': [], 'events': []},
                                               'swing': {'pivots': [], 'events': []}}
    sys.modules['detection.ob_detector'] = d1
    sys.modules['detection.smc_structure'] = d2
    _pkg.ob_detector, _pkg.smc_structure = d1, d2


def _install_db():
    """`storage.db_operations.get_db` — рахуємо upsert-и, нічого не пишемо."""
    calls = []
    st = sys.modules.get('storage') or types.ModuleType('storage')
    st.__path__ = getattr(st, '__path__', [])
    sys.modules['storage'] = st
    dbo = types.ModuleType('storage.db_operations')
    dbo.get_db = lambda: types.SimpleNamespace(
        upsert_smc_ob_state=lambda sym, tf, ob: calls.append((sym, tf, ob)),
        get_smc_ob_state=lambda sym, tf: None)
    sys.modules['storage.db_operations'] = dbo
    st.db_operations = dbo
    return calls


def _ob(bias='BULLISH', bar_time=1_700_000_000_000, created_at_t=None, tag='CHoCH'):
    """Блок у тому вигляді, що його віддає `detect_last_order_block`."""
    return {'bias': bias, 'bar_high': 10.0, 'bar_low': 9.0,
            'bar_time': bar_time, 'bar_idx': 5,
            'created_at_idx': 9,
            'created_at_t': bar_time if created_at_t is None else created_at_t,
            'created_by_tag': tag}


def _ns(tf='1h', htf='4h', enabled=True, pf=None):
    """Мінімальний сканер: лише те, чого торкається алерт."""
    ns = types.SimpleNamespace()
    ns._settings = {'ob_filter_timeframe': tf, 'ob_alert_enabled': enabled,
                    'ob_alert_htf': htf, 'ob_alert_max_lag_sec': 0,
                    'swing_size': 50, 'internal_size': 5}
    ns._ob_alert_seen = {}
    ns._ob_htf_cache = {}
    ns._ob_done_cycle = {}
    ns._ob_lane_bar = None
    ns._errors = 0
    ns._scan_count = 1
    ns._running = True
    ns._watchlist = ['BTCUSDT']
    ns._persisted = []
    ns._persist_ob_alert_state = lambda force=False: ns._persisted.append(force)
    ns.get_internal_size = lambda: 5
    ns._pf_calls = []

    def _pfk(md, sym, t, lim):
        ns._pf_calls.append((sym, t, lim))
        # ⚠️ Час бару мусить бути СВІЖИЙ: кеш старшого TF дійсний до закриття
        # свого бару, тож бар із 2023-го протухав би ще до першої перевірки і
        # тест «з кешу» падав би на фікстурі, а не на коді.
        if pf is not None:
            return pf
        return [{'t': int(time.time() * 1000), 'close': 78539.9}] * 300
    ns._pf_klines = _pfk
    ns._prefetch_klines = lambda md, syms, specs: None
    for name in ('_ob_alert_tick', '_ob_on_htf', '_ob_lane_due', '_ob_fast_lane'):
        setattr(ns, name, getattr(S, name).__get__(ns))
    return ns


# ═══════════ 1. «ЗʼЯВИВСЯ» = КОЛИ БЛОК СТАВ ВИДИМИЙ ══════════════════════
def test_appeared_is_the_close_of_the_bar_that_created_the_block():
    """⚠️ ГОЛОВНЕ РОЗРІЗНЕННЯ. У блоку ДВА часи, і «зʼявився» — НЕ `bar_time`.

    `bar_time` — свічка самого блоку (лежить назад у часі). Блок стає видимим
    на графіку, коли ЗАКРИЄТЬСЯ бар, на якому спрацював BOS/CHoCH, бо детекція
    працює по ЗАКРИТИХ барах. Отже `created_at_t + тривалість бару`."""
    ob = _ob(bar_time=1_000_000_000, created_at_t=1_000_010_000)
    got = oba.appeared_at(ob, '1h')
    _check(got == 1_000_010_000 + HOUR, f'очікував created_at_t+1год: {got}')
    _check(oba.appeared_at(ob, '4h') == 1_000_010_000 + 4 * HOUR, 'для 4h — 4 години')
    # мілісекунди теж приймаємо (бари приходять і так, і так)
    _check(oba.appeared_at(_ob(created_at_t=1_700_000_000_000), '1h')
           == 1_700_000_000 + HOUR, 'мс мусять звестись до секунд')
    print('✓ «зʼявився» = закриття бару, що створив блок (не bar_time свічки)')


def test_no_creation_time_means_we_never_claim_it_appeared():
    """Часу появи немає → 'stale', а не 'new': написати «зʼявився», не знаючи
    коли, означало б збрехати в самому рядку логу."""
    ob = _ob()
    ob.pop('created_at_t')
    _check(oba.appeared_at(ob, '1h') is None, 'без created_at_t часу не вигадуємо')
    _check(oba.outcome([], ob, '1h', time.time()) == 'stale',
           'невідомий час появи мусить дати stale')
    print('✓ немає часу появи → тиха база, «зʼявився» не пишемо')


# ═══════════ 2. «НОВИЙ» = ЩЕ НЕ ОПРАЦЬОВАНИЙ (пастка breaker) ════════════
def test_older_bar_time_is_still_new_when_not_processed():
    """⚠️ ПАСТКА BREAKER (той самий урок, що для VOB). Коли поточний блок стає
    breaker, він випадає зі списку і «останнім» стає СТАРІШИЙ блок із МЕНШИМ
    `bar_time`. Умова `bt > prev` не спрацювала б НІКОЛИ — реальна зміна блоку
    на графіку лишилась би невидимою."""
    now = 2_000_000_000.0
    newer, older = 1_999_000_000, 1_990_000_000
    seen = oba.seen_add([], newer)
    ob_old = _ob(bar_time=older, created_at_t=now - 60 - HOUR)   # щойно видимий
    _check(oba.outcome(seen, ob_old, '1h', now) == 'new',
           'СТАРІШИЙ, але не опрацьований блок — це НОВИЙ блок')
    _check(oba.outcome(oba.seen_add(seen, older), ob_old, '1h', now) == 'duplicate',
           'після опрацювання — duplicate')
    print('✓ новим є будь-який НЕ опрацьований блок (breaker не глушить алерт)')


def test_stable_state_is_not_an_event():
    """Той самий блок на кожному циклі → 'duplicate', у лог НЕ пишемо.
    Інакше лог заповнився б СТАНОМ (рівно той флуд, через який лог VOB уже
    довелось чистити)."""
    now = 2_000_000_000.0
    ob = _ob(bar_time=1_999_000_000, created_at_t=now - 60 - HOUR)
    seen = oba.seen_add([], ob['bar_time'])
    for _ in range(5):
        _check(oba.outcome(seen, ob, '1h', now) == 'duplicate', 'стан ≠ подія')
    print('✓ незмінний блок = СТАН → жодного рядка в лозі')


def test_old_block_is_a_silent_baseline():
    """Блок, що зʼявився давно, «щойно» не зʼявився → 'stale'. Саме це не дає
    рестарту вилити в лог поточний OB усіх монет watchlist."""
    now = 2_000_000_000.0
    ob = _ob(bar_time=1_900_000_000, created_at_t=now - 10 * HOUR)
    _check(oba.outcome([], ob, '1h', now) == 'stale', 'давній блок — не подія')
    print('✓ давній блок → тиха база (рестарт не флудить лог)')


def test_freshness_window_is_one_bar_by_default():
    """0 = АВТО = один бар TF (той самий принцип «0 = авто», що у
    `vob_alert_max_age_bars`). У межах поточного бару новішого блоку фізично
    існувати не може, тож один бар — це «найсвіжіший можливий стан»."""
    _check(oba.max_lag_for('1h', 0) == HOUR, 'авто для 1h = 3600с')
    _check(oba.max_lag_for('4h', 0) == 4 * HOUR, 'авто для 4h = 14400с')
    _check(oba.max_lag_for('1h', 90) == 90, 'явне значення має пріоритет')
    now = 2_000_000_000.0
    ob = _ob(bar_time=1_999_000_000, created_at_t=now - HOUR - 120)  # видимий 120с тому
    _check(oba.outcome([], ob, '1h', now) == 'new', 'у межах вікна — подія')
    _check(oba.outcome([], ob, '1h', now, 60) == 'stale', 'вужче вікно — тиха база')
    print('✓ вікно свіжості: 0 = один бар TF, явне значення поважається')


def test_seen_list_is_capped_and_does_not_mutate_input():
    src = [1, 2, 3]
    out = oba.seen_add(src, 4, cap=3)
    _check(src == [1, 2, 3], f'вхідний список мутувати НЕ можна: {src}')
    _check(out == [2, 3, 4], f'кеп мусить забути найдавніший: {out}')
    _check(oba.seen_add([5], 5) == [5], 'повтор не дублюється')
    print('✓ список опрацьованих: кеп працює, вхід не мутується')


# ═══════════ 3. ФОРМАТ РЯДКА ══════════════════════════════════════════════
def test_text_carries_everything_the_user_asked_for():
    """Порядок і зміст — як у прикладі користувача:
    «BTCUSDT LONG OB 1H LONG OB 4H зʼявився 09.09.26 о 21:25 $78,539.90».
    Символ і напрямок стоять ОКРЕМИМИ колонками таблиці, тож у `detail` —
    решта."""
    app = 1_757_453_100            # 09.09.2025 21:25 UTC
    p = oba.build_parts('BTCUSDT', '1h', 'LONG', 'CHoCH', '4h', 'LONG',
                        78539.9, app, app + 8)
    t = oba.build_text(p)
    for need in ('LONG OB 1H', 'LONG OB 4H', 'зʼявився', '09.09.25 о 21:25',
                 'UTC', '$78,539.90', '8с'):
        _check(need in t, f'у рядку мусить бути «{need}»: {t}')
    _check(t.index('OB 1H') < t.index('OB 4H') < t.index('зʼявився')
           < t.index('$78,539.90'), f'порядок як у вимозі: {t}')
    print(f'✓ рядок логу: {t}')


def test_both_times_are_shown_so_the_gap_is_explainable():
    """🐞 ПИТАННЯ КОРИСТУВАЧА (LITUSDT): «OB, який промальовувався на графіку
    орієнтовно о 12 годині, визначився аж у 21 — це нормально?»

    Так, і причина саме в ДВОХ РІЗНИХ часах: бокс МАЛЮЄТЬСЯ на свічці блоку
    (`bar_time`, «назад у часі»), а ІСНУВАТИ починає, коли закрився бар із
    BOS/CHoCH. Рядок мусить нести ОБИДВА — тоді питання відпадає само."""
    app = 1_757_453_100                       # 09.09.25 21:25 UTC — поява
    bar = app - 9 * HOUR                      # свічка блоку — на 9 год раніше
    p = oba.build_parts('LITUSDT', '1h', 'SHORT', 'CHoCH', '4h', 'LONG',
                        4.654, app, app + 187, bar_time=bar)
    t = oba.build_text(p)
    _check('свічка блоку 09.09.25 о 12:25' in t, f'час свічки блоку: {t}')
    _check('зʼявився 09.09.25 о 21:25' in t, f'час появи: {t}')
    _check(t.index('свічка блоку') < t.index('зʼявився'),
           f'спершу «де намальовано», потім «коли виник»: {t}')
    # Без bar_time сегмент просто відсутній — нічого не вигадуємо
    p2 = oba.build_parts('LITUSDT', '1h', 'SHORT', None, None, None,
                         4.654, app, app + 10)
    _check(p2['bar_time_txt'] is None and 'свічка блоку' not in oba.build_text(p2),
           'немає bar_time → сегмента немає')
    print(f'✓ обидва часи в рядку: {t}')


def test_missing_htf_is_said_out_loud():
    """4H не порахувався → пишемо «немає», а не мовчимо: інакше виглядало б,
    ніби старшого блоку не існує."""
    app = 1_757_453_100
    t = oba.build_text(oba.build_parts('BTCUSDT', '1h', 'SHORT', 'BOS', '4h',
                                       None, 1.2345, app, app + 3))
    _check('OB 4H: немає' in t, f'відсутність 4H мусить бути названа: {t}')
    print('✓ немає 4H → сказано прямо («OB 4H: немає»)')


def test_price_formatter_mirrors_the_page():
    """⚠️ ТА САМА ЦІНА — ОДИН ФОРМАТ. `fmt_price` мусить дзеркалити
    `fmtPriceJS` зі `smart_money.html`: та сама ціна в 🧾 Лозі й у таблиці не
    має писатись двома різними числами (урок PD-зони)."""
    html = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
                encoding='utf-8').read()
    i = html.index('function fmtPriceJS')
    js = html[i:i + 600]
    pairs = [(float(a), int(b)) for a, b in
             re.findall(r"if \(p < ([\d.]+)\) return '\$' \+ p\.toFixed\((\d+)\)", js)]
    _check(len(pairs) >= 4, f'не розпарсив пороги fmtPriceJS: {pairs}')
    for thr, dp in pairs:
        v = thr / 2.0
        got = oba.fmt_price(v)
        dec = len(got.split('.')[1]) if '.' in got else 0
        _check(dec == dp, f'{v}: Python дав {dec} знаків, JS обіцяє {dp} ({got})')
    _check('minimumFractionDigits: 2' in js, 'великі числа в JS — 2 знаки')
    _check(oba.fmt_price(78539.9) == '$78,539.90',
           f'великі числа — кома + 2 знаки: {oba.fmt_price(78539.9)}')
    _check(oba.fmt_price(0) == '—' and oba.fmt_price(None) == '—',
           'порожня ціна → прочерк, а не «$0»')
    print(f'✓ формат ціни дзеркалить fmtPriceJS ({len(pairs)} порогів звірено)')


def test_lag_is_human_readable():
    _check(oba.fmt_lag(8) == '8с', oba.fmt_lag(8))
    _check(oba.fmt_lag(200) == '3хв 20с', oba.fmt_lag(200))
    _check(oba.fmt_lag(180) == '3хв', oba.fmt_lag(180))
    _check(oba.fmt_lag(7200) == '2г', oba.fmt_lag(7200))
    _check(oba.fmt_lag(-5) == '0с', 'зсув годинника не дає відʼємну реакцію')
    print('✓ швидкість реакції читається людиною (8с / 3хв 20с / 2г)')


def test_parts_mirror_the_text():
    """ЗАМОК ВІД РОЗХОДЖЕННЯ: UI малює з `parts`, лог/CSV несе `text`. Якщо
    вони розійдуться, на екрані і в експорті буде РІЗНЕ (той самий замок, що
    `test_verdict_parts_mirror_the_text` для драбини)."""
    app = 1_757_453_100
    p = oba.build_parts('ETHUSDT', '1h', 'SHORT', 'BOS', '4h', 'LONG',
                        0.0456789, app, app + 45)
    t = oba.build_text(p)
    for k in ('side1', 'tf1', 'side4', 'tf4'):
        _check(str(p[k]) in t, f'{k}={p[k]} мусить бути в тексті: {t}')
    _check(p['appeared_txt'] in t and p['price_txt'] in t and p['lag_txt'] in t,
           f'усі відформатовані шматки мусять бути в тексті: {t}')
    print('✓ parts і text не розходяться')


# ═══════════ 4. ПОДІЯ В ЛОЗІ ══════════════════════════════════════════════
def test_price_key_is_p_not_close():
    """🐞 ДЕФЕКТ, ЗНАЙДЕНИЙ НА ПРОДІ (09.09). У КОЖНОМУ рядку логу замість ціни
    стояв прочерк: `⚡ виявлено за 4хв 46с · —`.

    Причина: `market_data.fetch_klines` віддає `[{p, v, b, s, h, l, o, t}, …]`
    — ціна закриття лежить у **`p`**, а перша версія читала `bar['close']`.
    Бари були на місці, ціна просто не діставалась. Тест фіксує САМЕ контракт
    fetch_klines, щоб помилка не повернулась."""
    _check(oba.close_of({'p': 78539.9, 't': 1}) == 78539.9,
           'ключ `p` — основний контракт fetch_klines')
    _check(oba.close_of({'close': 5.0}) == 5.0, 'ohlc чарта несе `close` — фолбек')
    _check(oba.close_of({'c': 7.0}) == 7.0, '`c` — теж фолбек')
    _check(oba.close_of({'h': 9.0}) is None, 'чужі ключі ціною НЕ вважаємо')
    for bad in (None, {}, {'p': 0}, {'p': 'abc'}, {'p': None}):
        _check(oba.close_of(bad) is None, f'сміття → None: {bad}')
    # І докстрінг market_data мусить і далі обіцяти `p` — інакше контракт поїхав
    md = open(os.path.join(_HERE, 'detection', 'market_data.py'),
              encoding='utf-8').read()
    _check('{p, v, b, s, h, l, o, t}' in md,
           'контракт fetch_klines змінився — перевірити close_of')
    print('✓ ціна береться з `p` (контракт fetch_klines), а не з `close`')


def test_new_ob_logs_exactly_once_with_time_and_price():
    """Повна вимога в одному тесті: НОВИЙ блок → ОДИН рядок, із часом появи,
    ціною і 4H; повтор — жодного рядка.

    ⚠️ Бари ТУТ — у форматі `fetch_klines` (`p`), а не вигаданому `close`:
    саме на вигаданому форматі перша версія тесту пропустила прод-дефект."""
    _install_log(); _install_db(); _install_detectors(ob4=_ob(bias='BULLISH'))
    now = time.time()
    ns = _ns()
    ns._ob_alert_seen['BTCUSDT'] = [1]        # база вже є → не «перший показ»
    ob = _ob(bias='BULLISH', bar_time=int((now - 2 * HOUR) * 1000),
             created_at_t=int((now - HOUR - 30) * 1000))
    out = ns._ob_alert_tick('BTCUSDT', None, '1h', ob,
                            [{'t': int(now * 1000), 'p': 78539.9}])
    _check(out == 'new', f'мусив бути new: {out}')
    _check(len(_LOGGED) == 1, f'рівно ОДИН рядок: {_LOGGED}')
    e = _LOGGED[0]
    _check(e['event'] == 'ob_new' and e['source'] == 'OB', e)
    _check(e['side'] == 'LONG', f'напрямок 1H-блоку: {e["side"]}')
    _check('$78,539.90' in e['detail'], f'ціна з ЖИВОГО бару: {e["detail"]}')
    _check('LONG OB 4H' in e['detail'], f'4H мусить бути дописаний: {e["detail"]}')
    _check(isinstance(e['extra'].get('parts'), dict), 'для UI потрібні parts')
    _check('⚡ виявлено за' in e['detail'],
           f'у робочому стані підпис — саме про швидкість: {e["detail"]}')
    # повтор того самого блоку — тиша
    ns._ob_alert_tick('BTCUSDT', None, '1h', ob, [{'t': 1, 'p': 78539.9}])
    _check(len(_LOGGED) == 1, f'повтор НЕ має писати нічого: {_LOGGED}')
    print(f'✓ новий OB → один рядок: {e["detail"]}')


def test_first_sight_after_restart_does_not_claim_our_speed():
    """🐞 ДРУГИЙ ПРОД-ДЕФЕКТ. Після рестарту 6 монет дали
    `⚡ виявлено за 4хв 43с` — і це читалось як «бот думав 4 хвилини». Насправді
    бот піднявся через 4хв після закриття бару: число міряло ВІК БЛОКУ, а не
    нашу реакцію. Перший показ монети мусить бути підписаний ІНАКШЕ."""
    _install_log(); _install_db(); _install_detectors(ob4=None)
    now = time.time()
    ns = _ns(htf='')                      # база порожня → перший показ
    ob = _ob(bias='SHORT' and 'BEARISH', bar_time=int((now - 2 * HOUR) * 1000),
             created_at_t=int((now - HOUR - 283) * 1000))
    out = ns._ob_alert_tick('LTCUSDT', None, '1h', ob,
                            [{'t': int(now * 1000), 'p': 104.5}])
    _check(out == 'new', out)
    d = _LOGGED[0]['detail']
    _check('перший показ після старту' in d, f'мусить бути чесний підпис: {d}')
    _check('⚡ виявлено за' not in d, f'⚡ про швидкість тут БРЕШЕ: {d}')
    _check('4хв 43с' in d, f'вік блоку все одно показуємо: {d}')
    print(f'✓ перший показ підписаний чесно: {d}')


def test_old_block_marks_baseline_without_logging():
    _install_log(); _install_db(); _install_detectors(ob4=None)
    now = time.time()
    ns = _ns()
    ob = _ob(created_at_t=int((now - 20 * HOUR) * 1000))
    out = ns._ob_alert_tick('BTCUSDT', None, '1h', ob, [{'t': 1, 'close': 5.0}])
    _check(out == 'stale', out)
    _check(not _LOGGED, f'давній блок у лог НЕ пишемо: {_LOGGED}')
    _check(ns._ob_alert_seen.get('BTCUSDT'), 'але мусить бути взятий за базу')
    print('✓ давній блок: база взята, лог чистий')


def test_toggle_off_does_nothing_at_all():
    _install_log(); _install_db(); _install_detectors()
    ns = _ns(enabled=False)
    out = ns._ob_alert_tick('BTCUSDT', None, '1h', _ob(), [{'t': 1, 'close': 5.0}])
    _check(out == 'off' and not _LOGGED and not ns._ob_alert_seen,
           f'вимкнено → жодних побічних ефектів: {out} {_LOGGED}')
    print('✓ тумблер OFF → нічого не рахуємо і нічого не пишемо')


def test_no_ob_is_silent():
    _install_log(); _install_db(); _install_detectors()
    ns = _ns()
    for bad in (None, {}, _ob(bias=None), _ob(bias='BULLISH', bar_time=0)):
        _check(ns._ob_alert_tick('BTCUSDT', None, '1h', bad, []) == 'no_ob',
               f'без блоку/напрямку нічого не пишемо: {bad}')
    _check(not _LOGGED, _LOGGED)
    print('✓ немає блоку або напрямку → тиша')


# ═══════════ 5. СТАРШИЙ TF (4H) ═══════════════════════════════════════════
def test_same_tf_does_not_ask_the_exchange_twice():
    """⚠️ Коли старший TF ЗБІГАЄТЬСЯ з TF воріт — беремо вже порахований блок.
    Два різні числа для того самого TF були б прямим протиріччям (і зайвим
    запитом)."""
    _install_log(); _install_db(); _install_detectors(ob4=_ob(bias='BEARISH'))
    ns = _ns(tf='4h', htf='4h')
    same = _ob(bias='BULLISH')
    got, note = ns._ob_on_htf('BTCUSDT', None, '4h', '4h', same)
    _check(got is same, 'мусить повернути ТОЙ САМИЙ блок')
    _check(not ns._pf_calls, f'жодного запиту барів: {ns._pf_calls}')
    _check('той самий TF' in note, note)
    print('✓ HTF == TF воріт → без другого запиту, те саме число')


def test_htf_is_computed_once_and_cached_until_its_bar_closes():
    """4H рахується ЛИШЕ в момент події, і повторний виклик у межах того самого
    4H-бару бере кеш — постійного тиску на біржу немає."""
    _install_log(); _install_db(); _install_detectors(ob4=_ob(bias='BEARISH'))
    ns = _ns()
    ob4, note = ns._ob_on_htf('BTCUSDT', None, '4h', '1h', None)
    _check((ob4 or {}).get('bias') == 'BEARISH', ob4)
    _check(len(ns._pf_calls) == 1 and ns._pf_calls[0][1] == '4h',
           f'рівно один запит 4h: {ns._pf_calls}')
    ns._ob_on_htf('BTCUSDT', None, '4h', '1h', None)
    _check(len(ns._pf_calls) == 1, f'другий раз — з кешу: {ns._pf_calls}')
    print('✓ 4H: один запит у момент події, далі кеш до закриття свого бару')


def test_htf_failure_is_reported_not_hidden():
    _install_log(); _install_db(); _install_detectors(fail=True)
    ns = _ns()
    ob4, note = ns._ob_on_htf('BTCUSDT', None, '4h', '1h', None)
    _check(ob4 is None and note, f'невдача мусить нести причину: {note}')
    ns2 = _ns(pf=[])
    ob4b, note2 = ns2._ob_on_htf('BTCUSDT', None, '4h', '1h', None)
    _check(ob4b is None and 'історії' in note2, f'мало барів → причина: {note2}')
    print('✓ 4H не вийшов → причина названа, а не проглинута')


# ═══════════ 6. ШВИДКІСТЬ ═════════════════════════════════════════════════
def test_fast_lane_runs_once_per_bar_not_every_cycle():
    """⚠️ КЛЮЧОВА ЕКОНОМІЯ. OB рахується по ЗАКРИТИХ барах, отже змінитись може
    ЛИШЕ коли закрився бар TF воріт. Тому смуга працює РІВНО на першому циклі
    після закриття бару, а решту циклів коштує один `if`."""
    ns = _ns()
    base = 1_757_448_000.0            # 09.09.25 20:00 UTC — РІВНА година
    _check(base % HOUR == 0, 'база тесту мусить лежати на межі бару')
    _check(ns._ob_lane_due('1h', base) is True, 'перший виклик — мусить узяти базу')
    _check(ns._ob_lane_due('1h', base + 10) is False, 'у межах бару — не треба')
    _check(ns._ob_lane_due('1h', base + 3599) is False, 'усе ще той самий бар')
    _check(ns._ob_lane_due('1h', base + HOUR) is True, 'новий бар → смуга потрібна')
    _check(ns._ob_lane_due('1h', base + HOUR + 5) is False, 'і знову тиша')
    print('✓ швидка смуга: рівно раз на бар TF воріт, не щоцикл')


def test_lane_also_fires_mid_cycle_not_only_at_the_start():
    """🐞 ПРОД-КЕЙС LITUSDT (09.09): бар закрився о 21:00, а реакція прийшла аж
    через **3хв 07с**. Причина не в детекції: смуга кликалась ЛИШЕ на початку
    циклу, тож закриття бару ПОСЕРЕД проходу чекало наступного циклу.

    Перевірка коштує один `if` на монету (`_ob_lane_due` — чиста арифметика над
    `now // tf_secs`, без I/O), а очікування зводить до однієї монети."""
    src = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    i = src.index('for symbol in list(self._watchlist):')
    body = src[i:i + 1200]
    _check('self._ob_fast_lane(md)' in body,
           'смуга мусить перевірятись і МІЖ монетами, а не лише на старті циклу')
    _check(body.index('self._ob_fast_lane(md)') < body.index('_pf_klines'),
           'перевірка мусить стояти ДО важкої роботи по монеті')
    # І це ДРУГИЙ виклик — перший лишається на початку циклу
    _check(src.count('self._ob_fast_lane(md)') >= 2,
           'потрібні ОБИДВА виклики: на старті циклу і в проході')
    print('✓ смуга ловить закриття бару і посеред циклу')


def test_new_bar_invalidates_work_already_done_this_cycle():
    """⚠️ НАЙТОНШЕ МІСЦЕ другого виклику. Монети, опрацьовані ДО закриття бару,
    мають `_ob_done_cycle == _scan_count`, і гейт у `_update_smc_ob` мовчки
    пропустив би саме їх — тобто найгірший випадок (бар закрився одразу після
    монети №1) лишився б без реакції на цілий цикл, попри другий виклик."""
    _install_log(); _install_db(); _install_detectors(ob4=None)
    ns = _ns()
    ns._watchlist = ['BTCUSDT', 'ETHUSDT']
    seen = []
    ns._update_smc_ob = lambda sym, md: seen.append(sym)
    # монета вже пройшла в ЦЬОМУ циклі
    ns._ob_done_cycle['BTCUSDT'] = ns._scan_count
    ns._ob_fast_lane(None)
    _check('BTCUSDT' in seen,
           f'після закриття бару вже опрацьовану монету треба ПЕРЕрахувати: {seen}')
    _check('ETHUSDT' in seen, f'решту — теж: {seen}')
    src = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    i = src.index('def _ob_fast_lane(')
    body = src[i:i + 2200]
    _check('self._ob_done_cycle.clear()' in body,
           'новий бар мусить знімати позначку «вже пораховано цим циклом»')
    _check(body.index('_ob_lane_due') < body.index('_ob_done_cycle.clear()'),
           'чистити гейт лише КОЛИ бар справді закрився, а не щоразу')
    print('✓ новий бар знімає позначку — жодна монета не лишається без перерахунку')


def test_fast_lane_marks_the_cycle_so_work_is_not_done_twice():
    """Смуга кличе ТОЙ САМИЙ `_update_smc_ob` (другої копії логіки немає), а
    позначка циклу не дає звичайному проходу зробити ту саму роботу вдруге."""
    _install_log(); _install_db(); _install_detectors(ob4=None)
    ns = _ns()
    ns._update_smc_ob = lambda sym, md: ns._pf_calls.append(('upd', sym, 0))
    n = ns._ob_fast_lane(None)
    _check(n == 1, f'опрацьовано монет: {n}')
    _check(ns._ob_done_cycle.get('BTCUSDT') == ns._scan_count,
           f'цикл мусить бути позначений: {ns._ob_done_cycle}')
    # у межах того самого бару друга смуга не потрібна
    _check(ns._ob_fast_lane(None) == 0, 'у межах бару смуга не повторюється')
    print('✓ смуга позначає цикл → подвійної роботи немає')


def test_lane_cycle_does_not_download_the_same_bars_twice():
    """⚠️ Смуга СПОЖИВАЄ (`pop`) бари `(ob_tf, 700)`, а `_update_smc_ob` цього
    циклу пропускається по гейту — отже повний префетч качав би ті самі 700
    барів на КОЖНУ монету вдруге, і ніхто їх не забрав би. Перевіряємо, що
    набір викидається саме тоді, коли смуга відпрацювала."""
    src = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    i = src.index('_lane_n = self._ob_fast_lane(md)')
    blk = src[i:i + 1400]
    _check('if _lane_n:' in blk, 'викидати набір лише коли смуга відпрацювала')
    _check('sp != _ob_spec' in blk, 'саме (ob_tf, 700) мусить бути прибраний')
    _check(blk.index('if _lane_n:') < blk.index('_prefetch_klines'),
           'фільтр специфікацій мусить стояти ДО повного префетчу')
    # ліміт 700 мусить лишатись унікальним для OB — інакше ми зняли б чужі бари
    _check(_m.KLINES_LIMIT != 700,
           'якщо головний ліміт стане 700, фільтр зніме ЧУЖІ бари — переробити')
    print('✓ на циклі зі смугою ті самі бари НЕ качаються вдруге')


def test_guard_stops_the_second_computation_in_the_same_cycle():
    """Сам гейт у `_update_smc_ob`: уже пораховано цим циклом → виходимо ДО
    будь-якої мережі (саме це робить ранню смугу безкоштовною)."""
    _install_log(); _install_db(); _install_detectors(ob4=None)
    ns = _ns()
    ns._update_smc_ob = S._update_smc_ob.__get__(ns)
    ns._ob_done_cycle['BTCUSDT'] = ns._scan_count
    ns._update_smc_ob('BTCUSDT', None)
    _check(not ns._pf_calls, f'жодного запиту барів після гейта: {ns._pf_calls}')
    print('✓ гейт циклу спрацьовує ДО мережі')


# ═══════════ 7. АЛЕРТ НІЧОГО НЕ ЗМІНЮЄ У ВОРОТАХ ══════════════════════════
def test_alert_is_message_only_and_does_not_touch_the_entry_gate():
    """⚠️ Алерт — ЛИШЕ повідомлення. `_ob_filter_allows` (ворота входу) не має
    жодної згадки про нього, інакше «додали рядок у лог» тихо змінило б
    ТОРГІВЛЮ."""
    import inspect
    src = inspect.getsource(S._ob_filter_allows)
    for bad in ('ob_alert', '_ob_alert_seen', 'ob_new'):
        _check(bad not in src, f'ворота входу не мусять знати про «{bad}»')
    _check('ob_alert' not in inspect.getsource(S._current_ob_bartime),
           'такт vob_one_per_ob теж недоторканий')
    print('✓ алерт не впливає ні на ворота входу, ні на такт VOB')


def test_settings_defaults_are_a_new_key_no_migration_needed():
    d = _m.DEFAULT_SETTINGS
    _check(d.get('ob_alert_enabled') is True, d.get('ob_alert_enabled'))
    _check(d.get('ob_alert_htf') == '4h', d.get('ob_alert_htf'))
    _check(d.get('ob_alert_max_lag_sec') == 0, d.get('ob_alert_max_lag_sec'))
    # TF воріт НЕ дублюємо окремим ключем — два джерела розійшлись би.
    _check('ob_alert_timeframe' not in d,
           'TF беремо з ob_filter_timeframe, окремого ключа бути не повинно')
    print('✓ дефолти: УВІМК, 4h, авто-вікно; окремого TF немає')


# ═══════════ 8. КОЛЬОРИ В UI ══════════════════════════════════════════════
def test_ui_paints_the_row_from_structured_fields():
    """Вимога «розмалюй кольорами, щоб не зливався текст»: рядок малюється зі
    СТРУКТУРНИХ `x.parts`, а НЕ розбором українського тексту, і кожна за
    природою різна річ має свій колір."""
    html = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
                encoding='utf-8').read()
    _check("ob_new: ['🆕'" in html, 'подія мусить мати значок і колір у _actMeta')
    _check('value="ob_new"' in html, 'і бути у випадайці фільтра логу')
    i = html.index('function _obNewHTML')
    fn = html[i:i + 2600]
    for need in ('p.side1', 'p.tf1', 'p.side4', 'p.tf4', 'p.appeared_txt',
                 'p.lag_txt', 'fmtPriceJS'):
        _check(need in fn, f'рендер мусить брати «{need}» зі структурних полів')
    _check('#22c55e' in html[html.index('_OB_SIDE_COL'):html.index('_OB_SIDE_COL') + 120]
           and '#ef4444' in html[html.index('_OB_SIDE_COL'):html.index('_OB_SIDE_COL') + 120],
           'LONG зелений / SHORT червоний')
    _check('#fbbf24' in fn, 'старший TF — окремий (бурштиновий) колір')
    _check('#93c5fd' in fn, 'TF воріт — свій колір')
    # і рендер справді підключений у таблиці логу
    _check("s.event === 'ob_new' && s.x && s.x.parts" in html,
           'розфарбований рендер мусить бути підключений до таблиці логу')
    _check('_obNewHTML(s.x.parts, esc(s.detail))' in html,
           'із фолбеком на плаский текст, якщо parts немає')
    print('✓ UI фарбує рядок зі структурних полів (1H / 4H / час / ціна)')


def test_ob_new_is_always_its_own_row():
    """🐞 ТРЕТІЙ ПРОД-ДЕФЕКТ + пряма вимога: «Зроби цей запис окремим, не в купі
    записів, де нічого не можна розібрати» → «окремою стрічкою».

    Таблиця логу ЗШИВАЄ події однієї монети+сторони в межах 180с в ОДИН рядок.
    На проді за 8 годин було signal 156 + rejected 134 + sltp 130 — тож рядок
    «🆕 Новий OB» гарантовано приклеювався до ланцюга угоди і зникав з очей.
    Це подія ІНШОЇ природи (зміна СТРУКТУРИ на графіку, а не крок угоди), тож
    зшивати її неправильно й по суті."""
    html = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
                encoding='utf-8').read()
    i = html.index('const SOLO_EVENTS')
    blk = html[i:i + 900]
    _check("'ob_new'" in blk, 'ob_new мусить бути у списку «окремою стрічкою»')
    _check('chains.push(' in blk and 'solo: true' in blk,
           'окрема подія мусить створювати ВЛАСНИЙ ланцюг')
    # ⚠️ Найтонше місце: після окремого рядка НЕ можна оновлювати lastByKey —
    # інакше наступна подія монети приклеїлась би вже до «Новий OB».
    _check('lastByKey' not in blk.split('return;')[0],
           'lastByKey у гілці solo чіпати НЕ можна')
    _check(blk.split('return;')[0].count('SOLO_EVENTS.has') == 1,
           'перевірка solo мусить стояти ДО звичайного групування')
    # І рядок мусить ЧИТАТИСЬ як окремий, а не лише формально ним бути
    _check('c.solo' in html and 'box-shadow:inset 3px 0 0' in html,
           'окрема стрічка мусить мати візуальний кант')
    print('✓ «Новий OB» — завжди ОКРЕМА стрічка (і візуально теж)')


if __name__ == '__main__':
    test_appeared_is_the_close_of_the_bar_that_created_the_block()
    test_no_creation_time_means_we_never_claim_it_appeared()
    test_older_bar_time_is_still_new_when_not_processed()
    test_stable_state_is_not_an_event()
    test_old_block_is_a_silent_baseline()
    test_freshness_window_is_one_bar_by_default()
    test_seen_list_is_capped_and_does_not_mutate_input()
    test_text_carries_everything_the_user_asked_for()
    test_both_times_are_shown_so_the_gap_is_explainable()
    test_missing_htf_is_said_out_loud()
    test_price_formatter_mirrors_the_page()
    test_lag_is_human_readable()
    test_parts_mirror_the_text()
    test_price_key_is_p_not_close()
    test_new_ob_logs_exactly_once_with_time_and_price()
    test_first_sight_after_restart_does_not_claim_our_speed()
    test_old_block_marks_baseline_without_logging()
    test_toggle_off_does_nothing_at_all()
    test_no_ob_is_silent()
    test_same_tf_does_not_ask_the_exchange_twice()
    test_htf_is_computed_once_and_cached_until_its_bar_closes()
    test_htf_failure_is_reported_not_hidden()
    test_fast_lane_runs_once_per_bar_not_every_cycle()
    test_lane_also_fires_mid_cycle_not_only_at_the_start()
    test_new_bar_invalidates_work_already_done_this_cycle()
    test_fast_lane_marks_the_cycle_so_work_is_not_done_twice()
    test_lane_cycle_does_not_download_the_same_bars_twice()
    test_guard_stops_the_second_computation_in_the_same_cycle()
    test_alert_is_message_only_and_does_not_touch_the_entry_gate()
    test_settings_defaults_are_a_new_key_no_migration_needed()
    test_ui_paints_the_row_from_structured_fields()
    test_ob_new_is_always_its_own_row()
    print('\nУсі тести алерту «новий OB» пройдено ✅')
