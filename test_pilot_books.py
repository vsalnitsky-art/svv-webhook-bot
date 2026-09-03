"""🎯 АВТОПІЛОТ: СВІЙ СТАН НА КОЖНУ КНИГУ + ЧЕСНИЙ ЛІЧИЛЬНИК ТРЕЙЛІВ.

Кейс TRXUSDT (03.09, знайдено на скріні таблиці paper-позицій):

    TRXUSDT LONG · вхід $0.32510 · Manual SL 0.3205 · ціль $0.34220
    у колонці «🎯 Автопілот»: 🛡 … 4.3% **114.38R** 🛡432

Обидва підсвічені числа — неправда, і це ДВА РІЗНІ дефекти:

1. **R = 114.38 арифметично неможливий.** З видимих чисел
   `(0.34220-0.32510)/(0.32510-0.32050) = 3.72R`. Для 114.38R стоп мусив би
   стояти за 0.046% від входу. Причина: `_pilot_state` і `_pilot_at` були
   ключовані ЛИШЕ символом, а по монеті можуть одночасно стояти РЕАЛЬНА і
   ПАПЕРОВА позиції з різними входом/стопом. `/api/tm/state` підмішував ОДИН
   знімок в ОБИДВІ таблиці → у paper-рядку світилось R реальної угоди.
   Побічно: спільний троттл блокував пілот другої книги на цілий PILOT_TTL,
   а закриття однієї книги стирало стан іншої.

2. **🛡432 — лічильник рахував НАМІРИ.** Приріст стояв у блоці запису стану
   (`act == 'trail'`), який виконується ДО самої спроби трейлу. Manual SL на
   скріні бурштиновий = його веде ОПЕРАТОР, тобто трейл щоразу впирався в
   замок і не застосовувався — стоп не зрушив жодного разу, а лічильник за
   2.4 год намотав 432 (такт 20с) і тултип писав «Стоп підтягнуто 432 раз(и)».
"""
import inspect
import os
import sys
import types
import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _n in ('pybit', 'pybit.unified_trading'):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules['pybit.unified_trading'].HTTP = object

_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_LOG = []


def _stubs():
    lg = types.ModuleType('detection.activity_log')
    lg.log_activity = lambda sym, kind, text, **kw: _LOG.append((kind, text))
    sys.modules['detection.activity_log'] = lg
    _pkg.activity_log = lg

    ff = types.ModuleType('detection.fuel_filter')
    ff.get_fuel_filter = lambda: types.SimpleNamespace(get_settings=lambda: {})
    sys.modules['detection.fuel_filter'] = ff

    st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
    sys.modules.setdefault('storage', st)
    db = types.ModuleType('storage.db_operations')
    db.get_db = lambda: types.SimpleNamespace(get_smc_ob_state=lambda s, tf: None)
    sys.modules['storage.db_operations'] = db

    sc = types.ModuleType('detection.smc_scanner')
    sc.get_smc_scanner = lambda: types.SimpleNamespace(get_settings=lambda: {})
    sys.modules['detection.smc_scanner'] = sc


_stubs()

# 🎯 План підміняємо, РЕШТУ математики лишаємо СПРАВЖНЮ: `risk_reward` і
# `progress` мусять рахувати ті самі числа, що на проді, — інакше тест довів
# би поведінку заглушки, а не бота.
_REAL = _load('_real_trade_pilot', 'detection/trade_pilot.py')
_FAKE = types.ModuleType('detection.trade_pilot')
for _k in dir(_REAL):
    if not _k.startswith('__'):
        setattr(_FAKE, _k, getattr(_REAL, _k))
_PLAN = {'action': 'hold', 'reasons': ['тест']}
_FAKE.plan = lambda *a, **k: dict(_PLAN)
# ⚠️ Пастка, на яку вже наступали з `tickr_core`: `from detection import
# trade_pilot` бере АТРИБУТ пакета, якщо він виставлений, тож підміни лише
# sys.modules замало.
sys.modules['detection.trade_pilot'] = _FAKE
_pkg.trade_pilot = _FAKE

tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
TM = tmmod.TradeManager


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ── стенд ──────────────────────────────────────────────────────────────────
OBJ = {'price': 0.34220, 'label': 'головний пул', 'kind': 'liq_main',
       'dist_pct': 4.33}


def _tm(sl_result=None, no_ttl=True):
    o = TM.__new__(TM)
    o._settings = {'pilot_enabled': True, 'pilot_autofill_tp': False}
    o._pilot_at = {}
    o._pilot_state = {}
    o._pilot_context = lambda sym, side: {
        'swing': None, 'runway': None, 'poc': None,
        'swing_tf': '1h', 'poc_hours': 72}
    o.sltp_calls = []

    def _upd(symbol, manual_sl=None, is_shadow=False, **kw):
        o.sltp_calls.append((symbol, manual_sl, is_shadow))
        return dict(sl_result or {'ok': True})
    o.update_manual_sl_tp = _upd
    if no_ttl:
        o.PILOT_TTL = 0.0            # у більшості тестів троттл лише заважає
    return o


def _pos(side='LONG', entry=0.32510, sl=0.32050, src=None):
    p = {'side': side, 'entry_price': entry, 'manual_sl': sl}
    if src:
        p['manual_sl_src'] = src
    return p


def _texts():
    return ' || '.join(t for _k, t in _LOG)


# ═══════════ 1. СВІЙ СТАН НА КОЖНУ КНИГУ ═══════════════════════════════════
def test_real_and_paper_keep_separate_state():
    """ГОЛОВНИЙ ЗАМОК КЕЙСУ TRXUSDT. Дві позиції по одній монеті, обидві з
    ТІЄЮ САМОЮ ціллю, але різними входом і стопом. Кожна таблиця мусить
    бачити СВОЄ R, а не сусіднє."""
    global _PLAN
    _PLAN = {'action': 'hold', 'objective': OBJ, 'reasons': ['тримаємо']}
    o = _tm()
    # Paper — те, що на скріні. Real — угода зі стопом майже в беззбитку.
    o._pilot_tick('TRXUSDT', _pos(entry=0.32510, sl=0.32050), 0.32800, True)
    o._pilot_tick('TRXUSDT', _pos(entry=0.32800, sl=0.32785), 0.32800, False)

    paper = o.get_pilot_state('TRXUSDT', True)
    real = o.get_pilot_state('TRXUSDT', False)
    _check(paper and real, 'обидві книги мусять мати свій знімок')
    _check(abs(paper['r'] - 3.72) < 0.01,
           f"paper мав показати 3.72R (як рахується вручну), а показав {paper['r']}")
    _check(real['r'] > 90, f"real зі стопом у беззбитку дає велике R: {real['r']}")
    _check(paper['r'] != real['r'], 'книги НЕ мусять ділити одне число')
    _check(paper['is_shadow'] is True and real['is_shadow'] is False,
           'знімок має нести ознаку книги')
    print(f"✓ paper {paper['r']}R і real {real['r']}R більше не змішуються")


def test_throttle_is_per_book_not_per_symbol():
    """Спільний `_pilot_at` означав: монітор, який встиг першим, глушить пілот
    другої книги на цілий PILOT_TTL. Тобто одна з позицій лишалась без
    супроводу — і це НЕ видно в UI."""
    global _PLAN
    _PLAN = {'action': 'hold', 'objective': OBJ, 'reasons': ['x']}
    o = _tm(no_ttl=False)            # СПРАВЖНІЙ троттл 20с
    o._pilot_tick('TRXUSDT', _pos(), 0.328, False)
    o._pilot_tick('TRXUSDT', _pos(entry=0.4, sl=0.39), 0.328, True)
    _check(o.get_pilot_state('TRXUSDT', False), 'реальна книга відпрацювала')
    _check(o.get_pilot_state('TRXUSDT', True),
           'паперова книга НЕ мусить глушитись троттлом реальної')
    # А в межах ОДНІЄЇ книги троттл лишається робочим.
    before = o.get_pilot_state('TRXUSDT', False)['at']
    o._pilot_tick('TRXUSDT', _pos(), 0.329, False)
    _check(o.get_pilot_state('TRXUSDT', False)['at'] == before,
           'повторний тік тієї самої книги в межах TTL мусить пропускатись')
    print('✓ троттл роздільний по книгах, але всередині книги діє')


def test_keys_never_collide():
    _check(TM._pilot_key('TRXUSDT', True) != TM._pilot_key('TRXUSDT', False),
           'ключі книг мусять різнитись')
    _check(TM._pilot_key('trxusdt', True) == TM._pilot_key('TRXUSDT', True),
           'регістр символу не має створювати другий ключ')
    print('✓ ключ = символ + книга')


def test_closing_one_book_does_not_wipe_the_other():
    """Закриття реальної позиції стирало стан ПАПЕРОВОЇ (і навпаки) — колонка
    сусідньої угоди раптом ставала порожньою."""
    real_src = inspect.getsource(TM._close_position)
    shadow_src = inspect.getsource(TM._close_shadow)
    _check('_pilot_key(symbol, False)' in real_src,
           'реальне закриття мусить чистити САМЕ реальний ключ')
    _check('_pilot_key(symbol, True)' not in real_src,
           'реальне закриття НЕ сміє чіпати паперовий стан')
    _check('_pilot_key(symbol, True)' in shadow_src,
           'паперове закриття мусить чистити паперовий ключ')
    _check('_pilot_key(symbol, False)' not in shadow_src,
           'паперове закриття НЕ сміє чіпати реальний стан')
    # І перевіряємо це поведінково на самому словнику стану.
    o = _tm()
    o._pilot_state[TM._pilot_key('X', True)] = {'r': 1}
    o._pilot_state[TM._pilot_key('X', False)] = {'r': 2}
    o._pilot_state.pop(TM._pilot_key('X', False), None)
    _check(o.get_pilot_state('X', True) == {'r': 1}, 'паперовий стан вцілів')
    _check(o.get_pilot_state('X', False) is None, 'реальний прибрано')
    print('✓ закриття однієї книги не стирає стан іншої')


def test_api_merges_state_per_book():
    """Замок на бекенді: `/api/tm/state` мусить питати стан ІЗ ОЗНАКОЮ книги,
    інакше розділення в TM нічого не дасть."""
    src = open(os.path.join(_ROOT, 'web', 'flask_app.py')).read()
    _check('get_pilot_state(pos.get(\'symbol\'), _shadow)' in src,
           'ендпоінт мусить передавати книгу в get_pilot_state')
    _check('get_pilot_state(pos.get(\'symbol\'))' not in src,
           'виклику БЕЗ книги більше бути не повинно')
    print('✓ /api/tm/state підмішує знімок кожній книзі окремо')


# ═══════════ 2. ЛІЧИЛЬНИК ТРЕЙЛІВ = ФАКТ, А НЕ НАМІР ═══════════════════════
def test_counter_grows_only_when_the_stop_really_moves():
    global _PLAN
    _PLAN = {'action': 'trail', 'stop': 0.3210, 'objective': OBJ,
             'reasons': ['структура']}
    o = _tm()
    for _ in range(3):
        o._pilot_tick('TRXUSDT', _pos(), 0.32800, False)
    st = o.get_pilot_state('TRXUSDT', False)
    _check(st['trails'] == 3, f"три успішні трейли → 3, а не {st['trails']}")
    _check(not st['trail_block'], f"блокування не було: {st['trail_block']!r}")
    _check(st['last_stop'] == 0.321, f"останній стоп: {st['last_stop']}")
    _check(len(o.sltp_calls) == 3, 'стоп мали переставити тричі')
    print('✓ успішний трейл рахується')


def test_operator_lock_does_not_inflate_the_counter():
    """🐞 САМЕ ЦЕ ДАЛО «🛡432». SL веде оператор → трейл не застосовується
    ЖОДНОГО разу, тож лічильник мусить лишитись НУЛЕМ, хай план хоч щотакту
    проситься трейлити."""
    global _PLAN
    _PLAN = {'action': 'trail', 'stop': 0.3210, 'objective': OBJ,
             'reasons': ['структура']}
    _LOG.clear()
    o = _tm()
    pos = _pos(src=TM.SRC_USER)
    for _ in range(20):
        o._pilot_tick('TRXUSDT', pos, 0.32800, False)
    st = o.get_pilot_state('TRXUSDT', False)
    _check(st['trails'] == 0,
           f"стоп не рухався жодного разу, а лічильник = {st['trails']}")
    _check(st['trail_block'], 'стан мусить пояснювати, ЧОМУ трейлу не було')
    _check('оператор' in st['trail_block'], st['trail_block'])
    _check(o.sltp_calls == [], 'ручний рівень не сміє переписуватись')
    # Лог лишається чистим — попередження пишеться ОДИН раз, а не 20.
    _check(_texts().count('веде ОПЕРАТОР') == 1,
           f'замок мав написати рівно один рядок: {_texts().count("веде ОПЕРАТОР")}')
    print('✓ операторський замок: 20 тіків → 0 трейлів, 1 рядок у лозі')


def test_rejected_level_does_not_count_either():
    """Trade Manager має власну валідацію боку рівня. Відмова — це теж
    «стоп не зрушив», і рахувати її як трейл не можна."""
    global _PLAN
    _PLAN = {'action': 'trail', 'stop': 0.3210, 'objective': OBJ,
             'reasons': ['структура']}
    o = _tm(sl_result={'ok': False, 'reason': 'SL нижче ціни для SHORT'})
    for _ in range(4):
        o._pilot_tick('TRXUSDT', _pos(), 0.32800, False)
    st = o.get_pilot_state('TRXUSDT', False)
    _check(st['trails'] == 0, f"відхилені рівні не рахуються: {st['trails']}")
    _check('не прийнято' in (st['trail_block'] or ''), st['trail_block'])
    print('✓ відхилений Trade Manager рівень не рахується трейлом')


def test_block_flag_clears_when_trail_goes_through():
    """Замок зняли (оператор очистив поле) → позначка мусить зникнути з
    НАСТУПНИМ тіком, інакше в таблиці назавжди лишиться 🔒."""
    global _PLAN
    _PLAN = {'action': 'trail', 'stop': 0.3210, 'objective': OBJ, 'reasons': ['x']}
    o = _tm()
    o._pilot_tick('TRXUSDT', _pos(src=TM.SRC_USER), 0.328, False)
    _check(o.get_pilot_state('TRXUSDT', False)['trail_block'], 'спершу заблоковано')
    o._pilot_tick('TRXUSDT', _pos(), 0.328, False)
    st = o.get_pilot_state('TRXUSDT', False)
    _check(not st['trail_block'], f"позначка мала зникнути: {st['trail_block']!r}")
    _check(st['trails'] == 1, f"а трейл — зарахуватись: {st['trails']}")
    print('✓ позначка блокування знімається, щойно трейл проходить')


def test_hold_never_touches_the_counter():
    global _PLAN
    _PLAN = {'action': 'hold', 'objective': OBJ, 'reasons': ['тримаємо']}
    o = _tm()
    o._pilot_state[TM._pilot_key('TRXUSDT', False)] = {'trails': 7}
    o._pilot_tick('TRXUSDT', _pos(), 0.328, False)
    st = o.get_pilot_state('TRXUSDT', False)
    _check(st['trails'] == 7, f"«тримаємо» не міняє лічильник: {st['trails']}")
    print('✓ «тримаємо» лічильник не чіпає')


# ═══════════ 3. UI: чесна іконка + єдиний формат чисел ═════════════════════
def test_ui_shows_blocked_trail_separately():
    html = open(os.path.join(_ROOT, 'templates', 'smart_money.html')).read()
    _check('_PILOT_BLOCKED' in html, 'потрібен окремий значок заблокованого трейлу')
    _check('pl.trail_block' in html, 'фронт мусить читати причину блокування')
    i = html.index('const [ic, lbl, col]')
    _check('trail_block' in html[i - 200:i + 200],
           'значок дії мусить враховувати блокування, а не показувати 🛡')
    print('✓ UI розрізняє «стоп підтягнуто» і «трейл заблоковано»')


def test_level_fields_keep_trailing_zeros():
    """Одне число — одне написання. Поле TP-2 показувало «0,3422», а колонка
    автопілота — «$0.34220»."""
    html = open(os.path.join(_ROOT, 'templates', 'smart_money.html')).read()
    i = html.index('function roundSlTp')
    body = html[i:i + 700]
    _check('String(Number(' not in body,
           'обгортка Number() обрізала кінцеві нулі — саме через неї формат рвався')
    _check('n.toFixed(_sltpDp(n))' in body, 'точність мусить бути фіксована')
    # І поля не мусять залежати від локалі браузера (кома замість крапки).
    j = html.index('function manualSlTpCellsHTML')
    cells = html[j:j + 2600]
    _check(cells.count('lang="en"') == 3,
           'усі три поля рівнів мусять мати lang="en" — інакше браузер малює кому')
    print('✓ поля рівнів: фіксована точність + крапка незалежно від локалі')


if __name__ == '__main__':
    test_real_and_paper_keep_separate_state()
    test_throttle_is_per_book_not_per_symbol()
    test_keys_never_collide()
    test_closing_one_book_does_not_wipe_the_other()
    test_api_merges_state_per_book()
    test_counter_grows_only_when_the_stop_really_moves()
    test_operator_lock_does_not_inflate_the_counter()
    test_rejected_level_does_not_count_either()
    test_block_flag_clears_when_trail_goes_through()
    test_hold_never_touches_the_counter()
    test_ui_shows_blocked_trail_separately()
    test_level_fields_keep_trailing_zeros()
    print('\nУсі тести автопілота (книги + лічильник трейлів) пройдено ✅')
