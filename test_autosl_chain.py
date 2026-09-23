"""Тест: 🛡 авто-SL з OB МУСИТЬ поставити стоп, а не «чекати».

Кейс MNTUSDT (23:28, paper SHORT @ 0.51360): у 🧾 Лозі —
    «Авто-SL з OB: OB на 15M протилежний (BULLISH) — чекаю BEARISH»
і поле Manual SL лишилось ПОРОЖНЄ. При цьому на графіку висів ★1H-OB, ВЕДМЕЖИЙ
(бейдж «🔒 OB Short 1H ★»), тобто ідеальний якір для стопа шорта — але старий код
дивився РІВНО ОДНЕ джерело (`q2_auto_ob_sl_tf`, 15m) і при невдачі просто виходив.
Угода лишалась БЕЗ стопа на невизначений час.

Тепер джерела пробуються ЛАНЦЮГОМ, і останній крок — гарантія:
    1) OB на `q2_auto_ob_sl_tf`      2) ★ OB на `ob_filter_timeframe`
    3) Volumized OB у бік угоди      4) % від входу
Кожен кандидат ще й перевіряється на БЕЗПЕЧНИЙ бік від поточної ціни.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules['pybit.unified_trading'].HTTP = object

_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_load('detection.signal_labels', 'detection/signal_labels.py')
_load('detection.setup_grader', 'detection/setup_grader.py')
ffmod = _load('detection.fuel_filter', 'detection/fuel_filter.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _near(got, want):
    """Порівняння з поправкою на `_round_sltp_value` (округлює рівень до
    «чистого» значення без хвоста float) — тому строга рівність тут не годиться."""
    try:
        return abs(float(got) - float(want)) <= max(1e-9, abs(float(want)) * 1e-4)
    except (TypeError, ValueError):
        return False


# ── стаби середовища, які читає `_auto_ob_manual_sl` ────────────────────────
_LOG = []
_OB_ROWS = {}         # tf → {bias, bar_high, bar_low} | None
_VOB = {}             # 'bullish_obs' / 'bearish_obs' → [ob]  (на будь-якому TF)
_VOB_TF = {}          # tf → такий самий dict, коли TF мусить РОЗРІЗНЯТИСЬ
_ASKED = []           # TF, за якими справді ходили по бари (порядок сходів)
_SETTINGS = {}


def _install_stubs():
    lg = types.ModuleType('detection.activity_log')
    lg.log_activity = lambda sym, kind, text, **kw: _LOG.append((kind, text))
    sys.modules['detection.activity_log'] = lg

    ff = types.ModuleType('detection.fuel_filter')
    ff.get_fuel_filter = lambda: types.SimpleNamespace(get_settings=lambda: dict(_SETTINGS))
    # ⚠️ Стаб мусить віддавати ТЕ САМЕ, що справжній модуль: TM імпортує звідси
    # сходи фолбеку SL. Без цього виклик падав би в `except` і 5m-крок не
    # перевірявся б узагалі (пастка стаба, на яку в проєкті вже наступали).
    ff.SL_FALLBACK_TFS = ('15m', '5m')
    sys.modules['detection.fuel_filter'] = ff

    db = types.ModuleType('storage.db_operations')
    db.get_db = lambda: types.SimpleNamespace(
        get_smc_ob_state=lambda sym, tf: _OB_ROWS.get(tf))
    st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
    sys.modules.setdefault('storage', st)
    sys.modules['storage.db_operations'] = db

    sc = types.ModuleType('detection.smc_scanner')
    sc.get_smc_scanner = lambda: types.SimpleNamespace(get_settings=lambda: {
        'ob_filter_timeframe': '1h', 'volumized_timeframe': '5m'})
    sys.modules['detection.smc_scanner'] = sc

    def _klines(s, limit=200, interval='5m'):
        # Запамʼятовуємо, за ЯКИЙ TF питали — інакше перевірити сходи
        # 15m→5m неможливо: детектор отримує лише бари.
        _ASKED.append(interval)
        return [{'o': 1}] * 50

    md = types.ModuleType('detection.market_data')
    md.get_market_data = lambda: types.SimpleNamespace(fetch_klines=_klines)
    sys.modules['detection.market_data'] = md

    vo = types.ModuleType('detection.volumized_ob')
    # `_VOB_TF[tf]` — блок САМЕ цього TF; `_VOB` — «однаково на будь-якому»
    # (так поводились усі наявні тести до появи сходів).
    vo.detect_volumized_obs = lambda *a, **k: dict(
        _VOB_TF.get(_ASKED[-1] if _ASKED else '', _VOB))
    sys.modules['detection.volumized_ob'] = vo


_install_stubs()
tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
TM = tmmod.TradeManager


def _tm():
    o = TM.__new__(TM)
    return o


def _reset(**over):
    _LOG.clear(); _OB_ROWS.clear(); _VOB.clear(); _VOB_TF.clear(); _ASKED.clear()
    _SETTINGS.clear()
    _SETTINGS.update({'q2_auto_ob_sl': True, 'q2_auto_ob_sl_buffer_pct': 0.2,
                      'q2_auto_ob_sl_tf': '15m', 'autosl_fallback_on': True,
                      'autosl_fallback_pct': 2.0, 'autosl_max_pct': 0.0})
    _SETTINGS.update(over)


def _pos(side='SHORT', entry=0.51360):
    return {'side': side, 'entry_price': entry}


def _text():
    return ' || '.join(t for _k, t in _LOG)


# ═══════════════════════════════ ТЕСТИ ══════════════════════════════════════
def test_mnt_case_star_1h_block_is_used_instead_of_waiting():
    """MNTUSDT: 15m-OB БИЧАЧИЙ при SHORT. Раніше — «чекаю BEARISH» і жодного
    стопа. Ланцюг мусить ПРОВАЛИТИСЬ далі й узяти ★1H-OB (ВЕДМЕЖИЙ).

    ⚠️ Обране джерело («🛑 SL з») тепер ПЕРШЕ для будь-якої угоди, тож щоб
    перевірити САМЕ прохід крізь непридатний 15m-OB, ставимо джерелом
    «15m Volumized» і не даємо жодного Volumized-блоку: ланцюг іде
    Volumized(порожньо) → 15m OB(протилежний) → ★1H OB(придатний).
    """
    _reset(queue4_sl_source='15m')
    _OB_ROWS['15m'] = {'bias': 'BULLISH', 'bar_high': 0.5150, 'bar_low': 0.5100}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(p.get('manual_sl'), f'СТОП МАВ БУТИ ПОСТАВЛЕНИЙ, лог: {_text()}')
    _check(_near(p['manual_sl'], 0.5285 * 1.002),
           f'SL мав стати над верхом ★1H-блоку, отримано {p["manual_sl"]}')
    _check('★1H' in _text(), f'у лозі має бути видно джерело: {_text()}')
    _check('блок протилежний (BULLISH)' in _text(),
           f'причина пропуску 15m має лишитись у лозі: {_text()}')
    print('✓ MNTUSDT: замість «чекаю BEARISH» узято ★1H-OB → SL поставлено')


def test_chosen_source_wins_for_every_trade():
    """🛑 «SL з» — ГЛОБАЛЬНЕ джерело (виправлено 08.09).

    РАНІШЕ вибір застосовувався ЛИШЕ до угод Черги-4 (`'Q4' in opened_by`), а
    на прямому відкритті мовчки бралося «OB TF» (деф. 15m) — те саме поле
    давало РІЗНИЙ стоп залежно від того, хто відкрив угоду. Тут позиція БЕЗ
    `opened_by`, тобто найзвичайніше пряме відкриття.
    """
    _reset()                       # queue4_sl_source за замовчуванням = '1h'
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    p = _pos()
    _check('opened_by' not in p, 'позиція навмисно БЕЗ походження — прямий шлях')
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5285 * 1.002),
           f'мав узятись ★1H (обране джерело), отримано {p["manual_sl"]}')
    _check('обране джерело' in _text(), f'у лозі має бути видно, що це вибір: {_text()}')
    print('✓ обране «🛑 SL з» діє на КОЖНУ угоду, не лише на Q4')


def test_ob_tf_is_the_fallback_when_chosen_source_has_nothing():
    """«OB TF» (`q2_auto_ob_sl_tf`) не зник — він став ФОЛБЕКОМ: спрацьовує,
    коли обране джерело нічого не дало."""
    _reset()                       # обране = '1h', але ★1H-рядка немає
    _OB_ROWS['1h'] = None
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5200 * 1.002),
           f'мав узятись фолбек 15m, отримано {p.get("manual_sl")}')
    print('✓ «OB TF» лишився робочим фолбеком')


def test_volumized_used_when_both_ob_rows_unusable():
    """⚠️ ЗМІНА КОНТРАКТУ (22.09, вимога користувача): «Якщо увімкнено "SL з" —
    1Н OB і немає можливості його отримати, то шукаємо на 15хв або на 5хв».
    Раніше тут очікувався сканерний TF (5m) — тепер після 1H першим іде 15m, і
    саме він мусить стояти в лозі. Тест ПЕРЕПИСАНО, а не «полагоджено»."""
    _reset()
    _OB_ROWS['15m'] = {'bias': 'BULLISH', 'bar_high': 0.5150, 'bar_low': 0.5100}
    _OB_ROWS['1h'] = None
    _VOB['bearish_obs'] = [{'top': 0.5250, 'bottom': 0.5200, 'breaker': False}]
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5250 * 1.002),
           f'мав узятись Volumized OB, отримано {p.get("manual_sl")}')
    _check('Volumized OB 15m' in _text(), f'джерело в лозі: {_text()}')
    print('✓ жодного придатного OB-рядка → Volumized OB у бік угоди')


# ═════ 🛑 СХОДИ 1H → 15m → 5m (вимога 22.09) ═══════════════════════════════
def test_chosen_1h_missing_walks_down_to_15m_then_5m():
    """ДОСЛІВНА вимога: немає 1H-блоку → шукаємо на 15хв, потім на 5хв.
    Раніше 5m-кроку не було взагалі: коли 15m нічого не давав, стоп тримався
    лише на гарантії «% від входу»."""
    _reset(queue4_sl_source='1h')
    _OB_ROWS['1h'] = None
    _OB_ROWS['15m'] = None
    _VOB_TF['15m'] = {}                                   # на 15m блоку немає
    _VOB_TF['5m'] = {'bearish_obs': [{'top': 0.5240, 'bottom': 0.5210,
                                      'breaker': False}]}
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5240 * 1.002),
           f'мав спуститись на 5m, отримано {p.get("manual_sl")}')
    _check('Volumized OB 5m' in _text(), f'у лозі має бути саме 5m: {_text()}')
    _check(_ASKED.index('15m') < _ASKED.index('5m'),
           f'15m мусить питатись ПЕРШИМ (тісніший стоп не має обганяти): {_ASKED}')
    print('✓ 1H немає → 15m → 5m, порядок спадний')


def test_ladder_does_not_ask_the_same_timeframe_twice():
    """Сканерний `volumized_timeframe` часто дорівнює одному зі сходів (у
    користувача 5m). Без дедупу той самий запит ішов би двічі, а в лог падала б
    дубльована причина пропуску."""
    _reset(queue4_sl_source='1h')
    _OB_ROWS['1h'] = None
    _OB_ROWS['15m'] = None
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(len(_ASKED) == len(set(_ASKED)),
           f'кожен TF мав питатись один раз, отримано {_ASKED}')
    print('✓ жоден таймфрейм не питається двічі')


def test_guarantee_still_last_when_no_timeframe_has_a_block():
    """Сходи НЕ скасовують «🛡 Стоп ЗАВЖДИ»: не дав жоден TF — лишається
    відсоток від входу, і угода не висить без стопа."""
    _reset(queue4_sl_source='1h', autosl_fallback_pct=2.0)
    _OB_ROWS['1h'] = None
    _OB_ROWS['15m'] = None
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.51360 * 1.02),
           f'мала спрацювати гарантія 2% від входу, отримано {p.get("manual_sl")}')
    print('✓ жоден TF не дав блоку → гарантія «% від входу»')


def test_volumized_skips_breaker():
    _reset()
    _OB_ROWS['15m'] = None
    _VOB['bearish_obs'] = [{'top': 0.5250, 'bottom': 0.5200, 'breaker': True},
                           {'top': 0.5300, 'bottom': 0.5280, 'breaker': False}]
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5300 * 1.002),
           f'breaker-зона знецінена і не може бути стопом, отримано {p.get("manual_sl")}')
    print('✓ breaker-блок пропускається')


def test_percent_fallback_guarantees_a_stop():
    """🛡 ГОЛОВНЕ: не знайшлось ЖОДНОГО блоку → стоп усе одно є."""
    _reset(autosl_fallback_pct=2.0)
    p = _pos(entry=0.51360)
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(p.get('manual_sl'), f'угода НЕ має лишатись без стопа: {_text()}')
    _check(_near(p['manual_sl'], 0.51360 * 1.02),
           f'SHORT → 2% над входом, отримано {p["manual_sl"]}')
    _check('% від входу' in _text(), f'джерело в лозі: {_text()}')
    print('✓ немає жодного блоку → гарантований відсотковий стоп')


def test_long_fallback_is_below_entry():
    _reset(autosl_fallback_pct=1.5)
    p = _pos(side='LONG', entry=100.0)
    _tm()._auto_ob_manual_sl('BTCUSDT', p, 101.0)
    _check(_near(p['manual_sl'], 98.5),
           f'LONG → 1.5% ПІД входом, отримано {p.get("manual_sl")}')
    print('✓ LONG-фолбек ставиться під входом')


def test_fallback_can_be_turned_off():
    """Вимкнений фолбек = стара поведінка «краще без стопа» (свідомий вибір)."""
    _reset(autosl_fallback_on=False)
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(not p.get('manual_sl'), 'фолбек вимкнено → стопа немає')
    _check('фолбек вимкнено' in _text(), f'причина має бути в лозі: {_text()}')
    print('✓ фолбек вимикається (повертає стару поведінку)')


def test_wrong_side_level_is_skipped_not_applied():
    """Рівень з неправильного боку закрив би угоду наступним тіком монітора."""
    _reset()
    # Ведмежий ★1H-блок (ОБРАНЕ джерело, іде першим), але ціна вже ВИЩЕ його
    # верху → стоп опинився б НИЖЧЕ ціни. Придатний рівень дає фолбек 15m.
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5100, 'bar_low': 0.5050}
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5285 * 1.002),
           f'мав перейти на фолбек 15m, отримано {p.get("manual_sl")}')
    _check('неправильного боку' in _text(), f'причина в лозі: {_text()}')
    print('✓ рівень із неправильного боку пропускається, а не ставиться')


def test_max_pct_clamps_a_far_block():
    """Стеля робить ризик угод порівнянним (кейс WIF 15.56% проти ALGO 1.40%)."""
    _reset(autosl_max_pct=3.0)
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.6000, 'bar_low': 0.5900}
    p = _pos(entry=0.51360)
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.51360 * 1.03),
           f'далекий блок мав підтягнутись до 3%, отримано {p.get("manual_sl")}')
    _check('підтягнуто до стелі' in _text(), f'це має бути видно в лозі: {_text()}')
    print('✓ стеля підтягує надто далекий рівень')


def test_max_pct_zero_changes_nothing():
    _reset(autosl_max_pct=0.0)
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.6000, 'bar_low': 0.5900}
    p = _pos(entry=0.51360)
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.6000 * 1.002),
           f'стеля=0 → рівень блоку без змін, отримано {p.get("manual_sl")}')
    _check('підтягнуто' not in _text(), 'без стелі нічого не підтягуємо')
    print('✓ стеля 0 = вимкнено (нічого не змінює)')


def test_set_once_is_preserved():
    """Стоп ставиться ОДИН раз — повторні тіки його не рухають."""
    _reset()
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    p = _pos()
    t = _tm()
    t._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    first = p['manual_sl']
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5180, 'bar_low': 0.5150}
    t._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(p['manual_sl'] == first, 'рівень не має рухатись після встановлення')
    print('✓ «один раз» збережено')


def test_user_typed_sl_is_never_touched():
    _reset()
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    p = _pos(); p['manual_sl'] = 0.5555      # рівень руками, без _auto_ob_sl_val
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(p['manual_sl'] == 0.5555, 'ручний рівень користувача чіпати не можна')
    print('✓ SL, введений руками, лишається недоторканим')


def test_disabled_feature_does_nothing():
    _reset(q2_auto_ob_sl=False)
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(not p.get('manual_sl') and not _LOG, 'тумблер OFF → жодних дій')
    print('✓ вимкнений авто-SL нічого не робить')


def test_defaults_guarantee_a_stop():
    _check(ffmod.DEFAULT_SETTINGS.get('autosl_fallback_on') is True,
           'гарантія стопа має бути УВІМКНЕНА за замовчуванням')
    _check(ffmod.DEFAULT_SETTINGS.get('autosl_fallback_pct') == 2.0, 'дефолт 2%')
    _check(ffmod.DEFAULT_SETTINGS.get('autosl_max_pct') == 0.0,
           'стеля за замовчуванням ВИМКНЕНА (не міняємо поведінку без запиту)')
    print('✓ дефолти: гарантія ON (2%), стеля OFF')


# ═══════════ 🏷 ПОХОДЖЕННЯ Manual SL/TP (авто ↔ вручну) ════════════════════
def test_autosl_marks_level_as_bot_origin():
    """Рівень, який поставив авто-SL, має нести позначку 'auto' — інакше UI не
    зможе пофарбувати поле, а лог не відрізнить бота від оператора."""
    _reset()
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(p.get('manual_sl_src') == TM.SRC_AUTO,
           f"очікували позначку '{TM.SRC_AUTO}', отримано {p.get('manual_sl_src')!r}")
    _check('Авто-SL' in (p.get('manual_sl_by') or ''),
           f'підпис джерела має бути людським: {p.get("manual_sl_by")!r}')
    print('✓ авто-SL позначає рівень як поставлений ботом')


class _StoreTM(TM):
    """Мінімальний TM для перевірки `update_manual_sl_tp` без біржі й БД."""
    def __init__(self, side='SHORT', price=100.0):
        import threading
        self._lock = threading.RLock()
        self._positions = {'BTCUSDT': {'symbol': 'BTCUSDT', 'side': side,
                                       'entry_price': 100.0}}
        self._shadow_positions = {}
        self._price = price
    def _get_current_price(self, symbol): return self._price
    def _persist_positions(self): pass
    def _persist_shadow_positions(self): pass


def test_user_edit_overrides_bot_origin():
    """Оператор вписав своє значення → позначка стає 'user', колір поля в UI
    міняється з блакитного на бурштиновий."""
    _LOG.clear()
    t = _StoreTM('SHORT', 100.0)
    t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0,
                          origin='auto', origin_label='Черга-4 · 1H OB')
    _check(t._positions['BTCUSDT']['manual_sl_src'] == TM.SRC_AUTO, 'спершу — бот')
    _check('🤖 Бот' in _text() and 'Черга-4 · 1H OB' in _text(),
           f'лог має назвати бота і джерело: {_text()}')
    _LOG.clear()
    t.update_manual_sl_tp('BTCUSDT', manual_sl=105.0)      # дефолт origin='user'
    _check(t._positions['BTCUSDT']['manual_sl_src'] == TM.SRC_USER,
           'після ручної правки позначка мусить стати user')
    _check('✏️ Користувач' in _text(), f'лог має назвати користувача: {_text()}')
    print('✓ ручна правка перекриває позначку бота (і це видно в лозі)')


def test_clearing_removes_origin_mark():
    t = _StoreTM('SHORT', 100.0)
    t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0, origin='auto')
    t.update_manual_sl_tp('BTCUSDT', manual_sl=0)          # 0 = зняти
    p = t._positions['BTCUSDT']
    _check('manual_sl' not in p and 'manual_sl_src' not in p,
           'знятий рівень не має лишати за собою позначку походження')
    print('✓ зняття рівня прибирає й позначку')


def test_tp_origin_tracked_separately():
    t = _StoreTM('SHORT', 100.0)
    t.update_manual_sl_tp('BTCUSDT', manual_sl=110.0, origin='auto', origin_label='бот')
    t.update_manual_sl_tp('BTCUSDT', manual_tp=90.0)       # TP — руками
    p = t._positions['BTCUSDT']
    _check(p['manual_sl_src'] == TM.SRC_AUTO and p['manual_tp_src'] == TM.SRC_USER,
           f"SL і TP мають мати НЕЗАЛЕЖНі позначки: {p.get('manual_sl_src')} / {p.get('manual_tp_src')}")
    print('✓ SL і TP мають незалежні позначки походження')


def test_rejected_level_leaves_no_mark():
    """Відхилений рівень нічого не мутує — позначки теж не має з'явитись."""
    t = _StoreTM('SHORT', 100.0)
    r = t.update_manual_sl_tp('BTCUSDT', manual_sl=90.0, origin='auto')   # нижче ціни
    _check(not r.get('ok'), 'SHORT зі стопом нижче ціни має бути відхилений')
    _check('manual_sl_src' not in t._positions['BTCUSDT'],
           'відхилений рівень не має лишати позначку')
    print('✓ відхилений рівень не лишає позначки')


# ═════ 🎯 ДЖЕРЕЛО SL = НАЛАШТУВАННЯ «🛑 SL з» (ГЛОБАЛЬНО, не лише Черга-4) ═
def test_q4_trade_uses_the_configured_1h_source_first():
    """🐞 Скарга: у Черзі-4 стоїть «SL з 1H OB», а в лозі — «SL з OB 15M».
    Причина: ДВА незалежні авто-SL зі СВОЇМИ таймфреймами; TM нічого не знав про
    вибір у Черзі-4 і брав власний `q2_auto_ob_sl_tf` (деф. 15m).
    Тепер для угод, відкритих Чергою-4, першим іде САМЕ обране джерело."""
    _reset(queue4_sl_source='1h')
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    p = _pos()
    p['opened_by'] = 'vob_alert → Q4'
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5285 * 1.002),
           f'мав узятись 1H-OB (як обрано в Черзі-4), отримано {p.get("manual_sl")}')
    _check('обране джерело: 1H OB' in _text(),
           f'джерело має бути назване в лозі: {_text()}')
    print('✓ угода з Черги-4 бере САМЕ обране джерело (1H OB), а не 15m')


def test_q4_trade_with_15m_choice_uses_volumized_15m():
    _reset(queue4_sl_source='15m')
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    _VOB['bearish_obs'] = [{'top': 0.5240, 'bottom': 0.5210, 'breaker': False}]
    p = _pos()
    p['opened_by'] = 'vob_alert → Q4'
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5240 * 1.002),
           f'мав узятись Volumized OB 15m, отримано {p.get("manual_sl")}')
    _check('Volumized OB 15m' in _text(), f'саме 15m, не сканерний TF: {_text()}')
    print('✓ вибір «15m Volumized OB» бере рівно 15m')


def test_every_trade_now_uses_the_chosen_source():
    """⚠️ ЗМІНА КОНТРАКТУ (08.09). Раніше цей тест звався
    `test_non_q4_trade_keeps_its_own_tf` і фіксував, що угоди НЕ з Черги-4
    йдуть за власним `q2_auto_ob_sl_tf`. Саме це й було дефектом: те саме
    поле «🛑 SL з» давало РІЗНИЙ стоп залежно від того, хто відкрив угоду, а
    при вимкнених чергах не діяло взагалі. Тепер обране джерело — глобальне.
    ⚠️ Стеля ризику від цього ЗМІНИЛАСЬ: не-Q4 угоди беруть 1H-блок замість
    15m, тобто стоп зазвичай ДАЛІ. Регулюється «🎚 Стеля SL» або вибором
    «15m Volumized OB»."""
    _reset(queue4_sl_source='1h', q2_auto_ob_sl_tf='15m')
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    for _origin in ('choch → Q2', 'manual', None):
        _reset(queue4_sl_source='1h', q2_auto_ob_sl_tf='15m')
        _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
        _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
        p = _pos()
        if _origin:
            p['opened_by'] = _origin
        _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
        _check(_near(p['manual_sl'], 0.5285 * 1.002),
               f'{_origin!r}: мав узятись ★1H (обране), отримано {p.get("manual_sl")}')
    print('✓ обране джерело діє на угоди БУДЬ-ЯКОГО походження')


def test_q4_source_unavailable_falls_back_and_says_so():
    _reset(queue4_sl_source='1h')
    _OB_ROWS['1h'] = None                     # обраного джерела немає
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    p = _pos()
    p['opened_by'] = 'vob_alert → Q4'
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5200 * 1.002), 'фолбек на 15m спрацював')
    _check('немає готового OB' in _text(),
           f'причина, чому обране джерело не спрацювало, має бути в лозі: {_text()}')
    print('✓ обране джерело недоступне → фолбек, і в лозі видно чому')


# ═════════ ⚖️ БЕЗЗБИТОК ПІСЛЯ TP-1 ═════════════════════════════════════════
def _tm_be(cur_sl=None, accept=True, buf=0.12):
    """TM з підміненим `update_manual_sl_tp` — перевіряємо САМЕ рішення."""
    o = _tm()
    o._settings = {'be_commission_buffer_pct': buf, 'tp1_move_to_be': True}
    o.calls = []

    def _upd(sym, manual_sl=None, is_shadow=False, origin='user',
             origin_label=None, **kw):
        o.calls.append((manual_sl, origin, origin_label))
        return ({'ok': True} if accept
                else {'ok': False, 'reason': 'wrong side', 'validation': True})
    o.update_manual_sl_tp = _upd
    pos = {'side': 'LONG', 'entry_price': 100.0}
    if cur_sl:
        pos['manual_sl'] = cur_sl
    return o, pos


def test_breakeven_level_covers_round_trip_fees():
    """Стоп РІВНО на вході — ще не беззбиток: комісії зроблять із нього мінус."""
    _check(_near(tmmod.breakeven_with_fees('LONG', 100.0, 0.12), 100.12),
           'LONG → трохи ВИЩЕ входу')
    _check(_near(tmmod.breakeven_with_fees('SHORT', 100.0, 0.12), 99.88),
           'SHORT → трохи НИЖЧЕ входу (дзеркально)')
    _check(_near(tmmod.breakeven_with_fees('LONG', 100.0, 0), 100.0),
           'буфер 0 = рівно вхід (стара поведінка)')
    for bad in (('LONG', 0, 0.12), ('LONG', None, 0.12), ('X', 100.0, 0.12),
                ('LONG', 'abc', 0.12)):
        _check(tmmod.breakeven_with_fees(*bad) is None, f'сміття {bad} → None')
    print('✓ рівень беззбитку = вхід + запас на комісії (обидва боки)')


def test_tp1_moves_stop_to_breakeven():
    o, pos = _tm_be()
    _LOG.clear()
    o._tp1_move_to_breakeven('BTCUSDT', pos, 103.0, False)
    _check(len(o.calls) == 1 and _near(o.calls[0][0], 100.12),
           f'SL мав переїхати в беззбиток: {o.calls}')
    _check(o.calls[0][1] == 'auto' and 'Беззбиток після TP-1' in (o.calls[0][2] or ''),
           f'рівень має бути позначений як БОТІВ і підписаний: {o.calls}')
    _check(pos.get('sl_breakeven') is True,
           'позиція має нести позначку — інакше поле не позеленіє')
    _check('БЕЗЗБИТОК' in _text(), f'подія мусить бути в лозі: {_text()}')
    print('✓ TP-1 → SL у беззбиток, позначка й запис у лозі є')


def test_breakeven_never_loosens_a_better_stop():
    """🔒 Головне правило: «захист» не має відсувати стоп НАЗАД. Якщо автопілот
    уже підтягнув стоп вище за беззбиток — лишаємо кращий рівень."""
    o, pos = _tm_be(cur_sl=101.5)          # уже краще за 100.12
    _LOG.clear()
    o._tp1_move_to_breakeven('BTCUSDT', pos, 103.0, False)
    _check(o.calls == [], f'кращий стоп чіпати не можна: {o.calls}')
    _check(pos.get('manual_sl') == 101.5, 'рівень лишився недоторканим')
    _check('уже кращий' in _text(), f'причина має бути в лозі: {_text()}')
    print('✓ беззбиток НЕ послаблює вже кращий стоп')


def test_breakeven_improves_a_worse_stop():
    o, pos = _tm_be(cur_sl=97.0)           # гірший за беззбиток
    o._tp1_move_to_breakeven('BTCUSDT', pos, 103.0, False)
    _check(len(o.calls) == 1 and _near(o.calls[0][0], 100.12),
           f'гірший стоп мусить підтягнутись: {o.calls}')
    print('✓ гірший стоп підтягується до беззбитку')


def test_rejected_breakeven_is_reported_not_faked():
    """Ціна встигла повернутись до входу → TM відхилить рівень. Лог мусить
    сказати правду (у проєкті вже був дефект «лог каже, що поставив»)."""
    o, pos = _tm_be(accept=False)
    _LOG.clear()
    o._tp1_move_to_breakeven('BTCUSDT', pos, 100.05, False)
    _check(pos.get('sl_breakeven') is not True,
           'відхилений рівень НЕ має лишати зелену позначку')
    _check('НЕ прийнято' in _text(), f'відмову треба показати: {_text()}')
    print('✓ відхилений беззбиток не вдає, що спрацював')


def test_every_frontend_position_route_exists():
    """🐞 КЛАС ПОМИЛКИ, який коштував мовчазної втрати рівня: `submitManualTp1`
    стукав у `/api/tm/positions/manual-sltp`, а маршрут зареєстровано як
    `manual-sl-tp`. 404 глушився в catch → поле «зберігалось» і зникало на
    наступному поллі. Тепер КОЖЕН фронтовий fetch звіряється з реальними
    маршрутами Flask — здогадуватись про URL більше не можна."""
    import re as _re
    html = open(os.path.join(_ROOT, 'templates/smart_money.html')).read()
    flask = open(os.path.join(_ROOT, 'web/flask_app.py')).read()
    used = set(_re.findall(r"['\"`](/api/tm/positions/[a-z0-9_-]+)", html))
    known = set(_re.findall(r"@app\.route\('(/api/tm/positions/[a-z0-9_-]+)'", flask))
    missing = used - known
    _check(not missing, f'фронт кличе неіснуючі маршрути: {sorted(missing)} '
                        f'(зареєстровані: {sorted(known)})')
    print(f'✓ усі {len(used)} фронтових маршрути позицій існують у Flask')


def test_breakeven_is_off_by_default():
    """Рішення про ризик — за користувачем. Дефолт: TP-1 стоп НЕ рухає."""
    _check(tmmod.DEFAULT_SETTINGS.get('tp1_move_to_be') is False,
           'переведення в беззбиток має бути ВИМКНЕНЕ за замовчуванням')
    o, pos = _tm_be()
    o._settings['tp1_move_to_be'] = False
    _LOG.clear()
    o._tp1_move_to_breakeven('BTCUSDT', pos, 103.0, False)
    _check(o.calls == [], f'тумблер OFF → стоп не чіпаємо взагалі: {o.calls}')
    _check(pos.get('sl_breakeven') is not True, 'і позначки не ставимо')
    _check(_text() == '', f'і в лог нічого не пишемо: {_text()}')
    print('✓ дефолт OFF: TP-1 не переводить стоп у беззбиток')


def test_tp1_calls_breakeven():
    """Замок звʼязку: частковий вихід і переведення в БЗ — одна дія."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('def _check_manual_tp1')
    j = src.index('def _tp1_move_to_breakeven')
    _check('self._tp1_move_to_breakeven(' in src[i:j],
           'TP-1 мусить кликати переведення в беззбиток')
    print('✓ TP-1 і беззбиток звʼязані в коді')


# ═════════ 📨 TELEGRAM: TP-1 / TP-2 не бувають порожні ══════════════════════
def _tm_msg(**st):
    o = _tm()
    o._settings = {'pilot_enabled': True, 'pilot_autofill_tp': True,
                   'pilot_tp1_close_pct': 50}
    o._settings.update(st)
    o.sent = []
    o._notify = lambda m, is_test=False, category=None: o.sent.append(m)
    return o


def test_tg_open_carries_both_tp_levels():
    o = _tm_msg()
    pos = {'symbol': 'FILUSDT', 'side': 'SHORT', 'entry_price': 0.70370,
           'manual_sl': 0.73032, 'manual_tp1': 0.69313, 'manual_tp': 0.6096}
    o._notify_open(pos)
    m = o.sent[0]
    _check('TP-1' in m and 'TP-2' in m, f'обидва рівні мають бути в TG: {m}')
    _check('0.69313' in m and '0.6096' in m,
           f'у повідомленні мусять бути САМІ ЧИСЛА: {m}')
    _check('(+1.50%)' in m and '(+13.37%)' in m,
           f'відсоток має стояти ОДРАЗУ після ціни кожного рівня: {m}')
    _check('(50%)' not in m and 'повний' not in m,
           f'підписи «(50%)» і «(повний)» прибрано на прохання: {m}')
    print('✓ TG-відкриття: TP-1/TP-2 з ціною і % одразу за нею, без підписів')


def test_tg_says_nothing_about_tp_when_there_is_nothing_to_say():
    """Вимога користувача: якщо рівнів ще немає — про них НЕ пишемо ЖОДНОГО
    слова (не «рахується», не прочерк). Повідомлення має лишатись коротким."""
    o = _tm_msg()
    pos = {'symbol': 'FILUSDT', 'side': 'SHORT', 'entry_price': 0.70370,
           'manual_sl': 0.73032}
    o._notify_open(pos)
    m = o.sent[0]
    _check('TP' not in m, f'жодної згадки про TP бути не повинно: {m}')
    _check('рахується' not in m and 'null' not in m, f'ні статусу, ні null: {m}')
    _check(m.count('\n') == 3, f'рівно 4 рядки: напрямок, монета, вхід, SL: {m!r}')
    print('✓ рівнів немає → у повідомленні про них ні слова')


def test_levels_message_has_no_service_tail():
    """Друге повідомлення — це САМІ рівні. Підсумкового рядка «вхід … · TP-1 …»
    користувач попросив прибрати: відсотки стоять біля цін."""
    o = _tm_msg()
    pos = {'side': 'SHORT', 'entry_price': 100.0,
           'manual_tp1': 98.0, 'manual_tp': 95.0}
    o._notify_pilot_levels('XUSDT', pos, False)
    m = o.sent[0]
    _check('(+2.00%)' in m and '(+5.00%)' in m, f'% біля кожної ціни: {m}')
    _check('вхід' not in m, f'службового рядка з входом бути не повинно: {m}')
    _check(m.count('\n') == 2, f'заголовок + два рівні, і все: {m!r}')
    print('✓ повідомлення про рівні = лише рівні з відсотками')


def test_levels_message_skipped_when_empty():
    o = _tm_msg()
    o._notify_pilot_levels('XUSDT', {'side': 'LONG', 'entry_price': 100.0}, False)
    _check(o.sent == [], 'нема чого показувати → повідомлення не шлемо взагалі')
    print('✓ порожніх повідомлень про рівні не буває')


def test_levels_message_is_sent_when_pilot_fills_them():
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('def _pilot_apply_tp')
    j = src.index('def get_pilot_state')
    _check('_notify_pilot_levels(symbol, pos, is_shadow)' in src[i:j],
           'після виставлення рівнів має піти повідомлення')
    print('✓ виставлення рівнів → повідомлення в Telegram')


# ═════════ ✏️ РУЧНИЙ SL: АВТОПІЛОТ НЕ ЧІПАЄ ════════════════════════════════
def test_pilot_does_not_overwrite_a_hand_set_stop():
    """🐞 КЕЙС SOLUSDT (01.09, знайдено в лозі). Користувач ТРИЧІ ставив
    Manual SL $96.82 — і щоразу через ~20 секунд трейл автопілота
    перезаписував його на $99.9094:

        16:58:14 ✏️ Користувач: Manual SL → $96.82
        16:58:38 🤖 Бот (Автопілот · структура): Manual SL → $99.9094
        16:59:09 ✏️ Користувач: Manual SL → $96.82
        16:59:31 🤖 Бот (Автопілот · структура): Manual SL → $99.9094

    Виглядало як «ручний SL не зберігається», хоча він зберігався і його
    одразу затирали: `plan()` отримує ручний рівень просто як `prev_stop`,
    ратчет вважає структурний стоп кращим (для LONG вище = ближче до
    прибутку) і застосовує його. Для Manual TP правило «оператора не чіпаємо»
    вже діяло — SL мусить поводитись так само."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index("if act == 'trail' and res.get('stop'):")
    j = src.index('lvl = self._round_sltp_value(res[', i)
    guard = src[i:j]
    _check("pos.get('manual_sl_src') == self.SRC_USER" in guard,
           'перед трейлом має стояти перевірка на ручний рівень')
    _check('return False' in guard,
           'ручний рівень → трейл НЕ застосовується')
    _check('Очистіть поле SL' in guard,
           'у лозі має бути сказано, ЯК повернути автопілоту керування')
    print('✓ автопілот не перезаписує стоп, виставлений руками')


def test_clearing_the_stop_returns_control_to_the_pilot():
    """Замок не вічний: очистив поле SL → автопілот знову веде стоп."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index("elif sl_op[0] == 'clear':")
    body = src[i:i + 700]
    _check("pos.pop('_pilot_sl_user_lock', None)" in body,
           'очищення рівня має знімати замок')
    # А нове ручне значення — заново дозволяє пояснення в лозі.
    # ⚠️ Якорити треба на ГІЛЦІ МУТАЦІЇ (`pos['manual_sl'] = ...`), а не на
    # першому `sl_op[0] == 'set'` у файлі — той належить ВАЛІДАЦІЇ.
    k = src.index("pos['manual_sl'] = _sv")
    _check("pos.pop('_pilot_sl_user_lock', None)" in src[k:k + 700],
           'нове ручне значення теж має скидати позначку «уже пояснили»')
    print('✓ очищення SL повертає керування автопілоту')


def test_the_lock_is_logged_once_not_every_tick():
    """Монітор тікає раз на 4с, автопілот — раз на 20с. Без анти-флуду
    пояснення сипалось би в лог сотнями рядків."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index("if act == 'trail' and res.get('stop'):")
    guard = src[i:i + 1600]
    _check("if not pos.get('_pilot_sl_user_lock'):" in guard,
           'рядок у лог має писатись ОДИН раз на рівень')
    print('✓ пояснення пишеться раз, а не щотіку')


# ═════════ 🔇 СПАМ ТИМ САМИМ РІВНЕМ (кейс ETHUSDT 09.09) ═══════════════════
def test_ratchet_compares_the_rounded_level_not_the_raw_one():
    """🐞 ГОЛОВНИЙ ДЕФЕКТ. У лозі щотакту (20-35с) стояли ДВА однакові рядки:

        22:45:32 🤖 Бот (Автопілот · структура): Manual SL → $2543.63
        22:45:32 🎯 Автопілот: SL → $2,543.63 · стоп підтягнуто: структура…
        22:46:08 (те саме)   22:46:45 (те саме) …

    — і так годинами, хоча стоп НЕ РУХАВСЯ.

    Корінь: ратчет порівнював СИРЕ число з ОКРУГЛЕНИМ. `plan()` рахує
    кандидата з повною точністю (swing×(1+buf) = 2543.62973), а на позиції
    лежить УЖЕ ОКРУГЛЕНИЙ рівень (2543.63). Для SHORT «краще» = нижче, тож
    2543.62973 < 2543.63 → `better_stop` каже ТАК → пишемо те саме 2543.63.
    Вічний цикл на дві сотих цента."""
    tp = _load('detection.trade_pilot', 'detection/trade_pilot.py')
    rnd = TM._round_sltp_value
    raw = 2539.82 * (1 + 0.0015)              # структурний стоп SHORT
    stored = rnd(raw)
    _check(stored == 2543.63, f'відтворення кейсу зі скріна: {stored}')
    _check(raw != stored, 'сире число МУСИТЬ відрізнятись — у цьому й пастка')
    # Ратчет на СИРОМУ vs збереженому — саме він і зациклював
    _check(tp.better_stop('SHORT', raw, stored) is True,
           'без фіксу ратчет вічно вважає сирий кандидат «кращим»')
    # А на ОКРУГЛЕНОМУ — рівність, тобто робити нічого
    _check(tp.better_stop('SHORT', rnd(raw), stored) is False,
           'округлений кандидат = чинний рівень → трейлу немає')
    print(f'✓ пастка відтворена: сире {raw} vs збережене {stored}')


def test_no_op_trail_writes_nothing_at_all():
    """Гейт у коді: якщо ОКРУГЛЕНИЙ кандидат дорівнює чинному рівню — ні
    запису, ні рядка в лозі, ні приросту 🛡. Порівнюємо саме ТЕ, ЩО РЕАЛЬНО
    ЗАПИШЕМО, з тим, ЩО ВЖЕ СТОЇТЬ — це закриває будь-яке джерело мікрошуму,
    а не лише конкретно це."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('lvl = self._round_sltp_value(res[')
    j = src.index('r = self.update_manual_sl_tp(', i)
    guard = src[i:j]
    _check("_cur = self._round_sltp_value(pos.get('manual_sl'))" in guard,
           'чинний рівень теж треба ОКРУГЛИТИ перед порівнянням')
    _check('_cur == lvl' in guard and 'return False' in guard,
           'рівні збіглись → вихід ДО запису')
    # ⚠️ Гейт мусить стояти ПЕРЕД викликом, інакше він нічого не економить
    _check(guard.index('_cur == lvl') < len(guard),
           'перевірка мусить бути ДО update_manual_sl_tp')
    print('✓ трейл «у те саме місце» не пише нічого')


def test_one_action_is_one_log_row():
    """Правило проєкту «один рядок = одна подія». Автопілот пише СВІЙ,
    змістовніший рядок (рівень + причина + ціль), тож загальний рядок TM на
    цьому шляху лише дублював подію — і подвоював той самий флуд."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    # 1) у TM є вимикач логу, і він гасить САМЕ лог, а не запис рівня
    i = src.index('if parts and not quiet:')
    _check('quiet: bool = False' in src, 'потрібен параметр `quiet` із дефолтом False')
    head = src[:i]
    _check("pos['manual_sl_by'] = origin_label or ''" in head,
           'позначка 🏷 походження мусить лишатись — гаситься лише лог')
    # 2) обидва шляхи автопілота (трейл SL і TP-2) кличуть з quiet=True
    for anchor in ("origin_label='Автопілот · структура'",
                   'origin_label=f"Автопілот · {tp2.get(\'label\')}"'):
        k = src.index(anchor)
        _check('quiet=True' in src[k:k + 400],
               f'виклик біля «{anchor[:34]}…» мусить бути quiet=True')
    # 3) власний рядок автопілота НЕ втратив структурні поля для CSV
    m = src.index("f'🎯 Автопілот: SL → {self._fmt_price(lvl)} · {_why}'")
    _check("'manual_sl': lvl" in src[m:m + 400] and "'origin'" in src[m:m + 400],
           'після quiet=True поля для CSV мусять бути у ВЛАСНОМУ рядку')
    print('✓ одна дія — один рядок (і CSV не втратив полів)')


# ═════════ 📨 ЧАСТКОВЕ ЗАКРИТТЯ: у ГРУПУ і КОРОТКО ═════════════════════════
def test_partial_close_goes_to_the_group_topic():
    """🐞 Повідомлення про часткове закриття йшли в ОСОБИСТИЙ бот: `_notify`
    без `category` шле адміну в приват, який зарезервований під службові
    повідомлення. Ринкові події мають іти в груповий топік."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    for fn in ('_partial_close', '_partial_close_shadow'):
        i = src.index(f'def {fn}(')
        j = src.index('def ', i + 10)
        body = src[i:j]
        k = body.index('self._notify(')
        call = body[k:k + 400]
        _check("category='trades'" in call,
               f'{fn}: повідомлення мусить іти в груповий топік, а не в приват')
    print('✓ часткове закриття йде в групу, не в особистий бот')


def test_partial_close_message_is_one_line():
    """Було шість рядків на рутинну подію. Має бути один."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    for fn in ('_partial_close', '_partial_close_shadow'):
        i = src.index(f'def {fn}(')
        j = src.index('def ', i + 10)
        body = src[i:j]
        k = body.index('self._notify(')
        call = body[k:k + 400]
        _check(call.count('\\n') <= 1,
               f'{fn}: повідомлення має бути компактним, а не багаторядковим')
        for gone in ('Reason:', 'Remaining:', 'Paper trade (no real close)',
                     'of position closed', 'of paper position closed'):
            _check(gone not in call, f'{fn}: «{gone}» мало зникнути')
    print('✓ повідомлення про часткове закриття — один рядок')


def test_tp_lines_have_no_labels():
    """Пункт 2: «(50%)» і «(повний)» прибрано з рядків рівнів."""
    o = _tm_msg()
    pos = {'side': 'SHORT', 'entry_price': 100.0,
           'manual_tp1': 98.0, 'manual_tp': 95.0}
    txt = o._tp_lines(pos)
    _check('TP-1' in txt and 'TP-2' in txt, f'самі рівні лишаються: {txt}')
    _check('(50%)' not in txt and '%)' in txt,
           f'частка прибрана, а відсоток від входу лишився: {txt}')
    _check('повний' not in txt, f'«(повний)» мало зникнути: {txt}')
    print(f'✓ рядки рівнів без підписів: {txt.splitlines()[0]}')


# ═════════ ✋ ЗНЯТИЙ Manual TP-2 = ЗНЯТИЙ АВТО-ВИХІД ════════════════════════
# Скарга дослівно: «Чому, коли я вручну видаляю Manual TP-1 Manual TP-2, вони
# все одно спрацьовують?» Поле очищалось ПРАВИЛЬНО — угоду закривали ДВА інші
# шляхи на ТІЙ САМІЙ ціні: стратегічний TP (`use_tp`/`tp_pct`, у REAL-книзі ще
# й НА БІРЖІ) і автопілот, для якого TP-2 і Є ціллю (🧲 магніт).
def test_the_rule_measures_tp2_not_both_fields():
    """ЄДИНЕ ДЖЕРЕЛО правила — `_tp2_cleared_by_user`.

    Міряємо САМЕ TP-2: TP-1 — ЧАСТКОВА фіксація, і зняти її, лишивши повний
    вихід, — нормальний сценарій, який не має вимикати автоматичний вихід."""
    f = TM._tp2_cleared_by_user
    _check(f({'pilot_tp_cleared': True}) is True,
           'знято TP-2 → авто-фіксації немає')
    _check(f({'pilot_tp_cleared': True, 'manual_tp': 1.23}) is False,
           'TP-2 стоїть → правило не діє, хоч би що робили з TP-1')
    _check(f({'pilot_tp_cleared': True, 'manual_tp1': 1.1}) is True,
           'знятий TP-2 лишається знятим, навіть коли TP-1 на місці')
    _check(f({}) is False, 'порожня позиція — не «оператор зняв»')
    _check(f({'manual_tp': 0}) is False,
           'рівня просто ще немає (не рахований) — це НЕ рішення оператора')
    print('✓ правило міряє саме TP-2, а не «обидва поля порожні»')


def test_pilot_does_not_close_at_a_level_the_operator_deleted():
    """Автопілот: `act='take'` на знятому TP-2 НЕ закриває угоду.

    Без цього гейта видалення рівня нічого не змінювало: угода закривалась на
    тій самій ціні, лише з причиною «🎯 Автопілот (ціль)» замість Manual TP —
    саме це й виглядало як «видалений TP усе одно спрацював»."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index("if act == 'take' and self._tp2_cleared_by_user(pos):")
    j = src.index("if act == 'take':", i)
    guard = src[i:j]
    _check('return False' in guard, 'знятий рівень → угоду НЕ фіксуємо')
    _check("_pilot_mark(pkey, take_block=" in guard,
           'стан має сказати ПРАВДУ: план хотів фіксацію, її заблоковано')
    _check("if not pos.get('_pilot_take_user_lock'):" in guard,
           'анти-флуд: пояснення пишеться раз, а не щотакту (такт 20с)')
    _check('Впишіть Manual TP-2' in guard,
           'у лозі має бути сказано, ЯК повернути автоматичну фіксацію')
    # ⚠️ Гейт мусить стояти ПЕРЕД гілкою закриття, інакше він марний.
    _check(i < src.index("self._close_position(symbol, px, reason='pilot_target')"),
           'гейт мусить стояти ДО закриття за ціллю')
    print('✓ автопілот не закриває на рівні, який оператор видалив')


def test_strategic_tp_respects_the_cleared_level_in_both_books():
    """Стратегічний TP (`use_tp` + `tp_pct`) — другий шлях на ту саму ціну.

    Паперова книга ДЗЕРКАЛИТЬ реальну: інакше та сама угода поводилась би
    по-різному залежно від книги."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    for close_call, book in (("self._close_position(symbol, current_price, reason='take_profit')", 'real'),
                             ("self._close_shadow(symbol, current_price, reason='take_profit')", 'paper')):
        k = src.index(close_call)
        head = src[max(0, k - 900):k]
        _check('self._tp2_cleared_by_user(pos)' in head,
               f'{book}: перед закриттям по стратегічному TP має стояти гейт')
        _check('_log_tp_cleared_once' in head,
               f'{book}: мовчазний пропуск читався б як «бот завис»')
    print('✓ стратегічний TP поважає знятий рівень в ОБОХ книгах')


def test_the_strategic_block_is_said_once_not_every_tick():
    """Монітор тікає раз на 4с, а ціна може стояти вище TP годинами."""
    _reset()
    o = _tm()
    pos = {'side': 'LONG', 'entry_price': 100.0, 'tp_price': 105.0,
           'pilot_tp_cleared': True}
    o._log_tp_cleared_once('XRPUSDT', pos, False)
    o._log_tp_cleared_once('XRPUSDT', pos, False)
    o._log_tp_cleared_once('XRPUSDT', pos, False)
    _check(len(_LOG) == 1, f'мав бути РІВНО один рядок, а не {len(_LOG)}')
    _check('ЗНЯТО ОПЕРАТОРОМ' in _text(), f'причина має бути названа: {_text()}')
    print('✓ стратегічний блок пояснюється один раз')


def test_putting_the_level_back_releases_the_lock():
    """Замок НЕ вічний: вписали Manual TP-2 → автофіксація повертається,
    і пояснення зʼявиться знову, якщо рівень знімуть удруге."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    k = src.index("pos['manual_tp'] = _tv")
    body = src[k:k + 900]
    _check("pos.pop('pilot_tp_cleared', None)" in body,
           'ручний рівень знімає позначку «оператор зняв TP-2»')
    _check("pos.pop('_pilot_take_user_lock', None)" in body
           and "pos.pop('_tp_cleared_logged', None)" in body,
           'позначки «про це вже сказано» теж скидаються')
    _check('_src == self.SRC_USER' in body,
           'знімає замок САМЕ оператор, а не автопілот')
    print('✓ повернутий рівень знімає замок')


def test_exchange_tp_can_be_cancelled_at_all():
    """🏦 РІВЕНЬ ЖИВ НЕ ЛИШЕ В БОТІ. Для реальної угоди тейк, поставлений при
    відкритті, стоїть НА BYBIT — і скасувати його було ФІЗИЧНО неможливо:
    `set_trading_stop` мав `if take_profit:`, тобто ковтав нуль, яким Bybit і
    скасовує тейк. Поле в інтерфейсі порожнє, а біржа закриває позицію сама."""
    src = open(os.path.join(_ROOT, 'core/bybit_connector.py')).read()
    i = src.index('def set_trading_stop')
    body = src[i:i + 1800]
    _check('if take_profit is not None:' in body,
           'нуль — ЗМІСТОВНЕ значення (скасувати), перевірка мусить бути is not None')
    # ⚠️ Коментар ПОЯСНЮЄ стару перевірку дослівно, тож замок мусить дивитись
    # на КОД, а не на текст (та сама пастка, що вже ловилась на `ensure_fresh`).
    code = '\n'.join(ln for ln in body.splitlines()
                     if not ln.strip().startswith('#'))
    _check('if take_profit:' not in code,
           'стара перевірка робила скасування неможливим')
    print('✓ біржовий тейк-профіт тепер можна скасувати')


def test_clearing_tp2_cancels_the_exchange_tp_and_says_what_happened():
    """І скасовуємо його САМЕ тоді, коли оператор зняв Manual TP-2 — інакше
    внутрішні гейти нічого не варті: біржа закриє позицію повз усю логіку."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('# 🏦 РЕАЛЬНА КНИГА: знятий TP-2 знімаємо Й НА БІРЖІ.')
    body = src[i:i + 2200]
    _check("not is_shadow" in body, 'паперова книга біржу не чіпає')
    _check("tp_op[0] == 'clear'" in body, 'лише на ЗНЯТТЯ рівня')
    _check("origin != self.SRC_AUTO" in body,
           'лише рішення ОПЕРАТОРА, не двигуна')
    _check('take_profit=0' in body, 'скасування — це явний нуль')
    _check('_exok' in body and 'НЕ вдалось скасувати' in body,
           'результат ЧИТАЄМО: відмова означає, що тейк лишився на біржі')
    print('✓ знятий TP-2 знімає й біржовий тейк, а відмова не мовчить')


def test_protective_exits_are_never_touched():
    """⚠️ Правило знімає ФІКСАЦІЮ ПРИБУТКУ, а не ЗАХИСТ. Стоп-лос і решта
    виходів мусять лишитись недоторканими — ризик не знімаємо ніколи."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    for close_call in ("self._close_position(symbol, current_price, reason='stop_loss')",
                       "self._close_shadow(symbol, current_price, reason='stop_loss')"):
        k = src.index(close_call)
        _check('_tp2_cleared_by_user' not in src[max(0, k - 400):k],
               'стоп-лос НЕ має залежати від знятого TP-2')
    print('✓ захисні виходи правило не зачіпає')


# ═════════ 🧮 ПРАВИЛО «МММ LiQ ⚖» ГАСИТЬ АВТОМАТИКУ АВТОПІЛОТА ══════════
# Вимога дослівно (17.09): «Якщо увімкнено "🧮 МММ LiQ ⚖ → вихід" — потрібно
# автоматично вимкнути все, що стосується автоматичного "🎯 Автопілот"; якщо
# Manual TP-1 або Manual TP-2 виставлені вручну, бот має реагувати на ручні
# дані. І тумблер ⚖️ TP-1 переводить SL у беззбиток має працювати.»
def test_the_rule_and_the_autopilot_automation_are_mutually_exclusive():
    """ЄДИНЕ джерело умови — щоб «вимкнено» означало те саме в логіці й в UI."""
    f = TM._pilot_auto_off
    _check(f({'use_mm_flat_exit': True}) is True, 'правило увімкнене → автоматики немає')
    _check(f({'use_mm_flat_exit': False}) is False, 'вимкнене правило нічого не гасить')
    _check(f({}) is False, 'порожні налаштування — не привід гасити автопілот')
    print('✓ правило і автоматика автопілота — взаємовиключні')


def test_the_gate_stops_every_DECISION_of_the_autopilot():
    """Правило гасить РІШЕННЯ автопілота: ціль із графіка, трейл стопа, `take`,
    автозаповнення TP-2, план поділу. Лишається РІВНО одна дія — 🧲 магніт у
    Manual TP-1 (вимога 17.09), і вона нічого не закриває.

    ⚠️ Раніше тут стояв замок «такту НЕ БУЛО взагалі» (`_pilot_at` порожній).
    Користувач ЗМІНИВ вимогу — редукований такт тепер потрібен, тож замок
    переписано під нову поведінку, а не «полагоджено»."""
    _reset()
    o = _tm()
    o._settings = {'pilot_enabled': True, 'use_mm_flat_exit': True,
                   'pilot_tp2_from_magnet': True, 'pilot_autofill_tp': True,
                   'pilot_tp1_close_pct': 50}
    o._pilot_state, o._pilot_at = {}, {}
    o._magnet_data_ok = True
    o._magnet_objective = lambda sym, side, entry: None
    pos = {'side': 'LONG', 'entry_price': 100.0}
    _check(o._pilot_tick('BTCUSDT', pos, 101.0, False) is False,
           'редукований такт НІКОЛИ не закриває угоду')
    st = o.get_pilot_state('BTCUSDT', False)
    _check(st and st.get('auto_off'),
           f'колонка мусить сказати ПРИЧИНУ, а не застигнути: {st}')
    _check(st.get('action') != 'take' and not st.get('trail_block'),
           'ні фіксації, ні трейлу в цьому режимі немає')
    _check('pilot_r_stop' not in pos, 'без стопа якір R не вигадуємо')
    # ⚠️ Реальна і паперова книги — РІЗНІ ключі (урок TRXUSDT).
    o._pilot_tick('BTCUSDT', pos, 101.0, True)
    _check(o.get_pilot_state('BTCUSDT', True) is not None
           and o.get_pilot_state('BTCUSDT', True) is not st,
           'паперова книга має власний знімок')
    print('✓ гейт гасить рішення автопілота, лишаючи тільки 🧲 → TP-1')


def test_the_gate_stands_before_any_pilot_work():
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('def _pilot_tick')
    # ⚠️ Зріз — РІВНО тіло `_pilot_tick`: редукований такт живе в
    # ОКРЕМОМУ методі нижче, і його рядки не мають плутати замок.
    j = src.index('def _pilot_auto_off', i)
    body = src[i:j]
    g = body.index('if self._pilot_auto_off(s):')
    _check(g < body.index('from detection import trade_pilot'),
           'гейт мусить стояти ДО завантаження модуля автопілота')
    _check(g < body.index('pilot_tp2_from_magnet'),
           'ДО 🧲 магніту — інакше запит до біржі робився б дарма')
    _check(g < body.index("if act == 'take'"),
           'ДО закриття за ціллю')
    _check(g < body.index('pilot_autofill_tp'),
           'ДО автозаповнення Manual TP-1/TP-2')
    print('✓ гейт стоїть найпершим — жодна автоматика повз нього не проходить')


def test_manual_levels_and_breakeven_are_outside_the_gate():
    """РУЧНІ рівні мусять працювати далі — це половина вимоги."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('def _check_manual_tp1')
    j = src.index('def _tp1_move_to_breakeven', i)
    tp1 = src[i:j]
    _check('_pilot_auto_off' not in tp1,
           'перевірка РУЧНОГО TP-1 не має залежати від правила')
    _check('_tp1_move_to_breakeven' in tp1,
           '⚖️ TP-1 → беззбиток викликається саме звідси і мусить лишитись')
    be = src[src.index('def _tp1_move_to_breakeven'):][:3000]
    _check('_pilot_auto_off' not in be,
           'беззбиток після TP-1 працює незалежно від правила')
    # І в МОНІТОРАХ обидві перевірки стоять ОКРЕМИМИ викликами після пілота.
    for anchor in ('self._check_manual_tp1(symbol, pos, current_price, False)',
                   'self._check_manual_tp1(symbol, pos, current_price, True)'):
        k = src.index(anchor)
        _check('_pilot_tick' not in src[k:k + 200],
               'ручний TP-1 — окремий крок монітора, не частина автопілота')
    _check('manual_reason = self._check_manual_sl_tp(pos, current_price)' in src,
           'ручний SL/TP теж лишається окремим кроком')
    print('✓ ручні Manual TP-1/TP-2 і ⚖️ беззбиток гейт не зачіпає')


def test_the_rule_does_not_rewrite_the_users_own_toggle():
    """Гасимо ПОВЕДІНКУ, а не чужий тумблер у БД: вимкнув правило — автопілот
    повернувся рівно таким, як його налаштували."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('def _pilot_auto_off')
    body = src[i:src.index('def _pilot_magnet_tp1', i)]
    _check("s.get('use_mm_flat_exit')" in body, 'умова читає саме це правило')
    _check("'pilot_enabled'" not in body.split('"""')[-1],
           'чужий тумблер НЕ переписуємо')
    print('✓ правило гасить поведінку, а не налаштування користувача')


# ═════════ 🧲 МАГНІТ → Manual TP-1 у режимі 🧮 (вимога 17.09) ═══════════════
# Дослівно: «при виборі 🧮 МММ LiQ ⚖ → вихід із 🎯 Автопілота беремо
# "🧲 найбільший магніт" і ставимо в Manual TP-1. І таким чином щоб не було
# пустим поле 🎯 Автопілот в таблицях відкритих угод — підраховуй дані для
# Manual TP-1.»
def _mm_tm(**over):
    o = _tm()
    o._settings = dict({'pilot_enabled': True, 'use_mm_flat_exit': True,
                        'pilot_tp2_from_magnet': True, 'pilot_autofill_tp': True,
                        'pilot_tp1_close_pct': 50}, **over)
    o._pilot_state, o._pilot_at = {}, {}
    o._magnet_data_ok = True
    o.db = None          # персист у стабі не потрібен
    o.calls = []

    def _mag(sym, side, entry):
        o.calls.append(sym)
        return {'price': 2.0, 'dist_pct': 8.51, 'kind': 'magnet',
                'label': '🧲 магніт $2.0000'}
    o._magnet_objective = _mag
    return o


def test_the_magnet_becomes_manual_tp1_and_the_column_is_filled():
    _reset()
    o = _mm_tm()
    pos = {'side': 'SHORT', 'entry_price': 2.1860, 'manual_sl': 2.3033}
    _check(o._pilot_tick('NEOUSDT', pos, 2.1500, False) is False,
           'редукований такт угоду не закриває')
    _check(_near(pos.get('manual_tp1'), 2.0),
           f'магніт мав лягти в Manual TP-1: {pos.get("manual_tp1")}')
    _check(pos.get('manual_tp') is None,
           'TP-2 у цьому режимі НЕ ставимо — просили саме TP-1')
    st = o.get_pilot_state('NEOUSDT', False)
    _check(st.get('objective') and st.get('progress'),
           f'колонка «🎯 Автопілот» не має бути порожньою: {st}')
    _check(st.get('r') and st['r'] > 1,
           f'R мусить рахуватись (від початкового стопа): {st.get("r")}')
    _check(st.get('r_stop') == 2.3033, 'якір R — стоп на момент появи')
    _check('магніт' in (st.get('why') or ''), f'у тултипі має бути причина: {st}')
    _check('TP-1' in _text() and 'Автопілот' in _text(),
           f'рівень мусить бути названий у 🧾 Лозі: {_text()}')
    print('✓ 🧲 магніт → Manual TP-1, колонка заповнена числами')


def test_the_exchange_is_asked_once_per_trade():
    """Магніт — це запит до біржі. Питаємо РІВНО один раз на угоду."""
    _reset()
    o = _mm_tm()
    pos = {'side': 'SHORT', 'entry_price': 2.1860, 'manual_sl': 2.3033}
    for _ in range(3):
        o._pilot_at = {}            # знімаємо тротл, імітуючи наступні такти
        o._pilot_tick('NEOUSDT', pos, 2.15, False)
    _check(o.calls == ['NEOUSDT'], f'мав бути ОДИН запит, а не {o.calls}')
    print('✓ біржу питаємо один раз на угоду')


def test_the_source_toggle_off_means_no_exchange_call_at_all():
    _reset()
    o = _mm_tm(pilot_tp2_from_magnet=False)
    pos = {'side': 'LONG', 'entry_price': 100.0}
    o._pilot_tick('BTCUSDT', pos, 101.0, False)
    _check(o.calls == [], 'вимкнене джерело → зайвого запиту до біржі немає')
    st = o.get_pilot_state('BTCUSDT', False)
    _check('вимкнено' in (st.get('tp_skip') or ''),
           f'причина мусить бути названа, а не порожнеча: {st}')
    _check(pos.get('manual_tp1') is None, 'рівня немає звідки взяти')
    print('✓ вимкнене джерело магніту — біржу не турбуємо, причину кажемо')


def test_the_operators_own_tp1_is_never_overwritten():
    _reset()
    o = _mm_tm()
    pos = {'side': 'SHORT', 'entry_price': 2.1860, 'manual_sl': 2.3033,
           'manual_tp1': 2.1000, 'manual_tp1_src': TM.SRC_USER}
    o._pilot_tick('NEOUSDT', pos, 2.15, False)
    _check(_near(pos['manual_tp1'], 2.1000),
           f'рівень оператора лишається недоторканим: {pos["manual_tp1"]}')
    print('✓ ручний TP-1 автопілот не перекриває')


def test_a_cleared_tp1_stays_cleared():
    """Оператор ЗНЯВ рівень → не відновлюємо, але кажемо про це в тултипі."""
    _reset()
    o = _mm_tm()
    pos = {'side': 'SHORT', 'entry_price': 2.1860, 'manual_sl': 2.3033,
           'pilot_tp_cleared': True}
    o._pilot_tick('NEOUSDT', pos, 2.15, False)
    _check(pos.get('manual_tp1') is None, 'знятий рівень не повертаємо')
    _check(o.get_pilot_state('NEOUSDT', False).get('tp_locked') is True,
           'стан «зняв оператор» мусить бути видно')
    print('✓ знятий TP-1 лишається знятим і це видно')


def test_no_magnet_means_a_reason_not_an_empty_cell():
    _reset()
    o = _mm_tm()
    o._magnet_objective = lambda *a: None
    o._magnet_skip = 'магніт лежить ПОЗАДУ входу'
    pos = {'side': 'LONG', 'entry_price': 100.0}
    o._pilot_tick('BTCUSDT', pos, 101.0, False)
    st = o.get_pilot_state('BTCUSDT', False)
    _check('ПОЗАДУ' in (st.get('tp_skip') or ''), f'причина зі сканера: {st}')
    _check('магніт' in (st.get('why') or ''), f'і в підсумку теж: {st}')
    print('✓ немає магніту — комірка каже ЧОМУ')


def test_the_exchange_silence_is_retried_not_remembered():
    """Біржа не відповіла → позначку «питали» НЕ ставимо: наступний такт
    спробує ще (зріз кешується, повтор безкоштовний)."""
    _reset()
    o = _mm_tm()
    o._magnet_objective = lambda *a: None
    o._magnet_data_ok = False
    pos = {'side': 'LONG', 'entry_price': 100.0}
    o._pilot_tick('BTCUSDT', pos, 101.0, False)
    _check(not pos.get('pilot_magnet_done'),
           'мовчання біржі не має закривати питання назавжди')
    print('✓ мовчання біржі — повторимо, а не забудемо')


def test_the_page_shows_the_magnet_state_separately():
    html = open(os.path.join(_ROOT, 'templates/smart_money.html')).read()
    _check('_PILOT_AUTO_TP1' in html,
           'коли магніт є — комірка мусить казати саме це, а не «вимкнено»')
    i = html.index('const [ic, lbl, col] = ')
    chain = html[i:i + 460]
    _check('pl.objective ? _PILOT_AUTO_TP1 : _PILOT_AUTO_OFF' in chain,
           'два РІЗНІ стани: є магніт / магніту немає')
    tip = html[html.index('pl.auto_off ?', i):][:700]
    _check('Manual TP-1' in tip and 'Manual TP-2' in tip,
           'тултип мусить сказати, що саме лишилось працювати')
    # Тумблери, які в цьому режимі ЗНОВУ мають сенс, гасити не можна.
    dim = html[html.index('const _PILOT_AUTO_IDS'):][:900]
    for _id in ('tm-pilot-autofill-tp', 'tm-pilot-tp2-magnet'):
        _check(f"'{_id}'" not in dim,
               f'{_id} керує магнітом у TP-1 — гасити його не можна')
    print('✓ сторінка розрізняє «магніт → TP-1» і «нічого не робимо»')


if __name__ == '__main__':
    test_mnt_case_star_1h_block_is_used_instead_of_waiting()
    test_chosen_source_wins_for_every_trade()
    test_ob_tf_is_the_fallback_when_chosen_source_has_nothing()
    test_volumized_used_when_both_ob_rows_unusable()
    test_chosen_1h_missing_walks_down_to_15m_then_5m()
    test_ladder_does_not_ask_the_same_timeframe_twice()
    test_guarantee_still_last_when_no_timeframe_has_a_block()
    test_volumized_skips_breaker()
    test_percent_fallback_guarantees_a_stop()
    test_long_fallback_is_below_entry()
    test_fallback_can_be_turned_off()
    test_wrong_side_level_is_skipped_not_applied()
    test_max_pct_clamps_a_far_block()
    test_max_pct_zero_changes_nothing()
    test_set_once_is_preserved()
    test_user_typed_sl_is_never_touched()
    test_disabled_feature_does_nothing()
    test_defaults_guarantee_a_stop()
    test_autosl_marks_level_as_bot_origin()
    test_user_edit_overrides_bot_origin()
    test_clearing_removes_origin_mark()
    test_tp_origin_tracked_separately()
    test_rejected_level_leaves_no_mark()
    test_q4_trade_uses_the_configured_1h_source_first()
    test_q4_trade_with_15m_choice_uses_volumized_15m()
    test_every_trade_now_uses_the_chosen_source()
    test_q4_source_unavailable_falls_back_and_says_so()
    test_breakeven_level_covers_round_trip_fees()
    test_tp1_moves_stop_to_breakeven()
    test_breakeven_never_loosens_a_better_stop()
    test_breakeven_improves_a_worse_stop()
    test_rejected_breakeven_is_reported_not_faked()
    test_every_frontend_position_route_exists()
    test_breakeven_is_off_by_default()
    test_tp1_calls_breakeven()
    test_tg_open_carries_both_tp_levels()
    test_tg_says_nothing_about_tp_when_there_is_nothing_to_say()
    test_levels_message_has_no_service_tail()
    test_levels_message_skipped_when_empty()
    test_levels_message_is_sent_when_pilot_fills_them()
    test_pilot_does_not_overwrite_a_hand_set_stop()
    test_clearing_the_stop_returns_control_to_the_pilot()
    test_the_lock_is_logged_once_not_every_tick()
    test_ratchet_compares_the_rounded_level_not_the_raw_one()
    test_no_op_trail_writes_nothing_at_all()
    test_one_action_is_one_log_row()
    test_partial_close_goes_to_the_group_topic()
    test_partial_close_message_is_one_line()
    test_tp_lines_have_no_labels()
    test_the_rule_measures_tp2_not_both_fields()
    test_pilot_does_not_close_at_a_level_the_operator_deleted()
    test_strategic_tp_respects_the_cleared_level_in_both_books()
    test_the_strategic_block_is_said_once_not_every_tick()
    test_putting_the_level_back_releases_the_lock()
    test_exchange_tp_can_be_cancelled_at_all()
    test_clearing_tp2_cancels_the_exchange_tp_and_says_what_happened()
    test_protective_exits_are_never_touched()
    test_the_rule_and_the_autopilot_automation_are_mutually_exclusive()
    test_the_gate_stops_every_DECISION_of_the_autopilot()
    test_the_gate_stands_before_any_pilot_work()
    test_manual_levels_and_breakeven_are_outside_the_gate()
    test_the_rule_does_not_rewrite_the_users_own_toggle()
    test_the_magnet_becomes_manual_tp1_and_the_column_is_filled()
    test_the_exchange_is_asked_once_per_trade()
    test_the_source_toggle_off_means_no_exchange_call_at_all()
    test_the_operators_own_tp1_is_never_overwritten()
    test_a_cleared_tp1_stays_cleared()
    test_no_magnet_means_a_reason_not_an_empty_cell()
    test_the_exchange_silence_is_retried_not_remembered()
    test_the_page_shows_the_magnet_state_separately()
    print('\nУсі тести гарантії авто-SL + походження рівнів пройдено ✅')
