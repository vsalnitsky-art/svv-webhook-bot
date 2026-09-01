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
_VOB = {}             # 'bullish_obs' / 'bearish_obs' → [ob]
_SETTINGS = {}


def _install_stubs():
    lg = types.ModuleType('detection.activity_log')
    lg.log_activity = lambda sym, kind, text, **kw: _LOG.append((kind, text))
    sys.modules['detection.activity_log'] = lg

    ff = types.ModuleType('detection.fuel_filter')
    ff.get_fuel_filter = lambda: types.SimpleNamespace(get_settings=lambda: dict(_SETTINGS))
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

    md = types.ModuleType('detection.market_data')
    md.get_market_data = lambda: types.SimpleNamespace(
        fetch_klines=lambda s, limit=200, interval='5m': [{'o': 1}] * 50)
    sys.modules['detection.market_data'] = md

    vo = types.ModuleType('detection.volumized_ob')
    vo.detect_volumized_obs = lambda *a, **k: dict(_VOB)
    sys.modules['detection.volumized_ob'] = vo


_install_stubs()
tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
TM = tmmod.TradeManager


def _tm():
    o = TM.__new__(TM)
    return o


def _reset(**over):
    _LOG.clear(); _OB_ROWS.clear(); _VOB.clear()
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
    стопа. Тепер має підхопитись ★1H-OB (ВЕДМЕЖИЙ), як на графіку."""
    _reset()
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


def test_primary_tf_still_wins_when_valid():
    """Стара поведінка збережена: придатний OB на обраному TF — перший у черзі."""
    _reset()
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5200 * 1.002),
           f'мав узятись 15m (обраний TF), отримано {p["manual_sl"]}')
    print('✓ придатний OB на обраному TF і далі має пріоритет')


def test_volumized_used_when_both_ob_rows_unusable():
    _reset()
    _OB_ROWS['15m'] = {'bias': 'BULLISH', 'bar_high': 0.5150, 'bar_low': 0.5100}
    _OB_ROWS['1h'] = None
    _VOB['bearish_obs'] = [{'top': 0.5250, 'bottom': 0.5200, 'breaker': False}]
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5250 * 1.002),
           f'мав узятись Volumized OB, отримано {p.get("manual_sl")}')
    _check('Volumized OB 5m' in _text(), f'джерело в лозі: {_text()}')
    print('✓ жодного придатного OB-рядка → Volumized OB у бік угоди')


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
    # Ведмежий 15m-блок, але ціна вже ВИЩЕ його верху → стоп нижче ціни.
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5100, 'bar_low': 0.5050}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    p = _pos()
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5285 * 1.002),
           f'мав перейти на ★1H, отримано {p.get("manual_sl")}')
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


# ═════ 🎯 ДЖЕРЕЛО SL МАЄ ВІДПОВІДАТИ НАЛАШТУВАННЮ ЧЕРГИ-4 ══════════════════
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
    _check('Черга-4: 1H OB' in _text(), f'джерело має бути назване в лозі: {_text()}')
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


def test_non_q4_trade_keeps_its_own_tf():
    """Угоди НЕ з Черги-4 і далі йдуть за власним `q2_auto_ob_sl_tf`."""
    _reset(queue4_sl_source='1h', q2_auto_ob_sl_tf='15m')
    _OB_ROWS['15m'] = {'bias': 'BEARISH', 'bar_high': 0.5200, 'bar_low': 0.5150}
    _OB_ROWS['1h'] = {'bias': 'BEARISH', 'bar_high': 0.5285, 'bar_low': 0.5250}
    p = _pos()
    p['opened_by'] = 'choch → Q2'
    _tm()._auto_ob_manual_sl('MNTUSDT', p, 0.51430)
    _check(_near(p['manual_sl'], 0.5200 * 1.002),
           f'не-Q4 угода лишається на своєму TF: {p.get("manual_sl")}')
    print('✓ угоди не з Черги-4 поведінку не змінили')


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


if __name__ == '__main__':
    test_mnt_case_star_1h_block_is_used_instead_of_waiting()
    test_primary_tf_still_wins_when_valid()
    test_volumized_used_when_both_ob_rows_unusable()
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
    test_non_q4_trade_keeps_its_own_tf()
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
    test_partial_close_goes_to_the_group_topic()
    test_partial_close_message_is_one_line()
    test_tp_lines_have_no_labels()
    print('\nУсі тести гарантії авто-SL + походження рівнів пройдено ✅')
