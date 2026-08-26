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
    print('\nУсі тести гарантії авто-SL пройдено ✅')
