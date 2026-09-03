"""🧲 TP-2 = МАГНІТ ЛІКВІДНОСТІ · TP-1 = РІВЕНЬ АВТОПІЛОТА · Черга-4 без 💧.

Вимога користувача дослівно: «💧 Фільтр ліквідності не будемо використовувати
в Черга-4, щоб також зменшити навантаження на біржу, а використаємо 💧 Фільтр
ліквідності для визначення Manual TP-2 при відкритті угоди, тобто ставимо туди
ціну із "🧲 найбільший магніт", а Manual TP-1 ставимо ціну яку визначить
🎯 Автопілот.»

⚠️ КЛЮЧОВЕ РІШЕННЯ, яке цей тест стереже: **магніт стає ЦІЛЛЮ УГОДИ**
(`pilot_objective`), а не просто рівнем TP-2. Якби ми поставили тільки рівень,
автопілот закрив би позицію на СВОЇЙ (ближчій) цілі через `action='take'` —
і TP-2 не спрацював би НІКОЛИ, тобто був би декоративним. Ставши ціллю, магніт
узгоджує все одразу: колонку «🎯 Автопілот», R, автозакриття і Manual TP-2.

⚠️ БЛИЖНЯ МЕЖА, а не підпис. Магніт — це СМУГА `[price..price_hi]`. LONG
зустрічає НИЖНЮ межу, SHORT — ВЕРХНЮ. Виходимо на першому дотику до кластера,
а не сподіваємось, що ціна прошиє його наскрізь.
"""
import importlib.util
import os
import sys
import types

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
    sys.modules['detection.activity_log'] = lg; _pkg.activity_log = lg
    ff = types.ModuleType('detection.fuel_filter')
    ff.get_fuel_filter = lambda: types.SimpleNamespace(get_settings=lambda: {})
    sys.modules['detection.fuel_filter'] = ff
    st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
    sys.modules.setdefault('storage', st)
    db = types.ModuleType('storage.db_operations')
    db.get_db = lambda: types.SimpleNamespace(get_smc_ob_state=lambda s, tf: None)
    sys.modules['storage.db_operations'] = db


_stubs()

# 🎯 Автопілот — СПРАВЖНІЙ: саме його `plan_targets` має дати TP-1, і саме
# його поведінку «ціль досягнуто → take» ми перевіряємо на магніті.
_REAL = _load('_real_trade_pilot', 'detection/trade_pilot.py')
_FAKE = types.ModuleType('detection.trade_pilot')
for _k in dir(_REAL):
    if not _k.startswith('__'):
        setattr(_FAKE, _k, getattr(_REAL, _k))
sys.modules['detection.trade_pilot'] = _FAKE
_pkg.trade_pilot = _FAKE

# Сканер підміняємо цілком — нам потрібен лише `get_liq_magnet`.
_MAGNET = {'ok': False}
_SC_CALLS = []


def _fake_scanner():
    def _get(sym):
        _SC_CALLS.append(sym)
        return dict(_MAGNET)
    return types.SimpleNamespace(get_liq_magnet=_get, get_settings=lambda: {})


_sc_mod = types.ModuleType('detection.smc_scanner')
_sc_mod.get_smc_scanner = _fake_scanner
sys.modules['detection.smc_scanner'] = _sc_mod
_pkg.smc_scanner = _sc_mod

tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
TM = tmmod.TradeManager
_SCAN_SRC = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py')).read()
_FF_SRC = open(os.path.join(_ROOT, 'detection', 'fuel_filter.py')).read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _magnet(lo, hi, pct='38.0%', ok=True):
    global _MAGNET
    _MAGNET = {'ok': ok, 'pct': pct, 'exchange': 'binance',
               'row': {'price': lo, 'price_hi': hi, 'pct': 38.0,
                       'dist_pct': 3.0, 'dir': 'up'},
               'reason': '' if ok else 'біржа не відповіла'}
    _SC_CALLS.clear()


def _tm(**over):
    o = TM.__new__(TM)
    o._settings = {'pilot_enabled': True, 'pilot_tp2_from_magnet': True,
                   'pilot_autofill_tp': False, 'pilot_tp_min_gap_pct': 0.5}
    o._settings.update(over)
    return o


# ═══════════ 1. ЦІЛЬ = МАГНІТ (головне) ════════════════════════════════════
def test_long_takes_the_lower_edge_short_the_upper():
    """Магніт — СМУГА. Ціна доходить до неї з одного боку: LONG зустрічає
    НИЖНЮ межу, SHORT — ВЕРХНЮ. Брати протилежну = чекати, поки ціна прошиє
    весь кластер наскрізь."""
    _magnet(lo=110.0, hi=112.0)
    o = _tm()
    m = o._magnet_objective('X', 'LONG', 100.0)
    _check(m and m['price'] == 110.0, f'LONG → нижня межа: {m}')
    _check(m['kind'] == 'magnet', m)
    _check(abs(m['dist_pct'] - 10.0) < 0.01, m)

    _magnet(lo=88.0, hi=90.0)
    m2 = o._magnet_objective('X', 'SHORT', 100.0)
    _check(m2 and m2['price'] == 90.0, f'SHORT → верхня межа: {m2}')
    print('✓ LONG бере нижню межу магніту, SHORT — верхню')


def test_magnet_behind_entry_is_rejected():
    """⚠️ Драбина СИМЕТРИЧНА навколо ціни, тож найбільша сходинка легко
    лежить ПОЗАДУ входу. Ціллю вона бути не може."""
    _magnet(lo=90.0, hi=92.0)
    o = _tm()
    _check(o._magnet_objective('X', 'LONG', 100.0) is None,
           'магніт під входом не може бути ціллю LONG')
    _check('ПОЗАДУ' in o._magnet_skip, o._magnet_skip)
    _magnet(lo=110.0, hi=112.0)
    _check(o._magnet_objective('X', 'SHORT', 100.0) is None,
           'магніт над входом не може бути ціллю SHORT')
    print('✓ магніт позаду входу відкидається з поясненням')


def test_too_close_magnet_is_rejected():
    """Рівень упритул до входу зʼїдає комісія — той самий `tp_min_gap_pct`,
    що вже боронить поділ TP-1/TP-2."""
    _magnet(lo=100.2, hi=100.4)
    o = _tm(pilot_tp_min_gap_pct=0.5)
    _check(o._magnet_objective('X', 'LONG', 100.0) is None, 'надто близько')
    _check('зазор' in o._magnet_skip, o._magnet_skip)
    # Той самий магніт при нульовому зазорі — годиться.
    _check(_tm(pilot_tp_min_gap_pct=0)._magnet_objective('X', 'LONG', 100.0),
           'зазор 0 → приймаємо')
    print('✓ надто близький магніт відкидається (комісія)')


def test_no_data_falls_back_silently():
    _magnet(lo=0, hi=0, ok=False)
    o = _tm()
    _check(o._magnet_objective('X', 'LONG', 100.0) is None, 'даних немає')
    _check('не відповіла' in o._magnet_skip, o._magnet_skip)
    print('✓ біржа не відповіла → None (ціль рахує автопілот)')


# ═══════════ 2. МАГНІТ СТАЄ ЦІЛЛЮ УГОДИ, А НЕ ПРОСТО РІВНЕМ ════════════════
def test_magnet_becomes_the_locked_objective():
    """ГОЛОВНИЙ ЗАМОК. Магніт мусить лягти в `pilot_objective` — інакше
    автопілот закриє позицію на СВОЇЙ ближчій цілі, і TP-2 буде декоративним."""
    _magnet(lo=110.0, hi=112.0)
    o = _tm()
    o._pilot_at = {}
    o._pilot_state = {}
    o._pilot_context = lambda sym, side: {'swing': None, 'runway': None,
                                          'poc': None, 'swing_tf': '1h',
                                          'poc_hours': 72}
    o.update_manual_sl_tp = lambda *a, **k: {'ok': True}
    o.PILOT_TTL = 0.0
    pos = {'side': 'LONG', 'entry_price': 100.0}
    o._pilot_tick('X', pos, 101.0, True)
    obj = pos.get('pilot_objective') or {}
    _check(obj.get('kind') == 'magnet', f'ціль мусить бути магнітом: {obj}')
    _check(obj.get('price') == 110.0, obj)
    _check(pos.get('pilot_magnet_done') is True, 'спроба мусить позначитись')
    _check(any('МАГНІТ' in t for _k, t in _LOG), 'у лозі мусить бути рядок')
    print('✓ магніт лягає в ціль угоди (а не лише в рівень TP-2)')


def test_asked_once_per_trade():
    """Біржу питаємо РІВНО ОДИН раз на угоду — не на кожному тіку монітора."""
    _magnet(lo=110.0, hi=112.0)
    o = _tm()
    o._pilot_at = {}; o._pilot_state = {}
    o._pilot_context = lambda sym, side: {'swing': None, 'runway': None,
                                          'poc': None, 'swing_tf': '1h',
                                          'poc_hours': 72}
    o.update_manual_sl_tp = lambda *a, **k: {'ok': True}
    o.PILOT_TTL = 0.0
    pos = {'side': 'LONG', 'entry_price': 100.0}
    for _ in range(5):
        o._pilot_tick('X', pos, 101.0, True)
    _check(len(_SC_CALLS) == 1, f'один запит на угоду, а не {len(_SC_CALLS)}')
    print('✓ магніт питається один раз на угоду')


def test_exchange_down_allows_retry():
    """⚠️ Позначку «спробували» ставимо ЛИШЕ коли біржа відповіла. Інакше
    угода, відкрита під час недоступності, назавжди лишилась би без магніту."""
    _magnet(lo=0, hi=0, ok=False)
    o = _tm()
    o._pilot_at = {}; o._pilot_state = {}
    o._pilot_context = lambda sym, side: {'swing': None, 'runway': None,
                                          'poc': None, 'swing_tf': '1h',
                                          'poc_hours': 72}
    o.update_manual_sl_tp = lambda *a, **k: {'ok': True}
    o.PILOT_TTL = 0.0
    pos = {'side': 'LONG', 'entry_price': 100.0}
    o._pilot_tick('X', pos, 101.0, True)
    _check(not pos.get('pilot_magnet_done'),
           'біржа лежала → мусимо спробувати ще раз пізніше')
    # Біржа піднялась — магніт застосовується.
    _magnet(lo=110.0, hi=112.0)
    pos.pop('pilot_objective', None)
    o._pilot_tick('X', pos, 101.0, True)
    _check((pos.get('pilot_objective') or {}).get('kind') == 'magnet',
           pos.get('pilot_objective'))
    print('✓ біржа лежала → повтор; піднялась → магніт застосовано')


def test_unusable_magnet_is_not_retried():
    """А от коли відповідь БУЛА і магніт просто не годиться (позаду входу) —
    питати ще раз немає сенсу."""
    _magnet(lo=90.0, hi=92.0)
    o = _tm()
    o._pilot_at = {}; o._pilot_state = {}
    o._pilot_context = lambda sym, side: {'swing': None, 'runway': None,
                                          'poc': None, 'swing_tf': '1h',
                                          'poc_hours': 72}
    o.update_manual_sl_tp = lambda *a, **k: {'ok': True}
    o.PILOT_TTL = 0.0
    pos = {'side': 'LONG', 'entry_price': 100.0}
    o._pilot_tick('X', pos, 101.0, True)
    _check(pos.get('pilot_magnet_done') is True, 'відповідь була → не питаємо знову')
    _check(not pos.get('pilot_objective'), 'ціль лишається за автопілотом')
    print('✓ непридатний магніт: одна спроба, далі ціль рахує автопілот')


def test_toggle_off_keeps_old_behaviour():
    _magnet(lo=110.0, hi=112.0)
    o = _tm(pilot_tp2_from_magnet=False)
    o._pilot_at = {}; o._pilot_state = {}
    o._pilot_context = lambda sym, side: {'swing': None, 'runway': None,
                                          'poc': None, 'swing_tf': '1h',
                                          'poc_hours': 72}
    o.update_manual_sl_tp = lambda *a, **k: {'ok': True}
    o.PILOT_TTL = 0.0
    pos = {'side': 'LONG', 'entry_price': 100.0}
    o._pilot_tick('X', pos, 101.0, True)
    _check(_SC_CALLS == [], 'вимкнено → біржу не смикаємо взагалі')
    _check((pos.get('pilot_objective') or {}).get('kind') != 'magnet',
           pos.get('pilot_objective'))
    print('✓ тумблер вимкнено → стара поведінка, без запитів')


# ═══════════ 3. TP-1 ЛИШАЄТЬСЯ ЗА АВТОПІЛОТОМ ══════════════════════════════
def test_tp1_is_still_the_pilots_level_inside_the_magnet_path():
    """TP-2 = магніт, TP-1 = те, що вибрав АВТОПІЛОТ у вікні шляху до нього.
    Перевіряємо на СПРАВЖНІЙ `plan_targets`."""
    tp = _REAL
    targets = [
        {'price': 103.0, 'dist_pct': 3.0, 'kind': 'swing', 'label': 'Weak High'},
        {'price': 106.0, 'dist_pct': 6.0, 'kind': 'liq_next', 'label': 'пул'},
        {'price': 130.0, 'dist_pct': 30.0, 'kind': 'poc', 'label': 'POC'},
    ]
    magnet = {'price': 110.0, 'dist_pct': 10.0, 'kind': 'magnet',
              'label': 'магніт ліквідності 38.0%'}
    r = tp.plan_targets('LONG', 100.0, 101.0, targets,
                        objective=magnet, stop=98.0)
    _check((r.get('tp2') or {}).get('price') == 110.0,
           f"TP-2 мусить бути магнітом: {r.get('tp2')}")
    t1 = r.get('tp1') or {}
    _check(t1, f'TP-1 мусить зʼявитись: {r}')
    _check(100.0 < t1['price'] < 110.0,
           f'TP-1 суворо між входом і магнітом: {t1}')
    _check(t1['kind'] != 'magnet', f'TP-1 — рівень автопілота, не магніт: {t1}')
    print(f"✓ TP-2 = магніт 110, TP-1 = автопілот {t1['price']} ({t1['kind']})")


def test_ui_knows_the_magnet_icon():
    html = open(os.path.join(_ROOT, 'templates', 'smart_money.html')).read()
    i = html.index('const _PILOT_KIND')
    _check("magnet: '🧲'" in html[i:i + 400],
           'колонка автопілота мусить малювати значок магніту')
    print('✓ UI знає значок 🧲 для цілі-магніту')


# ═══════════ 4. ЧЕРГА-4 БІЛЬШЕ НЕ СМИКАЄ БІРЖУ ═════════════════════════════
def test_queue4_recheck_skips_the_liquidity_filter():
    """Вимога: «💧 Фільтр ліквідності не будемо використовувати в Черга-4,
    щоб зменшити навантаження на біржу». Монета вже пройшла його НА ІНТЕЙКУ."""
    _check('skip_liq=True' in _FF_SRC,
           'Q4-recheck мусить просити сканер пропустити фільтр ліквідності')
    _check('skip_liq: bool = False' in _SCAN_SRC,
           'ворота мусять приймати параметр')
    _check('Черга-4 не ' in _SCAN_SRC,
           'у розкладі мусить бути чесно сказано, ЧОМУ фільтр не рахувався')
    # Фолбек на старіший сканер — щоб оновлення файлів у різному порядку
    # не поклало Чергу-4.
    # ⚠️ `rindex`, а не `index`: перше входження — у ПОЯСНЮВАЛЬНОМУ КОМЕНТАРІ,
    # а перевірити треба саме код виклику.
    i = _FF_SRC.rindex('skip_liq=True')
    _check('except TypeError' in _FF_SRC[i:i + 300],
           'потрібен фолбек, якщо сканер ще без параметра')
    print('✓ Черга-4 не перевіряє 💧 фільтр (і не падає на старому сканері)')


def test_intake_still_checks_liquidity():
    """⚠️ Зворотний бік: на ІНТЕЙКУ фільтр лишається — інакше він перестав би
    гейтити вхід узагалі, а це не те, про що просили."""
    _check('skip_liq=True' not in _SCAN_SRC.split('def _send_alert')[-1][:4000]
           if 'def _send_alert' in _SCAN_SRC else True,
           'шлях інтейку не має пропускати фільтр')
    i = _SCAN_SRC.index('def _signal_allowed')
    body = _SCAN_SRC[i:i + 12000]   # тіло воріт довге — вікно з запасом
    _check('if skip_liq:' in body, 'пропуск мусить бути ЯВНИМ і лише за прапорцем')
    print('✓ на інтейку фільтр працює як раніше')


# ═══════════ 5. TP-1 НІКОЛИ НЕ ПЕРЕВИЩУЄ TP-2 ══════════════════════════════
def test_tp1_never_exceeds_tp2_or_stays_empty():
    """Вимога користувача: «Manual TP-1 не має перевищувати Manual TP-2. Якщо
    так трапляється — TP-1 залишити пустим (не вірно визначився Автопілот)».

    Порожній TP-1 — КОРЕКТНИЙ стан (працюємо одним TP-2). А от «частковий»
    вихід ДАЛІ за повний — ні: половина позиції знімалась би після того, як
    угода вже мала закритись цілком."""
    tp = _REAL
    # Магніт БЛИЖЧЕ за всі обʼєкти графіка — TP-1 нема з чого зробити.
    targets = [
        {'price': 130.0, 'dist_pct': 30.0, 'kind': 'liq_next', 'label': 'далекий пул'},
        {'price': 150.0, 'dist_pct': 50.0, 'kind': 'poc', 'label': 'POC'},
    ]
    magnet = {'price': 104.0, 'dist_pct': 4.0, 'kind': 'magnet', 'label': 'магніт'}
    r = tp.plan_targets('LONG', 100.0, 101.0, targets,
                        objective=magnet, stop=98.0,
                        cfg={'tp1_fallback_path_pct': 0})
    _check((r.get('tp2') or {}).get('price') == 104.0, r)
    _check(r.get('tp1') is None,
           f'усі обʼєкти ДАЛІ за TP-2 → TP-1 мусить лишитись порожнім: {r.get("tp1")}')

    # SHORT-дзеркало: «перевищує» = ближче до нуля за TP-2.
    r2 = tp.plan_targets('SHORT', 100.0, 99.0,
                         [{'price': 70.0, 'dist_pct': 30.0, 'kind': 'poc', 'label': 'POC'}],
                         objective={'price': 96.0, 'dist_pct': 4.0,
                                    'kind': 'magnet', 'label': 'магніт'},
                         stop=102.0, cfg={'tp1_fallback_path_pct': 0})
    _check(r2.get('tp1') is None, f'SHORT: те саме правило: {r2.get("tp1")}')
    print('✓ TP-1 за межею TP-2 → поле лишається ПОРОЖНІМ')


def test_final_guard_exists_in_code():
    """Замок стоїть ОКРЕМО в кінці `plan_targets`, а не лише всередині гілок:
    правило надто дороге, щоб його загубила майбутня правка однієї з них."""
    src = open(os.path.join(_ROOT, 'detection', 'trade_pilot.py')).read()
    i = src.index('def plan_targets')
    j = src.index('def plan(', i)
    body = src[i:j]
    k = body.rindex('return {')
    tail = body[:k]
    _check("if tp1 is not None and not _ahead(side, tp1['price'], tp2['price'])" in tail,
           'потрібен фінальний замок ПЕРЕД поверненням')
    print('✓ фінальний замок «TP-1 перед TP-2» стоїть у коді')


if __name__ == '__main__':
    test_long_takes_the_lower_edge_short_the_upper()
    test_magnet_behind_entry_is_rejected()
    test_too_close_magnet_is_rejected()
    test_no_data_falls_back_silently()
    test_magnet_becomes_the_locked_objective()
    test_asked_once_per_trade()
    test_exchange_down_allows_retry()
    test_unusable_magnet_is_not_retried()
    test_toggle_off_keeps_old_behaviour()
    test_tp1_is_still_the_pilots_level_inside_the_magnet_path()
    test_ui_knows_the_magnet_icon()
    test_queue4_recheck_skips_the_liquidity_filter()
    test_intake_still_checks_liquidity()
    test_tp1_never_exceeds_tp2_or_stays_empty()
    test_final_guard_exists_in_code()
    print('\nУсі тести «TP-2 з магніту» пройдено ✅')
