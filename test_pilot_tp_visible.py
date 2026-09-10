"""Тест: ЧОМУ порожні Manual TP-1 / TP-2 — МУСИТЬ БУТИ СКАЗАНО.

Кейс NEOUSDT (09.09, paper SHORT @ $2.1860). У таблиці:
    Manual SL 2.3033 · Manual TP-1 — · Manual TP-2 —
    🎯 Автопілот: 🧲 $2.0000 · 6.8% · 1.59R
Тобто ціль БУЛА зафіксована (магніт $2.0000 попереду входу), R рахувався — а
обидва поля TP лишились порожні, і НІ в 🧾 Лозі, НІ в тултипі не було ЖОДНОГО
слова чому.

⚠️ Арифметика при цьому була ПРАВИЛЬНА: `plan_targets` із цими числами віддає
TP-2 = 2.0000 (1.59R). Отже збій стояв НЕ в розрахунку, а в тому, що жоден із
ЧОТИРЬОХ шляхів «рівня не буде» про себе не говорив:
    1) `pilot_autofill_tp` ВИМКНЕНО            → `tp_off`
    2) рівень ЗНЯВ оператор (`pilot_tp_cleared`) → `tp_locked`
    3) ЗБІЙ розрахунку (був лише `print` у stdout) → `tp_err` + рядок у 🧾 Лозі
    4) пілоту нема де взяти рівень               → `tp_skip` (це вже працювало)

Плюс окремо: новий kwarg `ladder=` мусить мати фолбек на СТАРІШИЙ
`trade_pilot` — інакше розбіжність версій файлів гасила TP повністю й МОВЧКИ.
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


_LOG = []


def _install_stubs():
    lg = types.ModuleType('detection.activity_log')
    lg.log_activity = lambda sym, kind, text, **kw: _LOG.append((kind, text))
    sys.modules['detection.activity_log'] = lg

    for nm in ('detection.fuel_filter', 'detection.smc_scanner',
               'detection.market_data', 'detection.volumized_ob'):
        sys.modules.setdefault(nm, types.ModuleType(nm))
    sys.modules['detection.fuel_filter'].get_fuel_filter = lambda: None
    sys.modules['detection.smc_scanner'].get_smc_scanner = lambda: None

    db = types.ModuleType('storage.db_operations')
    db.get_db = lambda: None
    st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
    sys.modules.setdefault('storage', st)
    sys.modules['storage.db_operations'] = db


_load('detection.signal_labels', 'detection/signal_labels.py')
_install_stubs()
tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
tp = _load('detection.trade_pilot', 'detection/trade_pilot.py')
_pkg.trade_pilot = tp
TM = tmmod.TradeManager


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ── РІВНО числа зі скріна NEOUSDT ──────────────────────────────────────────
ENTRY, PRICE, STOP = 2.1860, 2.1460, 2.3033
MAGNET = {'price': 2.0000, 'label': 'магніт $2.00', 'kind': 'magnet',
          'dist_pct': 6.8}
# Дилінг-діапазон навколо ціни: дає пілоту власні обʼєкти (Strong/Weak).
SWING = {'high': {'price': 2.30, 'label': 'Strong High'},
         'low': {'price': 2.05, 'label': 'Strong Low'}, 'tf': '1h'}


def _tm(**settings):
    o = TM.__new__(TM)
    o._settings = {'pilot_enabled': True, 'pilot_autofill_tp': True,
                   'pilot_tp2_from_magnet': False, 'pilot_tp1_close_pct': 50,
                   # Драбину в цих тестах не питаємо — перевіряємо саме
                   # прозорість, а не вибір рівня.
                   'pilot_tp1_from_liquidity': False}
    o._settings.update(settings)
    o._pilot_at = {}
    o._pilot_state = {}
    import threading
    o._lock = threading.RLock()
    o._positions = {}
    o._shadow_positions = {}
    o._manual_hist = {}
    o._pilot_context = lambda sym, side: {
        'swing': SWING, 'runway': None, 'poc': None, 'vah': None, 'val': None,
        'swing_tf': '1h', 'poc_hours': 72}
    o._liq_ladder_levels = lambda sym, side: []
    o._persist_shadow_positions = lambda: None
    o._persist_positions = lambda: None
    return o


def _pos(**over):
    p = {'side': 'SHORT', 'entry_price': ENTRY, 'manual_sl': STOP,
         'pilot_objective': dict(MAGNET)}
    p.update(over)
    return p


def _snap(o):
    return o._pilot_state.get('NEOUSDT|S') or {}


def _text():
    return ' || '.join(t for _k, t in _LOG)


# ═══════════════════════════════ ТЕСТИ ══════════════════════════════════════
def test_the_numbers_from_the_screenshot_do_produce_a_tp2():
    """ЗАМОК НА ДІАГНОЗ: із числами NEOUSDT рівень TP-2 рахується, і це 2.0000
    при 1.59R. Тобто порожні поля НЕ могли бути «правильним результатом
    розрахунку» — отже шукати треба було шлях, що до розрахунку не дійшов."""
    r = tp.plan_targets('SHORT', ENTRY, PRICE, [{'price': 2.05, 'kind': 'swing',
                                                 'label': 'Strong Low'}],
                        objective=dict(MAGNET), stop=STOP)
    _check(r['tp2'] and abs(r['tp2']['price'] - 2.0) < 1e-9,
           f"TP-2 мав порахуватись як 2.0000, отримано {r['tp2']}")
    _check(r['tp2']['r'] == 1.59, f"R мав бути 1.59 (як у колонці), got {r['tp2']['r']}")


def test_autofill_off_is_visible_in_the_snapshot():
    """1) Тумблер вимкнено — стан МУСИТЬ бути в знімку (тултип колонки),
    інакше порожні поля нічим не пояснені."""
    _LOG.clear()
    o = _tm(pilot_autofill_tp=False)
    p = _pos()
    o._pilot_tick('NEOUSDT', p, PRICE, True)
    s = _snap(o)
    _check(s.get('tp_off') is True, f"tp_off мусить бути True, знімок: {s}")
    _check(not p.get('manual_tp') and not p.get('manual_tp1'),
           'при вимкненому автозаповненні рівні ставити НЕ можна')


def test_operator_cleared_level_is_visible_not_vanished():
    """2) Рівень зняв ОПЕРАТОР → автопілот його не відновлює (правило лишається),
    але це мусить бути ВИДНО, а не виглядати як «поля просто зникли»."""
    _LOG.clear()
    o = _tm()
    p = _pos(pilot_tp_set=True, pilot_tp_cleared=True)
    o._pilot_tick('NEOUSDT', p, PRICE, True)
    s = _snap(o)
    _check(s.get('tp_locked') is True, f"tp_locked мусить бути True, знімок: {s}")
    _check(not p.get('manual_tp'), 'знятий оператором рівень НЕ відновлюємо')


def test_crash_goes_to_the_activity_log_not_only_stdout():
    """3) ГОЛОВНЕ. Збій автозаповнення раніше писався ЛИШЕ `print`-ом у stdout —
    на проді його не видно, тож порожні поля читались як «бот не працює»."""
    _LOG.clear()
    o = _tm()
    _orig = tp.plan_targets

    def _boom(*a, **k):
        raise ValueError('навмисний збій')

    tp.plan_targets = _boom
    try:
        p = _pos()
        o._pilot_tick('NEOUSDT', p, PRICE, True)
    finally:
        tp.plan_targets = _orig
    _check('збій автозаповнення TP' in _text(),
           f'збій МУСИТЬ бути в 🧾 Лозі, отримано: {_text()}')
    _check('навмисний збій' in _text(), f'причина мусить бути названа: {_text()}')
    s = _snap(o)
    _check('навмисний збій' in (s.get('tp_err') or ''),
           f"tp_err мусить їхати в знімок для тултипа: {s.get('tp_err')!r}")


def test_same_crash_is_logged_once_not_every_tick():
    """Анти-флуд: такт 20с → за годину було б 180 однакових рядків."""
    _LOG.clear()
    o = _tm()
    _orig = tp.plan_targets
    tp.plan_targets = lambda *a, **k: (_ for _ in ()).throw(ValueError('той самий'))
    try:
        p = _pos()
        for _ in range(5):
            o._pilot_at.clear()          # знімаємо тротл, імітуємо 5 тіків
            o._pilot_tick('NEOUSDT', p, PRICE, True)
    finally:
        tp.plan_targets = _orig
    n = sum(1 for _k, t in _LOG if 'збій автозаповнення TP' in t)
    _check(n == 1, f'однакова причина мусить писатись ОДИН раз, написано {n}')


def test_recovery_clears_the_error_mark():
    """Збій минув → позначка знімається, інакше ⚠️ світилось би вічно."""
    _LOG.clear()
    o = _tm()
    _orig = tp.plan_targets
    tp.plan_targets = lambda *a, **k: (_ for _ in ()).throw(ValueError('тимчасово'))
    p = _pos()
    try:
        o._pilot_tick('NEOUSDT', p, PRICE, True)
    finally:
        tp.plan_targets = _orig
    _check(p.get('pilot_tp_err'), 'позначка збою мала зʼявитись')
    o._pilot_at.clear()
    o._pilot_tick('NEOUSDT', p, PRICE, True)
    _check(not p.get('pilot_tp_err'),
           'після успішного розрахунку позначку збою треба зняти')
    _check(not _snap(o).get('tp_err'), 'у знімку збою теж бути не має')


def test_older_trade_pilot_without_ladder_kwarg_still_sets_levels():
    """⚠️ ФОЛБЕК НА СТАРІШИЙ МОДУЛЬ. Файли розпаковуються/деплояться в РІЗНОМУ
    порядку. Новий kwarg `ladder=` у ще не оновленому `trade_pilot` кидав
    TypeError на КОЖНОМУ такті — і обидва TP мовчки не ставились."""
    _LOG.clear()
    o = _tm(pilot_tp1_from_liquidity=True)
    _orig = tp.plan_targets
    seen = {'with_ladder': 0, 'without': 0}

    def _old(side, entry, price, targets, *, objective=None, stop=None, cfg=None):
        seen['without'] += 1
        return _orig(side, entry, price, targets,
                     objective=objective, stop=stop, cfg=cfg)

    def _probe(*a, **k):
        if 'ladder' in k:
            seen['with_ladder'] += 1
        return _old(*a, **{x: y for x, y in k.items() if x != 'ladder'}) \
            if 'ladder' not in k else _old(*a, **k)

    tp.plan_targets = _probe
    try:
        p = _pos()
        o._pilot_tick('NEOUSDT', p, PRICE, True)
    finally:
        tp.plan_targets = _orig
    _check(seen['with_ladder'] == 1, 'спершу пробуємо НОВИЙ виклик із ladder=')
    _check(seen['without'] == 1,
           'після TypeError мусить бути ПОВТОР без ladder= (фолбек)')
    _check(p.get('manual_tp1') or p.get('pilot_tp_set'),
           f'рівні мали виставитись через фолбек, позиція: {p}')
    _check('збій автозаповнення TP' not in _text(),
           f'фолбек — не збій, у лог його писати не треба: {_text()}')


def test_ui_tooltip_mirrors_all_four_states():
    """Бекенд дає чотири поля — сторінка МУСИТЬ їх показати, інакше знімок
    нікому не допоможе (той самий урок, що з `verdict.parts`)."""
    html = open(os.path.join(_ROOT, 'templates', 'smart_money.html'),
                encoding='utf-8').read()
    i = html.find('function pilotCellHTML(')
    _check(i > 0, 'pilotCellHTML не знайдено')
    body = html[i:i + 4500]
    for f in ('tp_err', 'tp_off', 'tp_locked', 'tp_skip'):
        _check(f'pl.{f}' in body, f'у тултипі немає стану {f}')
    _check('#f87171' in body,
           'ЗБІЙ мусить відрізнятись кольором від свідомого рішення')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f'  ok  {name}')
        except Exception as e:
            bad += 1; print(f'  FAIL {name}: {e}')
    print(f'\n{len(fns) - bad}/{len(fns)} passed')
    sys.exit(1 if bad else 0)
