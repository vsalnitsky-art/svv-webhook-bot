"""💧 НОВИЙ VOB + 🧮 БАНЕР + ТАБЛИЦЯ 💧 СКАНЕРА → СИГНАЛ (вимога 19.09).

**Вимога користувача, дослівно:** «Якщо увімкнено 📦 Volumized OB Trend і
увімкнено "Сканер ліквідності", то кожен новоутворений VOB звіряємо з таблицею
"Сканер ліквідності", і якщо співпадає напрямок VOB + напрямок банера
"МММ-монітор" і є така ж монета у таблиці "Сканер ліквідності", одразу маємо
сигнал і відправляємо монету далі по алгоритму в Чергу (якщо активно) або
відразу відкриваємо угоду, перед цим провіривши чи не існує вже відкрита така
угода.»

Що стережуть ці тести:
  • ТРИ умови мусять зійтись РАЗОМ (напрямок VOB = напрямок банера, монета в
    таблиці, і рядок таблиці зібрано в ТОЙ САМИЙ бік);
  • сигнал іде ЛИШЕ через спільні ворота `_signal_allowed` → `tm.on_signal`
    (урок ASTERUSDT: жоден шлях не відкриває угоду в обхід фільтрів);
  • «чи не існує вже відкрита така угода» — перевірка ПЕРЕД відправкою;
  • «новоутворений» = свіжий і ще НЕ опрацьований блок: старий блок дає тиху
    базу, той самий блок не фаєрить двічі;
  • ⚠️ РОЗБІГ НЕ КОВТАЄ БЛОК: немає збігу ЗАРАЗ (монети ще немає в таблиці,
    угода відкрита) → блок лишається неопрацьованим і спробує ще, поки свіжий;
  • вимкнений шлях/сканер блоки НЕ «зʼїдає»;
  • один блок не дає ДВА сигнали, коли увімкнений ще й 🟪 VOB-алерт.
"""
import ast
import importlib
import importlib.util
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

# Порожній пакет `detection`: справжній `__init__.py` тягне півпроєкту (pybit).
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


_lh = _load('detection.liq_hunter', 'liq_hunter.py')
_pkg.liq_hunter = _lh

_spec = importlib.util.spec_from_file_location(
    'smc_scanner_liqvob_test', os.path.join(_HERE, 'detection', 'smc_scanner.py'))
_sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sc)
S = _sc.SMCScanner

labels = _load('detection.signal_labels', 'signal_labels.py')

_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
            encoding='utf-8').read()
_LH_SRC = open(os.path.join(_HERE, 'detection', 'liq_hunter.py'),
               encoding='utf-8').read()
_FF_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'),
               encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
             encoding='utf-8').read()
_APPJS = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


def _fn_src(name):
    """Тіло методу сканера БЕЗ докстрінга — щоб замки не спрацьовували на
    власних поясненнях (задокументована пастка `ensure_fresh`)."""
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                     and isinstance(node.body[0].value, ast.Constant)
                                     ) else node.body
            return '\n'.join(ast.unparse(b) for b in body)
    raise AssertionError(f'метод {name} зник із smc_scanner')


# ── оточення ──────────────────────────────────────────────────────────────
_LOGGED, _OPENED = [], []


def _install_log():
    _LOGGED.clear()
    mod = types.ModuleType('detection.activity_log')

    def _log(symbol, event, detail='', side=None, source='', extra=None):
        _LOGGED.append({'symbol': symbol, 'event': event, 'detail': detail,
                        'side': side, 'source': source})
    mod.log_activity = _log
    sys.modules['detection.activity_log'] = mod
    _pkg.activity_log = mod


def _install_tm(open_syms=()):
    _OPENED.clear()
    mod = types.ModuleType('detection.trade_manager')
    tm = types.SimpleNamespace(
        _open=set(x.upper() for x in open_syms),
        on_signal=lambda **kw: _OPENED.append(kw) or {'status': 'queued'})
    tm.has_open_position = lambda s: (s or '').upper() in tm._open
    mod.get_trade_manager = lambda: tm
    sys.modules['detection.trade_manager'] = mod
    _pkg.trade_manager = mod
    return tm


class _DB:
    def __init__(self):
        self.d = {}
        self.reads = 0

    def get_setting(self, k, dflt=None):
        self.reads += 1
        return self.d.get(k, dflt)

    def set_setting(self, k, v):
        self.d[k] = v


class _FF:
    def __init__(self, direction='LONG'):
        self.direction = direction

    def mm_bias(self):
        return {'dir': self.direction}

    def symbols_in_work(self):
        return set()


def _hunter(rows=(), direction='LONG', db=None, **settings):
    """💧 Сканер із ГОТОВОЮ таблицею (скан не запускаємо — нас цікавить звірка).
    Кладемо його в синглтон модуля, бо саме звідти його бере сканер."""
    h = _lh.LiqHunterDaemon(db or _DB(), get_watchlist=lambda: [],
                            scan_fn=lambda s: {'ok': True, 'rows': []},
                            get_fuel_filter=lambda: _FF(direction))
    h.start = lambda: None          # фоновий цикл у тестах не піднімаємо
    base = {'enabled': True}
    base.update(settings)
    h.update_settings(base)
    h._rows = [dict(r) for r in rows]
    _lh._instance = h
    return h


def _table_row(sym='AAAUSDT', side='LONG', mass=72.0):
    return {'symbol': sym, 'side': side, 'mass_pct': mass,
            'mass_dir': 'up' if side == 'LONG' else 'down',
            'magnet_price': '$1.20000–1.25000', 'magnet_pct': '14.2%',
            'magnet_dist_pct': 5.58, 'magnet_dir': 'up', 'since': time.time()}


NOW = 1_700_000_000
BAR = 300          # 5m


def _vol(side='LONG', bars_old=2, breaker=False):
    """`vol_result` у формі, яку віддає `get_latest_ob_trend`: РІВНО один
    намальований на графіку блок (`latest_ob`) — те саме джерело, що в 🟪
    VOB-алерта."""
    ft = (NOW - bars_old * BAR) * 1000
    return {'latest_ob': {'type': 'Bull' if side == 'LONG' else 'Bear',
                          'formation_time': ft, 'top': 1.05, 'bottom': 1.0,
                          'breaker': breaker}}, ft


def _klines(bars_old=2):
    """Бари так, щоб вік блоку дорівнював `bars_old` (рахується як кількість
    барів із `t` > formation_time)."""
    return [{'t': (NOW - (bars_old - 1 - i) * BAR) * 1000}
            for i in range(bars_old)]


def _mk(allowed=True, price=1.02):
    """Мінімальний сканер: лише те, чого торкається шлях «VOB + таблиця»."""
    ns = types.SimpleNamespace()
    ns._settings = {'use_volumized_ob': True, 'volumized_swing_length': 5,
                    'vob_alert_max_age_bars': 0, 'volumized_timeframe': '5m'}
    ns._liqvob_seen, ns._liqvob_diag, ns._vob_fired_ft = {}, {}, {}
    ns._errors = 0
    ns._gate_calls = []

    def _gate(symbol, side, at_intake=False):
        ns._gate_calls.append((symbol, side, at_intake))
        return (allowed, 'OB-фільтр заблокував: 1H-блок BEARISH ПРОТИ LONG',
                'OB(1h):✓ · PD:✓')
    ns._signal_allowed = _gate
    ns._get_live_price = lambda s: price
    for nm in ('_vob_chart_candidates', '_vob_seen_list', '_vob_seen_add',
               '_vob_outcome', '_vob_age_bars', '_vob_age_label'):
        setattr(ns, nm, getattr(S, nm))
    ns._liq_vob_check = S._liq_vob_check.__get__(ns)
    return ns


def _run(ns, sym='AAAUSDT', side='LONG', bars_old=2, breaker=False):
    vr, ft = _vol(side, bars_old, breaker)
    ns._liq_vob_check(sym, vr, '5m', _klines(bars_old))
    return ft


# ═══════════ 1. ЧИСТЕ ПРАВИЛО: ТРИ УМОВИ РАЗОМ ═══════════════════════════
def test_the_three_conditions_must_hold_together():
    """ГОЛОВНИЙ ЗАМОК вимоги: напрямок VOB = напрямок банера І монета в
    таблиці. Жодної з умов не досить самої по собі."""
    row = _table_row(side='LONG')
    ok, note = _lh.vob_confluence(row, 'LONG', 'LONG')
    _check(ok, f'усі три умови зійшлись — мусить бути сигнал: {note}')
    _check('LONG' in note, 'причина мусить називати бік')

    ok, note = _lh.vob_confluence(row, 'SHORT', 'LONG')
    _check(not ok and 'ПРОТИ' in note, f'VOB проти банера → не сигнал: {note}')

    ok, note = _lh.vob_confluence(None, 'LONG', 'LONG')
    _check(not ok and 'таблиц' in note.lower(),
           f'монети немає в таблиці → не сигнал: {note}')

    ok, note = _lh.vob_confluence(row, 'LONG', None)
    _check(not ok and '⚖' in note, f'банер без напрямку → не сигнал: {note}')
    print('✓ три умови працюють ЛИШЕ разом')


def test_a_stale_table_row_is_not_a_match():
    """Одразу після розвороту банера таблиця ще стара: її рядки зібрані під
    ПРОТИЛЕЖНИЙ бік. Узяти такий рядок = назвати «збігом» пряме протиріччя."""
    ok, note = _lh.vob_confluence(_table_row(side='SHORT'), 'LONG', 'LONG')
    _check(not ok, 'рядок під інший бік — це НЕ збіг')
    _check('перескан' in note, f'причина мусить пояснити, чого чекаємо: {note}')
    print('✓ застарілий рядок таблиці збігом не вважається')


def test_the_path_has_its_own_switch_defaulting_on():
    """Умову ввімкнення задали САМІ наявні тумблери (📦 VOB Trend + 💧 Сканер),
    тож дефолт OFF означав би, що вимога не працює, доки тумблер не знайдуть.
    Потік угод це не розширює: сам сканер дефолтом ВИМКНЕНИЙ."""
    _check(_lh.DEFAULTS['vob_signal_on'] is True, 'дефолт шляху — УВІМКНЕНО')
    _check(_lh.DEFAULTS['enabled'] is False,
           'сам 💧 Сканер лишається вимкненим за замовчуванням')
    h = _hunter([_table_row()], enabled=False)
    _check(h.vob_signal_on() is False, 'вимкнений сканер → шлях не працює')
    h = _hunter([_table_row()], vob_signal_on=False)
    _check(h.vob_signal_on() is False, 'окремий тумблер вимикає лише сигнали')
    print('✓ тумблер шляху: дефолт ON, але сканер дефолтом OFF')


# ═══════════ 2. СКАНЕР: ЗБІГ → СИГНАЛ ДАЛІ ПО АЛГОРИТМУ ══════════════════
def test_the_scanner_fires_when_vob_banner_and_table_agree():
    _install_log(); _install_tm()
    _hunter([_table_row(sym='AAAUSDT', side='LONG')], direction='LONG')
    ns = _mk()
    _run(ns)
    _check(len(_OPENED) == 1, f'мусив піти РІВНО один сигнал: {_OPENED}')
    _check(_OPENED[0]['opened_by'] == 'liq_vob',
           f"код сигналу мусить бути власний: {_OPENED[0]['opened_by']}")
    _check(_OPENED[0]['side'] == 'LONG', 'бік — той самий, що у VOB і банера')
    _sig = [x for x in _LOGGED if x['event'] == 'signal']
    _check(_sig and 'Сканер ліквідності' in _sig[0]['detail'],
           f'у 🧾 Лог мусить піти рядок сигналу: {_LOGGED}')
    print('✓ збіг VOB + банер + таблиця → сигнал із кодом liq_vob')


def test_the_signal_goes_through_the_shared_gates():
    """Урок ASTERUSDT: жоден шлях не кличе `on_signal` повз `_signal_allowed`."""
    src = _fn_src('_liq_vob_check')
    _check('_signal_allowed' in src, 'ворота мусять бути у шляху')
    _check(src.index('_signal_allowed') < src.index('on_signal'),
           'ворота мусять стояти ДО on_signal')
    _install_log(); _install_tm()
    _hunter([_table_row()], direction='LONG')
    ns = _mk(allowed=False)
    _run(ns)
    _check(not _OPENED, 'ворота не пропустили — угоди бути не може')
    _check([x for x in _LOGGED if x['event'] == 'rejected'],
           'відмова мусить бути видимою в 🧾 Лозі')
    print('✓ сигнал іде лише через спільні ворота фільтрів')


def test_an_open_trade_blocks_the_signal():
    """Дослівна вимога: «перед цим провіривши чи не існує вже відкрита така
    угода». І блок при цьому НЕ «згоряє» — угода може закритись, поки він
    ще свіжий."""
    _install_log()
    tm = _install_tm(open_syms=['AAAUSDT'])
    _hunter([_table_row()], direction='LONG')
    ns = _mk()
    _run(ns)
    _check(not _OPENED, 'угода вже відкрита → сигналу не шлемо')
    _check(ns._liqvob_diag['AAAUSDT']['state'] == 'in_trade',
           'причина мусить бути названа')
    tm._open.clear()
    _run(ns)
    _check(len(_OPENED) == 1,
           'угоду закрито — ТОЙ САМИЙ свіжий блок мусить спрацювати')
    print('✓ відкрита угода блокує сигнал і НЕ зʼїдає блок')


def test_a_coin_outside_the_table_waits_instead_of_burning_the_block():
    """Таблиця оновлюється раз на 15 хв. Якщо монети в ній ЩЕ немає, блок не
    опрацьовуємо — інакше реальний збіг за хвилину вже нічого не дав би."""
    _install_log(); _install_tm()
    _hunter([_table_row(sym='BBBUSDT')], direction='LONG')   # іншої монети
    ns = _mk()
    _run(ns, sym='AAAUSDT')
    _check(not _OPENED, 'монети немає в таблиці → сигналу немає')
    _check(ns._liqvob_diag['AAAUSDT']['state'] == 'wait', 'стан — «чекаємо»')
    _hunter([_table_row(sym='AAAUSDT')], direction='LONG')   # скан додав монету
    _run(ns, sym='AAAUSDT')
    _check(len(_OPENED) == 1, 'монета зʼявилась у таблиці → блок спрацював')
    print('✓ «поки немає збігу» не спалює блок')


def test_the_same_block_fires_only_once():
    _install_log(); _install_tm()
    _hunter([_table_row()], direction='LONG')
    ns = _mk()
    _run(ns); _run(ns); _run(ns)
    _check(len(_OPENED) == 1, f'один блок = один сигнал: {len(_OPENED)}')
    print('✓ той самий блок не фаєрить двічі')


def test_an_old_block_is_a_silent_base_not_a_signal():
    """«Новоутворений» — це момент появи. Блок, старший за вікно свіжості
    (`swing_length + 2` = 7 барів), ми лише БАЧИМО на графіку."""
    _install_log(); _install_tm()
    _hunter([_table_row()], direction='LONG')
    ns = _mk()
    _run(ns, bars_old=40)
    _check(not _OPENED, 'старий блок сигналом не є')
    _check(ns._liqvob_diag['AAAUSDT']['state'] in ('first_sight', 'stale'),
           'старий блок → тиха база')
    _check(not [x for x in _LOGGED if x['event'] in ('signal', 'rejected')],
           'тиха база у 🧾 Лог не пишеться (це СТАН, а не подія)')
    print('✓ старий блок = тиха база, а не сигнал')


def test_a_breaker_block_is_never_a_signal():
    _install_log(); _install_tm()
    _hunter([_table_row()], direction='LONG')
    ns = _mk()
    _run(ns, breaker=True)
    _check(not _OPENED, 'breaker — знецінена зона, входу по ній немає')
    print('✓ breaker сигналу не дає')


def test_the_disabled_path_does_not_eat_blocks():
    """Вимкнений шлях мусить бути НЕВИДИМИМ для бази: інакше після вмикання
    перший же реальний блок виглядав би вже опрацьованим."""
    _install_log(); _install_tm()
    _hunter([_table_row()], direction='LONG', vob_signal_on=False)
    ns = _mk()
    _run(ns)
    _check(not _OPENED, 'шлях вимкнено — сигналу немає')
    _check(not ns._liqvob_seen.get('AAAUSDT'), 'база НЕ має бути зачеплена')
    _hunter([_table_row()], direction='LONG')      # увімкнули
    _run(ns)
    _check(len(_OPENED) == 1, 'після вмикання той самий блок мусить спрацювати')
    print('✓ вимкнений шлях блоків не зʼїдає')


def test_one_block_never_gives_two_signals():
    """Коли увімкнений ще й 🟪 VOB-алерт, ОДИН блок міг би дати ДВА сигнали по
    тій самій монеті. Другий не робимо — у черзі це був би той самий запис."""
    _install_log(); _install_tm()
    _hunter([_table_row()], direction='LONG')
    ns = _mk()
    vr, ft = _vol('LONG', 2)
    ns._vob_fired_ft['AAAUSDT'] = ft          # блок щойно пішов як 🟪 алерт
    ns._liq_vob_check('AAAUSDT', vr, '5m', _klines(2))
    _check(not _OPENED, 'дубля сигналу по одному блоку бути не може')
    _check(ns._liqvob_diag['AAAUSDT']['state'] == 'dup', 'причина названа')
    # А позначку ставить САМЕ шлях VOB-алерта — інакше гейт був би мертвий.
    _check(_SRC.count('_vob_fired_ft[symbol] = _ft') == 2,
           'обидві гілки 🟪 VOB-алерта мусять позначати блок відпрацьованим')
    print('✓ один блок — один сигнал, навіть із двома увімкненими шляхами')


def test_the_banner_direction_decides_the_side():
    """Банер 🔴 SHORT + бичачий VOB = розбіжність, а не сигнал."""
    _install_log(); _install_tm()
    _hunter([_table_row(side='SHORT')], direction='SHORT')
    ns = _mk()
    _run(ns, side='LONG')
    _check(not _OPENED, 'VOB проти банера → сигналу немає')
    _run(ns, side='SHORT')
    _check(len(_OPENED) == 1 and _OPENED[0]['side'] == 'SHORT',
           'VOB у бік банера → сигнал у той самий бік')
    print('✓ бік сигналу задає збіг VOB і банера')


# ═══════════ 3. ЗАМКИ НА ІНТЕГРАЦІЮ ══════════════════════════════════════
def test_settings_are_not_read_from_the_db_for_every_coin():
    """Шлях питає `vob_signal_on()` по КОЖНІЙ монеті КОЖНОГО циклу. Без кешу
    це сотні сесій SQLAlchemy на цикл — рівно та вада, через яку колись
    кешували `get_mm_settings`."""
    db = _DB()
    h = _hunter([_table_row()], db=db)
    before = db.reads
    for _ in range(200):
        h.vob_signal_on()
    _check(db.reads - before <= 2,
           f'налаштування мусять братись із кешу: {db.reads - before} читань')
    # ⚠️ Але зміна з UI мусить діяти ОДРАЗУ, а не через TTL.
    h.update_settings({'vob_signal_on': False})
    _check(h.vob_signal_on() is False, 'зміна налаштування діє негайно')
    print('✓ налаштування кешуються, але зміна діє миттєво')


def test_the_call_site_is_gated_by_the_volumized_block():
    """Друга половина умови користувача: «якщо увімкнено 📦 Volumized OB
    Trend». Вимкнений блок = VOB не рахується взагалі."""
    i = _SRC.index('self._liq_vob_check(')
    head = _SRC[max(0, i - 400):i]
    _check("use_volumized_ob" in head,
           'виклик мусить стояти під перевіркою 📦 Volumized OB Trend')
    print('✓ шлях працює лише при увімкненому 📦 Volumized OB Trend')


def test_the_hunter_still_opens_nothing_itself():
    """💧 Сканер лишається СПОСТЕРЕЖЕННЯМ: він тільки ВІДПОВІДАЄ про таблицю.
    Сигнал шле сканер — там, де VOB і виник."""
    tree = ast.parse(_LH_SRC)
    bad = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
           and n.attr in ('_open', 'on_signal', 'intercept')]
    _check(not bad, f'💧 Сканер не має відкривати угоди сам: {bad}')
    print('✓ сканер угод не відкриває — лише відповідає про таблицю')


def test_the_new_signal_code_is_labelled_everywhere():
    """Мітка мусить бути в Python і в ОБОХ JS-дзеркалах, інакше в таблицях
    стоятиме сирий код (задокументована вимога «скрізь означає скрізь»)."""
    _check('liq_vob' in labels.SIGNAL_BADGES, 'немає бейджа в signal_labels')
    _check('liq_vob' in labels.SIGNAL_ICONS, 'немає картинки в signal_labels')
    for src, name in ((_HTML, 'smart_money.html'), (_APPJS, 'infosite/app.js')):
        _check("liq_vob" in src, f'немає дзеркала мітки у {name}')
    _check("'liq_vob'" in _FF_SRC,
           'черга мусить знати назву сигналу (_kind_lbl у intercept)')
    _check(labels.icon_of('liq_vob → Q4') == '💧',
           'картинка сигналу мусить лишатись 💧 після проходу чергою')
    print('✓ мітка й картинка є скрізь')


def test_the_panel_shows_the_switch_and_the_signals():
    """Шлях мовчить, поки не зʼявиться новий VOB, — без рядка видимості це
    читалось би як «не працює» (урок «де сигнали?»)."""
    _check('id="lh-vob-signal"' in _HTML, 'немає тумблера шляху в панелі 💧')
    _check('vob_signal_on:' in _HTML, 'тумблер не зберігається')
    _check('id="lh-sig"' in _HTML and '_lhSignalsRender' in _HTML,
           'немає рядка видимості сигналів')
    _check('signals' in _LH_SRC and 'signal_count' in _LH_SRC,
           'бекенд мусить віддавати останні рішення шляху')
    print('✓ панель 💧 показує тумблер і останні сигнали')


if __name__ == '__main__':
    _fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
