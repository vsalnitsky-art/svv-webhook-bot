"""Тести правил ВИХОДУ з угоди.

1) 🧱 Opposite OB Exit має спрацьовувати на ПОЯВУ НОВОГО протилежного OB, а не
   на блок, що вже висів на графіку до входу.
   Кейс MNTUSDT (25.08): о 23:28:50 авто-SL писав «OB на 15M протилежний
   (BULLISH) — чекаю BEARISH», тобто протилежний блок УЖЕ був на момент
   відкриття шорта. О 23:41 правило закрило угоду «Ціна вдарилась у протилежний
   Order Block» — хоча (а) блок не новий, (б) ціну код узагалі не перевіряв.
2) 🔮 Forecast 1H / 4H і 🧠 Decision Center — три НЕЗАЛЕЖНІ самостійні правила
   виходу (усі дефолт OFF): протилежний вердикт → закриваємо одразу.
"""
import os, sys, types, importlib.util, threading, time

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

_LOG, _OB_ROWS = [], {}

lg = types.ModuleType('detection.activity_log')
lg.log_activity = lambda sym, kind, text, **kw: _LOG.append(text)
sys.modules['detection.activity_log'] = lg
db = types.ModuleType('storage.db_operations')
db.get_db = lambda: types.SimpleNamespace(get_smc_ob_state=lambda sym, tf: _OB_ROWS.get(tf))
st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
sys.modules.setdefault('storage', st)
sys.modules['storage.db_operations'] = db
ffm = types.ModuleType('detection.fuel_filter')
ffm.get_fuel_filter = lambda: None
sys.modules['detection.fuel_filter'] = ffm

tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
TM = tmmod.TradeManager
HOUR_MS = 3600 * 1000


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _TM(TM):
    """Мінімальний TM: справжні правила виходу, підроблені сховища й ціна."""
    def __init__(self, side='SHORT', opened_at=1000.0, shadow=False, **settings):
        self._lock = threading.RLock()
        self._opp_ob_base = {}
        self._signal_exit_at = {}
        self._mm_flat_since = {}
        # 🔄 Витримка розвороту (20.09) — нове поле стану ЗАВЖДИ додавати сюди,
        # інакше правило падає з AttributeError «на рівному місці».
        self._mm_against_since = {}
        self.closed = []
        pos = {'symbol': 'MNTUSDT', 'side': side, 'entry_price': 0.5136,
               'opened_at': opened_at}
        self._positions = {} if shadow else {'MNTUSDT': pos}
        self._shadow_positions = {'MNTUSDT': pos} if shadow else {}
        self._settings = {'use_opposite_ob_exit': True,
                          'opposite_ob_exit_timeframe': '15m',
                          'use_forecast_1h_exit': False,
                          'use_forecast_4h_exit': False,
                          'use_decision_exit': False}
        self._settings.update(settings)
        self._fc = {}
        self._dc = {}
    def _get_current_price(self, symbol): return 0.5147
    def _close_position(self, symbol, price, reason): self.closed.append(('real', reason))
    def _close_shadow(self, symbol, price, reason): self.closed.append(('paper', reason))
    def _get_forecast_both(self, symbol): return dict(self._fc)
    def compute_decision(self, symbol, price): return dict(self._dc)


def _reset(**kw):
    _LOG.clear(); _OB_ROWS.clear()
    return _TM(**kw)


def _ob(bias, bar_time, tag='CHoCH'):
    return {'bias': bias, 'bar_time': bar_time, 'created_by_tag': tag,
            'bar_high': 0.52, 'bar_low': 0.51}


# ═════════════ 1. 🧱 Opposite OB Exit — лише НОВИЙ блок ════════════════════
def test_mnt_case_preexisting_opposite_ob_must_not_close():
    """🐞 ГОЛОВНИЙ КЕЙС: протилежний блок стояв ЩЕ ДО входу → не закривати."""
    t = _reset(side='SHORT', opened_at=2_000_000.0)
    _OB_ROWS['15m'] = _ob('BULLISH', 1_500_000_000)     # блок був на вході
    t.on_main_ob_update('MNTUSDT')                       # 1-й прохід — фіксує базу
    _check(t.closed == [], f'перший прохід лише фіксує базу: {t.closed}')
    t.on_main_ob_update('MNTUSDT')                       # 2-й — той самий блок
    t.on_main_ob_update('MNTUSDT')                       # 3-й — і далі той самий
    _check(t.closed == [],
           f'блок, що висів ДО входу, НЕ має закривати угоду: {t.closed}')
    print('✓ MNTUSDT: наявний протилежний OB більше не вбиває свіжу угоду')


def test_new_opposite_ob_closes():
    t = _reset(side='SHORT')
    _OB_ROWS['15m'] = _ob('BULLISH', 1_500_000_000)
    t.on_main_ob_update('MNTUSDT')                       # база
    _OB_ROWS['15m'] = _ob('BULLISH', 1_500_000_000 + HOUR_MS)   # З'ЯВИВСЯ НОВИЙ
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [('real', 'opposite_ob_exit')],
           f'новий протилежний блок мав закрити угоду: {t.closed}')
    _check(any('НОВИЙ протилежний OB' in x for x in _LOG),
           f'подія має бути в лозі з поясненням: {_LOG}')
    print('✓ НОВИЙ протилежний OB закриває угоду (і це видно в лозі)')


def test_no_ob_at_open_then_opposite_appears_closes():
    """На вході блоку не було зовсім → перший протилежний і є «новим»."""
    t = _reset(side='LONG')
    _OB_ROWS['15m'] = None
    t.on_main_ob_update('MNTUSDT')          # немає рядка → нічого не робимо
    _OB_ROWS['15m'] = _ob('BEARISH', 1_600_000_000)
    t.on_main_ob_update('MNTUSDT')          # база (перший показ блоку)
    _OB_ROWS['15m'] = _ob('BEARISH', 1_600_000_000 + HOUR_MS)
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [('real', 'opposite_ob_exit')], f'мало закритись: {t.closed}')
    print('✓ блоку не було на вході → перший НОВІШИЙ протилежний закриває')


def test_same_side_new_ob_does_not_close_and_moves_baseline():
    """Новий блок У НАШ бік підтримує позицію — і стає новою базою."""
    t = _reset(side='SHORT')
    _OB_ROWS['15m'] = _ob('BEARISH', 1_500_000_000)
    t.on_main_ob_update('MNTUSDT')                                   # база
    _OB_ROWS['15m'] = _ob('BEARISH', 1_500_000_000 + HOUR_MS)        # новий, у бік
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [], f'блок у наш бік не закриває: {t.closed}')
    _check(t._opp_ob_base['MNTUSDT']['bar_time'] == 1_500_000_000 + HOUR_MS,
           'база мала зсунутись на новий блок')
    print('✓ новий блок у бік угоди не закриває і зсуває базу')


def test_older_block_never_closes():
    """OB може «відкотитись» на старіший (поточний став breaker) — це не подія."""
    t = _reset(side='SHORT')
    _OB_ROWS['15m'] = _ob('BEARISH', 1_600_000_000)
    t.on_main_ob_update('MNTUSDT')
    _OB_ROWS['15m'] = _ob('BULLISH', 1_500_000_000)   # СТАРІШИЙ протилежний
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [], f'старіший блок — не «з\'явився»: {t.closed}')
    print('✓ старіший протилежний блок не вважається появою')


def test_new_position_rebaselines():
    """Нова угода по монеті починає з чистої бази, а не тягне стару."""
    t = _reset(side='SHORT', opened_at=1000.0)
    _OB_ROWS['15m'] = _ob('BULLISH', 1_500_000_000)
    t.on_main_ob_update('MNTUSDT')
    t._positions['MNTUSDT'] = {'symbol': 'MNTUSDT', 'side': 'SHORT',
                               'entry_price': 0.5, 'opened_at': 9999.0}
    t.on_main_ob_update('MNTUSDT')      # інша угода → база фіксується заново
    _check(t.closed == [], f'нова угода не має закриватись одразу: {t.closed}')
    print('✓ нова угода перезаписує базу (не успадковує чужу)')


def test_toggle_off_disables_rule():
    t = _reset(side='SHORT', use_opposite_ob_exit=False)
    _OB_ROWS['15m'] = _ob('BULLISH', 1_500_000_000)
    t.on_main_ob_update('MNTUSDT')
    _OB_ROWS['15m'] = _ob('BULLISH', 1_500_000_000 + HOUR_MS)
    t.on_main_ob_update('MNTUSDT')
    _check(t.closed == [], 'вимкнене правило нічого не робить')
    print('✓ тумблер OFF вимикає правило')


def test_reason_text_no_longer_lies_about_price():
    """Код НІКОЛИ не перевіряв ціну — підпис «Ціна вдарилась…» був неправдою."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    _check('Ціна вдарилась у протилежний Order Block' not in src,
           'фальшивий підпис про ціну має бути прибраний')
    _check("З'явився НОВИЙ протилежний Order Block" in src,
           'підпис має описувати те, що код реально перевіряє')
    print('✓ підпис причини відповідає тому, що код перевіряє')


# ═════════════ 2-3. 🔮 Forecast 1H/4H · 🧠 Decision ════════════════════════
def _sig(side='SHORT', **kw):
    t = _reset(side=side, use_opposite_ob_exit=False, **kw)
    t.SIGNAL_EXIT_TTL = 0.0        # без тротлу в тесті
    return t


def test_forecast_1h_opposite_closes():
    t = _sig(use_forecast_1h_exit=True)
    t._fc = {'f1_side': 1, 'f1_conf': 72, 'f4_side': 0, 'f4_conf': 0}   # LONG проти SHORT
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'протилежний Forecast 1H мав закрити угоду')
    _check(t.closed == [('real', 'forecast_1h_exit')], f'{t.closed}')
    _check(any('Forecast 1H LONG' in x for x in _LOG), f'лог: {_LOG}')
    print('✓ Forecast 1H проти позиції → закриття')


def test_forecast_4h_opposite_closes_independently():
    """4H працює НЕЗАЛЕЖНО: 1H вимкнений і нейтральний, 4H — проти."""
    t = _sig(use_forecast_4h_exit=True)
    t._fc = {'f1_side': 0, 'f1_conf': 0, 'f4_side': 1, 'f4_conf': 65}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           '4H мав спрацювати самостійно')
    _check(t.closed == [('real', 'forecast_4h_exit')], f'{t.closed}')
    print('✓ Forecast 4H працює окремо від 1H')


def test_forecast_same_side_keeps_position():
    t = _sig(use_forecast_1h_exit=True, use_forecast_4h_exit=True)
    t._fc = {'f1_side': -1, 'f1_conf': 80, 'f4_side': -1, 'f4_conf': 70}  # обидва SHORT
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
           'прогноз У БІК позиції не має її закривати')
    print('✓ прогноз у бік позиції нічого не закриває')


def test_neutral_forecast_does_not_close():
    """Нейтраль / немає даних — НЕ привід виходити (інакше кожна пауза вибиває)."""
    t = _sig(use_forecast_1h_exit=True, use_forecast_4h_exit=True)
    for fc in ({'f1_side': 0, 'f4_side': 0}, {}, {'f1_side': None, 'f4_side': None}):
        t._fc = fc; t.closed.clear(); t._signal_exit_at.clear()
        _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
               f'нейтральний прогноз {fc} не має закривати')
    print('✓ нейтральний прогноз / немає даних → позиція живе')


def test_decision_opposite_closes():
    t = _sig(use_decision_exit=True)
    t._dc = {'recommended': 'LONG', 'confidence': 74}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'протилежний Decision мав закрити угоду')
    _check(t.closed == [('real', 'decision_exit')], f'{t.closed}')
    _check(any('Decision LONG' in x for x in _LOG), f'лог: {_LOG}')
    print('✓ Decision Center проти позиції → закриття')


def test_decision_neutral_keeps_position():
    t = _sig(use_decision_exit=True)
    for rec in ('NEUTRAL', '', None, 'SHORT'):
        t._dc = {'recommended': rec}; t.closed.clear(); t._signal_exit_at.clear()
        _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
               f'вердикт {rec!r} не має закривати SHORT')
    print('✓ NEUTRAL і власний напрямок позицію не чіпають')


def test_all_three_off_by_default_and_no_work_done():
    for k in ('use_forecast_1h_exit', 'use_forecast_4h_exit', 'use_decision_exit'):
        _check(tmmod.DEFAULT_SETTINGS.get(k) is False, f'{k} має бути OFF за замовчуванням')
    t = _sig()
    t.compute_decision = lambda *a: (_ for _ in ()).throw(
        AssertionError('Decision не має рахуватись, коли всі правила вимкнені'))
    t._get_forecast_both = lambda *a: (_ for _ in ()).throw(
        AssertionError('прогноз не має читатись, коли всі правила вимкнені'))
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
           'вимкнені правила → нічого не робимо')
    print('✓ усі три дефолт OFF і не витрачають ресурс, коли вимкнені')


def test_paper_position_closed_into_shadow_book():
    t = _TM(side='LONG', shadow=True, use_opposite_ob_exit=False,
            use_decision_exit=True)
    t.SIGNAL_EXIT_TTL = 0.0
    t._dc = {'recommended': 'SHORT', 'confidence': 60}
    _check(t._check_signal_exits('MNTUSDT', t._shadow_positions['MNTUSDT'], 1.0, True) is True,
           'paper-позиція теж має закриватись')
    _check(t.closed == [('paper', 'decision_exit')], f'{t.closed}')
    print('✓ paper-позиції закриваються в тіньову книгу')


def test_throttle_prevents_recompute_storm():
    """Перевірка не має бігати на кожен тік монітора (деф. 4с)."""
    t = _sig(use_decision_exit=True)
    t.SIGNAL_EXIT_TTL = 20.0
    calls = []
    t.compute_decision = lambda s_, p_: (calls.append(1), {'recommended': 'SHORT'})[1]
    for _ in range(5):
        t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False)
    _check(len(calls) == 1, f'очікували 1 розрахунок на вікно, отримано {len(calls)}')
    print(f'✓ тротл {t.SIGNAL_EXIT_TTL:.0f}с: 5 тіків → 1 розрахунок')


# ═════════════ 🔗 Режим комбінування AND / АБО ══════════════════════════════
def test_or_mode_is_default_any_rule_closes():
    """Дефолт 'or' — поведінка, яка була до появи режиму: спрацювало будь-яке."""
    _check(tmmod.DEFAULT_SETTINGS.get('signal_exit_mode') == 'or',
           "дефолт має бути 'or' (кожен окремо)")
    t = _sig(use_forecast_1h_exit=True, use_forecast_4h_exit=True, use_decision_exit=True)
    t._fc = {'f1_side': 1, 'f1_conf': 70, 'f4_side': 0}   # проти лише 1H
    t._dc = {'recommended': 'NEUTRAL'}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'у режимі АБО достатньо ОДНОГО правила')
    _check(t.closed == [('real', 'forecast_1h_exit')], f'{t.closed}')
    print('✓ АБО (дефолт): достатньо одного правила')


def test_and_mode_needs_every_enabled_rule():
    t = _sig(use_forecast_1h_exit=True, use_forecast_4h_exit=True,
             use_decision_exit=True, signal_exit_mode='and')
    # Лише 1H проти → у режимі AND угода ТРИМАЄТЬСЯ.
    t._fc = {'f1_side': 1, 'f1_conf': 70, 'f4_side': 0}
    t._dc = {'recommended': 'NEUTRAL'}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
           'AND: одного правила замало')
    # Тепер УСІ три проти.
    t._signal_exit_at.clear()
    t._fc = {'f1_side': 1, 'f1_conf': 70, 'f4_side': 1, 'f4_conf': 65}
    t._dc = {'recommended': 'LONG', 'confidence': 80}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'AND: усі три проти → вихід')
    _check(t.closed == [('real', 'signal_exit_and')], f'{t.closed}')
    _check(any('AND:' in x and 'усі проти' in x for x in _LOG),
           f'у лозі має бути видно ВСІ вердикти: {_LOG}')
    print('✓ AND: вихід лише коли ВСІ увімкнені правила проти')


def test_and_mode_neutral_breaks_the_agreement():
    """Нейтраль — це НЕ «проти». У режимі AND вона ламає збіг."""
    t = _sig(use_forecast_1h_exit=True, use_decision_exit=True, signal_exit_mode='and')
    t._fc = {'f1_side': 1, 'f1_conf': 70}      # проти
    t._dc = {'recommended': 'NEUTRAL'}          # нейтраль → збігу немає
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
           'нейтраль має ламати AND-збіг, а не рахуватись як «проти»')
    print('✓ AND: нейтраль ламає збіг (угода тримається)')


def test_and_mode_ignores_disabled_rules():
    """Вимкнене правило не бере участі — інакше AND ніколи б не зібрався."""
    t = _sig(use_forecast_1h_exit=True, use_decision_exit=True, signal_exit_mode='and')
    t._fc = {'f1_side': 1, 'f1_conf': 70, 'f4_side': 0}   # 4H нейтраль, але ВИМКНЕНИЙ
    t._dc = {'recommended': 'LONG', 'confidence': 80}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'вимкнений 4H не має блокувати AND-збіг увімкнених правил')
    print('✓ AND враховує ЛИШЕ увімкнені правила')


def test_and_with_single_rule_keeps_its_own_reason():
    """Одне увімкнене правило → AND == АБО, і бейдж лишається ВЛАСНИЙ."""
    t = _sig(use_decision_exit=True, signal_exit_mode='and')
    t._dc = {'recommended': 'LONG', 'confidence': 80}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'одне правило в AND має працювати як звичайно')
    _check(t.closed == [('real', 'decision_exit')],
           f'причина не має збіднюватись до загального AND: {t.closed}')
    print('✓ одне правило: AND = АБО, власна причина збережена')


def test_bad_mode_value_falls_back_to_or():
    t = _sig(use_forecast_1h_exit=True, signal_exit_mode='казна-що')
    t._fc = {'f1_side': 1, 'f1_conf': 70}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'невідомий режим → дефолт АБО, а не збій')
    print('✓ некоректне значення режиму → дефолт АБО')


# ═════ 🎯 «ЧІТКО ВИЗНАЧЕНЕ» значення — інакше ЧЕКАЄМО ══════════════════════
def test_neutral_never_closes_in_either_mode():
    """Правило користувача: значення має бути ЧІТКО LONG або SHORT; якщо
    нейтральне — ЧЕКАЄМО, поки визначиться. Це має діяти в ОБОХ режимах."""
    for mode in ('or', 'and'):
        t = _sig(use_forecast_1h_exit=True, use_forecast_4h_exit=True,
                 use_decision_exit=True, signal_exit_mode=mode)
        t._fc = {'f1_side': 0, 'f4_side': 0}
        t._dc = {'recommended': 'NEUTRAL'}
        _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
               f'режим {mode}: суцільна нейтраль не має закривати')
    print('✓ нейтраль → чекаємо (в обох режимах)')


def test_no_data_is_treated_as_undetermined():
    for mode in ('or', 'and'):
        t = _sig(use_forecast_1h_exit=True, use_decision_exit=True,
                 signal_exit_mode=mode)
        t._fc = {}            # прогнозу немає взагалі
        t._dc = {}            # Decision не порахувався
        _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
               f'режим {mode}: «немає даних» — це не вердикт')
    print('✓ «немає даних» = не визначено → чекаємо')


def test_weak_confidence_counts_as_undetermined():
    """Бік є, але впевненість нижча за поріг → це ще НЕ «чітке розуміння»."""
    t = _sig(use_forecast_1h_exit=True, signal_exit_min_conf=60)
    t._fc = {'f1_side': 1, 'f1_conf': 45}          # LONG, але лише 45%
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
           'слабкий вердикт не має закривати угоду')
    # Визначився чітко → закриваємо.
    t._signal_exit_at.clear()
    t._fc = {'f1_side': 1, 'f1_conf': 72}
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'вердикт вище порога → вихід')
    _check(t.closed == [('real', 'forecast_1h_exit')], f'{t.closed}')
    print('✓ впевненість нижче порога = не чітко → чекаємо, вище → вихід')


def test_missing_confidence_with_threshold_is_undetermined():
    """Поріг заданий, а відсотка немає → перевірити чіткість неможливо → чекаємо."""
    t = _sig(use_decision_exit=True, signal_exit_min_conf=50)
    t._dc = {'recommended': 'LONG'}                # без confidence
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
           'немає чим підтвердити чіткість → не закриваємо')
    print('✓ немає відсотка при заданому порозі → чекаємо')


def test_threshold_zero_keeps_any_explicit_side_clear():
    t = _sig(use_forecast_1h_exit=True, signal_exit_min_conf=0)
    t._fc = {'f1_side': 1, 'f1_conf': 12}          # слабко, але поріг вимкнено
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'при порозі 0 будь-який явний бік вважається чітким')
    print('✓ поріг 0 = будь-який явний LONG/SHORT чіткий (стара поведінка)')


def test_and_waits_when_one_verdict_is_weak():
    """У режимі AND слабкий вердикт ЛАМАЄ збіг так само, як нейтраль."""
    t = _sig(use_forecast_1h_exit=True, use_decision_exit=True,
             signal_exit_mode='and', signal_exit_min_conf=60)
    t._fc = {'f1_side': 1, 'f1_conf': 80}          # чітко проти
    t._dc = {'recommended': 'LONG', 'confidence': 40}   # проти, але слабко
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is False,
           'AND: слабкий вердикт має ламати збіг')
    t._signal_exit_at.clear()
    t._dc = {'recommended': 'LONG', 'confidence': 75}   # визначився чітко
    _check(t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5147, False) is True,
           'AND: коли всі чіткі й проти → вихід')
    print('✓ AND: чекаємо, доки КОЖЕН вердикт стане чітким')


def test_min_conf_default_is_off():
    _check(tmmod.DEFAULT_SETTINGS.get('signal_exit_min_conf') == 0,
           'поріг чіткості за замовчуванням вимкнено (поведінку не міняємо)')
    print('✓ дефолт порога чіткості — 0 (вимкнено)')



# ═════ 4. 🧮 МММ LiQ БЕЗ НАПРЯМКУ → ВИХІД (вимога 15.09) ═══════════════
# «Додай вихід із угоди по показнику "🧮 МММ LiQ" — якщо нейтраль,
# закриваємо угоду.» Показник беремо З ТОГО САМОГО знімка, що малює колонку
# «🧮 МММ LiQ» у таблиці угод і рядок 🧮 МММ-монітора.
_SRC_TM = open(os.path.join(_ROOT, 'detection', 'trade_manager.py'),
               encoding='utf-8').read()
_HTML_SM = open(os.path.join(_ROOT, 'templates', 'smart_money.html'),
                encoding='utf-8').read()


class _FF:
    """Фейковий Fuel Filter: віддає РІВНО той зріз, що `mm_snapshot_for`."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def mm_snapshot_for(self, symbols):
        self.calls.append(list(symbols))
        return {k: dict(v) for k, v in self.rows.items()}


def _use_ff(ff):
    ffm.get_fuel_filter = (lambda: ff)


def _mm(side='SHORT', rows=None, **settings):
    """TM з увімкненим правилом 🧮 + підставленим знімком МММ."""
    cfg = {'use_mm_flat_exit': True, 'mm_flat_exit_mode': 'flat',
           'mm_flat_exit_confirm_sec': 0, 'use_opposite_ob_exit': False}
    cfg.update(settings)
    t = _reset(side=side, **cfg)
    ff = _FF(rows if rows is not None else {})
    _use_ff(ff)
    # Доступ до зрізу з тесту: сценарії «МММ розвернувся → повернувся» мусять
    # міняти дані МІЖ тактами, а не створювати новий TM (таймери б зникли).
    t._ff = ff
    return t


def _row(mm, strength=5):
    return {'mm': mm, 'strength': strength, 'strength_prev': strength,
            'delta': 0, 'grow_since': None}


def test_flat_mm_closes_the_trade():
    """Дослівна вимога: МММ став ⚖ рівновагою → угоду закрито."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row(None, 4)})
    closed = t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(closed and t.closed == [('real', 'mm_flat_exit')],
           f'⚖ рівновага мусить закривати угоду: {t.closed}')
    _check(any('рівновага' in x for x in _LOG),
           f'причина не названа в 🧾 Лозі: {_LOG}')
    print('✓ 🧮 МММ ⚖ рівновага → вихід')


def test_directional_mm_keeps_the_trade():
    t = _mm(side='SHORT', rows={'MNTUSDT': _row('SHORT', 62)})
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'МММ тримає напрямок — виходу бути не може: {t.closed}')
    print('✓ 🧮 МММ у бік угоди → тримаємо')


def test_missing_snapshot_is_not_flat():
    """⚠️ ГОЛОВНИЙ ЗАПОБІЖНИК. Знімка по монеті може не бути (бот щойно
    піднявся, liq-map ще не зібрала рівні). Якби порожньо читалось як
    «рівновага», КОЖЕН рестарт закривав би ВСІ відкриті позиції."""
    t = _mm(side='SHORT', rows={})                     # знімок порожній
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'«немає даних» — це НЕ нейтраль: {t.closed}')
    # І сам Fuel Filter може ще не існувати.
    _use_ff(None)
    t._signal_exit_at.clear()
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'без Fuel Filter правило мусить мовчати: {t.closed}')
    print('✓ 🧮 «немає даних» ≠ «нейтраль» (рестарт не вбиває позиції)')


def test_opposite_mm_holds_by_default_and_closes_in_the_other_mode():
    """Дослівна вимога — про НЕЙТРАЛЬ, тож розворот МММ за замовчуванням угоду
    НЕ чіпає. Для тих, хто вважає розворот гіршим за згасання, є окремий режим."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)})
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'режим «лише рівновага» не закриває на розвороті: {t.closed}')
    t2 = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)},
             mm_flat_exit_mode='flat_or_against')
    t2._check_signal_exits('MNTUSDT', t2._positions['MNTUSDT'], 0.5, False)
    # ⚠️ ВИМОГА ЗМІНИЛАСЬ (18.09): розворот має ВЛАСНИЙ код причини —
    # `mm_against_exit`. Раніше обидві гілки віддавали `mm_flat_exit`, і в
    # історії угод розворот підписувався як «⚖ рівновага» (скарга зі скріном).
    _check(t2.closed == [('real', 'mm_against_exit')],
           f'режим «рівновага АБО проти» мусить закрити: {t2.closed}')
    print('✓ 🧮 розворот МММ: тримаємо (деф.) / закриваємо (окремий режим)')


def test_the_two_reasons_are_never_mixed_up():
    """Скарга 18.09: «Монета закрилась саме по протилежному значенню, а не
    рівновага — потрібно точно розрізнити причини закриття». Один код на дві
    різні події означав, що бейдж і рядок логу показували не те, що сталось."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row(None, 6)},
            mm_flat_exit_mode='flat_or_against')
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [('real', 'mm_flat_exit')],
           f'⚖ рівновага мусить лишитись `mm_flat_exit`: {t.closed}')
    t2 = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)},
             mm_flat_exit_mode='flat_or_against')
    t2._check_signal_exits('MNTUSDT', t2._positions['MNTUSDT'], 0.5, False)
    _check(t2.closed == [('real', 'mm_against_exit')],
           f'розворот мусить мати ВЛАСНИЙ код: {t2.closed}')
    # І підписи мусять бути РІЗНІ в усіх трьох місцях показу.
    for src, name in ((_SRC_TM, 'trade_manager'), (_HTML_SM, 'сторінка')):
        _check('mm_against_exit' in src, f'{name} не знає нової причини')
    _check("'mm_against_exit': '🧮 МММ LiQ РОЗВЕРНУВСЯ ПРОТИ позиції'" in _SRC_TM,
           'немає розгорнутого підпису розвороту')
    _check("'mm_against_exit': '🧮 МММ LiQ ПРОТИ'" in _SRC_TM
           and "'mm_against_exit': '🧮 МММ LiQ ПРОТИ'" in _HTML_SM,
           'бейдж розвороту мусить відрізнятись від бейджа рівноваги')
    print('✓ 🧮 ⚖ рівновага і РОЗВОРОТ — різні причини з різними підписами')


def test_bad_mode_value_falls_back_to_flat():
    t = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)},
            mm_flat_exit_mode='щось не те')
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'сміттєвий режим → дефолт «лише рівновага»: {t.closed}')
    print('✓ 🧮 некоректний режим → дефолт, а не збій')


def test_confirm_window_requires_the_state_to_hold():
    """Межа напрямку — сила ≈10%, і біля неї показник миготить. Тому є вікно
    підтвердження: перший такт рівноваги ще не закриває."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row(None, 6)},
            mm_flat_exit_confirm_sec=60)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'закрили, не дочекавшись підтвердження: {t.closed}')
    # «Відмотуємо» початок стану на 61с назад — стан протримався.
    t._mm_flat_since['MNTUSDT'] -= 61
    t._signal_exit_at.clear()
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [('real', 'mm_flat_exit')],
           f'після витримки мусить закрити: {t.closed}')
    _check(any('тримається' in x for x in _LOG),
           f'у причині не видно, скільки стан тримався: {_LOG}')
    print('✓ 🧮 вікно підтвердження: миготіння не вибиває з ринку')


def test_confirm_timer_resets_when_direction_returns():
    """Інакше короткі провали в рівновагу «накопичувались» би між епізодами і
    рано чи пізно дали б вихід там, де стан щоразу тримався секунди."""
    ff = _FF({'MNTUSDT': _row(None, 6)})
    t = _reset(side='SHORT', use_mm_flat_exit=True, mm_flat_exit_confirm_sec=60,
               use_opposite_ob_exit=False)
    _use_ff(ff)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check('MNTUSDT' in t._mm_flat_since, 'таймер не стартував')
    ff.rows = {'MNTUSDT': _row('SHORT', 44)}           # напрямок повернувся
    t._signal_exit_at.clear()
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check('MNTUSDT' not in t._mm_flat_since,
           'таймер не обнулено, коли напрямок повернувся')
    print('✓ 🧮 таймер підтвердження обнуляється на поверненні напрямку')


# ─── ⏱ ВИТРИМКА НАЛЕЖИТЬ ⚖ РІВНОВАЗІ, А НЕ РОЗВОРОТУ (вимога 18.09) ───────
# Дослівно: «⏱ Тримається ≥ — має відноситись до "Рівновага", а коли "проти"
# то закривати відразу». Сенс поля — межа напрямку |dir| ≤ 0.1, біля якої
# показник МИГОТИТЬ; зустрічний тиск миготінням не є.

def test_the_reversal_waits_out_its_own_window():
    """⚠️ ВИМОГА ЗМІНИЛАСЬ (20.09, дослівно: «потрібно трішки витримки, бо
    секундна зміна і вибиває з угоди»). Замок 18.09 «проти → закривати
    ОДРАЗУ» ПЕРЕПИСАНО, а не полагоджено: розворот тепер теж підтверджується
    часом — але СВОЇМ полем, не полем ⚖ рівноваги."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)},
            mm_flat_exit_mode='flat_or_against',
            mm_flat_exit_confirm_sec=0, mm_against_exit_confirm_sec=900)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'секундний розворот не має вибивати з угоди: {t.closed}')
    _check('MNTUSDT' in t._mm_against_since, 'відлік розвороту не стартував')
    # Розворот ТРИМАЄТЬСЯ → після витримки закриваємо (код причини свій).
    t._mm_against_since['MNTUSDT'] -= 901
    t._signal_exit_at.clear()
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [('real', 'mm_against_exit')],
           f'витриманий розворот мусить закрити угоду: {t.closed}')
    _check(any('тримається' in x for x in _LOG),
           f'у причині не видно, скільки розворот тримався: {_LOG}')
    print('✓ 🧮 розворот чекає ВЛАСНУ витримку, а потім закриває')


def test_the_reversal_timer_resets_when_mm_comes_back():
    """Повернувся напрямок (або ⚖) → відлік розвороту ОБНУЛЯЄТЬСЯ: інакше
    короткі зустрічні сплески «накопичувались» би між епізодами."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)},
            mm_flat_exit_mode='flat_or_against', mm_against_exit_confirm_sec=900)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check('MNTUSDT' in t._mm_against_since, 'відлік розвороту не стартував')
    t._ff.rows['MNTUSDT'] = _row('SHORT', 40)        # МММ знову в бік угоди
    t._signal_exit_at.clear()
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check('MNTUSDT' not in t._mm_against_since,
           'відлік розвороту мусить обнулятись, коли МММ повернувся')
    _check(t.closed == [], f'угода мусить лишитись відкритою: {t.closed}')
    print('✓ 🧮 повернення МММ обнуляє відлік розвороту')


def test_the_two_windows_never_inherit_each_other():
    """⚖ і 🔄 — РІЗНІ події, тож і витримки різні. Перехід між станами не має
    «дарувати» новому стану вже відпрацьований чужий час."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)},
            mm_flat_exit_mode='flat_or_against',
            mm_flat_exit_confirm_sec=900, mm_against_exit_confirm_sec=900)
    t._mm_flat_since['MNTUSDT'] = time.time() - 890  # ⚖ майже «дозрів»
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check('MNTUSDT' not in t._mm_flat_since,
           'таймер рівноваги лишився після розвороту')
    _check(t.closed == [], f'чужа витримка не має закривати угоду: {t.closed}')
    # І дзеркально: розворот → рівновага не успадковує його відлік.
    t._mm_against_since['MNTUSDT'] = time.time() - 890
    t._ff.rows['MNTUSDT'] = _row(None, 5)
    t._signal_exit_at.clear()
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check('MNTUSDT' not in t._mm_against_since,
           'таймер розвороту лишився після переходу в ⚖')
    _check(t.closed == [], f'⚖ мусить набирати СВОЮ витримку з нуля: {t.closed}')
    print('✓ 🧮 витримки ⚖ і 🔄 не перетікають одна в одну')


def test_zero_window_keeps_the_old_instant_behaviour():
    """0 = закривати одразу — щоб стара поведінка лишалась досяжною."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row('LONG', 55)},
            mm_flat_exit_mode='flat_or_against', mm_against_exit_confirm_sec=0)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [('real', 'mm_against_exit')],
           f'при 0 розворот мусить закривати одразу: {t.closed}')
    print('✓ 🧮 «0 с» повертає миттєве закриття на розвороті')


def test_the_window_still_guards_the_flat_state_in_the_same_mode():
    """⚠️ Половина вимоги, яку легко загубити: витримка мусить ПРАЦЮВАТИ для
    ⚖ рівноваги і в режимі «рівновага АБО проти» теж."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row(None, 6)},
            mm_flat_exit_mode='flat_or_against', mm_flat_exit_confirm_sec=900)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [], f'⚖ рівновага мусить чекати витримку: {t.closed}')
    t._mm_flat_since['MNTUSDT'] -= 901
    t._signal_exit_at.clear()
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [('real', 'mm_flat_exit')],
           f'після витримки ⚖ мусить закрити: {t.closed}')
    print('✓ 🧮 витримка і далі стереже саме ⚖ рівновагу')


def test_ui_has_a_separate_window_for_each_state():
    """⚠️ Замок ПЕРЕПИСАНО (вимога 20.09): полів тепер ДВА, і кожне мусить
    казати, про який саме стан воно — інакше користувач крутив би не те."""
    i = _HTML_SM.index('id="tm-mm-flat-exit-confirm"')
    _check('Рівновага тримається' in _HTML_SM[max(0, i - 900):i],
           'підпис поля не каже, що витримка — саме про ⚖ рівновагу')
    j = _HTML_SM.index('id="tm-mm-against-exit-confirm"')
    _check(j > i, 'поле витримки розвороту мусить стояти під полем ⚖')
    block = _HTML_SM[max(0, j - 1400):j]
    _check('Розворот тримається' in block,
           'друге поле не підписане як витримка РОЗВОРОТУ')
    _check('обнуляється' in block or 'обнуля' in block,
           'у підказці не сказано, що повернення МММ скидає відлік')
    _check('mm_against_exit_confirm_sec' in _HTML_SM,
           'ключ не їде на сервер')
    print('✓ UI: у ⚖ і 🔄 власні поля витримки, обидва підписані')


def test_the_reversal_window_has_a_sane_default():
    _check(tmmod.DEFAULT_SETTINGS['mm_against_exit_confirm_sec'] == 60,
           'дефолт витримки розвороту — 60с (одне оновлення джерела)')
    _check(tmmod.DEFAULT_SETTINGS['mm_flat_exit_confirm_sec'] == 0,
           'витримка ⚖ лишається окремим числом і не змінилась')
    print('✓ ⚙️ дефолт витримки розвороту — 60с, ⚖ не зачеплено')


def test_rule_is_off_by_default():
    _check(tmmod.DEFAULT_SETTINGS['use_mm_flat_exit'] is False,
           'нове правило виходу не має вмикатись саме')
    _check(tmmod.DEFAULT_SETTINGS['mm_flat_exit_mode'] == 'flat',
           'дефолтний режим мусить бути дослівною вимогою — лише рівновага')
    _check(tmmod.DEFAULT_SETTINGS['mm_flat_exit_confirm_sec'] == 0,
           'дефолт підтвердження — 0 (закривати одразу, як і просили)')
    ff = _FF({'MNTUSDT': _row(None, 3)})
    t = _reset(side='SHORT', use_opposite_ob_exit=False)   # усі правила OFF
    _use_ff(ff)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [] and ff.calls == [],
           f'вимкнене правило не має ні закривати, ні читати знімок: {ff.calls}')
    print('✓ 🧮 дефолт OFF і жодної роботи при вимкненому правилі')


def test_rule_stays_out_of_the_and_or_combination():
    """⚠️ У комбінуванні трьох вердиктів нейтраль ЛАМАЄ збіг, а тут вона і є
    підставою вийти. Змішати їх означало б прямо протилежні правила в одному
    вузлі, тому 🧮 працює ОКРЕМО — навіть у найсуворішому режимі 'and'."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row(None, 2)},
            use_forecast_1h_exit=True, use_decision_exit=True,
            signal_exit_mode='and')
    t._fc = {'f1_side': 0, 'f1_conf': 0, 'f4_side': 0, 'f4_conf': 0}
    t._dc = {'recommended': 'NEUTRAL'}
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(t.closed == [('real', 'mm_flat_exit')],
           f'🧮 мусить спрацювати незалежно від режиму комбінування: {t.closed}')
    # І навпаки: у самому комбінуванні МММ не згадується.
    i = _SRC_TM.index('def _signal_exit_reason(')
    fn = _SRC_TM[i:_SRC_TM.index('\n    def ', i + 10)]
    _check('mm_flat' not in fn and 'mm_snapshot_for' not in fn,
           'МММ просочився в комбінування трьох вердиктів')
    print('✓ 🧮 правило живе ОКРЕМО від АБО/AND-комбінування')


def test_source_is_the_shared_snapshot_and_nothing_is_recomputed():
    """Та сама метрика у двох місцях = ОДНЕ джерело: інакше бот закривав би
    угоду за числом, якого на екрані не видно (урок PD-зони)."""
    ff = _FF({'MNTUSDT': _row(None, 7)})
    t = _reset(side='SHORT', use_mm_flat_exit=True, use_opposite_ob_exit=False)
    _use_ff(ff)
    t._check_signal_exits('MNTUSDT', t._positions['MNTUSDT'], 0.5, False)
    _check(ff.calls == [['MNTUSDT']], f'знімок не запитано як треба: {ff.calls}')
    i = _SRC_TM.index('def _mm_flat_exit_reason(')
    fn = _SRC_TM[i:_SRC_TM.index('\n    def ', i + 10)]
    if '"""' in fn:                       # докстрінг пояснює — код не рахує
        fn = fn[fn.index('"""', fn.index('"""') + 3) + 3:]
    _check('mm_snapshot_for' in fn, 'правило не читає спільний знімок')
    for bad in ('_fuel_dir_legacy', 'compute_mm', '_liq_state', 'get_state('):
        _check(bad not in fn, f'правило рахує МММ саме ({bad}) — розійдеться з UI')
    print('✓ 🧮 джерело — спільний знімок, власних розрахунків немає')


def test_paper_position_closes_into_its_own_book():
    t = _mm(side='LONG', rows={'MNTUSDT': _row(None, 1)})
    t._positions, t._shadow_positions = {}, t._positions or t._shadow_positions
    pos = list(t._shadow_positions.values())[0] if t._shadow_positions else None
    if pos is None:                       # _mm створює реальну — зробимо паперову
        pos = {'symbol': 'MNTUSDT', 'side': 'LONG', 'entry_price': 0.5,
               'opened_at': 1000.0}
        t._shadow_positions = {'MNTUSDT': pos}
    t._check_signal_exits('MNTUSDT', pos, 0.5, True)
    _check(t.closed == [('paper', 'mm_flat_exit')],
           f'паперова угода мусить закритись у СВОЮ книгу: {t.closed}')
    print('✓ 🧮 paper-позиція закривається у свою книгу')


def test_throttle_is_per_book_not_per_symbol():
    """🐞 Було: тротл ключувався ЛИШЕ символом, тож по монеті з ДВОМА
    позиціями (real + paper) перевірку з'їдала та книга, чий монітор устиг
    першим, — друга лишалась БЕЗ правил виходу. Той самий клас помилки, що вже
    ловили на `_pilot_at` (кейс TRXUSDT)."""
    t = _mm(side='SHORT', rows={'MNTUSDT': _row(None, 3)})
    pos = t._positions['MNTUSDT']
    t._shadow_positions = {'MNTUSDT': dict(pos)}
    t._check_signal_exits('MNTUSDT', pos, 0.5, False)          # реальна
    t._check_signal_exits('MNTUSDT', t._shadow_positions['MNTUSDT'], 0.5, True)
    _check(t.closed == [('real', 'mm_flat_exit'), ('paper', 'mm_flat_exit')],
           f'обидві книги мусять перевірятись у тому самому такті: {t.closed}')
    _check(t._exit_key('AAA', True) != t._exit_key('AAA', False),
           'ключ тротлу не розрізняє книги')
    print('✓ тротл правил виходу — окремий на КОЖНУ книгу')


def test_reason_has_human_labels_everywhere():
    """Код причини мусить мати підпис у ВСІХ трьох місцях показу, інакше в
    історії угод стоятиме сире `mm_flat_exit`."""
    _check("'mm_flat_exit': '🧮 МММ LiQ втратив напрямок" in _SRC_TM,
           'немає розгорнутого підпису причини закриття')
    _check("'mm_flat_exit': '🧮 МММ LiQ ⚖'" in _SRC_TM,
           'немає короткого бейджа причини')
    _check("'mm_flat_exit': '🧮 МММ LiQ ⚖'" in _HTML_SM,
           'JS-мапа причин на сторінці не знає про нове правило')
    print('✓ причина має людський підпис у TM і на сторінці')


def test_ui_toggle_is_wired_both_ways():
    _check('id="tm-use-mm-flat-exit"' in _HTML_SM, 'немає тумблера правила')
    _check('id="tm-mm-flat-exit-mode"' in _HTML_SM, 'немає вибору режиму')
    _check('id="tm-mm-flat-exit-confirm"' in _HTML_SM, 'немає поля підтвердження')
    for key in ('use_mm_flat_exit', 'mm_flat_exit_mode', 'mm_flat_exit_confirm_sec'):
        _check(f'{key}:' in _HTML_SM, f'{key} не йде у збереження налаштувань')
        _check(f's.{key}' in _HTML_SM, f'{key} не відновлюється з налаштувань')
    # ⚠️ Блок мусить стояти ПІСЛЯ «Комбінувати» — інакше читався б як четверте
    # правило того збігу, у якому нейтраль означає ПРОТИЛЕЖНЕ.
    _check(_HTML_SM.index('id="tm-use-mm-flat-exit"')
           > _HTML_SM.index('id="tm-signal-exit-mode"'),
           'правило стоїть серед трьох вердиктів — читатиметься як частина AND/OR')
    print('✓ UI: тумблер + режим + підтвердження, і стоять окремо від AND/OR')


if __name__ == '__main__':
    test_mnt_case_preexisting_opposite_ob_must_not_close()
    test_new_opposite_ob_closes()
    test_no_ob_at_open_then_opposite_appears_closes()
    test_same_side_new_ob_does_not_close_and_moves_baseline()
    test_older_block_never_closes()
    test_new_position_rebaselines()
    test_toggle_off_disables_rule()
    test_reason_text_no_longer_lies_about_price()
    test_forecast_1h_opposite_closes()
    test_forecast_4h_opposite_closes_independently()
    test_forecast_same_side_keeps_position()
    test_neutral_forecast_does_not_close()
    test_decision_opposite_closes()
    test_decision_neutral_keeps_position()
    test_all_three_off_by_default_and_no_work_done()
    test_paper_position_closed_into_shadow_book()
    test_throttle_prevents_recompute_storm()
    test_or_mode_is_default_any_rule_closes()
    test_and_mode_needs_every_enabled_rule()
    test_and_mode_neutral_breaks_the_agreement()
    test_and_mode_ignores_disabled_rules()
    test_and_with_single_rule_keeps_its_own_reason()
    test_bad_mode_value_falls_back_to_or()
    test_neutral_never_closes_in_either_mode()
    test_no_data_is_treated_as_undetermined()
    test_weak_confidence_counts_as_undetermined()
    test_missing_confidence_with_threshold_is_undetermined()
    test_threshold_zero_keeps_any_explicit_side_clear()
    test_and_waits_when_one_verdict_is_weak()
    test_min_conf_default_is_off()
    test_flat_mm_closes_the_trade()
    test_directional_mm_keeps_the_trade()
    test_missing_snapshot_is_not_flat()
    test_opposite_mm_holds_by_default_and_closes_in_the_other_mode()
    test_the_two_reasons_are_never_mixed_up()
    test_bad_mode_value_falls_back_to_flat()
    test_confirm_window_requires_the_state_to_hold()
    test_confirm_timer_resets_when_direction_returns()
    test_the_reversal_waits_out_its_own_window()
    test_the_reversal_timer_resets_when_mm_comes_back()
    test_the_two_windows_never_inherit_each_other()
    test_zero_window_keeps_the_old_instant_behaviour()
    test_the_window_still_guards_the_flat_state_in_the_same_mode()
    test_ui_has_a_separate_window_for_each_state()
    test_the_reversal_window_has_a_sane_default()
    test_rule_is_off_by_default()
    test_rule_stays_out_of_the_and_or_combination()
    test_source_is_the_shared_snapshot_and_nothing_is_recomputed()
    test_paper_position_closes_into_its_own_book()
    test_throttle_is_per_book_not_per_symbol()
    test_reason_has_human_labels_everywhere()
    test_ui_toggle_is_wired_both_ways()
    print('\nУсі тести правил виходу пройдено ✅')
