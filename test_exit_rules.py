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
    print('\nУсі тести правил виходу пройдено ✅')
