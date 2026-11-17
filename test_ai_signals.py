"""🤖 ШІ: розряд заряджених «Пружин» — сигнал, вихід, читання аномалій (17.11).

Вимоги: «Сигнали ШІ» і «Вихід з угоди ШІ» на монетах «заряджених» Пружин;
ваги — з досліджень «⚠ Аномалії обсягу»; правила читання аномалій
користувача (🧽 розворот на рівні, 🔥 виснаження лише після руху, ⚡/📈 —
продовження, ⚡ проти тренду на рівні — зміна ініціативи, 🔥→🧽 — найсильніше,
підтвердження наступною свічкою або CHoCH, одна позначка без рівня — «придивитись»).

Запуск: python3 test_ai_signals.py
"""
import ast
import importlib.util
import os
import sys
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    sys.modules.setdefault(n, types.ModuleType(n))
sys.modules['pybit.unified_trading'].HTTP = object
if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
    sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_lg = types.ModuleType('detection.activity_log'); _lg.log_activity = lambda *a, **k: None
sys.modules['detection.activity_log'] = _lg
A = _load('detection.ai_signals', 'detection/ai_signals.py')
_SRC = open(os.path.join(_ROOT, 'detection', 'ai_signals.py'), encoding='utf-8').read()
_TM = open(os.path.join(_ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_SC = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
_APP = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
_JS = open(os.path.join(_ROOT, 'infosite', 'js', 'app.js'), encoding='utf-8').read()
_SL = open(os.path.join(_ROOT, 'detection', 'signal_labels.py'), encoding='utf-8').read()

NOW = 1_800_000_000.0
LV = [('межа коробки', 100.0), ('межа коробки', 90.0)]


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _an(kind, tf='1h', ago=600, **kw):
    a = {'kind': kind, 'tf': tf, 'ts': NOW - ago, 'price': 95.0}
    a.update(kw)
    return a


# ── 1. Класифікація за правилами користувача ─────────────────────────────
def test_absorb_is_reversal_strongest_at_level():
    on = A.classify_anomaly(_an('absorb', hint='up', price=90.1), LV)
    off = A.classify_anomaly(_an('absorb', hint='up', price=95.0), LV)
    _check(on['role'] == 'reversal' and on['dir'] == 'up' and on['level'], f'🧽 на рівні: {on}')
    _check(off['role'] == 'reversal' and not off['level'], f'🧽 без рівня: {off}')
    r_on = A.read_anomalies([_an('absorb', hint='up', price=90.1, o1=0.3)], NOW, LV)
    r_off = A.read_anomalies([_an('absorb', hint='up', price=95.0, o1=0.3)], NOW, LV)
    _check(r_on['up']['rev'] > r_off['up']['rev'], 'без рівня 🧽 — лише «придивитись» (слабше)')


def test_climax_reversal_only_after_long_move():
    after = A.classify_anomaly(_an('climax', bar_dir='up', pre_chg=3.0), LV)
    calm = A.classify_anomaly(_an('climax', bar_dir='up', pre_chg=0.2), LV)
    _check(after['role'] == 'reversal' and after['dir'] == 'dn',
           f'🔥 після руху вгору → розворот ПРОТИ свічки: {after}')
    _check(calm['role'] == 'start' and calm['dir'] == 'up',
           f'🔥 у спокої → старт руху в бік свічки: {calm}')
    _check(A.classify_anomaly(_an('climax', tf='15m', bar_dir='dn', pre_chg=-1.0), LV)['role'] == 'reversal',
           'поріг «тривалого руху» залежить від TF')


def test_side_and_spike_are_continuation_except_shift_at_level():
    _check(A.classify_anomaly(_an('side', hint='up'), LV)['role'] == 'continuation', '⚡ — продовження')
    _check(A.classify_anomaly(_an('spike', bar_dir='dn'), LV)['role'] == 'continuation', '📈 — продовження')
    sh = A.classify_anomaly(_an('side', hint='up', pre_chg=-2.0, price=90.2), LV)
    _check(sh['role'] == 'shift' and sh['dir'] == 'up', f'⚡ проти тренду на рівні → зміна ініціативи: {sh}')
    nl = A.classify_anomaly(_an('side', hint='up', pre_chg=-2.0, price=95.0), LV)
    _check(nl['role'] == 'continuation', '⚡ проти тренду БЕЗ рівня — не зміна ініціативи')


def test_combo_climax_then_absorb_is_strongest():
    an = [_an('climax', bar_dir='up', pre_chg=3.0, ago=3 * 3600 - 60),
          _an('absorb', hint='dn', price=100.1, ago=2 * 3600 - 60, o1=-0.3)]
    r = A.read_anomalies(an, NOW, LV, look=4 * 3600)
    _check(r['dn']['combo'] >= 0.8, f'🔥→🧽 на 1–3 свічках → найсильніший розворот: {r["dn"]}')
    _check(r['up']['exhaust'] > 0, '🔥 — виснаження руху вгору')
    far = [_an('climax', bar_dir='up', pre_chg=3.0, ago=6 * 3600),
           _an('absorb', hint='dn', price=100.1, ago=600)]
    _check(A.read_anomalies(far, NOW, LV, look=7 * 3600)['dn']['combo'] == 0,
           '🧽 пізніше ніж за 3 свічки — не сценарій 🔥→🧽')


def test_confirmation_next_candle_or_choch():
    base = _an('absorb', hint='up', price=90.1)
    w = lambda r: r['items'][0]['w']
    ok = A.read_anomalies([dict(base, o1=0.4)], NOW, LV)
    pend = A.read_anomalies([dict(base)], NOW, LV)
    bad = A.read_anomalies([dict(base, o1=-0.4)], NOW, LV)
    ch = A.read_anomalies([dict(base)], NOW, LV, choch={'dir': 'up', 't': NOW - 300})
    _check(w(ok) > w(pend) > w(bad), f'підтвердження свічкою: {w(ok)} > {w(pend)} > {w(bad)}')
    _check(w(ch) == w(ok), 'CHoCH у бік аномалії — теж підтвердження')
    _check(A.last_choch([{'tag': 'CHoCH', 'dir': 'bear', 'to_t': 5}, {'tag': 'BOS', 'dir': 'bull', 'to_t': 9},
                         {'tag': 'CHoCH', 'dir': 'bull', 'to_t': 7}]) == {'dir': 'up', 't': 7.0},
           'останній CHoCH (BOS не рахується)')


# ── 2. Бал входу / виходу ────────────────────────────────────────────────
_VW_UP = {'box_hi': 100.0, 'box_lo': 90.0, 'price': 101.0, 'beyond': 'LONG',
          'range_x': 2.0, 'vol_x': 2.0, 'range_need': 1.8, 'vol_need': 1.8}
_REC = {'charge': 80, 'state': 'charged', 'prob_long': 70, 'lean': 'LONG'}


def test_entry_reads_flow_and_vetoes_counter_combo():
    e = A.entry_score('LONG', _REC, _VW_UP, [_an('side', hint='up', o1=0.3)], NOW)
    rv = [_an('climax', bar_dir='dn', pre_chg=-3.0, ago=3000),
          _an('absorb', hint='up', price=90.1, ago=600, o1=0.3)]
    _check(A.entry_score('LONG', _REC, _VW_UP, rv, NOW)['score'] >= A.COMBO_ENTRY_SCORE,
           'підтверджений 🔥→🧽 у бік розряду + пробій → сильний вхід')
    _check(e['started'] and e['parts']['flow'] > 0, f'розряд вгору + ⚡ за: {e}')
    _check(e['ready'] and A.entry_decide(e) and e['scenario']['key'] == 'E2',
           f'17.11: вхід — за готовим сценарієм (💥 старт розряду пройдено): {e["scenario"]}')
    lone = A.entry_score('LONG', _REC, _VW_UP, [], NOW)
    _check(not lone['ready'] and not A.entry_decide(lone) and lone['score'] >= 50,
           f'високий бал без комплексу реакцій — НЕ вхід: {lone["scenario_text"]}')
    combo = [_an('climax', bar_dir='up', pre_chg=3.0, ago=3000),
             _an('absorb', hint='dn', price=100.1, ago=600, o1=-0.3)]
    v = A.entry_score('LONG', _REC, _VW_UP, combo, NOW)
    _check(v['veto'] and not v['ready'], f'зустрічний 🔥→🧽 — вето входу: {v}')


def test_exit_combo_ready_and_continuation_holds():
    vw = {'box_hi': 100.0, 'box_lo': 90.0, 'price': 99.0}
    combo = [_an('climax', bar_dir='up', pre_chg=3.0, ago=3000),
             _an('absorb', hint='dn', price=100.1, ago=600, o1=-0.3)]
    x = A.exit_score('LONG', vw, combo, NOW)
    _check(x['combo'] and x['combo_conf'] and x['ready'] and x['score'] >= A.COMBO_EXIT_SCORE,
           f'🔥 після руху за угодою + підтверджений 🧽 проти → вихід: {x}')
    nc = A.exit_score('LONG', vw, [dict(combo[0]), dict(combo[1], o1=None)], NOW)
    _check(nc['combo'] and not nc['combo_conf'] and nc['score'] < A.COMBO_EXIT_SCORE,
           f'без підтвердження — чекаємо: {nc}')
    one = A.exit_score('LONG', vw, [_an('absorb', hint='dn', price=95.0)], NOW)
    _check(not one['ready'], f'одна позначка без рівня — не причина виходити: {one}')
    hold = A.exit_score('LONG', vw, combo + [_an('side', hint='up', o1=0.3, ago=300),
                                             _an('spike', bar_dir='up', o1=0.3, ago=200)], NOW)
    _check(hold['hold'] > 0, 'продовження за угодою видно')
    h2 = A.exit_score('LONG', vw, [_an('absorb', hint='dn', price=100.1, o1=-0.2),
                                    _an('side', hint='up', o1=0.3, ago=300)], NOW)
    h1 = A.exit_score('LONG', vw, [_an('absorb', hint='dn', price=100.1, o1=-0.2)], NOW)
    _check(h2['score'] < h1['score'], 'продовження за угодою знижує бал виходу')


def test_exit_is_governed_by_anomalies_1711():
    """Вихід — за ВЕРДИКТОМ аномалій, а не за порогом балу (вимога 17.11)."""
    up = {'box_hi': 100.0, 'box_lo': 90.0, 'price': 99.0}
    dn = {'box_hi': 100.0, 'box_lo': 90.0, 'price': 89.0, 'beyond': 'SHORT'}
    # ціна проти + ціль пройдена, але аномалії мовчать → чекаємо, не виходимо
    w = A.exit_score('LONG', dn, [], NOW, exh={'pct': 100}, pnl=0.5, peak=2.0)
    _check(w['verdict'] == 'wait' and not w['ready'] and not A.exit_decide(w),
           f'без аномалій вихід ШІ не спрацьовує: {w}')
    # аномалії проти на рівні (підтверджені) + ціна проти → close
    ab = [_an('absorb', hint='dn', price=100.1, o1=-0.3)]
    c = A.exit_score('LONG', dn, ab, NOW)
    _check(c['verdict'] == 'close' and A.exit_decide(c, 'confirmed') and not A.exit_decide(c, 'all'),
           f'аномалії + ціна → вихід (режим confirmed), але не «всі»: {c}')
    # аномалії + ціна + прибуток → all
    a = A.exit_score('LONG', dn, ab, NOW, exh={'pct': 100}, pnl=0.5, peak=2.0)
    _check(a['verdict'] == 'all' and a['score'] >= A.ALL_EXIT_SCORE and A.exit_decide(a, 'all'),
           f'усі показники за вихід: {a}')
    _check(all(a['groups'].values()) and '✓ аномалії · ✓ ціна · ✓ прибуток' == A.exit_checklist_text(a),
           A.exit_checklist_text(a))
    # аномалії самі → watch
    o = A.exit_score('LONG', up, ab, NOW)
    _check(o['verdict'] == 'watch' and not A.exit_decide(o), f'аномалії без підтвердження: {o}')
    # свіже продовження за угодою тримає
    h = A.exit_score('LONG', dn, ab + [_an('side', hint='up', o1=0.3, ago=200),
                                       _an('spike', bar_dir='up', o1=0.3, ago=100)], NOW)
    _check(h['verdict'] == 'hold' and not h['ready'], f'⚡/📈 за угодою — тримаємо: {h}')
    # CHoCH проти угоди ПІСЛЯ входу — свідчення ціни; ДО входу — ні
    ch = {'dir': 'dn', 't': NOW - 300}
    x1 = A.exit_score('LONG', up, ab, NOW, choch=ch, opened_at=NOW - 3600)
    x0 = A.exit_score('LONG', up, ab, NOW, choch=ch, opened_at=NOW - 100)
    _check(x1['parts']['choch'] == 1.0 and x1['verdict'] in ('close', 'scenario') and x1['ready'],
           f'CHoCH після входу: {x1}')
    _check(x0['parts']['choch'] == 0.0, 'CHoCH до входу — не подія')
    # аномалії від входу: 🧽 3 год тому (до 6 год) враховується при look
    old = [_an('absorb', hint='dn', price=100.1, o1=-0.3, ago=3 * 3600)]
    _check(A.exit_score('LONG', dn, old, NOW)['verdict'] == 'wait', 'вікно 2 год його не бачить')
    _check(A.exit_score('LONG', dn, old, NOW, look=4 * 3600)['verdict'] == 'close',
           'вихід дивиться від входу')
    _check(not A.exit_decide({'ready': True, 'verdict': 'close'}, 'all')
           and A.exit_decide({'ready': True, 'verdict': 'combo'}, 'all'), 'режими')
    _check('function aiExitChipHTML' in _HTML and 'aiExitChipHTML(p)' in _HTML, 'значок 🤖 у 🚨')


def test_fresh_trade_is_not_closed_by_its_own_entry_reasons_1711():
    """Угода, що щойно відкрилась, не може «вже закриватись»: аномалії до входу
    і сплеск свічки входу — це причини ВХОДУ, а не виходу (скарга 17.11)."""
    oa = NOW - 600                                   # відкрито 10 хв тому
    entry_bar = {'box_hi': 100.0, 'box_lo': 90.0, 'price': 101.0, 'beyond': 'LONG',
                 'climax': 'LONG', 't': NOW - 1200}   # свічка входу — до відкриття
    pre = [_an('climax', bar_dir='up', pre_chg=3.0, ago=3000),
           _an('absorb', hint='dn', price=100.1, ago=1800, o1=-0.3)]
    e = A.exit_score('LONG', entry_bar, pre, NOW, exh={'pct': 100}, pnl=1.0, peak=1.0, opened_at=oa)
    _check(not e['ready'] and e['parts']['climax'] == 0.0 and e['parts']['flow'] == 0.0,
           f'щойно відкрита угода не закривається через причини входу: {e}')
    nb = dict(entry_bar, t=NOW - 60)                 # нова свічка вже після входу
    _check(A.exit_score('LONG', nb, [], NOW, opened_at=oa)['parts']['climax'] == 1.0,
           'сплеск НОВОЇ свічки після входу — свідчення')
    new = [_an('absorb', hint='dn', price=100.1, ago=120, o1=-0.3)]
    x = A.exit_score('LONG', {'box_hi': 100.0, 'box_lo': 90.0, 'price': 89.0, 'beyond': 'SHORT'},
                     pre + new, NOW, opened_at=oa)
    _check(x['ready'] and x['verdict'] in ('close', 'all'), f'нова аномалія після входу — вихід: {x}')


# ── 3. Калібровка з досліджень ───────────────────────────────────────────
def test_calibration_from_anomaly_research():
    st = [{'kind': 'absorb', 'tf': '1h', 'hint_n': 40, 'hint_pct': 70.0, 'decided': 40, 'reversal_pct': 70.0},
          {'kind': 'side', 'tf': '15m', 'hint_n': 10, 'hint_pct': 90.0},
          {'kind': 'climax', 'tf': '1h', 'decided': 50, 'reversal_pct': 100.0}]
    c = A.calibration(st)
    _check(abs(c['entry']['absorb'] - 1.4) < 1e-9 and c['entry']['absorb|1h'] == 1.4, f'🧽: {c}')
    _check(c['entry']['side'] == 1.0, 'вибірка < 30 → 1.0 (не вигадуємо)')
    _check(c['exit']['climax'] == 1.5, 'межа 1.5 для 🔥 розвороту')
    lo = A.calibration([{'kind': 'absorb', 'tf': '1h', 'hint_n': 100, 'hint_pct': 10.0}])
    _check(lo['entry']['absorb'] == 0.5, 'нижня межа 0.5')
    r1 = A.read_anomalies([_an('absorb', hint='up', price=90.1)], NOW, LV, cal={'absorb': 0.5})
    r2 = A.read_anomalies([_an('absorb', hint='up', price=90.1)], NOW, LV, cal={'absorb': 1.4})
    _check(r2['up']['rev'] > r1['up']['rev'], 'вага аномалії — з досліджень')


# ── 4. Рушій: сигнал ЛИШЕ через спільні ворота ───────────────────────────
class _Sc:
    def __init__(self, ok=True):
        self.ok, self.calls = ok, []
        self._ob_zones, self._cache = {}, {}

    def get_settings(self):
        return {'spring_enabled': True, 'ai_signal_enabled': True, 'ai_min_charge': 60}

    def scan_universe(self):
        return ['XUSDT']

    def spring_for(self, syms):
        return {'XUSDT': dict(_REC)}

    def _dir_paused(self, s, side):
        return ''

    def _signal_allowed(self, s, side, at_intake=False, origin=''):
        self.calls.append((s, side, at_intake, origin))
        return (self.ok, '' if self.ok else 'OB-фільтр', 'розклад')


class _Tm:
    def __init__(self):
        self.sig = []

    def on_signal(self, **kw):
        self.sig.append(kw)
        return 'queued'


class _Db:
    def __init__(self, rows):
        self.rows = rows

    def get_vol_anomalies(self, **kw):
        return [dict(r) for r in self.rows]

    def get_setting(self, k, d=None):
        return d

    def set_setting(self, k, v):
        pass


def _engine(ok=True):
    sc, tm = _Sc(ok), _Tm()
    db = _Db([dict(_an('side', hint='up', o1=0.3), symbol='XUSDT')])
    e = A.AiEngine(db=db, scanner=sc, tm=tm, clock=lambda: NOW)
    e._refresh_cal = lambda now: None
    e._fetch_live = lambda S, now: (dict(_REC), dict(_VW_UP))
    return e, sc, tm


def test_engine_signal_goes_through_shared_gates():
    e, sc, tm = _engine(True)
    e.tick(NOW)
    _check(sc.calls and sc.calls[0][3] == 'ai' and sc.calls[0][2] is True, f'ворота: {sc.calls}')
    _check(tm.sig and tm.sig[0]['opened_by'] == 'ai' and tm.sig[0]['side'] == 'LONG', f'on_signal: {tm.sig}')
    e.tick(NOW + 60)
    _check(len(tm.sig) == 1, 'один сигнал на епізод заряду')
    st = e.state()
    _check(st['coins'] and st['coins'][0]['signal'], 'стан для UI несе сигнал')


def test_engine_rejected_signal_never_opens():
    e, sc, tm = _engine(False)
    e.tick(NOW)
    _check(sc.calls and not tm.sig, 'ворота відхилили → on_signal не кличеться')
    e.tick(NOW + 60)
    _check(len(sc.calls) == 1, 'повтор лише через SIGNAL_RETRY_SEC')


def test_no_direct_open_path():
    tree = ast.parse(_SRC)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    _check('_open' not in names and 'manual_open' not in names and 'hand_open' not in names,
           'ШІ не відкриває угоди напряму — лише on_signal після воріт')


# ── 5. Вихід ШІ у Trade Manager ──────────────────────────────────────────
def _tm_class():
    st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
    sys.modules.setdefault('storage', st)
    db = types.ModuleType('storage.db_operations'); db.get_db = lambda: None
    sys.modules['storage.db_operations'] = db
    for nm, rel in (('detection.signal_labels', 'detection/signal_labels.py'),
                    ('detection.setup_grader', 'detection/setup_grader.py')):
        # Неповний архів користувача може не містити необов'язковий helper;
        # Trade Manager його не імпортує, тож не блокуємо предметний тест виходу.
        if nm not in sys.modules and os.path.exists(os.path.join(_ROOT, rel)):
            _load(nm, rel)
    lg = types.ModuleType('detection.activity_log'); lg.log_activity = lambda *a, **k: None
    sys.modules['detection.activity_log'] = lg
    ffm = types.ModuleType('detection.fuel_filter'); ffm.get_fuel_filter = lambda: None
    sys.modules['detection.fuel_filter'] = ffm
    return _load('detection.trade_manager', 'detection/trade_manager.py').TradeManager


def test_tm_ai_exit_reason():
    TM = _tm_class()
    sys.modules['detection'].ai_signals = A
    combo = [_an('climax', bar_dir='up', pre_chg=3.0, ago=3000),
             _an('absorb', hint='dn', price=100.1, ago=600, o1=-0.3)]
    for a in combo:                       # аномалії «зараз»
        a['ts'] = a['ts'] - NOW + __import__('time').time()
    stub = types.SimpleNamespace(
        _settings={'use_ai_exit': True, 'ai_exit_mode': 'all'}, AI_EXIT_TTL=20.0, scanner=None,
        _exit_key=TM._exit_key,
        _spring_fire_view=lambda s, fetch=False: {'box_hi': 100.0, 'box_lo': 90.0, 'price': 99.0},
        _trade_anomalies=lambda s, a, b: combo,
        trade_exh_for=lambda pos, vw: {})
    pos = {'side': 'LONG', 'entry_price': 95.0}
    hit = TM._ai_exit_reason(stub, 'XUSDT', pos, 99.0, False)
    _check(hit and hit[0] == 'ai_exit', f'вихід ШІ: {hit}')
    _check(pos['ai_exit']['ready'] and pos['ai_exit']['verdict'] == 'scenario'
           and pos['ai_exit']['scenario']['key'] == 'X1'
           and pos['ai_exit']['checklist'], f'знімок на позиції: {pos["ai_exit"]}')
    _check('бал ≥' not in hit[1] and '✓ аномалії' in hit[1], f'причина — вердикт, а не поріг: {hit[1]}')
    _check(TM._ai_exit_reason(stub, 'XUSDT', pos, 99.0, False) is None, 'власний тротл 20с')
    _check("'ai_exit': '🤖 Вихід ШІ" in _TM, 'підпис причини')


# ── 6. Налаштування / UI / мітки ─────────────────────────────────────────
def test_settings_and_ui():
    _check('ai_entry_min' not in _SC and 'sm-ai-entry-min' not in _HTML,
           '17.11: порогу «бал ≥» для входу більше немає — вхід за сценарієм')
    for k in ('ai_signal_enabled', 'ai_min_charge'):
        _check(f"'{k}'" in _SC, f'ключ сканера {k}')
    for k in ('use_ai_exit', 'ai_exit_mode'):
        _check(f"'{k}'" in _TM, f'ключ TM {k}')
    for i in ('sm-ai-signal', 'sm-ai-min-charge', 'tm-use-ai-exit', 'ai-rules-body',
              'tm-ai-exit-mode', 'ai-tbody', 'ai-cal'):
        _check(f'id="{i}"' in _HTML, f'UI {i}')
    _check('function updateAiSignal' in _HTML and 'function aiLoad' in _HTML, 'JS ШІ')
    _check("use_ai_exit: " in _HTML and "ai_exit_mode: " in _HTML, 'збереження виходу ШІ')
    _check('ai_exit_min' not in _TM and 'tm-ai-exit-min' not in _HTML,
           '17.11: порогу «бал ≥» для виходу більше немає')
    _check("@app.route('/api/ai/state')" in _APP and "@app.route('/api/ai/episodes-export')" in _APP, 'API ШІ')
    _check("'ai':" in _SL and 'ai_exit: "🤖 Вихід ШІ"' in _JS, 'мітки ШІ (бот + інфосайт)')


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)):
        try:
            fn()
            print('✓', name)
        except Exception as e:
            fails += 1
            print('✗', name, '—', repr(e))
    print('OK' if not fails else f'FAIL {fails}')
    sys.exit(1 if fails else 0)
