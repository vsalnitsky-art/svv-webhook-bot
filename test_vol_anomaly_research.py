"""⚠ Аномалії обсягу + 🔬 дослідження угод + «Хронологія угоди» (16.11).

Вимоги: «Хронологія угоди» — лише шапка + «Шлях монети»; у «Шлях монети» —
усе про аномалії; вести дослідження «Пружина вистрілила» і «Вихід на
кульмінації»; архівувати все потрібне для навчання бота рішенням.
"""
import ast, json, os, re, subprocess, sys, tempfile, threading, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules['pybit.unified_trading'].HTTP = object
if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
    sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


imb = _load('detection.imbalance', 'detection/imbalance.py')
va = _load('detection.vol_anomaly', 'detection/vol_anomaly.py')
tr = _load('detection.trade_research', 'detection/trade_research.py')

with open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8') as f:
    _HTML = f.read()
with open(os.path.join(_ROOT, 'infosite', 'js', 'app.js'), encoding='utf-8') as f:
    _APP = f.read()
with open(os.path.join(_ROOT, 'detection', 'trade_manager.py'), encoding='utf-8') as f:
    _TMSRC = f.read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _rows(vols, last=None):
    r = [{'v': v, 'buy': None, 'up': True} for v in vols]
    if last:
        r.append(last)
    return r


# ── 1. Детектор = графік кабінету ────────────────────────────────────────
def test_thresholds_mirror_the_chart():
    m = re.search(r'var VOL_AN = \{([^}]*)\}', _APP)
    _check(m, 'VOL_AN у app.js')
    js = dict((k.strip(), float(v)) for k, v in (p.split(':') for p in m.group(1).split(',')))
    for k, v in va.AN.items():
        _check(abs(js[k] - float(v)) < 1e-9, f'поріг {k}: бот {v} ≠ графік {js[k]}')


def test_anomaly_kinds_and_priority():
    base = [100.0] * 20
    a = va.anomaly_at(_rows(base, {'v': 200, 'buy': 0.7, 'up': False}), 20)
    _check(a and a['k'] == 'absorb' and a['by'] == 'sell', f'🧽 продажі-поглинання: {a}')
    a = va.anomaly_at(_rows(base, {'v': 200, 'buy': 0.3, 'up': True}), 20)
    _check(a and a['k'] == 'absorb' and a['by'] == 'buy', f'🧽 покупки-поглинання: {a}')
    a = va.anomaly_at(_rows(base, {'v': 350, 'buy': 0.55, 'up': True}), 20)
    _check(a and a['k'] == 'climax', f'🔥 ×3.5: {a}')
    a = va.anomaly_at(_rows(base, {'v': 160, 'buy': 0.8, 'up': True}), 20)
    _check(a and a['k'] == 'side' and a['dir'] == 'up', f'⚡: {a}')
    a = va.anomaly_at(_rows(base, {'v': 210, 'buy': None, 'up': True}), 20)
    _check(a and a['k'] == 'spike', f'📈 без розкладу: {a}')
    _check(va.anomaly_at(_rows(base, {'v': 120, 'buy': 0.9, 'up': True}), 20) is None, '×1.2 — не аномалія')
    _check(va.anomaly_at(_rows([100.0] * 5, {'v': 900, 'buy': None, 'up': True}), 5) is None,
           'база < 10 свічок → None')


def test_hints_and_verdict():
    _check(va.hint_of('absorb', 'sell', None) == 'dn', '🧽 купували, свічка червона → вниз')
    _check(va.hint_of('absorb', 'buy', None) == 'up', '🧽 продавали, свічка зелена → вгору')
    _check(va.hint_of('side', None, 'dn') == 'dn', '⚡ — бік тиску')
    _check(va.hint_of('climax', None, None) is None, '🔥 без напрямку')
    _check(va.verdict_of({'pre_chg': 2.0, 'o4': -1.0}) == 'reversal', 'розворот')
    _check(va.verdict_of({'pre_chg': 2.0, 'o4': 1.0}) == 'continuation', 'продовження')
    _check(va.verdict_of({'pre_chg': 0.1, 'o4': -1.0}) is None, 'без тренду — без вердикту')


def test_outcome_after_n_bars():
    rows = [{'t': i * 900, 'c': 100.0 + i, 'h': 100.5 + i, 'l': 99.5 + i} for i in range(12)]
    o = va.outcome_of(rows, 0)
    _check(o['o1'] == 1.0 and o['o4'] == 4.0 and o['o8'] == 8.0, f'рух: {o}')
    _check(o['done'] is True and o['hi'] == 8.5, f'макс і done: {o}')
    o = va.outcome_of(rows, 6 * 900)
    _check('o8' not in o and o['done'] is False, 'ще немає 8 свічок → не done')


def test_stats_reversal_and_hint():
    rows = [{'kind': 'absorb', 'tf': '15m', 'hint': 'dn', 'pre_chg': 2, 'o4': -1, 'o8': -2, 'done': True},
            {'kind': 'absorb', 'tf': '15m', 'hint': 'dn', 'pre_chg': 2, 'o4': 1, 'o8': 1, 'done': True},
            {'kind': 'climax', 'tf': '1h', 'hint': None, 'pre_chg': -3, 'o4': 2, 'done': False}]
    st = {(r['kind'], r['tf']): r for r in va.stats(rows)}
    a = st[('absorb', '15m')]
    _check(a['n'] == 2 and a['reversal_pct'] == 50.0 and a['hint_pct'] == 50.0, f'🧽: {a}')
    _check(a['avg4'] == 0.0, f'рух у бік підказки: {a}')
    _check(st[('climax', '1h')]['reversal_pct'] == 100.0, 'кульмінація → розворот')


# ── 2. Рушій: запис, підсвічки лише за потреби, наслідки ─────────────────
class _MD:
    def __init__(self, bars):
        self.bars = bars; self.split_calls = 0
        outer = self

        class _Sess:
            def get(self, url, params=None, timeout=None):
                outer.split_calls += 1
                t0 = int(params['start'])
                lst = [[str(t0 + i * 60000), '100', '101', '99', '101' if i < 12 else '99', '10', '0']
                       for i in range(15)]
                return types.SimpleNamespace(status_code=200, json=lambda: {'result': {'list': lst}})
        self._session = _Sess()

    def fetch_klines(self, sym, limit=60, interval='15m'):
        return self.bars


class _DB:
    def __init__(self):
        self.rows = []; self.upd = []

    def add_vol_anomaly(self, rec):
        self.rows.append(dict(rec, id=len(self.rows) + 1)); return True

    def get_vol_anomalies(self, **kw):
        return [r for r in self.rows if not r.get('done')]

    def update_vol_anomaly(self, rid, fields):
        self.upd.append((rid, fields))
        for r in self.rows:
            if r['id'] == rid:
                r.update(fields)

    def get_setting(self, k, d=None):
        return d


def _bars(n, live_t, last_v=1000.0, up=False):
    out = []
    for i in range(n):
        t = live_t - (n - i) * 900
        v = last_v if i == n - 1 else 1000.0 / 1.0
        o, c = (100.0, 99.0) if (i == n - 1 and not up) else (100.0, 100.5)
        out.append({'t': t * 1000, 'o': o, 'p': c, 'h': 101.0, 'l': 98.0, 'v': v * c, 'q': v})
    out.append({'t': live_t * 1000, 'o': 100, 'p': 100, 'h': 100, 'l': 100, 'v': 1, 'q': 1})
    return out


def test_process_stores_anomaly_with_context_and_split_only_when_needed():
    now = 1_000_000 * 900 + 10
    live = int(now // 900) * 900
    md = _MD(_bars(30, live, last_v=1000.0))
    db = _DB()
    m = va.VolAnomalyMonitor(db=db, md=md, get_watchlist=lambda: ['ETHUSDT'],
                             get_trades=lambda: {'ETHUSDT': [{'book': 'S', 'side': 'LONG', 'entry': 99}]})
    _check(m.process('ETHUSDT', '15m', now) is True, 'бар є → оброблено')
    _check(md.split_calls == 0 and not db.rows, 'звичайний обсяг → ні підсвічок, ні запису')
    md2 = _MD(_bars(30, live, last_v=2000.0, up=False))
    db2 = _DB()
    m2 = va.VolAnomalyMonitor(db=db2, md=md2, get_watchlist=lambda: [],
                              get_trades=lambda: {'ETHUSDT': [{'book': 'S', 'side': 'LONG', 'entry': 99}]})
    m2.process('ETHUSDT', '15m', now)
    _check(md2.split_calls == 1, 'обсяг ×2 → рівно один запит підсвічок')
    _check(len(db2.rows) == 1, f'записано: {db2.rows}')
    r = db2.rows[0]
    _check(r['kind'] == 'absorb' and r['hint'] == 'dn', f'покупці 80% (12 зелених підсвічок), свічка червона → 🧽 вниз: {r}')
    _check(r['ctx']['trades'][0]['side'] == 'LONG' and r['pre_chg'] is not None, 'контекст угоди і тренд до')
    _check(r['ts'] == r['bar_t'] + 900, 'ts = закриття свічки')


def test_bar_not_given_yet_retries():
    now = 1_000_000 * 900 + 10
    live = int(now // 900) * 900
    md = _MD(_bars(30, live - 900)[:-1])     # біржа ще не віддала останній закритий
    m = va.VolAnomalyMonitor(db=_DB(), md=md, get_watchlist=lambda: ['X'], get_trades=lambda: {})
    _check(m.process('X', '15m', now) is None, 'немає бару → повтор')


def test_monitor_never_trades():
    tree = ast.parse(open(os.path.join(_ROOT, 'detection', 'vol_anomaly.py'), encoding='utf-8').read())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for bad in ('on_signal', '_open', 'manual_open', '_close_position', '_close_shadow', 'log_activity'):
        _check(bad not in names, f'рушій аномалій не торгує і не флудить лог: {bad}')


# ── 3. Дослідження угоди ─────────────────────────────────────────────────
def test_research_peak_events_and_summary():
    pos = {'side': 'LONG', 'entry_price': 100.0}
    rs = {}
    vw = {'pct': 40, 'range_x': 1.0, 'vol_x': 1.1, 'parts': {'box': 30}, 'pre_sq': 2, 't': 3600}
    tr.step(rs, 'LONG', 1000, tr.snap_of(pos, 100.5, vw, {'prog': 10, 'pct': 10}), bar_t=3600)
    vw2 = dict(vw, pct=80, range_x=2.1, vol_x=2.2, climax='LONG')
    tr.step(rs, 'LONG', 1400, tr.snap_of(pos, 103.0, vw2, {'prog': 60, 'pct': 60}), bar_t=3600)
    tr.step(rs, 'LONG', 1500, tr.snap_of(pos, 103.0, vw2, {'prog': 60, 'pct': 60}), bar_t=3600)
    tr.step(rs, 'LONG', 1800, tr.snap_of(pos, 101.0, dict(vw, fired={'side': 'SHORT'}), {'prog': 20, 'pct': 20}), bar_t=7200)
    ks = [e['k'] for e in rs['ev']]
    _check(ks.count('climax') == 1, f'кульмінація — один раз на бар: {ks}')
    _check('fire' in ks and 'prog' in ks and 'fp' in ks, f'події: {ks}')
    _check(rs['pk']['pnl'] == 3.0 and rs['pk']['rx'] == 2.1 and rs['pk']['clx'] == 'LONG',
           f'пік із показниками в ту мить: {rs["pk"]}')
    _check(len(rs['s']) == 3, f'зрізи раз на 5 хв (1000/1400/1800): {len(rs["s"])}')
    sm = tr.summary('LONG', 100.0, 101.0, 1.0, rs,
                    [{'kind': 'absorb', 'hint': 'dn', 'price': 102.0, 'ts': 1700, 'tf': '15m'}])
    ia = sm['if_exit_at']
    _check(ia['climax_with']['pnl'] == 3.0 and ia['climax_with']['vs_exit'] == 2.0, f'якби вийти: {ia}')
    _check(ia['fire_against']['pnl'] == 1.0, 'вистріл проти')
    _check(ia['anom_absorb_against']['pnl'] == 2.0, f'аномалія проти угоди: {ia}')
    _check(sm['peak_share'] == 33.3, f'вихід узяв частку піку: {sm}')


def test_samples_are_bounded():
    rs = {}
    pos = {'side': 'SHORT', 'entry_price': 100.0}
    for i in range(tr.SAMPLES_MAX + 5):
        tr.step(rs, 'SHORT', i * tr.SAMPLE_SEC, tr.snap_of(pos, 99.0, {}, {}))
    _check(len(rs['s']) <= tr.SAMPLES_MAX, 'зрізи проріджуються')


def test_tm_wiring():
    _check(_TMSRC.count('self._research_tick(symbol, pos, current_price,') == 2, 'обидва монітори')
    _check(_TMSRC.count('self._research_attach(closed, pos)   # 🔬 до локу') == 3, 'усі три закриття, поза локом')
    _check('self._research_archive(closed, pos, is_paper)' in _TMSRC, 'архів у БД')
    a = _TMSRC.index('def _sample_trade_logs'); b = _TMSRC.index('def _trade_journey')
    body = _TMSRC[a:b]
    for bad in ('compute_decision', '_ctr_snapshot', 'get_fuel_strength_map', 'exit_grade'):
        _check(bad not in body, f'хронологія більше не аналізує: {bad}')
    _check("'research', 'research_summary'" in _TMSRC, 'дослідження не їде в /api/tm/state')


def test_journey_full_merges_everything():
    tmmod = sys.modules.get('detection.trade_manager')
    if tmmod is None:
        sys.modules.setdefault('detection.activity_log', types.ModuleType('detection.activity_log'))
        sys.modules['detection.activity_log'].log_activity = lambda *a, **k: None
        sys.modules['detection.activity_log'].get_activity_log = lambda: None
        st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
        sys.modules.setdefault('storage', st)
        dbm = types.ModuleType('storage.db_operations'); dbm.get_db = lambda: None
        sys.modules.setdefault('storage.db_operations', dbm)
        ffm = types.ModuleType('detection.fuel_filter'); ffm.get_fuel_filter = lambda: None
        sys.modules.setdefault('detection.fuel_filter', ffm)
        _load('detection.signal_labels', 'detection/signal_labels.py')
        _load('detection.setup_grader', 'detection/setup_grader.py')
        tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
    ns = types.SimpleNamespace(
        _trade_journey=lambda s, a, b: [{'t': 2000, 'event': 'opened', 'detail': 'x'}],
        _trade_anomalies=lambda s, a, b: [{'ts': 2500, 'kind': 'absorb', 'hint': 'dn', 'tf': '15m',
                                            'rv': 2.0, 'buy_share': 0.7, 'price': 99.0}])
    rec = {'side': 'LONG', 'entry_price': 100.0, 'opened_at': 2000,
           'research': {'ev': [{'t': 2600, 'k': 'climax', 'dir': 'LONG', 'rel': 'with', 'pnl': 1.0}]},
           'research_summary': {'pnl': 0.5}}
    j = tmmod.TradeManager._journey_full(ns, 'ETHUSDT', rec, 3000)
    ev = [e['event'] for e in j]
    _check(ev == ['opened', 'anomaly', 'research', 'rsummary'], f'порядок: {ev}')
    an = j[1]['x']
    _check(an['rel'] == 'against' and an['pnl_then'] == -1.0, f'аномалія відносно угоди: {an}')


# ── 4. Модалка: лише шапка + «Шлях монети» ───────────────────────────────
def _fn(name):
    i = _HTML.index('function ' + name + '(')
    j = _HTML.index('\n}\n', i)
    return _HTML[i:j + 2]


def test_modal_is_header_plus_journey_only():
    body = _fn('renderTradeHistoryChart')
    _check('return info + info2 + exitReasonLine + sltpLine + paperLine + journeyPanel;' in body,
           'лише шапка + шлях')
    for gone in ('pnlPanel', 'ctrStcPanel', 'revPanel', 'mmPanel', 'bannerStrip', 'Decision Center на вході',
                 'function panel('):
        _check(gone not in body, f'прибрано: {gone}')


def test_journey_renders_anomaly_and_summary_in_node():
    consts = ''.join(re.findall(r'(const _JR_[A-Z_]+ = \{.*?\};\n)', _HTML, re.S))
    code = ("function fmtPriceJS(v){return String(v);}\n" + consts + _fn('_jrPct') + _fn('_jrChip')
            + _fn('_jrIndChips') + _fn('journeyTree') + """
const j = [
 {t: 1000, event: 'anomaly', source: 'VOL 15M', detail: 'опис', x: {kind:'absorb', hint:'dn', rel:'against', rv:2.1, buy_share:0.7, o1:-0.3, o4:-1.2, hi:0.2, lo:-1.5, verdict:'reversal', done:true, price: 99}},
 {t: 2000, event: 'research', detail: '🎯 Кульмінація ▲ за угодою', x: {k:'climax', rel:'with', pnl: 1.5, rx: 2.1}},
 {t: 3000, event: 'rsummary', x: {peak: {pnl: 2, t: 1500, rx: 2.2}, peak_share: 50, giveback: 1, if_exit_at: {climax_with: {pnl: 1.5, vs_exit: 0.5}, anom_absorb_against: {pnl: -1, vs_exit: -2, tf: '15m'}}}}];
console.log(journeyTree(j, {opened_at: 500}));
""")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(code)
    out = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=20)
    _check(out.returncode == 0, out.stderr)
    h = out.stdout
    for s in ('🧽 Поглинання ▼', '⚠ ПРОТИ угоди', 'Наслідок', '↩ розворот', '×2.1',
              'Показники НА ПІКУ', 'Якби вийти на першій події', 'кульмінація за угодою',
              '⚠ аномалій 1 · 🔬 подій 1'):
        _check(s in h, f'у «Шляху монети» є: {s}')


def test_ui_panel_and_routes():
    for s in ("togglePanel('vanom')", 'vanomLoad', '/api/vol-anomaly/state', '/api/tm/research-export'):
        _check(s in _HTML, f'панель: {s}')
    app = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    for r in ("/api/vol-anomaly/state", "/api/vol-anomaly/settings", "/api/vol-anomaly/export",
              "/api/tm/research-export", 'init_vol_anomaly('):
        _check(r in app, f'маршрут / старт: {r}')
    models = open(os.path.join(_ROOT, 'storage', 'db_models.py'), encoding='utf-8').read()
    _check("vol_anomaly'" in models and "trade_research'" in models, 'таблиці БД')


if __name__ == '__main__':
    fails = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            try:
                v(); print('ok  ', k)
            except Exception as e:
                fails += 1; print('FAIL', k, '—', e)
    print('fails:', fails)
    sys.exit(1 if fails else 0)
