"""🔔 Session Band Reversal (SBR) — сигнали 1H/4H, книга МММ-SBR, TG «🔔 Сигнали».

Запуск: python3 test_sbr_signals.py
"""
import importlib.util
import json
import os
import re
import sys
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'sbr_signals_t', os.path.join(_ROOT, 'detection', 'sbr_signals.py'))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _fixture():
    rows = json.load(open(os.path.join(_ROOT, 'test_sbr_fixture_eth4h.json')))
    return [{'t': r[0], 'o': r[1], 'h': r[2], 'l': r[3], 'p': r[4]} for r in rows]


def _bar(t_ms, o, h, l, c):
    return {'t': t_ms, 'o': o, 'h': h, 'l': l, 'p': c}


# ── 1. Pine-семантика ────────────────────────────────────────────────────────
def test_pine_ema_and_rma_match_definitions():
    src = [1.0, 2, 3, 4, 5, 6, 7]
    e = S.pine_ema(src, 3)
    _check(e[:2] == [None, None] and abs(e[2] - 2.0) < 1e-12, f'старт EMA = SMA: {e}')
    _check(abs(e[3] - (0.5 * 4 + 0.5 * 2.0)) < 1e-12, 'alpha EMA = 2/(len+1)')
    r = S.pine_rma(src, 3)
    _check(abs(r[3] - (4 / 3 + 2.0 * 2 / 3)) < 1e-12, 'alpha RMA = 1/len')
    atr = S.pine_atr([3, 5, 4], [1, 2, 1], [2, 4, 3], 2)
    # TR: 2, max(3,3,0)=3, max(3,0,3)=3 → RMA2: (2+3)/2=2.5 → 0.5*3+0.5*2.5
    _check(atr[1] == 2.5 and abs(atr[2] - 2.75) < 1e-12, f'ATR = RMA(TR): {atr}')


# ── 2. Опорні точки оригіналу (ETH 4H) ───────────────────────────────────────
def _sigs_by_kyiv():
    out = {}
    for x in S.compute_signals(_fixture(), 14400):
        out[S.kyiv_dt(x['t'] / 1000).strftime('%d.%m %H:%M')] = x
    return out


def test_reference_signals_of_the_original():
    g = _sigs_by_kyiv()
    a = g.get('20.10 19:00')
    _check(a and a['side'] == 'SHORT' and abs(a['trigger'] - 2455.85) < 1e-9
           and a['n'] == 1, f'20.10 19:00 SHORT: {a}')
    b = g.get('21.10 19:00')
    _check(b and b['side'] == 'SHORT' and abs(b['trigger'] - 2527.63) < 1e-9, f'21.10: {b}')
    c = g.get('24.10 19:00')
    _check(c and c['side'] == 'LONG' and abs(c['trigger'] - 2293.0) < 1e-9
           and c['score'] == 100.0 and c['risk'] == '1R + Add', f'24.10: {c}')
    # Score пропорційний пробою: 76.39 / 97.75 (OKX), у оригіналі 76.27 / 97.70.
    _check(abs(a['score'] - 76.39) < 0.01 and abs(b['score'] - 97.75) < 0.01,
           f'Score {a["score"]} / {b["score"]}')
    # 22.10 19:00 — пробій зі Score 100, але свічка ЧЕРВОНА → мітки немає.
    _check('22.10 19:00' not in g, 'червона свічка не дає LONG')


def test_sl_is_close_plus_atr_of_the_bar():
    # Калібровка 04.11: SL = close ± 3.7·ATR14 поточного бару. Еталон (інша
    # біржа): 20.10 2582.65 · 21.10 2634.52; на барах OKX — у межах 0.5%.
    g = _sigs_by_kyiv()
    for k, ref in (('20.10 19:00', 2582.65), ('21.10 19:00', 2634.52)):
        sl = g[k]['sl']
        _check(abs(sl - ref) / ref < 0.005, f'{k}: SL {sl:.2f} проти еталона {ref}')
    _check(S.DEFAULTS['sl_k'] == 3.7, 'sl_k = 3.7')


def test_history_start_does_not_change_the_signal():
    bars = _fixture()
    tail = bars[60:]
    a = {x['t']: x for x in S.compute_signals(bars, 14400)}
    b = {x['t']: x for x in S.compute_signals(tail, 14400)}
    for t, x in a.items():
        if t < tail[100]['t']:
            continue
        y = b.get(t)
        # Навіть на 100 барах розгону різниця SL — соті частки долара; на
        # робочих 1000 барах вона ~1e-30 (затухання (13/14)^N).
        _check(y and y['side'] == x['side'] and y['trigger'] == x['trigger']
               and abs(y['sl'] - x['sl']) < 0.05,
               f'рекурсивні EMA/ATR сходяться: старт історії не змінює сигналу {x} {y}')


def test_live_bar_is_ignored():
    bars = _fixture()
    last = bars[-1]
    now = last['t'] / 1000 + 14400 - 60        # останній бар ще живий
    cb = S.closed_bars(bars, 14400, now)
    _check(len(cb) == len(bars) - 1, 'живий бар відкинуто')


# ── 3. Один рядок на монету ──────────────────────────────────────────────────
def _sig(side, close_t, score=70.0, n=1):
    return {'side': side, 't': int((close_t - 3600) * 1000), 'close_t': close_t,
            'trigger': 1.0, 'score': score, 'risk': '1R', 'sl': 0.9, 'n': n,
            'time_kyiv': '19:00'}


def test_combine_tf_column():
    c = S.combine(_sig('LONG', 1000, 60), _sig('LONG', 2000, 80))
    _check(c['side'] == 'LONG' and c['tf_label'] == '1H+4H' and c['strength'] == 80,
           f'1H потім 4H у той самий бік — ОДИН рядок «1H+4H»: {c}')
    c = S.combine(_sig('SHORT', 3000), _sig('LONG', 2000))
    _check(c['side'] == 'SHORT' and c['tf_label'] == '1H', f'найсвіжіший вирішує: {c}')
    c = S.combine(_sig('SHORT', 2000), _sig('LONG', 2000))
    _check(c['side'] == 'LONG' and c['tf_label'] == '4H', 'одночасно — вирішує 4H')
    _check(S.combine(None, None) is None, 'немає сигналів — None')


# ── 4. Рушій: свіжий сигнал → Telegram, без дублів ───────────────────────────
class _DB:
    def __init__(self):
        self.d = {}

    def get_setting(self, k, default=None):
        return self.d.get(k, default)

    def set_setting(self, k, v):
        self.d[k] = json.loads(json.dumps(v))


def _series(tf_sec, n, end_open, signal_last=None):
    """Рівні бари + за бажанням сигнальний останній бар (SHORT/LONG)."""
    out = []
    t0 = end_open - (n - 1) * tf_sec
    for i in range(n):
        t = t0 + i * tf_sec
        out.append(_bar(t * 1000, 100.0, 100.5, 99.5, 100.0 + (0.01 if i % 2 else -0.01)))
    if signal_last == 'SHORT':
        out[-1] = _bar(out[-1]['t'], 101.0, 104.0, 100.6, 100.9)
    elif signal_last == 'LONG':
        out[-1] = _bar(out[-1]['t'], 99.0, 99.4, 96.0, 99.1)
    return out


def _engine(now, sig1=None, sig4=None, tg=('BTCUSDT',)):
    sent = []
    clock = {'t': now}

    def fetch(sym, tf, limit):
        tf_sec = S.TF_SEC[tf]
        last_closed_open = int(clock['t'] // tf_sec) * tf_sec - tf_sec
        bars = _series(tf_sec, 300, last_closed_open, sig1 if tf == '1h' else sig4)
        bars.append(_bar((last_closed_open + tf_sec) * 1000, 100, 100, 100, 100))   # живий
        return bars

    db = _DB()
    db.d[S.DB_KEY_SETTINGS] = {'tg_symbols': list(tg)}
    e = S.SbrEngine(db, lambda: ['BTCUSDT', 'SOLUSDT'], fetch,
                    lambda text: (sent.append(text) or True, ''), clock=lambda: clock['t'])
    return e, sent, clock, db


def test_fresh_signal_goes_to_telegram_once():
    now = 14400 * 500 + 3600 * 2 + 5          # 1H закрився, 4H — ні
    e, sent, clock, db = _engine(now, sig1='SHORT')
    e.tick()
    _check(len(sent) == 1 and 'BTCUSDT' in sent[0] and 'SHORT' in sent[0] and '1H' in sent[0],
           f'свіжий 1H SHORT відмічених монет → TG: {sent}')
    _check(not any('SOLUSDT' in x for x in sent), 'не відмічена монета в TG не йде')
    rows = e.rows()
    _check(rows['SOLUSDT']['combo']['side'] == 'SHORT', 'але в книзі вона є')
    clock['t'] += 30
    e.tick()
    _check(len(sent) == 1, 'той самий бар вдруге не шлеться')
    # Рестарт: позначка «надіслано» персиститься.
    e2 = S.SbrEngine(db, lambda: ['BTCUSDT'], e._fetch, lambda t: (sent.append(t) or True, ''),
                     clock=lambda: clock['t'])
    e2._state = {}
    e2.tick()
    _check(len(sent) == 1, 'після рестарту дубля немає')


def test_1h_and_4h_on_one_close_make_one_message():
    now = 14400 * 500 + 5                      # закрились і 1H, і 4H
    e, sent, clock, _ = _engine(now, sig1='LONG', sig4='LONG')
    e.tick()
    _check(len(sent) == 1 and '1H+4H' in sent[0], f'один запис 1H+4H: {sent}')
    _check(e.rows()['BTCUSDT']['combo']['tf_label'] == '1H+4H', 'колонка TF = 1H+4H')


def test_late_start_is_a_silent_base():
    now = 14400 * 500 + 3600 * 2 + 3600 * 0.5   # пів години після закриття 1H
    e, sent, _, _ = _engine(now, sig1='SHORT')
    e.tick()
    _check(not sent, 'бот лежав на закритті — старий сигнал не спамимо')


def test_tg_symbols_default_and_toggle():
    db = _DB()
    e = S.SbrEngine(db, lambda: [], None, None)
    _check(e.get_settings()['tg_symbols'] == ['BTCUSDT', 'ETHUSDT'], 'дефолт BTC+ETH')
    e.set_tg_symbol('solusdt', True)
    e.set_tg_symbol('ETHUSDT', False)
    _check(db.d[S.DB_KEY_SETTINGS]['tg_symbols'] == ['BTCUSDT', 'SOLUSDT'], 'зберігається в БД')


def test_tg_text_like_the_tv_label():
    sig = {'side': 'SHORT', 't': 0, 'close_t': 0, 'trigger': 2455.85, 'score': 76.39,
           'risk': '1R', 'sl': 2576.36, 'n': 1, 'time_kyiv': '19:00'}
    t = S.tg_text('ETHUSDT', [('4h', sig)])
    lines = t.split('\n')
    _check(lines == [
        '🔔 <b>SBR</b> · 🔴 <b>SHORT #1</b>',
        '<b>#ETHUSDT</b> · <code>4H</code> · 📊 <b>76.39%</b>',
        '🎯 Trigger: <code>2,455.85</code>',
        '🛑 SL: <code>2,576.36</code>',
    ], f'формат 30.10: {lines}')
    # 1H+4H на одному закритті — ОДНЕ повідомлення, рівні старшого TF, Score більший
    s1 = dict(sig, score=80.0, trigger=2450.0, sl=2500.0, n=3)
    t2 = S.tg_text('ETHUSDT', [('4h', sig), ('1h', s1)])
    _check('<code>1H+4H</code>' in t2 and '<b>80.00%</b>' in t2
           and '2,455.85' in t2 and 'SHORT #1' in t2 and t2.count('\n') == 3, t2)


# ── 5. Книга МММ-SBR у fuel_filter ───────────────────────────────────────────
def _ff_src():
    return open(os.path.join(_ROOT, 'detection', 'fuel_filter.py'), encoding='utf-8').read()


def test_book_direction_rule():
    src = _ff_src()
    a = src.index('def sbr_mm_dir(')
    b = src.index('\ndef ', a + 10)
    import typing
    ns = {'Optional': typing.Optional, 'Dict': typing.Dict}
    exec(src[a:b], ns)
    f = ns['sbr_mm_dir']
    row = {'1h': {'last': _sig('LONG', 2000, 81)}, '4h': {'last': None},
           'combo': S.combine(_sig('LONG', 2000, 81), None), 'fresh': {'1h': True}}
    c = f(row)
    _check(c['status'] == 'LONG' and c['strength'] == 81 and c['sbr_tfs'] == '1H'
           and c['sbr_fresh'], f'напрямок = бік сигналу, сила = Score: {c}')
    _check(f({'1h': None, '4h': None}) is None, 'немає даних → None')
    z = f({'1h': {'last': None}, '4h': None, 'combo': None})
    _check(z['status'] is None and z['strength'] == 0, 'без сигналів → ⚖ сила 0')


def test_book_is_wired_everywhere():
    src = _ff_src()
    for needle in ("'sbr': '_mms'", "'mm_sbr_enabled': True", "book='sbr'",
                   "'mms': _want_mms", "'mm_sbr':", "'MMS'"):
        _check(needle in src, f'fuel_filter: {needle}')
    _check("'mms'" in src[src.index('HEAVY_SECTIONS'):src.index('HEAVY_SECTIONS') + 200],
           'секція mms важка')


def test_ui_panel_columns_and_watchlist_button():
    a = _HTML.index('id="mms-panel"')
    blk = _HTML[a:_HTML.index('id="mms-tbody"', a)]
    th = len(re.findall(r'<th[\s>]', blk))
    m = re.search(r"_MMB\.sbr\.cols = \[([^\]]*)\]", _HTML)
    cols = [x.strip().strip("'") for x in m.group(1).split(',')]
    _check(th == len(cols) + 3, f'<th> {th} = cols {len(cols)} + TRD/☑/символ')
    for k in cols:
        _check(re.search(rf"\b{k}:\s+r =>", _HTML), f'комірка {k}')
    _check('📶 TF' in blk and '1H+4H' in blk, 'колонка «який сигнал прийшов»')
    _check('${sbrFlagHTML(sym)}' in _HTML and "fetch('/api/sbr/tg-symbol'" in _HTML,
           'кнопка 🔔 у рядку WATCHLIST')
    _check("_served.mms && d.mm_sbr" in _HTML and "want.push('mms')" in _HTML, 'полл секції')


def test_tg_category_and_routes():
    tg = open(os.path.join(_ROOT, 'web', 'tg_bot.py'), encoding='utf-8').read()
    _check("'sbr':      '🔔 Сигнали'" in tg and "'#Сигнали'" in tg, 'категорія «🔔 Сигнали»')
    app = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    for r in ('/api/sbr/state', '/api/sbr/settings', '/api/sbr/tg-symbol'):
        _check(f"@app.route('{r}'" in app, r)
    _check('init_sbr_engine' in app, 'рушій стартує з застосунком')
    _check(S.TG_CATEGORY == 'sbr', 'рушій шле саме в цю тему')


def test_bars_without_economy():
    _check(S.BARS == 1000 and S.TFS == ('1h', '4h'), '1000 барів на кожен TF')


def test_js_cells_render():
    """РЕАЛЬНИЙ JS комірок «📶 TF» і «🔔 Сигнал» під node."""
    import subprocess, tempfile
    def fn(name):
        a = _HTML.index('function ' + name + '(')
        return _HTML[a:_HTML.index('\n}', a) + 2]
    esc = _HTML[_HTML.index('const _mmEsc'):]
    esc = esc[:esc.index(';\n') + 2]
    pc = _HTML[_HTML.index('function _pz('):_HTML.index('function _pcTick(')]
    js = ('const fmtPriceJS = p => String(p);\n' + esc + '\n' + pc + fn('_sbrCt') + '\n' + fn('_sbrTfTip')
          + '\n' + fn('_mmSbrTfCell') + '\n' + fn('_mmSbrSigCell') + r"""
const g = {t: 1792857600000, close_t: 1792872000, trigger: 2293, score: 100, risk: '1R + Add', sl: 2184.43, n: 1, time_kyiv: '07:00'};
console.log(_mmSbrTfCell({mm:'LONG', sbr_tfs:'1H+4H', sbr_fresh:true, sbr_by_tf:{'1h':g,'4h':g}}));
console.log('|'); console.log(_mmSbrSigCell({mm:'LONG', sbr_sig:g, sbr_by_tf:{'4h':g}}));
console.log('|'); console.log(_mmSbrTfCell({}));
""")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30,
                           env=dict(os.environ, TZ='Europe/Kyiv'))
    finally:
        os.unlink(path)
    p = r.stdout.split('|')
    _check(len(p) == 3, r.stderr)
    # 🕐 Час бару — годинником ПК (тут TZ процесу = Kyiv), а не серверне time_kyiv:
    # 1792857600 = 24.10.2026 16:00 UTC = 19:00 Kyiv (+3); серверне time_kyiv
    # навмисно інше ('07:00') — сторінка його більше не читає.
    # Показуємо момент СПРАЦЮВАННЯ (close_t = 20:00 UTC = 23:00 Kyiv), а не
    # відкриття бару 19:00 — той самий час, що в Telegram (вимога 10.11).
    _check('24.10 23:00' in p[1] and '24.10 19:00' not in p[1] and '07:00' not in p[1], p[1])
    _check('1H+4H' in p[0] and '🆕' in p[0], p[0])
    _check('LONG #1' in p[1] and '2293' in p[1] and '100.00' in p[1]
           and '1R + Add' in p[1] and 'SL 2184.43' in p[1], p[1])
    _check('—' in p[2], p[2])


# ── 6. Вибір TF (банер + Telegram), ворота і TG банера (29.10) ────────────────
def _ff_fn(name):
    src = _ff_src()
    a = src.index('def ' + name + '(')
    b = src.index('\ndef ', a + 10)
    import typing
    ns = {'Optional': typing.Optional, 'Dict': typing.Dict,
          '__file__': os.path.join(_ROOT, 'detection', 'fuel_filter.py')}
    exec(src[a:b], ns)
    return ns[name]


def test_banner_tf_choice():
    tfs_of = _ff_fn('sbr_tfs_of')
    _check(tfs_of({}) == ('1h', '4h'), 'дефолт — обидва TF')
    _check(tfs_of({'mms_tf_1h': False}) == ('4h',), 'лише 4H')
    _check(tfs_of({'mms_tf_1h': False, 'mms_tf_4h': False}) == ('1h', '4h'),
           'обидва вимкнені → обидва')
    f = _ff_fn('sbr_mm_dir')
    row = {'1h': {'last': _sig('SHORT', 3000, 90)}, '4h': {'last': _sig('LONG', 2000, 70)},
           'combo': S.combine(_sig('SHORT', 3000, 90), _sig('LONG', 2000, 70))}
    _check(f(row)['status'] == 'SHORT', 'обидва TF — найсвіжіший 1H SHORT')
    c = f(row, ('4h',))
    _check(c['status'] == 'LONG' and c['sbr_tfs'] == '4H' and c['strength'] == 70,
           f'лише 4H — 1H не враховується: {c}')
    _check(f({'1h': {'last': _sig('LONG', 1)}, '4h': None}, ('4h',)) is None,
           'обраного TF немає → «невідомо»')


def test_tg_tf_choice():
    _check(S.tg_tfs_of({'tg_4h': False}) == ('1h',), 'лише 1H')
    _check(S.tg_tfs_of({'tg_1h': False, 'tg_4h': False}) == S.TFS, 'обидва вимкнені → обидва')
    now = 14400 * 500 + 3600 * 2 + 5          # закрився лише 1H
    e, sent, clock, db = _engine(now, sig1='SHORT')
    e.update_settings({'tg_1h': False})
    e.tick()
    _check(not sent, f'1H вимкнено для TG → сигнал 1H не шлеться: {sent}')
    _check(e.rows()['BTCUSDT']['combo']['side'] == 'SHORT', 'але в книзі він є')


def test_gates_and_banner_tg_wired():
    sc = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    _check("('mms_gate_enabled', 'mms_gate_banner', 'sbr')" in sc, 'пара воріт у MM_GATES')
    _check("'sbr': ('🔔МММ-SBR', '🔔 МММ-SBR')" in sc, 'підписи воріт')
    ff = _ff_src()
    _check("'key': 'mms_bias_tg'" in ff and "book='sbr')" in ff, 'TG банера SBR')
    for i in ('sm-mms-gate', 'sm-mms-gate-banner', 'ff-mms-bias-tg', 'mms-tf-1h',
              'mms-tf-4h', 'mms-tg-tfs'):
        _check(f'id="{i}"' in _HTML, f'контрол {i}')
    _check("'mms_gate_enabled', 'mms_gate_banner']" in _HTML, 'ворота в _SM_MM_GATES')
    _check("mms_tf_1h: _c('mms-tf-1h')" in _HTML and "mms_bias_tg: _c('ff-mms-bias-tg')" in _HTML,
           'збереження')
    _check("tg_tfs: tfs" in _HTML and 'data-ttf="1d"' in _HTML, 'TG TF (усі) у sbrSaveTg')
    _check("'pchg'" not in re.search(r"_MMB\.sbr\.cols = \[[^\]]*\]", _HTML).group(0),
           'колонку «Рух» прибрано')


# ── 🔒 ворота SBR не судять за старим станом (кейс OKBUSDT 31.10) ──
def test_stale_state_is_fetched_for_one_coin_and_signal_reaches_tg():
    now = 14400 * 500 + 3600 * 2 + 5          # 1H щойно закрився
    e, sent, clock, db = _engine(now - 3600, sig1=None)
    e.tick()                                   # стан на ПОПЕРЕДНЬОМУ барі
    clock['t'] = now
    e._fetch_sig = 'SHORT'
    _orig = e._fetch

    def fetch(sym, tf, limit):
        tf_sec = S.TF_SEC[tf]
        last_closed_open = int(clock['t'] // tf_sec) * tf_sec - tf_sec
        bars = _series(tf_sec, 300, last_closed_open, 'SHORT' if tf == '1h' else None)
        bars.append(_bar((last_closed_open + tf_sec) * 1000, 100, 100, 100, 100))
        return bars
    e._fetch = fetch
    _check(not e.is_current('BTCUSDT', ('1h',)), 'одразу після закриття бару стан застарілий')
    _check(e.ensure_current('BTCUSDT', ('1h',)) is True, 'бар монети дотягнуто')
    _check(e.is_current('BTCUSDT', ('1h',)), 'тепер актуальний')
    _check(e.rows()['BTCUSDT']['combo']['side'] == 'SHORT', 'новий SHORT у стані')
    _check(len(sent) == 1 and 'SHORT' in sent[0], f'свіжий сигнал пішов у TG: {sent}')
    _check(e.ensure_current('BTCUSDT', ('1h',)) is False, 'повторно біржу не питаємо')
    e.tick()
    _check(len(sent) == 1, 'прохід tick не дублює сигнал')
    _check(_orig is not None, '')


def test_gate_rejects_while_sbr_state_is_stale():
    src = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    a = src.index('_MM_GATE_TAG')
    b = src.index('\ndef mm_gate_decide(')
    c = src.index('\n# 🧮 ЗАГАЛЬНІ МММ-ВОРОТА')
    ns = {}
    exec(src[src.index('_MM_GATE_TAG = '):b] + src[b:c], ns)
    f = ns['mm_gate_decide']
    view = {'on': True, 'dir': 'LONG', 'coin': 'LONG', 'strength': 90, 'str_min': 0}
    _check(f(view, 'LONG', True, 'sbr')[1] is True, 'актуальний стан LONG пропускає LONG')
    app, ok, chip, why = f({**view, 'stale': True}, 'LONG', True, 'sbr')
    _check(app and not ok and 'оновлюється' in chip, f'застарілий стан — не пропускаємо: {chip}')
    _check(a > 0, '')
    ff = open(os.path.join(_ROOT, 'detection', 'fuel_filter.py'), encoding='utf-8').read()
    g = ff[ff.index('def mm_gate_view('):ff.index('def symbols_in_work(')]
    _check("_bk == 'sbr'" in g and '_sbr_gate_fresh' in g and "out['stale']" in g,
           'mm_gate_view звіряє актуальність SBR')
    h = ff[ff.index('def _sbr_gate_fresh('):ff.index('def symbols_in_work(')]
    _check('ensure_current' in h and 'on_sbr_update' in h, 'дотягує бар і перебудовує книгу')


# ── ⏰ БУДИЛЬНИК SBR → головна кнопка напрямку (вимога 01.11) ──────────────────
def _asig(side, score=80.0):
    return {'side': side, 'score': score, 'trigger': 1.0, 't': 0, 'close_t': 0, 'n': 1, 'sl': 0.9}


def test_alarm_match_rule():
    st = {'alarm_on': True, 'alarm_symbol': 'btc', 'alarm_tf': 'any', 'alarm_side': 'LONG'}
    _check(S.alarm_norm_symbol('btc') == 'BTCUSDT' and S.alarm_norm_symbol('') == 'BTCUSDT'
           and S.alarm_norm_symbol('eth-usdt') == 'ETHUSDT', 'нормалізація монети')
    h = S.alarm_match(st, 'BTCUSDT', [('4h', _asig('LONG')), ('1h', _asig('LONG'))])
    _check(h and h[0] == '1h', f'будь-який TF → молодший першим: {h}')
    _check(S.alarm_match(st, 'BTCUSDT', [('1h', _asig('SHORT'))]) is None, 'інший бік')
    _check(S.alarm_match(st, 'ETHUSDT', [('1h', _asig('LONG'))]) is None, 'інша монета')
    _check(S.alarm_match({**st, 'alarm_tf': '4h'}, 'BTCUSDT', [('1h', _asig('LONG'))]) is None,
           'лише 4H — сигнал 1H не будить')
    _check(S.alarm_match({**st, 'alarm_on': False}, 'BTCUSDT', [('1h', _asig('LONG'))]) is None,
           'головний тумблер вимкнено — нічого')


class _MemDB:
    def __init__(self): self.d = {}
    def get_setting(self, k, d=None): return self.d.get(k, d)
    def set_setting(self, k, v): self.d[k] = v


def test_alarm_enables_the_main_button_and_keeps_working():
    calls = []
    dg = types.ModuleType('detection.direction_gate')
    state = {'LONG': False}

    def live_enable(side, source=''):
        calls.append(side)
        if state.get(side):
            return True, 'вже увімкнено'
        state[side] = True
        return True, 'увімкнено'
    dg.live_enable = live_enable
    pkg = sys.modules.setdefault('detection', types.ModuleType('detection'))
    old = sys.modules.get('detection.direction_gate')
    sys.modules['detection.direction_gate'] = dg
    pkg.direction_gate = dg
    sent = []
    group = []
    try:
        e = S.SbrEngine(_MemDB(), lambda: [], None, lambda t: (group.append(t) or True, ''))
        e._admin_notify = lambda t: sent.append(t) or True
        e.update_settings({'alarm_on': True, 'alarm_symbol': 'BTC', 'alarm_tf': 'any',
                           'alarm_side': 'LONG'})
        _check('BTCUSDT' in e._universe(), 'монета будильника рахується рушієм')
        e._on_signals('BTCUSDT', [('1h', _asig('SHORT'))], 100.0)
        _check(calls == [], 'SHORT не будить LONG-будильник')
        e._on_signals('BTCUSDT', [('1h', _asig('LONG'))], 200.0)
        st = e.get_settings()
        _check(calls == ['LONG'] and state['LONG'], 'кнопку LONG увімкнено')
        _check(st['alarm_on'] is True, 'головний тумблер лишається увімкненим (не одноразовий)')
        _check(st['alarm_last']['ok'] and st['alarm_last']['side'] == 'LONG', st['alarm_last'])
        _check(any('Будильник SBR' in x for x in sent), 'повідомлення особисто адміну')
        _check(not any('Будильник' in x for x in group), 'у групу будильник НЕ йде')
        _check(not any('кнопку' in x.lower() for x in sent), 'без рядка про головну кнопку')
        n = len(sent)
        e._on_signals('BTCUSDT', [('4h', _asig('LONG'))], 300.0)
        _check(len(sent) == n, 'кнопка вже увімкнена → без повторного TG')
        e.update_settings({'alarm_on': False})
        e._on_signals('BTCUSDT', [('1h', _asig('LONG'))], 400.0)
        _check(calls == ['LONG', 'LONG'], 'вимкнений будильник нічого не робить')
    finally:
        if old is not None:
            sys.modules['detection.direction_gate'] = old
            pkg.direction_gate = old


def test_alarm_ui_and_route():
    # 16.11: випадайку TF замінили чипи (кілька TF) + вибір дії над TM.
    for k in ('id="sbr-alarm-on"', 'id="sbr-alarm-sym"', 'id="sbr-alarm-tfs"',
              'id="sbr-alarm-side"', 'id="sbr-alarm-tm"', 'id="sbr-alarm-status"',
              'function sbrSaveAlarm(', 'function _sbrApplyAlarm('):
        _check(k in _HTML, k)
    i = _HTML.index('id="sbr-alarm-box"')
    _check(_HTML.index('id="sm-side-gates-state"') < i, 'будильник стоїть у рядку Trade Direction')
    fl = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    r = fl[fl.index("def api_sbr_settings("):fl.index("def api_sbr_tg_symbol(")]
    for k in ('alarm_on', 'alarm_symbol', 'alarm_tf', 'alarm_tfs', 'alarm_side', 'alarm_tm'):
        _check(f"'{k}'" in r, k)
    dg = open(os.path.join(_ROOT, 'detection', 'direction_gate.py'), encoding='utf-8').read()
    _check('def live_enable(' in dg, 'вмикання кнопки живе в direction_gate')
    sb = open(os.path.join(_ROOT, 'detection', 'sbr_signals.py'), encoding='utf-8').read()
    _check('allow_long_entries' not in sb and 'allow_short_entries' not in sb,
           'ключі кнопок не набираються руками')


# ── ⏰ Будильник v2 (16.11): 15m / 30m / 45m · дія над Trade Manager ─────────
def test_alarm_tfs_migration_and_choice():
    _check(S.alarm_tfs_of({}) == ('1h', '4h'), 'дефолт = 1H+4H (як старе «any»)')
    _check(S.alarm_tfs_of({'alarm_tf': '4h'}) == ('4h',), 'старий alarm_tf=4h')
    _check(S.alarm_tfs_of({'alarm_tfs': ['4h', '15M', 'x', '45m']}) == ('15m', '45m', '4h'),
           'кілька TF, від молодшого, сміття відкинуто')
    _check(S.alarm_tfs_of({'alarm_tfs': [], 'alarm_tf': '1h'}) == ('1h',), 'порожньо → старий ключ')
    s = {'alarm_on': True, 'alarm_symbol': 'BTC', 'alarm_side': 'LONG', 'alarm_tfs': ['15m', '4h']}
    _check(S.alarm_match(s, 'BTCUSDT', [('15m', _asig('LONG'))])[0] == '15m', '15M будить')
    _check(S.alarm_match(s, 'BTCUSDT', [('1h', _asig('LONG'))]) is None, '1H не обрано')
    _check(S.alarm_match(s, 'BTCUSDT', [('4h', _asig('LONG')), ('15m', _asig('LONG'))])[0] == '15m',
           'молодший TF першим')
    e = S.SbrEngine(_MemDB(), lambda: [], None, None)
    e.update_settings({'alarm_tf': '4h'})
    _check(e.get_settings()['alarm_tfs'] == ['4h'], 'старий UI (alarm_tf) переводиться в список')
    e.update_settings({'alarm_tfs': ['30m', '1h'], 'alarm_tm': 'ON'})
    st = e.get_settings()
    _check(st['alarm_tfs'] == ['30m', '1h'] and st['alarm_tm'] == 'on', st)
    e.update_settings({'alarm_tm': 'зламане'})
    _check(e.get_settings()['alarm_tm'] == '', 'невідома дія → не чіпати TM')


def test_45m_is_built_from_15m_aligned_to_utc_day():
    day = 1_763_251_200          # 16.11.2025 00:00 UTC
    bars = []
    for i in range(7):           # 7 барів 15m → 2 повні групи 45m + 1 неповна
        t = day + i * 900
        bars.append({'t': t * 1000, 'o': 100 + i, 'h': 110 + i, 'l': 90 - i, 'p': 101 + i,
                     'v': 10, 'q': 1})
    out = S.aggregate_bars(list(reversed(bars)), 2700, 900)
    _check(len(out) == 2, f'лише повні групи: {out}')
    a = out[0]
    _check(a['t'] == day * 1000 and a['o'] == 100 and a['p'] == 103 and a['h'] == 112
           and a['l'] == 88 and a['v'] == 30 and a['q'] == 3, a)
    _check(out[1]['t'] == (day + 2700) * 1000, 'межі 45m від 00:00 UTC')
    _check(86400 % 2700 == 0, '32 бари 45m на добу — епоха й доба вирівняні')


def test_all_tfs_for_group_and_personal_1711():
    _check(S.ALL_TFS == ('15m', '30m', '45m', '1h', '4h', '1d') and S.ALARM_TF_ALL == S.ALL_TFS,
           'усі TF рушія')
    _check(S.tg_tfs_of({'tg_tfs': ['1d', '15m', 'zz']}) == ('15m', '1d'), 'набір із ALL_TFS')
    _check(S.tg_tfs_of({'tg_tfs': []}) == S.TFS, 'порожньо → 1H+4H')
    e = S.SbrEngine(_MemDB(), lambda: ['ETHUSDT'], None, None,
                    get_extra_tfs=lambda: {'30m': ['XRPUSDT'], '1h': ['ZZZUSDT']})
    e.start = lambda: None
    e.update_settings({'tg_tfs': ['15m', '4h'], 'tg_symbols': ['BTCUSDT']})
    st = e.get_settings()
    _check(st['tg_tfs'] == ['15m', '4h'] and st['tg_1h'] is False and st['tg_4h'] is True,
           f'дзеркало tg_1h/tg_4h: {st}')
    e.update_settings({'tg_1h': True})              # старий UI не губить 15M
    _check(e.get_settings()['tg_tfs'] == ['15m', '1h', '4h'], e.get_settings()['tg_tfs'])
    plan = dict(e._plan(['ETHUSDT', 'BTCUSDT']))
    _check(plan.get('15m') == ['BTCUSDT'], f'15M — лише монети групи з 🔔: {plan}')
    _check(plan.get('30m') == ['XRPUSDT'], 'особиста підписка 30M')
    _check('1d' not in plan and '45m' not in plan, 'не обрано → не рахується')
    _check(plan['1h'] == ['ETHUSDT', 'BTCUSDT'], '1H/4H — як і раніше')
    e._safe_fetch = lambda *a: None
    e.update_settings({'tg_enabled': False})
    _check('15m' not in dict(e._plan(['BTCUSDT'])), 'TG вимкнено → 15M групи не рахується')


def test_young_tf_routes_to_group_and_personal_1711():
    group, pers, spring = [], [], []
    e = S.SbrEngine(_MemDB(), lambda: [], None, lambda t: (group.append(t) or True, ''),
                    personal=lambda S_, it: pers.append(it) or (0, 0),
                    spring=lambda S_, it: spring.append(it),
                    get_extra_tfs=lambda: {'45m': ['ETHUSDT']})
    e.start = lambda: None
    e.update_settings({'tg_symbols': ['BTCUSDT'], 'tg_tfs': ['15m']})
    e._on_signals('BTCUSDT', [('15m', _asig('LONG'))], 100.0)
    _check(group and '15M' in group[-1], f'15M обрано для групи → іде в тему: {group}')
    _check(not spring, '🔭 скан — лише 1H/4H')
    e._on_signals('ETHUSDT', [('45m', _asig('SHORT')), ('30m', _asig('SHORT'))], 200.0)
    _check(pers and [x[0] for x in pers[-1]] == ['45m'], f'особисто — лише підписаний TF: {pers}')
    _check(len(group) == 1, 'ETH без 🔔 у групі → тема мовчить')


def test_low_tfs_only_for_alarm_coin_and_45m_fetches_15m():
    asked = []

    def fetch(sym, tf, limit):
        asked.append((sym, tf, limit))
        return None
    e = S.SbrEngine(_MemDB(), lambda: ['ETHUSDT'], fetch, None)
    e.start = lambda: None       # без фонового потоку: він сам почав би тягнути бари
    plan = dict((tf, pool) for tf, pool in e._plan(['ETHUSDT', 'BTCUSDT']))
    _check(set(plan) == {'1h', '4h'}, 'будильник вимкнено → лише 1H/4H')
    e.update_settings({'alarm_on': True, 'alarm_symbol': 'SOL',
                       'alarm_tfs': ['15m', '45m', '1h']})
    plan = dict((tf, pool) for tf, pool in e._plan(['ETHUSDT', 'SOLUSDT']))
    _check(plan.get('15m') == ['SOLUSDT'] and plan.get('45m') == ['SOLUSDT'], plan)
    _check('30m' not in plan, '30M не обрано → не рахується')
    _check(plan['1h'] == ['ETHUSDT', 'SOLUSDT'], '1H — увесь всесвіт')
    e._safe_fetch('SOLUSDT', '45m')
    e._safe_fetch('SOLUSDT', '15m')
    _check(asked == [('SOLUSDT', '15m', S.ALARM_BARS * 3), ('SOLUSDT', '15m', S.ALARM_BARS)],
           f'45M — із 15m: {asked}')


def test_low_tf_signal_reaches_only_the_alarm():
    group, pers, spring, hit = [], [], [], []
    e = S.SbrEngine(_MemDB(), lambda: [], None, lambda t: (group.append(t) or True, ''),
                    personal=lambda S_, it: pers.append(it) or (0, 0),
                    spring=lambda S_, it: spring.append(it))
    e.update_settings({'tg_symbols': ['BTCUSDT']})
    e._alarm_check = lambda S_, items, now, s: hit.append(items)
    e._on_signals('BTCUSDT', [('15m', _asig('LONG'))], 100.0)
    _check(hit and hit[0][0][0] == '15m', 'будильник бачить 15M')
    _check(not group and not pers and not spring and not e._events,
           '15M не йде в групу / особисто / 🔭 скан / журнал')
    e._on_signals('BTCUSDT', [('15m', _asig('LONG')), ('1h', _asig('LONG'))], 200.0)
    _check(pers and [x[0] for x in pers[-1]] == ['1h'] and spring
           and [e_['tfs'] for e_ in e._events] == ['1H'], '1H іде як і раніше')


class _FakeTM:
    def __init__(self, on=False, accept=True):
        self.on, self.accept, self.calls = on, accept, []

    def is_enabled(self):
        return self.on

    def update_settings(self, patch):
        self.calls.append(dict(patch))
        if self.accept:
            self.on = bool(patch.get('enabled', self.on))
        return {'enabled': self.on}


def test_alarm_set_tm_verifies_the_real_state():
    tm = _FakeTM(on=False)
    _check(S.alarm_set_tm(True, tm) == (True, 'увімкнено', True) and tm.on, 'увімкнено')
    _check(tm.calls == [{'enabled': True}], 'той самий шлях, що тумблер на сторінці')
    _check(S.alarm_set_tm(True, tm) == (True, 'вже увімкнено', False), 'вже в стані — без запису')
    _check(len(tm.calls) == 1, 'зайвого запису немає')
    _check(S.alarm_set_tm(False, tm) == (True, 'вимкнено', True) and not tm.on, 'вимкнено')
    deaf = _FakeTM(on=False, accept=False)
    ok, why, changed = S.alarm_set_tm(True, deaf)
    _check(not ok and not changed and 'не прийняв' in why,
           'запит пішов, а стан не змінився → невдача, а не «увімкнено»')
    src = open(os.path.join(_ROOT, 'detection', 'sbr_signals.py'), encoding='utf-8').read()
    fn = src[src.index('def alarm_set_tm('):src.index('def alarm_match(')]
    _check(fn.count('is_enabled()') >= 2, 'стан читається і ДО, і ПІСЛЯ запису')


def test_alarm_switches_tm_and_reports_it():
    dg = types.ModuleType('detection.direction_gate')
    dg.live_enable = lambda side, source='': (True, 'вже увімкнено')
    pkg = sys.modules.setdefault('detection', types.ModuleType('detection'))
    old = sys.modules.get('detection.direction_gate')
    sys.modules['detection.direction_gate'] = dg
    pkg.direction_gate = dg
    sent = []
    tm = _FakeTM(on=False)
    try:
        e = S.SbrEngine(_MemDB(), lambda: [], None, None)
        e._admin_notify = lambda t: sent.append(t) or True
        e._tm_switch = lambda on: S.alarm_set_tm(on, tm)
        e.update_settings({'alarm_on': True, 'alarm_symbol': 'BTC', 'alarm_tfs': ['30m'],
                           'alarm_side': 'LONG', 'alarm_tm': 'on'})
        e._on_signals('BTCUSDT', [('30m', _asig('LONG'))], 100.0)
        L = e.get_settings()['alarm_last']
        _check(tm.on and L['tm'] == 'on' and L['tm_ok'] and L['tf'] == '30M', L)
        _check(sent and 'Trade Manager → <b>УВІМКНЕНО</b>' in sent[-1],
               'зміна TM іде в TG навіть коли кнопка вже була увімкнена')
        n = len(sent)
        e._on_signals('BTCUSDT', [('30m', _asig('LONG'))], 200.0)
        _check(len(sent) == n, 'усе вже в стані → тиша')
        e.update_settings({'alarm_tm': 'off'})
        e._on_signals('BTCUSDT', [('30m', _asig('LONG'))], 300.0)
        _check(not tm.on and 'ВИМКНЕНО' in sent[-1], 'вимкнення TM')
        deaf = _FakeTM(on=True, accept=False)
        e._tm_switch = lambda on: S.alarm_set_tm(on, deaf)
        e._on_signals('BTCUSDT', [('30m', _asig('LONG'))], 400.0)
        L = e.get_settings()['alarm_last']
        _check(L['ok'] is False and L['tm_ok'] is False and 'НЕ змінено' in sent[-1], L)
    finally:
        if old is not None:
            sys.modules['detection.direction_gate'] = old
            pkg.direction_gate = old


def test_page_never_overwrites_tm_with_a_stale_checkbox():
    sv = _HTML[_HTML.index('async function saveTMSettings()'):]
    sv = sv[:sv.index('\n}\n')]
    _check("enabled: document.getElementById('tm-enabled').checked," not in sv,
           '`enabled` більше не їде з кожною правкою TM')
    _check('if (_sendEnabled) payload.enabled' in sv, '…лише після кліку людини')
    tg = _HTML[_HTML.index('async function onTMToggle()'):]
    tg = tg[:tg.index('\n}\n')]
    _check('_tmEnabledDirty = true' in tg, 'клік по тумблеру ставить позначку')
    st = _HTML[_HTML.index('async function loadTMState('):]
    _check('_tmSyncMasterToggle(d.settings.enabled)' in st[:st.index('\n}\n')],
           'полл ставить чекбокс за сервером')
    import subprocess
    js = _HTML[_HTML.index('const _SBR_ALARM_TFS'):_HTML.index('function sbrAlarmTmChange')]
    r = subprocess.run(['node', '-e', js + """
        console.log(JSON.stringify([_sbrAlarmTfsOf({alarm_tfs: ['4h', '15m']}),
            _sbrAlarmTfsOf({alarm_tf: '4h'}), _sbrAlarmTfsOf({})]));"""],
        capture_output=True, text=True, timeout=30)
    _check(r.returncode == 0, r.stderr)
    _check(json.loads(r.stdout) == [['15m', '4h'], ['4h'], ['1h', '4h']], r.stdout)


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v()
            print('✓', k)
            n += 1
    print(f'OK — {n} тестів')
