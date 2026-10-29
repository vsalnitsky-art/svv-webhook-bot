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
    for part in ('#ETHUSDT', 'SHORT #1', '4H', 'Time: 19:00 Kyiv', 'Trigger:', 'Score: 76.39',
                 'Risk: 1R', 'SL:'):
        _check(part in t, f'«{part}» у повідомленні: {t}')


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
    js = ('const fmtPriceJS = p => String(p);\n' + esc + '\n' + fn('_sbrTfTip')
          + '\n' + fn('_mmSbrTfCell') + '\n' + fn('_mmSbrSigCell') + r"""
const g = {t: 1792857600000, close_t: 1792872000, trigger: 2293, score: 100, risk: '1R + Add', sl: 2184.43, n: 1, time_kyiv: '19:00'};
console.log(_mmSbrTfCell({mm:'LONG', sbr_tfs:'1H+4H', sbr_fresh:true, sbr_by_tf:{'1h':g,'4h':g}}));
console.log('|'); console.log(_mmSbrSigCell({mm:'LONG', sbr_sig:g, sbr_by_tf:{'4h':g}}));
console.log('|'); console.log(_mmSbrTfCell({}));
""")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    p = r.stdout.split('|')
    _check(len(p) == 3, r.stderr)
    _check('1H+4H' in p[0] and '🆕' in p[0], p[0])
    _check('LONG #1' in p[1] and '19:00' in p[1] and '2293' in p[1] and '100.00' in p[1]
           and '1R + Add' in p[1] and 'SL 2184.43' in p[1], p[1])
    _check('—' in p[2], p[2])


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v()
            print('✓', k)
            n += 1
    print(f'OK — {n} тестів')
