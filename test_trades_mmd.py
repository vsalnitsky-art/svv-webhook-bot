"""🧠 МММ-Decision + 🚨 сигнал у таблицях відкритих угод · OB 4H ліворуч ·
Opposite OB Exit на 5m (вимога 22.10).

Дослівно: «Прибери в таблицях відкритих угод колонки 🧮 МММ LiQ і 🧮 МММ-Trend, а
замість них продублюй показник МММ-Decision і поруч додай сигнальну колонку,
яка має інформувати сигнальним кольоровим попередженням, що показник у полі
МММ-Decision змінився у слабшу сторону або взагалі став протилежним
відповідно відкритій угоді» · «В колонці "Side" показник OB 4Н розмісти з
лівої сторони» · «додай вибір 5хв таймфрейму» (Opposite OB Exit).
"""
import os
import subprocess
import sys
import tempfile
import threading
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)

_m = _mon._m
_HTML = _mon._HTML
_SRC = _mon._SRC
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _dec(reco, pl, verdict='good'):
    return {'recommended': reco, 'prob_long': pl, 'prob_short': round(1 - pl, 4),
            'verdict': verdict, 'confidence': 'high',
            'headline': (f'LONG {round(pl * 100)}%' if pl >= 0.5
                         else f'SHORT {round((1 - pl) * 100)}%')}


# ═══════════ 1. ЧИСТЕ ПРАВИЛО СИГНАЛУ ═══════════════════════════════════
def test_alert_levels():
    A = _m.dec_trade_alert
    cur = _m.dec_mm_dir(_dec('LONG', 0.85))
    _check(A('LONG', cur, None)['level'] == 'ok', 'за угодою → ok')
    _check(A('SHORT', cur, None)['level'] == 'against', 'протилежний → against')
    neu = _m.dec_mm_dir(_dec('NEUTRAL', 0.52, 'marginal'))
    _check(A('LONG', neu, None)['level'] == 'neutral', 'NEUTRAL → neutral')
    weak = _m.dec_mm_dir(_dec('LONG', 0.62, 'good'))
    r = A('LONG', weak, cur)
    _check(r['level'] == 'weaker' and r['align'] == 24 and r['peak_align'] == 70
           and r['drop'] == 46, f'перевага впала від піку → weaker: {r}')
    small = _m.dec_mm_dir(_dec('LONG', 0.80, 'good'))
    _check(A('LONG', small, cur)['level'] == 'ok',
           f'падіння < {_m.MMD_WEAK_PP} п.п. — ще не «слабшає»')
    word = _m.dec_mm_dir(_dec('LONG', 0.84, 'marginal'))
    _check(A('LONG', word, cur)['level'] == 'weaker',
           'слово сили опустилось (СИЛЬНИЙ → СЕРЕДНІЙ) → weaker')
    _check(A('LONG', None, cur) is None and A('', cur, None) is None,
           'немає даних → None («невідомо» ≠ «все гаразд»)')
    print('✓ сигнал: ok / weaker (від піку угоди) / neutral / against, без даних — None')


def test_short_mirrors():
    A = _m.dec_trade_alert
    peak = _m.dec_mm_dir(_dec('SHORT', 0.10))
    cur = _m.dec_mm_dir(_dec('SHORT', 0.35))
    r = A('SHORT', cur, peak)
    _check(r['align'] == 30 and r['peak_align'] == 80 and r['level'] == 'weaker', r)
    print('✓ SHORT дзеркальний: перевага рахується у бік угоди')


# ═══════════ 2. ДВИГУН: вердикт для угод навіть при вимкненому блоці ═══
class _TM:
    def __init__(self, data, positions=None, shadow=None):
        self.data, self.calls = data, []
        self._lock = threading.RLock()
        self._positions = positions or {}
        self._shadow_positions = shadow or {}

    def compute_decision(self, sym, price):
        self.calls.append(sym)
        return self.data.get(sym)


def _ff(data, positions=None, shadow=None, dec_on=False):
    ff = _mon._mk()
    tm = _TM(data, positions, shadow)
    _mod = types.ModuleType('detection.trade_manager')
    _mod.get_trade_manager = lambda: tm
    sys.modules['detection.trade_manager'] = _mod
    if 'detection' in sys.modules:
        setattr(sys.modules['detection'], 'trade_manager', _mod)
    ff._get_tm = lambda: tm
    ff._mmd_trade_peak = {}
    for k in ('mm_ctr_enabled', 'mm_fc_enabled'):
        ff._settings[k] = False
    ff._settings['mm_dec_enabled'] = dec_on
    return ff, tm


def test_open_trades_get_trend_even_when_the_block_is_off():
    """⚠️ 26.10: колонки 🧠/🚨 угод живуть на 🧮 МММ-Trend (ті самі `fuels`
    двигуна, що таблиця блоку). Decision для угод більше НЕ рахується."""
    pos = {'BTCUSDT': {'side': 'LONG', 'opened_at': 1000.0}}
    ff, tm = _ff({'BTCUSDT': _dec('SHORT', 0.1)}, positions=pos)
    ff._settings['mm_new_enabled'] = False
    ff._legacy = {'BTCUSDT': 0.3, 'ETHUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.42, ETHUSDT=0.3), now=2000.0)
    _check(tm.calls == [], f'Decision для угод не рахується: {tm.calls}')
    r = ff.dec_trade_for([('BTCUSDT', 'LONG', 1000.0), ('ETHUSDT', 'LONG', 1000.0)])
    k = _m.mmd_trade_key('BTCUSDT', 'LONG', 1000.0)
    _check(set(r) == {k}, f'лише монета в угоді: {r}')
    _check(r[k]['mm'] == 'LONG' and r[k]['strength'] == 42
           and r[k]['src'] == 'trend' and r[k]['alert']['level'] == 'ok', r)
    print('✓ монета в угоді має стан МММ-Trend навіть при вимкненому блоці')


def test_weakening_is_measured_from_the_trade_peak():
    pos = {'BTCUSDT': {'side': 'LONG', 'opened_at': 1000.0}}
    ff, tm = _ff({}, positions=pos)
    k = _m.mmd_trade_key('BTCUSDT', 'LONG', 1000.0)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.6), now=2000.0)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=2030.0)
    a = ff.dec_trade_for([('BTCUSDT', 'LONG', 1000.0)])[k]['alert']
    _check(a['level'] == 'weaker' and a['peak_align'] == 60, a)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.05), now=2060.0)
    a = ff.dec_trade_for([('BTCUSDT', 'LONG', 1000.0)])[k]['alert']
    _check(a['level'] == 'neutral' and 'МММ-Trend' in a['text'], a)
    ff._mm_capture(_mon._fuels(BTCUSDT=-0.4), now=2090.0)
    a = ff.dec_trade_for([('BTCUSDT', 'LONG', 1000.0)])[k]['alert']
    _check(a['level'] == 'against', a)
    _check(ff._mmt_trade_peak[k]['dir'] == 0.6, ff._mmt_trade_peak)
    print('✓ «слабшає» міряється від ПІКУ угоди, ⚖ → neutral, протилежний → against')


def test_new_trade_does_not_inherit_old_peak_and_restart_keeps_it():
    ff, _ = _ff({})
    ff._mmt_trade = {'BTCUSDT': (1.0, _m.trend_mm_cur({'dir': 0.5, 'status': 'LONG'}))}
    old = _m.mmd_trade_key('BTCUSDT', 'LONG', 1000)
    ff._mmd_track_peaks({old}, now=5000.0)
    new = _m.mmd_trade_key('BTCUSDT', 'LONG', 9000)
    _check(old != new, 'ключ несе opened_at')
    ff._mmd_track_peaks(set(), now=5000.0 + 60)
    _check(old in ff._mmt_trade_peak, 'короткий пропуск (TM ще не піднявся) пік не стирає')
    ff._mmd_track_peaks(set(), now=5000.0 + _m.MMD_PEAK_FORGET_SEC + 61)
    _check(old not in ff._mmt_trade_peak, 'пік закритої угоди забувається')
    _check("'mmt_trade_peak'" in _SRC and "st.get('mmt_trade_peak')" in _SRC,
           'пік персиститься в стан FF (власний ключ — шкала Decision не змішується)')
    print('✓ нова угода — нова база, пік переживає рестарт і забувається згодом')


def test_route_only_reads():
    i = _FLASK.index("_dec_map = ff.dec_trade_for(")
    blk = _FLASK[i - 800:i + 1200]
    _check("pos['mmd']" in blk and 'mmd_trade_key' in blk, blk[:300])
    for bad in ('compute_decision', 'evaluate_entry', '_fuel_dir_legacy',
                "pos['old_mm']", "pos['new_mm']"):
        _check(bad not in blk, f'зайве на гарячому ендпоінті: {bad}')
    a = _SRC.index('def dec_trade_for(')
    fn = _SRC[a:_SRC.index('\n    def ', a + 10)]
    _check('compute_decision' not in fn.split('"""')[2], 'dec_trade_for — лише читання')
    print('✓ /api/tm/state лише читає кеш вердиктів (без розрахунків)')


# ═══════════ 3. UI ═════════════════════════════════════════════════════
def _js(body):
    src = 'const _MMD_V_UA = { good: "СИЛЬНИЙ", marginal: "СЕРЕДНІЙ", poor: "СЛАБКИЙ" };\n'
    for f in ('function mmdTradeCellHTML(', 'function mmdAlertCellHTML('):
        a = _HTML.index(f)
        src += _HTML[a:_HTML.index('\n}', a) + 2] + '\n'
    for c in ('const _MMD_ALERT',):
        a = _HTML.index(c)
        src = _HTML[a:_HTML.index('};', a) + 2] + '\n' + src
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as fh:
        fh.write(src + body)
        path = fh.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr
    finally:
        os.unlink(path)


def test_js_cells():
    out = _js(r'''
const m = {mm:'LONG', strength:70, prob:85, verdict:'good', headline:'LONG 85%',
           alert:{level:'against', align:-20, peak_align:70, drop:90, text:'ПРОТИ угоди: Decision SHORT'}};
console.log(mmdTradeCellHTML({mmd:m}));
console.log('|');
console.log(mmdAlertCellHTML({mmd:m}));
console.log('|');
console.log(mmdAlertCellHTML({mmd:Object.assign({}, m, {alert:{level:'weaker', align:24, peak_align:70, text:'слабшає'}})}));
console.log('|');
console.log(mmdAlertCellHTML({}) + mmdTradeCellHTML({}));
''')
    p = out.split('|')
    _check(len(p) == 4, out)
    _check('▲' in p[0] and '70%' in p[0] and 'width:70%' in p[0] and 'width:36px' in p[0], p[0])
    _check('ПРОТИ' in p[1] and 'tm-mmd-alert-hot' in p[1] and '#dc2626' in p[1], p[1])
    _check('СЛАБШ' in p[2] and '#fbbf24' in p[2] and 'hot' not in p[2], p[2])
    _check(p[3].count('>—<') == 2, 'немає даних → «—» в обох комірках')
    print('✓ JS: 🧮 комірка + 🚨 ПРОТИ (червоний, блимає) / СЛАБШ (бурштин)')


def test_blink_css_exists():
    _check('.tm-mmd-alert-hot' in _HTML and '@keyframes tmMmdHot' in _HTML, 'немає блимання')
    print('✓ CSS: «ПРОТИ» блимає')


# ═══════════ 4. Opposite OB Exit — 5m ═════════════════════════════════
def test_opposite_ob_exit_accepts_5m():
    _check("ALLOWED_EXIT_TFS = ('5m', '15m', '30m', '1h', '4h')" in _TM_SRC,
           'TM мусить приймати 5m')
    i = _HTML.index('id="tm-opposite-ob-exit-tf"')
    sel = _HTML[i:_HTML.index('</select>', i)]
    _check('<option value="5m">5M</option>' in sel, 'немає опції 5M')
    _check("['5m', '15m', '30m', '1h', '4h'].includes(s.opposite_ob_exit_timeframe)" in _HTML,
           'сторінка скидала б 5m на 15m при завантаженні')
    print('✓ Opposite OB Exit: TF 5M доступний (бекенд + сторінка)')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
