"""🔔 МММ-SBR (з 29.10; до того 🧮 МММ-Trend / 🧠 МММ-Decision) + 🚨 сигнал у таблицях відкритих угод · OB 4H ліворуч ·
Opposite OB Exit на 5m (вимога 22.10).

Дослівно: «Прибери в таблицях відкритих угод колонки 🧮 МММ LiQ і 🧮 МММ-Trend, а
замість них продублюй показник МММ-Decision і поруч додай сигнальну колонку,
яка має інформувати сигнальним кольоровим попередженням, що показник у полі
МММ-Decision змінився у слабшу сторону або взагалі став протилежним
відповідно відкритій угоді» · «В колонці "Side" показник OB 4Н розмісти з
лівої сторони» · «додай вибір 5хв таймфрейму» (Opposite OB Exit).

29.10: «замість 🧮 МММ-Trend бери показник МММ-SBR, а в 🚨 став коли вже
з'явиться протилежна позначка із TF і якщо увімкнено тумблер активуй 🧱 OB».
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


# 🔔 З 29.10 колонки угод живуть на МММ-SBR (рушій Session Band Reversal).
import importlib.util as _iu   # noqa: E402
_sp = _iu.spec_from_file_location('sbr_signals_trades_t',
                                  os.path.join(_HERE, 'detection', 'sbr_signals.py'))
_S = _iu.module_from_spec(_sp)
_sp.loader.exec_module(_S)


def _sig(side, close_t, score=70.0, n=1):
    return {'side': side, 't': int((close_t - 3600) * 1000), 'close_t': close_t,
            'trigger': 1.0, 'score': score, 'risk': '1R', 'sl': 0.9, 'n': n,
            'time_kyiv': '19:00'}


def _row(s1=None, s4=None):
    return {'1h': {'last': s1}, '4h': {'last': s4}, 'combo': _S.combine(s1, s4),
            'fresh': {}}


# ═══════════ 1. ЧИСТЕ ПРАВИЛО СИГНАЛУ ═══════════════════════════════════
def test_alert_levels():
    A, D = _m.sbr_trade_alert, _m.sbr_mm_dir
    long_ = D(_row(_sig('LONG', 2000, 80)))
    _check(A('LONG', long_, 1000)['level'] == 'ok', 'сигнал за угодою → ok')
    r = A('SHORT', long_, 1000)
    _check(r['level'] == 'against' and r['tf'] == '1H' and '1H' in r['text'],
           f'протилежний сигнал ПІСЛЯ входу → against із TF: {r}')
    r = A('SHORT', long_, 3000)
    _check(r['level'] == 'pre', f'протилежний, але ще до входу → pre (не «зʼявився»): {r}')
    both = D(_row(_sig('SHORT', 1500), _sig('SHORT', 2000)))
    _check(A('LONG', both, 1000)['tf'] == '1H+4H', 'TF = комбінація, що в колонці')
    empty = D({'1h': {'last': None}, '4h': None, 'combo': None})
    _check(A('LONG', empty, 1000)['level'] == 'none', 'сигналу немає → none')
    _check(A('LONG', None, 1000) is None and A('', long_, 1000) is None,
           'немає даних → None («невідомо» ≠ «все гаразд»)')
    print('✓ сигнал: ok / against (+TF, після входу) / pre (до входу) / none; без даних — None')


def test_only_against_drives_the_automat():
    _check("MMD_CLOSE_LEVELS = ('against',)" in _TM_SRC,
           'тенденцію до закриття дає ЛИШЕ протилежний сигнал SBR')
    _check("'pre'" in _TM_SRC and "'none'" in _TM_SRC, 'pre/none запамʼятовуються')
    print('✓ 🧱 автомат реагує лише на ⚠ ПРОТИ')


# ═══════════ 2. ДВИГУН: стан SBR для угод навіть при вимкненому блоці ══
class _TM:
    def __init__(self, data, positions=None, shadow=None):
        self.data, self.calls = data, []
        self._lock = threading.RLock()
        self._positions = positions or {}
        self._shadow_positions = shadow or {}

    def compute_decision(self, sym, price):
        self.calls.append(sym)
        return self.data.get(sym)


def _ff(data, positions=None, shadow=None, dec_on=False, sbr=None):
    ff = _mon._mk()
    tm = _TM(data, positions, shadow)
    _mod = types.ModuleType('detection.trade_manager')
    _mod.get_trade_manager = lambda: tm
    sys.modules['detection.trade_manager'] = _mod
    if 'detection' in sys.modules:
        setattr(sys.modules['detection'], 'trade_manager', _mod)
    ff._get_tm = lambda: tm
    for k in ('mm_ctr_enabled', 'mm_fc_enabled'):
        ff._settings[k] = False
    ff._settings['mm_dec_enabled'] = dec_on
    ff._sbr_rows = lambda: dict(sbr or {})
    return ff, tm


def test_open_trades_get_sbr_even_when_the_block_is_off():
    pos = {'BTCUSDT': {'side': 'LONG', 'opened_at': 1000.0}}
    sbr = {'BTCUSDT': _row(_sig('SHORT', 1500, 60), _sig('SHORT', 1800, 77)),
           'ETHUSDT': _row(_sig('LONG', 1500))}
    ff, tm = _ff({'BTCUSDT': _dec('SHORT', 0.1)}, positions=pos, sbr=sbr)
    ff._settings['mm_sbr_enabled'] = False
    ff._mm_capture(_mon._fuels(BTCUSDT=0.42, ETHUSDT=0.3), now=2000.0)
    _check(tm.calls == [], f'Decision для угод не рахується: {tm.calls}')
    r = ff.dec_trade_for([('BTCUSDT', 'LONG', 1000.0), ('ETHUSDT', 'LONG', 1000.0)])
    k = _m.mmd_trade_key('BTCUSDT', 'LONG', 1000.0)
    _check(set(r) == {k}, f'лише монета в угоді: {r}')
    v = r[k]
    _check(v['mm'] == 'SHORT' and v['tf'] == '1H+4H' and v['strength'] == 77
           and v['src'] == 'sbr' and v['alert']['level'] == 'against'
           and v['alert']['tf'] == '1H+4H', v)
    print('✓ монета в угоді має стан МММ-SBR навіть при вимкненому блоці; 🚨 ПРОТИ + TF')


def test_trade_column_uses_the_banner_tf_choice():
    pos = {'BTCUSDT': {'side': 'LONG', 'opened_at': 1000.0}}
    sbr = {'BTCUSDT': _row(_sig('SHORT', 2500), _sig('LONG', 1800, 66))}
    ff, _ = _ff({}, positions=pos, sbr=sbr)
    ff._settings['mms_tf_1h'] = False
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=3000.0)
    k = _m.mmd_trade_key('BTCUSDT', 'LONG', 1000.0)
    v = ff.dec_trade_for([('BTCUSDT', 'LONG', 1000.0)])[k]
    _check(v['tf'] == '4H' and v['alert']['level'] == 'ok',
           f'обрано лише 4H — свіжий 1H SHORT не рахується (як у таблиці блоку): {v}')
    print('✓ колонка угод бере ті самі TF, що банер МММ-SBR')


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
    src = ''
    for f in ('function chargeColor(', 'function springTradeCellHTML(',
              'function sbrTrailBadgeHTML(', 'function mmdAlertCellHTML('):
        a = _HTML.index(f)
        src += _HTML[a:_HTML.index('\n}', a) + 2] + '\n'
    for c in ('const _MMD_ALERT', 'const _CHG_STOPS'):
        a = _HTML.index(c)
        e = _HTML.index(';\n', a) + 2
        src = _HTML[a:e] + '\n' + src
    src = "function fmtPriceJS(v){return String(Number(v));}\n" + src
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as fh:
        fh.write(src + body)
        path = fh.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
        return r.stdout + r.stderr
    finally:
        os.unlink(path)


def test_js_cells():
    """⚠️ 31.10: колонку 🔔 МММ-SBR замінено на 🌀 Заряд (на вході → зараз,
    градієнт), а 🚨 показує ще й стан 🔔⏳ трейлу (тест переписано)."""
    out = _js(r"""
const m = {alert:{level:'against', side:'SHORT', tf:'1H+4H', text:'ПРОТИ угоди: 🔔 МММ-SBR SHORT 1H+4H'}};
console.log(springTradeCellHTML({side:'LONG', spring_open:30, spring:{charge:70, pot_long:50, label:'🌀 накручена'}}));
console.log('|');
console.log(mmdAlertCellHTML({mmd:m}));
console.log('|');
console.log(mmdAlertCellHTML({mmd:{alert:{level:'pre', tf:'4H', text:'до входу'}}}));
console.log('|');
console.log(mmdAlertCellHTML({}) + springTradeCellHTML({}));
console.log('|');
console.log(mmdAlertCellHTML({side:'LONG', current_price:102, mmd:m,
  sbr_trail:{armed_px:100, best_px:103, stop_px:101, tf:'1H', sig_side:'SHORT', charge_ok:true, pot:55}}));
""")
    p = out.split('|')
    _check(len(p) == 5, out)
    _check('>30<' in p[0] and '>70<' in p[0] and '+40' in p[0] and 'linear-gradient' in p[0]
           and 'width:70%' in p[0] and 'left:calc(30%' in p[0], p[0])
    _check('ПРОТИ 1H+4H' in p[1] and 'tm-mmd-alert-hot' in p[1] and '#dc2626' in p[1], p[1])
    _check('ДО ВХОДУ' in p[2] and '#fbbf24' in p[2] and 'hot' not in p[2], p[2])
    _check(p[3].count('>—<') == 2, 'немає даних → «—» в обох комірках')
    _check('🔔⏳ 1H' in p[4] and '🛡101' in p[4] and '+2.00%' in p[4] and '🌀' in p[4]
           and 'ПРОТИ' not in p[4].split('title=')[0], p[4])
    print('✓ JS: 🌀 заряд (вхід → зараз, градієнт) + 🚨 ПРОТИ / ДО ВХОДУ / 🔔⏳ трейл')


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
