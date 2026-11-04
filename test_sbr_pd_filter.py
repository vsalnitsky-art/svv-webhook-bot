"""📍 Фільтр Premium / Discount для 🔔 МММ-SBR (вимога 04.11).

Рішення користувача: (а) лише показ — колонка «📍 PD» + фільтр «лише у вигідній
зоні» поруч із «Сила ≥»; один TF PD (як зараз, `pd_zone_timeframe`);
рівновага відсіюється.
"""
import os
import re
import subprocess
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)

_m = _mon._m
_HTML = _mon._HTML
_SRC = _mon._SRC


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ═══════════ 1. ЧИСТЕ ПРАВИЛО ══════════════════════════════════════════
def test_short_in_premium_and_long_in_discount_fit():
    f = _m.sbr_pd_fit
    a = f('SHORT', 90, 75, 25)
    _check(a['zone'] == 'premium' and a['fit'] is True and a['depth'] == 0.6, a)
    b = f('LONG', 10, 75, 25)
    _check(b['zone'] == 'discount' and b['fit'] is True and b['depth'] == 0.6, b)
    print('✓ SHORT у Premium і LONG у Discount — вигідне положення з глибиною')


def test_against_and_equilibrium_do_not_fit():
    f = _m.sbr_pd_fit
    _check(f('LONG', 90, 75, 25)['fit'] is False, 'LONG у Premium — проти')
    _check(f('SHORT', 10, 75, 25)['fit'] is False, 'SHORT у Discount — проти')
    e = f('LONG', 50, 75, 25)
    _check(e['zone'] == 'eq' and e['fit'] is False and e['depth'] is None,
           'рівновага відсіюється (рішення користувача)')
    print('✓ зона проти сигналу і рівновага — не вигідні')


def test_unknown_data_is_not_against():
    f = _m.sbr_pd_fit
    _check(f('LONG', None) is None, 'немає PD → None, а не «проти»')
    n = f(None, 80)
    _check(n['zone'] == 'premium' and n['fit'] is None, '⚖ монета без боку → fit None')
    _check(f('SHORT', 75, 75, 25)['fit'] is True, 'поріг Premium включно')
    print('✓ «невідомо» ≠ «проти»; пороги включні')


# ═══════════ 2. ДЖЕРЕЛО = БЕЙДЖ PD ГРАФІКА ═════════════════════════════
def test_pd_comes_from_the_scanner_cache_only():
    class _Sc:
        def get_pd_pct(self, s):
            return {'BTCUSDT': 88.0, 'ETHUSDT': None}.get(s)

        def get_pd_thresholds(self):
            return (70.0, 30.0)
    mod = types.ModuleType('detection.smc_scanner')
    mod.get_smc_scanner = lambda: _Sc()
    old = sys.modules.get('detection.smc_scanner')
    sys.modules['detection.smc_scanner'] = mod
    try:
        ff = _mon._mk()
        out = ff._mm_pd_marks(['BTCUSDT', 'ETHUSDT'])
    finally:
        if old is not None:
            sys.modules['detection.smc_scanner'] = old
        else:
            sys.modules.pop('detection.smc_scanner', None)
    _check(out == {'BTCUSDT': (88.0, 70.0, 30.0)}, out)
    i = _SRC.index('def _mm_pd_marks')
    body = _SRC[i:_SRC.index('\n    def ', i + 10)]
    _check('get_pd_pct' in body and 'get_pd_thresholds' in body
           and '_pd_zone_cache' not in body and 'fetch' not in body,
           'лише публічні читачі сканера — без власного розрахунку й мережі')
    print('✓ PD = scanner.get_pd_pct + get_pd_thresholds (те саме, що бейдж графіка)')


def test_only_the_sbr_book_gets_pd_and_setting_is_validated():
    i = _SRC.index("if _bk == 'sbr' and rows:")
    _check('sbr_pd_fit' in _SRC[i:i + 500], 'рядки SBR несуть pd')
    s = dict(_m.DEFAULT_SETTINGS)
    _check(s.get('mms_pd_only') is False, 'фільтр за замовчуванням вимкнений')
    _check("'pd_only':" in _SRC, 'стан віддає pd_only')
    print('✓ PD лише в книзі SBR, mms_pd_only деф. вимкнено')


# ═══════════ 3. UI ═════════════════════════════════════════════════════
def test_ui_column_and_checkbox_next_to_strength():
    m = re.search(r"_MMB\.sbr\.cols = \[([^\]]*)\]", _HTML)
    cols = [x.strip().strip("'") for x in m.group(1).split(',')]
    _check('pd' in cols and cols.index('pd') == cols.index('sbrsig') + 1, cols)
    a = _HTML.index('id="mms-table"')
    th = len(re.findall(r'<th[ >]', _HTML[a:_HTML.index('</thead>', a)]))
    _check(th == len(cols) + 3, f'<th> {th} = cols {len(cols)} + TRD/☑/символ')
    _check('colspan="10" class="tm-empty-msg">Немає сигналів SBR' in _HTML, 'colspan 10')
    s1 = _HTML.index('id="mms-min-str"')
    s2 = _HTML.index('id="mms-pd-only"')
    s3 = _HTML.index('id="ff-mms-bias-confirm"')
    _check(s1 < s2 < s3, 'чекбокс одразу за «Сила ≥»')
    _check('mms_pd_only: !!_MMB.sbr.pdOnly' in _HTML, 'зберігається в налаштуваннях бота')
    _check("(!B.pdOnly || (r.pd && r.pd.fit === true))" in _HTML,
           'фільтр діє в _mmTabRows → таблиця і лічильники вкладок')
    print('✓ колонка «📍 PD» + чекбокс поруч із «Сила ≥», фільтр у лічильниках')


def test_js_cell_and_filter():
    esc = _HTML[_HTML.index('const _mmEsc'):]
    esc = esc[:esc.index(';\n') + 2]
    a = _HTML.index('const _MM_PD_UA')
    b = _HTML.index('\n}', _HTML.index('function _mmPdCell(', a)) + 2
    js = esc + _HTML[a:b] + (
        "\nconsole.log(_mmPdCell({mm:'SHORT',pd:{pct:88,zone:'premium',fit:true,depth:0.5}}));"
        "console.log('|');console.log(_mmPdCell({mm:'LONG',pd:{pct:50,zone:'eq',fit:false,depth:null}}));"
        "console.log('|');console.log(_mmPdCell({mm:'LONG'}));")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    p = r.stdout.split('|')
    _check(len(p) == 3, r.stderr)
    _check('Premium' in p[0] and '88%' in p[0] and '✓' in p[0] and '#4ade80' in p[0], p[0])
    _check('Рівновага' in p[1] and '✓' not in p[1], p[1])
    _check('—' in p[2], 'немає PD → «—»')
    print('✓ JS: комірка PD — зона, %, ✓/✗ і колір рамки за збігом')


if __name__ == '__main__':
    fns = [v for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
    print(f'{len(fns)}/{len(fns)} passed')
