"""🌀 «Пружина» — заряд/напрямок/вистріл монет WATCHLIST + пріоритет і ворота
Черги-4 (вимога 29.10).

Запуск: python3 test_spring_coil.py
"""
import importlib.util
import os
import random
import re

_ROOT = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location('spring_coil_t', os.path.join(_ROOT, 'detection', 'spring_coil.py'))
S = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(S)
_FF = open(os.path.join(_ROOT, 'detection', 'fuel_filter.py'), encoding='utf-8').read()
_SC = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
_APP = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _coil(seed=1, n=240, up=True):
    """Тренд, потім вузька коробка біля верху з сухим обʼємом."""
    random.seed(seed)
    p, out = 100.0, []
    for i in range(n):
        if i < n - 60:
            p *= 1 + random.gauss(0.001 if up else -0.001, 0.01)
            amp, vol = 0.012, 1000
        else:
            # детермінований дрейф у коробці: висхідні мінімуми (up) або
            # спадні максимуми (down) + ±шум, без випадковості напрямку
            p *= 1 + (0.0003 if up else -0.0003) + (0.0015 if i % 2 else -0.0015)
            amp, vol = 0.003, 500
        out.append({'t': i * 3600000, 'o': p, 'h': p * (1 + amp), 'l': p * (1 - amp), 'p': p, 'v': vol})
    return out


def _walk(seed):
    random.seed(seed)
    p, out = 100.0, []
    for i in range(240):
        o = p
        p *= 1 + random.gauss(0, 0.008)
        hi = max(o, p) * (1 + abs(random.gauss(0, 0.004)))
        lw = min(o, p) * (1 - abs(random.gauss(0, 0.004)))
        out.append({'t': i * 3600000, 'o': o, 'h': hi, 'l': lw, 'p': p,
                    'v': 1000 * (1 + abs(random.gauss(0, 0.5)))})
    return out


def test_coil_is_charged_with_direction():
    r = S.analyze(_coil(up=True))
    _check(r['ok'] and r['state'] == 'charged' and r['charge'] >= S.CHARGED, r)
    _check(r['dir'] == 'LONG' and r['pot_long'] > r['pot_short'], r)
    _check(r['squeeze_bars'] > 0 and r['vol_ratio'] < 1.0, r)
    d = S.analyze(_coil(up=False))
    _check(d['dir'] == 'SHORT' and d['pot_short'] > d['pot_long'], f'дзеркально вниз: {d}')


def test_noise_is_not_a_spring():
    ch = [S.analyze(_walk(s))['charge'] for s in range(120)]
    charged = sum(1 for c in ch if c >= S.CHARGED)
    _check(sum(ch) / len(ch) < 40, f'випадкове блукання в середньому без заряду: {sum(ch)/len(ch):.1f}')
    _check(charged <= 6, f'«заряджена» на шумі трапляється рідко: {charged}/120')


def test_fired_spring_points_the_way():
    b = _coil()
    last = b[-1]['p']
    b[-1] = dict(b[-1], p=last * 1.04, h=last * 1.045, l=last * 0.999, v=5000)
    r = S.analyze(b)
    _check(r['state'] == 'fired' and r['fired']['side'] == 'LONG' and r['pot_short'] == 0, r)


def test_insufficient_data_is_honest():
    r = S.analyze(_coil()[:100])
    _check(not r['ok'] and 'замало' in r['reason'], r)


def test_potential_is_charge_times_direction_share():
    r = S.analyze(_coil())
    _check(abs(r['pot_long'] + r['pot_short'] - r['charge']) < 0.2, 'сума потенціалів = заряд')


def test_gate_and_priority():
    rec = {'ok': True, 'pot_long': 70, 'pot_short': 10, 'label': '🌀', 'charge': 80}
    _check(S.gate(rec, 'LONG', 'require', 40)[0], 'LONG 70 ≥ 40')
    ok, why = S.gate(rec, 'SHORT', 'require', 40)
    _check(not ok and 'чекає' in why, why)
    _check(S.gate(rec, 'SHORT', 'priority', 40)[0], 'priority не блокує')
    _check(S.gate(None, 'SHORT', 'require', 40)[0], 'немає даних → не блокуємо')
    _check(S.priority_key(rec, 'LONG') == 70 and S.priority_key(None, 'LONG') == 50, 'ключ')


def test_scanner_reuses_the_1h_bars_and_watchlist_only():
    _check("td_sp = _get_tf_data(SPRING_TF)" in _SC, 'дані — ті самі закриті 1h-бари сканера')
    _check("(getattr(self, '_spring', None) or {}).pop(symbol, None)" in _SC, 'видалення з watchlist чистить запис')
    i = _SC.index('def spring_for(')
    fn = _SC[i:_SC.index('\n    def ', i + 10)]
    _check('get_watchlist' in fn, 'віддаємо лише WATCHLIST')
    i = _SC.index('def _spring_update(')
    fn = _SC[i:_SC.index('\n    def ', i + 10)]
    _check("prev.get('bar_t') == _t" in fn, 'перерахунок лише на новому закритому барі')
    _check("'spring': spr" in _SC, 'персист через рестарт')


def test_queue4_priority_and_gate_are_wired():
    i = _FF.index('def _engine_tick_queue4(')
    fn = _FF[i:_FF.index('\n    def ', i + 10)]
    _check('priority_key' in fn and 'items.sort' in fn, 'найзарядженіші першими')
    g = fn.index("_sp_mod.gate(_rec, d, 'require'")
    _check(g < fn.index('self._q4_snapshot(sym, s)'), 'ворота стоять ДО важкого знімка')
    _check(fn.index("_sp_mod.gate(_sp.get(str(sym).upper()), _open_dir")
           < fn.index('self._q4_recheck_filters('), 'бік відкриття — до повторної перевірки')
    _check("'spring_mode': 'require'" in _FF and "'spring_min_pot': 40" in _FF, 'дефолти')


def test_ui_panel_and_route():
    for i in ('id="spring-panel"', 'id="spr-mode"', 'id="spr-min"', 'id="spr-tbody"'):
        _check(i in _HTML, i)
    _check("spring_mode: _v('spr-mode'" in _HTML and "setIf('spr-mode'" in _HTML, 'збереження')
    _check("@app.route('/api/spring/state')" in _APP and "fetch('/api/spring/state')" in _HTML, 'маршрут')
    i = _HTML.index('id="spr-table"')
    tbl = _HTML[i:_HTML.index('</table>', i)]
    n = len(re.findall(r'<th[\s>]', tbl))
    _check('colspan="10"' in tbl and n == 10, f'{n} колонок')


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('✓', k); n += 1
    print(f'OK — {n} тестів')
