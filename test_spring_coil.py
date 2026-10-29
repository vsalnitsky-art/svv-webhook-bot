"""🌀 «Пружина» — заряд/напрямок/вистріл монет WATCHLIST + окремі ворота
сигналів «Через Пружину» (вимога 29.10; не прив'язані до черг).

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
    _check("_get_tf_data(SPRING_TF)" in _SC, 'дані — ті самі закриті 1h-бари сканера')
    _check("(getattr(self, '_spring', None) or {}).pop(symbol, None)" in _SC, 'видалення з watchlist чистить запис')
    i = _SC.index('def spring_for(')
    fn = _SC[i:_SC.index('\n    def ', i + 10)]
    _check('get_watchlist' in fn, 'віддаємо лише WATCHLIST')
    i = _SC.index('def _spring_update(')
    fn = _SC[i:_SC.index('\n    def ', i + 10)]
    _check("prev.get('bar_t') == _t" in fn, 'перерахунок лише на новому закритому барі')
    _check("'spring': spr" in _SC, 'персист через рестарт')


def _scanner(settings, spring):
    """Справжній SMCScanner без __init__ (ізольовано) — лише воріт «Пружини»."""
    spec = importlib.util.spec_from_file_location(
        'smc_scanner_spring_t', os.path.join(_ROOT, 'detection', 'smc_scanner.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sc = m.SMCScanner.__new__(m.SMCScanner)
    sc._settings = dict(settings)
    sc._spring = dict(spring)
    return sc


def test_signal_gate_is_separate_from_the_queues():
    """⚠️ 29.10: «Зроби це окремо від Черги-4» — ворота живуть у сканері
    (`_signal_allowed`), а в fuel_filter про «Пружину» немає ні слова."""
    _check('spring' not in _FF.lower(), 'fuel_filter (черги) «Пружину» не знає')
    i = _SC.index('def _signal_allowed(')
    fn = _SC[i:_SC.index('\n    def ', i + 10)]
    g = fn.index('_sg(symbol, side_label)')
    _check(g < fn.index('# OB'), 'ворота стоять ДО OB/PD/Forecast/…')
    _check("'spring_gate_enabled': True" in _SC and "'spring_min_pot': 40" in _SC, 'дефолти')


def test_signal_gate_behaviour():
    rec = {'ok': True, 'pot_long': 70, 'pot_short': 10, 'label': '🌀 заряджена', 'charge': 80}
    sc = _scanner({'spring_min_pot': 40}, {'ETHUSDT': rec})
    app, ok, chip, why = sc._spring_gate('ETHUSDT', 'LONG')
    _check(app and ok and 'LONG 70' in chip and '✓' in chip, chip)
    app, ok, chip, why = sc._spring_gate('ETHUSDT', 'SHORT')
    _check(app and not ok and '✗' in chip and 'SHORT 10 < 40' in why, (chip, why))
    app, ok, chip, why = sc._spring_gate('XRPUSDT', 'SHORT')
    _check(ok and 'немає даних' in chip, 'немає даних → не блокуємо')



def test_gate_decisions_are_counted_for_the_status_line():
    rec = {'ok': True, 'pot_long': 70, 'pot_short': 10, 'label': '🌀', 'charge': 80}
    sc = _scanner({'spring_min_pot': 40}, {'ETHUSDT': rec})
    sc._spring_gate('ETHUSDT', 'LONG'); sc._spring_gate('ETHUSDT', 'SHORT')
    sc._spring_gate('XRPUSDT', 'LONG')
    st = sc.spring_stats()
    _check((st['pass'], st['block'], st['nodata']) == (1, 1, 1), st)
    _check(st['last'][0]['symbol'] == 'XRPUSDT' and st['last'][1]['outcome'] == 'block', st['last'])


def test_scan_progress_is_tracked_even_without_a_new_bar():
    """👁 «Чи сканує» — монета позначається ПЕРЕВІРЕНОЮ кожним проходом,
    навіть коли бар той самий і перерахунку немає (прогрес циклу)."""
    sc = _scanner({}, {})
    sc._scan_count = 7
    bars = _coil(seed=3)
    sc._spring_update('ETHUSDT', bars)
    sc._spring_update('BTCUSDT', bars)
    sc._spring_update('ETHUSDT', bars)          # той самий бар → без перерахунку
    st = sc.spring_stats()
    _check(st['tick_cycle'] == 7 and st['tick_n'] == 3 and st['tick_sym'] == 'ETHUSDT', st)
    _check(st['calc_n'] == 2 and st['calc_at'] > 0, 'перерахунок лише двічі (новий бар)')
    sc._scan_count = 8
    sc._spring_update('ETHUSDT', bars)
    _check(sc.spring_stats()['tick_n'] == 1, 'новий цикл — прогрес з нуля')


def test_master_switch_turns_the_whole_technology_off():
    """🔌 Головний тумблер (деф. УВІМК): ВИМКНЕНО → не рахується, ворота не
    діють, таблиця порожня."""
    _check("'spring_enabled': True" in _SC, 'дефолт УВІМК')
    rec = {'ok': True, 'pot_long': 70, 'pot_short': 10, 'label': '🌀', 'charge': 80}
    sc = _scanner({'spring_enabled': False}, {'ETHUSDT': rec})
    sc.get_watchlist = lambda: ['ETHUSDT']
    _check(sc.spring_for() == {}, 'вимкнено → таблиця порожня')
    sc._settings['spring_enabled'] = True
    _check('ETHUSDT' in sc.spring_for(), 'увімкнено → видно')
    i = _SC.index('def _signal_allowed(')
    fn = _SC[i:_SC.index('\n    def ', i + 10)]
    _check("self._settings.get('spring_enabled', True)" in fn, 'ворота гасить головний тумблер')
    j = _SC.index('td_sp = (_get_tf_data(SPRING_TF)')
    _check("spring_enabled" in _SC[j:j + 160], 'розрахунок у скані гасить головний тумблер')


def test_ui_panel_toggle_and_status():
    for i in ('id="spring-panel"', 'id="spr-enabled"', 'id="spr-gate"', 'id="spr-min"',
              'id="spr-tbody"', 'id="spr-status"', 'id="spr-off"', 'id="spr-live"'):
        _check(i in _HTML, i)
    _check('spring_enabled: !(en && !en.checked)' in _HTML, 'тумблер пишеться в налаштування сканера')
    _check('spring_gate_enabled: !!(el && el.checked)' in _HTML, 'ворота пишуться')
    _check('spring_mode' not in _HTML, 'режиму Черги-4 немає')
    _check("'spr'" in _HTML and 'function _sprStatusPaint' in _HTML, 'гармошка + рядок стану')
    for k in ('spr-bar', 'spr-dot', 'tick_n', 'calc_at', 'ff-countdown'):
        _check(k in _HTML, k)
    _check("'/api/spring/state'" in _APP and "'enabled': enabled" in _APP and "'stats': stats" in _APP, 'маршрут')



def test_route_does_not_use_unimported_names():
    """🐞 29.10: маршрут кликав `time.time()`, а `time` у flask_app глобально не
    імпортовано → NameError → сторінка вічно «Завантаження…». Кожна функція,
    що вживає `time.`, мусить мати власний `import time`."""
    import ast
    t = ast.parse(_APP)
    fn = next(n for n in ast.walk(t) if isinstance(n, ast.FunctionDef) and n.name == 'api_spring_state')
    uses = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == 'time']
    imp = any(isinstance(n, ast.Import) and any(a.name == 'time' and a.asname in (None, 'time') for a in n.names)
              for n in ast.walk(fn))
    _check(not uses or imp, 'api_spring_state вживає time без імпорту')


def test_page_never_hangs_on_loading_silently():
    i = _HTML.index('async function sprLoad(')
    fn = _HTML[i:_HTML.index('\n}\n', i)]
    for k in ("fail('сервер відповів HTTP", "fail('помилка на сервері", "fail('немає звʼязку"):
        _check(k in fn, k)
    _check('запит стану «Пружини» до бота' in _HTML, 'початковий напис у рядку стану')


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('✓', k); n += 1
    print(f'OK — {n} тестів')
