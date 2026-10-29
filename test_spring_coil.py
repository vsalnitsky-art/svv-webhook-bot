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


# ── ⚖ КУДИ НАКРУЧЕНО + 🎯 ЙМОВІРНИЙ ХІД (вимога 29.10) ──────────────────────
def test_projection_points_where_the_spring_is_wound():
    r = S.analyze(_coil(up=True))
    _check(r['lean'] == 'LONG' and r['prob_long'] > 50, r)
    _check(abs(r['prob_long'] - 100 * r['pot_long'] / (r['pot_long'] + r['pot_short'])) < 0.2, 'ймовірність = частка натягу')
    _check(abs(r['lean_pp'] - abs(r['pot_long'] - r['pot_short'])) < 0.2, 'перевага в пунктах')
    _check(r['tgt_long'] > r['box_hi'] and r['tgt_short'] < r['box_lo'], 'цілі за межами коробки')
    d = S.analyze(_coil(up=False))
    _check(d['lean'] == 'SHORT' and d['prob_long'] < 50, d)


def test_projection_is_a_measured_move_with_atr_floor():
    p = S.projection(100.0, 102.0, 98.0, 1.0, 60, 20)
    _check(p['tgt_long'] == 106.0 and p['tgt_short'] == 94.0 and p['move_long_pct'] == 6.0, p)
    tiny = S.projection(100.0, 100.2, 99.8, 1.0, 10, 10)
    _check(tiny['tgt_long'] == 100.2 + 1.5 and tiny['lean'] is None and tiny['prob_long'] == 50.0, tiny)
    z = S.projection(100.0, 101.0, 99.0, 1.0, 0, 0)
    _check(z['prob_long'] == 50.0, 'нуль заряду → сторону не вигадуємо')
    f = S.projection(105.0, 102.0, 98.0, 1.0, 60, 0, {'side': 'LONG', 'level': 102.0})
    _check(f['tgt_long'] == 106.0, 'після вистрілу ціль від рівня пробою')


def test_aggregate_ignores_flat_coins_and_picks_the_strongest():
    recs = {'A': {'ok': True, 'state': 'charged', 'charge': 80, 'pot_long': 70, 'pot_short': 10, 'move_long_pct': 3.0},
            'B': {'ok': True, 'state': 'building', 'charge': 50, 'pot_long': 10, 'pot_short': 40, 'move_short_pct': 2.0},
            'C': {'ok': True, 'state': 'flat', 'charge': 20, 'pot_long': 0, 'pot_short': 20},
            'D': {'ok': True, 'state': 'charged', 'charge': 70, 'pot_long': 40, 'pot_short': 30}}
    a = S.aggregate(recs)
    _check(a['coins'] == 3 and a['sum_long'] == 120 and a['sum_short'] == 80 and a['enough'], a)
    _check(a['lean'] == 'LONG' and a['n_long'] == 2 and a['n_short'] == 1, a)
    _check(a['top_long']['symbol'] == 'A' and a['top_short']['symbol'] == 'B', a)
    _check(S.aggregate({})['lean'] is None, 'порожньо → рівновага')


def test_route_and_page_show_the_tug_bar_and_move():
    for k in ("'lean'", "'prob_long'", "'move_long_pct'", "'tgt_short'", "_spc.aggregate(full, total=len(full))", "_spc.projection("):
        _check(k in _APP, k)
    for k in ('function _sprTugHTML(', 'function _sprMoveCell(', 'function _sprAggPaint(', 'id="spr-agg"',
              "sprSort('lean')", "sprSort('move')", 'agg: d.agg'):
        _check(k in _HTML, k)
    i = _HTML.index('<table class="tm-table" id="spr-table">')
    t = _HTML[i:_HTML.index('</table>', i)]
    _check(len(re.findall(r'<th[ >]', t)) == 12 and 'colspan="12"' in t, 'колонок 12')


def test_tug_cells_render_in_node():
    import shutil, subprocess
    if not shutil.which('node'):
        print('  (node немає — пропуск)'); return
    fns = []
    for name in ('_sprLeanMove', '_sprTugHTML', '_sprTugCell', '_sprMoveCell', '_sprAggPaint'):
        i = _HTML.index('function ' + name + '(')
        fns.append(_HTML[i:_HTML.index('\n}\n', i) + 2])
    js = ("const _el={style:{},innerHTML:''};const document={getElementById:()=>_el};"
          "let _sprMeta={agg:{coins:2,n_long:1,n_short:1,sum_long:80,sum_short:50,prob_long:61.5,lean:'LONG',"
          "top_long:{symbol:'AAA',pot:70,move_pct:3},top_short:null}};\n" + '\n'.join(fns) +
          "\nconst r={pot_long:70,pot_short:10,lean:'LONG',prob_long:87.5,lean_pp:60,move_long_pct:3.2,move_short_pct:1,tgt_long:106,tgt_short:99};"
          "console.log(_sprTugCell(r));console.log(_sprMoveCell(r));console.log(_sprMoveCell({}));_sprAggPaint();console.log(_el.innerHTML);")
    out = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
    _check(out.returncode == 0, out.stderr)
    o = out.stdout
    _check('width:35%' in o and 'width:5%' in o and '▲ 88%' in o, o)
    _check('+3.20%' in o and 'width:92px' in o and '—' in o and 'AAA 70' in o and 'LONG 62%' in o, o)


# ── АУДИТ 29.10: «вистріл» лише після стиснення · флет не накручений ─────────
def test_dump_without_squeeze_is_not_a_fired_spring():
    for seed in range(20):
        b = _walk(seed)
        last = b[-1]['p']
        b[-1] = dict(b[-1], p=last * 0.95, h=last * 1.001, l=last * 0.948, v=9000)
        r = S.analyze(b)
        _check(r['state'] != 'fired', f'злив без стиснення ≠ вистріл (seed {seed}): {r["fired"]}')


def test_fired_target_uses_the_pre_breakout_box():
    b = _coil()
    last = b[-1]['p']
    b[-1] = dict(b[-1], p=last * 1.04, h=last * 1.045, l=last * 0.999, v=5000)
    r = S.analyze(b)
    f = r['fired']
    _check(f['pre_sq'] >= S.FIRE_MIN_SQ or f['pre_ratio'] <= S.FIRE_MAX_BOX_RATIO, f)
    w = max(f['box_hi'] - f['box_lo'], S.MOVE_MIN_ATR * r['atr'])
    _check(abs(r['tgt_long'] - (f['level'] + w)) < 1e-9, 'ціль від коробки ДО пробою')
    _check(r['tgt_long'] < r['box_hi'] + (r['box_hi'] - r['box_lo']), 'бар пробою не роздуває хід')


def test_flat_spring_has_no_lean():
    p = S.projection(100.0, 101.0, 99.0, 1.0, 1.0, 0.9, None, charge=2.0)
    _check(p['lean'] is None and p['prob_long'] == 50.0, p)
    q = S.projection(100.0, 101.0, 99.0, 1.0, 40, 10, None, charge=50.0)
    _check(q['lean'] == 'LONG', q)


def test_old_record_version_is_recomputed_on_the_same_bar():
    _check("prev.get('ver') == getattr(sc, 'VERSION', None)" in _SC, 'версія в гейті перерахунку')
    _check(S.analyze(_coil())['ver'] == S.VERSION, 'запис несе версію')


def test_fired_spring_is_released_not_wound():
    b = _coil()
    last = b[-1]['p']
    b[-1] = dict(b[-1], p=last * 1.04, h=last * 1.045, l=last * 0.999, v=5000)
    r = S.analyze(b)
    _check(r['state'] == 'fired' and r['released'] and r['lean'] == 'LONG', r)
    c = S.analyze(_coil(up=False))
    a = S.aggregate({'F': r, 'C': c})
    _check(a['fired_long'] == 1 and a['coins'] == 1 and a['sum_long'] == c['pot_long'],
           f'вистрілена не входить у натяг: {a}')
    i = _HTML.index('function _sprTugCell(')
    fn = _HTML[i:_HTML.index('\n}\n', i)]
    _check('розпрямилась' in fn and fn.index('r.released') < fn.index('_sprTugHTML'), 'вистрілена — без шкали натягу')


# ── АУДИТ 29.10 (2): флет без напрямку · шкала = ефективний натяг · банер ───
def test_flat_coin_has_no_direction():
    for seed in range(10):
        r = S.analyze(_walk(seed))
        if r['state'] == 'flat':
            _check(r['dir'] is None and r['lean'] is None, f'флет без напрямку: {r["dir"]} {r["lean"]}')
            _check(r['tension_long'] == 0 and r['tension_short'] == 0, 'флет без натягу')
    c = S.analyze(_coil())
    _check(c['state'] != 'flat' and c['dir'] == 'LONG', 'накручена пружина зберігає напрямок')
    _check(c['tension_long'] == c['pot_long'] and c['tension_short'] == c['pot_short'], c)
    _check("if r.get('state') == 'flat':" in _APP and "r['dir'] = None" in _APP, 'маршрут для старих записів')


def test_banner_needs_enough_charged_coins():
    recs = {'JST': {'ok': True, 'state': 'building', 'charge': 48, 'pot_long': 42, 'pot_short': 6,
                    'move_long_pct': 2.06, 'move_short_pct': 3.85}}
    for i in range(54):
        recs[f'F{i}'] = {'ok': True, 'state': 'flat', 'charge': 5, 'pot_long': 1, 'pot_short': 4}
    a = S.aggregate(recs, total=55)
    _check(a['coins'] == 1 and not a['enough'] and a['need'] == 6 and a['lean'] is None, a)
    _check(a['top_long']['symbol'] == 'JST' and a['top_short'] is None,
           f'LONG-монета не може бути «найсильнішою SHORT»: {a["top_short"]}')


def test_flat_cells_and_thin_banner_render_in_node():
    import shutil, subprocess
    if not shutil.which('node'):
        print('  (node немає — пропуск)'); return
    fns = []
    for name in ('_sprTugHTML', '_sprTugCell', '_sprPotCell', '_sprAggPaint'):
        i = _HTML.index('function ' + name + '(')
        fns.append(_HTML[i:_HTML.index('\n}\n', i) + 2])
    js = ("const _el={style:{},innerHTML:''};const document={getElementById:()=>_el};"
          "let _sprMeta={agg:{coins:1,total:55,need:6,enough:false,n_long:1,n_short:0,sum_long:42,sum_short:6,"
          "prob_long:87.5,lean:null,top_long:{symbol:'JST',pot:42,move_pct:2.06},top_short:null}};\n" + '\n'.join(fns) +
          "\nconst r={state:'flat',charge:31,pot_long:4,pot_short:27,tension_long:0,tension_short:0,lean:null,prob_long:50};"
          "console.log(_sprTugCell(r));console.log(_sprPotCell(27,'SHORT',40,'require',true));_sprAggPaint();console.log(_el.innerHTML);")
    out = subprocess.run(['node', '-e', js], capture_output=True, text=True, timeout=20)
    _check(out.returncode == 0, out.stderr)
    o = out.stdout
    _check('width:0%' in o and 'width:13.5%' not in o and '⚖ 50%' in o, f'флет — порожня шкала: {o}')
    _check('#4b5563' in o and 'не накручена' in o, 'флет — приглушений потенціал')
    _check('замало заряджених' in o and '88%' not in o and 'JST 42' in o, f'банер без вердикту: {o}')


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('✓', k); n += 1
    print(f'OK — {n} тестів')
