"""🔭 «Пружина — скан біржі» (08.11): ручний скан усіх USDT-перпів біржі,
у таблиці — лише заряджені; той самий `spring_coil.analyze`; через спільну
чергу сканів; прогрес для візуалізації; лише показ."""
import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util  # noqa: E402


def _load(name):
    sp = importlib.util.spec_from_file_location(
        '_t_' + name, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection', name + '.py'))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


ss = _load('spring_scan')
sc = ss._fs()._sc()

ROOT = os.path.dirname(os.path.abspath(__file__))
HTML = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _bars(kind, n=300, now=None):
    """Закриті 1h-бари + один живий. kind='coil' — тиснеться (заряджена),
    'wild' — розхитана (без заряду)."""
    now = now or time.time()
    t0 = (int(now // 3600) - n) * 3600
    out, p = [], 100.0
    for i in range(n + 1):
        if kind == 'coil':
            amp = 3.0 if i < n - 60 else 0.15
            vol = 1000 if i < n - 60 else 300
        else:
            amp = 3.0 + (i % 7) * 0.4
            vol = 1000 + (i % 5) * 200
        p = p + math.sin(i * 0.7) * amp * 0.3
        out.append({'t': (t0 + i * 3600) * 1000, 'o': p, 'h': p + amp, 'l': p - amp,
                    'p': p + amp * 0.1 * math.sin(i), 'v': vol})
    return out


class _DB:
    def __init__(self):
        self.d = {}

    def get_setting(self, k, default=None):
        return self.d.get(k, default)

    def set_setting(self, k, v):
        self.d[k] = json.loads(json.dumps(v))


TICKERS = {
    'COILUSDT': {'vol_usd': 80e6, 'last': 100},
    'WILDUSDT': {'vol_usd': 90e6, 'last': 100},
    'LOWUSDT': {'vol_usd': 10e6, 'last': 100},       # нижче порогу
    'USDCUSDT': {'vol_usd': 900e6, 'last': 1},       # стейбл
    'BTCUSDT251226': {'vol_usd': 900e6, 'last': 1},  # поставочний
    'BTCUSDC': {'vol_usd': 900e6, 'last': 1},        # не USDT
}


def _mk(db=None, queued=None, notify=None, clock=None, kinds=None):
    calls = {'fetch': [], 'queue': []}

    def fetch_factory(ex):
        def f(sym):
            calls['fetch'].append((ex, sym))
            k = (kinds or {}).get(sym) or ('coil' if sym.startswith('COIL') else 'wild')
            return _bars(k)
        return f

    def run_q(name, fn, source):
        calls['queue'].append(name)
        return fn()

    eng = ss.SpringScan(db=db, tickers=lambda ex: dict(TICKERS),
                        fetch_factory=fetch_factory, run_queued=queued or run_q, workers=2,
                        notify=notify, clock=clock)
    return eng, calls


def test_fixture_coin_is_actually_charged():
    rec = sc.analyze([b for b in _bars('coil') if b['t'] / 1000 + 3600 <= time.time()])
    assert rec and rec['state'] == 'charged', rec and (rec['state'], rec['charge'])
    rec2 = sc.analyze([b for b in _bars('wild') if b['t'] / 1000 + 3600 <= time.time()])
    assert rec2 and rec2['state'] != 'charged'


def test_pick_symbols_volume_and_filters():
    syms, vols, brk = ss.pick_symbols(TICKERS, 50e6)
    assert syms == ['WILDUSDT', 'COILUSDT']          # за обігом ↓
    assert brk['low_vol'] == 1 and brk['stable'] == 1 and brk['not_usdt'] == 2
    assert brk['kept'] == 2 and brk['total'] == 6


def test_only_charged_rows_land_in_table():
    eng, calls = _mk()
    res = eng.scan('bybit', 50e6)
    assert [r['symbol'] for r in res['rows']] == ['COILUSDT']
    assert res['rows'][0]['state'] == 'charged'
    assert res['scanned'] == 2 and res['charged'] == 1 and res['added'] == 1
    assert {s for _, s in calls['fetch']} == {'COILUSDT', 'WILDUSDT'}
    assert 'LOWUSDT' not in {s for _, s in calls['fetch']}


def test_same_shape_as_the_spring_tables():
    fs = ss._fs()
    eng, _ = _mk()
    row = eng.scan('bybit', 50e6)['rows'][0]
    for k in fs.ROW_KEYS:
        assert k in row, k
    assert row['vol_usd'] == 80e6


def test_live_bar_is_dropped():
    fs = ss._fs()
    now = time.time()
    b = _bars('coil', now=now)
    assert len(fs.closed_only(b, now)) == len(b) - 1


def test_start_runs_through_shared_queue_and_tracks_progress():
    db = _DB()
    eng, calls = _mk(db=db)
    r = eng.start('binance', 60e6)
    assert r['started']
    for _ in range(200):
        if not eng.is_running():
            break
        time.sleep(0.02)
    assert calls['queue'] == [ss.QUEUE_NAME]
    p = eng.progress()
    assert p['phase'] == 'done' and p['done'] == 2 and p['total'] == 2 and p['found'] == 1
    assert p['found_syms'] == ['COILUSDT'] and len(p['recent']) == 2
    st0 = eng.settings()
    assert st0['exchange'] == 'binance' and st0['min_vol_usd'] == 60e6
    assert st0['min_charge'] == 80 and st0['auto_on'] is False and st0['auto_min'] == 30
    # персист і відновлення після рестарту
    eng2 = ss.SpringScan(db=db)
    st = eng2.state()
    assert st['result']['rows'][0]['symbol'] == 'COILUSDT' and st['result']['restored']
    assert st['settings']['exchange'] == 'binance'


def test_second_click_does_not_double_the_scan():
    import threading
    gate = threading.Event()

    def slow_q(name, fn, source):
        gate.wait(2)
        return fn()

    eng, _ = _mk(queued=slow_q)
    assert eng.start()['started']
    assert eng.start()['started'] is False
    gate.set()


def test_error_is_visible_in_progress():
    def bad_q(name, fn, source):
        raise RuntimeError('біржа лежить')
    eng, _ = _mk(queued=bad_q)
    eng.start()
    for _ in range(100):
        if not eng.is_running():
            break
        time.sleep(0.02)
    p = eng.progress()
    assert p['phase'] == 'error' and 'біржа' in p['error']


def test_norm_settings():
    assert ss.norm_exchange('MEXC') == 'bybit' and ss.norm_exchange('Binance') == 'binance'
    assert ss.norm_min_vol('abc') == ss.DEFAULT_MIN_VOL and ss.norm_min_vol(5) == ss.MIN_VOL_FLOOR


def test_show_only_never_trades():
    src = open(os.path.join(ROOT, 'detection', 'spring_scan.py'), encoding='utf-8').read()
    tree = ast.parse(src)
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | \
            {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for bad in ('on_signal', 'intercept', '_open', 'manual_open', 'tg_send'):
        assert bad not in names, bad
    # Telegram — лише попередження через ОДНЕ місце (_default_notify → тема «🔔 Сигнали»).
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    users = [k for k, f in fns.items()
             if any(isinstance(x, ast.Name) and x.id == 'notify_category' for x in ast.walk(f))]
    assert users == ['_default_notify'], users


def test_routes_and_ui():
    fl = open(os.path.join(ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    assert "/api/spring-scan/state" in fl and "/api/spring-scan/run" in fl
    i = fl.index('def api_spring_scan_state')
    assert 'import time' in fl[i:i + 400]
    assert '/api/spring-scan/settings' in fl and '/api/spring-scan/tg' in fl
    assert "init_spring_scan(db=get_db(), start_loop=True)" in fl
    for id_ in ('xspr-panel', 'xspr-exchange', 'xspr-minvol', 'xspr-run', 'xspr-viz',
                'xspr-tbody', 'xspr-agg', 'xspr-caret', 'xspr-body',
                'xspr-mincharge', 'xspr-auto', 'xspr-auto-min'):
        assert f'id="{id_}"' in HTML, id_
    assert "'xspr'" in HTML.split('const _PANEL_IDS = ')[1].split('\n')[0]
    assert '<option value="50000000" selected>' in HTML
    assert '<option value="5000000">$5M</option>' in HTML
    assert ss.norm_min_vol(5e6) == 5e6
    assert 'value="bybit"' in HTML and 'value="binance"' in HTML
    # рядок — той самий спільний рендер «Пружини»
    j = HTML.index('function xsprRender(')
    assert '_sprRowHTML(' in HTML[j:HTML.index('function _xsCostHint')]
    # колонок у заголовку стільки ж, скільки colspan порожнього рядка
    t = HTML[HTML.index('id="xspr-table"'):]
    t = t[:t.index('</thead>')]
    assert len(re.findall(r'<th[ >]', t)) == 16 and 'colspan="16"' in HTML and 'colspan="15"' not in HTML
    assert 'id="xspr-mincharge" value="80"' in HTML and 'id="xspr-auto-min" value="30"' in HTML


def test_page_script_parses():
    node = subprocess.run(['which', 'node'], capture_output=True, text=True).stdout.strip()
    if not node:
        return
    js = HTML[HTML.index('// ═══ 🔭 «ПРУЖИНА — СКАН БІРЖІ»'):]
    js = js[:js.index("setTimeout(() => { try { xsprLoad(false); }")]
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
    r = subprocess.run([node, '--check', f.name], capture_output=True, text=True)
    os.unlink(f.name)
    assert r.returncode == 0, r.stderr



# ═══ 09.11: фільтр заряду · автоскан · перерахунок · TG по SBR ═══════════════

class _Clock:
    def __init__(self, t):
        self.t = float(t)

    def __call__(self):
        return self.t


def test_min_charge_filter_gates_the_entry():
    eng, _ = _mk()
    eng.update_settings({'min_charge': 99.5})
    res = eng.scan('bybit', 50e6)
    assert res['charged'] == 1 and res['added'] == 0 and res['rows'] == []
    eng.update_settings({'min_charge': 80})
    assert [r['symbol'] for r in eng.scan('bybit', 50e6)['rows']] == ['COILUSDT']
    assert ss.takes_in_work({'state': 'charged', 'charge': 80}, 80)
    assert not ss.takes_in_work({'state': 'charged', 'charge': 79.9}, 80)
    assert not ss.takes_in_work({'state': 'building', 'charge': 99}, 80)


def test_repeat_scan_supplements_the_table():
    kinds = {}
    eng, _ = _mk(kinds=kinds)
    eng.scan('bybit', 50e6)
    assert [r['symbol'] for r in eng.rows()] == ['COILUSDT']
    first_added = eng.rows()[0]['added_at']
    kinds['WILDUSDT'] = 'coil'                      # друга монета зарядилась
    res = eng.scan('bybit', 50e6)
    assert sorted(r['symbol'] for r in eng.rows()) == ['COILUSDT', 'WILDUSDT']
    assert res['added'] == 1
    assert [r for r in eng.rows() if r['symbol'] == 'COILUSDT'][0]['added_at'] == first_added
    # монета таблиці, що вже не заряджена, вибуває і на повному скані
    kinds['COILUSDT'] = 'wild'
    eng.scan('bybit', 50e6)
    assert [r['symbol'] for r in eng.rows()] == ['WILDUSDT']


def test_recalc_touches_only_table_coins_and_drops_uncharged():
    kinds = {}
    eng, calls = _mk(kinds=kinds)
    eng.scan('bybit', 50e6)
    eng.set_tg('COILUSDT', ['1h'])
    calls['fetch'].clear()
    out = eng.recalc()
    assert [s for _, s in calls['fetch']] == ['COILUSDT']   # лише таблиця
    assert out['checked'] == 1 and out['removed'] == []
    kinds['COILUSDT'] = 'wild'
    out = eng.recalc()
    assert out['removed'] == ['COILUSDT'] and eng.rows() == []
    assert 'COILUSDT' not in eng.settings()['tg']              # кнопки — разом із монетою
    assert eng.state()['removed'][0]['symbol'] == 'COILUSDT'


def test_recalc_keeps_charged_below_entry_filter():
    """Фільтр заряду — лише на вхід; лишаємося, поки «🌀 заряджена»."""
    assert ss.stays_in_table({'state': 'charged', 'charge': 66})
    assert not ss.stays_in_table({'state': 'fired', 'charge': 90})
    assert not ss.stays_in_table(None)


def test_fetch_failure_does_not_remove_a_coin():
    eng, _ = _mk()
    eng.scan('bybit', 50e6)

    def boom(ex):
        def f(sym):
            raise RuntimeError('net')
        return f
    eng._fetch_factory = boom
    out = eng.recalc()
    assert out['errors'] == 1 and [r['symbol'] for r in eng.rows()] == ['COILUSDT']


def test_tick_runs_recalc_on_new_bar_and_autoscan_on_schedule():
    clk = _Clock(1_800_000_000 + 30 * 60)
    eng, calls = _mk(clock=clk)
    started = []
    eng.start_recalc = lambda source='bar': started.append('recalc') or {'started': True}
    eng.start = lambda **kw: started.append('scan') or {'started': True}
    assert eng.tick() is None                      # таблиця порожня, авто вимкнено
    eng.update_settings({'auto_on': True, 'auto_min': 30})
    assert eng.tick() == 'scan'
    eng._result = {'ts': clk.t}
    assert eng.tick() is None                      # ще не минуло 30 хв
    clk.t += 30 * 60
    assert eng.tick() == 'scan'
    eng._result = {'ts': clk.t}
    eng._table = {'COILUSDT': {'symbol': 'COILUSDT', 'state': 'charged'}}
    eng._recalc = {'bar': ss.next_bar_due(clk.t) - 3600}
    assert eng.tick() == 'recalc'
    eng._recalc = {'bar': ss.next_bar_due(clk.t)}
    assert eng.tick() is None                      # цей бар уже перераховано
    eng._table = {}
    clk.t += 30 * 60 + 1
    eng._auto_fail_at = clk.t
    eng._result = {'ts': 0}
    assert eng.tick() is None                      # після невдачі — пауза
    clk.t += ss.FAIL_RETRY_SEC
    assert eng.tick() == 'scan'


def test_sbr_signal_in_coin_direction_sends_once():
    sent = []
    eng, _ = _mk(notify=lambda txt: (sent.append(txt) or True, ''))
    eng.scan('bybit', 50e6)
    side = ss.row_side(eng.rows()[0])
    assert side in ('LONG', 'SHORT')
    opp = 'SHORT' if side == 'LONG' else 'LONG'
    sig = lambda s, t=1000: {'side': s, 'close_t': t, 'trigger': 1.5, 'sl': 1.4, 'n': 1, 'score': 80}
    assert eng.on_sbr('COILUSDT', [('1h', sig(side))]) == 0         # кнопки не увімкнені
    eng.set_tg('COILUSDT', ['1h'])
    assert eng.on_sbr('COILUSDT', [('4h', sig(side))]) == 0         # не той TF
    assert eng.on_sbr('COILUSDT', [('1h', sig(opp))]) == 0          # проти напрямку
    assert eng.on_sbr('COILUSDT', [('1h', sig(side))]) == 1
    assert eng.on_sbr('COILUSDT', [('1h', sig(side))]) == 0         # той самий сигнал
    assert len(sent) == 1 and '#COILUSDT' in sent[0] and side in sent[0]
    assert eng.on_sbr('WILDUSDT', [('1h', sig(side))]) == 0         # монети нема в таблиці
    assert eng.set_tg('WILDUSDT', ['1h'])['ok'] is False
    assert eng.tg_symbols() == ['COILUSDT']


def test_table_and_tg_survive_restart():
    db = _DB()
    eng, _ = _mk(db=db)
    eng.scan('bybit', 50e6)
    eng.set_tg('COILUSDT', ['4h', '1h'])
    eng.recalc()
    eng2 = ss.SpringScan(db=db)
    assert [r['symbol'] for r in eng2.rows()] == ['COILUSDT']
    assert eng2.rows()[0]['tg'] == ['1h', '4h']
    assert eng2.state()['recalc'].get('checked') == 1


def test_manual_remove_and_clear():
    eng, _ = _mk(db=_DB(), kinds={'COILUSDT': 'coil', 'COIL2USDT': 'coil'})
    TICKERS['COIL2USDT'] = {'vol_usd': 70e6, 'last': 100}
    try:
        eng.scan('bybit', 50e6)
    finally:
        TICKERS.pop('COIL2USDT', None)
    assert set(eng._table) == {'COILUSDT', 'COIL2USDT'}
    eng.set_tg('COILUSDT', ['1h'])
    r = eng.remove('coilusdt')
    assert r['ok'] and 'COILUSDT' not in eng._table and 'COILUSDT' not in eng.settings()['tg']
    assert eng.state()['removed'][-1]['why'].startswith('✕')
    assert not eng.remove('NOPEUSDT')['ok']
    eng.set_tg('COIL2USDT', ['4h'])
    c = eng.clear()
    assert c['ok'] and c['cleared'] == 1 and not eng._table and not eng.settings()['tg']
    assert eng.rows() == []
    fl = open(os.path.join(ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    assert '/api/spring-scan/remove' in fl
    assert 'id="xspr-clear"' in HTML
    assert 'function xsprRemove(' in HTML and 'function xsprClear(' in HTML and '_xsDelCell(x)' in HTML


def test_recalc_does_not_resurrect_a_removed_coin():
    eng, _ = _mk()
    eng.scan('bybit', 50e6)
    orig = eng._analyze_many

    def am(syms, fetch, now):
        out = orig(syms, fetch, now)
        eng.remove('COILUSDT')          # прибрали вручну посеред перерахунку
        return out
    eng._analyze_many = am
    eng.recalc()
    assert 'COILUSDT' not in eng._table


def test_sbr_engine_feeds_and_calls_spring_scan():
    src = open(os.path.join(ROOT, 'detection', 'sbr_signals.py'), encoding='utf-8').read()
    assert 'from detection.spring_scan import sbr_symbols' in src
    assert 'spring=_default_spring' in src and 'self._spring(S, items)' in src


# ── ✋ ВІДКРИТТЯ УГОДИ З ТАБЛИЦІ СКАНУ (вимога 10.11) ──
def test_open_plan_side_and_refusals():
    assert ss.open_plan(None)[0] is False
    ok, sd, _ = ss.open_plan({'symbol': 'AUSDT', 'dir': 'LONG'})
    assert ok and sd == 'LONG'
    ok, sd, _ = ss.open_plan({'symbol': 'AUSDT', 'dir': None, 'lean': 'SHORT'})
    assert ok and sd == 'SHORT'
    # людина обрала бік ПРОТИ рядка — її вибір сильніший
    ok, sd, _ = ss.open_plan({'symbol': 'AUSDT', 'dir': 'LONG'}, 'short')
    assert ok and sd == 'SHORT'
    ok, sd, why = ss.open_plan({'symbol': 'AUSDT'})
    assert not ok and sd is None and '⚖' in why
    ok, sd, why = ss.open_plan({'symbol': 'AUSDT', 'dir': 'LONG', 'bybit': False})
    assert not ok and 'Bybit' in why


def test_open_route_goes_through_hand_open_with_label():
    src = open(os.path.join(ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    i = src.index("'/api/spring-scan/open'")
    body = src[i:src.index('@app.route', i + 10)]
    assert 'open_plan(' in body and "ff.hand_open(" in body and "'XSPR'" in body
    ff = open(os.path.join(ROOT, 'detection', 'fuel_filter.py'), encoding='utf-8').read()
    j = ff.index('def hand_open(')
    hb = ff[j:ff.index('\n    def ', j + 10)]
    assert 'by_hand=True' in hb and "_ob_compose('manual', engine)" in hb
    assert '_q4_set_vob_sl' in hb and '_tm_blocks_open' in hb


def test_open_button_rendered():
    assert 'function _xsOpenCell(' in HTML and '_xsOpenCell(x) + _xsDelCell(x)' in HTML
    assert "fetch('/api/spring-scan/open'" in HTML
    lab = open(os.path.join(ROOT, 'detection', 'signal_labels.py'), encoding='utf-8').read()
    assert "'XSPR':             '🔭 Пружина-скан'" in lab


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('ok', fn.__name__)
    print(f'{len(fns)} passed')
