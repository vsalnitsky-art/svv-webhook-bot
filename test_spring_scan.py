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


def _mk(db=None, queued=None):
    calls = {'fetch': [], 'queue': []}

    def fetch_factory(ex):
        def f(sym):
            calls['fetch'].append((ex, sym))
            return _bars('coil' if sym.startswith('COIL') else 'wild')
        return f

    def run_q(name, fn, source):
        calls['queue'].append(name)
        return fn()

    eng = ss.SpringScan(db=db, tickers=lambda ex: dict(TICKERS),
                        fetch_factory=fetch_factory, run_queued=queued or run_q, workers=2)
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
    assert res['scanned'] == 2 and res['charged'] == 1
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
    assert eng.settings() == {'exchange': 'binance', 'min_vol_usd': 60e6}
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
    for bad in ('on_signal', 'intercept', '_open', 'manual_open', 'notify_category', 'tg_send'):
        assert bad not in names, bad


def test_routes_and_ui():
    fl = open(os.path.join(ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    assert "/api/spring-scan/state" in fl and "/api/spring-scan/run" in fl
    i = fl.index('def api_spring_scan_state')
    assert 'import time' in fl[i:i + 400]
    for id_ in ('xspr-panel', 'xspr-exchange', 'xspr-minvol', 'xspr-run', 'xspr-viz',
                'xspr-tbody', 'xspr-agg', 'xspr-caret', 'xspr-body'):
        assert f'id="{id_}"' in HTML, id_
    assert "'xspr'" in HTML.split('const _PANEL_IDS = ')[1].split('\n')[0]
    assert '<option value="50000000" selected>' in HTML
    assert 'value="bybit"' in HTML and 'value="binance"' in HTML
    # рядок — той самий спільний рендер «Пружини»
    j = HTML.index('function xsprRender(')
    assert '_sprRowHTML(' in HTML[j:HTML.index('function _xsCostHint')]
    # колонок у заголовку стільки ж, скільки colspan порожнього рядка
    t = HTML[HTML.index('id="xspr-table"'):]
    t = t[:t.index('</thead>')]
    assert len(re.findall(r'<th[ >]', t)) == 13 and 'colspan="13"' in HTML


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


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print('ok', fn.__name__)
    print(f'{len(fns)} passed')
