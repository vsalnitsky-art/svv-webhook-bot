"""⚠ Сигнали «Аномалії обсягу» на графіку кабінету · TF High/Low · «Обсяг»
вимкнений за замовчуванням (вимога 24.11).

Дослівно: «Зроби для кнопок (Обсяг / Аномалії обсягу) вимкнений режим постійно,
спрацьовує разово… запамʼятовувати вибір не потрібно. Також… для High Low додай
вибір таймфреймів» · «Для цих сигналів (сценарій входу LONG / SHORT) також
створи алгоритм відображення на графіку і кнопку по типу SBR».

python test_anom_chart_2411.py
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import test_anom_alarm as T      # noqa: E402  (ai_signals + info_portal зі стабами)

A, IP, NOW = T.A, T.IP, T.NOW
JS = open(os.path.join(ROOT, 'infosite', 'js', 'app.js'), encoding='utf-8').read()
TOOLS = open(os.path.join(ROOT, 'infosite', 'js', 'chart-tools.js'), encoding='utf-8').read()


def _r(key='E2', last_t=NOW - 300):
    return {'score': 82, 'scenario': {'key': key, 'name': '💥 Старт розряду', 'last_t': last_t,
                                      'stages': [{'label': 'заряд', 'done': True},
                                                 {'label': 'імпульс', 'done': True},
                                                 {'label': 'ще ні', 'done': False}]}}


def test_engine_remembers_marks_once_per_scenario():
    eng = A.AiEngine(db=None, clock=lambda: NOW)
    assert eng._anom_mark('BTCUSDT', 'LONG', _r(), 100.0, {'state': 'dir', 'dir': 'LONG'}, NOW)
    # той самий сценарій (бік · ключ · остання реакція) — без дубля
    assert not eng._anom_mark('BTCUSDT', 'LONG', _r(), 101.0, None, NOW + 30)
    # новий сценарій — нова позначка
    assert eng._anom_mark('BTCUSDT', 'SHORT', _r('E1', NOW - 60), 99.0, None, NOW + 60)
    ms = eng.anom_marks('btcusdt')
    assert [m['side'] for m in ms] == ['LONG', 'SHORT']
    assert ms[0]['stages'] == ['заряд', 'імпульс'] and ms[0]['adir'] == 'LONG' and ms[0]['px'] == 100.0
    # старші за тиждень — не віддаються
    old = A.AiEngine(db=None, clock=lambda: NOW + A.ANOM_MARK_KEEP + 3600)
    old._anom_marks = eng._anom_marks
    assert old.anom_marks('BTCUSDT') == []


def test_marks_survive_restart():
    store = {}
    db = types.SimpleNamespace(get_setting=lambda k, d=None: store.get(k, d),
                               set_setting=lambda k, v: store.__setitem__(k, v))
    e1 = A.AiEngine(db=db, clock=lambda: NOW)
    e1._anom_mark('ETHUSDT', 'LONG', _r(), 2000.0, None, NOW)
    e1._persist()
    e2 = A.AiEngine(db=db, clock=lambda: NOW + 60)
    e2._load()
    assert [m['side'] for m in e2.anom_marks('ETHUSDT')] == ['LONG']


def test_portal_marks_only_inside_chart_window():
    eng = A.AiEngine(db=None, clock=lambda: NOW)
    eng._anom_mark('BTCUSDT', 'LONG', _r(last_t=NOW - 7200), 100.0, None, NOW - 7000)
    eng._anom_mark('BTCUSDT', 'SHORT', _r('E1', NOW - 60), 99.0, None, NOW)
    mod = types.SimpleNamespace(get_ai_engine=lambda: eng)
    sys.modules['detection.ai_signals'] = mod
    try:
        got = IP.anom_marks_of('BTCUSDT', [{'time': int(NOW - 3600)}])
    finally:
        sys.modules.pop('detection.ai_signals', None)
    assert [g['side'] for g in got] == ['SHORT']           # раніша — за межею графіка
    assert got[0]['ct'] == int(NOW) and got[0]['name']     # момент спрацювання


def test_prefs_shl_tf_and_anom_layer():
    info = IP.user_info({}, ['BTCUSDT'])
    assert info['shl_tf'] == '1h' and info['show_anom'] is False
    cur, err = IP.apply_change(info, {'shl_tf': '4h', 'show_anom': True}, ['BTCUSDT'])
    assert not err and cur['shl_tf'] == '4h' and cur['show_anom'] is True
    assert IP.apply_change(cur, {'shl_tf': 'zz'}, ['BTCUSDT'])[0]['shl_tf'] == '4h'


def test_page_volume_off_and_buttons():
    # «Обсяг» / «Аномалії обсягу» — завжди вимкнені на відкритті, вибір не зберігається
    assert 'function volInd(k) { return false; }' in JS
    assert 'T.ind.volume=false;T.ind.volanom=false;' in TOOLS
    # High/Low — свій TF; ⚠ Сигнали — кнопка шару, позначки іншим кольором
    for k in ('id="iw-shl-tf"', '"iw-shl-tf": "shl_tf"', 'data-layer="show_anom"', 'function anomOf(c)',
              'ANOM_LONG', 'ANOM_SHORT', 'kind: "anom"', 'function anomCard(', '"&anom=1"'):
        assert k in JS, k
    assert 'ly_anom' in open(os.path.join(ROOT, 'web', 'user_perms.py'), encoding='utf-8').read()


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
            n += 1
    print(f'OK {n}')
