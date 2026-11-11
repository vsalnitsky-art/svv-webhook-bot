"""⚖ Банер «SMC Forecast» у БОТІ (вимога 11.11): таймер веде бот і пам'ятає
після рестарту, ворота за банером, UI бота й інфосайт — одне джерело."""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def _mod(name='sb_test'):
    p = os.path.join(HERE, 'detection', 'spring_banner.py')
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _DB:
    def __init__(self, data=None):
        self.data = dict(data or {})

    def get_setting(self, k, d=None):
        return self.data.get(k, d)

    def set_setting(self, k, v):
        self.data[k] = v


AGG_L = {'lean': 'LONG', 'prob_long': 70.0, 'enough': True, 'coins': 5, 'total': 40, 'need': 4}
AGG_S = {'lean': 'SHORT', 'prob_long': 30.0, 'enough': True, 'coins': 5, 'total': 40, 'need': 4}


class _Eng:
    def __init__(self, agg, ts=1000.0):
        self.agg, self.ts = agg, ts

    def state(self):
        return {'result': {'exchange': 'binance', 'ts': self.ts}}

    def banner_agg(self):
        return self.agg


class _Sc:
    def __init__(self, agg):
        self.agg = agg

    def spring_banner(self):
        return self.agg

    def spring_stats(self):
        return {'calc_at': 1000.0}


def test_timer_survives_restart():
    db = _DB()
    m = _mod('sb1')
    m._db = lambda: db
    m._BOOT = 0
    out = m.compute(_Sc(AGG_L), _Eng(AGG_L), now=5000.0, force=True)
    a = [x for x in out if x['key'] == 'all'][0]
    assert a['lean'] == 'LONG' and a['since'] == 1000.0
    assert db.data[m.DB_KEY]['all']['since'] == 1000.0
    # «рестарт»: новий екземпляр, сторона WL ще без даних (прогрів) → таймер той самий
    m2 = _mod('sb2')
    m2._db = lambda: db
    m2._BOOT = 9000.0
    out = m2.compute(_Sc(None), _Eng(AGG_S), now=9010.0, force=True)
    a = [x for x in out if x['key'] == 'all'][0]
    assert a['lean'] == 'LONG' and a['since'] == 1000.0, a


def test_no_data_is_not_flat():
    m = _mod('sb3')
    assert m.lean_of(None) is None
    assert m.lean_of({'lean': None, 'coins': 0, 'total': 0}) is None
    assert m.lean_of({'lean': None, 'coins': 1, 'total': 30}) == 'FLAT'


def test_banner_gate_decide():
    p = os.path.join(HERE, 'detection', 'spring_gates.py')
    spec = importlib.util.spec_from_file_location('sg_t', p)
    sg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sg)
    assert sg.banner_decide('⚖Binance', 'LONG', 'LONG')[1] is True
    app, ok, chip, why = sg.banner_decide('⚖Binance', 'LONG', 'SHORT')
    assert app and not ok and 'SHORT' in why
    assert sg.banner_decide('⚖Binance', None, 'LONG')[1] is False
    src = open(os.path.join(HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    assert "'spbx_gate_enabled': False" in src and "'spbw_gate_enabled': False" in src
    assert 'banner_decide' in src


def test_bot_owns_the_banner():
    ss = open(os.path.join(HERE, 'detection', 'spring_scan.py'), encoding='utf-8').read()
    assert '_spring_banner().compute(eng=self)' in ss, 'таймер веде потік бота'
    fa = open(os.path.join(HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    assert "/api/spring-banner" in fa
    ip = open(os.path.join(HERE, 'web', 'info_portal.py'), encoding='utf-8').read()
    assert 'return _sb().compute(sc, eng, now)' in ip
    html = open(os.path.join(HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
    assert 'id="smcf-panel"' in html and "fetch('/api/spring-banner'" in html
    assert html.index('id="smcf-panel"') < html.index('id="spring-panel"')
    assert 'sm-spbx-gate' in html and 'sm-spbw-gate' in html


def test_infosite_fits_screen():
    css = open(os.path.join(HERE, 'infosite', 'style.css'), encoding='utf-8').read()
    assert 'grid-template-columns: minmax(0, 1fr)' in css and '.wrap > * { min-width: 0; }' in css
    assert '#rec-tbl .g1 { background' not in css, 'групи без окремого тла — без «кусків»'


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK — {len(fns)} тестів')
