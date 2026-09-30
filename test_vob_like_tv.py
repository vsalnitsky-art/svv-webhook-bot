"""🖼 VOB НА ГРАФІКУ ЯК У TRADINGVIEW + ▲/▼ НЕ МІНЯЄ КОЛІР ПІСЛЯ ПЕРЕХОДУ (30.09).

1) TV (Zone Count One) малює ОДИН бичачий + ОДИН ведмежий блок, а бот малював
   лише останній. Тепер `volumized_obs` = усі видимі блоки обох боків.
2) Відкрили монету → панель рахувала свіжий VOB (кеш скану > 30с), трикутник
   ставав кольору останнього VOB; перейшли на іншу → повертався СТАРИЙ колір
   скану, бо свіжий результат не писався в `_volumized_trend_cache`.
"""
import os, sys, types, random, threading, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


vob = _load('vob_tv', 'detection/volumized_ob.py')
# `_latest_volumized_ob` імпортує `detection.volumized_ob` — підкладаємо файл
# напряму (пакетний імпорт тягне pybit).
if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection'); _pkg.__path__ = []
    sys.modules['detection'] = _pkg
sys.modules['detection.volumized_ob'] = vob
if 'detection.market_data' not in sys.modules:
    _md = types.ModuleType('detection.market_data'); _md.get_market_data = lambda: None
    sys.modules['detection.market_data'] = _md
sm = _load('smc_scanner_tv', 'detection/smc_scanner.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _walk(seed, n=1500):
    rnd = random.Random(seed); p = 100.0; out = []
    for i in range(n):
        o = p; p = max(1.0, p * (1 + rnd.gauss(0, 0.004)))
        h = max(o, p) * (1 + abs(rnd.gauss(0, 0.002)))
        l = min(o, p) * (1 - abs(rnd.gauss(0, 0.002)))
        out.append({'t': 1_700_000_000_000 + i * 900_000, 'o': o, 'h': h, 'l': l,
                    'p': p, 'c': p, 'v': rnd.uniform(50, 500)})
    return out


def _sc():
    sc = sm.SMCScanner.__new__(sm.SMCScanner)
    sc._lock = threading.RLock()
    sc._settings = {'use_volumized_ob': True, 'volumized_timeframe': '15m',
                    'volumized_swing_length': 3, 'volumized_zone_count': 'One',
                    'volumized_ob_end_method': 'Wick', 'volumized_max_atr_mult': 3.5,
                    'volumized_combine_obs': True}
    sc._volumized_trend_cache = {}
    return sc


def test_trend_wrapper_exposes_both_sides():
    r = vob.get_latest_ob_trend(_walk(1), swing_length=3, zone_count='One')
    _check('bullish_obs' in r and 'bearish_obs' in r, 'немає списків блоків')
    _check(len(r['bullish_obs']) <= 1 and len(r['bearish_obs']) <= 1, 'One ≠ 1 на бік')
    print('✓ get_latest_ob_trend віддає блоки обох боків')


def test_chart_gets_both_sides_like_tv():
    found = False
    for seed in range(10):
        sc = _sc()
        latest = sc._latest_volumized_ob('X', klines=_walk(seed))
        zones = sc._volumized_zones('X')
        types_ = {z['type'] for z in zones}
        _check(len(zones) <= 2, f'Zone Count One дав {len(zones)} блоків')
        if latest:
            _check(zones and zones[0]['type'] == latest['type']
                   and zones[0]['top'] == latest['top'], 'перший бокс ≠ останній VOB')
        if types_ == {'Bull', 'Bear'}:
            found = True
    _check(found, 'жоден прогін не дав обох боків')
    print('✓ на графік ідуть обидва боки, найновіший першим')


def test_panel_result_is_what_the_watchlist_shows():
    sc = _sc()
    latest = sc._latest_volumized_ob('X', klines=_walk(3))
    want = ('LONG' if latest['type'] == 'Bull' else 'SHORT') if latest else None
    _check(sc._volumized_trend_cache.get('X', {}).get('trend') == want,
           'свіжий VOB панелі не записано в кеш ▲/▼ watchlist')
    print('✓ ▲/▼ у watchlist = колір останнього VOB і після переходу')


def test_page_draws_all_and_keeps_color():
    html = open(os.path.join(_ROOT, 'templates/smart_money.html'), encoding='utf-8').read()
    _check('d.volumized_obs' in html and 'volObPrim.setOBs(_boxes)' in html,
           'графік не малює всі блоки')
    _check('wlVolumizedTrends[currentSymbol] = volTrend' in html,
           'колір трикутника повертається після переходу')
    print('✓ сторінка малює всі блоки і тримає колір трикутника')


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v()
