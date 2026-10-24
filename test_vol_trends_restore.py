"""💾 ▲/▼ Volumized-тренд watchlist ПЕРЕЖИВАЄ РЕСТАРТ (вимога 25.10).

Скарга: після рестарту watchlist і «Watchlist консенсус · ▲ Volumized» щоразу
починали з нуля — ● тренд-точки вже зберігались у БД, а ▲/▼ (з них фронт
рахує консенсус) жили лише в памʼяті. Тепер останній відомий стан пишеться в
кінці кожного скан-циклу і показується одразу після рестарту (приглушено, з
позначкою ↻), доки монету не просканує перший свіжий прохід.
"""
import os, sys, types, threading, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection'); _pkg.__path__ = []
    sys.modules['detection'] = _pkg
if 'detection.market_data' not in sys.modules:
    _md = types.ModuleType('detection.market_data'); _md.get_market_data = lambda: None
    sys.modules['detection.market_data'] = _md
sm = _load('smc_scanner_volrest', 'detection/smc_scanner.py')
_SRC = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _DB:
    def __init__(self):
        self.d = {}

    def get_setting(self, k, default=None):
        return self.d.get(k, default)

    def set_setting(self, k, v):
        self.d[k] = v


def _sc(db, tf='5m', wl=('BTCUSDT', 'ETHUSDT', 'SOLUSDT')):
    s = sm.SMCScanner.__new__(sm.SMCScanner)
    s.db = db
    s._lock = threading.RLock()
    s._settings = {'volumized_timeframe': tf, 'use_volumized_ob': True}
    s._watchlist = list(wl)
    s._volumized_trend_cache = {}
    s._vol_trends_persisted = {}
    s._vol_trends_persisted_tf = None
    return s


def test_trends_survive_a_restart():
    db = _DB()
    a = _sc(db)
    a._volumized_trend_cache = {'BTCUSDT': {'trend': 'LONG', 'tf': '5m'},
                                'ETHUSDT': {'trend': 'SHORT', 'tf': '5m'}}
    a._persist_vol_trends_state()
    b = _sc(db)                      # «рестарт»: памʼять порожня
    b._load_vol_trends_state()
    view, restored = b._vol_trends_view()
    _check(view == {'BTCUSDT': 'LONG', 'ETHUSDT': 'SHORT'}, view)
    _check(sorted(restored) == ['BTCUSDT', 'ETHUSDT'], restored)
    print('✓ ▲/▼ відновлюються після рестарту')


def test_fresh_scan_wins_over_the_snapshot():
    db = _DB()
    a = _sc(db)
    a._volumized_trend_cache = {'BTCUSDT': {'trend': 'LONG', 'tf': '5m'}}
    a._persist_vol_trends_state()
    b = _sc(db); b._load_vol_trends_state()
    b._volumized_trend_cache = {'BTCUSDT': {'trend': 'SHORT', 'tf': '5m'}}
    view, restored = b._vol_trends_view()
    _check(view['BTCUSDT'] == 'SHORT' and not restored, (view, restored))
    print('✓ свіжий скан сильніший за знімок минулого запуску')


def test_snapshot_never_reaches_the_trading_cache():
    db = _DB()
    a = _sc(db)
    a._volumized_trend_cache = {'BTCUSDT': {'trend': 'LONG', 'tf': '5m'}}
    a._persist_vol_trends_state()
    b = _sc(db); b._load_vol_trends_state()
    _check(b._volumized_trend_cache == {},
           'відновлене лише для ПОКАЗУ — дедуп сигналів читає _volumized_trend_cache')
    print('✓ знімок минулого запуску не впливає на торгівлю')


def test_other_timeframe_or_old_snapshot_is_ignored():
    db = _DB()
    a = _sc(db, tf='5m')
    a._volumized_trend_cache = {'BTCUSDT': {'trend': 'LONG', 'tf': '5m'}}
    a._persist_vol_trends_state()
    b = _sc(db, tf='15m'); b._load_vol_trends_state()
    _check(b._vol_trends_view()[0] == {}, 'інший TF → чужий тренд не показуємо')
    db.d[sm.DB_KEY_VOL_TRENDS_STATE]['ts'] -= sm.VOL_TRENDS_RESTORE_MAX_AGE + 1
    c = _sc(db, tf='5m'); c._load_vol_trends_state()
    _check(c._vol_trends_view()[0] == {}, 'надто старий знімок не показуємо')
    print('✓ інший TF і давній знімок ігноруються')


def test_unscanned_coins_keep_their_old_value_on_the_next_persist():
    db = _DB()
    a = _sc(db)
    a._volumized_trend_cache = {'BTCUSDT': {'trend': 'LONG', 'tf': '5m'},
                                'ETHUSDT': {'trend': 'SHORT', 'tf': '5m'}}
    a._persist_vol_trends_state()
    b = _sc(db); b._load_vol_trends_state()
    b._volumized_trend_cache = {'BTCUSDT': {'trend': None, 'tf': '5m'}}
    b._persist_vol_trends_state()    # цикл після рестарту ще не дійшов до ETH
    _check(db.d[sm.DB_KEY_VOL_TRENDS_STATE]['trends'] == {'ETHUSDT': 'SHORT'},
           db.d[sm.DB_KEY_VOL_TRENDS_STATE])
    print('✓ ще не скановані монети не губляться при наступному записі')


def test_wiring():
    _check('self._persist_vol_trends_state()' in _SRC, 'запис у кінці циклу')
    i = _SRC.index('self._settings = self._load_settings()')
    _check(_SRC.index('self._load_vol_trends_state()') > i,
           'завантаження ПІСЛЯ налаштувань (звіряється TF)')
    _check("'volumized_restored': _vol_restored" in _SRC, 'state віддає відновлені')
    _check('wlVolRestored' in _HTML and "opacity:0.5" in _HTML,
           'відновлені ▲/▼ приглушені на сторінці')
    print('✓ провідка бекенд → сторінка')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for f in fns:
        f()
    print(f'\nOK — {len(fns)} тестів')
