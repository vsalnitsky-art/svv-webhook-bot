"""🏷 Позначка «останній OB 4H» ПЕРЕЖИВАЄ РЕСТАРТ (вимога 25.10).

Скарга: після рестарту OB 4H у таблицях угод і МММ довго відновлювався —
позначка жила лише в памʼяті й зʼявлялась, коли скан доходив до монети (і
тягнув 700 4h-барів). Тепер знімок пишеться в кінці кожного циклу; поки
4h-бар не закрився, блок відновлюється ТОЧНО (і скан не ходить у мережу),
після закриття — показується як ↻ до першого свіжого перерахунку.
"""
import os, sys, time, types, threading, importlib.util

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
sm = _load('smc_scanner_obmark', 'detection/smc_scanner.py')
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


_OB = {'bias': 'BEARISH', 'bar_high': 101.0, 'bar_low': 99.0, 'bar_time': 1700000000000,
       'bar_idx': 5, 'created_at_idx': 7, 'created_at_t': 1700000600000,
       'created_by_tag': 'CHoCH'}


def _sc(db, wl=('BTCUSDT', 'ETHUSDT')):
    s = sm.SMCScanner.__new__(sm.SMCScanner)
    s.db = db
    s._lock = threading.RLock()
    s._settings = {}
    s._watchlist = list(wl)
    s._ob_mark = {}
    s._ob_mark_restored = set()
    s._ob_htf_cache = {}
    return s


def _seed(s, sym, until, ob=_OB):
    """Як після справжнього `_ob_on_htf`: кеш до закриття бару + позначка."""
    s._ob_htf_cache[(sym, sm.OB_MARK_TF)] = (until, ob)
    s._ob_on_htf = lambda symbol, md, tf, a, b: s._ob_htf_cache[(symbol, tf)][1:] + ('',)
    s._ob_mark_update(sym, None)


def test_current_bar_restores_exactly_and_skips_the_network():
    db = _DB()
    a = _sc(db)
    _seed(a, 'BTCUSDT', time.time() + 3600)
    a._persist_ob_mark_state()
    b = _sc(db); b._load_ob_mark_state()
    m = b.ob_mark_for(['BTCUSDT'])['BTCUSDT']
    _check(m['bias'] == 'BEARISH' and not m.get('restored'), m)
    _check('ob' not in m and 'until' not in m, 'службові поля назовні не йдуть')
    hit = b._ob_htf_cache.get(('BTCUSDT', sm.OB_MARK_TF))
    _check(hit and hit[1]['bias'] == 'BEARISH',
           'бар ще не закрився → блок у кеші, скан не тягне 4h-бари')
    print('✓ поточний 4h-бар: позначка точна, мережа не потрібна')


def test_closed_bar_is_shown_as_restored_until_rescanned():
    db = _DB()
    a = _sc(db)
    _seed(a, 'BTCUSDT', time.time() - 10)
    a._persist_ob_mark_state()
    b = _sc(db); b._load_ob_mark_state()
    _check(b.ob_mark_for(['BTCUSDT'])['BTCUSDT'].get('restored'), 'бар закрився → ↻')
    _check(('BTCUSDT', sm.OB_MARK_TF) not in b._ob_htf_cache,
           'закритий бар у кеш не кладемо — блок мусить перерахуватись')
    _seed(b, 'BTCUSDT', time.time() + 3600)
    _check(not b.ob_mark_for(['BTCUSDT'])['BTCUSDT'].get('restored'),
           'свіжий перерахунок знімає ↻')
    print('✓ закритий бар: ↻ до першого свіжого скану')


def test_no_block_is_restored_too():
    db = _DB()
    a = _sc(db)
    _seed(a, 'ETHUSDT', time.time() + 3600, ob=None)
    a._persist_ob_mark_state()
    b = _sc(db); b._load_ob_mark_state()
    _check(b.ob_mark_for(['ETHUSDT'])['ETHUSDT']['bias'] is None, '«4H —» теж відновлюється')
    _check(b._ob_htf_cache[('ETHUSDT', sm.OB_MARK_TF)][1] is None, 'кеш: блоку немає')
    print('✓ «блоку немає» теж переживає рестарт')


def test_old_snapshot_and_foreign_coins_are_ignored():
    db = _DB()
    a = _sc(db)
    _seed(a, 'BTCUSDT', time.time() + 3600)
    a._persist_ob_mark_state()
    db.d[sm.DB_KEY_OB_MARK_STATE]['marks']['BTCUSDT']['ts'] -= sm.OB_MARK_RESTORE_MAX_AGE + 1
    b = _sc(db); b._load_ob_mark_state()
    _check(b.ob_mark_for(['BTCUSDT']) == {}, 'давню позначку не показуємо')
    a2 = _sc(db); _seed(a2, 'BTCUSDT', time.time() + 3600); a2._persist_ob_mark_state()
    c = _sc(db, wl=('ETHUSDT',)); c._load_ob_mark_state()
    _check(c.ob_mark_for(['BTCUSDT']) == {}, 'монета поза watchlist не відновлюється')
    print('✓ давній знімок і монети поза watchlist ігноруються')


def test_wiring():
    _check('self._persist_ob_mark_state()' in _SRC, 'запис у кінці циклу')
    i = _SRC.index('self._watchlist = self._load_watchlist()')
    _check(_SRC.index('self._load_ob_mark_state()') > i, 'завантаження ПІСЛЯ watchlist')
    _check('o.restored' in _HTML and "r.ob4h.restored" in _HTML,
           'сторінка приглушує ↻ і бачить його в сигнатурі таблиць МММ')
    print('✓ провідка бекенд → сторінка')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for f in fns:
        f()
    print(f'\nOK — {len(fns)} тестів')
