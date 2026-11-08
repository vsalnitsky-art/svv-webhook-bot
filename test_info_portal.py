"""👤 Інфосайт — особиста сторінка користувача (вимога 08.11).

Стережемо: (1) особистий список монет — свій на кожного, без дублів і зі
стелею; (2) монета з WATCHLIST бота читається ЛИШЕ з памʼяті сканера (нуль
запитів до біржі); (3) монета ПОЗА ботом рахується фоновим потоком тими самими
функціями, а запит сторінки біржу не чекає; (4) спільний кеш + `sig`;
(5) сторінка більше не тягне старих важких блоків.
"""
import os
import sys
import threading
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Пакет `detection` без його __init__ (той тягне pybit/requests): підмодулі
# smc_structure / ob_detector / volumized_ob вантажаться самі, як у сканері.
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg
# Модуль вантажимо напряму: `web/__init__.py` тягне весь Flask-застосунок.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('info_portal', os.path.join(_HERE, 'web', 'info_portal.py'))
ip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ip)


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _k(t, p):
    return {'t': t * 1000, 'o': p, 'h': p * 1.01, 'l': p * 0.99, 'p': p, 'v': 1}


class _Sc:
    VOB_KLINES_LIMIT = 3000
    OB_ZONES_SHOW = 8

    def __init__(self):
        self._lock = threading.RLock()
        self._settings = {'pd_zone_timeframe': '1h', 'ob_filter_timeframe': '1h',
                          'use_volumized_ob': True, 'volumized_timeframe': '5m',
                          'poc_filter_market': 'futures', 'poc_filter_tf': '1h',
                          'poc_filter_window_days': 3}
        self._cache = {'BTCUSDT': {'klines': [_k(1000 + i * 900, 100 + i) for i in range(500)],
                                   'updated_at': time.time()}}
        self._ob_zones = {'BTCUSDT': {'tf': '1h', 'internal': [
            {'bias': 'BULLISH', 'bar_high': 120.0, 'bar_low': 110.0,
             'bar_time': 2_000_000_000_000, 'created_by_tag': 'CHoCH'}], 'swing': [
            {'bias': 'BULLISH', 'bar_high': 120.0, 'bar_low': 110.0,
             'bar_time': 2_000_000_000_000, 'created_by_tag': 'CHoCH'}]}}

    def get_watchlist(self):
        return ['BTCUSDT', 'ETHUSDT']

    def get_pd_pct(self, s):
        return 18.0 if s == 'BTCUSDT' else None

    def get_pd_thresholds(self):
        return (75.0, 25.0)

    def get_display_label(self):
        return '15M'

    def get_timeframe(self):
        return '15m'

    def get_internal_size(self):
        return 5

    def _volumized_zones(self, s):
        return [{'type': 'Bear', 'top': 130.0, 'bottom': 125.0, 'start_time_sec': 2_000_000_000,
                 'breaker': False, 'tf': '5m', 'pct': 60}] if s == 'BTCUSDT' else []

    def _format_vob(self, o, vtf):
        return {'type': 'Bull', 'top': 1, 'bottom': 0.5, 'start_time_sec': 1, 'breaker': False,
                'tf': vtf, 'pct': 50}

    def _compute_pd_pct(self, kl, piv, price):
        return 81.0


def _reset():
    with ip._lock:
        ip._coin_cache.clear(); ip._poc_cache.clear()
        ip._ext.clear(); ip._ext_want.clear(); ip._poc_want.clear()


def test_personal_list_rules():
    _check(ip.norm_symbol('btc') == 'BTCUSDT' and ip.norm_symbol('eth-usdt') == 'ETHUSDT', 'назва')
    _check(ip.iwl_norm(['btc', 'BTCUSDT', 'x!', 'SOL']) == ['BTCUSDT', 'SOLUSDT'], 'дублі / сміття')
    _check(len(ip.iwl_norm(['C%dUSDT' % i for i in range(100)])) == ip.IWL_MAX, 'стеля')
    cur = ip.user_info({}, ['BTCUSDT'])
    _check(cur == {'wl': ['BTCUSDT', 'ETHUSDT'], 'sel': 'BTCUSDT'}, 'стартовий набір для нового користувача')
    cur, err = ip.apply_change(cur, {'add': 'pepe'}, ['BTCUSDT'])
    _check(not err and cur['wl'][-1] == 'PEPEUSDT' and cur['sel'] == 'PEPEUSDT',
           'монету поза ботом теж можна додати (вимога 08.11)')
    cur, err = ip.apply_change(cur, {'add': '!!'}, [])
    _check(err, 'некоректна назва → помилка')
    cur, _ = ip.apply_change(cur, {'remove': 'PEPEUSDT'}, [])
    _check('PEPEUSDT' not in cur['wl'] and cur['sel'] in cur['wl'], 'видалення обраної → обирається інша')
    cur, _ = ip.apply_change(cur, {'sel': 'ETHUSDT'}, [])
    _check(cur['sel'] == 'ETHUSDT', 'вибір монети')
    print('✓ особистий список: дублі, стеля, будь-яка монета, обрана монета')


def test_bot_coin_reads_memory_only():
    _reset()
    sc = _Sc()
    calls = []
    md = types.ModuleType('detection.market_data')
    md.get_market_data = lambda: calls.append(1)
    sys.modules['detection.market_data'] = md
    ip._kick = lambda: None
    p = ip.coin_payload('btc', sc)
    _check(p['ok'] and p['src'] == 'bot' and len(p['ohlc']) == ip.OHLC_BARS, p.get('reason'))
    _check(p['pd'] == {'pct': 18.0, 'zone': 'discount', 'tf': '1H'}, p['pd'])
    _check(len(p['ob']) == 1 and p['ob'][0]['start'] == 2_000_000_000, 'OB без дублів між сім\'ями, час у секундах')
    _check(p['vob'][0]['side'] == 'SHORT' and p['vob'][0]['tf'] == '5M', p['vob'])
    _check(not calls, 'монета бота — жодного звернення до біржі')
    _check('BTCUSDT' in ip._poc_want, 'POC поставлено фоновому потоку, а не рахується в запиті')
    _check(ip.coin_payload('BTCUSDT', sc) is p, 'спільний кеш на монету')
    print('✓ монета бота: лише памʼять сканера, спільний кеш')


def test_ext_coin_never_blocks_the_request():
    _reset()
    sc = _Sc()
    ip._kick = lambda: None
    p = ip.coin_payload('PEPE', sc)
    _check(not p['ok'] and p['src'] == 'ext' and '⏳' in p['reason'], p)
    _check('PEPEUSDT' in ip._ext_want, 'монету поставлено в чергу фонового потоку')
    _check('PEPEUSDT' not in ip._coin_cache, '«рахується» не кешуємо')
    rows = ip.rows_for(['BTCUSDT', 'PEPEUSDT'], sc)
    _check([r['src'] for r in rows] == ['bot', 'ext'], rows)
    print('✓ монета поза ботом: запит не чекає біржу, «⏳ рахується»')


def test_ext_compute_uses_the_scanner_functions():
    _reset()
    sc = _Sc()
    bars = [_k(1000 + i * 900, 100 + (i % 7)) for i in range(800)]
    md = types.ModuleType('detection.market_data')
    md.get_market_data = lambda: types.SimpleNamespace(
        fetch_klines=lambda s, limit=0, interval='': bars[-limit:])
    sys.modules['detection.market_data'] = md
    upd = ip._ext_compute(sc, 'PEPEUSDT', {'main', 'vob', 'slow'})
    _check(len(upd['main']) == ip.OHLC_BARS and upd['pd_pct'] == 81.0, 'бари + PD тією ж _compute_pd_pct')
    _check(upd['zones'].get('tf') == '1h' and 'internal' in upd['zones'], 'OB — detect_order_blocks')
    with ip._lock:
        ip._ext['PEPEUSDT'] = dict(upd, err='')
        ip._ext_want['PEPEUSDT'] = time.time()
    p = ip.coin_payload('PEPEUSDT', sc)
    _check(p['ok'] and p['pd']['pct'] == 81.0 and p['pd']['zone'] == 'premium', p)
    print('✓ монета поза ботом: ті самі функції детекції й параметри сканера')


def test_ext_due_schedule():
    now = 10_000.0
    _check(ip.ext_due({}, now, '5m', '1h') == {'main', 'vob', 'slow'}, 'перший раз — усе')
    rec = {'main_at': now - 5, 'vob_at': now - 5, 'slow_at': now - 5}
    _check(ip.ext_due(rec, now, '5m', '1h') == set(), 'свіже — нічого')
    rec = {'err': 'x', 'err_at': now - 5}
    _check(ip.ext_due(rec, now, '5m', '1h') == set(), 'після помилки — пауза')
    print('✓ розклад перерахунку монет поза ботом')


def test_sig_skips_unchanged_payload():
    _reset()
    sc = _Sc()
    ip._kick = lambda: None
    p = ip.coin_payload('BTCUSDT', sc)
    _check(p['sig'] and p['sig'] == ip.sig_of(p), 'sig')
    print('✓ сигнатура відповіді графіка')


def test_wiring_and_page():
    fl = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    au = open(os.path.join(_HERE, 'web', 'auth.py'), encoding='utf-8').read()
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    html = open(os.path.join(_HERE, 'infosite', 'index.html'), encoding='utf-8').read()
    _check('register_info_portal(app)' in fl, 'маршрути зареєстровано')
    _check("'GET, POST, HEAD, OPTIONS' if _own" in fl, 'крос-доменний POST лише для /api/me')
    _check("'/api/info/'" in au, 'доступ без bot_access')
    for u in ('/api/me/watchlist', '/api/info/rows', '/api/info/coin', '/api/info/universe'):
        _check(u in js, u)
    for old in ('/api/fuel-filter/state', '/api/tm/state', '/api/sm/bias', 'funding-table', 'btc-card'):
        _check(old not in js and old not in html, f'старе наповнення лишилось: {old}')
    _check('id="iw-list"' in html and 'id="iw-box"' in html and 'lightweight-charts@4.1.3' in html, 'розмітка')
    _check('vsv-logo.png' in html and 'id="auth-links"' in html, 'логотип і кабінет лишились')
    print('✓ маршрути, доступ, нова сторінка без старих блоків')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
