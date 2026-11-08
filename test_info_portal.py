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
                                   'analysis': {'internal': {'events': [
                                       {'from_t': 1_000_000, 'to_t': 1_400_000, 'level': 150.0,
                                        'tag': 'CHoCH', 'dir': 'bull'}]}},
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
        ip._comp.clear(); ip._want.clear(); ip._poc_want.clear()


def test_personal_list_rules():
    _check(ip.IWL_MAX == 20, 'стеля 20 монет (вимога 08.11)')
    _check(ip.norm_symbol('btc') == 'BTCUSDT' and ip.norm_symbol('eth-usdt') == 'ETHUSDT', 'назва')
    _check(ip.iwl_norm(['btc', 'BTCUSDT', 'x!', 'SOL']) == ['BTCUSDT', 'SOLUSDT'], 'дублі / сміття')
    _check(len(ip.iwl_norm(['C%dUSDT' % i for i in range(100)])) == ip.IWL_MAX, 'стеля')
    cur = ip.user_info({}, ['BTCUSDT'])
    _check(cur['wl'] == ['BTCUSDT', 'ETHUSDT'] and cur['sel'] == 'BTCUSDT', 'стартовий набір')
    _check((cur['tf'], cur['ob_tf'], cur['vob_tf']) == ('15m', '1h', '5m'), 'TF за замовчуванням')
    cur, err = ip.apply_change(cur, {'add': 'pepe'}, ['BTCUSDT'])
    _check(not err and cur['wl'][-1] == 'PEPEUSDT' and cur['sel'] == 'PEPEUSDT',
           'монету поза ботом теж можна додати')
    cur, err = ip.apply_change(cur, {'add': '!!'}, [])
    _check(err, 'некоректна назва → помилка')
    cur, _ = ip.apply_change(cur, {'remove': 'PEPEUSDT'}, [])
    _check('PEPEUSDT' not in cur['wl'] and cur['sel'] in cur['wl'], 'видалення обраної → обирається інша')
    cur, _ = ip.apply_change(cur, {'sel': 'ETHUSDT', 'tf': '4h', 'ob_tf': 'off', 'vob_tf': 'zz'}, [])
    _check(cur['sel'] == 'ETHUSDT' and cur['tf'] == '4h' and cur['ob_tf'] == 'off'
           and cur['vob_tf'] == '5m', 'вибір монети і TF (некоректний TF ігнорується)')
    full = {'wl': ['C%dUSDT' % i for i in range(20)], 'sel': 'C0USDT'}
    _c, err = ip.apply_change(full, {'add': 'ZZZ'}, [])
    _check('20' in err, 'понад 20 монет — відмова')
    print('✓ особистий список: дублі, стеля 20, будь-яка монета, TF графіка/OB/VOB')


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
    _check(len(p['ob']) == 1 and p['ob'][0]['start'] == 2_000_000_000, 'OB без дублів, час у секундах')
    _check(p['vob'][0]['side'] == 'SHORT' and p['vob'][0]['tf'] == '5M', p['vob'])
    _check(p['events'] and p['events'][-1]['tag'] == 'CHoCH', 'структура CHoCH/BOS з памʼяті сканера')
    _check(not calls and not ip._want, 'монета бота + TF бота — жодного звернення до біржі')
    _check('BTCUSDT' in ip._poc_want, 'POC поставлено фоновому потоку')
    _check(ip.coin_payload('BTCUSDT', sc) is p, 'спільний кеш на монету')
    print('✓ монета бота на TF бота: лише памʼять сканера, спільний кеш')


def test_own_tf_goes_to_background():
    _reset()
    sc = _Sc()
    ip._kick = lambda: None
    p = ip.coin_payload('BTCUSDT', sc, tf='4h', ob_tf='4h', vob_tf='off')
    _check(not p['ok'] and p['tf'] == '4h' and p['vob'] == [] and p['vob_tf'] == 'off', p)
    _check(('bars', 'BTCUSDT', '4h') in ip._want and ('ob', 'BTCUSDT', '4h') in ip._want,
           'свій TF → фоновий потік, запит біржу не чекає')
    _check(not any(k[0] == 'vob' for k in ip._want), 'VOB вимкнено — не рахуємо')
    p = ip.coin_payload('BTCUSDT', sc, tf='15m', ob_tf='4h')
    _check(p['ok'] and p['ob_wait'] and p not in [v[1] for v in ip._coin_cache.values()],
           'бари з памʼяті, OB 4H ще рахується — не кешуємо')
    print('✓ свій TF графіка / OB / VOB: фоновий розрахунок')


def test_ext_coin_never_blocks_the_request():
    _reset()
    sc = _Sc()
    ip._kick = lambda: None
    p = ip.coin_payload('PEPE', sc)
    _check(not p['ok'] and p['src'] == 'ext' and '⏳' in p['reason'], p)
    _check(('bars', 'PEPEUSDT', '15m') in ip._want, 'монету поставлено в чергу фонового потоку')
    _check(not ip._coin_cache, '«рахується» не кешуємо')
    rows = ip.rows_for(['BTCUSDT', 'PEPEUSDT'], sc)
    _check([r['src'] for r in rows] == ['bot', 'ext'], rows)
    _check(rows[0]['price'] and 'pd' not in rows[0] and 'poc' not in rows[0],
           'у списку лише ціна (PD/POC — на графіку)')
    print('✓ монета поза ботом: запит не чекає біржу, «⏳ рахується»')


def test_compute_uses_the_scanner_functions():
    _reset()
    sc = _Sc()
    bars = [_k(1000 + i * 900, 100 + (i % 7)) for i in range(800)]
    md = types.ModuleType('detection.market_data')
    md.get_market_data = lambda: types.SimpleNamespace(
        fetch_klines=lambda s, limit=0, interval='': bars[-limit:])
    sys.modules['detection.market_data'] = md
    d, until = ip._compute(sc, 'bars', 'PEPEUSDT', '15m')
    _check(len(d['kl']) == ip.OHLC_BARS and isinstance(d['events'], list), 'бари + структура')
    d, until = ip._compute(sc, 'ob', 'PEPEUSDT', '1h')
    _check(d['zones'].get('tf') == '1h' and 'internal' in d['zones'], 'OB — detect_order_blocks')
    _check(until > time.time(), 'кеш OB до закриття бару')
    d, _u = ip._compute(sc, 'pd', 'PEPEUSDT', '1h')
    _check(d['pd_pct'] == 81.0, 'PD тією ж _compute_pd_pct')
    now = time.time()
    with ip._lock:
        for k, t in (('bars', '15m'), ('ob', '1h'), ('vob', '5m'), ('pd', '1h')):
            ip._want[(k, 'PEPEUSDT', t)] = now
        ip._comp[('bars', 'PEPEUSDT', '15m')] = {'data': {'kl': bars, 'events': []}, 'at': now}
        ip._comp[('ob', 'PEPEUSDT', '1h')] = {'data': {'zones': {}}, 'at': now}
        ip._comp[('vob', 'PEPEUSDT', '5m')] = {'data': {'vz': []}, 'at': now}
        ip._comp[('pd', 'PEPEUSDT', '1h')] = {'data': {'pd_pct': 81.0}, 'at': now}
    p = ip.coin_payload('PEPEUSDT', sc)
    _check(p['ok'] and p['pd']['pct'] == 81.0 and p['pd']['zone'] == 'premium', p)
    print('✓ розрахунок — ті самі функції детекції й параметри сканера')


def test_due_schedule():
    now = 10_000.0
    _check(ip.due('bars', {}, now), 'перший раз — рахуємо')
    _check(not ip.due('bars', {'at': now - 5}, now) and ip.due('bars', {'at': now - 40}, now), 'бари — 30с')
    _check(not ip.due('ob', {'at': now - 5, 'until': now + 100}, now), 'OB — до закриття бару')
    _check(ip.due('ob', {'at': now - 5, 'until': now - 1}, now), 'бар закрився — перерахунок')
    _check(not ip.due('vob', {'err': 'x', 'err_at': now - 5}, now), 'після помилки — пауза')
    print('✓ розклад перерахунку частин')


def test_sig_skips_unchanged_payload():
    _reset()
    sc = _Sc()
    ip._kick = lambda: None
    p = ip.coin_payload('BTCUSDT', sc)
    _check(p['sig'] and p['sig'] == ip.sig_of(p), 'sig')
    print('✓ сигнатура відповіді графіка')


def test_logout_revokes_tokens():
    src = open(os.path.join(_HERE, 'web', 'auth.py'), encoding='utf-8').read()
    import ast
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'info_token_revoked')
    g = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'auth', 'exec'), g)
    rev = g['info_token_revoked']
    _check(rev({'info_logout_at': 1000}, 999) and not rev({'info_logout_at': 1000}, 1000),
           'токен, виданий ДО виходу, — недійсний; новий вхід — дійсний')
    _check(not rev({}, 5), 'без виходу — дійсний')
    _check("route('/api/me/logout'" in src and "_revoke_info_tokens(u)" in src
           and "request.args.get('next') == 'info'" in src, 'маршрути виходу')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('"/api/me/logout"' in js and '/logout?next=info' in js, 'сайт закриває і токен, і сесію бота')
    print('✓ вихід: токени відкликано, сесію бота закрито')


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
    _check('class="brand-name">VSV Bot<' in html, '«VSV Bot» виділено на банері')
    _check('id="iw-tfs"' in html and 'id="iw-ob-tf"' in html and 'id="iw-vob-tf"' in html, 'вибір TF')
    _check('createPriceLine' in js and 'title: "POC"' in js and 'setMarkers' in js, 'POC і CHoCH/BOS на графіку')
    _check('pdBadge(r.pd)' not in js and 'pocBadge(r.poc' not in js, 'PD/POC прибрано зі списку')
    print('✓ маршрути, доступ, нова сторінка без старих блоків')


def test_imbalance_detection_and_fill():
    import importlib.util as iu
    sp = iu.spec_from_file_location('imbalance', os.path.join(_HERE, 'detection', 'imbalance.py'))
    m = iu.module_from_spec(sp); sp.loader.exec_module(m)
    T = 1_700_000_000_000
    b = [{'t': T + i * 60000, 'o': 100, 'h': 101, 'l': 99, 'p': 100} for i in range(20)]
    b += [{'t': T + 20 * 60000, 'o': 100, 'h': 101, 'l': 99, 'p': 101},
          {'t': T + 21 * 60000, 'o': 101, 'h': 106, 'l': 100.5, 'p': 105.5},
          {'t': T + 22 * 60000, 'o': 105.5, 'h': 107, 'l': 103, 'p': 106}]
    z = m.detect_imbalances(b)
    _check(len(z) == 1 and z[0]['side'] == 'LONG' and (z[0]['bottom'], z[0]['top']) == (101, 103), z)
    _check(z[0]['t'] == (T + 21 * 60000) // 1000, 'початок зони = середня свічка, секунди')
    b.append({'t': T + 23 * 60000, 'o': 106, 'h': 106.5, 'l': 102, 'p': 102.5})
    z = m.detect_imbalances(b)
    _check(z[0]['top'] == 102 and z[0]['filled_pct'] == 50, 'часткове заповнення зменшує зону')
    b.append({'t': T + 24 * 60000, 'o': 102, 'h': 102.5, 'l': 100, 'p': 100.5})
    _check(all(x['side'] != 'LONG' for x in m.detect_imbalances(b)), 'заповнений розрив зникає')
    tiny = [{'t': T + i * 60000, 'o': 100, 'h': 101 + (i % 2) * 0.001, 'l': 99, 'p': 100} for i in range(40)]
    _check(m.detect_imbalances(tiny) == [], 'шум без розриву')
    print('✓ IMB: визначення, часткове й повне заповнення')


def test_payload_carries_imb_and_sbr():
    _reset()
    sc = _Sc()
    ip._kick = lambda: None
    now = 1000 + 500 * 900 + 10
    kl = sc._cache['BTCUSDT']['klines']
    _check(isinstance(ip.imb_of(kl, '15m', now), list), 'imb_of')
    _check(ip.sbr_of(kl, '15m', now) == [], 'SBR лише на 1H / 4H (вимога 09.11)')
    kl1h = [dict(k, t=(3600 * 400 + i * 3600) * 1000) for i, k in enumerate(kl)]
    sig = ip.sbr_of(kl1h, '1h', 3600 * 400 + 500 * 3600 + 10)
    _check(isinstance(sig, list) and all(set(('side', 't', 'trigger', 'score', 'sl', 'tf')) <= set(x) for x in sig), sig[:1])
    p = ip.coin_payload('BTCUSDT', sc)
    _check('imb' in p and 'sbr' in p, 'відповідь графіка несе IMB і SBR')
    cur, _ = ip.apply_change(ip.user_info({}, []), {'show_imb': False}, [])
    _check(cur['show_imb'] is False and cur['show_sbr'] is True, 'перемикачі шарів зберігаються')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('subscribeClick' in js and 'hitAt' in js and 'sigCard' in js, 'клік по позначці S/B → картка')
    _check('overlaps(' in js and 'taken.push' in js, 'підписи розкладаються без накладань')
    _check('"S" : "B"' in js, 'позначки S / B')
    print('✓ графік: IMB, сигнали SBR, підписи без накладань')


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(_HERE, *rel.split('/')))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def test_imb_carries_volume_and_impulse_share():
    m = _load('imbalance2', 'detection/imbalance.py')
    T = 1_700_000_000_000
    b = [{'t': T + i * 60000, 'o': 100, 'h': 101, 'l': 99, 'p': 100, 'v': 10} for i in range(20)]
    b += [{'t': T + 20 * 60000, 'o': 100, 'h': 101, 'l': 99, 'p': 101, 'v': 100},
          {'t': T + 21 * 60000, 'o': 101, 'h': 106, 'l': 100.5, 'p': 105.5, 'v': 800},
          {'t': T + 22 * 60000, 'o': 105.5, 'h': 107, 'l': 103, 'p': 106, 'v': 100}]
    z = m.detect_imbalances(b)[0]
    _check(z['vol'] == 1000 and z['pct'] == 80, z)
    nov = [dict(x, v=0) for x in b]
    _check(m.detect_imbalances(nov)[0]['pct'] is None, 'немає обсягу → None, а не вигадане число')
    _check(ip.imb_of(b, '1m', (T + 30 * 60000) / 1000)[0]['vol'] == 1000, 'обсяг доходить до графіка')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('_imbBody' in js and 'volTxt' in js, 'IMB: обсяг + % у зоні')
    print('✓ IMB: обсяг 3 свічок і % імпульсної свічки')


def test_personal_sbr_subscriptions():
    cur = ip.user_info({'info': {'wl': ['BTCUSDT', 'ETHUSDT'],
                                 'sbr': {'BTCUSDT': ['1h', '15m'], 'XRPUSDT': ['4h']}}}, [])
    _check(cur['sbr'] == {'BTCUSDT': ['1h']}, 'лише 1H/4H і лише монети списку')
    cur, _ = ip.apply_change(cur, {'sbr': {'symbol': 'eth', 'tfs': ['4h', '1h']}}, [])
    _check(cur['sbr'] == {'BTCUSDT': ['1h'], 'ETHUSDT': ['1h', '4h']}, cur['sbr'])
    cur, _ = ip.apply_change(cur, {'sbr': {'symbol': 'BTCUSDT', 'tfs': []}}, [])
    _check('BTCUSDT' not in cur['sbr'], 'порожній вибір = відписка')
    cur, _ = ip.apply_change(cur, {'remove': 'ETHUSDT'}, [])
    _check(cur['sbr'] == {}, 'прибрали монету → і підписку')
    s1, s4 = {'side': 'LONG'}, {'side': 'LONG'}
    rc = ip.sbr_recipients([('11', {'BTCUSDT': ['1h']}), ('22', {'BTCUSDT': ['4h']}),
                            ('33', {'ETHUSDT': ['1h']})], 'BTCUSDT', [('1h', s1), ('4h', s4)])
    _check(rc == {'11': [('1h', s1)], '22': [('4h', s4)]}, rc)
    print('✓ особисті SBR-підписки: 1H/4H по монеті, отримувачі')


def test_sbr_engine_delivers_personally_and_tracks_extra():
    S = _load('sbr_signals_t', 'detection/sbr_signals.py')
    got = []
    e = S.SbrEngine(db=None, get_watchlist=lambda: ['BTCUSDT'], fetch=lambda *a: None,
                    notify=None, get_extra=lambda: ['ZZZUSDT'],
                    personal=lambda sym, items: (got.append((sym, items)) or (1, 0)))
    _check('ZZZUSDT' in e._universe(), 'монета підписки рахується і поза WATCHLIST бота')
    sig = {'side': 'SHORT', 'score': 70.0, 'trigger': 1.0, 'n': 1, 't': 0, 'close_t': 0}
    e._on_signals('ZZZUSDT', [('1h', sig)], 0)
    _check(got and got[0][0] == 'ZZZUSDT', 'свіжий сигнал → особиста доставка')
    print('✓ рушій SBR: особиста доставка + монети підписок')


def test_blocked_user_has_no_access_and_no_group():
    tg = _load('tg_bot_t', 'web/tg_bot.py')
    _check(tg.group_action(True, False, None) == 'restrict', 'заблокований → лишається, але без прав')
    _check(tg.group_action(True, False, 'restricted') is None, 'вже обмежений — без повторів')
    _check(tg.group_action(True, False, 'banned') == 'restrict', 'старий бан → повертаємо обмеженим')
    _check(tg.group_action(False, True, 'restricted') == 'restore', 'знову активний → права назад')
    _check(tg.group_action(False, False, None) is None, 'очікує схвалення — не чіпаємо')
    _check(all(v is False for v in tg._NO_RIGHTS.values()) and 'can_send_messages' in tg._NO_RIGHTS,
           'обмеження знімає всі права учасника')
    src = open(os.path.join(_HERE, 'web', 'tg_bot.py'), encoding='utf-8').read()
    fb = src.split('def forget_bot_user')[1][:900]
    _check('group_ban(' in fb and 'forget_bot_contact' in fb, 'видалений → бан у групі + слід у боті')
    auth = open(os.path.join(_HERE, 'web', 'auth.py'), encoding='utf-8').read()
    _check("not path.startswith(('/api/me', '/api/info/'))" in auth,
           'API-ключ не відкриває особистий кабінет')
    _check("'blocked' if" in auth and 'sync_user_group(uid)' in auth, 'причина 403 + синхронізація групи')
    _check('forget_bot_user(target.telegram_chat_id)' in auth, 'видалення акаунта кличе повне видалення')
    _check('if not _is_active(u):' in auth.split('def info_sbr_subscribers')[1][:900],
           'особисті сигнали — лише активним')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('DENY_TXT' in js and 'blocked:' in js, 'сайт пояснює закритий доступ')
    print('✓ заблокований: без кабінету, без розсилок, поза групою')


def test_comment_button():
    tg = _load('tg_bot_c', 'web/tg_bot.py')
    link = tg.comment_link('sbr', 123, 'vsv_bot')
    _check(link == 'https://t.me/vsv_bot?start=cm_sbr_123', link)
    _check(len(link.split('start=')[1]) <= 64, 'start-параметр ≤ 64')
    _check(tg.parse_comment_start('cm_sbr_123') == ('sbr', 123), 'розбір')
    _check(tg.parse_comment_start('cm_register_1') is None, 'службові теми не коментуються')
    _check(tg.comment_link('register', 1, 'b') is None, 'без кнопки в службових темах')
    src = open(os.path.join(_HERE, 'web', 'tg_bot.py'), encoding='utf-8').read()
    _check("_add_comment_button(category, chat" in src and "'reply_parameters'" in src,
           'кнопка після відправки + коментар відповіддю під повідомленням')
    _check("send to {category} failed" not in src.split('def tg_send')[1].split('def _send_get_id')[0],
           'tg_send більше не падає NameError')
    print('✓ 💬 кнопка «Коментар» під повідомленнями групи')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
