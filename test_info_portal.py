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
    # 🏁 High/Low бот у памʼяті не тримає — його рахує фоновий потік (не запит сторінки)
    _check(not calls and all(k[0] in ('shl', 'sbrlast') for k in ip._want),
           'монета бота + TF бота — жодного звернення до біржі із запиту сторінки')
    _check(('BTCUSDT', 7, '1h') in ip._poc_want, 'POC поставлено фоновому потоку (деф. 7 днів · 1H)')
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
    """Обсяг — у МОНЕТАХ (обіг / типова ціна) імпульсної свічки, як у TV
    («11.461K», а не «3.64B»); розклад продаж/купівля — за кольором 3 свічок."""
    m = _load('imbalance2', 'detection/imbalance.py')
    T = 1_700_000_000_000
    b = [{'t': T + i * 60000, 'o': 100, 'h': 101, 'l': 99, 'p': 100, 'v': 10} for i in range(20)]
    b += [{'t': T + 20 * 60000, 'o': 101, 'h': 101, 'l': 99, 'p': 100, 'v': 100},      # червона: 1 монета
          {'t': T + 21 * 60000, 'o': 101, 'h': 106, 'l': 100.5, 'p': 105.5, 'v': 832},  # зелена: 8 монет
          {'t': T + 22 * 60000, 'o': 105.5, 'h': 107, 'l': 103, 'p': 106, 'v': 316}]    # зелена: 3 монети
    z = m.detect_imbalances(b)[0]
    _check(abs(z['vol'] - 8.0) < 1e-6, ('обсяг імпульсної свічки в монетах', z))
    _check(z['sell_pct'] == 8 and z['buy_pct'] == 92 and z['pct'] == 92, z)
    nov = [dict(x, v=0) for x in b]
    zz = m.detect_imbalances(nov)[0]
    _check(zz['pct'] is None and zz['sell_pct'] is None and zz['vol'] is None,
           'немає обсягу → None, а не вигадане число')
    r = ip.imb_of(b, '1m', (T + 30 * 60000) / 1000)[0]
    _check(abs(r['vol'] - 8.0) < 1e-6 and r['sell_pct'] == 8 and r['buy_pct'] == 92,
           'обсяг і розклад доходять до графіка')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('_imbBody' in js and 'volTxt' in js and 'topPct: z.sell_pct' in js
           and 'botPct: z.buy_pct' in js, 'IMB: обсяг + дві смуги продаж/купівля')
    _check('IMB_SELL' in js and 'IMB_BUY' in js, 'смуги різних кольорів')
    print('✓ IMB: обсяг у монетах і розклад продаж/купівля')


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


def test_no_group_comments_0811():
    """💬 Коментарі в групі ПРИБРАНО (вимога 08.11)."""
    src = open(os.path.join(_HERE, 'web', 'tg_bot.py'), encoding='utf-8').read()
    for w in ('_add_comment_button', '_comment_begin', '_comment_take',
              'comments_status', 'editMessageReplyMarkup', '_COMMENT_CATS'):
        _check(w not in src, f'залишок коментарів: {w}')
    app = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    html = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
    _check('/api/tg/comments' not in app and 'tgComments' not in html, 'маршрут і кнопки прибрано')
    _check("send to {category} failed" not in src.split('def tg_send')[1].split('def _send_get_id')[0],
           'tg_send більше не падає NameError')
    print('✓ 💬 коментарів у групі немає')


def test_cabinet_polish_0911():
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    html = open(os.path.join(_HERE, 'infosite', 'index.html'), encoding='utf-8').read()
    _check('brand-sub' not in html and '· Мій кабінет' not in html, 'напис «Мій кабінет» прибрано')
    _check('<a class="who"' in js and '>Кабінет</a>' not in js, '👤 імʼя = посилання на кабінет')
    _check('" VOB" + (v.breaker' in js, 'підпис «15 Min VOB»')
    _check('id="notify-head"' in html and 'style="display:none"' in html.split('id="notify-body"')[1][:30]
           and 'ntfApply' in js, '🔔 сповіщення — гармошка, згорнуто')
    card = js.split('function sigCard')[1].split('document.addEventListener')[0]
    _check('row("Бар"' not in card and 'розворот від смуги' not in js and '(сила сигналу)' in card
           and '"%"' in card, 'картка SBR: без «Бар» і пояснень, Score у %')
    _check('function livePx' in js and js.count('livePx(sym)') >= 3, 'ціна списку = ціна шапки')
    _check('dirCls(sym)' in js, 'ціна за напрямком')
    print('✓ кабінет: шапка, VOB, гармошка, картка SBR, одна ціна')


def test_mm_babl_colour_and_tv_like_chart():
    """🧭 Назва монети кольором «МММ-бабло» (та сама формула, що 🧮 МММ LiQ):
    спершу знімок двигуна, далі `_fuel_dir_legacy`; немає даних → без кольору.
    🖱 Графік: інерція миші, вертикальний рух вимикає автомасштаб, кнопка «A»,
    оновлення хвостом (`update`), а не повною заміною даних."""
    class FF:
        calls = []
        def mm_snapshot_for(self, syms):
            return {'BTCUSDT': {'mm': 'SHORT', 'strength': 34}}
        def _fuel_dir_legacy(self, sym):
            self.calls.append(sym)
            return {'status': 'LONG', 'strength': 21} if sym == 'ETHUSDT' else None
    ip._mm_cache.clear()
    ff = FF()
    m = ip.mm_marks(['btcusdt', 'ETHUSDT', 'XRPUSDT'], ff=ff, now=1000.0)
    _check(m['BTCUSDT'] == {'dir': 'SHORT', 'str': 34}, m)
    _check(m['ETHUSDT'] == {'dir': 'LONG', 'str': 21}, m)
    _check('XRPUSDT' not in m, 'немає даних → без кольору (не «рівновага»)')
    _check('BTCUSDT' not in ff.calls, 'монета зі знімка не перераховується')
    ip.mm_marks(['ETHUSDT'], ff=ff, now=1010.0)
    _check(ff.calls.count('ETHUSDT') == 1, 'фолбек кешується')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    css = open(os.path.join(_HERE, 'infosite', 'style.css'), encoding='utf-8').read()
    _check('mmCls(sym)' in js and "' mm-short'" not in js and 'mm-short' in js
           and '.iw-sym.mm-short' in css and '.iw-title.mm-long' in css, 'колір назви в списку й шапці')
    _check(js.count('mmCls(sym)') >= 3, 'і список, і шапка')
    _check('kineticScroll: { mouse: true' in js, 'інерція миші як у TradingView')
    _check('function tvPanInit' in js and 'autoScale: false' in js and 'id = "iw-auto"' in js,
           'вертикальний рух → вільний рух, кнопка A')
    _check('IW.series.update(o[u])' in js, 'оновлення хвостом, без setData посеред руху')
    print('✓ МММ-бабло на назві монети · графік як у TradingView')


def test_swing_hl_btc_colour_and_price_wheel():
    """🏁 High/Low як у боті (та сама `_swing_hl_for_tf`), колір МММ-бабло лише
    для BTC, колесо над шкалою цін — масштаб по висоті."""
    v = ip.shl_view({'high': {'price': 81199.9, 'label': 'Strong High'},
                     'low': {'price': 75143.5, 'label': 'Weak Low'}}, '1h')
    _check(v == {'high': {'price': 81199.9, 'label': 'Strong High'},
                 'low': {'price': 75143.5, 'label': 'Weak Low'}, 'tf': '1H'}, v)
    _check(ip.shl_view(None, '1h') is None and ip.shl_view({'high': {'price': 0}}, '1h') is None,
           'немає рівнів → нічого не малюємо')
    src = open(os.path.join(_HERE, 'web', 'info_portal.py'), encoding='utf-8').read()
    _check("sc._swing_hl_for_tf(sym, tf)" in src, 'джерело — функція бота, своєї копії немає')
    _check("swing_hl_timeframe" in src, 'TF — той самий, що в боті')
    _check(ip.MM_SYMBOLS == ('BTCUSDT',), 'МММ-бабло лише BTC')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('function setShl' in js and 'data-layer="show_shl"' in js, 'лінії High/Low + перемикач шару')
    _check('var MM_SYMS = ["BTCUSDT"]' in js, 'колір назви — лише BTC')
    _check('addEventListener("wheel"' in js and 'autoscaleInfoProvider: priceRangeProvider' in js
           and 'passive: false' in js, 'колесо над шкалою цін масштабує висоту')
    print('✓ High/Low · колір лише BTC · колесо над шкалою цін')


def test_sbr_last_badges_1h_4h():
    """🔔 Останній сигнал SBR 1H/4H у шапці графіка: % від Trigger до ціни."""

    v = ip.sbr_last_view({'side': 'SHORT', 't': 1762560000000, 'n': 2,
                          'trigger': 100.0, 'score': 76.27, 'time_kyiv': '08.11 10:00'}, '4h')
    assert v['tf'] == '4H' and v['trigger'] == 100.0 and v['t'] == 1762560000
    assert ip.sbr_last_view(None, '1h') is None
    m = ip.sbr_move('SHORT', 100.0, 98.0)
    assert m['pct'] == -2.0 and m['fav'] is True
    m = ip.sbr_move('LONG', 100.0, 98.0)
    assert m['fav'] is False
    assert ip.sbr_move('LONG', 0, 98) is None
    src = open(ip.__file__, encoding='utf-8').read()
    assert "'sbr_last': sbr_last" in src and "('sbrlast', sym, t)" in src
    assert 'get_sbr_engine' in src          # монета в рушії → памʼять, без мережі
    js = open('infosite/app.js', encoding='utf-8').read()
    assert 'sbrLastBadge("1h"' in js and 'sbrLastBadge("4h"' in js
    assert 'function sbrMove(' in js


def test_paper_fresh_and_watchlist_height_0811():
    """Видалена paper-угода не висить у кеші; WATCHLIST до низу графіка."""
    import types
    pass
    tm = types.SimpleNamespace(_shadow_closed=[{'symbol': 'A', 'closed_at': 1}],
                               _shadow_positions={})
    s1 = ip.paper_book_sig(tm)
    tm._shadow_closed = []
    assert ip.paper_book_sig(tm) != s1
    src = open(os.path.join(_HERE, 'web', 'info_portal.py'), encoding='utf-8').read()
    assert "sig == _paper_cache['sig']" in src and src.count("'no-store'") >= 2
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    assert 'cache: "no-store"' in js and 'loadPaper(); }, 5000)' in js
    css = open(os.path.join(_HERE, 'infosite', 'style.css'), encoding='utf-8').read()
    assert 'align-items: stretch' in css and '.iw-list { flex: 1 1 0; min-height: 0;' in css


def test_imb_counts_the_live_bar_like_tv():
    """Кейс BTC 1H 08.11: розрив після обвалу видно ВЖЕ на живій свічці."""
    T = 1_700_000_000_000
    H = 3600_000
    b = [{'t': T + i * H, 'o': 100, 'h': 100.5, 'l': 99.5, 'p': 100, 'v': 1000} for i in range(30)]
    b += [{'t': T + 30 * H, 'o': 100, 'h': 100.4, 'l': 99.6, 'p': 99.8, 'v': 1000},
          {'t': T + 31 * H, 'o': 99.8, 'h': 99.9, 'l': 95.0, 'p': 95.5, 'v': 9000},   # обвал
          {'t': T + 32 * H, 'o': 95.5, 'h': 97.0, 'l': 94.8, 'p': 96.0, 'v': 2000}]   # ЖИВА
    now = (T + 32 * H) / 1000 + 600   # жива свічка відкрита 10 хв
    z = ip.imb_of(b, '1h', now)
    _check(z and z[0]['side'] in ('bear', 'bearish', 'SHORT', 'short') or
           (z and z[0]['top'] > z[0]['bottom']), ('IMB на живому барі', z))
    _check(abs(z[0]['top'] - 99.6) < 1e-9 and abs(z[0]['bottom'] - 97.0) < 1e-9, z[0])
    print('✓ IMB видно на живій свічці, як у TV')


def test_imb_slivers_are_hidden():
    """09.11: мізерні смужки IMB (майже заповнені / тонші за 0.25·ATR) не малюються."""
    vis = ip.imb_visible
    _check(vis({'top': 102, 'bottom': 100, 'filled_pct': 0}, 2.0), 'нормальний розрив видно')
    _check(not vis({'top': 102, 'bottom': 100, 'filled_pct': 85}, 2.0), '≥80% заповнено — сховано')
    _check(not vis({'top': 100.3, 'bottom': 100, 'filled_pct': 10}, 2.0), 'залишок < 0.25·ATR — сховано')
    _check(vis({'top': 100.3, 'bottom': 100, 'filled_pct': 10}, None), 'ATR невідомий → лише заповнення')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('b.kind === "imb" && (yBt - yT) < IMB_MIN_PX * vr' in js, 'фронт: тонкі IMB на екрані не малюються')
    print('✓ IMB-смужки прибрано')


def test_tfs_poc_params_ob_declutter_0811():
    """1m/5m прибрано з графіка, 1m — з VOB; POC — свої параметри (деф. 7 днів ·
    1H); OB-бокси без купи перекритих; підписи графіка без накладань."""
    _check('1m' not in ip.CHART_TFS and '5m' not in ip.CHART_TFS and '1m' not in ip.VOB_TFS, 'TF прибрано')
    _check(ip.norm_tf('1m', ip.CHART_TFS, '15m') == '15m', 'збережене 1m → дефолт')
    _check(ip.poc_norm() == (7, '1h') and ip.poc_norm(14, '4h') == (14, '4h')
           and ip.poc_norm(9, '1m') == (7, '1h'), 'POC: дозволені значення, деф. 7 · 1H')
    cur = ip.user_info({}, [], None)
    _check(cur['poc_days'] == 7 and cur['poc_tf'] == '1h', 'новий користувач — 7 днів')
    out, err = ip.apply_change(cur, {'poc_days': 3}, [])
    _check(not err and out['poc_days'] == 3 and out['poc_tf'] == '1h', 'зміна вікна POC')
    z = {'internal': [{'bar_high': 101, 'bar_low': 100, 'bar_time': i * 3600 * 1000, 'bias': 'BULLISH'}
                      for i in range(1, 9)] +
                     [{'bar_high': 120, 'bar_low': 118, 'bar_time': 50 * 3600 * 1000, 'bias': 'BULLISH'}],
         'swing': []}
    bx = ip.ob_boxes(z)
    _check(len(bx) == 2, ('перекриті блоки того самого боку не дублюються', len(bx)))
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('"1m"' not in js.split('var CHART_TFS')[1][:80], 'фронт без 1m')
    _check('iw-poc-days' in js and '&poc_days=' in js, 'вибір POC на графіку')
    _check('sLbl.push' in js and 'text: e.tag, up: bull' in js, 'CHoCH/BOS — підпис на лінії, а не маркер')
    _check('pct <= 0' in js, 'IMB: 0% не малюємо')
    _check('self._lv.forEach' in js, 'чипи обходять підписи POC / High / Low')
    print('✓ TF без 1m/5m · POC 7 днів · OB без купи · підписи без накладань')



def test_sig_card_drag_and_arrow():
    """↔ Картку SBR тягнуть за шапку; стрілка-виноска вказує на свою позначку
    S/B (за тими самими `_hits`, що й клік) і йде за нею при прокрутці графіка."""
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    css = open(os.path.join(_HERE, 'infosite', 'style.css'), encoding='utf-8').read()
    _check('function sigDragInit' in js and 'setPointerCapture' in js, 'картку тягнуть за шапку')
    _check('function sigAnchor' in js and 'IW.prim._hits' in js and 'g.t === sel.t && g.side === sel.side' in js,
           'стрілка шукає СВОЮ позначку (час + бік)')
    _check('function sigArrowLoop' in js and 'requestAnimationFrame' in js, 'стрілка йде за графіком щокадру')
    pan = js.split('function tvPanInit')[1][:900]
    _check('closest("#iw-sig")' in pan, 'перетягування картки не рухає графік')
    _check('.iw-sig-arrow' in css and 'pointer-events: none' in css.split('.iw-sig-arrow {')[1][:200],
           'стрілка не перехоплює кліки')
    print('✓ картка SBR: перетягування + стрілка до позначки')


def _liq_raw(price=100.0, n=200, ex='bybit', sym='ETHUSDT'):
    import math
    kl = []
    for k in range(n):
        p = price * (1 + 0.05 * math.sin(k / 9.0))
        kl.append({'h': p * 1.01, 'l': p * 0.99, 'c': p, 'v': 1000 + (k % 7) * 300})
    return {'ok': True, 'exchange': ex, 'symbol': sym, 'price': price,
            'oi_usd': 5e8, 'bars': n, 'klines': kl, 'took_sec': 0.4,
            'fetched_at': 1000.0}


def _liq_reset():
    ip._liq_cache.clear(); ip._liq_user_at.clear(); ip._liq_calc_at.clear()
    ip._liq_pause.clear()


def test_liq_profile_button_only_and_cached():
    """💧 Профіль: параметри нормалізуються; кеш — СИРІ дані на (монета,
    біржа, глибина), крок застосовується без нового запиту; відмова НЕ
    кешується."""
    _check(ip.liq_params('BYBIT', '720', 'fine') == ('bybit', 720, 'fine'), 'параметри')
    _check(ip.liq_params('mexc', '999', 'x') == ('binance', ip.LIQ_BARS_DEFAULT, ip.LIQ_RES_DEFAULT),
           'невідоме → дефолт')
    calls = []

    def runner(sym, ex, b):
        calls.append((sym, ex, b))
        return _liq_raw(ex=ex, sym=sym)
    _liq_reset()
    v = ip.liq_profile('eth', 'bybit', 168, 'mid', now=1000, runner=runner)
    _check(v['ok'] and v['symbol'] == 'ETHUSDT' and v['rows'], v)
    _check(set(v['rows'][0]) == {'lo', 'hi', 'usd', 'pct', 'dir', 'dist'}, 'форма рядка')
    _check(calls == [('ETHUSDT', 'bybit', 168)], 'крок у запит до біржі НЕ входить')
    _check(v['src'] == 'server' and v['age'] == 0, 'джерело і вік даних')
    v2 = ip.liq_profile('ETHUSDT', 'bybit', 168, 'mid', now=1030, runner=runner, step='0.5')
    _check(len(calls) == 1, 'інший крок — той самий кеш, без нового запиту')
    _check(v2['step'] == 0.5 and v2['age'] == 30, f'крок від графіка застосовано: {v2.get("step")}')
    _check(ip.liq_step('-1') is None and ip.liq_step('abc') is None, 'сміттєвий крок відкинуто')
    ip.liq_profile('ETHUSDT', 'bybit', 168, 'fine', now=1000 + ip.LIQ_TTL - 1, runner=runner)
    _check(len(calls) == 1, 'кеш до LIQ_TTL')
    ip.liq_profile('ETHUSDT', 'binance', 168, 'mid', now=1001, runner=runner)
    _check(len(calls) == 2, 'інша біржа — свій ключ')
    bad = ip.liq_profile('SOLUSDT', now=1, runner=lambda *a: {'ok': False, 'reason': 'немає OI'})
    _check(not bad['ok'] and bad['reason'] == 'немає OI', 'причина відмови')
    _check(not any(k[0] == 'SOLUSDT' for k in ip._liq_cache), 'відмова не кешується')
    print('✓ 💧 профіль: сирий кеш, крок без запиту, причина')


def test_liq_cache_gives_same_numbers_as_direct_scan():
    """Кеш сирих даних НЕ змінює чисел: драбина з кешу == прямий scan_one."""
    from detection import liq_scan
    _liq_reset()
    raw = _liq_raw()
    for step in (None, '0.5', '2'):
        v = ip.liq_profile('ETHUSDT', 'bybit', 168, 'mid', now=1000, runner=lambda *a: raw,
                           step=step)
        direct = liq_scan.summarise_raw(raw, rows=ip.LIQ_ROWS, step_pct=ip.LIQ_RES['mid'],
                                        step_usd=ip.liq_step(step), min_row_pct=ip.LIQ_MIN_ROW_PCT)
        ref = ip.liq_view(direct, 'ETHUSDT', 'bybit', 168, 'mid')
        for k in ('rows', 'magnet', 'above_pct', 'below_pct', 'pull', 'step', 'total_usd', 'verdict'):
            _check(v[k] == ref[k], f'{k} (крок {step})')
    print('✓ 💧 кеш сирих даних дає ті самі числа, що прямий розрахунок')


def test_liq_user_cooldown_and_exchange_pause():
    _liq_reset()
    calls = []

    def runner(sym, ex, b):
        calls.append(sym)
        return _liq_raw(sym=sym)
    v = ip.liq_profile('ETHUSDT', 'bybit', 168, now=1000, runner=runner, user=7)
    _check(v['ok'], v)
    w = ip.liq_profile('SOLUSDT', 'bybit', 168, now=1010, runner=runner, user=7)
    _check(not w['ok'] and w.get('retry_in') and len(calls) == 1, f'ліміт користувача: {w}')
    c = ip.liq_profile('ETHUSDT', 'bybit', 168, now=1010, runner=runner, user=7, step='1')
    _check(c['ok'] and len(calls) == 1, 'кеш — без ліміту')
    o = ip.liq_profile('SOLUSDT', 'bybit', 168, now=1010, runner=runner, user=8)
    _check(o['ok'] and len(calls) == 2, 'інший користувач — свій ліміт')
    ok = ip.liq_profile('SOLUSDT', 'bybit', 168, now=1000 + ip.LIQ_USER_COOLDOWN + 1,
                        runner=runner, user=7)
    _check(ok['ok'], 'після паузи — знову можна')
    # Біржа відповіла лімітом → пауза всім запитам кабінету до неї.
    _liq_reset()
    b = ip.liq_profile('ADAUSDT', 'binance', 168, now=2000,
                       runner=lambda *a: {'ok': False, 'blocked': 429, 'reason': 'x'}, user=1)
    _check(not b['ok'] and b.get('paused'), b)
    n = []
    p = ip.liq_profile('XRPUSDT', 'binance', 168, now=2100, user=2,
                       runner=lambda *a: n.append(1) or _liq_raw())
    _check(not p['ok'] and p.get('paused') and not n, 'під час паузи біржу не питаємо')
    q = ip.liq_profile('XRPUSDT', 'bybit', 168, now=2100, user=3, runner=lambda *a: _liq_raw())
    _check(q['ok'], 'пауза — лише для своєї біржі')
    r = ip.liq_profile('XRPUSDT', 'binance', 168, now=2000 + ip.LIQ_PAUSE_SEC + 1, user=4,
                       runner=lambda *a: _liq_raw())
    _check(r['ok'], 'після паузи — знову')
    print('✓ 💧 ліміт на користувача + пауза при блокуванні біржею')


def test_liq_from_browser_data_matches_server():
    """Дані, які взяв браузер, рахуються ТІЄЮ САМОЮ формулою; сміття відкинуто;
    чужого кешу не наповнюють."""
    _liq_reset()
    raw = _liq_raw()
    srv = ip.liq_profile('ETHUSDT', 'bybit', 168, 'mid', now=1000, runner=lambda *a: raw, step='0.5')
    body = {'symbol': 'eth', 'exchange': 'bybit', 'bars': 168, 'res': 'mid', 'step': 0.5,
            'price': raw['price'], 'oi_usd': raw['oi_usd'],
            'klines': [[k['h'], k['l'], k['c'], k['v']] for k in raw['klines']]}
    _liq_reset()
    br = ip.liq_from_client(body, user=5, now=1000)
    _check(br['ok'] and br['src'] == 'browser', br)
    for k in ('rows', 'magnet', 'above_pct', 'pull', 'step', 'total_usd'):
        _check(br[k] == srv[k], f'браузерний шлях == серверний: {k}')
    _check(not ip._liq_cache, 'дані від клієнта не йдуть у спільний кеш')
    _check(not ip.liq_from_client(body, user=5, now=1000.5)['ok'], 'ліміт частоти розрахунку')
    for bad in ({'price': 0}, {'oi_usd': 'x'}, {'klines': [[1, 1, 1, 1]] * 5},
                {'klines': [[1, 1, 1, 1]] * (ip.LIQ_CLIENT_MAX_BARS + 1)}, {'symbol': '!!'}):
        b2 = dict(body); b2.update(bad)
        _check(not ip.liq_from_client(b2, now=5000)['ok'], f'відкинуто: {list(bad)}')
    print('✓ 💧 дані з браузера: та сама формула, перевірка входу')


def test_exchange_block_is_detected():
    from detection import liq_scan

    class R:
        def __init__(self, st, j=None):
            self.status_code, self._j, self.text = st, j or {}, ''

        def json(self):
            return self._j
    for ex, r in (('binance', R(429)), ('binance', R(418)), ('bybit', R(403)),
                  ('bybit', R(200, {'retCode': 10006}))):
        try:
            liq_scan._guard(ex, r)
            raise AssertionError(f'не впізнано блок {ex}')
        except liq_scan.ExchangeBlocked:
            pass
    liq_scan._guard('binance', R(451))     # регіон ≠ ліміт
    liq_scan._guard('bybit', R(200, {'retCode': 0}))
    print('✓ 💧 ліміт / блок IP біржі впізнається, регіон — ні')


def test_liq_step_pct_is_finer():
    """Крок сходинок профілю — частка ціни, округлена до 1/2/5 × 10^k."""
    spec = importlib.util.spec_from_file_location('ladder_t', os.path.join(_HERE, 'detection', 'liquidation_map', 'ladder.py'))
    ld = importlib.util.module_from_spec(spec); spec.loader.exec_module(ld)
    _check(ld.step_for(100000) == 2000, 'дефолт банера не змінився')
    _check(ld.step_for(100000, pct=0.005) == 500, '0.5% → 500')
    _check(ld.step_for(4000, pct=0.0025) == 10, '0.25% → 10')
    print('✓ 💧 крок профілю')


def test_liq_frontend_contract():
    """Кнопка вмикає лише за натисканням; зміна монети прибирає профіль і гасить
    кнопку; зони й чипи закінчуються перед смугою профілю."""
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    html = open(os.path.join(_HERE, 'infosite', 'index.html'), encoding='utf-8').read()
    for i in ('iw-liq-btn', 'iw-liq-ex', 'iw-liq-bars', 'iw-liq-res'):
        _check('id="' + i + '"' in html, i)
    _check('value="binance"' in html and 'value="bybit"' in html, 'дві біржі')
    _check('/api/info/liq?' in js and js.count('/api/info/liq?') == 1, 'один виклик — у liqFromServer')
    ld = js.split('function liqLoad()')[1][:1500]
    _check(ld.index('liqFromBrowser(') < ld.index('liqFromServer('), 'спершу браузер, фолбек — сервер')
    _check('/api/me/liq-calc' in js and 'fapi.binance.com/fapi/v1/' in js and '"klines?symbol="' in js
           and 'api.bybit.com/v5/market/' in js and '"kline?category=linear' in js, 'браузер тягне ті самі ендпоінти')
    _check('x[7]' in js.split('binance: function')[1][:900] and 'x[6]' in js.split('bybit: function')[1][:1200],
           'обсяг у доларах — як на сервері')
    lc = js.split('function loadCoin')[1][:1500]
    _check('LQ.sym !== sym) liqOff()' in lc, 'інша монета → профіль прибрано')
    _check('LQ.sym !== IW.sel) return' in js, 'запізніла відповідь для іншої монети ігнорується')
    _check('X2 = XR -' in js and '(XR - w - (LW ? 10 : 8) * hr)' in js, 'зони й VOB-чипи лівіше профілю')
    _check('rightOffset' in js.split('function liqOffset')[1][:600], 'свічки не під профілем')
    _check('"&step=" + st' in js and 'function liqStepFromChart' in js, 'крок — з видимого діапазону графіка')
    ch = js.split('if (id !== "iw-liq-ex"')[1][:700]
    _check('liqLoad' not in ch, 'зміна параметра біржу НЕ питає — лише кнопка')
    _check(js.count('lineVisible: !_lvTitleOff()') == 2, 'лінії High/Low/POC не заходять у профіль')
    _check("'magnet': mag" in open(os.path.join(_HERE, 'web', 'info_portal.py'), encoding='utf-8').read(),
           'магніт віддається сирими числами')
    i0 = html.index('id="iw-tf-wait"')
    _check(html.index('id="iw-chart"') < i0 < html.index('id="iw-tfbar"'), 'статус — поза панеллю кнопок')
    _check('class="iw-status"' in html, 'загальний статус — у куті блоку графіка')
    print('✓ 💧 фронт: кнопка, зміна монети, без накладань')


def test_vob_drawn_like_imb_without_changing_the_math():
    """📊 VOB «як IMB» (вимога 10.11): дві смуги-частки + обсяг; самі числа VOB
    (обсяг, %, межі) рахуються як раніше — додано лише розклад для показу."""
    import importlib.util, types
    sc = types.SimpleNamespace()
    src = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    i = src.index('    def _format_vob(self, lob, vtf):')
    j = src.index('\n    def ', i + 10)
    ns = {}
    exec('import textwrap\n' + 'def _f' + src[i + len('    def _format_vob'):j].replace('\n    ', '\n'), ns)
    z = ns['_f'](sc, {'type': 'Bull', 'top': 110.0, 'bottom': 90.0, 'start_time': 1000,
                       'ob_volume': 3000.0, 'ob_low_volume': 1000.0, 'ob_high_volume': 3000.0}, '15m')
    _check(z['volume'] == 30 and z['pct'] == 33, ('обсяг і % — як раніше', z))
    _check(z['hi_pct'] == 75 and z['lo_pct'] == 25, ('розклад половин', z))
    vb = ip.vob_boxes([z])[0]
    _check(vb['hi_pct'] == 75 and vb['lo_pct'] == 25, 'розклад доходить до графіка')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('topPct: v.hi_pct' in js and 'botPct: v.lo_pct' in js and 'volTxt: fmtVol(v.volume, 3)' in js,
           'VOB малюється смугами + обсягом')
    _check('b.kind === "imb" || b.kind === "vob"' in js, 'той самий рендер, що IMB')
    sm = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
    _check('function _vobBody' in sm and 'if (ob.vobBars) _vobBody(' in sm
           and 'topPct: vob.hi_pct' in sm and 'botPct: vob.lo_pct' in sm,
           'графік бота малює VOB так само')
    print('✓ 📊 VOB як IMB (бот + інфосайт), математика без змін')


def test_ob_vob_toggles_and_no_label_pileup():
    """10.11: OB/VOB — кнопки увімкнення; водяні знаки не налазять; VOB прозоріший."""
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    html = open(os.path.join(_HERE, 'infosite', 'index.html'), encoding='utf-8').read()
    sm = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
    _check('data-tgl="ob_tf"' in html and 'data-tgl="vob_tf"' in html, 'кнопки OB/VOB у панелі')
    _check('list.concat(["off"])' not in js and 'tfLast(' in js, 'випадайка — лише TF, «вимк» — кнопкою')
    _check('function wmDraw' in js and 'wmTaken.forEach' in js, 'водяні знаки без накладань, CHoCH їх обходить')
    _check('vobBodies' in js, 'чип OB шукає місце поза тілом VOB')
    _check('Math.max(v.hi_pct, v.lo_pct)' in js and 'Math.max(vob.hi_pct, vob.lo_pct)' in sm,
           'у чипі VOB — частка переможної половини')
    _check('rgba(34,171,94,.07)' in js and "rgba(22,163,74,0.07)" in sm, 'VOB прозоріший (бот + інфосайт)')
    # розрахунок не змінено: перемикач лише міняє ob_tf/vob_tf ('off' уже підтримувався)
    cur, _ = ip.apply_change(ip.user_info({}, []), {'vob_tf': 'off'}, [])
    _check(cur['vob_tf'] == 'off', 'вимкнення VOB зберігається')


def test_stale_vob_hidden_and_symbol_on_price_label():
    """10.11: VOB, що вже не відповідає ситуації, не малюється; назва монети —
    на мітці поточної ціни (як у TradingView)."""
    def bars(closes):
        return [{'time': 100 + i, 'close': c} for i, c in enumerate(closes)]
    L = {'side': 'LONG', 'top': 110.0, 'bottom': 100.0, 'start': 100, 'breaker': False}
    S = dict(L, side='SHORT')
    _check(ip.vob_stale(L, bars([115, 120, 118])) == '', 'LONG тримає — показуємо')
    _check(ip.vob_stale(dict(L, breaker=True), bars([115])) == 'breaker', 'breaker — ховаємо')
    _check(ip.vob_stale(L, bars([115, 95])) == 'below', 'LONG пробито донизу — ховаємо')
    _check(ip.vob_stale(S, bars([95, 115])) == 'above', 'SHORT пробито вгору — ховаємо')
    _check(ip.vob_stale(L, bars([115, 95, 116, 104])) == 'chop', 'ціна пиляє зону — ховаємо')
    _check(ip.vob_stale(L, bars([115, 104, 112, 108])) == '', 'лише заходи в зону — ще зона')
    _check(ip.vob_stale(L, [{'time': 50, 'close': 90}]) == '', 'бари ДО появи не важать')
    _check(len(ip.vob_fresh([L, dict(L, breaker=True)], bars([115]))) == 1, 'vob_fresh фільтрує')
    src = open(os.path.join(_HERE, 'web', 'info_portal.py'), encoding='utf-8').read()
    _check("vob_fresh(vob_boxes(" in src, 'фільтр стоїть на шляху графіка')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('applyOptions({ title: ttl })' in js, 'назва монети на мітці ціни')



def test_one_clock_everywhere():
    """🕐 10.11: графік, перехрестя і SBR — годинником ПК (бот + кабінет)."""
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    sm = open(os.path.join(_HERE, 'templates', 'smart_money.html'), encoding='utf-8').read()
    _check('tickMarkFormatter: locTick' in js and 'timeFormatter: locFull' in js, 'кабінет: шкала й перехрестя')
    _check('time_kyiv' not in js and 'Europe/Kyiv' not in js, 'кабінет не показує серверний Київ')
    _check('tickMarkFormatter: _pcTick' in sm and 'timeFormatter: _pcFull' in sm, 'бот: шкала й перехрестя')
    _check('time_kyiv' not in sm and 'Europe/Kyiv' not in sm and 'getUTCHours' not in sm, 'бот: без UTC/Київ')
    _check('_pcFull(p.appeared)' in sm, '🆕 Новий OB у лозі — годинником ПК')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
