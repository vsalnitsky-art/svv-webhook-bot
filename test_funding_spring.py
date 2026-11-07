"""💰🌀 «Funding — Пружина» (вимога 06.11).

Дослівно: «Перероби повністю Funding — МММ по монетах, видали попередні
налаштування, таблицю організуй по алгоритму "Пружина" всі монети які будуть
потрапляти із 💰 Funding Rate Scanner опрацьовуй і відображай згідно алгоритму
"Пружина".» Рішення користувача: видалити УСЕ (шари, скальп, VOB→Черга-4,
авто-відкриття, усі TG блоку); монети — УСІ з 💰 Funding Rate Scanner.
"""
import importlib.util
import os
import random
import re
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'funding_spring_t', os.path.join(_ROOT, 'detection', 'funding_spring.py'))
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)
_FF = open(os.path.join(_ROOT, 'detection', 'fuel_filter.py'), encoding='utf-8').read()
_APP = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()

H = 3600
NOW0 = (1_800_000_000 // H) * H + 30     # 30с після закриття бару


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _bars(n, live_open, seed=1):
    rnd = random.Random(seed)
    p, out = 100.0, []
    for i in range(n):
        t = live_open - (n - 1 - i) * H
        p *= 1 + rnd.uniform(-0.01, 0.01)
        out.append({'t': t * 1000, 'o': p, 'h': p * 1.004, 'l': p * 0.996, 'p': p, 'v': 1000})
    return out


class _Net:
    def __init__(self, now, lag=0):
        self.now, self.lag, self.calls = now, lag, []

    def __call__(self, sym, tf, limit):
        self.calls.append((sym, tf, limit))
        # біржа «запізнюється» на lag барів із віддачею закритого бару
        return _bars(limit, (int(self.now // H) - self.lag) * H)


class _DB:
    def __init__(self):
        self.d = {}

    def get_setting(self, k, default=None):
        return self.d.get(k, default)

    def set_setting(self, k, v):
        self.d[k] = v


def _eng(syms, net, wl=None, info=None, db=None):
    return fs.FundingSpring(db, lambda: list(syms), net, lambda s: dict(wl or {}),
                            lambda: dict(info or {}))


# ═══════════ 1. РУШІЙ ═══════════════════════════════════════════════════
def test_only_closed_bars_and_one_request_per_bar():
    net = _Net(NOW0)
    e = _eng(['AAAUSDT'], net)
    e.tick(NOW0)
    _check(len(net.calls) == 1 and net.calls[0][1] == '1h', net.calls)
    _check(net.calls[0][2] >= 220, 'глибина ≥ MIN_BARS «Пружини»')
    rec = e._recs['AAAUSDT']
    _check(rec['bar_t'] == (NOW0 // H) * H - H, 'останній ЗАКРИТИЙ бар (живий відкинуто)')
    net.calls.clear()
    e.tick(NOW0 + 600)
    _check(net.calls == [], 'у межах того самого бару — жодного запиту')
    net.now = NOW0 + H
    e.tick(NOW0 + H)
    _check(len(net.calls) == 1, 'новий закритий бар → один запит')
    print('✓ лише закриті бари, ОДИН запит на монету на закритий бар')


def test_waits_when_exchange_has_not_closed_the_bar():
    net = _Net(NOW0, lag=2)                     # біржа ще не віддала щойно закритий бар
    e = _eng(['AAAUSDT'], net)
    e.tick(NOW0)
    _check('AAAUSDT' not in e._recs, 'старий бар не видаємо за свіжий')
    _check(e._fail['AAAUSDT'][1] == NOW0 + fs.RETRY_SEC, 'повтор через RETRY_SEC')
    net.lag = 0
    e.tick(NOW0 + fs.RETRY_SEC)
    _check('AAAUSDT' in e._recs, 'бар закрився — пораховано')
    print('✓ біржа ще не закрила бар → чекаємо й повторюємо')


def test_watchlist_coin_takes_the_scanner_record_without_requests():
    net = _Net(NOW0)
    wl = {'WLUSDT': {'ok': True, 'charge': 70, 'state': 'charged', 'bar_t': 1,
                     'pot_long': 50, 'pot_short': 20}}
    e = _eng(['WLUSDT', 'AAAUSDT'], net, wl=wl)
    e.tick(NOW0)
    _check([c[0] for c in net.calls] == ['AAAUSDT'], f'WATCHLIST без запиту: {net.calls}')
    st = e.state(NOW0)
    row = next(r for r in st['rows'] if r['symbol'] == 'WLUSDT')
    _check(row['src'] == 'watchlist' and row['charge'] == 70, 'той самий запис, що в 🌀 Пружині')
    print('✓ монета WATCHLIST — готовий запис сканера (одне число в обох таблицях)')


def test_rows_carry_funding_and_follow_the_scanner_list():
    net = _Net(NOW0)
    syms = ['AAAUSDT', 'BBBUSDT']
    e = _eng(syms, net, info={'AAAUSDT': {'rate': -1.25, 'next_ms': 1, 'f_trend': -1}})
    e.tick(NOW0)   # трекер другого рядка наповнює тік
    st = e.state(NOW0)
    a = next(r for r in st['rows'] if r['symbol'] == 'AAAUSDT')
    _check(a['funding']['rate'] == -1.25 and a['funding']['f_trend'] == -1, a['funding'])
    _check(st['coins'] == 2 and st['computed'] == 2, st)
    syms.remove('BBBUSDT')                       # монета вийшла зі 💰 сканера
    e.tick(NOW0 + 60)
    _check('BBBUSDT' not in e._recs and len(e.state(NOW0 + 60)['rows']) == 1,
           'монета поза сканером зникає з таблиці')
    print('✓ рядок несе ставку funding; таблиця = список 💰 сканера')


def test_flat_has_no_direction_and_short_history_is_named():
    net = _Net(NOW0)
    e = _eng(['AAAUSDT'], net)
    e.tick(NOW0)
    r = e.state(NOW0)['rows'][0]
    if r.get('state') == 'flat':
        _check(r['dir'] is None, 'у флеті напрямок не визначаємо (те саме правило)')
    short = fs.FundingSpring(None, lambda: ['NEWUSDT'],
                             lambda s, tf, n: _bars(50, (NOW0 // H) * H), None, None)
    short.tick(NOW0)
    row = short.state(NOW0)['rows'][0]
    _check(row.get('pending') and 'замало барів' in row.get('reason', ''), row)
    print('✓ флет без напрямку; нова монета — причина «замало барів», а не порожнеча')


def test_failure_backs_off_and_is_counted():
    def boom(*a):
        raise RuntimeError('418')
    e = fs.FundingSpring(None, lambda: ['AAAUSDT'], boom, None, None)
    e.tick(NOW0)
    _check(e._stats['errors'] == 1 and '418' in e._stats['last_err'], e._stats)
    _check(e._fail['AAAUSDT'][1] > NOW0 + fs.RETRY_SEC, 'експоненційний backoff')
    print('✓ збій біржі — backoff і лічильник у рядку стану')


def test_records_survive_a_restart():
    db = _DB()
    net = _Net(NOW0)
    e = _eng(['AAAUSDT'], net, db=db)
    e.tick(NOW0)
    _check(fs.DB_KEY in db.d and 'AAAUSDT' in db.d[fs.DB_KEY]['recs'], 'персист')
    e2 = _eng(['AAAUSDT'], _Net(NOW0), db=db)
    _check('AAAUSDT' in e2._recs, 'після рестарту таблиця не порожня')
    _check('AAAUSDT' not in e2._restored, 'бар ще відкритий — запис точний')
    print('✓ записи переживають рестарт')


def test_disable_stops_work_and_clears():
    db = _DB()
    net = _Net(NOW0)
    e = _eng(['AAAUSDT'], net, db=db)
    e.tick(NOW0)
    e.update_settings({'enabled': False})
    net.calls.clear()
    e.tick(NOW0 + H)
    _check(net.calls == [] and not e._recs, 'вимкнено — біржу не чіпаємо, таблиця порожня')
    _check(db.d[fs.DB_SETTINGS]['enabled'] is False, 'тумблер зберігається')
    print('✓ тумблер вимикає розрахунок повністю')


def test_engine_is_display_only():
    src = open(os.path.join(_ROOT, 'detection', 'funding_spring.py'), encoding='utf-8').read()
    for bad in ('on_signal', 'intercept', '_open(', 'notify_category', 'manual_open'):
        _check(bad not in src, f'рушій лише показує: знайдено «{bad}»')
    print('✓ рушій нічого не відкриває і не шле в Telegram')


# ═══════════ 2. СТАРИЙ БЛОК ВИДАЛЕНО ПОВНІСТЮ ═════════════════════════════
def test_old_funding_block_is_gone_from_the_engine():
    for name in ('self._anomalies', '_funding_layers', '_layer_signal_alert', '_vob_open_or_trail',
                 '_scalp_setup_alert', '_refresh_setup_scalp_cache', '_send_gold_alert',
                 '_opportunity_alert', '_opportunity_for', '_scan_funding', '_notify_funding',
                 '_funding_alert', 'mute_funding', 'get_funding_history', '_signal_open',
                 '_price_spike', '_opp_episode', '_funding_vob', 'clear_anomalies'):
        _check(name not in _FF, f'у fuel_filter лишилось «{name}»')
    run = _FF[_FF.index('def _run_alerts'):]
    run = run[:run.index('\n    def ', 10)]
    _check('_btc_start_alert' in run and 'opportunity' not in run.lower(),
           'цикл алертів — лише ₿ START/STOP')
    print('✓ шари / скальп / VOB→Черга-4 / авто-відкриття / TG / архів — видалено')


def test_dead_settings_are_purged_from_the_blob():
    i = _FF.index('PURGED_SETTING_PREFIXES')
    blk = _FF[i:i + 900]
    for k in ("'funding_'", "'scalp_tg_'", "'layer_tg_'", "'opportunity_'", "'ff_tg_'",
              "'spike_tg'", "'spike_auto_open'", "'queue3_vob_open'"):
        _check(k in blk, f'міграція не прибирає {k}')
    a = _FF.index('DEFAULT_SETTINGS')
    dflt = _FF[a:_FF.index('\n}\n', a)]
    _check(not re.search(r"^    '(funding_|scalp_tg_|layer_tg_|opportunity_|ff_tg_)", dflt, re.M),
           'у DEFAULT_SETTINGS лишився ключ старого блоку')
    _check("'queue3_vob_sl_buffer_pct'" in dflt, 'буфер SL Черги-4 НЕ мав зникнути')
    _check("'fund'" not in _FF[_FF.index('HEAVY_SECTIONS'):_FF.index('HEAVY_SECTIONS') + 120],
           'секції fund у get_state більше немає')
    print('✓ ключі старого блоку прибираються з БД; буфер SL Черги-4 на місці')


def test_routes_old_out_new_in():
    for bad in ('/api/fuel-filter/funding-archive', '/api/fuel-filter/funding-mute',
                '/api/fuel-filter/anomaly/', '/api/fuel-filter/funding-history'):
        _check(bad not in _APP, f'маршрут лишився: {bad}')
    for good in ("'/api/funding-spring/state'", "'/api/funding-spring/settings'",
                 'init_funding_spring'):
        _check(good in _APP, good)
    print('✓ маршрути: старі прибрано, нові додано, рушій стартує з застосунком')


# ═══════════ 3. СТОРІНКА ═════════════════════════════════════════════════
def test_page_has_only_the_new_table():
    for bad in ('ff-anomaly', 'ff-funding-', 'showFundingMuted', 'showFundingArchive',
                'clearAnomalies', 'ffLayersCell', 'ff-layer-tg', 'ff-scalp-tg', 'ff-gold-',
                'ff-queue3-vob-open', 'oppStatsRow', "want.push('fund')"):
        _check(bad not in _HTML, f'на сторінці лишилось «{bad}»')
    for good in ('id="fspr-table"', 'function fsprLoad(', "fetch('/api/funding-spring/state')",
                 'id="fspr-enabled"', 'id="ff-queue3-vob-buffer"'):
        _check(good in _HTML, good)
    th = _HTML[_HTML.index('id="fspr-table"'):]
    th = th[:th.index('</thead>')]
    _check(len(re.findall(r'<th[ >]', th)) == 12, '12 колонок (ставка — у другому рядку)')
    _check('colspan="12" class="tm-empty-msg">Завантаження' in _HTML, 'colspan 12')
    _check('fsprSort(\'funding\')' not in _HTML, 'окремої колонки 💰 Funding немає — ставка в другому рядку')
    _check(_HTML.count('_sprRowHTML(') >= 3, 'обидві таблиці малює ОДИН рендер рядка')
    print('✓ сторінка: лише нова таблиця, спільний рендер рядка з 🌀 Пружиною')


def _run_funding_js(row_js):
    a = _HTML.index("let _sprErr = ''")
    b = _HTML.index('\n}\n', _HTML.index('function _sprRowHTML(')) + 3
    parts = [_HTML[a:b]]
    for fn in ('function fTrendBadge(', 'function fmtVolShort(', 'function _fsprSubRow('):
        i = _HTML.index(fn)
        parts.append(_HTML[i:_HTML.index('\n}\n', i) + 3])
    js = ''.join(parts) + "\nfunction fmtFundLeft(){return '2г 10хв';}\n" + row_js
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        res = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    _check(res.returncode == 0, res.stderr[-400:])
    return res.stdout


def test_js_spring_row_plus_funding_second_row():
    """Вимога 07.11: другий рядок повернуто (крім «Дій»)."""
    row = ("{symbol:'AAAUSDT',state:'charged',label:'🌀 заряджена',charge:72,dir:'SHORT',"
           "pot_long:20,pot_short:52,lean:'SHORT',prob_long:28,tension_long:20,tension_short:52,"
           "funding:{rate:-1.2345,prev_rate:-1.10,f_trend:-1,next_ms:1900000000000,"
           "vol24h:12500000,vol24h_prev:10000000,vol24h_min:9000000,vol24h_max:13000000,"
           "price:{dir:'down',chg:-0.42,dir_overall:'up',chg_overall:1.3}}}")
    out = _run_funding_js(f"const r={row};console.log(_sprRowHTML(r,101,'off','','') + '@@SPLIT@@' + _fsprSubRow(r,12,''));")
    main, sub = out.split('@@SPLIT@@', 1)
    tds = main.split('<td')[1:]
    _check(len(tds) == 12, f'12 комірок, а не {len(tds)}')
    _check('заряджена' in tds[1] and '72' in tds[2], 'стан і заряд «Пружини»')
    _check('colspan="12"' in sub, 'другий рядок на всю ширину')
    _check('-1.234%' in sub and '#f87171' in sub, 'ставка глибшає → червона')
    _check('F🔻' in sub and 'ff-fundnext' in sub and '2г 10хв' in sub, 'F-Trend + відлік до виплати')
    _check('Vol 24h' in sub and '▲' in sub and '$9.0M–$13.0M' in sub, 'обсяг: стрілка + діапазон')
    _check('Ціна:' in sub and '-0.42%' in sub and 'Загалом:' in sub and '+1.30%' in sub, 'ціна 15хв + ~2год')
    _check('Приглушити' not in sub and 'Прибрати' not in sub, '«Дії» не повертали')
    print('✓ JS: рядок «Пружини» + другий рядок 💰 Funding (без «Дій»)')


def test_second_row_tracker():
    info = {'AAAUSDT': {'rate': -1.0, 'vol24h': 100.0, 'next_ms': (NOW0 + 3600) * 1000,
                        'price': {'dir': 'up', 'chg': 0.5}}}
    e = _eng(['AAAUSDT'], _Net(NOW0), info=info)
    e._track_funding(info, ['AAAUSDT'], NOW0)
    info['AAAUSDT'].update(rate=-1.2, vol24h=130.0)
    e._track_funding(info, ['AAAUSDT'], NOW0 + 10)
    v = e.funding_view('AAAUSDT', NOW0 + 10)
    _check(v['prev_rate'] == -1.0 and v['rate'] == -1.2, 'попередня ставка тримається ~30с')
    e._track_funding(info, ['AAAUSDT'], NOW0 + 40)
    _check(e.funding_view('AAAUSDT', NOW0 + 40)['prev_rate'] == -1.2, 'через 30с база зсувається')
    _check(v['vol24h_prev'] == 100.0 and v['vol24h_min'] == 100.0 and v['vol24h_max'] == 130.0, v)
    e._track_funding(info, ['AAAUSDT'], NOW0 + 200)
    _check(e.funding_view('AAAUSDT', NOW0 + 200)['vol24h_prev'] == 100.0, 'база обсягу ~2 хв')
    _check(v['price']['dir'] == 'up' and v['next_ms'], 'ціна і час виплати')
    _check(e.funding_view('AAAUSDT', NOW0 + 7200)['next_ms'] is None, 'минулий час виплати не показуємо')
    e._track_funding(info, [], NOW0 + 300)
    _check('AAAUSDT' not in e._fund, 'монета вийшла зі сканера — трекер забуто')
    print('✓ трекер другого рядка: тренд ставки, база обсягу, мін/макс, ціна')


if __name__ == '__main__':
    fns = [v for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
    print(f'{len(fns)}/{len(fns)} passed')
