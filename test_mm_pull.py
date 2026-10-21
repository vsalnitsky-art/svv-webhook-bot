"""🧭 МММ-Pull — п'ятий банер/монітор на скані ліквідності «куди тягне» (вимога 20.10).

Дослівно: «Додай ще один МММ монітор і банер як і попередні, візьми за основу
дані [📡 Tickr · 💧 скан: Binance · Watchlist бота · 720 год · 🧭 куди тягне
(звіт)]. Додай ці налаштування і вистав як на скріні за замовчуванням.» +
«додай до налаштувань [Через … · за банером …]».
"""
import os
import re
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)

_m = _mon._m
_HTML = _mon._HTML
_SRC = _mon._SRC
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
_LQ_SRC = open(os.path.join(_HERE, 'detection', 'liq_scan.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _row(sym, above, price=100.0, ok=True):
    return {'symbol': sym, 'ok': ok, 'price': price,
            'above_pct': above, 'below_pct': (None if above is None else 100 - above),
            'magnet_price': '$95.00–96.00', 'magnet_pct': '14.2%',
            'magnet_dist': '↓4.5%', 'magnet_dir': 'down', 'exchange': 'binance',
            'magnet_row': {'price': 95.0, 'price_hi': 96.0, 'pct': 14.2, 'dist_pct': 4.5},
            'fallback': False}


# ═══════════ 1. ПРАВИЛО НАПРЯМКУ (чиста функція) ═════════════════════════
def test_mass_above_pulls_up_and_is_long():
    c = _m.pull_mm_dir(_row('X', 62.3))
    _check(c['status'] == 'LONG' and c['strength'] == 25, c)
    c2 = _m.pull_mm_dir(_row('X', 30.0))
    _check(c2['status'] == 'SHORT' and c2['strength'] == 40, c2)
    print('✓ маса ВИЩЕ 62.3% → LONG 25% · НИЖЧЕ 70% → SHORT 40%')


def test_balance_uses_the_same_ten_points_as_the_report():
    _check(_m.pull_mm_dir(_row('X', 54.9))['status'] is None, 'перевага 9.8 п.п. → ⚖')
    _check(_m.pull_mm_dir(_row('X', 55.2))['status'] == 'LONG', '10.4 п.п. → напрямок')
    print('✓ ≤ 10 п.п. → ⚖ рівновага (той самий поріг, що у звіті)')


def test_no_data_is_unknown_not_flat():
    _check(_m.pull_mm_dir(_row('X', 60, ok=False)) is None, 'ok False')
    _check(_m.pull_mm_dir(_row('X', None)) is None, 'немає часток')
    _check(_m.pull_mm_dir(None) is None, 'None')
    print('✓ немає даних → None («невідомо» ≠ «рівновага»)')


# ═══════════ 2. ДЕФОЛТИ (як на скріні) ══════════════════════════════════
def test_defaults_match_the_screenshot():
    d = _m.DEFAULT_SETTINGS
    _check(d['mmp_exchange'] == 'binance' and d['mmp_universe'] == 'watchlist'
           and d['mmp_bars'] == 720 and d['mmp_sort'] == 'pull_report', d)
    _check(d['mm_pull_enabled'] is False, 'блок ходить на біржу — дефолт ВИМК')
    _check(d['mmp_bias_tg'] is True and d['mmp_interval_min'] == 15, d)
    a = _HTML.index('id="mmp-panel"')
    blk = _HTML[a:_HTML.index('</table>', a)]
    for sel, val in (('mmp-exchange', 'binance'), ('mmp-universe', 'watchlist'),
                     ('mmp-bars', '720'), ('mmp-sort', 'pull_report')):
        k = blk.index(f'id="{sel}"')
        seg = blk[k:blk.index('</select>', k)]
        _check(f'value="{val}" selected' in seg, f'{sel}: за замовчуванням {val}')
    _check('id="ff-mm-pull-enabled" ' in blk and 'id="ff-mm-pull-enabled" checked' not in blk,
           'тумблер блоку за замовчуванням ВИМК')
    print('✓ дефолти: Binance · Watchlist бота · 720 год · 🧭 куди тягне (звіт)')


# ═══════════ 3. СКАН У ФОНІ ═════════════════════════════════════════════
def _ff(rows=None):
    ff = _mon._mk()
    ff._settings.update({'mm_pull_enabled': True, 'mmp_bias_confirm_sec': 0,
                         'mm_monitor_enabled': False, 'mm_new_enabled': False,
                         'mm_ctr_enabled': False, 'mm_fc_enabled': False})
    ff.launched = []
    ff._mmp_scan_launch = lambda s, cfg: ff.launched.append(cfg)
    if rows is not None:
        ff._mmp_scan = {'rows': {r['symbol']: r for r in rows}, 'ts': 900.0,
                        'cfg': ff._mmp_cfg(ff.get_settings()), 'ok': True, 'scanned': len(rows)}
        ff._mmp_scan_at = 990.0
    return ff


def test_scan_starts_once_and_waits_for_the_interval():
    ff = _ff()
    s = ff.get_settings()
    _check(ff._mmp_scan_tick(s, 1000.0) and len(ff.launched) == 1, 'перший такт запускає скан')
    _check(not ff._mmp_scan_tick(s, 1001.0), 'поки біжить — другий не запускаємо')
    ff._mmp_scan_running = False
    ff._mmp_scan = {'rows': {}, 'cfg': ff._mmp_cfg(s), 'ok': True}
    ff._mmp_scan_at = 1010.0
    _check(not ff._mmp_scan_tick(s, 1010.0 + 14 * 60), 'до інтервалу — тихо')
    _check(ff._mmp_scan_tick(s, 1010.0 + 15 * 60), 'інтервал минув — знову')
    print('✓ скан у фоні: один за раз, інтервал від кінця попереднього')


def test_changed_params_rescan_immediately():
    ff = _ff([_row('BTCUSDT', 70)])
    s = ff.get_settings()
    _check(not ff._mmp_scan_tick(s, 1000.0), 'свіжий результат — не скануємо')
    ff._settings['mmp_bars'] = 336
    _check(ff._mmp_scan_tick(ff.get_settings(), 1000.0), 'інша глибина → скан одразу')
    print('✓ зміна біржі/списку/глибини/сортування → скан одразу')


def test_scan_run_uses_the_shared_scanner_and_keeps_flat():
    calls = {}
    fake = types.ModuleType('detection.liq_scan')

    def _scan(**kw):
        calls.update(kw)
        return {'ok': True, 'scanned': 2, 'rows': [_row('BTCUSDT', 70), _row('ETHUSDT', 50)]}
    fake.scan_liquidity = _scan
    old = sys.modules.get('detection.liq_scan')
    oldq = sys.modules.get('detection.scan_queue')
    sys.modules['detection.liq_scan'] = fake
    setattr(sys.modules['detection'], 'liq_scan', fake)
    q = types.ModuleType('detection.scan_queue')
    q.run = lambda name, fn, source='', timeout=0: (calls.setdefault('job', name), fn())[1]
    sys.modules['detection.scan_queue'] = q
    setattr(sys.modules['detection'], 'scan_queue', q)
    try:
        ff = _ff()
        ff._mmp_watchlist = lambda: ['BTCUSDT', 'ETHUSDT']
        ff._mmp_scan_running = True
        s = ff.get_settings()
        ff._mmp_scan_run(s, ff._mmp_cfg(s))
    finally:
        if old is not None:
            sys.modules['detection.liq_scan'] = old
        if oldq is not None:
            sys.modules['detection.scan_queue'] = oldq
    _check(calls.get('job') == 'mm-pull', 'скан іде через спільну чергу сканів')
    _check(calls['keep_flat'] is True and calls['universe'] == 'watchlist'
           and calls['exchange'] == 'binance' and calls['bars'] == 720
           and calls['sort_by'] == 'pull_report' and calls['symbols'] == ['BTCUSDT', 'ETHUSDT'], calls)
    _check(set(ff._mmp_scan['rows']) == {'BTCUSDT', 'ETHUSDT'} and not ff._mmp_scan_running, ff._mmp_scan)
    _check('keep_flat: bool = False' in _LQ_SRC, 'звіт на Tickr рівновагу й далі ріже')
    print('✓ скан — готовий liq_scan через чергу `mm-pull`, ⚖ рядки лишаються')


def test_failed_scan_keeps_the_previous_rows():
    ff = _ff([_row('BTCUSDT', 70)])
    fake = types.ModuleType('detection.liq_scan')
    fake.scan_liquidity = lambda **kw: {'ok': False, 'reason': 'binance 418'}
    sys.modules['detection.liq_scan'] = fake
    setattr(sys.modules['detection'], 'liq_scan', fake)
    q = types.ModuleType('detection.scan_queue')
    q.run = lambda name, fn, source='', timeout=0: fn()
    sys.modules['detection.scan_queue'] = q
    setattr(sys.modules['detection'], 'scan_queue', q)
    ff._mmp_watchlist = lambda: ['BTCUSDT']
    s = ff.get_settings()
    ff._mmp_scan_run(s, ff._mmp_cfg(s))
    _check('BTCUSDT' in ff._mmp_scan['rows'] and ff._mmp_scan['ok'] is False
           and '418' in ff._mmp_scan['reason'], ff._mmp_scan)
    print('✓ невдалий скан не стирає попередній результат, причина видима')


# ═══════════ 4. КНИГА В ТАКТІ ДВИГУНА ══════════════════════════════════
def test_capture_builds_the_pull_book_and_its_banner():
    ff = _ff([_row('BTCUSDT', 70), _row('ETHUSDT', 30), _row('XRPUSDT', 52)])
    ff._mm_capture({}, now=1000.0)
    st = ff.mm_monitor_state(book='pull')
    by = {r['symbol']: r for r in st['rows']}
    _check(st['book'] == 'pull' and st['enabled'] and not st['limited'], st)
    _check(by['BTCUSDT']['mm'] == 'LONG' and by['BTCUSDT']['strength'] == 40, by['BTCUSDT'])
    _check(by['ETHUSDT']['mm'] == 'SHORT' and by['XRPUSDT']['mm'] is None, by)
    _check(by['BTCUSDT']['above_pct'] == 70.0 and by['BTCUSDT']['magnet_price'], by['BTCUSDT'])
    b = st['bias']
    _check(b['dir'] is None and b.get('coins') == 3, b)   # 40 vs 40 → рівновага
    _check(st['scan'].get('exchange') == 'binance' and st['scan'].get('bars') == 720, st['scan'])
    print('✓ такт будує книгу Pull і її банер (той самий mm_bias_step)')


def test_disabled_block_clears_everything_and_never_scans():
    ff = _ff([_row('BTCUSDT', 70)])
    ff._mm_capture({}, now=1000.0)
    ff._settings['mm_pull_enabled'] = False
    ff.launched.clear()
    ff._mm_capture({}, now=1030.0)
    st = ff.mm_monitor_state(book='pull')
    _check(not st['rows'] and not st['bias'] and not st['enabled'] and not ff.launched, st)
    _check(ff._mmp_scan == {}, 'результат скану теж забуто')
    print('✓ вимкнений блок гасить знімок і банер, скан не запускається')


def test_banner_change_goes_to_telegram():
    ff = _ff([_row('BTCUSDT', 70)])
    sent = []
    ff._broadcast_users = lambda cat, key, msg: (sent.append(msg) or (True, ''))
    ff._mm_capture({}, now=1000.0)
    _check(not [m for m in sent if 'МММ-PULL' in m], 'перший такт мовчки')
    ff._mmp_scan['rows'] = {'BTCUSDT': _row('BTCUSDT', 25)}
    ff._mm_capture({}, now=1030.0)
    m = [x for x in sent if 'МММ-PULL' in x]
    _check(len(m) == 1 and 'SHORT' in m[0] and 'монет:' not in m[0], m)
    _check("'mmp_bias_tg': True" in _SRC, 'тумблер TG деф. УВІМК')
    print('✓ 🧭 банер → Telegram: підтверджена зміна, перший такт мовчки')


def test_group_open_from_pull_uses_its_own_direction_and_label():
    ff = _ff([_row('BTCUSDT', 75)])
    ff._mm_capture({}, now=1000.0)
    res = ff.group_open(['BTCUSDT'], book='pull')
    _check(res['opened'] == 1 and ff.opened[0]['side'] == 'LONG', (res, ff.opened))
    _check('MMP' in ff.opened[0]['kw'].get('opened_by', ''), ff.opened[0])
    print('✓ ✋ групове відкриття бере напрямок Pull, мітка MMP')


# ═══════════ 5. ВОРОТА «ЧЕРЕЗ МММ-PULL» ════════════════════════════════
def test_gate_view_and_rule():
    ff = _ff([_row('BTCUSDT', 30)])
    ff._mm_capture({}, now=1000.0)
    v = ff.mm_gate_view('BTCUSDT', book='pull')
    _check(v['on'] and v['coin'] == 'SHORT', v)
    sys.path.insert(0, _HERE)
    _check("('mmp_gate_enabled', 'mmp_gate_banner', 'pull')" in _SC_SRC, 'пара в MM_GATES')
    _check("'pull': ('🧭МММ-Pull', '🧭 МММ-PULL')" in _SC_SRC, 'підписи воріт')
    for k in ('mmp_gate_enabled', 'mmp_gate_banner'):
        _check(f"'{k}': False" in _SC_SRC, f'дефолт ВИМК: {k}')
    _check("'mmp_gate_enabled', 'mmp_gate_banner'," in _SC_SRC, 'білий список')
    for need in ('id="sm-mmp-gate"', 'id="sm-mmp-gate-banner"',
                 "['sm-mmp-gate', 'sm-mmp-gate-banner', 'mmp_gate_enabled', 'mmp_gate_banner']",
                 "'🧭 МММ-Pull']].forEach", '🧭 Через МММ-Pull', 'за банером МММ-Pull'):
        _check(need in _HTML, f'на сторінці немає «{need}»')
    print('✓ ворота «🧭 Через МММ-Pull» + «за банером»: книга, підписи, деф. ВИМК, UI')


# ═══════════ 6. UI ═════════════════════════════════════════════════════
def test_ui_block_is_a_twin_with_its_own_columns():
    a = _HTML.index('id="mmp-panel"')
    blk = _HTML[a:_HTML.index('</table>', a)]
    th = len(re.findall(r'<th[\s>]', blk))
    m = re.search(r"_MMB\.pull\.cols = \[([^\]]*)\]", _HTML)
    ncols = len(m.group(1).split(','))
    _check(th == ncols + 2, f'<th> {th} ≠ cols {ncols}+2')
    _check(f'colspan="{ncols + 2}"' in _HTML[a:_HTML.index('</table>', a) + 400], 'colspan')
    for need in ("togglePanel('mmp')", "togglePanel('mmplist')", 'mmp-bias-banner',
                 "mmSetDir('LONG','pull')", 'ff-mmp-bias-confirm', 'mmp-scan-status',
                 "mmSort('mass','pull')", "mmSort('magnet','pull')"):
        _check(need in blk, need)
    _check("'mmplist'" in _HTML.split('_PANEL_CLOSED_BY_DEFAULT = ')[1].split(';')[0], 'список згорнутий')
    _check("mmApplyState(d.mm_pull, 'pull')" in _HTML and "want.push('mmp')" in _HTML, 'полл')
    for need in ("mm_pull_enabled: _c('ff-mm-pull-enabled')", "mmp_exchange: _v('mmp-exchange'",
                 "mmp_bars: parseInt(_v('mmp-bars'", "mmp_sort: _v('mmp-sort'",
                 "mmp_bias_tg: _c('ff-mmp-bias-tg')", 'id="ff-mmp-bias-tg"', "'🧭 банер Pull'"):
        _check(need in _HTML, need)
    _check(_HTML.index('id="mmf-panel"') < a < _HTML.index('СКАНЕР ЛІКВІДНОСТІ (власний'), 'під 🔮 Local')
    _check("'mmp'" in _SRC.split('HEAVY_SECTIONS = ')[1].split('\n')[0], 'секція mmp')
    print('✓ UI: блок-двійник під 🔮, колонки маса/магніт, налаштування скану, TG, гармошки')


def test_labels_everywhere():
    sl = open(os.path.join(_HERE, 'detection', 'signal_labels.py'), encoding='utf-8').read()
    app = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check("'MMP':              '🧭 МММ-Pull'" in sl and sl.count("'MMP': '🧭'") == 2, 'labels')
    _check('"MMP": "🧭 МММ-Pull"' in app and "'MMP': '🧭 МММ-Pull'" in _HTML, 'JS-дзеркала')
    print('✓ мітка MMP «🧭 МММ-Pull» у Python і обох JS-дзеркалах')


# ═══════════ 20.10 · скрін першого запуску: 32751/900с, «$10–10» ═════════
def test_updated_is_the_last_tick_not_the_scan():
    ff = _ff([_row('BTCUSDT', 70)])
    ff._mm_capture({}, now=5000.0)
    _check(ff._mmp_snapshot_ts == 5000.0, f'«оновлено» = такт, а не скан: {ff._mmp_snapshot_ts}')
    print('✓ «оновлено» у блоці Pull = останній такт (час скану — окремим рядком)')


def test_band_under_a_dollar_keeps_its_decimals():
    sys.path.insert(0, os.path.join(_HERE, 'detection', 'liquidation_map'))
    import ladder as L
    _check(L.fmt_band_ua(10.0, 10.1) == '$10.0–10.1', L.fmt_band_ua(10.0, 10.1))
    _check(L.fmt_band_ua(14.0, 14.2) == '$14.0–14.2', L.fmt_band_ua(14.0, 14.2))
    _check(L.fmt_band_ua(62, 63) == '$62–63' and L.fmt_band_ua(76000, 78000) == '$76 000–78 000',
           'смуга ≥ $1 — як і раніше цілими')
    _check('function _liqBandDec(' in _HTML, 'JS-дзеркало на Smart Money')
    print('✓ смуга вужча за $1: «$10.0–10.1», а не «$10–10»')


def test_timers_use_the_server_clock():
    _check("'server_now': time.time()" in _SRC, 'get_state віддає годинник сервера')
    _check('_srvSkewNote(d.server_now)' in _HTML, 'фронт бере поправку з кожної відповіді')
    a = _HTML.index('window._ffTimerTick = setInterval(')
    _check('const nowS = _nowS();' in _HTML[a:a + 300], 'глобальний 1с-тікер — серверне «зараз»')
    body = [l for l in _HTML.split('\n') if 'Date.now()' in l and '1000' in l
            and not l.strip().startswith('//')]
    _check(all('_SRV_SKEW' in l for l in body), f'сирий Date.now()/1000 лишився: {body}')
    print('✓ таймери рахують від годинника сервера (поправка server_now)')


# ═══════════ 21.10 · годинник ПК · OB 4H у таблицях · відлік скану ═══════
def test_banner_times_use_the_pc_clock():
    _check('function _srvToPc(' in _HTML, 'хелпер годинника ПК')
    for need in ('new Date(_srvToPc(B.meta.ts) * 1000)', 'new Date(_srvToPc(t) * 1000)',
                 'new Date(_srvToPc(d.ts) * 1000)'):
        _check(need in _HTML, need)
    print('✓ «оновлено / скан / наступний» — за годинником ПК')


def test_every_book_row_carries_the_4h_ob_mark():
    ff = _ff([_row('BTCUSDT', 70), _row('ETHUSDT', 30)])
    ff._mm_ob4h_marks = lambda syms: {'BTCUSDT': {'bias': 'BEARISH', 'bar_time': 1}}
    ff._mm_capture({}, now=1000.0)
    rows = {r['symbol']: r for r in ff.mm_monitor_state(ff.get_settings(), book='pull')['rows']}
    _check(rows['BTCUSDT']['ob4h']['bias'] == 'BEARISH', rows['BTCUSDT'])
    _check(rows['ETHUSDT']['ob4h'] is None, 'немає позначки → None, не вигадуємо')
    _check("ob4hMarkHTML({ ob4h: r.ob4h, side: r.mm }, 'МММ у рядку')" in _HTML, 'рендер біля символу')
    _check('ob_mark_for' in _SRC.split('def _mm_ob4h_marks')[1].split('def ')[0], 'джерело — сканер')
    print('✓ кожен рядок МММ-таблиць несе «останній OB 4H» (те саме джерело, що угоди)')


def test_next_scan_has_a_live_countdown():
    a = _HTML.index('function _mmpScanPaint(')
    seg = _HTML[a:_HTML.index('function _fmtCountdown(', a)]
    _check('ff-countdown' in seg and 'data-until' in seg, 'відлік у рядку скану')
    _check(".ff-countdown[data-until]" in _HTML, 'глобальний тікер веде відлік')
    print('✓ «наступний … · через 4:59» — живий зворотний відлік, кольорами')



# ═══════════ 21.10 · OB 4H ліворуч від назви · сортування магніту ═══════
def test_ob4h_mark_sits_left_of_the_symbol():
    a = _HTML.index("ob4hMarkHTML({ ob4h: r.ob4h, side: r.mm }, 'МММ у рядку')")
    b = _HTML.index('${tvSym(r.symbol)}', a - 400)
    _check(a < b, 'позначка OB 4H мусить стояти ПЕРЕД назвою монети')
    _check('mm-ob4h-slot' in _HTML[a - 300:a], 'слот сталої ширини — назви в стовпчик')
    _check('.mm-ob4h-slot {' in _HTML and 'width:38px' in _HTML, 'CSS слоту')
    print('✓ 🏷 OB 4H — ліворуч від назви, у слоті сталої ширини')


def test_magnet_sorts_by_share_and_distance_numbers():
    ff = _ff([_row('BTCUSDT', 70), _row('ETHUSDT', 30)])
    ff._mm_capture({}, now=1000.0)
    r = ff.mm_monitor_state(ff.get_settings(), book='pull')['rows'][0]
    _check(r['magnet_share'] == 14.2 and r['magnet_dist_pct'] == 4.5, r)
    _check(_m._mrow_num({'magnet_row': None}, 'pct') is None, 'немає сходинки → None')
    a = _HTML.index('function _mmSortVal(')
    seg = _HTML[a:_HTML.index('\n}', a)]
    _check("col === 'magnet') return (r.magnet_share" in seg, 'частка — сирим числом')
    _check("col === 'magdist') return (r.magnet_dist_pct" in seg, 'відстань — сирим числом')
    _check('parseFloat(String(r.magnet_pct)' not in seg, 'форматований підпис не парсимо')
    for need in ("mmSort('magnet','pull')", "mmSort('magdist','pull')"):
        _check(need in _HTML, need)
    _check("magdist: 'magnet'" in _HTML, 'збережене сортування відновлюється для pull')
    _check('#${B.P}-table [data-mmsort]' in _HTML, 'стрілки й на підписах усередині заголовка')
    print('✓ 🧲 Магніт сортується за «частка %» і «відстань %» (сирі числа)')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
