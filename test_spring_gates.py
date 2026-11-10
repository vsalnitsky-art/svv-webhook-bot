"""🌀/🔭 Вимога 10.11: ворота «Через Пружину-WL / -скан» (як «🔔 Через МММ-SBR»),
таблиця скану без монет WATCHLIST, монети Binance без Bybit, виділення
монет, що проходять ворота, і «Рекомендовані ботом» на інфосайті."""
import importlib.util
import re
import json
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def _load(name, path):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, *path.split('/')))
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


sg = _load('_t_spring_gates', 'detection/spring_gates.py')
import test_spring_scan as T  # noqa: E402  (фікстури скану)

ss = T.ss
HTML = T.HTML
APP = open(os.path.join(ROOT, 'infosite', 'app.js'), encoding='utf-8').read()
IDX = open(os.path.join(ROOT, 'infosite', 'index.html'), encoding='utf-8').read()


def _rec(pl, ps, state='charged', charge=70):
    return {'ok': True, 'state': state, 'pot_long': pl, 'pot_short': ps, 'charge': charge}


# ── 1. чисті правила ────────────────────────────────────────────────────────
def test_tab_side_mirrors_the_page_tabs():
    assert sg.tab_side(_rec(16, 58)) == 'SHORT'
    assert sg.tab_side(_rec(40, 10)) == 'LONG'
    assert sg.tab_side(_rec(30, 30)) is None
    assert sg.tab_side(_rec(40, 10, state='flat')) is None
    assert sg.tab_side(None) is None
    js = HTML[HTML.index('function _sprFilter('):]
    assert "r.state !== 'flat'" in js[:400], 'правило вкладок — те саме, що на сторінці'


def test_banner_side_needs_enough_coins():
    assert sg.banner_side({'lean': 'SHORT', 'enough': True}) == 'SHORT'
    assert sg.banner_side({'lean': 'SHORT', 'enough': False}) is None
    assert sg.banner_side({'lean': None}) is None
    assert sg.banner_side(None) is None


def test_book_decide_like_mm_gates():
    ok = sg.book_decide(sg.TAG_WL, _rec(10, 50), None, 'SHORT')
    assert ok[0] and ok[1] and ok[2].endswith(':✓')
    no = sg.book_decide(sg.TAG_WL, _rec(10, 50), None, 'LONG')
    assert no[0] and not no[1] and 'SHORT' in no[3]
    miss = sg.book_decide(sg.TAG_XS, None, None, 'LONG')
    assert not miss[1] and 'немає в таблиці' in miss[3], '«невідомо» ≠ «за напрямком»'
    bn = sg.book_decide(sg.TAG_WL, _rec(10, 50), 'LONG', 'SHORT', need_banner=True)
    assert not bn[1] and 'банер' in bn[3]
    bn2 = sg.book_decide(sg.TAG_WL, _rec(10, 50), None, 'SHORT', need_banner=True)
    assert not bn2[1], 'банер ⚖ → відмова'
    assert sg.book_decide(sg.TAG_WL, _rec(10, 50), 'SHORT', 'SHORT', need_banner=True)[1]


def test_passes_gates_uses_enabled_spring_gates():
    s = {'spring_gate_enabled': True, 'spring_min_pot': 40, 'spring_charge_gate_enabled': False}
    assert sg.passes_gates(_rec(10, 58), s)[:2] == (True, 'SHORT')
    assert not sg.passes_gates(_rec(10, 35), s)[0], 'потенціал < порогу'
    s2 = dict(s, spring_charge_gate_enabled=True, spring_min_charge=80)
    assert not sg.passes_gates(_rec(10, 58, charge=70), s2)[0], 'заряд < порогу'
    assert not sg.passes_gates(_rec(10, 58, state='flat'), s)[0], 'без напрямку'
    pot = lambda rec, side: 99
    assert sg.passes_gates(_rec(10, 35), s, pot)[0], 'потенціал — через spring_coil.potential'


def test_scan_member_for_watchlist_coins():
    row = _rec(5, 60)
    assert sg.scan_member(row, None, 80) == (row, 'скан')
    wl = _rec(5, 60, charge=85)
    assert sg.scan_member(None, wl, 80) == (wl, 'WATCHLIST')
    assert sg.scan_member(None, None, 80) == (None, '')
    r, why = sg.scan_member(None, _rec(5, 60, charge=70), 80)
    assert r is None and '70 < 80' in why
    r, why = sg.scan_member(None, _rec(5, 60, state='building', charge=90), 80)
    assert r is None and 'заряджається' in why


def test_watchlist_miss_names_the_real_reason():
    # Кейс DOGE/MNT 10.11: заряд 66-68 ≥ 🔋 60, але < фільтра скану 80 —
    # відмова мусить казати це, а не «монети немає в таблиці».
    rec, why = sg.scan_member(None, _rec(5, 60, charge=66), 80)
    ok = sg.book_decide(sg.TAG_XS, rec, None, 'SHORT', src=why)
    assert ok[0] and not ok[1]
    assert '66 < 80' in ok[2] and '66 < 80' in ok[3]
    assert 'немає в таблиці' not in ok[2]


# ── 2. сканер: ворота в _signal_allowed ─────────────────────────────────────
def _scanner(settings, spring, wl=None):
    m = _load('_t_smc_spring_gates', 'detection/smc_scanner.py')
    sc = m.SMCScanner.__new__(m.SMCScanner)
    sc._settings = dict(m.DEFAULT_SETTINGS if hasattr(m, 'DEFAULT_SETTINGS') else {}, **settings)
    sc._spring = dict(spring)
    sc.get_watchlist = lambda: list(wl or spring)
    return m, sc


def test_scanner_book_gates_wl():
    m, sc = _scanner({'spring_enabled': True, 'sprw_gate_enabled': True},
                     {'AAAUSDT': _rec(10, 60), 'BBBUSDT': _rec(60, 10)})
    r = sc._spring_book_gates('AAAUSDT', 'SHORT')
    assert len(r) == 1 and r[0][1]
    assert not sc._spring_book_gates('AAAUSDT', 'LONG')[0][1]
    assert not sc._spring_book_gates('ZZZUSDT', 'LONG')[0][1], 'монети немає → відмова'
    sc._settings['spring_enabled'] = False
    assert sc._spring_book_gates('AAAUSDT', 'LONG') == [], 'вимкнена Пружина → не діє'
    m2, off = _scanner({}, {'AAAUSDT': _rec(10, 60)})
    assert off._spring_book_gates('AAAUSDT', 'LONG') == [], 'деф. ВИМК'


def test_scanner_book_gates_scan_uses_the_live_engine():
    m, sc = _scanner({'spring_enabled': True, 'sprx_gate_enabled': True, 'sprx_gate_banner': True},
                     {'AAAUSDT': _rec(10, 60, charge=90)})
    fake = types.ModuleType('detection.spring_scan')

    class E:
        def row_for(self, s): return _rec(60, 5) if s == 'XYZUSDT' else None
        def settings(self): return {'min_charge': 80}
        def banner_agg(self): return {'lean': 'SHORT', 'enough': True}
    fake.get_spring_scan = lambda: E()
    old = sys.modules.get('detection.spring_scan')
    sys.modules['detection.spring_scan'] = fake
    try:
        r = sc._spring_book_gates('AAAUSDT', 'SHORT')[0]
        assert r[1] and 'WATCHLIST' in r[2], 'монета WATCHLIST судиться за відбором скану'
        assert not sc._spring_book_gates('XYZUSDT', 'LONG')[0][1], 'банер SHORT проти LONG'
        sc._settings['sprx_gate_banner'] = False
        assert sc._spring_book_gates('XYZUSDT', 'LONG')[0][1]
        fake.get_spring_scan = lambda: None
        assert sc._spring_book_gates('XYZUSDT', 'LONG') == [], 'рушія немає → не діє'
    finally:
        if old is not None:
            sys.modules['detection.spring_scan'] = old
        else:
            sys.modules.pop('detection.spring_scan', None)


def test_gate_sits_in_signal_allowed_and_settings():
    src = open(os.path.join(ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    i = src.index('def _signal_allowed(')
    fn = src[i:src.index('\n    def ', i + 10)]
    assert fn.index('_spring_book_gates') < fn.index('# OB'), 'ворота стоять ДО OB/PD/…'
    for k in ('sprw_gate_enabled', 'sprw_gate_banner', 'sprx_gate_enabled', 'sprx_gate_banner'):
        assert f"'{k}': False" in src, f'дефолт {k} = ВИМК'
        assert src.count(f"'{k}'") >= 3, f'{k} у білому списку і валідації'


def test_pass_map_marks_gate_passers():
    m, sc = _scanner({'spring_enabled': True, 'spring_gate_enabled': True, 'spring_min_pot': 40},
                     {'AAAUSDT': _rec(10, 60), 'BBBUSDT': _rec(30, 10)})
    pm = sc.spring_pass_map()
    assert pm['AAAUSDT']['ok'] and pm['AAAUSDT']['side'] == 'SHORT'
    assert not pm['BBBUSDT']['ok']


# ── 3. скан біржі: без WATCHLIST · Bybit ────────────────────────────────────
def test_scan_skips_watchlist_coins():
    eng, calls = T._mk()
    eng._watchlist = lambda: ['COILUSDT']
    res = eng.scan('bybit', 50e6)
    assert res['rows'] == [] and 'COILUSDT' not in {s for _, s in calls['fetch']}
    assert res['breakdown']['in_wl'] == 1


def test_coin_added_to_watchlist_leaves_the_scan_table():
    eng, _ = T._mk()
    eng.scan('bybit', 50e6)
    assert [r['symbol'] for r in eng.rows()] == ['COILUSDT']
    eng._watchlist = lambda: ['COILUSDT']
    assert eng.rows() == [], 'одразу не показуємо'
    eng.recalc()
    assert 'COILUSDT' not in eng._table
    assert any('WATCHLIST' in x['why'] for x in eng._removed)


def test_split_bybit():
    keep, flag, n = ss.split_bybit(['A', 'B'], {'A'}, False)
    assert keep == ['A'] and n == 1 and flag == {'A': True, 'B': False}
    keep, flag, n = ss.split_bybit(['A', 'B'], {'A'}, True)
    assert keep == ['A', 'B'] and n == 0
    keep, flag, n = ss.split_bybit(['A', 'B'], None, False)
    assert keep == ['A', 'B'] and flag == {'A': None, 'B': None}, '«невідомо» ≠ «немає»'


def _mk_binance(show):
    eng, calls = T._mk()

    def tick(ex):
        return dict(T.TICKERS) if ex == 'binance' else {'WILDUSDT': {}}   # COIL на Bybit немає
    eng._tickers = tick
    eng.update_settings({'show_non_bybit': show})
    return eng, calls


def test_binance_scan_hides_coins_missing_on_bybit():
    eng, calls = _mk_binance(False)
    res = eng.scan('binance', 50e6)
    assert res['rows'] == [] and res['breakdown']['not_bybit'] == 1
    assert 'COILUSDT' not in {s for _, s in calls['fetch']}, 'не коштує запиту барів'
    assert eng.settings()['show_non_bybit'] is False


def test_binance_scan_shows_them_with_a_flag_when_asked():
    eng, _ = _mk_binance(True)
    res = eng.scan('binance', 50e6)
    assert [(r['symbol'], r['bybit']) for r in res['rows']] == [('COILUSDT', False)]
    eng.update_settings({'show_non_bybit': False})
    assert eng.rows() == [] and eng.row_for('COILUSDT') is None


def test_bybit_list_failure_is_fail_open():
    eng, _ = T._mk()

    def tick(ex):
        if ex == 'bybit':
            raise RuntimeError('403')
        return dict(T.TICKERS)
    eng._tickers = tick
    res = eng.scan('binance', 50e6)
    assert [r['symbol'] for r in res['rows']] == ['COILUSDT']
    assert res['breakdown']['bybit_check'] is False and res['rows'][0]['bybit'] is None


# ── 4. інфосайт: «Рекомендовані ботом» ──────────────────────────────────────
def test_recommended_rows():
    sys.modules.setdefault('detection', types.ModuleType('detection')).__path__ = [os.path.join(ROOT, 'detection')]
    ip = _load('_t_info_portal_rec', 'web/info_portal.py')
    wl = {'AAAUSDT': _rec(10, 60), 'BBBUSDT': _rec(30, 10)}
    pm = {'AAAUSDT': {'ok': True, 'side': 'SHORT'}, 'BBBUSDT': {'ok': False, 'side': 'LONG'}}
    scan = [dict(_rec(70, 5), symbol='XYZUSDT', bybit=False, vol_usd=1e7),
            dict(_rec(70, 5), symbol='AAAUSDT')]
    rows = ip.recommended_rows(wl, pm, scan)
    got = {(r['symbol'], r['src'], r['side']) for r in rows}
    assert got == {('AAAUSDT', 'wl', 'SHORT'), ('XYZUSDT', 'scan', 'LONG')}
    assert next(r for r in rows if r['symbol'] == 'XYZUSDT')['bybit'] is False


def _ipl(name):
    sys.modules.setdefault('detection', types.ModuleType('detection')).__path__ = [os.path.join(ROOT, 'detection')]
    return _load(name, 'web/info_portal.py')


def _ob(bias, hi, lo, t):
    return {'bias': bias, 'bar_high': hi, 'bar_low': lo, 'bar_time': t}


def test_rec_sl_ladder_ob_1h_then_lower_tf():
    """🛑 SL = OB у бік угоди: пріоритет 1H, далі 15m → 5m; не VOB (вимога 10.11)."""
    ip = _ipl('_t_info_portal_rec2')
    z1 = {'internal': [_ob('BULLISH', 96, 94, 2000)]}            # під ціною — не для SHORT
    z15 = {'internal': [_ob('BEARISH', 104, 102, 3000)]}
    r = ip.rec_sl_from_obs({'1h': z1, '15m': z15, '5m': {}}, 'SHORT', 100.0,
                           {'queue3_vob_sl_buffer_pct': 0.0})
    assert r['sl'] == 104.0 and r['tf'] == '15m' and r['fb'] is True, r
    r2 = ip.rec_sl_from_obs({'1h': z1}, 'LONG', 100.0, {'queue3_vob_sl_buffer_pct': 0.0})
    assert r2['sl'] == 94.0 and r2['tf'] == '1h' and r2['fb'] is False
    # старший TF ще рахується → чекаємо, молодший не підставляємо
    assert ip.rec_sl_from_obs({'1h': None, '15m': z15}, 'SHORT', 100.0, {})['wait']
    # блок, через який ціна пройшла, стопом не є; нічого → гарантія %
    gone = {'internal': [_ob('BEARISH', 99, 98, 1)]}
    fb = ip.rec_sl_from_obs({'1h': gone, '15m': {}, '5m': {}}, 'SHORT', 100.0,
                            {'autosl_fallback_pct': 2})
    assert abs(fb['sl'] - 102.0) < 1e-9 and fb['tf'] == '%'
    # найновіший internal першим
    zz = {'internal': [_ob('BEARISH', 110, 108, 1), _ob('BEARISH', 105, 103, 9)]}
    assert ip.rec_ob_level(zz, 'SHORT', 100.0)[0] == 105.0


def test_rec_levels_and_sort_by_probability():
    ip = _ipl('_t_info_portal_rec3')
    row = dict(_rec(10, 60), symbol='AAAUSDT', side='SHORT', tgt_short=90.0, move_short_pct=-10.0)
    lv = ip.rec_levels(row, {'sl': 105.0, 'pct': 5.0, 'label': 'OB 1H', 'tf': '1h'})
    assert lv['tp'] == 90.0 and lv['sl'] == 105.0 and lv['rr'] == 2.0 and lv['sl_tf'] == '1h'
    scan = [dict(_rec(70, 5), symbol='AUSDT', prob_long=60),
            dict(_rec(5, 70), symbol='BUSDT', prob_long=10),
            dict(_rec(70, 5), symbol='CUSDT', prob_long=80)]
    rows = ip.recommended_rows({}, {}, scan)
    assert [r['symbol'] for r in rows] == ['BUSDT', 'CUSDT', 'AUSDT']   # 90 · 80 · 60


def test_rec_table_layout():
    """Без колонок «Джерело» / «Стан» / «Потенціал» і без пояснень; є SL і TP."""
    i = IDX.index('id="rec-tbl"'); head = IDX[i:IDX.index('</thead>', i)]
    for gone in ('Джерело', 'Стан', 'Потенціал'):
        assert gone not in head, gone
    assert 'SL' in head and 'TP' in head
    cols = re.findall(r'<th[^>]*>([^<]*)</th>', head)
    assert cols[-1] == 'Заряд', cols
    assert cols.index('🎯 TP') == cols.index('🛑 SL') + 1, cols
    assert 'rec-foot' not in IDX
    assert 'function recPath' in APP and 'rec-ic' in APP
    assert 'rec-wrap' in IDX and '.rec-wrap { max-height' in open(os.path.join(ROOT, 'infosite', 'style.css'), encoding='utf-8').read()


def test_gate_mark_is_left_of_the_coin():
    i = HTML.index('r.gate_ok ? h.replace(')
    seg = HTML[i:i + 600]
    assert "h.replace('<td style=\"white-space:nowrap\">'" in seg
    assert 'spr-gate-ok' not in seg and '✅' not in seg, 'без квадрата ✅ — лише кант (вимога 10.11)'
    assert '▲' not in seg and '▼' not in seg, 'без трикутників — напрямок показує кант'
    assert '#f87171' in seg, 'кант SHORT червоний'


def test_infosite_has_the_table_with_tabs():
    assert 'id="rec-card"' in IDX and 'id="rec-tabs"' in IDX
    for d in ('all', 'LONG', 'SHORT'):
        assert f'data-d="{d}"' in IDX
    assert '/api/info/recommended' in APP and 'function recRender' in APP
    assert "'/api/info/recommended'" in open(os.path.join(ROOT, 'web', 'info_portal.py'), encoding='utf-8').read()


# ── 5. сторінка бота ────────────────────────────────────────────────────────
def test_page_controls():
    for i in ('sm-sprw-gate', 'sm-sprw-gate-banner', 'sm-sprx-gate', 'sm-sprx-gate-banner', 'xspr-nonbybit'):
        assert f'id="{i}"' in HTML, i
    assert '_SM_SPR_GATES' in HTML and "concat(_SM_SPR_GATES)" in HTML
    assert 'show_non_bybit' in HTML and '_xsBybitBadge' in HTML
    assert 'r.gate_ok' in HTML, 'монети, що проходять ворота, виділені'


def test_no_double_dollar():
    """Кейс скріна: «$$13.4M» — fmtVolShort уже ставить «$»."""
    i = HTML.index('function _xsVol(')
    assert "replace(/^\\$/, '')" in HTML[i:i + 400]


def test_gone_chips_are_grouped():
    assert 'function _xsGoneGroups' in HTML and "'🗑 ' + g.symbol" in HTML


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); n += 1
            print('ok', k)
    print(f'{n} passed')
