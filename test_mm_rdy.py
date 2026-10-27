"""🧩 МММ-ГОТОВНІСТЬ — сьомий банер/монітор на SMC-«Готовності» (25.10).

Дослівно: «Створи ще один МММ банер (монітор) взявши за основу показники
"Готовність"» + «Оновлення даних організуй в чергу з іншими банерами, за
замовчуванням раз на 15хв. Щоб не було навантаження на сервер і біржі. Із
збереженням даних та їх відновленням на момент перезапуску сервера.»
"""
import os
import re
import subprocess
import sys
import tempfile
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)
import test_mm_restart as _rst   # noqa: E402  (фейкова БД)

_m = _mon._m
_HTML = _mon._HTML
_SRC = _mon._SRC
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _g(score, grade='ХОРОШИЙ', hot=False):
    return {'ok': True, 'score': score, 'grade': grade, 'hot': hot}


# ═══════════ 1. ПРАВИЛО НАПРЯМКУ (чиста функція) ═════════════════════════
def test_higher_side_wins_and_strength_is_the_margin():
    c = _m.rdy_mm_dir(_g(57, 'ХОРОШИЙ', True), _g(31, 'СЛАБКИЙ'))
    _check(c['status'] == 'LONG' and c['strength'] == 26, c)
    _check(c['rdy_score'] == 57 and c['rdy_grade'] == 'ХОРОШИЙ' and c['rdy_hot'], c)
    s = _m.rdy_mm_dir(_g(20, 'ВИЧЕРПАНО'), _g(72, 'ВІДМІННИЙ'))
    _check(s['status'] == 'SHORT' and s['dir'] < 0 and s['strength'] == 52, s)
    print('✓ бік = вищий бал, сила = перевага |L − S|')


def test_weak_winner_or_small_margin_is_balance():
    mid = _m._grade_min('СЕРЕДНІЙ')
    c = _m.rdy_mm_dir(_g(mid - 1, 'СЛАБКИЙ'), _g(5, 'ВИЧЕРПАНО'))
    _check(c['status'] is None, 'переможець нижче СЕРЕДНІЙ → ⚖')
    c2 = _m.rdy_mm_dir(_g(60), _g(52))
    _check(c2['status'] is None and c2['strength'] == 8, 'перевага ≤ 10 п.п. → ⚖')
    c3 = _m.rdy_mm_dir(_g(mid), _g(mid - 11))
    _check(c3['status'] == 'LONG', 'рівно СЕРЕДНІЙ і перевага > 10 → напрямок')
    print('✓ переможець < СЕРЕДНІЙ або перевага ≤ 10 п.п. → ⚖ рівновага')


def test_missing_is_unknown():
    _check(_m.rdy_mm_dir(None, None) is None, 'порожньо')
    _check(_m.rdy_mm_dir({'ok': False, 'score': 0}, None) is None, 'невдалий грейд')
    c = _m.rdy_mm_dir(_g(70), None)
    _check(c['status'] == 'LONG' and c['rdy_s'] is None, c)
    print('✓ немає грейдів → None («невідомо» ≠ «рівновага»)')


# ═══════════ 2. СКАН У СПІЛЬНІЙ ЧЕРЗІ + КНИГА ════════════════════════════
def _ff(rows=None):
    ff = _mon._mk()
    ff._settings.update({'mm_rdy_enabled': True, 'mmr_bias_confirm_sec': 0,
                         'mm_monitor_enabled': False, 'mm_new_enabled': False,
                         'mm_ctr_enabled': False, 'mm_fc_enabled': False})
    ff._mmr_scan, ff._mmr_scan_running, ff._mmr_scan_at = {}, False, 0.0
    ff.launched = []
    ff._mmr_scan_launch = lambda s, syms: ff.launched.append(list(syms))
    if rows is not None:
        ff._mmr_scan = {'rows': rows, 'ts': 900.0, 'ok': True, 'scanned': len(rows)}
        ff._mmr_scan_at = 990.0
    return ff


def test_defaults():
    d = _m.DEFAULT_SETTINGS
    _check(d['mm_rdy_enabled'] is False, 'важкий блок деф. ВИМК')
    _check(d['mmr_interval_min'] == 15 and _m.mmr_interval_sec({}) == 900.0, 'деф. 15 хв')
    _check(d['mmr_bias_tg'] is True, 'TG банера деф. УВІМК')
    _check(_m.mmr_interval_sec({'mmr_interval_min': 1}) == 300.0
           and _m.mmr_interval_sec({'mmr_interval_min': 999}) == 240 * 60.0, 'межі 5…240')
    print('✓ дефолти: блок ВИМК, скан раз на 15 хв (межі 5…240), TG УВІМК')


def test_scan_runs_once_and_waits_for_the_interval():
    ff = _ff()
    s = ff.get_settings()
    _check(ff._mmr_scan_tick(s, 1000.0, ['BTCUSDT']) and ff.launched == [['BTCUSDT']], 'запуск')
    _check(not ff._mmr_scan_tick(s, 1001.0, ['BTCUSDT']), 'поки біжить — другий не запускаємо')
    ff._mmr_scan_running, ff._mmr_scan_at = False, 1000.0
    ff._mmr_scan = {'ok': True, 'rows': {}}
    _check(not ff._mmr_scan_tick(s, 1000.0 + 899, ['BTCUSDT']), 'до інтервалу — ні')
    _check(ff._mmr_scan_tick(s, 1000.0 + 900, ['BTCUSDT']), 'через 15 хв — так')
    ff._mmr_scan_running, ff._mmr_scan_at = False, 2000.0
    ff._mmr_scan = {'ok': False, 'rows': {}}
    _check(ff._mmr_scan_tick(s, 2000.0 + _m.MMR_FAIL_RETRY, ['BTCUSDT']), 'невдалий — повтор за 5 хв')
    print('✓ один скан за раз, інтервал від кінця попереднього, невдалий — через 5 хв')


def test_scan_goes_through_the_shared_queue():
    a = _SRC.index('    def _mmr_scan_run(')
    fn = _SRC[a:_SRC.index('\n    def ', a + 10)]
    _check('scan_queue.run(MMR_SCAN_JOB' in fn and _m.MMR_SCAN_JOB == 'mm-rdy', 'спільна черга')
    _check('_compute_setup_both(' in fn and 'MMR_COIN_PAUSE_SEC' in fn, 'той самий грейд + пауза')
    print('✓ скан іде через спільну чергу сканів (mm-rdy), з паузою між монетами')


def test_scan_run_grades_both_sides_and_keeps_old_on_failure():
    ff = _ff()
    calls = []
    ff._compute_setup_both = lambda sym, s: (calls.append(sym) or
                                             {'LONG': _g(60), 'SHORT': _g(20, 'ВИЧЕРПАНО')})
    ff._live_price = lambda sym: 1.5
    _sq = types.ModuleType('detection.scan_queue')
    _sq.run = lambda name, fn, source='', timeout=0: fn()
    sys.modules['detection.scan_queue'] = _sq
    if 'detection' in sys.modules:
        setattr(sys.modules['detection'], 'scan_queue', _sq)
    _orig = _m.MMR_COIN_PAUSE_SEC
    _m.MMR_COIN_PAUSE_SEC = 0
    try:
        ff._mmr_scan_running = True
        ff._mmr_scan_run(ff.get_settings(), ['BTCUSDT', 'ETHUSDT'])
        _check(calls == ['BTCUSDT', 'ETHUSDT'] and not ff._mmr_scan_running, calls)
        r = ff._mmr_scan['rows']['BTCUSDT']
        _check(r['l']['score'] == 60 and r['s']['score'] == 20 and 'checks' not in r['l'], r)
        ff._compute_setup_both = lambda sym, s: None
        ff._mmr_scan_run(ff.get_settings(), ['BTCUSDT'])
        _check(ff._mmr_scan['ok'] is False and 'BTCUSDT' in ff._mmr_scan['rows'],
               'невдалий скан не стирає попередній результат')
    finally:
        _m.MMR_COIN_PAUSE_SEC = _orig
    print('✓ скан грейдить обидва боки; невдалий скан не стирає результат')


def test_capture_builds_the_book_and_banner_from_the_scan():
    rows = {'BTCUSDT': {'l': _g(66, 'ХОРОШИЙ', True), 's': _g(20, 'ВИЧЕРПАНО'), 'price': 1.0},
            'ETHUSDT': {'l': _g(15, 'ВИЧЕРПАНО'), 's': _g(58), 'price': 2.0},
            'XRPUSDT': {'l': _g(45, 'СЕРЕДНІЙ'), 's': _g(41, 'СЕРЕДНІЙ'), 'price': 3.0}}
    ff = _ff(rows)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3, ETHUSDT=0.3, XRPUSDT=0.3), now=1000.0)
    _check(not ff.launched, 'свіжий скан не повторюється')
    st = ff.mm_monitor_state(book='rdy')
    by = {r['symbol']: r for r in st['rows']}
    _check(st['book'] == 'rdy' and st['enabled'], st)
    _check(by['BTCUSDT']['mm'] == 'LONG' and by['BTCUSDT']['strength'] == 46, by['BTCUSDT'])
    _check(by['BTCUSDT']['rdy_score'] == 66 and by['BTCUSDT']['rdy_hot'], by['BTCUSDT'])
    _check(by['ETHUSDT']['mm'] == 'SHORT' and by['XRPUSDT']['mm'] is None, by)
    _check(st['scan'].get('interval_min') == 15 and st['scan'].get('graded') == 3, st['scan'])
    # важіль = (46 − 43) / (46 + 43 + 4) ≈ 3% → ⚖
    _check(st['bias'].get('dir') is None and st['bias'].get('coins') == 3, st['bias'])
    print('✓ такт будує книгу з готового скану і банер (той самий mm_bias_step)')


def test_disabled_block_clears_everything():
    ff = _ff({'BTCUSDT': {'l': _g(70), 's': _g(10), 'price': 1.0}})
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)
    ff._settings['mm_rdy_enabled'] = False
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1030.0)
    st = ff.mm_monitor_state(book='rdy')
    _check(not st['rows'] and not st['bias'] and not ff._mmr_scan, st)
    _check(not ff.launched, 'вимкнений блок скан не запускає')
    print('✓ вимкнений блок не сканує і гасить знімок, банер, результат скану')


def test_group_open_and_gate_view_use_the_rdy_book():
    ff = _ff({'BTCUSDT': {'l': _g(15, 'ВИЧЕРПАНО'), 's': _g(64), 'price': 1.0}})
    ff._mm_capture(_mon._fuels(BTCUSDT=0.5), now=1000.0)
    v = ff.mm_gate_view('BTCUSDT', book='rdy')
    _check(v['on'] and v['coin'] == 'SHORT', v)
    res = ff.group_open(['BTCUSDT'], book='rdy')
    _check(res['opened'] == 1 and ff.opened[0]['side'] == 'SHORT', (res, ff.opened))
    _check('MMR' in ff.opened[0]['kw'].get('opened_by', ''), ff.opened[0])
    print('✓ ворота і ✋ групове відкриття читають книгу rdy, мітка MMR')


# ═══════════ 3. ЗБЕРЕЖЕННЯ / ВІДНОВЛЕННЯ ═════════════════════════════════
def test_scan_survives_restart_and_is_not_repeated():
    db = _rst._DB()
    ff = _rst._mk(db)
    now = time.time()
    ff._mmr_scan = {'rows': {'BTCUSDT': {'l': _g(66), 's': _g(20), 'price': 1.0}},
                    'ts': now - 120, 'ok': True, 'scanned': 1}
    ff._mmr_scan_at = now - 120
    _check(ff._mm_books_persist(now, force=True), 'запис')
    _check('rdy_scan' in db.store[_m._DB_MM_BOOKS], 'скан у блобі')
    ff2 = _rst._boot(db)
    _check('BTCUSDT' in ff2._mmr_scan.get('rows', {}) and ff2._mmr_scan.get('restored'), ff2._mmr_scan)
    launched = []
    ff2._mmr_scan_launch = lambda *a: launched.append(a)
    ff2._mmr_scan_running = False
    _check(ff2._mmr_scan_tick({'mmr_interval_min': 15}, time.time(), ['BTCUSDT']) is False
           and not launched, 'свіжий відновлений скан не повторюється')
    _check(ff2._mmr_scan_tick({'mmr_interval_min': 15}, time.time() + 900, ['BTCUSDT']),
           'після інтервалу — скан за розкладом')
    print('✓ 💾 результат скану переживає рестарт, свіжий скан не повторюється')


def test_old_scan_is_restored_beyond_the_raw_limit_but_not_forever():
    db = _rst._DB()
    ff = _rst._mk(db)
    now = time.time()
    ff._mmr_scan = {'rows': {'BTCUSDT': {'l': _g(66), 's': _g(20)}}, 'ts': now, 'ok': True}
    ff._mmr_scan_at = now
    ff._mm_books_persist(now, force=True)
    blob = db.store[_m._DB_MM_BOOKS]
    blob['ts'] = now - 3600          # старше за 30 хв «сирих» кешів
    _check('BTCUSDT' in _rst._boot(db)._mmr_scan.get('rows', {}), 'година — відновлюємо')
    blob['ts'] = now - _m.MMR_RESTORE_MAX_AGE - 60
    _check(not _rst._boot(db)._mmr_scan, 'старше 4 год — ні')
    print('✓ скан відновлюється до 4 год (інтервал довший за 30 хв «сирих» кешів)')


def test_banner_state_persists():
    _check("'mm_bias_rdy':" in _SRC and "('mmr_state_since', '_mmr_state_since')" in _SRC,
           'банер і таймери переживають рестарт')
    _check("'rdy': '_mmr'" in _SRC, 'книга в _MM_BOOK_PREFIX (знімок для показу)')
    print('✓ банер, таймери стану і знімок для показу персистяться')


# ═══════════ 4. ГРЕЙД — ТОЙ САМИЙ, ЩО «🎯 Готовність» ═════════════════════
def test_both_sides_share_one_analysis():
    ff = _mon._mk()
    n = []
    ff._setup_inputs = lambda sym, s, *a: (n.append(sym) or {'kl1_raw': [], 'smc': None})
    ff._setup_sig = staticmethod(lambda inp, d: {'d': d})
    sg = types.ModuleType('detection.setup_grader')
    sg.grade_setup = lambda d, sig, cfg: {'ok': True, 'score': 50 if d == 'LONG' else 30,
                                          'grade': 'x', 'dir': d}
    sys.modules['detection.setup_grader'] = sg
    try:
        out = _m.FuelFilterDaemon._compute_setup_both(ff, 'BTCUSDT', {})
    finally:
        sys.modules.pop('detection.setup_grader', None)
    _check(n == ['BTCUSDT'], 'свічки й SMC-аналіз — ОДИН раз на монету')
    _check(out['LONG']['score'] == 50 and out['SHORT']['score'] == 30, out)
    a = _SRC.index('    def _compute_setup(self')
    fn = _SRC[a:_SRC.index('\n    def ', a + 10)]
    _check('_setup_inputs(' in fn and '_setup_sig(' in fn and 'grade_setup(d, sig' in fn,
           'колонка «Готовність» рахується тими самими хелперами')
    print('✓ обидва боки — з одного аналізу; той самий grade_setup, що колонка')


def test_labels_and_gates():
    sl = open(os.path.join(_HERE, 'detection', 'signal_labels.py'), encoding='utf-8').read()
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check("'MMR':              '🧩 МММ-Готовність'" in sl and sl.count("'MMR': '🧩'") == 2, 'labels')
    _check('"MMR": "🧩 МММ-Готовність"' in js and js.count('"MMR": "🧩"') == 2, 'infosite')
    _check("'MMR': '🧩 МММ-Готовність'" in _HTML and _HTML.count("'MMR': '🧩'") == 2, 'page')
    _check("('mmr_gate_enabled', 'mmr_gate_banner', 'rdy')" in _SC_SRC, 'пара в MM_GATES')
    _check("'mmr_gate_enabled': False" in _SC_SRC and
           _SC_SRC.count("'mmr_gate_enabled', 'mmr_gate_banner',") == 3, 'деф. ВИМК + MM_GATES + списки')
    for need in ('id="sm-mmr-gate"', 'id="sm-mmr-gate-banner"',
                 "['sm-mmr-gate', 'sm-mmr-gate-banner', 'mmr_gate_enabled', 'mmr_gate_banner']"):
        _check(need in _HTML, need)
    print('✓ мітка MMR = 🧩 у Python і JS-дзеркалах; ворота «Через МММ-Готовність»')


# ═══════════ 5. UI ═════════════════════════════════════════════════════
def test_ui_block_is_a_twin_under_decision():
    a = _HTML.index('id="mmr-panel"')
    blk = _HTML[a:_HTML.index('</table>', a)]
    th = len(re.findall(r'<th[\s>]', blk))
    m = re.search(r"_MMB\.rdy\.cols = \[([^\]]*)\]", _HTML)
    ncols = len(m.group(1).split(','))
    _check(th == ncols + 3, f'<th> {th} ≠ cols {ncols}+3 (TRD + ☑ + символ)')
    _check(f'colspan="{ncols + 3}"' in blk, 'colspan')
    _check('mmd' not in blk.replace('data-mmdir', '') and "'dec'" not in blk, 'лишки двійника')
    _check('id="mmrlist-body"' in blk and 'id="mmrlist-caret"' in blk, 'гармошка списку — свої id')
    for need in ("togglePanel('mmr')", "togglePanel('mmrlist')", 'ff-mm-rdy-enabled',
                 'ff-mmr-bias-confirm', 'id="mmr-interval"', 'id="mmr-scan-status"',
                 "mmSetDir('LONG','rdy')", 'mmr-bias-banner', "mmSort('rdy','rdy')"):
        _check(need in blk, need)
    _check("'mmrlist'" in _HTML.split('_PANEL_CLOSED_BY_DEFAULT = ')[1].split(';')[0], 'список згорнутий')
    _check("mmApplyState(d.mm_rdy, 'rdy'); _mmrScanPaint(d.mm_rdy.scan)" in _HTML, 'стан')
    _check("want.push('mmr')" in _HTML and 'mmr_interval_min: parseInt(' in _HTML, 'секція/збереження')
    _check('ff-mmr-bias-tg' in _HTML and '🧩 банер Готовність' in _HTML, 'TG')
    _check(_HTML.index('id="mmd-panel"') < _HTML.index('id="mmr-panel"'), 'під 🧠 Decision')
    print('✓ UI: блок-двійник під 🧠 Decision, інтервал і стан скану, гармошки')


def test_js_cell():
    a = _HTML.index('const _MMR_GRADE_COL')
    b = _HTML.index('\n}', _HTML.index('function _mmRdyCell(', a)) + 2
    esc = _HTML[_HTML.index('const _mmEsc'):]
    esc = esc[:esc.index(';\n') + 2]
    js = esc + _HTML[a:b] + (
        "\nconsole.log(_mmRdyCell({mm:'LONG',rdy_score:57,rdy_grade:'ХОРОШИЙ',rdy_hot:true,rdy_l:57,rdy_s:31,rdy_side:'LONG'}));"
        "console.log('|');console.log(_mmRdyCell({mm:null,rdy_score:40,rdy_grade:'СЕРЕДНІЙ',rdy_l:40,rdy_s:35}));"
        "console.log('|');console.log(_mmRdyCell({}));")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    p = r.stdout.split('|')
    _check(len(p) == 3, r.stderr)
    _check('🟢' in p[0] and '57%' in p[0] and 'ХОРОШИЙ' in p[0] and '🎯' in p[0] and 'L57/S31' in p[0], p[0])
    _check('⚖' in p[1] and 'L40/S35' in p[1], p[1])
    _check('—' in p[2], p[2])
    print('✓ JS: «🟢 LONG 57% ХОРОШИЙ 🎯 · L57/S31»')


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
    print(f'OK — {len(tests)} тестів')
