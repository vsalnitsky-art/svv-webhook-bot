"""🧠 МММ-DECISION — шостий банер/монітор на вердикті банера Decision (22.10).

Дослівно: «Будуємо ще один МММ банер (монітор), за основу беремо ось ці
показники» (банер Decision «SHORT 85% СИЛЬНИЙ») «І фільтр угод також додаємо
до налаштувань ("Через МММ ..." і "за банером МММ ...")».
"""
import os
import re
import subprocess
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)

_m = _mon._m
_HTML = _mon._HTML
_SRC = _mon._SRC
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _dec(reco, pl, verdict='good', conf='high'):
    return {'recommended': reco, 'prob_long': pl, 'prob_short': round(1 - pl, 4),
            'verdict': verdict, 'confidence': conf,
            'headline': (f'LONG {round(pl * 100)}%' if pl >= 0.5
                         else f'SHORT {round((1 - pl) * 100)}%')}


# ═══════════ 1. ПРАВИЛО НАПРЯМКУ (чиста функція) ═════════════════════════
def test_screenshot_case_is_a_short():
    c = _m.dec_mm_dir(_dec('SHORT', 0.15))           # «SHORT 85% СИЛЬНИЙ»
    _check(c['status'] == 'SHORT' and c['strength'] == 70 and c['prob'] == 85, c)
    _check(c['verdict'] == 'good' and c['headline'] == 'SHORT 85%', c)
    print('✓ скрін: SHORT 85% СИЛЬНИЙ → SHORT, сила 70 (перевага 85−15), ймовірність 85')


def test_direction_is_exactly_the_banner_verdict():
    c = _m.dec_mm_dir(_dec('NEUTRAL', 0.51, verdict='marginal'))
    _check(c['status'] is None and c['prob'] is None and c['strength'] == 2, c)
    c2 = _m.dec_mm_dir(_dec('LONG', 0.53, verdict='marginal'))
    _check(c2['status'] == 'LONG', 'слабкий, але явний LONG банера лишається LONG')
    print('✓ напрямок = recommended банера (NEUTRAL → ⚖), власного порогу немає')


def test_missing_is_unknown_not_balance():
    _check(_m.dec_mm_dir(None) is None and _m.dec_mm_dir({}) is None, 'порожньо')
    _check(_m.dec_mm_dir({'recommended': 'LONG'}) is None, 'без ймовірностей')
    print('✓ немає вердикту → None («невідомо» ≠ «рівновага»)')


# ═══════════ 2. КНИГА В ТАКТІ ДВИГУНА ══════════════════════════════════
class _TM:
    def __init__(self, data):
        self.data, self.calls = data, []

    def compute_decision(self, sym, price):
        self.calls.append(sym)
        return self.data.get(sym)


def _ff_with(data):
    ff = _mon._mk()
    tm = _TM(data)
    _mod = types.ModuleType('detection.trade_manager')
    _mod.get_trade_manager = lambda: tm
    sys.modules['detection.trade_manager'] = _mod
    if 'detection' in sys.modules:
        setattr(sys.modules['detection'], 'trade_manager', _mod)
    for k in ('mm_ctr_enabled', 'mm_fc_enabled'):
        ff._settings[k] = False
    ff._settings['mm_dec_enabled'] = True
    return ff, tm


def test_disabled_by_default():
    _check(_m.DEFAULT_SETTINGS['mm_dec_enabled'] is False, 'новий важкий блок деф. ВИМК')
    _check(_m.DEFAULT_SETTINGS['mmd_bias_tg'] is True, 'TG банера деф. УВІМК (як у решти)')
    print('✓ тумблер блоку деф. ВИМК (найважчий розрахунок), TG банера деф. УВІМК')


def test_capture_builds_the_dec_book_and_its_banner():
    ff, tm = _ff_with({'BTCUSDT': _dec('SHORT', 0.15), 'ETHUSDT': _dec('LONG', 0.90),
                       'XRPUSDT': _dec('NEUTRAL', 0.51, 'marginal')})
    ff._legacy = {'BTCUSDT': 0.3, 'ETHUSDT': 0.3, 'XRPUSDT': 0.3}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3, ETHUSDT=0.3, XRPUSDT=0.3), now=1000.0)
    st = ff.mm_monitor_state(book='dec')
    by = {r['symbol']: r for r in st['rows']}
    _check(st['book'] == 'dec' and st['enabled'], st)
    _check(by['BTCUSDT']['mm'] == 'SHORT' and by['BTCUSDT']['strength'] == 70, by['BTCUSDT'])
    _check(by['BTCUSDT']['dec_prob'] == 85 and by['BTCUSDT']['dec_verdict'] == 'good', by['BTCUSDT'])
    _check(by['ETHUSDT']['mm'] == 'LONG' and by['XRPUSDT']['mm'] is None, by)
    # важіль = (80 − 70) / (80 + 70 + 2) = 6.6% → ⚖ (поріг 10%)
    _check(st['bias'].get('dir') is None, st['bias'])
    print('✓ такт будує книгу Decision і її банер (той самий mm_bias_step)')


def test_even_rotation_refreshes_every_coin_once_per_period():
    """Вимога 24.10: «плавний інтервал рівномірного навантаження, головне щоб
    дані чітко оновлювались». Кожна монета — рівно раз за `mmd_refresh_min`,
    за такт однакова порція (ceil(N × 30с / період))."""
    syms = {f'C{i}USDT': _dec('LONG', 0.8) for i in range(75)}
    ff, tm = _ff_with(syms)
    ff._mmd_boot_at = 0.0
    ff._settings['mmd_refresh_min'] = 5
    fuels = _mon._fuels(**{k: 0.3 for k in syms})
    q = _m.mmd_tick_quota(75, 300)
    _check(q == 8, f'75 монет / 5 хв = 8 за такт: {q}')
    ff._mm_capture(fuels, now=1000.0)
    _check(len(tm.calls) == 75, 'перший вердикт — одразу всім (таблиця не порожня)')
    _check(len(ff.mm_monitor_state(book='dec')['rows']) == 75, 'усі 75 у таблиці')
    t = 1030.0
    for rnd in range(2):                     # два повні періоди після старту
        n_start, per_tick = len(tm.calls), []
        for _ in range(10):                  # 10 тактів × 30с = 5 хв
            n0 = len(tm.calls)
            ff._mm_capture(fuels, now=t)
            per_tick.append(len(tm.calls) - n0)
            t += 30
        got = tm.calls[n_start:]
        _check(max(per_tick) <= q, f'не більше квоти за такт: {per_tick}')
        _check(sum(1 for n in per_tick if n == 0) == 0, f'жодного порожнього такту: {per_tick}')
        _check(sorted(got) == sorted(syms), f'період {rnd + 1}: кожна монета рівно раз')
    print('✓ Decision: рівномірна ротація — кожна монета раз за період, порівну щотакту')


def test_refresh_setting_is_validated_and_drives_the_quota():
    _check(_m.DEFAULT_SETTINGS['mmd_refresh_min'] == 5, 'дефолт 5 хв')
    _check(_m.mmd_refresh_sec({'mmd_refresh_min': 0}) == 60, 'нижня межа 1 хв')
    _check(_m.mmd_refresh_sec({'mmd_refresh_min': 999}) == 3600, 'верхня межа 60 хв')
    _check(_m.mmd_refresh_sec({'mmd_refresh_min': 'x'}) == 300, 'сміття → дефолт')
    _check(_m.mmd_tick_quota(75, 900) == 3 and _m.mmd_tick_quota(5, 300) == 1,
           'квота масштабується з періодом, мінімум 1')
    _check(_m.mmd_tick_quota(0, 300) == 0, 'немає монет — нічого не рахуємо')
    _check(not hasattr(_m, 'MMD_PER_TICK'), 'фіксованого ліміту 12 більше немає')
    print('✓ період оновлення валідується і задає порцію за такт')


def test_stale_verdict_is_dropped_only_after_failed_retries():
    """Свіже або невідомо: вердикт, який не вдалось оновити 2 періоди, зникає;
    відновлений, до якого черга ще не дійшла, — лишається."""
    data = {'AUSDT': _dec('LONG', 0.8), 'BUSDT': _dec('SHORT', 0.2)}
    ff, tm = _ff_with(data)
    ff._mmd_boot_at = 0.0
    ff._settings['mmd_refresh_min'] = 1
    fuels = _mon._fuels(AUSDT=0.3, BUSDT=0.3)
    ff._mm_capture(fuels, now=1000.0)
    data.pop('BUSDT')                       # далі compute_decision → None
    ff._mm_capture(fuels, now=1060.0)
    _check('BUSDT' in ff._mmd_dec, 'одна невдача в межах 2 періодів — ще тримаємо')
    ff._mm_capture(fuels, now=1130.0)
    rows = {r['symbol'] for r in ff.mm_monitor_state(book='dec')['rows']}
    _check('BUSDT' not in ff._mmd_dec and rows == {'AUSDT'}, f'застарілий викинуто: {rows}')
    print('✓ застарілий вердикт зникає лише після невдалих спроб')


def test_trade_coins_go_first_among_due():
    syms = {f'C{i}USDT': _dec('LONG', 0.8) for i in range(20)}
    ff, tm = _ff_with(syms)
    ff._mmd_boot_at = 0.0
    ff._settings['mmd_refresh_min'] = 5
    fuels = _mon._fuels(**{k: 0.3 for k in syms})
    ff._mmd_decisions(fuels, 1000.0, extra={'C19USDT'}, period=300)
    _check(tm.calls[0] == 'C19USDT', f'монета в угоді — першою: {tm.calls}')
    print('✓ монети в угоді — першими серед тих, кому час')


def test_disabled_block_clears_everything():
    ff, tm = _ff_with({'BTCUSDT': _dec('SHORT', 0.15)})
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)
    ff._settings['mm_dec_enabled'] = False
    n = len(tm.calls)
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1030.0)
    st = ff.mm_monitor_state(book='dec')
    _check(not st['rows'] and not st['bias'] and not st['enabled'], st)
    _check(len(tm.calls) == n and not ff._mmd_dec, 'вимкнений — не рахує і забуває кеш')
    print('✓ вимкнений блок нічого не рахує і гасить знімок, банер, кеш')


def test_group_open_from_dec_uses_its_own_direction_and_label():
    ff, _ = _ff_with({'BTCUSDT': _dec('SHORT', 0.15)})
    ff._legacy = {'BTCUSDT': 0.5}      # 🧮 каже LONG — Decision має перемогти
    ff._mm_capture(_mon._fuels(BTCUSDT=0.5), now=1000.0)
    res = ff.group_open(['BTCUSDT'], book='dec')
    _check(res['opened'] == 1 and ff.opened[0]['side'] == 'SHORT', (res, ff.opened))
    _check('MMD' in ff.opened[0]['kw'].get('opened_by', ''), ff.opened[0])
    print('✓ ✋ групове відкриття бере напрямок Decision, мітка MMD')


def test_gate_view_reads_the_dec_book():
    ff, _ = _ff_with({'BTCUSDT': _dec('SHORT', 0.15)})
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)
    v = ff.mm_gate_view('BTCUSDT', book='dec')
    _check(v['on'] and v['coin'] == 'SHORT', v)
    ok = _mon._m  # noqa
    print('✓ mm_gate_view(book="dec") читає вкладку монети з книги Decision')


def test_state_persists_and_exposes_the_section():
    _check("'mmd'" in _SRC.split('HEAVY_SECTIONS = ')[1].split('\n')[0], 'секція mmd')
    _check("'mm_dec': (self.mm_monitor_state(settings, book='dec')" in _SRC, 'поле mm_dec')
    _check("'mmd': _want_mmd" in _SRC, 'served.mmd')
    _check("'mm_bias_dec':" in _SRC and "('mmd_state_since', '_mmd_state_since')" in _SRC,
           'банер і таймери переживають рестарт')
    print('✓ get_state: секція mmd → mm_dec; персист банера')


def test_source_is_the_banner_compute_decision():
    a = _SRC.index('def _mmd_decisions(')
    fn = _SRC[a:_SRC.index('\n    def ', a + 10)]
    _check('compute_decision(' in fn, 'джерело — той самий compute_decision, що банер')
    _check('evaluate_entry' not in fn and 'build_decision' not in fn, 'свого вердикту немає')
    print('✓ джерело — TradeManager.compute_decision (той самий, що банер)')


def test_labels_mirrors():
    sl = open(os.path.join(_HERE, 'detection', 'signal_labels.py'), encoding='utf-8').read()
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check("'MMD':              '🧠 МММ-Decision'" in sl and sl.count("'MMD': '🧠'") == 2, 'labels')
    _check('"MMD": "🧠 МММ-Decision"' in js and js.count('"MMD": "🧠"') == 2, 'infosite')
    _check("'MMD': '🧠 МММ-Decision'" in _HTML and _HTML.count("'MMD': '🧠'") == 2, 'page')
    print('✓ мітка MMD = 🧠 МММ-Decision у Python і обох JS-дзеркалах')


# ═══════════ 3. ВОРОТА «ЧЕРЕЗ МММ-DECISION» ════════════════════════════
def test_gate_pair_is_declared_and_off_by_default():
    _check("('mmd_gate_enabled', 'mmd_gate_banner', 'dec')" in _SC_SRC, 'пара в MM_GATES')
    _check("'mmd_gate_enabled': False" in _SC_SRC and "'mmd_gate_banner': False" in _SC_SRC,
           'деф. ВИМК')
    _check(_SC_SRC.count("'mmd_gate_enabled', 'mmd_gate_banner',") == 3, 'MM_GATES + білий список + валідація')
    _check("'dec': ('🧠МММ-Decision', '🧠 МММ-DECISION')" in _SC_SRC, 'підписи')
    for need in ('id="sm-mmd-gate"', 'id="sm-mmd-gate-banner"',
                 "['sm-mmd-gate', 'sm-mmd-gate-banner', 'mmd_gate_enabled', 'mmd_gate_banner']",
                 "['sm-mmd-gate', 'mmd_gate_enabled']", "'🧠 МММ-Decision']"):
        _check(need in _HTML, f'на сторінці немає «{need}»')
    print('✓ «Через МММ-Decision» + «за банером» — деф. ВИМК, білий список, UI')


# ═══════════ 4. UI ═════════════════════════════════════════════════════
def test_ui_block_is_a_twin_with_its_own_columns():
    a = _HTML.index('id="mmd-panel"')
    blk = _HTML[a:_HTML.index('</table>', a)]
    th = len(re.findall(r'<th[\s>]', blk))
    m = re.search(r"_MMB\.dec\.cols = \[([^\]]*)\]", _HTML)
    ncols = len(m.group(1).split(','))
    _check(th == ncols + 3, f'<th> {th} ≠ cols {ncols}+3 (TRD + ☑ + символ)')
    _check(f'colspan="{ncols + 3}"' in blk, 'colspan порожнього рядка')
    _check('mmp' not in blk and "'pull'" not in blk, 'у двійнику лишились id/книга Pull')
    for need in ("togglePanel('mmd')", "togglePanel('mmdlist')", 'ff-mm-dec-enabled',
                 'ff-mmd-bias-confirm', "mmSetDir('LONG','dec')", 'mmd-bias-banner',
                 "mmSort('dec','dec')"):
        _check(need in blk or need in _HTML, need)
    _check("'mmd', 'mmdlist'" in _HTML and "'mmdlist'" in _HTML.split('_PANEL_CLOSED_BY_DEFAULT = ')[1].split(';')[0], 'гармошки')
    _check("mmApplyState(d.mm_dec, 'dec')" in _HTML, 'стан доїжджає до книги')
    _check("want.push('mmd')" in _HTML, 'секцію просимо лише розгорнутою')
    _check('mm_dec_enabled: _c(' in _HTML and 'mmd_str_mins:' in _HTML, 'збереження')
    _check('_MMB.pull, _MMB.dec, _MMB.rdy].forEach' in _HTML, 'банер малює 1с-тікер')
    _check('ff-mmd-bias-tg' in _HTML and "🧠 банер Decision" in _HTML, 'TG-тумблер і шапка')
    i = _HTML.index('id="mmp-panel"'); j = _HTML.index('id="mmd-panel"')
    _check(i < j, 'блок 🧠 стоїть під 🧭 МММ-Pull')
    print('✓ UI: блок-двійник під 🧭 Pull, колонка 🧠 Вердикт, гармошки, збереження')


def test_js_cell_matches_the_banner():
    a = _HTML.index('const _MMD_V_UA')
    b = _HTML.index('\n}', _HTML.index('function _mmDecCell(', a)) + 2
    esc = _HTML[_HTML.index('const _mmEsc'):]
    esc = esc[:esc.index(';\n') + 2]
    js = esc + _HTML[a:b] + (
        "\nconsole.log(_mmDecCell({mm:'SHORT',dec_prob:85,dec_verdict:'good',dec_headline:'SHORT 85%'}));"
        "console.log('|');console.log(_mmDecCell({mm:null,dec_prob:null,dec_verdict:'marginal',dec_headline:'LONG 51%'}));"
        "console.log('|');console.log(_mmDecCell({}));")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    p = r.stdout.split('|')
    _check(len(p) == 3, r.stderr)
    _check('🔴' in p[0] and 'SHORT' in p[0] and '85%' in p[0] and 'СИЛЬНИЙ' in p[0], p[0])
    _check('NEUTRAL' in p[1], p[1])
    _check('—' in p[2], 'немає вердикту → «—»')
    print('✓ JS: комірка «🔴 SHORT 85% СИЛЬНИЙ» як банер Decision')


# ═══════════ 5. ФІЛЬТР «ВЕРДИКТ ≥» (вимога 22.10) ═══════════════════════
def test_verdict_filter_rule_is_pure_and_ordered():
    ok = _m.dec_verdict_ok
    _check(ok('poor', '') and ok(None, ''), "'' = фільтр не задіяно")
    _check(ok('good', 'marginal') and ok('marginal', 'marginal') and not ok('poor', 'marginal'),
           'СЕРЕДНІЙ+ пускає СЕРЕДНІЙ і СИЛЬНИЙ')
    _check(ok('good', 'good') and not ok('marginal', 'good'), 'СИЛЬНИЙ — лише СИЛЬНИЙ')
    _check(not ok(None, 'good'), 'немає вердикту ≠ СИЛЬНИЙ')
    _check(_m.DEFAULT_SETTINGS['mmd_verdict_min'] == '', 'деф. — усі')
    print('✓ «Вердикт ≥»: СЛАБКИЙ < СЕРЕДНІЙ < СИЛЬНИЙ, деф. — усі')


def test_verdict_filter_is_saved_and_served():
    ff, _ = _ff_with({'BTCUSDT': _dec('SHORT', 0.15)})
    _check("s['mmd_verdict_min'] = _vm if _vm in DEC_VERDICT_MINS else ''" in _SRC,
           'невідоме значення → «усі», а не збій')
    ff._settings['mmd_verdict_min'] = 'good'
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3), now=1000.0)
    st = ff.mm_monitor_state(book='dec')
    _check(st['verdict_min'] == 'good', st.get('verdict_min'))
    _check(ff.mm_monitor_state(book='fc')['verdict_min'] == '', 'інші книги без фільтра')
    for need in ('id="mmd-verdict-min"', "mmd_verdict_min: _mmVerdictMin(_MMB.dec, 'all')",
                 'mmd_verdict_mins: Object.assign({}, _MMB.dec.verdictMins)',
                 '&& _mmVerdictOk(B, r, d)', 'JSON.stringify(B.verdictMins || {})'):
        _check(need in _HTML, need)
    a = _HTML.index('id="mmd-min-str"'); b = _HTML.index('id="mmd-verdict-min"'); c = _HTML.index('id="mmd-str-out"')
    _check(a < b < c, 'випадайка стоїть поруч із «Сила ≥»')
    print('✓ «Вердикт ≥» зберігається в налаштуваннях і їде на сторінку поряд із «Сила ≥»')


def test_gate_with_tab_threshold_respects_verdict():
    import importlib.util
    spec = importlib.util.spec_from_file_location('_sc_for_dec', os.path.join(_HERE, 'detection', 'smc_scanner.py'))
    try:
        sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
        decide = sc.mm_gate_decide
    except Exception:
        print('· smc_scanner не вантажиться ізольовано — перевірка лише за текстом')
        _check("vmin = str(v.get('verdict_min') or '') if need_strength else ''" in _SC_SRC, 'умова в mm_gate_decide')
        return
    v = {'on': True, 'dir': 'SHORT', 'coin': 'SHORT', 'strength': 70, 'str_min': 0,
         'verdict': 'marginal', 'verdict_min': 'good'}
    _check(decide(v, 'SHORT', book='dec', need_strength=True)[1] is False, 'слабший вердикт ріже')
    _check(decide(v, 'SHORT', book='dec', need_strength=False)[1] is True, 'без ⚑ — не діє')
    v['verdict'] = 'good'
    _check(decide(v, 'SHORT', book='dec', need_strength=True)[1] is True, 'СИЛЬНИЙ пропускає')
    print('✓ ворота з «⚑ порогом вкладки» вимагають і «Вердикт ≥»')


def test_js_tab_rows_filter_by_verdict():
    a = _HTML.index('function _mmTabRows(')
    b = _HTML.index('function mmVerdictChanged(', a)
    js = ("const _MM_TABS=['all','LONG','SHORT','flat'];const _mmClampStr=v=>Math.max(0,Math.min(100,parseFloat(v)||0));"
          "function _mmMinStr(B,d){return _mmClampStr(B.strMins[d||B.dir]);}"
          "const _mmTabOf=r=>(r.mm==='LONG'?'LONG':(r.mm==='SHORT'?'SHORT':'flat'));"
          + _HTML[a:b] +
          "\nconst _MMB={};const B={dir:'all',strMins:{all:0,LONG:0,SHORT:0,flat:0},"
          "verdictMins:{all:'',LONG:'',SHORT:'',flat:''},rows:["
          "{mm:'SHORT',strength:70,dec_verdict:'good'},{mm:'LONG',strength:40,dec_verdict:'marginal'},"
          "{mm:'LONG',strength:30,dec_verdict:'poor'}]};"
          "console.log(_mmTabRows(B,'all').length);B.verdictMins.all='marginal';console.log(_mmTabRows(B,'all').length);"
          "B.verdictMins.all='good';console.log(_mmTabRows(B,'all').length);"
          # ⚠️ Вкладка LONG має СВІЙ фільтр: «Всі = СИЛЬНИЙ» на неї НЕ діє.
          "console.log(_mmTabRows(B,'LONG').length);"
          "B.verdictMins.LONG='good';console.log(_mmTabRows(B,'LONG').length);"
          "console.log(_mmTabRows(B,'SHORT').length);")
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False, encoding='utf-8') as f:
        f.write(js)
        path = f.name
    try:
        r = subprocess.run(['node', path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    _check(r.stdout.split() == ['3', '2', '1', '2', '0', '1'], (r.stdout, r.stderr))
    print('✓ JS: кожна вкладка фільтрується СВОЇМ «Вердикт ≥»')


# ═══════════ 6. «ВЕРДИКТ ≥» — СВІЙ НА КОЖНУ ВКЛАДКУ (вимога 22.10) ══════════
def test_verdict_mins_norm_is_per_tab_with_legacy():
    n = _m.dec_verdict_mins_norm
    _check(n(None, 'good') == {'all': 'good', 'LONG': 'good', 'SHORT': 'good', 'flat': 'good'},
           'старий блоб → спільне значення на всі вкладки')
    got = n({'LONG': 'marginal', 'SHORT': 'сміття'}, '')
    _check(got == {'all': '', 'LONG': 'marginal', 'SHORT': '', 'flat': ''}, got)
    _check(n('x', 'bad') == {'all': '', 'LONG': '', 'SHORT': '', 'flat': ''}, 'сміття → усі')
    print('✓ dec_verdict_mins_norm: свій поріг на вкладку, міграція зі спільного')


def test_state_and_gate_use_the_coin_tab_verdict():
    ff, _ = _ff_with({'BTCUSDT': _dec('SHORT', 0.15, verdict='marginal'),
                      'ETHUSDT': _dec('LONG', 0.85, verdict='marginal')})
    ff._settings['mmd_verdict_mins'] = {'all': '', 'LONG': '', 'SHORT': 'good', 'flat': ''}
    ff._mm_capture(_mon._fuels(BTCUSDT=0.3, ETHUSDT=0.3), now=1000.0)
    st = ff.mm_monitor_state(book='dec')
    _check(st['verdict_mins']['SHORT'] == 'good' and st['verdict_mins']['LONG'] == '', st['verdict_mins'])
    _check(ff.mm_gate_view('BTCUSDT', book='dec')['verdict_min'] == 'good', 'SHORT-монета → поріг SHORT')
    _check(ff.mm_gate_view('ETHUSDT', book='dec')['verdict_min'] == '', 'LONG-монета → поріг LONG')
    print('✓ стан віддає пороги вкладок, ворота беруть поріг ВКЛАДКИ монети')


def test_tab_switch_shows_that_tabs_verdict():
    a = _HTML.index('function mmSetDir(')
    body = _HTML[a:_HTML.index('\n}', a)]
    _check('_mmVerdictToInput(B)' in body, 'перемикання вкладки підставляє її «Вердикт ≥»')
    a = _HTML.index('function mmVerdictChanged(')
    body = _HTML[a:_HTML.index('\n}', a)]
    _check('B.verdictMins[B.dir]' in body, 'випадайка пише поріг АКТИВНОЇ вкладки')
    _check('_vs[d]' in _HTML and 'mm.verdict_mins' in _HTML, 'пороги вкладок їдуть із сервера')
    _check('🧠≥' in _HTML, 'позначка фільтра на вкладці')
    print('✓ UI: випадайка «Вердикт ≥» — поріг активної вкладки, позначка 🧠≥ на вкладці')


if __name__ == '__main__':
    n = 0
    for _name, _fn in list(globals().items()):
        if _name.startswith('test_') and callable(_fn):
            _fn(); n += 1
    print(f'\nOK — {n} тестів')
