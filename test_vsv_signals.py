"""🛰 VSV SIGNALS (вимога 24.11): радар волатильності ОБОХ Пружин.
25.11: + ЙМОВІРНИЙ напрямок (`hint`, vsv_direction) з фактичною точністю.

— монета береться, коли вона «🌀 заряджена» (≥65), уже почала імпульс або VSV·AI
  бачить сетап; дубль — один раз, WATCHLIST сильніший;
— ранг — VSV Score (заряд · стиснення · низька волатильність · висихання обсягу · AI);
— у рядку немає ані LONG/SHORT, ані SL/ціль (лише інформація);
— поруч із назвою кнопка «ⓘ Що це?» з поясненням у термінах трейдера.
"""
import importlib.util
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))


def _ip():
    spec = importlib.util.spec_from_file_location('_t_ip_vsv', os.path.join(ROOT, 'web', 'info_portal.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


IP = _ip()
IDX = open(os.path.join(ROOT, 'infosite', 'index.html'), encoding='utf-8').read()
APP = open(os.path.join(ROOT, 'infosite', 'js', 'app.js'), encoding='utf-8').read()
CSS = open(os.path.join(ROOT, 'infosite', 'css', 'style.css'), encoding='utf-8').read()

DIR_KEYS = ('side', 'dir', 'lean', 'prob_long', 'prob_short', 'tgt_long', 'tgt_short',
            'sl', 'tp', 'm15', 'pot_long', 'pot_short', 'move_long_pct', 'move_short_pct')


def _rec(charge, state='charged', sq=0, rank=50, vr=1.0, ml=3.0, ms=-2.0, **kw):
    r = {'charge': charge, 'state': state, 'squeeze_bars': sq, 'atr_rank': rank,
         'vol_ratio': vr, 'move_long_pct': ml, 'move_short_pct': ms, 'price': 1.0,
         'dir': 'LONG', 'lean': 'LONG', 'prob_long': 70, 'pot_long': 60, 'pot_short': 10,
         'tgt_long': 1.03, 'tgt_short': 0.98}
    r.update(kw)
    return r


def test_view_has_no_direction_and_scores():
    v = IP.vsv_signal_view(_rec(90, sq=24, rank=10, vr=0.5), {'ready': True, 'scenario_name': 'E2'})
    assert v['score'] >= 80 and v['grade'] == 'A' and v['ai_ready'] and v['ai_scenario'] == 'E2'
    assert v['amp_pct'] == 2.5, 'амплітуда — середній |хід| в обидва боки'
    for k in DIR_KEYS:
        assert k not in v, k
    lo = IP.vsv_signal_view(_rec(66, rank=90, vr=2.0))
    assert lo['score'] < v['score'] and lo['grade'] == 'C'
    assert IP.vsv_signal_view({'charge': None}) is None
    assert IP.vsv_signal_view({'ok': False, 'charge': 80}) is None


def test_rows_select_both_springs_rank_and_drop_direction():
    wl = {'AAAUSDT': _rec(92, sq=20, rank=8, vr=0.6),
          'BBBUSDT': _rec(40, state='flat'),                       # без заряду → геть
          'DUPUSDT': _rec(70)}
    scan = [dict(_rec(85, sq=10, rank=20, vr=0.8), symbol='CCCUSDT', vol_usd=6e7, bybit=True),
            dict(_rec(99), symbol='DUPUSDT'),                     # дубль — WATCHLIST сильніший
            dict(_rec(30, state='fired', fired=True), symbol='IMPUSDT'),  # імпульс береться
            dict(_rec(20, state='flat'), symbol='AIUSDT')]        # AI бачить сетап → береться
    rows, cnt = IP.vsv_signal_rows(wl, scan, {'AIUSDT': {'ready': True}}, 'binance')
    syms = [r['symbol'] for r in rows]
    assert 'BBBUSDT' not in syms and set(syms) == {'AAAUSDT', 'CCCUSDT', 'DUPUSDT', 'IMPUSDT', 'AIUSDT'}
    assert syms[0] == 'AAAUSDT' and [r['score'] for r in rows] == sorted((r['score'] for r in rows), reverse=True)
    dup = next(r for r in rows if r['symbol'] == 'DUPUSDT')
    assert dup['src'] == 'wl' and dup['ex'] == 'bybit' and dup['charge'] == 70
    c = next(r for r in rows if r['symbol'] == 'CCCUSDT')
    assert c['ex'] == 'binance' and c['vol_usd'] == 6e7
    assert next(r for r in rows if r['symbol'] == 'IMPUSDT')['impulse'] is True
    for r in rows:
        for k in DIR_KEYS:
            assert k not in r, (r['symbol'], k)
    assert cnt == {'total': 5, 'shown': 5, 'impulse': 1, 'ai': 1, 'charged': 4}
    rows2, cnt2 = IP.vsv_signal_rows(wl, scan, {'AIUSDT': {'ready': True}}, 'binance', limit=2)
    assert len(rows2) == 2 and cnt2['total'] == 5


def test_infosite_info_button_and_no_direction():
    head = IDX[IDX.index('id="rec-card"'):IDX.index('id="rec-tbl"')]
    assert 'id="rec-info-btn"' in head and 'aria-controls="rec-info"' in head
    info = IDX[IDX.index('id="rec-info"'):IDX.index('</div>', IDX.index('id="rec-info"'))]
    for w in ('squeeze', 'volatility expansion', 'range breakout'):
        assert w in info, w
    full = IDX[IDX.index('id="rec-info"'):IDX.index('id="rec-tbl"')]
    for w in ('Чому напрямок не є 100% вірним і кінцевим', 'liquidity sweep', 'може змінитися',
              '50% — це рівень випадкового вгадування', 'не торгова рекомендація'):
        assert w in full, w
    tbl = IDX[IDX.index('id="rec-tbl"'):IDX.index('</table>', IDX.index('id="rec-tbl"'))]
    for gone in ('LONG', 'SHORT', '1H · OB 1H', 'OB 15m', 'SL', 'ЦІЛЬ'):
        assert gone not in tbl, gone
    blk = APP[APP.index('var REC = '):APP.index('function recRender()')]
    assert 'r.side' not in blk and 'pp-long' not in blk and 'pp-short' not in blk, \
        'бік — лише зведенням r.hint (25.11), а не сирими полями Пружини'
    assert 'rec-info' in APP and 'aria-expanded' in APP
    for c in ('.rec-info-btn', '.rec-info[hidden]', '.rec-warn', '.rec-score', '.rec-grade.g-A', '.rec-phase.imp'):
        assert c in CSS, c
    assert re.search(r'demo\.js\?v=\d+', IDX) and 'demo.js?v=6' not in IDX


def test_tg_new_coins_only_after_base_and_with_cooldown():
    r = lambda S: {'symbol': S, 'score': 80, 'grade': 'A', 'charge': 85}
    new, st = IP.vsv_tg_new([r('A'), r('B')], {}, 1000)
    assert new == [] and st['init'] and st['in'] == ['A', 'B'], 'перший прохід — тиха база'
    new, st2 = IP.vsv_tg_new([], st, 1060)
    assert new == [] and st2 == st, 'порожній список — немає даних, стан не чіпаємо'
    new, st = IP.vsv_tg_new([r('A'), r('B'), r('C')], st, 1100)
    assert [x['symbol'] for x in new] == ['C']
    new, st = IP.vsv_tg_new([r('A'), r('B')], st, 1200)          # C вийшла
    new, st = IP.vsv_tg_new([r('A'), r('B'), r('C')], st, 1300)  # блимнула назад
    assert new == [], 'та сама монета не частіше за cooldown'
    new, st = IP.vsv_tg_new([r('A')], st, 1400)
    new, st = IP.vsv_tg_new([r('A'), r('C')], st, 1300 + IP.VSV_TG_COOLDOWN)
    assert [x['symbol'] for x in new] == ['C']


def test_tg_text_is_compact_and_has_direction_score_and_real_accuracy():
    rows = [
        {'symbol': 'AAAUSDT', 'score': 86.4, 'grade': 'A',
         'hint': {'side': 'LONG', 'prob': 78.2, 'acc': {'acc': 71.4, 'n': 28}}},
        {'symbol': 'BBBUSDT', 'score': 72.0, 'grade': 'B',
         'hint': {'side': 'SHORT', 'prob': 64.9}},
        {'symbol': 'CCCUSDT', 'score': 55.0, 'grade': 'C', 'hint': {'side': None}},
    ]
    t = IP.vsv_tg_text(rows)
    assert t.startswith('🛰 <b>VSV SIGNALS</b>')
    assert '🚀 <b>#AAAUSDT</b> · 🟢 <b>LONG</b>' in t
    assert '🎯 78%' in t and '🏆 VSV <b>86 A</b>' in t and '📈 71% · 28' in t
    assert '🚀 <b>#BBBUSDT</b> · 🔴 <b>SHORT</b>' in t and '🎯 65%' in t
    assert '🚀 <b>#CCCUSDT</b> · ⚖ <b>NEUTRAL</b>' in t
    for old in ('Напрямок не вказується', 'заряд ', 'squeeze ', 'амплітуд'):
        assert old not in t, old


def test_tg_tick_sends_new_a_grade_coins_to_signals_theme():
    sent = []
    IP._vsv_tg.update({'boot': 0, 'set': {'enabled': True, 'grade': 'A'},
                       'st': {'init': True, 'in': ['AAAUSDT'], 'sent': {}}})
    rows = [{'symbol': 'AAAUSDT', 'score': 85, 'grade': 'A'},
            {'symbol': 'BBBUSDT', 'score': 86, 'grade': 'A',
             'hint': {'side': 'LONG', 'prob': 75}},
            {'symbol': 'CCCUSDT', 'score': 70, 'grade': 'B'}]
    old = IP.recommended_payload
    try:
        IP.recommended_payload = lambda now=None: {'rows': rows}
        assert IP.vsv_tg_tick(now=10_000, send=lambda t: sent.append(t) or True) == ['BBBUSDT']
        assert len(sent) == 1 and '#BBBUSDT' in sent[0] and '#AAAUSDT' not in sent[0]
        assert '#CCCUSDT' not in sent[0], 'Telegram за замовчуванням — лише клас A'
        IP._vsv_tg['set'] = {'enabled': False, 'grade': 'A'}
        rows.append({'symbol': 'DDDUSDT', 'score': 88, 'grade': 'A'})
        assert IP.vsv_tg_tick(now=10_100, send=lambda t: sent.append(t) or True) == []
        assert len(sent) == 1, 'тумблер вимкнено — нічого не шлемо'
    finally:
        IP.recommended_payload = old
        IP._vsv_tg['set'] = None
    src = open(os.path.join(ROOT, 'web', 'info_portal.py'), encoding='utf-8').read()
    assert "notify_category('sbr'" in src and "'/api/vsv-signals/settings'" in src
    assert "notify_category_animation('sbr'" not in src, 'VSV SIGNALS — компактний текст без GIF'
    assert "VSV_TG_SET_KEY" in src and "{'enabled': True, 'grade': 'A'}" in src, 'дефолт — A, увімкнено'
    html = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    assert 'id="vsv-tg"' in html and 'function vsvTgSave' in html and '/api/vsv-signals/settings' in html


def test_dblclick_fullscreen_chart_and_vsv_row():
    assert 'function chartDblFree(' in APP and 'setFull(!isFull())' in APP
    f = APP[APP.index('function chartDblFree('):APP.index('function recOpenOnChart(')]
    assert 'hitAt' in f and 'VOL_TOP' in f and '#iw-sig' in f, 'позначки, Σ-зона і картка — не чистий простір'
    assert '" data-sym="' in APP and 'esc(r.symbol)' in APP
    assert '#rec-tbl tr.rec-row[data-sym]' in APP and 'recOpenOnChart(tr.getAttribute("data-sym"))' in APP
    r = APP[APP.index('function recOpenOnChart('):APP.index('function setFull(')]
    assert 'IW.view = sym' in r and 'setFull(true)' in r and 'saveMine({ sel: sym })' in r
    sf = APP[APP.index('function setFull('):APP.index('function isFull(')]
    assert 'IW.view = ""' in sf, 'вихід із повного екрана повертає вашу монету'
    assert 'function curSym()' in APP and 'var sym = curSym();\n    if (!sym) { setNote(' in APP


def test_menu_cabinet_and_anom_telegram_label():
    assert 'id="nav-cabinet"' in IDX and 'Мій кабінет' in IDX
    assert 'nc.href = botUrl("/cabinet?from=info")' in APP
    assert '.main-nav a[hidden]{display:none}' in CSS
    assert '⚠ Аномалії обсягу у Telegram' in APP


VD = IP._vd()


def test_direction_combines_evidence_and_never_100():
    rec = _rec(80, prob_long=75, move24_pct=2.0)
    adir = {'state': 'dir', 'up': 0.9, 'dn': 0.1}
    h = VD.direction(rec, adir, {'ready': True, 'side': 'LONG', 'confidence': 70})
    assert h['side'] == 'LONG' and 50 < h['prob'] <= VD.P_MAX and h['agree'] == 4 and h['against'] == 0
    # джерела суперечать → ⚖
    h2 = VD.direction(_rec(80, prob_long=85), {'state': 'dir', 'up': 0.0, 'dn': 0.56})
    assert h2['side'] is None and '⚖' in h2['why']
    # без натягу і без order flow — не вгадуємо
    h3 = VD.direction({'state': 'flat', 'move24_pct': 4.0})
    assert h3['side'] is None and 'недостатньо' in h3['why']
    # імпульс уже пішов — його бік
    h4 = VD.direction(_rec(30, state='fired', fired={'side': 'SHORT'}, prob_long=80))
    assert h4['side'] == 'SHORT'
    # mixed order flow ослаблює, а не голосує
    hm = VD.direction(_rec(80, prob_long=75), {'state': 'mixed', 'up': 0.9, 'dn': 0.8})
    assert hm['prob'] < VD.direction(_rec(80, prob_long=75), {'state': 'dir', 'up': 0.9, 'dn': 0.1})['prob']


def test_journal_verifies_direction_by_market():
    rows = [{'symbol': 'AAAUSDT', 'price': 100, 'amp_pct': 4.0, 'hint': {'side': 'LONG', 'prob': 72}},
            {'symbol': 'BBBUSDT', 'price': 50, 'amp_pct': 1.0, 'hint': {'side': 'SHORT', 'prob': 64}},
            {'symbol': 'CCCUSDT', 'price': 10, 'hint': {'side': None}}]
    j = VD.journal_open([], rows, 1000, {'AAAUSDT': 100.0, 'BBBUSDT': 50.0, 'CCCUSDT': 10.0})
    assert [e['S'] for e in j] == ['AAAUSDT', 'BBBUSDT'] and j[0]['th'] == 2.0 and j[1]['th'] == 1.0
    assert VD.journal_open([], rows, 1000, {}) == [], 'без живої ціни запис не відкривається'
    j = VD.journal_open(j, rows, 1100, {'AAAUSDT': 100.0, 'BBBUSDT': 50.0})
    assert len(j) == 2, 'той самий бік монети — один відкритий запис'
    j = VD.journal_update(j, {'AAAUSDT': 102.1, 'BBBUSDT': 50.6}, 2000)
    assert j[0]['st'] == 'win' and j[1]['st'] == 'loss', 'LONG +2.1% ✅, SHORT проти −1.2% ❌'
    j2 = VD.journal_open([], rows[:1], 0, {'AAAUSDT': 100.0})
    j2 = VD.journal_update(j2, {'AAAUSDT': 100.5}, VD.EVAL_SEC + 1)
    assert j2[0]['st'] == 'flat'
    acc = VD.accuracy(j + j2)
    assert acc['n'] == 2 and acc['acc'] == 50.0 and acc['flat'] == 1
    assert acc['buckets']['70-80']['win'] == 1 and acc['buckets']['60-70']['loss'] == 1


def test_row_accuracy_honest_until_enough_data():
    few = [{'st': 'win', 'prob': 72}] * 5
    assert VD.row_accuracy(VD.accuracy(few), 72) == {'acc': None, 'n': 5, 'scope': None, 'need': VD.MIN_N}
    j = [{'st': 'win', 'prob': 72}] * 8 + [{'st': 'loss', 'prob': 72}] * 4 + [{'st': 'loss', 'prob': 55}] * 10
    a = VD.accuracy(j)
    assert VD.row_accuracy(a, 75)['scope'] == 'bucket' and VD.row_accuracy(a, 75)['acc'] == 66.7
    r = VD.row_accuracy(a, 66)
    assert r['scope'] == 'all' and r['n'] == 22 and r['acc'] == round(800 / 22, 1)


def test_direction_can_flip_and_is_marked():
    st = {}
    st['X'] = VD.track_flip(st, 'X', 'LONG', 100)
    assert st['X']['prev'] is None
    st['X'] = VD.track_flip(st, 'X', 'LONG', 200)
    assert st['X']['since'] == 100
    st['X'] = VD.track_flip(st, 'X', None, 300)          # ⚖
    assert st['X']['prev'] is None
    st['X'] = VD.track_flip(st, 'X', 'SHORT', 400)       # LONG → ⚖ → SHORT = розворот
    assert st['X']['prev'] == 'LONG' and st['X']['flip_t'] == 400
    v = VD.flip_view(st['X'], 400 + 1800)
    assert v == {'prev': 'LONG', 't': 400, 'ago_min': 30}
    assert VD.flip_view(st['X'], 400 + VD.FLIP_SHOW_SEC + 1) is None


def test_rows_and_page_show_probable_direction():
    rows, _ = IP.vsv_signal_rows({'AAAUSDT': _rec(92, sq=20, rank=8, vr=0.6)}, [], {}, 'binance',
                                 adir_map={'AAAUSDT': {'state': 'dir', 'up': 0.9, 'dn': 0.0}})
    h = rows[0]['hint']
    assert h['side'] == 'LONG' and h['parts'] and 'prob_long' not in rows[0]
    old = (IP._vsv_dir['j'], IP._vsv_dir['flip'], IP._vsv_dir['acc'])
    try:
        IP._vsv_dir.update(j=[{'st': 'win', 'prob': 72}] * 15 + [{'st': 'loss', 'prob': 72}] * 5,
                           flip={}, acc=None)
        IP._vsv_dir['acc'] = VD.accuracy(IP._vsv_dir['j'])
        da = IP.vsv_dir_attach(rows, 5000)
        assert da['acc'] == 75.0 and rows[0]['hint']['acc']['acc'] == 75.0
    finally:
        IP._vsv_dir['j'], IP._vsv_dir['flip'], IP._vsv_dir['acc'] = old
    assert 'function recDirCell(' in APP and 'recDirCell(r)' in APP and 'rec-flip' in APP
    assert '>Ймовірний напрямок</th>' in IDX and 'colspan="9" class="muted">Завантаження' in IDX
    assert '.rec-dir-s.l' in CSS
    src = open(os.path.join(ROOT, 'web', 'info_portal.py'), encoding='utf-8').read()
    loop = src[src.index('def _vsv_tg_loop'):src.index('def vsv_tg_start')]
    assert 'vsv_dir_tick()' in loop, 'журнал веде потік VSV незалежно від тумблера Telegram'


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f(); n += 1; print('ok', k)
    print(f'OK — {n} тестів')
