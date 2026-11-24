"""🛰 VSV SIGNALS (вимога 24.11): радар волатильності ОБОХ Пружин БЕЗ напрямку.

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
    for w in ('squeeze', 'volatility expansion', 'range breakout', 'Напрямок свідомо не вказується'):
        assert w in info, w
    tbl = IDX[IDX.index('id="rec-tbl"'):IDX.index('</table>', IDX.index('id="rec-tbl"'))]
    for gone in ('LONG', 'SHORT', '1H · OB 1H', 'OB 15m', 'SL', 'ЦІЛЬ'):
        assert gone not in tbl, gone
    blk = APP[APP.index('var REC = '):APP.index('function recRender()')]
    assert 'r.side' not in blk and 'pp-long' not in blk and 'pp-short' not in blk
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


def test_tg_text_has_no_direction():
    rows, _ = IP.vsv_signal_rows({'AAAUSDT': _rec(92, sq=20, rank=8, vr=0.6)}, [], {}, 'binance')
    t = IP.vsv_tg_text(rows)
    assert '#AAAUSDT' in t and 'VSV' in t and 'Напрямок не вказується' in t
    for w in ('LONG', 'SHORT', '🟢', '🔴', 'SL', 'TP'):
        assert w not in t, w


def test_tg_tick_sends_new_coins_to_signals_theme():
    sent = []
    IP._vsv_tg.update({'boot': 0, 'set': {'enabled': True}, 'st': {'init': True, 'in': ['AAAUSDT'], 'sent': {}}})
    rows = [{'symbol': 'AAAUSDT', 'score': 70}, {'symbol': 'BBBUSDT', 'score': 66}]
    old = IP.recommended_payload
    try:
        IP.recommended_payload = lambda now=None: {'rows': rows}
        assert IP.vsv_tg_tick(now=10_000, send=lambda t: sent.append(t) or True) == ['BBBUSDT']
        assert len(sent) == 1 and '#BBBUSDT' in sent[0] and '#AAAUSDT' not in sent[0]
        IP._vsv_tg['set'] = {'enabled': False}
        rows.append({'symbol': 'CCCUSDT', 'score': 66})
        assert IP.vsv_tg_tick(now=10_100, send=lambda t: sent.append(t) or True) == []
        assert len(sent) == 1, 'тумблер вимкнено — нічого не шлемо'
    finally:
        IP.recommended_payload = old
        IP._vsv_tg['set'] = None
    src = open(os.path.join(ROOT, 'web', 'info_portal.py'), encoding='utf-8').read()
    assert "notify_category('sbr'" in src and "'/api/vsv-signals/settings'" in src
    assert "VSV_TG_SET_KEY" in src and "{'enabled': True}" in src, 'дефолт — увімкнено'
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


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f(); n += 1; print('ok', k)
    print(f'OK — {n} тестів')
