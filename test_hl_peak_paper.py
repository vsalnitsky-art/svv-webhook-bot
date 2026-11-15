"""📈 HH/HL/LH/LL у кабінеті · 📈 Пік у закритих угодах · 🧪 paper-таблиця інфосайту як у боті (15.11)."""
import importlib.util, json, os, re, subprocess, sys, types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
if 'detection' not in sys.modules:   # пакет без __init__ (той тягне pybit)
    _pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_HERE, 'detection')]
    sys.modules['detection'] = _pkg
_spec = importlib.util.spec_from_file_location('info_portal', os.path.join(_HERE, 'web', 'info_portal.py'))
_IPM = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_IPM)
APP = open('infosite/js/app.js', encoding='utf-8').read()
SM = open('templates/smart_money.html', encoding='utf-8').read()
IP = open('web/info_portal.py', encoding='utf-8').read()


def _block(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


def _node(code):
    r = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _ip():
    return _IPM


def test_hl_points_types_and_colors():
    ip = _ip()
    pts = ip.hl_points([
        {'t': 1_700_003_000_000, 'price': 9, 'type': 'LL'},
        {'t': 1_700_001_000_000, 'price': 10, 'type': 'HH'},
        {'t': 1_700_002_000_000, 'price': 8, 'type': 'HL'},
        {'t': 1_700_002_500_000, 'price': 11, 'type': 'LH'},
        {'t': 1_700_004_000_000, 'price': 1, 'type': 'XX'},
    ])
    assert [p['type'] for p in pts] == ['HH', 'HL', 'LH', 'LL']          # за часом
    assert [p['dir'] for p in pts] == ['bull', 'bull', 'bear', 'bear']
    assert [p['high'] for p in pts] == [True, False, True, False]
    assert pts[0]['t'] == 1_700_001_000                                            # мс → с
    assert len(ip.hl_points([{'t': i * 1000, 'price': 1, 'type': 'HH'} for i in range(1, 100)], n=5)) == 5


def test_hl_tf_pref_default_off_and_change():
    ip = _ip()
    u = ip.user_info({'info': {'wl': ['BTCUSDT']}}, [])
    assert u['hl_tf'] == 'off'
    ch = ip.apply_change(u, {'hl_tf': '4h'}, [])[0]
    assert ch['hl_tf'] == '4h'
    assert ip.apply_change(ch, {'hl_tf': '7m'}, [])[0]['hl_tf'] == '4h'   # невідоме → лишаємо
    assert ip.apply_change(ch, {'hl_tf': 'off'}, [])[0]['hl_tf'] == 'off'


def test_hl_compute_uses_swing_pivots_of_closed_bars():
    ip = _ip()
    import detection.smc_structure as ss
    seen = {}

    def fake_detect(kl, internal_size=5, swing_size=50):
        seen['n'] = len(kl); seen['sw'] = swing_size
        return {'swing': {'pivots': [{'t': 1_700_005_000_000, 'price': 2, 'type': 'HH'}]},
                'internal': {'pivots': [{'t': 6_000_000, 'price': 3, 'type': 'LL'}]}}

    md = types.SimpleNamespace(fetch_klines=lambda sym, limit, interval: [{'t': i} for i in range(100)])
    sc = types.SimpleNamespace(_settings={'swing_size': 50}, get_internal_size=lambda: 5)
    old = ss.detect_smc_structure
    mdm = types.ModuleType('detection.market_data'); mdm.get_market_data = lambda: md
    keep = sys.modules.get('detection.market_data')
    sys.modules['detection.market_data'] = mdm
    ss.detect_smc_structure = fake_detect
    try:
        data, until = ip._compute(sc, 'hl', 'BTCUSDT', '1h')
    finally:
        ss.detect_smc_structure = old
        if keep is not None:
            sys.modules['detection.market_data'] = keep
    assert seen == {'n': 99, 'sw': 50}                                    # живий бар відкинуто
    assert data['pts'] == [{'t': 1_700_005_000, 'price': 2.0, 'type': 'HH', 'high': True, 'dir': 'bull'}]
    assert until > 0


def test_hl_wired_into_payload_and_page():
    assert "'hl_tf': hl_tf, 'hl': hl, 'hl_wait': hl_wait" in IP
    assert "('hl', hl_tf)" in IP and "hl_tf=a.get('hl_tf')" in IP
    assert '"&hl_tf=" + IW.hl_tf' in APP and 'hlCtl() +' in APP
    lay = _block(APP, 'var lay = $("#iw-layers");', 'liqUi();')
    assert lay.index('show_imb') < lay.index('hlCtl()')                    # кнопка поруч з IMB
    fns = (_block(APP, 'function snapSec(', '\n  }\n') + '\n  }\n'
           + _block(APP, 'function structOf(', 'function setShl('))
    out = _node(fns + """
      var c = {ohlc: [{time: 100}, {time: 200}, {time: 300}], events: [],
               hl: [{t: 50, type: 'HH', high: true, dir: 'bull'}, {t: 250, type: 'LL', high: false, dir: 'bear'},
                    {t: 300, type: 'HL', high: false, dir: 'bull'}]};
      console.log(JSON.stringify(structOf(c).marks));""")
    assert [m['text'] for m in out] == ['LL', 'HL']                        # старіші за графік — ні
    assert out[0]['time'] == 200 and out[0]['position'] == 'belowBar' and out[0]['color'] == '#ef4444'
    assert out[1]['color'] == '#22c55e'


def test_bot_closed_tables_have_peak():
    assert SM.count('>Пік</th>') == 2
    assert SM.count('${peakCellHTML(c)}') == 3                             # real · paper · архів
    fn = _block(SM, 'function closedPeakPct(', 'function closedRealPnlCellHTML(')
    out = _node(fn + """
      console.log(JSON.stringify([peakCellHTML({peak_pnl_pct: 3.2, pnl_pct: 1.0}),
                                  peakCellHTML({peak_pnl_pct: 0.5, pnl_pct: 1.4}),
                                  peakCellHTML({pnl_pct: 1})]));""")
    assert '+3.20%' in out[0] and '2.20 п.п.' in out[0] and '#20b26c' in out[0]
    assert '+1.40%' in out[1]                                             # пік не менший за результат
    assert '>—<' in out[2]


def test_infosite_paper_like_bot():
    ip = _ip()
    for k in ('ob4h', 'ob1h', 'manual_mode', 'manual_sl_tf', 'sbr_trail', 'sl_breakeven'):
        assert k in ip._PAPER_POS_KEYS, k
    assert 'peak_pnl_pct' in ip._PAPER_CLOSED_KEYS
    assert "sc.ob_mark_for(syms, tf='1h')" in IP
    row = _block(APP, 'function ppRenderOpen(', 'var PP_REASON')
    assert 'ppObMark(p, "4h") + ppObMark(p, "1h")' in row and 'ppPnlCell(p)' in row and 'ppAlertCell(p)' in row
    assert 'ppPeak(c)' in APP and '>Пік</th>' in open('infosite/index.html', encoding='utf-8').read()
    fns = _block(APP, 'function ppPnlCell(', 'var PP_PIN')
    out = _node("""
      var PP = {shown: true}; function esc(x){return String(x);} function fmtPrice(v){return '$' + v;}
      function locDM(){return '15.11';} function locHM(){return '10:00';}
      var PP_ALERT = { ok: ["✓ OK", "ok"], against: ["⚠ ПРОТИ", "against"] };
      function ppAmt(h){return h;} function ppUsd(v){v=Number(v||0);return (v>=0?'+':'−')+'$'+Math.abs(v).toFixed(2);}
      function ppCls(v){v=Number(v||0);return v>0?'pp-up':(v<0?'pp-dn':'');}
      function ppPct(v){v=Number(v||0);return (v>=0?'+':'−')+Math.abs(v).toFixed(2)+'%';}
      """ + fns + """
      var p = {side: 'LONG', pnl_pct: 1.5, pnl_net: {net_usd: 3.1, roi_pct: 15.2, fees_usd: .4}, paper_size: {lev: 10, margin: 20},
               ob4h: {bias: 'BULLISH'}, ob1h: {bias: 'BEARISH'}};
      console.log(JSON.stringify([ppPnlCell(p), ppObMark(p, '4h') + ppObMark(p, '1h'),
        ppAlertCell({mmd: {alert: {level: 'pre', side: 'SHORT', tf: '4H', text: 'x'}}}),
        ppAlertCell({mmd: {alert: {level: 'none', text: 'немає'}}}),
        ppAlertCell({side: 'LONG', current_price: 102, sbr_trail: {armed_px: 100, stop_px: 101, tf: '1H'}})]));""")
    assert '+1.50%' in out[0] and '+$3.10' in out[0] and 'ROI +15.2% ×10' in out[0] and '<br>' in out[0]
    assert '4H▲' in out[1] and '1H▼' in out[1] and '#fbbf24' in out[1]   # 1H проти угоди — бурштин
    assert '▼ SHORT 4H' in out[2] and '—' not in out[2]
    assert 'без сигналу' in out[3]
    assert '🔔⏳ 1H' in out[4] and '+2.00%' in out[4]


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); n += 1; print('✓', k)
    print(f'OK — {n} тестів')
