"""Причини закриття на інфосайті · 🎯 «Досягнення вершини» % · 🏷 бейдж угоди."""
import json, re, subprocess
SM = open('templates/smart_money.html', encoding='utf-8').read()
APP = open('infosite/js/app.js', encoding='utf-8').read()
TM = open('detection/trade_manager.py', encoding='utf-8').read()


def _node(js):
    return json.loads(subprocess.run(['node', '-e', js], capture_output=True, text=True, check=True).stdout)


def _fn(src, name):
    i = src.index('function ' + name + '(')
    j = src.index('\n}', i) + 2
    return src[i:j]


def test_infosite_explains_every_reason_the_bot_knows():
    bot = set(re.findall(r"'(\w+)': '", _fn(SM, 'reasonLabel')))
    i = APP.index('var PP_REASON = {'); site = set(re.findall(r'(\w+): "', APP[i:APP.index('\n', i)]))
    assert bot <= site, bot - site
    assert 'reverse_signal' in site and 'exh_exit' in site
    assert 'ppReason(c) + "</td></tr>"' in APP


def test_culmination_is_renamed_everywhere():
    assert 'Кульмінація' not in _fn(SM, 'reasonLabel')
    assert "'exh_exit': '🎯 Досягнення вершини'" in SM and "'exh_exit': '🎯 Досягнення вершини'" in TM
    assert 'exh_exit: "🎯 Досягнення вершини"' in APP


def test_peak_share_is_result_over_peak():
    js = _fn(SM, 'closedPeakPct') + _fn(SM, 'peakShareOf') + """
    console.log(JSON.stringify([
      peakShareOf({reason:'exh_exit', pnl_pct:3.43, peak_pnl_pct:4.55}),
      peakShareOf({reason:'exh_exit', pnl_pct:-1, peak_pnl_pct:2}),
      peakShareOf({reason:'exh_exit', pnl_pct:0, peak_pnl_pct:0}),
      peakShareOf({reason:'manual', pnl_pct:3, peak_pnl_pct:4}),
      peakShareOf({reason:'exh_exit', pnl_pct:5, peak_pnl_pct:4})]));"""
    assert _node(js) == [75, 0, None, None, 100]
    i = APP.index('function ppPeakShare('); j = APP.index('function ppReason(', i)
    out = _node(APP[i:j] + "console.log(JSON.stringify([ppPeakShare({reason:'exh_exit',pnl_pct:3.43,peak_pnl_pct:4.55}),ppPeakShare({reason:'manual',pnl_pct:1,peak_pnl_pct:2})]));")
    assert out == [75, None]                      # бот і інфосайт — одне число


def test_badge_link_and_data():
    assert 'onclick="tmBadgeOpen(${i});return false;"' in SM and 'window._tmClosedRows = closed;' in SM
    js = _fn(SM, 'tmBadgeData') + """
    console.log(JSON.stringify([
      tmBadgeData({symbol:'BTCUSDT', side:'SHORT', pnl_pct:0.44, entry_price:64600, exit_price:64315,
                   exchange:{roi_pct:12.47, leverage:29, entry_price:64600, exit_price:64315.1}}),
      tmBadgeData({symbol:'ETHUSDT', side:'LONG', pnl_pct:-1.2, entry_price:10, exit_price:9.88})]));"""
    a, b = _node(js)
    assert (a['label'], a['val'], a['lev'], a['exit']) == ('ROI', 12.47, 29, 64315.1)
    assert (b['label'], b['val'], b['lev']) == ('Рух ціни', -1.2, None)   # без факту біржі ROI не вигадуємо


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); n += 1
    print('OK', n)
