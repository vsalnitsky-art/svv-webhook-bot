"""⚡ Жива свічка (WebSocket Bybit) у кабінеті й графіку бота + обсяг IMB під смугами (15.11)."""
import json, re, subprocess

APP = open('infosite/js/app.js', encoding='utf-8').read()
SM = open('templates/smart_money.html', encoding='utf-8').read()


def _block(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


def _node(code):
    r = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


LIVE = _block(APP, 'var LIVE_WS_URL', 'LiveKline.prototype.ok')


def test_kline_parse_and_interval_map():
    out = _node(LIVE + """
      console.log(JSON.stringify({
        iv: [liveIv('15m'), liveIv('1h'), liveIv('4h'), liveIv('1d'), liveIv('7x')],
        b: liveBarOf({start: 1763200800000, open: '95000', high: '95500', low: '94800', close: '95210.5', confirm: false}),
        bad: liveBarOf({start: 0, close: '1'})
      }));""")
    assert out['iv'] == ['15', '60', '240', 'D', None]
    assert out['b'] == {'time': 1763200800, 'open': 95000, 'high': 95500, 'low': 94800, 'close': 95210.5, 'confirm': False}
    assert out['bad'] is None


def test_live_bar_updates_same_bar_and_appends_new():
    fn = _block(APP, 'function liveKeyOf', 'function drawCoin')
    out = _node(LIVE + "LiveKline.prototype.ok=function(){return true};" + """
      var calls = [];
      var IW = {series: {update: function (b) { calls.push(b); }}, coin: {symbol: 'BTCUSDT', tf: '15m',
               ohlc: [{time: 100, open: 1, high: 2, low: 1, close: 1.5}]}};
      function renderHead() {} function renderList() {}
      """ + fn + """
      var k = liveKeyOf('BTCUSDT', '15m');
      onLiveBar('kline.60.BTCUSDT', {time: 100, open: 1, high: 9, low: 1, close: 9});  // чужий TF — ігнор
      onLiveBar(k, {time: 100, open: 1, high: 2.2, low: 0.9, close: 2.1});
      onLiveBar(k, {time: 99, open: 1, high: 1, low: 1, close: 1});                  // старий — ігнор
      onLiveBar(k, {time: 1000, open: 2.1, high: 2.3, low: 2.0, close: 2.2});
      var c = {symbol: 'BTCUSDT', tf: '15m', ohlc: [{time: 100, open: 1, high: 2, low: 1, close: 1.5}]};
      IW.live = new LiveKline(function(){});
      var m = liveMerge(c);
      console.log(JSON.stringify({n: calls.length, last: IW.coin.ohlc, merge: m.ahead, srv: c.ohlc.length}));""")
    assert out['n'] == 2
    assert out['last'][0]['close'] == 2.1 and out['last'][1]['time'] == 1000
    assert out['merge'] is True and out['srv'] == 2   # сервер ще не бачив новий бар → дописано


def test_wired_into_draw_and_bot_chart():
    draw = _block(APP, 'function drawCoin', 'function loadCoin')
    assert 'liveMerge(c)' in draw and 'liveSet(c)' in draw
    assert 'IW.liveBar = null' in _block(APP, 'function loadCoin', 'document.addEventListener')
    assert 'wss://stream.bybit.com/v5/public/linear' in SM
    lc = _block(SM, 'async function loadChart', '// 🏁 Strong High')
    assert '_smLiveMerge(ohlc, d)' in lc and '_smLive.set(currentSymbol' in lc
    assert "'tf': self.get_timeframe()" in open('detection/smc_scanner.py', encoding='utf-8').read()
    # дзеркало: той самий розбір свічки в обох файлах
    assert re.sub(r'\s+', '', _block(SM, 'function liveBarOf', 'function LiveKline')) == \
        re.sub(r'\s+', '', _block(APP, 'function liveBarOf', 'function LiveKline'))


def test_ha_live_update_uses_previous_bar():
    pro = open('infosite/js/chart-pro.js', encoding='utf-8').read()
    assert 'PRO.prevHA' in pro and 'PRO.lastHA.time === bar.time' in pro


def test_imb_volume_drawn_behind_bars():
    body = _block(APP, 'function _imbBody', 'function _imbVol')
    assert body.index('_imbVol(ctx') < body.index('_imbBar(ctx')


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v(); print('ok', k)
