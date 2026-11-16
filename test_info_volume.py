"""📊 Обсяг на графіку інфосайту: покупці / продавці в кожній свічці (16.11).

Історія — підсвічки Bybit (сервер, поступове заповнення), жива свічка —
справжні угоди publicTrade у браузері. Малює рідна серія графіка."""
import importlib.util, json, re, subprocess
from test_hl_peak_paper import _ip

ip = _ip()
_im = importlib.util.spec_from_file_location('_imb_vol_t', 'detection/imbalance.py')
_imb = importlib.util.module_from_spec(_im); _im.loader.exec_module(_imb)
pine_sub_minutes, split_from_subbars = _imb.pine_sub_minutes, _imb.split_from_subbars

APP = open('infosite/js/app.js', encoding='utf-8').read()
TOOLS = open('infosite/js/chart-tools.js', encoding='utf-8').read()


def _block(src, start, end):
    i = src.index(start)
    return src[i:src.index(end, i)]


def _node(code):
    r = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


# ── сервер ────────────────────────────────────────────────────────────────
def test_ohlc_carries_coin_volume():
    o = ip.ohlc_of([{'t': 1000_000, 'o': 10, 'h': 12, 'l': 9, 'p': 11, 'q': 55.5, 'v': 600},
                    {'t': 1900_000, 'o': 11, 'h': 12, 'l': 10, 'p': 10, 'v': 500}])
    assert o[0]['volume'] == 55.5                 # поле біржі в монетах
    assert o[1]['volume'] == 50.0                 # немає q → обіг / ціна
    assert ip.bar_volume({}) == 0.0


def test_sub_plan_matches_pine_up_to_1h():
    for tf in ('15m', '30m', '1h'):
        sub, base = ip.vol_sub_plan(tf)
        assert sub == pine_sub_minutes(ip.bar_sec(tf)) and sub % base == 0
    assert ip.vol_sub_plan('15m') == (2, 1) and ip.vol_sub_plan('1h') == (6, 3)
    # старші TF — крупніша підсвічка: перше заповнення ≤ ~10 запитів
    for tf in ('4h', '1d'):
        sub, base = ip.vol_sub_plan(tf)
        assert ip.OHLC_BARS * (ip.bar_sec(tf) // 60 // base) <= 10 * ip.VOL_PAGE


def _mk_fetch(tf, base, calls, first_ms=0):
    """Біржа: бар base хв у кожну мить; зелений, якщо індекс парний. Обсяг 1."""
    step = base * 60000

    def fetch(b, end_ms):
        assert b == base
        calls.append(end_ms)
        last = end_ms // step * step
        out = []
        for i in range(ip.VOL_PAGE):
            t = last - i * step
            if t < first_ms:
                break
            up = (t // step) % 2 == 0
            out.append({'t': t, 'o': 10.0, 'p': 11.0 if up else 9.0, 'v': 1.0})
        return out
    return fetch


def test_split_is_the_imb_function():
    tf = '1h'
    sub, base = ip.vol_sub_plan(tf)
    t = 1_700_000_000 // 3600 * 3600
    rows = [{'t': t * 1000 + i * base * 60000, 'o': 10, 'p': 11 if i % 3 else 9, 'v': 2.0 + i}
            for i in range(3600 // 60 // base)]
    got = ip.vol_split_rows(rows, tf, sub, base)[t]
    exp = split_from_subbars(ip.group_subbars(rows, base, sub, t * 1000))
    assert got == (ip._sig6(exp[0]), ip._sig6(exp[1])) and got[0] > 0 and got[1] > 0


def test_step_fills_newest_first_and_never_refetches_closed():
    tf = '1h'
    sub, base = ip.vol_sub_plan(tf)
    now = 1_700_000_000 + 1234.0
    calls = []
    f = _mk_fetch(tf, base, calls)
    d = ip.vol_step(None, tf, now, f)
    t_live = int(now // 3600) * 3600
    assert len(calls) == ip.VOL_PAGES_PER_PASS and not d['done']
    assert d['live']['t'] == t_live and d['live']['asof'] == int(now * 1000)
    assert d['hi'] == t_live and max(d['sp']) == t_live - 3600        # найновіші закриті — першими
    assert min(d['sp']) == d['lo'] > t_live - (ip.OHLC_BARS - 1) * 3600
    n = 0
    while not d['done']:
        calls.clear()
        d = ip.vol_step(d, tf, now, f)
        assert len(calls) <= ip.VOL_PAGES_PER_PASS
        n += 1
        assert n < 20
    assert len(d['sp']) == ip.OHLC_BARS - 1 and d['lo'] == t_live - (ip.OHLC_BARS - 1) * 3600
    # кожна закрита свічка: 20 базових барів по 3 хв → 10 підсвічок по 6 хв
    assert all(abs(b + s - 20.0) < 1e-9 or (b + s) <= 20.0 for b, s in d['sp'].values())
    # готово → лише ОДИН запит (жива свічка), закриті не перезапитуються
    calls.clear()
    d2 = ip.vol_step(d, tf, now + 30, f)
    assert len(calls) == 1 and d2['done'] and d2['sp'] == d['sp']
    # закрилась нова свічка → вона додається, найстаріша випадає
    calls.clear()
    d3 = ip.vol_step(d2, tf, now + 3600, f)
    assert len(calls) == 1 and t_live in d3['sp'] and d3['live']['t'] == t_live + 3600
    assert min(d3['sp']) == t_live + 3600 - (ip.OHLC_BARS - 1) * 3600


def test_step_short_history_and_empty_exchange():
    tf = '15m'
    sub, base = ip.vol_sub_plan(tf)
    now = 1_700_000_000.0
    t_live = int(now // 900) * 900
    calls = []
    d = ip.vol_step(None, tf, now, _mk_fetch(tf, base, calls, first_ms=(t_live - 5 * 900) * 1000))
    assert d['done'] and len(calls) == 1 and sorted(d['sp']) == [t_live - i * 900 for i in range(5, 0, -1)]
    d = ip.vol_step(None, tf, now, lambda b, e: [])
    assert d['done'] and d['sp'] == {}
    assert ip.vol_view(d, [{'time': t_live}])['m'] == []


def test_view_only_chart_bars():
    data = {'sub': 6, 'lo': 100, 'hi': 400, 'done': False,
            'sp': {100: (1.0, 2.0), 200: (3.0, 1.0), 300: (0.0, 0.0)},
            'live': {'t': 400, 'b': 1, 's': 1, 'asof': 5}}
    v = ip.vol_view(data, [{'time': 200}, {'time': 300}, {'time': 400}])
    assert v['m'] == [[200, 3.0, 1.0]] and v['sub'] == 6 and v['fill'] and v['live']['t'] == 400
    assert ip.vol_view(None, []) is None and ip.vol_view({'lo': None}, []) is None


def test_payload_wiring():
    src = open('web/info_portal.py', encoding='utf-8').read()
    cp = src[src.index('def coin_payload'):src.index('# ── Особисті дані користувача')]
    assert "_vp['pk'], vol)" in cp                              # розклад — у ключі кешу відповіді
    assert "('vol', tf if vol else OFF)" in cp                  # вимкнено → частина не рахується
    assert "'vsplit': vsplit, 'vol_wait': vol_wait" in cp
    assert "vol=str(a.get('vol') or '') in ('1', 'true')" in src
    comp = src[src.index('def _compute'):src.index('def _worker')]
    assert "if kind == 'vol':" in comp and 'vol_step(' in comp
    assert "payload.get('vsplit')" in src[src.index('def sig_of'):src.index('_lock = threading.Lock()')] \
        or "payload.get('vsplit')" in src


# ── сторінка ──────────────────────────────────────────────────────────────
VOLJS = _block(APP, 'var VOL_UP =', 'function VolRenderer')


def test_tape_rules():
    out = _node("function esc(s){return String(s)} var localStorage={getItem:function(){return null}};" + VOLJS + """
      var bs = 900, t1 = 9000, t2 = 9900;
      var tp = volTapeNew('k', t1 * 1000 + 300000);            // потік із середини свічки t1
      volTapeAdd(tp, t1 * 1000 + 310000, 'Buy', 4, bs);        // до asof
      volTapeAdd(tp, t1 * 1000 + 400000, 'Buy', 6, bs);
      volTapeAdd(tp, t1 * 1000 + 400500, 'Sell', 2, bs);
      volTapeAdd(tp, t2 * 1000 + 10, 'Sell', 30, bs);          // t2 покрита з відкриття
      volTapeAdd(tp, t2 * 1000 + 20, 'Buy', 10, bs);
      volTapeAdd(tp, 0, 'Buy', 5, bs); volTapeAdd(tp, t2 * 1000, 'Buy', -1, bs);
      var base = {t: t1, b: 50, s: 50, asof: t1 * 1000 + 350000};
      var srv = {}; srv[8100] = [7, 3]; srv[t1] = [1, 1];
      var fresh = volTapeCarry(tp, volTapeNew('k', t2 * 1000 + 500000), t2 + bs);
      var cur = volTapeCarry(tp, volTapeNew('k', t2 * 1000 + 500000), t2);
      console.log(JSON.stringify({
        full: [tp.bars[t1].full, tp.bars[t2].full],
        mix: volSplit(t1, srv, base, tp), tape: volSplit(t2, srv, base, tp),
        sub: volSplit(8100, srv, base, tp), none: volSplit(7200, srv, base, tp),
        nobase: volSplit(t1, srv, null, tp), part: volSplit(t1, {}, null, tp),
        baseonly: volSplit(t1, srv, base, null),
        carry: [Object.keys(fresh.bars), Object.keys(cur.bars)],
        row: volRow({time: t2, volume: 80}, srv, base, tp),
        row0: volRow({time: 7200, volume: 5}, srv, base, tp),
        tf: [volTfSec('15m'), volTfSec('1h'), volTfSec('4h'), volTfSec('1d')]
      }));""")
    assert out['full'] == [False, True]
    assert out['tape'] == {'buy': 10, 'sell': 30, 'src': 'tape'}           # точні угоди
    assert out['mix'] == {'buy': 56, 'sell': 52, 'src': 'mix'}             # оцінка до asof + угоди після
    assert out['sub'] == {'buy': 7, 'sell': 3, 'src': 'sub'} and out['none'] is None
    assert out['nobase']['src'] == 'sub' and out['part'] == {'buy': 10, 'sell': 2, 'src': 'part'}
    assert out['baseonly'] == {'buy': 50, 'sell': 50, 'src': 'sub'}
    assert out['carry'] == [[str(9900)], []]            # розрив: закрита точна свічка лишається, поточна — ні
    assert out['row']['value'] == 80 and out['row']['buy'] == 0.25 and out['row']['src'] == 'tape'
    assert out['row0']['buy'] is None and out['row0']['value'] == 5
    assert out['tf'] == [900, 3600, 14400, 86400]


def test_bar_geometry_and_width():
    out = _node("function esc(s){return String(s)} var localStorage={getItem:function(){return null}};" + VOLJS + """
      var g = [volBarGeom(500, 400, 0.64), volBarGeom(500, 400, null), volBarGeom(500, 497, 0.5),
               volBarGeom(500, 499, 0.9), volBarGeom(500, 400, 0), volBarGeom(500, 400, 1), volBarGeom(500, 400, 0.001)];
      var w = []; for (var s = 0.5; s <= 40; s += 0.5) w.push(volBarW(s, 1), volBarW(s, 2));
      console.log(JSON.stringify({g: g, wmin: Math.min.apply(null, w), mono: volBarW(30, 1) > volBarW(6, 1), le: volBarW(10, 1) <= 10}));""")
    g = out['g']
    assert g[0] == {'h': 100, 'buy': 64, 'sell': 35, 'gap': 1}            # покупці знизу, 1px проміжок
    assert g[1] == {'h': 100, 'buy': 0, 'sell': 100, 'gap': 0}
    for x in g:
        assert x['buy'] + x['sell'] + x['gap'] == x['h'] and x['buy'] >= 0 and x['sell'] >= 0
    assert g[4]['buy'] == 0 and g[5]['sell'] == 0 and g[6]['buy'] == 1    # мала частка не зникає
    assert out['wmin'] >= 1 and out['mono'] and out['le']


def test_legend_text():
    out = _node("function esc(s){return String(s)} var localStorage={getItem:function(){return null}};" +
                _block(APP, 'function fmtVol', 'function pdBadge') + VOLJS + _block(APP, 'var VOL_SRC', 'function volLegend()') + """
      console.log(JSON.stringify({
        a: volLegendHTML({value: 8952, buy: 0.36, src: 'tape'}, 6, ''),
        b: volLegendHTML({value: 100, buy: 0.7, src: 'sub'}, 6, ''),
        c: volLegendHTML({value: 100, buy: null, src: ''}, 6, 'чекаємо'),
        d: volLegendHTML(null, 6, '')}));""")
    a = re.sub(r'<[^>]+>', '', out['a'])
    assert 'Обсяг 8.952K' in a and 'покупці 36%' in a and 'продавці 64%' in a and 'Δ −' in a and '⚡ угоди' in a
    assert 'vl-dn vl-win' in out['a'] and 'vl-up vl-win' in out['b'] and 'підсвічки 6 хв' in out['b']
    assert 'чекаємо' in out['c'] and 'покупці' not in out['c'] and out['d'] == ''


def test_live_socket_also_streams_trades():
    live = _block(APP, 'var LIVE_WS_URL', 'LiveKline.prototype.ok')
    out = _node("""
      var socks = [];
      function WebSocket(u) { this.sent = []; socks.push(this); }
      WebSocket.prototype.send = function (x) { this.sent.push(x); }; WebSocket.prototype.close = function () {};
      function setInterval() { return 1; } function clearInterval() {}
      """ + live + """
      var bars = [], tr = [];
      var L = new LiveKline(function (k, b) { bars.push(b); }, function (k, d, t0) { tr.push([k, d.length, t0]); });
      L.set('BTCUSDT', '1h', true); socks[0].onopen();
      socks[0].onmessage({data: JSON.stringify({topic: 'kline.60.BTCUSDT', ts: 111, data: [{start: 3600000, open: '1', high: '2', low: '1', close: '2', volume: '12.5'}]})});
      socks[0].onmessage({data: JSON.stringify({topic: 'publicTrade.BTCUSDT', ts: 222, data: [{T: 1, S: 'Buy', v: '1'}, {T: 2, S: 'Sell', v: '2'}]})});
      socks[0].onmessage({data: JSON.stringify({topic: 'publicTrade.ETHUSDT', ts: 222, data: [{T: 1}]})});
      L.set('BTCUSDT', '1h', true); var same = socks.length;
      L.set('BTCUSDT', '1h', false); socks[1].onopen();
      console.log(JSON.stringify({a: JSON.parse(socks[0].sent[0]).args, b: JSON.parse(socks[1].sent[0]).args,
        vol: bars[0].volume, tr: tr, same: same, n: socks.length}));""")
    assert out['a'] == ['kline.60.BTCUSDT', 'publicTrade.BTCUSDT'] and out['b'] == ['kline.60.BTCUSDT']
    assert out['vol'] == 12.5
    assert out['tr'] == [['publicTrade.BTCUSDT', 2, 111]]        # t0 — час БІРЖІ першого повідомлення
    assert out['same'] == 1 and out['n'] == 2                    # ті самі параметри → без перепідключення


def test_native_series_and_wiring():
    # рідна серія графіка: перемальовування при прокрутці / масштабі веде бібліотека
    assert 'IW.chart.addCustomSeries(new VolPane(), opt)' in APP
    assert 'VolPane.prototype.priceValueBuilder' in APP and 'useBitmapCoordinateSpace' in _block(APP, 'VolRenderer.prototype.draw', 'function VolPane')
    assert 'customSeriesDefaultOptions' in open('infosite/js/vendor/lightweight-charts.js', encoding='utf-8').read()
    draw = _block(APP, 'function drawCoin', 'function loadCoin')
    assert 'volApply(c)' in draw
    load = _block(APP, 'function loadCoin', 'document.addEventListener')
    assert '(VOL.on ? "&vol=1" : "")' in load and 'c.vol_wait' in load
    assert 'volume: volToggle' in APP
    # chart-tools більше не малює свій (однокольоровий) обсяг
    assert 'window.VSV_HOOK.volume(!!T.ind.volume)' in TOOLS and 'T.series.volume' not in TOOLS
    # вимкнули → потік угод зупинено, стрічка скинута
    tg = _block(APP, 'function volToggle', 'function drawCoin')
    assert 'VOL.tape = null' in tg and 'liveSet(IW.coin)' in tg
    assert 'IW.live.set(c && c.symbol, c && c.tf, VOL.on)' in APP
    css = open('infosite/css/style.css', encoding='utf-8').read()
    assert '.iw-vol-lg{' in css and 'pointer-events:none' in css[css.index('.iw-vol-lg{'):]


if __name__ == '__main__':
    n = 0
    for k, f in sorted(globals().items()):
        if k.startswith('test_') and callable(f):
            f(); n += 1
    print(f'OK {n}')
