"""Integration checks: professional brain -> existing AI gates -> Trade Manager.

Run after test_professional_ai.py: python test_professional_integration.py
"""
import copy
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import test_ai_signals as L
import test_professional_ai as H


def _check(value, message):
    if not value:
        raise AssertionError(message)


class _Sc(L._Sc):
    def __init__(self, ok=True, **settings):
        super().__init__(ok)
        self.extra = settings

    def get_settings(self):
        return dict(super().get_settings(), ai_learning_enabled=False, **self.extra)


class _Md:
    def __init__(self, market):
        self.market = market
        self.calls = []

    def fetch_klines(self, symbol, limit=300, interval='15m'):
        self.calls.append((symbol, limit, interval))
        return copy.deepcopy(self.market[interval])


def _engine(direction=1, **settings):
    bars = H.make_bars(1100, direction=direction)
    shift = int(L.NOW - 900 - bars[-1]['t'])
    for bar in bars:
        bar['t'] += shift
        # Достатній ATR, щоб реальна комісія не робила тестовий R:R невигідним.
        bar['h'] += 0.5
        bar['l'] -= 0.5
    market = {'15m': bars, '1h': L.A and __import__('detection.professional_ai', fromlist=['resample_ohlcv']).resample_ohlcv(bars, '1h'),
              '4h': __import__('detection.professional_ai', fromlist=['resample_ohlcv']).resample_ohlcv(bars, '4h')}
    sc = _Sc(True, **settings)
    tm = L._Tm()
    db = L._Db([dict(L._an('side', hint='up', o1=0.3), symbol='XUSDT')])
    engine = L.A.AiEngine(db=db, md=_Md(market), scanner=sc, tm=tm, clock=lambda: L.NOW)
    engine._refresh_cal = lambda now: None
    engine._fetch_live = lambda S, now: (dict(L._REC), dict(L._VW_UP))
    # 23.11: E2 — лише з підтвердженим розрядом (тут перевіряється мозок графіка)
    engine.discharge_for = lambda S, side, now=None, live_price=None: {'state': 'confirmed', 'edge': 100.0,
                                                                       'vob1h': {'ok': True, 'side': side}}
    engine.discharge_both = lambda S, side, now=None, live_price=None: {
        '1h': {'state': 'confirmed', 'edge': 100.0, 'tf': '1h'}, '15m': {'state': 'confirmed', 'edge': 100.0, 'tf': '15m'},
        'vob1h': {'ok': True, 'side': side}}
    engine.rdy_view = lambda S: None
    engine._ai_s = L.A.ai_settings_norm({'enabled': True, 'scenarios': ['E1', 'E2', 'E3', 'E4']})
    engine._ai_s_at = 1e15
    return engine, sc, tm


def test_professional_entry_passes_all_existing_gates():
    # 17.11 «єдиний мозок»: графік 15m/1h/4h — частина рішення 🤖 завжди
    # (окремого тумблера немає); санітарні перевірки пройдено → сигнал іде
    # через спільні ворота, ознаки й план — в епізод.
    # 22.11: ШІ — власний сценарій: угоду відкриває tm.ai_open, ворота бота
    # (`_signal_allowed`) його не зупиняють.
    engine, scanner, tm = _engine(1)
    engine.tick(L.NOW)
    _check(tm.opened and tm.opened[0]['side'] == 'LONG', 'Valid chart should open the AI trade')
    _check(not scanner.calls and not tm.sig, 'AI does not go through the bot gates')
    row = engine.state()['coins'][0]
    _check(row['brain']['ok'] and row['brain']['plan'], row['brain'])
    signal = engine._eps['XUSDT']['signal']
    _check(signal['professional']['features']['15m']['trend'] > 0,
           'Entry features must be archived for later learning')
    _check(signal['brain']['plan']['stop'] and signal['brain']['checks'], signal['brain'])


def test_stale_chart_blocks_before_shared_gates():
    engine, scanner, tm = _engine(1)
    for tf in engine._md.market:
        for bar in engine._md.market[tf]:
            bar['t'] -= 7200
    engine.tick(L.NOW)
    _check(not tm.opened and not scanner.calls, 'Stale 15m data must fail closed')
    row = engine.state()['coins'][0]
    _check(not row['brain']['ok'] and any(c['k'] == 'data' and c['ok'] is False
                                          for c in row['brain']['checks']), row['brain'])


def test_untrained_model_does_not_block_entry():
    # Модель, що ще вчиться (OOS < 200), на рішення НЕ впливає (раніше був
    # окремий тумблер «лише з OOS-моделлю» — прибрано 17.11).
    engine, scanner, tm = _engine(1)
    engine.tick(L.NOW)
    row = engine.state()['coins'][0]
    m = [c for c in row['brain']['checks'] if c['k'] == 'model'][0]
    _check(m['ok'] is None and not m['hard'] and tm.opened, m)


def _tm_stub(TM, engine, settings, calls=None):
    calls = calls if calls is not None else []

    def _anoms(s, a, b):
        calls.append('anoms')
        return []
    stub = types.SimpleNamespace(
        _settings=dict(settings), AI_EXIT_TTL=20.0, scanner=None, _exit_key=TM._exit_key,
        _spring_fire_view=lambda s, fetch=False: {'box_hi': 100.0, 'box_lo': 90.0, 'price': 99.0},
        _trade_anomalies=_anoms, trade_exh_for=lambda pos, vw: {})
    stub._ai_plan_follow = lambda *a, **k: TM._ai_plan_follow(stub, *a, **k)
    return stub


def test_trade_manager_plan_follow_is_part_of_ai_exit():
    # «🧠 Професійний вихід» більше не окреме правило: супровід за планом —
    # частина «🤖 Вихід з угоди ШІ» (один тумблер), кеш барів рушія.
    engine, _scanner, _tm = _engine(-1)
    L.A._instance = engine
    TM = L._tm_class()
    sys.modules['detection'].ai_signals = L.A
    calls = []
    stub = _tm_stub(TM, engine, {'use_ai_exit': True}, calls)
    opened = engine._md.market['15m'][0]['t']
    current = engine._md.market['15m'][-1]['c']
    pos = {'side': 'LONG', 'entry_price': 140.0, 'opened_at': opened,
           'peak_pnl_pct': 100.0 * (145.0 / 140.0 - 1.0)}
    # Монітор у мережу НЕ ходить: без кешу рушія — даних графіка немає.
    n0 = len(engine._md.calls)
    TM._ai_exit_reason(stub, 'XUSDT', pos, current, False)
    _check(len(engine._md.calls) == n0, 'Exit must read cache only')
    _check(pos['ai_pro']['close'] is False and calls == ['anoms'], pos.get('ai_pro'))
    # Кеш наповнює такт рушія → план угоди є, ціна далеко нижче стопу плану.
    engine._pro_market('XUSDT', L.NOW)
    stub._ai_exit_at = {}
    hit = TM._ai_exit_reason(stub, 'XUSDT', pos, current, False)
    _check(hit and hit[0] == 'ai_exit' and 'стоп плану' in hit[1], hit)
    _check(pos['ai_exit']['verdict'] == 'invalid' and pos['ai_exit']['plan']['stop']
           and 'analysis' not in pos['ai_pro'], pos['ai_exit'])
    # Вимкнено «вихід ШІ» (налаштування ШІ) → нічого не рахується і не закривається.
    engine._ai_s = dict(engine._ai_s, exit_enabled=False)
    stub2 = _tm_stub(TM, engine, {}, [])
    pos2 = dict(pos)
    pos2.pop('ai_exit', None); pos2.pop('ai_pro', None)
    _check(TM._ai_exit_reason(stub2, 'XUSDT', pos2, current, False) is None
           and 'ai_pro' not in pos2, pos2)


def test_engine_prefetches_open_trades_and_learning_is_gated():
    engine, _scanner, _tm = _engine(1)
    sb = types.ModuleType('detection.sbr_signals')
    sb._open_trade_symbols = lambda: ['YUSDT']
    old = sys.modules.get('detection.sbr_signals')
    sys.modules['detection.sbr_signals'] = sb
    try:
        # 22.11: бари готуються лише для ВЛАСНИХ угод ШІ
        engine._tm = types.SimpleNamespace(ai_open_positions=lambda: [('YUSDT', True)])
        _check(engine._pro_prefetch_open(L.NOW) == 1, 'One open trade')
        got = {(c[0], c[2]) for c in engine._md.calls}
        _check({('YUSDT', '15m'), ('YUSDT', '1h'), ('YUSDT', '4h')} <= got, got)
        engine._pro_prune(())
        _check('YUSDT' in engine._pro_bars, 'Open-trade bars survive pruning')
    finally:
        if old is not None:
            sys.modules['detection.sbr_signals'] = old
        else:
            sys.modules.pop('detection.sbr_signals', None)

    class _Db2:
        n = 0

        def get_ai_episodes(self, **k):
            _Db2.n += 1
            return []

        def get_research(self, **k):
            return []
    e2, sc2, _ = _engine(1)
    e2.db = _Db2()
    e2._tm = types.SimpleNamespace(_settings={})
    e2._ai_s = L.A.ai_settings_norm({'enabled': False, 'exit_enabled': False, 'learning_enabled': True})
    _check(e2._learn_professional(L.NOW, force=True) is None and _Db2.n == 0,
           'No archive reads while the brain is unused')
    e2._ai_s = dict(e2._ai_s, enabled=True)
    rep = e2._learn_professional(L.NOW, force=True)
    _check(_Db2.n == 1 and rep['status'] == 'no_new_labels', rep)


def test_reason_labels_and_light_payload():
    tm_src = open(os.path.join(ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    html = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    site = open(os.path.join(ROOT, 'infosite', 'js', 'app.js'), encoding='utf-8').read()
    _check(tm_src.count("'ai_pro_exit':") >= 2, 'TM close detail + badge')
    _check("'ai_pro_exit': '🧠 Проф. вихід ШІ'" in html and 'ai_pro_exit: "🧠 Проф. вихід ШІ"' in site,
           'Page and infosite labels')
    TM = L._tm_class()
    lite = sys.modules[TM.__module__]._lite_trade({'professional_entry': {'x': 1}, 'side': 'LONG'})
    _check('professional_entry' not in lite, 'Heavy entry context stays out of /api/tm/state')


def test_one_brain_one_switch_per_side_of_the_trade():
    # 17.11: окремих 🧠-тумблерів немає — один на вхід, один на вихід.
    source = open(os.path.join(ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    tm_source = open(os.path.join(ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    html = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    _check("'ai_learning_enabled'" in source, 'learning key')
    _check("'ai_professional_enabled': False" not in source
           and "'ai_professional_require_model': False" not in source, 'pro filter keys removed')
    _check("'use_ai_professional_exit'" not in tm_source, 'pro exit key removed')
    _check('_ai_pro_exit_reason' not in tm_source and 'def _ai_plan_follow' in tm_source, 'plan follow')
    for element in ('sm-ai-professional', 'sm-ai-require-model', 'tm-use-ai-professional-exit'):
        _check(f'id="{element}"' not in html, element)
    # 22.11: один тумблер на вхід і один на вихід — у ВЛАСНІЙ панелі ШІ.
    for element in ('aip-enabled', 'aip-learn', 'aip-exit', 'aip-exit-mode'):
        _check(f'id="{element}"' in html, element)
    _check('function aiProChipHTML' not in html and 'function _aiBrainCell' in html, 'UI мозку')
    a, m = (html.index(f'id="{x}"') for x in ('aip-exit', 'aip-exit-mode'))
    _check(a < m, 'випадайка режиму — біля «вихід ШІ»')


def test_js_exit_chip_shows_r_and_plan():
    import json as _json
    import subprocess
    import tempfile
    html = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    a = html.index('const _AIX_ICON')
    b = html.index('function aiExitChipHTML(')
    src = ("function fmtPriceJS(v){return String(Number(v));}\nconst AI_BRAND_IMG='<img>';\n" + html[a:html.index('function aiExitChipHTML(', a)]
           + html[b:html.index('\n}', b) + 2])
    pos = {'ai_exit': {'verdict': 'hold', 'verdict_ua': 'тримаємо', 'r': 1.234, 'peak_r': 1.8,
                       'groups': {'anom': False, 'price': True, 'profit': False},
                       'plan': {'stop': 95, 'target': 111, 'net_rr': 2.1}, 'plan_src': 'ai'}}
    src += "\nconsole.log(JSON.stringify([aiExitChipHTML(%s), aiExitChipHTML({})]));" % _json.dumps(pos)
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(src)
    out = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=30)
    _check(out.returncode == 0, out.stderr)
    lines = _json.loads(out.stdout)
    _check('+1.2R' in lines[0] and 'ціль 111' in lines[0] and 'сигналу ШІ' in lines[0], lines[0])
    _check(lines[1] == '', 'No chip without data')


if __name__ == '__main__':
    failures = 0
    for name, fn in sorted((k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)):
        try:
            fn()
            print('✓', name)
        except Exception as exc:
            failures += 1
            import traceback
            traceback.print_exc()
            print('✗', name, '—', repr(exc))
    print('OK' if not failures else f'FAIL {failures}')
    raise SystemExit(1 if failures else 0)
