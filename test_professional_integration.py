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
    sc = _Sc(True, ai_professional_enabled=True, **settings)
    tm = L._Tm()
    db = L._Db([dict(L._an('side', hint='up', o1=0.3), symbol='XUSDT')])
    engine = L.A.AiEngine(db=db, md=_Md(market), scanner=sc, tm=tm, clock=lambda: L.NOW)
    engine._refresh_cal = lambda now: None
    engine._fetch_live = lambda S, now: (dict(L._REC), dict(L._VW_UP))
    return engine, sc, tm


def test_professional_entry_passes_all_existing_gates():
    engine, scanner, tm = _engine(1)
    engine.tick(L.NOW)
    _check(tm.sig and tm.sig[0]['side'] == 'LONG', 'Valid chart should allow LONG through on_signal')
    _check(scanner.calls and scanner.calls[0][2] is True and scanner.calls[0][3] == 'ai',
           'Existing shared gates remain mandatory')
    row = engine.state()['coins'][0]
    _check(row['professional']['ok'] and row['professional_allowed'],
           'Professional snapshot must be visible in state')
    signal = engine._eps['XUSDT']['signal']
    _check(signal['professional']['features']['15m']['trend'] > 0,
           'Entry features must be archived for later learning')


def test_stale_chart_blocks_before_shared_gates():
    engine, scanner, tm = _engine(1)
    for tf in engine._md.market:
        for bar in engine._md.market[tf]:
            bar['t'] -= 7200
    engine.tick(L.NOW)
    _check(not tm.sig and not scanner.calls, 'Stale 15m data must fail closed')
    row = engine.state()['coins'][0]
    _check(not row['professional']['ok'] and not row['professional_allowed'], row['professional'])


def test_unready_required_model_blocks_entry():
    engine, scanner, tm = _engine(1, ai_professional_require_model=True)
    engine.tick(L.NOW)
    _check(not tm.sig and not scanner.calls, 'Required untrained model cannot open a trade')
    row = engine.state()['coins'][0]
    _check(any('модель' in x for x in row['professional']['vetoes']), row['professional'])


def _tm_stub(TM, engine, settings, calls=None):
    calls = calls if calls is not None else []

    def _anoms(s, a, b):
        calls.append('anoms')
        return []
    stub = types.SimpleNamespace(
        _settings=dict(settings), AI_EXIT_TTL=20.0, scanner=None, _exit_key=TM._exit_key,
        _spring_fire_view=lambda s, fetch=False: {'box_hi': 100.0, 'box_lo': 90.0, 'price': 99.0},
        _trade_anomalies=_anoms, trade_exh_for=lambda pos, vw: {})
    stub._ai_pro_exit_reason = lambda *a, **k: TM._ai_pro_exit_reason(stub, *a, **k)
    return stub


def test_trade_manager_professional_exit():
    engine, _scanner, _tm = _engine(-1)
    L.A._instance = engine
    TM = L._tm_class()
    sys.modules['detection'].ai_signals = L.A
    calls = []
    stub = _tm_stub(TM, engine, {'use_ai_exit': False, 'use_ai_professional_exit': True}, calls)
    opened = engine._md.market['15m'][0]['t']
    current = engine._md.market['15m'][-1]['c']
    pos = {'side': 'LONG', 'entry_price': 140.0, 'opened_at': opened,
           'peak_pnl_pct': 100.0 * (145.0 / 140.0 - 1.0)}
    # Монітор у мережу НЕ ходить: без кешу рушія — тримаємо (даних немає).
    n0 = len(engine._md.calls)
    hit = TM._ai_exit_reason(stub, 'XUSDT', pos, current, False)
    _check(hit is None and len(engine._md.calls) == n0, 'Exit must read cache only')
    _check(pos['ai_pro']['close'] is False, pos.get('ai_pro'))
    # Кеш наповнює такт рушія → тепер рішення є.
    engine._pro_market('XUSDT', L.NOW)
    stub._ai_exit_at = {}
    hit = TM._ai_exit_reason(stub, 'XUSDT', pos, current, False)
    _check(hit and hit[0] == 'ai_pro_exit' and 'Професійний вихід' in hit[1], hit)
    _check(pos['ai_pro']['close'] and 'analysis' not in pos['ai_pro'], pos['ai_pro'])
    # «🤖 Вихід ШІ» за аномаліями ВИМКНЕНО — він не рахувався і не закривав.
    _check(not calls and 'ai_exit' not in pos, 'Legacy AI exit must stay off')


def test_legacy_exit_runs_only_with_its_own_switch():
    engine, _scanner, _tm = _engine(1)
    L.A._instance = engine
    TM = L._tm_class()
    sys.modules['detection'].ai_signals = L.A
    calls = []
    stub = _tm_stub(TM, engine, {'use_ai_exit': True, 'use_ai_professional_exit': False}, calls)
    pos = {'side': 'LONG', 'entry_price': 100.0, 'opened_at': L.NOW - 3600}
    TM._ai_exit_reason(stub, 'XUSDT', pos, 101.0, False)
    _check(calls == ['anoms'] and 'ai_exit' in pos and 'ai_pro' not in pos, pos)


def test_engine_prefetches_open_trades_and_learning_is_gated():
    engine, _scanner, _tm = _engine(1)
    sb = types.ModuleType('detection.sbr_signals')
    sb._open_trade_symbols = lambda: ['YUSDT']
    old = sys.modules.get('detection.sbr_signals')
    sys.modules['detection.sbr_signals'] = sb
    try:
        engine._tm = types.SimpleNamespace(_settings={'use_ai_professional_exit': True})
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
    sc2.extra.update(ai_professional_enabled=False, ai_learning_enabled=True)
    sc2.get_settings = lambda: dict(L._Sc.get_settings(sc2), **sc2.extra)
    _check(e2._learn_professional(L.NOW, force=True) is None and _Db2.n == 0,
           'No archive reads while the brain is unused')
    sc2.extra['ai_professional_enabled'] = True
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


def test_new_settings_are_explicit_and_default_safe():
    source = open(os.path.join(ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    tm_source = open(os.path.join(ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    html = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    for key in ('ai_professional_enabled', 'ai_professional_require_model', 'ai_learning_enabled'):
        _check(f"'{key}'" in source, key)
    _check("'use_ai_professional_exit': False" in tm_source, 'Professional exit must default OFF')
    for element in ('sm-ai-professional', 'sm-ai-require-model', 'sm-ai-learning',
                    'tm-use-ai-professional-exit'):
        _check(f'id="{element}"' in html, element)
    # 17.11: «закривати:» належить 🤖 виходу — стоїть ОДРАЗУ за ним, до 🧠.
    a, m, b = (html.index(f'id="{x}"') for x in
               ('tm-use-ai-exit', 'tm-ai-exit-mode', 'tm-use-ai-professional-exit'))
    _check(a < m < b, 'випадайка режиму — біля «🤖 Вихід з угоди ШІ», а не біля 🧠')


def test_js_pro_chip_renders():
    import json as _json
    import subprocess
    import tempfile
    html = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    a = html.index('function aiProChipHTML(')
    src = "function fmtPriceJS(v){return String(Number(v));}\n" + html[a:html.index('\n}', a) + 2]
    pos = {'ai_pro': {'action': 'HOLD', 'close': False, 'r_multiple': 1.234, 'peak_r': 1.8,
                      'data_ok': True, 'plan': {'stop': 95, 'tp2': 111}, 'explanation': 'тримаємо'}}
    src += "\nconsole.log(JSON.stringify([aiProChipHTML(%s), aiProChipHTML({})]));" % _json.dumps(pos)
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(src)
    out = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=30)
    _check(out.returncode == 0, out.stderr)
    lines = _json.loads(out.stdout)
    _check('🧠+1.2R' in lines[0] and 'ціль 111' in lines[0], lines[0])
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
