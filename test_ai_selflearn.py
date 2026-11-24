"""🎓 Самонавчання ШІ по всіх напрямках (вимога 24.11).

Дослівно: «Бот має самостійно тренувати і вдосконалювати свій мозок по всіх
напрямках, з якими він працює.»

python test_ai_selflearn.py
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import test_ai_signals as L      # noqa: E402  (ai_signals + стаби)

A, NOW = L.A, L.NOW
_sp = importlib.util.spec_from_file_location('detection.ai_selflearn',
                                             os.path.join(ROOT, 'detection', 'ai_selflearn.py'))
SL = importlib.util.module_from_spec(_sp)
sys.modules['detection.ai_selflearn'] = SL
_sp.loader.exec_module(SL)


def _b(t, h, lo, c):
    return {'t': int(t * 1000), 'h': h, 'l': lo, 'p': c}


def test_entry_first_touch_win_loss_flat():
    r = SL.entry_new('X', 'LONG', 'E2', NOW, 100.0, risk_pct=1.0)
    # +1.5% раніше за −1% → ✅
    assert SL.entry_update(r, [_b(NOW + 3600, 101.6, 99.5, 101.0)], NOW + 7200)
    assert r['res'] == 'win' and r['r'] == SL.WIN_R and r['done']
    r = SL.entry_new('X', 'LONG', 'E2', NOW, 100.0, risk_pct=1.0)
    # обидві межі в одній свічці → ❌ (консервативно)
    SL.entry_update(r, [_b(NOW + 3600, 102.0, 98.5, 100.0)], NOW + 7200)
    assert r['res'] == 'loss' and r['r'] == -1.0
    # SHORT дзеркально; свічки ДО входу не рахуються
    r = SL.entry_new('X', 'SHORT', 'E1', NOW, 100.0, risk_pct=1.0)
    SL.entry_update(r, [_b(NOW - 3600, 120, 80, 100), _b(NOW + 3600, 100.4, 98.4, 98.6)], NOW + 7200)
    assert r['res'] == 'win'
    # горизонт минув без дотику → flat із кінцевим рухом у R
    r = SL.entry_new('X', 'LONG', 'E3', NOW, 100.0, risk_pct=1.0)
    SL.entry_update(r, [_b(NOW + 3600, 100.6, 99.6, 100.5)], NOW + SL.HORIZON_SEC + 10)
    assert r['res'] == 'flat' and r['r'] == 0.5


def test_exit_early_or_right():
    x = SL.exit_new('X', 'LONG', 'reversal', 100.0, NOW, risk_pct=1.0)
    SL.exit_update(x, [_b(NOW + 3600, 101.2, 99.8, 101.0)], NOW + 7200)
    assert x['res'] == 'early'                    # ціна пішла ще на 1R у бік угоди
    x = SL.exit_new('X', 'LONG', 'reversal', 100.0, NOW, risk_pct=1.0)
    SL.exit_update(x, [_b(NOW + 3600, 100.3, 99.4, 99.6)], NOW + 7200)
    assert x['res'] == 'right'


def _done(scen, side, res, n, conf=50):
    out = []
    for i in range(n):
        r = SL.entry_new('X%d' % i, side, scen, NOW + i, 100.0, 1.0,
                         feat={'charge': .8, 'pot': .6, 'prob': .7, 'lean': .5, 'break': .9}, conf=conf)
        r.update(done=True, res=res, r={'win': 1.5, 'loss': -1.0, 'flat': 0.0}[res])
        out.append(r)
    return out


def test_learn_scenario_factor_and_shrinkage():
    good = _done('E2', 'LONG', 'win', 30)
    bad = _done('E1', 'SHORT', 'loss', 30)
    few = _done('E3', 'LONG', 'win', 5)
    L_ = SL.learn(good + bad + few, [])
    assert L_['scen']['E2|LONG']['factor'] > 1.0 and L_['scen']['E2|LONG']['factor'] <= SL.FACT_HI
    assert L_['scen']['E1|SHORT']['factor'] < 1.0 and L_['scen']['E1|SHORT']['factor'] >= SL.FACT_LO
    assert L_['scen']['E3|LONG']['factor'] == 1.0        # мала вибірка — не вигадуємо
    c, f = SL.conf_adjust(50, 'E1', 'SHORT', L_)
    assert f < 1 and c < 50
    assert SL.conf_adjust(50, 'E4', 'LONG', L_) == (50, 1.0)


def test_exit_trust_never_touches_protective():
    ex = [dict(SL.exit_new('X', 'LONG', 'time', 100, NOW + i), done=True, res='early') for i in range(20)]
    ex += [dict(SL.exit_new('Y', 'LONG', 'invalid', 100, NOW + i), done=True, res='early') for i in range(20)]
    L_ = SL.learn([], ex)
    ok, why = SL.exit_allowed('time', L_)
    assert not ok and 'зарано' in why
    assert SL.exit_allowed('invalid', L_) == (True, '')
    assert SL.exit_allowed('false_start', L_) == (True, '')
    assert SL.exit_allowed('close', L_)[0]                # даних немає → як і раніше


def test_virtual_etalons_feed_the_main_brain():
    vg, va = SL.virtual_etalons(_done('E2', 'LONG', 'win', 20) + _done('E2', 'LONG', 'loss', 10))
    assert len(vg) == 20 and len(va) == 10 and all(e['virtual'] for e in vg + va)
    feat = {'charge': .8, 'pot': .6, 'prob': .7, 'lean': .5, 'break': .9}
    eb = A.etalon_brain(feat, [], [], 'LONG', A.ai_settings_norm({}), vgood=vg, vanti=va)
    assert eb['mature'] and eb['n_vgood'] == 20 and eb['p_good'] is not None
    # без віртуального досвіду — вчиться
    assert not A.etalon_brain(feat, [], [], 'LONG', A.ai_settings_norm({}))['mature']


def test_dis_tf_auto():
    L_ = SL.learn([], [], {'1h': {'done': 30, 'avg_last': 0.4}, '15m': {'done': 30, 'avg_last': 0.9},
                           '5m': {'done': 5, 'avg_last': 3.0}})
    assert L_['dis']['best'] == '15m'
    e = A.AiEngine(db=None, clock=lambda: NOW)
    e._sl_learned = L_
    assert e.dis_tf(A.ai_settings_norm({'discharge_tf': 'auto'})) == '15m'
    assert e.dis_tf(A.ai_settings_norm({'discharge_tf': '1h'})) == '1h'
    e._sl_learned = {}
    assert e.dis_tf(A.ai_settings_norm({'discharge_tf': 'auto'})) == '1h'


def test_engine_records_shadow_and_learns():
    e, sc, tm = L._engine(True)
    e.tick(NOW)
    assert e._sl_entries, 'кожна готова точка входу пишеться в тінь'
    n = len(e._sl_entries)
    e.tick(NOW + 30)
    assert len(e._sl_entries) == n, 'той самий сценарій — без дубля'
    r = e._sl_entries[0]
    assert r['scen'] and r['feat'] and r['risk'] > 0
    # після закриття угоди ШІ — тінь виходу
    assert e.selflearn_exit({'symbol': 'XUSDT', 'side': 'LONG', 'exit_price': 101.0, 'closed_at': NOW,
                             'ai_exit': {'verdict': 'time'}, 'ai_plan': {'risk_pct': 1.0}})
    st = e.selflearn_state()
    assert st['enabled'] and st['entries']['n'] == n and st['exits']['n'] == 1


def test_wiring():
    tmsrc = open(os.path.join(ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
    assert 'exit_trust(' in tmsrc and 'learn_hold' in tmsrc
    page = open(os.path.join(ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
    for k in ('aip-self', 'self_learning:', '_aipSelfLearnHTML', 'value="auto"'):
        assert k in page, k
    assert A.ai_settings_norm({})['self_learning'] is True
    assert A.ai_settings_norm({'discharge_tf': 'auto'})['discharge_tf'] == 'auto'


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
            n += 1
    print(f'OK {n}')
