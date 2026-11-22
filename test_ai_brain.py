"""🤖🧠 ЄДИНИЙ МОЗОК ШІ (17.11): вхід = напрямок → старт → санітарні перевірки
графіка → план; вихід = вердикт аномалій + супровід за планом угоди.

python test_ai_brain.py
"""
import importlib.util
import os
import sys
import time
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_sp = importlib.util.spec_from_file_location('detection.ai_brain',
                                             os.path.join(ROOT, 'detection', 'ai_brain.py'))
B = importlib.util.module_from_spec(_sp)
sys.modules['detection.ai_brain'] = B
_sp.loader.exec_module(B)

import test_ai_signals as L          # noqa: E402  (рушій + стаби)


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _er(ready=True, veto='', score=70):
    sc = {'key': 'E2', 'name': '💥 Старт розряду Пружини', 'done': 3, 'total': 3,
          'ready': ready, 'stale': False, 'broken': ''}
    return {'side': 'LONG', 'score': score, 'veto': veto, 'scenario': sc}


def _k(r, k):
    return [c for c in r['checks'] if c['k'] == k][0]


# ─────────────────────────────── ВХІД ───────────────────────────────
def test_trend_vetoes_do_not_block_a_discharge_start():
    """Строгий трендовий фільтр проти (тренд ще плаский), але санітарні
    перевірки пройдено → вхід ДОЗВОЛЕНО, а строгий вердикт — у тіні."""
    r = B.entry_check('LONG', _er(), L.pro_ok(), {'prob_long': 70})
    _check(r['ok'] and r['strict'] and not r['strict']['ready'], r)
    _check(r['plan']['stop'] == 99.0 and r['plan']['target_src'] == 'r'
           and r['plan']['net_rr'] >= B.MIN_NET_RR, r['plan'])
    _check('тренд' not in r['reason'], r['reason'])


def test_direction_and_start_come_first():
    r = B.entry_check('LONG', _er(veto='ціна пробиває коробку в ПРОТИЛЕЖНИЙ бік'), L.pro_ok(), {})
    _check(not r['ok'] and _k(r, 'direction')['ok'] is False, r)
    r = B.entry_check('LONG', _er(ready=False), L.pro_ok(), {})
    _check(not r['ok'] and _k(r, 'start')['ok'] is False and 'формується' in _k(r, 'start')['note'], r)


def test_sanitary_checks_fail_closed():
    r = B.entry_check('LONG', _er(), {'ok': False, 'data_issues': ['15m: застарілі дані']}, {})
    _check(not r['ok'] and 'застарілі' in _k(r, 'data')['note'] and r['plan'] is None, r)
    r = B.entry_check('LONG', _er(), L.pro_ok(mom=-0.6), {})
    _check(not r['ok'] and _k(r, 'impulse')['ok'] is False, 'імпульс 15m проти — старт спростовано')
    r = B.entry_check('LONG', _er(), L.pro_ok(atr=4.5), {})
    _check(not r['ok'] and _k(r, 'volatility')['ok'] is False, 'волатильність поза межами')
    r = B.entry_check('LONG', _er(), L.pro_ok(ready_model=True, p=0.4), {})
    _check(not r['ok'] and _k(r, 'model')['ok'] is False, 'готова модель проти')
    r = B.entry_check('LONG', _er(), L.pro_ok(ready_model=False, p=0.1), {})
    _check(r['ok'] and _k(r, 'model')['ok'] is None, 'модель, що вчиться, не впливає')


def test_room_to_target_uses_the_spring_move():
    # ціль Пружини близько: +0.8% при стопі −1% → R < мін → входу немає
    r = B.entry_check('LONG', _er(), L.pro_ok(), {'tgt_long': 100.8})
    _check(not r['ok'] and _k(r, 'room')['ok'] is False and r['plan']['target_src'] == 'spring', r)
    # ціль далеко: +4% → R ≈ 3.4
    r = B.entry_check('LONG', _er(), L.pro_ok(), {'tgt_long': 104.0})
    _check(r['ok'] and r['plan']['target'] == 104.0 and r['plan']['net_rr'] > 3, r['plan'])
    # ціль ПОЗАДУ ціни (не в бік угоди) → фолбек 2.2R
    r = B.entry_check('LONG', _er(), L.pro_ok(), {'tgt_long': 97.0})
    _check(r['plan']['target_src'] == 'r', r['plan'])


# ─────────────────────────────── ВИХІД ───────────────────────────────
_PLAN = {'entry': 100.0, 'stop': 99.0, 'target': 104.0, 'net_rr': 3.0}


def _anom(verdict='none', ready=False, groups=None, hold=0.0):
    return {'verdict': verdict, 'ready': ready, 'hold': hold,
            'groups': groups or {'anom': False, 'price': False, 'profit': False}, 'checklist': []}


def test_plan_stop_closes_in_both_modes_even_against_continuation():
    for mode in ('confirmed', 'all'):
        m = B.exit_merge('LONG', _anom(hold=1.0), None, mode, _PLAN, 98.9)
        _check(m['verdict'] == 'invalid' and m['close'], (mode, m))


def test_reversal_and_profit_rules_follow_the_mode():
    rev = {'close': True, 'reason': 'professional_reversal', 'r_multiple': 0.4, 'peak_r': 0.6}
    m = B.exit_merge('LONG', _anom(), rev, 'confirmed', _PLAN, 100.4)
    _check(m['verdict'] == 'reversal' and m['close'] and m['groups']['price'], m)
    m = B.exit_merge('LONG', _anom(), rev, 'all', _PLAN, 100.4)
    _check(not m['close'], 'режим «всі показники» розворот сам не закриває')
    # свіже продовження за угодою тримає розворот, але не захисне
    m = B.exit_merge('LONG', _anom(hold=0.8), rev, 'confirmed', _PLAN, 100.4)
    _check(m['verdict'] == 'hold' and not m['close'], m)
    m = B.exit_merge('LONG', _anom(), {'close': False, 'r_multiple': 4.1}, 'confirmed', _PLAN, 104.1)
    _check(m['verdict'] == 'target' and m['close'] and m['groups']['profit'], m)


def test_chart_evidence_completes_all_groups():
    """Аномалії кажуть «закривати», графік дає розворот, прибуток — ціль →
    ✅ усі групи (закриває навіть у суворому режимі)."""
    an = _anom('watch', False, {'anom': True, 'price': False, 'profit': False})
    pro = {'close': True, 'reason': 'professional_reversal', 'r_multiple': 4.0, 'peak_r': 4.0}
    m = B.exit_merge('LONG', an, pro, 'all', _PLAN, 104.2)
    _check(m['verdict'] == 'all' and m['close'], m)


def test_own_plan_beats_the_chart_brains_2_2r():
    """Угода сигналу ШІ має свій план (ціль Пружини 104); власна ціль 2.2R
    мозку графіка тоді не закриває."""
    pro = {'close': True, 'reason': 'professional_target', 'r_multiple': 2.3}
    m = B.exit_merge('LONG', _anom(), pro, 'confirmed', _PLAN, 102.3)
    _check(m['verdict'] != 'target' and not m['close'], m)
    # без власного плану — план мозку графіка і є планом угоди
    pro2 = dict(pro, entry_risk={'entry': 100, 'stop': 99, 'tp2': 102.2})
    m = B.exit_merge('LONG', _anom(), pro2, 'confirmed', None, 102.3)
    _check(m['verdict'] == 'target' and m['close'], m)


def test_no_chart_data_anomalies_still_work():
    an = _anom('scenario', True, {'anom': True, 'price': True, 'profit': False})
    m = B.exit_merge('SHORT', an, None, 'all', None, 100.0)
    _check(m['verdict'] == 'scenario' and m['close'], m)


# ─────────────────────────────── РУШІЙ / TM ───────────────────────────────
def test_engine_brain_veto_blocks_signal_and_is_logged_once():
    e, sc, tm = L._engine(True)
    e._professional_snapshot = lambda S, now: L.pro_ok(mom=-0.6)
    e.tick(L.NOW)
    e.tick(L.NOW + 30)
    ep = e._eps['XUSDT']
    _check(not tm.opened and not tm.sig and not sc.calls, 'мозок проти → угоди немає')
    vetoes = [x for x in ep.get('ev') or [] if x.get('k') == 'brain_veto']
    _check(len(vetoes) == 1, f'подія вето — один раз, поки причина та сама: {vetoes}')
    row = e.state()['coins'][0]
    _check(not row['brain']['ok'] and 'brain_rules' in e.state(), row['brain'])


def test_engine_signal_carries_plan_to_trade():
    e, sc, tm = L._engine(True)
    e.tick(L.NOW)
    # 22.11: план кладе на угоду сам `tm.ai_open` (власний шлях ШІ)
    got = tm.opened[0] if tm.opened else {}
    _check(got.get('plan') and got['plan']['stop'] == 99.0, got)
    sig = e._eps['XUSDT']['signal']
    _check(sig['brain']['plan'] and sig['brain']['strict'] is not None, sig)


def test_trade_plan_attaches_only_to_a_fresh_trade():
    TM = L._tm_class()
    import threading
    now = time.time()
    stub = types.SimpleNamespace(_lock=threading.Lock(),
                                 _positions={'XUSDT': {'side': 'LONG', 'opened_at': now - 10},
                                             'YUSDT': {'side': 'LONG', 'opened_at': now - 3600}},
                                 _shadow_positions={})
    _check(TM.set_ai_professional_context(stub, 'XUSDT', 'LONG', {}, plan=_PLAN), 'свіжа угода')
    _check(stub._positions['XUSDT']['ai_plan']['stop'] == 99.0, stub._positions)
    _check(not TM.set_ai_professional_context(stub, 'YUSDT', 'LONG', {}, plan=_PLAN)
           and 'ai_plan' not in stub._positions['YUSDT'], 'давня чужа угода плану не отримує')


def test_tm_exit_uses_the_trade_plan():
    TM = L._tm_class()
    sys.modules['detection'].ai_signals = L.A
    stub = types.SimpleNamespace(
        _settings={'use_ai_exit': True, 'ai_exit_mode': 'all'}, AI_EXIT_TTL=20.0, scanner=None,
        _exit_key=TM._exit_key, _ai_plan_follow=lambda *a, **k: None,
        _spring_fire_view=lambda s, fetch=False: {}, _trade_anomalies=lambda s, a, b: [],
        trade_exh_for=lambda pos, vw: {})
    pos = {'side': 'LONG', 'entry_price': 100.0, 'opened_at': time.time() - 600, 'ai_plan': _PLAN}
    _check(TM._ai_exit_reason(stub, 'XUSDT', pos, 100.5, False) is None
           and pos['ai_exit']['plan_src'] == 'ai', pos.get('ai_exit'))
    stub._ai_exit_at = {}
    hit = TM._ai_exit_reason(stub, 'XUSDT', pos, 98.8, False)
    _check(hit and hit[0] == 'ai_exit' and 'стоп плану' in hit[1], hit)


if __name__ == '__main__':
    fails = 0
    for name, fn in sorted((k, v) for k, v in dict(globals()).items()
                           if k.startswith('test_') and callable(v) and getattr(v, '__module__', '') == __name__):
        try:
            fn()
            print('✓', name)
        except Exception as e:
            fails += 1
            import traceback
            traceback.print_exc()
            print('✗', name, '—', repr(e))
    print('OK' if not fails else f'FAIL {fails}')
    sys.exit(1 if fails else 0)
