"""🧩 Сценарії аномалій: комплекс реакцій до точки входу і до виходу (17.11).

Вимога: «бот має аналізувати весь процес перед входом… не по одній реакції, а
комплексно… і таким самим чином супроводжувати угоду… комплекс реакцій, які
дадуть розуміння, що потрібно закривати» + «заложи ШІ ці правила».

Запуск: python3 test_anomaly_scenarios.py
"""
import importlib.util
import os
import sys
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
if 'detection' not in sys.modules:
    _p = types.ModuleType('detection'); _p.__path__ = [os.path.join(_ROOT, 'detection')]
    sys.modules['detection'] = _p


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


S = _load('detection.anomaly_scenarios', 'detection/anomaly_scenarios.py')
_lg = types.ModuleType('detection.activity_log'); _lg.log_activity = lambda *a, **k: None
sys.modules['detection.activity_log'] = _lg
A = _load('detection.ai_signals', 'detection/ai_signals.py')
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()

NOW = 1_800_000_000.0
H = 3600
LV = [('межа коробки', 100.0), ('межа коробки', 90.0)]


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _an(kind, ago, tf='1h', **kw):
    a = {'kind': kind, 'tf': tf, 'ts': NOW - ago, 'price': 95.0}
    a.update(kw)
    return a


def _items(anoms, look=12 * H):
    return A.read_anomalies(anoms, NOW, LV, look=look)['items']


def _get(scens, key):
    return next(s for s in scens if s['key'] == key)


# ── ВХІД ─────────────────────────────────────────────────────────────────
def test_e1_reversal_full_sequence_is_the_entry_point():
    an = [_an('climax', 5 * H, bar_dir='dn', pre_chg=-3.0, price=90.5),     # 🔥 виснаження зливу
          _an('absorb', 3 * H, hint='up', price=90.1, o1=0.3),              # 🧽 на рівні, підтв.
          ]
    e = _get(S.match_entry(_items(an), 'LONG', {'price': 91.0}, None, NOW), 'E1')
    _check(e['done'] == 4 and e['complete'] and not e['ready'] and e['stale'],
           f'процес пройдено, але 3 год тому — точку входу пропущено: {S.stages_text(e)}')
    an2 = [_an('climax', 3 * H, bar_dir='dn', pre_chg=-3.0, price=90.5),
           _an('absorb', 1 * H, hint='up', price=90.1, o1=0.3)]
    e2 = _get(S.match_entry(_items(an2), 'LONG', {'price': 91.0}, None, NOW), 'E1')
    _check(e2['ready'], f'свіжий повний сценарій — точка входу: {S.stages_text(e2)}')
    one = _get(S.match_entry(_items(an2[:1]), 'LONG', {'price': 91.0}, None, NOW), 'E1')
    _check(one['done'] == 1 and not one['ready'], 'одна реакція — лише перший етап')


def test_e1_broken_by_counter_reaction_or_lost_level():
    an = [_an('climax', 3 * H, bar_dir='dn', pre_chg=-3.0, price=90.5),
          _an('absorb', 2 * H, hint='up', price=90.1, o1=0.3),
          _an('side', 1 * H, hint='dn', o1=-0.3, price=89.5)]                # ⚡ продавці знову
    e = _get(S.match_entry(_items(an), 'LONG', {'price': 91.0}, None, NOW), 'E1')
    _check(e['broken'] and not e['ready'], f'зустрічна підтверджена реакція — злам: {e}')
    lost = _get(S.match_entry(_items(an[:2]), 'LONG', {'price': 88.0}, None, NOW), 'E1')
    _check(lost['broken'] and 'не втримала' in lost['broken'], f'ціна втратила рівень: {lost}')


def test_e1_choch_confirms():
    an = [_an('climax', 3 * H, bar_dir='dn', pre_chg=-3.0, price=90.5),
          _an('absorb', 2 * H, hint='up', price=90.1)]                       # без o1
    e0 = _get(S.match_entry(_items(an), 'LONG', {'price': 91.0}, None, NOW), 'E1')
    _check(e0['done'] == 2, f'без підтвердження — 2/4: {S.stages_text(e0)}')
    e = _get(S.match_entry(_items(an), 'LONG', {'price': 91.0},
                           {'dir': 'up', 't': NOW - H}, NOW), 'E1')
    _check(e['ready'], f'CHoCH у бік входу — підтвердження: {S.stages_text(e)}')


def test_e2_discharge_and_absorbed_breakout():
    ctx = {'price': 101.0, 'beyond': 'LONG', 'charged': True, 'bar_t': NOW - 1800}
    an = [_an('side', 600, hint='up', o1=0.3, price=101.0)]
    e = _get(S.match_entry(_items(an), 'LONG', ctx, None, NOW), 'E2')
    _check(e['ready'], f'заряд → пробій → ⚡ продовження → ніхто не поглинає: {S.stages_text(e)}')
    ab = an + [_an('absorb', 300, hint='dn', price=101.0)]
    b = _get(S.match_entry(_items(ab), 'LONG', ctx, None, NOW), 'E2')
    _check(b['broken'] and not b['ready'], f'пробій поглинають — злам: {b}')
    nc = _get(S.match_entry(_items(an), 'LONG', dict(ctx, charged=False), None, NOW), 'E2')
    _check(not nc['ready'], 'без заряду пружини сценарій розряду не завершується')


def test_e3_pullback_continuation():
    an = [_an('side', 5 * H, hint='up', o1=0.4, price=95.0),                # рух угору
          _an('side', 1 * H, hint='up', o1=0.2, price=90.2)]                # відкат до рівня 90
    e = _get(S.match_entry(_items(an), 'LONG', {'price': 91.0}, None, NOW), 'E3')
    _check(e['ready'], f'рух → тихий відкат → реакція на рівні: {S.stages_text(e)}')
    bad = an[:1] + [_an('absorb', 3 * H, hint='dn', o1=-0.3, price=96.0)] + an[1:]
    b = _get(S.match_entry(_items(bad), 'LONG', {'price': 91.0}, None, NOW), 'E3')
    _check(b['broken'], 'відкат із зустрічним тиском — злам')


def test_entry_decision_needs_complete_fresh_scenario():
    rec = {'charge': 80, 'state': 'charged', 'prob_long': 70}
    vw = {'box_hi': 100.0, 'box_lo': 90.0, 'price': 101.0, 'beyond': 'LONG',
          'range_x': 2.0, 'vol_x': 2.0, 'range_need': 1.8, 'vol_need': 1.8, 't': NOW - 1800}
    hi = A.entry_score('LONG', rec, vw, [], NOW)
    _check(hi['score'] >= 50 and not A.entry_decide(hi), 'бал є, сценарію немає — не входимо')
    ok = A.entry_score('LONG', rec, vw, [_an('side', 600, hint='up', o1=0.3, price=101.0)], NOW)
    _check(A.entry_decide(ok) and 'Пробій після стиснення' in ok['scenario_text'], ok['scenario_text'])


def test_e4_spring_false_break():
    ctx = {'price': 91.0, 'box_lo': 90.0, 'box_hi': 100.0}
    an = [_an('climax', 3 * H, bar_dir='dn', pre_chg=-1.0, price=89.2),        # вибили межу 90
          _an('absorb', 2 * H, hint='up', price=89.8, o1=0.3)]                  # поглинули, повернули
    e = _get(S.match_entry(_items(an), 'LONG', ctx, None, NOW), 'E4')
    _check(e['ready'], f'Spring: вибили межу → поглинання → повернення → підтвердження: {S.stages_text(e)}')
    fail = an + [_an('side', H, hint='dn', o1=-0.3, price=88.5)]              # новий мінімум
    b = _get(S.match_entry(_items(fail), 'LONG', dict(ctx, price=88.4), None, NOW), 'E4')
    _check(b['broken'], f'новий екстремум — пробій справжній: {b}')


# ── ВИХІД ────────────────────────────────────────────────────────────────
def test_exit_scenarios_after_entry_only():
    oa = NOW - 6 * H
    x1 = [_an('climax', 3 * H, bar_dir='up', pre_chg=3.0, price=110.0),     # 🔥 наш рух вичерпано
          _an('absorb', 2 * H, hint='dn', price=110.2, o1=-0.3)]
    r = _get(S.match_exit(_items(x1), 'LONG', {'price': 109.0}, None, NOW, oa), 'X1')
    _check(r['complete'] and r['ready'], f'X1 кульмінація руху угоди: {S.stages_text(r)}')
    late = _get(S.match_exit(_items(x1), 'LONG', {'price': 109.0}, None, NOW, NOW - 2.5 * H), 'X1')
    _check(not late['complete'], '🔥 до входу — не етап виходу')
    x4 = [_an('absorb', 3 * H, hint='dn', price=105.0), _an('absorb', H, hint='dn', price=105.5)]
    r4 = _get(S.match_exit(_items(x4), 'LONG', {'price': 104.0}, None, NOW, oa), 'X4')
    _check(r4['complete'], f'X4 подвійне поглинання проти: {S.stages_text(r4)}')
    x3 = [_an('side', 4 * H, hint='up', o4=0.05, price=105.0),
          _an('spike', 3 * H, bar_dir='up', o4=-0.1, price=105.1),
          _an('side', H, hint='dn', price=104.8)]
    r3 = _get(S.match_exit(_items(x3), 'LONG', {'price': 104.5}, None, NOW, oa), 'X3')
    _check(r3['complete'], f'X3 зусилля без результату: {S.stages_text(r3)}')
    x2 = [_an('side', 3 * H, hint='dn', price=100.1, pre_chg=2.0)]          # ⚡ проти тренду на рівні
    r2 = _get(S.match_exit(_items(x2), 'LONG', {'price': 99.0},
                           {'dir': 'dn', 't': NOW - H}, NOW, oa), 'X2')
    _check(r2['complete'], f'X2 перехоплення ініціативи: {S.stages_text(r2)}')
    x5 = [_an('spike', 3 * H, bar_dir='up', price=112.5),                     # вибили верх 112
          _an('absorb', 2 * H, hint='dn', price=111.8, o1=-0.3)]
    r5 = _get(S.match_exit(_items(x5), 'LONG', {'price': 111.0, 'box_hi': 112.0, 'box_lo': 100.0},
                           None, NOW, oa), 'X5')
    _check(r5['complete'], f'X5 аптраст: {S.stages_text(r5)}')


def test_exit_score_uses_scenarios():
    oa = NOW - 6 * H
    vw = {'box_hi': 112.0, 'box_lo': 100.0, 'price': 104.0}
    x4 = [_an('absorb', 3 * H, hint='dn', price=105.0), _an('absorb', H, hint='dn', price=105.5)]
    r = A.exit_score('LONG', vw, x4, NOW, opened_at=oa, look=6 * H)
    _check(r['verdict'] == 'scenario' and A.exit_decide(r, 'all') and r['scenario']['key'] == 'X4',
           f'сценарій виходу завершено → вихід: {r["verdict"]} {r.get("scenario_text")}')


# ── 📘 Довідник правил = код ─────────────────────────────────────────────
def test_catalog_matches_code_and_page():
    cat = S.CATALOG
    _check([x['key'] for x in cat['entry']] == ['E1', 'E2', 'E3', 'E4']
           and [x['key'] for x in cat['exit']] == ['X1', 'X2', 'X3', 'X4', 'X5'], 'усі сценарії в довіднику')
    for sc in S.match_entry([], 'LONG', {}, None, NOW):
        _check(len(sc['stages']) == len(next(x for x in cat['entry'] if x['key'] == sc['key'])['stages']),
               f'етапи {sc["key"]} = довідник')
    for sc in S.match_exit([], 'LONG', {}, None, NOW):
        _check(len(sc['stages']) == len(next(x for x in cat['exit'] if x['key'] == sc['key'])['stages']),
               f'етапи {sc["key"]} = довідник')
    _check('function _aiRulesHTML' in _HTML and 'id="ai-rules-body"' in _HTML
           and 'function _aiScenCell' in _HTML and 'function _aiDecision' in _HTML,
           'сторінка: довідник правил, сценарії LONG/SHORT, рішення')


def test_e2_old_impulse_is_not_a_fresh_entry_point_1811():
    """🐞 18.11 (кейс AVAX/TRUMP): етап «🌀 заряджена» (t = now) робив свіжим
    будь-який старий ланцюг — сигнал ішов щопівгодини по імпульсу 8-год давності."""
    ctx = {'charged': True}
    old = [_an('spike', 8 * H, tf='15m', bar_dir='up', o1=0.3, price=95.0),
           _an('side', 3 * H, tf='15m', hint='up', price=96.0)]
    e = _get(S.match_entry(_items(old), 'LONG', ctx, None, NOW), 'E2')
    _check(e['complete'] and not e['ready'] and e['stale'], f'запізно, а не точка входу: {e}')
    _check(e['last_t'] is not None and e['last_t'] < NOW - 2 * H,
           'свіжість — за останньою ПОДІЄЮ, не за станом пружини')


def test_newer_chain_is_not_shadowed_by_an_older_stale_one_1811():
    """Сценарій чіплявся за найстарішу реакцію вікна — свіжий ланцюг губився."""
    ctx = {'charged': True}
    an = [_an('spike', 9 * H, tf='15m', bar_dir='up', o1=0.3, price=95.0),
          _an('side', 8 * H, tf='15m', hint='up', price=95.5),
          _an('spike', 1200, tf='15m', bar_dir='up', o1=0.3, price=97.0),
          _an('side', 300, tf='15m', hint='up', price=97.5)]
    e = _get(S.match_entry(_items(an), 'LONG', ctx, None, NOW), 'E2')
    _check(e['ready'] and e['last_t'] >= NOW - 1200, f'свіжий ланцюг: {S.stages_text(e)}')


if __name__ == '__main__':
    fails = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]:
        try:
            fn(); print('✓', name)
        except Exception as e:
            fails += 1; print('✗', name, '—', repr(e)[:600])
    print('OK' if not fails else f'FAIL {fails}')
    sys.exit(1 if fails else 0)
