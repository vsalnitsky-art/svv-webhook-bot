"""🧭 НАПРЯМОК ВОРІТ «ЧЕРЕЗ ПРУЖИНУ» = ⚠ АНОМАЛІЇ ОБСЯГУ (вимога 18.11).

Дослівно: «напрямок руху, який пропонує таблиця пружин, часто не встигає
оновитись і суперечить… орієнтуватись саме по показниках аномалії».
Пружина відповідає за «в таблиці / накручена», БІК — за аномаліями;
аномалії мовчать → вкладка Пружини (фолбек).

python test_spring_anom_dir.py
"""
import ast
import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import test_ai_signals as L      # noqa: E402  (ai_signals + стаби)

A = L.A


def _load(name, rel):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
    return m


G = _load('detection.spring_gates', 'detection/spring_gates.py')
V = _load('detection.vol_anomaly', 'detection/vol_anomaly.py')
NOW = L.NOW


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding='utf-8').read()


_UP = [L._an('absorb', hint='up', price=90.1, o1=0.3), L._an('side', hint='up', o1=0.2, ago=300)]
_DN = [L._an('absorb', hint='dn', price=100.0, o1=-0.3), L._an('side', hint='dn', o1=-0.2, ago=300)]


# ── 1. Чисте правило напрямку ─────────────────────────────────────────────
def test_direction_follows_the_anomalies():
    up = A.anomaly_direction(_UP, NOW, L.LV)
    dn = A.anomaly_direction(_DN, NOW, L.LV)
    _check(up['state'] == 'dir' and up['dir'] == 'LONG', up)
    _check(dn['state'] == 'dir' and dn['dir'] == 'SHORT', dn)
    none = A.anomaly_direction([], NOW, L.LV)
    _check(none['state'] == 'none' and none['dir'] is None, none)


def test_moves_both_ways_have_no_direction():
    mixed = A.anomaly_direction(_UP + _DN, NOW, L.LV)
    _check(mixed['state'] == 'mixed' and mixed['dir'] is None, mixed)


def test_old_moves_fall_out_of_the_window():
    old = [dict(a, ts=NOW - A.DIR_LOOK_SEC - 60) for a in _UP]
    _check(A.anomaly_direction(old, NOW, L.LV)['state'] == 'none', 'старше за вікно не рахується')


# ── 2. Ворота: бік — за аномаліями, фолбек — вкладка ─────────────────────
_FLAT = {'state': 'flat', 'pot_long': 2, 'pot_short': 1, 'charge': 5}
_TAB_LONG = {'state': 'charged', 'pot_long': 60, 'pot_short': 10, 'charge': 80}


def test_flat_coin_passes_when_anomalies_point_the_signal_way():
    """Кейс 18.11: «монета у вкладці ⚖ без напрямку, а сигнал LONG»."""
    ad = A.anomaly_direction(_UP, NOW, L.LV)
    ap, ok, chip, why = G.book_decide(G.TAG_WL, _FLAT, None, 'LONG', adir=ad)
    _check(ap and ok and 'аномалії LONG' in chip, (chip, why))


def test_anomalies_override_a_lagging_spring_tab():
    ad = A.anomaly_direction(_DN, NOW, L.LV)
    ap, ok, chip, why = G.book_decide(G.TAG_WL, _TAB_LONG, None, 'LONG', adir=ad)
    _check(ap and not ok and 'аномалії обсягу показують рух SHORT' in why, why)
    _check(G.book_decide(G.TAG_WL, _TAB_LONG, None, 'SHORT', adir=ad)[1], 'SHORT пропускається')


def test_mixed_moves_reject_and_silence_falls_back_to_the_tab():
    mixed = A.anomaly_direction(_UP + _DN, NOW, L.LV)
    _check(not G.book_decide(G.TAG_WL, _TAB_LONG, None, 'LONG', adir=mixed)[1], 'суперечливі → відмова')
    none = A.anomaly_direction([], NOW, L.LV)
    ap, ok, chip, why = G.book_decide(G.TAG_WL, _TAB_LONG, None, 'LONG', adir=none)
    _check(ok and 'вкладка LONG' in chip, chip)
    ap, ok, chip, why = G.book_decide(G.TAG_WL, _FLAT, None, 'LONG', adir=none)
    _check(not ok and 'аномалій немає' in why, why)
    # рушій ШІ недоступний (adir=None) — стара поведінка
    _check(G.book_decide(G.TAG_WL, _TAB_LONG, None, 'LONG')[1], 'без adir — вкладка')


def test_no_row_is_still_no_row():
    ad = A.anomaly_direction(_UP, NOW, L.LV)
    _check(not G.book_decide(G.TAG_XS, None, None, 'LONG', adir=ad)[1], 'монети немає в таблиці')


def test_highlight_uses_the_same_direction():
    s = {'spring_gate_enabled': False}
    ok, side, why = G.passes_gates(_TAB_LONG, s, None, A.anomaly_direction(_DN, NOW, L.LV))
    _check(ok and side == 'SHORT', (ok, side, why))


# ── 3. Скан-шлях і показ ──────────────────────────────────────────────────
def test_scanner_feeds_the_gates_from_the_ai_engine_cache():
    src = _src('detection/smc_scanner.py')
    _check('def _anom_dir' in src and 'adir=ad' in src and 'anomaly_dir(symbol)' in src, 'сканер')
    e, sc, tm = L._engine(True)
    e.tick(NOW)
    r = e.anomaly_dir('XUSDT')
    _check(r and r['state'] in ('dir', 'weak', 'mixed', 'none'), r)
    _check(e.anomaly_dir('NOPEUSDT')['state'] == 'none', 'монети без рядків — «аномалій немає»')
    e._clock = lambda: NOW + A.DIR_TTL + 5
    _check(e.anomaly_dir('XUSDT') is None, 'застарілий кеш → None (ворота беруть вкладку)')


def test_tables_show_the_gate_direction():
    html = _src('templates/smart_money.html')
    app = _src('web/flask_app.py')
    _check('function _sprAdirChip' in html and '${d}${adir}' in html, 'значок у колонці «Напрямок»')
    _check("row['adir'] = _adir_slim" in app and "r['adir'] = _adir_slim" in app, 'обидві таблиці')


# ── 4. Покриття аномалій для WATCHLIST — поза дослідженням ────────────────
def test_watchlist_direction_coins_stay_out_of_research_stats():
    syms, roles, _ = V.research_universe({}, {'AUSDT': 'wl'}, {}, NOW, extra=['AUSDT', 'BUSDT'])
    _check(roles['BUSDT'] == 'dir' and roles['AUSDT'] == 'entry', roles)
    rows = [{'kind': 'side', 'tf': '1h', 'ctx': {'role': 'dir'}},
            {'kind': 'side', 'tf': '1h', 'ctx': {'role': 'entry'}}]
    st = V.stats(rows)
    _check(len(st) == 1 and st[0]['n'] == 1, st)
    mon = V.VolAnomalyMonitor(get_trades=lambda: {}, get_charged=lambda: {},
                              get_dir_syms=lambda: ['CUSDT'])
    _check(mon.universe(NOW) == ['CUSDT'] and mon.state()['universe']['dir'] == 1, mon.state())


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
