"""Тест: VOB — «блок на графіку = сигнал» + чистий лог.

Перевіряє:
  • _vob_edge_outcome — СВІЖИЙ блок фаєриться і на ПЕРШОМУ показі ('fresh'),
    старий на першому показі → 'first_sight' (тиха база), duplicate, stale;
  • _vob_log_decision — оновлює self._vob_diag (для get_state/UI) і НІКОЛИ не
    пише в 🧾 Лог роботи бота (лог лишаємо чистим: туди йдуть лише реальні
    сигнали fired + вердикт rejected з основного шляху).

Модуль вантажиться ізольовано (без важкого detection/__init__).
"""
import os
import sys
import types
import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))

_pkg = types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg

# Фейковий detection.activity_log — щоб довести, що логер VOB туди НЕ пише.
_LOGS = []
_al = types.ModuleType('detection.activity_log')
_al.log_activity = lambda *a, **k: _LOGS.append((a, k))
sys.modules['detection.activity_log'] = _al


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


scmod = _load('detection.smc_scanner', 'detection/smc_scanner.py')
SC = scmod.SMCScanner


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ── 1. «Блок на графіку = сигнал»: свіжий фаєриться і на першому показі ──────
def test_edge_outcome_fresh_first_sight_fires():
    f = SC._vob_edge_outcome
    # ПЕРШИЙ показ + свіжий → 'fresh' (обробляємо поточний блок, НЕ ковтаємо).
    _check(f(None, 200, 3, 7) == 'fresh', 'перший показ + свіжий → fresh (сигнал!)')
    _check(f(None, 200, 7, 7) == 'fresh', 'перший показ + вік==поріг → fresh')
    # ПЕРШИЙ показ + старий → 'first_sight' (старий блок не сигнал → тиха база).
    _check(f(None, 200, 8, 7) == 'first_sight', 'перший показ + старий → first_sight')
    print('✓ свіжий блок фаєриться і на першому показі; старий → тиха база')


def test_edge_outcome_running():
    f = SC._vob_edge_outcome
    _check(f(100, 100, 0, 7) == 'duplicate', 'ft==prev → duplicate')
    _check(f(100, 90, 0, 7) == 'duplicate', 'ft<prev → duplicate')
    _check(f(100, 200, 3, 7) == 'fresh', 'новіший + свіжий → fresh')
    _check(f(100, 200, 8, 7) == 'stale', 'новіший + старий → stale')
    print('✓ у процесі роботи: duplicate/fresh/stale — коректно')


# ── 2. Логер НЕ засмічує лог; усе — лише в діагностику для UI ────────────────
def _mk():
    obj = SC.__new__(SC)
    obj._vob_diag = {}
    obj._lock = __import__('threading').RLock()
    return obj


def test_log_decision_never_writes_activity_log():
    _LOGS.clear()
    o = _mk()
    for oc in ('stale', 'first_sight', 'epoch', 'duplicate', 'no_candidate',
               'fired', 'filtered'):
        o._vob_log_decision('BTCUSDT', 'SHORT', oc, age=5, max_age=7, ft=1,
                            vol_tf='5m', detail='x', log_it=True)  # log_it ІГНОРУЄТЬСЯ
    _check(len(_LOGS) == 0, 'логер VOB НЕ пише в activity_log (лог чистий)')
    print('✓ логер нічого не ліпить у лог (усе — лише в діагностику)')


def test_log_decision_updates_diag():
    o = _mk()
    o._vob_log_decision('ETHUSDT', 'SHORT', 'stale', age=25, max_age=7, ft=99,
                        vol_tf='5m', detail='старий')
    d = o._vob_diag['ETHUSDT']
    _check(d['outcome'] == 'stale' and d['age'] == 25 and d['ft'] == 99,
           'діагностику збережено для get_state/UI')
    _check(d['label'].startswith('⌛'), 'людський підпис проставлено')
    print('✓ діагностика оновлюється (джерело для UI-колонки «стан VOB»)')


# ── 3. Catch-all: обидва напрямки за один цикл, жоден не губиться ────────────
def test_pick_candidates_both_sides():
    f = SC._vob_pick_candidates
    bull = {'formation_time': 100}
    bear = {'formation_time': 105}
    out = f(bull, bear)
    _check(len(out) == 2, 'обидва боки взято (не лише максимальний за часом)')
    # відсортовано за часом: старіший (bull 100) першим, новіший (bear 105) останнім
    _check(out[0][0] == 'LONG' and out[0][2] == 100, 'старіший (LONG 100) першим')
    _check(out[1][0] == 'SHORT' and out[1][2] == 105, 'новіший (SHORT 105) останнім')
    print('✓ обидва напрямки за один цикл (раніше брався лише один → інший губився)')


def test_pick_candidates_skips_breaker_and_invalid():
    f = SC._vob_pick_candidates
    _check(f({'formation_time': 100, 'breaker': True}, None) == [], 'breaker пропущено')
    _check(f({'formation_time': 0}, None) == [], 'ft<=0 пропущено')
    _check(f(None, None) == [], 'немає OB → порожньо')
    _check(len(f({'formation_time': 50}, None)) == 1, 'лише бичачий → один кандидат')
    print('✓ breaker/невалідні пропускаються, один бік теж ок')


def test_per_side_base_independent():
    # Симуляція пер-напрямкової бази: рух бази SHORT НЕ маскує новий LONG.
    seen = {'SHORT': 105}
    # новий LONG ft=100: prev(LONG)=None → перший показ; свіжий → 'fresh' (сигнал!),
    # хоча SHORT-база вже 105. Раніше єдина база=105 з'їла б цей LONG як duplicate.
    out_long = SC._vob_edge_outcome(seen.get('LONG'), 100, 2, 7)
    _check(out_long == 'fresh', 'новий LONG не маскується SHORT-базою → fresh')
    out_short = SC._vob_edge_outcome(seen.get('SHORT'), 105, 2, 7)
    _check(out_short == 'duplicate', 'той самий SHORT (105) → duplicate')
    print('✓ бази LONG/SHORT незалежні — протилежний бік не з’їдається')


# ── 4. «1 VOB на 1H OB»: свіжий 1H-OB скидає 5m і чекає новий 5m-VOB ─────────
def test_epoch_reset_needed():
    f = SC._vob_epoch_reset_needed
    _check(f(None, 1000) is True, 'перший 1H-OB (prev None) → reset')
    _check(f(1000, 2000) is True, 'інший bar_time → СВІЖИЙ 1H-OB → reset')
    _check(f(1000, 1000) is False, 'той самий 1H-OB → без reset')
    _check(f(1000, None) is False, 'немає 1H-OB → без reset (сигналу нема)')
    print('✓ свіжий 1H-OB (інший bar_time) скидає 5m-базу; той самий — ні')


def test_epoch_already_fired():
    f = SC._vob_epoch_already_fired
    _check(f(1000, 1000) is True, 'у цій епосі вже фаєрили → чекаємо новий 1H-OB')
    _check(f(1000, 2000) is False, 'нова епоха → можна фаєрити')
    _check(f(None, 1000) is False, 'ще не фаєрили → можна')
    _check(f(1000, None) is False, 'немає 1H-OB → не рахуємо як fired')
    print('✓ «1 сигнал на 1 1H-OB»: повторний fire у тій самій епосі блокується')


def test_epoch_flow_reset_then_new_5m():
    # Симуляція повного такту користувача:
    #   свіжий 1H-OB(A) скидає наявний 5m(100)→duplicate → чекаємо новий 5m
    #   новий 5m(200) у епосі A → fresh (сигнал) → епоха A fired
    #   свіжий 1H-OB(B) знову скидає → чекаємо новий 5m → новий 5m(400) → fresh
    edge = SC._vob_edge_outcome
    # epoch A: reset базує наявний 5m=100 → duplicate
    seen = {'LONG': 100}
    _check(edge(seen.get('LONG'), 100, 2, 7) == 'duplicate', 'наявний 5m після reset → duplicate')
    # новий 5m=200 → fresh (сигнал)
    _check(edge(seen.get('LONG'), 200, 2, 7) == 'fresh', 'новий 5m у епосі A → fresh (сигнал)')
    # epoch B (новий 1H-OB): reset_needed True, базуємо наявний 5m=200
    _check(SC._vob_epoch_reset_needed(1000, 2000) is True, 'новий 1H-OB → reset')
    seen_b = {'LONG': 200}
    _check(edge(seen_b.get('LONG'), 200, 2, 7) == 'duplicate', 'після reset епохи B наявний 5m → duplicate')
    _check(edge(seen_b.get('LONG'), 400, 2, 7) == 'fresh', 'новий 5m у епосі B → fresh (сигнал)')
    print('✓ такт: 1H-OB скидає 5m → чекає новий 5m → сигнал → новий 1H-OB → знову')


if __name__ == '__main__':
    test_edge_outcome_fresh_first_sight_fires()
    test_edge_outcome_running()
    test_log_decision_never_writes_activity_log()
    test_log_decision_updates_diag()
    test_pick_candidates_both_sides()
    test_pick_candidates_skips_breaker_and_invalid()
    test_per_side_base_independent()
    test_epoch_reset_needed()
    test_epoch_already_fired()
    test_epoch_flow_reset_then_new_5m()
    print('\nУсі тести VOB (1H-OB такт + catch-all + блок=сигнал + чистий лог) пройдено ✅')
