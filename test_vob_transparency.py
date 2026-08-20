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


if __name__ == '__main__':
    test_edge_outcome_fresh_first_sight_fires()
    test_edge_outcome_running()
    test_log_decision_never_writes_activity_log()
    test_log_decision_updates_diag()
    print('\nУсі тести VOB (блок=сигнал + чистий лог) пройдено ✅')
