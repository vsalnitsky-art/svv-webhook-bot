"""Тест: ПРОЗОРІСТЬ VOB-алерту (жодного тихого відкидання).

Перевіряє:
  • _vob_edge_outcome — чисте рішення first_sight/duplicate/stale/fresh;
  • _vob_log_decision — оновлює self._vob_diag ЗАВЖДИ, а в 🧾 Лог пише лише коли
    log_it=True (stale/epoch/first_sight видимі; duplicate/no_candidate/fired/
    filtered — без дубля-логу, але в діагностиці є).

Модуль вантажиться ізольовано (без важкого detection/__init__).
"""
import os
import sys
import types
import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Порожній пакет-заглушка `detection`, щоб не тягнути важкий __init__.
_pkg = types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg

# Фейковий detection.activity_log — ловимо всі log_activity(...) виклики.
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


# ── 1. Чисте рішення edge/свіжості ──────────────────────────────────────────
def test_edge_outcome():
    f = SC._vob_edge_outcome
    _check(f(None, 100, 0, 7) == 'first_sight', 'prev None → first_sight')
    _check(f(100, 100, 0, 7) == 'duplicate', 'ft==prev → duplicate')
    _check(f(200, 100, 0, 7) == 'duplicate', 'ft<prev → duplicate')
    _check(f(100, 200, 3, 7) == 'fresh', 'новіший + свіжий → fresh')
    _check(f(100, 200, 7, 7) == 'fresh', 'вік == поріг → fresh (межа включна)')
    _check(f(100, 200, 8, 7) == 'stale', 'новіший але старий → stale')
    print('✓ _vob_edge_outcome: first_sight/duplicate/fresh/stale — коректно')


# ── 2. Логер: діагностика ЗАВЖДИ, лог — керовано ────────────────────────────
def _mk():
    obj = SC.__new__(SC)
    obj._vob_diag = {}
    obj._lock = __import__('threading').RLock()
    return obj


def test_log_decision_updates_diag_and_logs():
    _LOGS.clear()
    o = _mk()
    # stale → має і оновити діагностику, і написати в лог
    o._vob_log_decision('BTCUSDT', 'LONG', 'stale', age=25, max_age=7,
                        ft=123, vol_tf='5m', detail='OB старий')
    _check('BTCUSDT' in o._vob_diag, 'діагностику оновлено')
    d = o._vob_diag['BTCUSDT']
    _check(d['outcome'] == 'stale' and d['age'] == 25 and d['max_age'] == 7,
           'поля діагностики збережено')
    _check(d['label'].startswith('⌛'), 'людський підпис проставлено')
    _check(len(_LOGS) == 1 and _LOGS[0][1].get('source') == 'VOB',
           'stale пише ОДИН рядок у лог із source=VOB')
    print('✓ stale: діагностика + видимий рядок у лозі (source=VOB)')


def test_log_decision_silent_but_tracked():
    _LOGS.clear()
    o = _mk()
    # duplicate / no_candidate / fired / filtered → log_it=False: у лог НЕ пишемо,
    # але діагностику оновлюємо (видно в get_state).
    o._vob_log_decision('ETHUSDT', 'SHORT', 'duplicate', age=2, max_age=7,
                        ft=50, vol_tf='5m', log_it=False)
    o._vob_log_decision('SOLUSDT', None, 'no_candidate', vol_tf='5m', log_it=False)
    _check(len(_LOGS) == 0, 'log_it=False → у лог НЕ пишемо')
    _check(o._vob_diag['ETHUSDT']['outcome'] == 'duplicate', 'duplicate у діагностиці')
    _check(o._vob_diag['SOLUSDT']['outcome'] == 'no_candidate', 'no_candidate у діагностиці')
    print('✓ duplicate/no_candidate: без дубля-логу, але в діагностиці є')


def test_first_sight_and_epoch_are_visible():
    _LOGS.clear()
    o = _mk()
    o._vob_log_decision('AAAUSDT', 'LONG', 'first_sight', age=1, max_age=7, ft=10, vol_tf='5m')
    o._vob_log_decision('BBBUSDT', 'LONG', 'epoch', age=1, max_age=7, ft=10, vol_tf='5m',
                        detail='цей 1H-OB уже дав сигнал')
    _check(len(_LOGS) == 2, 'first_sight та epoch — обидва видимі в лозі')
    print('✓ first_sight та epoch лишають видимий слід (раніше гинули тихо)')


if __name__ == '__main__':
    test_edge_outcome()
    test_log_decision_updates_diag_and_logs()
    test_log_decision_silent_but_tracked()
    test_first_sight_and_epoch_are_visible()
    print('\nУсі тести VOB-прозорості пройдено ✅')
