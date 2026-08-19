"""Тест: гейт «1 VOB на 1H OB» на РІВНІ ВІДКРИТТЯ (fuel_filter).

Перевіряє, що монета НЕ ре-відкривається на тому самому 1H-OB після закриття/
видалення (баг GRAMUSDT: угода відкрилась знову за ~54с після закриття через
новий 5m VOB на тому самому 1H-OB). Скид позначки — лише на НОВОМУ 1H-OB.

Логіка тестується на РЕАЛЬНИХ методах FuelFilterDaemon (важкі залежності —
застабовані), інстанс створюється в обхід __init__.
"""
import os
import sys
import types
import importlib.util

# fuel_filter має ЛИШЕ один легкий пакетний імпорт (detection.signal_labels).
# Щоб НЕ тягнути важкий detection/__init__ (біржові пакети), реєструємо `detection`
# як порожній пакет і завантажуємо потрібні модулі напряму з файлів.
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Порожній пакет-заглушка `detection` (без __init__ з важкими імпортами).
_pkg = types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg

_load('detection.signal_labels', 'detection/signal_labels.py')
ffmod = _load('detection.fuel_filter', 'detection/fuel_filter.py')

FF = ffmod.FuelFilterDaemon


def _make():
    """Інстанс FF в обхід __init__ — лише поля/методи, потрібні для гейта."""
    obj = FF.__new__(FF)
    obj._opened_ob_epoch = {}
    obj._persist_state = lambda: None
    # Керовані стани для стабів:
    obj._t_enabled = True     # vob_one_per_ob ON/OFF
    obj._t_bt = None          # поточний bar_time 1H-OB (None = немає OB)
    # Підмінюємо два джерела даних (scanner-налаштування + БД OB-стану):
    obj._vob_one_per_ob_on = types.MethodType(lambda self: self._t_enabled, obj)
    obj._ob_epoch_now = types.MethodType(lambda self, sym: self._t_bt, obj)
    return obj


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_disabled_never_blocks():
    o = _make(); o._t_enabled = False; o._t_bt = 1000
    o._opened_ob_epoch['BTCUSDT'] = 1000     # навіть якщо позначено
    _check(o._ob_epoch_already_opened('BTCUSDT') is False,
           'вимкнений гейт не має блокувати')
    print('✓ disabled → ніколи не блокує')


def test_first_open_then_block_same_ob():
    o = _make(); o._t_bt = 1000
    # Перший показ — ще не відкривали → не блок.
    _check(o._ob_epoch_already_opened('GRAMUSDT') is False, 'перший показ не блок')
    # Відкрили угоду → позначаємо епоху.
    o._mark_ob_epoch_opened('GRAMUSDT')
    _check(o._opened_ob_epoch.get('GRAMUSDT') == 1000, 'епоху позначено')
    # (закриття угоди епоху НЕ чистить — блокує повторне відкриття)
    _check(o._ob_epoch_already_opened('GRAMUSDT') is True,
           'той самий 1H-OB після закриття → блок')
    print('✓ той самий 1H-OB після відкриття/закриття → блок (баг GRAM)')


def test_new_ob_resets():
    o = _make(); o._t_bt = 1000
    o._mark_ob_epoch_opened('GRAMUSDT')
    _check(o._ob_epoch_already_opened('GRAMUSDT') is True, 'блок на старому OB')
    # З'явився НОВИЙ 1H-OB (інший bar_time) → скид, знову можна.
    o._t_bt = 2000
    _check(o._ob_epoch_already_opened('GRAMUSDT') is False,
           'новий 1H-OB → скид (можна відкривати)')
    _check('GRAMUSDT' not in o._opened_ob_epoch, 'застарілу позначку прибрано')
    print('✓ новий 1H-OB → скид позначки')


def test_no_ob_no_block():
    o = _make(); o._t_bt = None
    o._opened_ob_epoch['X'] = 1000
    _check(o._ob_epoch_already_opened('X') is False, 'немає валідного OB → не блок')
    # mark без OB — no-op.
    o._mark_ob_epoch_opened('Y')
    _check('Y' not in o._opened_ob_epoch, 'mark без OB — no-op')
    print('✓ немає 1H-OB → не блокує, mark no-op')


def test_case_insensitive():
    o = _make(); o._t_bt = 1000
    o._mark_ob_epoch_opened('gramusdt')
    _check(o._ob_epoch_already_opened('GRAMUSDT') is True, 'ключ незалежний від регістру')
    print('✓ символ незалежний від регістру')


if __name__ == '__main__':
    test_disabled_never_blocks()
    test_first_open_then_block_same_ob()
    test_new_ob_resets()
    test_no_ob_no_block()
    test_case_insensitive()
    print('\nУсі тести VOB-open-epoch пройдено ✅')
