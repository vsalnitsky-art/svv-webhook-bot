"""Тест: 🔁 повторна перевірка фільтрів на ВИХОДІ з Черги-4.

Монета може стояти в черзі годинами (TTL 15+ год). За цей час 1H-OB може
перевернутись, прогноз/Decision — змінитись. Перед відкриттям проганяємо ТОЙ
САМИЙ ланцюг `_signal_allowed`, але СТРОГО (at_intake=False).
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
if 'pybit' not in sys.modules:
    for n in ('pybit', 'pybit.unified_trading'):
        m = types.ModuleType(n); sys.modules[n] = m
    sys.modules['pybit.unified_trading'].HTTP = object

_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_load('detection.signal_labels', 'detection/signal_labels.py')
_load('detection.setup_grader', 'detection/setup_grader.py')
ffmod = _load('detection.fuel_filter', 'detection/fuel_filter.py')
FF = ffmod.FuelFilterDaemon


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _Scanner:
    def __init__(self, res): self.res = res; self.calls = []
    def _signal_allowed(self, sym, side, at_intake=False):
        self.calls.append((sym, side, at_intake))
        return self.res


def _ff(scanner):
    o = FF.__new__(FF)
    o._get_tm = (lambda: types.SimpleNamespace(scanner=scanner)) if scanner else None
    return o


def test_passes_when_filters_ok():
    sc = _Scanner((True, '', 'OB:✓ Прогноз:✓'))
    ok, why = _ff(sc)._q4_recheck_filters('BTCUSDT', 'LONG')
    _check(ok is True and why == '', 'фільтри пройдені → відкриття дозволено')
    _check(sc.calls == [('BTCUSDT', 'LONG', False)],
           'перевірка СТРОГА (at_intake=False), а не як на інтейку')
    print('✓ фільтри пройдені → дозвіл, перевірка строга')


def test_blocks_and_reports_reason():
    sc = _Scanner((False, 'OB-фільтр заблокував', 'OB(1h):✗'))
    ok, why = _ff(sc)._q4_recheck_filters('ETHUSDT', 'SHORT')
    _check(ok is False, 'фільтр проти → відкриття НЕ дозволено')
    _check('OB-фільтр' in why and 'OB(1h):✗' in why,
           f'причина + розклад у тексті (маємо: {why})')
    print('✓ фільтр проти → блок із конкретною причиною для логу')


def test_no_scanner_does_not_block():
    ok, why = _ff(None)._q4_recheck_filters('XRPUSDT', 'LONG')
    _check(ok is True, 'немає сканера → не блокуємо торгівлю')
    print('✓ відсутній сканер не блокує')


def test_scanner_error_does_not_block():
    class _Boom:
        def _signal_allowed(self, *a, **k): raise RuntimeError('boom')
    ok, why = _ff(_Boom())._q4_recheck_filters('SOLUSDT', 'LONG')
    _check(ok is True, 'помилка перевірки НЕ повинна блокувати відкриття')
    print('✓ помилка перевірки безпечна (fail-open)')


def test_default_setting_is_on():
    _check(ffmod.DEFAULT_SETTINGS.get('queue4_recheck_filters') is True,
           'за замовчуванням УВІМКНЕНО')
    print('✓ тумблер за замовчуванням увімкнено')


if __name__ == '__main__':
    test_passes_when_filters_ok()
    test_blocks_and_reports_reason()
    test_no_scanner_does_not_block()
    test_scanner_error_does_not_block()
    test_default_setting_is_on()
    print('\nУсі тести повторної перевірки Черги-4 пройдено ✅')
