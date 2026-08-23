"""Тест: ПАРАЛЕЛЬНИЙ ПРЕФЕТЧ барів (мінімізація періоду між перевірками).

Мережа займала ~53% циклу (~7 HTTP на монету: 3000 барів = 3 сторінки на TF).
Обробка лишається ПОСЛІДОВНОЮ (спільний стан/сигнали/БД), а завантаження —
паралельне: результат кладеться в `self._prefetch`, звідки його бере скан.
"""
import os, sys, types, importlib.util, threading, time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg
_al = types.ModuleType('detection.activity_log'); _al.log_activity = lambda *a, **k: None
sys.modules['detection.activity_log'] = _al

spec = importlib.util.spec_from_file_location('detection.smc_scanner',
                                              os.path.join(_ROOT, 'detection/smc_scanner.py'))
scmod = importlib.util.module_from_spec(spec)
sys.modules['detection.smc_scanner'] = scmod
spec.loader.exec_module(scmod)
SC = scmod.SMCScanner


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _MD:
    """Імітує біржу: кожен запит «коштує» 50мс мережевого очікування."""
    def __init__(self):
        self.calls = []
        self._lk = threading.Lock()

    def fetch_klines(self, symbol, limit=60, interval='1m'):
        time.sleep(0.05)
        with self._lk:
            self.calls.append((symbol, interval, limit))
        return [{'t': 1000 + i, 'p': 1.0, 'h': 1.0, 'l': 1.0, 'o': 1.0, 'v': 1.0}
                for i in range(limit)]


def _mk():
    o = SC.__new__(SC)
    o._settings = {'timeframe': '15m', 'volumized_timeframe': '5m',
                   'ob_filter_timeframe': '1h', 'pd_zone_timeframe': '1h',
                   'use_volumized_ob': True}
    o._prefetch = {}
    return o


def test_specs_match_what_scan_consumes():
    o = _mk()
    specs = set(o._prefetch_specs())
    _check(('15m', scmod.KLINES_LIMIT) in specs, 'головний TF у префетчі')
    _check(('5m', 3000) in specs, 'Volumized TF у префетчі')
    _check(('1h', 700) in specs, '1H OB-стан у префетчі')
    _check(('1h', 3000) in specs, 'TF PD-зони у префетчі')
    print('✓ префетч тягне рівно ті набори, що споживає скан')


def test_volumized_off_not_prefetched():
    o = _mk()
    o._settings['use_volumized_ob'] = False
    o._settings['vob_alert_enabled'] = False
    _check(('5m', 3000) not in set(o._prefetch_specs()),
           'вимкнений Volumized НЕ вантажимо (економія мережі)')
    print('✓ вимкнена функція не тягне бари')


def test_prefetch_is_parallel_and_used():
    o = _mk()
    md = _MD()
    syms = [f'S{i}USDT' for i in range(8)]
    specs = [('15m', 100), ('5m', 100)]        # 16 запитів × 50мс = 0.8с послідовно
    t0 = time.time()
    o._prefetch_klines(md, syms, specs)
    el = time.time() - t0
    _check(len(o._prefetch) == 16, f'усі 16 наборів завантажено (маємо {len(o._prefetch)})')
    _check(el < 0.5, f'паралельно швидше за послідовні 0.8с (вийшло {el:.2f}с)')

    # Скан бере з префетчу — БЕЗ мережі.
    before = len(md.calls)
    kl = o._pf_klines(md, 'S0USDT', '15m', 100)
    _check(kl and len(kl) == 100, 'бари отримано з префетчу')
    _check(len(md.calls) == before, 'мережу НЕ смикали (взяли з кешу)')
    # Повторний виклик того самого — префетч спожито, іде звичайний запит.
    kl2 = o._pf_klines(md, 'S0USDT', '15m', 100)
    _check(kl2 and len(md.calls) == before + 1, 'фолбек на прямий запит працює')
    print('✓ префетч паралельний, скан читає з нього, фолбек не зламаний')


def test_missing_prefetch_falls_back():
    o = _mk(); md = _MD()
    kl = o._pf_klines(md, 'NOPEUSDT', '1h', 700)
    _check(kl and len(kl) == 700, 'без префетчу — звичайний запит')
    _check(md.calls == [('NOPEUSDT', '1h', 700)], 'запит із правильними параметрами')
    print('✓ монета поза префетчем довантажується коректно')


if __name__ == '__main__':
    test_specs_match_what_scan_consumes()
    test_volumized_off_not_prefetched()
    test_prefetch_is_parallel_and_used()
    test_missing_prefetch_falls_back()
    print('\nУсі тести префетчу пройдено ✅')
