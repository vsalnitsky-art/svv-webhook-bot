"""⏰ Будильник SBR вмикає головну кнопку → бот ОДРАЗУ працює (вимога 01.11).

Дослівно: «Проконтролюй, щоб алгоритм будильника умикнувши кнопку, запустив
всю роботу алгоритму відповідно увімкненої кнопки, а не лише засвітив вмикання
кнопки, а більше нічого не відбуватиметься.»

**Корінь:** поки кнопку вимкнено, шляхи сигналів ВІДХИЛЯЛИ сигнал і при цьому
вважали його ОПРАЦЬОВАНИМ: CHoCH «спалював» дедуп тренду (до фліпу 1H-OB), а
VOB-блок ставав опрацьованим. Тож після вмикання будильником бот мовчав би до
нових подій. Тепер `SMCScanner._dir_paused` питають ДО бази — сигнал на паузі
не опрацьовано, і щойно кнопка увімкнена, він іде в роботу.
"""
import ast
import os
import sys
import types
import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules['pybit.unified_trading'].HTTP = object

if 'detection' not in sys.modules:
    _pkg = types.ModuleType('detection')
    _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
    sys.modules['detection'] = _pkg
_pkg = sys.modules['detection']


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DG = _load('detection.direction_gate', 'detection/direction_gate.py')
_pkg.direction_gate = DG
SC = _load('smc_scanner_alarm', 'detection/smc_scanner.py')
SC._DG_CACHE = DG
S = SC.SMCScanner
SRC = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'), encoding='utf-8').read()

_LOG = []


def _install(settings):
    log = types.ModuleType('detection.activity_log')
    log.log_activity = lambda sym, ev, detail='', **k: _LOG.append((sym, ev, detail))
    sys.modules['detection.activity_log'] = log
    _pkg.activity_log = log

    class _TM:
        def get_settings(self):
            return dict(settings)

        def update_settings(self, upd):
            settings.update(upd)

    tm = _TM()
    mod = types.ModuleType('detection.trade_manager')
    mod.get_trade_manager = lambda: tm
    sys.modules['detection.trade_manager'] = mod
    _pkg.trade_manager = mod
    return tm


def _check(c, m):
    if not c:
        raise AssertionError(m)


def test_paused_choch_does_not_burn_dedup_and_resumes_after_alarm():
    _LOG.clear()
    st = {DG.KEY_LONG: False, DG.KEY_SHORT: False}
    _install(st)
    ns = types.SimpleNamespace(_last_signal_dir={}, _dir_pause_logged=set(),
                               _settings={})
    ev = {'dir': 'bull', 'level': 100.0}
    S._send_alert(ns, 'BTCUSDT', ev, 'choch')
    S._send_alert(ns, 'BTCUSDT', ev, 'choch')
    _check('BTCUSDT' not in ns._last_signal_dir,
           'на паузі дедуп тренду НЕ витрачається')
    rej = [x for x in _LOG if x[1] == 'rejected']
    _check(len(rej) == 1, f'анти-флуд: один рядок за паузу, а не {len(rej)}')
    ok, note = DG.live_enable('LONG', source='SBR alarm')
    _check(ok and note == 'увімкнено' and st[DG.KEY_LONG] is True, 'кнопку увімкнено')
    _check(S._dir_paused(ns, 'BTCUSDT', 'LONG') == '',
           'після будильника LONG більше не на паузі')
    _check(S._dir_paused(ns, 'BTCUSDT', 'SHORT') != '', 'SHORT лишився вимкненим')
    print('✓ CHoCH на паузі не спалює дедуп, будильник одразу відкриває напрямок')


def test_unknown_state_is_fail_open():
    mod = types.ModuleType('detection.trade_manager')
    mod.get_trade_manager = lambda: None
    sys.modules['detection.trade_manager'] = mod
    _pkg.trade_manager = mod
    ns = types.SimpleNamespace()
    _check(S._dir_paused(ns, 'X', 'LONG') == '', 'стан невідомий → не пауза')
    print('✓ невідомий стан кнопок не зупиняє бота')


def _fn(name):
    tree = ast.parse(SRC)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(SRC, n)
    raise AssertionError(name)


def test_every_signal_path_checks_pause_before_marking_processed():
    src = _fn('_send_alert')
    _check(src.index('_dir_paused') < src.index('self._last_signal_dir[symbol] = side_label'),
           'CHoCH: пауза ДО позначки дедупу')
    lv = _fn('_liq_vob_check')
    _check(lv.index('_dir_paused') < lv.index("self._vob_seen_add(self._liqvob_seen[symbol], side, ft)\n"
                                             "            try:"),
           '💧 liq_vob: пауза ДО опрацювання блоку')
    oa = _fn('_ob_alert_signal')
    _check(oa.index('_dir_paused') < oa.index('self._signal_allowed('), '🆕 Новий OB: пауза першою')
    _check(SRC.count('SMCScanner._dir_paused(self') >= 5, 'усі шляхи VOB/CHoCH/OB під паузою')
    print('✓ кожен шлях сигналу питає паузу ДО того, як вважати сигнал опрацьованим')


if __name__ == '__main__':
    fns = [v for k, v in dict(globals()).items() if k.startswith('test_') and callable(v)]
    for f in fns:
        f()
    print(f'{len(fns)}/{len(fns)} passed')
