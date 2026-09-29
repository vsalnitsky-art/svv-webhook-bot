"""🐞 ПОТІК ІСТОРИЧНИХ CHoCH+BOS + ВХІД ЗА МІСЯЧНИМ РІВНЕМ (кейс 29.09).

Лог 20:38:48–20:39:39: по RENDER/SKY/VVV/PONS/QNT/STABLE/M ~35 «свіжих
сигналів» на монету за кілька секунд, угоди відкрито за рівнями давніх подій
(RENDER $1.4250 при ціні $1.93 → фальшивий «+35%»).
Три діри:
  1) ранній вихід (`не має подій` / вимкнений тумблер) зараховував перший скан
     без водяного знаку → наступний прохід вважав новою ВСЮ історію;
  2) у режимах 'choch_bos' / 'choch_or_bos' BOS-гілка не мала гейта
     «лише найновіша подія напрямку»;
  3) ціна входу = рівень події, навіть коли він далеко від ринку.
"""
import importlib.util
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg

spec = importlib.util.spec_from_file_location(
    'smc_scanner_flood_test', os.path.join(_HERE, 'detection', 'smc_scanner.py'))
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)
S = sc.SMCScanner


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


def _mk(mode='choch_bos', enabled=True):
    s = S.__new__(S)
    s._settings = {'alert_mode': mode, 'choch_alerts_enabled': enabled,
                   'recency_minutes': 0}
    s._seen_events, s._first_scan_done, s._event_hwm = {}, {}, {}
    s._pending_choch = {}
    s.fired = []
    s._send_alert = lambda sym, ev, mode, choch_event=None: s.fired.append(ev)
    s._htf_allows = lambda sym, d: True
    s._dedup_allows = lambda sym, d: True
    return s


def _history(n=12):
    """Чергування CHoCH→BOS в обидва боки, бари щогодини в минулому."""
    now = int(time.time() * 1000)
    evs = []
    for i in range(n):
        d = 'bull' if (i // 2) % 2 == 0 else 'bear'
        tag = 'CHoCH' if i % 2 == 0 else 'BOS'
        t = now - (n - i) * 3600_000
        evs.append({'from_t': t - 60_000, 'to_t': t, 'dir': d, 'tag': tag,
                    'level': 100 + i})
    return evs


def _res(evs):
    return {'internal': {'events': evs}}


def test_toggle_off_then_on_does_not_dump_history():
    s = _mk(enabled=False)
    evs = _history()
    s._process_alerts('RENDERUSDT', _res(evs))           # тумблер вимкнено
    s._settings['choch_alerts_enabled'] = True
    s._process_alerts('RENDERUSDT', _res(evs))           # увімкнули
    _check(not s.fired, f'історія фаєрнула {len(s.fired)} сигналів')
    print('✓ увімкнення тумблера не вивалює історію')


def test_empty_first_scan_does_not_dump_history():
    s = _mk()
    s._process_alerts('SKYUSDT', _res([]))               # бари не прийшли
    s._process_alerts('SKYUSDT', _res(_history()))
    _check(not s.fired, f'історія фаєрнула {len(s.fired)} сигналів')
    print('✓ порожній перший прохід не обнуляє водяний знак')


def test_bos_batch_fires_at_most_newest_per_direction():
    for mode in ('choch_bos', 'choch_or_bos'):
        s = _mk(mode)
        s._first_scan_done['VVVUSDT'] = True             # стан «з діркою»
        s._process_alerts('VVVUSDT', _res(_history(20)))
        bos = [e for e in s.fired if e['tag'] == 'BOS']
        _check(len(bos) <= 2, f'{mode}: {len(bos)} BOS-сигналів за прохід')
        dirs = [e['dir'] for e in bos]
        _check(len(dirs) == len(set(dirs)), f'{mode}: дубль напрямку {dirs}')
    print('✓ пачка подій → щонайбільше один BOS-сигнал на напрямок')


def test_fresh_bos_still_fires():
    s = _mk()
    evs = _history()
    s._process_alerts('QNTUSDT', _res(evs))              # перший скан (тихо)
    now = int(time.time() * 1000)
    evs2 = evs + [
        {'from_t': now - 120_000, 'to_t': now - 60_000, 'dir': 'bear', 'tag': 'CHoCH', 'level': 90},
        {'from_t': now - 50_000, 'to_t': now - 10_000, 'dir': 'bear', 'tag': 'BOS', 'level': 89}]
    s._process_alerts('QNTUSDT', _res(evs2))
    _check(len(s.fired) == 1 and s.fired[0]['level'] == 89, s.fired)
    print('✓ справжній свіжий CHoCH+BOS і далі дає сигнал')


def test_entry_level_guard():
    _check(sc.entry_level_ok(1.93, 1.929), 'рівень біля ціни')
    _check(not sc.entry_level_ok(1.425, 1.929), 'RENDER: рівень −26% від ціни')
    _check(not sc.entry_level_ok(17.179, 27.07), 'VVV')
    _check(sc.entry_level_ok(1.425, 0), 'немає живої ціни → fail-open')
    src = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()
    body = src.split('def _send_alert')[1][:6000]
    _check('entry_level_ok(entry_price, _live)' in body, 'гейт у _send_alert')
    print('✓ рівень далеко від ринку → вхід за живою ціною')


if __name__ == '__main__':
    _fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
