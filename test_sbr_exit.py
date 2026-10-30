"""🔔 Вихід із угоди по протилежному сигналу МММ-SBR 1H / 4H (вимога 30.10).

Дослівно: «Додай ще один варіант виходу із угоди. Вихід по протилежному
сигналу від 🔔 МММ-SBR 1Н або 4Н.»
Правило: після ВХОДУ на обраному TF зʼявився сигнал Session Band Reversal
протилежного боку → закриваємо. Сигнал, що був ДО входу, — не подія.
"""
import os, sys, types, importlib.util, threading

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules['pybit.unified_trading'].HTTP = object

_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_load('detection.signal_labels', 'detection/signal_labels.py')
_load('detection.setup_grader', 'detection/setup_grader.py')
_LOG = []
lg = types.ModuleType('detection.activity_log')
lg.log_activity = lambda sym, kind, text, **kw: _LOG.append(text)
sys.modules['detection.activity_log'] = lg
db = types.ModuleType('storage.db_operations')
db.get_db = lambda: types.SimpleNamespace(get_smc_ob_state=lambda sym, tf: None)
st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
sys.modules.setdefault('storage', st)
sys.modules['storage.db_operations'] = db
ffm = types.ModuleType('detection.fuel_filter')
ffm.get_fuel_filter = lambda: None
sys.modules['detection.fuel_filter'] = ffm
sbr = _load('detection.sbr_signals', 'detection/sbr_signals.py')
tmmod = _load('detection.trade_manager', 'detection/trade_manager.py')
TM = tmmod.TradeManager

with open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8') as f:
    _HTML = f.read()
with open(os.path.join(_ROOT, 'detection', 'trade_manager.py'), encoding='utf-8') as f:
    _TMSRC = f.read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _sig(side, close_t, n=1, score=87.0):
    return {'side': side, 'close_t': close_t, 'n': n, 'score': score, 'time_kyiv': '14:00'}


class _Eng:
    def __init__(self, snaps): self._s = snaps
    def snaps(self, symbol): return self._s.get(symbol, {})


class _TM(TM):
    def __init__(self, side='LONG', opened_at=10_000.0, **settings):
        self._lock = threading.RLock()
        self._signal_exit_at = {}
        self._mm_flat_since = {}
        self._mm_against_since = {}
        self.closed = []
        pos = {'symbol': 'ETHUSDT', 'side': side, 'entry_price': 100.0, 'opened_at': opened_at}
        self._positions = {'ETHUSDT': pos}
        self._shadow_positions = {}
        self._settings = {'use_sbr_exit': True}
        self._settings.update(settings)
    def _get_current_price(self, symbol): return 101.0
    def _close_position(self, symbol, price, reason): self.closed.append(('real', reason))
    def _close_shadow(self, symbol, price, reason): self.closed.append(('paper', reason))
    def _get_forecast_both(self, symbol): return {}
    def compute_decision(self, symbol, price): return {}


def _run(t, snaps):
    sbr._instance = _Eng({'ETHUSDT': snaps})
    _LOG.clear()
    pos = t._positions['ETHUSDT']
    return t._check_signal_exits('ETHUSDT', pos, 101.0, False)


# ── чиста функція ──────────────────────────────────────────────────────────
def test_only_opposite_signal_after_entry_counts():
    snaps = {'1h': {'tail': [_sig('SHORT', 9_000), _sig('LONG', 11_000)]},
             '4h': {'tail': [_sig('SHORT', 12_000, n=2)]}}
    hit = TM.sbr_exit_hit('LONG', 10_000, snaps, ('1h', '4h'))
    _check(hit and hit[0] == '4h' and hit[1]['close_t'] == 12_000, hit)
    _check(TM.sbr_exit_hit('LONG', 10_000, snaps, ('1h',)) is None,
           '1H SHORT був ДО входу — не подія; 1H LONG — за угодою')
    _check(TM.sbr_exit_hit('SHORT', 10_000, snaps, ('1h',))[1]['side'] == 'LONG', 'дзеркально')


def test_unknown_entry_time_never_closes():
    snaps = {'1h': {'tail': [_sig('SHORT', 99_999)]}}
    _check(TM.sbr_exit_hit('LONG', 0, snaps, ('1h',)) is None, 'час входу невідомий')
    _check(TM.sbr_exit_hit('', 10, snaps, ('1h',)) is None, 'бік невідомий')
    _check(TM.sbr_exit_hit('LONG', 10, {}, ('1h', '4h')) is None, 'немає даних')


def test_timeframe_selection():
    _check(TM.sbr_exit_tfs({'sbr_exit_1h': True, 'sbr_exit_4h': False}) == ('1h',), '1H')
    _check(TM.sbr_exit_tfs({'sbr_exit_1h': False, 'sbr_exit_4h': True}) == ('4h',), '4H')
    _check(TM.sbr_exit_tfs({'sbr_exit_1h': False, 'sbr_exit_4h': False}) == ('1h', '4h'),
           'обидва зняті → обидва')
    _check(TM.sbr_exit_tfs({}) == ('1h', '4h'), 'дефолт — обидва')


# ── правило в моніторі ─────────────────────────────────────────────────────
def test_rule_closes_on_new_opposite_signal():
    t = _TM(side='LONG', opened_at=10_000.0)
    closed = _run(t, {'4h': {'tail': [_sig('SHORT', 14_400 + 10_000)]}})
    _check(closed and t.closed == [('real', 'sbr_exit')], t.closed)
    _check(any('МММ-SBR 4H' in x and 'SHORT' in x for x in _LOG), _LOG)


def test_rule_respects_the_tf_choice_and_the_switch():
    t = _TM(side='LONG', sbr_exit_4h=False)
    _check(not _run(t, {'4h': {'tail': [_sig('SHORT', 20_000)]}}) and t.closed == [],
           'обрано лише 1H — сигнал 4H не закриває')
    t = _TM(side='LONG', use_sbr_exit=False)
    _check(not _run(t, {'1h': {'tail': [_sig('SHORT', 20_000)]}}), 'правило вимкнене')


def test_signal_before_entry_does_not_close():
    t = _TM(side='SHORT', opened_at=50_000.0)
    _check(not _run(t, {'1h': {'tail': [_sig('LONG', 40_000)]}}) and t.closed == [], t.closed)


def test_default_is_off_and_labels_exist_everywhere():
    _check(tmmod.DEFAULT_SETTINGS['use_sbr_exit'] is False, 'дефолт OFF')
    _check(_TMSRC.count("'sbr_exit':") >= 2, 'підпис причини в _close_detail і бейджі TM')
    for k in ("'sbr_exit': '🔔 SBR проти'", 'id="tm-use-sbr-exit"', 'id="tm-sbr-exit-1h"',
              'id="tm-sbr-exit-4h"', 'use_sbr_exit:', 'sbr_exit_1h:', 'sbr_exit_4h:'):
        _check(k in _HTML, k)


def test_engine_snaps_reads_memory_only():
    e = sbr.SbrEngine.__new__(sbr.SbrEngine)
    e._lock = threading.RLock()
    e._state = {'ETHUSDT': {'1h': {'tail': [1]}, '4h': None}}
    _check(e.snaps('ethusdt') == {'1h': {'tail': [1]}}, e.snaps('ethusdt'))
    _check(e.snaps('XXX') == {}, 'невідома монета')


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v(); print('✓', k); n += 1
    print(f'OK — {n} тестів')
