"""🎯 «Вихід на кульмінації (Виснаження)» для угод ШІ (вимога 24.11).

Запуск: python3 test_ai_exh_exit.py
"""
import importlib.util
import os
import sys
import types

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
_TMSRC = open(os.path.join(_ROOT, 'detection', 'trade_manager.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'), encoding='utf-8').read()
_BRAIN = open(os.path.join(_ROOT, 'detection', 'ai_brain.py'), encoding='utf-8').read()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for _n in ('pybit', 'pybit.unified_trading'):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules['pybit.unified_trading'].HTTP = object
_pkg = types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg
_load('detection.signal_labels', 'detection/signal_labels.py')
_load('detection.setup_grader', 'detection/setup_grader.py')
_lg = types.ModuleType('detection.activity_log')
_LOG = []
_lg.log_activity = lambda *a, **k: _LOG.append(a)
sys.modules['detection.activity_log'] = _lg
_db = types.ModuleType('storage.db_operations')
_db.get_db = lambda: None
_st = types.ModuleType('storage')
_st.__path__ = [os.path.join(_ROOT, 'storage')]
sys.modules.setdefault('storage', _st)
sys.modules['storage.db_operations'] = _db
_ff = types.ModuleType('detection.fuel_filter')
_ff.get_fuel_filter = lambda: None
sys.modules['detection.fuel_filter'] = _ff
T = _load('detection.trade_manager', 'detection/trade_manager.py')
A = _load('detection.ai_signals', 'detection/ai_signals.py')
_AI_S = {}
A.ai_settings_now = lambda: dict(A.AI_DEFAULTS, **_AI_S)


def _tm(use_exh=False):
    tm = object.__new__(T.TradeManager)
    tm._settings = dict(T.DEFAULT_SETTINGS, use_exh_exit=use_exh)
    tm._persist_book = lambda is_shadow: None
    tm._view = {'pct': 10, 'atr': 2.0, 'bar_hi': 112.0, 'bar_lo': 105.0, 'climax': 'LONG',
                't': 1, 'price': 110.0}
    tm._spring_fire_view = lambda symbol, fetch=True: dict(tm._view)
    tm.closed = []

    def _close(symbol, pos, px, is_shadow, hit):
        tm.closed.append(hit)
        return True
    tm._signal_exit_close = _close
    return tm


def _pos():
    return {'side': 'LONG', 'entry_price': 100.0, 'opened_by': 'ai', 'ai_owned': True,
            'spring_open_tgt': 120.0}


def test_defaults_and_settings():
    assert A.AI_DEFAULTS['exh_exit'] is True
    assert A.ai_settings_norm({'exh_exit': False})['exh_exit'] is False
    assert A.ai_settings_norm({})['exh_exit'] is True
    _AI_S.clear()
    assert T.TradeManager._ai_exh_on() is True
    _AI_S['exh_exit'] = False
    assert T.TradeManager._ai_exh_on() is False
    _AI_S.clear()


def test_ai_trade_is_led_without_the_global_toggle():
    tm = _tm(use_exh=False)
    pos = _pos()
    # звичайна угода при вимкненому глобальному тумблері — правило мовчить
    assert tm._exh_exit_tick('X', pos, 110.0, True, 109.5, 110.5) is False
    assert 'exh_trail' not in pos
    # угода ШІ (force) — озброюємо від вершини свічки 112 на 0.5·ATR(2) → стоп 111
    _LOG.clear()
    assert tm._exh_exit_tick('X', pos, 110.0, True, 109.5, 110.5, force=True) is False
    st = pos.get('exh_trail')
    assert st and st['best_px'] == 112.0 and st['stop_px'] == 111.0, st
    assert _LOG and '🤖 ШІ' in _LOG[-1][2]
    # відкат нижче стопа → закриття з причиною exh_exit
    assert tm._exh_exit_tick('X', pos, 110.8, True, 110.8, 111.2, force=True) is True
    assert tm.closed and tm.closed[-1][0] == 'exh_exit' and '🤖 ШІ' in tm.closed[-1][1]


def test_wiring_in_both_monitors():
    assert _TMSRC.count('self._exh_exit_tick(symbol, pos, current_price,') == 2
    assert _TMSRC.count("(self._ai_exh_on() if _ai else True)") == 2
    assert _TMSRC.count('force=_ai)') == 2
    # трейл SBR для угод ШІ і далі не діє
    assert _TMSRC.count('if not _manual and not _ai and self._sbr_trail_tick(') == 2


def test_ui_and_rules():
    assert 'id="aip-exh-exit"' in _HTML
    assert "ck('aip-exh-exit', a.exh_exit !== false)" in _HTML
    assert "exh_exit: !!(g('aip-exh-exit') || {}).checked" in _HTML
    assert '🎯 Вихід на кульмінації (налаштування ШІ' in _BRAIN


if __name__ == '__main__':
    n = 0
    for k, f in list(globals().items()):
        if k.startswith('test_') and callable(f):
            f()
            n += 1
    print(f'OK {n}')
