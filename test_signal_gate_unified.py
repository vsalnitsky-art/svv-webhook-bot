"""Unified signal gate — regression tests.

Guards the fix for the ASTERUSDT case: a Volumized-OB (5m) alert opened a SHORT
that the user's Forecast Filter (Require 1H+4H match, AND, Сильний ≥66%) should
have blocked — because the VOB-alert path called on_signal() DIRECTLY, bypassing
the OB/PD/Forecast filters that only lived inside _send_alert (the CHoCH path).

The fix: a single gate SMCScanner._signal_allowed(symbol, side) applies all
three filters (each respecting its own toggle) and is called from BOTH the
CHoCH path and the Volumized-OB-alert path. So "if a filter is enabled, it
does its job" — on every entry, no exceptions.

_signal_allowed reads only self._settings + the three filter helpers, so we test
the REAL orchestration with a lightweight stub `self` (no heavy deps / __init__).
"""
import importlib.util
import os
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smc_scanner_gate_test",
    os.path.join(_HERE, "detection", "smc_scanner.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
gate = _mod.SMCScanner._signal_allowed  # unbound; call gate(self, symbol, side)


def _self(settings, ob_ok=True, pd_ok=True, fc_ok=True, str_ok=True, poc_ok=True):
    ns = types.SimpleNamespace()
    ns._settings = settings
    # Filter helpers accept **kw so the at_intake kwarg the gate passes is absorbed.
    ns._ob_filter_allows = lambda sym, side, **kw: ob_ok
    ns._pd_zone_filter_allows = lambda sym, side, **kw: pd_ok
    ns._forecast_filter_allows = lambda sym, side, **kw: fc_ok
    ns._forecast_strength_allows = lambda sym, side, **kw: str_ok
    ns._poc_filter_allows = lambda sym, side, **kw: poc_ok
    # Value helpers used by the detail breakdown.
    # `_ob_state_label` живить розклад «OB(1h лише CHoCH BEARISH/BOS):✗» —
    # рішення воно НЕ ухвалює, тому тут порожній стан.
    ns._ob_state_label = lambda sym: 'BULLISH/CHoCH'
    ns._forecast_pair = lambda sym: ('—', '—')
    ns.get_pd_pct = lambda sym: None
    ns._decision_gate = lambda sym, side, at_intake=False: (True, 'LONG 80%')
    return ns


def test_all_filters_off_allows():
    ok, reason, _ = gate(_self({}), 'X', 'SHORT')
    assert ok is True and reason == ''


def test_forecast_blocks_short_against_long_forecast():
    # The ASTER case: forecast filter ON, forecast disagrees → block.
    s = {'forecast_1h_filter_enabled': True, 'forecast_4h_filter_enabled': True}
    ok, reason, _ = gate(_self(s, fc_ok=False), 'ASTERUSDT', 'SHORT')
    assert ok is False
    assert 'Forecast' in reason, reason


def test_forecast_toggle_respected_when_off():
    # forecast helper would block, but neither TF enabled → filter skipped.
    ok, reason, _ = gate(_self({}, fc_ok=False), 'X', 'SHORT')
    assert ok is True and reason == ''


def test_ob_filter_blocks_when_enabled():
    s = {'ob_filter_enabled': True}
    ok, reason, _ = gate(_self(s, ob_ok=False), 'X', 'SHORT')
    assert ok is False and 'OB' in reason, reason


def test_ob_toggle_respected_when_off():
    ok, reason, _ = gate(_self({'ob_filter_enabled': False}, ob_ok=False), 'X', 'LONG')
    assert ok is True and reason == ''


def test_pd_blocks_unconditionally_helper_owns_toggle():
    # PD helper is called every time (it checks use_pd_zone_filter internally);
    # here the stub says "blocked" → gate must block with the PD reason.
    ok, reason, _ = gate(_self({}, pd_ok=False), 'X', 'LONG')
    assert ok is False and 'PD' in reason, reason


def test_priority_ob_before_pd_before_forecast():
    # All three would block; OB is checked first → its reason wins.
    s = {'ob_filter_enabled': True,
         'forecast_1h_filter_enabled': True}
    ok, reason, _ = gate(_self(s, ob_ok=False, pd_ok=False, fc_ok=False), 'X', 'SHORT')
    assert ok is False and 'OB' in reason, reason


def test_all_enabled_and_passing_allows():
    s = {'ob_filter_enabled': True, 'forecast_1h_filter_enabled': True}
    ok, reason, _ = gate(_self(s, ob_ok=True, pd_ok=True, fc_ok=True), 'X', 'LONG')
    assert ok is True and reason == ''


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
