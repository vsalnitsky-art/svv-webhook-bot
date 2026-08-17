"""VOB-alert freshness gate — regression tests.

Requirement (user): a VOB signal must fire ONLY when a genuinely NEW, VALID
Volumized OB has JUST appeared — never a stale/phantom one — and as fast as
possible (no wait for bar close). A Volumized OB is only valid AFTER its swing
confirms (~volumized_swing_length bars after its candle), so the gate fires the
instant it becomes valid and skips OBs whose confirmation is too old.

`_vob_age_bars` = how many bars formed after the OB's formation_time. Small age
= just confirmed (fire); large age = stale (skip). Loaded via importlib (only
stdlib imports at module top).
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smc_scanner_vob_test", os.path.join(_HERE, "detection", "smc_scanner.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
age = _m.SMCScanner._vob_age_bars


def _klines(n, step_ms=900_000, start=1_000_000_000_000):
    return [{'t': start + i * step_ms} for i in range(n)]


def test_age_counts_bars_after_formation():
    kl = _klines(12)
    assert age(kl, kl[5]['t']) == 6      # bars 6..11 are after index 5


def test_fresh_ob_small_age():
    kl = _klines(12)
    assert age(kl, kl[-2]['t']) == 1     # just confirmed → age 1 (fresh)


def test_stale_ob_large_age():
    kl = _klines(30)
    assert age(kl, kl[0]['t']) == 29     # very old OB → large age (would be skipped)


def test_empty_and_bad_inputs():
    assert age([], 123) == 0
    assert age(_klines(5), 0) == 0
    assert age(_klines(5), None) == 0


def test_gate_decision_fresh_vs_stale():
    # Simulate the gate: swing_length=5 → max_age auto = 7.
    kl = _klines(20)
    max_age = 5 + 2
    # OB confirmed just now (age ~5) → fire
    fresh_ft = kl[-6]['t']            # 5 bars after
    assert age(kl, fresh_ft) <= max_age
    # OB from long ago (age 15) → skip
    stale_ft = kl[-16]['t']
    assert age(kl, stale_ft) > max_age


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
