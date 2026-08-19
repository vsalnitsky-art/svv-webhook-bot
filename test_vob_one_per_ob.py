"""«1 VOB на 1H OB» (vob_one_per_ob) — regression tests.

Strategy: one signal = one 1H OB + one 5m VOB. The 1H OB must be NEW (not the
one already present when we started watching). `_vob_epoch_decision(seen, bt)`:
  'skip'     — no valid 1H OB (bt None) → don't fire;
  'baseline' — first sight (seen None): record the EXISTING OB, DON'T fire (it's
               not "new"); counting starts from the NEXT OB;
  'used'     — same OB already baselined/consumed → ignore;
  'fire'     — a NEW 1H OB (different bar_time) → allowed to signal.
Variant (A): the epoch is consumed only when a signal actually fires.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smc_scanner_vob1_test", os.path.join(_HERE, "detection", "smc_scanner.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
D = _m.SMCScanner._vob_epoch_decision


def test_no_ob_skip():
    assert D(None, None) == 'skip'
    assert D(1000, None) == 'skip'


def test_first_sight_baselines_existing_ob():
    # pre-existing OB must NOT fire — it's baselined
    assert D(None, 1000) == 'baseline'


def test_same_ob_used():
    assert D(1000, 1000) == 'used'


def test_new_ob_fires():
    assert D(1000, 2000) == 'fire'


def test_new_ob_any_direction_fires():
    # bar_time changes on ANY new OB (even same-direction) → fire
    assert D(1000, 1500) == 'fire'


def test_flow_no_phantom_on_existing_then_one_per_new():
    """Existing OB=1000 → baseline (no signal). New OB=2000 → fire (one).
    Same OB 2000 → used (ignore). New OB=3000 → fire again."""
    seen = None
    # first sight of existing OB
    assert D(seen, 1000) == 'baseline'
    seen = 1000                                   # baselined
    # more 5m VOBs on the SAME (existing) OB → still used, never fires
    assert D(seen, 1000) == 'used'
    # NEW OB appears → the ONE signal
    assert D(seen, 2000) == 'fire'
    seen = 2000                                   # consumed on fire
    assert D(seen, 2000) == 'used'                # rest of this OB ignored
    # next NEW OB → one more
    assert D(seen, 3000) == 'fire'


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
