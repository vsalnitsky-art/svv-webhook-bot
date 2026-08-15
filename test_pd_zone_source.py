"""PD-zone SINGLE SOURCE OF TRUTH — regression tests.

Guards the fix for the banner/badge contradiction: the chart badge showed
"Discount 17% (1H)" while the decision banner said "PD Premium 89%" for the
SAME symbol at the SAME time. Root cause: the banner recomputed Premium/Discount
from a 20-bar 15m rolling window instead of the authoritative dealing-range %
(pd_zone_timeframe, default 1H) that the badge and the PD filter use.

The fix: evaluate_entry / _score_pd_zone accept `pd_pct` (+ thresholds) and,
when given, classify THAT value instead of range_high/range_low. These tests
verify the banner can never again disagree with the badge, and that the sign
of the PD contribution is correct for both sides.

Loaded via importlib by file path so it runs without the heavy `detection`
package __init__ (pybit / binance) being importable.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "position_evaluator_under_test",
    os.path.join(_HERE, "detection", "position_evaluator.py"))
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)
_score_pd_zone = pe._score_pd_zone

W = 15.0            # PD weight
PREM, DISC = 75.0, 25.0   # get_pd_thresholds() defaults (premium_min, discount_max)


def test_discount_matches_badge_and_penalises_short():
    """The screenshot case: badge = Discount 17% (1H). The banner must agree,
    and Discount must be a NEGATIVE for a SHORT (bad place to short)."""
    score, label = _score_pd_zone('SHORT', 63089.0, None, None, W,
                                  pd_pct=17.0, premium_min=PREM, discount_max=DISC)
    assert 'Discount' in label and '17%' in label, label
    assert 'Premium' not in label, label
    assert score < 0, f"Discount must penalise SHORT, got {score}"


def test_discount_rewards_long():
    score, label = _score_pd_zone('LONG', 63089.0, None, None, W,
                                  pd_pct=17.0, premium_min=PREM, discount_max=DISC)
    assert 'Discount' in label, label
    assert score > 0, f"Discount must reward LONG, got {score}"


def test_premium_rewards_short_penalises_long():
    s_short, l_short = _score_pd_zone('SHORT', 0, None, None, W,
                                      pd_pct=89.0, premium_min=PREM, discount_max=DISC)
    s_long, _ = _score_pd_zone('LONG', 0, None, None, W,
                               pd_pct=89.0, premium_min=PREM, discount_max=DISC)
    assert 'Premium' in l_short and '89%' in l_short, l_short
    assert s_short > 0 and s_long < 0, (s_short, s_long)


def test_equilibrium_is_neutral():
    score, label = _score_pd_zone('SHORT', 0, None, None, W,
                                  pd_pct=50.0, premium_min=PREM, discount_max=DISC)
    assert 'Equilibrium' in label, label
    assert abs(score) <= W * 0.2, score


def test_pd_pct_overrides_misleading_range():
    """Even when a (wrong) short-window range would say Premium, an explicit
    pd_pct=17 must win — the banner follows the dealing range, not the window."""
    # range 62400..63200, price 63089 → ~86% (the OLD buggy 'Premium').
    score, label = _score_pd_zone('SHORT', 63089.0, 63200.0, 62400.0, W,
                                  pd_pct=17.0, premium_min=PREM, discount_max=DISC)
    assert 'Discount' in label, label
    assert score < 0, score


def test_fallback_when_no_pd_pct():
    """Health path passes no pd_pct → graceful fallback to range_high/low
    with the classic Fib levels (still produces a valid label)."""
    score, label = _score_pd_zone('SHORT', 63089.0, 63200.0, 62400.0, W)
    assert label.startswith('PD '), label
    # 63089 in 62400..63200 → top of window → Premium under the Fib fallback.
    assert 'Premium' in label, label


def test_unavailable_when_nothing_provided():
    score, label = _score_pd_zone('LONG', 100.0, None, None, W)
    assert score == 0.0
    assert 'недоступний' in label, label


def test_threshold_config_respected():
    """pd_pct=30 is Discount under default 25/75? No — 30 > 25, so Equilibrium.
    Confirms the banner uses the SAME configurable thresholds as the badge,
    not the hard-coded Fib 38.2 (which would wrongly call 30 'Discount')."""
    _, label = _score_pd_zone('LONG', 0, None, None, W,
                              pd_pct=30.0, premium_min=PREM, discount_max=DISC)
    assert 'Equilibrium' in label, label


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
