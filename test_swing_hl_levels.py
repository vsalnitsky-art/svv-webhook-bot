"""Swing High/Low display — regression tests (TradingView-faithful).

Guards the fix for the ATOMUSDT case: the chart drew "Strong High 1.3937 /
Weak Low 1.3252" — WRONG levels (intermediate pivots, both below price) and
WRONG names (always Strong High / Weak Low). TradingView drew "Weak High 1.566
(top) / Strong Low 1.33 (bottom)".

Root cause: the levels were the last raw HH/LH & HL/LL pivots; and the labels
were hard-coded. Fix:
  - levels = trailing swing extremes of the dealing range (_swing_trailing_range,
    the SAME source as the PD-zone %, so the lines match the badge);
  - names by swing trend (_swing_hl_labels), exactly like LuxAlgo/TradingView.

Loaded via importlib by file path — smc_scanner.py has only stdlib top-level
imports (heavy deps are imported lazily inside methods), so the real static
methods run without the detection package __init__.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smc_scanner_under_test",
    os.path.join(_HERE, "detection", "smc_scanner.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
S = _mod.SMCScanner

trailing = S._swing_trailing_range
labels = S._swing_hl_labels
pd_pct = S._compute_pd_pct


def _kl(seq):
    """Build klines from (high, low) pairs; close = midpoint."""
    out = []
    for h, l in seq:
        out.append({'t': 0, 'o': l, 'h': h, 'l': l, 'p': (h + l) / 2, 'v': 1.0})
    return out


# ATOM-like swing structure: swing HIGH pivot at 1.566, swing LOW pivot at 1.33.
# Bars after each pivot never exceed/undercut them → trailing range = 1.33..1.566.
ATOM_PIVOTS = [
    {'idx': 5,  'price': 1.33,  'type': 'LL'},   # swing low (Strong Low in TV)
    {'idx': 60, 'price': 1.566, 'type': 'HH'},   # swing high (Weak High in TV)
]
# 80 bars: keep highs <= 1.566 and lows >= 1.33 so trailing doesn't drift.
ATOM_KL = _kl([(1.40, 1.33)] * 6 + [(1.50, 1.40)] * 54 + [(1.566, 1.47)] +
              [(1.52, 1.47)] * 19)


def test_trailing_range_is_dealing_range_not_intermediate():
    top, bottom = trailing(ATOM_KL, ATOM_PIVOTS)
    assert top is not None and bottom is not None
    assert abs(top - 1.566) < 1e-6, top       # real swing high, NOT 1.3937
    assert abs(bottom - 1.33) < 1e-6, bottom  # real swing low,  NOT 1.3252


def test_pd_pct_matches_badge_60_percent():
    # Equilibrium 60% (1H) on the screenshot → confirms the shared range.
    pct = pd_pct(ATOM_KL, ATOM_PIVOTS, current_price=1.472)
    assert pct is not None and abs(pct - 60.0) < 1.5, pct


def test_labels_bullish_like_tradingview():
    # Bullish swing → Weak High on top, Strong Low on bottom (ATOM case).
    hi, lo = labels(1)
    assert hi == 'Weak High' and lo == 'Strong Low', (hi, lo)


def test_labels_bearish():
    hi, lo = labels(-1)
    assert hi == 'Strong High' and lo == 'Weak Low', (hi, lo)


def test_labels_undefined_defaults_bullish():
    hi, lo = labels(0)
    assert hi == 'Weak High' and lo == 'Strong Low', (hi, lo)


def test_trailing_extends_above_pivot_high():
    # If price creeps above the last swing-high pivot, the range grows to it.
    kl = _kl([(1.60, 1.33)] * 3 + [(1.62, 1.50)] * 3)  # later high 1.62 > pivot
    piv = [{'idx': 0, 'price': 1.33, 'type': 'LL'},
           {'idx': 1, 'price': 1.55, 'type': 'HH'}]
    top, bottom = trailing(kl, piv)
    assert top >= 1.62 - 1e-9, top
    assert abs(bottom - 1.33) < 1e-6, bottom


def test_no_pivots_returns_none():
    assert trailing(ATOM_KL, []) == (None, None)


def test_only_one_side_returns_none():
    top, bottom = trailing(ATOM_KL, [{'idx': 5, 'price': 1.33, 'type': 'LL'}])
    assert (top, bottom) == (None, None)


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
