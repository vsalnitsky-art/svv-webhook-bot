"""Independent filters — Min-strength & POC — regression tests.

User requirement: (1) «Мін. сила» must be a STANDALONE filter with its own
toggle (not coupled to 1H/4H direction); (2) a NEW standalone POC filter
(«краще LONG / краще SHORT») with the SAME params as the chart (market/TF/window,
defaults FUTURES/1H/3d) so its verdict matches the chart badge 1:1; (3) both
toggleable and part of the whole chain (`_signal_allowed`). The POC filter is
directional — it follows the whole filter direction (LONG below POC / SHORT
above POC), never a fixed single side.

We test the two gates with a lightweight stub `self` and fake `detection.*`
modules injected into sys.modules (avoids heavy package __init__).
"""
import importlib.util
import os
import sys
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smc_scanner_filters_test", os.path.join(_HERE, "detection", "smc_scanner.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
S = _m.SMCScanner


def _ensure_detection_pkg():
    if 'detection' not in sys.modules:
        pkg = types.ModuleType('detection'); pkg.__path__ = []
        sys.modules['detection'] = pkg


# ---------------- POC filter (uses compute_poc + price_vs_poc) ----------------
def _price_vs_poc(poc, price, side=None, tol_pct=0.05):
    """Mirror of detection.volume_profile.price_vs_poc semantics."""
    if poc is None or price is None or poc <= 0 or price <= 0:
        return {'ok': None, 'rel': None}
    dist = (price - poc) / poc * 100.0
    rel = 'at' if abs(dist) <= tol_pct else ('above' if price > poc else 'below')
    s = (side or '').upper()
    ok = (rel == 'below') if s == 'LONG' else ((rel == 'above') if s == 'SHORT' else None)
    return {'ok': ok, 'rel': rel, 'dist_pct': dist}


def _inject_poc(poc_value, ok=True, last_close=None):
    _ensure_detection_pkg()
    vp = types.ModuleType('detection.volume_profile')
    vp.compute_poc = lambda symbol, hours=None, interval='1h', market='futures': {
        'ok': ok, 'poc': poc_value, 'last_close': last_close}
    vp.price_vs_poc = _price_vs_poc
    sys.modules['detection.volume_profile'] = vp


def _poc_stub(settings, price):
    ns = types.SimpleNamespace()
    ns._settings = settings
    ns._get_live_price = lambda sym: price
    return ns


def test_poc_filter_disabled_passes():
    _inject_poc(100.0)
    st = _poc_stub({'poc_filter_enabled': False}, 90)
    assert S._poc_filter_allows(st, 'BTCUSDT', 'SHORT') is True


def test_poc_below_allows_long_blocks_short():
    _inject_poc(100.0)
    st = _poc_stub({'poc_filter_enabled': True}, 90)   # price below POC → краще LONG
    assert S._poc_filter_allows(st, 'BTCUSDT', 'LONG') is True
    assert S._poc_filter_allows(st, 'BTCUSDT', 'SHORT') is False


def test_poc_above_allows_short_blocks_long():
    _inject_poc(100.0)
    st = _poc_stub({'poc_filter_enabled': True}, 110)  # price above POC → краще SHORT
    assert S._poc_filter_allows(st, 'BTCUSDT', 'SHORT') is True
    assert S._poc_filter_allows(st, 'BTCUSDT', 'LONG') is False


def test_poc_on_poc_is_neutral_allows_both():
    _inject_poc(100.0)
    st = _poc_stub({'poc_filter_enabled': True}, 100.0)  # exactly on POC → 'at'
    assert S._poc_filter_allows(st, 'BTCUSDT', 'LONG') is True
    assert S._poc_filter_allows(st, 'BTCUSDT', 'SHORT') is True


def test_poc_no_data_blocks():
    _inject_poc(None, ok=False)
    st = _poc_stub({'poc_filter_enabled': True}, 100)
    assert S._poc_filter_allows(st, 'BTCUSDT', 'LONG') is False


def test_poc_uses_last_close_when_no_live_price():
    _inject_poc(100.0, last_close=90.0)
    st = _poc_stub({'poc_filter_enabled': True}, None)  # no live price → use last_close 90
    assert S._poc_filter_allows(st, 'BTCUSDT', 'LONG') is True   # 90 < 100 → LONG ok


# ---------------- Forecast-strength filter ----------------
def _inject_forecast(f1_side, f1_conf, f4_side=None, f4_conf=None):
    _ensure_detection_pkg()
    fe_mod = types.ModuleType('detection.forecast_engine')
    cache = {'forecast_1h': ({'side': f1_side, 'confidence': f1_conf} if f1_side is not None else None),
             'forecast_4h': ({'side': f4_side, 'confidence': f4_conf} if f4_side is not None else None)}
    fe_mod.get_forecast_engine = lambda: types.SimpleNamespace(get=lambda s: cache)
    sys.modules['detection.forecast_engine'] = fe_mod


def _str_stub(settings):
    ns = types.SimpleNamespace()
    ns._settings = settings
    return ns


def test_strength_disabled_passes():
    _inject_forecast(-1, 90)
    st = _str_stub({'forecast_strength_filter_enabled': False})
    assert S._forecast_strength_allows(st, 'BTCUSDT', 'LONG') is True


def test_strength_passes_when_strong_agreeing():
    _inject_forecast(1, 80)
    st = _str_stub({'forecast_strength_filter_enabled': True, 'forecast_min_strength': 'strong'})
    assert S._forecast_strength_allows(st, 'BTCUSDT', 'LONG') is True


def test_strength_blocks_when_too_weak():
    _inject_forecast(1, 60)  # 60 < 66
    st = _str_stub({'forecast_strength_filter_enabled': True, 'forecast_min_strength': 'strong'})
    assert S._forecast_strength_allows(st, 'BTCUSDT', 'LONG') is False


def test_strength_moderate_threshold():
    _inject_forecast(1, 45)  # ≥40 moderate, <66 strong
    st = _str_stub({'forecast_strength_filter_enabled': True, 'forecast_min_strength': 'moderate'})
    assert S._forecast_strength_allows(st, 'BTCUSDT', 'LONG') is True


def test_strength_blocks_opposite():
    _inject_forecast(-1, 90)  # strong SHORT, signal LONG
    st = _str_stub({'forecast_strength_filter_enabled': True, 'forecast_min_strength': 'strong'})
    assert S._forecast_strength_allows(st, 'BTCUSDT', 'LONG') is False


# ---------------- Decision-verdict filter ----------------
def _inject_tm(reco):
    _ensure_detection_pkg()
    tm_mod = types.ModuleType('detection.trade_manager')
    tm_mod.get_trade_manager = lambda: types.SimpleNamespace(
        compute_decision=lambda sym, price: {'recommended': reco, 'headline': f'{reco} 80%'})
    sys.modules['detection.trade_manager'] = tm_mod


def _dec_stub(settings):
    ns = types.SimpleNamespace()
    ns._settings = settings
    ns._get_live_price = lambda sym: 100.0
    # Bind the REAL _decision_gate so _decision_filter_allows exercises actual logic.
    ns._decision_gate = types.MethodType(S._decision_gate, ns)
    return ns


def test_decision_disabled_passes():
    _inject_tm('SHORT')
    st = _dec_stub({'decision_filter_enabled': False})
    assert S._decision_filter_allows(st, 'X', 'LONG') is True


def test_decision_match_passes():
    _inject_tm('LONG')
    st = _dec_stub({'decision_filter_enabled': True})
    assert S._decision_filter_allows(st, 'X', 'LONG') is True


def test_decision_opposite_blocks():
    _inject_tm('SHORT')
    st = _dec_stub({'decision_filter_enabled': True})
    assert S._decision_filter_allows(st, 'X', 'LONG') is False


def test_decision_neutral_blocks_at_open():
    _inject_tm('NEUTRAL')
    st = _dec_stub({'decision_filter_enabled': True})
    assert S._decision_filter_allows(st, 'X', 'LONG') is False   # strict at open


def test_decision_neutral_waits_at_intake():
    _inject_tm('NEUTRAL')
    st = _dec_stub({'decision_filter_enabled': True})
    assert S._decision_filter_allows(st, 'X', 'LONG', at_intake=True) is True  # wait


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
