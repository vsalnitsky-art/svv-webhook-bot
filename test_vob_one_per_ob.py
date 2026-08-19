"""«1 VOB на 1H OB» (vob_one_per_ob) — regression tests.

Strategy (user): one signal = one 1H OB + one 5m VOB. The FIRST 5m VOB that
passes all filters within a 1H-OB "epoch" (identified by its bar_time) fires
ONE signal; every later 5m VOB on the SAME 1H OB is ignored until a NEW 1H OB
appears (any direction → different bar_time). Variant (A): the epoch is marked
used only when a signal actually fires.

`_vob_epoch_fresh(symbol, ob_bt)` is the core gate: True iff no VOB signal has
fired yet for this 1H-OB bar_time. Loaded via importlib (smc_scanner has only
stdlib top-level imports).
"""
import importlib.util
import os
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "smc_scanner_vob1_test", os.path.join(_HERE, "detection", "smc_scanner.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
S = _m.SMCScanner


def _stub():
    ns = types.SimpleNamespace()
    ns._vob_ob_epoch = {}
    ns._vob_epoch_fresh = types.MethodType(S._vob_epoch_fresh, ns)
    return ns


def test_no_ob_never_fresh():
    st = _stub()
    assert st._vob_epoch_fresh('X', None) is False   # немає валідного OB → не фаєримо


def test_first_ob_is_fresh():
    st = _stub()
    assert st._vob_epoch_fresh('X', 1000) is True    # перший раз на цьому OB → сигнал


def test_same_ob_not_fresh_after_consumed():
    st = _stub()
    st._vob_ob_epoch['X'] = 1000                      # епоха вже спожита (сигнал був)
    assert st._vob_epoch_fresh('X', 1000) is False   # той самий OB → ігнор


def test_new_ob_is_fresh_again():
    st = _stub()
    st._vob_ob_epoch['X'] = 1000
    assert st._vob_epoch_fresh('X', 2000) is True     # НОВИЙ 1H-OB → знову один сигнал


def test_new_ob_same_direction_still_fresh():
    # bar_time змінюється на БУДЬ-який новий OB (навіть той самий напрямок) → епоха нова
    st = _stub()
    st._vob_ob_epoch['X'] = 1000
    assert st._vob_epoch_fresh('X', 1500) is True


def test_per_symbol_independent():
    st = _stub()
    st._vob_ob_epoch['X'] = 1000
    assert st._vob_epoch_fresh('Y', 1000) is True     # інша монета — своя епоха


def test_epoch_flow_one_signal_per_ob():
    """Проганяємо потік: OB=1000 → перший VOB fresh (фаєримо, споживаємо) →
    решта VOB на 1000 ігнор → OB=2000 → знову один."""
    st = _stub()
    # OB 1000: перший VOB
    assert st._vob_epoch_fresh('X', 1000) is True
    st._vob_ob_epoch['X'] = 1000                      # сигнал спрацював → епоха спожита
    # ще кілька VOB на тому ж OB — усі ігнор
    assert st._vob_epoch_fresh('X', 1000) is False
    assert st._vob_epoch_fresh('X', 1000) is False
    # новий OB 2000 — знову один
    assert st._vob_epoch_fresh('X', 2000) is True
    st._vob_ob_epoch['X'] = 2000
    assert st._vob_epoch_fresh('X', 2000) is False


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
