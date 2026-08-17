"""Signal-label registry — «Сигнал → Двигун» — regression tests.

Guards the fix for "не зрозуміло від якого сигналу пішла угода": the origin
signal was lost when a signal passed through the FF queue (the engine stamped
its own label). Now opened_by stores machine codes "<signal> → <engine>" and
pretty_opened_by() renders badges; signal_code_of() extracts the origin for
logic. Substrings needed by other code ('funding', 'POC-сетап', 'external')
must survive in the composed machine string.
"""
import importlib.util
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "signal_labels_under_test", os.path.join(_HERE, "detection", "signal_labels.py"))
sl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sl)


def test_compose_signal_and_engine():
    assert sl.compose('vob_alert', 'Q4') == 'vob_alert → Q4'


def test_compose_signal_only():
    assert sl.compose('choch', None) == 'choch'
    assert sl.compose('choch', '') == 'choch'


def test_pretty_signal_engine():
    assert sl.pretty_opened_by('vob_alert → Q4') == '🟪 Volumized OB → 🎯 Черга-4'
    assert sl.pretty_opened_by('choch_bos → Q1') == '🟦 CHoCH+BOS → 🎯 Черга-1'


def test_pretty_legacy_raw_code():
    assert sl.pretty_opened_by('choch') == '🟦 CHoCH'


def test_pretty_keeps_verdict_tail():
    r = sl.pretty_opened_by('vob_alert → Q4 · 🤪 SHORT 66% (помірний)')
    assert r == '🟪 Volumized OB → 🎯 Черга-4 · 🤪 SHORT 66% (помірний)'


def test_pretty_unknown_passthrough():
    # unknown engine string is left as-is (nothing "disappears")
    assert sl.pretty_opened_by('🎯 Черга-4 (усі 4 шари)') == '🎯 Черга-4 (усі 4 шари)'


def test_signal_code_of_extracts_origin():
    assert sl.signal_code_of('vob_alert → Q4') == 'vob_alert'
    assert sl.signal_code_of('choch_bos → Q1 · 🤪 x') == 'choch_bos'
    assert sl.signal_code_of('choch') == 'choch'
    assert sl.signal_code_of('external') == 'external'


def test_funding_substring_preserved():
    # /funding/i detection must still match the composed machine string
    s = sl.compose('vob', 'Q3-VOB(funding)')
    assert 'funding' in s.lower(), s


def test_poc_substring_preserved():
    assert 'POC-сетап' in sl.compose('poc', 'POC-сетап')


def test_external_stays_exact_code():
    # external is never composed (kept exact for `== 'external'` checks)
    assert sl.signal_code_of('external') == 'external'


def test_poc_label_not_choch():
    # Regression: POC-setup routed into the FF queue used to hard-code
    # kind='choch' → showed as a phantom «CHoCH» while CHoCH alerts were OFF.
    # Now kind='poc' → its own badge, никакого CHoCH.
    assert sl.pretty_opened_by('poc') == '🎯 POC-сетап'
    assert sl.pretty_opened_by('poc → Q4') == '🎯 POC-сетап → 🎯 Черга-4'
    assert 'CHoCH' not in sl.pretty_opened_by('poc → Q4')


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
