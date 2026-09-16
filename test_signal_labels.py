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


# ═══ 🧮 КАРТИНКА УГОДИ З «МММ-МОНІТОРА» (вимога 15.09) ═══════════════════
# «🟪 заміни на 🧮, щоб видно було, що ця угода із "🧮 МММ-монітор"».
# Угоди монітора несуть ТОЙ САМИЙ сигнал, що й угоди Черги-4 (`vob_alert`),
# тож картинка сигналу їх не розрізняла. Тепер картинку дає ДВИГУН.

def test_mmm_trade_shows_the_monitor_icon_not_the_signal_one():
    assert sl.icon_of('vob_alert → MMM') == '🧮'
    # ✋ групове відкриття з монітора — теж 🧮 (двигун той самий)
    assert sl.icon_of('manual → MMM') == '🧮'


def test_other_engines_keep_the_signal_icon():
    # нічого, крім МММ-монітора, не змінилось
    assert sl.icon_of('vob_alert → Q4') == '🟪'
    assert sl.icon_of('choch_bos → Q1') == '🟦'
    assert sl.icon_of('manual → Q4') == '✋'
    assert sl.icon_of('poc → Q4') == '🎯'


def test_single_code_is_a_signal_not_an_engine():
    # запис черги несе лише СИГНАЛ — його не можна читати як двигун
    assert sl.engine_code_of('vob_alert') == ''
    assert sl.icon_of('vob_alert') == '🟪'
    assert sl.engine_code_of('vob_alert → MMM') == 'MMM'
    assert sl.engine_code_of('vob_alert → MMM · 🤪 LONG 70%') == 'MMM'


def test_icon_change_does_not_touch_the_label_or_the_logic():
    raw = 'vob_alert → MMM'
    # повна мітка (підказка) лишається як була
    assert sl.pretty_opened_by(raw) == '🟪 Volumized OB → 🧮 МММ-монітор'
    # код сигналу для логіки — теж
    assert sl.signal_code_of(raw) == 'vob_alert'


def test_unknown_code_falls_back_to_the_tag_icon():
    assert sl.icon_of('🎯 Черга-4 (усі 4 шари)') == sl.FALLBACK_ICON


# ═══ JS-ДЗЕРКАЛА мусять збігатися з бекендом ═════════════════════════════
# Іконки малює ФРОНТ, тож без цього замка мапи розійшлись би мовчки — і на
# сторінці стояло б 🟪, хоча бекенд уже каже 🧮.

def _js_map(text, name):
    """Витягти { 'k': 'v', … } із JS-джерела за іменем константи."""
    import re
    i = text.index(name)
    start = text.index('{', i)
    depth, end = 0, start
    for j in range(start, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                end = j
                break
    body = text[start:end + 1]
    return dict(re.findall(r"""['"]([^'"]+)['"]\s*:\s*['"]([^'"]+)['"]""", body))


def _js_files():
    for rel in ('templates/smart_money.html', 'infosite/app.js'):
        with open(os.path.join(_HERE, rel), encoding='utf-8') as f:
            yield rel, f.read()


def test_both_js_mirrors_match_the_python_icon_maps():
    for rel, src in _js_files():
        assert _js_map(src, 'SIGNAL_ICON_JS') == sl.SIGNAL_ICONS, rel
        assert _js_map(src, 'ENGINE_ICON_OVERRIDE_JS') == sl.ENGINE_ICON_OVERRIDE, rel


def test_both_js_mirrors_apply_the_override_before_the_signal_icon():
    # порядок важливий: спершу двигун, потім сигнал — інакше 🧮 ніколи не
    # переможе 🟪 і правка не діяла б.
    for rel, src in _js_files():
        i = src.index('function signalIconHtml')
        body = src[i:i + 500]
        assert 'ENGINE_ICON_OVERRIDE_JS' in body, rel
        assert body.index('ENGINE_ICON_OVERRIDE_JS') < body.index('SIGNAL_ICON_JS'), rel


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
