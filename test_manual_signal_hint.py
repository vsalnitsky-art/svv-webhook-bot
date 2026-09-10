"""Тест: ✋ РУЧНИЙ СИГНАЛ не залежить від майстер-тумблера Trade Manager.

**Питання користувача (10.09):** «Чому не відправляється в ручному режимі
сигнал? До чого тут "⚠️ Trade Manager disabled — turn it ON in TM Settings
first"? Якщо сигнал мав би як мінімум потрапити у Чергу.»

Мав — і потрапляє. Кнопка ✋ кличе `on_signal(opened_by='manual')` →
`intercept` → черга; майстер-тумблер TM гасить лише ПРЯМЕ відкриття (⬆LONG /
⬇SHORT). Зламаним був ПОКАЗ, і зразу у двох місцях:

1. **Підпис стосувався не тих кнопок.** Один спільний елемент
   `sm-decision-actions-hint` стояв ВПРИТУЛ до ✋ Сигналу, а текст «Trade
   Manager disabled» належав лише двом кнопкам прямого відкриття.
2. **Відповідь на натискання СТИРАЛАСЬ.** `sendManualSignal` писала результат
   у ТОЙ САМИЙ елемент, а `updateManualEntryButtons` перезаписує його на
   КОЖНОМУ перемальунку банера (~10с) — тож підтвердження зникало, і кнопка
   виглядала як мертва.
3. **Відповідь нічого не казала:** результат `on_signal` ІГНОРУВАВСЯ, тож
   «відправлено в обробку» друкувалось і коли сигнал став у чергу, і коли був
   відхилений як дубль.
"""
import os, sys, re, ast

_ROOT = os.path.dirname(os.path.abspath(__file__))
_HTML = open(os.path.join(_ROOT, 'templates', 'smart_money.html'),
             encoding='utf-8').read()
_FLASK = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _fn(html, name):
    i = html.find(f'function {name}(')
    _check(i > 0, f'{name} не знайдено')
    return html[i:i + 2600]


def test_signal_has_its_own_hint_element():
    """Власний елемент — інакше 10-секундний перемальунок стирає відповідь."""
    _check('id="sm-decision-signal-hint"' in _HTML,
           'у ✋ Сигналу немає власного підпису')
    body = _fn(_HTML, 'sendManualSignal')
    _check("getElementById('sm-decision-signal-hint')" in body,
           'sendManualSignal і далі пише у СПІЛЬНИЙ елемент')
    _check("getElementById('sm-decision-actions-hint')" not in body,
           'sendManualSignal не має чіпати підпис кнопок прямого відкриття')


def test_periodic_refresh_cannot_wipe_the_answer():
    """`updateManualEntryButtons` бігає щополла — він мусить чіпати РІВНО
    підпис ⬆LONG/⬇SHORT і не знати про підпис сигналу."""
    body = _fn(_HTML, 'updateManualEntryButtons')
    _check("getElementById('sm-decision-actions-hint')" in body,
           'перевіряємо не ту функцію')
    _check('sm-decision-signal-hint' not in body,
           'перемальунок банера не має торкатись підпису ✋ Сигналу')


def test_tm_disabled_text_names_the_buttons_it_is_about():
    """Підпис мусить сказати, ЧОГО він стосується, і що ✋ Сигнал працює."""
    body = _fn(_HTML, 'updateManualEntryButtons')
    j = body.find('if (!tmEnabled)')
    _check(j > 0, 'не знайдено гілку вимкненого TM')
    seg = body[j:body.find('} else if', j)]
    m = re.search(r"hintText = ((?:.|\n)*?);", seg)
    _check(m, 'не знайдено текст підпису при вимкненому TM')
    txt = m.group(1)
    _check('LONG' in txt and 'SHORT' in txt,
           f'підпис не називає кнопки, яких стосується: {txt}')
    _check('Сигнал' in txt, f'підпис не каже, що ✋ Сигнал працює: {txt}')


def test_signal_button_is_never_disabled_by_the_tm_switch():
    """Кнопку ✋ майстер-тумблер TM не гасить — і не мусить."""
    body = _fn(_HTML, 'updateManualEntryButtons')
    _check('sm-decision-btn-signal' not in body,
           'updateManualEntryButtons не має керувати кнопкою ✋ Сигнал')


def test_endpoint_reports_what_actually_happened():
    """Результат `on_signal` мусить доїжджати до користувача."""
    i = _FLASK.find('def api_smc_manual_signal(')
    _check(i > 0, 'маршрут ручного сигналу не знайдено')
    body = _FLASK[i:_FLASK.find('@app.route', i + 10)]
    _check(re.search(r'res\s*=\s*tm\.on_signal\(', body),
           'результат on_signal і далі ігнорується')
    for st in ('queued', 'opened', 'duplicate', 'rejected'):
        _check(f"'{st}'" in body, f'статус {st} не перекладено для показу')
    _check("'outcome'" in body, 'у відповіді немає підсумку')
    body_js = _fn(_HTML, 'sendManualSignal')
    _check('d.outcome' in body_js, 'сторінка не показує підсумок')


def test_endpoint_does_not_gate_on_the_tm_master_switch():
    """ЗАМОК НА СУТЬ: ручний сигнал не має перевіряти `is_enabled()`/`enabled`
    — інакше повернеться саме та поведінка, про яку питав користувач."""
    i = _FLASK.find('def api_smc_manual_signal(')
    body = _FLASK[i:_FLASK.find('@app.route', i + 10)]
    for bad in ('is_enabled()', "get('enabled')", '.enabled'):
        _check(bad not in body,
               f'ручний сигнал не має залежати від майстер-тумблера ({bad})')


# ══════ 🖐 РУЧНИЙ ВИБІР НАПРЯМКУ + ПІДПИС НЕ ЛАМАЄ КНОПКИ (вимога 10.09) ══════
#
# Дві вимоги користувача, обидві по цьому самому рядку кнопок:
#   1. «Текст розмісти так, щоб кнопки не втрачали свою форму. Його не потрібно
#      багато, не більше ніж на дві стрічки.»
#   2. «Зроби сигнал також щоб був ручний вибір LONG чи SHORT.»

def _css(html, sel):
    i = html.find(sel + ' {')
    if i < 0:
        i = html.find(sel + '{')
    _check(i > 0, f'CSS-правило {sel} не знайдено')
    return html[i:html.find('}', i)]


def test_buttons_keep_their_shape_next_to_a_long_hint():
    """У flex-рядку текст зʼїдав ширину кнопок, і «⬆ LONG» ламалось на два
    рядки. Кнопка не має ні стискатись, ні переносити свій підпис."""
    btn = _css(_HTML, '.sm-decision-btn')
    _check('flex: 0 0 auto' in btn or 'flex:0 0 auto' in btn,
           f'кнопка мусить бути flex:0 0 auto — {btn}')
    _check('white-space: nowrap' in btn or 'white-space:nowrap' in btn,
           'підпис кнопки не має переноситись')
    row = _css(_HTML, '.sm-decision-actions')
    _check('flex-wrap: wrap' in row or 'flex-wrap:wrap' in row,
           'довгий підпис мусить переноситись ЦІЛИМ блоком, а не тиснути кнопки')


def test_hint_is_capped_at_two_lines():
    """«не більше ніж на дві стрічки» — і це мусить триматись CSS-ом, а не
    надією на коротку фразу."""
    h = _css(_HTML, '.sm-decision-actions-hint')
    _check('-webkit-line-clamp: 2' in h or '-webkit-line-clamp:2' in h,
           f'немає обмеження у два рядки: {h}')
    _check('overflow: hidden' in h or 'overflow:hidden' in h,
           'без overflow:hidden обрізання не спрацює')


def test_the_tm_off_text_is_actually_short():
    """Замок на ДОВЖИНУ: попередній варіант був на 160+ символів і в три
    рядки. Деталі мусять піти в `title`, а не в сам підпис."""
    body = _fn(_HTML, 'updateManualEntryButtons')
    j = body.find('if (!tmEnabled)')
    seg = body[j:body.find('} else if', j)]
    m = re.search(r"hintText = ((?:.|\n)*?);", seg)
    txt = ''.join(re.findall(r"'([^']*)'", m.group(1)))
    _check(len(txt) <= 90, f'підпис задовгий ({len(txt)} символів): {txt}')
    _check('hintFull' in seg, 'повний текст мусить лишитись у title')
    _check("hint.title" in body, 'title підпису не виставляється')


def test_signal_direction_can_be_chosen_by_hand():
    """Селектор напрямку: «за вердиктом» (як було) + власні LONG / SHORT."""
    _check('id="sm-signal-dir"' in _HTML, 'немає селектора напрямку сигналу')
    for v in ('"auto"', '"LONG"', '"SHORT"'):
        _check(f'value={v}' in _HTML, f'немає опції {v}')
    body = _fn(_HTML, '_smSignalSide')
    _check("'LONG'" in body and "'SHORT'" in body,
           'ручний вибір не повертається як є')
    _check('_smDecisionDir' in body,
           'режим «за вердиктом» мусить лишитись (стара поведінка)')


def test_send_uses_the_selector_not_the_verdict_directly():
    """ЄДИНЕ місце вибору: якби `sendManualSignal` читав вердикт сам, кнопка
    показувала б один напрямок, а на сервер летів би інший."""
    body = _fn(_HTML, 'sendManualSignal')
    _check('_smSignalSide()' in body, 'sendManualSignal не питає селектор')
    _check('_smDecisionDir' not in body,
           'sendManualSignal не має читати вердикт повз _smSignalSide')


def test_button_is_labelled_with_the_direction_that_will_be_sent():
    """У режимі «за вердиктом» напрямок змінюється разом із банером — без
    підпису натискання було б наосліп."""
    body = _fn(_HTML, '_smSignalDirSync')
    _check('_smSignalSide()' in body, 'підпис не спирається на той самий вибір')
    _check('Сигнал ${side}' in body, 'кнопка не підписана напрямком')
    _check('_smSignalDirSync()' in _fn(_HTML, 'renderDecision')
           or '_smSignalDirSync' in _HTML.split('updateManualEntryButtons(decision);')[1][:400],
           'підпис не оновлюється на перемальунку банера')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for name, fn in fns:
        try:
            fn(); print(f'  ok  {name}')
        except Exception as e:
            bad += 1; print(f'  FAIL {name}: {e}')
    print(f'\n{len(fns) - bad}/{len(fns)} passed')
    sys.exit(1 if bad else 0)
