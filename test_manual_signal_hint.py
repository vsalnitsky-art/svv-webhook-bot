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
