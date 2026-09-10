"""Тест: 🌊 ПОТОП СИГНАЛІВ У ЛОЗІ (кейс 10.09) — два незалежні дефекти.

**Скарга користувача дослівно:** «Що сталось із ботом? Йому знесло дах… Що за
жахіття в логу?» Плюс вимога: «потрібно лише при появі нового OB 1H CHoCH
відправити сигнал і проконтролювати Deduplicate Signals (1 per trend) і все».

**Що показав експорт (500 рядків за 5.6 хв, УСЬОГО 4 монети):**

    ASTERUSDT LONG  → 44 сигнали      ASTERUSDT SHORT → 1
    FILUSDT   LONG  → 42              FILUSDT   SHORT → 1
    BCHUSDT   LONG  → 39              BCHUSDT   SHORT → 1
    ADAUSDT   LONG  → 35              ADAUSDT   SHORT → 1
    OB-фільтр заблокував: 165 · дедуплікація заблокувала: 161 · у чергу: 4

**Асиметрія 44:1 і є діагнозом.** Напрямок, який ПРОЙШОВ фільтри, дав рівно
ОДИН сигнал і далі чесно різався дедупом. Напрямок, який фільтр ВІДХИЛИВ,
фаєрився знову й знову — тобто позначку дедупу отримував ЛИШЕ успішний шлях.

**ДЕФЕКТ 1 — позначка дедупу стояла ПІСЛЯ раннього `return`.**
`_send_alert`: рядок `self._last_signal_dir[symbol] = side_label` лежав ~70
рядків НИЖЧЕ за `return` у гілці «фільтр не пропустив». Відхилений напрямок
ніколи не ставав «опрацьованим».

**ДЕФЕКТ 2 — один прохід фаєрив ПАЧКУ подій.** BCHUSDT: 40 «сигналів» за 0.8с
з інтервалом 0.02с — це ОДИН виклик `_process_alerts` із десятками подій у
`new_events`, а не 40 циклів скану (цикл по 230 монетах стільки не встиг би).
При `Свіжість сигналу: без ліміту` вікно віку не відсіювало нічого, тож кожна
ланка старого ланцюга CHoCH давала власний алерт.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules['pybit.unified_trading'].HTTP = object


def _load_scanner():
    """Сканер вантажимо САМОСТІЙНО (без пакета `detection`) — пакетний імпорт
    тягне `__init__.py → pybit` і півпроєкту (задокументований урок)."""
    spec = importlib.util.spec_from_file_location(
        'smc_scanner_flood', os.path.join(_ROOT, 'detection', 'smc_scanner.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['smc_scanner_flood'] = mod
    spec.loader.exec_module(mod)
    return mod


SC = _load_scanner()
S = SC.SMCScanner


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _ev(from_t, to_t, d, tag='CHoCH'):
    return {'from_t': from_t, 'to_t': to_t, 'dir': d, 'tag': tag, 'level': 100.0}


def _pairs(evs):
    return [(f"{e['from_t']}:{e['dir']}", e) for e in evs]


# ═══════ ДЕФЕКТ 2 — ОДИН АЛЕРТ НА НАПРЯМОК ЗА ПРОХІД ═══════════════════════
def test_a_chain_of_events_gives_at_most_two_alerts():
    """Відтворює BCHUSDT: довгий ланцюг CHoCH, що чергується напрямками."""
    evs = []
    for i in range(40):
        evs.append(_ev(1000 + i, 2000 + i, 'bull' if i % 2 == 0 else 'bear'))
    ids = S._alertable_ids(_pairs(evs))
    _check(len(ids) == 2, f'мало лишитись 2 алерти (по одному на бік), а не {len(ids)}')


def test_the_alert_goes_to_the_NEWEST_event_of_each_side():
    """⚠️ Новизна за `to_t`, а не за `from_t`: `from_t` — це СТАРИЙ півот, що
    ламається, тож за ним «найновішою» стала б не та подія, що на графіку."""
    old = _ev(from_t=9999, to_t=100, d='bull')      # from_t найбільший, бар СТАРИЙ
    new = _ev(from_t=10, to_t=900, d='bull')        # from_t малий, бар НОВИЙ
    ids = S._alertable_ids(_pairs([old, new]))
    _check(ids == {'10:bull'}, f'алерт мав дістатись НАЙНОВІШІЙ події: {ids}')


def test_both_directions_survive():
    """Протилежний бік НЕ маскується — по одному алерту на кожен."""
    ids = S._alertable_ids(_pairs([
        _ev(1, 100, 'bull'), _ev(2, 500, 'bull'),
        _ev(3, 200, 'bear'), _ev(4, 700, 'bear')]))
    _check(ids == {'2:bull', '4:bear'}, f'по одному на бік, найновіші: {ids}')


def test_empty_and_garbage_do_not_raise_and_do_not_open_the_gate():
    """Порожній вхід → порожня множина (нічого не фаєримо, а НЕ «все»)."""
    _check(S._alertable_ids([]) == set(), 'порожній вхід → порожньо')
    _check(S._alertable_ids(None) == set(), 'None → порожньо')
    _check(S._alertable_ids([('x',), 'junk', ('y', None)]) == set(),
           'сміття не має ні падати, ні відкривати ворота')


def test_single_event_still_alerts():
    """Звичайний робочий випадок — одна нова подія — не має постраждати."""
    ids = S._alertable_ids(_pairs([_ev(5, 500, 'bear')]))
    _check(ids == {'5:bear'}, f'одна подія мусить лишитись алертабельною: {ids}')


def test_gate_is_applied_in_both_alert_modes():
    """Замок у КОДІ: обидві гілки, що кличуть `_send_alert` на CHoCH, мусять
    питати `_alertable`. Інакше режим `choch_or_bos` лишився б із потопом."""
    src = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    i = src.find('def _process_alerts(')
    body = src[i:src.find('\n    def ', i + 10)]
    _check(body.count('in _alertable') >= 2,
           'гейт стоїть менш ніж у двох гілках алертів')
    _check('_alertable = self._alertable_ids(new_events)' in body,
           'множина алертабельних подій не рахується')


def test_the_cap_does_not_touch_pending_or_tm_hooks():
    """⚠️ НАЙТОНШЕ: обмежуємо САМЕ алерт. `_pending_choch`, `on_bos_event` і
    `queue2_on_choch` мусять бачити КОЖНУ подію — інакше зламалась би логіка
    CHoCH+BOS і виходи TM."""
    src = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    i = src.find('def _process_alerts(')
    body = src[i:src.find('\n    def ', i + 10)]
    for line in body.splitlines():
        if 'in _alertable' in line:
            _check('_pending_choch' not in line and 'on_bos_event' not in line
                   and 'queue2_on_choch' not in line,
                   f'гейт алерту не має гейтити стан/хуки: {line.strip()}')
    for marker in ('on_bos_event', 'queue2_on_choch', "self._pending_choch[symbol]"):
        _check(marker in body, f'{marker} зник із циклу — це регресія')


# ═══════ ДЕФЕКТ 1 — ДЕДУП СТАВИТЬСЯ Й НА ВІДХИЛЕНОМУ СИГНАЛІ ═══════════════
def test_rejected_signal_marks_dedup_before_returning():
    """ГОЛОВНИЙ ЗАМОК. У гілці «фільтр не пропустив» позначка `_last_signal_dir`
    мусить стояти ДО `return` — інакше повертається потоп 44:1."""
    import ast
    src = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == '_send_alert')
    # Знаходимо `if not _allowed:` і перевіряємо ПОРЯДОК усередині нього.
    blk = None
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp) \
                and isinstance(node.test.op, ast.Not) \
                and getattr(node.test.operand, 'id', '') == '_allowed':
            blk = node
            break
    _check(blk is not None, 'гілку `if not _allowed:` не знайдено')
    stamp_line = ret_line = None
    for node in ast.walk(blk):
        if isinstance(node, ast.Assign):
            tgt = node.targets[0]
            if isinstance(tgt, ast.Subscript) and \
                    getattr(tgt.value, 'attr', '') == '_last_signal_dir':
                stamp_line = node.lineno
        if isinstance(node, ast.Return):
            ret_line = node.lineno
    _check(stamp_line is not None,
           'ВІДХИЛЕНИЙ сигнал не позначає напрямок — потоп повернеться')
    _check(ret_line is not None and stamp_line < ret_line,
           f'позначка ({stamp_line}) мусить бути ДО return ({ret_line})')


def test_rejected_signal_persists_the_dedup_state():
    """Позначка мусить ПЕРЕЖИТИ рестарт — інакше `botupdate` (робиться часто)
    щоразу відкривав би потоп заново."""
    import ast
    src = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == '_send_alert')
    blk = next(n for n in ast.walk(fn)
               if isinstance(n, ast.If) and isinstance(n.test, ast.UnaryOp)
               and getattr(n.test.operand, 'id', '') == '_allowed')
    _check('_persist_dedup_state' in ast.dump(blk),
           'стан дедупу не персиститься на відхиленому сигналі')


def test_dedup_reset_still_belongs_to_the_1h_ob_flip():
    """Вимога користувача: «лише при появі нового OB 1H CHoCH відправити
    сигнал». Скид позначки лишається ТАМ, де й був — на фліпі 1H-OB, а не
    десь ще: інакше «новий 1H-OB → новий сигнал» перестало б виконуватись."""
    src = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    i = src.find('def _update_smc_ob(')
    _check(i > 0, '_update_smc_ob не знайдено')
    body = src[i:src.find('\n    def ', i + 10)]
    _check('_last_signal_dir' in body,
           'скид дедупу на фліпі 1H-OB зник — це регресія')


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
