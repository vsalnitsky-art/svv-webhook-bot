"""Тест: «ЩО ЦЕ ЗА OB? ЇХ НЕМАЄ НА ГРАФІКУ» (кейс 10.09) — три дефекти показу.

**Питання користувача дослівно:**
 1. «Подія "Сигнал" це має бути одне і те, що і 🆕 Новий OB? Тобто це одні і ті
    ж OB просто по різному визначаються. То чому вони не співпадають?»
 2. «OB-фільтр заблокував (Order Block проти напрямку), що це за напрямок?
    Що це за OB? Їх немає на графіку.»

**ВІДПОВІДЬ НА (1): це РІЗНІ події на РІЗНИХ таймфреймах.**
 • `🆕 Новий OB` (`event='ob_new'`) — **1H Order Block** (`ob_filter_timeframe`),
   детекція на ЗАКРИТИХ 1h-барах;
 • `Сигнал` (`event='signal'`, `Свіжий сигнал choch`) — **CHoCH на 15m**
   (`timeframe` сканера, `Internal Size`).
Збігатися вони не мусять і не можуть. Але лог ПІДШТОВХУВАВ до хибного
висновку — див. ДЕФЕКТ 3.

**ДЕФЕКТ 1 — причина відмови не називала НІ бік, НІ блок.**
`OB-фільтр заблокував (Order Block проти напрямку)` — жодного орієнтира.
Поруч подія «🆕 Новий OB» про ТОЙ САМИЙ рядок БД несла і свічку блоку, і ціну.
Два повідомлення про одне й те саме описували його ПО-РІЗНОМУ.

**ДЕФЕКТ 2 — `_ob_state_label` давав лише `BEARISH/BOS`** (бік + тег), без
свічки й меж — знайти блок очима було неможливо.

**ДЕФЕКТ 3 — експорт СКЛЕЮВАВ `ob_new` із чужим ланцюгом.** Сесія різалась
ЛИШЕ на `signal`, без межі за часом: ORDIUSDT `blocked` об 11:01 і `ob_new` о
17:00 — **через 6 годин** — опинились в ОДНІЙ сесії, а `outcome` усього
ланцюга став `ob_new`. У таблиці логу це вже було виправлено (`SOLO_EVENTS`),
а експорт лишився зі старим правилом.
"""
import os, sys, types, importlib.util, re

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules['pybit.unified_trading'].HTTP = object


def _load_scanner():
    spec = importlib.util.spec_from_file_location(
        'smc_scanner_obclar', os.path.join(_ROOT, 'detection', 'smc_scanner.py'))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['smc_scanner_obclar'] = mod
    spec.loader.exec_module(mod)
    return mod


SC = _load_scanner()
S = SC.SMCScanner
_ROW = {}


def _install_db():
    db = types.ModuleType('storage.db_operations')
    db.get_db = lambda: types.SimpleNamespace(
        get_smc_ob_state=lambda s, tf: (dict(_ROW) if _ROW else None))
    st = types.ModuleType('storage'); st.__path__ = [os.path.join(_ROOT, 'storage')]
    sys.modules.setdefault('storage', st)
    sys.modules['storage.db_operations'] = db


_install_db()


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _sc(**over):
    o = S.__new__(S)
    o._settings = {'ob_filter_timeframe': '1h', 'ob_filter_choch_only': True}
    o._settings.update(over)
    return o


def _set_row(**kw):
    """Рядок `sob_smc_ob_state`, як на проді (AAVEUSDT зі скріна).
    Без аргументів — БАЗОВИЙ блок; `_ROW.clear()` окремо = «рядка немає»."""
    _ROW.clear()
    _ROW.update({'bias': 'BEARISH', 'created_by_tag': 'BOS',
                 'bar_time': 1789016400000,      # 10.09.26 05:00 UTC
                 'bar_low': 122.50, 'bar_high': 123.40})
    _ROW.update(kw)


# ═══════ ДЕФЕКТ 2 — БЛОК МУСИТЬ БУТИ ЗНАЙДЕНИЙ ОЧИМА ══════════════════════
def test_label_identifies_the_block_not_just_its_side():
    """`BEARISH/BOS` саме по собі не дає знайти блок на графіку."""
    _set_row()
    lbl = _sc()._ob_state_label('AAVEUSDT')
    _check('BEARISH/BOS' in lbl, f'бік і тег мусять лишитись: {lbl}')
    _check('свічка' in lbl, f'немає свічки блоку — блок не знайти: {lbl}')
    _check('122.5' in lbl and '123.4' in lbl,
           f'немає меж блоку — блок не звірити з графіком: {lbl}')


def test_label_uses_the_same_formatters_as_the_ob_new_line():
    """⚠️ ОДИН формат на два повідомлення про ТОЙ САМИЙ рядок БД.
    Свій формат тут дав би два різні написання одного числа."""
    import ast
    src = open(os.path.join(_ROOT, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == '_ob_state_info')
    body = ast.dump(fn)
    _check('fmt_utc' in body and 'fmt_price' in body,
           'мітка мусить форматувати тими самими функціями, що «🆕 Новий OB»')
    _check('strftime' not in body and "f'%'" not in body,
           'власного форматування тут бути не має')


def test_label_stays_honest_when_there_is_nothing_to_show():
    """Немає рядка / немає блоку → чесний текст, а не вигаданий орієнтир."""
    _set_row(bias=None)
    _check(_sc()._ob_state_label('X') == 'нема блоку', 'порожній bias')
    _ROW.clear()
    _check(_sc()._ob_state_label('X') == 'не рахувався', 'немає рядка')


def test_label_never_writes_CHOCH_in_caps():
    """У всьому проєкті тег пишеться «CHoCH» — `.upper()` зробив би з нього
    іншу сутність."""
    _set_row(created_by_tag='CHOCH')
    lbl = _sc()._ob_state_label('X')
    _check('CHoCH' in lbl and 'CHOCH' not in lbl, f'написання тега: {lbl}')


# ═══════ ДЕФЕКТ 1 — ПРИЧИНА ВІДМОВИ НАЗИВАЄ БІК І БЛОК ═══════════════════
def _reason(side, **row):
    """Витягнути причину саме OB-фільтра з `_signal_allowed`."""
    _set_row(**row)
    o = _sc()
    # Решта фільтрів вимкнена, тож OB — єдиний, що може відмовити.
    o._settings.update({'ob_filter_enabled': True, 'use_pd_zone_filter': False,
                        'forecast_1h_filter_enabled': False,
                        'forecast_4h_filter_enabled': False,
                        'forecast_strength_filter_enabled': False,
                        'decision_filter_enabled': False,
                        'poc_filter_enabled': False,
                        'liq_filter_enabled': False})
    o._pd_zone_filter_allows = lambda *a, **k: True
    ok, reason, detail = o._signal_allowed('AAVEUSDT', side, at_intake=True)
    return ok, reason, detail


def test_wrong_direction_reason_names_both_sides_and_the_block():
    """Кейс зі скріна: LONG-сигнал, а 1H-блок BEARISH."""
    ok, reason, _d = _reason('LONG')
    _check(not ok, 'сигнал мав бути зарізаний')
    _check('LONG' in reason, f'немає боку СИГНАЛУ: {reason}')
    _check('BEARISH' in reason, f'немає боку БЛОКУ: {reason}')
    _check('BULLISH' in reason, f'не сказано, який блок ПОТРІБЕН: {reason}')
    _check('свічка' in reason and '122.5' in reason,
           f'блок не ідентифіковано — його не знайти на графіку: {reason}')
    _check('1H' in reason, f'не названо TF блоку: {reason}')


def test_bos_reason_also_names_the_block():
    """Другий тип відмови: бік ЗБІГАЄТЬСЯ, але блок створено BOS."""
    ok, reason, _d = _reason('SHORT')     # BEARISH-блок + SHORT = бік ок
    _check(not ok, 'BOS-блок при «лише CHoCH» мусить різати')
    _check('BOS' in reason and 'CHoCH' in reason, f'тип події: {reason}')
    _check('свічка' in reason and '123.4' in reason,
           f'блок не ідентифіковано: {reason}')
    _check('ПРОТИ' not in reason,
           f'це НЕ відмова за напрямком — причини не змішувати: {reason}')


def test_no_block_reason_says_the_signal_side_too():
    _set_row(bias=None)
    ok, reason, _d = _reason('LONG', bias=None)
    _check(not ok and 'нема блоку' in reason, f'{reason}')
    _check('LONG' in reason, f'бік сигналу мусить бути названий: {reason}')


def test_three_reasons_stay_distinct():
    """«немає блоку» / «проти напрямку» / «створено BOS» — ТРИ РІЗНІ відмови."""
    r_none = _reason('LONG', bias=None)[1]
    r_dir = _reason('LONG')[1]
    r_bos = _reason('SHORT')[1]
    _check(len({r_none, r_dir, r_bos}) == 3,
           f'причини злились: {r_none!r} {r_dir!r} {r_bos!r}')


# ═══════ ДЕФЕКТ 3 — ЕКСПОРТ НЕ СКЛЕЮЄ «Новий OB» ІЗ ЧУЖИМ ЛАНЦЮГОМ ═══════
def test_export_cuts_the_chain_on_ob_new_and_on_a_long_gap():
    """Відтворює ORDIUSDT: `blocked` 11:01 і `ob_new` 17:00 — 6 годин."""
    src = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    i = src.find('groups, cur = [], None')
    _check(i > 0, 'групування сесій не знайдено')
    body = src[i:i + 1800]
    _check("_SOLO" in body and "'ob_new'" in body,
           'ob_new не починає власний ланцюг в експорті')
    _check('_MAX_GAP' in body,
           'немає межі за часом — події через години склеюються')
    m = re.search(r'_MAX_GAP\s*=\s*(\d+)\s*\*\s*3600', body)
    _check(m and int(m.group(1)) <= 6,
           f'межа розриву задовга, 6-годинний склей лишиться: {body[:200]}')


def test_export_does_not_glue_the_event_after_a_solo_one():
    """⚠️ НАЙТОНШЕ: після solo-події наступна теж починає СВІЙ ланцюг —
    інакше вона приклеїлась би вже до рядка «Новий OB» (той самий момент,
    що й у таблиці логу)."""
    src = open(os.path.join(_ROOT, 'web', 'flask_app.py'), encoding='utf-8').read()
    i = src.find('groups, cur = [], None')
    body = src[i:i + 1800]
    _check('if _solo:' in body and 'cur = None' in body,
           'після solo-події ланцюг не закривається')


def test_ui_and_export_agree_on_what_is_solo():
    """Два подання ОДНОГО логу не мають суперечити одне одному."""
    html = open(os.path.join(_ROOT, 'templates', 'smart_money.html'),
                encoding='utf-8').read()
    _check("SOLO_EVENTS = new Set(['ob_new'])" in html,
           'у таблиці логу ob_new більше не solo — розійшлось із експортом')


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
