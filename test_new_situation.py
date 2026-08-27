"""Тест: 🧬 «НОВА СИТУАЦІЯ ДЛЯ НОВОЇ УГОДИ» — жодних угод «із повітря».

Скарга (APTUSDT, 26.08): угоду закрито ВРУЧНУ о 03:03, а о 03:10:53 бот відкрив
її знову — «невідомо звідки». Корінь не в одному конкретному шляху, а в тому, що
НІЩО не вимагало НОВОЇ підстави для входу:

  • закриття угоди не лишало жодної позначки — для логіки входу монета одразу
    ставала «чистою», ніби угоди й не було;
  • запис, що вже лежав у черзі, відкривався наступним тіком двигуна незалежно
    від того, коли він туди потрапив (а з ♾ «Без терміну» він живе вічно);
  • у рядку 🧾 Логу «Відкрито» не було НІЧОГО про походження — ні коли зайшов
    сигнал, ні скільки монета чекала, тож зв'язок доводилось шукати очима.

Правило тепер: наступна угода по монеті відкривається ЛИШЕ на сигналі, що виник
ПІСЛЯ закриття попередньої. Ручні шляхи (✋) правило не стосується.
"""
import os, sys, types, importlib.util, threading, time

_ROOT = os.path.dirname(os.path.abspath(__file__))
if 'pybit' not in sys.modules:
    for n in ('pybit', 'pybit.unified_trading'):
        sys.modules[n] = types.ModuleType(n)
    sys.modules['pybit.unified_trading'].HTTP = object

_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_load('detection.signal_labels', 'detection/signal_labels.py')
_load('detection.setup_grader', 'detection/setup_grader.py')
ffmod = _load('detection.fuel_filter', 'detection/fuel_filter.py')
FF = ffmod.FuelFilterDaemon
MIN = 60.0


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _ff(**settings):
    o = FF.__new__(FF)
    o._lock = threading.RLock()
    o._manual_closed_at = {}
    o._last_trade_end = {}
    o._pending = {}; o._pending2 = {}; o._pending3 = {}; o._pending4 = {}
    o._q4_layers_cache = {}; o._q4_lit_at = {}
    o._engine_skip = {}
    o._new_situation_logged = None
    o._consumed_signal_at = {}
    o._persist_state = lambda: None
    s = {'require_new_situation': True, 'open_max_signal_age_min': 0,
         'manual_close_lock_min': 0}
    s.update(settings)
    o.get_settings = lambda: dict(s)
    return o, s


# ═════════════ 🧬 «Нова ситуація» — чиста логіка ════════════════════════════
def test_signal_older_than_last_close_is_refused():
    """Головний кейс APTUSDT: сигнал/запис у черзі СТАРІШИЙ за закриття."""
    o, s = _ff()
    now = time.time()
    o._last_trade_end['APTUSDT'] = now - 5 * MIN          # закрили 5 хв тому
    ok, why = o._is_new_situation('APTUSDT', now - 40 * MIN, s)   # сигнал 40 хв тому
    _check(ok is False, 'сигнал ДО закриття не має відкривати нову угоду')
    _check('СТАРІШИЙ за закриття' in why, f'причина має бути зрозумілою: {why}')
    print('✓ сигнал, старіший за закриття попередньої угоди, відхиляється')


def test_signal_after_close_is_accepted():
    o, s = _ff()
    now = time.time()
    o._last_trade_end['APTUSDT'] = now - 30 * MIN
    ok, why = o._is_new_situation('APTUSDT', now - 2 * MIN, s)
    _check(ok is True, f'свіжий сигнал ПІСЛЯ закриття — це нова ситуація: {why}')
    print('✓ сигнал після закриття = нова ситуація, вхід дозволено')


def test_first_ever_trade_is_not_blocked():
    """По монеті ще не було жодної угоди → нема з чим порівнювати."""
    o, s = _ff()
    ok, _ = o._is_new_situation('NEWUSDT', time.time() - 10 * MIN, s)
    _check(ok is True, 'перша угода по монеті не має блокуватись')
    print('✓ перша угода по монеті проходить')


def test_missing_signal_time_does_not_invent_a_refusal():
    o, s = _ff()
    o._last_trade_end['APTUSDT'] = time.time()
    for bad in (None, 0, '', 'abc'):
        ok, _ = o._is_new_situation('APTUSDT', bad, s)
        _check(ok is True, f'без часу сигналу ({bad!r}) відмову не вигадуємо')
    print('✓ немає часу сигналу → не блокуємо навмання')


def test_rule_can_be_switched_off():
    o, s = _ff(require_new_situation=False)
    o._last_trade_end['APTUSDT'] = time.time()
    ok, _ = o._is_new_situation('APTUSDT', time.time() - 60 * MIN, s)
    _check(ok is True, 'вимкнене правило = стара поведінка')
    print('✓ правило вимикається (повертає стару поведінку)')


def test_stale_signal_age_limit():
    o, s = _ff(open_max_signal_age_min=30)
    ok, why = o._is_new_situation('BTCUSDT', time.time() - 90 * MIN, s)
    _check(ok is False and 'ліміт' in why, f'сигнал 90хв при ліміті 30хв: {why}')
    ok2, _ = o._is_new_situation('BTCUSDT', time.time() - 5 * MIN, s)
    _check(ok2 is True, 'свіжий сигнал у межах ліміту проходить')
    print('✓ окремий ліміт віку сигналу працює (0 = вимкнено)')


def test_age_limit_off_by_default():
    o, s = _ff()
    ok, _ = o._is_new_situation('BTCUSDT', time.time() - 10_000 * MIN, s)
    _check(ok is True, 'за замовчуванням ліміт віку вимкнений')
    print('✓ ліміт віку за замовчуванням вимкнено')


def test_defaults():
    _check(ffmod.DEFAULT_SETTINGS.get('require_new_situation') is True,
           'правило «нова ситуація» має бути УВІМКНЕНЕ за замовчуванням')
    _check(ffmod.DEFAULT_SETTINGS.get('open_max_signal_age_min') == 0,
           'ліміт віку — вимкнений (не міняємо поведінку без запиту)')
    _check(ffmod.DEFAULT_SETTINGS.get('manual_close_lock_min') == 0,
           'блокування після ручного закриття — ОПЦІЯ, дефолт вимкнено')
    print('✓ дефолти: правило ON, ліміт віку OFF, блок ручного закриття OFF')


# ═════════════ ✋ Ручне закриття чистить черги ══════════════════════════════
def test_manual_close_purges_all_queues():
    """Запис, що ЛИШИВСЯ в черзі після ручного закриття, відкрив би угоду
    наступним тіком — без жодного сигналу. Це і є «угода з повітря»."""
    o, _ = _ff()
    for q in (o._pending, o._pending2, o._pending3, o._pending4):
        q['APTUSDT'] = {'dir': 'SHORT', 'added_at': time.time() - 3600}
    o._q4_layers_cache['APTUSDT'] = (time.time(), {'SHORT': {}})
    o._q4_lit_at['APTUSDT'] = time.time()
    o.note_manual_close('APTUSDT')
    for name, q in (('Черга-1', o._pending), ('Черга-2', o._pending2),
                    ('Черга-3', o._pending3), ('Черга-4', o._pending4)):
        _check('APTUSDT' not in q, f'{name}: запис мав бути прибраний')
    _check('APTUSDT' not in o._q4_layers_cache, 'кеш шарів мав очиститись')
    _check(o._manual_closed_at.get('APTUSDT'), 'мітка ручного закриття мала стати')
    print('✓ ручне закриття чистить УСІ черги (нема чому відкриватись)')


def test_manual_close_lock_when_enabled():
    o, _ = _ff(manual_close_lock_min=60)
    o._manual_closed_at['APTUSDT'] = time.time() - 10 * MIN
    blocked, left = o.manual_close_block('APTUSDT')
    _check(blocked is True and 49 < left < 51,
           f'мало лишитись ~50 хв блокування, отримано {left:.1f}')
    print('✓ блокування після ручного закриття рахує залишок часу')


def test_manual_close_lock_expires_and_clears():
    o, _ = _ff(manual_close_lock_min=30)
    o._manual_closed_at['APTUSDT'] = time.time() - 31 * MIN
    blocked, _ = o.manual_close_block('APTUSDT')
    _check(blocked is False, 'термін вийшов → блокування знято')
    _check('APTUSDT' not in o._manual_closed_at, 'протермінована мітка має зникнути')
    print('✓ блокування саме спливає й прибирає мітку')


def test_manual_close_lock_off_by_default():
    o, _ = _ff()
    o._manual_closed_at['APTUSDT'] = time.time()
    blocked, _ = o.manual_close_block('APTUSDT')
    _check(blocked is False, 'дефолт 0 = блокування вимкнено')
    print('✓ дефолт: блокування після ручного закриття вимкнено')


def test_lock_can_be_cleared_by_hand():
    o, _ = _ff(manual_close_lock_min=60)
    o._manual_closed_at['APTUSDT'] = time.time()
    _check(o.clear_manual_close_block('APTUSDT') is True, 'мітка мала бути знята')
    _check(o.manual_close_block('APTUSDT')[0] is False, 'після зняття — не блоковано')
    print('✓ блокування можна зняти достроково')


# ═════════════ 🧬 Походження угоди в рядку логу ════════════════════════════
def test_origin_trace_answers_where_it_came_from():
    """Рядок «Відкрито» має САМ пояснювати походження — без пошуку по логу."""
    o, _ = _ff()
    o._ob_epoch_now = lambda sym: 1787536500000
    now = time.time()
    info = {'kind': 'vob_alert', 'added_at': now - 3 * 3600 - 12 * MIN,
            'added_price': 0.5673}
    lay = {'count': 4, 'required': 4,
           'layers': [{'label': 'Новий МММ', 'detail': 'SHORT 34%'},
                      {'label': 'Готовність', 'detail': 'СЕРЕДНІЙ 44'}]}
    t = o._origin_trace('APTUSDT', info, lay, now)
    for token in ('сигнал:', 'у черзі з', 'чекала', '3г 12хв',
                  'шари 4/4', 'Новий МММ', 'такт 1H-OB'):
        _check(token in t, f'у ланцюгу походження бракує «{token}»:\n{t}')
    print(f'✓ ланцюг походження самодостатній:\n    {t}')


def test_origin_trace_marks_opposite_replacement():
    o, _ = _ff()
    o._ob_epoch_now = lambda sym: None
    t = o._origin_trace('X', {'kind': 'choch', 'added_at': time.time(),
                              'gate_exempt': True}, {}, time.time())
    _check('ПРОТИЛЕЖНИЙ сигнал' in t, f'заміна протилежним має бути видна: {t}')
    print('✓ заміну протилежним сигналом видно в походженні')


def test_origin_trace_survives_missing_data():
    o, _ = _ff()
    o._ob_epoch_now = lambda sym: None
    t = o._origin_trace('X', {}, None, time.time())
    _check(isinstance(t, str) and t, 'порожні дані не мають ламати рядок логу')
    _check('невідом' in t, f'брак даних має бути названий, а не прихований: {t}')
    print('✓ брак даних у походженні не ламає лог і чесно позначається')


def test_wait_formatting():
    _check(FF._fmt_wait(45) == '45с', FF._fmt_wait(45))
    _check(FF._fmt_wait(47 * 60) == '47хв', FF._fmt_wait(47 * 60))
    _check(FF._fmt_wait(3 * 3600 + 12 * 60) == '3г 12хв', FF._fmt_wait(3 * 3600 + 12 * 60))
    _check(FF._fmt_wait(None) == '—', 'сміття → прочерк')
    print('✓ тривалість очікування форматується читабельно')


# ═════════════ 🧟 МЕРТВИЙ ЗАПИС У ЧЕРЗІ (кейс LDOUSDT) ══════════════════════
def test_ldo_case_close_purges_spent_queue_records():
    """🐞 КЕЙС LDOUSDT (26.08). Правило «нова ситуація» ВІДМОВЛЯЛО у відкритті,
    але запис лишався в Черзі-4. Умова `signal_at <= last_trade_end` НЕЗМІННА —
    запис не міг стати валідним НІКОЛИ, а з ♾ «Без терміну» його не виселяв ні
    TTL, ні застій. Результат: висів 11.5 год, а двигун кожні ~22с писав ту саму
    відмову (30 однакових рядків у лозі за 10 хвилин).
    Тепер закриття угоди ПРИБИРАЄ відпрацьовані записи негайно."""
    o, _ = _ff()
    sig = time.time() - 11.5 * 3600          # сигнал 25.08 23:47
    o._pending4['LDOUSDT'] = {'dir': 'SHORT', 'kind': 'vob_alert', 'added_at': sig}
    o._q4_layers_cache['LDOUSDT'] = (time.time(), {'SHORT': {}})
    o.note_trade_closed('LDOUSDT')           # угода закрилась о 10:29
    _check('LDOUSDT' not in o._pending4,
           'відпрацьований запис мав ЗНИКНУТИ з Черги-4, а не висіти 11 годин')
    _check('LDOUSDT' not in o._q4_layers_cache, 'кеш шарів теж має піти')
    _check(o._last_trade_end.get('LDOUSDT'), 'риска часу мала стати')
    print('✓ LDOUSDT: закриття угоди прибирає відпрацьований запис із черги')


def test_close_purges_every_queue():
    o, _ = _ff()
    sig = time.time() - 3600
    for q in (o._pending, o._pending2, o._pending3, o._pending4):
        q['LDOUSDT'] = {'dir': 'SHORT', 'added_at': sig}
    o.note_trade_closed('LDOUSDT')
    for name, q in (('Черга-1', o._pending), ('Черга-2', o._pending2),
                    ('Черга-3', o._pending3), ('Черга-4', o._pending4)):
        _check('LDOUSDT' not in q, f'{name}: запис мав бути прибраний')
    print('✓ чистка стосується ВСІХ чотирьох черг')


def test_close_keeps_record_created_after_it():
    """Запис, що зʼявився ПІСЛЯ закриття, — це вже нова ситуація, не чіпаємо."""
    o, _ = _ff()
    o._pending4['LDOUSDT'] = {'dir': 'SHORT', 'added_at': time.time() + 5}
    o.note_trade_closed('LDOUSDT')
    _check('LDOUSDT' in o._pending4, 'новіший за закриття запис має лишитись')
    print('✓ запис, новіший за закриття, не чіпаємо')


def test_close_without_queue_records_is_quiet():
    o, _ = _ff()
    o.note_trade_closed('LDOUSDT')
    _check(o._last_trade_end.get('LDOUSDT'), 'риска все одно ставиться')
    print('✓ закриття без записів у чергах працює тихо')


def test_refusal_is_logged_once_not_every_tick():
    """30 однакових рядків за 10 хвилин — це не лог, це шум."""
    o, s = _ff()
    o._last_trade_end['LDOUSDT'] = time.time() - 60
    logged = []
    mod = types.ModuleType('detection.activity_log')
    mod.log_activity = lambda sym, kind, text, **kw: logged.append(text)
    sys.modules['detection.activity_log'] = mod
    sig = time.time() - 3600
    for _ in range(10):
        ok, why = o._is_new_situation('LDOUSDT', sig, s)
        _check(ok is False, 'відмова має лишатись стабільною')
        _fk = ('LDOUSDT', why)
        if o._new_situation_logged != _fk:
            o._new_situation_logged = _fk
            mod.log_activity('LDOUSDT', 'skipped', why)
    _check(len(logged) == 1, f'очікували 1 запис у лог, отримано {len(logged)}')
    print('✓ незмінна відмова пишеться в лог ОДИН раз, а не щотіку')


# ═════ 🔥 ВИГОРІЛИЙ СИГНАЛ + 🚨 ЗАСТРЯГ (вимоги користувача) ═══════════════
def test_consumed_signal_never_accepted_again():
    """«Запис пішов далі по алгоритму → назад у чергу він потрапити НЕ може».
    Сигнал, за яким відкрилась угода, вигорає НАЗАВЖДИ — навіть поки та угода
    ще ВІДКРИТА (тобто до будь-якого закриття)."""
    o, s = _ff()
    sig = time.time() - 3600
    o._pending4['LDOUSDT'] = {'dir': 'SHORT', 'added_at': sig}
    o.note_signal_consumed('LDOUSDT', sig)
    _check('LDOUSDT' not in o._pending4, 'монета мала вийти з черги')
    ok, why = o._is_new_situation('LDOUSDT', sig, s)
    _check(ok is False and 'УЖЕ ВІДПРАЦЮВАВ' in why,
           f'вигорілий сигнал не має прийматись знову: {why}')
    print('✓ сигнал, за яким відкрилась угода, вигорає назавжди')


def test_consumed_purges_every_queue_not_just_one():
    """«З Черги має бути ПОВНІСТЮ видалена ця монета» — з усіх, не лише з тієї,
    що відкрила."""
    o, _ = _ff()
    sig = time.time() - 600
    for q in (o._pending, o._pending2, o._pending3, o._pending4):
        q['LDOUSDT'] = {'dir': 'SHORT', 'added_at': sig}
    o._q4_layers_cache['LDOUSDT'] = (time.time(), {})
    o.note_signal_consumed('LDOUSDT', sig)
    for name, q in (('Черга-1', o._pending), ('Черга-2', o._pending2),
                    ('Черга-3', o._pending3), ('Черга-4', o._pending4)):
        _check('LDOUSDT' not in q, f'{name}: монета мала бути прибрана ПОВНІСТЮ')
    _check('LDOUSDT' not in o._q4_layers_cache, 'кеш шарів теж')
    print('✓ відкриття прибирає монету з УСІХ черг, а не лише з однієї')


def test_new_signal_after_burn_is_accepted():
    """Повернення можливе — але ЛИШЕ з НОВИМ сигналом (новий added_at)."""
    o, s = _ff()
    burned = time.time() - 3600
    o.note_signal_consumed('LDOUSDT', burned)
    ok, _ = o._is_new_situation('LDOUSDT', time.time(), s)
    _check(ok is True, 'НОВИЙ сигнал після вигоряння має проходити')
    print('✓ новий сигнал (новий added_at) приймається — шлях відкритий')


def test_startup_purge_drops_restored_zombies():
    """Після рестарту стан приїжджає з БД — відпрацьований запис міг повернутись
    у чергу і далі крутитись у СЕРЕДИНІ алгоритму. Чистимо ДО першого тіку."""
    o, _ = _ff()
    old = time.time() - 7200
    o._pending4['LDOUSDT'] = {'dir': 'SHORT', 'added_at': old}
    o._pending2['ETHUSDT'] = {'dir': 'LONG', 'added_at': old}
    o._pending4['FRESHUSDT'] = {'dir': 'LONG', 'added_at': time.time()}
    o._consumed_signal_at['LDOUSDT'] = old          # уже відпрацював
    o._last_trade_end['ETHUSDT'] = old + 60         # угода закрилась після сигналу
    n = o._purge_spent_queue_records()
    _check(n == 2, f'мали прибрати 2 зомбі, прибрано {n}')
    _check('LDOUSDT' not in o._pending4 and 'ETHUSDT' not in o._pending2, 'зомбі мали піти')
    _check('FRESHUSDT' in o._pending4, 'свіжий запис чіпати не можна')
    print('✓ стартова чистка прибирає відновлених зомбі, свіжі лишає')


def test_stuck_record_is_evicted_even_with_no_ttl():
    """«Якщо монета застрягає у Черга-4 — її також потрібно повністю викинути».
    ♾ «Без терміну» цю межу НЕ вимикає."""
    now = time.time()
    _check(FF._q4_stuck(now - 25 * 3600, now, 24.0) is True,
           '25 год при межі 24 → застряг')
    _check(FF._q4_stuck(now - 5 * 3600, now, 24.0) is False,
           '5 год — ще не застряг')
    _check(FF._q4_stuck(now - 999 * 3600, now, 0.0) is False,
           '0 = запобіжник вимкнено (свідомий вибір)')
    for bad in (None, '', 'abc'):
        _check(FF._q4_stuck(bad, now, 24.0) is False, f'сміття {bad!r} не виселяє')
    print('✓ жорстка стеля застрягання працює незалежно від ♾')


def test_hard_cap_default():
    _check(ffmod.DEFAULT_SETTINGS.get('queue4_hard_max_hours') == 24,
           'стеля застрягання за замовчуванням 24 год')
    print('✓ дефолт стелі застрягання — 24 год')


# ═════ 🧬 ПОХОДЖЕННЯ ЇДЕ РАЗОМ ІЗ ВІДКРИТТЯМ (кейс OPUSDT) ════════════════
def test_origin_trace_is_handed_to_trade_manager_before_open():
    """🐞 Кейс OPUSDT: Черга-4 писала походження ОКРЕМИМ рядком, і того рядка
    в експортованому лозі не було ЗОВСІМ — лишалось «Відкрито paper …» без
    жодного контексту. Тепер ланцюг передається в TM ДО відкриття і друкується
    в ТОМУ САМОМУ рядку, тож загубитись не може."""
    o, s = _ff()
    o._fuel_managed = {}
    o._engine_skip = {}
    handed = {}

    class _TM:
        scanner = None
        def set_origin_trace(self, sym, text): handed[sym] = text
        def manual_open(self, *a, **k): return {'ok': False, 'reason': 'stub'}
    o._get_tm = lambda: _TM()
    o._tm_has_position = lambda sym, real: False
    o._exhaustion = lambda sym, side: 0
    o._soft_safeguard = lambda *a, **k: (True, '')
    o._live_price = lambda sym: 100.0
    try:
        o._open('OPUSDT', 'SHORT', {'mark_price': 0.09714}, dict(s, enabled=True),
                opened_by='vob_alert → Q4', skip_safeguard=True,
                skip_ctr_safeguard=True,
                origin_trace='Черга-4 · сигнал: 🟪 Volumized OB · шари 3/3')
    except Exception:
        pass          # відкриття тут не потрібне — важливо, що трасу передали
    _check(handed.get('OPUSDT'),
           f'ланцюг походження мав дійти до TM ДО відкриття: {handed}')
    _check('шари 3/3' in handed['OPUSDT'], f'розклад шарів має бути в ньому: {handed}')
    print('✓ походження передається в TM до відкриття (одним рядком, не окремим)')


if __name__ == '__main__':
    test_signal_older_than_last_close_is_refused()
    test_signal_after_close_is_accepted()
    test_first_ever_trade_is_not_blocked()
    test_missing_signal_time_does_not_invent_a_refusal()
    test_rule_can_be_switched_off()
    test_stale_signal_age_limit()
    test_age_limit_off_by_default()
    test_defaults()
    test_manual_close_purges_all_queues()
    test_manual_close_lock_when_enabled()
    test_manual_close_lock_expires_and_clears()
    test_manual_close_lock_off_by_default()
    test_lock_can_be_cleared_by_hand()
    test_origin_trace_answers_where_it_came_from()
    test_origin_trace_marks_opposite_replacement()
    test_origin_trace_survives_missing_data()
    test_wait_formatting()
    test_ldo_case_close_purges_spent_queue_records()
    test_close_purges_every_queue()
    test_close_keeps_record_created_after_it()
    test_close_without_queue_records_is_quiet()
    test_refusal_is_logged_once_not_every_tick()
    test_consumed_signal_never_accepted_again()
    test_consumed_purges_every_queue_not_just_one()
    test_new_signal_after_burn_is_accepted()
    test_startup_purge_drops_restored_zombies()
    test_stuck_record_is_evicted_even_with_no_ttl()
    test_hard_cap_default()
    test_origin_trace_is_handed_to_trade_manager_before_open()
    print('\nУсі тести «нової ситуації» пройдено ✅')
