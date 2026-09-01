"""Тести двох правок за кейсом ARBUSDT (31.08).

🔄 1. ПРОТИЛЕЖНИЙ СИГНАЛ НЕ ЗНИКАЄ.
О 15:02 по ARBUSDT прийшов LONG, який пройшов УСІ фільтри (OB(1h)✓ ·
Прогноз[AND] 1H LONG 75% + 4H LONG 50%✓ · Decision LONG 82%✓). Але бот тримав
збитковий SHORT, а «Reverse on opposite signal» вимкнено — TM написав
«тримаємо SHORT» і сигнал ЗНИК НАЗАВЖДИ. Бот досидів у шорті до стопа −8.58%,
а рух пішов саме в бік того LONG. Тепер такий сигнал іде в Чергу-4 з міткою
`opp_wait` і чекає закриття позиції.

📐 2. ГЕЙТ ЗА R ПЕРЕД ВІДКРИТТЯМ.
Той самий ARBUSDT зайшов у шорт зі стопом 8.23% від входу і ціллю 1.60% —
0.19R. Тепер очікуваний R рахується ЩЕ ДО входу, з ТИХ САМИХ джерел, що
дадуть реальний стоп і реальну ціль, і угода з R нижче порогу не відкривається.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
for n in ('pybit', 'pybit.unified_trading'):
    if n not in sys.modules:
        sys.modules[n] = types.ModuleType(n)
sys.modules['pybit.unified_trading'].HTTP = object

_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg

_LOG = []
_lg = types.ModuleType('detection.activity_log')
_lg.log_activity = lambda sym, kind, text, **kw: _LOG.append((kind, text))
sys.modules['detection.activity_log'] = _lg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


_load('detection.signal_labels', 'detection/signal_labels.py')
_load('detection.setup_grader', 'detection/setup_grader.py')
_load('detection.trade_pilot', 'detection/trade_pilot.py')
ffmod = _load('detection.fuel_filter', 'detection/fuel_filter.py')
FF = ffmod.FuelFilterDaemon


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _text():
    return ' || '.join(t for _, t in _LOG)


# ═════════ 🔄 ПРОТИЛЕЖНИЙ СИГНАЛ ═══════════════════════════════════════════
def test_opposite_signal_is_routed_to_the_queue_not_dropped():
    """Замок на розводку в Trade Manager: коли реверс вимкнено, сигнал мусить
    іти в чергу, а не повертати «duplicate»."""
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    i = src.index('_route_ff = (_pos is None)')
    j = src.index('if _route_ff:', i)
    body = src[i:j]
    _check("elif self._settings.get('queue_opposite_signal', True):" in body,
           'протилежний сигнал має маршрутизуватись у чергу')
    _check('_opp_wait = True' in body, 'і позначатись як «чекає закриття»')
    _check("'queue_opposite_signal': True," in src,
           'тумблер має бути УВІМКНЕНИЙ за замовчуванням — інакше сигнал і далі '
           'зникатиме, а саме через це й загубився ARBUSDT LONG')
    _check('opp_wait=_opp_wait' in src, 'прапорець мусить доїхати до intercept')
    print('✓ протилежний сигнал іде в Чергу-4, а не зникає')


def test_open_position_no_longer_blocks_the_opposite_record():
    """`_already_open` блокує постановку в Чергу-4, поки угода відкрита. Для
    протилежного сигналу це саме та ситуація, заради якої він і ставиться."""
    src = open(os.path.join(_ROOT, 'detection/fuel_filter.py')).read()
    i = src.index('_already_open = bool(sym in self._fuel_managed')
    body = src[i:i + 1400]
    _check('if opp_wait:' in body and '_already_open = False' in body,
           'opp_wait має знімати блок «вже в угоді»')
    _check("'opp_wait': bool(opp_wait)" in src,
           'запис у Черзі-4 мусить нести прапорець')
    print('✓ відкрита угода більше не блокує запис протилежного сигналу')


def test_close_purge_keeps_the_waiting_record():
    """🐞 ПАСТКА, яку легко було не помітити: `note_trade_closed` прибирає з
    черг усе, що старше за закриття. Запис протилежного сигналу за
    визначенням старіший — і його змело б рівно тоді, коли він нарешті став
    актуальним."""
    o = FF.__new__(FF)
    import threading
    o._lock = threading.RLock()
    o._last_trade_end = {}
    o._pending = {}; o._pending2 = {}; o._pending3 = {}
    o._pending4 = {'ARBUSDT': {'dir': 'LONG', 'added_at': 100.0, 'opp_wait': True}}
    o._q4_cache_drop = lambda s: None
    o._q4_lit_at = {}
    o._persist_state = lambda: None
    o._fmt_wait = FF._fmt_wait
    _LOG.clear()
    o.note_trade_closed('ARBUSDT')
    _check('ARBUSDT' in o._pending4,
           'запис, що чекав саме цього закриття, прибирати НЕ можна')

    # А звичайний (не opp_wait) запис так само має зникати, як і раніше.
    o._pending4 = {'ARBUSDT': {'dir': 'LONG', 'added_at': 100.0}}
    o.note_trade_closed('ARBUSDT')
    _check('ARBUSDT' not in o._pending4,
           'звичайний відпрацьований запис мусить прибиратись, як і раніше')
    print('✓ чистка на закритті не з’їдає запис протилежного сигналу')


def test_new_situation_rule_exempts_the_waiting_record():
    """Правило «нова ситуація» відхиляє сигнали, старіші за закриття. Для
    запису, що ЧЕКАВ цього закриття, виняток обовʼязковий — інакше він ніколи
    не відкриється. Але «сигнал уже ВІДПРАЦЮВАВ» (burned) лишається в силі."""
    o = FF.__new__(FF)
    o._consumed_signal_at = {}
    o._last_trade_end = {'ARBUSDT': 500.0}
    o._fmt_wait = FF._fmt_wait
    s = {'require_new_situation': True, 'open_max_signal_age_min': 0}

    ok, why = o._is_new_situation('ARBUSDT', 100.0, s)
    _check(not ok, f'звичайний старий запис має відхилятись: {why}')

    ok2, why2 = o._is_new_situation('ARBUSDT', 100.0, s, opp_wait=True)
    _check(ok2, f'запис, що чекав закриття, мусить пройти: {why2}')

    o._consumed_signal_at['ARBUSDT'] = 200.0
    ok3, why3 = o._is_new_situation('ARBUSDT', 100.0, s, opp_wait=True)
    _check(not ok3 and 'ВІДПРАЦЮВАВ' in why3,
           f'вигорілий сигнал не рятує навіть opp_wait: {why3}')
    print('✓ виняток діє точково: тільки для очікування закриття')


def test_telegram_notification_exists():
    src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    _check('def _notify_opposite_queued' in src, 'потрібне сповіщення в TG')
    i = src.index('def _notify_opposite_queued')
    body = src[i:i + 1800]
    _check('ПРОТИЛЕЖНИЙ СИГНАЛ' in body, 'заголовок має називати подію')
    _check('Чергу-4' in body, 'і казати, що сигнал не втрачено')
    _check('is_test=is_shadow' in body, 'paper має йти у свій топік')
    _check('self._notify_opposite_queued(symbol, side, _pos_side, _pos)' in src,
           'виклик має стояти на шляху постановки в чергу')
    print('✓ Telegram-сповіщення про протилежний сигнал є')


# ═════════ 📐 ГЕЙТ ЗА R ═════════════════════════════════════════════════════
class _TM:
    def __init__(self, price, ctx):
        self._price, self._ctx = price, ctx
    def _get_current_price(self, sym):
        return self._price
    def _pilot_context(self, sym, side):
        return self._ctx


def _ff_rr(price, sl_bounds, ctx):
    o = FF.__new__(FF)
    o._q4_rr_objective = {}
    o._get_tm = lambda: _TM(price, ctx)
    o._fuel_dir_smoothed = lambda s: {'mark_price': price}
    o._q4_ob_bounds_1h = lambda s: sl_bounds
    o._q4_ob_bounds_vob = lambda s, side, tf='15m': None
    o._q4_sl_side_ok = FF._q4_sl_side_ok
    o._fmt_price = lambda v: f'{v}'
    o._q4_sl_candidates = lambda sym, side, src: FF._q4_sl_candidates(o, sym, side, src)
    o._q4_pick_sl = lambda *a: FF._q4_pick_sl(o, *a)
    return o


ARB_CTX = {'swing': {'high': {'price': 0.0900, 'label': 'Weak High'},
                     'low': {'price': 0.08288, 'label': 'Weak Low'}},
           'runway': None, 'poc': None, 'vah': None, 'val': None}


def test_arb_case_is_rejected_by_the_r_gate():
    """Реальний кейс: вхід 0.08423, стоп із 1H OB 0.09116 (8.23%), найдальша
    ціль — Weak Low 0.08288 (1.60%). Це 0.19R."""
    o = _ff_rr(0.08423, (0.09093, 0.08500, '1h', 'BEARISH'), ARB_CTX)
    r, detail = o._q4_expected_r('ARBUSDT', 'SHORT',
                                 {'queue4_sl_source': '1h',
                                  'queue3_vob_sl_buffer_pct': 0.25,
                                  'autosl_max_pct': 0})
    _check(r is not None, f'R мав порахуватись: {detail}')
    _check(r < 1.0, f'кейс ARB мусить давати R<1, отримано {r} ({detail})')
    _check('стоп' in detail and 'ціль' in detail,
           f'розклад має називати обидві відстані: {detail}')
    print(f'✓ кейс ARBUSDT: {detail} → відсіюється порогом 1.0R')


def test_good_setup_passes_the_r_gate():
    """Дзеркально: той самий стоп, але далека ціль → R проходить.
    ⚠️ Ціль мусить бути В МЕЖАХ вікна автопілота (`max_target_dist_pct`=15%),
    інакше `collect_targets` її відкине як «вже не ціль цієї угоди» — саме на
    цьому впала перша версія фікстури."""
    ctx = {'swing': {'high': {'price': 0.0900, 'label': 'Weak High'},
                     'low': {'price': 0.0730, 'label': 'Weak Low'}},
           'runway': None, 'poc': None, 'vah': None, 'val': None}
    o = _ff_rr(0.08423, (0.09093, 0.08500, '1h', 'BEARISH'), ctx)
    r, detail = o._q4_expected_r('ARBUSDT', 'SHORT',
                                 {'queue4_sl_source': '1h',
                                  'queue3_vob_sl_buffer_pct': 0.25,
                                  'autosl_max_pct': 0})
    _check(r is not None and r > 1.0, f'далека ціль → R>1: {r} ({detail})')
    print(f'✓ той самий стоп із далекою ціллю: {r}R — проходить')


def test_stop_ceiling_changes_the_r():
    """Стеля SL підтягує стоп → ризик менший → R більший. Це той самий
    `autosl_max_pct`, що діє і на реальне виставлення стопа."""
    o = _ff_rr(0.08423, (0.09093, 0.08500, '1h', 'BEARISH'), ARB_CTX)
    base = {'queue4_sl_source': '1h', 'queue3_vob_sl_buffer_pct': 0.25}
    r_off, _ = o._q4_expected_r('ARBUSDT', 'SHORT', dict(base, autosl_max_pct=0))
    r_cap, _ = o._q4_expected_r('ARBUSDT', 'SHORT', dict(base, autosl_max_pct=2.5))
    _check(r_cap > r_off, f'стеля мусить покращити R: {r_off} → {r_cap}')
    print(f'✓ стеля SL 2.5% піднімає R з {r_off} до {r_cap}')


def test_no_target_means_no_verdict_not_a_refusal():
    """Нема з чого рахувати → `(None, причина)`. Гейт тоді НЕ блокує:
    вигаданої відмови не даємо."""
    o = _ff_rr(0.08423, (0.09093, 0.08500, '1h', 'BEARISH'),
               {'swing': None, 'runway': None, 'poc': None})
    r, why = o._q4_expected_r('ARBUSDT', 'SHORT',
                              {'queue4_sl_source': '1h',
                               'queue3_vob_sl_buffer_pct': 0.25})
    _check(r is None and 'цілі' in why, f'мала бути чесна причина: {why}')

    o2 = _ff_rr(0.08423, None, ARB_CTX)          # немає блоку для стопа
    r2, why2 = o2._q4_expected_r('ARBUSDT', 'SHORT',
                                 {'queue4_sl_source': '1h',
                                  'queue3_vob_sl_buffer_pct': 0.25})
    _check(r2 is None and 'стоп' in why2, f'мала бути чесна причина: {why2}')
    print('✓ без даних R не вигадується, гейт не блокує')


def test_r_and_real_sl_come_from_one_source():
    """ЗАМОК: якби R рахувався зі свого «приблизного» стопа, гейт відсіював би
    за одним числом, а стоп ставав би за іншим. Обидва шляхи мусять кликати
    `_q4_pick_sl`."""
    src = open(os.path.join(_ROOT, 'detection/fuel_filter.py')).read()
    _check(src.count('self._q4_pick_sl(') >= 2,
           'і розрахунок R, і виставлення стопа мусять брати ОДНУ функцію')
    i = src.index('def _q4_expected_r')
    body = src[i:i + 3200]
    _check('self._q4_pick_sl(' in body, 'R бере стоп зі спільної функції')
    _check('clamp_sl_distance' in body,
           'і ту саму стелю, що діє на реальний стоп')
    print('✓ R і реальний стоп рахуються з одного джерела')


def test_gate_objective_is_locked_into_the_trade():
    """🐞 ЗНАЙДЕНО НА ПРОДІ (VETUSDT 01.09). Гейт пропустив шорт із вердиктом
    «1.0R · ціль головний пул ліквідності +2.75%», а через 90 секунд автопілот
    обрав ІНШУ ціль — Strong Low +1.25% — і угода стала 0.43R. Тобто число, ЗА
    ЯКИМ пропустили, не було числом, яке потім реально стоїть в угоді.

    Причина: ціль спирається на `runway`, а він береться лише коли напрямок
    МММ збігається з боком угоди. МММ фліпнув — пул зник із контексту.

    Замок: ціль, за якою гейт ухвалив рішення, мусить доїхати до угоди."""
    vet_ctx = {'swing': {'high': {'price': 0.006900, 'label': 'Weak High'},
                         'low': {'price': 0.006551, 'label': 'Strong Low'}},
               'runway': None, 'poc': None, 'vah': None, 'val': None}
    o = _ff_rr(0.006634, (0.006809, 0.006500, '1h', 'BEARISH'), vet_ctx)
    r, _d = o._q4_expected_r('VETUSDT', 'SHORT',
                             {'queue4_sl_source': '1h',
                              'queue3_vob_sl_buffer_pct': 0.25})
    _check(r is not None, 'R мав порахуватись')
    _check('VETUSDT' in o._q4_rr_objective,
           'ціль рішення мусить зберігатись, а не губитись')

    src = open(os.path.join(_ROOT, 'detection/fuel_filter.py')).read()
    i = src.index('# 📐 ГЕЙТ ЗА R')
    j = src.index('opened = self._open(sym, _open_dir', i)
    _check('set_pending_objective' in src[i:j],
           'ціль мусить передаватись у Trade Manager перед відкриттям')

    tm_src = open(os.path.join(_ROOT, 'detection/trade_manager.py')).read()
    _check('def set_pending_objective' in tm_src, 'TM має приймати ціль')
    _check(tm_src.count("pilot_objective'] = _obj0") >= 2,
           'ціль має лягати на позицію в ОБОХ шляхах відкриття (real + paper)')
    print('✓ ціль гейта фіксується в угоді — R рішення = R у колонці')


def test_gate_defaults_and_placement():
    src = open(os.path.join(_ROOT, 'detection/fuel_filter.py')).read()
    _check("'queue4_min_rr': 1.0," in src, 'поріг за замовчуванням 1.0R')
    i = src.index('# 📐 ГЕЙТ ЗА R')
    j = src.index('opened = self._open(sym, _open_dir', i)
    body = src[i:j]
    _check('continue' in body, 'не пройшов R → відкриття відкладено')
    _check('_q4_rr_logged' in body, 'потрібен анти-флуд, як у повторної перевірки')
    _check('_pending4.pop' not in body,
           'запис НЕ виселяємо: R рухомий, монета пробує знову')
    print('✓ гейт стоїть перед відкриттям, дефолт 1.0R, запис лишається в черзі')


if __name__ == '__main__':
    test_opposite_signal_is_routed_to_the_queue_not_dropped()
    test_open_position_no_longer_blocks_the_opposite_record()
    test_close_purge_keeps_the_waiting_record()
    test_new_situation_rule_exempts_the_waiting_record()
    test_telegram_notification_exists()
    test_arb_case_is_rejected_by_the_r_gate()
    test_good_setup_passes_the_r_gate()
    test_stop_ceiling_changes_the_r()
    test_no_target_means_no_verdict_not_a_refusal()
    test_r_and_real_sl_come_from_one_source()
    test_gate_objective_is_locked_into_the_trade()
    test_gate_defaults_and_placement()
    print('\nУсі тести протилежного сигналу + гейта за R пройдено ✅')
