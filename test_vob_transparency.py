"""Тест: VOB — «блок на графіку = сигнал» + чистий лог.

Перевіряє:
  • _vob_edge_outcome — СВІЖИЙ блок фаєриться і на ПЕРШОМУ показі ('fresh'),
    старий на першому показі → 'first_sight' (тиха база), duplicate, stale;
  • _vob_log_decision — оновлює self._vob_diag (для get_state/UI) і НІКОЛИ не
    пише в 🧾 Лог роботи бота (лог лишаємо чистим: туди йдуть лише реальні
    сигнали fired + вердикт rejected з основного шляху).

Модуль вантажиться ізольовано (без важкого detection/__init__).
"""
import os
import sys
import types
import importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))

_pkg = types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg

# Фейковий detection.activity_log — щоб довести, що логер VOB туди НЕ пише.
_LOGS = []
_al = types.ModuleType('detection.activity_log')
_al.log_activity = lambda *a, **k: _LOGS.append((a, k))
sys.modules['detection.activity_log'] = _al


def _load(mod_name, rel_path):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_ROOT, rel_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


scmod = _load('detection.smc_scanner', 'detection/smc_scanner.py')
SC = scmod.SMCScanner


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ── 1. «Блок на графіку = сигнал»: свіжий фаєриться і на першому показі ──────
def test_edge_outcome_fresh_first_sight_fires():
    f = SC._vob_edge_outcome
    # ПЕРШИЙ показ + свіжий → 'fresh' (обробляємо поточний блок, НЕ ковтаємо).
    _check(f(None, 200, 3, 7) == 'fresh', 'перший показ + свіжий → fresh (сигнал!)')
    _check(f(None, 200, 7, 7) == 'fresh', 'перший показ + вік==поріг → fresh')
    # ПЕРШИЙ показ + старий → 'first_sight' (старий блок не сигнал → тиха база).
    _check(f(None, 200, 8, 7) == 'first_sight', 'перший показ + старий → first_sight')
    print('✓ свіжий блок фаєриться і на першому показі; старий → тиха база')


def test_edge_outcome_running():
    f = SC._vob_edge_outcome
    _check(f(100, 100, 0, 7) == 'duplicate', 'ft==prev → duplicate')
    _check(f(100, 90, 0, 7) == 'duplicate', 'ft<prev → duplicate')
    _check(f(100, 200, 3, 7) == 'fresh', 'новіший + свіжий → fresh')
    _check(f(100, 200, 8, 7) == 'stale', 'новіший + старий → stale (коли вікно задане)')
    print('✓ у процесі роботи: duplicate/fresh/stale — коректно')


def test_no_age_gate_default_fires_any_new_vob():
    # ДЕФОЛТ vob_alert_max_age_bars=0 → _max_age=None → БЕЗ вікна свіжості:
    # блок будь-якого віку, що новий (edge за ft), фаєрить → «блок на графіку = сигнал».
    f = SC._vob_edge_outcome
    _check(f(None, 200, 9999, None) == 'fresh', 'перший показ, будь-який вік, БЕЗ вікна → fresh')
    _check(f(100, 200, 9999, None) == 'fresh', 'новий ft, будь-який вік, БЕЗ вікна → fresh (не stale)')
    _check(f(100, 100, 9999, None) == 'duplicate', 'той самий ft → duplicate (fire раз)')
    print('✓ БЕЗ вікна свіжості (деф.): будь-який НОВИЙ VOB фаєрить, старий блок теж')


# ── 2. Логер НЕ засмічує лог; усе — лише в діагностику для UI ────────────────
def _mk():
    obj = SC.__new__(SC)
    obj._vob_diag = {}
    obj._lock = __import__('threading').RLock()
    return obj


def test_log_decision_never_writes_activity_log():
    _LOGS.clear()
    o = _mk()
    for oc in ('stale', 'first_sight', 'epoch', 'duplicate', 'no_candidate',
               'fired', 'filtered'):
        o._vob_log_decision('BTCUSDT', 'SHORT', oc, age=5, max_age=7, ft=1,
                            vol_tf='5m', detail='x', log_it=True)  # log_it ІГНОРУЄТЬСЯ
    _check(len(_LOGS) == 0, 'логер VOB НЕ пише в activity_log (лог чистий)')
    print('✓ логер нічого не ліпить у лог (усе — лише в діагностику)')


def test_log_decision_updates_diag():
    o = _mk()
    o._vob_log_decision('ETHUSDT', 'SHORT', 'stale', age=25, max_age=7, ft=99,
                        vol_tf='5m', detail='старий')
    d = o._vob_diag['ETHUSDT']
    _check(d['outcome'] == 'stale' and d['age'] == 25 and d['ft'] == 99,
           'діагностику збережено для get_state/UI')
    _check(d['label'].startswith('⌛'), 'людський підпис проставлено')
    print('✓ діагностика оновлюється (джерело для UI-колонки «стан VOB»)')


# ── 3. Catch-all: обидва напрямки за один цикл, жоден не губиться ────────────
def test_pick_candidates_both_sides():
    f = SC._vob_pick_candidates
    bull = {'formation_time': 100}
    bear = {'formation_time': 105}
    out = f(bull, bear)
    _check(len(out) == 2, 'обидва боки взято (не лише максимальний за часом)')
    # відсортовано за часом: старіший (bull 100) першим, новіший (bear 105) останнім
    _check(out[0][0] == 'LONG' and out[0][2] == 100, 'старіший (LONG 100) першим')
    _check(out[1][0] == 'SHORT' and out[1][2] == 105, 'новіший (SHORT 105) останнім')
    print('✓ обидва напрямки за один цикл (раніше брався лише один → інший губився)')


def test_pick_candidates_skips_breaker_and_invalid():
    f = SC._vob_pick_candidates
    _check(f({'formation_time': 100, 'breaker': True}, None) == [], 'breaker пропущено')
    _check(f({'formation_time': 0}, None) == [], 'ft<=0 пропущено')
    _check(f(None, None) == [], 'немає OB → порожньо')
    _check(len(f({'formation_time': 50}, None)) == 1, 'лише бичачий → один кандидат')
    print('✓ breaker/невалідні пропускаються, один бік теж ок')


def test_per_side_base_independent():
    # Симуляція пер-напрямкової бази: рух бази SHORT НЕ маскує новий LONG.
    seen = {'SHORT': 105}
    # новий LONG ft=100: prev(LONG)=None → перший показ; свіжий → 'fresh' (сигнал!),
    # хоча SHORT-база вже 105. Раніше єдина база=105 з'їла б цей LONG як duplicate.
    out_long = SC._vob_edge_outcome(seen.get('LONG'), 100, 2, 7)
    _check(out_long == 'fresh', 'новий LONG не маскується SHORT-базою → fresh')
    out_short = SC._vob_edge_outcome(seen.get('SHORT'), 105, 2, 7)
    _check(out_short == 'duplicate', 'той самий SHORT (105) → duplicate')
    print('✓ бази LONG/SHORT незалежні — протилежний бік не з’їдається')


# ── 4. «1 VOB на 1H OB»: свіжий 1H-OB скидає 5m і чекає новий 5m-VOB ─────────
def test_epoch_reset_needed():
    f = SC._vob_epoch_reset_needed
    _check(f(None, 1000) is True, 'перший 1H-OB (prev None) → reset')
    _check(f(1000, 2000) is True, 'інший bar_time → СВІЖИЙ 1H-OB → reset')
    _check(f(1000, 1000) is False, 'той самий 1H-OB → без reset')
    _check(f(1000, None) is False, 'немає 1H-OB → без reset (сигналу нема)')
    print('✓ свіжий 1H-OB (інший bar_time) скидає 5m-базу; той самий — ні')


def test_epoch_already_fired():
    f = SC._vob_epoch_already_fired
    _check(f(1000, 1000) is True, 'у цій епосі вже фаєрили → чекаємо новий 1H-OB')
    _check(f(1000, 2000) is False, 'нова епоха → можна фаєрити')
    _check(f(None, 1000) is False, 'ще не фаєрили → можна')
    _check(f(1000, None) is False, 'немає 1H-OB → не рахуємо як fired')
    print('✓ «1 сигнал на 1 1H-OB»: повторний fire у тій самій епосі блокується')


def test_vob_numbering():
    # Нумерація VOB у межах 1H-OB: свіжий 1H-OB обнуляє лічильник; VOB #1 = сигнал,
    # #2,3,… — лише номер; наявний блок при reset = #0 (чекаємо НОВИЙ = #1); усі
    # нові VOB рахуються (жодного не пропускаємо).
    # РЕЗУЛЬТАТИВНІСТЬ: кандидатом є КОЖЕН новий VOB, доки на цьому 1H-OB не
    # було спрацювання, що пройшло фільтри. Відсів фільтром такт НЕ витрачає.
    cand = SC._vob_is_signal_candidate
    _check(cand(None, 1000) is True, 'ще не було результативного → кандидат')
    _check(cand(1000, 1000) is False, 'результативний уже був на цьому 1H-OB → ні')
    _check(cand(1000, 2000) is True, 'новий 1H-OB → знову кандидат')

    # Симуляція лічильника як у скані (обидва боки рахуються в один лічильник):
    reset = SC._vob_epoch_reset_needed
    counter = 0
    seen = {}
    def new_vob(side, ft):
        # правило скану: новий, якщо prev None або ft > prev
        nonlocal counter
        prev = seen.get(side)
        if not (prev is None or ft > prev):
            return None  # не новий → #0
        seen[side] = ft
        counter += 1
        return counter
    # Свіжий 1H-OB(A): reset → лічильник=0, наявні базуються (#0)
    _check(reset(None, 1000) is True, 'новий 1H-OB → reset')
    counter = 0; seen = {'LONG': 100, 'SHORT': 90}   # наявні при reset → #0
    _check(new_vob('LONG', 100) is None, 'наявний LONG при reset → #0 (чекаємо новий)')
    # Далі йдуть НОВІ VOB (обидва боки), рахуються всі:
    _check(new_vob('SHORT', 110) == 1, 'перший НОВИЙ VOB → #1 (кандидат)')
    _check(new_vob('LONG', 120) == 2, 'наступний НОВИЙ VOB → #2')
    _check(new_vob('SHORT', 130) == 3, 'ще НОВИЙ → #3')
    # Сценарій «#1 і #2 зарубали фільтри, #3 пройшов»: такт витрачає ЛИШЕ #3.
    fired = None                       # ще не було результативного
    _check(cand(fired, 1000) is True, '#1 кандидат')
    # #1 відсіяно фільтром → fired НЕ змінюється
    _check(cand(fired, 1000) is True, '#2 знову кандидат (відсів такт не витратив)')
    fired = 1000                       # #3 пройшов усі фільтри → такт витрачено
    _check(cand(fired, 1000) is False, 'після результативного — більше не сигнал')
    # Новий 1H-OB(B): reset → лічильник=0 знову
    _check(reset(1000, 2000) is True, 'новий 1H-OB(B) → reset')
    counter = 0; seen = {'SHORT': 130}
    _check(new_vob('LONG', 200) == 1, 'у новій епосі перший НОВИЙ VOB → #1 (сигнал) знову')
    print('✓ нумерація: reset на 1H-OB, #1=сигнал, #2,3=номер, усі рахуються')


def test_epoch_flow_reset_then_new_5m():
    # Симуляція повного такту користувача:
    #   свіжий 1H-OB(A) скидає наявний 5m(100)→duplicate → чекаємо новий 5m
    #   новий 5m(200) у епосі A → fresh (сигнал) → епоха A fired
    #   свіжий 1H-OB(B) знову скидає → чекаємо новий 5m → новий 5m(400) → fresh
    edge = SC._vob_edge_outcome
    # epoch A: reset базує наявний 5m=100 → duplicate
    seen = {'LONG': 100}
    _check(edge(seen.get('LONG'), 100, 2, 7) == 'duplicate', 'наявний 5m після reset → duplicate')
    # новий 5m=200 → fresh (сигнал)
    _check(edge(seen.get('LONG'), 200, 2, 7) == 'fresh', 'новий 5m у епосі A → fresh (сигнал)')
    # epoch B (новий 1H-OB): reset_needed True, базуємо наявний 5m=200
    _check(SC._vob_epoch_reset_needed(1000, 2000) is True, 'новий 1H-OB → reset')
    seen_b = {'LONG': 200}
    _check(edge(seen_b.get('LONG'), 200, 2, 7) == 'duplicate', 'після reset епохи B наявний 5m → duplicate')
    _check(edge(seen_b.get('LONG'), 400, 2, 7) == 'fresh', 'новий 5m у епосі B → fresh (сигнал)')
    print('✓ такт: 1H-OB скидає 5m → чекає новий 5m → сигнал → новий 1H-OB → знову')


def test_vob_state_persist_roundtrip():
    """Стан такту (лічильник/епоха/fired/база) мусить ПЕРЕЖИВАТИ рестарт —
    інакше після кожного botupdate такт починався заново («VOB 0 · чекаємо #1»)
    і бот міг відкрити ДРУГУ угоду на тому самому 1H-OB."""
    store = {}

    class _DB:
        def set_setting(self, k, v): store[k] = v
        def get_setting(self, k, d=None): return store.get(k, d)

    a = SC.__new__(SC)
    a.db = _DB()
    a._vob_alert_seen = {'BTCUSDT': {'LONG': 111, 'SHORT': 222}}
    a._vob_ob_epoch = {'BTCUSDT': 1000}
    a._vob_epoch_fired = {'BTCUSDT': 1000}
    a._vob_counter = {'BTCUSDT': 3}
    a._persist_vob_state()
    _check(scmod.DB_KEY_VOB_STATE in store, 'стан записано в БД')

    # «рестарт»: новий інстанс читає той самий сховок
    b = SC.__new__(SC)
    b.db = _DB()
    b._vob_alert_seen = {}; b._vob_ob_epoch = {}
    b._vob_epoch_fired = {}; b._vob_counter = {}
    b._load_vob_state()
    _check(b._vob_counter.get('BTCUSDT') == 3, 'лічильник відновлено (не 0)')
    _check(b._vob_epoch_fired.get('BTCUSDT') == 1000, 'fired-епоха відновлена')
    _check(b._vob_ob_epoch.get('BTCUSDT') == 1000, 'такт 1H-OB відновлено')
    _check(b._vob_alert_seen.get('BTCUSDT') == {'LONG': 111, 'SHORT': 222},
           'пер-напрямкова база відновлена')
    # і головне: після рестарту повторного сигналу на тому самому 1H-OB НЕ буде
    _check(SC._vob_is_signal_candidate(b._vob_epoch_fired.get('BTCUSDT'), 1000) is False,
           'після рестарту такт НЕ обнуляється → другої угоди на тому ж 1H-OB немає')
    print('✓ стан VOB переживає рестарт (лічильник/епоха/fired/база)')


if __name__ == '__main__':
    test_edge_outcome_fresh_first_sight_fires()
    test_edge_outcome_running()
    test_no_age_gate_default_fires_any_new_vob()
    test_log_decision_never_writes_activity_log()
    test_log_decision_updates_diag()
    test_pick_candidates_both_sides()
    test_pick_candidates_skips_breaker_and_invalid()
    test_per_side_base_independent()
    test_epoch_reset_needed()
    test_epoch_already_fired()
    test_vob_numbering()
    test_epoch_flow_reset_then_new_5m()
    test_vob_state_persist_roundtrip()
    print('\nУсі тести VOB (1H-OB такт + catch-all + блок=сигнал + чистий лог) пройдено ✅')
