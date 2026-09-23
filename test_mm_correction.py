"""🔻 ДЕТЕКТОР КОРЕКЦІЇ банера «🧮 МММ-монітор» + ворота відкриття (19.09).

**Вимоги користувача, дослівно:**
  1. «Потрібно щоб бот моніторив графіки і відслідковував саме корекцію по
     монетах… (наприклад на молодших таймфреймах VOB, OB або FVG почали
     зʼявлятись на графіках в протилежному напрямку від банера "МММ-монітор").
     Бот має професійно аналізувати графіки і візуалізувати загальний вердикт
     на банері "МММ-монітор" стосовно чи почалась або закінчилась корекція.»
  2. «Зроби професійний таймер "2д 09:57:19" для банера, щоб все виглядало в
     одному стилі.»
  3. «В період корекції потрібно обмежити відкриття угод.»

Що стережуть ці тести:
  • корекція = рух ПРОТИ банера, видний по БАГАТЬОХ монетах (ширина ринку), а
    не «одна монета впала»;
  • шар без вибірки — НЕ ознака (мала вибірка ≠ «корекції немає» і ≠ «є»);
  • початок і кінець підтверджуються ЧАСОМ (симетрично), а пороги входу й
    виходу різні (гістерезис) — вердикт не миготить;
  • ⚖ банер без напрямку → корекції не буває за визначенням;
  • ворота відкриття стоять у ДВОХ вузлах (`_open` і `on_signal`), ✋ ручне їх
    обходить, а лог не флудить;
  • детектор не робить ЖОДНОГО запиту до біржі (усі числа вже пораховані);
  • таймер банера — в ОДНОМУ стилі (доба такою самою плиткою, як цифри часу).
"""
import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import threading
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

# Порожній пакет `detection`: справжній `__init__.py` тягне півпроєкту (pybit).
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, 'detection', fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mc = _load('detection.mm_correction', 'mm_correction.py')
_pkg.mm_correction = mc

_spec = importlib.util.spec_from_file_location(
    'fuel_filter_corr_test', os.path.join(_HERE, 'detection', 'fuel_filter.py'))
_ffm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ffm)
FF = _ffm.FuelFilterDaemon

_FF_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'),
               encoding='utf-8').read()
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'),
               encoding='utf-8').read()
_MC_SRC = open(os.path.join(_HERE, 'detection', 'mm_correction.py'),
               encoding='utf-8').read()
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
             encoding='utf-8').read()

NOW = 10_000.0


def _mk_ff(stored):
    """Двигун із ЗАПИСУВАНИМ сховищем налаштувань — його потребують і
    валідація, і міграція дефолтів."""
    ff = FF.__new__(FF)

    class _Stored:
        def __init__(self, data):
            self.data = dict(data)

        def get_setting(self, key, default=None):
            return self.data if key == 'fuel_filter_settings' else default

        def set_setting(self, key, val):
            if key == 'fuel_filter_settings':
                self.data = dict(val)

    ff._db = _Stored(stored)
    return ff


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


def _fn_src(src, name):
    """Тіло функції/методу БЕЗ докстрінга — щоб замки не спрацьовували на
    власних поясненнях (задокументована пастка `ensure_fresh`)."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                                     and isinstance(node.body[0].value, ast.Constant)
                                     ) else node.body
            return '\n'.join(ast.unparse(b) for b in body)
    raise AssertionError(f'функція {name} зникла')


_LOGGED = []


def _install_log():
    _LOGGED.clear()
    mod = types.ModuleType('detection.activity_log')

    def _log(symbol, event, detail='', side=None, source='', extra=None):
        _LOGGED.append({'symbol': symbol, 'event': event, 'detail': detail,
                        'side': side, 'source': source})
    mod.log_activity = _log
    sys.modules['detection.activity_log'] = mod
    _pkg.activity_log = mod


# 🧾 СИРИЙ ЛОГ КОРЕКЦІЇ: підміняємо шар БД ще НА ІМПОРТІ модуля — інакше
# `_mm_corr_log_write` потягнув би справжній `storage.db_operations` (а той —
# конфіг і живе зʼєднання) просто заради тесту детектора.
_DBROWS = []


def _install_db(fail=False):
    _DBROWS.clear()
    st = sys.modules.get('storage') or types.ModuleType('storage')
    st.__path__ = [os.path.join(_HERE, 'storage')]
    sys.modules['storage'] = st
    mod = types.ModuleType('storage.db_operations')

    class _DB:
        def log_mm_correction(self, **row):
            if fail:
                raise RuntimeError('БД лягла')
            _DBROWS.append(dict(row))
    mod.get_db = lambda: _DB()
    sys.modules['storage.db_operations'] = mod
    st.db_operations = mod


_install_db()


def _rows(kind=None):
    return [r for r in _DBROWS if kind is None or r.get('kind') == kind]


def _snap(**coins):
    """{SYM: (статус, сила, напрямок ціни)} → знімок монітора."""
    out = {}
    for sym, (st, stren, pdir) in coins.items():
        out[sym] = {'status': st, 'strength': stren,
                    'dir': (stren / 100.0) * (1 if st == 'LONG' else -1),
                    'price_dir': pdir}
    return out


def _mk(trends=None, vob_on=True, **settings):
    """Мінімальний двигун: лише те, чого торкаються важіль і детектор корекції.

    ⚠️ Тренди VOB підміняємо на рівні `_mm_vob_trends` — тобто НЕ лізучи в
    сканер: тест доводить поведінку детектора, а не роботу сканера."""
    ff = FF.__new__(FF)
    ff._lock = threading.RLock()
    ff._mm_bias, ff._mm_bias_since, ff._mm_bias_cand = {}, 0.0, {}
    ff._mm_corr_st, ff._mm_corr, ff._mm_lever_hist = {}, {}, []
    ff._mm_corr_skip_logged = {}
    # 🧾 Таймер сирого логу (20.09) і знімок монітора (звідки лог бере ціну в
    # момент блокування). Нове поле стану ЗАВЖДИ додавати сюди.
    ff._mm_corr_log_at = 0.0
    ff._mm_corr_override = 0.0   # ⏸ ручна пауза (нове поле стану — див. пастку `_mk`)
    ff._mm_snapshot = {}
    ff._engine_skip = {}
    s = {'mm_bias_confirm_sec': 0, 'mm_corr_confirm_sec': 0,
         'mm_corr_min_layers': 2, 'mm_corr_vob_pct': 60.0,
         'mm_corr_price_pct': 60.0, 'mm_corr_lever_drop': 15.0,
         'mm_corr_enabled': True, 'mm_corr_block_open': True,
         'mm_corr_log_enabled': True, 'mm_corr_log_every_sec': 300,
         'mm_monitor_enabled': True, 'enabled': True}
    s.update(settings)
    ff._settings = s
    ff.get_settings = lambda: dict(ff._settings)
    ff._mm_vob_trends = lambda: {'on': vob_on, 'tf': '5m',
                                 'trends': dict(trends or {})}
    return ff


def _tick(ff, snap, now=NOW):
    """Один такт: важіль банера → вердикт про корекцію (як у `_mm_capture`)."""
    s = ff.get_settings()
    ff._mm_track_bias(snap, now, s)
    ff._mm_track_correction(snap, now, s)
    return dict(ff._mm_corr)


# ═══════════ 1. ЧИСТІ ШАРИ: «ПРОТИ БАНЕРА» ═══════════════════════════════
def test_vob_layer_counts_the_share_of_coins_looking_the_other_way():
    """📦 ГОЛОВНА ОЗНАКА: на молодшому TF блоки пішли ПРОТИ банера. Це частка
    монет, а не «десь один ведмежий блок»."""
    trends = {f'C{i}USDT': 'SHORT' for i in range(7)}
    trends.update({f'L{i}USDT': 'LONG' for i in range(3)})
    syms = list(trends)
    lay = mc.vob_layer(trends, syms, 'LONG', 60.0)
    _check(lay['ok'], 'вибірки вистачає — шар мусить бути визначеним')
    _check(lay['pct'] == 70.0, f'7 із 10 проти = 70%: {lay}')
    _check(lay['lit'], 'поріг 60% — 70% мусить його перекрити')
    # Дзеркало: для банера SHORT «проти» — це вже LONG-блоки.
    _check(mc.vob_layer(trends, syms, 'SHORT', 60.0)['pct'] == 30.0,
           'бік «проти» мусить рахуватись від банера, а не бути зашитим')
    print('✓ 📦 VOB: частка монет, чий блок дивиться ПРОТИ банера')


def test_a_tiny_sample_is_not_a_signal():
    """«Немає даних» ≠ «немає корекції» і ≠ «корекція є». Три монети з
    двохсот — не ринок, і оголошувати по них вердикт не можна."""
    lay = mc.vob_layer({'AUSDT': 'SHORT', 'BUSDT': 'SHORT'}, ['AUSDT', 'BUSDT'],
                       'LONG', 60.0)
    _check(not lay['ok'], f'мала вибірка мусить лишатись НЕВИЗНАЧЕНОЮ: {lay}')
    _check(not lay['lit'], 'невизначений шар не може бути «засвіченим»')
    _check(mc.MIN_SAMPLE >= 5, 'мінімальна вибірка не має бути символічною')
    print('✓ 🔎 мала вибірка — шар НЕ визначений, а не «корекція»')


def test_price_layer_ignores_flat_coins():
    """💹 Монета без свіжого руху (▬) нікуди не тягне — у частку вона не
    входить ні «за», ні «проти». Інакше рівний ринок читався б як корекція."""
    snap = _snap(A=('LONG', 50, 'down'), B=('LONG', 40, 'down'),
                 C=('LONG', 30, 'down'), D=('SHORT', 20, 'down'),
                 E=('LONG', 10, 'up'), F=('LONG', 10, 'flat'),
                 G=('LONG', 10, None))
    lay = mc.price_layer(snap, 'LONG', 60.0)
    _check(lay['n'] == 5, f'у частку йдуть лише монети з рухом: {lay}')
    _check(lay['pct'] == 80.0, f'4 із 5 проти = 80%: {lay}')
    _check(lay['lit'], 'поріг 60% перекрито')
    print('✓ 💹 Ціна: рахуються лише монети зі СВІЖИМ рухом')


def test_lever_drop_is_measured_towards_the_banner_side():
    """📉 Важіль міряємо У БІК банера: якщо він устиг перевернутись,
    просідання чесно виходить більшим за сам пік."""
    _check(mc.lever_layer(40.0, 55.0, 15.0)['lit'], '55 → 40 це −15 п.п.')
    _check(not mc.lever_layer(50.0, 55.0, 15.0)['lit'], '−5 п.п. — це не просідання')
    flip = mc.lever_layer(-20.0, 55.0, 15.0)
    _check(flip['pct'] == 75.0, f'важіль перевернувся → 75 п.п.: {flip}')
    _check(not mc.lever_layer(None, 55.0, 15.0)['ok'],
           'без історії важеля шар мусить бути НЕВИЗНАЧЕНИМ')
    print('✓ 📉 Важіль: просідання від піку, з урахуванням перевороту')


def test_evaluate_needs_the_configured_number_of_layers():
    snap = _snap(**{f'C{i}': ('LONG', 50, 'down') for i in range(6)})
    trends = {f'C{i}': 'SHORT' for i in range(6)}
    cfg = {'mm_corr_min_layers': 2, 'mm_corr_vob_pct': 60.0,
           'mm_corr_price_pct': 60.0, 'mm_corr_lever_drop': 15.0}
    r = mc.evaluate(snap, trends, 'LONG', 50.0, 52.0, cfg)
    _check(r['lit'] == 2, f'📦 і 💹 проти, важіль майже не просів: {r["lit"]}')
    _check(r['need'] == 2, 'потрібна кількість — з налаштувань')
    r2 = mc.evaluate(snap, trends, 'LONG', 50.0, 52.0,
                     dict(cfg, mm_corr_min_layers=3))
    _check(r2['lit'] == 2 and r2['need'] == 3,
           'при «3 з 3» тих самих ознак уже НЕ досить')
    print('✓ 🔻 вердикт збирається з ознак за налаштованим порогом')


# ═══════════ 2. МАШИНА СТАНІВ: ПОЧАЛАСЬ / ЗАВЕРШИЛАСЬ ════════════════════
def test_correction_start_is_confirmed_by_time():
    st = mc.next_state({}, 2, 2, 2, 1000.0, 120)
    _check(st['state'] == 'pending', f'спершу — відлік підтвердження: {st}')
    st = mc.next_state(st, 2, 2, 2, 1060.0, 120)
    _check(st['state'] == 'pending', 'за 60с зі 120 корекція ще не оголошена')
    st = mc.next_state(st, 2, 2, 2, 1130.0, 120)
    _check(st['state'] == 'on' and st['since'] == 1130.0,
           f'після вікна — корекція, і таймер від цього моменту: {st}')
    print('✓ ⏳ початок корекції підтверджується часом')


def test_the_end_is_confirmed_too_and_then_reported():
    """«Корекція закінчилась» — така сама ПОДІЯ, як і «почалась». Оголосити
    кінець на одному спокійному такті означало б повернути бота в ринок
    рівно на відкаті всередині корекції."""
    st = {'state': 'on', 'since': 1000.0, 'cand_since': 0.0,
          'ended_at': 0.0, 'lasted': 0.0}
    st = mc.next_state(st, 0, 0, 2, 2000.0, 120)
    _check(st['state'] == 'ending', f'спершу відлік кінця: {st}')
    st = mc.next_state(st, 0, 0, 2, 2060.0, 120)
    _check(st['state'] == 'ending', 'на півдорозі кінець НЕ оголошуємо')
    st = mc.next_state(st, 0, 0, 2, 2130.0, 120)
    _check(st['state'] == 'ended', f'після вікна — завершилась: {st}')
    _check(st['lasted'] >= 1130.0, f'тривалість мусить бути збережена: {st}')
    # Далі стан сам собою повертається в ТРЕНД — «завершилась» це подія,
    # а не вічний підпис.
    st2 = mc.next_state(st, 0, 0, 2, 2130.0 + mc.ENDED_SHOW_SEC + 1, 120)
    _check(st2['state'] == 'trend', f'через вікно показу — знову тренд: {st2}')
    print('✓ ✅ кінець корекції теж підтверджується і показується подією')


def test_a_blip_back_does_not_end_the_correction():
    """ГІСТЕРЕЗИС: тримаємо корекцію за ПОСЛАБЛЕНИМИ порогами. Інакше вердикт
    смикався б рівно там, де ринок зависає найчастіше."""
    st = {'state': 'on', 'since': 1000.0, 'cand_since': 0.0,
          'ended_at': 0.0, 'lasted': 0.0}
    st = mc.next_state(st, 1, 2, 2, 2000.0, 120)   # суворо 1, послаблено 2
    _check(st['state'] == 'on', f'за послабленими порогами корекція триває: {st}')
    _check(mc.EXIT_MARGIN_PCT > 0 and mc.EXIT_MARGIN_PP > 0,
           'гістерезис мусить бути ненульовим')
    print('✓ 🔁 гістерезис: короткий відкат корекцію не закриває')


def test_zero_confirm_switches_instantly():
    st = mc.next_state({}, 2, 2, 2, 500.0, 0)
    _check(st['state'] == 'on', f'0 с = миттєво: {st}')
    st = mc.next_state(st, 0, 0, 2, 900.0, 0)
    _check(st['state'] == 'ended', f'і кінець теж миттєво: {st}')
    print('✓ ⏱ «0 с» повертає миттєве перемикання')


# ═══════════ 3. ДВИГУН: ВЕРДИКТ НА БАНЕРІ ════════════════════════════════
def test_the_engine_declares_a_correction_against_the_banner():
    """Кейс користувача: банер LONG, а на графіках усі монети пішли в SHORT."""
    _install_log()
    coins = {f'C{i}USDT': ('LONG', 60, 'down') for i in range(8)}
    snap = _snap(**coins)
    trends = {s: 'SHORT' for s in coins}
    ff = _mk(trends=trends)
    c = _tick(ff, snap)
    _check(ff.mm_bias()['dir'] == 'LONG', 'банер мусить лишатись LONG')
    _check(c['state'] == 'on', f'корекція мусить бути оголошена: {c}')
    _check(c['lit'] >= c['need'], f'ознак замало: {c}')
    _check(c['blocking'], 'при увімкненому тумблері відкриття мусять блокуватись')
    ev = [x for x in _LOGGED if x['event'] == 'event']
    _check(ev and 'КОРЕКЦІЯ' in ev[0]['detail'],
           f'подія мусить піти в 🧾 Лог: {_LOGGED}')
    print('✓ 🔻 двигун бачить корекцію проти банера і пише подію')


def test_a_market_that_follows_the_banner_is_not_a_correction():
    coins = {f'C{i}USDT': ('LONG', 60, 'up') for i in range(8)}
    ff = _mk(trends={s: 'LONG' for s in coins})
    c = _tick(ff, _snap(**coins))
    _check(c['state'] == 'trend', f'ринок іде за банером — корекції немає: {c}')
    _check(not c['blocking'], 'без корекції відкриття не блокуються')
    print('✓ ▶️ ринок за банером — вердикт «тренд»')


def test_without_a_banner_direction_there_is_no_correction():
    """⚖ Корекція означає «рух ПРОТИ тренду». Немає напрямку — немає тренду,
    і коригувати нема чого; причина при цьому НАЗВАНА."""
    coins = {'AUSDT': ('LONG', 30, 'down'), 'BUSDT': ('SHORT', 30, 'down'),
             'CUSDT': ('LONG', 20, 'down'), 'DUSDT': ('SHORT', 20, 'down')}
    ff = _mk(trends={s: 'SHORT' for s in coins})
    c = _tick(ff, _snap(**coins))
    _check(ff.mm_bias().get('dir') is None, 'банер мусить бути без напрямку')
    _check(c['state'] == 'trend', f'без напрямку корекції бути не може: {c}')
    _check('напрям' in (c.get('reason') or ''), f'причина не названа: {c}')
    print('✓ ⚖ банер без напрямку — корекція не рахується, причина названа')


def test_the_verdict_survives_a_restart():
    """`botupdate` роблять часто, а «корекція триває 1г 20хв» — це стан РИНКУ.
    Разом із таймером мусить пережити рестарт і блокування відкриттів."""
    ff = _mk()
    ff._mm_corr_st = {'state': 'on', 'since': 500.0, 'cand_since': 0.0,
                      'ended_at': 0.0, 'lasted': 0.0}
    blob = {}
    with ff._lock:
        blob['mm_corr_st'] = dict(ff._mm_corr_st)
    _check('mm_corr_st' in _fn_src(_FF_SRC, '_persist_state'),
           'стан корекції мусить персиститись у тому самому блобі')
    _check('mm_corr_st' in _fn_src(_FF_SRC, '_load_state'),
           'стан корекції мусить відновлюватись')
    coins = {f'C{i}USDT': ('LONG', 60, 'down') for i in range(8)}
    ff2 = _mk(trends={s: 'SHORT' for s in coins})
    ff2._mm_corr_st = dict(blob['mm_corr_st'])
    c = _tick(ff2, _snap(**coins), now=5000.0)
    _check(c['state'] == 'on' and c['since'] == 500.0,
           f'після рестарту таймер корекції мусить продовжитись: {c}')
    print('✓ 💾 вердикт і таймер корекції переживають рестарт')


def test_a_banner_flip_resets_the_verdict():
    """Фліп банера — це ЗМІНА ТРЕНДУ, а не корекція. Лишити «корекцію проти
    LONG» під банером SHORT означало б блокувати відкриття за вердиктом про
    ринок, якого вже немає."""
    up = {f'C{i}USDT': ('LONG', 70, 'down') for i in range(8)}
    ff = _mk(trends={s: 'SHORT' for s in up})
    c = _tick(ff, _snap(**up))
    _check(c['state'] == 'on', f'спершу мусить бути корекція: {c["state"]}')
    # Ринок перевернувся: тепер і сам МММ по монетах SHORT — банер фліпає.
    dn = {f'C{i}USDT': ('SHORT', 70, 'down') for i in range(8)}
    c2 = _tick(ff, _snap(**dn), now=NOW + 30)
    _check(ff.mm_bias()['dir'] == 'SHORT', 'банер мусив перевернутись')
    _check(c2['state'] != 'on', f'після фліпу вердикт мусить обнулитись: {c2}')
    _check(not c2['blocking'], 'і блокування відкриттів теж')
    print('✓ 🔁 фліп банера обнуляє вердикт про корекцію')


def test_the_detector_never_touches_the_exchange():
    """Ознаки беруться з УЖЕ порахованих даних: знімок монітора + кеш сканера.
    Жодного походу на біржу — інакше «аналіз графіків» коштував би сотні HTTP."""
    src = _fn_src(_FF_SRC, '_mm_track_correction')
    for bad in ('fetch_klines', 'scan_liquidity', 'get_latest_ob_trend',
                'detect_volumized_obs', 'requests'):
        _check(bad not in src, f'детектор не має ходити на біржу: {bad}')
    _check('_mm_vob_trends' in src, 'тренди VOB беремо з кешу сканера')
    _check('volumized_trends' in _fn_src(_FF_SRC, '_mm_vob_trends'),
           'читати треба ПУБЛІЧНИМ методом сканера')
    _check('def volumized_trends' in _SC_SRC,
           'сканер мусить мати публічний `volumized_trends`')
    _check('_volumized_trend_cache' not in src,
           'лізти у приватний кеш сканера не можна')
    print('✓ 🛰 детектор не робить жодного запиту до біржі')


def test_the_verdict_is_read_not_recomputed_by_the_state_reader():
    """Той самий урок B2, що з шарами Черги-4: рахує ДВИГУН, `mm_monitor_state`
    лише читає — інакше банер і ворота судили б різними числами."""
    src = _fn_src(_FF_SRC, 'mm_monitor_state')
    _check('_mm_corr' in src, 'стан мусить віддавати поле correction')
    for bad in ('evaluate(', 'next_state('):
        _check(bad not in src, f'читач не має рахувати вердикт: {bad}')
    print('✓ 🧭 вердикт рахує двигун — читач лише віддає готове')


# ═══════════ 4. ВОРОТА ВІДКРИТТЯ ═════════════════════════════════════════
def test_only_a_confirmed_correction_blocks_opening():
    ff = _mk()
    ff._mm_corr = {'state': 'pending', 'blocking': False, 'bias': 'LONG',
                   'since': 0, 'lit': 2, 'need': 2, 'layers': []}
    _check(ff.correction_blocks_open()[0] is False,
           'відлік підтвердження — ще НЕ подія, блокувати на ньому не можна')
    ff._mm_corr = {'state': 'on', 'blocking': True, 'bias': 'LONG',
                   'since': NOW, 'lit': 2, 'need': 2, 'layers': []}
    ok, why = ff.correction_blocks_open()
    _check(ok and 'КОРЕКЦІЯ' in why, f'підтверджена корекція мусить блокувати: {why}')
    ff._mm_corr = {'state': 'on', 'blocking': False, 'bias': 'LONG',
                   'since': NOW, 'lit': 2, 'need': 2, 'layers': []}
    _check(ff.correction_blocks_open()[0] is False,
           'вимкнений тумблер блокування мусить лишати торгівлю')
    print('✓ 🚫 блокує лише ПІДТВЕРДЖЕНА корекція і лише за тумблером')


def test_the_gate_stands_in_both_opening_nodes():
    """Автоматичні відкриття йдуть ДВОМА шляхами: черги (`_open`) і прямий
    опен (`on_signal`, коли черги вимкнені). Один вузол накрив би лише
    половину — рівно так колись «проспали» 🚦 головні кнопки."""
    o = _fn_src(_FF_SRC, '_open')
    _check('correction_blocks_open' in o, 'у `_open` немає воріт корекції')
    _check(o.index('correction_blocks_open') < o.index("fuel.get('mark_price')"),
           'ворота мусять стояти ДО будь-якої роботи з відкриття')
    _check('by_hand' in o.split('correction_blocks_open')[0].rsplit('if', 1)[-1]
           or 'if not by_hand' in o,
           '✋ ручне відкриття мусить обходити ворота корекції')
    t = _fn_src(_TM_SRC, 'on_signal')
    _check('correction_blocks_open' in t, 'у `on_signal` немає воріт корекції')
    _check(t.index('correction_blocks_open') < t.index('open_plan'),
           'корекція мусить перевірятись ДО розрахунку рівнів (він важкий)')
    _check('manual' in t.split('correction_blocks_open')[0][-400:],
           '✋ ручний сигнал мусить обходити ворота корекції')
    print('✓ 🚪 ворота стоять в обох вузлах відкриття, ✋ ручне їх обходить')


def test_the_blocked_open_is_logged_once_per_episode():
    """АНТИ-ФЛУД. Двигун смикає `_open` щотакту по кожній монеті черги, а
    корекція триває годинами — без ключа «епізод» це був би рівно той потоп,
    який уже чистили в Q4-recheck."""
    _install_log()
    ff = _mk()
    ff._mm_corr = {'state': 'on', 'blocking': True, 'bias': 'LONG',
                   'since': NOW, 'lit': 2, 'need': 2, 'layers': []}
    ff._open = FF._open.__get__(ff)
    for _ in range(5):
        _check(ff._open('AAAUSDT', 'LONG', {}, ff.get_settings()) is False,
               'під час корекції відкриття мусить бути відхилене')
    lines = [x for x in _LOGGED if x['event'] == 'skipped']
    _check(len(lines) == 1, f'мусить бути РІВНО один рядок на епізод: {len(lines)}')
    # Нова корекція — новий епізод, і про неї треба сказати знову.
    ff._mm_corr['since'] = NOW + 9999
    ff._open('AAAUSDT', 'LONG', {}, ff.get_settings())
    _check(len([x for x in _LOGGED if x['event'] == 'skipped']) == 2,
           'новий епізод корекції мусить дати новий рядок')
    # ✋ Ручне відкриття ворота обходить (далі його зупинить уже ціна/розмір).
    _LOGGED.clear()
    ff._open('AAAUSDT', 'LONG', {}, ff.get_settings(), by_hand=True)
    _check(not [x for x in _LOGGED if 'КОРЕКЦІЯ' in x['detail']],
           '✋ ручне відкриття не має впиратись у ворота корекції')
    print('✓ 🔇 один рядок на епізод · ✋ ручне проходить')


def test_a_broken_detector_never_stops_trading():
    """FAIL-OPEN: ці ворота вміють зупинити торгівлю повністю, тож збій
    читання не має ставати тихою зупинкою бота (той самий принцип, що в
    🚦 воріт напрямку)."""
    ff = _mk()

    def _boom():
        raise RuntimeError('нема стану')
    ff.mm_correction = _boom
    _check(ff.correction_blocks_open() == (False, ''),
           'на помилці ворота мусять ПРОПУСКАТИ')
    print('✓ 🛟 збій детектора не зупиняє торгівлю')


def test_the_switches_have_the_documented_defaults():
    _check(mc.DEFAULTS['mm_corr_enabled'] is True, 'детектор дефолтом УВІМК')
    _check(mc.DEFAULTS['mm_corr_block_open'] is True,
           'дослівна вимога: «в період корекції обмежити відкриття угод»')
    _check(mc.DEFAULTS['mm_corr_min_layers'] == 2, 'дефолт — 2 ознаки з 3')
    # ⚠️ ПЕРЕПИСАНО (21.09), а не «полагоджено»: раніше тут стояло 120 із
    # поясненням «те саме вікно, що в антиспамі банера». Лог показав, що цього
    # замало — 9 із 33 стартів були миготінням усередині епізоду, тож вікно
    # корекції СВІДОМО довше за банерне (`mm_bias_confirm_sec` = 120). Це різні
    # події: банер гасить підпис, а тут ми зупиняємо ТОРГІВЛЮ.
    _check(mc.DEFAULTS['mm_corr_confirm_sec'] == 300,
           'підтвердження корекції — власне вікно, довше за банерне')
    # Ключі мусять доїхати в налаштування бота (інакше UI нікуди не збереже).
    for k in mc.DEFAULTS:
        _check(k in _ffm.DEFAULT_SETTINGS, f'ключа {k} немає в налаштуваннях FF')
    print('✓ ⚙️ дефолти на місці й доїжджають у налаштування бота')


def test_thresholds_are_clamped_not_crashed():
    """Сміття з БД/UI не має валити бота — і не має мовчки ставати «корекція
    завжди» (0 шарів) чи «ніколи» (99 шарів)."""
    ff = FF.__new__(FF)
    ff._db = types.SimpleNamespace(get_setting=lambda *a, **k: {
        'mm_corr_min_layers': 99, 'mm_corr_vob_pct': 'абв',
        'mm_corr_confirm_sec': -5})
    s = FF.get_settings(ff)
    _check(isinstance(s, dict), 'валідацію налаштувань не знайдено')
    _check(s['mm_corr_min_layers'] == 3, f'шарів лише три: {s["mm_corr_min_layers"]}')
    _check(s['mm_corr_vob_pct'] == 60.0, 'сміття мусить падати на дефолт')
    _check(s['mm_corr_confirm_sec'] == 0, 'відʼємне вікно = миттєво')
    print('✓ 🧮 пороги обрізаються, а не валять бота')


# ═══════════ 5. UI: ВЕРДИКТ НА БАНЕРІ + ТАЙМЕР В ОДНОМУ СТИЛІ ════════════
def test_ui_has_the_verdict_row_and_its_settings():
    for el in ('mm-corr-row', 'mm-corr-state', 'mm-corr-timer',
               'mm-corr-layers', 'mm-corr-block', 'mm-corr-sum'):
        _check(f'id="{el}"' in _HTML, f'немає елемента {el}')
    for el in ('ff-mm-corr-enabled', 'ff-mm-corr-block', 'ff-mm-corr-layers',
               'ff-mm-corr-vob', 'ff-mm-corr-price', 'ff-mm-corr-drop',
               'ff-mm-corr-confirm'):
        _check(f'id="{el}"' in _HTML, f'немає контрола {el}')
        _check(f"'{el}'" in _HTML, f'контрол {el} нікуди не зберігається')
    for key in ('mm_corr_enabled', 'mm_corr_block_open', 'mm_corr_min_layers',
                'mm_corr_vob_pct', 'mm_corr_price_pct', 'mm_corr_lever_drop',
                'mm_corr_confirm_sec'):
        _check(key in _HTML, f'ключ {key} не їде на сервер')
    # Вердикт мусить ДОЇЖДЖАТИ на сторінку (та сама пастка, що з `coverage`).
    _check('mm.correction' in _HTML, 'поле `correction` не читається зі стану')
    print('✓ 🖥 UI: рядок вердикту, налаштування і зв\'язок зі станом')


def test_the_timer_is_one_style_for_days_and_hours():
    """Вимога 2: «професійний таймер "2д 09:57:19" … в одному стилі». Раніше
    доба була окремим ЖОВТИМ текстом без плитки (інший розмір і колір)."""
    _check('.fday' not in _HTML, 'старий «висячий» сегмент днів мусив зникнути')
    _check('fd fdd' in _HTML, 'доба мусить бути ТАКОЮ САМОЮ плиткою, що й цифри')
    css = _HTML[_HTML.index('.ff-flip {'):_HTML.index('.tm-pnl-pos')]
    _check('.ff-flip .fd.fdd' in css, 'немає стилю плитки днів')
    _check('background' not in css.split('.fd.fdd')[1].split('}')[0],
           'плитка днів не має перевизначати тло — інакше це знову інший стиль')
    print('✓ ⏱ таймер: доба — така сама плитка, що й цифри часу')


def _run_js(body):
    """Ганяємо САМ код сторінки (зріз рендера вердикту) під node."""
    i = _HTML.index('let _mmCorr = null;')
    j = _HTML.index('function mmApplyState(')
    src = _HTML[i:j]
    for fn in ('function flipTimerHTML(', 'function fmtTimer('):
        a = _HTML.index(fn)
        b = _HTML.index('\n}', a) + 2
        src = _HTML[a:b] + '\n' + src
    pre = '''
const _els = {};
function _el(id) {
  // ⚠️ `dataset` теж мусить бути: справжній елемент його має ЗАВЖДИ, і саме
  // через нього кнопка ⏸ паузи захищається від перемальовування під час
  // запиту. Без нього тест падав би «на рівному місці» (та сама пастка
  // фейк-DOM, що вже ловили на `querySelectorAll`).
  if (!_els[id]) _els[id] = {id, style:{}, dataset:{}, innerHTML:'', textContent:'', title:''};
  return _els[id];
}
const document = { getElementById: _el };
'''
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False,
                                     encoding='utf-8') as f:
        f.write(pre + src + '\n' + body)
        p = f.name
    try:
        r = subprocess.run(['node', p], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise AssertionError(f'node: {r.stderr[:600]}')
        return r.stdout.strip()
    finally:
        os.unlink(p)


def test_js_draws_all_four_verdicts():
    out = _run_js(r'''
const now = Math.floor(Date.now()/1000);
const L = [{key:'vob',icon:'📦',name:'VOB проти',pct:72,need:60,ok:true,lit:true,n:20,note:''},
           {key:'price',icon:'💹',name:'Ціна проти',pct:80,need:60,ok:true,lit:true,n:18,note:''},
           {key:'lever',icon:'📉',name:'Важіль просів',pct:4,need:15,ok:true,lit:false,n:1,note:''}];
const g = id => document.getElementById(id);
const seen = {};
mmRenderCorr({state:'on', since: now-4000, lit:2, need:2, layers:L, blocking:true,
              enabled:true, confirm_sec:120, bias:'LONG', vob_tf:'5m'});
seen.on = {st:g('mm-corr-state').textContent, t:g('mm-corr-timer').innerHTML,
           lay:g('mm-corr-layers').innerHTML, blk:g('mm-corr-block').textContent,
           show:g('mm-corr-row').style.display};
mmRenderCorr({state:'pending', cand_since: now-45, confirm_sec:120, lit:2, need:2,
              layers:L, enabled:true, blocking:false});
seen.pending = {st:g('mm-corr-state').textContent, t:g('mm-corr-timer').innerHTML};
mmRenderCorr({state:'ended', ended_at: now-300, lasted:3600, lit:0, need:2,
              layers:L, enabled:true, blocking:false});
seen.ended = {st:g('mm-corr-state').textContent, t:g('mm-corr-timer').innerHTML,
              blk:g('mm-corr-block').style.display};
mmRenderCorr(null);
seen.off = {show:g('mm-corr-row').style.display};
console.log(JSON.stringify(seen));
''')
    import json
    d = json.loads(out)
    _check('КОРЕКЦІЯ' in d['on']['st'], f'стан «корекція» не показано: {d["on"]}')
    _check(d['on']['show'] == 'flex', 'рядок вердикту мусить бути видимим')
    _check('01:06:40' in re.sub(r'<[^>]*>', '', d['on']['t']),
           f'таймер корекції не той: {d["on"]["t"]}')
    _check('class="fd"' in d['on']['t'], 'таймер мусить бути в тих самих плитках')
    _check('72%' in d['on']['lay'] and '80%' in d['on']['lay'],
           f'розклад ознак не показано: {d["on"]["lay"]}')
    _check('зупинено' in d['on']['blk'], f'блокування не показано: {d["on"]}')
    _check('45/120' in d['pending']['t'], f'відлік не показано: {d["pending"]}')
    _check('ЗАВЕРШИЛАСЬ' in d['ended']['st'], f'кінець не показано: {d["ended"]}')
    _check(d['ended']['blk'] == 'none', 'після кінця блокування не показуємо')
    _check(d['off']['show'] == 'none', 'без даних рядок мусить зникати')
    print('✓ 🎨 JS: усі чотири стани вердикту малюються на банері')


def test_js_countdown_is_whole_seconds_not_a_raw_float():
    """🐞 Зі скріна: «82.72762799263/120с». `cand_since` — серверний
    `time.time()` (дробовий), тож віднімання давало сирий float прямо в UI."""
    out = _run_js(r'''
const now = Math.floor(Date.now()/1000);
mmRenderCorr({state:'pending', cand_since: now - 82.72762799263, confirm_sec:120,
              lit:2, need:2, layers:[], enabled:true, blocking:false});
console.log(document.getElementById('mm-corr-timer').innerHTML);
''').strip()
    out = re.sub(r'<[^>]*>', '', out).strip()
    _check(re.fullmatch(r'\d+/\d+с', out), f'відлік мусить бути цілим: {out}')
    _check('.' not in out, f'у відліку лишився дробовий залишок: {out}')
    print('✓ ⏱ відлік підтвердження — цілі секунди, без «хвоста»')


def test_js_shows_the_unit_each_layer_really_uses():
    """📉 Просідання важеля міряється в П.П., а не у %. Підписати його
    відсотком означало б назвати число не тим, що воно є (і саме так було
    видно на скріні: «Важіль просів 24.7%/15%»)."""
    _check(mc.lever_layer(10.0, 34.7, 15.0)['unit'] == 'п.п.',
           'шар важеля мусить сам казати свою одиницю')
    _check(mc.price_layer({}, 'LONG', 60.0)['unit'] == '%',
           'часткові ознаки лишаються у відсотках')
    out = _run_js(r'''
const L = [{key:'price',icon:'💹',name:'Ціна проти',pct:91.9,need:60,ok:true,lit:true,n:18,unit:'%'},
           {key:'lever',icon:'📉',name:'Важіль просів',pct:24.7,need:15,ok:true,lit:true,n:1,unit:'п.п.'}];
mmRenderCorr({state:'on', since: Math.floor(Date.now()/1000)-60, lit:2, need:2,
              layers:L, enabled:true, blocking:true});
console.log(document.getElementById('mm-corr-layers').innerHTML);
''')
    out = re.sub(r'<[^>]*>', '', out)
    _check('24.7п.п./15п.п.' in out, f'важіль підписано не в п.п.: {out}')
    _check('91.9%/60%' in out, f'частка мусить лишитись у %: {out}')
    print('✓ 📉 одиницю дає бекенд: частки — %, важіль — п.п.')


def test_js_timer_puts_the_day_in_the_same_tile():
    out = _run_js(r'''console.log(flipTimerHTML(2*86400 + 9*3600 + 57*60 + 19));''')
    _check('fd fdd' in out, f'доба не в плитці: {out}')
    _check(re.sub(r'<[^>]*>', '', out) == '2д09:57:19',
           f'таймер читається не так: {re.sub(r"<[^>]*>", "", out)}')
    _check('fday' not in out, 'старий стиль дня мусив зникнути')
    print('✓ ⏱ JS: «2д 09:57:19» — доба тією самою плиткою')


# ═══════════ 6. 🧾 ЛОГУВАННЯ КОРЕКЦІЇ (вимога 20.09) ═════════════════════
# «Зроби логування по "Корекції" для подальшого аналізу і коригування
# налаштувань.» Тобто потрібен не ще один рядок для очей, а СИРИЙ РЯД значень
# трьох ознак у часі + пороги, що діяли, + що бот тоді зробив.
_MODELS_SRC = open(os.path.join(_HERE, 'storage', 'db_models.py'),
                   encoding='utf-8').read()
_DBOPS_SRC = open(os.path.join(_HERE, 'storage', 'db_operations.py'),
                  encoding='utf-8').read()
_FLASK_SRC = open(os.path.join(_HERE, 'web', 'flask_app.py'),
                  encoding='utf-8').read()


def _model_columns(name):
    """Імена колонок моделі з AST — без імпорту SQLAlchemy і конфігу."""
    for node in ast.walk(ast.parse(_MODELS_SRC)):
        if isinstance(node, ast.ClassDef) and node.name == name:
            out = []
            for b in node.body:
                if (isinstance(b, ast.Assign) and isinstance(b.value, ast.Call)
                        and getattr(b.value.func, 'id', '') == 'Column'):
                    out.append(b.targets[0].id)
            return out
    raise AssertionError(f'модель {name} зникла')


def _busy_market(ff=None, now=NOW):
    """Ринок, який ПРОТИ банера LONG (обидві часткові ознаки засвічені)."""
    snap = _snap(**{f'C{i}': ('LONG', 70, 'down') for i in range(8)})
    trends = {f'C{i}': 'SHORT' for i in range(8)}
    return (ff or _mk(trends=trends)), snap


def test_every_sample_carries_the_thresholds_that_were_in_force():
    """ГОЛОВНЕ для калібрування: у рядку мусять бути і ЗНАЧЕННЯ ознак, і
    ПОРОГИ, що діяли в ту мить. Пороги ж і крутитимуть за підсумками аналізу —
    без їх знімка старі рядки стануть нечитабельними («60% це багато чи мало
    було тоді?»)."""
    _install_db()
    ff, snap = _busy_market()
    _tick(ff, snap)
    r = _rows()[-1]
    for k in ('vob_pct', 'price_pct', 'lever', 'lever_peak',
              'vob_need', 'price_need', 'lever_need', 'need_layers',
              'confirm_sec', 'lit', 'lit_hold', 'determined', 'state', 'bias'):
        _check(k in r, f'у рядку логу немає поля {k}: {sorted(r)}')
    _check(r['vob_need'] == 60.0 and r['price_need'] == 60.0,
           f'пороги мусять бути ТІ, що діяли: {r}')
    _check(r['vob_pct'] == 100.0 and r['price_pct'] == 100.0,
           f'значення ознак мусять бути справжніми: {r}')
    _check(r['bias'] == 'LONG' and r['coins'] == 8,
           f'контекст банера теж потрібен для аналізу: {r}')
    print('✓ 🧾 рядок логу несе і значення ознак, і пороги, що діяли')


def test_the_quiet_market_is_logged_too():
    """НЕГАТИВНІ СЕМПЛИ ОБОВʼЯЗКОВІ. Без рядків «корекції немає» видно лише
    те, де детектор спрацював, і неможливо побачити, де він спрацював БИ з
    іншим порогом — тобто калібрувати нема на чому."""
    _install_db()
    snap = _snap(**{f'C{i}': ('LONG', 70, 'up') for i in range(8)})
    ff = _mk(trends={f'C{i}': 'LONG' for i in range(8)})
    _tick(ff, snap)
    r = _rows()
    _check(len(r) == 1 and r[0]['state'] == 'trend',
           f'спокійний ринок теж мусить писатись: {r}')
    _check(r[0]['lit'] == 0 and r[0]['vob_pct'] == 0.0,
           f'у спокійному семплі мусять бути реальні нулі, а не порожнеча: {r[0]}')
    print('✓ 🧾 «корекції немає» теж у логу — інакше калібрувати нема на чому')


def test_samples_are_throttled_but_events_never_are():
    """Такт двигуна 30с — писати щотакту означало б 2880 рядків на добу. Але
    ПОДІЯ (початок/кінець) не має губитись через те, що семпл щойно писався."""
    _install_db()
    ff, snap = _busy_market()
    quiet = _snap(**{f'C{i}': ('LONG', 70, 'up') for i in range(8)})
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m',
                                 'trends': {f'C{i}': 'LONG' for i in range(8)}}
    _tick(ff, quiet, NOW)                      # семпл №1 (trend)
    _tick(ff, quiet, NOW + 30)                 # ще такт — писати нема чого
    _tick(ff, quiet, NOW + 60)
    _check(len(_rows()) == 1, f'семпли мусять троттлитись: {len(_rows())}')
    # А тепер ринок розвернувся — це ПОДІЯ, і вона пише одразу.
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m',
                                 'trends': {f'C{i}': 'SHORT' for i in range(8)}}
    _tick(ff, snap, NOW + 90)
    _check(len(_rows()) == 2 and _rows()[-1]['kind'] == 'start',
           f'початок корекції мусить писатись повз троттл: {_rows()}')
    _check(_rows()[-1]['prev_state'] == 'trend',
           f'перехід мусить бути видно з рядка: {_rows()[-1]}')
    print('✓ ⏱ семпли — за інтервалом, події — завжди')


def test_the_start_and_the_end_are_both_in_the_log():
    """Обидві межі епізоду мусять бути в одній вибірці: без `end` неможливо
    порахувати ні тривалість, ні «чи не запізно ми відпустили ринок»."""
    _install_db()
    ff, snap = _busy_market()
    _tick(ff, snap, NOW)
    quiet = _snap(**{f'C{i}': ('LONG', 70, 'up') for i in range(8)})
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m',
                                 'trends': {f'C{i}': 'LONG' for i in range(8)}}
    _tick(ff, quiet, NOW + 600)
    kinds = [r['kind'] for r in _rows()]
    _check('start' in kinds and 'end' in kinds, f'бракує меж епізоду: {kinds}')
    _end = [r for r in _rows() if r['kind'] == 'end'][-1]
    _check(float(_end['lasted'] or 0) == 600.0,
           f'тривалість корекції мусить бути в рядку: {_end}')
    print('✓ 🔚 початок і кінець епізоду — обидва в логу, з тривалістю')


def test_a_blocked_open_is_written_with_the_symbol_and_the_price():
    """ЦІНА БЛОКУВАННЯ — головне число post-hoc аналізу: «а що зробив ринок
    після того, як ми не зайшли». Без неї «блокування врятувало чи коштувало»
    лишається здогадкою."""
    _install_db()
    _install_log()
    ff = _mk()
    ff._mm_corr = {'state': 'on', 'blocking': True, 'bias': 'LONG',
                   'since': NOW, 'lit': 2, 'need': 2, 'layers': []}
    ff._open = FF._open.__get__(ff)
    for _ in range(4):
        ff._open('AAAUSDT', 'LONG', {'mark_price': 1.25}, ff.get_settings())
    b = _rows('block')
    _check(len(b) == 1, f'рядок блокування — ОДИН на монету за епізод: {len(b)}')
    _check(b[0]['symbol'] == 'AAAUSDT' and b[0]['side'] == 'LONG',
           f'рядок мусить називати монету і бік: {b[0]}')
    _check(b[0]['price'] == 1.25, f'ціна в момент блокування: {b[0]}')
    # Друга монета того самого епізоду — свій рядок, і лічильник росте.
    ff._open('BBBUSDT', 'SHORT', {'mark_price': 9.0}, ff.get_settings())
    b = _rows('block')
    _check(len(b) == 2 and b[-1]['blocked_n'] == 2,
           f'`blocked_n` мусить показувати ціну епізоду: {b}')
    print('✓ 🚫 блокування: монета · бік · ціна · скільки їх за епізод')


def test_both_gates_write_through_one_writer():
    """Вузлів воріт ДВА (`_open` і `on_signal`), а писач мусить лишатись ОДИН:
    дві копії анти-флуду розійшлися б, і `blocked_n` рахував би половину."""
    t = _fn_src(_TM_SRC, 'on_signal')
    _check('note_correction_block' in t,
           'прямий опен не фіксує блокування в сирому логу')
    o = _fn_src(_FF_SRC, '_open')
    _check('note_correction_block' in o, 'черговий опен не фіксує блокування')
    _check('_mm_corr_skip_logged' not in t,
           'TM не має вести власний анти-флуд — ключ живе у FF')
    n = _fn_src(_FF_SRC, 'note_correction_block')
    _check('_mm_corr_skip_logged' in n and "'block'" in n,
           'писач мусить робити і анти-флуд, і рядок логу')
    print('✓ 🚪 обидва вузли воріт пишуть через ОДИН писач')


def test_the_log_can_be_switched_off():
    """Тумблер мусить гасити САМ ЗАПИС, а не лише показ: інакше він брехав би
    про економію (той самий принцип, що з тумблером монітора)."""
    _install_db()
    _install_log()
    ff, snap = _busy_market()
    ff._settings['mm_corr_log_enabled'] = False
    _tick(ff, snap)
    _check(not _rows(), f'вимкнений лог не має писати нічого: {_rows()}')
    ff._mm_corr = {'state': 'on', 'blocking': True, 'bias': 'LONG',
                   'since': NOW, 'lit': 2, 'need': 2, 'layers': []}
    ff._open = FF._open.__get__(ff)
    ff._open('AAAUSDT', 'LONG', {'mark_price': 1.0}, ff.get_settings())
    _check(not _rows('block'), 'вимкнений лог не має писати і блокування')
    _check([x for x in _LOGGED if x['event'] == 'skipped'],
           'але 🧾 Лог роботи бота мусить лишитись — це різні речі')
    print('✓ 🔌 тумблер гасить САМ запис, а подію для людини лишає')


def test_a_broken_log_never_breaks_the_tick():
    """Лог — не привід зупинити торгівлю. Збій БД мусить лишити і вердикт, і
    ворота на місці (той самий принцип, що в `log_readiness`)."""
    _install_db(fail=True)
    ff, snap = _busy_market()
    out = _tick(ff, snap)
    _check(out.get('state') in ('on', 'pending'),
           f'вердикт мусить рахуватись попри збій логу: {out}')
    _check(ff._mm_corr_log_at == 0.0,
           'невдалий запис не має вважатись зробленим (інакше троттл зʼїв би '
           'наступну спробу)')
    _install_db()
    print('✓ 🛟 збій логу не чіпає ні вердикт, ні ворота')


def test_the_engine_never_writes_a_field_the_table_cannot_store():
    """Мовчазна втрата поля = дірка в аналізі. Тому: кожне поле, яке пише
    двигун, мусить бути і колонкою таблиці, і в білому списку шару БД."""
    _install_db()
    ff, snap = _busy_market()
    _tick(ff, snap)
    ff._mm_corr = {'state': 'on', 'blocking': True, 'bias': 'LONG',
                   'since': NOW, 'lit': 2, 'need': 2, 'layers': []}
    ff._open = FF._open.__get__(ff)
    ff._open('AAAUSDT', 'LONG', {'mark_price': 2.0}, ff.get_settings())
    cols = set(_model_columns('MmCorrectionLog'))
    white = set(re.findall(r"'(\w+)'", re.search(
        r'_MM_CORR_FIELDS = \((.*?)\n    \)', _DBOPS_SRC, re.S).group(1)))
    for r in _DBROWS:
        for k in r:
            _check(k in cols, f'поле {k} пишеться, але колонки для нього немає')
            _check(k in white, f'поле {k} відріже білий список шару БД')
    _check('timestamp' in cols and 'kind' in cols, 'бракує базових колонок')
    print('✓ 🗄 кожне записане поле має колонку і проходить білий список')


def test_the_csv_export_lists_every_column():
    """CSV — робочий формат аналізу. Колонка, яку забули в експорті, робить
    дані неповними МОВЧКИ."""
    # ⚠️ Беремо функцію ЗА ТОЧНИМ ІМЕНЕМ через AST, а не регуляркою по
    # префіксу: поруч зʼявився `api_fuel_filter_mm_corr_log_clear`, і пошук
    # `find`/`search` за префіксом чіплявся вже за нього.
    body = _fn_src(_FLASK_SRC, 'api_fuel_filter_mm_corr_log')
    _check("format" in body and 'csv' in body, 'немає CSV-експорту')
    listed = set(re.findall(r"'(\w+)'", body.split('cols = [')[1].split(']')[0]))
    for c in _model_columns('MmCorrectionLog'):
        if c == 'id':
            continue
        _check(c in listed, f'колонки {c} немає в CSV-експорті')
    _check('reversed(rows)' in body,
           'ряд у часі мусить читатись згори вниз (найстаріші зверху)')
    print('✓ 📤 CSV віддає ВСІ колонки, найстаріші зверху')


def test_the_table_is_pruned_like_every_other_service_log():
    """Append-таблиця без чистки одного дня стане проблемою БД — у проєкті це
    вже проходили з логом «Готовності»."""
    _check("'sob_mm_corr_log': ('timestamp', 'dt')" in _FLASK_SRC,
           'таблиця не в переліку службових — її не чистить ні ручна '
           '«Службові», ні DB-autoclean')
    _check('clear_old_mm_corr' in _DBOPS_SRC, 'немає чистки за віком')
    print('✓ 🗑 таблиця чиститься як решта службових логів')


def test_log_defaults_and_clamps():
    _check(mc.DEFAULTS['mm_corr_log_enabled'] is True,
           'без логу пороги калібрувати нема на чому — дефолт УВІМК')
    _check(mc.DEFAULTS['mm_corr_log_every_sec'] == 300,
           '300с ≈ 288 рядків на добу — достатньо і не роздуває БД')
    ff = FF.__new__(FF)
    ff._db = types.SimpleNamespace(get_setting=lambda *a, **k: {
        'mm_corr_log_every_sec': 1})
    s = FF.get_settings(ff)
    _check(s['mm_corr_log_every_sec'] == 30,
           'частіше за такт двигуна писати нема чого — мусить обрізатись до 30с')
    print('✓ ⚙️ дефолти логу і нижня межа інтервалу')


def test_ui_has_the_log_controls_and_the_csv_link():
    for el in ('ff-mm-corr-log', 'ff-mm-corr-log-every', 'mm-corr-csv'):
        _check(f'id="{el}"' in _HTML, f'немає контрола {el}')
    for el in ('ff-mm-corr-log', 'ff-mm-corr-log-every'):
        _check(f"'{el}'" in _HTML, f'контрол {el} нікуди не зберігається')
    for key in ('mm_corr_log_enabled', 'mm_corr_log_every_sec'):
        _check(key in _HTML, f'ключ {key} не їде на сервер')
    _check('/api/fuel-filter/mm-corr-log' in _HTML and 'format=csv' in _HTML,
           'немає посилання на вивантаження CSV')
    # Згорнута гармошка мусить казати, чи лог узагалі пишеться.
    _sum = _HTML.split('function _mmCorrSummary(')[1].split('\nfunction ')[0]
    _check('mm_corr_log_enabled' in _sum,
           'вимкнений лог мусить бути видно, не розгортаючи гармошку')
    print('✓ 🖥 UI: тумблер логу, інтервал, CSV і розклад у шапці')


# ═══════ 7. 🎚 ПЕРЕКАЛІБРУВАННЯ ПОРОГІВ + 🗑 ОЧИЩЕННЯ ЛОГУ (21.09) ═══════
def test_new_defaults_come_from_the_log_analysis():
    """Пороги змінені за підсумками 26-год логу: 💹 Ціна — найшумніший шар
    (медіанний стрибок 14.4 п.п. проти 1.4 у VOB) і запускала 30 із 33
    корекцій; 9 стартів були миготінням усередині епізоду."""
    _check(mc.DEFAULTS['mm_corr_price_pct'] == 80.0,
           f"поріг ціни: {mc.DEFAULTS['mm_corr_price_pct']}")
    _check(mc.DEFAULTS['mm_corr_confirm_sec'] == 300,
           f"підтвердження: {mc.DEFAULTS['mm_corr_confirm_sec']}")
    # Решту НЕ чіпали: VOB стабільний і саме він ТРИМАЄ корекцію.
    _check(mc.DEFAULTS['mm_corr_vob_pct'] == 60.0, 'поріг VOB не мали чіпати')
    _check(mc.DEFAULTS['mm_corr_min_layers'] == 2,
           '«3 з 3» дало б 3 епізоди за добу — це вимкнений детектор')
    print('✓ нові дефолти: 💹 80% · ⏱ 300с (VOB і «2 з 3» без змін)')


class _MigDB:
    """Шар БД лише для міграції: один блоб налаштувань."""
    def __init__(self, stored):
        self.store = {'fuel_filter_settings': stored}
    def get_setting(self, k, d=None):
        return self.store.get(k, d)
    def set_setting(self, k, v):
        self.store[k] = v


def _migrate(stored):
    ff = FF.__new__(FF)
    ff._db = _MigDB(dict(stored))
    FF._migrate_settings(ff)
    return ff._db.store['fuel_filter_settings']


def test_a_saved_blob_really_gets_the_new_thresholds():
    """⚠️ ГОЛОВНЕ: сторінка шле УСІ ключі одним блобом, тож старий дефолт уже
    лежить у БД і перекрив би новий — правка в коді не зробила б НІЧОГО
    (урок `pilot_autofill_migrated_v1`)."""
    out = _migrate({'mm_corr_price_pct': 60.0, 'mm_corr_confirm_sec': 120,
                    'enabled': True})
    _check(out['mm_corr_price_pct'] == 80.0, f'ціна не переїхала: {out}')
    _check(out['mm_corr_confirm_sec'] == 300, f'підтвердження не переїхало: {out}')
    _check(out['enabled'] is True, 'решта налаштувань мусить лишитись')
    print('✓ збережений блоб зі СТАРИМИ дефолтами перекалібровано')


def test_a_value_the_user_set_by_hand_is_never_touched():
    """Переписуємо ЛИШЕ те, що дорівнює старому дефолту: 65 чи 240 людина
    поставила руками, і мовчки зрушити це означало б змінити чужу настройку."""
    out = _migrate({'mm_corr_price_pct': 65.0, 'mm_corr_confirm_sec': 240})
    _check(out['mm_corr_price_pct'] == 65.0, f'ручне значення зрушили: {out}')
    _check(out['mm_corr_confirm_sec'] == 240, f'ручне значення зрушили: {out}')
    print('✓ значення, задане руками, міграція не чіпає')


def test_the_migration_runs_exactly_once():
    """Інакше вона щоразу відкочувала б вибір користувача назад до 80/300."""
    out = _migrate({'mm_corr_price_pct': 60.0})
    _check(out.get(FF.MM_CORR_TUNE_FLAG) is True, 'позначку не поставлено')
    out['mm_corr_price_pct'] = 60.0          # користувач свідомо повернув 60
    again = _migrate(out)
    _check(again['mm_corr_price_pct'] == 60.0,
           f'міграція спрацювала ВДРУГЕ і перебила вибір: {again}')
    print('✓ міграція одноразова — другий старт вибір не перебиває')


def test_a_clean_install_is_stamped_too():
    """⚠️ Порожній блоб теж отримує позначку: інакше міграція спрацювала б на
    ДРУГОМУ старті й перекрила б вибір, який користувач уже встиг зробити."""
    out = _migrate({})
    _check(out.get(FF.MM_CORR_TUNE_FLAG) is True,
           f'чиста установка мусить бути позначена: {out}')
    print('✓ чиста установка позначається одразу')


def test_a_broken_db_never_breaks_boot():
    """Міграція живе в `__init__` — виняток там поклав би весь рушій."""
    class _Boom:
        def get_setting(self, *a, **k): raise RuntimeError('БД лягла')
        def set_setting(self, *a, **k): raise RuntimeError('БД лягла')
    ff = FF.__new__(FF); ff._db = _Boom()
    FF._migrate_settings(ff)          # не має підняти виняток
    print('✓ збій БД під час міграції не валить старт')


def test_ui_fallbacks_match_the_new_defaults():
    """Хардкоджені фолбеки в HTML — друге написання того самого числа. Якщо
    вони відстануть, поле показуватиме 60 там, де бекенд рахує 80."""
    for old in ('_mmcNum(\'ff-mm-corr-price\', 60)',
                '_mmcNum(\'ff-mm-corr-confirm\', 120)',
                's.mm_corr_price_pct != null ? s.mm_corr_price_pct : 60',
                's.mm_corr_confirm_sec != null ? s.mm_corr_confirm_sec : 120'):
        _check(old not in _HTML, f'у сторінці лишився старий фолбек: {old}')
    _check("_mmcNum('ff-mm-corr-price', 80)" in _HTML, 'фолбек ціни не оновлено')
    _check("_mmcNum('ff-mm-corr-confirm', 300)" in _HTML, 'фолбек вікна не оновлено')
    _check('id="ff-mm-corr-price" min="0" max="100" step="5" value="80"' in _HTML,
           'value= у полі ціни не оновлено')
    _check('id="ff-mm-corr-confirm" min="0" max="3600" step="30" value="300"' in _HTML,
           'value= у полі підтвердження не оновлено')
    print('✓ фолбеки UI збігаються з новими дефолтами')


def test_clear_button_exists_and_its_route_is_registered():
    """Урок submitManualTp1: вигаданий URL виглядає як «збережено»."""
    _check('mmCorrLogClear(' in _HTML, 'немає кнопки очищення логу')
    _check('id="mm-corr-clear"' in _HTML, 'немає самої кнопки')
    _check('id="mm-corr-clear-note"' in _HTML, 'немає місця для відповіді')
    _check('/api/fuel-filter/mm-corr-log/clear' in _HTML, 'кнопка нікуди не шле')
    fl = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    _check("@app.route('/api/fuel-filter/mm-corr-log/clear'" in fl,
           'маршрут очищення не зареєстровано')
    i = _HTML.find('async function mmCorrLogClear')
    body = _HTML[i:i + 1600]
    _check('confirm(' in body, 'незворотна дія мусить питати підтвердження')
    _check('if (!r.ok)' in body, 'без перевірки HTTP 404 виглядав би як успіх')
    _check('d.deleted' in body, 'результат мусить називати КІЛЬКІСТЬ видаленого')
    print('✓ кнопка очищення: підтвердження · маршрут · відповідь числом')


def test_clearing_reuses_the_single_delete_implementation():
    """Другої копії DELETE не заводимо — розійшлись би (та сама чистка за
    віком уже живе в `clear_old_mm_corr`)."""
    fl = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    # ⚠️ Ріжемо ДОКСТРІНГ: він сам ПОЯСНЮЄ, чому власного DELETE тут немає —
    # та сама пастка, що вже ловили на `ensure_fresh`.
    body = _fn_src(fl, 'api_fuel_filter_mm_corr_log_clear')
    _check('clear_old_mm_corr' in body, 'маршрут мусить кликати наявний метод')
    # ⚠️ Шукаємо САМЕ SQL, а не слово: поле відповіді `deleted` містить його
    # як підрядок, і груба перевірка падала на власній назві поля.
    _check('DELETE FROM' not in body.upper() and 'text(' not in body,
           'власного SQL у маршруті бути не має')
    print('✓ очищення йде через ЄДИНУ наявну реалізацію')



# ═══════ 8. 🧭 ПРОФЕСІЙНИЙ ПОЧАТОК/КІНЕЦЬ: ШИРИНА РИНКУ ВИРІШУЄ (22.09) ════
# **Скарга користувача, дослівно:** «Мені не подобається як працює корекція.
# Це точно не професійному рівні якщо VOB 80% і корекція завершилась. Це як
# так завершилась, коли майже всі монети у протилежному стані? А ціна, як
# можна визначити за 15хв, що корекція завершилась, коли на проміжку
# наприклад в один день корекція продовжується.»
#
# Лог проду (12 год, 144 семпли) підтвердив дослівно: 11 «завершень» при
# 📦 VOB 63-90% проти банера, одне — при 90.1% (64 монети з 71). Причина:
# три ознаки були РІВНИМИ голосами, тож кінець ухвалювали два найшвидші шари
# (💹 15 хв, 📉 30 хв), перекриваючи повільну структурну ширину.


def _six(price_dir='down'):
    """Шість монет: банер LONG, усі йдуть проти нього (щоб вибірка ≥ MIN_SAMPLE)."""
    return _snap(**{f'C{i}': ('LONG', 50, price_dir) for i in range(6)})


def _cfg(**kw):
    c = {'mm_corr_min_layers': 2, 'mm_corr_vob_pct': 60.0,
         'mm_corr_price_pct': 60.0, 'mm_corr_lever_drop': 15.0}
    c.update(kw)
    return c


def test_breadth_vetoes_the_end_while_most_coins_are_against():
    """ГОЛОВНИЙ ЗАМОК СКАРГИ: при 📦 90% проти банера корекція завершитись
    НЕ МОЖЕ, хай навіть 💹 ціна і 📉 важіль зовсім заспокоїлись.

    Числа взяті з реального рядка логу 22.09 03:32 — саме там стара версія
    оголосила кінець: vob 90.1% (64/71), price 7.7%, lever_drop 6.4.
    """
    trends = {f'C{i}': 'SHORT' for i in range(9)}
    trends['C9'] = 'LONG'                       # 9 з 10 проти → 90%
    snap = _snap(**{f'C{i}': ('LONG', 50, 'up') for i in range(10)})  # ціна ЗА банер
    r = mc.evaluate(snap, trends, 'LONG', 50.0, 51.0, _cfg())
    _check(r['layers'][0]['pct'] == 90.0, f'📦 мусить бути 90%: {r["layers"][0]}')
    _check(not r['layers'][1]['lit'], '💹 ціна заспокоїлась')
    _check(not r['layers'][2]['lit'], '📉 важіль майже не просів')
    _check(r['lit_hold'] < r['need'],
           'за СТАРИМ правилом голосів корекція вже «завершилась» би')
    _check(r['stay'] is True,
           'ширина 90% мусить ЗАБОРОНИТИ кінець — це і є суть скарги')
    _check(r['breadth_ok'] is False, f'вето ширини: {r["breadth_ok"]}')
    # ⚠️ Числа беремо з РЕЗУЛЬТАТУ, а не магічні: і дефолт, і САМА ШКАЛА
    # порогу вже мінялись (проти <50 → проти <65 → за ≥70), і зашите число
    # ламало б тест на рівному місці. Причина мусить називати те, що РЕАЛЬНО
    # вирішує кінець, — частку ТИХ, ХТО ПОВЕРНУВСЯ, і поріг.
    _check(str(r['breadth_for_pct']) in r['why']
           and str(r['exit_pct']) in r['why'],
           f'причина мусить називати ЧИСЛА: {r["why"]}')
    print('✓ 🧭 при 📦 90% проти банера кінець корекції ЗАБЛОКОВАНО')


def test_the_correction_ends_only_when_breadth_recovers():
    """Кінець настає САМЕ тоді, коли ЗА банером повернулось ≥ порога монет.

    ⚠️ ПЕРЕПИСАНО 22.09: поріг тепер міряє тих, хто ПОВЕРНУВСЯ, а не тих,
    хто ще проти (див. розділ 10)."""
    snap = _snap(**{f'C{i}': ('LONG', 50, 'up') for i in range(10)})
    # ⚠️ Поріг беремо з КОНФІГА, а не числом: і дефолт, і сама ШКАЛА вже
    # мінялись, тож зашите число ламало б тест на рівному місці.
    _cfg_now = _cfg()
    _exit = float(_cfg_now.get('mm_corr_vob_exit_pct',
                              mc.DEFAULTS['mm_corr_vob_exit_pct']))

    def _back(n):
        """n монет ПОВЕРНУЛОСЬ за банером (решта — проти)."""
        t = {f'C{i}': 'LONG' for i in range(n)}
        t.update({f'C{i}': 'SHORT' for i in range(n, 10)})
        return mc.evaluate(snap, t, 'LONG', 50.0, 51.0, _cfg_now)

    _few = max(0, int(_exit // 10) - 1)     # повернулось МЕНШЕ порога
    _many = min(10, int(_exit // 10) + 1)   # повернулось БІЛЬШЕ порога
    _check(_back(_few)['stay'] is True,
           f'повернулось {_few * 10}% (< {_exit}) — корекція мусить тривати')
    r = _back(_many)
    _check(r['stay'] is False and r['breadth_ok'] is True,
           f'повернулось {_many * 10}% — кінець дозволено: '
           f'{r["stay"]}/{r["breadth_ok"]}')
    # Рівно НА порозі кінець УЖЕ дозволено — поріг це «скільки досить» («≥»),
    # так само як на старті.
    _check(mc.breadth_exit_ok(
        {'ok': True, 'pct': 100.0 - _exit, 'for_pct': _exit}, _exit) is True,
        f'рівно {_exit}% повернутих — цього вже досить')
    print('✓ 🧭 кінець настає, коли повернулось ≥ порога монет')


def test_fast_layers_no_longer_hold_a_correction_after_breadth_recovered():
    """⚠️ ПЕРЕПИСАНО 22.09 — правило СКАСОВАНО, а не «полагоджено».

    Було: «💹 ціна і 📉 важіль можуть корекцію лише ПРОДОВЖИТИ». Саме це й
    тримало корекцію нескінченно (скарга «немає повідомлень про вихід»):
    ширина рахувалась ДВІЧІ — у вето і серед голосів `lit_hold`, тож знявши
    вето, вона сама себе й тримала. Тепер ширина визначена → ВОНА й вирішує.
    """
    snap = _snap(**{f'C{i}': ('LONG', 50, 'down') for i in range(10)})  # 💹 100%
    # 8 із 10 ПОВЕРНУЛИСЬ → за 80% ≥ 70% (дефолт)
    t = {f'C{i}': 'LONG' for i in range(8)}
    t.update({f'C{i}': 'SHORT' for i in range(8, 10)})
    r = mc.evaluate(snap, t, 'LONG', 30.0, 60.0, _cfg())   # важіль −30 п.п.
    _check(r['breadth_ok'] is True, f'ширина відновилась: {r["why"]}')
    _check(r['lit_hold'] >= r['need'],
           f'голоси 💹+📉 ще горять: {r["lit_hold"]}/{r["need"]}')
    _check(r['stay'] is False,
           'і все одно кінець дозволено — вирішує ШИРИНА, а не голоси')
    print('✓ 🧭 після відновлення ширини голоси корекцію не тримають')


def test_the_price_and_lever_layers_are_marked_start_only():
    """Роль шару віддає БЕКЕНД (`role`), а не вгадує фронт — той самий
    принцип, що `unit`. Інакше «💹 7.7%/80%» поруч із живою корекцією
    виглядало б як суперечність."""
    r = mc.evaluate(_six(), {f'C{i}': 'SHORT' for i in range(6)},
                    'LONG', 50.0, 52.0, _cfg())
    roles = {x['key']: x.get('role') for x in r['layers']}
    _check(roles['vob'] == 'both', f'📦 ширина вирішує і початок, і кінець: {roles}')
    _check(roles['price'] == 'start', f'💹 ціна — лише початок: {roles}')
    _check(roles['lever'] == 'start', f'📉 важіль — лише початок: {roles}')
    print('✓ 🧭 роль кожного шару приходить із бекенда')


def test_breadth_is_required_to_start_a_correction():
    """Корекція — подія ШИРИНИ. 💹+📉 на спокійній структурі її не оголошують."""
    # Ширина ЗА банер (усі 6 монет LONG), але ціна й важіль проти.
    snap = _six()
    trends = {f'C{i}': 'LONG' for i in range(6)}
    r = mc.evaluate(snap, trends, 'LONG', 20.0, 60.0, _cfg())
    _check(r['lit'] >= r['need'], 'голосів 💹+📉 формально досить')
    _check(r['start_ok'] is False,
           'але без ширини корекцію оголошувати не можна')
    off = mc.evaluate(snap, trends, 'LONG', 20.0, 60.0,
                      _cfg(mm_corr_vob_required=False))
    _check(off['start_ok'] is True,
           'вимкнений тумблер мусить повертати стару поведінку «2 з 3»')
    print('✓ 🧭 📦 ширина обовʼязкова, щоб ПОЧАТИ корекцію')


def test_no_breadth_data_falls_back_to_the_old_vote_rule():
    """«Немає даних» ≠ «ширина відновилась». Коли 📦 визначити нічим (блок
    вимкнено / мала вибірка), рішення ухвалює стара логіка голосів — вигадувати
    кінець корекції на порожньому місці не можна."""
    snap = _six()
    r = mc.evaluate(snap, {}, 'LONG', 30.0, 60.0, _cfg())   # трендів немає зовсім
    _check(r['breadth_ok'] is None, f'ширини немає: {r["breadth_ok"]}')
    _check(not r['layers'][0]['ok'], '📦 шар мусить бути НЕВИЗНАЧЕНИМ')
    _check(r['stay'] == (r['lit_hold'] >= r['need']),
           'без ширини працює стара арифметика голосів')
    _check('нічим' in r['why'], f'причина мусить це називати: {r["why"]}')
    print('✓ 🧭 немає даних ширини → фолбек на старе правило голосів')


def test_the_toggle_restores_the_old_behaviour_exactly():
    """`mm_corr_breadth_exit=False` мусить повертати РІВНО стару поведінку —
    інакше тумблер обіцяв би те, чого не робить."""
    snap = _snap(**{f'C{i}': ('LONG', 50, 'up') for i in range(10)})
    trends = {f'C{i}': 'SHORT' for i in range(9)}
    trends['C9'] = 'LONG'                                   # 📦 90% проти
    off = mc.evaluate(snap, trends, 'LONG', 50.0, 51.0,
                      _cfg(mm_corr_breadth_exit=False))
    _check(off['breadth_ok'] is None, 'вимкнено → ширина не голосує на вихід')
    _check(off['stay'] == (off['lit_hold'] >= off['need']),
           'вимкнено → рішення рівно за голосами (стара логіка)')
    _check(off['stay'] is False,
           'саме так стара версія і «завершувала» корекцію при 90% проти')
    print('✓ 🧭 тумблер повертає стару поведінку рівно такою, якою вона була')


def test_state_machine_takes_the_ready_made_decisions():
    """`next_state` — машина станів, а не суддя ознак: `start_ok`/`stay`
    приходять готові з `evaluate`. Дефолт `None` лишає стару арифметику, тож
    усі наявні виклики поводяться як раніше."""
    # Голосів немає (0/2), але `start_ok=True` мусить перекрити підрахунок.
    st = mc.next_state({}, 0, 0, 2, 1000.0, 0, start_ok=True)
    _check(st['state'] == 'on', f'рішення ззовні мусить діяти: {st}')
    # Голосів немає, але `stay=True` (вето ширини) тримає корекцію.
    st2 = mc.next_state(st, 0, 0, 2, 1100.0, 0, stay=True)
    _check(st2['state'] == 'on', f'вето ширини тримає корекцію: {st2}')
    st3 = mc.next_state(st2, 0, 0, 2, 1200.0, 0, stay=False)
    _check(st3['state'] == 'ended', f'ширина відпустила → кінець: {st3}')
    # Стара сигнатура — стара поведінка.
    _check(mc.next_state({}, 2, 2, 2, 1000.0, 0)['state'] == 'on',
           'без нових аргументів машина мусить рахувати голоси як раніше')
    print('✓ 🧭 машина станів приймає готові рішення, стара сигнатура жива')


def test_the_engine_passes_the_decisions_to_the_state_machine():
    """Замок на ПРОВОДКУ: правила мають сенс лише якщо двигун їх передає."""
    body = _fn_src(_FF_SRC, '_mm_track_correction')
    _check('start_ok=res' in body and 'stay=res' in body,
           'двигун мусить передавати ГОТОВІ рішення в next_state')
    print('✓ 🧭 двигун передає рішення `evaluate` у машину станів')


def test_a_long_correction_survives_a_calm_price_window():
    """ПОВНИЙ ПРОГІН ЧЕРЕЗ ДВИГУН — дослівна відповідь на скаргу «як можна
    визначити за 15хв, що корекція завершилась»."""
    trends = {f'C{i}': 'SHORT' for i in range(9)}
    trends['C9'] = 'LONG'                                   # 📦 90% проти
    ff = _mk(trends=trends, mm_corr_confirm_sec=0)
    # Такт 1: усе проти банера → корекція почалась.
    _tick(ff, _snap(**{f'C{i}': ('LONG', 60, 'down') for i in range(10)}), NOW)
    _check(ff._mm_corr['state'] == 'on', f'корекція мусить початись: {ff._mm_corr}')
    # Такт 2: ціна повністю заспокоїлась і пішла ЗА банером, важіль стабільний.
    _tick(ff, _snap(**{f'C{i}': ('LONG', 60, 'up') for i in range(10)}), NOW + 60)
    _check(ff._mm_corr['state'] == 'on',
           f'спокійні 15 хв ціни НЕ мають завершувати корекцію: {ff._mm_corr}')
    _check(ff._mm_corr.get('breadth_ok') is False, 'тримає саме ширина')
    # Такт 3: структура монет реально розвернулась → кінець.
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m',
                                 'trends': {f'C{i}': 'LONG' for i in range(10)}}
    _tick(ff, _snap(**{f'C{i}': ('LONG', 60, 'up') for i in range(10)}), NOW + 120)
    _check(ff._mm_corr['state'] == 'ended',
           f'ширина відновилась → кінець: {ff._mm_corr}')
    print('✓ 🧭 багатогодинна корекція переживає спокійне 15-хв вікно ціни')


def test_the_end_event_names_the_breadth_number():
    """Рядок «завершилась» мусить нести ЧИСЛО ширини: саме на мовчазному
    «завершилась» стару версію і спіймали."""
    ff = _mk(trends={f'C{i}': 'SHORT' for i in range(6)}, mm_corr_confirm_sec=0)
    _tick(ff, _six(), NOW)
    _check(ff._mm_corr['state'] == 'on', 'корекція почалась')
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m',
                                 'trends': {f'C{i}': 'LONG' for i in range(6)}}
    _tick(ff, _snap(**{f'C{i}': ('LONG', 50, 'up') for i in range(6)}), NOW + 60)
    ends = [m for m in _LOGGED if 'ЗАВЕРШИЛАСЬ' in str(m)]
    _check(ends, f'подія кінця мусить бути в лозі: {_LOGGED}')
    _check('ширина' in str(ends[-1]),
           f'рядок кінця мусить називати ширину: {ends[-1]}')
    print('✓ 🧭 подія «завершилась» називає число ширини')


def test_the_two_thresholds_are_independent():
    """⚠️ ПЕРЕПИСАНО 22.09, а не «полагоджено»: вимога змінилась.

    Раніше тут стояв замок «вихід ≤ вхід». Користувач попросив 60/65 і має
    рацію — обмеження було зайвим (див. розділ 10). Лишається лише 0..100."""
    ff = _mk_ff({'mm_corr_vob_pct': 60.0, 'mm_corr_vob_exit_pct': 90.0})
    s = FF.get_settings(ff)
    _check(float(s['mm_corr_vob_exit_pct']) == 90.0,
           f'поріг виходу НЕ можна зводити: {s["mm_corr_vob_exit_pct"]}')
    _check(s['mm_corr_vob_required'] is True and s['mm_corr_breadth_exit'] is True,
           'нові тумблери мусять проходити валідацію з дефолтами')
    print('✓ 🧭 пороги незалежні, лишилось лише 0..100')


def test_new_keys_have_professional_defaults():
    """Дефолти: ширина обовʼязкова, кінець вирішує ширина.
    ⚠️ Саме ЧИСЛО порога виходу перевіряє розділ 10 (50 → 65 на вимогу 22.09)."""
    d = mc.DEFAULTS
    _check(d['mm_corr_vob_required'] is True, '📦 обовʼязкова на старт')
    _check(d['mm_corr_breadth_exit'] is True, '🧭 кінець вирішує ширина')
    # ⚠️ Звʼязку «вихід = вхід − EXIT_MARGIN_PCT» БІЛЬШЕ НЕМАЄ: 22.09 користувач
    # обрав 60/65, тобто вихід ВИЩЕ за вхід. Гістерезис лишився в `lit_hold`
    # (послаблені пороги голосів) — а поріг ширини тепер незалежне число.
    print('✓ 🧭 дефолти: ширина обовʼязкова · кінець за шириною · 50%')


def test_the_raw_log_carries_the_exit_threshold_too():
    """Поріг ВИХОДУ тепер вирішує кінець, тож без нього в рядку старі семпли
    стануть нечитабельними так само, як без `*_need`."""
    _check('vob_exit' in _fn_src(_FF_SRC, '_mm_corr_log_write'),
           'писач логу мусить класти поріг виходу в рядок')
    cols = set(_model_columns('MmCorrectionLog'))
    _check('vob_exit' in cols, 'у моделі БД немає колонки vob_exit')
    dbo = open(os.path.join(_HERE, 'storage', 'db_operations.py'),
               encoding='utf-8').read()
    _check("'vob_exit'" in dbo, 'білий список шару БД відріже нове поле')
    models = open(os.path.join(_HERE, 'storage', 'db_models.py'),
                  encoding='utf-8').read()
    # ⚠️ Міграція стала СПИСКОМ колонок (до `vob_exit` додались числа 📐 OB),
    # тож замок дивиться на ЗАПИС у списку, а не на готовий рядок ALTER.
    _check("('vob_exit', 'FLOAT')" in models,
           'таблиця вже могла бути створена без колонки — потрібна міграція')
    _check('ADD COLUMN {col_name} {col_type}' in models,
           'міграція mm_corr_log мусить виконувати ALTER для кожної колонки')
    print('✓ 🧭 поріг виходу є і в рядку логу, і в БД, і в міграції')


def test_ui_exposes_the_new_rules_and_says_which_layer_is_start_only():
    """Мовчазна зміна правил читалась би як «бот зламався»: обидва тумблери,
    поріг виходу і позначка «лише старт» мусять бути на сторінці."""
    for _id in ('ff-mm-corr-vobreq', 'ff-mm-corr-breadth', 'ff-mm-corr-vobexit'):
        _check(f'id="{_id}"' in _HTML, f'немає контрола {_id}')
    for key in ('mm_corr_vob_required', 'mm_corr_breadth_exit',
                'mm_corr_vob_exit_pct'):
        _check(key in _HTML, f'ключ {key} не зберігається зі сторінки')
    i = _HTML.find('function mmRenderCorr')
    body = _HTML[i:_HTML.find('function _mmcNum', i)]
    _check("l.role === 'start'" in body,
           'рядок мусить показувати, що шар лише для старту')
    _check('exit_pct' in body,
           'поки корекція триває, мусить бути видно, за якої ширини вона скінчиться')
    print('✓ 🧭 UI показує нові правила і роль кожного шару')



# ═══ 9. 📨 ПОДІЯ ≠ ЗМІНА СТАНУ · 📐 OB ЯК ДРУГЕ ДЖЕРЕЛО ШИРИНИ (22.09) ════
# **Дві скарги користувача одним пакетом:**
#  1) «Проаналізуй оповіщення, вони не зовсім нормально відпрацьовують.»
#     Скрін Telegram за 14 год: «✅ ЗАВЕРШИЛАСЬ (тривала 17хв)» ДВІЧІ
#     (21:20 і 21:33), «(тривала 23хв)» двічі (0:16 і 0:31), «(тривала 27хв)»
#     двічі (1:58 і 2:10); два 🔻 підряд без ✅ між ними (8:08 і 8:30);
#     «🔻 КОРЕКЦІЯ … ознак 1/2» — менше, ніж потрібно.
#  2) «Зверни увагу на VOB, вони не відразу зʼявляються, навіть коли монета
#     отримала протилежний рух. Додай ще відслідковування OB.»


def test_returning_from_ending_is_not_a_new_correction():
    """`ending → on` — ТОЙ САМИЙ епізод (`since` не змінився), а не старт.
    На скріні це давало два 🔻 підряд без ✅ між ними."""
    st = mc.next_state({}, 2, 2, 2, 1000.0, 0)
    _check(st['event'] == 'start' and st['state'] == 'on', f'старт: {st}')
    _since = st['since']
    st = mc.next_state(st, 0, 0, 2, 1100.0, 300)      # ознаки зникли → ending
    _check(st['state'] == 'ending' and st['event'] == '', f'відлік кінця: {st}')
    st = mc.next_state(st, 2, 2, 2, 1200.0, 300)      # повернулись → on
    _check(st['state'] == 'on', f'корекція триває далі: {st}')
    _check(st['event'] == 'resume',
           f'це ПОВЕРНЕННЯ, а не новий старт: {st["event"]}')
    _check(st['since'] == _since, 'таймер епізоду НЕ перезапускається')
    print('✓ 📨 повернення з `ending` — не новий старт (два 🔻 підряд)')


def test_an_aborted_false_start_does_not_re_announce_the_end():
    """`pending → ended` — несправдливий старт згас, а бейдж «завершилась»
    просто повернувся. Саме тут народжувалось ДРУГЕ «✅ ЗАВЕРШИЛАСЬ» із ТІЄЮ
    САМОЮ тривалістю через 10-15 хв (21:20 і 21:33 «тривала 17хв»)."""
    st = mc.next_state({}, 2, 2, 2, 0.0, 0)           # on
    st = mc.next_state(st, 0, 0, 2, 1020.0, 0)        # → ended (17 хв)
    _check(st['event'] == 'end', f'справжній кінець: {st}')
    _lasted = st['lasted']
    st = mc.next_state(st, 2, 2, 2, 1800.0, 300)      # смик ознак → pending
    _check(st['state'] == 'pending' and st['event'] == '', f'{st}')
    st = mc.next_state(st, 0, 0, 2, 1860.0, 300)      # смик минув
    _check(st['state'] == 'ended', f'бейдж повертається: {st}')
    _check(st['event'] == 'abort',
           f'але це НЕ подія «завершилась»: {st["event"]}')
    _check(st['lasted'] == _lasted, 'тривалість та сама — тому й було видно дубль')
    print('✓ 📨 згаслий несправдливий старт НЕ шле друге «✅ завершилась»')


def test_only_real_events_reach_the_log_and_telegram():
    """Замок на ПРОВОДКУ: читач мусить дивитись на `event`, а не на зміну
    стану — інакше повернеться рівно те, що на скріні."""
    body = _fn_src(_FF_SRC, '_mm_track_correction')
    _check("_ev in ('start', 'end')" in body,
           'подію мусить вирішувати `event`, а не порівняння станів')
    _check("st.get('state') in ('on', 'ended')" not in body,
           'стара умова «стан змінився» мусить зникнути')
    _check('_broadcast_users' in body, 'Telegram-гілка на місці')
    print('✓ 📨 у Telegram і 🧾 Лог ідуть ЛИШЕ справжні події')


def test_the_raw_log_kind_follows_the_event_too():
    """Калібрувальний лог рахував `resume` як `start`, а `abort` як `end` —
    тобто ЗАВИЩУВАВ кількість епізодів, і аналіз за ним рахував не те."""
    body = _fn_src(_FF_SRC, '_mm_track_correction')
    # ⚠️ `_fn_src` віддає AST-unparse, тож дужки/переноси нормалізовані —
    # шукаємо СУТЬ, а не дослівний рядок із файлу.
    _kl = [l for l in body.splitlines() if '_kind' in l and '=' in l]
    _check(_kl and "_ev in ('start', 'end')" in _kl[0],
           f'kind сирого логу мусить іти від події: {_kl}')
    print('✓ 📨 kind сирого логу теж від події, а не від стану')


def test_a_real_start_can_never_print_fewer_signs_than_needed():
    """«ознак 1/2» на скріні бралось із повернення `ending → on`: воно йде за
    ПОСЛАБЛЕНИМ порогом, а в текст друкувався СУВОРИЙ лічильник."""
    st = mc.next_state({}, 2, 2, 2, 0.0, 0)
    st = mc.next_state(st, 0, 0, 2, 100.0, 300)       # ending
    st = mc.next_state(st, 1, 2, 2, 200.0, 300)       # суворих 1, послаблених 2
    _check(st['state'] == 'on' and st['event'] == 'resume',
           f'повернення за послабленим: {st}')
    # А справжній старт завжди має lit >= need.
    st2 = mc.next_state({}, 2, 2, 2, 0.0, 0)
    _check(st2['event'] == 'start', 'справжній старт')
    print('✓ 📨 «ознак 1/2» більше не потрапляє в рядок старту')


def test_the_end_message_does_not_promise_openings_that_were_never_blocked():
    """«відкриття знову дозволені» при вимкнених воротах суперечило власному
    рядку старту («відкриття НЕ блокуємо»)."""
    body = _fn_src(_FF_SRC, '_mm_track_correction')
    _check("_blk_on = bool(settings.get('mm_corr_block_open', True))" in body,
           'кінець мусить питати, чи ворота взагалі були увімкнені')
    _check("', відкриття знову дозволені' if _blk_on else ''" in body,
           'фраза мусить бути умовною')
    print('✓ 📨 кінець не обіцяє зняти блок, якого не було')


def test_telegram_does_not_repeat_the_topic_name_three_times():
    """На скріні кожне повідомлення несло назву тричі: тег #МММ_монітор,
    власний заголовок 🧮 МММ-МОНІТОР і сама назва теми."""
    body = _fn_src(_FF_SRC, '_mm_track_correction')
    _check('МММ-МОНІТОР' not in body,
           'власний заголовок у повідомленні корекції зайвий — тег уже є')
    print('✓ 📨 назва теми більше не дублюється в тілі повідомлення')


# ───────────────────── 📐 OB як друге джерело ширини ─────────────────────

def _t(n_against, n_for, side='SHORT', other='LONG'):
    """Тренди по тих САМИХ символах, що й `_snap` (C0…): `evaluate` ріже
    тренди за ключами знімка, тож чужі імена дали б порожню вибірку."""
    d = {f'C{i}': side for i in range(n_against)}
    d.update({f'C{n_against + i}': other for i in range(n_for)})
    return d


def test_ob_layer_mirrors_the_vob_layer_in_shape():
    """Форма 1-в-1 — інакше UI і лог довелось би вчити двом розкладам."""
    v = mc.vob_layer(_t(7, 3), None, 'LONG', 60.0, tf='5m')
    o = mc.ob_layer(_t(7, 3), None, 'LONG', 60.0, tf='5m')
    _check(set(v.keys()) == set(o.keys()), f'різні поля: {set(v) ^ set(o)}')
    _check(o['key'] == 'ob' and o['icon'] == '📐', f'{o}')
    _check(o['role'] == 'both', 'ширина вирішує і початок, і кінець')
    _check(o['pct'] == 70.0 and o['lit'], f'7 з 10 проти: {o}')
    print('✓ 📐 шар OB за формою — дзеркало шару VOB')


def test_breadth_is_the_most_alarming_of_the_two_sensors():
    """МАКСИМУМ, а не середнє: одне правило на обидва кінці."""
    v = mc.vob_layer(_t(5, 5), None, 'LONG', 60.0)        # 50%
    o = mc.ob_layer(_t(8, 2), None, 'LONG', 60.0)         # 80%
    b = mc.breadth_of(v, o)
    _check(b['ok'] and b['pct'] == 80.0, f'максимум із двох: {b}')
    _check(b['src'] == '📐', f'джерело мусить бути назване: {b}')
    _check(mc.breadth_of(v, None)['pct'] == 50.0, 'лише VOB — його число')
    _check(mc.breadth_of(None, o)['pct'] == 80.0, 'лише OB — його число')
    _check(mc.breadth_of(None, None)['ok'] is False,
           'жодного → НЕ визначено (а не «ширина відновилась»)')
    print('✓ 📐 ширина = найтривожніший із двох сенсорів')


def test_ob_sees_the_move_that_vob_has_not_drawn_yet():
    """ДОСЛІВНА скарга: «VOB не відразу зʼявляються, навіть коли монета
    отримала протилежний рух». 📦 ще показує старий бік, 📐 вже перевернувся —
    ширина мусить це ПОБАЧИТИ і корекція почнеться."""
    snap = _snap(**{f'C{i}': ('LONG', 50, 'down') for i in range(10)})
    vob_t = {f'C{i}': 'LONG' for i in range(10)}          # 📦 ще «за банером»
    ob_t = {f'C{i}': 'SHORT' for i in range(10)}          # 📐 уже проти
    r = mc.evaluate(snap, vob_t, 'LONG', 50.0, 52.0, _cfg(),
                    ob_trends=ob_t, ob_tf='5m')
    _check(r['breadth_pct'] == 100.0, f'ширину бачить 📐: {r["breadth_pct"]}')
    _check(r['breadth_lit'] and r['start_ok'],
           f'корекція мусить стартувати: {r}')
    off = mc.evaluate(snap, vob_t, 'LONG', 50.0, 52.0,
                      _cfg(mm_corr_ob_on=False), ob_trends=ob_t, ob_tf='5m')
    _check(not off['breadth_lit'] and not off['start_ok'],
           'вимкнене джерело мусить повертати стару (сліпу) поведінку')
    print('✓ 📐 OB ловить рух, якого 📦 VOB ще не намалював')


def test_ob_also_holds_the_correction_open():
    """Друге джерело працює і на ВИХІД: 📦 вже відпустив, 📐 ще ні."""
    snap = _snap(**{f'C{i}': ('LONG', 50, 'up') for i in range(10)})
    vob_t = _t(3, 7, 'SHORT', 'LONG')                     # 📦 30% — відпустив
    ob_t = _t(7, 3, 'SHORT', 'LONG')                      # 📐 70% — тримає
    r = mc.evaluate(snap, vob_t, 'LONG', 50.0, 51.0, _cfg(),
                    ob_trends=ob_t, ob_tf='5m')
    _check(r['breadth_ok'] is False and r['stay'] is True,
           f'кінець мусить бути заблокований: {r["breadth_ok"]}/{r["stay"]}')
    _check('70.0' in r['why'] and '📐' in r['why'],
           f'причина мусить називати число І джерело: {r["why"]}')
    print('✓ 📐 OB тримає корекцію, коли 📦 VOB уже відпустив')


def test_the_second_source_is_not_a_fourth_vote():
    """«2 з 3» не сміє мовчки стати «2 з 4»: 📦 і 📐 міряють ОДНЕ Й ТЕ САМЕ."""
    snap = _snap(**{f'C{i}': ('LONG', 50, 'up') for i in range(10)})   # 💹 не горить
    both = _t(9, 1, 'SHORT', 'LONG')
    r = mc.evaluate(snap, both, 'LONG', 50.0, 51.0, _cfg(),
                    ob_trends=both, ob_tf='5m')
    _check(len(r['layers']) == 4, f'показуємо ЧОТИРИ шари: {len(r["layers"])}')
    _check(r['lit'] == 1,
           f'але голос лише ОДИН (ширина), а не два: lit={r["lit"]}')
    _check(not r['start_ok'], 'одного голосу при need=2 не досить')
    print('✓ 📐 друге джерело — не четвертий голос, а той самий «ширина»')


def test_the_scanner_feeds_ob_trends_for_free():
    """Замок на ДЖЕРЕЛО: бари вже завантажені, мережі це коштувати не може."""
    sc = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
              encoding='utf-8').read()
    _check('def ob_trends(' in sc, 'сканер мусить мати ПУБЛІЧНИЙ читач')
    body = _fn_src(sc, '_update_ob_trend')
    for bad in ('fetch_klines', '_pf_klines', 'requests', 'session'):
        _check(bad not in body, f'жодної мережі у {bad}')
    _check('detect_order_blocks' in body and 'detect_smc_structure' in body,
           'мусить рахувати ТОЙ САМИЙ OB, що малюється на графіку')
    _check('limit=1' in body, 'потрібен РІВНО поточний блок, як і у VOB')
    ff = _fn_src(_FF_SRC, '_mm_ob_trends')
    _check('ob_trends' in ff, 'FF читає публічним методом, а не чужим кешем')
    _check('_ob_trend_cache' not in ff, 'лізти у приватний кеш сканера не можна')
    print('✓ 📐 джерело OB — той самий скан, нуль запитів до біржі')


def test_older_scanner_or_module_does_not_break_the_detector():
    """Файли деплояться в різному порядку — обидва фолбеки мусять бути."""
    body = _fn_src(_FF_SRC, '_mm_track_correction')
    _check('except TypeError' in body,
           'старіший mm_correction без ob_trends= мусить працювати')
    ff = _fn_src(_FF_SRC, '_mm_ob_trends')
    _check("hasattr(sc, 'ob_trends')" in ff,
           'старіший сканер без ob_trends() → шар НЕ визначений, а не «немає проти»')
    print('✓ 📐 старіші файли не ламають детектор')


def test_the_raw_log_carries_both_sensors_and_the_decider():
    """Без чисел 📐 у рядку неможливо буде перевірити, чий сенсор вирішив."""
    body = _fn_src(_FF_SRC, '_mm_corr_log_write')
    for f in ('ob_pct', 'ob_n', 'ob_against', 'breadth_pct', 'breadth_src'):
        _check(f"'{f}'" in body, f'писач логу не кладе {f}')
    cols = set(_model_columns('MmCorrectionLog'))
    for f in ('ob_pct', 'ob_n', 'ob_against', 'breadth_pct', 'breadth_src'):
        _check(f in cols, f'у моделі БД немає колонки {f}')
    dbo = open(os.path.join(_HERE, 'storage', 'db_operations.py'),
               encoding='utf-8').read()
    for f in ('ob_pct', 'breadth_src'):
        _check(f"'{f}'" in dbo, f'білий список шару БД відріже {f}')
    models = open(os.path.join(_HERE, 'storage', 'db_models.py'),
                  encoding='utf-8').read()
    _check("('ob_pct', 'FLOAT')" in models and "('breadth_src', 'VARCHAR(8)')" in models,
           'наявна таблиця потребує ALTER для нових колонок')
    print('✓ 📐 сирий лог несе обидва сенсори і те, чиє число вирішило')


def test_ui_exposes_the_second_source():
    _check('id="ff-mm-corr-ob"' in _HTML, 'немає тумблера 📐')
    _check('mm_corr_ob_on' in _HTML, 'ключ не зберігається зі сторінки')
    _check(mc.DEFAULTS['mm_corr_ob_on'] is True,
           'користувач просив саме додати джерело — дефолт УВІМК')
    print('✓ 📐 друге джерело видно і керовано з UI')



# ═══ 10. 🧭 ОБИДВА ПОРОГИ — «СКІЛЬКИ МОНЕТ» (вимога 22.09) ═════════════════
# Дослівно: «мені потрібно саме розуміння початку корекції це — наприклад 60%,
# то це має бути 60% монет, які втратили напрямок, і вихід із корекції
# наприклад 70% — то це мається на увазі 70% монет повернулися за напрямком».
#
# ⚠️ РОЗДІЛ ПЕРЕПИСАНО, а не «полагоджено»: раніше поріг виходу міряв тих, хто
# ЩЕ ПРОТИ, тож 65 означало «за ≥ 35%» — удвічі мʼякше, ніж читалось.


def _js_fn(name):
    """Витягти ОДНУ функцію зі сторінки (шукаємо по `function <name>(`)."""
    i = _HTML.index('function ' + name + '(')
    depth, j, started = 0, i, False
    while j < len(_HTML):
        if _HTML[j] == '{':
            depth += 1
            started = True
        elif _HTML[j] == '}':
            depth -= 1
            if started and depth == 0:
                return _HTML[i:j + 1]
        j += 1
    raise AssertionError('не знайшов кінець функції ' + name)


def _br(against_pct):
    """Ширина у формі, яку віддає `breadth_of`."""
    return {'ok': True, 'pct': against_pct,
            'for_pct': round(100.0 - against_pct, 1), 'src': '📦'}


def test_the_exit_threshold_counts_the_coins_that_came_back():
    """70 = «повернулось 70% монет», а НЕ «проти менше 70%»."""
    _check(mc.breadth_exit_ok(_br(30.0), 70.0) is True,
           'проти 30% → за 70% → рівно поріг, кінець дозволено')
    _check(mc.breadth_exit_ok(_br(31.0), 70.0) is False,
           'за 69% — ще замало')
    # ⚠️ Головне: СТАРЕ читання дало б протилежну відповідь.
    _check(mc.breadth_exit_ok(_br(60.0), 70.0) is False,
           'за старим правилом «проти < 70» це був би КІНЕЦЬ — тепер ні')
    _check(mc.breadth_exit_ok(None, 70.0) is None
           and mc.breadth_exit_ok({'ok': False}, 70.0) is None,
           'немає даних → None, а не вигаданий кінець')
    print('✓ 🧭 поріг кінця рахує тих, хто ПОВЕРНУВСЯ')


def test_start_and_exit_read_on_the_same_scale():
    """Обидва пороги — «скільки монет», просто з різних боків.

    ⚠️ Знаменник ОДИН, тож `за + проти = 100`: два числа не можуть
    суперечити одне одному."""
    b = mc.breadth_of({'ok': True, 'pct': 82.0}, {'ok': True, 'pct': 74.0})
    _check(b['pct'] == 82.0, f'проти — МАКСИМУМ із двох сенсорів: {b}')
    _check(b['for_pct'] == 18.0, f'за = 100 − проти: {b}')
    _check(round(b['pct'] + b['for_pct'], 1) == 100.0, 'сума мусить бути 100')
    print('✓ 🧭 «за» і «проти» — одна шкала з двох боків')


def test_votes_no_longer_hold_the_correction_once_breadth_recovered():
    """🐞 КОРІНЬ СКАРГИ «немає повідомлень про вихід із корекції».

    Ширина рахувалась ДВІЧІ — і у вето, і серед голосів `lit_hold`. Тож навіть
    коли вето знімалось, та сама ширина ще «світилась» голосом і тримала
    корекцію. Відтворено: при порозі старту 60 послаблений — 50, а в реальному
    логу ширина нижче 53% не опускалась ЖОДНОГО разу."""
    # За ≥45% = проти ≤55%. Беремо РІВНО профіль із реального логу: ширина
    # завмерла на 53% проти (47% за) — кінець уже дозволено, але 53 ≥ 50
    # (послаблений поріг старту), тож голос ширини ще світиться в `lit_hold`.
    cfg = _cfg(mm_corr_vob_exit_pct=45.0)
    trends = {f'C{i}': ('SHORT' if i < 53 else 'LONG') for i in range(100)}
    snap = _snap(**{f'C{i}': ('LONG', 50, 'down') for i in range(100)})
    r = mc.evaluate(snap, trends, 'LONG', 90.0, 90.0, cfg)
    _check(r['breadth_ok'] is True, f'ширина відновилась: {r["breadth_pct"]}%')
    _check(r['stay'] is False,
           f'голоси НЕ мають тримати корекцію після відновлення: '
           f'stay={r["stay"]} lit_hold={r["lit_hold"]}/{r["need"]}')
    # ⚠️ `_fn_src` віддає AST-unparse, і дужки він ЗНІМАЄ: шукаємо суть
    # (`b_ok is not None` → інакше голоси), а не дослівний рядок.
    src = _fn_src(_MC_SRC, 'evaluate')
    _check('b_ok is not None' in src and 'lit_hold >= need' in src,
           'голоси мусять лишитись ФОЛБЕКОМ, коли ширини немає')
    print('✓ 🧭 після відновлення ширини рішення за нею, а не за голосами')


def test_the_veto_still_holds_while_too_few_came_back():
    """Зворотний бік: поки повернулось мало — кінець ЗАБОРОНЕНО."""
    cfg = _cfg(mm_corr_vob_exit_pct=70.0)
    trends = {f'C{i}': ('SHORT' if i < 90 else 'LONG') for i in range(100)}
    snap = _snap(**{f'C{i}': ('LONG', 50, 'up') for i in range(100)})
    r = mc.evaluate(snap, trends, 'LONG', 50.0, 51.0, cfg)
    _check(r['breadth_ok'] is False and r['stay'] is True,
           f'при 10% повернутих кінець неможливий: {r["why"]}')
    _check('10.0%' in r['why'] and '70' in r['why'],
           f'причина мусить називати ОБИДВА числа: {r["why"]}')
    print('✓ 🧭 поки повернулось мало — вето тримає')


def test_the_new_default_is_seventy_percent_back():
    """«вихід із корекції наприклад 70%» — дослівно."""
    _check(float(mc.DEFAULTS['mm_corr_vob_exit_pct']) == 70.0,
           mc.DEFAULTS['mm_corr_vob_exit_pct'])
    _check(float(mc.DEFAULTS['mm_corr_vob_pct']) == 60.0,
           'поріг старту лишається 60% «втратили напрямок»')
    _check(_HTML.count("s.mm_corr_vob_exit_pct : 70") == 2,
           'фолбеки на сторінці мусять бути 70')
    _check("_mmcNum('ff-mm-corr-vobexit', 70)" in _HTML, 'save шле старий фолбек')
    _check('id="ff-mm-corr-vobexit" min="0" max="100" step="5" value="70"' in _HTML,
           'value= у розмітці не оновлено')
    print('✓ 🧭 дефолт 60 «втратили» / 70 «повернулись»')


def test_the_migration_carries_both_old_defaults_to_the_new_meaning():
    """⚠️ Сенс числа змінився, тож ОБИДВА старі дефолти (50 і 65) ведемо на 70.

    Дописати ключ у стару позначку було б мертвим кодом — вона вже `True`."""
    _flags = [f for f, _ in FF.MM_CORR_TUNE_WAVES]
    _wave = dict(FF.MM_CORR_TUNE_WAVES)[_flags[-1]]
    _olds, _new = _wave['mm_corr_vob_exit_pct']
    _check(_new == 70.0 and set(_olds) == {50.0, 65.0}, _wave)
    for _old in (50.0, 65.0):
        ff = _mk_ff({'mm_corr_tuned_v1': True, 'mm_corr_exit65_v2': True,
                     'mm_corr_vob_exit_pct': _old})
        ff._migrate_settings()
        _got = ff._db.get_setting('fuel_filter_settings', {})
        _check(float(_got['mm_corr_vob_exit_pct']) == 70.0,
               f'{_old} мусить перейти на 70: {_got}')
    # ручний вибір НЕ чіпаємо (правило 21.09)
    ff = _mk_ff({'mm_corr_tuned_v1': True, 'mm_corr_exit65_v2': True,
                 'mm_corr_vob_exit_pct': 58.0})
    ff._migrate_settings()
    _check(float(ff._db.get_setting('fuel_filter_settings', {})
                 ['mm_corr_vob_exit_pct']) == 58.0, 'ручне значення чіпати не можна')
    print('✓ 🧭 міграція веде обидва старі дефолти на нову шкалу')


def test_the_raw_log_carries_the_number_that_decides_the_end():
    """Поріг `vob_exit` міряє «за», тож у рядку мусить стояти САМЕ «за».

    ⚠️ Читаємо файли ТЕКСТОМ (як і решта замків цього файлу): імпортувати
    `storage.db_operations` тут не можна — він тягне конфіг і зʼєднання."""
    src = _fn_src(_FF_SRC, '_mm_corr_log_write')
    _check("'breadth_for'" in src, 'писач мусить класти частку «за»')
    _check('breadth_for' in _model_columns('MmCorrectionLog'),
           'у моделі БД немає колонки breadth_for')
    dbo = open(os.path.join(_HERE, 'storage', 'db_operations.py'),
               encoding='utf-8').read()
    _i = dbo.index('_MM_CORR_FIELDS = (')
    wl = dbo[_i:dbo.index(')', _i)]
    _check("'breadth_for'" in wl, 'білий список шару БД відріже breadth_for')
    _check(wl.count("'breadth_pct'") == 1,
           f'дублів у білому списку бути не має: {wl}')
    models = open(os.path.join(_HERE, 'storage', 'db_models.py'),
                  encoding='utf-8').read()
    _check("('breadth_for', 'FLOAT')" in models,
           'ідемпотентний ALTER TABLE не додасть колонку на старій БД')
    _check("'breadth_for'" in open(
        os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read(),
        'CSV-експорт мусить нести число, що вирішує кінець')
    print('✓ 🧭 сирий лог несе число, що вирішує кінець')


def _run_exit_sync(entry, exit_val):
    """Прогнати СПРАВЖНІЙ JS сторінки під node на крихітному фейк-DOM."""
    import json
    import subprocess
    js = _js_fn('_mmcExitSync') + (
        "\nconst _els = {"
        f"'ff-mm-corr-vob': {{value: '{entry}'}},"
        f"'ff-mm-corr-vobexit': {{value: '{exit_val}'}},"
        "'mm-corr-exit-note': {style: {display: 'none'}, textContent: ''}};\n"
        "const document = { getElementById: (id) => _els[id] || null };\n"
        "_mmcExitSync();\n"
        "console.log(JSON.stringify({exit: _els['ff-mm-corr-vobexit'].value,"
        " shown: _els['mm-corr-exit-note'].style.display !== 'none',"
        " text: _els['mm-corr-exit-note'].textContent,"
        " color: _els['mm-corr-exit-note'].style.color}));\n")
    r = subprocess.run(['node', '-e', js], capture_output=True, text=True,
                       timeout=30)
    _check(r.returncode == 0, 'JS упав: ' + r.stderr[:400])
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_page_translates_the_exit_threshold_into_the_other_side():
    """60/70 — коректна пара, і сторінка мусить це ПЕРЕКЛАСТИ.

    Без перекладу «70» поруч із «60» читається як мʼякше, хоча воно суворіше."""
    out = _run_exit_sync(60, 70)
    _check(out['exit'] == '70', f'число чіпати не можна: {out}')
    _check(out['shown'], 'переклад мусить бути видимий')
    for _must in ('60', '70', '30'):
        _check(_must in out['text'], f'немає «{_must}»: {out["text"]}')
    _check('93c5fd' in out['color'].replace('#', '') or 'rgb' in out['color'],
           f'коректна пара — довідка, а не попередження: {out["color"]}')
    print('✓ 🧭 сторінка перекладає «70% за» у «≤30% проти»')


def test_the_page_warns_when_the_two_bands_overlap():
    """⚠️ Вивернута пара: кінець дозволений там, де старт ЩЕ діє."""
    out = _run_exit_sync(60, 30)          # за ≥30% = проти ≤70% ≥ старт 60%
    _check(out['shown'] and '⚠️' in out['text'],
           f'перекриття смуг мусить бути назване: {out}')
    _check('fbbf24' in out['color'].replace('#', '') or 'rgb' in out['color'],
           f'це попередження, а не довідка: {out["color"]}')
    _check(out['exit'] == '30', 'число все одно НЕ зводимо')
    print('✓ 🧭 перекриття смуг названо, але число не зводиться')




# ═══ 11. ⏸ РУЧНА ПАУЗА ВЕРДИКТУ + ДРУГИЙ РЯДОК (вимога 23.09) ════════════
# Дослівно: «Додай кнопку ручного зупинення корекції, тобто все залишається
# рахуватись відображатись як і зазвичай, а це лише дасть можливість раніше
# дати можливість відслідковувати VOB в таблиці "Сканер ліквідності". Кнопка
# вдавлена — то працюємо ніби як немає стану "Корекція". Стан кнопки
# змінюється або ще одним натиском або автоматично коли Корекція дійсно
# закінчилась.»
def _in_correction(**over):
    """Двигун із ЖИВОЮ підтвердженою корекцією проти банера LONG."""
    coins = {f'C{i}USDT': ('LONG', 60, 'down') for i in range(8)}
    ff = _mk(trends={s: 'SHORT' for s in coins}, **over)
    ff._persist_state = lambda: None
    c = _tick(ff, _snap(**coins))
    _check(c['state'] == 'on' and c['blocking'], f'потрібна жива корекція: {c}')
    return ff, coins


def test_pause_lifts_the_gate_but_leaves_the_verdict_alone():
    """ГОЛОВНЕ: пауза знімає ВОРОТА, а не детектор. Стан, ознаки й таймер
    мусять лишитись такими самими — інакше «працюємо ніби немає корекції»
    перетворилось би на «корекцію не рахуємо»."""
    ff, coins = _in_correction()
    before = dict(ff._mm_corr)
    r = ff.set_correction_override()
    _check(r['ok'] and r['override'], f'кнопка мусить вдавитись: {r}')
    _check(ff.correction_blocks_open()[0] is False,
           'при вдавленій кнопці ворота мусять бути зняті')
    c = dict(ff._mm_corr)
    _check(c['override'] is True and c['blocking'] is False, f'{c}')
    for k in ('state', 'since', 'lit', 'need', 'layers', 'exit_pct'):
        _check(c.get(k) == before.get(k), f'«{k}» змінилось паузою: {c.get(k)}')
    # І далі рахується: наступний такт не знімає паузу і не ламає вердикт.
    c2 = _tick(ff, _snap(**coins))
    _check(c2['state'] == 'on' and c2['override'] and not c2['blocking'],
           f'такт мусить зберегти і корекцію, і паузу: {c2}')
    print('✓ ⏸ пауза знімає лише ворота — вердикт рахується як зазвичай')


def test_second_press_returns_the_gate():
    ff, coins = _in_correction()
    ff.set_correction_override()
    r = ff.set_correction_override()
    _check(r['ok'] and r['override'] is False, f'другий натиск відпускає: {r}')
    _check(_tick(ff, _snap(**coins))['blocking'],
           'ворота мусять повернутись')
    _check(ff.correction_blocks_open()[0] is True, 'і блокувати відкриття')
    print('✓ ⏸ другий натиск повертає ворота')


def test_pause_releases_itself_when_the_correction_really_ends():
    """«Автоматично, коли Корекція дійсно закінчилась» — без цього наступна,
    ВЖЕ ІНША корекція мовчки не блокувала б відкриття."""
    ff, coins = _in_correction()
    ff.set_correction_override()
    # Ринок повернувся за банером → корекція завершується.
    up = {s: ('LONG', 60, 'up') for s in coins}
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m',
                                 'trends': {s: 'LONG' for s in coins}}
    c = _tick(ff, _snap(**up), now=NOW + 600)
    _check(c['state'] != 'on', f'корекція мусила завершитись: {c}')
    _check(not c.get('override'), f'пауза мусить зніматись САМА: {c}')
    _check(float(ff._mm_corr_override or 0) == 0.0, 'позначку теж знято')
    print('✓ ⏸ пауза знімається сама, коли корекція завершилась')


def test_pause_belongs_to_the_episode_not_to_a_flag():
    """Пауза привʼязана до `since` ЕПІЗОДУ: НОВА корекція блокує знову, навіть
    якщо кнопку не відпускали руками."""
    ff, coins = _in_correction()
    ff.set_correction_override()
    _check(ff.correction_blocks_open()[0] is False, 'пауза діє')
    # Той самий стан, але інший епізод (машину станів скинуто).
    ff._mm_corr_st = {}
    ff._mm_lever_hist = []
    c = _tick(ff, _snap(**coins), now=NOW + 3600)
    _check(c['state'] == 'on' and float(c['since']) > NOW,
           f'це вже НОВА корекція: {c}')
    _check(c['blocking'] and not c.get('override'),
           f'нова корекція мусить блокувати: {c}')
    print('✓ ⏸ пауза не переноситься на НАСТУПНУ корекцію')


def test_pause_cannot_be_armed_without_a_correction():
    """Вдавити кнопку над станом, якого немає, означало б, що наступна
    корекція стартує вже призупиненою."""
    coins = {f'C{i}USDT': ('LONG', 60, 'up') for i in range(8)}
    ff = _mk(trends={s: 'LONG' for s in coins})
    ff._persist_state = lambda: None
    _tick(ff, _snap(**coins))
    r = ff.set_correction_override(True)
    _check(not r['ok'] and 'нема' in r['reason'],
           f'мусить бути ЧЕСНА відмова з причиною: {r}')
    _check(float(ff._mm_corr_override or 0) == 0.0, 'нічого не запамʼятали')
    print('✓ ⏸ без корекції паузу поставити не можна — і сказано чому')


def test_pause_survives_a_restart():
    """`botupdate` роблять часто; без персисту кнопка «відпускалась» сама, а
    ворота мовчки вмикались назад посеред того самого епізоду."""
    ff, _ = _in_correction()
    ff.set_correction_override()
    _check('mm_corr_override' in _fn_src(_FF_SRC, '_persist_state'),
           'паузу не зберігають у блобі стану')
    _check('mm_corr_override' in _fn_src(_FF_SRC, '_load_state'),
           'паузу не відновлюють на старті')
    print('✓ ⏸ пауза переживає рестарт')


def test_the_raw_log_says_why_the_gate_was_off():
    """Семпл із `blocking=0` посеред живої корекції інакше читався б як дефект
    детектора. Окремої КОЛОНКИ не заводимо — для цього є `note`."""
    _DBROWS.clear()
    ff, coins = _in_correction()
    ff.set_correction_override()
    ff._mm_corr_log_at = 0.0
    ff._mm_corr_log_write('sample')
    rows = _rows('sample')
    _check(rows and rows[-1].get('blocking') is False,
           f'ворота зняті — це має бути в рядку: {rows}')
    _check('пауза' in (rows[-1].get('note') or ''),
           f'причина мусить бути названа: {rows[-1]}')
    print('✓ 🧾 у сирому лозі видно, що ворота зняла ручна пауза')


def test_ui_moved_the_summary_to_a_second_line():
    """Вимога 1 (23.09): «🧭 кінець коли ЗА… · ознак 1/2, 🚫 відкриття
    зупинено — перенеси на другий рядок»."""
    i = _HTML.index('id="mm-corr-row"')
    j = _HTML.index('id="mm-limited-hint"')
    row = _HTML[i:j]
    _check('id="mm-corr-line2"' in row, 'другого рядка немає')
    line2 = row[row.index('id="mm-corr-line2"'):]
    for el in ('mm-corr-exit', 'mm-corr-count', 'mm-corr-block',
               'mm-corr-override'):
        _check(f'id="{el}"' in line2, f'{el} мусить стояти в ДРУГОМУ рядку')
    _check('flex-basis:100%' in line2.split('>')[0],
           'перенос ЦІЛИМ блоком тримає flex-basis, а не надія на ширину')
    # Перший рядок лишає РІВНО стан/таймер/розклад шарів.
    head = row[:row.index('id="mm-corr-line2"')]
    for el in ('mm-corr-state', 'mm-corr-timer', 'mm-corr-layers'):
        _check(f'id="{el}"' in head, f'{el} мусить лишитись у ПЕРШОМУ рядку')
    print('✓ 🖥 підсумок і ворота переїхали на другий рядок')


def test_js_draws_the_pressed_button_and_its_own_state():
    """Стан «пауза» мусить відрізнятись від «тумблер вимкнено» — інакше
    причина знову стає здогадкою."""
    out = _run_js(r'''
const now = Math.floor(Date.now()/1000);
const L = [{key:'vob',icon:'📦',name:'VOB проти',pct:72,need:60,ok:true,lit:true,n:20,note:'',role:'both'},
           {key:'price',icon:'💹',name:'Ціна проти',pct:80,need:60,ok:true,lit:true,n:18,note:'',role:'start'}];
const g = id => document.getElementById(id);
const seen = {};
mmRenderCorr({state:'on', since: now-900, lit:2, need:2, layers:L, enabled:true,
              blocking:true, exit_pct:70, breadth_for_pct:4.2, breadth_exit_on:true});
seen.gate = {txt:g('mm-corr-block').textContent, btn:g('mm-corr-override').textContent,
             show:g('mm-corr-override').style.display};
mmRenderCorr({state:'on', since: now-900, lit:2, need:2, layers:L, enabled:true,
              blocking:false, override:true, exit_pct:70, breadth_for_pct:4.2,
              breadth_exit_on:true});
seen.paused = {txt:g('mm-corr-block').textContent, btn:g('mm-corr-override').textContent,
               exit:g('mm-corr-exit').innerHTML, cnt:g('mm-corr-count').innerHTML,
               lay:g('mm-corr-layers').innerHTML};
mmRenderCorr({state:'trend', lit:0, need:2, layers:[], enabled:true, blocking:false});
seen.trend = {btn:g('mm-corr-override').style.display};
console.log(JSON.stringify(seen));
''')
    import json
    d = json.loads(out)
    _check(d['gate']['btn'] == '⏸ Пауза' and d['gate']['show'] != 'none',
           f'кнопка мусить бути видна в корекції: {d["gate"]}')
    _check('зупинено' in d['gate']['txt'], f'ворота: {d["gate"]}')
    _check('ВДАВЛЕНА' in d['paused']['btn'], f'вдавлений стан: {d["paused"]}')
    _check('пауза вручну' in d['paused']['txt'],
           f'стан воріт мусить називати ПРИЧИНУ: {d["paused"]}')
    _check('кінець коли ЗА' in d['paused']['exit'], f'{d["paused"]}')
    _check('ознак' in d['paused']['cnt'], f'{d["paused"]}')
    # ⚠️ Шукаємо САМЕ маркери підсумку, а не слова: «кінець» трапляється ще й
    # у ПІДКАЗЦІ шару ширини («вирішує і початок, і кінець»).
    _check('🧭 кінець коли' not in d['paused']['lay']
           and 'ознак <b' not in d['paused']['lay'],
           f'у першому рядку лишився підсумок: {d["paused"]["lay"]}')
    _check(d['trend']['btn'] == 'none',
           'без корекції кнопки бути не повинно')
    print('✓ 🖥 кнопка паузи: вдавлений стан і власний підпис воріт')


def test_the_button_goes_through_its_own_route():
    _check('/api/fuel-filter/mm-corr/override' in _HTML, 'сторінка не кличе маршрут')
    _check("@app.route('/api/fuel-filter/mm-corr/override'" in _FLASK_SRC,
           'маршруту немає у flask_app (урок «URL фронта ≠ маршрут»)')
    fn = _fn_src(_FLASK_SRC, 'api_fuel_filter_mm_corr_override')
    _check('set_correction_override' in fn,
           'маршрут мусить лише делегувати — правило живе в двигуні')
    for bad in ('_mm_corr_override =', 'state ==', 'since'):
        _check(bad not in fn, f'логіка паузи протекла в маршрут: {bad}')
    _check('if (!r.ok)' in _HTML.split('async function mmCorrOverride')[1][:1200],
           'HTTP-статус мусить перевірятись окремо (урок submitManualTp1)')
    print('✓ 🚪 кнопка ходить своїм маршрутом, логіка лишилась у двигуні')


if __name__ == '__main__':
    _fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
