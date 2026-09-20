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
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
               encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
             encoding='utf-8').read()

NOW = 10_000.0


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
    ff._engine_skip = {}
    s = {'mm_bias_confirm_sec': 0, 'mm_corr_confirm_sec': 0,
         'mm_corr_min_layers': 2, 'mm_corr_vob_pct': 60.0,
         'mm_corr_price_pct': 60.0, 'mm_corr_lever_drop': 15.0,
         'mm_corr_enabled': True, 'mm_corr_block_open': True,
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
    _check(mc.DEFAULTS['mm_corr_confirm_sec'] == 120,
           'підтвердження — те саме вікно, що в антиспамі банера')
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
  if (!_els[id]) _els[id] = {id, style:{}, innerHTML:'', textContent:'', title:''};
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


def test_js_timer_puts_the_day_in_the_same_tile():
    out = _run_js(r'''console.log(flipTimerHTML(2*86400 + 9*3600 + 57*60 + 19));''')
    _check('fd fdd' in out, f'доба не в плитці: {out}')
    _check(re.sub(r'<[^>]*>', '', out) == '2д09:57:19',
           f'таймер читається не так: {re.sub(r"<[^>]*>", "", out)}')
    _check('fday' not in out, 'старий стиль дня мусив зникнути')
    print('✓ ⏱ JS: «2д 09:57:19» — доба тією самою плиткою')


if __name__ == '__main__':
    _fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in _fns:
        fn()
    print(f'\n{len(_fns)}/{len(_fns)} passed')
