"""🆕 «НОВИЙ OB» ЯК ОКРЕМИЙ ТИП СИГНАЛУ: другий TF (ЗБІГ) + дедуп 1-на-тренд.

**Вимога користувача (10.09), дослівно:**
 1. «Додай до налаштувань вибір другого таймфрейму для Нового OB… Другий
    таймфрейм зроби щоб його можна було вимикати при нагоді (за замовчуванням
    другий таймфрейм 4Н). Тобто, якщо вибрано другий таймфрейм, то Новий OB має
    тепер спрацювати, коли зійдуться обидва таймфрейми (1Н і 4Н тільки тоді
    вважається, що Новий OB зʼявився), інакше якщо вибрано основний таймфрейм,
    а це 1Н, то мають іти всі Нові OB 1Н.»
 2. «Додай також алгоритм Deduplicate Signals (1 per trend) для Новий OB. Тобто
    якщо вибрано лише основний таймфрейм 1Н то при такому варіанті SHORT має
    право перебити лише протилежний Новий OB LONG. А якщо задіяно два
    таймфрейми наприклад 1Н+4Н то 1Н+4Н SHORT має право перебити лише 1Н+4Н
    LONG.»
 3. «Організуй все це як ще один вид сигналу. З можливістю вмикати і вимикати.»

Що стережуть ці тести:
  • ЗБІГ: обидва TF мусять дати ОДИН напрямок, інакше події НЕМА;
  • «немає блоку на старшому TF» ≠ «узгоджено» (невизначеність не пропускаємо);
  • ⚠️ розбіг НЕ КОВТАЄ блок: збіг може настати за кілька хвилин, тож блок
    лишається неопрацьованим, поки він у вікні свіжості;
  • ДЕДУП прив'язаний до КОМБІНАЦІЇ TF (1H vs 1H+4H), переживає рестарт;
  • придушені стани (`wait_htf`/`dedup`) у 🧾 Лог НЕ пишуться — це СТАНИ
    (урок флуду VOB), вони живуть у `ob_alert_diag`;
  • торговий сигнал іде ЛИШЕ через спільні ворота `_signal_allowed`;
  • ворота ВХОДУ лишаються недоторканими.
"""
import ast
import importlib
import importlib.util
import os
import sys
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))

# Порожній пакет `detection` з правильним `__path__`: справжній `__init__.py`
# тягне `sleeper_scanner → core → pybit`, тобто півпроєкту.
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg

_spec = importlib.util.spec_from_file_location(
    'smc_scanner_obsig_test', os.path.join(_HERE, 'detection', 'smc_scanner.py'))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
S = _m.SMCScanner

oba = importlib.import_module('detection.ob_alert')
labels = importlib.import_module('detection.signal_labels')

HOUR = 3600
_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'),
            encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
             encoding='utf-8').read()


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


# ── оточення ──────────────────────────────────────────────────────────────
_LOGGED = []
_OPENED = []


def _install_log():
    _LOGGED.clear()
    mod = types.ModuleType('detection.activity_log')

    def _log(symbol, event, detail='', side=None, source='', extra=None):
        _LOGGED.append({'symbol': symbol, 'event': event, 'detail': detail,
                        'side': side, 'source': source, 'extra': extra or {}})
    mod.log_activity = _log
    sys.modules['detection.activity_log'] = mod
    _pkg.activity_log = mod


def _install_tm():
    """Фейковий `detection.trade_manager` — збираємо відкриття."""
    _OPENED.clear()
    mod = types.ModuleType('detection.trade_manager')
    tm = types.SimpleNamespace(
        on_signal=lambda **kw: _OPENED.append(kw) or 'queued')
    mod.get_trade_manager = lambda: tm
    sys.modules['detection.trade_manager'] = mod
    _pkg.trade_manager = mod


def _ob(bias='BULLISH', bar_time=None, created_at_t=None, tag='CHoCH',
        now=None):
    """Блок у вигляді `detect_last_order_block`, СВІЖИЙ за замовчуванням
    (зʼявився 30с тому на 1h), щоб `outcome` давав саме 'new'."""
    n = now or time.time()
    bt = bar_time if bar_time is not None else int((n - 2 * HOUR) * 1000)
    ct = created_at_t if created_at_t is not None else int((n - HOUR - 30) * 1000)
    return {'bias': bias, 'bar_high': 10.0, 'bar_low': 9.0, 'bar_time': bt,
            'bar_idx': 5, 'created_at_idx': 9, 'created_at_t': ct,
            'created_by_tag': tag}


def _mk(htf_on=True, htf='4h', dedup=True, as_sig=False, ob4_bias='BULLISH',
        allowed=True, tf='1h'):
    """Мінімальний сканер: лише те, чого торкається 🆕 Новий OB."""
    ns = types.SimpleNamespace()
    ns._settings = {
        'ob_filter_timeframe': tf, 'ob_alert_enabled': True,
        'ob_alert_htf': htf, 'ob_alert_htf_enabled': htf_on,
        'ob_alert_dedup': dedup, 'ob_alert_signal': as_sig,
        'ob_alert_max_lag_sec': 0, 'swing_size': 50, 'internal_size': 5,
    }
    ns._ob_alert_seen = {}
    ns._ob_alert_fired = {}
    ns._ob_alert_diag = {}
    ns._ob_htf_cache = {}
    ns._errors = 0
    ns._persisted = []
    ns._persist_ob_alert_state = lambda force=False: ns._persisted.append(force)
    ns._htf_calls = []
    ns._htf_bias = ob4_bias

    def _htf_stub(symbol, md, _tf, ob_tf, ob_same):
        ns._htf_calls.append((symbol, _tf))
        if ns._htf_bias is None:
            return None, 'мало історії 4h'
        return {'bias': ns._htf_bias}, ''
    ns._ob_on_htf = _htf_stub
    ns._gate_calls = []

    def _gate(symbol, side, at_intake=False):
        ns._gate_calls.append((symbol, side, at_intake))
        return (allowed, 'OB-фільтр заблокував: 1H-блок BEARISH ПРОТИ сигналу LONG',
                'OB(1h):✓ · PD:✓')
    ns._signal_allowed = _gate
    ns._get_live_price = lambda s: 78540.0
    for name in ('_ob_alert_tick', '_ob_alert_signal'):
        setattr(ns, name, getattr(S, name).__get__(ns))
    return ns


def _tick(ns, sym='BTCUSDT', ob=None, price=78539.9, now=None):
    n = now or time.time()
    return ns._ob_alert_tick(sym, None, ns._settings['ob_filter_timeframe'],
                             ob if ob is not None else _ob(now=n),
                             [{'t': int(n * 1000), 'p': price}])


# ═══════════ 1. ЧИСТІ ФУНКЦІЇ: ЗБІГ І КОМБІНАЦІЯ TF ══════════════════════
def test_combo_label_names_the_active_combination():
    """Підпис комбінації — і він же «лінія тренду» для дедупу."""
    _check(oba.combo_label('1h', '4h', False) == '1H',
           'другий TF вимкнено → комбінація з ОДНОГО TF')
    _check(oba.combo_label('1h', '4h', True) == '1H+4H', 'увімкнено → обидва')
    _check(oba.combo_label('1h', '', True) == '1H',
           'увімкнено, але TF не заданий → нічого не вигадуємо')
    _check(oba.combo_label('15m', '1h', True) == '15M+1H', 'будь-яка пара')
    print('✓ combo_label: 1H / 1H+4H')


def test_converge_truth_table():
    """Другий TF ВИМКНЕНО → вирішує один основний (ідуть ВСІ нові OB).
    УВІМКНЕНО → потрібен ЗБІГ напрямків."""
    _check(oba.converge('LONG', None, False)[0] is True,
           'вимкнено → старший TF узагалі не бере участі')
    _check(oba.converge('SHORT', 'LONG', False)[0] is True, 'те саме і для SHORT')
    _check(oba.converge('LONG', 'LONG', True)[0] is True, 'збіг → подія є')
    ok, note = oba.converge('LONG', 'SHORT', True)
    _check(not ok and 'розбіг' in note, f'розбіг мусить різати з причиною: {note}')
    _check(oba.converge(None, 'LONG', True)[0] is False,
           'немає напрямку основного TF → немає події')
    print('✓ converge: збіг обох TF = «новий OB зʼявився»')


def test_missing_htf_block_is_not_agreement():
    """⚠️ «Немає OB на старшому TF» ≠ «узгоджено». Якби відсутність даних
    пропускала сигнал, увімкнений другий TF мовчки НЕ робив би нічого там, де
    4H-блок ще не порахувався — налаштування виглядало б робочим, а не діяло."""
    ok, note = oba.converge('LONG', None, True)
    _check(not ok, 'без блоку старшого TF збігу немає')
    _check('немає OB на старшому TF' in note, f'причина мусить бути названа: {note}')
    print('✓ немає блоку на другому TF → збігу НЕМА (невизначеність не пускаємо)')


# ═══════════ 2. ЧИСТИЙ ДЕДУП «1 НА ТРЕНД» ════════════════════════════════
def test_same_side_in_the_same_combo_is_suppressed():
    """Правило користувача: перебити може ЛИШЕ протилежний напрямок."""
    prev = {'side': 'SHORT', 'combo': '1H'}
    ok, note = oba.dedup_allows(prev, 'SHORT', '1H')
    _check(not ok and 'дедуп' in note, f'той самий бік мусить глушитись: {note}')
    ok2, note2 = oba.dedup_allows(prev, 'LONG', '1H')
    _check(ok2 and 'перебито' in note2, f'протилежний мусить проходити: {note2}')
    print('✓ 1H: SHORT перебиває лише протилежний LONG')


def test_the_combination_keys_the_trend():
    """«1H+4H SHORT має право перебити лише 1H+4H LONG» — тобто дедуп міряє
    тренд САМЕ в межах активної комбінації TF."""
    prev = {'side': 'SHORT', 'combo': '1H+4H'}
    _check(oba.dedup_allows(prev, 'SHORT', '1H+4H')[0] is False,
           '1H+4H SHORT після 1H+4H SHORT — глушиться')
    _check(oba.dedup_allows(prev, 'LONG', '1H+4H')[0] is True,
           '1H+4H LONG перебиває')
    # ⚠️ Інша комбінація = інший за визначенням сигнал: позначка від часів
    # «лише 1H» не має глушити перший же 1H+4H-сигнал після вмикання збігу.
    ok, note = oba.dedup_allows({'side': 'SHORT', 'combo': '1H'},
                                'SHORT', '1H+4H')
    _check(ok and 'інша комбінація' in note, f'перемикання TF: {note}')
    print('✓ дедуп прив\'язаний до комбінації TF (1H ≠ 1H+4H)')


def test_no_stamp_means_allowed():
    """Порожня/сміттєва позначка нічого не глушить (відмову не вигадуємо)."""
    for prev in (None, {}, 'junk', {'side': ''}):
        _check(oba.dedup_allows(prev, 'LONG', '1H')[0] is True, f'{prev!r}')
    print('✓ немає позначки → сигнал проходить')


# ═══════════ 3. ТІК: ЗБІГ ДРУГОГО TF ══════════════════════════════════════
def test_second_tf_off_fires_every_new_ob_and_never_asks_the_exchange():
    """«Якщо вибрано основний таймфрейм, а це 1Н, то мають іти всі Нові OB 1Н».
    ⚠️ І старший TF тоді НЕ питаємо взагалі — ні запиту до біржі, ні рядка 4H."""
    _install_log(); _install_tm()
    ns = _mk(htf_on=False)
    _check(_tick(ns) == 'new', 'вимкнений другий TF → подія є')
    _check(not ns._htf_calls, f'зайвий запит старшого TF: {ns._htf_calls}')
    _check(len(_LOGGED) == 1 and _LOGGED[0]['event'] == 'ob_new', _LOGGED)
    _check('OB 4H' not in _LOGGED[0]['detail'],
           f'4H у рядку не має бути: {_LOGGED[0]["detail"]}')
    print('✓ другий TF OFF → усі нові OB, без зайвих запитів')


def test_agreement_of_both_tfs_is_what_makes_it_an_event():
    _install_log(); _install_tm()
    ns = _mk(htf_on=True, ob4_bias='BULLISH')
    _check(_tick(ns, ob=_ob(bias='BULLISH')) == 'new', 'обидва LONG → подія')
    d = _LOGGED[0]['detail']
    _check('LONG OB 4H' in d and 'збіг обох TF' in d, f'збіг мусить бути видний: {d}')
    _check(_LOGGED[0]['extra']['parts'].get('combo') == '1H+4H',
           f'комбінація в parts: {_LOGGED[0]["extra"]["parts"]}')
    print(f'✓ збіг 1H+4H → подія: {d}')


def test_disagreement_is_not_an_event_and_is_not_logged():
    _install_log(); _install_tm()
    ns = _mk(htf_on=True, ob4_bias='BEARISH')
    out = _tick(ns, ob=_ob(bias='BULLISH'))
    _check(out == 'wait_htf', f'розбіг → не подія: {out}')
    _check(not _LOGGED, f'⚠️ придушений СТАН у 🧾 Лог НЕ пишемо: {_LOGGED}')
    diag = ns._ob_alert_diag.get('BTCUSDT') or {}
    _check(diag.get('outcome') == 'wait_htf' and 'розбіг' in (diag.get('note') or ''),
           f'але причина мусить лишитись у діагностиці: {diag}')
    print('✓ розбіг TF → тихо, причина — у ob_alert_diag')


def test_disagreement_does_not_swallow_the_block():
    """⚠️ НАЙТОНШЕ МІСЦЕ. Старший TF оновлюється СВОЇМ баром, тож збіг може
    настати за кілька хвилин — і це буде ТА САМА поява блоку. Якби розбіг
    позначав блок «опрацьованим», подія загубилась би назавжди (та сама помилка,
    що вже траплялась у VOB: «скидання ковтало блок»)."""
    _install_log(); _install_tm()
    ns = _mk(htf_on=True, ob4_bias='BEARISH')
    now = time.time()
    ob = _ob(bias='BULLISH', now=now)
    _check(_tick(ns, ob=ob, now=now) == 'wait_htf', 'спершу розбіг')
    _check(not ns._ob_alert_seen.get('BTCUSDT'),
           f'блок НЕ має бути опрацьований: {ns._ob_alert_seen}')
    # старший TF довернувся в наш бік — той самий блок мусить вистрілити
    ns._htf_bias = 'BULLISH'
    _check(_tick(ns, ob=ob, now=now) == 'new', 'після збігу — подія')
    _check(len(_LOGGED) == 1, f'і рівно ОДИН рядок: {_LOGGED}')
    print('✓ розбіг лише ЧЕКАЄ: збіг пізніше дає ту саму подію')


def test_stale_block_is_still_a_silent_baseline():
    """Вікно свіжості лишається головним запобіжником від вічного перебору:
    блок, що випав із вікна, тихо стає базою — навіть якщо збігу так і не було."""
    _install_log(); _install_tm()
    ns = _mk(htf_on=True, ob4_bias='BEARISH')
    now = time.time()
    old = _ob(bias='BULLISH', created_at_t=int((now - 20 * HOUR) * 1000), now=now)
    _check(_tick(ns, ob=old, now=now) == 'stale', 'давній блок → тиха база')
    _check(ns._ob_alert_seen.get('BTCUSDT'), 'база мусить бути взята')
    _check(not _LOGGED and not ns._htf_calls,
           'ні рядка, ні запиту старшого TF на давньому блоці')
    print('✓ старий блок: база взята, збіг навіть не питаємо')


# ═══════════ 4. ТІК: ДЕДУП ════════════════════════════════════════════════
def test_second_same_direction_block_is_deduped():
    _install_log(); _install_tm()
    ns = _mk(htf_on=False)
    now = time.time()
    _check(_tick(ns, ob=_ob(bias='BEARISH', now=now), now=now) == 'new', 'перший')
    ob2 = _ob(bias='BEARISH', bar_time=int((now - HOUR) * 1000), now=now)
    out = _tick(ns, ob=ob2, now=now)
    _check(out == 'dedup', f'другий SHORT підряд → дедуп: {out}')
    _check(len(_LOGGED) == 1, f'другого рядка бути не має: {_LOGGED}')
    # ⚠️ Придушений блок мусить бути ОПРАЦЬОВАНИЙ, інакше перевірявся б щоцикл
    _check(oba.is_processed(ns._ob_alert_seen['BTCUSDT'], ob2['bar_time']),
           'придушений блок усе одно опрацьований')
    # протилежний — перебиває
    ob3 = _ob(bias='BULLISH', bar_time=int((now - 3 * HOUR) * 1000), now=now)
    _check(_tick(ns, ob=ob3, now=now) == 'new', 'протилежний мусить пройти')
    _check(len(_LOGGED) == 2, f'і дати свій рядок: {_LOGGED}')
    print('✓ тік: 1 на тренд — перебиває лише протилежний')


def test_dedup_off_lets_every_new_block_through():
    _install_log(); _install_tm()
    ns = _mk(htf_on=False, dedup=False)
    now = time.time()
    for i in range(3):
        ob = _ob(bias='BEARISH', bar_time=int((now - (i + 1) * HOUR) * 1000), now=now)
        _check(_tick(ns, ob=ob, now=now) == 'new', f'блок #{i}')
    _check(len(_LOGGED) == 3, f'дедуп вимкнено → три рядки: {len(_LOGGED)}')
    print('✓ дедуп OFF → кожен новий блок дає подію')


def test_dedup_stamp_survives_restart():
    """⚠️ Позначка мусить ПЕРЕЖИТИ рестарт: `botupdate` робиться часто, і без
    персисту той самий тренд давав би другий сигнал після кожного оновлення."""
    store = {}
    ns = types.SimpleNamespace()
    ns._settings = {'ob_filter_timeframe': '1h'}
    ns._ob_alert_seen = {'BTCUSDT': [123]}
    ns._ob_alert_fired = {'BTCUSDT': {'side': 'SHORT', 'combo': '1H+4H', 'ts': 1.0}}
    ns.db = types.SimpleNamespace(
        set_setting=lambda k, v: store.__setitem__(k, v),
        get_setting=lambda k, d=None: store.get(k, d))
    for n in ('_persist_ob_alert_state', '_load_ob_alert_state'):
        setattr(ns, n, getattr(S, n).__get__(ns))
    ns._persist_ob_alert_state(force=True)
    _check(store[_m.DB_KEY_OB_ALERT]['fired']['BTCUSDT']['combo'] == '1H+4H',
           f'комбінація мусить зберігатись: {store}')
    fresh = types.SimpleNamespace()
    fresh._settings = {'ob_filter_timeframe': '1h'}
    fresh._ob_alert_seen, fresh._ob_alert_fired = {}, {}
    fresh.db = ns.db
    fresh._load_ob_alert_state = S._load_ob_alert_state.__get__(fresh)
    fresh._load_ob_alert_state()
    got = fresh._ob_alert_fired.get('BTCUSDT') or {}
    _check(got.get('side') == 'SHORT' and got.get('combo') == '1H+4H',
           f'позначка не відновилась: {fresh._ob_alert_fired}')
    _check(fresh._ob_alert_seen.get('BTCUSDT') == [123], 'seen теж на місці')
    # сміття не має ламати завантаження
    store[_m.DB_KEY_OB_ALERT] = {'tf': '1h', 'fired': {'X': {'side': '?'},
                                                       'Y': 'junk'}}
    fresh._ob_alert_fired = {}
    fresh._load_ob_alert_state()
    _check(not fresh._ob_alert_fired, f'сміття не пишемо: {fresh._ob_alert_fired}')
    print('✓ позначка дедупу переживає рестарт (разом із комбінацією TF)')


# ═══════════ 5. ОКРЕМИЙ ТИП СИГНАЛУ ══════════════════════════════════════
def test_signal_goes_through_the_shared_gate_and_opens():
    """📨 Тумблер УВІМК → сигнал іде в ту саму обробку, що й решта типів."""
    _install_log(); _install_tm()
    ns = _mk(htf_on=False, as_sig=True, allowed=True)
    _check(_tick(ns, ob=_ob(bias='BEARISH')) == 'new', 'подія мусить бути')
    _check(ns._gate_calls and ns._gate_calls[0][1] == 'SHORT',
           f'спільні ворота мусять бути спитані: {ns._gate_calls}')
    _check(len(_OPENED) == 1 and _OPENED[0]['opened_by'] == 'ob_alert',
           f'мітка походження: {_OPENED}')
    _check(_OPENED[0]['side'] == 'SHORT' and _OPENED[0]['entry_price'] > 0, _OPENED)
    evs = [e['event'] for e in _LOGGED]
    _check('ob_new' in evs and 'signal' in evs, f'події в лозі: {evs}')
    _sig = [e for e in _LOGGED if e['event'] == 'signal'][0]
    _check('🆕 Новий OB (1H)' in _sig['detail'] and 'OB(1h):✓' in _sig['detail'],
           f'рядок сигналу мусить нести комбінацію TF і РОЗКЛАД фільтрів: {_sig}')
    print(f'✓ тип сигналу: {_sig["detail"]}')


def test_blocked_signal_is_not_opened_and_the_reason_is_written():
    _install_log(); _install_tm()
    ns = _mk(htf_on=False, as_sig=True, allowed=False)
    _check(_tick(ns, ob=_ob(bias='BULLISH')) == 'new',
           'сама ПОДІЯ «новий OB» лишається (фільтр ріже лише торгівлю)')
    _check(not _OPENED, f'заблокований сигнал НЕ відкривається: {_OPENED}')
    _check([e for e in _LOGGED if e['event'] == 'rejected'],
           f'причина мусить бути в лозі: {_LOGGED}')
    print('✓ фільтр зарізав → угоди немає, причина в лозі')


def test_signal_toggle_off_keeps_it_a_message_only():
    """⚠️ Дефолт OFF: новий тип сигналу не має мовчки розширити потік угод."""
    _install_log(); _install_tm()
    ns = _mk(htf_on=False, as_sig=False)
    _check(_tick(ns, ob=_ob(bias='BULLISH')) == 'new', 'рядок у лозі лишається')
    _check(not ns._gate_calls and not _OPENED,
           f'ні воріт, ні відкриття: {ns._gate_calls} {_OPENED}')
    _check(_m.DEFAULT_SETTINGS.get('ob_alert_signal') is False,
           'дефолт «у торгівлю» мусить бути ВИМКНЕНО')
    print('✓ тумблер «у торгівлю» OFF → лишається повідомленням')


def test_signal_never_bypasses_the_shared_gate():
    """ЗАМОК У КОДІ (урок ASTERUSDT): VOB-alert колись кликав `on_signal`
    напряму й обходив УСІ фільтри. Кожен новий шлях мусить заходити в
    `_signal_allowed` — і РАНІШЕ за `on_signal`."""
    fn = next(n for n in ast.walk(ast.parse(_SRC))
              if isinstance(n, ast.FunctionDef) and n.name == '_ob_alert_signal')
    body = ast.dump(fn)
    _check('_signal_allowed' in body, 'спільні ворота не кличуться')
    _check('on_signal' in body, 'сигнал мусить доходити до TM')
    i_gate = min(n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Attribute) and n.attr == '_signal_allowed')
    i_open = min(n.lineno for n in ast.walk(fn)
                 if isinstance(n, ast.Attribute) and n.attr == 'on_signal')
    _check(i_gate < i_open, f'ворота ({i_gate}) мусять стояти ДО {i_open}')
    print('✓ сигнал нового типу не може обійти спільні ворота')


def test_label_is_mirrored_everywhere():
    """🏷 Мітка сигналу — ЄДИНЕ джерело + ОБОВʼЯЗКОВІ JS-дзеркала."""
    _check(labels.SIGNAL_BADGES.get('ob_alert') == '🆕 Новий OB',
           labels.SIGNAL_BADGES.get('ob_alert'))
    _check(labels.pretty_opened_by('ob_alert → Q4') == '🆕 Новий OB → 🎯 Черга-4',
           labels.pretty_opened_by('ob_alert → Q4'))
    _check(labels.signal_code_of('ob_alert → Q4') == 'ob_alert', 'код для логіки')
    _check("'ob_alert': '🆕 Новий OB'" in _HTML and "'ob_alert': '🆕'" in _HTML,
           'JS-дзеркало у smart_money.html не синхронне')
    js = open(os.path.join(_HERE, 'infosite', 'app.js'), encoding='utf-8').read()
    _check('"ob_alert": "🆕 Новий OB"' in js and '"ob_alert": "🆕"' in js,
           'JS-дзеркало в infosite/app.js не синхронне')
    print('✓ мітка 🆕 Новий OB: бекенд + обидва JS-дзеркала')


def test_entry_gates_still_know_nothing_about_the_alert():
    """⚠️ Новий ТИП СИГНАЛУ не зробив алерт воротами: `_ob_filter_allows`,
    «1H OB лише з CHoCH» і такт `vob_one_per_ob` читають ТОЙ САМИЙ рядок БД."""
    import inspect
    for fn in (S._ob_filter_allows, S._current_ob_bartime):
        src = inspect.getsource(fn)
        for bad in ('ob_alert', 'ob_new', 'combo_label', 'dedup_allows'):
            _check(bad not in src, f'{fn.__name__} знає про «{bad}» — це регресія')
    print('✓ ворота входу і такт VOB лишились недоторканими')


# ═══════════ 6. НАЛАШТУВАННЯ ТА UI ═══════════════════════════════════════
def test_defaults_are_new_keys_so_no_migration_is_needed():
    """⚠️ НОВІ ключі → `merged.update(stored)` віддає саме дефолт, міграція не
    потрібна (на відміну від кейсу `pilot_autofill_migrated_v1`)."""
    d = _m.DEFAULT_SETTINGS
    _check(d.get('ob_alert_htf_enabled') is True, 'збіг із другим TF — УВІМК')
    _check(d.get('ob_alert_htf') == '4h', 'другий TF за замовчуванням 4H')
    _check(d.get('ob_alert_dedup') is True, 'дедуп — УВІМК')
    _check(d.get('ob_alert_signal') is False, '«у торгівлю» — ВИМК')
    print('✓ дефолти: збіг ON · 4H · дедуп ON · торгівля OFF')


def test_settings_are_accepted_and_validated():
    ns = types.SimpleNamespace()
    _check("'ob_alert_htf_enabled', 'ob_alert_dedup'" in _SRC,
           'нові ключі мусять бути у білому списку update_settings')
    i = _SRC.index("# === 🆕 Алерт «новий OB»: валідація ===")
    blk = _SRC[i:i + 1400]
    for k in ('ob_alert_htf_enabled', 'ob_alert_dedup', 'ob_alert_signal'):
        _check(f"self._settings['{k}'] = bool(" in blk,
               f'{k} мусить зводитись до bool (UI шле і рядки, і None)')
    print('✓ нові ключі: приймаються і валідуються')


def test_ui_has_its_own_signal_type_block():
    """«Організуй все це як ще один вид сигналу. З можливістю вмикати і
    вимикати» — чотири контроли ОДНИМ блоком у ряду типів сигналів."""
    for cid in ('sm-ob-alert', 'sm-ob-alert-htf', 'sm-ob-alert-htf-tf',
                'sm-ob-alert-dedup', 'sm-ob-alert-signal'):
        _check(f'id="{cid}"' in _HTML, f'немає контрола {cid}')
    i = _HTML.index('id="sm-obalert-group"')
    blk = _HTML[i:i + 3200]
    _check(blk.count('updateObAlert()') >= 5,
           'кожен контрол мусить зберігати налаштування')
    # Блок мусить стояти в ряду ТИПІВ СИГНАЛІВ, поряд із 🟪 Volumized OB
    _check(_HTML.index('id="sm-vob-alert"') < i, 'блок не в ряду типів сигналів')
    # Випадайка другого TF — той самий набір, що у воріт (два набори розійшлись би)
    for tf in ('15m', '30m', '1h', '4h'):
        _check(f'value="{tf}"' in blk, f'немає варіанта {tf} у другому TF')
    _check('value="4h" selected' in blk, 'за замовчуванням 4H')
    print('✓ UI: 🆕 Новий OB — окремий блок типу сигналу')


def test_ui_posts_all_keys_together_and_greys_out_dead_controls():
    """⚠️ Частковий стан не має доїжджати до бекенда (принцип `updateOBFilter`),
    а активний контрол, який ні на що не впливає, вводить в оману (той самий
    урок, що з полями TTL при ♾ «Без терміну»)."""
    i = _HTML.index('async function updateObAlert()')
    fn = _HTML[i:i + 1600]
    for k in ('ob_alert_enabled', 'ob_alert_htf_enabled', 'ob_alert_htf',
              'ob_alert_dedup', 'ob_alert_signal'):
        _check(k in fn, f'{k} не шлеться разом з іншими')
    j = _HTML.index('function _smObAlertSync()')
    sync = _HTML[j:j + 1200]
    _check('disabled' in sync and 'opacity' in sync,
           'мертві контроли мусять гаснути')
    _check('_smObAlertSync()' in _HTML[:i] or '_smObAlertSync();' in _HTML,
           'синк мусить кликатись і при завантаженні налаштувань')
    print('✓ UI: усі ключі разом, мертві контроли погашені')


def test_header_summary_shows_the_active_combination():
    """Комбінація TF — це те, за чим бот реагує і чим міряє тренд дедуп, тож
    вона має бути видна в шапці, а не лише в розгорнутій гармошці."""
    i = _HTML.index("summary += ` · 🆕 OB ${_combo}`")
    blk = _HTML[max(0, i - 700):i + 300]
    _check("_oaH.checked" in blk and "'+'" in blk,
           'у шапці мусить зʼявлятись другий TF, коли збіг увімкнено')
    print('✓ шапка налаштувань показує активну комбінацію (🆕 OB 1H+4H)')


if __name__ == '__main__':
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_')]
    bad = 0
    for name, fn in fns:
        try:
            fn()
        except Exception as e:
            bad += 1
            print(f'  FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - bad}/{len(fns)} passed')
    sys.exit(1 if bad else 0)
