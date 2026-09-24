"""📨 TELEGRAM: тема «🧮 МММ-монітор» + угоди в групу (вимога 21.09).

**Вимоги користувача, дослівно:**
  1. «Перероби все оповіщення в Телеграм, яке стосувалось ₿ BTCUSDT тепер на
     "🧮 МММ-МОНІТОР", зміни в групі назву і напрявляй туди як і було раніше
     ₿ BTCUSDT якщо увімкнено відправку, а основним тепер сюди напрявляй
     оповіщення стосовно змін банера "🧮 МММ-МОНІТОР".»
  2. «Також проаналізуй повідомлення, які ідуть в основний телеграм бот
     стосовно відкритих угод поза потом "External", напрляй їх в основну
     групу, а не в телеграм бот.»
  3. «Стосовно корекцій також зроби оповіщення. Додай можливість вмикати та
     вимикати, тобто можливість контролювати відправку всіх оповіщень.»

Що стережуть ці тести:
  • підпис теми став «🧮 МММ-монітор», а КЛЮЧ категорії лишився `btc` —
    його знають env-змінні, збережені id тем і тумблер кабінету `notify_btc`;
  • ВЖЕ СТВОРЕНА тема РЕАЛЬНО перейменовується в групі (`editForumTopic`), і
    рівно один раз — інакше вимога «зміни в групі назву» не виконана;
  • зміна ПІДТВЕРДЖЕНОГО статусу банера йде в ту саму тему, а перший такт
    після рестарту — мовчки (`botupdate` не подія);
  • події корекції (почалась / завершилась) ідуть тим самим текстом, що в
    🧾 Лог, а відліки (⏳) — ні;
  • обидва потоки мають власні тумблери, і збій Telegram не ламає такт;
  • повідомлення про ВІДКРИТІ УГОДИ йдуть у груповий топік (`category`), а
    службові помилки лишаються в приватному боті адміна.
"""
import ast
import importlib.util
import os
import re
import sys
import threading
import types

_HERE = os.path.dirname(os.path.abspath(__file__))


def _check(c, msg):
    if not c:
        raise AssertionError(msg)


# ───────────────────────────────────────────────────────────────────────────
# Завантаження модулів БЕЗ важких пакетних __init__ (pybit тощо).
# ───────────────────────────────────────────────────────────────────────────
_pkg = sys.modules.get('detection') or types.ModuleType('detection')
_pkg.__path__ = [os.path.join(_HERE, 'detection')]
sys.modules['detection'] = _pkg


class _DB:
    """Мінімальний шар БД: `tg_forum_topics` / `tg_forum_topic_names`."""
    store = {}

    def get_setting(self, k, d=None):
        return _DB.store.get(k, d)

    def set_setting(self, k, v):
        _DB.store[k] = v


_stg = sys.modules.get('storage') or types.ModuleType('storage')
_stg.__path__ = [os.path.join(_HERE, 'storage')]
sys.modules['storage'] = _stg
_dbmod = types.ModuleType('storage.db_operations')
_dbmod.get_db = lambda: _DB()
sys.modules['storage.db_operations'] = _dbmod
_stg.db_operations = _dbmod

if 'web' not in sys.modules:
    _wpkg = types.ModuleType('web')
    _wpkg.__path__ = [os.path.join(_HERE, 'web')]
    sys.modules['web'] = _wpkg
_tspec = importlib.util.spec_from_file_location(
    'web.tg_bot', os.path.join(_HERE, 'web', 'tg_bot.py'))
tg = importlib.util.module_from_spec(_tspec)
sys.modules['web.tg_bot'] = tg
_tspec.loader.exec_module(tg)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_HERE, 'detection', fname))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mc = _load('detection.mm_correction', 'mm_correction.py')
_pkg.mm_correction = mc

_fspec = importlib.util.spec_from_file_location(
    'fuel_filter_tg_test', os.path.join(_HERE, 'detection', 'fuel_filter.py'))
_ffm = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(_ffm)
FF = _ffm.FuelFilterDaemon

_FF_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'),
               encoding='utf-8').read()
_TM_SRC = open(os.path.join(_HERE, 'detection', 'trade_manager.py'),
               encoding='utf-8').read()
_TG_SRC = open(os.path.join(_HERE, 'web', 'tg_bot.py'), encoding='utf-8').read()
_HTML = open(os.path.join(_HERE, 'templates', 'smart_money.html'),
             encoding='utf-8').read()

NOW = 10_000.0


def _fn_src(src, name):
    """Тіло функції БЕЗ докстрінга — щоб замок не спрацював на поясненні."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = (node.body[1:] if (node.body
                                      and isinstance(node.body[0], ast.Expr)
                                      and isinstance(node.body[0].value, ast.Constant))
                    else node.body)
            return '\n'.join(ast.unparse(b) for b in body)
    raise AssertionError(f'функція {name} зникла')


# ───────────────────────────────────────────────────────────────────────────
# Стенди
# ───────────────────────────────────────────────────────────────────────────
_LOGGED = []


def _install_log():
    _LOGGED.clear()
    mod = types.ModuleType('detection.activity_log')

    def _log(symbol, event, detail='', side=None, source='', extra=None):
        _LOGGED.append({'event': event, 'detail': detail, 'source': source})
    mod.log_activity = _log
    sys.modules['detection.activity_log'] = mod
    _pkg.activity_log = mod


_install_log()

_DBROWS = []
_dbmod_corr = types.ModuleType('storage.db_operations')


class _DB2(_DB):
    def log_mm_correction(self, **row):
        _DBROWS.append(dict(row))


_dbmod.get_db = lambda: _DB2()


def _snap(**coins):
    out = {}
    for sym, (st, stren, pdir) in coins.items():
        out[sym] = {'status': st, 'strength': stren,
                    'dir': (stren / 100.0) * (1 if st == 'LONG' else -1),
                    'price_dir': pdir}
    return out


def _mk(trends=None, vob_on=True, **settings):
    """Двигун із перехопленим `_broadcast_users` — тести дивляться на те, ЩО і
    В ЯКУ КАТЕГОРІЮ пішло, а не на роботу самого Telegram."""
    ff = FF.__new__(FF)
    ff._lock = threading.RLock()
    ff._mm_bias, ff._mm_bias_since, ff._mm_bias_cand = {}, 0.0, {}
    ff._mm_bias_tg_last = '__none__'
    ff._mm_corr_st, ff._mm_corr, ff._mm_lever_hist = {}, {}, []
    ff._mm_corr_skip_logged = {}
    ff._mm_corr_log_at = 0.0
    ff._mm_snapshot = {}
    ff._engine_skip = {}
    s = {'mm_bias_confirm_sec': 0, 'mm_corr_confirm_sec': 0,
         'mm_corr_min_layers': 2, 'mm_corr_vob_pct': 60.0,
         'mm_corr_price_pct': 60.0, 'mm_corr_lever_drop': 15.0,
         'mm_corr_enabled': True, 'mm_corr_block_open': True,
         'mm_corr_log_enabled': False, 'mm_corr_log_every_sec': 300,
         'mm_bias_tg': True, 'mm_corr_tg': True,
         'mm_monitor_enabled': True, 'enabled': True}
    s.update(settings)
    ff._settings = s
    ff.get_settings = lambda: dict(ff._settings)
    ff._mm_vob_trends = lambda: {'on': vob_on, 'tf': '5m',
                                 'trends': dict(trends or {})}
    ff.sent = []
    # ⚠️ Стаб мусить повертати ТЕ САМЕ, що справжній метод — `(ok, причина)`.
    # Поки він віддавав None, розпакування в `_mm_track_correction` падало в
    # `except Exception`, і нова гілка «Telegram не прийняв» не виконувалась
    # НІКОЛИ, хоча тести зеленіли (задокументована пастка стабів).
    ff.tg_ok = True

    def _bc(cat, pref, text):
        ff.sent.append((cat, pref, text))
        return (True, '') if ff.tg_ok else (False, 'CHAT_WRITE_FORBIDDEN')

    ff._broadcast_users = _bc
    return ff


def _tick(ff, snap, now=NOW):
    s = ff.get_settings()
    ff._mm_track_bias(snap, now, s)
    ff._mm_track_correction(snap, now, s)
    return dict(ff._mm_corr)


def _banner(ff):
    """Лише повідомлення БАНЕРА (у них є важіль) — щоб тест про банер не
    рахував заразом події корекції, які летять у ту саму тему."""
    return [t for c, p, t in ff.sent if 'важіль' in t]


def _long(n=6, strength=60):
    return _snap(**{f'C{i}': ('LONG', strength, 'up') for i in range(n)})


def _short(n=6, strength=60):
    return _snap(**{f'C{i}': ('SHORT', strength, 'down') for i in range(n)})


# ═══════════ 1. ТЕМА В ГРУПІ: ПІДПИС ЗМІНЕНО, КЛЮЧ — НІ ══════════════════
def test_topic_is_labelled_the_monitor_now():
    """Підпис теми і хештег стосуються 🧮 МММ-монітора — саме туди тепер іде
    головний потік (банер + корекція), а ₿-сеанс лишається сусідом."""
    _check(tg._CAT_LABEL['btc'] == '🧮 МММ-монітор',
           f'підпис теми мусить бути 🧮 МММ-монітор: {tg._CAT_LABEL["btc"]!r}')
    _check('₿ BTCUSDT' not in tg._CAT_LABEL.values(),
           'старого підпису «₿ BTCUSDT» у темах лишатись не має')
    _check(tg._CAT_TAG['btc'] == '#МММ_монітор',
           f'хештег теж мусить назвати монітор: {tg._CAT_TAG["btc"]!r}')
    print('✓ тема називається «🧮 МММ-монітор»')


def test_the_category_key_itself_never_changed():
    """⚠️ КЛЮЧ `btc` перейменувати НЕ МОЖНА: його знають env-змінні, збережені
    id тем і тумблер кабінету. Перейменування загубило б і тему, і тумблер
    (той самий прецедент, що з `q2_auto_ob_sl*`)."""
    _check(tg._CAT_ENV['btc'] == ('TELEGRAM_CHAT_BTC', 'TELEGRAM_TOPIC_BTC'),
           f'env-змінні мусять лишитись тими самими: {tg._CAT_ENV["btc"]}')
    _check('btc' in tg._PROTECTED_CATS, 'тема лишається захищеною від пересилки')
    src = _fn_src(_TG_SRC, '_cat_enabled')
    _check("'btc': 'notify_btc'" in src,
           'тумблер кабінету мусить лишитись `notify_btc`')
    print('✓ ключ категорії лишився `btc` — env, id тем і тумблер живі')


def test_an_already_created_topic_is_actually_renamed_in_the_group():
    """Тема створюється ОДИН раз, і її id персиститься — тож зміни `_CAT_LABEL`
    МАЛО. Без `editForumTopic` у групі й далі висіло б «₿ BTCUSDT»."""
    _DB.store.clear()
    _DB.store['tg_forum_topics'] = {'-100500': {'btc': 7}}
    tg._forum_topics_cache = None
    tg._forum_names_cache = None
    calls = []
    tg._api = lambda m, p=None: (calls.append((m, dict(p or {}))) or {'ok': True})
    os.environ['TELEGRAM_FORUM_CHAT'] = '-100500'
    chat, tid = tg._forum_thread('btc')
    _check((chat, tid) == ('-100500', 7), f'тема мусить лишитись тією самою: {chat}/{tid}')
    _check(any(m == 'editForumTopic' and p.get('name') == '🧮 МММ-монітор'
               and int(p.get('message_thread_id')) == 7 for m, p in calls),
           f'мусив піти editForumTopic із новою назвою: {calls}')
    print('✓ наявна тема РЕАЛЬНО перейменовується в групі')


def test_the_rename_is_asked_for_only_once():
    """⚠️ Назву запамʼятовуємо в БД — інакше `editForumTopic` смикався б на
    КОЖНЕ повідомлення (зайвий запит до API на рівному місці)."""
    _DB.store.clear()
    _DB.store['tg_forum_topics'] = {'-100500': {'btc': 7}}
    tg._forum_topics_cache = None
    tg._forum_names_cache = None
    calls = []
    tg._api = lambda m, p=None: (calls.append(m) or {'ok': True})
    os.environ['TELEGRAM_FORUM_CHAT'] = '-100500'
    tg._forum_thread('btc')
    tg._forum_thread('btc')
    tg._forum_thread('btc')
    _check(calls.count('editForumTopic') == 1,
           f'перейменування мусить бути ОДНЕ на всі виклики: {calls}')
    saved = _DB.store.get('tg_forum_topic_names') or {}
    _check(saved.get('-100500', {}).get('btc') == '🧮 МММ-монітор',
           f'назву мусили запамʼятати в БД (переживає рестарт): {saved}')
    # Рестарт процесу: кеш порожній, але БД памʼятає → API не смикаємо.
    tg._forum_names_cache = None
    calls.clear()
    tg._forum_thread('btc')
    _check('editForumTopic' not in calls,
           f'після рестарту перейменування повторюватись не має: {calls}')
    print('✓ перейменування рівно одне, і воно переживає рестарт')


def test_a_failed_rename_never_breaks_sending():
    """Best-effort, як і створення теми: не вийшло перейменувати — тема
    лишається зі старою назвою, а повідомлення йде як ішло. І НЕ
    запамʼятовуємо назву, якої не поставили, — спробуємо ще раз."""
    _DB.store.clear()
    _DB.store['tg_forum_topics'] = {'-100500': {'btc': 7}}
    tg._forum_topics_cache = None
    tg._forum_names_cache = None

    def _boom(m, p=None):
        if m == 'editForumTopic':
            raise RuntimeError('Telegram лежить')
        return {'ok': True}
    tg._api = _boom
    os.environ['TELEGRAM_FORUM_CHAT'] = '-100500'
    chat, tid = tg._forum_thread('btc')
    _check((chat, tid) == ('-100500', 7),
           f'відправка мусить працювати попри невдале перейменування: {chat}/{tid}')
    _check((_DB.store.get('tg_forum_topic_names') or {}) == {},
           'назву, якої не поставили, запамʼятовувати не можна')
    # Те саме для «ok: False» (Telegram відповів, але відмовив).
    tg._api = lambda m, p=None: ({'ok': False} if m == 'editForumTopic' else {'ok': True})
    tg._forum_thread('btc')
    _check((_DB.store.get('tg_forum_topic_names') or {}) == {},
           'відмову API теж не можна вважати успіхом')
    print('✓ невдале перейменування не ламає відправку і не «залипає»')


# ═══════════ 2. БАНЕР 🧮 → TELEGRAM ══════════════════════════════════════
def test_a_confirmed_banner_change_is_announced():
    """ГОЛОВНИЙ ПОТІК теми: банер перемкнув ПІДТВЕРДЖЕНИЙ статус."""
    ff = _mk()
    _tick(ff, _long(), NOW)                       # перший такт — тиха база
    ff.sent.clear()
    _tick(ff, _short(), NOW + 60)
    _msgs = _banner(ff)
    _check(len(_msgs) == 1, f'мусить піти РІВНО одне повідомлення: {ff.sent}')
    txt = _msgs[0]
    cat, pref = [(c, p) for c, p, t in ff.sent if t == txt][0]
    _check(cat == 'btc' and pref == 'notify_btc',
           f'іти має в ту саму тему, що й раніше ₿: {cat}/{pref}')
    _check('МММ-МОНІТОР' in txt, f'повідомлення мусить назвати джерело: {txt}')
    _check('SHORT' in txt and 'LONG' in txt,
           f'мусить бути видно і новий статус, і попередній: {txt}')
    _check('монет:' in txt, f'розклад монет — це і є підстава статусу: {txt}')
    print('✓ зміна статусу банера → повідомлення в тему 🧮')


def test_the_first_tick_after_a_restart_is_silent():
    """⚠️ `_mm_bias` ПЕРСИСТИТЬСЯ, тож після `botupdate` перший такт міг би
    «побачити зміну» і надіслати повідомлення про подію, якої не було."""
    ff = _mk()
    ff._mm_bias = {'dir': 'LONG', 'since': NOW - 3600, 'pct': 50.0}
    ff._mm_bias_since = NOW - 3600            # відновлено з БД
    _tick(ff, _short(), NOW)                  # ринок уже інший
    _check(_banner(ff) == [],
           f'перший такт після рестарту мусить лише запамʼятати стан: {ff.sent}')
    # А ось НАСТУПНА зміна — уже подія.
    _tick(ff, _long(), NOW + 60)
    _check(len(_banner(ff)) == 1, f'друга зміна вже мусить піти: {ff.sent}')
    print('✓ перший такт після рестарту — мовчки, друга зміна — подія')


def test_only_a_confirmed_status_is_announced():
    """⚠️ Кандидат (⏳) — це ще відлік. Сповіщати про нього означало б
    повернути те саме миготіння, від якого банер і захищали."""
    ff = _mk(mm_bias_confirm_sec=120)
    _tick(ff, _long(), NOW)
    ff.sent.clear()
    _tick(ff, _short(), NOW + 30)             # кандидат, вікно ще не вийшло
    _check(ff._mm_bias.get('cand_dir') == 'SHORT',
           f'кандидат мусить зʼявитись: {ff._mm_bias}')
    _check(ff._mm_bias.get('dir') == 'LONG', 'підтверджений статус ще старий')
    _check(_banner(ff) == [], f'про кандидата не сповіщаємо: {ff.sent}')
    _tick(ff, _short(), NOW + 200)            # вікно вийшло
    _check(len(_banner(ff)) == 1,
           f'після підтвердження — одне повідомлення: {ff.sent}')
    print('✓ шлемо лише ПІДТВЕРДЖЕНИЙ статус, не кандидата')


def test_the_same_status_is_never_repeated():
    ff = _mk()
    _tick(ff, _long(), NOW)
    ff.sent.clear()
    _tick(ff, _long(), NOW + 60)
    _tick(ff, _long(), NOW + 120)
    _check(_banner(ff) == [], f'той самий статус — не подія: {ff.sent}')
    print('✓ незмінний статус повідомлень не породжує')


def test_the_banner_toggle_turns_the_alerts_off():
    """«Додай можливість вмикати та вимикати» — для потоку банера."""
    ff = _mk(mm_bias_tg=False)
    _tick(ff, _long(), NOW)
    _tick(ff, _short(), NOW + 60)
    _tick(ff, _long(), NOW + 120)
    _check(_banner(ff) == [], f'тумблер вимкнено — нічого не шлемо: {ff.sent}')
    print('✓ тумблер ⚖️ банера гасить потік')


def test_a_broken_telegram_never_stops_the_tick():
    """Збій відправки не має підійматись у такт двигуна: банер мусить
    порахуватись, навіть коли Telegram лежить."""
    ff = _mk()
    _tick(ff, _long(), NOW)

    def _boom(*a, **k):
        raise RuntimeError('Telegram лежить')
    ff._broadcast_users = _boom
    _tick(ff, _short(), NOW + 60)
    _check(ff._mm_bias.get('dir') == 'SHORT',
           f'банер мусить оновитись попри збій відправки: {ff._mm_bias}')
    print('✓ збій Telegram не ламає розрахунок банера')


# ═══════════ 3. КОРЕКЦІЯ → TELEGRAM ══════════════════════════════════════
def _corr_on(ff, now=NOW):
    """Довести детектор до підтвердженої корекції проти банера LONG."""
    trends = {f'C{i}': 'SHORT' for i in range(6)}
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m', 'trends': trends}
    _tick(ff, _long(strength=70), now)                 # банер LONG, важіль 70
    down = _snap(**{f'C{i}': ('LONG', 20, 'down') for i in range(6)})
    return _tick(ff, down, now + 60)                   # ознаки проти


def test_correction_start_and_end_are_announced():
    """«Стосовно корекцій також зроби оповіщення» — подія зупиняє відкриття
    угод, і дізнаватись про неї лише з логу запізно."""
    ff = _mk()
    _install_log()
    _corr_on(ff)
    _check(ff._mm_corr.get('state') == 'on',
           f'корекція мусила оголоситись: {ff._mm_corr}')
    corr = [t for c, p, t in ff.sent if 'КОРЕКЦІЯ' in t]
    _check(len(corr) == 1, f'мусить піти одне повідомлення про початок: {ff.sent}')
    # Вимога 24.09: у TG ЛИШЕ напрямок корекції (протилежний банеру LONG) і
    # час старту — розклад ознак лишається в 🧾 Лозі.
    _check(corr[0].startswith('🔴 SHORT КОРЕКЦІЯ · почалась ')
           and 'ознак' not in corr[0],
           f'коротке повідомлення про старт: {corr[0]}')
    # …і завершення.
    ff.sent.clear()
    trends = {f'C{i}': 'LONG' for i in range(6)}
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m', 'trends': trends}
    _tick(ff, _long(strength=70), NOW + 120)
    ended = [t for c, p, t in ff.sent if 'ЗАВЕРШИЛАСЬ' in t]
    _check(len(ended) == 1, f'завершення теж мусить піти: {ff.sent}')
    _check(ended[0].startswith('🔴 SHORT КОРЕКЦІЯ ЗАВЕРШИЛАСЬ (тривала ')
           and '→' in ended[0] and 'банер' not in ended[0],
           f'коротке повідомлення про кінець з часом від → до: {ended[0]}')
    print('✓ початок і кінець корекції → повідомлення в тему 🧮')


def test_the_telegram_text_is_the_same_event_as_the_log_line():
    """⚠️ Два різні тексти про ОДНУ подію розійшлися б. Тому Telegram несе
    РІВНО те, що вже пішло в 🧾 Лог роботи бота."""
    ff = _mk()
    _install_log()
    _corr_on(ff)
    log_txt = [r['detail'] for r in _LOGGED
               if r['source'] == 'MMM' and 'КОРЕКЦІЯ' in r['detail']]
    tg_txt = [t for c, p, t in ff.sent if 'КОРЕКЦІЯ' in t]
    _check(log_txt and tg_txt, f'мусять бути обидва канали: {log_txt} / {tg_txt}')
    # 24.09: TG став КОРОТКИМ, а лог лишив розклад — тож TG-текст мусить
    # бути ПОЧАТКОМ рядка логу (одна подія — один текст, лог лише довший).
    _check(log_txt[0].startswith(tg_txt[0]),
           f'TG мусить бути початком рядка логу:\n  лог: {log_txt[0]}\n  TG:  {tg_txt[0]}')
    print('✓ Telegram і 🧾 Лог описують подію ОДНИМ текстом')


def test_countdowns_are_not_announced():
    """⚠️ `pending` / `ending` — це ще не подія (той самий принцип, що з
    кандидатом банера)."""
    ff = _mk(mm_corr_confirm_sec=120)
    _install_log()
    trends = {f'C{i}': 'SHORT' for i in range(6)}
    ff._mm_vob_trends = lambda: {'on': True, 'tf': '5m', 'trends': trends}
    _tick(ff, _long(strength=70), NOW)
    ff.sent.clear()
    down = _snap(**{f'C{i}': ('LONG', 20, 'down') for i in range(6)})
    _tick(ff, down, NOW + 60)
    _check(ff._mm_corr.get('state') == 'pending',
           f'мусить бути відлік: {ff._mm_corr.get("state")}')
    _check([t for c, p, t in ff.sent if 'КОРЕКЦІЯ' in t] == [],
           f'про відлік не сповіщаємо: {ff.sent}')
    print('✓ відліки підтвердження в Telegram не йдуть')


def test_the_correction_toggle_turns_it_off_but_the_log_stays():
    """Тумблер гасить САМЕ ВІДПРАВКУ: 🧾 Лог роботи бота лишається — інакше
    подія зникла б і з історії."""
    ff = _mk(mm_corr_tg=False)
    _install_log()
    _corr_on(ff)
    _check([t for c, p, t in ff.sent if 'КОРЕКЦІЯ' in t] == [],
           f'тумблер вимкнено — у Telegram нічого: {ff.sent}')
    _check(any('КОРЕКЦІЯ' in r['detail'] for r in _LOGGED),
           '🧾 Лог мусить лишитись — тумблер про відправку, а не про історію')
    print('✓ тумблер 🔻 корекції гасить відправку, не історію')


def test_the_unit_in_the_telegram_text_is_the_one_the_layer_uses():
    """📉 Важіль міряється в П.П., а частки — у %. Одиницю дає БЕКЕНД
    (`layer['unit']`), і в тексті вона мусить бути та сама."""
    src = _fn_src(_FF_SRC, '_mm_track_correction')
    _check("x.get('unit')" in src or 'x.get("unit")' in src,
           'текст події мусить брати одиницю з шару, а не зашивати «%»')
    lev = mc.lever_layer(30.0, 55.0, 15.0)
    _check(lev.get('unit') == 'п.п.', f'важіль — у п.п.: {lev}')
    _check(mc.vob_layer({f'C{i}': 'SHORT' for i in range(6)},
                        [f'C{i}' for i in range(6)], 'LONG',
                        60.0).get('unit') == '%', 'частка — у %')
    print('✓ одиниця в тексті — та сама, що в шарі (% vs п.п.)')


# ═══════════ 4. УГОДИ — В ГРУПУ, НЕ В ПРИВАТНИЙ БОТ ══════════════════════
def _notify_calls(fn_name):
    """Усі `self._notify(...)` усередині методу → список kwargs-імен."""
    out = []
    for node in ast.walk(ast.parse(_TM_SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            for n in ast.walk(node):
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == '_notify'):
                    out.append({k.arg: ast.unparse(k.value) for k in n.keywords})
    return out


def test_open_trade_events_go_to_the_group_topic():
    """Вимога 2: «повідомлення… стосовно відкритих угод… напрявляй їх в
    основну групу, а не в телеграм бот». Без `category` `_notify` шле адміну
    в ПРИВАТ, зарезервований під службові повідомлення."""
    for fn in ('_check_breakeven', 'on_bos_event', '_on_bos_event_shadow',
               '_adopt_external_position'):
        calls = _notify_calls(fn)
        if not calls:
            continue
        for kw in calls:
            _check(kw.get('category') == "'trades'",
                   f'{fn}: подія відкритої угоди мусить іти в групу: {kw}')
    print('✓ BE-move · трейл після BOS-2 · прийнята зовнішня позиція → група')


def test_the_external_close_message_goes_to_the_group_too():
    """«…поза потоком External» — саме це повідомлення теж мусить іти в групу
    (раніше воно було єдиним із цієї сімʼї, що сипалось у приват)."""
    src = _TM_SRC
    i = src.find('🔄 External close')
    _check(i > 0, 'повідомлення про зовнішнє закриття зникло')
    _check("category='trades'" in src[i:i + 700],
           'External close мусить нести category=trades')
    print('✓ 🔄 External close → груповий топік')


def test_service_errors_still_go_to_the_private_bot():
    """⚠️ СВІДОМО НЕ ЧІПАЄМО: помилки розміру / відхилений ордер / збій
    закриття — це СЛУЖБОВІ повідомлення адміну, а не подія відкритої угоди.
    Приватний бот саме під них і зарезервований."""
    for marker in ('❌ Sizing error', '❌ Order rejected', '⚠️ Close API error'):
        i = _TM_SRC.find(marker)
        _check(i > 0, f'службове повідомлення зникло: {marker}')
        tail = _TM_SRC[i:i + 200].split('\n')[0]
        _check('category=' not in tail,
               f'{marker} мусить лишитись у приваті адміна: {tail}')
    print('✓ службові помилки лишились у приватному боті')


def test_notify_without_a_category_means_the_private_chat():
    """Замок на СЕМАНТИКУ: якщо це правило зміниться, усі рішення вище
    перестануть означати те, що означають зараз."""
    src = _fn_src(_TM_SRC, '_notify')
    _check('notify_category' in src,
           '`category` мусить вести в груповий топік через notify_category')
    print('✓ без `category` — приватний чат адміна, з нею — тема групи')


# ═══════════ 5. UI: КОНТРОЛЬ НАД УСІМА ПОТОКАМИ В ОДНОМУ МІСЦІ ═══════════
def test_ui_has_both_toggles_and_they_reach_the_server():
    for cid, key in (('ff-mm-bias-tg', 'mm_bias_tg'),
                     ('ff-mm-corr-tg', 'mm_corr_tg')):
        _check(f'id="{cid}"' in _HTML, f'немає контрола {cid}')
        _check(f"setIf('{cid}'" in _HTML, f'{cid} не читається з налаштувань')
        _check(f"{key}: _c('{cid}')" in _HTML, f'{cid} не доїжджає на сервер')
    print('✓ обидва тумблери є в UI і доїжджають на сервер')


def test_ui_names_every_stream_of_the_topic():
    """«можливість контролювати відправку ВСІХ оповіщень» — згорнута гармошка
    мусить казати, що саме зараз шлеться, включно з ₿-сеансом (його тумблер
    стоїть біля банера ₿, і про це теж сказано)."""
    i = _HTML.find('function _mmTgSummary')
    _check(i > 0, 'підсумок сповіщень зник')
    body = _HTML[i:i + 800]
    for token in ('mm_bias_tg', 'mm_corr_tg', 'start_signal_tg_alerts'):
        _check(token in body, f'підсумок мусить враховувати {token}')
    sec = _HTML[_HTML.find('id="mm-tg-sum"') - 600:_HTML.find('id="mm-tg-sum"') + 2000]
    _check('банера ₿' in sec, 'секція мусить пояснити, де тумблер ₿-сеансу')
    _check('кабінет' in sec, 'секція мусить назвати майстер-вимикач теми')
    print('✓ UI називає всі три потоки і майстер-вимикач')


def test_defaults_are_on_and_validation_coerces():
    """Дефолт УВІМК свідомо: тема існує САМЕ заради цих подій, а нові ключі
    → `merged.update(stored)` віддає дефолт (міграція не потрібна)."""
    _check(_ffm.DEFAULT_SETTINGS.get('mm_bias_tg') is True,
           'банерні сповіщення мусять бути увімкнені за замовчуванням')
    _check(mc.DEFAULTS.get('mm_corr_tg') is True,
           'сповіщення про корекцію мусять бути увімкнені за замовчуванням')
    ff = FF.__new__(FF)
    ff._lock = threading.RLock()

    class _Stored:
        def get_setting(self, k, d=None):
            return {'mm_bias_tg': 'yes', 'mm_corr_tg': 0}
    ff._db = _Stored()
    s = FF.get_settings(ff)
    _check(s['mm_bias_tg'] is True and s['mm_corr_tg'] is False,
           f'значення мусять зводитись до булевих: {s["mm_bias_tg"]!r}/'
           f'{s["mm_corr_tg"]!r}')
    print('✓ дефолти УВІМК, значення зводяться до булевих')


def test_the_cabinet_still_calls_the_topic_by_its_new_name():
    """Кабінет адміна — єдине місце майстер-вимикача теми; він мусить
    називати її так само, як сама тема, інакше вимкнути «не те» дуже легко."""
    src = open(os.path.join(_HERE, 'web', 'auth.py'), encoding='utf-8').read()
    i = src.find("id=\"nbtc\"")
    _check(i > 0, 'тумблер теми зник із кабінету')
    _check('МММ-монітор' in src[i - 400:i],
           'підпис тумблера мусить назвати 🧮 МММ-монітор')
    print('✓ кабінет називає тему її новим імʼям')


# ═══════════ 6. 🐞 «НАЗВА СТАРА, СПОВІЩЕНЬ НЕМАЄ» (скарга 21.09) ══════════
def test_the_first_real_change_after_a_restart_is_not_swallowed():
    """ГОЛОВНИЙ ДЕФЕКТ скарги. Позначка «перший показ після старту» живе в
    `_mm_bias_alert`, а викликався він ЛИШЕ на зміні статусу — тож позначку
    зʼїдала ПЕРША РЕАЛЬНА зміна після кожного рестарту. `botupdate` роблять
    часто, тому на проді сповіщень не було взагалі."""
    ff = _mk()
    ff._mm_bias = {'dir': 'LONG', 'since': NOW - 3600, 'pct': 50.0}
    ff._mm_bias_since = NOW - 3600            # відновлено з БД
    _tick(ff, _long(), NOW)                   # такт 1: той самий LONG
    _check(_banner(ff) == [], f'перший такт мусить лише запамʼятати: {ff.sent}')
    _tick(ff, _short(), NOW + 60)             # ПЕРША реальна зміна
    _check(len(_banner(ff)) == 1,
           f'перша ж реальна зміна мусить піти, а не бути зʼїденою: {ff.sent}')
    print('✓ перша реальна зміна після рестарту БІЛЬШЕ не ковтається')


def test_the_alert_is_called_every_tick_not_only_on_a_change():
    """Тест-замок на САМ корінь: якщо виклик знову заведуть під
    `if side != prev`, позначка знову зʼїдатиме першу зміну."""
    src = _fn_src(_FF_SRC, '_mm_track_bias')
    _check('_mm_bias_alert' in src, 'виклик алерта зник')
    for line in src.splitlines():
        if '_mm_bias_alert' in line:
            _check('if ' not in line,
                   f'виклик мусить бути безумовним: {line.strip()}')
    _check('if side != prev:\n    self._mm_bias_alert' not in src,
           'виклик знову стоїть під умовою зміни')
    print('✓ алерт кличеться на КОЖНОМУ такті — рішення ухвалює він сам')


def test_a_topic_set_by_env_is_renamed_too():
    """ДРУГИЙ ДЕФЕКТ: тему можна задати не автостворенням, а напряму
    (`TELEGRAM_CHAT_BTC` + `TELEGRAM_TOPIC_BTC`). `_forum_thread` тоді виходить
    першим рядком, і перейменування, що стояло лише там, НЕ спрацьовувало —
    саме тому «назва теми залишається старою»."""
    _DB.store.clear()
    tg._forum_topics_cache = None
    tg._forum_names_cache = None
    calls = []
    tg._api = lambda m, p=None: (calls.append((m, dict(p or {}))) or {'ok': True})
    os.environ.pop('TELEGRAM_FORUM_CHAT', None)     # форум-групи НЕМАЄ
    os.environ['TELEGRAM_CHAT_BTC'] = '-100777'
    os.environ['TELEGRAM_TOPIC_BTC'] = '42'
    try:
        chat, thread = tg._cat_chat('btc')
        _check((chat, thread) == ('-100777', '42'), f'маршрут: {chat}/{thread}')
        _check(any(m == 'editForumTopic' and p.get('name') == '🧮 МММ-монітор'
                   and int(p.get('message_thread_id')) == 42 for m, p in calls),
               f'тему з env теж мусить перейменувати: {calls}')
    finally:
        os.environ.pop('TELEGRAM_CHAT_BTC', None)
        os.environ.pop('TELEGRAM_TOPIC_BTC', None)
    print('✓ тема, задана через env, теж перейменовується')


def test_the_rename_cache_is_keyed_by_chat_and_category():
    """Теми різних категорій можуть жити в РІЗНИХ чатах, тож спільний ключ
    писав би назву однієї теми в запис іншої."""
    src = _fn_src(_TG_SRC, '_forum_rename_if_needed')
    _check('(str(chat), category)' in src,
           'кеш назв мусить бути ключований парою (чат, категорія)')
    print('✓ кеш назв ключований чатом і категорією')


def test_every_silent_reason_is_named_by_the_checker():
    """«Сповіщень немає» мало ЧОТИРИ мовчазні причини. Перевірка мусить
    називати кожну, інакше користувач знову лишиться зі здогадкою."""
    _DB.store.clear()
    tg._forum_topics_cache = None
    tg._forum_names_cache = None
    tg._api = lambda m, p=None: {'ok': True}
    os.environ.pop('TELEGRAM_FORUM_CHAT', None)
    os.environ.pop('TELEGRAM_CHAT_BTC', None)
    os.environ.pop('TELEGRAM_TOPIC_BTC', None)
    os.environ['TELEGRAM_BOT_TOKEN'] = 'test:token'
    os.environ['TELEGRAM_CHAT_ID'] = '999'
    # 1) тумблер кабінету вимкнено
    tg._cat_enabled = lambda c: False
    r = tg.category_check('btc')
    _check(not r['enabled'] and 'кабінет' in r['reason'],
           f'вимкнений тумблер мусить бути НАЗВАНИЙ: {r}')
    # 2) власного чату немає → приват адміна, а не тема групи
    tg._cat_enabled = lambda c: True
    r = tg.category_check('btc')
    _check(r['route'] == 'admin' and 'ПРИВАТНИЙ' in r['reason'],
           f'відсутність теми мусить бути названа: {r}')
    # 3) Telegram відмовив на самій відправці
    os.environ['TELEGRAM_CHAT_BTC'] = '-100777'
    os.environ['TELEGRAM_TOPIC_BTC'] = '42'
    tg._api = lambda m, p=None: ({'ok': True} if m == 'editForumTopic'
                                 else {'ok': False, 'description': 'CHAT_WRITE_FORBIDDEN'})
    try:
        r = tg.category_check('btc', send_test=True)
        _check(r.get('sent') is False and 'CHAT_WRITE_FORBIDDEN' in r['reason'],
               f'відмову Telegram мусить бути видно дослівно: {r}')
        # 4) успіх — теж однозначний
        tg._api = lambda m, p=None: {'ok': True}
        r = tg.category_check('btc', send_test=True)
        _check(r.get('sent') is True, f'успішна відправка: {r}')
        _check(r['chat'] == '-100777' and r['thread'] == '42',
               f'маршрут мусить бути видний: {r}')
    finally:
        os.environ.pop('TELEGRAM_CHAT_BTC', None)
        os.environ.pop('TELEGRAM_TOPIC_BTC', None)
    print('✓ перевірка називає КОЖНУ з чотирьох мовчазних причин')


def test_ui_has_the_checker_button_and_route_exists():
    _check('mmTgCheck(' in _HTML, 'немає кнопки перевірки теми')
    _check('id="mm-tg-check"' in _HTML, 'немає місця для результату перевірки')
    _check("/api/tg/topic-check" in _HTML, 'кнопка нікуди не шле')
    _fl = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()
    _check("@app.route('/api/tg/topic-check'" in _fl,
           'маршрут перевірки не зареєстровано (урок submitManualTp1)')
    _check('if (!r.ok)' in _HTML[_HTML.find('async function mmTgCheck'):
                                 _HTML.find('async function mmTgCheck') + 1400],
           'без перевірки HTTP-статусу 404 виглядав би як успіх')
    print('✓ кнопка, місце для відповіді і маршрут — на місці')



# ═══ 7. 🏷 НАЗВА ТЕМИ НЕ ЗАЛЕЖИТЬ ВІД ВІДПРАВКИ · 📨 ВІДМОВА ГОВОРИТЬ (22.09) ══
# **Скарга дослівно:** «Тема в телеграм групі знову називається чомусь
# "₿ BTCUSDT" і повідомлень про "✅ КОРЕКЦІЯ ЗАВЕРШИЛАСЬ" не було.»
# Обидва симптоми мали ОДИН корінь: перейменування жило ВСЕРЕДИНІ `_cat_chat`,
# тобто спрацьовувало лише як побічний ефект відправки, а `notify_category`
# виходить на тумблері кабінету ДО `_cat_chat`. Немає повідомлень → назва не
# оновлюється НІКОЛИ, і зовні це виглядає як дві різні поломки.


def test_topic_rename_no_longer_depends_on_sending():
    """Назва теми — властивість ТЕМИ, а не сповіщення."""
    _check(hasattr(tg, 'sync_topic_names'),
           'потрібен окремий синхронізатор назв, не привʼязаний до відправки')
    src = _fn_src(_TG_SRC, 'sync_topic_names')
    _check('_forum_rename_if_needed' in src, 'він мусить саме перейменовувати')
    _check('createForumTopic' not in src,
           'створювати тему для вимкненої категорії НЕ можна — тумблер має '
           'гасити групу, а не наповнювати її')
    _check('_cat_enabled' not in src,
           'тумблер сповіщень назви теми не стосується')
    print('✓ 🏷 назва теми оновлюється незалежно від відправки')


def test_startup_syncs_topic_names():
    """Без виклику на СТАРТІ тиха група назавжди лишалась би зі старою назвою."""
    src = _fn_src(_TG_SRC, 'start_tg_bot')
    _check('sync_topic_names' in src, 'старт бота мусить синхронізувати назви')
    print('✓ 🏷 назви тем синхронізуються на старті бота')


def test_the_health_check_repairs_the_name_even_when_muted():
    """🩺 кнопка мусить лагодити назву ДО раннього виходу на тумблері —
    інакше користувач із вимкненою темою не має ЖОДНОГО способу її оновити."""
    src = _fn_src(_TG_SRC, 'category_check')
    _i_sync = src.find('sync_topic_names')
    _i_gate = src.find("out['enabled']")
    _check(_i_sync > 0, '🩺 перевірка мусить кликати синхронізатор')
    # перший ранній `return` на тумблері йде ПІСЛЯ перевірки `not out['enabled']`
    _i_ret = src.find('return out', src.find("if not out['enabled']"))
    _check(_i_sync < _i_ret,
           'синхронізація мусить стояти ДО виходу на вимкненому тумблері')
    print('✓ 🩺 перевірка теми лагодить назву навіть при вимкненому тумблері')


def test_a_refused_send_is_written_into_the_bot_log():
    """Подія сталась, а Telegram не прийняв — це мусить бути ВИДНО в 🧾 Лозі,
    а не лише в stdout (урок «невидимий збій читається як бот не працює»)."""
    _check(hasattr(tg, 'last_send_error'),
           'потрібен публічний читач причини відмови')
    bc = _fn_src(_FF_SRC, '_broadcast_users')
    # ⚠️ `_fn_src` віддає AST-unparse, тож `return ok, x` нормалізується в
    # `return (ok, ...)` — шукаємо СУТЬ, а не дослівний рядок.
    _check('last_send_error' in bc and 'return (ok' in bc,
           f'_broadcast_users мусить віддавати (ok, причина): {bc[-200:]}')
    for fn in ('_mm_track_correction', '_mm_bias_alert'):
        body = _fn_src(_FF_SRC, fn)
        _check('_broadcast_users' in body and 'if not _ok' in body,
               f'{fn} мусить ЧИТАТИ результат відправки')
        _check('НЕ прийняв' in body, f'{fn} мусить писати причину в 🧾 Лог')
    print('✓ 📨 відмова Telegram більше не мовчазна')


def test_a_refused_send_really_reaches_the_log():
    """Не лише текст у коді — прогін: відмова мусить дати рядок у 🧾 Лозі."""
    _LOGGED.clear()
    ff = _mk(trends={f'C{i}': 'SHORT' for i in range(6)}, mm_corr_confirm_sec=0)
    ff.tg_ok = False
    _tick(ff, _snap(**{f'C{i}': ('LONG', 50, 'down') for i in range(6)}))
    _bad = [m for m in _LOGGED if 'НЕ прийняв' in str(m.get('detail'))]
    _check(_bad, f'рядок про відмову мусить бути в лозі: {_LOGGED}')
    _check('CHAT_WRITE_FORBIDDEN' in str(_bad[-1]['detail']),
           f'і нести ДОСЛІВНУ причину: {_bad[-1]}')
    _LOGGED.clear()
    ff2 = _mk(trends={f'C{i}': 'SHORT' for i in range(6)}, mm_corr_confirm_sec=0)
    _tick(ff2, _snap(**{f'C{i}': ('LONG', 50, 'down') for i in range(6)}))
    _check(not [m for m in _LOGGED if 'НЕ прийняв' in str(m.get('detail'))],
           'успішна відправка зайвих рядків не пише')
    print('✓ 📨 прогін: відмова дає рядок у 🧾 Лозі, успіх — ні')



# ═══ 8. 🩺 ЗВІТ ПЕРЕВІРКИ НЕ БРЕШЕ (реальний прогін на проді, 22.09) ═════════
# Перша ж перевірка кнопкою віддала:
#   тема «🧮 МММ-монітор» · форум-група · чат … · тема #42 · тумблер ✓ ·
#   ✅ повідомлення надіслано · перейменування: trades: Bad Request:
#   TOPIC_NOT_MODIFIED
# Тобто канал ПРАЦЮЄ, а «помилка» — подвійний дефект МОГО ж коду.


def test_topic_not_modified_is_success_not_a_failure():
    """«Назва вже така, як треба» — це УСПІХ.

    Telegram відмовляє на editForumTopic, коли міняти нічого. Рахуючи це
    збоєм, ми (а) писали в 🩺 звіт неіснуючу проблему і (б) НЕ запамʼятовували
    назву, тобто били в API на КОЖНІЙ синхронізації — вічно."""
    src = _fn_src(_TG_SRC, '_forum_rename_if_needed')
    _check('TOPIC_NOT_MODIFIED' in src, 'цей код відповіді мусить оброблятись')
    _i_ok = src.find('TOPIC_NOT_MODIFIED')
    _i_cache = src.find('_forum_names_cache[ckey] = want')
    _check(0 < _i_ok < _i_cache,
           'перевірка мусить стояти ДО запамʼятовування назви — інакше '
           'наступна синхронізація знову піде в API')
    print('✓ 🏷 TOPIC_NOT_MODIFIED = назва вже правильна, а не збій')


def test_the_check_reports_this_categorys_error_not_someone_elses():
    """🩺 перевірка ОДНІЄЇ теми мусить казати ПРО НЕЇ.

    Глобальний рядок помилки давав звіт про «🧮 МММ-монітор» із причиною від
    категорії `trades` — тобто перевірка звинувачувала не ту тему."""
    _check(hasattr(tg, 'rename_error'), 'потрібен читач причини ПО КАТЕГОРІЇ')
    _check(isinstance(tg._forum_rename_err, dict),
           'причини мусять зберігатись ПО КАТЕГОРІЯХ, а не одним рядком')
    tg._forum_rename_err.clear()
    tg._forum_rename_err['trades'] = 'Bad Request: TOPIC_NOT_MODIFIED'
    _check(tg.rename_error('btc') == '',
           'чужа помилка НЕ має зʼявлятись у звіті про btc')
    _check('TOPIC_NOT_MODIFIED' in tg.rename_error('trades'),
           'своя — має')
    tg._forum_rename_err.clear()
    src = _fn_src(_TG_SRC, 'category_check')
    _check('_forum_rename_err' not in src,
           'category_check мусить читати через rename_error(category), а не '
           'глобальний рядок')
    _check(src.count('rename_error(category)') >= 3,
           f'усі три місця звіту: {src.count("rename_error(category)")}')
    print('✓ 🩺 звіт називає помилку САМЕ своєї теми')


def test_a_successful_rename_clears_only_its_own_error():
    """Успіх по одній категорії не має стирати причину по іншій."""
    tg._forum_rename_err.clear()
    tg._forum_rename_err['trades'] = 'бот не адмін теми'
    tg._forum_rename_err['btc'] = 'старе'
    tg._forum_rename_err.pop('btc', None)      # те, що робить успішний шлях
    _check(tg.rename_error('trades') == 'бот не адмін теми',
           'чужа причина мусить лишитись')
    tg._forum_rename_err.clear()
    print('✓ 🏷 успіх стирає причину лише своєї категорії')


def test_correction_tg_text_is_short_colored_and_timed():
    """Вимога 24.09: «🔴 SHORT КОРЕКЦІЯ · почалась <дата час>» і «… ЗАВЕРШИЛАСЬ
    (тривала …)» з коректним часом. Напрямок — ПРОТИЛЕЖНИЙ банеру."""
    import datetime as _dt
    from zoneinfo import ZoneInfo
    t0 = _dt.datetime(2026, 9, 22, 23, 20, tzinfo=ZoneInfo('Europe/Kyiv')).timestamp()
    t1 = t0 + 40 * 3600 + 47 * 60
    s = _ffm.corr_tg_text('start', 'LONG', t0)
    _check(s == '🔴 SHORT КОРЕКЦІЯ · почалась 22.09 23:20', s)
    s = _ffm.corr_tg_text('start', 'SHORT', t0)
    _check(s.startswith('🟢 LONG КОРЕКЦІЯ'), s)
    e = _ffm.corr_tg_text('end', 'LONG', t0, t1, t1 - t0, FF._fmt_wait)
    _check(e.startswith('🔴 SHORT КОРЕКЦІЯ ЗАВЕРШИЛАСЬ (тривала ')
           and '(тривала 1д 16г 47хв)' in e
           and e.endswith('22.09 23:20 → 24.09 16:07'), e)
    print('✓ 📨 TG корекції: коротко, колір протилежного боку, дата й час')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for t in tests:
        t()
    print(f'\nAll Telegram-routing tests passed ✓ ({len(tests)} tests)')
