"""💾 МММ-КНИГИ ПЕРЕЖИВАЮТЬ РЕСТАРТ (вимога 22.10) + 🧹 автоочистка БД.

Дослівно: «Після кожного рестарту бота, більшість банерів дуже довго
відновлюються. Є можливість зберігати їх стан, щоб після рестарту лише
обновити інформацію, а не чекати поки все відновиться?» + «Проконтролюй, що
автоочистка баз теж все відпрацьовувала.»
"""
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter + стаби)

_m = _mon._m
_HTML = _mon._HTML
_SRC = _mon._SRC
_FLASK = open(os.path.join(_HERE, 'web', 'flask_app.py'), encoding='utf-8').read()


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _DB:
    def __init__(self):
        self.store = {}

    def get_setting(self, k, d=None):
        return self.store.get(k, d)

    def set_setting(self, k, v):
        import json
        self.store[k] = json.loads(json.dumps(v))   # як справжня БД: лише JSON
        return True


def _mk(db):
    ff = _mon._mk()
    ff._db = db
    ff._mm_books_load()
    return ff


def _boot(db):
    """«Рестарт»: новий демон над тією самою БД."""
    return _mk(db)


# ═══════════ 1. ЗНІМКИ ДЛЯ ПОКАЗУ ═══════════════════════════════════════
def test_tables_come_back_after_restart_marked_restored():
    db = _DB()
    ff = _mk(db)
    _mon._cap(ff, BTCUSDT=0.42, ETHUSDT=-0.30)
    _check(ff._mm_books_persist(time.time(), force=True), 'запис')
    ff2 = _boot(db)
    st = ff2.mm_monitor_state(book='old')
    syms = sorted(r['symbol'] for r in st['rows'])
    _check(syms == ['BTCUSDT', 'ETHUSDT'], syms)
    _check(st['restored'] is True and st['ts'] > 0, st['restored'])
    _check(not st.get('pending'), 'з відновленим знімком «⏳ рахується» не потрібне')
    print('✓ після рестарту таблиця одразу з рядками, позначка restored')


def test_trading_paths_never_see_the_restored_snapshot():
    db = _DB()
    ff = _mk(db)
    _mon._cap(ff, BTCUSDT=0.05)                      # ⚖ рівновага
    ff._mm_books_persist(time.time(), force=True)
    ff2 = _boot(db)
    _check(ff2._mm_snapshot == {}, 'живий знімок порожній до першого такту')
    _check(ff2._mmn_snapshot == {}, 'МММ-new теж')
    _check(ff2.mm_snapshot_for(['BTCUSDT']) == {},
           'правило «🧮 ⚖ → вихід» не має закрити угоду за числом минулого запуску')
    print('✓ ворота / колонки угод / правила виходу відновлений знімок НЕ бачать')


def test_first_fresh_capture_replaces_the_restored_one():
    db = _DB()
    ff = _mk(db)
    _mon._cap(ff, BTCUSDT=0.42)
    ff._mm_books_persist(time.time(), force=True)
    ff2 = _boot(db)
    ff2._clock = [time.time()]
    _mon._cap(ff2, SOLUSDT=0.50)
    st = ff2.mm_monitor_state(book='old')
    _check(st['restored'] is False, 'свіжий такт знімає позначку')
    _check([r['symbol'] for r in st['rows']] == ['SOLUSDT'], st['rows'])
    _check('old' not in ff2._mm_books_shown, 'відновлений знімок забуто')
    print('✓ перший свіжий такт замінює відновлений знімок')


def test_disabled_book_does_not_show_restored_rows():
    db = _DB()
    ff = _mk(db)
    _mon._cap(ff, BTCUSDT=0.42)
    ff._mm_books_persist(time.time(), force=True)
    ff2 = _boot(db)
    ff2._settings['mm_monitor_enabled'] = False
    st = ff2.mm_monitor_state(book='old')
    _check(st['rows'] == [] and st['restored'] is False, st)
    print('✓ вимкнений блок відновлених рядків не показує')


def test_persist_is_throttled_and_single_row():
    db = _DB()
    ff = _mk(db)
    _mon._cap(ff, BTCUSDT=0.42)
    t0 = ff._mm_books_saved_at
    _check(t0 > 0, 'такт сам пише блоб')
    _check(ff._mm_books_persist(t0 + 10) is False, 'частіше за хвилину — ні')
    _check(ff._mm_books_persist(t0 + _m.MM_BOOKS_PERSIST_SEC + 1) is True, 'раз на хвилину')
    _check(list(db.store) == [_m._DB_MM_BOOKS], db.store.keys())
    print('✓ запис раз на MM_BOOKS_PERSIST_SEC, ОДИН ключ (рядків у БД не додається)')


def test_old_blob_is_ignored_and_future_blob_distrusted():
    db = _DB()
    db.store[_m._DB_MM_BOOKS] = {'ts': time.time() - _m.MM_BOOKS_SHOW_MAX_AGE - 10,
                                 'books': {'old': {'snap': {'BTCUSDT': {'status': 'LONG'}},
                                                   'ts': 1.0}}}
    _check(_mk(db)._mm_books_shown == {}, 'старіше за 6 год — не показуємо')
    db.store[_m._DB_MM_BOOKS]['ts'] = time.time() + 3600
    _check(_mk(db)._mm_books_shown == {}, 'час із майбутнього — не довіряємо')
    print('✓ задавній блоб і блоб «із майбутнього» ігноруються')


# ═══════════ 2. СИРІ КЕШІ ДЖЕРЕЛ ═════════════════════════════════════════
class _FE:
    """Двигун прогнозу після рестарту: кеш STC порожній, кожен запит — рахуємо."""
    def __init__(self):
        self.fetched = []

    def peek_ctr_tf(self, s, tf):
        return None

    def get_ctr_tf(self, s, tf):
        self.fetched.append((s, tf))
        return {'stc': 10.0}


def test_ctr_readings_survive_and_refresh_within_budget():
    db = _DB()
    ff = _mk(db)
    now = time.time()
    ff._mmc_rd_last = {f'C{i}USDT': {'1h': [now - 100, 20.0], '4h': [now - 100, 30.0]}
                       for i in range(30)}
    ff._mm_books_persist(now, force=True)
    ff2 = _boot(db)
    fe = _FE()
    rd = ff2._ctr_mm_readings([f'C{i}USDT' for i in range(30)], fe)
    _check(len(rd) == 30, f'усі 30 монет одразу мають STC, а не 5: {len(rd)}')
    _check(len(fe.fetched) == _m.CTR_MM_FETCH_PER_TICK, 'бюджет запитів не перевищено')
    print('✓ ⚡ STC 1H/4H переживає рестарт: усі монети одразу, оновлення — в межах бюджету')


def test_decision_cache_survives_with_its_own_age():
    db = _DB()
    ff = _mk(db)
    t = time.time() - 50
    ff._mmd_dec = {'BTCUSDT': (t, {'recommended': 'SHORT', 'prob_long': 0.15,
                                   'prob_short': 0.85, 'verdict': 'good'})}
    ff._mm_books_persist(time.time(), force=True)
    ff2 = _boot(db)
    _check(abs(ff2._mmd_dec['BTCUSDT'][0] - t) < 1, 'вік вердикту збережено (черга оновлення за віком)')
    _check(ff2._mmd_dec['BTCUSDT'][1]['recommended'] == 'SHORT', ff2._mmd_dec)
    print('✓ 🧠 кеш вердиктів Decision переживає рестарт зі СВОЇМ часом')


def test_pull_scan_survives_and_is_not_repeated_while_fresh():
    db = _DB()
    ff = _mk(db)
    now = time.time()
    s = {'mmp_interval_min': 15}
    cfg = ff._mmp_cfg(s)
    ff._mmp_scan = {'rows': {'BTCUSDT': {'ok': True, 'above_pct': 70, 'below_pct': 30,
                                         'rows': [{'big': 1}]}},
                    'ts': now - 60, 'cfg': cfg, 'ok': True}
    ff._mmp_scan_at = now - 60
    ff._mm_books_persist(now, force=True)
    blob = db.store[_m._DB_MM_BOOKS]
    _check('rows' not in blob['pull_scan']['rows']['BTCUSDT'], 'драбину в блоб не пишемо')
    ff2 = _boot(db)
    launched = []
    ff2._mmp_scan_launch = lambda *a: launched.append(a)
    _check(ff2._mmp_scan_tick(s, time.time()) is False and not launched,
           'свіжий відновлений скан не повторюється')
    _check(ff2._mmp_scan['cfg'] == cfg, 'cfg відновлено як tuple')
    print('✓ 🧭 результат скану Pull переживає рестарт, повторний скан не потрібен')


def test_raw_caches_have_a_tighter_age_limit():
    db = _DB()
    ff = _mk(db)
    now = time.time()
    _mon._cap(ff, BTCUSDT=0.42)
    ff._mmd_dec = {'BTCUSDT': (now, {'recommended': 'LONG', 'prob_long': 0.8,
                                     'prob_short': 0.2})}
    ff._mm_books_persist(now, force=True)
    db.store[_m._DB_MM_BOOKS]['ts'] = now - _m.MM_BOOKS_RAW_MAX_AGE - 60
    ff2 = _boot(db)
    _check(ff2._mmd_dec == {}, 'старі вердикти в РОЗРАХУНОК не йдуть')
    _check('old' in ff2._mm_books_shown, 'а таблиця для показу — так')
    print('✓ сирі кеші (у розрахунок) — ≤30 хв, знімок для показу — ≤6 год')


def test_forecast_fallback_only_right_after_boot():
    db = _DB()
    ff = _mk(db)
    now = time.time()
    pair = {'f1': {'side': -1, 'pct': -12.0, 'conf': 50}, 'f4': None}
    _check(ff._mm_fc_remember('BTCUSDT', pair, now) is pair, 'живий прогноз')
    ff._mm_books_persist(now, force=True)
    ff2 = _boot(db)
    empty = {'f1': None, 'f4': None}
    got = ff2._mm_fc_remember('BTCUSDT', empty, time.time())
    _check(got.get('fc_stale') and got['f1']['side'] == -1, got)
    late = ff2._boot_ts + _m.MM_FC_RESTORE_SEC + 1
    _check(ff2._mm_fc_remember('BTCUSDT', empty, late) == empty,
           'після вікна старту застарілий прогноз не підставляємо')
    print('✓ 🔮 відновлена пара прогнозу — лише перші 15 хв, позначена fc_stale')


def test_capture_calls_persist():
    src = _mon._fn_src('_mm_capture') if hasattr(_mon, '_fn_src') else _SRC
    _check('_mm_books_persist(_now)' in src, 'такт пише блоб')
    _check('self._mm_books_load()' in _SRC, '__init__ відновлює блоб')
    print('✓ такт пише блоб, старт його читає')


# ═══════════ 3. СТОРІНКА ═════════════════════════════════════════════════
def test_page_shows_the_restored_marker():
    _check('restored: !!(mm && mm.restored)' in _HTML, 'поле доїжджає в B.meta')
    _check('з минулого запуску' in _HTML, 'шапка каже, що числа не свіжі')
    print('✓ шапка таблиці: «↻ … з минулого запуску»')


# ═══════════ 4. АВТООЧИСТКА БД ═══════════════════════════════════════════
def _func(src, name):
    i = src.index(f'def {name}(')
    j = src.index('\ndef ', i + 1)
    return src[i:j]


def test_cleanup_uses_one_transaction_per_table():
    f = _func(_FLASK, '_db_service_cleanup')
    loop = f[f.index('for tbl'):]
    _check('engine.begin()' in loop, 'кожна таблиця — своя транзакція')
    _check('conn.commit()' not in f, 'спільного commit наприкінці більше немає')
    print('✓ автоочистка: помилка однієї таблиці більше не відкочує всю чистку')


def test_vacuum_covers_the_settings_table():
    f = _func(_FLASK, '_db_vacuum_full')
    _check("'sob_bot_settings'" in f, 'VACUUM прибирає мертві версії блобів')
    print('✓ VACUUM охоплює sob_bot_settings (блоби стану перезаписуються)')


if __name__ == '__main__':
    n = 0
    for k, v in list(globals().items()):
        if k.startswith('test_') and callable(v):
            v()
            n += 1
    print(f'\nOK — {n} тестів')
