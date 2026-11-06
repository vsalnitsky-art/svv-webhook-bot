"""💾 Що ще губилось на рестарті (аналіз 25.10) — і як це закрито.

1. 💰 Funding-VOB: перший показ після рестарту приймав будь-який наявний блок
   за НОВИЙ сигнал → тиха база + персист опрацьованих блоків.
2. 🧱 Opposite OB Exit: база блоку на вході фіксувалась заново з ПОТОЧНОГО
   блоку → протилежний OB, що зʼявився під час простою, губився. База тепер
   живе на позиції (а позиції персистяться).
3. 🔮 Прогноз 1H/4H: порожній до проходу сканера → знімок кешу рушія.
4. PD-зона: рахується по закритих барах → до закриття бару відновлюється ТОЧНО.
5. 🎯 «Готовність»: не персиститься (TTL 90с), але монети Черги-4 тепер
   перераховуються першими.
"""
import importlib.util
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import test_mm_monitor as _mon   # noqa: E402  (ізольований fuel_filter)
import test_autosl_chain as _tmb  # noqa: E402  (ізольований trade_manager)

_ff = _mon._m
FF = _ff.FuelFilterDaemon
TM = _tmb.TM
_FF_SRC = open(os.path.join(_HERE, 'detection', 'fuel_filter.py'), encoding='utf-8').read()
_SC_SRC = open(os.path.join(_HERE, 'detection', 'smc_scanner.py'), encoding='utf-8').read()


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


fe_mod = _load('forecast_engine_rs', 'detection/forecast_engine.py')
sm = _load('smc_scanner_rs', 'detection/smc_scanner.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


class _DB:
    def __init__(self):
        self.d = {}

    def get_setting(self, k, default=None):
        return self.d.get(k, default)

    def set_setting(self, k, v):
        self.d[k] = v


# ═══════════ 1. 💰 FUNDING-VOB — ВИДАЛЕНО 06.11 ═══════════════════════════
# Старий блок «💰 Funding — МММ» (і його VOB→Черга-4) видалено повністю —
# переживати рестарт нема чому. Замок: від нього не лишилось ні правила, ні стану.
def test_funding_vob_is_gone():
    _check(not hasattr(_ff, 'funding_vob_outcome'), 'правило funding-VOB видалено')
    _check('funding_vob_seen' not in _FF_SRC and '_vob_seen' not in _FF_SRC,
           'стану опрацьованих funding-VOB більше немає')
    print('✓ funding-VOB видалено разом зі старим блоком')


# ═══════════ 2. 🧱 OPPOSITE OB EXIT ══════════════════════════════════════
class _TmStub:
    def __init__(self):
        self._opp_ob_base = {}
        self.persisted = []

    def _exit_key(self, symbol, is_shadow):
        return f"{symbol.upper()}|{'S' if is_shadow else 'R'}"

    def _persist_book(self, is_shadow):
        self.persisted.append(is_shadow)


def test_opposite_ob_baseline_survives_a_restart():
    pos = {'side': 'SHORT', 'opened_at': 1000.0}
    a = _TmStub()
    _check(TM._opp_ob_baseline(a, 'BTCUSDT', pos, '15m', {'bar_time': 500}, True) is None,
           'перший прохід лише фіксує базу')
    _check(pos['opp_ob_base']['bar_time'] == 500 and a.persisted == [True],
           'база лягла на позицію і позицію збережено')
    b = _TmStub()                                    # «рестарт»: памʼять порожня
    base = TM._opp_ob_baseline(b, 'BTCUSDT', pos, '15m', {'bar_time': 900}, True)
    _check(base is not None and base['bar_time'] == 500,
           'після рестарту — стара база, а блок 900 з простою лишається ПОДІЄЮ')
    base['bar_time'] = 900
    _check(pos['opp_ob_base']['bar_time'] == 900, 'памʼять і позиція — один обʼєкт')
    print('✓ Opposite OB: протилежний блок, що зʼявився під час простою, не губиться')


def test_a_new_trade_gets_a_fresh_baseline():
    pos = {'side': 'LONG', 'opened_at': 2000.0,
           'opp_ob_base': {'key': 'SHORT|1000.0|15m', 'bar_time': 500}}
    s = _TmStub()
    _check(TM._opp_ob_baseline(s, 'BTCUSDT', pos, '15m', {'bar_time': 700}) is None,
           'чужа угода / інший TF — база фіксується заново')
    _check(pos['opp_ob_base']['bar_time'] == 700, pos)
    print('✓ Opposite OB: нова угода чи інший TF починає з чистої бази')


# ═══════════ 3. 🔮 ПРОГНОЗ ═══════════════════════════════════════════════
def _fc(ts):
    return {'symbol': 'BTCUSDT', 'computed_at': ts,
            'forecast_1h': {'side': 1, 'pct': 40, 'conf': 75},
            'forecast_4h': {'side': -1, 'pct': -12, 'conf': 50},
            'ctr': {'stc': 22.5}}


def test_forecast_cache_survives_a_restart():
    db = _DB()
    now = time.time()
    a = fe_mod.ForecastEngine()
    a._cache = {'BTCUSDT': _fc(now - 120), 'ETHUSDT': _fc(now - 60)}
    _check(a.persist(db) == 2, 'знімок записано')
    b = fe_mod.ForecastEngine()
    _check(b.restore(db, now=now) == 2, 'відновлено')
    c = b.get('BTCUSDT')
    _check(c['forecast_1h']['pct'] == 40 and c['restored'], c)
    _check(abs(c['computed_at'] - (now - 120)) < 1e-6,
           'computed_at ОРИГІНАЛЬНИЙ — ensure_fresh перерахує за звичним правилом')
    print('✓ прогноз 1H/4H відновлюється одразу після рестарту')


def test_old_forecast_is_not_restored_and_fresh_wins():
    db = _DB()
    now = time.time()
    a = fe_mod.ForecastEngine()
    a._cache = {'BTCUSDT': _fc(now - fe_mod.ForecastEngine.RESTORE_MAX_AGE - 5),
                'ETHUSDT': _fc(now - 30)}
    a.persist(db)
    b = fe_mod.ForecastEngine()
    b._cache['ETHUSDT'] = _fc(now)                  # уже порахований цим запуском
    _check(b.restore(db, now=now) == 0, 'давній не відновлюємо, свіжий не перебиваємо')
    _check(b.get('BTCUSDT') is None and not b.get('ETHUSDT').get('restored'), b._cache)
    print('✓ давній прогноз не відновлюється, свіжий не перебивається')


# ═══════════ 4. PD-ЗОНА ═════════════════════════════════════════════════
def _sc(db, tf='1h', wl=('BTCUSDT', 'ETHUSDT')):
    s = sm.SMCScanner.__new__(sm.SMCScanner)
    s.db = db
    s._lock = threading.RLock()
    s._settings = {'pd_zone_timeframe': tf}
    s._watchlist = list(wl)
    s._pd_zone_cache = {}
    return s


def test_pd_zone_restores_only_while_its_bar_is_open():
    db = _DB()
    now = time.time()
    a = _sc(db)
    a._pd_zone_cache = {
        'BTCUSDT': {'pct': 82.0, 'tf': '1h', 'updated_at': now, 'until': now + 600},
        'ETHUSDT': {'pct': 12.0, 'tf': '1h', 'updated_at': now, 'until': now - 5},
    }
    a._persist_derived_state()
    b = _sc(db)
    b._load_derived_state()
    _check(b.get_pd_pct('BTCUSDT') == 82.0, 'бар не закрився — PD точна')
    _check(b.get_pd_pct('ETHUSDT') is None, 'бар закрився — чекаємо скану')
    c = _sc(db, tf='4h')
    c._load_derived_state()
    _check(c.get_pd_pct('BTCUSDT') is None, 'інший pd_zone_timeframe — не наш')
    print('✓ PD-зона відновлюється лише поки її бар відкритий')


def test_scanner_wiring():
    _check("'until': _pd_until" in _SC_SRC, 'PD несе момент закриття бару')
    i = _SC_SRC.index('self._watchlist = self._load_watchlist()')
    _check(_SC_SRC.index('self._load_derived_state()') > i, 'завантаження ПІСЛЯ watchlist')
    _check('self._persist_derived_state()' in _SC_SRC, 'запис у кінці циклу')
    print('✓ провідка сканера')


# ═══════════ 5. 🎯 «ГОТОВНІСТЬ» ══════════════════════════════════════════
def test_queue4_coins_are_graded_first():
    f = FF.__new__(FF)
    f._setup_at, f._setup_cache, f._exit_cache = {}, {}, {}
    f._pending4 = {'ZECUSDT': {}}
    done = []
    f._compute_setup = lambda sym, d, s: done.append(sym) or {'score': 50}
    targets = {s: ('LONG',) for s in ('AAAUSDT', 'BBBUSDT', 'ZECUSDT')}
    FF._refresh_setup_cache(f, {'setup_max_per_cycle': 1, 'setup_ttl': 90}, targets)
    _check(done == ['ZECUSDT'], f'Черга-4 першою: {done}')
    print('✓ після рестарту «Готовність» спершу для монет Черги-4')


if __name__ == '__main__':
    fns = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'\nOK — {len(fns)} тестів')
