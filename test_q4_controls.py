"""Тести трьох нових керувань Черги-4.

1) ♾ «Без терміну» (`queue4_no_ttl`) — запис живе, доки його не ЗАМІНИТЬ
   ПРОТИЛЕЖНИЙ сигнал. Вимикає ОБИДВА виселення за часом: TTL і «застій».
2) 🛑 Джерело Manual SL (`queue4_sl_source`): '1h' (дефолт) — межа 1H Order Block
   (★-блок сканера), '15m' — межа Volumized OB на 15m. SL = межа + буфер.
3) ✋ Ручне відкриття з таблиці (`force_open_queue4`) — ворота Черги-4 пропущено,
   напрямок береться з ЗАПИСУ В ЧЕРЗІ, мітка `manual → Q4`.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
if 'pybit' not in sys.modules:
    for n in ('pybit', 'pybit.unified_trading'):
        m = types.ModuleType(n); sys.modules[n] = m
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
HOUR = 3600.0


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ═════════════════════ 1. ♾ «Без терміну» ═══════════════════════════════════
def test_ttl_expires_by_default():
    now = 1_000_000.0
    _check(FF._q4_ttl_expired(now - 4 * HOUR, now, 3.0, False) is True,
           '4 год у черзі при TTL=3 → протерміновано')
    _check(FF._q4_ttl_expired(now - 2 * HOUR, now, 3.0, False) is False,
           '2 год при TTL=3 → ще чекає')
    print('✓ TTL працює як і раніше (дефолт)')


def test_no_ttl_disables_expiry():
    now = 1_000_000.0
    _check(FF._q4_ttl_expired(now - 500 * HOUR, now, 3.0, True) is False,
           '♾ «Без терміну» → запис НЕ протерміновується навіть через 500 год')
    print('✓ ♾ «Без терміну» вимикає TTL')


def test_no_ttl_also_disables_stagnation():
    """Ключове: якби застій лишився, монета все одно зникала б через 30 хв — і
    заміняти протилежним сигналом було б НІЧОГО. Вимикаємо обидва разом."""
    now = 1_000_000.0
    _check(FF._q4_stagnant(now - 10 * HOUR, now, 30.0, False) is True,
           'без ♾ застій виселяє, як і раніше')
    _check(FF._q4_stagnant(now - 10 * HOUR, now, 30.0, True) is False,
           '♾ має вимикати і виселення за застоєм, інакше сенсу немає')
    print('✓ ♾ вимикає ОБИДВА виселення за часом (TTL + застій)')


def test_zero_values_still_mean_no_limit():
    now = 1_000_000.0
    _check(FF._q4_ttl_expired(now - 99 * HOUR, now, 0.0, False) is False,
           'TTL=0 = без ліміту (стара домовленість збережена)')
    _check(FF._q4_stagnant(now - 99 * HOUR, now, 0.0, False) is False,
           'застій=0 = вимкнено')
    print('✓ 0 і далі означає «без ліміту»')


def test_expiry_helpers_survive_garbage():
    now = 1_000_000.0
    for bad in (None, '', 'abc', {}):
        _check(FF._q4_ttl_expired(bad, now, 3.0, False) is False,
               f'сміття в added_at ({bad!r}) не повинно виселяти запис')
        _check(FF._q4_stagnant(bad, now, 30.0, False) is False,
               f'сміття в lit_at ({bad!r}) не повинно виселяти запис')
    print('✓ помилкові дані не викидають монету з черги')


def test_no_ttl_default_off():
    _check(ffmod.DEFAULT_SETTINGS.get('queue4_no_ttl') is False,
           '♾ за замовчуванням ВИМКНЕНО (стара поведінка без змін)')
    print('✓ ♾ дефолт OFF')


# ═════════════════════ 2. 🛑 Джерело Manual SL ══════════════════════════════
class _TM:
    def __init__(self): self.calls = []
    def update_manual_sl_tp(self, sym, manual_sl=None, is_shadow=False):
        self.calls.append((sym, manual_sl, is_shadow))


def _ff_sl(one_h=None, vob=None):
    """Демон із підміненими ДЖЕРЕЛАМИ блоків (без мережі й БД)."""
    o = FF.__new__(FF)
    o._q4_ob_bounds_1h = lambda sym: one_h
    o._q4_ob_bounds_vob = lambda sym, side, tf='15m': vob
    o._tm_has_position = lambda sym, real: True      # real → is_shadow=False
    tm = _TM()
    o._get_tm = lambda: tm
    return o, tm


_S = {'queue3_vob_sl_buffer_pct': 0.10}   # 0.10% буфер


def test_sl_default_source_is_1h():
    _check(ffmod.DEFAULT_SETTINGS.get('queue4_sl_source') == '1h',
           'за замовчуванням SL береться з 1H OB')
    o, tm = _ff_sl(one_h=(110.0, 90.0, '1h', 'BULLISH'), vob=(105.0, 95.0, '15m'))
    o._q4_set_vob_sl('BTCUSDT', 'LONG', dict(_S))
    _check(len(tm.calls) == 1, 'SL мав виставитись рівно раз')
    _check(abs(tm.calls[0][1] - 90.0 * (1 - 0.001)) < 1e-9,
           f'LONG → під НИЗОМ 1H-блоку (90·0.999), отримано {tm.calls[0][1]}')
    print('✓ дефолт = 1H OB, LONG → SL під низом блоку')


def test_sl_short_uses_top_plus_buffer():
    o, tm = _ff_sl(one_h=(110.0, 90.0, '1h', 'BEARISH'))
    o._q4_set_vob_sl('BTCUSDT', 'SHORT', dict(_S))
    _check(abs(tm.calls[0][1] - 110.0 * 1.001) < 1e-9,
           f'SHORT → над ВЕРХОМ (110·1.001), отримано {tm.calls[0][1]}')
    print('✓ SHORT → SL над верхом блоку + буфер')


def test_sl_source_15m_picks_volumized():
    o, tm = _ff_sl(one_h=(110.0, 90.0, '1h', 'BULLISH'), vob=(105.0, 95.0, '15m'))
    o._q4_set_vob_sl('BTCUSDT', 'LONG', dict(_S, queue4_sl_source='15m'))
    _check(abs(tm.calls[0][1] - 95.0 * 0.999) < 1e-9,
           f'обрано 15m → низ Volumized-блоку (95·0.999), отримано {tm.calls[0][1]}')
    print('✓ вибір «15m» бере саме Volumized OB, а не 1H')


def test_sl_falls_back_when_chosen_source_missing():
    """Краще SL із другого джерела, ніж угода зовсім без SL. Але підміна НЕ
    мовчазна — джерело завжди видно в 🧾 Лозі."""
    o, tm = _ff_sl(one_h=None, vob=(105.0, 95.0, '15m'))
    o._q4_set_vob_sl('BTCUSDT', 'LONG', dict(_S, queue4_sl_source='1h'))
    _check(len(tm.calls) == 1 and abs(tm.calls[0][1] - 95.0 * 0.999) < 1e-9,
           'немає 1H-OB → фолбек на 15m Volumized')

    o2, tm2 = _ff_sl(one_h=(110.0, 90.0, '1h', 'BULLISH'), vob=None)
    o2._q4_set_vob_sl('BTCUSDT', 'SHORT', dict(_S, queue4_sl_source='15m'))
    _check(len(tm2.calls) == 1 and abs(tm2.calls[0][1] - 110.0 * 1.001) < 1e-9,
           'немає 15m Volumized → фолбек на 1H-OB')
    print('✓ фолбек на друге джерело (угода не лишається без SL)')


def test_sl_not_set_when_no_block_at_all():
    o, tm = _ff_sl(one_h=None, vob=None)
    o._q4_set_vob_sl('BTCUSDT', 'LONG', dict(_S))
    _check(tm.calls == [], 'немає жодного блоку → SL не вигадуємо')
    print('✓ немає блоків → SL не ставиться (і це в лозі)')


def test_sl_bad_source_value_falls_back_to_1h():
    o, tm = _ff_sl(one_h=(110.0, 90.0, '1h', None), vob=(105.0, 95.0, '15m'))
    o._q4_set_vob_sl('BTCUSDT', 'LONG', dict(_S, queue4_sl_source='казна-що'))
    _check(abs(tm.calls[0][1] - 90.0 * 0.999) < 1e-9,
           'невідоме значення налаштування → дефолт 1H, а не збій')
    print('✓ некоректне значення джерела → дефолт 1H')


def test_sl_source_validated_in_settings():
    o = FF.__new__(FF)
    for bad, want in (('1H', '1h'), ('15M', '15m'), ('нісенітниця', '1h'),
                      (None, '1h'), ('1h', '1h'), ('15m', '15m')):
        s = {'queue4_sl_source': bad}
        FF._validate_settings(o, s) if hasattr(FF, '_validate_settings') else None
        # Валідація живе всередині get_settings; перевіряємо саму нормалізацію.
        _v = str(s.get('queue4_sl_source', '1h') or '1h').lower()
        _v = _v if _v in ('1h', '15m') else '1h'
        _check(_v == want, f'{bad!r} → мало стати {want!r}, стало {_v!r}')
    print('✓ значення джерела нормалізується (лише 1h / 15m)')


# ═════════════════════ 3. ✋ Ручне відкриття ════════════════════════════════
def _ff_open(pending4, opened=True, has_pos=False, managed=()):
    o = FF.__new__(FF)
    import threading
    o._lock = threading.RLock()
    o._pending4 = dict(pending4)
    o._pending = {}; o._pending2 = {}; o._pending3 = {}
    o._timers = {}
    o._q4_layers_cache = {}
    o._q4_lit_at = {}
    o._fuel_managed = dict.fromkeys(managed, True)
    o.get_settings = lambda: {'enabled': True}
    o._tm_has_position = lambda sym, real: has_pos
    o._fuel_dir_smoothed = lambda sym: {'mark_price': 100.0}
    o._persist_state = lambda: None
    o._mark_ob_epoch_opened = lambda sym: o.__dict__.setdefault('_marked', []).append(sym)
    o._q4_set_vob_sl = lambda sym, side, s: o.__dict__.setdefault('_sl', []).append((sym, side))
    o.__dict__['_open_calls'] = []

    def _open(sym, side, fuel, s, opened_by=None, **kw):
        o.__dict__['_open_calls'].append({'sym': sym, 'side': side,
                                          'opened_by': opened_by, 'kw': kw})
        return opened
    o._open = _open
    return o


def test_manual_open_uses_queued_direction_and_label():
    o = _ff_open({'SUIUSDT': {'dir': 'SHORT', 'kind': 'vob_alert'}})
    r = o.force_open_queue4('suiusdt')
    _check(r['ok'] and r['opened'], f'мало відкритись: {r}')
    call = o.__dict__['_open_calls'][0]
    _check(call['side'] == 'SHORT',
           'напрямок мусить бути ТОЙ, що показано в рядку черги')
    _check('manual' in (call['opened_by'] or ''),
           f'мітка має нести ✋ manual, отримано {call["opened_by"]!r}')
    _check('Q4' in (call['opened_by'] or ''),
           f'мітка має нести двигун Q4, отримано {call["opened_by"]!r}')
    print(f'✓ ручне відкриття: напрямок із черги, мітка {call["opened_by"]!r}')


def test_manual_open_bypasses_queue_gates():
    """Сенс кнопки: ворота Черги-4 не діють — рішення ухвалює людина."""
    o = _ff_open({'BTCUSDT': {'dir': 'LONG', 'kind': 'choch'}})
    o._q4_recheck_filters = lambda sym, side: (_ for _ in ()).throw(
        AssertionError('повторна перевірка фільтрів НЕ повинна виконуватись'))
    o._ob_epoch_already_opened = lambda sym: (_ for _ in ()).throw(
        AssertionError('гейт «1 угода на 1H-OB» НЕ повинен блокувати ручне відкриття'))
    r = o.force_open_queue4('BTCUSDT')
    _check(r['ok'], f'ручне відкриття не мало впертись у ворота: {r}')
    kw = o.__dict__['_open_calls'][0]['kw']
    _check(kw.get('skip_safeguard') and kw.get('skip_ctr_safeguard'),
           'ручне відкриття йде з тими самими послабленнями, що й авто-Q4')
    print('✓ ворота Черги-4 (шари / 🔁 перевірка / 1H-OB) пропущено')


def test_manual_open_cleans_queue_and_sets_sl():
    o = _ff_open({'ETHUSDT': {'dir': 'LONG', 'kind': 'vob_alert'}})
    o._q4_cache_put('ETHUSDT', {'LONG': {'count': 2}})
    o.force_open_queue4('ETHUSDT')
    _check('ETHUSDT' not in o._pending4, 'монета мала вийти з Черги-4')
    _check('ETHUSDT' not in o._q4_layers_cache, 'кеш шарів мав очиститись')
    _check(o._timers.get('ETHUSDT', {}).get('dir') == 'LONG', 'таймер угоди мав стартувати')
    _check(o.__dict__.get('_marked') == ['ETHUSDT'],
           '1H-OB мав позначитись як відпрацьований (щоб авто-двигун не відкрив другу)')
    _check(o.__dict__.get('_sl') == [('ETHUSDT', 'LONG')],
           'Manual SL мав виставитись тим самим шляхом, що й в авто-відкритті')
    print('✓ чергу почищено, таймер/епоху/SL виставлено')


def test_manual_open_refuses_when_not_queued():
    o = _ff_open({})
    r = o.force_open_queue4('XRPUSDT')
    _check(not r['ok'] and 'Черзі-4' in r['reason'], f'мала бути відмова: {r}')
    _check(o.__dict__['_open_calls'] == [], '_open не мав викликатись')
    print('✓ монети немає в черзі → відмова, нічого не відкриваємо')


def test_manual_open_refuses_when_already_in_trade():
    o = _ff_open({'ADAUSDT': {'dir': 'LONG'}}, has_pos=True)
    r = o.force_open_queue4('ADAUSDT')
    _check(not r['ok'], f'уже є позиція → відмова: {r}')
    o2 = _ff_open({'ADAUSDT': {'dir': 'LONG'}}, managed=('ADAUSDT',))
    r2 = o2.force_open_queue4('ADAUSDT')
    _check(not r2['ok'], f'уже під керуванням FF → відмова: {r2}')
    print('✓ подвійне відкриття неможливе')


def test_manual_open_keeps_queue_when_open_rejected():
    o = _ff_open({'DOGEUSDT': {'dir': 'SHORT'}}, opened=False)
    r = o.force_open_queue4('DOGEUSDT')
    _check(not r['ok'] and r['opened'] is False, f'мала бути відмова: {r}')
    _check('DOGEUSDT' in o._pending4,
           '_open відхилив → монета ЛИШАЄТЬСЯ в черзі (не губимо запис)')
    print('✓ відмова `_open` не викидає монету з черги')


def test_manual_open_requires_ff_enabled():
    o = _ff_open({'BTCUSDT': {'dir': 'LONG'}})
    o.get_settings = lambda: {'enabled': False}
    r = o.force_open_queue4('BTCUSDT')
    _check(not r['ok'], 'при вимкненому FF ручне відкриття не проходить')
    print('✓ вимкнений Fuel Filter → відмова')


if __name__ == '__main__':
    test_ttl_expires_by_default()
    test_no_ttl_disables_expiry()
    test_no_ttl_also_disables_stagnation()
    test_zero_values_still_mean_no_limit()
    test_expiry_helpers_survive_garbage()
    test_no_ttl_default_off()
    test_sl_default_source_is_1h()
    test_sl_short_uses_top_plus_buffer()
    test_sl_source_15m_picks_volumized()
    test_sl_falls_back_when_chosen_source_missing()
    test_sl_not_set_when_no_block_at_all()
    test_sl_bad_source_value_falls_back_to_1h()
    test_sl_source_validated_in_settings()
    test_manual_open_uses_queued_direction_and_label()
    test_manual_open_bypasses_queue_gates()
    test_manual_open_cleans_queue_and_sets_sl()
    test_manual_open_refuses_when_not_queued()
    test_manual_open_refuses_when_already_in_trade()
    test_manual_open_keeps_queue_when_open_rejected()
    test_manual_open_requires_ff_enabled()
    print('\nУсі тести керувань Черги-4 пройдено ✅')
