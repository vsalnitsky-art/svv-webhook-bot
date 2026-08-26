"""Тест: 🎯 «Як у Watchlist» — відбір монет тим самим принципом, що зібрав
робочий watchlist, але з УСІХ монет біржі.

Аудит поточних 33 монет watchlist (ETH, XRP, BNB, SOL, DOGE, TRX, LINK, AVAX,
DOT, HYPE, LTC, SUI, NEAR, APT, ONDO, UNI, MNT, TAO, ENA, ALGO, INJ, LDO, HBAR,
OP, WIF, 1000PEPE, ORDI, NEO, STRK, ZEC, BTC…) показав, що їх об'єднує НЕ сектор
(там і L1, і L2, і DeFi, і меми, і приватність), а рівно три речі:
    1) USDT-ПЕРПЕТУАЛ        — спот бот не торгує;
    2) топ-N КАПІТАЛІЗАЦІЇ   — приблизно топ-150 CoinGecko;
    3) реальна ЛІКВІДНІСТЬ   — обіг 24h + відкритий інтерес.
Плюс те, чого в списку НЕМАЄ: стейблкоїни й обгортки (WBTC, stETH, PAXG…) —
у них немає власної структури.
"""
import os, sys, types, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
_pkg = types.ModuleType('detection'); _pkg.__path__ = [os.path.join(_ROOT, 'detection')]
sys.modules['detection'] = _pkg


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


tk = _load('detection.tickr_core', 'detection/tickr_core.py')


def _check(c, m):
    if not c:
        raise AssertionError(m)


# ── синтетична біржа: перпи, споти, стейбли, обгортки, мілкі монети ─────────
def _sym(base, market='swap', quote='USDT'):
    return {'exchange': 'bybit', 'symbol': f'{base}{quote}',
            'exchange_symbol': f'{base}{quote}', 'base_asset': base,
            'quote_asset': quote, 'market_type': market, 'contract_type': '',
            'status': 'Trading', 'is_active': True,
            'is_spot': market == 'spot', 'is_swap': market == 'swap',
            'is_usdt': quote == 'USDT', 'is_usdc': quote == 'USDC'}


_UNIVERSE = [
    _sym('ETH'), _sym('SOL'), _sym('LINK'), _sym('STRK'), _sym('ZEC'),
    _sym('ETH', market='spot'),          # спот — має відсіятись
    _sym('SOL', quote='USDC'),           # USDC-перп — дубль тієї самої монети
    _sym('USDC'), _sym('WBTC'), _sym('PAXG'),   # стейбл / обгортка / золото
    _sym('NONAME'),                      # поза топ-N капіталізації
    _sym('THIN'),                        # у топі, але без обсягу
    _sym('NOOI'),                        # обсяг є, відкритого інтересу немає
    _sym('BTCUSDT-26MAR27'.replace('USDT-26MAR27', '')),   # заглушка, не dated
]

_ACT = {
    'ETH': {'vol_usd': 9e9, 'oi_usd': 5e9, 'trades': 900_000, 'change_pct': 1.2},
    'SOL': {'vol_usd': 3e9, 'oi_usd': 1e9, 'trades': 500_000, 'change_pct': -2.0},
    'LINK': {'vol_usd': 4e8, 'oi_usd': 2e8, 'trades': 90_000, 'change_pct': 0.5},
    'STRK': {'vol_usd': 8e7, 'oi_usd': 3e7, 'trades': 20_000, 'change_pct': 3.0},
    'ZEC': {'vol_usd': 2e8, 'oi_usd': 9e7, 'trades': 40_000, 'change_pct': 5.0},
    'USDC': {'vol_usd': 9e9, 'oi_usd': 9e9, 'trades': 999_999, 'change_pct': 0.0},
    'WBTC': {'vol_usd': 5e8, 'oi_usd': 3e8, 'trades': 30_000, 'change_pct': 1.0},
    'PAXG': {'vol_usd': 3e8, 'oi_usd': 2e8, 'trades': 10_000, 'change_pct': 0.3},
    'NONAME': {'vol_usd': 9e8, 'oi_usd': 5e8, 'trades': 80_000, 'change_pct': 9.0},
    'THIN': {'vol_usd': 1e6, 'oi_usd': 5e7, 'trades': 900, 'change_pct': 1.0},
    'NOOI': {'vol_usd': 8e8, 'oi_usd': 1e6, 'trades': 70_000, 'change_pct': 2.0},
}

# Ранги: усі, крім NONAME, у «топ-150».
_RANKS = {'ETH': 2, 'SOL': 6, 'LINK': 13, 'ZEC': 48, 'STRK': 140,
          'USDC': 5, 'WBTC': 11, 'PAXG': 90, 'THIN': 120, 'NOOI': 130}


def _install(monkey_ranks=None):
    tk.fetch = lambda ex, cats, active_only=True: {'ok': True, 'symbols': list(_UNIVERSE)}
    tk._ADAPTERS['bybit'] = object()

    def _activity(market_type):
        out = {}
        for s in _UNIVERSE:
            if s['market_type'] != market_type:
                continue
            b = s['base_asset']
            a = _ACT.get(b)
            if a:
                out[tk._canon(s['symbol'])] = dict(a, funding=0.0001, last=1.0)
        return out
    tk._ACTIVITY['bybit'] = _activity
    ranks = _RANKS if monkey_ranks is None else monkey_ranks
    tk.top_mcap_map = lambda n=100: {
        b: {'rank': r, 'mcap': 1e12 / max(1, r)} for b, r in ranks.items()}


def _run(sort_by='watchlist_like', **kw):
    _install(kw.pop('ranks', None))
    return tk.top_active('bybit', ['crypto'], sort_by=sort_by, top_n=100, **kw)


def _bases(res):
    return [r['base_asset'] for r in res['symbols']]


# ═══════════════════════════════ ТЕСТИ ══════════════════════════════════════
def test_mode_is_registered():
    _check('watchlist_like' in tk.SORT_KEYS, 'режим має бути у списку сортувань')
    _check(tk.WL_LIKE['mcap_universe'] == 150, 'універсум — топ-150 капіталізації')
    print('✓ режим зареєстровано (топ-150 капіталізації)')


def test_keeps_only_liquid_usdt_perps_in_mcap_top():
    r = _run()
    _check(r['ok'], f'виклик мав пройти: {r}')
    got = set(_bases(r))
    _check(got == {'ETH', 'SOL', 'LINK', 'STRK', 'ZEC'},
           f'очікували 5 «watchlist-подібних» монет, отримано {sorted(got)}')
    print(f'✓ відібрано саме ліквідні USDT-перпи з топ-капіталізації: {sorted(got)}')


def test_spot_and_usdc_are_dropped():
    r = _run()
    for row in r['symbols']:
        _check(row['is_swap'], f'спот не мав пройти: {row["symbol"]}')
        _check(row['is_usdt'], f'не-USDT перп не мав пройти: {row["symbol"]}')
    print('✓ спот і USDC-перпи відсіяно (бот торгує лише USDT-перпи)')


def test_stables_and_wrapped_are_excluded():
    got = set(_bases(_run()))
    for bad in ('USDC', 'WBTC', 'PAXG'):
        _check(bad not in got, f'{bad} не має власної структури — має бути відсіяний')
    print('✓ стейбли, обгортки й токенізоване золото виключено')


def test_out_of_mcap_universe_dropped():
    got = set(_bases(_run()))
    _check('NONAME' not in got,
           'монета поза топ-N капіталізації не має проходити, хай навіть ліквідна')
    print('✓ поза топ-150 капіталізації — не проходить')


def test_binance_case_no_bulk_oi_must_not_wipe_everything():
    """🐞 РЕАЛЬНИЙ БАГ: на BINANCE вибірка віддавала «залишилось 0».
    `_activity_binance` не заповнює `oi_usd` (у Binance Futures немає bulk-
    ендпоінта OI), тож гейт по OI викошував УСІХ, хто пройшов інші перевірки:
    «мілкий OI 37 → залишилось 0». Тепер, коли біржа не дає OI ЖОДНІЙ монеті,
    гейт вимикається, а в статусі про це прямо сказано."""
    global _ACT
    _saved = _ACT
    try:
        _ACT = {b: dict(v, oi_usd=0.0) for b, v in _saved.items()}   # біржа без OI
        r = _run()
        got = set(_bases(r))
        _check(got == {'ETH', 'SOL', 'LINK', 'STRK', 'ZEC', 'NOOI'},
               f'без OI-даних відбір має триматись на капіталізації+обігу: {sorted(got)}')
        w = ' '.join(r.get('warnings') or [])
        _check('фільтр по OI вимкнено' in w,
               f'вимкнення гейта має бути ЯВНО написане в статусі: {w}')
    finally:
        _ACT = _saved
    print('✓ біржа без bulk-OI (Binance) більше не дає порожній результат')


def test_multiplied_ticker_matches_base_coin():
    """1000PEPE/1000BONK — той самий актив, що PEPE/BONK у CoinGecko. Без
    нормалізації вони вилітали як «поза топ-N», хоча 1000PEPE РЕАЛЬНО є в
    робочому watchlist."""
    _check(tk.wl_base_variants('1000PEPE') == ['1000PEPE', 'PEPE'],
           f"очікували варіанти [1000PEPE, PEPE], отримано {tk.wl_base_variants('1000PEPE')}")
    _check(tk.wl_base_variants('1MBABYDOGE')[-1] == 'BABYDOGE', 'множник 1M теж знімається')
    _check(tk.wl_base_variants('ETH') == ['ETH'], 'звичайний тікер не чіпаємо')

    global _UNIVERSE, _ACT
    _u, _a = _UNIVERSE, _ACT
    try:
        _UNIVERSE = _u + [_sym('1000PEPE')]
        _ACT = dict(_a, **{'1000PEPE': {'vol_usd': 7e8, 'oi_usd': 2e8,
                                        'trades': 60_000, 'change_pct': 4.0}})
        r = _run(ranks=dict(_RANKS, PEPE=25))     # у CoinGecko вона зветься PEPE
        _check('1000PEPE' in set(_bases(r)),
               f'помножений контракт мав знайтись за базовою монетою: {_bases(r)}')
    finally:
        _UNIVERSE, _ACT = _u, _a
    print('✓ 1000PEPE знаходиться за базовим тікером PEPE')


def test_illiquid_dropped_by_volume_and_oi():
    got = set(_bases(_run()))
    _check('THIN' not in got, 'малий обіг 24h → відсів')
    _check('NOOI' not in got, 'обіг є, але немає відкритого інтересу → відсів')
    print('✓ мілкий обіг і мілкий OI відсіюються окремо')


def test_sorted_by_market_cap_rank():
    got = _bases(_run())
    _check(got == sorted(got, key=lambda b: _RANKS[b]),
           f'порядок має бути за рангом капіталізації (найбільші зверху): {got}')
    _check(got[0] == 'ETH', f'зверху мала бути найбільша монета, а не {got[0]}')
    print(f'✓ сортування за капіталізацією: {got}')


def test_user_volume_threshold_overrides_default():
    """Поле «VOL 24H ≥» має ПЕРЕКРИВАТИ дефолтний поріг режиму."""
    r = _run(min_vol_usd=1e9)      # лишає тільки ETH і SOL
    _check(set(_bases(r)) == {'ETH', 'SOL'},
           f'жорсткіший поріг обсягу мав звузити список: {_bases(r)}')
    print('✓ поріг обсягу з UI перекриває дефолт режиму')


def test_default_volume_floor_applies_when_user_left_zero():
    r = _run(min_vol_usd=0)
    _check('THIN' not in set(_bases(r)),
           'при порожньому полі має діяти ДЕФОЛТНИЙ поріг, інакше пролізе неліквід')
    print(f'✓ порожнє поле → дефолтний поріг ${tk.WL_LIKE["min_vol_usd"]/1e6:.0f}M')


def test_reports_why_coins_were_dropped():
    r = _run()
    w = ' '.join(r.get('warnings') or [])
    _check('Як у Watchlist' in w, f'має бути пояснення відбору: {w}')
    for token in ('не USDT-перп', 'поза топ-', 'стейбл/обгортка', 'мілкий OI'):
        _check(token in w, f'у поясненні бракує «{token}»: {w}')
    print('✓ у статусі видно, СКІЛЬКИ і ЧОМУ відсіяно')


def test_no_coingecko_is_reported_not_silent():
    r = _run(ranks={})
    _check(r['symbols'] == [], 'без рангів капіталізації відбір неможливий')
    _check(any('CoinGecko' in x for x in r.get('warnings') or []),
           f'відсутність CoinGecko має бути ЯВНО повідомлена: {r.get("warnings")}')
    print('✓ немає CoinGecko → чесне попередження, а не порожня таблиця без причини')


def test_other_sorts_unaffected():
    """Регресія: новий режим не має міняти поведінку наявних сортувань."""
    r = _run(sort_by='vol_usd')
    got = _bases(r)
    _check('USDC' in got and 'ETH' in got,
           f'звичайний «Обсяг 24h» відбирає ВСЕ, як і раніше: {got}')
    _check(got[0] in ('ETH', 'USDC'), f'сортування за обсягом: {got[:3]}')
    print('✓ інші режими сортування не зачеплені')


if __name__ == '__main__':
    test_mode_is_registered()
    test_keeps_only_liquid_usdt_perps_in_mcap_top()
    test_spot_and_usdc_are_dropped()
    test_stables_and_wrapped_are_excluded()
    test_out_of_mcap_universe_dropped()
    test_illiquid_dropped_by_volume_and_oi()
    test_binance_case_no_bulk_oi_must_not_wipe_everything()
    test_multiplied_ticker_matches_base_coin()
    test_sorted_by_market_cap_rank()
    test_user_volume_threshold_overrides_default()
    test_default_volume_floor_applies_when_user_left_zero()
    test_reports_why_coins_were_dropped()
    test_no_coingecko_is_reported_not_silent()
    test_other_sorts_unaffected()
    print('\nУсі тести режиму «Як у Watchlist» пройдено ✅')
