"""💧 СКАН ЛІКВІДНОСТІ ПО СПИСКУ МОНЕТ — куди тягне ринок, одразу по всіх.

Запит користувача: «список монет, відсортований за ліквідністю: де найбільший
перекіс і найближчий магніт — щоб бачити, куди тягне ринок по всьому списку
одразу», з вибором біржі, разовим запуском зі сторінки 📡 Tickr.

⚠️ ЧОМУ ЦЕ ОКРЕМИЙ РОЗРАХУНОК, А НЕ liq-map (принципово — не плутати!).
Живий `liquidation_map` будує рівні з ПРИРОСТУ відкритого інтересу між своїми
тіками: він знає, скільки OI додалось саме зараз і за якою ціною. Це точніше,
але працює ЛИШЕ для монет, які демон уже веде (BTC/ETH + замовлені), і
потребує годин накопичення. Для разового скану сотні монет таких даних немає
і бути не може.

Тому тут — **МОМЕНТАЛЬНИЙ ЗРІЗ**: беремо ПОТОЧНИЙ сукупний OI (одним bulk-
запитом на всю біржу) і розкладаємо його по історії цін, вважаючи, що позиції
відкривались там, де торгувався обсяг. Далі — та сама формула ліквідації і та
сама драбина, що й у живій карті.

Що це означає чесно:
  • напрямок перекосу і розташування магнітів — надійні (вони визначаються
    геометрією рівнів, а не тим, коли саме позиція відкрилась);
  • абсолютні суми — грубіші, ніж у живій карті;
  • монети без bulk-OI (Binance, BingX) порахувати НЕМОЖЛИВО — так і кажемо,
    а не підставляємо нулі.

Розрахункове ядро — ЧИСТІ функції; мережа лише в `scan_liquidity`.
"""

from typing import Dict, List, Optional
import time
from concurrent.futures import ThreadPoolExecutor

from detection.liquidation_map.liquidation_math import (
    liquidation_price, DEFAULT_LEVERAGE_WEIGHTS,
)
from detection.liquidation_map import ladder as _ladder


# Скільки барів історії беремо для розкладки OI. 1h × 168 = 7 діб: достатньо,
# щоб охопити актуальні позиції, і не тягне зайвих сторінок (ліміт 1000/запит).
DEFAULT_INTERVAL = '60'
DEFAULT_BARS = 168
# Стеля кандидатів у скані — щоб разовий запуск не перетворився на годинний.
MAX_SYMBOLS = 120
PARALLEL = 8
# Рівні далі за це від ціни в драбину не потрапляють: вони не «магніти».
WINDOW_PCT = 12.0


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if x == x and x not in (float('inf'), float('-inf')) else None
    except (TypeError, ValueError):
        return None


def build_levels(bars: List[Dict], oi_usd: float, price: float,
                 symbol: str = 'BTCUSDT',
                 window_pct: float = WINDOW_PCT) -> List[Dict]:
    """ЧИСТА функція: розкласти сукупний OI по історії й отримати рівні
    ліквідації у форматі, який розуміє `ladder.build_ladder`.

    `bars` — [{h, l, c, v}], найстаріший першим.

    Логіка:
      1. Позиції відкривались там, де ТОРГУВАВСЯ ОБСЯГ. Тому вагу кожного
         бару беремо за його обсягом, а «ціну входу» — за типовою ціною
         (h+l+c)/3.
      2. Частку OI цього бару ділимо між плечима 25×/50×/100× тими самими
         вагами, що й жива карта, і порівну між лонгами та шортами.
         ⚠️ 50/50 — свідоме припущення: співвідношення лонг/шорт по КОЖНІЙ
         монеті пачкою не віддає жодна біржа. Перекіс, який ми показуємо,
         виникає з ГЕОМЕТРІЇ рівнів, а не з вгаданого співвідношення.
      3. Ціну ліквідації рахуємо ТІЄЮ САМОЮ формулою, що й жива карта.
      4. **МІТИГАЦІЯ**: якщо після свого бару ціна вже проходила крізь рівень,
         позиції там немає — рівень викидаємо. Без цього драбина показувала б
         давно знесені кластери.
    """
    p = _f(price) or 0.0
    oi = _f(oi_usd) or 0.0
    if p <= 0 or oi <= 0 or not bars:
        return []

    clean = []
    for b in bars:
        # Свічки приходять із чужого API — приймаємо ЛИШЕ словники. Падати на
        # сміттєвому елементі не можна: один кривий бар вибив би монету зі
        # скану цілком.
        if not isinstance(b, dict):
            continue
        h, l, c = _f(b.get('h')), _f(b.get('l')), _f(b.get('c'))
        v = _f(b.get('v')) or 0.0
        if h is None or l is None or c is None or h <= 0 or l <= 0 or c <= 0:
            continue
        clean.append({'h': h, 'l': l, 'c': c, 'v': max(0.0, v),
                      'tp': (h + l + c) / 3.0})
    if not clean:
        return []
    tot_v = sum(b['v'] for b in clean)
    if tot_v <= 0:
        # Обсягу немає — розподіляємо рівномірно, щоб не втратити монету зовсім.
        for b in clean:
            b['v'] = 1.0
        tot_v = float(len(clean))

    lo_bound = p * (1.0 - window_pct / 100.0)
    hi_bound = p * (1.0 + window_pct / 100.0)

    # Майбутні екстремуми: чи ходила ціна крізь рівень ПІСЛЯ його бару.
    n = len(clean)
    fut_hi = [0.0] * n
    fut_lo = [0.0] * n
    run_hi, run_lo = 0.0, float('inf')
    for i in range(n - 1, -1, -1):
        fut_hi[i], fut_lo[i] = run_hi, run_lo
        run_hi = max(run_hi, clean[i]['h'])
        run_lo = min(run_lo, clean[i]['l'])

    out: List[Dict] = []
    for i, b in enumerate(clean):
        share = oi * (b['v'] / tot_v)
        if share <= 0:
            continue
        for lev, w in DEFAULT_LEVERAGE_WEIGHTS.items():
            notional = share * w / 2.0          # порівну між лонгами й шортами
            if notional <= 0:
                continue
            for side in ('long', 'short'):
                try:
                    lq = liquidation_price(side.upper(), b['tp'], lev,
                                           notional, symbol=symbol)
                except Exception:
                    continue
                if not lq or lq <= 0 or lq < lo_bound or lq > hi_bound:
                    continue
                # Мітигація: рівень уже знесений рухом ціни після цього бару.
                if side == 'long' and fut_lo[i] <= lq:
                    continue
                if side == 'short' and fut_hi[i] >= lq:
                    continue
                out.append({'price': lq, 'usd': notional, 'side': side})
    return out


def summarise(levels: List[Dict], price: float, symbol: str,
              step_usd=None) -> Dict:
    """Звести рівні монети до рядка списку. Драбина й вердикт — ТІ САМІ
    функції, що й у блоці на сторінці Smart Money: щоб «тягне вниз» означало
    одне й те саме в обох місцях."""
    lad = _ladder.build_ladder(levels, price, step_usd=step_usd, top_n=6)
    if not lad.get('ok'):
        return {'symbol': symbol, 'ok': False,
                'reason': lad.get('reason', 'немає рівнів')}
    v = lad.get('verdict') or {}
    m = (v.get('parts') or {}).get('magnet') or {}
    top = (lad.get('rows') or [])
    # Найближчий магніт — НЕ найбільший: серед значущих сходинок беремо ту,
    # до якої ціні йти найменше. Саме вона спрацює першою.
    nearest = min(top, key=lambda r: r['dist_pct']) if top else None
    return {
        'symbol': symbol, 'ok': True,
        'price': lad.get('mark_price'),
        'step': lad.get('step'),
        'total_usd': lad.get('total_usd'),
        'above_pct': (lad.get('above') or {}).get('pct'),
        'below_pct': (lad.get('below') or {}).get('pct'),
        'pull': lad.get('pull'),
        'pull_pct': lad.get('pull_pct'),
        'strength': v.get('strength'),
        'verdict': v.get('text'),
        # Найбільший за масою магніт (те саме, що у вердикті блоку).
        'magnet_price': m.get('price'), 'magnet_pct': m.get('pct'),
        'magnet_dist': m.get('dist'), 'magnet_dir': m.get('dir'),
        # І найближчий за відстанню — перша перепона на шляху ціни.
        'near_price': nearest['price'] if nearest else None,
        'near_pct': nearest['pct'] if nearest else None,
        'near_dist': nearest['dist_pct'] if nearest else None,
        'near_dir': nearest['dir'] if nearest else None,
        'rows': top,
    }


def sort_rows(rows: List[Dict], by: str = 'pull') -> List[Dict]:
    """Порядок списку. `pull` — за силою перекосу (де ринок тягне найдужче),
    `magnet` — за розміром найбільшого магніту, `near` — за близькістю
    найближчого магніту (що спрацює першим)."""
    ok = [r for r in rows if r.get('ok')]
    bad = [r for r in rows if not r.get('ok')]

    def _pct(v):
        try:
            return float(str(v).replace('%', ''))
        except (TypeError, ValueError):
            return 0.0

    if by == 'magnet':
        ok.sort(key=lambda r: -_pct(r.get('magnet_pct')))
    elif by == 'near':
        ok.sort(key=lambda r: (r.get('near_dist') if r.get('near_dist')
                               is not None else 9e9))
    else:
        ok.sort(key=lambda r: -(r.get('pull_pct') or 0))
    return ok + bad


# ── мережа ────────────────────────────────────────────────────────────────
def _klines_bybit(session, symbol, interval, limit):
    r = session.get('https://api.bybit.com/v5/market/kline',
                    params={'category': 'linear', 'symbol': symbol,
                            'interval': interval, 'limit': limit}, timeout=10)
    lst = ((r.json().get('result') or {}).get('list') or [])
    # Bybit віддає найновіший першим — розвертаємо.
    return [{'h': x[2], 'l': x[3], 'c': x[4], 'v': x[6]} for x in reversed(lst)]


def _klines_mexc(session, symbol, interval, limit):
    sym = symbol.replace('USDT', '_USDT')
    r = session.get(f'https://contract.mexc.com/api/v1/contract/kline/{sym}',
                    params={'interval': 'Min60', 'limit': limit}, timeout=10)
    d = (r.json() or {}).get('data') or {}
    hs, ls, cs, vs = d.get('high') or [], d.get('low') or [], d.get('close') or [], d.get('amount') or []
    return [{'h': hs[i], 'l': ls[i], 'c': cs[i], 'v': vs[i] if i < len(vs) else 0}
            for i in range(min(len(hs), len(ls), len(cs)))]


_KLINES = {'bybit': _klines_bybit, 'mexc': _klines_mexc}

# Біржі, які НЕ віддають відкритий інтерес пачкою. Без OI рівні ліквідації
# не побудувати, і вигадувати їх не можна — кажемо прямо.
NO_BULK_OI = {
    'binance': 'Binance не віддає відкритий інтерес пачкою (лише по одній '
               'монеті — це сотні запитів), тож моментальний скан неможливий',
    'bingx': 'BingX не віддає відкритий інтерес у публічних тікерах',
}


def scan_liquidity(exchange: str = 'bybit', top_n: int = 40,
                   min_vol_usd: float = 20_000_000,
                   min_oi_usd: float = 5_000_000,
                   bars: int = DEFAULT_BARS,
                   sort_by: str = 'pull') -> Dict:
    """Разовий скан: список монет із перекосом ліквідності й магнітами.

    Вартість: 1 bulk-запит на всю біржу + 1 запит свічок на КОЖНУ монету,
    що пройшла відсів. При `top_n=40` це ~41 HTTP і 15-30 секунд.
    """
    import requests
    from detection import tickr_core

    exchange = (exchange or 'bybit').lower()
    if exchange in NO_BULK_OI:
        return {'ok': False, 'exchange': exchange,
                'reason': NO_BULK_OI[exchange]}
    if exchange not in _KLINES:
        return {'ok': False, 'exchange': exchange,
                'reason': f'скан ліквідності для {exchange} не підтримується'}

    try:
        metrics = tickr_core._ACTIVITY[exchange](tickr_core.MARKET_SWAP)
    except Exception as e:
        return {'ok': False, 'exchange': exchange,
                'reason': f'не вдалось отримати тікери: {e}'}

    # Відсів: ліквідні монети з реальним OI. Без цього в списку буде сміття,
    # у якому «перекіс 90%» побудований на трьох доларах.
    cands, drop_vol, drop_oi = [], 0, 0
    for sym, m in metrics.items():
        vol = _f(m.get('vol_usd')) or 0.0
        oi = _f(m.get('oi_usd')) or 0.0
        price = _f(m.get('last')) or 0.0
        if price <= 0:
            continue
        if vol < min_vol_usd:
            drop_vol += 1
            continue
        if oi < min_oi_usd:
            drop_oi += 1
            continue
        cands.append({'symbol': sym, 'price': price, 'oi_usd': oi,
                      'vol_usd': vol})
    cands.sort(key=lambda c: -c['oi_usd'])
    n = max(1, min(int(top_n or 40), MAX_SYMBOLS))
    cands = cands[:n]
    if not cands:
        return {'ok': False, 'exchange': exchange,
                'reason': f'жодна монета не пройшла відсів '
                          f'(обіг <{min_vol_usd/1e6:.0f}M: {drop_vol}, '
                          f'OI <{min_oi_usd/1e6:.0f}M: {drop_oi})'}

    session = requests.Session()
    fetch = _KLINES[exchange]
    bars = max(24, min(int(bars or DEFAULT_BARS), 1000))

    def _one(c):
        try:
            kl = fetch(session, c['symbol'], DEFAULT_INTERVAL, bars)
        except Exception as e:
            return {'symbol': c['symbol'], 'ok': False,
                    'reason': f'свічки недоступні: {str(e)[:60]}'}
        lv = build_levels(kl, c['oi_usd'], c['price'], symbol=c['symbol'])
        row = summarise(lv, c['price'], c['symbol'])
        row['oi_usd'] = round(c['oi_usd'], 0)
        row['vol_usd'] = round(c['vol_usd'], 0)
        return row

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
        for r in pool.map(_one, cands):
            rows.append(r)
    rows = sort_rows(rows, sort_by)
    ok_rows = [r for r in rows if r.get('ok')]
    return {'ok': True, 'exchange': exchange,
            'scanned': len(cands), 'with_data': len(ok_rows),
            'dropped_vol': drop_vol, 'dropped_oi': drop_oi,
            'sort_by': sort_by, 'bars': bars,
            'took_sec': round(time.time() - t0, 1),
            'rows': rows, 'fetched_at': time.time()}
