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
  • співвідношення лонг/шорт — припущення 50/50 (див. `build_levels`).

⚠️ BULK-OI: ОБМЕЖЕННЯ БУЛО НЕ В БІРЖІ, А В МАСШТАБІ (виправлено).
Раніше Binance і BingX відмовляли повністю: вони не віддають відкритий інтерес
ПАЧКОЮ. Але «пачкою» заважає лише тоді, коли монет СОТНІ. Після відсіву за
обігом у скані лишаються ДЕСЯТКИ монет, а на кожну ми й так робимо запит
свічок — тож ще один запит OI на монету цілком по кишені (`_OI_ONE`).
Тому:
  • біржа з bulk-OI (Bybit, MEXC) — 1 запит тікерів на всю біржу;
  • біржа без нього (Binance, BingX) — 1 запит тікерів (обіг+ціна) + по
    одному запиту OI на КОЖНУ монету, що пройшла відсів, зі стелею
    `PER_SYMBOL_OI_CAP`.
І окремий режим `scan_one(exchange, symbol)` — аналіз ОДНІЄЇ монети: там
питання bulk-OI не стоїть узагалі (2-3 запити разом зі свічками), тож
працюють УСІ чотири біржі.

Розрахункове ядро — ЧИСТІ функції; мережа лише в `scan_liquidity`/`scan_one`.
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
# Стеля для бірж БЕЗ bulk-OI: там на кожну монету йде ЩЕ ОДИН запит (OI),
# тобто скан коштує вдвічі. Десятки монет — нормально, сотні — вже ні.
PER_SYMBOL_OI_CAP = 60
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
              step_usd=None, top_n: int = 6) -> Dict:
    """Звести рівні монети до рядка списку. Драбина й вердикт — ТІ САМІ
    функції, що й у блоці на сторінці Smart Money: щоб «тягне вниз» означало
    одне й те саме в обох місцях.

    `top_n` — скільки сходинок лишити: 6 для рядка списку, більше — для
    режиму однієї монети, де драбина малюється повністю."""
    lad = _ladder.build_ladder(levels, price, step_usd=step_usd, top_n=top_n)
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
        # Вердикт ПО ШМАТКАХ — щоб режим однієї монети розфарбував числа
        # окремо від тексту, як це вже робить блок 💧 на Smart Money.
        'verdict_parts': v.get('parts'),
        'verdict_tone': v.get('tone'),
        # Найбільший за масою магніт (те саме, що у вердикті блоку).
        'magnet_price': m.get('price'), 'magnet_pct': m.get('pct'),
        'magnet_dist': m.get('dist'), 'magnet_dir': m.get('dir'),
        # ⚠️ І ТА САМА сходинка СИРИМИ ЧИСЛАМИ — для логіки (напр. TP-2 від
        # магніту). `magnet_price` вище — ФОРМАТОВАНИЙ рядок для показу;
        # парсити його заради числа не можна.
        'magnet_row': lad.get('magnet_row'),
        # І найближчий за відстанню — перша перепона на шляху ціни.
        'near_price': nearest['price'] if nearest else None,
        # ⚠️ Верхня межа теж — щоб UI підписав СМУГУ, а не одну межу: `near_dist`
        # (як і `magnet_dist`) міряється від СЕРЕДИНИ сходинки, тож підпис
        # однією межею вказував би на іншу точку, ніж відсоток поруч.
        'near_price_hi': nearest.get('price_hi') if nearest else None,
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
                    params={'category': 'linear',
                            'symbol': _ex_symbol('bybit', symbol),
                            'interval': interval, 'limit': limit}, timeout=10)
    lst = ((r.json().get('result') or {}).get('list') or [])
    # Bybit віддає найновіший першим — розвертаємо.
    return [{'h': x[2], 'l': x[3], 'c': x[4], 'v': x[6]} for x in reversed(lst)]


def _klines_mexc(session, symbol, interval, limit):
    sym = _ex_symbol('mexc', symbol)
    r = session.get(f'https://contract.mexc.com/api/v1/contract/kline/{sym}',
                    params={'interval': 'Min60', 'limit': limit}, timeout=10)
    d = (r.json() or {}).get('data') or {}
    hs, ls, cs, vs = d.get('high') or [], d.get('low') or [], d.get('close') or [], d.get('amount') or []
    return [{'h': hs[i], 'l': ls[i], 'c': cs[i], 'v': vs[i] if i < len(vs) else 0}
            for i in range(min(len(hs), len(ls), len(cs)))]


def _klines_binance(session, symbol, interval, limit):
    r = session.get('https://fapi.binance.com/fapi/v1/klines',
                    params={'symbol': _ex_symbol('binance', symbol),
                            'interval': '1h', 'limit': limit}, timeout=10)
    lst = r.json()
    if not isinstance(lst, list):
        raise RuntimeError(str(lst)[:80])
    # [openTime, o, h, l, c, volume, closeTime, quoteVolume, …] — беремо
    # QUOTE-обсяг (idx 7): вага бару має бути в доларах, як і OI.
    return [{'h': x[2], 'l': x[3], 'c': x[4], 'v': x[7]} for x in lst]


def _klines_bingx(session, symbol, interval, limit):
    r = session.get('https://open-api.bingx.com/openApi/swap/v3/quote/klines',
                    params={'symbol': _ex_symbol('bingx', symbol),
                            'interval': '1h', 'limit': limit}, timeout=10)
    rows = (r.json() or {}).get('data') or []
    # BingX віддає найновіший першим — упорядковуємо за часом.
    rows = sorted(rows, key=lambda x: _f(x.get('time')) or 0)
    return [{'h': x.get('high'), 'l': x.get('low'), 'c': x.get('close'),
             'v': x.get('volume')} for x in rows]


_KLINES = {'bybit': _klines_bybit, 'mexc': _klines_mexc,
           'binance': _klines_binance, 'bingx': _klines_bingx}


def _ex_symbol(exchange: str, symbol: str) -> str:
    """Канонічний `BTCUSDT` → формат конкретної біржі."""
    s = norm_symbol(symbol)
    if exchange == 'mexc':
        return s[:-4] + '_USDT' if s.endswith('USDT') else s
    if exchange == 'bingx':
        return s[:-4] + '-USDT' if s.endswith('USDT') else s
    return s


def norm_symbol(symbol: str) -> str:
    """Що б користувач не ввів — `btc`, `BTC-USDT`, `btc_usdt`, `BTCUSDT` —
    зводимо до канонічного `BTCUSDT`. Порожнє → BTCUSDT (дефолт за
    домовленістю)."""
    s = ''.join(ch for ch in str(symbol or '').upper() if ch.isalnum())
    if not s:
        return 'BTCUSDT'
    for q in ('USDT', 'USDC', 'USD'):
        if s.endswith(q):
            return s[:-len(q)] + 'USDT'
    return s + 'USDT'


# ── OI по ОДНІЙ монеті ────────────────────────────────────────────────────
# Кожен фетчер повертає (ціна, OI у доларах). Саме це знімає обмеження
# «немає bulk-OI»: пачкою його не дають, а поштучно — дають усі чотири біржі.
def _oi_one_bybit(session, symbol):
    r = session.get('https://api.bybit.com/v5/market/tickers',
                    params={'category': 'linear',
                            'symbol': _ex_symbol('bybit', symbol)}, timeout=10)
    lst = ((r.json().get('result') or {}).get('list') or [])
    if not lst:
        return None, None
    t = lst[0]
    px = _f(t.get('lastPrice')) or 0.0
    return px, (_f(t.get('openInterest')) or 0.0) * px


def _oi_one_binance(session, symbol):
    sym = _ex_symbol('binance', symbol)
    oi = session.get('https://fapi.binance.com/fapi/v1/openInterest',
                     params={'symbol': sym}, timeout=10).json()
    px_j = session.get('https://fapi.binance.com/fapi/v1/ticker/price',
                       params={'symbol': sym}, timeout=10).json()
    px = _f((px_j or {}).get('price')) or 0.0
    # `openInterest` — у БАЗОВІЙ монеті, тож переводимо в долари ціною.
    return px, (_f((oi or {}).get('openInterest')) or 0.0) * px


def _oi_one_mexc(session, symbol):
    r = session.get('https://contract.mexc.com/api/v1/contract/ticker',
                    params={'symbol': _ex_symbol('mexc', symbol)}, timeout=10)
    d = (r.json() or {}).get('data') or {}
    if isinstance(d, list):
        d = d[0] if d else {}
    px = _f(d.get('lastPrice')) or 0.0
    return px, (_f(d.get('holdVol')) or 0.0) * px


def _oi_one_bingx(session, symbol):
    sym = _ex_symbol('bingx', symbol)
    oi = session.get('https://open-api.bingx.com/openApi/swap/v2/quote/openInterest',
                     params={'symbol': sym}, timeout=10).json()
    tk = session.get('https://open-api.bingx.com/openApi/swap/v2/quote/ticker',
                     params={'symbol': sym}, timeout=10).json()
    t = (tk or {}).get('data') or {}
    if isinstance(t, list):
        t = t[0] if t else {}
    px = _f(t.get('lastPrice')) or 0.0
    return px, (_f(((oi or {}).get('data') or {}).get('openInterest')) or 0.0) * px


_OI_ONE = {'bybit': _oi_one_bybit, 'binance': _oi_one_binance,
           'mexc': _oi_one_mexc, 'bingx': _oi_one_bingx}

# Біржі, які віддають відкритий інтерес ПАЧКОЮ (один запит на всю біржу).
# Решта — через `_OI_ONE`, по запиту на монету (див. шапку модуля).
BULK_OI = {'bybit', 'mexc'}
# Пояснення для UI: чому на цих біржах скан списку дорожчий.
NO_BULK_OI = {
    'binance': 'Binance не віддає відкритий інтерес пачкою — беремо його '
               'по одному запиту на монету (скан трохи довший)',
    'bingx': 'BingX не віддає відкритий інтерес у публічних тікерах — '
             'беремо його по одному запиту на монету (скан трохи довший)',
}


def scan_one(exchange: str = 'binance', symbol: str = 'BTCUSDT',
             bars: int = DEFAULT_BARS, rows: int = 12) -> Dict:
    """💧 Аналіз ЛІКВІДНОСТІ ОДНІЄЇ МОНЕТИ — повна драбина + вердикт.

    Тут питання bulk-OI не стоїть узагалі: на одну монету потрібні 2-3
    запити (OI + ціна + свічки), тож режим працює на ВСІХ біржах, зокрема
    на Binance і BingX.

    Розрахунок — ТІ САМІ `build_levels`/`build_ladder`/`make_verdict`, що й у
    скані списку і в блоці 💧 на Smart Money: одна монета не має показувати
    інші числа, ніж та сама монета в списку.
    """
    import requests

    exchange = (exchange or 'binance').lower()
    sym = norm_symbol(symbol)
    if exchange not in _KLINES or exchange not in _OI_ONE:
        return {'ok': False, 'exchange': exchange, 'symbol': sym,
                'reason': f'біржа {exchange} не підтримується'}

    session = requests.Session()
    t0 = time.time()
    try:
        price, oi = _OI_ONE[exchange](session, sym)
    except Exception as e:
        return {'ok': False, 'exchange': exchange, 'symbol': sym,
                'reason': f'не вдалось отримати відкритий інтерес: {str(e)[:80]}'}
    if not price or price <= 0:
        return {'ok': False, 'exchange': exchange, 'symbol': sym,
                'reason': f'{sym} не знайдено на {exchange} (перевір назву)'}
    if not oi or oi <= 0:
        return {'ok': False, 'exchange': exchange, 'symbol': sym,
                'reason': f'{exchange} не віддає відкритий інтерес по {sym} — '
                          f'без нього рівні ліквідації не побудувати'}

    bars = max(24, min(int(bars or DEFAULT_BARS), 1000))
    try:
        kl = _KLINES[exchange](session, sym, DEFAULT_INTERVAL, bars)
    except Exception as e:
        return {'ok': False, 'exchange': exchange, 'symbol': sym,
                'reason': f'свічки недоступні: {str(e)[:80]}'}

    lv = build_levels(kl, oi, price, symbol=sym)
    out = summarise(lv, price, sym, top_n=max(3, int(rows or 12)))
    out.update({'exchange': exchange, 'oi_usd': round(oi, 0),
                'bars': bars, 'levels': len(lv),
                'took_sec': round(time.time() - t0, 1),
                'fetched_at': time.time()})
    return out


def scan_liquidity(exchange: str = 'binance', top_n: int = 40,
                   min_vol_usd: float = 20_000_000,
                   min_oi_usd: float = 5_000_000,
                   bars: int = DEFAULT_BARS,
                   sort_by: str = 'pull') -> Dict:
    """Разовий скан: список монет із перекосом ліквідності й магнітами.

    Вартість:
      • біржа з bulk-OI (Bybit, MEXC): 1 запит тікерів + 1 запит свічок на
        монету → при `top_n=40` це ~41 HTTP, 15-30 секунд;
      • без bulk-OI (Binance, BingX): + ще 1 запит OI на монету → ~81 HTTP.
        Саме тому там діє стеля `PER_SYMBOL_OI_CAP`.
    """
    import requests
    from detection import tickr_core

    exchange = (exchange or 'binance').lower()
    if exchange not in _KLINES:
        return {'ok': False, 'exchange': exchange,
                'reason': f'скан ліквідності для {exchange} не підтримується'}

    try:
        metrics = tickr_core._ACTIVITY[exchange](tickr_core.MARKET_SWAP)
    except Exception as e:
        return {'ok': False, 'exchange': exchange,
                'reason': f'не вдалось отримати тікери: {e}'}

    session = requests.Session()
    bulk_oi = exchange in BULK_OI
    warnings = []
    n = max(1, min(int(top_n or 40), MAX_SYMBOLS))
    if not bulk_oi:
        n = min(n, PER_SYMBOL_OI_CAP)
        warnings.append(NO_BULK_OI.get(
            exchange, 'відкритий інтерес береться по одному запиту на монету'))

    # Відсів: ліквідні монети з реальним OI. Без цього в списку буде сміття,
    # у якому «перекіс 90%» побудований на трьох доларах.
    # ⚠️ Порядок кроків залежить від того, чи є OI ОДРАЗУ: коли його немає,
    # спершу відбираємо кандидатів за ОБІГОМ (він є в тікерах завжди), і лише
    # для них питаємо OI поштучно — інакше довелось би питати всю біржу.
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
        if bulk_oi and oi < min_oi_usd:
            drop_oi += 1
            continue
        cands.append({'symbol': sym, 'price': price, 'oi_usd': oi,
                      'vol_usd': vol})
    # З bulk-OI сортуємо за самим OI (це і є «найважчі» монети), без нього —
    # за обігом: іншого мірила на цьому кроці просто немає.
    cands.sort(key=lambda c: -(c['oi_usd'] if bulk_oi else c['vol_usd']))
    cands = cands[:n]

    if cands and not bulk_oi:
        fetch_oi = _OI_ONE[exchange]

        def _oi(c):
            try:
                px, oi = fetch_oi(session, c['symbol'])
            except Exception:
                return c
            if px and px > 0:
                c['price'] = px
            c['oi_usd'] = oi or 0.0
            return c

        with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
            cands = list(pool.map(_oi, cands))
        keep = [c for c in cands if c['oi_usd'] >= min_oi_usd]
        drop_oi = len(cands) - len(keep)
        cands = keep

    if not cands:
        return {'ok': False, 'exchange': exchange, 'warnings': warnings,
                'reason': f'жодна монета не пройшла відсів '
                          f'(обіг <{min_vol_usd/1e6:.0f}M: {drop_vol}, '
                          f'OI <{min_oi_usd/1e6:.0f}M: {drop_oi})'}

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
            'bulk_oi': bulk_oi, 'warnings': warnings,
            'took_sec': round(time.time() - t0, 1),
            'rows': rows, 'fetched_at': time.time()}
