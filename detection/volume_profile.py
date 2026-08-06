"""
volume_profile — Volume Profile (VPVR) → POC з даних Binance SPOT.

Відтворює «білу лінію» POC (Point of Control) як у MobChart: рівень ціни з
найбільшим торгованим обсягом у заданому діапазоні. Джерело — Binance SPOT
(api.binance.com), як на скріні користувача.

ТОЧНІСТЬ: беремо ДРІБНІ 1m-klines у вікні і розподіляємо обсяг КОЖНОЇ свічки
рівномірно по цінових бінах, які вона перекриває [low..high] — стандартний
метод VPVR (як у TradingView), але на 1-хв роздільності → дуже близько до
реального профілю за угодами, значно легше за aggTrades.

Використання:
  • Візуал: біла горизонтальна лінія POC на графіку (видимий діапазон).
  • Сигнал-гейт: у момент сигналу порівняти поточну ціну з POC —
    LONG «ОК», коли POC ВИЩЕ ціни (є куди рости до вартості);
    SHORT «ОК», коли POC НИЖЧЕ ціни. Інакше — вхід проти вартості.

Публічне API:
  compute_poc(symbol, from_sec=None, to_sec=None, hours=None, bins=..., ...)
    → {ok, poc, vah, val, price_high, price_low, bins, klines, src, symbol,
       market, value_area_pct, computed_at}
  price_vs_poc(poc, price, side) → {'ok': bool, 'rel': 'above'|'below'|'at',
       'reason': str}
"""

import time
import threading
from typing import Optional, Dict, List

import requests

# Binance klines — на вибір ринок: SPOT (як на скріні) або USDT-Futures (perp).
# Формат рядка klines однаковий для обох ендпоінтів.
_SPOT_KLINES = 'https://api.binance.com/api/v3/klines'
_FUT_KLINES = 'https://fapi.binance.com/fapi/v1/klines'


def _klines_url(market: str) -> str:
    return _FUT_KLINES if (market or 'spot').lower().startswith('fut') else _SPOT_KLINES

_DEFAULT_BINS = 150          # роздільність профілю (рядків)
_DEFAULT_HOURS = 72.0        # фолбек-вікно, якщо не передано from/to (≈ як видно на скріні)
_MAX_REQUESTS = 10           # кап пагінації klines (10×1000 = 10000 хв ≈ 7 діб 1m)
_VALUE_AREA = 0.70           # частка обсягу для зони вартості (VAH/VAL)

_cache: Dict[str, tuple] = {}   # key -> (ts, result)
_cache_lock = threading.Lock()
_TTL = 25.0


def _norm_symbol(symbol: str) -> str:
    s = (symbol or '').upper().strip()
    if s.endswith('.P'):
        s = s[:-2]
    return s


def _fetch_klines(symbol: str, interval: str, limit: int = 1000,
                  start_ms: Optional[int] = None,
                  end_ms: Optional[int] = None, market: str = 'spot') -> List[list]:
    params = {'symbol': symbol, 'interval': interval, 'limit': min(1000, int(limit))}
    if start_ms is not None:
        params['startTime'] = int(start_ms)
    if end_ms is not None:
        params['endTime'] = int(end_ms)
    r = requests.get(_klines_url(market), params=params, timeout=12)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    return data


def _fetch_window(symbol: str, interval: str, start_ms: int, end_ms: int,
                  market: str = 'spot') -> List[list]:
    """Пагінація 1m-klines у вікні [start_ms, end_ms]. Binance ліміт 1000/запит."""
    out: List[list] = []
    cur = int(start_ms)
    for _ in range(_MAX_REQUESTS):
        batch = _fetch_klines(symbol, interval, 1000, start_ms=cur, end_ms=end_ms,
                              market=market)
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        last_open = int(batch[-1][0])
        nxt = last_open + 1
        if nxt >= end_ms or nxt <= cur:
            break
        cur = nxt
    return out


def _profile_from_klines(klines: List[list], bins: int):
    """Побудувати профіль обсягу з klines. Кожен рядок kline:
    [openTime, open, high, low, close, volume, closeTime, quoteVol, trades, ...].
    Обсяг свічки РІВНОМІРНО розподіляється по бінах, які перекриває [low..high].
    Повертає (vol_by_bin[], lo, hi, binw) або (None, ...) при браку даних."""
    lows, highs = [], []
    parsed = []
    for k in klines:
        try:
            hi = float(k[2]); lo = float(k[3]); vol = float(k[5])
        except (IndexError, TypeError, ValueError):
            continue
        if hi <= 0 or lo <= 0 or vol < 0 or hi < lo:
            continue
        parsed.append((lo, hi, vol))
        lows.append(lo); highs.append(hi)
    if not parsed:
        return None, 0.0, 0.0, 0.0
    lo = min(lows); hi = max(highs)
    if hi <= lo:
        hi = lo * 1.0001 + 1e-9
    bins = max(10, int(bins))
    binw = (hi - lo) / bins
    if binw <= 0:
        return None, lo, hi, 0.0
    vol_by_bin = [0.0] * bins
    for (kl, kh, kv) in parsed:
        if kv <= 0:
            continue
        i0 = int((kl - lo) / binw)
        i1 = int((kh - lo) / binw)
        if i0 < 0:
            i0 = 0
        if i1 > bins - 1:
            i1 = bins - 1
        if i1 < i0:
            i1 = i0
        n = (i1 - i0 + 1)
        share = kv / n
        for i in range(i0, i1 + 1):
            vol_by_bin[i] += share
    return vol_by_bin, lo, hi, binw


def _value_area(vol_by_bin: List[float], poc_idx: int, target_frac: float):
    """Зона вартості: розширюємось від POC назовні, поки не набрано target_frac
    сумарного обсягу. Повертає (val_idx, vah_idx)."""
    total = sum(vol_by_bin)
    if total <= 0:
        return poc_idx, poc_idx
    acc = vol_by_bin[poc_idx]
    lo_i = hi_i = poc_idx
    n = len(vol_by_bin)
    need = total * target_frac
    while acc < need and (lo_i > 0 or hi_i < n - 1):
        below = vol_by_bin[lo_i - 1] if lo_i > 0 else -1.0
        above = vol_by_bin[hi_i + 1] if hi_i < n - 1 else -1.0
        if above >= below:
            hi_i += 1
            acc += max(0.0, above)
        else:
            lo_i -= 1
            acc += max(0.0, below)
    return lo_i, hi_i


def compute_poc(symbol: str, from_sec: Optional[float] = None,
                to_sec: Optional[float] = None, hours: Optional[float] = None,
                bins: int = _DEFAULT_BINS, interval: str = '1m',
                market: str = 'spot') -> Dict:
    """POC (+VAH/VAL) з Binance (SPOT або USDT-Futures) за вікном.
    Пріоритет вікна: [from_sec, to_sec] (видимий діапазон графіка) →
    інакше останні `hours` годин (дефолт 72). `bins` — роздільність профілю.
    `market`: 'spot' (дефолт, як на скріні) або 'futures' (perp)."""
    sym = _norm_symbol(symbol)
    mkt = 'futures' if (market or 'spot').lower().startswith('fut') else 'spot'
    out = {'ok': False, 'symbol': sym, 'market': mkt, 'poc': None,
           'vah': None, 'val': None, 'price_high': None, 'price_low': None,
           'bins': int(bins), 'klines': 0,
           'src': f'binance_{mkt}_{interval}',
           'value_area_pct': int(_VALUE_AREA * 100), 'computed_at': None,
           'reason': ''}
    if not sym:
        out['reason'] = 'no symbol'
        return out
    now_ms = int(time.time() * 1000)
    if from_sec and to_sec and to_sec > from_sec:
        start_ms = int(float(from_sec) * 1000)
        end_ms = int(float(to_sec) * 1000)
    else:
        h = float(hours or _DEFAULT_HOURS)
        end_ms = now_ms
        start_ms = end_ms - int(h * 3600 * 1000)
    if end_ms > now_ms:
        end_ms = now_ms
    ck = f"{mkt}|{sym}|{interval}|{start_ms // 1000}|{end_ms // 1000}|{int(bins)}"
    with _cache_lock:
        hit = _cache.get(ck)
        if hit and (time.time() - hit[0]) < _TTL:
            return hit[1]
    try:
        klines = _fetch_window(sym, interval, start_ms, end_ms, market=mkt)
    except Exception as e:
        out['reason'] = f'binance fetch error: {e}'
        return out
    if not klines:
        out['reason'] = f'no klines (symbol not on Binance {mkt}?)'
        return out
    vol_by_bin, lo, hi, binw = _profile_from_klines(klines, bins)
    if not vol_by_bin or binw <= 0:
        out['reason'] = 'insufficient data'
        return out
    poc_idx = max(range(len(vol_by_bin)), key=lambda i: vol_by_bin[i])
    poc = lo + (poc_idx + 0.5) * binw
    val_i, vah_i = _value_area(vol_by_bin, poc_idx, _VALUE_AREA)
    val = lo + (val_i + 0.0) * binw       # нижня межа нижнього біна зони
    vah = lo + (vah_i + 1.0) * binw       # верхня межа верхнього біна зони
    out.update({
        'ok': True, 'poc': round(poc, 10), 'vah': round(vah, 10),
        'val': round(val, 10), 'price_high': round(hi, 10),
        'price_low': round(lo, 10), 'klines': len(klines),
        'computed_at': time.time(),
    })
    with _cache_lock:
        _cache[ck] = (time.time(), out)
    return out


def price_vs_poc(poc: Optional[float], price: Optional[float],
                 side: Optional[str] = None, tol_pct: float = 0.05) -> Dict:
    """Вердикт «ціна ↔ POC» для напрямку.
      LONG «ОК»  → POC ВИЩЕ ціни (є куди рости до вартості);
      SHORT «ОК» → POC НИЖЧЕ ціни.
    `tol_pct` — мертва зона навколо POC (%), у якій кажемо 'at' (на рівні)."""
    res = {'ok': None, 'rel': None, 'reason': 'немає даних', 'dist_pct': None}
    try:
        if poc is None or price is None or poc <= 0 or price <= 0:
            return res
        dist_pct = (price - poc) / poc * 100.0
        res['dist_pct'] = round(dist_pct, 3)
        band = abs(float(tol_pct))
        if abs(dist_pct) <= band:
            rel = 'at'
        elif price > poc:
            rel = 'above'      # ціна вище POC
        else:
            rel = 'below'      # ціна нижче POC
        res['rel'] = rel
        s = (side or '').upper()
        if s == 'LONG':
            # LONG ОК, коли POC вище ціни (rel == 'below')
            res['ok'] = (rel == 'below')
            res['reason'] = ('POC вище ціни — LONG ОК' if rel == 'below'
                             else ('ціна на POC' if rel == 'at'
                                   else 'POC нижче ціни — LONG проти вартості'))
        elif s == 'SHORT':
            res['ok'] = (rel == 'above')
            res['reason'] = ('POC нижче ціни — SHORT ОК' if rel == 'above'
                             else ('ціна на POC' if rel == 'at'
                                   else 'POC вище ціни — SHORT проти вартості'))
        else:
            res['reason'] = ('ціна вище POC' if rel == 'above'
                             else ('ціна нижче POC' if rel == 'below' else 'ціна на POC'))
        return res
    except Exception:
        return res
