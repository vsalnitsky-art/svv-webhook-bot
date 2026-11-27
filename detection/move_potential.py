"""
move_potential — objective "how much gas is left" analytics for a coin.

Given a direction (LONG/SHORT) and recent klines, this computes a set of
VERIFIABLE technical measurements about the current move — never a forecast
of a future % gain, which no honest tool can promise. Every number here is
either a raw market value (ATR) or a measured distance/ratio.

Outputs (all best-effort; any field may be None on insufficient data):

  atr_abs        — ATR(14) in price units (current bar)
  atr_pct        — ATR as % of price (per-bar volatility)
  stretch_atr    — distance of price from its VWAP/EMA mean, in ATRs
                   (how extended the move is — high = exhausted)
  stretch_pct    — same distance as % of price
  adr_pct        — average daily range %, last N days
  adr_used_pct   — how much of today's ADR has already been travelled
                   (>100% = day already over-extended)
  runway_pct     — distance to the next real obstacle in the trade
                   direction (liquidity level / range edge), as % of price
  runway_atr     — same runway expressed in ATRs (how many "bars" of room)
  exhaustion     — 0..100 composite: higher = more exhausted / less room
  verdict        — 'FRESH' | 'MATURE' | 'EXHAUSTED'
  notes          — short human strings explaining each contributor

The exhaustion score is a transparent blend (documented inline). It is an
ASSESSMENT of remaining room, not a probability of continuation.
"""

from typing import List, Dict, Optional

try:
    from detection.forecast_engine import _atr, _ema
except Exception:  # pragma: no cover
    _atr = None
    _ema = None


def _vwap(klines: List[Dict]) -> Optional[float]:
    """Session-agnostic rolling VWAP over the supplied window."""
    num = 0.0
    den = 0.0
    for k in klines:
        try:
            h = float(k.get('h', k['p']))
            l = float(k.get('l', k['p']))
            c = float(k['p'])
            v = float(k.get('v', 0) or 0)
        except (KeyError, TypeError, ValueError):
            continue
        tp = (h + l + c) / 3.0
        num += tp * v
        den += v
    if den <= 0:
        return None
    return num / den


def _adr_pct(klines: List[Dict], days: int = 14, bars_per_day: int = 96) -> Optional[float]:
    """Average daily range as % of price, from intraday bars.

    bars_per_day defaults to 96 (15m bars). We bucket the most recent
    `days` days, take (high-low)/open per day, average. Falls back
    gracefully when there isn't enough history.
    """
    if not klines or len(klines) < bars_per_day:
        return None
    ranges = []
    n = len(klines)
    for d in range(days):
        end = n - d * bars_per_day
        start = end - bars_per_day
        if start < 0:
            break
        day = klines[start:end]
        if not day:
            continue
        hi = max(float(k.get('h', k['p'])) for k in day)
        lo = min(float(k.get('l', k['p'])) for k in day)
        op = float(day[0].get('o', day[0]['p'])) or float(day[0]['p'])
        if op > 0:
            ranges.append((hi - lo) / op * 100.0)
    if not ranges:
        return None
    return sum(ranges) / len(ranges)


def _adr_used_pct(klines: List[Dict], adr_pct: float, bars_per_day: int = 96) -> Optional[float]:
    """How much of today's ADR has been travelled so far (high-low of the
    current day's bars, as % of ADR)."""
    if not klines or adr_pct is None or adr_pct <= 0:
        return None
    day = klines[-bars_per_day:] if len(klines) >= bars_per_day else klines
    if not day:
        return None
    hi = max(float(k.get('h', k['p'])) for k in day)
    lo = min(float(k.get('l', k['p'])) for k in day)
    op = float(day[0].get('o', day[0]['p'])) or float(day[0]['p'])
    if op <= 0:
        return None
    today_range_pct = (hi - lo) / op * 100.0
    return today_range_pct / adr_pct * 100.0


def analyze_move_potential(
    side: str,
    klines: List[Dict],
    liquidation_levels: Optional[List[Dict]] = None,
    range_high: Optional[float] = None,
    range_low: Optional[float] = None,
    bars_per_day: int = 96,
) -> Dict:
    """Compute the move-potential snapshot. See module docstring.

    side: 'LONG' | 'SHORT' — direction whose remaining room we assess.
    klines: market_data bars {p,o,h,l,v}, oldest→newest.
    liquidation_levels: optional [{price, usd}, ...] obstacles.
    range_high/low: optional swing-range bounds (fallback obstacle).
    """
    out = {
        'ok': False, 'side': None, 'atr_abs': None, 'atr_pct': None,
        'stretch_atr': None, 'stretch_pct': None,
        'adr_pct': None, 'adr_used_pct': None,
        'runway_pct': None, 'runway_atr': None,
        'exhaustion': None, 'verdict': None, 'notes': [],
    }
    side = (side or '').upper()
    out['side'] = side
    dir_sign = 1 if side == 'LONG' else -1 if side == 'SHORT' else 0
    if dir_sign == 0 or not klines or len(klines) < 20 or _atr is None:
        out['notes'].append('недостатньо даних')
        return out

    try:
        price = float(klines[-1]['p'])
    except (KeyError, TypeError, ValueError):
        return out
    if price <= 0:
        return out

    notes = []

    # ---- ATR ----
    atr_series = _atr(klines, 14)
    atr_abs = atr_series[-1] if atr_series else None
    atr_pct = (atr_abs / price * 100.0) if atr_abs else None
    if atr_pct is not None:
        out['atr_abs'] = round(atr_abs, 8)
        out['atr_pct'] = round(atr_pct, 3)
        vol_word = ('низька' if atr_pct < 0.5 else
                    'помірна' if atr_pct < 1.5 else 'висока')
        notes.append(f'ATR {atr_pct:.2f}%/бар ({vol_word} волатильність)')

    # ---- Stretch from mean (exhaustion of the move) ----
    mean_ref = _vwap(klines[-bars_per_day:]) if len(klines) >= 20 else None
    if mean_ref is None:
        ema_series = _ema([float(k['p']) for k in klines], 21)
        mean_ref = ema_series[-1] if ema_series and ema_series[-1] is not None else None
    stretch_atr = None
    if mean_ref and atr_abs and atr_abs > 0:
        signed_dist = (price - mean_ref) * dir_sign  # +ve = moved our way
        stretch_atr = signed_dist / atr_abs
        out['stretch_atr'] = round(stretch_atr, 2)
        out['stretch_pct'] = round(signed_dist / price * 100.0, 3)
        if stretch_atr > 3:
            notes.append(f'розтягнуто на {stretch_atr:.1f} ATR від середнього (виснаження)')
        elif stretch_atr < 0:
            notes.append(f'ціна ще не подолала середнє ({stretch_atr:.1f} ATR)')
        else:
            notes.append(f'{stretch_atr:.1f} ATR від середнього')

    # ---- ADR + how much used today ----
    adr_pct = _adr_pct(klines, days=14, bars_per_day=bars_per_day)
    if adr_pct is not None:
        out['adr_pct'] = round(adr_pct, 2)
        used = _adr_used_pct(klines, adr_pct, bars_per_day=bars_per_day)
        if used is not None:
            out['adr_used_pct'] = round(used, 1)
            if used >= 100:
                notes.append(f'денний хід вичерпано ({used:.0f}% ADR)')
            elif used >= 70:
                notes.append(f'пройдено {used:.0f}% денного ходу')
            else:
                notes.append(f'денний хід має запас ({used:.0f}% ADR)')

    # ---- Runway: distance to next obstacle in trade direction ----
    target = None
    obstacles = []
    for lev in (liquidation_levels or []):
        try:
            lp = float(lev['price'])
        except (KeyError, TypeError, ValueError):
            continue
        # Obstacle must be ahead in the trade direction
        if dir_sign == 1 and lp > price:
            obstacles.append(lp)
        elif dir_sign == -1 and lp < price:
            obstacles.append(lp)
    if obstacles:
        # Nearest obstacle ahead
        target = min(obstacles) if dir_sign == 1 else max(obstacles)
        target_src = 'ліквідний рівень'
    else:
        # Fallback to range edge
        edge = range_high if dir_sign == 1 else range_low
        if edge is not None:
            edge = float(edge)
            if (dir_sign == 1 and edge > price) or (dir_sign == -1 and edge < price):
                target = edge
                target_src = 'край діапазону'
    if target is not None:
        runway_pct = abs(target - price) / price * 100.0
        out['runway_pct'] = round(runway_pct, 3)
        if atr_abs and atr_abs > 0:
            out['runway_atr'] = round(abs(target - price) / atr_abs, 1)
            notes.append(f'до {target_src}: {runway_pct:.2f}% ({out["runway_atr"]:.1f} ATR)')
        else:
            notes.append(f'до {target_src}: {runway_pct:.2f}%')

    # ---- Composite exhaustion (transparent blend) ----
    # Each sub-signal contributes 0..1 (1 = exhausted). We average the ones
    # we actually have, so missing data degrades gracefully.
    subs = []
    # Stretch: >4 ATR is very exhausted, <=0 is fresh.
    if stretch_atr is not None:
        s = max(0.0, min(1.0, stretch_atr / 4.0))
        subs.append(s)
    # ADR used: 100%+ is exhausted.
    if out['adr_used_pct'] is not None:
        subs.append(max(0.0, min(1.0, out['adr_used_pct'] / 100.0)))
    # Runway: <1 ATR ahead is exhausted, >=5 ATR is fresh.
    if out['runway_atr'] is not None:
        r = 1.0 - max(0.0, min(1.0, out['runway_atr'] / 5.0))
        subs.append(r)
    if subs:
        exhaustion = round(sum(subs) / len(subs) * 100.0, 1)
        out['exhaustion'] = exhaustion
        if exhaustion >= 66:
            out['verdict'] = 'EXHAUSTED'
        elif exhaustion >= 40:
            out['verdict'] = 'MATURE'
        else:
            out['verdict'] = 'FRESH'

    out['notes'] = notes
    out['ok'] = True
    return out
