"""
reversal_pressure — 4H reversal-pressure index for a coin.

Given a side (the CURRENT move's direction) and 4H market data from Bybit,
this computes a 0..100 index of how much PRESSURE has built up for a reversal
AGAINST that move. It is explicitly NOT a probability of the future — it is a
weighted count of conditions that historically PRECEDE reversals. The label
must always say "схильність до розвороту", never "ймовірність".

Components (each 0..1, then weighted), all from verifiable 4H data:
  rsi_extreme    20  — RSI(14) overbought (>70) for a top / oversold (<30) for a bottom
  divergence     22  — price/RSI divergence (price new extreme, RSI not) — strongest classic warning
  stretch        16  — distance from EMA/VWAP mean in ATRs (overextension)
  funding        14  — funding-rate extreme in the move's direction (crowded carry)
  oi_divergence  12  — price extends while OI falls (move losing fuel)
  ls_extreme     10  — long/short account ratio one-sided (contrarian)
  wick           6   — long rejection wicks on recent 4H candles (exhaustion)
                ---
                100

Output: {ok, index 0..100, level ('LOW'|'BUILDING'|'HIGH'), components, notes,
         sources_used}. Any missing source degrades gracefully (its weight is
         dropped and the rest are renormalised, so the index stays 0..100).
"""

from typing import List, Dict, Optional


W = {
    'rsi_extreme': 20, 'divergence': 22, 'stretch': 16, 'funding': 14,
    'oi_divergence': 12, 'ls_extreme': 10, 'wick': 6,
}


def _rsi(closes: List[float], period: int = 14) -> Optional[List[float]]:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    # Wilder's smoothing
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    rsis = []
    for i in range(period, len(gains)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = (avg_g / avg_l) if avg_l > 0 else 999
        rsis.append(100 - 100 / (1 + rs))
    return rsis


def _ema(vals: List[float], period: int) -> Optional[float]:
    if len(vals) < period:
        return None
    k = 2 / (period + 1)
    e = vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e


def _atr(klines: List[Dict], period: int = 14) -> Optional[float]:
    if len(klines) < period + 1:
        return None
    trs = []
    for i in range(1, len(klines)):
        h = klines[i]['high']; l = klines[i]['low']; pc = klines[i - 1]['close']
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    # Wilder
    atr = sum(trs[:period]) / period
    for t in trs[period:]:
        atr = (atr * (period - 1) + t) / period
    return atr


def analyze_reversal_pressure(
    side: str,
    klines_4h: List[Dict],
    funding_rate: Optional[float] = None,
    oi_history: Optional[List[Dict]] = None,
    long_pct: Optional[float] = None,
) -> Dict:
    """Compute the 4H reversal-pressure index against `side`.

    side: 'LONG'|'SHORT' — direction of the current move. Reversal pressure
          is the pressure to turn the OTHER way.
    klines_4h: Bybit 4H bars {open,high,low,close,volume,timestamp}, old→new.
    funding_rate / oi_history / long_pct: optional extra signals.
    """
    out = {'ok': False, 'index': None, 'level': None,
           'components': {}, 'notes': [], 'sources_used': []}
    side = (side or '').upper()
    dir_sign = 1 if side == 'LONG' else -1 if side == 'SHORT' else 0
    if dir_sign == 0 or not klines_4h or len(klines_4h) < 30:
        out['notes'].append('недостатньо 4H даних')
        return out

    # The reversal is AGAINST the current move: a LONG move reverses DOWN
    # (to SHORT), a SHORT move reverses UP (to LONG). Surface this explicitly
    # so the UI can say which way, not just "how much".
    reversal_to = 'SHORT' if side == 'LONG' else 'LONG'
    out['from_side'] = side
    out['reversal_to'] = reversal_to

    closes = [k['close'] for k in klines_4h]
    price = closes[-1]
    comp = {}
    notes = []
    used = ['4H klines']

    # 1. RSI extreme (in the move's direction → reversal risk)
    rsis = _rsi(closes, 14)
    rsi_now = rsis[-1] if rsis else None
    if rsi_now is not None:
        if dir_sign == 1:   # uptrend → overbought is reversal risk
            comp['rsi_extreme'] = max(0.0, min(1.0, (rsi_now - 60) / 25))
            if rsi_now >= 70:
                notes.append(f'RSI 4H перекуплений ({rsi_now:.0f})')
        else:               # downtrend → oversold is reversal risk
            comp['rsi_extreme'] = max(0.0, min(1.0, (40 - rsi_now) / 25))
            if rsi_now <= 30:
                notes.append(f'RSI 4H перепроданий ({rsi_now:.0f})')

    # 2. Price / RSI divergence over the last ~10 bars
    if rsis and len(rsis) >= 10 and len(closes) >= 12:
        recent_p = closes[-10:]
        recent_r = rsis[-10:]
        if dir_sign == 1:
            # price higher high but RSI lower high → bearish divergence
            if recent_p[-1] >= max(recent_p[:-1]) and recent_r[-1] < max(recent_r[:-1]) - 2:
                comp['divergence'] = 1.0
                notes.append('ведмежа дивергенція ціна/RSI на 4H')
            else:
                comp['divergence'] = 0.0
        else:
            if recent_p[-1] <= min(recent_p[:-1]) and recent_r[-1] > min(recent_r[:-1]) + 2:
                comp['divergence'] = 1.0
                notes.append('бичача дивергенція ціна/RSI на 4H')
            else:
                comp['divergence'] = 0.0

    # 3. Stretch from mean (in ATRs)
    ema = _ema(closes, 21)
    atr = _atr(klines_4h, 14)
    if ema and atr and atr > 0:
        signed = (price - ema) * dir_sign
        stretch_atr = signed / atr
        comp['stretch'] = max(0.0, min(1.0, stretch_atr / 4.0))
        if stretch_atr > 3:
            notes.append(f'ціна розтягнута на {stretch_atr:.1f} ATR (4H)')

    # 4. Funding extreme (crowded in the move's direction)
    if funding_rate is not None:
        used.append('funding')
        # Positive funding = longs pay shorts (crowd long). For an uptrend,
        # high positive funding is reversal risk; for a downtrend, very
        # negative funding is reversal risk.
        fr = funding_rate
        if dir_sign == 1:
            comp['funding'] = max(0.0, min(1.0, fr / 0.0008))  # 0.08% = hot
            if fr >= 0.0005:
                notes.append(f'funding перегрітий ({fr*100:.3f}%)')
        else:
            comp['funding'] = max(0.0, min(1.0, -fr / 0.0008))
            if fr <= -0.0005:
                notes.append(f'funding різко негативний ({fr*100:.3f}%)')

    # 5. OI divergence (price extends but OI falls = move losing fuel)
    if oi_history and len(oi_history) >= 6:
        used.append('open interest')
        try:
            oi_vals = [o['open_interest'] for o in oi_history[-6:]]
            oi_chg = (oi_vals[-1] - oi_vals[0]) / oi_vals[0] if oi_vals[0] else 0
            price_chg = (closes[-1] - closes[-6]) / closes[-6] if len(closes) >= 6 else 0
            price_moved_our_way = (price_chg * dir_sign) > 0
            if price_moved_our_way and oi_chg < -0.02:
                comp['oi_divergence'] = min(1.0, abs(oi_chg) / 0.1)
                notes.append('ціна йде, а OI падає (рух без палива)')
            else:
                comp['oi_divergence'] = 0.0
        except Exception:
            pass

    # 6. Long/short ratio extreme (contrarian)
    if long_pct is not None:
        used.append('long/short ratio')
        if dir_sign == 1 and long_pct >= 65:
            comp['ls_extreme'] = min(1.0, (long_pct - 65) / 25)
            notes.append(f'натовп {long_pct:.0f}% LONG (контра)')
        elif dir_sign == -1 and long_pct <= 35:
            comp['ls_extreme'] = min(1.0, (35 - long_pct) / 25)
            notes.append(f'натовп {100-long_pct:.0f}% SHORT (контра)')
        else:
            comp['ls_extreme'] = 0.0

    # 7. Rejection wicks on last 3 candles
    wick_score = 0.0
    for k in klines_4h[-3:]:
        rng = k['high'] - k['low']
        if rng <= 0:
            continue
        body_top = max(k['open'], k['close'])
        body_bot = min(k['open'], k['close'])
        if dir_sign == 1:
            upper = (k['high'] - body_top) / rng
            if upper > 0.5:
                wick_score = max(wick_score, upper)
        else:
            lower = (body_bot - k['low']) / rng
            if lower > 0.5:
                wick_score = max(wick_score, lower)
    if wick_score > 0:
        comp['wick'] = wick_score
        notes.append('довгі тіні відторгнення на 4H')

    # ---- Weighted, renormalised over available components ----
    total_w = sum(W[k] for k in comp)
    if total_w <= 0:
        out['notes'] = notes or ['немає сигналів розвороту']
        out['ok'] = True
        out['index'] = 0
        out['level'] = 'LOW'
        out['sources_used'] = used
        return out
    weighted = sum(comp[k] * W[k] for k in comp)
    index = round(weighted / total_w * 100, 1)
    out['index'] = index
    out['level'] = 'HIGH' if index >= 60 else 'BUILDING' if index >= 35 else 'LOW'
    out['components'] = {k: round(v, 2) for k, v in comp.items()}
    out['notes'] = notes or ['сигналів розвороту мало']
    out['sources_used'] = used
    out['ok'] = True
    return out
