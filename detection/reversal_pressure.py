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


def detect_4h_trend(klines_4h: List[Dict]) -> Dict:
    """Determine the ACTUAL 4H trend using MULTIPLE professional factors,
    not a single structure check that too often returns NEUTRAL.

    A real market almost always has a directional bias — true range-bound
    flat is rare and short-lived. We score four independent signals and
    combine them; NEUTRAL is returned ONLY when they genuinely conflict
    (near-zero net), which is the honest definition of "no trend".

    Factors (each votes -1..+1):
      1. EMA structure: fast(21) vs slow(55) vs price location
      2. Linear-regression slope of closes (normalised by ATR)
      3. SMC swing structure bias (HH/HL vs LH/LL)
      4. Position vs longer EMA(100) — where price sits in the bigger picture

    Returns {trend: 'LONG'|'SHORT'|'NEUTRAL', label, strength 0..1}.
    """
    out = {'trend': 'NEUTRAL', 'label': '↔ ВБІК', 'strength': 0.0}
    if not klines_4h or len(klines_4h) < 30:
        return out

    closes = [float(k['close']) for k in klines_4h]
    price = closes[-1]
    votes = []   # each in [-1, +1]

    # --- Factor 1: EMA structure (21 vs 55) + price location ---
    def _ema(vals, p):
        if len(vals) < p:
            return None
        k = 2 / (p + 1)
        e = sum(vals[:p]) / p
        for v in vals[p:]:
            e = v * k + e * (1 - k)
        return e
    ema_f = _ema(closes, 21)
    ema_s = _ema(closes, 55)
    if ema_f and ema_s:
        # fast above slow → up; plus price above both reinforces
        v = 0.0
        v += 0.6 if ema_f > ema_s else -0.6
        v += 0.4 if price > ema_f else -0.4
        votes.append(max(-1.0, min(1.0, v)))

    # --- Factor 2: linear-regression slope of last N closes (ATR-normalised) ---
    n = min(50, len(closes))
    seg = closes[-n:]
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(seg) / n
    num = sum((xs[i] - mx) * (seg[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n)) or 1e-9
    slope = num / den  # price units per bar
    # normalise by ATR so it's comparable across coins
    trs = []
    for i in range(1, len(klines_4h)):
        h = float(klines_4h[i]['high']); l = float(klines_4h[i]['low'])
        pc = closes[i - 1]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = (sum(trs[-14:]) / min(14, len(trs))) if trs else 0
    if atr > 0:
        slope_atr = slope / atr  # ATRs per bar
        # ±0.15 ATR/bar ≈ a clear trend
        votes.append(max(-1.0, min(1.0, slope_atr / 0.15)))

    # --- Factor 3: SMC swing structure bias ---
    try:
        import numpy as np
        from detection.smc_structure_filter import SMCStructureDetector
        det = SMCStructureDetector(swing_length=min(50, len(klines_4h) // 3))
        res = det.update(
            np.array([float(k['high']) for k in klines_4h]),
            np.array([float(k['low']) for k in klines_4h]),
            np.array(closes, dtype=float))
        tb = res.get('trend_bias')
        val = getattr(tb, 'value', tb)
        if val in (1, -1):
            votes.append(float(val))
    except Exception:
        pass

    # --- Factor 4: position vs EMA(100) (bigger-picture bias) ---
    ema_l = _ema(closes, min(100, len(closes) - 1))
    if ema_l:
        dist = (price - ema_l) / ema_l
        votes.append(max(-1.0, min(1.0, dist / 0.03)))  # ±3% = full vote

    if not votes:
        return out

    net = sum(votes) / len(votes)   # -1..+1
    strength = min(1.0, abs(net))
    # Threshold for calling a trend: net must clear a dead-zone. 0.25 keeps
    # genuine flat (conflicting factors) NEUTRAL while still labelling normal
    # directional drift correctly.
    if net >= 0.25:
        return {'trend': 'LONG', 'label': '↑ ВГОРУ', 'strength': round(strength, 2)}
    if net <= -0.25:
        return {'trend': 'SHORT', 'label': '↓ ВНИЗ', 'strength': round(strength, 2)}
    out['strength'] = round(strength, 2)
    return out


def analyze_reversal_pressure(
    side: str,
    klines_4h: List[Dict],
    funding_rate: Optional[float] = None,
    oi_history: Optional[List[Dict]] = None,
    long_pct: Optional[float] = None,
    use_actual_trend: bool = True,
    tf_label: str = '4H',
) -> Dict:
    """Compute the 4H reversal-pressure index.

    By default (use_actual_trend=True) the pressure is measured against the
    ACTUAL 4H trend derived from swing structure — answering "how close is
    the current 4H trend to flipping?". This is what a trader wants: first
    where the 4H trend is, then how near it is to breaking.

    When use_actual_trend=False, pressure is measured against the supplied
    `side` (e.g. the bot's verdict) instead — kept for callers that need that.

    klines_4h: Bybit 4H bars {open,high,low,close,volume,timestamp}, old→new.
    funding_rate / oi_history / long_pct: optional extra signals.
    """
    out = {'ok': False, 'index': None, 'level': None,
           'components': {}, 'notes': [], 'sources_used': [],
           'trend_4h': None, 'trend_4h_label': None}

    # Determine the reference direction the pressure is measured against.
    trend_info = detect_4h_trend(klines_4h) if klines_4h else {'trend': 'NEUTRAL', 'label': '↔ ВБІК'}
    out['trend_4h'] = trend_info['trend']
    out['trend_4h_label'] = trend_info['label']

    # Guard: need enough data first.
    if not klines_4h or len(klines_4h) < 30:
        out['notes'].append('недостатньо 4H даних')
        return out

    # If the 4H trend is sideways/undefined there is no trend to "reverse",
    # but the bar should still be meaningful: show pressure toward whichever
    # range edge is nearer to breaking. We pick the side by where price sits
    # in its recent range (upper half → pressure to break UP / LONG).
    if use_actual_trend and trend_info['trend'] not in ('LONG', 'SHORT'):
        try:
            window = klines_4h[-60:] if len(klines_4h) >= 60 else klines_4h
            hi = max(k['high'] for k in window)
            lo = min(k['low'] for k in window)
            px = klines_4h[-1]['close']
            pos_pct = (px - lo) / (hi - lo) * 100 if hi > lo else 50
        except Exception:
            pos_pct = 50
        # Nearer to top → likelier to break up (LONG); nearer bottom → SHORT.
        break_to = 'LONG' if pos_pct >= 50 else 'SHORT'
        # Pressure = how close to the edge (0 at mid-range, ~100 at the edge)
        edge_idx = round(abs(pos_pct - 50) / 50 * 100, 1)
        out['ok'] = True
        out['index'] = edge_idx
        out['level'] = 'NONE'
        out['reversal_to'] = break_to
        out['from_side'] = None
        arrow = '↑ вгору (LONG)' if break_to == 'LONG' else '↓ вниз (SHORT)'
        out['verdict_text'] = (f'{tf_label} у консолідації (тренду немає). Ціна в '
                               f'{"верхній" if pos_pct >= 50 else "нижній"} частині діапазону — '
                               f'ймовірніший вихід {arrow}.')
        out['notes'] = [f'{tf_label} флет — ціна на {pos_pct:.0f}% діапазону']
        return out

    if use_actual_trend and trend_info['trend'] in ('LONG', 'SHORT'):
        side = trend_info['trend']
    side = (side or '').upper()
    dir_sign = 1 if side == 'LONG' else -1 if side == 'SHORT' else 0
    if dir_sign == 0:
        out['notes'].append('тренд невизначений')
        return out

    # The reversal is AGAINST the reference trend: an UP trend reverses DOWN
    # (to SHORT), a DOWN trend reverses UP (to LONG). Surface this explicitly.
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
            notes.append(f'ціна розтягнута на {stretch_atr:.1f} ATR ({tf_label})')

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

    # Single coherent verdict sentence — trend + how close to breaking + which
    # way it would flip. No contradictions: this only runs when a real trend
    # exists.
    trend_word = 'висхідний (LONG)' if side == 'LONG' else 'низхідний (SHORT)'
    flip_word = '↑ вгору (LONG)' if reversal_to == 'LONG' else '↓ вниз (SHORT)'
    if index >= 60:
        out['verdict_text'] = (f'{tf_label} тренд {trend_word} ВИСНАЖУЄТЬСЯ — '
                               f'високий тиск зламу, розворот {flip_word} ймовірний.')
    elif index >= 35:
        out['verdict_text'] = (f'{tf_label} тренд {trend_word} ще тримається, але злам '
                               f'визріває — стежити за розворотом {flip_word}.')
    else:
        out['verdict_text'] = (f'{tf_label} тренд {trend_word} міцний — тиск зламу низький, '
                               f'розворот {flip_word} поки не на часі.')
    return out
