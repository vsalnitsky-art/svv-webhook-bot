"""
hold_analyzer — SMC Hold-Confidence for open positions.

For an open trade (LONG/SHORT at some entry), score 0..100 how strongly the
current Smart-Money-Concepts picture supports HOLDING the trade in its
direction. This is pure analysis + a verdict — it never closes trades, never
alerts, never auto-acts. The trader decides.

Direction-relative: the same bullish structure yields a HIGH score for a LONG
and a LOW score for a SHORT.

Weights (HTF structure dominates, per design):
    HTF structure / trend ...... 35   (the trend decision)
    LTF CHoCH / BOS ............ 25   (reversal against you = strongest alarm)
    Premium/Discount zone ...... 15
    Order Block support ........ 15
    Trade context (PnL / TP-SL)  10
                               -----
                                100

Verdict:
    >= 65  HOLD     (green)  — SMC backs the trade
    45..64 NEUTRAL  (amber)  — mixed; no edge either way
    < 45   WEAK     (red)    — SMC works against the trade

Bars use market_data schema {p=close, o, h, l, v, t}. SMCAnalyzer wants
{high, low, close} so we adapt before calling it.
"""

from typing import List, Dict, Optional

try:
    from detection.smc_analyzer import (
        SMCAnalyzer, MarketBias, StructureSignal, PriceZone,
    )
except Exception:  # pragma: no cover - import guard for isolated tests
    SMCAnalyzer = None


# Component weights (sum = 100)
W_HTF_STRUCT = 35
W_LTF_BREAK = 25
W_ZONE = 15
W_OB = 15
W_CONTEXT = 10

HOLD_MIN = 65
NEUTRAL_MIN = 45


def _adapt_bars(klines: List[Dict]) -> List[Dict]:
    """market_data {p,o,h,l,v} -> SMCAnalyzer {high,low,close}. Tolerates
    bars that already use high/low/close."""
    out = []
    for k in klines or []:
        try:
            high = k.get('h', k.get('high'))
            low = k.get('l', k.get('low'))
            close = k.get('p', k.get('close', k.get('c')))
            if high is None or low is None or close is None:
                continue
            open_ = k.get('o', k.get('open', close))
            out.append({
                'open': float(open_),
                'high': float(high),
                'low': float(low),
                'close': float(close),
                'volume': float(k.get('v', k.get('volume', 0)) or 0),
                'timestamp': k.get('t', k.get('timestamp', 0)),
            })
        except (TypeError, ValueError):
            continue
    return out


def _sign(bias) -> int:
    """+1 bullish, -1 bearish, 0 neutral — robust to enum or string."""
    v = getattr(bias, 'value', bias)
    if v in ('BULLISH', 1):
        return 1
    if v in ('BEARISH', -1):
        return -1
    return 0


def analyze_hold(position: Dict,
                 ltf_klines: List[Dict],
                 htf_klines: Optional[List[Dict]] = None,
                 analyzer: Optional["SMCAnalyzer"] = None) -> Dict:
    """Score how strongly SMC supports holding `position`.

    position: {'symbol', 'side' ('LONG'|'SHORT'), 'entry_price',
               optional 'tp_price','sl_price','pnl_pct'/current price}
    Returns a dict: {ok, score(0..100), verdict, color, components, reasons}.
    """
    side = (position.get('side') or '').upper()
    dir_sign = 1 if side == 'LONG' else -1 if side == 'SHORT' else 0
    if dir_sign == 0:
        return {'ok': False, 'reason': 'unknown side'}

    ltf = _adapt_bars(ltf_klines)
    if SMCAnalyzer is None:
        return {'ok': False, 'reason': 'analyzer unavailable'}
    if len(ltf) < 30:
        return {'ok': False, 'reason': 'insufficient ltf data'}

    az = analyzer or SMCAnalyzer()
    htf = _adapt_bars(htf_klines) if htf_klines else None
    res = az.analyze(ltf, htf_klines=htf if htf and len(htf) >= 20 else None)

    reasons = []
    comp = {}

    # ---- 1. HTF structure / trend (35) — the trend decision ----
    # If we passed HTF klines, analyze() already folded HTF bias into result;
    # we also run a dedicated HTF pass for a clean trend read when available.
    htf_sign = 0
    if htf and len(htf) >= 30:
        htf_res = az.analyze(htf)
        htf_sign = _sign(htf_res.market_bias)
    # Fall back to LTF market_bias as the structural read if no HTF.
    struct_sign = htf_sign if htf_sign != 0 else _sign(res.market_bias)
    # Aligned with trade dir -> full marks; against -> zero; neutral -> half.
    if struct_sign == 0:
        c_htf = W_HTF_STRUCT * 0.5
        reasons.append(('HTF структура нейтральна', 'neutral'))
    elif struct_sign == dir_sign:
        c_htf = W_HTF_STRUCT
        reasons.append(('HTF тренд за позицією', 'good'))
    else:
        c_htf = 0
        reasons.append(('HTF тренд ПРОТИ позиції', 'bad'))
    comp['htf_structure'] = round(c_htf, 1)

    # ---- 2. LTF CHoCH / BOS (25) — reversal alarm ----
    sig = getattr(res.structure_signal, 'value', res.structure_signal)
    c_break = W_LTF_BREAK * 0.5  # default: no decisive break -> half
    if sig in ('BULLISH_BOS', 'BEARISH_BOS'):
        bos_sign = 1 if sig == 'BULLISH_BOS' else -1
        if bos_sign == dir_sign:
            c_break = W_LTF_BREAK
            reasons.append(('BOS підтверджує напрямок', 'good'))
        else:
            c_break = W_LTF_BREAK * 0.25
            reasons.append(('BOS проти позиції', 'bad'))
    elif sig in ('BULLISH_CHOCH', 'BEARISH_CHOCH'):
        choch_sign = 1 if sig == 'BULLISH_CHOCH' else -1
        if choch_sign == dir_sign:
            c_break = W_LTF_BREAK * 0.8
            reasons.append(('CHoCH у бік позиції (свіжий розворот на користь)', 'good'))
        else:
            c_break = 0  # reversal against you — the strongest alarm
            reasons.append(('CHoCH ПРОТИ позиції (розворот!)', 'bad'))
    else:
        reasons.append(('Без рішучого зламу структури', 'neutral'))
    comp['ltf_break'] = round(c_break, 1)

    # ---- 3. Premium/Discount zone (15) ----
    zone = getattr(res.price_zone, 'value', res.price_zone)
    zlvl = getattr(res, 'zone_level', 0.5)
    c_zone = W_ZONE * 0.5
    if dir_sign == 1:   # LONG wants discount (cheap)
        if zone == 'DISCOUNT':
            c_zone = W_ZONE; reasons.append(('LONG у discount-зоні (дешево)', 'good'))
        elif zone == 'PREMIUM':
            c_zone = W_ZONE * 0.2; reasons.append(('LONG у premium-зоні (куплено дорого)', 'bad'))
    else:               # SHORT wants premium (expensive)
        if zone == 'PREMIUM':
            c_zone = W_ZONE; reasons.append(('SHORT у premium-зоні (дорого)', 'good'))
        elif zone == 'DISCOUNT':
            c_zone = W_ZONE * 0.2; reasons.append(('SHORT у discount-зоні (продано дешево)', 'bad'))
    comp['zone'] = round(c_zone, 1)

    # ---- 4. Order Block support (15) ----
    c_ob = W_OB * 0.5
    at_bull = getattr(res, 'price_at_bullish_ob', False)
    at_bear = getattr(res, 'price_at_bearish_ob', False)
    n_bull = len(getattr(res, 'active_bullish_obs', []) or [])
    n_bear = len(getattr(res, 'active_bearish_obs', []) or [])
    if dir_sign == 1:
        if at_bull:
            c_ob = W_OB; reasons.append(('Ціна на бичачому OB (опора)', 'good'))
        elif n_bull > n_bear:
            c_ob = W_OB * 0.7; reasons.append(('Переважають бичачі OB нижче', 'good'))
        elif at_bear:
            c_ob = W_OB * 0.2; reasons.append(('Ціна під ведмежим OB (опір)', 'bad'))
    else:
        if at_bear:
            c_ob = W_OB; reasons.append(('Ціна на ведмежому OB (опір)', 'good'))
        elif n_bear > n_bull:
            c_ob = W_OB * 0.7; reasons.append(('Переважають ведмежі OB вище', 'good'))
        elif at_bull:
            c_ob = W_OB * 0.2; reasons.append(('Ціна над бичачим OB (підтримка проти SHORT)', 'bad'))
    comp['order_block'] = round(c_ob, 1)

    # ---- 5. Trade context: PnL + TP/SL proximity (10) ----
    c_ctx = W_CONTEXT * 0.5
    entry = position.get('entry_price') or 0
    cur = position.get('current_price') or position.get('mark_price') or 0
    pnl_pct = position.get('pnl_pct')
    if pnl_pct is None and entry and cur:
        pnl_pct = (cur - entry) / entry * 100 * dir_sign
    if pnl_pct is not None:
        if pnl_pct > 0.5:
            c_ctx = W_CONTEXT; reasons.append((f'У прибутку (+{pnl_pct:.2f}%)', 'good'))
        elif pnl_pct < -0.5:
            c_ctx = W_CONTEXT * 0.3; reasons.append((f'У збитку ({pnl_pct:.2f}%)', 'bad'))
        else:
            reasons.append(('Біля беззбитку', 'neutral'))
    comp['context'] = round(c_ctx, 1)

    score = sum(comp.values())
    score = max(0, min(100, round(score, 1)))
    if score >= HOLD_MIN:
        verdict, color = 'HOLD', 'good'
    elif score >= NEUTRAL_MIN:
        verdict, color = 'NEUTRAL', 'neutral'
    else:
        verdict, color = 'WEAK', 'bad'

    return {
        'ok': True,
        'symbol': position.get('symbol'),
        'side': side,
        'score': score,
        'verdict': verdict,
        'color': color,
        'components': comp,
        'reasons': reasons,
    }
