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

Verdict (4 actionable tiers — each tells the trader what to do):
    >= 75  STRONG HOLD  (green)   — structure strongly backs the trade
    60..74 HOLD         (lime)    — favourable; keep holding
    40..59 REDUCE       (amber)   — edge fading; trim / tighten stop
    < 40   EXIT         (red)     — structure against; little upside left

Scoring is deliberately decisive: a component with NO evidence for the trade
does not sit at 50% — absence of edge pulls the score down, because "no
reason to hold" is not neutral. Neutral defaults are ~30-35% of weight, and
good/bad outcomes are polarised so verdicts separate cleanly.

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

# Verdict thresholds (4 tiers)
STRONG_HOLD_MIN = 75
HOLD_MIN = 60
REDUCE_MIN = 40
# below REDUCE_MIN -> EXIT


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
    # Aligned with trade dir -> full marks; against -> zero; neutral -> low.
    # Neutral is NOT half: a trendless HTF gives the trade no edge, so it
    # should pull the score down, not park it at the midpoint.
    if struct_sign == 0:
        c_htf = W_HTF_STRUCT * 0.3
        reasons.append(('HTF структура нейтральна (немає тренду на користь)', 'neutral'))
    elif struct_sign == dir_sign:
        c_htf = W_HTF_STRUCT
        reasons.append(('HTF тренд за позицією', 'good'))
    else:
        c_htf = 0
        reasons.append(('HTF тренд ПРОТИ позиції', 'bad'))
    comp['htf_structure'] = round(c_htf, 1)

    # ---- 2. LTF CHoCH / BOS (25) — reversal alarm ----
    sig = getattr(res.structure_signal, 'value', res.structure_signal)
    c_break = W_LTF_BREAK * 0.35  # no decisive break -> below-mid (no confirm)
    if sig in ('BULLISH_BOS', 'BEARISH_BOS'):
        bos_sign = 1 if sig == 'BULLISH_BOS' else -1
        if bos_sign == dir_sign:
            c_break = W_LTF_BREAK
            reasons.append(('BOS підтверджує напрямок', 'good'))
        else:
            c_break = W_LTF_BREAK * 0.15
            reasons.append(('BOS проти позиції', 'bad'))
    elif sig in ('BULLISH_CHOCH', 'BEARISH_CHOCH'):
        choch_sign = 1 if sig == 'BULLISH_CHOCH' else -1
        if choch_sign == dir_sign:
            c_break = W_LTF_BREAK * 0.85
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
    c_zone = W_ZONE * 0.35
    if dir_sign == 1:   # LONG wants discount (cheap)
        if zone == 'DISCOUNT':
            c_zone = W_ZONE; reasons.append(('LONG у discount-зоні (дешево)', 'good'))
        elif zone == 'PREMIUM':
            c_zone = W_ZONE * 0.1; reasons.append(('LONG у premium-зоні (куплено дорого)', 'bad'))
    else:               # SHORT wants premium (expensive)
        if zone == 'PREMIUM':
            c_zone = W_ZONE; reasons.append(('SHORT у premium-зоні (дорого)', 'good'))
        elif zone == 'DISCOUNT':
            c_zone = W_ZONE * 0.1; reasons.append(('SHORT у discount-зоні (продано дешево)', 'bad'))
    comp['zone'] = round(c_zone, 1)

    # ---- 4. Order Block support (15) ----
    c_ob = W_OB * 0.35
    at_bull = getattr(res, 'price_at_bullish_ob', False)
    at_bear = getattr(res, 'price_at_bearish_ob', False)
    n_bull = len(getattr(res, 'active_bullish_obs', []) or [])
    n_bear = len(getattr(res, 'active_bearish_obs', []) or [])
    if dir_sign == 1:
        if at_bull:
            c_ob = W_OB; reasons.append(('Ціна на бичачому OB (опора)', 'good'))
        elif n_bull > n_bear:
            c_ob = W_OB * 0.65; reasons.append(('Переважають бичачі OB нижче', 'good'))
        elif at_bear:
            c_ob = W_OB * 0.1; reasons.append(('Ціна під ведмежим OB (опір)', 'bad'))
    else:
        if at_bear:
            c_ob = W_OB; reasons.append(('Ціна на ведмежому OB (опір)', 'good'))
        elif n_bear > n_bull:
            c_ob = W_OB * 0.65; reasons.append(('Переважають ведмежі OB вище', 'good'))
        elif at_bull:
            c_ob = W_OB * 0.1; reasons.append(('Ціна над бичачим OB (підтримка проти SHORT)', 'bad'))
    comp['order_block'] = round(c_ob, 1)

    # ---- 5. Trade context: PnL + TP/SL proximity (10) ----
    c_ctx = W_CONTEXT * 0.4
    entry = position.get('entry_price') or 0
    cur = position.get('current_price') or position.get('mark_price') or 0
    pnl_pct = position.get('pnl_pct')
    if pnl_pct is None and entry and cur:
        pnl_pct = (cur - entry) / entry * 100 * dir_sign
    if pnl_pct is not None:
        if pnl_pct > 0.5:
            c_ctx = W_CONTEXT; reasons.append((f'У прибутку (+{pnl_pct:.2f}%)', 'good'))
        elif pnl_pct < -0.5:
            c_ctx = W_CONTEXT * 0.2; reasons.append((f'У збитку ({pnl_pct:.2f}%)', 'bad'))
        else:
            reasons.append(('Біля беззбитку', 'neutral'))
    comp['context'] = round(c_ctx, 1)

    score = sum(comp.values())
    score = max(0, min(100, round(score, 1)))

    # Four actionable tiers — each maps to a concrete decision.
    if score >= STRONG_HOLD_MIN:
        verdict, color, action = 'STRONG HOLD', 'good', 'Тримати — структура сильно за угоду'
    elif score >= HOLD_MIN:
        verdict, color, action = 'HOLD', 'good', 'Тримати — картина сприятлива'
    elif score >= REDUCE_MIN:
        verdict, color, action = 'REDUCE', 'neutral', 'Слабшає — зменшити або підтягнути стоп'
    else:
        verdict, color, action = 'EXIT', 'bad', 'Структура проти — перспектив мало'

    # Headline: the single most decisive reason behind the verdict. Pick the
    # component that moved the score furthest from its own neutral baseline,
    # so the trader sees WHY at a glance.
    baselines = {'htf_structure': W_HTF_STRUCT * 0.3, 'ltf_break': W_LTF_BREAK * 0.35,
                 'zone': W_ZONE * 0.35, 'order_block': W_OB * 0.35,
                 'context': W_CONTEXT * 0.4}
    labels = {'htf_structure': 'HTF тренд', 'ltf_break': 'Злам структури',
              'zone': 'Зона ціни', 'order_block': 'Order Block', 'context': 'P&L'}
    dominant = max(comp, key=lambda k: abs(comp[k] - baselines.get(k, 0)))
    dom_dir = 'за угоду' if comp[dominant] >= baselines.get(dominant, 0) else 'проти угоди'
    headline = f'{labels[dominant]}: {dom_dir}'

    return {
        'ok': True,
        'symbol': position.get('symbol'),
        'side': side,
        'score': score,
        'verdict': verdict,
        'color': color,
        'action': action,
        'headline': headline,
        'components': comp,
        'reasons': reasons,
    }
