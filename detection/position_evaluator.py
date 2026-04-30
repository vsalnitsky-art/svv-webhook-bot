"""Position Health Score — rule-based expert system that aggregates every
indicator the bot tracks into a single -100..+100 score for an open position.

The score is purely advisory in this iteration: it is computed and surfaced
in the UI/Telegram, but actual position closing remains the responsibility
of the existing exit rules in trade_manager.py (Reverse SMC, Forecast 1H
Confluence, HTF Flip, Time Stop, Trailing, BE, BOS-N partials).

Design principles:
  - PURE: no side effects, no I/O, no DB writes — easy to test and reason
    about. Caller assembles the inputs, evaluator returns a dict.
  - TRANSPARENT: every contribution to the final score comes back in
    `components` so the user sees exactly why a position is "healthy" or
    "in trouble".
  - CONFIGURABLE: weights live in EvaluationConfig and are intended to be
    UI-tunable. Defaults below come from the Balanced preset.
  - DIRECTIONAL: same indicator means different things for LONG vs SHORT.
    The signed_for(side, raw_dir) helper handles the sign flipping in one
    place so individual scorers stay readable.

Score scale:
    +100  = strongest possible alignment with position direction (rare)
    +50   = solid alignment, all indicators favorable
       0  = neutral / no consensus
    -50   = several signals against the position
    -100  = all indicators screaming reverse (very rare)

The threshold for advisory CLOSE recommendation is configurable. Common
presets:
    Aggressive:   threshold = -20  (close at first sign of trouble)
    Balanced:     threshold = -40  (default — needs consensus of negatives)
    Conservative: threshold = -60  (only close on strong reversal signal)
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import time


# ============================================================
# Configuration
# ============================================================

@dataclass
class EvaluationConfig:
    """Weight configuration for each scoring component.
    
    Each weight represents the MAXIMUM points a component can contribute,
    in either direction. The final per-component contribution is in the
    range [-weight, +weight]. The total score is clamped to [-100, +100].
    
    Defaults are tuned for "Balanced" preset:
      - HTF and Forecast carry the most weight (they describe the macro)
      - LTF structure (CHoCH/BOS) and CTR carry medium weight
      - PnL/time decay carry the least — they are descriptive of the
        position state itself, not the market.
    """
    # Macro alignment (highest impact)
    weight_htf_alignment: float = 25.0       # HTF Bias direction
    weight_forecast_alignment: float = 30.0  # Forecast 1H side · confidence
    
    # Microstructure
    weight_ltf_choch: float = 25.0           # opposite CHoCH after entry
    weight_ltf_bos: float = 12.0             # BOS in our direction (continuation)
    
    # Cycle / momentum
    weight_ctr_alignment: float = 15.0       # CTR signal direction
    weight_ctr_zone: float = 10.0            # Overbought/Oversold risk
    
    # Position state
    weight_pnl_momentum: float = 10.0        # PnL trajectory (peak vs current)
    weight_time_decay: float = 10.0          # stale position with no progress
    
    # Threshold for advisory CLOSE recommendation
    threshold: float = -40.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'EvaluationConfig':
        valid = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)


PRESETS = {
    'aggressive': {
        'threshold': -20.0,
    },
    'balanced': {
        'threshold': -40.0,
    },
    'conservative': {
        'threshold': -60.0,
    },
}


# ============================================================
# Helpers
# ============================================================

def _signed_for_side(side: str, raw_dir: str) -> int:
    """Convert a raw direction ('bull'/'bear', 'LONG'/'SHORT', or 1/-1) into
    a signed value relative to the position's side.
    
    Returns +1 if raw_dir aligns WITH the position direction,
            -1 if it aligns AGAINST,
             0 if neutral/unknown.
    """
    if raw_dir is None:
        return 0
    
    # Normalize raw_dir
    if isinstance(raw_dir, (int, float)):
        if raw_dir > 0:
            raw = 'bull'
        elif raw_dir < 0:
            raw = 'bear'
        else:
            return 0
    else:
        s = str(raw_dir).lower()
        if s in ('bull', 'long', 'buy', 'up', '1', '+1'):
            raw = 'bull'
        elif s in ('bear', 'short', 'sell', 'down', '-1'):
            raw = 'bear'
        else:
            return 0
    
    if side == 'LONG':
        return 1 if raw == 'bull' else -1
    elif side == 'SHORT':
        return 1 if raw == 'bear' else -1
    return 0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ============================================================
# Individual scorers — each returns (signed_score, label)
# ============================================================

def _score_htf_alignment(side: str, htf_bias: str, weight: float) -> Tuple[float, str]:
    """HTF Bias is the macro direction filter. If position aligns with HTF
    we get the full positive weight. If opposite — full negative. Neutral
    HTF returns 0 (no information either way)."""
    if not htf_bias or htf_bias == 'neutral':
        return 0.0, 'HTF: neutral'
    
    sign = _signed_for_side(side, htf_bias)
    if sign == 0:
        return 0.0, 'HTF: unknown'
    
    score = weight * sign
    label = f"HTF {htf_bias}: {'+' if sign > 0 else ''}{score:.0f}"
    return score, label


def _score_forecast_alignment(side: str, forecast: Optional[Dict], weight: float) -> Tuple[float, str]:
    """Forecast 1H. Score combines side AND confidence:
        sign × (confidence/100) × weight
    
    Forecast confidence ranges 50/60/75/90 for non-trivial signals,
    or 0 for "no signal". A confidence-90 LONG forecast on a LONG
    position gives +0.9·weight. Opposite SHORT forecast gives -0.9·weight.
    """
    if not forecast:
        return 0.0, 'Forecast: —'
    
    side_n = forecast.get('side', 0)  # +1 / 0 / -1
    conf = forecast.get('confidence', 0) or 0
    if conf <= 0 or side_n == 0:
        return 0.0, 'Forecast: neutral'
    
    sign = _signed_for_side(side, side_n)
    if sign == 0:
        return 0.0, 'Forecast: unclear'
    
    score = weight * sign * (conf / 100.0)
    fc_side = 'LONG' if side_n > 0 else 'SHORT'
    label = f"Forecast {fc_side} {conf}%: {'+' if score > 0 else ''}{score:.0f}"
    return score, label


def _score_ltf_choch(side: str, recent_choch: Optional[Dict], weight: float) -> Tuple[float, str]:
    """A CHoCH on the LTF (15M) is the bot's primary reversal warning.
    A CHoCH AGAINST our position direction means structure is breaking
    against us — full negative weight. A CHoCH WITH our direction is
    a fresh reaffirmation — small positive (we don't double-count, BOS
    handles continuation more reliably).
    
    `recent_choch` is the most recent CHoCH event observed AFTER the
    position was opened: {dir, t, level} or None.
    """
    if not recent_choch:
        return 0.0, 'CHoCH: none after open'
    
    sign = _signed_for_side(side, recent_choch.get('dir'))
    if sign == 0:
        return 0.0, 'CHoCH: unclear'
    
    if sign > 0:
        # Same-direction CHoCH after open is rare and weak as a "good news"
        # signal, but worth a small bump
        score = weight * 0.3
        label = f"CHoCH same-dir: +{score:.0f}"
    else:
        # Opposite CHoCH — this is the canonical reversal warning
        score = -weight
        label = f"CHoCH against: {score:.0f}"
    return score, label


def _score_ltf_bos(side: str, bos_count_with: int, bos_count_against: int,
                   weight: float) -> Tuple[float, str]:
    """BOS counts AFTER position entry. Each continuation BOS in our
    direction is positive (trend confirmation). BOS against — negative,
    but capped (CHoCH already handled the structural reversal; BOS
    against without prior CHoCH is unusual and we shouldn't double-penalize).
    
    Diminishing returns: 1st BOS = full weight, 2nd = +50%, 3rd+ = +25%
    of the weight per event, capped at 1.5× weight.
    """
    pos_score = 0.0
    if bos_count_with > 0:
        # Diminishing returns
        pos_score = weight * (1.0 + 0.5 * min(bos_count_with - 1, 1) +
                              0.25 * max(0, bos_count_with - 2))
        pos_score = min(pos_score, weight * 1.5)
    
    neg_score = 0.0
    if bos_count_against > 0:
        # Capped at -weight, regardless of count (CHoCH carries reversal)
        neg_score = -min(weight, weight * 0.5 * bos_count_against)
    
    score = pos_score + neg_score
    parts = []
    if bos_count_with:
        parts.append(f"{bos_count_with}× with")
    if bos_count_against:
        parts.append(f"{bos_count_against}× against")
    detail = ', '.join(parts) if parts else 'none'
    label = f"BOS ({detail}): {'+' if score >= 0 else ''}{score:.0f}"
    return score, label


def _score_ctr_alignment(side: str, ctr: Optional[Dict], weight: float) -> Tuple[float, str]:
    """CTR (STC-based cycle) momentum. A CTR signal in our direction
    that fired recently is a tailwind. An opposite recent signal is a
    headwind — momentum is rotating against us.
    
    Age decay: signal at age 0-5 bars = full weight, 5-15 = half,
    15+ bars = quarter. After ~30 bars the CTR signal is too stale
    to count.
    """
    if not ctr:
        return 0.0, 'CTR: —'
    
    last_dir = ctr.get('last_dir')
    age = ctr.get('last_signal_age_bars')
    if not last_dir or age is None:
        return 0.0, 'CTR: no signal'
    
    sign = _signed_for_side(side, last_dir)
    if sign == 0:
        return 0.0, 'CTR: unclear'
    
    # Age decay
    if age <= 5:
        decay = 1.0
    elif age <= 15:
        decay = 0.5
    elif age <= 30:
        decay = 0.25
    else:
        return 0.0, f'CTR: too stale ({age} bars)'
    
    score = weight * sign * decay
    label = f"CTR {last_dir} ({age}b): {'+' if score > 0 else ''}{score:.0f}"
    return score, label


def _score_ctr_zone(side: str, ctr: Optional[Dict], weight: float) -> Tuple[float, str]:
    """CTR zone is reversal-risk indicator (independent of cycle direction):
        STC ≥ 75 (Overbought) → reversal-down risk
            → bad for LONG, good for SHORT
        STC ≤ 25 (Oversold)   → reversal-up risk
            → good for LONG, bad for SHORT
        Mid (25-75)           → neutral
    
    The intensity scales linearly with how far past the threshold STC sits.
    """
    if not ctr:
        return 0.0, 'CTR zone: —'
    
    stc = ctr.get('stc')
    if stc is None:
        return 0.0, 'CTR zone: —'
    
    try:
        stc_v = float(stc)
    except Exception:
        return 0.0, 'CTR zone: —'
    
    if stc_v >= 75:
        # Overbought — reversal risk to the downside
        intensity = (stc_v - 75) / 25.0  # 0 at 75, 1 at 100
        intensity = _clamp(intensity, 0.0, 1.0)
        zone = 'Overbought'
        # If LONG: risk of reversal down = bad → negative
        # If SHORT: same risk = good → positive
        sign = -1 if side == 'LONG' else 1
        score = weight * intensity * sign
    elif stc_v <= 25:
        intensity = (25 - stc_v) / 25.0  # 0 at 25, 1 at 0
        intensity = _clamp(intensity, 0.0, 1.0)
        zone = 'Oversold'
        sign = 1 if side == 'LONG' else -1
        score = weight * intensity * sign
    else:
        return 0.0, f'CTR zone: Mid ({stc_v:.0f})'
    
    label = f"CTR {zone} ({stc_v:.0f}): {'+' if score >= 0 else ''}{score:.0f}"
    return score, label


def _score_pnl_momentum(pnl_pct: float, peak_pnl_pct: Optional[float],
                        weight: float) -> Tuple[float, str]:
    """How is the PnL trajectory? Three regimes:
    
    1) Position currently profitable AND at peak (or within 80% of peak):
       Strong momentum, give bonus → +weight × (pnl/peak ratio)
    
    2) Position profitable but well off peak (giving back gains):
       Trail risk — we should be locking in. Negative scaled by giveback.
    
    3) Position currently losing:
       Mild negative — magnitude based on how deep we are.
    """
    if peak_pnl_pct is None or peak_pnl_pct == 0:
        peak_pnl_pct = max(pnl_pct, 0)
    
    # Regime 1: solidly profitable
    if pnl_pct > 0.5 and peak_pnl_pct > 0.5:
        ratio = pnl_pct / peak_pnl_pct  # 1.0 = at peak, 0.5 = halfway down
        if ratio >= 0.8:
            # Holding gains — positive bonus
            score = weight * 0.6 * ratio
            return score, f"PnL holding peak ({pnl_pct:+.2f}%): +{score:.0f}"
        else:
            # Giving back gains — negative
            giveback = 1.0 - ratio
            score = -weight * giveback
            return score, f"PnL giving back ({pnl_pct:+.2f}%, peak {peak_pnl_pct:+.2f}%): {score:.0f}"
    
    # Regime 3: in loss
    if pnl_pct < 0:
        # Mild negative scaling — we don't want to compound stop-loss logic
        magnitude = min(abs(pnl_pct) / 3.0, 1.0)  # capped at -3% loss
        score = -weight * 0.5 * magnitude
        return score, f"PnL losing ({pnl_pct:+.2f}%): {score:.0f}"
    
    # Regime 2: small profit, no clear peak yet
    return 0.0, f"PnL small ({pnl_pct:+.2f}%)"


def _score_time_decay(opened_at: float, pnl_pct: float, weight: float) -> Tuple[float, str]:
    """Time-in-position penalty for stale unproductive positions.
    
    A position open for >4 hours with PnL still near zero (or negative)
    is consuming opportunity cost — even if no exit signal has fired.
    Scale linearly: at 4h with PnL ≤ 0.5%, score = -0.5×weight.
    At 8h+, score = -weight.
    
    Productive positions (PnL > 1%) get a small bonus for the first
    couple of hours then drift toward 0.
    """
    if not opened_at:
        return 0.0, 'Time: —'
    
    age_hours = (time.time() - float(opened_at)) / 3600.0
    
    # Stale unproductive
    if age_hours > 4 and pnl_pct < 0.5:
        # Linear from 0 at 4h to -1 at 8h+
        intensity = _clamp((age_hours - 4) / 4.0, 0.0, 1.0)
        score = -weight * intensity
        return score, f"Stale {age_hours:.1f}h, PnL {pnl_pct:+.2f}%: {score:.0f}"
    
    # Fresh productive
    if age_hours < 2 and pnl_pct > 1.0:
        score = weight * 0.3
        return score, f"Fresh productive ({age_hours:.1f}h): +{score:.0f}"
    
    return 0.0, f"Time {age_hours:.1f}h: 0"


# ============================================================
# Main entry point
# ============================================================

def evaluate_position(
    side: str,
    pnl_pct: float,
    opened_at: float,
    htf_bias: str,
    forecast: Optional[Dict],
    ctr: Optional[Dict],
    recent_choch: Optional[Dict],
    bos_count_with: int,
    bos_count_against: int,
    peak_pnl_pct: Optional[float] = None,
    config: Optional[EvaluationConfig] = None,
) -> Dict:
    """Compute the Position Health Score for an open position.
    
    Args:
        side: 'LONG' or 'SHORT'
        pnl_pct: current PnL in percent
        opened_at: unix timestamp of position open
        htf_bias: 'bull' | 'bear' | 'neutral'
        forecast: forecast_engine cache for this symbol or None
        ctr: CTR result dict or None
        recent_choch: last CHoCH event observed AFTER opened_at, or None
        bos_count_with: BOS events in our direction since open
        bos_count_against: BOS events against us since open
        peak_pnl_pct: highest PnL reached during this position's lifetime
        config: optional weight configuration (uses balanced defaults if None)
    
    Returns:
        {
            'score': float in [-100, 100],
            'verdict': 'healthy' | 'caution' | 'close',
            'threshold': float (config.threshold),
            'components': [{'name': str, 'value': float, 'label': str}, ...],
            'summary': human-readable one-liner
        }
    """
    cfg = config or EvaluationConfig()
    
    components: List[Dict] = []
    
    def add(name: str, scorer_result: Tuple[float, str]):
        v, lbl = scorer_result
        components.append({
            'name': name,
            'value': round(v, 2),
            'label': lbl,
        })
    
    add('htf', _score_htf_alignment(side, htf_bias, cfg.weight_htf_alignment))
    add('forecast', _score_forecast_alignment(side, forecast, cfg.weight_forecast_alignment))
    add('ltf_choch', _score_ltf_choch(side, recent_choch, cfg.weight_ltf_choch))
    add('ltf_bos', _score_ltf_bos(side, bos_count_with, bos_count_against, cfg.weight_ltf_bos))
    add('ctr_align', _score_ctr_alignment(side, ctr, cfg.weight_ctr_alignment))
    add('ctr_zone', _score_ctr_zone(side, ctr, cfg.weight_ctr_zone))
    add('pnl', _score_pnl_momentum(pnl_pct, peak_pnl_pct, cfg.weight_pnl_momentum))
    add('time', _score_time_decay(opened_at, pnl_pct, cfg.weight_time_decay))
    
    raw_total = sum(c['value'] for c in components)
    score = round(_clamp(raw_total, -100.0, 100.0), 1)
    
    # Verdict
    if score <= cfg.threshold:
        verdict = 'close'
    elif score < 0:
        verdict = 'caution'
    else:
        verdict = 'healthy'
    
    # Summary — two strongest negative components, for at-a-glance review
    sorted_neg = sorted([c for c in components if c['value'] < 0],
                        key=lambda c: c['value'])
    sorted_pos = sorted([c for c in components if c['value'] > 0],
                        key=lambda c: c['value'], reverse=True)
    bullets = []
    for c in sorted_neg[:2]:
        bullets.append('▼ ' + c['label'])
    for c in sorted_pos[:1]:
        bullets.append('▲ ' + c['label'])
    summary = ' · '.join(bullets) if bullets else 'No strong signals'
    
    return {
        'score': score,
        'verdict': verdict,
        'threshold': cfg.threshold,
        'components': components,
        'summary': summary,
    }


# ============================================================
# ENTRY-SIDE EVALUATION
# ============================================================
# This is the dual of evaluate_position(): instead of asking "should we
# stay in this position?", it asks "should we open this position?"
#
# Key differences from exit-side scoring:
#   - PnL momentum and time decay are gone (no position exists yet)
#   - SMC structure quality is added: how fresh is the CHoCH, how close
#     to a Strong Low / Weak High pivot, how strong is the BOS leg
#   - Premium/Discount: opening a LONG in Premium (top 38.2%) is bad RR;
#     opening in Discount is great. Mirror image for SHORT.
#   - Volume confirmation: was there a volume spike on the signal bar?
#   - ATR volatility: is current ATR healthy or anaemic / extreme?
#
# Like exit-side, this is ADVISORY in this iteration. It produces a score
# in [-100, +100] which is logged, surfaced in Telegram OPEN messages, and
# stored on the position so the UI can show the "score at entry" badge.
# It does NOT block opens.
# ============================================================


@dataclass
class EntryEvaluationConfig:
    """Weights for the Entry Score components.
    
    Different from EvaluationConfig because the factor mix is different.
    Each weight = max points the component can contribute. Final score
    is clamped to [-100, +100].
    
    Defaults below are tuned for the "Balanced" preset, which requires a
    moderate consensus of positives. Aggressive lowers the threshold
    (more signals pass), Conservative raises it.
    """
    # Macro confluence
    weight_htf_alignment: float = 25.0       # HTF Bias direction
    weight_forecast_alignment: float = 25.0  # Forecast 1H side · confidence
    weight_ctr_alignment: float = 15.0       # CTR signal direction
    weight_ctr_zone: float = 10.0            # Overbought (LONG bad) etc.
    
    # SMC structure quality
    weight_choch_freshness: float = 12.0     # how fresh is the CHoCH
    weight_pivot_proximity: float = 12.0     # near Strong Low / Weak High?
    weight_pd_zone: float = 15.0             # Premium/Discount/Equilibrium
    
    # Market state
    weight_volume_confirmation: float = 8.0  # signal bar volume vs avg
    weight_atr_health: float = 8.0           # ATR not too low / extreme
    
    # Threshold for advisory ENTRY recommendation
    threshold: float = 30.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'EntryEvaluationConfig':
        valid = {k: v for k, v in (d or {}).items()
                 if k in cls.__dataclass_fields__}
        return cls(**valid)


# Three presets — same shape as exit, different magnitudes. Threshold is
# POSITIVE for entry (we want a signal good enough to act on), unlike exit
# where we want a NEGATIVE score that's bad enough to bail.
ENTRY_PRESETS = {
    'aggressive':   {'threshold': 10.0},   # take almost any signal
    'balanced':     {'threshold': 30.0},   # require modest consensus
    'conservative': {'threshold': 50.0},   # only A-grade setups
}


# ============================================================
# Entry-side scorers
# ============================================================

def _score_choch_freshness(choch_age_bars: Optional[int],
                           weight: float) -> Tuple[float, str]:
    """A fresh CHoCH (just happened) is the strongest version of the signal.
    Older CHoCH means structure has had time to evolve — weaker conviction.
    
    age 0-2:  full positive
    age 3-5:  half
    age 6-15: quarter
    age 16+:  zero (signal got stale)
    """
    if choch_age_bars is None:
        return 0.0, 'CHoCH age: unknown'
    if choch_age_bars <= 2:
        score, lvl = weight, 'fresh'
    elif choch_age_bars <= 5:
        score, lvl = weight * 0.5, 'recent'
    elif choch_age_bars <= 15:
        score, lvl = weight * 0.25, 'older'
    else:
        return 0.0, f'CHoCH stale ({choch_age_bars}b)'
    return score, f'CHoCH {lvl} ({choch_age_bars}b): +{score:.0f}'


def _score_pivot_proximity(side: str, entry_price: float,
                           strong_low: Optional[float],
                           weak_high: Optional[float],
                           atr: Optional[float],
                           weight: float) -> Tuple[float, str]:
    """LONG entries are best when near a Strong Low (HL on bullish trend) —
    structurally well-defended. SHORT entries are best near a Weak High.
    
    Distance is measured in ATRs to be price-scale invariant. Within 2 ATR
    of the relevant pivot = full weight, 2-5 ATR = half, 5-10 ATR = quarter,
    farther = zero. No relevant pivot = zero (no information either way,
    we don't penalize).
    """
    if not atr or atr <= 0:
        return 0.0, 'pivot: ATR unavailable'
    
    if side == 'LONG':
        if strong_low is None:
            return 0.0, 'pivot: no Strong Low'
        dist_atr = abs(entry_price - strong_low) / atr
        target = 'Strong Low'
    else:  # SHORT
        if weak_high is None:
            return 0.0, 'pivot: no Weak High'
        dist_atr = abs(weak_high - entry_price) / atr
        target = 'Weak High'
    
    if dist_atr <= 2:
        score, lvl = weight, 'near'
    elif dist_atr <= 5:
        score, lvl = weight * 0.5, 'medium'
    elif dist_atr <= 10:
        score, lvl = weight * 0.25, 'far'
    else:
        return 0.0, f'pivot: too far ({dist_atr:.1f} ATR)'
    return score, f'{target} {lvl} ({dist_atr:.1f} ATR): +{score:.0f}'


def _score_pd_zone(side: str, entry_price: float,
                   range_high: Optional[float],
                   range_low: Optional[float],
                   weight: float) -> Tuple[float, str]:
    """Premium/Discount classifies where price sits in the active range:
        > 61.8%: Premium (top of range — bad LONG entry, good SHORT)
        < 38.2%: Discount (bottom of range — good LONG, bad SHORT)
        38.2-61.8%: Equilibrium (mild signal either way)
    
    The 38.2 / 61.8 levels mirror Pine's PD zone convention exactly (Fib).
    
    For LONG: Discount = +full, Equilibrium = +0.3·, Premium = -full
    For SHORT: mirror.
    """
    if range_high is None or range_low is None or range_high <= range_low:
        return 0.0, 'PD: range unavailable'
    
    rng = range_high - range_low
    pos = (entry_price - range_low) / rng  # 0 = bottom, 1 = top
    pos = max(0.0, min(1.0, pos))
    
    if pos < 0.382:
        zone, intensity = 'Discount', (0.382 - pos) / 0.382
        # LONG good, SHORT bad
        sign = 1 if side == 'LONG' else -1
    elif pos > 0.618:
        zone, intensity = 'Premium', (pos - 0.618) / 0.382
        # LONG bad, SHORT good
        sign = -1 if side == 'LONG' else 1
    else:
        # Equilibrium — mild bonus toward "fair" entry, negligible
        zone, intensity = 'Equilibrium', 0.0
        sign = 0
    
    intensity = min(1.0, intensity)
    score = weight * sign * intensity if intensity > 0 else weight * 0.1 * (1 if zone == 'Equilibrium' else 0)
    pct = pos * 100
    return score, f'PD {zone} ({pct:.0f}%): {"+" if score >= 0 else ""}{score:.0f}'


def _score_volume_confirmation(signal_volume: Optional[float],
                               avg_volume: Optional[float],
                               weight: float) -> Tuple[float, str]:
    """Did the signal bar have meaningful volume?
        ≥2.0× avg: strong confirmation, +full
        1.5-2.0×: moderate, +0.6
        1.0-1.5×: average, +0.2
        <1.0×: weak, mild negative (low conviction signal)
    """
    if not signal_volume or not avg_volume or avg_volume <= 0:
        return 0.0, 'volume: unavailable'
    
    ratio = signal_volume / avg_volume
    if ratio >= 2.0:
        score = weight
        lvl = f'{ratio:.1f}× strong'
    elif ratio >= 1.5:
        score = weight * 0.6
        lvl = f'{ratio:.1f}× moderate'
    elif ratio >= 1.0:
        score = weight * 0.2
        lvl = f'{ratio:.1f}× average'
    else:
        score = -weight * 0.5
        lvl = f'{ratio:.1f}× weak'
    return score, f'Vol {lvl}: {"+" if score >= 0 else ""}{score:.0f}'


def _score_atr_health(atr: Optional[float], price: float,
                      weight: float) -> Tuple[float, str]:
    """Volatility sanity check. We measure ATR as % of price:
        0.5% - 2.5%: healthy range, full positive
        0.2% - 0.5%: low volatility, half (small moves, harder TP)
        2.5% - 5.0%: elevated, half (whippy)
        <0.2% or >5.0%: extreme, negative
    """
    if not atr or not price or price <= 0:
        return 0.0, 'ATR: unavailable'
    
    atr_pct = (atr / price) * 100
    if 0.5 <= atr_pct <= 2.5:
        score = weight
        lvl = 'healthy'
    elif 0.2 <= atr_pct < 0.5:
        score = weight * 0.5
        lvl = 'low'
    elif 2.5 < atr_pct <= 5.0:
        score = weight * 0.5
        lvl = 'elevated'
    elif atr_pct > 5.0:
        score = -weight * 0.5
        lvl = 'extreme'
    else:  # < 0.2%
        score = -weight * 0.3
        lvl = 'flat'
    return score, f'ATR {atr_pct:.2f}% ({lvl}): {"+" if score >= 0 else ""}{score:.0f}'


# ============================================================
# Entry main entry point
# ============================================================

def evaluate_entry(
    side: str,
    entry_price: float,
    htf_bias: str,
    forecast: Optional[Dict],
    ctr: Optional[Dict],
    choch_age_bars: Optional[int] = None,
    strong_low: Optional[float] = None,
    weak_high: Optional[float] = None,
    range_high: Optional[float] = None,
    range_low: Optional[float] = None,
    atr: Optional[float] = None,
    signal_volume: Optional[float] = None,
    avg_volume: Optional[float] = None,
    config: Optional[EntryEvaluationConfig] = None,
) -> Dict:
    """Compute the Entry Score for a fresh SMC signal.
    
    Args:
        side: 'LONG' or 'SHORT'
        entry_price: planned entry price (typically the latest close)
        htf_bias: 'bull' | 'bear' | 'neutral'
        forecast: forecast_engine 1H result or None
        ctr: CTR result dict or None
        choch_age_bars: bars elapsed since the seeding CHoCH
        strong_low: highest HL pivot in current bullish leg, or None
        weak_high: highest LH pivot in current bearish leg, or None
        range_high / range_low: bounds of the active swing range for PD calc
        atr: most recent ATR value (period 14)
        signal_volume: volume of the signal bar
        avg_volume: average volume over recent N bars (e.g. last 20)
        config: weight configuration; uses Balanced defaults if None
    
    Returns:
        {
            'score': float in [-100, 100],
            'verdict': 'good' | 'marginal' | 'poor',
            'threshold': float (config.threshold),
            'components': [{'name', 'value', 'label'}, ...],
            'summary': human-readable headline of best/worst factors
        }
    """
    cfg = config or EntryEvaluationConfig()
    components: List[Dict] = []
    
    def add(name: str, scorer_result: Tuple[float, str]):
        v, lbl = scorer_result
        components.append({
            'name': name,
            'value': round(v, 2),
            'label': lbl,
        })
    
    # Macro factors — reuse exit-side scorers (they're side-aware already)
    add('htf', _score_htf_alignment(side, htf_bias, cfg.weight_htf_alignment))
    add('forecast', _score_forecast_alignment(side, forecast,
                                                cfg.weight_forecast_alignment))
    add('ctr_align', _score_ctr_alignment(side, ctr, cfg.weight_ctr_alignment))
    add('ctr_zone', _score_ctr_zone(side, ctr, cfg.weight_ctr_zone))
    
    # Entry-specific factors
    add('choch_fresh', _score_choch_freshness(choch_age_bars,
                                                cfg.weight_choch_freshness))
    add('pivot_prox', _score_pivot_proximity(side, entry_price,
                                              strong_low, weak_high, atr,
                                              cfg.weight_pivot_proximity))
    add('pd_zone', _score_pd_zone(side, entry_price,
                                    range_high, range_low, cfg.weight_pd_zone))
    add('volume', _score_volume_confirmation(signal_volume, avg_volume,
                                              cfg.weight_volume_confirmation))
    add('atr', _score_atr_health(atr, entry_price, cfg.weight_atr_health))
    
    raw_total = sum(c['value'] for c in components)
    score = round(_clamp(raw_total, -100.0, 100.0), 1)
    
    # Verdict — entry threshold is positive (we want a signal worth taking)
    if score >= cfg.threshold:
        verdict = 'good'
    elif score >= 0:
        verdict = 'marginal'
    else:
        verdict = 'poor'
    
    # Summary
    sorted_pos = sorted([c for c in components if c['value'] > 0],
                        key=lambda c: c['value'], reverse=True)
    sorted_neg = sorted([c for c in components if c['value'] < 0],
                        key=lambda c: c['value'])
    bullets = []
    for c in sorted_pos[:2]:
        bullets.append('▲ ' + c['label'])
    for c in sorted_neg[:1]:
        bullets.append('▼ ' + c['label'])
    summary = ' · '.join(bullets) if bullets else 'No signals available'
    
    return {
        'score': score,
        'verdict': verdict,
        'threshold': cfg.threshold,
        'components': components,
        'summary': summary,
    }
