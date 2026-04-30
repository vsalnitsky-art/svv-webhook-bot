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
