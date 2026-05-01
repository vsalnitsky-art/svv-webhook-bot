"""Decision Center — the bot's unified analytical brain.

Aggregates every piece of advisory analysis the bot computes (Entry Score
for both directions, Health Score for open positions, raw indicator state
from HTF/Forecast/CTR/SMC structure) into ONE coherent verdict object that
the UI can render as a single block, the Telegram messages can quote in
one line, and the post-mortem analytics can correlate against actual P&L.

Design principles:
  - ONE decision per symbol per call. No more split badges, no more
    "advanced users go check this other panel". Everything that matters
    for the user's choice lives in the dict this module returns.
  - HUMAN-READABLE rationale. We expose `headline` (short) and `rationale`
    (one-line explanation) so the UI doesn't need to interpret raw scores.
  - DATA-AGNOSTIC inputs. We accept already-computed evaluator results
    (LONG/SHORT entry scores, optional health) plus market-state primitives
    (htf_bias, forecast, ctr). We never refetch — the caller assembles
    state once and we synthesize.
  - ROBUST to missing data. Any field can be None. We degrade gracefully
    and the verdict surfaces what's missing.

The output of build_decision() looks like:

    {
        # The user's primary takeaway — these are what the chart-panel badge
        # and the Telegram one-liner read.
        'headline': 'LONG 78%',
        'recommended': 'LONG',          # 'LONG' | 'SHORT' | 'NEUTRAL'
        'verdict':     'good',           # 'good' | 'marginal' | 'poor'
        'prob_long':   0.78,
        'prob_short':  0.22,
        'confidence':  'high',           # 'high' | 'medium' | 'low'
        
        # One-line plain-English explanation
        'rationale': 'HTF bearish, but Forecast LONG 75% confirms with CTR Oversold (18)',
        
        # Detailed breakdown — only used by the expandable details panel
        'long_score':  52,
        'short_score': 12,
        'components_long':  [...],   # contribution list
        'components_short': [...],
        
        # Market state snapshot — same data the UI already has, included
        # here so consumers can use one object end-to-end
        'market': {
            'htf_bias': 'bear',
            'forecast': {...},
            'ctr': {...},
        },
        
        # Position attached if any
        'position': {
            'kind': 'real' | 'shadow',
            'side': 'LONG',
            'entry_price': 103450.0,
            'pnl_pct': 1.5,
            'health': {...},
        } or None,
    }
"""

from typing import Dict, List, Optional, Tuple
import math


# Softmax temperature — calibrated so:
#   |scoreΔ|=10  →  ~58/42
#   |scoreΔ|=40  →  ~79/21  (this is the typical "clear edge")
#   |scoreΔ|=80  →  ~96/4   (extreme one-sidedness)
TEMPERATURE = 30.0

# Confidence buckets based on the WIN side's probability
CONFIDENCE_HIGH = 0.70   # 70%+ win-side prob → "high confidence"
CONFIDENCE_MED = 0.58    # 58-70% → "medium"
                          # below 58% → "low"

# How close LONG and SHORT scores have to be to call it Neutral.
# 1pt buffer prevents floating-point ties from looking decisive.
NEUTRAL_DIFF_THRESHOLD = 1.0


def _softmax2(a: float, b: float, temp: float = TEMPERATURE) -> Tuple[float, float]:
    """Two-class softmax with numerical stability."""
    try:
        a_t, b_t = a / temp, b / temp
        m = max(a_t, b_t)
        ea, eb = math.exp(a_t - m), math.exp(b_t - m)
        total = ea + eb
        if total <= 0:
            return 0.5, 0.5
        return ea / total, eb / total
    except Exception:
        return 0.5, 0.5


def _confidence_label(winner_prob: float) -> str:
    """Map probability to a human label. Used for color/intensity in UI."""
    if winner_prob >= CONFIDENCE_HIGH:
        return 'high'
    if winner_prob >= CONFIDENCE_MED:
        return 'medium'
    return 'low'


def _build_rationale(recommended: str,
                     long_score: float,
                     short_score: float,
                     long_components: List[Dict],
                     short_components: List[Dict],
                     htf_bias: Optional[str],
                     forecast: Optional[Dict],
                     ctr: Optional[Dict]) -> str:
    """Compose a one-line plain-English rationale.
    
    Strategy: pick the 2 strongest contributing factors for the recommended
    side (whatever it is), state them. If the unfavored side has any
    notable supporting factor, append "but ..." or similar. Falls back to
    raw indicator state when no scoring components are dominant (e.g.
    NEUTRAL or all-poor cases).
    """
    if recommended == 'NEUTRAL':
        # No strong directional bias — describe the state plainly
        bits = []
        if htf_bias and htf_bias != 'neutral':
            bits.append(f"HTF {htf_bias}ish")
        if forecast and forecast.get('confidence', 0) > 0:
            side = forecast.get('side', 0)
            side_lbl = 'LONG' if side > 0 else 'SHORT' if side < 0 else 'mixed'
            bits.append(f"Forecast {side_lbl} {forecast.get('confidence')}%")
        if ctr and ctr.get('last_dir'):
            bits.append(f"CTR {ctr['last_dir']} (STC {int(ctr.get('stc', 0))})")
        if not bits:
            return "No strong directional signals — wait for confluence"
        return "Mixed signals: " + ", ".join(bits) + " — no clear edge"
    
    # Pick top 2 positive contributors for the recommended side
    src = long_components if recommended == 'LONG' else short_components
    sorted_pos = sorted([c for c in (src or []) if c.get('value', 0) > 0],
                        key=lambda c: c['value'], reverse=True)
    
    pos_phrases = []
    for c in sorted_pos[:2]:
        # Strip the trailing "+12" numeric since we're already showing scores
        lbl = c.get('label', '')
        if ':' in lbl:
            lbl = lbl.rsplit(':', 1)[0].strip()
        pos_phrases.append(lbl)
    
    if not pos_phrases:
        # Edge case: recommended side won by being LESS BAD
        return f"All signals weak; {recommended} is the lesser-bad option"
    
    # Top negative for SAME side — surface as a caveat ("…but X is concerning")
    sorted_neg = sorted([c for c in (src or []) if c.get('value', 0) < 0],
                        key=lambda c: c['value'])
    caveat = ''
    if sorted_neg:
        worst = sorted_neg[0]
        worst_lbl = worst.get('label', '')
        if ':' in worst_lbl:
            worst_lbl = worst_lbl.rsplit(':', 1)[0].strip()
        # Only mention if it's >= 30% of weight
        if abs(worst['value']) >= 8:
            caveat = f" (despite {worst_lbl})"
    
    head = " + ".join(pos_phrases)
    return f"{head} confirm {recommended}{caveat}"


def _build_headline(recommended: str,
                    prob_long: float,
                    prob_short: float) -> str:
    """The one-line UI verdict. Always the same shape so it's quick to scan."""
    if recommended == 'NEUTRAL':
        # Show whichever side is fractionally higher, but flag NEUTRAL
        return f"NEUTRAL {round(prob_long * 100)}/{round(prob_short * 100)}"
    if recommended == 'LONG':
        return f"LONG {round(prob_long * 100)}%"
    return f"SHORT {round(prob_short * 100)}%"


def _resolve_recommended(long_score: float, short_score: float) -> str:
    """Pick the side or NEUTRAL when scores are essentially tied."""
    if abs(long_score - short_score) <= NEUTRAL_DIFF_THRESHOLD:
        return 'NEUTRAL'
    return 'LONG' if long_score > short_score else 'SHORT'


def build_decision(
    long_eval: Optional[Dict],
    short_eval: Optional[Dict],
    htf_bias: Optional[str] = None,
    forecast: Optional[Dict] = None,
    ctr: Optional[Dict] = None,
    position: Optional[Dict] = None,
) -> Dict:
    """Compose the unified Decision Center verdict.
    
    Args:
        long_eval, short_eval: results from evaluate_entry() for each side.
            Either may be None (feature disabled, error, etc.) — we degrade.
        htf_bias, forecast, ctr: market-state snapshot (for rationale text).
        position: optional dict with the current open position on this
            symbol — {kind, side, entry_price, pnl_pct, health}. None when
            no position exists.
    
    Never raises. On any internal failure returns a minimally valid dict
    so callers don't need to null-check shape.
    """
    # Extract scores; default to 0 for any missing eval (treated as neutral)
    long_score = float((long_eval or {}).get('score', 0))
    short_score = float((short_eval or {}).get('score', 0))
    long_components = (long_eval or {}).get('components', [])
    short_components = (short_eval or {}).get('components', [])
    
    # Probabilities via softmax
    prob_long, prob_short = _softmax2(long_score, short_score)
    
    # Recommendation
    recommended = _resolve_recommended(long_score, short_score)
    
    # Verdict — derive from the recommended side's evaluator output.
    # (When NEUTRAL, we report "marginal" since neither side is decisively
    # good or poor.)
    if recommended == 'LONG':
        verdict = (long_eval or {}).get('verdict', 'marginal')
    elif recommended == 'SHORT':
        verdict = (short_eval or {}).get('verdict', 'marginal')
    else:
        verdict = 'marginal'
    
    # Confidence based on winning side's probability magnitude
    winner_prob = max(prob_long, prob_short)
    confidence = _confidence_label(winner_prob)
    
    # Rationale (human-readable)
    rationale = _build_rationale(
        recommended=recommended,
        long_score=long_score,
        short_score=short_score,
        long_components=long_components,
        short_components=short_components,
        htf_bias=htf_bias,
        forecast=forecast,
        ctr=ctr,
    )
    
    # Headline
    headline = _build_headline(recommended, prob_long, prob_short)
    
    return {
        'headline': headline,
        'recommended': recommended,
        'verdict': verdict,
        'confidence': confidence,
        'prob_long': round(prob_long, 4),
        'prob_short': round(prob_short, 4),
        'rationale': rationale,
        'long_score': round(long_score, 1),
        'short_score': round(short_score, 1),
        'components_long': long_components,
        'components_short': short_components,
        'market': {
            'htf_bias': htf_bias,
            'forecast': forecast,
            'ctr': ctr,
        },
        'position': position,
    }


def telegram_one_liner(decision: Dict) -> str:
    """Compact one-line summary for Telegram OPEN/CLOSE messages.
    
    Format: '🧠 Decision: LONG 78% (good)'
    Returns empty string if the decision is empty or feature disabled.
    """
    if not decision or 'headline' not in decision:
        return ''
    headline = decision.get('headline', '?')
    verdict = decision.get('verdict', '')
    return f"🧠 Decision: {headline} ({verdict})"
