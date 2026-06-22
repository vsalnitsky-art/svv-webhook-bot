# Quality Gate V2 — Professional Trade Scoring System

**Version:** 2.0  
**Date:** 2026-06-22  
**Status:** Production

## Executive Summary

Quality Gate V2 is a complete rewrite of the trade quality scoring system, fixing critical methodological flaws in V1 that made it ineffective:

- **V1 Problem**: 80% of scoring weight was one variable counted three times (ADR room + exhaustion + verdict)
- **V2 Solution**: Independent factors with no overlap (ADR room, HTF alignment, ATR, Decision score)
- **V1 Problem**: Factors contradicted each other (ADR rewarded fresh, exhaustion/verdict punished fresh)
- **V2 Solution**: Consistent directional logic, exhaustion as kill-switch (not scored)
- **V1 Problem**: Overfitting on 142 trades (marginal>good paradox, no train/test split)
- **V2 Solution**: Cleaned noise weights, included validation tooling for out-of-sample testing
- **V1 Problem**: Filter threshold 40 blocked almost nothing (grades started at 40)
- **V2 Solution**: Threshold 50 blocks bottom half (grades 75/60/45), meaningful filtering

## Scoring Formula

**Total: 100 points** from 4 independent factors:

### 1. ADR Room (35 points) — Primary Predictor
**Metric:** `100 - adr_used_pct`  
**Empirical basis:** Winners had 36.5% room, losers 9.8% (Δ=-26.67%)

```
≥35% room  → 35 pts
≥25% room  → 28 pts
≥15% room  → 18 pts
≥5% room   → 8 pts
<5% room   → 0 pts
```

**Why this matters:** When most of the daily range is already used up, there's no physical room for the trade to profit. This was the strongest empirical predictor in archive analysis.

### 2. HTF Trend Alignment (30 points) — New in V2
**Metric:** Smart Direction result (trading WITH vs. AGAINST 4H trend)  
**Empirical basis:** Contra-trend trades have lower success rates

```
Aligned with 4H         → 30 pts
Contra-trend (blocked)  → 0 pts
WAIT mode / no data     → 15 pts (neutral)
```

**Why this matters:** Higher timeframe trend provides directional edge. V1 had no HTF component.

**Current status:** Not yet integrated in Trade Manager (returns 15 pts). Can be enabled by passing `compute_smart_direction()` result to `calculate_quality_score_v2()`.

### 3. ATR Volatility (20 points)
**Metric:** `move.atr_pct` (percentage)  
**Empirical basis:** Winners 0.53%, losers 0.69% — calmer is tradeable

```
<0.4%   → 20 pts
<0.55%  → 15 pts
<0.7%   → 10 pts
<0.9%   → 5 pts
≥0.9%   → 0 pts
```

**Why this matters:** Excessive volatility increases slippage and stop-hunt risk.

### 4. Decision Center Score (15 points)
**Metric:** Numeric score from `evaluate_entry()` (composite of HTF/Forecast/CTR)  
**Range:** Typically -10 to +30, threshold usually 10

```
≥20  → 15 pts
≥10  → 12 pts
≥0   → 8 pts
<0   → 0 pts
```

**Why this matters:** Decision score already aggregates multiple independent signals (HTF bias, forecast alignment, CTR extremes). Using the numeric score (not verdict) avoids the noise present in small-sample categorical labels.

**V1 mistake:** Used `decision.verdict` ('good'|'marginal'|'poor'), which showed "marginal > good" paradox due to overfitting on 142 trades.

### Kill-Switch: Exhaustion >85%
**Not scored** — instant block with grade `BLOCKED` and score 0.

**Why separate:** Exhaustion is already a composite of ADR used, stretch, and runway. Scoring it would triple-count ADR. Instead, use it as a hard safety cutoff.

## Grades

```
EXCELLENT:  75-100  (top quartile, strong setup)
GOOD:       60-74   (above average)
FAIR:       45-59   (mediocre, borderline)
POOR:       0-44    (weak setup, avoid)
BLOCKED:    exhaustion >85% (auto-reject)
```

**Default filter threshold:** 50 (blocks POOR + lower FAIR)

## Usage

### In Trade Manager (Automated)

Quality Gate V2 is already integrated into `detection/trade_manager.py`:

```python
# Automatically called in on_signal() when use_quality_gate=True
quality_result = self._calculate_quality_score(symbol, side)
# Returns: {'score': int, 'grade': str, 'breakdown': dict, 'reason': str, 'blocked': bool}
```

**Settings (via UI or API):**
```python
{
  'use_quality_gate': False,           # Master toggle
  'quality_gate_mode': 'advisory',     # 'off' | 'advisory' | 'filter'
  'quality_gate_threshold': 50,        # Min score for filter mode
}
```

### Standalone Validation

```python
from detection.quality_gate_v2 import validate_on_archive

# Run train/test split validation on trade archive
validate_on_archive('/path/to/trade_archive.json', test_size=0.3)
```

**Output:**
- Train set metrics (70% of trades)
- Test set metrics (30% of trades, out-of-sample)
- Win rate by grade bucket
- Score distribution

**Interpretation:**
- Train/test win rates similar by grade → model generalizes
- Large divergence → overfitting (need more data or simpler formula)

## V1 Audit (What Was Wrong)

### Problem 1: Triple-Counting ADR

**V1 scoring:**
- ADR room: 35 pts (based on `100 - adr_used_pct`)
- Exhaustion: 25 pts (composite that **includes** `adr_used_pct` as a sub-component)
- Move verdict: 20 pts (derived from exhaustion thresholds: <40%=FRESH, 40-66%=MATURE, >66%=EXHAUSTED)

**Result:** 80 of 100 points were the same variable measured three times. This is multicollinearity — the model thinks it's looking at 3 independent signals but it's really just one.

**V2 fix:** ADR room (35 pts) represents runway. Exhaustion is a kill-switch (>85% = block). Verdict removed entirely.

### Problem 2: Contradictory Factors

**V1 logic:**
- ADR room ≥30% → max points (rewards fresh moves)
- Exhaustion <30% → penalty (punishes fresh moves)
- Verdict = FRESH → 12 pts, MATURE → 20 pts (punishes fresh)

**Result:** Truly fresh setups were simultaneously rewarded (ADR) and punished (exhaustion + verdict). Scores bunched toward the middle, filter did nothing useful.

**V2 fix:** Consistent logic. High ADR room = good. Exhaustion >85% = hard stop. No contradiction.

### Problem 3: Overfitting on 142 Trades

**V1 empirical claims:**
- `marginal` verdict = 72.4% win rate
- `good` verdict = 57.6% win rate
- MATURE > FRESH

**Reality:** With 142 trades × 5 weighted factors, these percentages are **in-sample noise**, not predictive patterns. The "marginal > good" paradox is logically impossible on healthy data.

**V2 fix:**
- Use numeric `decision.score` (not categorical verdict)
- Remove MATURE>FRESH noise
- Included `validate_on_archive()` for train/test split

### Problem 4: Threshold That Blocks Nothing

**V1:**
- Grade FAIR starts at 40
- Default threshold: 40
- Filter mode blocks only POOR (<40)

**Problem:** With triple-counting compressing scores toward middle, almost nothing scored <40. Filter was effectively disabled.

**V2 fix:**
- Grade thresholds: 75/60/45
- Default threshold: 50 (blocks bottom half)
- Real filtering

## Integration Checklist

- [x] V2 scoring function (`quality_gate_v2.py`)
- [x] Integrated into Trade Manager (`trade_manager.py`)
- [x] Updated default threshold (40 → 50)
- [x] Updated UI hints (smart_money.html)
- [x] Validation tooling (train/test split)
- [ ] Smart Direction integration (currently neutral 15/30 pts)
- [ ] Backtest on new archive data (when available)

## Future Work

1. **HTF Alignment Integration**: Pass `compute_smart_direction()` result to unlock full 30-point HTF component
2. **Live Validation**: Collect new archive with V2 scores, run out-of-sample validation
3. **Adaptive Thresholds**: Adjust threshold based on recent win rates (online learning)
4. **Per-Symbol Calibration**: Different ADR/ATR norms per asset class

## Technical Details

**File:** `/home/user/svv-webhook-bot/detection/quality_gate_v2.py`  
**Function:** `calculate_quality_score_v2(symbol, side, scanner, smart_direction_result=None)`  
**Dependencies:**
- `scanner.get_bias(symbol)` → provides `move` and `decision` data
- `smart_direction_result` (optional) → provides HTF alignment

**Error handling:** Returns neutral score (50, FAIR) on any error, never raises.

## References

- Original analysis: 142 closed trades (archive)
- Strongest predictor: ADR room (Δ=-26.67% between winners/losers)
- V1 implementation: trade_manager.py lines 964-1109 (deprecated 2026-06-22)
- V2 implementation: quality_gate_v2.py (production since 2026-06-22)
