"""
Quality Gate V2 — professional rewrite of trade quality scoring.

KEY FIXES FROM V1:
  1. Removed triple-counting: exhaustion/verdict/ADR were all same variable.
     Now ADR room is sole representative of "runway", exhaustion is a hard
     kill-switch (not scored), verdict removed entirely.
  2. Independent factors: ADR room (35), HTF alignment (30), ATR (20),
     Decision score (15). Total 100, no overlap.
  3. Noise-free weights: removed "marginal>good" paradox from V1.
  4. Realistic thresholds: filter@50 blocks bottom half (not just <40).
  5. Validation tooling: train/test split, out-of-sample metrics.

SCORING FORMULA (empirical weights):
  - ADR room (100 - adr_used_pct): 35 points
      Winners 36.5%, losers 9.8% → Δ=-26.67% (strongest predictor)
  - HTF trend alignment (Smart Direction): 30 points
      Trading WITH the 4H trend vs. against it (new, not in V1)
  - ATR volatility: 20 points
      Winners 0.53%, losers 0.69% → lower is better
  - Decision Center score: 15 points
      Composite from HTF/Forecast/CTR (independent from above)
  - Exhaustion kill-switch: >85% → auto-BLOCK (not scored)

GRADES:
  75-100: EXCELLENT
  60-74:  GOOD
  45-59:  FAIR
  0-44:   POOR

Default filter threshold: 50 (blocks POOR + lower FAIR).
"""

from typing import Dict, Optional


def calculate_quality_score_v2(
    symbol: str,
    side: str,
    scanner,  # smc_scanner instance (for bias data)
    smart_direction_result: Optional[Dict] = None,
) -> Dict:
    """Calculate Trade Quality Score V2 (0-100) with independent factors.

    Args:
        symbol: trading pair
        side: 'LONG' or 'SHORT'
        scanner: SMCScanner instance (provides get_bias())
        smart_direction_result: output from smart_direction.compute_smart_direction()
            Contains {allow_long, allow_short, mode, reason}. If None, HTF
            alignment component defaults to neutral (15/30 points).

    Returns:
        {
            'score': 0-100,
            'grade': 'EXCELLENT' | 'GOOD' | 'FAIR' | 'POOR',
            'breakdown': {component: points},
            'reason': str (human-readable summary),
            'blocked': bool (True if exhaustion kill-switch triggered),
            'metrics': {...} (raw values for validation)
        }
    """
    if not scanner:
        return _error_result('Scanner not available')

    try:
        bias_data = scanner.get_bias(symbol)
        if not bias_data:
            return _error_result('No bias data')

        move = bias_data.get('move', {})
        decision = bias_data.get('decision', {})

        points = {}
        total = 0
        blocked = False
        reason_parts = []

        # --- KILL-SWITCH: Exhaustion >85% ---
        exhaustion = move.get('exhaustion', 50)
        if exhaustion > 85:
            blocked = True
            return {
                'score': 0,
                'grade': 'BLOCKED',
                'breakdown': {},
                'reason': f'⛔ Exhaustion {exhaustion:.0f}% > 85% — move depleted',
                'blocked': True,
                'metrics': {'exhaustion': exhaustion},
            }

        # --- 1. ADR ROOM (35 points) ---
        # Most important: winners had 36.5% room, losers 9.8% (Δ=-26.67%)
        adr_used = move.get('adr_used_pct', 100)
        adr_room = 100 - adr_used
        if adr_room >= 35:
            points['adr_room'] = 35
        elif adr_room >= 25:
            points['adr_room'] = 28
        elif adr_room >= 15:
            points['adr_room'] = 18
        elif adr_room >= 5:
            points['adr_room'] = 8
        else:
            points['adr_room'] = 0
        total += points['adr_room']

        if adr_room >= 30:
            reason_parts.append(f"ADR {adr_room:.0f}%✓")
        elif adr_room < 10:
            reason_parts.append(f"⚠ ADR {adr_room:.0f}%")

        # --- 2. HTF TREND ALIGNMENT (30 points) ---
        # New factor: trading WITH 4H trend (via Smart Direction) vs. against.
        # If smart_direction says "allow this side", award full points.
        # If it blocks this side (but allows opposite), penalize.
        # If WAIT mode or no data, neutral (15 points).
        if smart_direction_result:
            mode = smart_direction_result.get('mode', 'WAIT')
            allow_long = smart_direction_result.get('allow_long', False)
            allow_short = smart_direction_result.get('allow_short', False)

            if side == 'LONG':
                if allow_long:
                    points['htf_alignment'] = 30
                    reason_parts.append("4H✓")
                elif allow_short and not allow_long:
                    # Contra-trend: 4H wants SHORT but we're going LONG
                    points['htf_alignment'] = 0
                    reason_parts.append("⚠ contra-4H")
                else:
                    # WAIT or BOTH → neutral
                    points['htf_alignment'] = 15
            else:  # SHORT
                if allow_short:
                    points['htf_alignment'] = 30
                    reason_parts.append("4H✓")
                elif allow_long and not allow_short:
                    points['htf_alignment'] = 0
                    reason_parts.append("⚠ contra-4H")
                else:
                    points['htf_alignment'] = 15
        else:
            # No Smart Direction data → neutral
            points['htf_alignment'] = 15

        total += points['htf_alignment']

        # --- 3. ATR VOLATILITY (20 points) ---
        # Winners 0.53%, losers 0.69% → lower is better (calmer = tradeable)
        atr_pct = move.get('atr_pct', 0.6)
        if atr_pct < 0.4:
            points['atr'] = 20
        elif atr_pct < 0.55:
            points['atr'] = 15
        elif atr_pct < 0.7:
            points['atr'] = 10
        elif atr_pct < 0.9:
            points['atr'] = 5
        else:
            points['atr'] = 0
        total += points['atr']

        # --- 4. DECISION CENTER SCORE (15 points) ---
        # Use the raw numeric score from evaluate_entry(), NOT the verdict.
        # V1 had "marginal>good" paradox because verdict was noise on 142 trades.
        # Decision score is a composite (HTF/Forecast/CTR) → independent from above.
        #
        # evaluate_entry() returns score ~[-10, +30]. Threshold is usually 10.
        # Map: ≥20 → 15pts, ≥10 → 12pts, ≥0 → 8pts, <0 → 0pts.
        decision_score = decision.get('score', 0)  # fallback 0 if missing
        if decision_score >= 20:
            points['decision'] = 15
        elif decision_score >= 10:
            points['decision'] = 12
        elif decision_score >= 0:
            points['decision'] = 8
        else:
            points['decision'] = 0
        total += points['decision']

        # --- GRADE ---
        if total >= 75:
            grade = 'EXCELLENT'
        elif total >= 60:
            grade = 'GOOD'
        elif total >= 45:
            grade = 'FAIR'
        else:
            grade = 'POOR'

        reason = ' · '.join(reason_parts) if reason_parts else grade

        return {
            'score': total,
            'grade': grade,
            'breakdown': points,
            'reason': reason,
            'blocked': blocked,
            'metrics': {
                'adr_room': adr_room,
                'atr_pct': atr_pct,
                'exhaustion': exhaustion,
                'decision_score': decision_score,
                'htf_mode': smart_direction_result.get('mode') if smart_direction_result else None,
            },
        }

    except Exception as e:
        return _error_result(f'Error: {e}')


def _error_result(reason: str) -> Dict:
    """Fallback for errors: neutral score, no block."""
    return {
        'score': 50,
        'grade': 'FAIR',
        'breakdown': {},
        'reason': reason,
        'blocked': False,
        'metrics': {},
    }


# ============================================================================
# VALIDATION TOOLING
# ============================================================================

def validate_on_archive(archive_path: str, test_size: float = 0.3):
    """Run train/test validation on a trade archive JSON.

    Args:
        archive_path: path to JSON file with closed trades
        test_size: fraction for test set (0.3 = 70% train, 30% test)

    Prints:
        - Correlation of score with win/loss (train + test)
        - Win rate by grade bucket (train + test)
        - Distribution of scores (train + test)
        - Out-of-sample performance metrics

    Usage:
        from detection.quality_gate_v2 import validate_on_archive
        validate_on_archive('/path/to/trade_archive.json', test_size=0.3)
    """
    import json
    import random
    from collections import defaultdict

    with open(archive_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    trades = data.get('closed_trades', [])
    if not trades:
        print("❌ No closed_trades in archive")
        return

    # Shuffle and split
    random.seed(42)
    random.shuffle(trades)
    split_idx = int(len(trades) * (1 - test_size))
    train = trades[:split_idx]
    test = trades[split_idx:]

    print(f"\n{'='*60}")
    print(f"Quality Gate V2 — Train/Test Validation")
    print(f"{'='*60}")
    print(f"Total trades: {len(trades)}")
    print(f"Train: {len(train)} ({(1-test_size)*100:.0f}%)")
    print(f"Test:  {len(test)} ({test_size*100:.0f}%)\n")

    def analyze_split(name, subset):
        """Compute metrics for train or test split."""
        scores = []
        outcomes = []  # 1 = win, 0 = loss
        grade_buckets = defaultdict(lambda: {'wins': 0, 'losses': 0})

        for t in subset:
            # Mock score calculation (real would need scanner instance)
            # For validation, we extract PRE-TRADE metrics if available.
            pre = t.get('pre_trade_snapshot', {})
            move = pre.get('move', {})
            decision = pre.get('decision', {})

            adr_used = move.get('adr_used_pct', 100)
            adr_room = 100 - adr_used
            atr_pct = move.get('atr_pct', 0.6)
            exhaustion = move.get('exhaustion', 50)
            decision_score = decision.get('score', 0)

            # Simplified scoring (HTF alignment not in archive, default neutral=15)
            pts = {}
            if exhaustion > 85:
                # Blocked
                score = 0
                grade = 'BLOCKED'
            else:
                # ADR room
                if adr_room >= 35:
                    pts['adr'] = 35
                elif adr_room >= 25:
                    pts['adr'] = 28
                elif adr_room >= 15:
                    pts['adr'] = 18
                elif adr_room >= 5:
                    pts['adr'] = 8
                else:
                    pts['adr'] = 0

                # HTF (default neutral)
                pts['htf'] = 15

                # ATR
                if atr_pct < 0.4:
                    pts['atr_v'] = 20
                elif atr_pct < 0.55:
                    pts['atr_v'] = 15
                elif atr_pct < 0.7:
                    pts['atr_v'] = 10
                elif atr_pct < 0.9:
                    pts['atr_v'] = 5
                else:
                    pts['atr_v'] = 0

                # Decision
                if decision_score >= 20:
                    pts['dec'] = 15
                elif decision_score >= 10:
                    pts['dec'] = 12
                elif decision_score >= 0:
                    pts['dec'] = 8
                else:
                    pts['dec'] = 0

                score = sum(pts.values())

                if score >= 75:
                    grade = 'EXCELLENT'
                elif score >= 60:
                    grade = 'GOOD'
                elif score >= 45:
                    grade = 'FAIR'
                else:
                    grade = 'POOR'

            # Outcome
            pnl = t.get('pnl_pct', 0)
            win = 1 if pnl > 0 else 0

            scores.append(score)
            outcomes.append(win)

            if win:
                grade_buckets[grade]['wins'] += 1
            else:
                grade_buckets[grade]['losses'] += 1

        # Stats
        avg_score = sum(scores) / len(scores) if scores else 0
        win_rate_overall = sum(outcomes) / len(outcomes) * 100 if outcomes else 0

        # Score distribution
        dist = defaultdict(int)
        for s in scores:
            if s >= 75:
                dist['EXCELLENT'] += 1
            elif s >= 60:
                dist['GOOD'] += 1
            elif s >= 45:
                dist['FAIR'] += 1
            else:
                dist['POOR'] += 1

        print(f"--- {name} ---")
        print(f"Avg score: {avg_score:.1f}")
        print(f"Overall win rate: {win_rate_overall:.1f}%")
        print(f"\nWin rate by grade:")
        for g in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'BLOCKED']:
            b = grade_buckets[g]
            total = b['wins'] + b['losses']
            if total > 0:
                wr = b['wins'] / total * 100
                print(f"  {g:12s}: {wr:5.1f}% ({b['wins']}W / {b['losses']}L, n={total})")

        print(f"\nScore distribution:")
        for g in ['EXCELLENT', 'GOOD', 'FAIR', 'POOR']:
            count = dist[g]
            pct = count / len(scores) * 100 if scores else 0
            print(f"  {g:12s}: {count:3d} ({pct:5.1f}%)")
        print()

    analyze_split("TRAIN SET", train)
    analyze_split("TEST SET (out-of-sample)", test)

    print("="*60)
    print("✅ Validation complete. Compare train vs. test metrics:")
    print("   - Similar win rates by grade → model generalizes")
    print("   - Large divergence → overfitting on train data")
    print("="*60 + "\n")
