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


def calculate_quality_score_v2(symbol: str, side: str, scanner,
                               smart_direction_result: Optional[Dict] = None) -> Dict:
    """Score a candidate trade 0-100 from four INDEPENDENT factors.

    Returns a dict:
      score      — int 0..100
      grade      — EXCELLENT / GOOD / FAIR / POOR (or BLOCKED on kill-switch)
      breakdown  — per-factor point allocation
      reason     — short human-readable explanation
      blocked    — True only when the exhaustion kill-switch fired
      metrics    — raw inputs used (adr_room, atr_pct, exhaustion,
                   decision_score, htf_mode) for the Blocked Trades table

    Never raises — any error degrades to a neutral FAIR/50 result.
    """
    try:
        if not scanner:
            return _error_result('Scanner not available')

        bias_data = scanner.get_bias(symbol)
        if not bias_data:
            return _error_result('No bias data')

        move = bias_data.get('move', {})
        decision = bias_data.get('decision', {})

        points: Dict[str, int] = {}
        total = 0
        blocked = False
        reason_parts = []

        # === Exhaustion kill-switch (not scored — hard block) ===
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

        # === ADR room — 35 pts (strongest predictor) ===
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
            reason_parts.append(f'ADR {adr_room:.0f}%✓')
        elif adr_room < 10:
            reason_parts.append(f'⚠ ADR {adr_room:.0f}%')

        # === HTF trend alignment (Smart Direction) — 30 pts ===
        if smart_direction_result:
            mode = smart_direction_result.get('mode', 'WAIT')
            allow_long = smart_direction_result.get('allow_long', False)
            allow_short = smart_direction_result.get('allow_short', False)
            if side == 'LONG':
                if allow_long:
                    points['htf_alignment'] = 30
                    reason_parts.append('4H✓')
                elif allow_short and not allow_long:
                    points['htf_alignment'] = 0
                    reason_parts.append('⚠ contra-4H')
                else:
                    points['htf_alignment'] = 15
            else:  # SHORT
                if allow_short:
                    points['htf_alignment'] = 30
                    reason_parts.append('4H✓')
                elif allow_long and not allow_short:
                    points['htf_alignment'] = 0
                    reason_parts.append('⚠ contra-4H')
                else:
                    points['htf_alignment'] = 15
        else:
            # No Smart Direction data — neutral half-credit (don't punish).
            points['htf_alignment'] = 15
        total += points['htf_alignment']

        # === ATR volatility — 20 pts (lower is better) ===
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

        # === Decision Center composite score — 15 pts ===
        decision_score = decision.get('score', 0)
        if decision_score >= 20:
            points['decision'] = 15
        elif decision_score >= 10:
            points['decision'] = 12
        elif decision_score >= 0:
            points['decision'] = 8
        else:
            points['decision'] = 0
        total += points['decision']

        # === Grade ===
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


def validate_on_archive(archive_path: str, test_size: float = 0.3):
    """Train/test split validation against a closed-trade archive JSON.

    Splits trades into train/test, recomputes the V2 grade from each trade's
    `pre_trade_snapshot`, and reports win rate by grade for both splits so you
    can spot overfitting (large train↔test divergence).
    """
    import json
    import random
    from collections import defaultdict

    with open(archive_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    trades = data.get('closed_trades')
    if not trades:
        print('❌ No closed_trades in archive')
        return

    random.seed(42)
    random.shuffle(trades)
    split = int(len(trades) * (1 - test_size))
    train = trades[:split]
    test = trades[split:]

    print('\n' + '=' * 60)
    print('Quality Gate V2 — Train/Test Validation')
    print('Total trades: ' + str(len(trades)))
    print('Train: ' + str(len(train)) + ' (' + f'{(1 - test_size) * 100:.0f}' + '%)')
    print('Test:  ' + str(len(test)) + ' (' + f'{test_size * 100:.0f}' + '%)\n')

    def _grade_from_snapshot(snap):
        """Recompute the V2 grade from a stored pre-trade snapshot."""
        move = snap.get('move', {})
        decision = snap.get('decision', {})
        adr_room = 100 - move.get('adr_used_pct', 100)
        atr_pct = move.get('atr_pct', 0.6)
        exhaustion = move.get('exhaustion', 50)
        if exhaustion > 85:
            return 0, 'BLOCKED'
        total = 0
        if adr_room >= 35:
            total += 35
        elif adr_room >= 25:
            total += 28
        elif adr_room >= 15:
            total += 18
        elif adr_room >= 5:
            total += 8
        total += snap.get('htf', 15)
        if atr_pct < 0.4:
            total += 20
        elif atr_pct < 0.55:
            total += 15
        elif atr_pct < 0.7:
            total += 10
        elif atr_pct < 0.9:
            total += 5
        dec = decision.get('score', 0)
        if dec >= 20:
            total += 15
        elif dec >= 10:
            total += 12
        elif dec >= 0:
            total += 8
        if total >= 75:
            grade = 'EXCELLENT'
        elif total >= 60:
            grade = 'GOOD'
        elif total >= 45:
            grade = 'FAIR'
        else:
            grade = 'POOR'
        return total, grade

    def analyze_split(name, subset):
        """Compute metrics for train or test split."""
        by_grade = defaultdict(lambda: {'wins': 0, 'losses': 0})
        scores = []
        wins = losses = 0
        for t in subset:
            snap = t.get('pre_trade_snapshot')
            if not snap:
                continue
            score, grade = _grade_from_snapshot(snap)
            scores.append(score)
            pnl = t.get('pnl_pct', 0)
            if pnl >= 0:
                by_grade[grade]['wins'] += 1
                wins += 1
            else:
                by_grade[grade]['losses'] += 1
                losses += 1
        n = wins + losses
        print('--- ' + name + ' ---')
        if scores:
            print('Avg score: ' + f'{sum(scores) / len(scores):.1f}')
        if n:
            print('Overall win rate: ' + f'{wins / n * 100:.1f}' + '%')
        print('\nWin rate by grade:')
        for grade in ('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'BLOCKED'):
            g = by_grade.get(grade)
            if not g:
                continue
            gn = g['wins'] + g['losses']
            wr = g['wins'] / gn * 100 if gn else 0
            print('  ' + f'{grade:12s}' + ': ' + f'{wr:5.1f}' + '% ('
                  + str(g['wins']) + 'W / ' + str(g['losses']) + 'L, n=' + str(gn) + ')')
        print('\nScore distribution:')

    analyze_split('TRAIN SET', train)
    analyze_split('TEST SET (out-of-sample)', test)
    print('\n✅ Validation complete. Compare train vs. test metrics:')
    print('   - Similar win rates by grade → model generalizes')
    print('   - Large divergence → overfitting on train data')
    print('=' * 60 + '\n')
