"""
smart_direction — intelligent trade-direction algorithm for the bias panel.

Determines which direction (LONG / SHORT / both / neither) is MOST EFFECTIVE
to trade right now, based on:
  1. LTF verdict (LONG/SHORT/WAIT) — the bot's entry signal
  2. 4H trend + exhaustion — is the higher-TF trend aligned and healthy?
  3. 1H reversal pressure (optional) — early reversal confirmation

Strategy (user-selected "allow early reversal"):
  - Align verdict with the 4H trend when the 4H is fresh/healthy.
  - When 4H is EXHAUSTED + opposing 1H/verdict show confluence → allow
    the REVERSAL direction (contra-trend play, catching the turn early).
  - Flat 4H (NEUTRAL) → verdict must be strong enough (high confidence).
  - WAIT verdict → both gates off (no new entries).

Output: {'allow_long': bool, 'allow_short': bool, 'mode': str, 'reason': str}

The mode is single-directional ('LONG_ONLY' / 'SHORT_ONLY'), both-directional
('BOTH', rare, when 4H flat and verdict confidence is marginal), or paused
('WAIT', no new entries). Reason is a short Ukrainian string for the UI banner.
"""

from typing import Dict, Optional


def compute_smart_direction(
    verdict_data: Dict,
    allow_early_reversal: bool = True,
    require_confidence: int = 60,
) -> Dict:
    """Compute the optimal trade direction from the bias verdict payload.

    verdict_data: output of web.flask_app's bias endpoint (/api/sm/bias) —
        contains {verdict, confidence, reversal (4H), reversal_1h (1H), move}.
    allow_early_reversal: if True, open the reversal direction when 4H is
        exhausted + confluence from 1H. If False, only trade WITH the 4H
        trend (more conservative, waits for the new trend to fully establish).
    require_confidence: verdict confidence threshold for WAIT → SINGLE-SIDE.
        Below this, the gates close (WAIT mode) even if 4H is OK.

    Returns:
        {
            'allow_long': bool,
            'allow_short': bool,
            'mode': 'LONG_ONLY' | 'SHORT_ONLY' | 'BOTH' | 'WAIT',
            'reason': str (Ukrainian explanation for UI),
            'ok': bool,
        }
    """
    out = {'allow_long': False, 'allow_short': False, 'mode': 'WAIT',
           'reason': 'немає даних', 'ok': False}
    if not verdict_data or not verdict_data.get('ok'):
        return out

    verdict = verdict_data.get('verdict', 'WAIT')
    confidence = verdict_data.get('confidence', 0)
    rev_4h = verdict_data.get('reversal') or {}
    rev_1h = verdict_data.get('reversal_1h') or {}
    move = verdict_data.get('move') or {}

    # Guard: WAIT verdict → no entries, regardless of 4H/1H state.
    if verdict == 'WAIT':
        out['ok'] = True
        out['reason'] = 'вердикт WAIT — жоден напрямок не підтверджений'
        return out

    # Guard: low confidence → treat as WAIT even if verdict is directional.
    if confidence < require_confidence:
        out['ok'] = True
        out['reason'] = (f'вердикт {verdict} впевненість низька '
                         f'({confidence}%, потрібно ≥{require_confidence}%)')
        return out

    # --- Read 4H trend + exhaustion ---
    trend_4h = rev_4h.get('trend_4h')          # 'LONG' | 'SHORT' | 'NEUTRAL'
    exhaustion_4h = rev_4h.get('index') or 0   # 0..100
    # Reversal_to is the flip direction if the current 4H trend breaks.
    reversal_to_4h = rev_4h.get('reversal_to')

    # --- Scenarios ---

    # Scenario A: 4H is FLAT (no trend) — verdict must carry alone.
    if trend_4h == 'NEUTRAL' or rev_4h.get('level') == 'NONE':
        # Without a higher-TF trend to lean on, we need high conviction from
        # the verdict. Already guarded by confidence threshold above, but if
        # the verdict is directional and strong enough, open that side.
        if verdict == 'LONG':
            out['allow_long'], out['allow_short'] = True, False
            out['mode'] = 'LONG_ONLY'
            out['reason'] = (f'4H консолідація — йдемо за {verdict}, '
                             f'{confidence}% впевненість')
        elif verdict == 'SHORT':
            out['allow_long'], out['allow_short'] = False, True
            out['mode'] = 'SHORT_ONLY'
            out['reason'] = (f'4H консолідація — йдемо за {verdict}, '
                             f'{confidence}% впевненість')
        else:
            out['mode'] = 'WAIT'
            out['reason'] = '4H флет, вердикт нечіткий — чекаємо тренду'
        out['ok'] = True
        return out

    # Scenario B: 4H trend AGREES with verdict — straightforward.
    if verdict == trend_4h:
        # Check exhaustion: if the 4H trend is critically exhausted (≥60),
        # we're still in the trend but it's late-stage. Open the direction
        # but flag the risk in the reason.
        if exhaustion_4h >= 60:
            side = 'LONG' if verdict == 'LONG' else 'SHORT'
            out['allow_long'] = (verdict == 'LONG')
            out['allow_short'] = (verdict == 'SHORT')
            out['mode'] = f'{side}_ONLY'
            out['reason'] = (f'{verdict} узгоджено з 4H, але тренд виснажений '
                             f'({exhaustion_4h:.0f}%) — обережно')
        else:
            side = 'LONG' if verdict == 'LONG' else 'SHORT'
            out['allow_long'] = (verdict == 'LONG')
            out['allow_short'] = (verdict == 'SHORT')
            out['mode'] = f'{side}_ONLY'
            exh_word = 'свіжий' if exhaustion_4h < 35 else 'дозрілий'
            out['reason'] = (f'{verdict} за 4H трендом ({exh_word}, '
                             f'{exhaustion_4h:.0f}% виснаження) — запас є')
        out['ok'] = True
        return out

    # Scenario C: verdict OPPOSES the 4H trend.
    # Sub-case C1: 4H exhausted (≥60) + early-reversal allowed → open the
    #              flip direction (verdict's side). This is the aggressive
    #              "catch the reversal" play.
    # Sub-case C2: 4H not exhausted or early-reversal disabled → WAIT (the
    #              old trend is still too strong to fight).

    if exhaustion_4h >= 60 and allow_early_reversal:
        # Check 1H confirmation if available: if 1H also shows reversal
        # pressure in the same direction (reversal_to_1h == verdict), that's
        # stronger confluence. Not mandatory, but we can mention it.
        reversal_to_1h = rev_1h.get('reversal_to') if rev_1h.get('ok') else None
        conf_1h = ''
        if reversal_to_1h == verdict:
            idx_1h = rev_1h.get('index', 0)
            if idx_1h >= 35:
                conf_1h = f' + 1H підтверджує ({idx_1h:.0f}%)'

        out['allow_long'] = (verdict == 'LONG')
        out['allow_short'] = (verdict == 'SHORT')
        out['mode'] = f'{verdict}_ONLY'
        out['reason'] = (f'4H тренд {trend_4h} виснажений ({exhaustion_4h:.0f}%) — '
                         f'дозволяємо ранній розворот {verdict}{conf_1h}')
        out['ok'] = True
        return out

    # No early-reversal allowed OR 4H not exhausted enough → contra-trend is
    # too risky, close gates.
    out['mode'] = 'WAIT'
    out['reason'] = (f'вердикт {verdict} ПРОТИ 4H тренду ({trend_4h}), а той '
                     f'ще міцний ({exhaustion_4h:.0f}%) — чекаємо зламу')
    out['ok'] = True
    return out
