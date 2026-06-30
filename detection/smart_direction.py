"""
smart_direction — trade-direction gate from the bias verdict.

Simple, transparent rule (exactly what the banner shows → which buttons are on):
    • verdict LONG  → LONG button ON,  SHORT off   (mode LONG_ONLY)
    • verdict SHORT → SHORT button ON, LONG off    (mode SHORT_ONLY)
    • verdict WAIT  → both OFF                       (mode WAIT)

The bias verdict (compute_bias) already folds confidence, forecast agreement,
fuel direction, book imbalance, manipulation and sentiment into a single
LONG / SHORT / WAIT call — so the gate just MIRRORS it. There is no separate
higher-TF (4H) requirement here: the panel no longer emits 4H reversal data,
and the user's intent is a 1:1 mirror of the main banner.

Output: {'allow_long', 'allow_short', 'mode', 'reason', 'ok'}.
`reason` is a short Ukrainian string for the UI banner.

Signature is kept compatible with auto_gate's call (allow_early_reversal /
require_confidence are accepted but not needed for the mirror).
"""

from typing import Dict


def compute_smart_direction(
    verdict_data: Dict,
    allow_early_reversal: bool = True,
    require_confidence: int = 60,
) -> Dict:
    """Mirror the bias verdict onto the LONG/SHORT direction gates.

    verdict_data: output of compute_bias / the /api/sm/bias endpoint —
        uses {verdict, confidence}.
    allow_early_reversal, require_confidence: accepted for call-site
        compatibility; the mirror does not need them (the verdict already
        encodes confidence — it is WAIT whenever conviction is too low).

    Returns:
        {'allow_long': bool, 'allow_short': bool,
         'mode': 'LONG_ONLY' | 'SHORT_ONLY' | 'WAIT',
         'reason': str, 'ok': bool}
    """
    out = {'allow_long': False, 'allow_short': False, 'mode': 'WAIT',
           'reason': 'немає даних', 'ok': False}
    if not verdict_data or not verdict_data.get('ok'):
        return out

    verdict = verdict_data.get('verdict', 'WAIT')
    confidence = verdict_data.get('confidence', 0)

    if verdict == 'LONG':
        out['allow_long'], out['allow_short'] = True, False
        out['mode'] = 'LONG_ONLY'
        out['reason'] = f'банер LONG ({confidence}%) → дозволяємо LONG'
    elif verdict == 'SHORT':
        out['allow_long'], out['allow_short'] = False, True
        out['mode'] = 'SHORT_ONLY'
        out['reason'] = f'банер SHORT ({confidence}%) → дозволяємо SHORT'
    else:
        out['allow_long'], out['allow_short'] = False, False
        out['mode'] = 'WAIT'
        out['reason'] = 'банер WAIT → обидва напрямки вимкнено'

    out['ok'] = True
    return out
