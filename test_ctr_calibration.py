"""CTR calibration tests for grade_setup («Готовність»).

Pins the fix that stops CTR from cutting reversal setups: in 'soft' (default)
mode an overbought/oversold CTR no longer zeroes the timing block nor triggers
the «CTR проти входу» veto (which capped the score to 43); 'normal' keeps the
old hard behavior; 'off' makes CTR neutral. Pure module — no heavy deps.
"""

import sys, importlib.util
sys.path.insert(0, '.')

_spec = importlib.util.spec_from_file_location(
    'setup_grader', 'detection/setup_grader.py')
sg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sg)


def _sig(stc):
    """A genuinely good LONG reversal (Discount zone, OB, liquidity, MM up)
    but with CTR overbought (stc≥75) — the exact shape CTR was over-punishing."""
    return {
        'smc': {'structure_signal': 'BULLISH_CHOCH', 'market_bias': 'BULLISH',
                'price_at_bullish_ob': True, 'active_bullish_obs': 2,
                'active_fvgs': [], 'price_zone': 'DISCOUNT', 'zone_level': 0.14},
        'htf_bias': 'BULLISH', 'mp': {'exhaustion': 50},
        'mm_dir': 0.3, 'mm_strength': 45, 'mm_conflict': False,
        'mm_runway': {'room_pct': 1.2},
        'ctr1h': {'stc': stc, 'last_dir': 'down', 'age': 2},
        'liq_levels': [{'price': 1.0, 'side': 'long', 'age_min': 30},
                       {'price': 2.0, 'side': 'short', 'age_min': 999}],
        'mark_price': 1.5, 'btc_dir': 'LONG', 'btc_start': False,
        'funding_rate': -0.5, 'funding_trend': -0.1, 'vol_up': True, 'spike': False,
    }


def test_normal_mode_vetoes_overbought():
    r = sg.grade_setup('LONG', _sig(80), {'ctr_mode': 'normal'})
    assert 'CTR проти входу' in r['vetoes'], r['vetoes']
    assert r['score'] <= 43, r['score']
    print(f"✓ normal: CTR-вето кап ≤43 (score={r['score']})")


def test_soft_mode_no_veto_scores_higher():
    normal = sg.grade_setup('LONG', _sig(80), {'ctr_mode': 'normal'})
    soft = sg.grade_setup('LONG', _sig(80), {'ctr_mode': 'soft'})
    assert 'CTR проти входу' not in soft['vetoes'], soft['vetoes']
    assert soft['score'] > normal['score'], (soft['score'], normal['score'])
    assert soft['score'] >= 53, f"good reversal should now reach ХОРОШИЙ, got {soft['score']}"
    print(f"✓ soft: без вето, {normal['score']} → {soft['score']} (розворот проходить)")


def test_off_mode_neutral():
    off = sg.grade_setup('LONG', _sig(80), {'ctr_mode': 'off'})
    assert 'CTR проти входу' not in off['vetoes'], off['vetoes']
    # timing block neutralised → its detail says CTR off
    tim = next((c for c in off['checks'] if c['key'] == 'timing'), None)
    assert tim and 'вимкнено' in tim['detail'], tim
    print(f"✓ off: CTR нейтральний, не впливає (score={off['score']})")


def test_default_is_soft():
    # No cfg → _DEFAULTS ctr_mode='soft' → no veto at overbought.
    r = sg.grade_setup('LONG', _sig(80))
    assert 'CTR проти входу' not in r['vetoes'], r['vetoes']
    print('✓ дефолт grade_setup = soft (без CTR-вето)')


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
    print(f'\nAll CTR-calibration tests passed ✓ ({len(tests)} tests)')
