"""🐞 VOB ПІСЛЯ «Combine Zones» — НАЙНОВІШИЙ БЛОК ПЕРШИМ (кейс BTCUSDT 30.09).

На TradingView (Volumized OB, 15m, Swing 5, Low, Combine) останнім був
бичачий блок 30.09 (82 500–83 100) → LONG, а бот показував SHORT і червоний
блок від 23.09. Корінь: `_combine_obs_func` ставив КОЖЕН злитий блок на
початок списку, тож старі злиті зони опинялись ПОПЕРЕДУ свіжого блоку, а
обрізання до Zone Count (`[:3]`) викидало саме НАЙНОВІШИЙ. Тепер після злиття
списки сортуються за `formation_time` (найновіший першим).
"""
import os, random, importlib.util

_ROOT = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('vob', os.path.join(_ROOT, 'detection/volumized_ob.py'))
vob = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(vob)


def _check(c, m):
    if not c:
        raise AssertionError(m)


def _walk(seed, n=3000):
    rnd = random.Random(seed)
    p = 100.0; out = []
    for i in range(n):
        o = p
        p = max(1.0, p * (1 + rnd.gauss(0, 0.004)))
        h = max(o, p) * (1 + abs(rnd.gauss(0, 0.002)))
        l = min(o, p) * (1 - abs(rnd.gauss(0, 0.002)))
        out.append({'t': 1_700_000_000_000 + i * 900_000, 'o': o, 'h': h, 'l': l,
                    'p': p, 'c': p, 'v': rnd.uniform(50, 500)})
    return out


def _ft(ob):
    return ob.get('formation_time', ob['start_time'])


def test_lists_are_newest_first_after_combine():
    for seed in range(12):
        for sw in (3, 5, 10):
            r = vob.detect_volumized_obs(_walk(seed), swing_length=sw, combine_obs=True, zone_count='Low')
            for key in ('all_bullish', 'all_bearish'):
                fts = [_ft(o) for o in r[key]]
                _check(fts == sorted(fts, reverse=True), f'{key} не newest-first (seed {seed}, sw {sw})')
    print('✓ після злиття обидва списки — найновіший першим')


def test_newest_block_is_never_trimmed_away():
    for seed in range(12):
        r = vob.detect_volumized_obs(_walk(seed), swing_length=5, combine_obs=True, zone_count='Low')
        for full, vis in (('all_bullish', 'bullish_obs'), ('all_bearish', 'bearish_obs')):
            if r[full]:
                _check(r[vis] and _ft(r[vis][0]) == max(_ft(o) for o in r[full]),
                       f'найновіший блок {full} випав із видимих (seed {seed})')
        allv = r['bullish_obs'] + r['bearish_obs']
        if allv:
            _check(_ft(r['latest_ob']) == max(_ft(o) for o in r['all_bullish'] + r['all_bearish']),
                   f'latest_ob не найновіший блок (seed {seed})')
    print('✓ найновіший блок завжди серед видимих і саме він — «останній VOB»')


def test_merge_puts_old_block_first_so_detect_must_resort():
    """Документує саму пастку: злиття ставить злитий (старий) блок на початок."""
    def ob(ft, lo, hi):
        return {'type': 'Bull', 'top': hi, 'bottom': lo, 'start_time': ft - 5,
                'formation_time': ft, 'ob_volume': 1, 'breaker': False, 'break_time': None}
    obs = [ob(300, 50, 51), ob(200, 10, 12), ob(100, 11, 13)]   # newest-first; 200+100 перекриваються
    merged = vob._combine_obs_func(obs)
    _check(merged[0]['formation_time'] == 200 and merged[1]['formation_time'] == 300,
           'очікували пастку: злитий старий блок попереду свіжого')
    print('✓ злиття справді псує порядок — сортування в detect обовʼязкове')


if __name__ == '__main__':
    test_lists_are_newest_first_after_combine()
    test_newest_block_is_never_trimmed_away()
    test_merge_puts_old_block_first_so_detect_must_resort()
    print('\nУсі тести «порядок VOB після злиття» пройдено ✅')
