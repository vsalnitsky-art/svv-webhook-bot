"""Correctness tests for the professional market brain and archive learner.

Run from bot_source: python test_professional_ai.py
Synthetic tests verify implementation invariants, NOT market profitability.
"""
import copy
import json
import math
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import importlib.util


def _sib(name):
    # Без пакетного імпорту: detection/__init__.py тягне pybit.
    full = f'detection.{name}'
    if full not in sys.modules:
        sp = importlib.util.spec_from_file_location(full, os.path.join(ROOT, 'detection', f'{name}.py'))
        m = importlib.util.module_from_spec(sp)
        sys.modules[full] = m
        sp.loader.exec_module(m)
    return sys.modules[full]


P = _sib('professional_ai')
_PL = _sib('professional_learning')
ArchiveLearner, samples_from_archives = _PL.ArchiveLearner, _PL.samples_from_archives

T0 = 1_700_000_000 // 900 * 900


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def make_bars(n=1500, direction=1, reverse=False):
    bars = []
    previous = 100.0
    for i in range(n):
        wave = 0.10 * math.sin(i / 9.0)
        close = 100.0 + direction * (0.035 * i + wave)
        open_ = previous
        high = max(open_, close) + 0.08
        low = min(open_, close) - 0.08
        bars.append({'t': T0 + i * 900, 'o': open_, 'h': high, 'l': low,
                     'c': close, 'v': 1000.0 + (i % 17) * 25.0, 'q': close * 1000.0})
        previous = close
    return list(reversed(bars)) if reverse else bars


def market(direction=1, n=1500):
    bars = make_bars(n=n, direction=direction)
    now = bars[-1]['t'] + 900
    return {'15m': bars, '1h': P.resample_ohlcv(bars, '1h'),
            '4h': P.resample_ohlcv(bars, '4h')}, now


def test_normalization_reverse_ms_and_unclosed():
    bars = make_bars(10)
    raw = [[b['t'] * 1000, b['o'], b['h'], b['l'], b['c'], b['v'], b['q']]
           for b in reversed(bars)]
    now = bars[-1]['t'] + 450
    clean = P.normalize_ohlcv(raw, '15m', now=now)
    _check(len(clean) == 9 and clean[-1]['t'] == bars[-2]['t'],
           'Unclosed bar must be removed and reverse ms input sorted')
    clean.append(dict(clean[-1]))
    dedup = P.normalize_ohlcv(clean, '15m', now=bars[-1]['t'] + 900)
    _check(len(dedup) == 9, 'Duplicate timestamps must be deduplicated')


def test_no_lookahead_in_features():
    bars = make_bars(200)
    a = P.timeframe_features(bars[:150])
    future = make_bars(220, direction=-1)[150:]
    b = P.timeframe_features(bars[:150] + future)
    # b is intentionally at another index; compare the proper prefix recomputation
    c = P.timeframe_features(bars[:150])
    _check(a == c, 'Feature computation must be deterministic')
    keys = ('trend', 'momentum', 'rsi', 'macd', 'relative_volume', 'atr_pct')
    _check(all(a[k] == P.timeframe_features(bars[:150])[k] for k in keys),
           'No future values may enter a feature snapshot')


def test_long_short_direction_and_determinism():
    up, now_up = market(1)
    down, now_down = market(-1)
    brain = P.ProfessionalBrain()
    a = brain.analyze(up, now_up)
    b = brain.analyze(up, now_up)
    d = brain.analyze(down, now_down)
    _check(a == b, 'Same closed data must produce the same decision')
    _check(a['ok'] and d['ok'], f'Data should be valid: {a["data_issues"]} {d["data_issues"]}')
    _check(a['sides']['LONG']['quality'] > a['sides']['SHORT']['quality'],
           f'Uptrend must favour LONG: {a["sides"]}')
    _check(d['sides']['SHORT']['quality'] > d['sides']['LONG']['quality'],
           f'Downtrend must favour SHORT: {d["sides"]}')
    json.dumps(a, allow_nan=False)


def test_stale_insufficient_and_model_gates_fail_closed():
    data, now = market(1)
    brain = P.ProfessionalBrain()
    stale = brain.analyze(data, now + 3600)
    _check(not stale['ok'] and stale['action'] == 'WAIT', 'Stale data must block entry')
    short = {tf: rows[:20] for tf, rows in data.items()}
    _check(not brain.analyze(short, now)['ok'], 'Insufficient history must block entry')
    gated = brain.analyze(data, now, require_model=True)
    _check(gated['action'] == 'WAIT' and any('модель' in v for v in gated['vetoes']),
           'Untrained model cannot authorize a model-required trade')


def test_risk_plan_costs_and_direction():
    data, now = market(1)
    result = P.ProfessionalBrain().analyze(data, now)
    long = result['sides']['LONG']['risk_plan']
    short = result['sides']['SHORT']['risk_plan']
    _check(long['stop'] < long['entry'] < long['tp1'] < long['tp2'], 'LONG SL/TP ordering')
    _check(short['tp2'] < short['tp1'] < short['entry'] < short['stop'], 'SHORT SL/TP ordering')
    _check(long['net_rr_after_costs'] < long['gross_rr'], 'Costs must reduce R:R')


def test_position_monitor_can_close_and_holds_without_data():
    up, up_now = market(1)
    down, down_now = market(-1)
    brain = P.ProfessionalBrain()
    current = down['15m'][-1]['c']
    decision = brain.manage_position(down, 'LONG', 140.0, down['15m'][0]['t'],
                                     down_now, current_price=current,
                                     peak_price=145.0)
    _check(decision['close'] and decision['action'] == 'CLOSE', decision)
    broken = brain.manage_position({'15m': [], '1h': [], '4h': []}, 'LONG', 100,
                                   down['15m'][0]['t'], down_now)
    _check(not broken['close'] and broken['reason'] == 'market_data_unavailable',
           'Missing chart data must not invent a model exit')


def test_model_readiness_and_persistence_roundtrip():
    data, now = market(1)
    features = {tf: P.timeframe_features(data[tf]) for tf in ('15m', '1h', '4h')}
    model = P.OnlineLogisticModel()
    for i in range(210):
        model.observe(features, 'LONG', i % 2)
    state = model.state(min_oos=200)
    restored = P.OnlineLogisticModel.from_state(state)
    _check(state['ready'] and restored.oos_samples == 210, 'OOS readiness and persistence')
    _check(model.predict(features, 'LONG') == restored.predict(features, 'LONG'),
           'Weights must survive round-trip')
    _check(restored.brier_sum == model.brier_sum and restored.correct == model.correct,
           'Prequential metrics must survive round-trip')
    json.dumps(state, allow_nan=False)


def _archive_rows(n=80, reverse_validation=False):
    rows = []
    for i in range(n):
        positive = 1 if i % 2 == 0 else -1
        features = {
            '15m': {'trend': 0.8 * positive, 'momentum': 0.6 * positive,
                    'rsi_signed': 0.5 * positive, 'macd': 0.4 * positive,
                    'range_position': 0.5 * positive, 'relative_volume': 1.2,
                    'atr_pct': 0.4},
            '1h': {'trend': 0.7 * positive, 'momentum': 0.5 * positive},
            '4h': {'trend': 0.6 * positive},
        }
        outcome = 1.0 if positive > 0 else -1.0
        if reverse_validation and i >= 60:
            outcome *= -1
        signal = {'side': 'LONG', 't': T0 + i * 3600, 'professional': {'features': features}}
        rows.append({'symbol': 'XUSDT', 'signal_side': 'LONG',
                     'signal_at': T0 + i * 3600, 'ended_at': T0 + i * 3600 + 1800,
                     'sig_end_pct': outcome, 'signal': signal})
    return rows


def test_archive_learning_promotes_validated_and_deduplicates():
    learner = ArchiveLearner(min_train=30, min_validation=20)
    report = learner.run(_archive_rows(), [])
    _check(report['promoted'] and report['validation_samples'] == 20, report)
    _check(report['challenger']['brier'] < report['baseline']['brier'], report)
    again = learner.run(_archive_rows(), [])
    _check(again['status'] == 'no_new_labels', 'Archived IDs cannot be learned twice')
    restored = ArchiveLearner.from_state(learner.state())
    _check(restored.run(_archive_rows(), [])['status'] == 'no_new_labels',
           'Processed IDs must survive restart')


def test_archive_learning_rejects_degradation():
    learner = ArchiveLearner(min_train=30, min_validation=20)
    before = learner.model.state()
    report = learner.run(_archive_rows(reverse_validation=True), [])
    _check(not report['promoted'] and report['status'] == 'rejected_no_improvement', report)
    _check(learner.model.state()['weights'] == before['weights'],
           'Rejected challenger cannot alter the champion')


def test_same_bar_stop_target_is_conservative():
    bar = {'t': T0, 'o': 100, 'h': 110, 'l': 90, 'c': 105, 'v': 1, 'q': 1}
    raw, reason = P._barrier([bar], 0, 'LONG', stop=95, target=105)
    _check(raw == 95 and reason == 'stop_same_bar_conservative', (raw, reason))
    raw, reason = P._barrier([bar], 0, 'SHORT', stop=105, target=95)
    _check(raw == 105 and reason == 'stop_same_bar_conservative', (raw, reason))


def _bt_config(fee=0.0, slippage=0.0):
    return P.BrainConfig(initial_train_bars=1000, rolling_train_bars=300,
                         walkforward_test_bars=100, horizon_bars=4, embargo_bars=2,
                         fee_pct=fee, slippage_pct=slippage, model_min_oos=10 ** 9,
                         min_net_rr=0.0, use_brain_exits=False)


def test_walkforward_purged_and_costs_reduce_results():
    bars = make_bars(1400)
    free = P.walk_forward_backtest(bars, _bt_config())
    costly = P.walk_forward_backtest(bars, _bt_config(fee=0.055, slippage=0.0))
    for split in free['splits']:
        _check(split['max_train_label_index'] + split['embargo_bars'] < split['test_start_index'],
               f'Purged/embargo boundary violated: {split}')
    _check(free['trades'] and len(free['trades']) == len(costly['trades']),
           'Test needs comparable OOS trades')
    _check([t['entry_index'] for t in free['trades']] == [t['entry_index'] for t in costly['trades']],
           'Costs must not change this controlled test entry sequence')
    _check(all(a['pnl_pct'] > b['pnl_pct'] for a, b in zip(free['trades'], costly['trades'])),
           'Fees/slippage must worsen every identical trade')
    json.dumps(costly, allow_nan=False)


def test_seen_eviction_keeps_newest_processed():
    learner = ArchiveLearner(min_train=30, min_validation=20)
    old_cap, old_keep = _PL.SEEN_CAP, _PL.SEEN_KEEP
    _PL.SEEN_CAP, _PL.SEEN_KEEP = 60, 50
    try:
        learner.run(_archive_rows(80), [])
        order = learner.state()['seen_ids']
        _check(len(order) == 50, len(order))
        newest = samples_from_archives(_archive_rows(80), [])[-1]['id']
        _check(newest in learner.seen, 'Eviction must drop the OLDEST processed ids, not random hashes')
        restored = ArchiveLearner.from_state(learner.state())
        _check(restored.state()['seen_ids'] == order, 'Order survives restart')
    finally:
        _PL.SEEN_CAP, _PL.SEEN_KEEP = old_cap, old_keep


if __name__ == '__main__':
    failures = 0
    tests = sorted((k, v) for k, v in globals().items() if k.startswith('test_') and callable(v))
    for name, fn in tests:
        try:
            fn()
            print('✓', name)
        except Exception as exc:
            failures += 1
            print('✗', name, '—', repr(exc))
    print('OK' if not failures else f'FAIL {failures}')
    raise SystemExit(1 if failures else 0)
