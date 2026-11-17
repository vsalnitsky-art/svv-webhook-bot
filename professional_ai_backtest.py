#!/usr/bin/env python3
"""Run the professional AI walk-forward test on a closed-bar 15m CSV.

CSV columns (case-insensitive aliases are handled by the normalizer):
  t,o,h,l,c,v,q  OR  timestamp,open,high,low,close,volume,turnover
Timestamps may be UTC seconds or milliseconds. Rows must be CLOSED candles.

Example:
  python professional_ai_backtest.py BTCUSDT_15m.csv --output btc_oos.json \
      --model-output btc_model.json
"""
import argparse
import json
import os
import sys

import importlib.util

# Модуль напряму з файлу: `from detection import …` виконав би
# detection/__init__.py (pybit, сканери) — CLI бектесту це не потрібно.
_P_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detection', 'professional_ai.py')
_spec = importlib.util.spec_from_file_location('detection.professional_ai', _P_PATH)
_P = importlib.util.module_from_spec(_spec)
sys.modules['detection.professional_ai'] = _P
_spec.loader.exec_module(_P)
BrainConfig, OnlineLogisticModel = _P.BrainConfig, _P.OnlineLogisticModel
load_csv_ohlcv, walk_forward_backtest = _P.load_csv_ohlcv, _P.walk_forward_backtest


def main(argv=None):
    parser = argparse.ArgumentParser(description='Professional AI purged walk-forward backtest')
    parser.add_argument('csv', help='CSV with CLOSED 15m OHLCV bars')
    parser.add_argument('--output', help='Write full OOS report JSON here')
    parser.add_argument('--model-output', help='Write the last walk-forward online model JSON here')
    parser.add_argument('--require-model', action='store_true',
                        help='Require the online model to be OOS-ready before entries')
    parser.add_argument('--fee-pct', type=float, default=0.055, help='Fee per fill, percent')
    parser.add_argument('--slippage-pct', type=float, default=0.02, help='Slippage per fill, percent')
    parser.add_argument('--initial-train', type=int, default=1200)
    parser.add_argument('--test-bars', type=int, default=200)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--embargo', type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.fee_pct, args.slippage_pct) < 0:
        parser.error('Fees and slippage cannot be negative')
    if min(args.initial_train, args.test_bars, args.horizon) < 1 or args.embargo < 0:
        parser.error('Invalid walk-forward window')
    cfg = BrainConfig(fee_pct=args.fee_pct, slippage_pct=args.slippage_pct,
                      initial_train_bars=args.initial_train,
                      walkforward_test_bars=args.test_bars,
                      horizon_bars=args.horizon, embargo_bars=args.embargo)
    try:
        rows = load_csv_ohlcv(args.csv)
        report = walk_forward_backtest(rows, cfg, require_model=args.require_model)
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as handle:
                handle.write(payload + '\n')
        if args.model_output:
            # Persist exactly the model that accompanied the final OOS split.
            model_state = report['model']
            # Reconstructing validates the public persistence schema. Runtime-only
            # accumulators are optional and are not needed for future predictions.
            model = OnlineLogisticModel.from_state(model_state)
            with open(args.model_output, 'w', encoding='utf-8') as handle:
                handle.write(model.to_json() + '\n')
        print(json.dumps({'ok': True, 'bars': report['bars'], 'splits': len(report['splits']),
                          'metrics': report['metrics'], 'output': args.output,
                          'model_output': args.model_output,
                          'limitations': report['limitations']}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
