"""Professional, explainable market brain for entries and position exits.

Design rules
------------
* Closed OHLCV bars only; exchange arrays and this project's dict bars are OK.
* No order execution and no exchange dependency in this module.
* 15m execution, 1h confirmation and 4h regime, with explicit data vetoes.
* A deterministic market/risk layer works without pretending to be an ML model.
* The online logistic model is prequential: prediction first, delayed label later.
* Walk-forward tests purge labels and embargo samples, enter on the NEXT bar,
  charge fees/slippage and resolve an ambiguous same-bar SL/TP as SL first.

This is decision support software, not a promise of profit. Live deployment still
requires symbol-specific out-of-sample tests, paper forward testing and the existing
Trade Manager limits. Funding, liquidation, latency and real order-book fills are not
simulated by the lightweight backtester.
"""
from __future__ import annotations

import bisect
import copy
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TF_SEC = {'15m': 900, '1h': 3600, '4h': 14400}
SIDES = ('LONG', 'SHORT')
MODEL_VERSION = 'professional-online-logistic-v1'
MODEL_FEATURES = (
    'trend_15m', 'trend_1h', 'trend_4h', 'momentum_15m', 'momentum_1h',
    'rsi_15m', 'macd_15m', 'range_position_15m', 'relative_volume_15m',
    'atr_pct_15m', 'alignment',
)


@dataclass(frozen=True)
class BrainConfig:
    min_bars_15m: int = 120
    min_bars_1h: int = 100
    min_bars_4h: int = 60
    max_gap_mult: int = 4
    entry_quality: float = 62.0
    reversal_quality: float = 62.0
    stop_atr: float = 1.5
    max_stop_atr: float = 3.0
    target_r: float = 2.2
    min_net_rr: float = 1.55
    min_atr_pct: float = 0.02
    max_atr_pct: float = 3.0
    min_relative_volume: float = 0.25
    fee_pct: float = 0.055             # per fill, percent of notional
    slippage_pct: float = 0.02         # per fill, percent of notional
    model_lr: float = 0.035
    model_l2: float = 0.0001
    model_min_oos: int = 200
    model_min_probability: float = 0.56
    horizon_bars: int = 8
    embargo_bars: int = 4
    initial_train_bars: int = 1200
    rolling_train_bars: int = 1000
    walkforward_test_bars: int = 200
    use_brain_exits: bool = True


DEFAULT_CONFIG = BrainConfig()


def _finite(v, default=None):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _clamp(v, lo=-1.0, hi=1.0):
    x = _finite(v, lo)
    return max(lo, min(hi, x))


def _sign(side):
    return 1.0 if str(side).upper() == 'LONG' else -1.0


def _opposite(side):
    return 'SHORT' if str(side).upper() == 'LONG' else 'LONG'


def _round(v, digits=6):
    x = _finite(v)
    return None if x is None else round(x, digits)


def _timestamp(v):
    x = _finite(v)
    if x is None:
        return None
    # Bybit normally uses ms. Also tolerate us/ns without confusing seconds with ms.
    if x > 1e17:
        x /= 1e9
    elif x > 1e14:
        x /= 1e6
    elif x > 1e11:
        x /= 1e3
    return int(x)


def _field(row, names, index=None, default=None):
    if isinstance(row, dict):
        for name in names:
            if name in row and row[name] is not None:
                return row[name]
        return default
    if isinstance(row, (list, tuple)) and index is not None and len(row) > index:
        return row[index]
    return default


def normalize_ohlcv(rows: Iterable, interval: str = '15m', now: Optional[float] = None,
                    drop_unclosed: bool = True) -> List[Dict]:
    """Return unique, chronological, finite bars with timestamps in UTC seconds.

    Accepted input: Bybit arrays [ms,o,h,l,c,v,turnover] or dictionaries using
    t/o/h/l/c/v/q and common long key names. Malformed bars are skipped, never
    replaced with synthetic prices. The current, still-forming bar is removed.
    """
    if interval not in TF_SEC:
        raise ValueError(f'Unsupported interval: {interval}')
    sec = TF_SEC[interval]
    unique = {}
    for row in rows or []:
        t = _timestamp(_field(row, ('t', 'time', 'timestamp', 'open_time', 'start'), 0))
        o = _finite(_field(row, ('o', 'open'), 1))
        h = _finite(_field(row, ('h', 'high'), 2))
        l = _finite(_field(row, ('l', 'low'), 3))
        c = _finite(_field(row, ('c', 'close', 'p'), 4))
        v = _finite(_field(row, ('v', 'volume'), 5, 0.0), 0.0)
        q = _finite(_field(row, ('q', 'turnover', 'quote_volume'), 6, 0.0), 0.0)
        if None in (t, o, h, l, c) or min(o, h, l, c) <= 0:
            continue
        if h < max(o, c) or l > min(o, c) or h < l or v < 0 or q < 0:
            continue
        if now is not None:
            if t > now + 1:
                continue
            if drop_unclosed and t + sec > now + 1e-6:
                continue
        unique[t] = {'t': int(t), 'o': o, 'h': h, 'l': l, 'c': c,
                     'v': v, 'q': q}
    return [unique[t] for t in sorted(unique)]


def resample_ohlcv(rows: Iterable, interval: str) -> List[Dict]:
    """Aggregate chronological 15m bars; caller filters incomplete higher-TF bars."""
    sec = TF_SEC[interval]
    groups = []
    current_t = None
    current = None
    for b in rows or []:
        t = int(b['t']) // sec * sec
        if t != current_t:
            if current is not None:
                groups.append(current)
            current_t = t
            current = dict(b, t=t)
        else:
            current['h'] = max(current['h'], b['h'])
            current['l'] = min(current['l'], b['l'])
            current['c'] = b['c']
            current['v'] += b.get('v', 0.0)
            current['q'] += b.get('q', 0.0)
    if current is not None:
        groups.append(current)
    return groups


def _ema(values: Sequence[float], period: int) -> List[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    out = [float(values[0])]
    for value in values[1:]:
        out.append(alpha * float(value) + (1.0 - alpha) * out[-1])
    return out


def _rsi(closes: Sequence[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for a, b in zip(closes[-period - 1:-1], closes[-period:]):
        d = b - a
        gains.append(max(0.0, d))
        losses.append(max(0.0, -d))
    ag = sum(gains) / period
    al = sum(losses) / period
    # Wilder smoothing over the available history avoids a one-window RSI bias.
    all_gains, all_losses = [], []
    for a, b in zip(closes[:-1], closes[1:]):
        d = b - a
        all_gains.append(max(0.0, d))
        all_losses.append(max(0.0, -d))
    if len(all_gains) > period:
        ag = sum(all_gains[:period]) / period
        al = sum(all_losses[:period]) / period
        for g, l in zip(all_gains[period:], all_losses[period:]):
            ag = (ag * (period - 1) + g) / period
            al = (al * (period - 1) + l) / period
    if al <= 1e-15:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(rows: Sequence[Dict], period: int = 14) -> List[float]:
    if not rows:
        return []
    trs = []
    for i, b in enumerate(rows):
        if i == 0:
            tr = b['h'] - b['l']
        else:
            pc = rows[i - 1]['c']
            tr = max(b['h'] - b['l'], abs(b['h'] - pc), abs(b['l'] - pc))
        trs.append(max(0.0, tr))
    n = min(period, len(trs))
    value = sum(trs[:n]) / n
    out = [value] * n
    for tr in trs[n:]:
        value = (value * (period - 1) + tr) / period
        out.append(value)
    return out


def timeframe_features(rows: Sequence[Dict]) -> Dict:
    """Features at the last supplied bar only; no future data is consulted."""
    if len(rows) < 55:
        raise ValueError('At least 55 closed bars are required')
    closes = [b['c'] for b in rows]
    volumes = [b.get('v', 0.0) for b in rows]
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema(macd, 9)
    atrs = _atr(rows, 14)
    atr = max(atrs[-1], closes[-1] * 1e-8)
    base_vols = volumes[-21:-1] if len(volumes) > 1 else volumes
    positive_vols = [v for v in base_vols if v > 0]
    med_vol = statistics.median(positive_vols) if positive_vols else 0.0
    rel_vol = volumes[-1] / med_vol if med_vol > 0 else 1.0
    window = rows[-20:]
    hi = max(b['h'] for b in window)
    lo = min(b['l'] for b in window)
    range_pos = 2.0 * (closes[-1] - lo) / (hi - lo) - 1.0 if hi > lo else 0.0
    swing = rows[-12:]
    rsi = _rsi(closes)
    candle_range = rows[-1]['h'] - rows[-1]['l']
    body = ((closes[-1] - rows[-1]['o']) / candle_range) if candle_range > 0 else 0.0
    ema_trend = (ema20[-1] - ema50[-1]) / (2.0 * atr)
    price_trend = (closes[-1] - ema50[-1]) / (3.0 * atr)
    return {
        'price': closes[-1], 'atr': atr, 'atr_pct': 100.0 * atr / closes[-1],
        'ema_fast': ema20[-1], 'ema_slow': ema50[-1],
        'trend': _clamp(0.65 * ema_trend + 0.35 * price_trend),
        'momentum': _clamp((closes[-1] - closes[-4]) / (2.0 * atr)),
        'rsi': rsi, 'rsi_signed': _clamp((rsi - 50.0) / 25.0),
        'macd': _clamp((macd[-1] - signal[-1]) / atr),
        'relative_volume': max(0.0, rel_vol), 'range_position': _clamp(range_pos),
        'body': _clamp(body), 'swing_high': max(b['h'] for b in swing),
        'swing_low': min(b['l'] for b in swing), 'bar_time': int(rows[-1]['t']),
    }


def model_vector(features: Dict[str, Dict], side: str) -> Dict[str, float]:
    """Direction-normalized vector: one model can evaluate LONG and SHORT."""
    sg = _sign(side)
    f15, f1h, f4h = features['15m'], features['1h'], features['4h']
    alignment = sg * (0.25 * _clamp(f15['trend']) + 0.35 * _clamp(f1h['trend'])
                      + 0.40 * _clamp(f4h['trend']))
    return {
        'trend_15m': sg * f15['trend'], 'trend_1h': sg * f1h['trend'],
        'trend_4h': sg * f4h['trend'], 'momentum_15m': sg * f15['momentum'],
        'momentum_1h': sg * f1h['momentum'], 'rsi_15m': sg * f15['rsi_signed'],
        'macd_15m': sg * f15['macd'], 'range_position_15m': sg * f15['range_position'],
        'relative_volume_15m': _clamp(math.log(max(0.05, f15['relative_volume'])) / math.log(3.0)),
        'atr_pct_15m': _clamp(f15['atr_pct'] / 3.0, 0.0, 1.0),
        'alignment': alignment,
    }


class OnlineLogisticModel:
    """Small, inspectable SGD logistic model with L2 and prequential metrics."""

    def __init__(self, learning_rate: float = 0.035, l2: float = 0.0001):
        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.weights = {name: 0.0 for name in MODEL_FEATURES}
        self.bias = 0.0
        self.train_samples = 0
        self.oos_samples = 0
        self.brier_sum = 0.0
        self.correct = 0

    def predict_vector(self, vector: Dict[str, float]) -> float:
        z = self.bias + sum(self.weights[k] * _finite(vector.get(k), 0.0)
                            for k in MODEL_FEATURES)
        z = _clamp(z, -30.0, 30.0)
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, features: Dict[str, Dict], side: str) -> float:
        return self.predict_vector(model_vector(features, side))

    def update_vector(self, vector: Dict[str, float], y: int) -> float:
        if y not in (0, 1):
            raise ValueError('Label must be 0 or 1')
        p = self.predict_vector(vector)
        error = float(y) - p
        for name in MODEL_FEATURES:
            w = self.weights[name]
            x = _finite(vector.get(name), 0.0)
            self.weights[name] = w + self.learning_rate * (error * x - self.l2 * w)
        self.bias += self.learning_rate * error
        self.train_samples += 1
        return p

    def update(self, features: Dict[str, Dict], side: str, y: int) -> float:
        return self.update_vector(model_vector(features, side), y)

    def observe(self, features: Dict[str, Dict], side: str, y: int) -> float:
        """Prequential update: score first, only then learn the matured label."""
        vector = model_vector(features, side)
        p = self.predict_vector(vector)
        self.oos_samples += 1
        self.brier_sum += (float(y) - p) ** 2
        self.correct += int((p >= 0.5) == bool(y))
        self.update_vector(vector, y)
        return p

    def state(self, min_oos: int = 200) -> Dict:
        return {
            'version': MODEL_VERSION, 'feature_names': list(MODEL_FEATURES),
            'weights': dict(self.weights), 'bias': self.bias,
            'learning_rate': self.learning_rate, 'l2': self.l2,
            'train_samples': self.train_samples, 'oos_samples': self.oos_samples,
            'ready': self.oos_samples >= int(min_oos), 'min_oos': int(min_oos),
            'brier': (self.brier_sum / self.oos_samples) if self.oos_samples else None,
            'accuracy': (self.correct / self.oos_samples) if self.oos_samples else None,
            'brier_sum': self.brier_sum, 'correct': self.correct,
            'probability_note': 'Online score; not claimed calibrated without external calibration.',
        }

    def to_json(self) -> str:
        return json.dumps(self.state(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_state(cls, state: Dict) -> 'OnlineLogisticModel':
        if not isinstance(state, dict) or state.get('version') != MODEL_VERSION:
            raise ValueError('Unsupported or missing professional AI model version')
        if tuple(state.get('feature_names') or ()) != MODEL_FEATURES:
            raise ValueError('Professional AI feature schema mismatch')
        model = cls(state.get('learning_rate', 0.035), state.get('l2', 0.0001))
        weights = state.get('weights') or {}
        for name in MODEL_FEATURES:
            value = _finite(weights.get(name))
            if value is None or abs(value) > 1e6:
                raise ValueError('Invalid professional AI model weight')
            model.weights[name] = value
        model.bias = _finite(state.get('bias'))
        if model.bias is None:
            raise ValueError('Invalid professional AI model bias')
        model.train_samples = max(0, int(state.get('train_samples') or 0))
        model.oos_samples = max(0, int(state.get('oos_samples') or 0))
        model.brier_sum = max(0.0, _finite(state.get('brier_sum'), 0.0))
        model.correct = max(0, int(state.get('correct') or 0))
        return model


class ProfessionalBrain:
    def __init__(self, config: BrainConfig = DEFAULT_CONFIG,
                 model: Optional[OnlineLogisticModel] = None):
        self.config = config
        self.model = model or OnlineLogisticModel(config.model_lr, config.model_l2)

    def _prepare(self, market: Dict[str, Iterable], now: float) -> Tuple[Dict[str, List[Dict]], List[str]]:
        prepared, issues = {}, []
        for tf in ('15m', '1h', '4h'):
            bars = normalize_ohlcv((market or {}).get(tf) or [], tf, now=now)
            prepared[tf] = bars
            minimum = getattr(self.config, f'min_bars_{tf}')
            if len(bars) < minimum:
                issues.append(f'{tf}: лише {len(bars)} закритих барів, потрібно {minimum}')
                continue
            last_close = bars[-1]['t'] + TF_SEC[tf]
            if now - last_close > 1.5 * TF_SEC[tf]:
                issues.append(f'{tf}: застарілі дані')
            # Recent holes can invalidate indicators. Old listing gaps are not relevant.
            for a, b in zip(bars[-61:-1], bars[-60:]):
                if b['t'] - a['t'] > self.config.max_gap_mult * TF_SEC[tf]:
                    issues.append(f'{tf}: розрив історії біля {b["t"]}')
                    break
        return prepared, issues

    def _risk_plan(self, entry: float, side: str, atr: float,
                   rows: Sequence[Dict]) -> Dict:
        cfg = self.config
        sg = _sign(side)
        swing = rows[-12:] if rows else []
        if side == 'LONG':
            structural = entry - min((b['l'] for b in swing), default=entry - atr) + 0.20 * atr
        else:
            structural = max((b['h'] for b in swing), default=entry + atr) - entry + 0.20 * atr
        distance = max(cfg.stop_atr * atr, structural)
        distance = min(cfg.max_stop_atr * atr, distance)
        stop = entry - sg * distance
        tp1 = entry + sg * distance
        tp2 = entry + sg * cfg.target_r * distance
        risk_pct = 100.0 * distance / entry
        cost_pct = 2.0 * (cfg.fee_pct + cfg.slippage_pct)
        net_rr = ((cfg.target_r * risk_pct - cost_pct) / (risk_pct + cost_pct)) if risk_pct > 0 else 0.0
        return {
            'entry': _round(entry, 10), 'stop': _round(stop, 10),
            'tp1': _round(tp1, 10), 'tp2': _round(tp2, 10),
            'risk_distance': _round(distance, 10), 'risk_pct': _round(risk_pct),
            'gross_rr': _round(cfg.target_r, 3), 'net_rr_after_costs': _round(net_rr, 3),
            'assumed_roundtrip_cost_pct': _round(cost_pct, 4),
            'invalidation': '15m close beyond structural/ATR stop; live protective stop remains in Trade Manager',
        }

    def _side(self, side: str, features: Dict[str, Dict], require_model: bool) -> Dict:
        cfg = self.config
        sg = _sign(side)
        f15, f1h, f4h = features['15m'], features['1h'], features['4h']
        trend = sg * (0.20 * f15['trend'] + 0.35 * f1h['trend'] + 0.45 * f4h['trend'])
        alignment = sg * (0.25 * f15['trend'] + 0.35 * f1h['trend'] + 0.40 * f4h['trend'])
        momentum = sg * (0.45 * f15['momentum'] + 0.35 * f15['macd'] + 0.20 * f1h['momentum'])
        breakout = sg * f15['range_position']
        volume_score = _clamp(math.log(max(0.05, f15['relative_volume']) / 0.5)
                              / math.log(6.0), 0.0, 1.0)
        if f15['relative_volume'] > 6.0:
            volume_score *= 0.8
        components = {
            'trend': (trend + 1.0) / 2.0, 'momentum': (momentum + 1.0) / 2.0,
            'alignment': (alignment + 1.0) / 2.0, 'breakout': (breakout + 1.0) / 2.0,
            'participation': volume_score, 'candle': (sg * f15['body'] + 1.0) / 2.0,
        }
        quality = 100.0 * (0.32 * components['trend'] + 0.22 * components['momentum']
                           + 0.20 * components['alignment'] + 0.10 * components['breakout']
                           + 0.08 * components['participation'] + 0.08 * components['candle'])
        vetoes = []
        if quality < cfg.entry_quality:
            vetoes.append(f'якість {quality:.1f} нижча за {cfg.entry_quality:.1f}')
        if trend < 0.12:
            vetoes.append('багатотаймфреймовий тренд не підтверджує бік')
        if sg * f4h['trend'] < -0.30:
            vetoes.append('сильний протилежний режим 4h')
        if sg * f1h['trend'] < -0.20 and sg * f4h['trend'] < 0.05:
            vetoes.append('конфлікт 1h і 4h')
        if momentum < -0.25:
            vetoes.append('імпульс 15m спрямований проти входу')
        if not (cfg.min_atr_pct <= f15['atr_pct'] <= cfg.max_atr_pct):
            vetoes.append('волатильність поза допустимим діапазоном')
        if f15['relative_volume'] < cfg.min_relative_volume:
            vetoes.append('недостатня участь обсягу')
        p = self.model.predict(features, side)
        model_state = self.model.state(cfg.model_min_oos)
        if require_model and not model_state['ready']:
            vetoes.append(f'модель ще не готова: OOS {model_state["oos_samples"]}/{cfg.model_min_oos}')
        if model_state['ready']:
            threshold = cfg.model_min_probability if require_model else 0.50
            if p < threshold:
                vetoes.append(f'модель проти: score {p:.3f} < {threshold:.3f}')
        plan = self._risk_plan(f15['price'], side, f15['atr'], [])
        # Supply the actual execution bars for structural stops in analyze().
        return {'side': side, 'quality': _round(quality, 2), 'components': _json_safe(components),
                'trend_signed': _round(trend), 'momentum_signed': _round(momentum),
                'model_probability': _round(p), 'model_is_ready': model_state['ready'],
                'vetoes': vetoes, 'risk_plan': plan, 'ready': not vetoes}

    def analyze(self, market: Dict[str, Iterable], now: float,
                require_model: bool = False) -> Dict:
        prepared, issues = self._prepare(market, now)
        model_state = self.model.state(self.config.model_min_oos)
        base = {'ok': False, 'action': 'WAIT', 'side': None, 'ts': _round(now, 3),
                'data_issues': issues, 'vetoes': list(issues), 'model': model_state,
                'regime': {'trend': 'unknown', 'volatility': 'unknown'},
                'limitations': 'Closed bars; no funding, liquidation or order-book guarantee.'}
        if issues:
            return base
        features = {tf: timeframe_features(prepared[tf]) for tf in ('15m', '1h', '4h')}
        f15, f1h, f4h = features['15m'], features['1h'], features['4h']
        sides = {}
        for side in SIDES:
            result = self._side(side, features, require_model)
            result['risk_plan'] = self._risk_plan(f15['price'], side, f15['atr'], prepared['15m'])
            if result['risk_plan']['net_rr_after_costs'] < self.config.min_net_rr:
                result['vetoes'].append('R:R після комісії й проковзування занизький')
            result['ready'] = not result['vetoes']
            sides[side] = result
        best = max(SIDES, key=lambda sd: sides[sd]['quality'])
        composite = 0.25 * f15['trend'] + 0.35 * f1h['trend'] + 0.40 * f4h['trend']
        regime = 'up' if composite > 0.18 else ('down' if composite < -0.18 else 'range')
        volatility = ('extreme' if f15['atr_pct'] > self.config.max_atr_pct else
                      'high' if f15['atr_pct'] > 1.5 else
                      'low' if f15['atr_pct'] < 0.08 else 'normal')
        ready = [sd for sd in SIDES if sides[sd]['ready']]
        action = best if best in ready else 'WAIT'
        return {
            'ok': True, 'action': action, 'side': best if action != 'WAIT' else None,
            'ts': _round(now, 3), 'data_issues': [], 'vetoes': list(sides[best]['vetoes']),
            'best_side': best, 'quality': sides[best]['quality'], 'sides': sides,
            'risk_plan': sides[best]['risk_plan'], 'features': _json_safe(features),
            'regime': {'trend': regime, 'composite': _round(composite), 'volatility': volatility,
                       'atr_pct_15m': _round(f15['atr_pct'])},
            'data': {tf: {'bars': len(prepared[tf]), 'last_open': prepared[tf][-1]['t'],
                          'last_close': prepared[tf][-1]['t'] + TF_SEC[tf],
                          'age_sec': _round(now - prepared[tf][-1]['t'] - TF_SEC[tf], 3)}
                     for tf in ('15m', '1h', '4h')},
            'model': model_state,
            'limitations': 'Closed bars; model score is not a guaranteed or externally calibrated win rate.',
        }

    def manage_position(self, market: Dict[str, Iterable], side: str, entry_price: float,
                        opened_at: float, now: float, current_price: Optional[float] = None,
                        peak_price: Optional[float] = None, require_model: bool = False) -> Dict:
        """Return HOLD or CLOSE with evidence. ML alone can never close a trade."""
        side = str(side).upper()
        if side not in SIDES:
            raise ValueError('side must be LONG or SHORT')
        analysis = self.analyze(market, now, require_model=require_model)
        result = {'action': 'HOLD', 'close': False, 'reason': 'no_confirmed_exit',
                  'side': side, 'ts': _round(now, 3), 'analysis': analysis}
        if not analysis.get('ok'):
            result['reason'] = 'market_data_unavailable'
            result['explanation'] = 'Немає надійного нового рішення; аварійні стопи Trade Manager лишаються активними.'
            return result
        bars = normalize_ohlcv(market.get('15m') or [], '15m', now=now)
        f15 = analysis['features']['15m']
        px = _finite(current_price, f15['price'])
        entry = _finite(entry_price)
        if entry is None or entry <= 0 or px is None or px <= 0:
            result['reason'] = 'invalid_position_data'
            return result
        times = [b['t'] for b in bars]
        idx = max(0, bisect.bisect_right(times, float(opened_at)) - 1)
        entry_bars = bars[:idx + 1]
        try:
            entry_features = timeframe_features(entry_bars)
            atr = entry_features['atr']
            risk_source = 'ATR_at_entry'
        except (ValueError, IndexError):
            atr = f15['atr']
            risk_source = 'current_ATR_approximation'
        initial = self._risk_plan(entry, side, atr, entry_bars)
        distance = max(float(initial['risk_distance']), entry * 1e-8)
        sg = _sign(side)
        r_multiple = sg * (px - entry) / distance
        peak = _finite(peak_price, px)
        peak_r = sg * (peak - entry) / distance if peak and peak > 0 else r_multiple
        result.update({'entry_risk': initial, 'risk_source': risk_source,
                       'current_price': _round(px, 10), 'r_multiple': _round(r_multiple, 3),
                       'peak_r': _round(peak_r, 3)})
        own = analysis['sides'][side]
        opp_side = _opposite(side)
        opp = analysis['sides'][opp_side]

        def close(reason, explanation):
            result.update({'action': 'CLOSE', 'close': True, 'reason': reason,
                           'explanation': explanation})
            return result

        # Protective price exits can use the live price. Pattern decisions below use
        # closed bars only, so a forming candle cannot trigger a reversal exit.
        if sg * (px - initial['stop']) <= 0:
            return close('professional_invalidation', 'Ціна зламала початковий ATR/структурний стоп сценарію.')
        if r_multiple >= self.config.target_r:
            return close('professional_target', f'Досягнуто ціль {self.config.target_r:.1f}R за планом ШІ.')

        previous_confirms = False
        if len(bars) >= 56:
            try:
                prev = timeframe_features(bars[:-1])
                previous_confirms = sg * prev['momentum'] < 0.05
            except ValueError:
                previous_confirms = False
        trend_against = (sg * f15['trend'] < -0.08 and sg * analysis['features']['1h']['trend'] < -0.05)
        momentum_against = sg * f15['momentum'] < -0.15
        if (own['quality'] < 48.0 and opp['quality'] >= self.config.reversal_quality
                and trend_against and momentum_against and previous_confirms):
            return close('professional_reversal',
                         f'Підтверджений розворот: власна якість {own["quality"]:.1f}, '
                         f'протилежна {opp["quality"]:.1f}; 15m/1h і дві свічки імпульсу проти угоди.')
        if peak_r >= 1.0 and r_multiple <= 0.5 * peak_r and momentum_against:
            return close('professional_profit_giveback',
                         f'Віддача прибутку: пік {peak_r:.2f}R → {r_multiple:.2f}R, імпульс 15m проти угоди.')
        if (now - float(opened_at) >= 32 * TF_SEC['15m'] and abs(r_multiple) < 0.15
                and own['quality'] < 50.0 and opp['quality'] > own['quality'] + 8.0):
            return close('professional_time_stop',
                         'Сценарій не реалізувався за 32 закриті 15m бари; краща якість уже в протилежному боці.')
        result['explanation'] = 'Підтвердженого виходу немає: стоп/ціль не досягнуто, розворот не підтверджений.'
        result['monitor'] = {'own_quality': own['quality'], 'opposite_quality': opp['quality'],
                             'trend_against': trend_against, 'momentum_against': momentum_against,
                             'two_bar_confirmation': previous_confirms}
        return result

    def model_state(self) -> Dict:
        return self.model.state(self.config.model_min_oos)

    def export_model(self) -> str:
        return self.model.to_json()

    def import_model(self, state: Dict) -> None:
        self.model = OnlineLogisticModel.from_state(state)


def _json_safe(value):
    """Recursively convert to finite JSON values (never NaN/Infinity)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _market_at(primary: Sequence[Dict], index: int, higher: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    close_time = primary[index]['t'] + TF_SEC['15m']
    return {
        '15m': list(primary[:index + 1]),
        '1h': [b for b in higher['1h'] if b['t'] + TF_SEC['1h'] <= close_time],
        '4h': [b for b in higher['4h'] if b['t'] + TF_SEC['4h'] <= close_time],
    }


def _label(primary: Sequence[Dict], index: int, side: str, horizon: int,
           roundtrip_cost_pct: float) -> int:
    future = primary[index + horizon]['c']
    move = 100.0 * _sign(side) * (future / primary[index]['c'] - 1.0)
    return int(move > roundtrip_cost_pct)


def _fill(raw_price: float, side: str, is_entry: bool, slippage_pct: float) -> float:
    # Buy orders pay more, sell orders receive less. SHORT entry is a sell.
    buy = (side == 'LONG') == is_entry
    return raw_price * (1.0 + slippage_pct / 100.0 if buy else 1.0 - slippage_pct / 100.0)


def _barrier(primary: Sequence[Dict], index: int, side: str, stop: float, target: float):
    """Conservative intrabar fill: stop first when SL and TP share one OHLC bar."""
    b = primary[index]
    sg = _sign(side)
    stop_hit_open = (b['o'] <= stop) if side == 'LONG' else (b['o'] >= stop)
    target_hit_open = (b['o'] >= target) if side == 'LONG' else (b['o'] <= target)
    if stop_hit_open:
        return b['o'], 'stop_gap'
    if target_hit_open:
        return b['o'], 'target_gap'
    stop_hit = (b['l'] <= stop) if side == 'LONG' else (b['h'] >= stop)
    target_hit = (b['h'] >= target) if side == 'LONG' else (b['l'] <= target)
    if stop_hit:
        return stop, 'stop' if not target_hit else 'stop_same_bar_conservative'
    if target_hit:
        return target, 'target'
    return None


def _metrics(trades: Sequence[Dict], initial_equity: float = 100000.0) -> Dict:
    pnl = [float(t['pnl_pct']) for t in trades]
    equity = float(initial_equity)
    peak = equity
    max_dd = 0.0
    curve = []
    for p in pnl:
        equity *= max(0.0, 1.0 + p / 100.0)
        peak = max(peak, equity)
        max_dd = max(max_dd, 100.0 * (peak - equity) / peak if peak > 0 else 0.0)
        curve.append(_round(equity, 4))
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    mean = statistics.fmean(pnl) if pnl else 0.0
    sd = statistics.stdev(pnl) if len(pnl) > 1 else 0.0
    return {
        'trades': len(pnl), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': _round(100.0 * len(wins) / len(pnl), 3) if pnl else None,
        'expectancy_pct': _round(mean), 'profit_factor': _round(sum(wins) / abs(sum(losses)), 4) if losses else None,
        'max_drawdown_pct': _round(max_dd, 4), 'sharpe_per_trade': _round(math.sqrt(len(pnl)) * mean / sd, 4) if sd > 0 else None,
        'total_return_pct': _round(100.0 * (equity / initial_equity - 1.0), 4),
        'final_equity': _round(equity, 4), 'equity_curve': curve,
    }


def walk_forward_backtest(rows: Iterable, config: BrainConfig = DEFAULT_CONFIG,
                          require_model: bool = False) -> Dict:
    """Purged, embargoed, chronological OOS simulation on closed 15m bars.

    Higher timeframes are built only from 15m bars already closed at decision time.
    Each split is trained afresh on labels fully matured before its test window.
    Within the test window labels become learnable only after horizon + embargo.
    """
    cfg = config
    raw = list(rows or [])
    inferred_now = None
    # Do not silently discard the last historical row merely because it is old now.
    # A caller's last bar is treated as closed for an offline dataset; normalize with
    # drop_unclosed=False, then explicitly remove a still-forming live bar beforehand.
    primary = normalize_ohlcv(raw, '15m', now=inferred_now, drop_unclosed=False)
    if len(primary) < cfg.initial_train_bars + cfg.horizon_bars + 2:
        raise ValueError(f'Need at least {cfg.initial_train_bars + cfg.horizon_bars + 2} closed 15m bars')
    higher = {'1h': resample_ohlcv(primary, '1h'), '4h': resample_ohlcv(primary, '4h')}
    brain = ProfessionalBrain(cfg)
    trades, split_reports = [], []
    cost = 2.0 * (cfg.fee_pct + cfg.slippage_pct)
    n = len(primary)
    split_start = cfg.initial_train_bars
    split_id = 0
    while split_start < n - cfg.horizon_bars - 1:
        split_id += 1
        split_end = min(n - cfg.horizon_bars - 1, split_start + cfg.walkforward_test_bars)
        model = OnlineLogisticModel(cfg.model_lr, cfg.model_l2)
        brain.model = model
        train_end = split_start - cfg.embargo_bars - cfg.horizon_bars
        train_begin = max(0, train_end - cfg.rolling_train_bars)
        trained = 0
        max_train_label_index = -1
        for i in range(train_begin, train_end):
            market = _market_at(primary, i, higher)
            if len(market['4h']) < cfg.min_bars_4h or len(market['1h']) < cfg.min_bars_1h:
                continue
            features = {tf: timeframe_features(market[tf]) for tf in ('15m', '1h', '4h')}
            for side in SIDES:
                model.update(features, side, _label(primary, i, side, cfg.horizon_bars, cost))
            trained += 1
            max_train_label_index = max(max_train_label_index, i + cfg.horizon_bars)
        split_reports.append({'split': split_id, 'test_start_index': split_start,
                              'test_end_index_exclusive': split_end,
                              'train_start_index': train_begin, 'train_end_index_exclusive': train_end,
                              'max_train_label_index': max_train_label_index,
                              'embargo_bars': cfg.embargo_bars, 'trained_samples_per_side': trained,
                              'oos_trades': 0})
        pending = []
        next_entry = split_start
        for i in range(split_start, split_end):
            matured = [j for j in pending if j + cfg.horizon_bars + cfg.embargo_bars <= i]
            pending = [j for j in pending if j + cfg.horizon_bars + cfg.embargo_bars > i]
            for j in matured:
                market_j = _market_at(primary, j, higher)
                features_j = {tf: timeframe_features(market_j[tf]) for tf in ('15m', '1h', '4h')}
                for side in SIDES:
                    model.observe(features_j, side, _label(primary, j, side, cfg.horizon_bars, cost))
            pending.append(i)
            if i < next_entry or i + 1 >= n:
                continue
            analysis = brain.analyze(_market_at(primary, i, higher),
                                     primary[i]['t'] + TF_SEC['15m'], require_model=require_model)
            if analysis['action'] == 'WAIT':
                continue
            side = analysis['action']
            plan = analysis['sides'][side]['risk_plan']
            entry_index = i + 1
            entry = _fill(primary[entry_index]['o'], side, True, cfg.slippage_pct)
            distance = float(plan['risk_distance'])
            sg = _sign(side)
            stop = entry - sg * distance
            target = entry + sg * cfg.target_r * distance
            last_index = min(n - 1, entry_index + cfg.horizon_bars - 1)
            exit_price = None
            exit_index = last_index
            reason = 'time_exit'
            peak_price = entry
            for j in range(entry_index, last_index + 1):
                b = primary[j]
                peak_price = max(peak_price, b['h']) if side == 'LONG' else min(peak_price, b['l'])
                barrier = _barrier(primary, j, side, stop, target)
                if barrier:
                    exit_price, reason = barrier
                    exit_index = j
                    break
                if cfg.use_brain_exits:
                    decision = brain.manage_position(_market_at(primary, j, higher), side, entry,
                                                     primary[entry_index]['t'], b['t'] + TF_SEC['15m'],
                                                     current_price=b['c'], peak_price=peak_price,
                                                     require_model=require_model)
                    if decision['close']:
                        exit_price, exit_index = b['c'], j
                        reason = decision['reason']
                        break
            if exit_price is None:
                exit_price = primary[last_index]['c']
            exit_fill = _fill(exit_price, side, False, cfg.slippage_pct)
            pnl = 100.0 * sg * (exit_fill / entry - 1.0) - 2.0 * cfg.fee_pct
            risk_pct = 100.0 * distance / entry
            trades.append({'split': split_id, 'side': side,
                           'signal_index': i, 'entry_index': entry_index, 'exit_index': exit_index,
                           'entry_time': primary[entry_index]['t'], 'exit_time': primary[exit_index]['t'],
                           'entry': _round(entry, 10), 'exit': _round(exit_fill, 10),
                           'stop': _round(stop, 10), 'target': _round(target, 10),
                           'pnl_pct': _round(pnl, 6), 'r_net': _round(pnl / risk_pct if risk_pct > 0 else 0.0, 5),
                           'reason': reason, 'entry_quality': analysis['sides'][side]['quality']})
            split_reports[-1]['oos_trades'] += 1
            next_entry = exit_index + 1
        split_start = split_end
    result = {'ok': True, 'mode': 'purged_walk_forward', 'config': asdict(cfg),
              'bars': n, 'splits': split_reports, 'metrics': _metrics(trades), 'trades': trades,
              'model': brain.model_state(), 'limitations': [
                  'Funding, borrow/liquidation, variable spread, latency and order-book depth are not simulated.',
                  'OHLC cannot establish intrabar order; SL wins every ambiguous SL/TP bar.',
                  'Synthetic or in-sample success is not evidence of a live market edge.',
                  'Use real unseen data and paper forward testing before enabling real orders.',
              ]}
    return _json_safe(result)


def load_csv_ohlcv(path: str) -> List[Dict]:
    """CSV helper for the CLI: timestamp/open/high/low/close/volume[/turnover]."""
    import csv
    with open(path, 'r', encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))
