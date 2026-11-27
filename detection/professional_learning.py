"""Champion/challenger learning from archived AI episodes and closed trades.

No live I/O and no order routing here. AiEngine supplies database rows and persists
the returned state. A record is processed at most once; a challenger replaces the
champion only after improving on a later, untouched validation subset.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Dict, Iterable, List, Optional, Tuple


def _sibling(name):
    """Сусідній модуль БЕЗ пакетного імпорту: `from detection import …`
    виконує detection/__init__.py, а той тягне півпроєкту (pybit) —
    ізольовані тести й CLI бектесту мусять вантажитись без нього."""
    import importlib.util
    import os
    import sys
    full = f'detection.{name}'
    m = sys.modules.get(full)
    if m is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.py')
        sp = importlib.util.spec_from_file_location(full, p)
        m = importlib.util.module_from_spec(sp)
        sys.modules[full] = m
        sp.loader.exec_module(m)
    return m


_P = _sibling('professional_ai')
MODEL_FEATURES = _P.MODEL_FEATURES
OnlineLogisticModel = _P.OnlineLogisticModel
_finite = _P._finite
model_vector = _P.model_vector


SEEN_CAP = 20000
SEEN_KEEP = 12000


def _loads(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return {}
    return value if isinstance(value, dict) else {}


def _id(symbol, side, event_at, label_at, source):
    raw = f'{source}|{symbol}|{side}|{float(event_at):.3f}|{float(label_at):.3f}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]


def _sample(symbol, side, event_at, label_at, features, outcome_pct, source,
            cost_pct: float) -> Optional[Dict]:
    side = str(side or '').upper()
    outcome = _finite(outcome_pct)
    event = _finite(event_at)
    label = _finite(label_at)
    if side not in ('LONG', 'SHORT') or outcome is None or event is None or label is None:
        return None
    if not isinstance(features, dict) or not all(tf in features for tf in ('15m', '1h', '4h')):
        return None
    try:
        vector = model_vector(features, side)
        if any(_finite(vector.get(k)) is None for k in MODEL_FEATURES):
            return None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return {
        'id': _id(symbol, side, event, label, source), 'symbol': str(symbol or '').upper(),
        'side': side, 'event_at': event, 'label_at': label, 'features': features,
        'y': int(outcome > cost_pct), 'outcome_pct': outcome, 'source': source,
    }


def samples_from_archives(episodes: Iterable[Dict], research: Iterable[Dict],
                          roundtrip_cost_pct: float = 0.15) -> List[Dict]:
    """Extract matured, direction-labelled samples; never infer missing outcomes."""
    rows = []
    for rec in research or []:
        ctx = _loads(rec.get('entry_ctx'))
        snapshot = _loads(rec.get('entry_snapshot'))
        professional = (ctx.get('professional') or snapshot.get('professional') or {})
        features = professional.get('features') or {}
        row = _sample(rec.get('symbol'), rec.get('side'), rec.get('opened_at'),
                      rec.get('closed_at'), features, rec.get('pnl_pct'),
                      'trade_research', roundtrip_cost_pct)
        if row:
            rows.append(row)
    for rec in episodes or []:
        signal = _loads(rec.get('signal'))
        professional = signal.get('professional') or {}
        row = _sample(rec.get('symbol'), rec.get('signal_side') or signal.get('side'),
                      rec.get('signal_at') or signal.get('t'), rec.get('ended_at'),
                      professional.get('features') or {}, rec.get('sig_end_pct'),
                      'ai_episode', roundtrip_cost_pct)
        if row:
            rows.append(row)
    # If the same opening exists in both datasets, prefer the realized trade result.
    dedup = {}
    for row in rows:
        key = (row['symbol'], row['side'], int(row['event_at'] // 60))
        if key not in dedup or row['source'] == 'trade_research':
            dedup[key] = row
    return sorted(dedup.values(), key=lambda r: (r['label_at'], r['event_at'], r['id']))


def _metrics(model: OnlineLogisticModel, samples: List[Dict]) -> Dict:
    if not samples:
        return {'n': 0, 'brier': None, 'logloss': None, 'accuracy': None}
    brier = 0.0
    logloss = 0.0
    correct = 0
    for row in samples:
        p = model.predict(row['features'], row['side'])
        y = row['y']
        brier += (y - p) ** 2
        logloss += -(y * math.log(max(1e-12, p)) + (1 - y) * math.log(max(1e-12, 1.0 - p)))
        correct += int((p >= 0.5) == bool(y))
    n = len(samples)
    return {'n': n, 'brier': brier / n, 'logloss': logloss / n, 'accuracy': correct / n}


class ArchiveLearner:
    def __init__(self, model: Optional[OnlineLogisticModel] = None,
                 min_train: int = 30, min_validation: int = 20,
                 min_brier_improvement: float = 0.001):
        self.model = model or OnlineLogisticModel()
        self.min_train = max(5, int(min_train))
        self.min_validation = max(5, int(min_validation))
        self.min_brier_improvement = max(0.0, float(min_brier_improvement))
        self.seen = set()
        self._order: List[str] = []       # порядок обробки: витісняємо найстаріші
        self.history = []
        self.last_status = 'not_run'
        self.last_report = {}

    def run(self, episodes: Iterable[Dict], research: Iterable[Dict],
            roundtrip_cost_pct: float = 0.15) -> Dict:
        all_rows = samples_from_archives(episodes, research, roundtrip_cost_pct)
        new = [row for row in all_rows if row['id'] not in self.seen]
        if not new:
            report = {'status': 'no_new_labels', 'new_samples': 0,
                      'archive_samples': len(all_rows), 'promoted': False}
            self.last_status, self.last_report = report['status'], report
            return report
        minimum = self.min_train + self.min_validation
        if len(new) < minimum:
            report = {'status': 'insufficient_new_labels', 'new_samples': len(new),
                      'required': minimum, 'archive_samples': len(all_rows), 'promoted': False}
            # Do not mark these as seen: once enough independent outcomes exist,
            # the original untouched batch can still form its first train/validation split.
            self.last_status, self.last_report = report['status'], report
            return report
        n_validation = max(self.min_validation, int(round(len(new) * 0.25)))
        n_validation = min(n_validation, len(new) - self.min_train)
        train = new[:-n_validation]
        validation = new[-n_validation:]
        # Validation outcomes must become available after every training outcome.
        if train[-1]['label_at'] > validation[0]['label_at']:
            report = {'status': 'chronology_veto', 'new_samples': len(new), 'promoted': False}
            self.last_status, self.last_report = report['status'], report
            return report
        baseline = OnlineLogisticModel.from_state(self.model.state())
        challenger = OnlineLogisticModel.from_state(self.model.state())
        for row in train:
            challenger.update(row['features'], row['side'], row['y'])
        before = _metrics(baseline, validation)
        after = _metrics(challenger, validation)
        brier_gain = before['brier'] - after['brier']
        promoted = (brier_gain >= self.min_brier_improvement
                    and after['logloss'] <= before['logloss'])
        if promoted:
            # Metrics above are untouched holdout metrics. Only AFTER evaluation may
            # the accepted model observe these labels and update prequential counters.
            for row in validation:
                challenger.observe(row['features'], row['side'], row['y'])
            self.model = challenger
            status = 'promoted'
        else:
            status = 'rejected_no_improvement'
        # Every validated batch is consumed exactly once, even when its challenger is
        # rejected. This prevents accidental repeated overweighting of old archives.
        for row in new:
            if row['id'] not in self.seen:
                self.seen.add(row['id'])
                self._order.append(row['id'])
        if len(self._order) > SEEN_CAP:
            # Витісняємо НАЙСТАРІШІ оброблені (а не «найменші хеші», як було:
            # сортування хешів = випадкова вибірка, і свіжі записи могли
            # опрацюватись повторно). Рушій читає ≤5000 найновіших на джерело.
            self._order = self._order[-SEEN_KEEP:]
            self.seen = set(self._order)
        report = {
            'status': status, 'promoted': promoted, 'new_samples': len(new),
            'train_samples': len(train), 'validation_samples': len(validation),
            'train_label_range': [train[0]['label_at'], train[-1]['label_at']],
            'validation_label_range': [validation[0]['label_at'], validation[-1]['label_at']],
            'baseline': before, 'challenger': after,
            'brier_improvement': brier_gain, 'required_improvement': self.min_brier_improvement,
            'model': self.model.state(),
        }
        self.history.append(report)
        self.history = self.history[-20:]
        self.last_status, self.last_report = status, report
        return report

    def state(self) -> Dict:
        return {'version': 'archive-champion-challenger-v1', 'model': self.model.state(),
                'seen_ids': list(self._order), 'history': copy.deepcopy(self.history),
                'last_status': self.last_status, 'last_report': copy.deepcopy(self.last_report),
                'min_train': self.min_train, 'min_validation': self.min_validation,
                'min_brier_improvement': self.min_brier_improvement}

    @classmethod
    def from_state(cls, state: Dict) -> 'ArchiveLearner':
        if not isinstance(state, dict) or state.get('version') != 'archive-champion-challenger-v1':
            raise ValueError('Unsupported archive learner state')
        learner = cls(OnlineLogisticModel.from_state(state.get('model') or {}),
                      state.get('min_train', 30), state.get('min_validation', 20),
                      state.get('min_brier_improvement', 0.001))
        learner._order = [str(x) for x in (state.get('seen_ids') or [])][-SEEN_KEEP:]
        learner.seen = set(learner._order)
        learner.history = list(state.get('history') or [])[-20:]
        learner.last_status = str(state.get('last_status') or 'not_run')
        learner.last_report = dict(state.get('last_report') or {})
        return learner
