"""
ctr_direction — режим «CTR_STC» для Smart Direction.

Керує кнопками LONG/SHORT на основі Schaff Trend Cycle (STC) по 15m/1H/4H.
Логіка РОЗВОРОТНА (варіант А, як і сам CTR-нахил):
  • STC > 50 (перекупленість) → SHORT-нахил (чекаємо відкат вниз);
  • STC < 50 (перепроданість)  → LONG-нахил (чекаємо відскок угору).
Голоси ТФ зважуються (старший ТФ важливіший): 15m 0.20 · 1H 0.35 · 4H 0.45.
Впевненість кожного ТФ = |STC−50|/50·100. Підсумковий бал = Σ вага·знак·впевн.
При незгоді ТФ бал近 нуля → WAIT (кнопки вимкнені).

Вихід сумісний зі smart_direction: {allow_long, allow_short, mode, reason, ok}
+ per_tf для UI (напрямок · нахил · впевненість · обґрунтування).
"""
from typing import Dict, Optional

# Вага таймфреймів (старший = важливіший). Тюняться за потреби.
_TF_WEIGHTS = [('15m', 0.20), ('1h', 0.35), ('4h', 0.45)]
_THRESHOLD = 0.25          # |score| ≥ цього → напрямок, інакше WAIT
_TF_LABEL = {'15m': 'CTR 15m', '1h': 'CTR 1H', '4h': 'CTR 4H'}


def _zone(stc: float) -> str:
    if stc >= 75:
        return 'перекупленість'
    if stc <= 25:
        return 'перепроданість'
    return 'нейтральна зона'


def _tf_read(stc: Optional[float]) -> Optional[Dict]:
    """Один ТФ → {stc, lean, conf, zone, sign}. None коли даних немає."""
    if stc is None:
        return None
    stc = float(stc)
    # Розворотно: перекупленість (>50) → SHORT; перепроданість (<50) → LONG.
    if stc > 50:
        lean, sign = 'SHORT', -1.0
    elif stc < 50:
        lean, sign = 'LONG', 1.0
    else:
        lean, sign = None, 0.0
    conf = round(abs(stc - 50.0) / 50.0 * 100.0)
    return {'stc': round(stc, 1), 'lean': lean, 'conf': conf,
            'zone': _zone(stc), 'sign': sign}


def compute_ctr_direction(symbol: str) -> Dict:
    """Напрямок для кнопок LONG/SHORT з CTR/STC 15m/1H/4H (розворотно).

    Return: {allow_long, allow_short, mode, reason, ok, score, per_tf:[...]}.
    per_tf: [{tf, label, stc, lean, conf, zone}] — для показу в UI.
    """
    out = {'allow_long': False, 'allow_short': False, 'mode': 'WAIT',
           'reason': 'немає даних CTR', 'ok': False, 'score': 0.0, 'per_tf': []}
    try:
        from detection.forecast_engine import get_forecast_engine
        fe = get_forecast_engine()
        if not fe:
            return out
        num = 0.0
        wsum = 0.0
        per_tf = []
        for tf, w in _TF_WEIGHTS:
            try:
                r = fe.get_ctr_tf(symbol, tf)
            except Exception:
                r = None
            rd = _tf_read((r or {}).get('stc'))
            if rd is None:
                continue
            num += w * rd['sign'] * (rd['conf'] / 100.0)
            wsum += w
            per_tf.append({'tf': tf, 'label': _TF_LABEL.get(tf, tf),
                           'stc': rd['stc'], 'lean': rd['lean'],
                           'conf': rd['conf'], 'zone': rd['zone']})
        if wsum <= 0:
            return out
        score = max(-1.0, min(1.0, num / wsum))   # ренормовано → [-1..1]
        out['score'] = round(score, 3)
        out['per_tf'] = per_tf
        out['ok'] = True
        if score >= _THRESHOLD:
            out['allow_long'], out['allow_short'] = True, False
            out['mode'] = 'LONG_ONLY'
            out['reason'] = f'CTR перепроданість (bal {score:+.2f}) → LONG'
        elif score <= -_THRESHOLD:
            out['allow_long'], out['allow_short'] = False, True
            out['mode'] = 'SHORT_ONLY'
            out['reason'] = f'CTR перекупленість (bal {score:+.2f}) → SHORT'
        else:
            out['reason'] = f'CTR ТФ незгодні (bal {score:+.2f}) → WAIT'
        return out
    except Exception as e:
        try:
            print(f"[ctr_direction] error {symbol}: {e}")
        except Exception:
            pass
        return out
