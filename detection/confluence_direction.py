"""
confluence_direction — режим «Confluence» для Smart Direction.

Найсильніший сигнал = ЗГОДА кількох НЕЗАЛЕЖНИХ вимірів (щоб не подвоювати одне
й те саме). Керує кнопками LONG/SHORT лише коли виміри узгоджені й рух НЕ
виснажений.

Виміри (кожен голосує LONG/+1 · SHORT/−1 · нейтр/0, зі своєю вагою):
  • Тренд (CTR-кросовер STC 15m/1H/4H) — напрямок тренду (last_dir, не рівень);
  • Forecast (Fib 1H/4H)             — проєкція ціни × впевненість;
  • Бабло (ММ-модель)                — де застряг капітал (funding/L/S/кластери).
Гейт ВИСНАЖЕННЯ (не голос, а стоп): якщо рух у бік сигналу вже виснажений
(exhaustion ≥ поріг) → сигнал «пізно» → WAIT.

Вихід сумісний зі smart_direction: {allow_long, allow_short, mode, reason, ok}
+ score/score_pct + per_dim[] (для таблиці в UI) + exhausted.
"""
from typing import Dict, Optional

_TF_W = [('15m', 0.20), ('1h', 0.35), ('4h', 0.45)]   # ваги ТФ для CTR-тренду
_DIM_W = {'trend': 0.40, 'forecast': 0.30, 'bablo': 0.30}
_THRESHOLD = 0.35            # |score| ≥ цього → напрямок
_EXH_LATE = 80.0            # exhaustion ≥ цього → рух «пізно» → WAIT


def _ctr_trend(symbol: str):
    """Тренд за КРОСОВЕРОМ STC (last_dir), а не за рівнем. → (vote[-1..1], detail)."""
    try:
        from detection.forecast_engine import get_forecast_engine
        fe = get_forecast_engine()
        if not fe:
            return 0.0, '—'
        num = wsum = 0.0
        parts = []
        for tf, w in _TF_W:
            try:
                r = fe.get_ctr_tf(symbol, tf)
            except Exception:
                r = None
            ld = (r or {}).get('last_dir')
            age = (r or {}).get('age')
            if ld not in ('LONG', 'SHORT'):
                continue
            sign = 1.0 if ld == 'LONG' else -1.0
            # Свіжий кросовер важить повністю, старий — менше (загасання за віком).
            fresh = 1.0 if (age is None or age <= 3) else max(0.4, 1.0 - (age - 3) * 0.1)
            num += w * sign * fresh
            wsum += w
            parts.append({'name': tf, 'dir': ld})
        if wsum <= 0:
            return 0.0, []
        return max(-1.0, min(1.0, num / wsum)), parts
    except Exception:
        return 0.0, []


def _forecast_vote(symbol: str, verdict_data: Optional[Dict]):
    """Fib-прогноз 1H/4H × впевненість. → (vote[-1..1], detail)."""
    try:
        fc1 = fc4 = None
        if verdict_data:
            comp = (verdict_data.get('components') or {})
            fc1 = comp.get('forecast')          # {side, confidence} (1H)
        # Прямо з движка (свіже, кешоване), щоб мати і 4H.
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            fc = fe.get(symbol) if fe else {}
            fc1 = fc.get('forecast_1h') or fc1
            fc4 = fc.get('forecast_4h')
        except Exception:
            pass
        num = wsum = 0.0
        parts = []
        for fcx, w, lbl in ((fc1, 0.4, '1H'), (fc4, 0.6, '4H')):
            if not fcx:
                continue
            side = fcx.get('side')
            if side not in (1, -1):
                continue
            conf = float(fcx.get('confidence') or 0) / 100.0
            num += w * (1.0 if side == 1 else -1.0) * conf
            wsum += w
            parts.append({'name': lbl, 'dir': 'LONG' if side == 1 else 'SHORT',
                          'val': f"{int(fcx.get('confidence') or 0)}%"})
        if wsum <= 0:
            return 0.0, []
        return max(-1.0, min(1.0, num / wsum)), parts
    except Exception:
        return 0.0, []


def _bablo_vote(db, symbol: str):
    """ММ-модель (funding/L/S/кластери). → (vote[-1..1], detail)."""
    try:
        from detection.mm_model import compute_mm
        r = compute_mm(db, symbol, with_confirmations=True)
        if not r:
            return 0.0, '—'
        st = r.get('status')
        strg = float(r.get('strength') or 0) / 100.0
        vv = f"{r.get('strength')}%"
        if st == 'LONG':
            return min(1.0, strg), [{'name': 'ММ', 'dir': 'LONG', 'val': vv}]
        if st == 'SHORT':
            return -min(1.0, strg), [{'name': 'ММ', 'dir': 'SHORT', 'val': vv}]
        return 0.0, [{'name': 'ММ', 'dir': None, 'val': 'рівновага'}]
    except Exception:
        return 0.0, []


def _exhaustion_for(verdict_data: Optional[Dict], direction: Optional[str]) -> Optional[float]:
    try:
        if not verdict_data or direction not in ('LONG', 'SHORT'):
            return None
        mv = verdict_data.get('move_long') if direction == 'LONG' else verdict_data.get('move_short')
        if mv and mv.get('ok'):
            return mv.get('exhaustion')
    except Exception:
        pass
    return None


def compute_confluence(db, symbol: str, verdict_data: Optional[Dict] = None) -> Dict:
    """Конфлюенс незалежних вимірів → напрямок для кнопок LONG/SHORT."""
    out = {'allow_long': False, 'allow_short': False, 'mode': 'WAIT',
           'reason': 'немає даних', 'ok': False, 'score': 0.0, 'score_pct': 0,
           'per_dim': [], 'exhausted': False}
    try:
        t_vote, t_det = _ctr_trend(symbol)
        f_vote, f_det = _forecast_vote(symbol, verdict_data)
        b_vote, b_det = _bablo_vote(db, symbol)
        dims = [
            ('Тренд (CTR)', _DIM_W['trend'], t_vote, t_det),
            ('Forecast', _DIM_W['forecast'], f_vote, f_det),
            ('Бабло (ММ)', _DIM_W['bablo'], b_vote, b_det),
        ]
        per_dim = []
        num = wsum = 0.0
        agree_l = agree_s = 0
        for name, w, vote, det in dims:
            d = 'LONG' if vote > 0.15 else ('SHORT' if vote < -0.15 else None)
            if d == 'LONG':
                agree_l += 1
            elif d == 'SHORT':
                agree_s += 1
            num += w * vote
            wsum += w
            per_dim.append({'name': name, 'dir': d,
                            'conf': int(round(abs(vote) * 100)), 'detail': det})
        out['per_dim'] = per_dim
        if wsum <= 0:
            return out
        score = max(-1.0, min(1.0, num / wsum))
        out['score'] = round(score, 3)
        out['score_pct'] = round(score * 100)
        out['ok'] = True
        agree = max(agree_l, agree_s)

        direction = None
        if score >= _THRESHOLD and agree_l >= 2:
            direction = 'LONG'
        elif score <= -_THRESHOLD and agree_s >= 2:
            direction = 'SHORT'

        # Гейт виснаження: не входимо в бік, що вже майже вичерпаний.
        exh = _exhaustion_for(verdict_data, direction)
        if direction and exh is not None and exh >= _EXH_LATE:
            out['exhausted'] = True
            out['reason'] = f'{direction} {agree}/3, але виснажено {int(exh)}% → WAIT (пізно)'
            return out

        if direction == 'LONG':
            out['allow_long'] = True
            out['mode'] = 'LONG_ONLY'
            out['reason'] = f'конфлюенс {agree}/3 → LONG ({out["score_pct"]:+d}%)'
        elif direction == 'SHORT':
            out['allow_short'] = True
            out['mode'] = 'SHORT_ONLY'
            out['reason'] = f'конфлюенс {agree}/3 → SHORT ({out["score_pct"]:+d}%)'
        else:
            out['reason'] = f'виміри незгодні ({out["score_pct"]:+d}%) → WAIT'
        return out
    except Exception as e:
        try:
            print(f"[confluence] error {symbol}: {e}")
        except Exception:
            pass
        return out
