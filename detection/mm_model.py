"""
mm_model.py — професійна модель показника «ММ» (куди тягне ціну).

ОСНОВА — LIQMAP. Замість примітивної різниці «сума палива вище − сума нижче»
рахуємо Liquidity Pull Vector (LPV): вектор до домінантного, НАЙБЛИЖЧОГО, свіжого
пулу ліквідності. Кожен кластер стопів/ліквідацій — це магніт: близький важить
експоненційно більше (ціна першою його досягає), і найбільший найближчий кластер
дає ціль max-pain. Для монет з активним потоком (насамперед BTC) додаються
ПІДТВЕРДЖЕННЯ:
  • whale-tape  — реальний крупний капітал (агресивний потік buy/sell);
  • стакан      — дисбаланс лімітів (де сидить MM), лише якщо символ уже активний;
  • funding     — переповнена сторона = ймовірна ціль (контр-сигнал).

Ваги LIQMAP-домінантні й РЕНОРМУЮТЬСЯ за наявними сигналами, тож монета лише з
liq-даними теж отримує чистий бал. Усе best-effort: будь-яка помилка чи брак
даних → сигнал просто випадає з міксу, модель не падає.

Повертає dict, сумісний зі старим _fuel_dir ({dir, mark_price, status}) + додаткові
поля для банера/логу: {strength, target, components, weights, data_quality}.
"""
import math
import time
from typing import Dict, Optional

# Дефолтні ваги/пороги (LIQMAP-домінантний мікс). Тюняться через налаштування FF.
_DEFAULTS = {
    'mm_liq_weight': 0.55,      # ядро — Liquidity Pull Vector
    'mm_whale_weight': 0.20,    # реальний крупний капітал
    'mm_book_weight': 0.15,     # ліміти стакана (MM)
    'mm_pos_weight': 0.10,      # позиціювання (funding, контр)
    'mm_prox_pct': 2.0,         # D0 — масштаб проксіміті-загасання, %
    'mm_threshold': 0.15,       # |score| поріг для LONG/SHORT
    'mm_funding_scale': 0.05,   # |funding %|, що дає повний контр-сигнал
    'mm_reach_pct': 15.0,       # далі за цю відстань кластер уже не магніт
    # 🧭 Реконсиляція з РУХОМ ціни. Магніт-модель контртрендова — сама по собі
    # показує, ДЕ ліквідність, а не куди ЙДЕ ціна. Тренд коригує напрямок:
    'mm_trend_weight': 0.30,    # вага м'якого блендингу тренду у спокійному ринку
    'mm_trend_override': 0.5,   # |імпульс| ≥ цього і проти магніту → тренд ВЕДЕ
}

# TTL-кеш funding, щоб не смикати біржу щоцикл на кожну монету.
_funding_cache: Dict[str, tuple] = {}   # symbol -> (ts, rate)
_FUNDING_TTL = 300.0


def get_mm_settings(db) -> Dict:
    """Ваги/пороги з БД поверх дефолтів (усі числові, best-effort)."""
    out = dict(_DEFAULTS)
    try:
        for k in _DEFAULTS:
            v = db.get_setting(k, None)
            if v is not None:
                out[k] = float(v)
    except Exception:
        pass
    return out


def _liq_pull(levels, mark: float, d0: float, reach: float):
    """Liquidity Pull Vector з рівнів liq-map.

    → (pull[-1..+1], target|None, mass). pull > 0 = тягне ВГОРУ (кластери зверху
    домінують → магніт угору → LONG), < 0 = вниз. target = найсильніший найближчий
    кластер (ймовірна ціль ходу)."""
    num = 0.0
    den = 0.0
    best = None
    for lev in levels or []:
        try:
            price = float(lev['price'])
            usd = float(lev.get('usd') or 0.0)
        except Exception:
            continue
        if usd <= 0 or not mark or mark <= 0:
            continue
        dist = abs(price - mark) / mark * 100.0
        if dist > reach:
            continue
        prox = math.exp(-dist / max(d0, 1e-6))
        wgt = usd * prox
        s = 1.0 if price > mark else -1.0
        num += s * wgt
        den += wgt
        if best is None or wgt > best[0]:
            best = (wgt, price, 'up' if s > 0 else 'down')
    if den <= 0:
        return 0.0, None, 0.0
    target = {'price': round(best[1], 10), 'side': best[2]} if best else None
    return max(-1.0, min(1.0, num / den)), target, den


def _whale_term(symbol: str) -> Optional[float]:
    """Агресивний потік крупного капіталу → [-1..+1] (buy важче = +). None коли
    whale-tape не веде саме цей символ (він односимвольний, зазвичай BTC)."""
    try:
        from detection.whale_tape import get_whale_tape
        wt = get_whale_tape()
        if not wt:
            return None
        st = wt.get_state(window_minutes=60)
        if not st or (st.get('symbol') or '').upper() != symbol.upper():
            return None
        stats = st.get('stats') or {}
        total = float(stats.get('total_volume') or 0.0)
        if total <= 0:
            return None
        nd = float(stats.get('net_delta') or 0.0)
        return max(-1.0, min(1.0, nd / total))
    except Exception:
        return None


def _book_term(symbol: str) -> Optional[float]:
    """Дисбаланс лімітів стакана → [-1..+1] (більше bid = +). Лише якщо символ уже
    активний у колекторі — НЕ підписуємо нові WS на кожну монету."""
    try:
        from detection.orderbook_collector import get_orderbook_collector
        ob = get_orderbook_collector()
        if not ob:
            return None
        try:
            active = {(a.get('symbol') or '').upper() for a in (ob.active_symbols() or [])}
        except Exception:
            active = set()
        if symbol.upper() not in active:
            return None
        snap = ob.request(symbol)          # активний → повертає наявний знімок
        if not snap:
            return None
        walls = ob.compute_walls(snap)
        if not walls:
            return None
        imb = walls.get('imbalance_pct')
        if imb is None:
            return None
        return max(-1.0, min(1.0, float(imb) / 100.0))
    except Exception:
        return None


def _pos_term(symbol: str, fscale: float) -> Optional[float]:
    """Позиціювання за funding → [-1..+1], КОНТР: сильно додатний funding =
    переповнені лонги = ймовірна ціль знизу = ведмежий нахил (−). TTL-кеш."""
    try:
        now = time.time()
        c = _funding_cache.get(symbol)
        if c and (now - c[0]) < _FUNDING_TTL:
            rate = c[1]
        else:
            from detection.exchange_router import get_funding_rate
            rate, _src = get_funding_rate(symbol)
            _funding_cache[symbol] = (now, rate)
        if rate is None:
            return None
        # rate у частках (напр. 0.0001 = 0.01%). Переводимо у % і контр-знак.
        pct = float(rate) * 100.0
        return max(-1.0, min(1.0, -pct / max(fscale, 1e-6)))
    except Exception:
        return None


def _dq_factor(dq) -> float:
    """data_quality → множник впевненості [0.4..1.0]."""
    try:
        conf = (dq or {}).get('confidence')
        return {'high': 1.0, 'medium': 0.7, 'low': 0.4}.get(conf, 0.5)
    except Exception:
        return 0.5


def compute_mm(db, symbol: str, liq_state: Optional[Dict] = None,
               with_confirmations: bool = True,
               with_funding: bool = False,
               momentum: Optional[float] = None) -> Optional[Dict]:
    """Головна функція. `liq_state` можна передати готовим (щоб не фетчити двічі).
    `with_confirmations=False` → лише LPV-ядро (дешево, для масового скану).
    `with_funding` — чи робити funding-запит (єдиний мережевий; вмикати лише для
    BTC / сфокусованої монети, щоб масовий скан не смикав біржу).
    `momentum` — знаковий імпульс ЦІНИ [-1..+1] (+ вгору): передає викликач (у
    нього є свічки). Магніт-модель контртрендова, тож імпульс КОРИГУЄ напрямок —
    при сильному русі проти магніту тренд ВЕДЕ (див. mm_trend_override).

    Return: {dir, status, strength, mark_price, target, components, weights,
             data_quality, trend_override} або None коли зовсім немає даних."""
    try:
        st = get_mm_settings(db)
        lst = liq_state
        if lst is None:
            from detection.liquidation_map.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if lm:
                try:
                    prof = db.get_setting('liqmap_decay_profile', 'tori')
                except Exception:
                    prof = 'tori'
                lst = lm.get_state(symbol, lookback_hours=24, profile=prof)
        mark = (lst or {}).get('mark_price')
        if not mark:
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                if md:
                    tk = md.get_ticker(symbol)
                    mark = tk.get('last') if tk else None
            except Exception:
                pass
        if not mark or mark <= 0:
            return None

        pull, target, mass = _liq_pull((lst or {}).get('levels') or [], mark,
                                       st['mm_prox_pct'], st['mm_reach_pct'])

        # Мікс: LIQMAP завжди (якщо є маса), решта — best-effort, ваги ренормуються.
        terms = []  # (name, weight, value)
        if mass > 0:
            terms.append(('liq', st['mm_liq_weight'], pull))
        comp = {'liq': round(pull, 3) if mass > 0 else None,
                'whale': None, 'book': None, 'pos': None}
        if with_confirmations:
            wf = _whale_term(symbol)
            if wf is not None:
                terms.append(('whale', st['mm_whale_weight'], wf))
                comp['whale'] = round(wf, 3)
            bi = _book_term(symbol)
            if bi is not None:
                terms.append(('book', st['mm_book_weight'], bi))
                comp['book'] = round(bi, 3)
            if with_funding:
                ps = _pos_term(symbol, st['mm_funding_scale'])
                if ps is not None:
                    terms.append(('pos', st['mm_pos_weight'], ps))
                    comp['pos'] = round(ps, 3)

        wsum = sum(w for _n, w, _v in terms)
        if wsum <= 0:
            return None
        score = sum(w * v for _n, w, v in terms) / wsum   # ренормовано → [-1..1]
        score = max(-1.0, min(1.0, score))
        liq_score = score   # чистий магніт (до реконсиляції з трендом) — для логу

        # 🧭 Реконсиляція з РУХОМ ціни. Магніт контртрендовий → сам показує, ДЕ
        # ліквідність, а не куди ЙДЕ ціна. При сильному русі ПРОТИ магніту тренд
        # веде напрямок (інакше «монета росте, а ММ = SHORT»); у спокійному ринку
        # — м'який блендинг, магніт лишається основою.
        trend_override = False
        if momentum is not None:
            m = max(-1.0, min(1.0, float(momentum)))
            comp['trend'] = round(m, 3)
            opposes = (m > 0) != (score > 0)
            if abs(m) >= st['mm_trend_override'] and opposes and abs(score) > 0.05:
                score = m                       # тренд ВЕДЕ напрямок
                trend_override = True
            else:
                tw = st['mm_trend_weight']
                score = (1.0 - tw) * score + tw * m
            score = max(-1.0, min(1.0, score))

        thr = st['mm_threshold']
        status = ('LONG' if score > thr else ('SHORT' if score < -thr else None))
        dqf = _dq_factor((lst or {}).get('data_quality'))
        strength = int(round(min(1.0, abs(score)) * 100 * dqf))

        return {
            'dir': round(score, 3),          # сумісно зі старим fuel_dir (після тренду)
            'status': status,
            'strength': strength,
            'mark_price': mark,
            'target': target,                # {price, side} — ціль-магніт
            'components': comp,               # внесок кожного сигналу [-1..1]
            'weights': {n: round(w, 3) for n, w, _v in terms},
            'data_quality': dqf,
            'liq_score': round(liq_score, 3),  # чистий магніт до реконсиляції (лог)
            'trend_override': trend_override,  # тренд переважив магніт
        }
    except Exception as e:
        try:
            print(f"[mm_model] compute_mm error {symbol}: {e}")
        except Exception:
            pass
        return None
