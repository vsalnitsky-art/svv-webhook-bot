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
    # 💰 ЧИСТА БАБЛО-МОДЕЛЬ (без тренду): усе крутиться навколо ліквідності,
    # ліквідацій, стопів і позиціювання капіталу. Напрямок = куди тягне «бабло».
    'mm_liq_weight': 0.35,      # кластери ліквідацій/стопів (сторона + проксіміті)
    'mm_pos_weight': 0.38,      # позиціювання: funding + long/short (max-pain, контр)
    'mm_whale_weight': 0.15,    # реальний крупний капітал (whale-tape)
    'mm_book_weight': 0.12,     # ліміти стакана (MM)
    'mm_prox_pct': 2.0,         # D0 — масштаб проксіміті-загасання, %
    'mm_threshold': 0.15,       # |score| поріг для LONG/SHORT
    'mm_funding_scale': 0.05,   # |funding %|, що дає повний контр-сигнал
    'mm_reach_pct': 15.0,       # далі за цю відстань кластер уже не магніт
}

# TTL-кеші позиціювання, щоб не смикати біржу щоцикл на кожну монету.
_funding_cache: Dict[str, tuple] = {}   # symbol -> (ts, rate)
_ls_cache: Dict[str, tuple] = {}        # symbol -> (ts, ls_long_pct)
_POS_TTL = 300.0


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
        # Сторона кластера: 'short'-позиції ліквідуються, коли ціна РОСТЕ (магніт
        # угору, +), 'long' — коли ПАДАЄ (вниз, −). Фолбек на позицію відносно mark.
        sd = str(lev.get('side') or '').lower()
        if sd == 'short':
            s = 1.0
        elif sd == 'long':
            s = -1.0
        else:
            s = 1.0 if price > mark else -1.0
        num += s * wgt
        den += wgt
        if best is None or wgt > best[0]:
            best = (wgt, price, 'up' if s > 0 else 'down')
    if den <= 0:
        return 0.0, None, 0.0
    target = {'price': round(best[1], 10), 'side': best[2]} if best else None
    return max(-1.0, min(1.0, num / den)), target, den


def _runway(levels, mark: float, move_dir: Optional[str], reach: float,
            min_usd: float = 100_000.0) -> Optional[Dict]:
    """«Запас ходу» — скільки ще простору до значущої ліквідності ПОПЕРЕДУ руху.
    move_dir: 'LONG'(ціль зверху)/'SHORT'(ціль знизу). Логіка: ціна тягнеться до
    великих пулів ліквідації; найближчий великий кластер попереду — де рух може
    сповільнитись, головний (найбільший) — ймовірна кінцева ціль.
    → {room_pct, label, next:{price,dist_pct,usd}, main:{...}} або None."""
    if move_dir not in ('LONG', 'SHORT') or not mark or mark <= 0:
        return None
    ahead = []
    for lev in levels or []:
        try:
            price = float(lev['price'])
            usd = float(lev.get('usd') or 0.0)
        except Exception:
            continue
        if usd < min_usd:
            continue
        up = price > mark
        if (move_dir == 'LONG' and not up) or (move_dir == 'SHORT' and up):
            continue                      # не попереду руху
        dist = abs(price - mark) / mark * 100.0
        if dist <= 0 or dist > reach:
            continue
        ahead.append((dist, price, usd))
    if not ahead:
        return {'dir': move_dir, 'room_pct': None,
                'label': 'простір відкритий (немає великих цілей попереду)',
                'next': None, 'main': None}
    ahead.sort(key=lambda x: x[0])         # за відстанню
    nxt = ahead[0]
    main = max(ahead, key=lambda x: x[2])  # найбільший пул попереду = ймовірна ціль
    room = main[0]
    label = ('малий запас' if room < 0.7 else
             ('помірний запас' if room < 2.5 else 'великий запас'))
    return {
        'dir': move_dir,                   # у ЯКИЙ бік рахується запас (LONG/SHORT)
        'room_pct': round(room, 2),
        'label': label,
        'next': {'price': round(nxt[1], 10), 'dist_pct': round(nxt[0], 2), 'usd': round(nxt[2], 0)},
        'main': {'price': round(main[1], 10), 'dist_pct': round(main[0], 2), 'usd': round(main[2], 0)},
    }


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


def _funding_signed(symbol: str, fscale: float) -> Optional[float]:
    """funding → [-1..+1] КОНТР: додатний funding = переповнені ЛОНГИ (платять) =
    їх вичавлюють ВНИЗ (−); мінусовий = переповнені ШОРТИ = сквіз УГОРУ (+)."""
    try:
        now = time.time()
        c = _funding_cache.get(symbol)
        if c and (now - c[0]) < _POS_TTL:
            rate = c[1]
        else:
            from detection.exchange_router import get_funding_rate
            rate, _src = get_funding_rate(symbol)
            _funding_cache[symbol] = (now, rate)
        if rate is None:
            return None
        pct = float(rate) * 100.0
        return max(-1.0, min(1.0, -pct / max(fscale, 1e-6)))
    except Exception:
        return None


def _ls_signed(symbol: str) -> Optional[float]:
    """long/short-ratio → [-1..+1] КОНТР: натовп у ЛОНГАХ (ls_long>50) → ціль вниз
    (−); натовп у ШОРТАХ (<50) → сквіз угору (+). TTL-кеш."""
    try:
        now = time.time()
        c = _ls_cache.get(symbol)
        if c and (now - c[0]) < _POS_TTL:
            ls_long = c[1]
        else:
            from detection.market_data import get_market_data
            md = get_market_data()
            ls_long = None
            if md and hasattr(md, 'fetch_ls_ratio'):
                data, _src = md.fetch_ls_ratio(symbol)
                if data and data.get('ls_long') is not None:
                    ls_long = float(data['ls_long'])
            _ls_cache[symbol] = (now, ls_long)
        if ls_long is None:
            return None
        return max(-1.0, min(1.0, -(ls_long - 50.0) / 50.0))
    except Exception:
        return None


def _pos_term(symbol: str, fscale: float):
    """Позиціювання капіталу (max-pain, КОНТР): переповнений бік вичавлюють у
    протилежний. Комбінує funding + long/short-ratio. → (value[-1..1], detail)
    або (None, {}). detail — для логу/тултипа."""
    f = _funding_signed(symbol, fscale)
    ls = _ls_signed(symbol)
    vals = [v for v in (f, ls) if v is not None]
    if not vals:
        return None, {}
    return sum(vals) / len(vals), {'funding': f, 'ls': ls}


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
               momentum: Optional[float] = None,
               live_price: Optional[float] = None) -> Optional[Dict]:
    """ЧИСТА БАБЛО-МОДЕЛЬ напрямку ММ (без тренду). Напрямок визначає лише капітал:
    кластери ліквідацій/стопів (сторона+проксіміті) + позиціювання (funding+L/S,
    контр/max-pain) + whale-потік + ліміти стакана. `liq_state` можна передати
    готовим. `with_confirmations=False` → лише магніт-ядро (для масового скану).
    `with_funding`/`momentum` — застарілі, ігноруються (тренд прибрано з напрямку;
    момент лишається лише як контекст у SCORE, не тут).

    Return: {dir, status, strength, mark_price, target, components, weights,
             data_quality, runway} або None коли зовсім немає даних."""
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
        # Пріоритет ЖИВОЇ ціни: liq-map mark_price лагає на швидких рухах, і всі %
        # (запас, відстані) виходили б від застарілої точки. Fallback: liq-map → ticker.
        mark = live_price if (live_price and live_price > 0) else (lst or {}).get('mark_price')
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

        # 💰 Бабло-мікс: кластери ліквідацій + позиціювання (funding+L/S) + whale +
        # стакан. Ваги ренормуються за наявними сигналами. Тренд НЕ бере участі —
        # напрямок визначає лише «бабло». Позиціювання рахується для ВСІХ монет.
        terms = []  # (name, weight, value)
        if mass > 0:
            terms.append(('liq', st['mm_liq_weight'], pull))
        comp = {'liq': round(pull, 3) if mass > 0 else None,
                'pos': None, 'whale': None, 'book': None,
                'funding': None, 'ls': None}
        if with_confirmations:
            ps, pdet = _pos_term(symbol, st['mm_funding_scale'])
            if ps is not None:
                terms.append(('pos', st['mm_pos_weight'], ps))
                comp['pos'] = round(ps, 3)
                comp['funding'] = pdet.get('funding')
                comp['ls'] = pdet.get('ls')
            wf = _whale_term(symbol)
            if wf is not None:
                terms.append(('whale', st['mm_whale_weight'], wf))
                comp['whale'] = round(wf, 3)
            bi = _book_term(symbol)
            if bi is not None:
                terms.append(('book', st['mm_book_weight'], bi))
                comp['book'] = round(bi, 3)

        wsum = sum(w for _n, w, _v in terms)
        if wsum <= 0:
            return None
        score = sum(w * v for _n, w, v in terms) / wsum   # ренормовано → [-1..1]
        score = max(-1.0, min(1.0, score))

        thr = st['mm_threshold']
        status = ('LONG' if score > thr else ('SHORT' if score < -thr else None))
        dqf = _dq_factor((lst or {}).get('data_quality'))
        strength = int(round(min(1.0, abs(score)) * 100 * dqf))

        # 🎯 «Запас ходу» — відстань до ліквідності попереду руху, СТРОГО в бік
        # напрямку ММ. Якщо напрямку немає (⚪ рівновага) — запас невизначений (None),
        # бо «куди» немає сенсу.
        runway = _runway((lst or {}).get('levels') or [], mark, status,
                         st['mm_reach_pct']) if status in ('LONG', 'SHORT') else None

        return {
            'dir': round(score, 3),          # сумісно зі старим fuel_dir
            'status': status,
            'strength': strength,
            'mark_price': mark,
            'target': target,                # {price, side} — ціль-магніт
            'components': comp,               # внесок кожного сигналу [-1..1]
            'weights': {n: round(w, 3) for n, w, _v in terms},
            'data_quality': dqf,
            'runway': runway,                  # запас ходу до ліквідності попереду
        }
    except Exception as e:
        try:
            print(f"[mm_model] compute_mm error {symbol}: {e}")
        except Exception:
            pass
        return None
