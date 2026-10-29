"""🌀 «ПРУЖИНА» — заряд монети перед імпульсом (вимога 29.10).

Дослівно: «Десь має бути алгоритм, який визначав по монеті, що вона готова до
"взриву", тобто заряджена пружина для хорошого старту… Ту стару технологію
задіювати не варто, а втіль вже новий алгоритм на його базі… Я хочу перед
відкриттям угод мати картину які монети мають дати хороший потенціал… і
відкривати саме по ним угоди в пріоритеті, а по монетам які не мають
потенціалу, не витрачати ресурси. Нехай чекають своєї черги.»

Стара технологія — `sleeper_scanner_v3` («5-Day Sleeper»: BB-стиснення + сухий
обʼєм + ріст OI + стакан, окремі запити до біржі на кожну монету, своя
черга станів IDLE→READY і HP). Ідею взято, реалізацію — ні:

• ДАНІ — лише ЗАКРИТІ 1h-бари, які SMC-сканер УЖЕ тримає для монет WATCHLIST
  (`_get_tf_data('1h')`, кеш до закриття бару). Жодного додаткового запиту.
• ЗАРЯД (0-100) — п'ять вимірів стиснення енергії, кожен 0..1:
    ttm  — TTM-стиснення: ширина BB(20, 2σ) у одиницях ATR (як `squeeze.py`);
    atr  — ATR% зараз проти власної історії (низький перцентиль = стиснуто);
    box  — коробка останніх 24 барів проти діапазону 96 барів (вузька = заряд);
    dry  — обʼєм висихає (12 барів проти попередніх 72);
    dur  — скільки барів поспіль триває стиснення (довше = сильніший вихід).
• НАПРЯМОК (bias −1…+1) — куди ймовірніше «вистрілить»:
    pos    — де ціна в коробці (біля верху → вгору);
    struct — нахил мінімумів/максимумів у коробці (висхідні мінімуми → вгору);
    trend  — EMA50 нахил + бік від EMA200 (контекст старшої хвилі);
    mom    — TTM-моментум (ціна проти рівноваги коробки).
• ВИСТРІЛ — останні 3 бари: бар із діапазоном > 1.8×ATR, закриттям ЗА межею
  коробки й обʼємом > 1.8× середнього → пружина вже розпрямилась (рух почався).
• ПОТЕНЦІАЛ за боком = заряд × частка напрямку:
    pot_long = charge × (1+bias)/2 · pot_short = charge × (1−bias)/2.
  Нейтральна заряджена монета має по половині заряду в кожен бік — вона
  вистрілить, але куди, ще невідомо.

ЧИСТІ функції, лише stdlib — модуль бере ізольований тест.
Бари — стандартні словники `market_data.fetch_klines`: {p, o, h, l, v, t}.
"""

import math
from typing import Dict, List, Optional, Tuple

TF = '1h'
LOOKBACK = 240          # 10 діб 1h — історія для перцентиля ATR і EMA200
MIN_BARS = 220          # менше — заряд не рахуємо (EMA200 + розігрів ATR)
BOX_BARS = 24           # «коробка» — останні 24 години
WIDE_BARS = 96          # ширший діапазон для порівняння (4 доби)
ATR_LEN = 14
BB_LEN = 20
TTM_OFF = 2.0           # ширина BB у одиницях ATR, вище — стиснення немає
TTM_FULL = 0.5          # нижче — стиснення максимальне
DUR_FULL = 24           # 24 бари поспіль у стисненні = повний бал тривалості

W_CHARGE = {'ttm': 0.30, 'atr': 0.25, 'box': 0.20, 'dry': 0.15, 'dur': 0.10}
W_BIAS = {'pos': 0.30, 'struct': 0.30, 'trend': 0.25, 'mom': 0.15}

CHARGED = 65            # ≥ — 🌀 заряджена
BUILDING = 45           # ≥ — ⏳ заряджається
DIR_MIN = 0.20          # |bias| ≥ — напрямок визначений

FIRE_RANGE_ATR = 1.8
FIRE_VOL_X = 1.8
FIRE_LOOK = 3

# 🎯 Ймовірний хід після вистрілу = «виміряний рух»: висота коробки,
# відкладена від її межі (класична проєкція стиснення). Підлога — ATR×N, бо
# надто вузька коробка дала б «хід» менший за звичайний шум бару.
MOVE_MIN_ATR = 1.5

MODES = ('off', 'priority', 'require')
STATE_LABEL = {'fired': '💥 вистрілила', 'charged': '🌀 заряджена',
               'building': '⏳ заряджається', 'flat': '😴 без заряду'}


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _f(bar: Dict, *keys) -> float:
    for k in keys:
        v = bar.get(k)
        if v not in (None, ''):
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _atr_series(h: List[float], l: List[float], c: List[float], n: int) -> List[Optional[float]]:
    """Wilder ATR по всій серії (як `ta.atr`); None до розігріву."""
    out: List[Optional[float]] = [None] * len(c)
    if len(c) < n + 1:
        return out
    trs = [h[0] - l[0]]
    for i in range(1, len(c)):
        trs.append(max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1])))
    atr = sum(trs[1:n + 1]) / n
    out[n] = atr
    for i in range(n + 1, len(c)):
        atr = (atr * (n - 1) + trs[i]) / n
        out[i] = atr
    return out


def _ema(vals: List[float], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(vals)
    if len(vals) < n:
        return out
    k = 2.0 / (n + 1)
    e = sum(vals[:n]) / n
    out[n - 1] = e
    for i in range(n, len(vals)):
        e = vals[i] * k + e * (1 - k)
        out[i] = e
    return out


def _unit_r(c: List[float], atr: List[Optional[float]], i: int) -> Optional[float]:
    """Ширина BB(20, 2σ) у одиницях ATR на барі i: bbw / (2·ATR)."""
    if i + 1 < BB_LEN or atr[i] is None or atr[i] <= 0:
        return None
    w = c[i + 1 - BB_LEN:i + 1]
    m = sum(w) / BB_LEN
    sd = math.sqrt(sum((x - m) ** 2 for x in w) / BB_LEN)
    return (4.0 * sd) / (2.0 * atr[i])


def _slope(vals: List[float]) -> float:
    """Нахил лінійної регресії на бар."""
    n = len(vals)
    if n < 2:
        return 0.0
    mx = (n - 1) / 2.0
    my = sum(vals) / n
    num = sum((i - mx) * (v - my) for i, v in enumerate(vals))
    den = sum((i - mx) ** 2 for i in range(n))
    return num / den if den else 0.0


def analyze(klines: List[Dict]) -> Dict:
    """Заряд + напрямок + вистріл по ЗАКРИТИХ барах (найстаріший першим)."""
    if not klines or len(klines) < MIN_BARS:
        return {'ok': False, 'reason': f'замало барів ({len(klines or [])} < {MIN_BARS})'}
    kl = klines[-LOOKBACK:]
    c = [_f(b, 'p', 'c', 'close') for b in kl]
    h = [_f(b, 'h', 'high') for b in kl]
    lo = [_f(b, 'l', 'low') for b in kl]
    v = [_f(b, 'v', 'volume') for b in kl]
    if not all(c) or not all(h) or not all(lo):
        return {'ok': False, 'reason': 'нульові ціни в барах'}
    n = len(c)
    last = n - 1
    atr = _atr_series(h, lo, c, ATR_LEN)
    a_now = atr[last]
    if not a_now or a_now <= 0:
        return {'ok': False, 'reason': 'ATR не порахувався'}
    price = c[last]

    # ── ЗАРЯД ──────────────────────────────────────────────────────────────
    ur = _unit_r(c, atr, last)
    ttm = _clamp((TTM_OFF - ur) / (TTM_OFF - TTM_FULL)) if ur is not None else 0.0
    band = ('off' if ur is None or ur >= TTM_OFF else
            'high' if ur < 1.0 else 'mid' if ur < 1.5 else 'low')

    atrp = [atr[i] / c[i] for i in range(max(ATR_LEN, n - 120), n) if atr[i]]
    cur = a_now / price
    rank = sum(1 for x in atrp if x <= cur) / len(atrp) if atrp else 0.5
    atr_c = _clamp(1.0 - rank)

    box_h = max(h[-BOX_BARS:])
    box_l = min(lo[-BOX_BARS:])
    wide = max(h[-WIDE_BARS:]) - min(lo[-WIDE_BARS:])
    box_w = box_h - box_l
    ratio = box_w / wide if wide > 0 else 1.0
    box_c = _clamp((0.6 - ratio) / 0.45)
    box_pct = box_w / price * 100.0

    v_now = sum(v[-12:]) / 12.0
    base = v[-84:-12]
    v_base = sum(base) / len(base) if base else 0.0
    vol_ratio = v_now / v_base if v_base > 0 else 1.0
    dry = _clamp((1.1 - vol_ratio) / 0.6)

    sq_bars = 0
    for back in range(0, 72):
        r = _unit_r(c, atr, last - back)
        if r is None or r >= TTM_OFF:
            break
        sq_bars += 1
    dur = _clamp(sq_bars / DUR_FULL)

    parts_c = {'ttm': ttm, 'atr': atr_c, 'box': box_c, 'dry': dry, 'dur': dur}
    charge = 100.0 * sum(W_CHARGE[k] * parts_c[k] for k in W_CHARGE)

    # ── НАПРЯМОК ───────────────────────────────────────────────────────────
    pos = ((price - box_l) / box_w) if box_w > 0 else 0.5
    pos_t = _clamp((pos - 0.5) * 2.0, -1.0, 1.0)
    sl_h = _slope(h[-BOX_BARS:]) * BOX_BARS / a_now
    sl_l = _slope(lo[-BOX_BARS:]) * BOX_BARS / a_now
    struct_t = _clamp((sl_h + sl_l) / 4.0, -1.0, 1.0)
    e50 = _ema(c, 50)
    e200 = _ema(c, 200)
    tr_t = 0.0
    if e50[last] is not None and e50[last - 10] is not None:
        tr_t += 0.6 * _clamp(((e50[last] - e50[last - 10]) / a_now) / 3.0, -1.0, 1.0)
    if e200[last] is not None:
        tr_t += 0.4 if price > e200[last] else -0.4
    mid = ((max(h[-BB_LEN:]) + min(lo[-BB_LEN:])) / 2.0 + sum(c[-BB_LEN:]) / BB_LEN) / 2.0
    mom_t = _clamp((price - mid) / a_now, -1.0, 1.0)
    parts_b = {'pos': pos_t, 'struct': struct_t, 'trend': tr_t, 'mom': mom_t}
    bias = sum(W_BIAS[k] * parts_b[k] for k in W_BIAS)

    # ── ВИСТРІЛ ────────────────────────────────────────────────────────────
    fired = None
    for back in range(0, FIRE_LOOK):
        i = last - back
        j0 = i - BOX_BARS
        if j0 < 1 or atr[i - 1] is None:
            break
        bh = max(h[j0:i])
        bl = min(lo[j0:i])
        bv = v[max(0, i - 84):i - 12] or v[max(0, i - 24):i]
        vb = sum(bv) / len(bv) if bv else 0.0
        rng = h[i] - lo[i]
        big = rng > FIRE_RANGE_ATR * atr[i - 1] and vb > 0 and v[i] > FIRE_VOL_X * vb
        if big and c[i] > bh:
            fired = {'side': 'LONG', 'bars_ago': back, 'level': bh}
            break
        if big and c[i] < bl:
            fired = {'side': 'SHORT', 'bars_ago': back, 'level': bl}
            break

    if fired:
        bias = 1.0 if fired['side'] == 'LONG' else -1.0
        state = 'fired'
        p = max(0.0, 60.0 - 12.0 * fired['bars_ago'])
        pot_long = p if fired['side'] == 'LONG' else 0.0
        pot_short = p if fired['side'] == 'SHORT' else 0.0
    else:
        state = ('charged' if charge >= CHARGED else
                 'building' if charge >= BUILDING else 'flat')
        pot_long = charge * (1.0 + bias) / 2.0
        pot_short = charge * (1.0 - bias) / 2.0

    d = 'LONG' if bias >= DIR_MIN else ('SHORT' if bias <= -DIR_MIN else None)
    proj = projection(price, box_h, box_l, a_now, pot_long, pot_short, fired)
    t = kl[last].get('t')
    try:
        t = int(float(t))
        t = t // 1000 if t > 10 ** 12 else t
    except (TypeError, ValueError):
        t = None
    return {
        'ok': True, 'state': state, 'label': STATE_LABEL[state],
        'charge': round(charge, 1), 'bias': round(bias, 3), 'dir': d,
        'pot_long': round(pot_long, 1), 'pot_short': round(pot_short, 1),
        'squeeze_bars': sq_bars, 'band': band,
        'unit_r': round(ur, 2) if ur is not None else None,
        'box_pct': round(box_pct, 2), 'box_hi': box_h, 'box_lo': box_l,
        'vol_ratio': round(vol_ratio, 2), 'atr_rank': round(rank * 100.0, 1),
        'fired': fired, 'price': price, 'bar_t': t, 'atr': a_now,
        **proj,
        'parts': {k: round(x, 3) for k, x in {**parts_c, **parts_b}.items()},
    }


def projection(price: float, box_h: float, box_l: float, atr: float,
               pot_long: float, pot_short: float, fired: Optional[Dict] = None) -> Dict:
    """🎯 Куди і наскільки найсильніше накручено пружину. ЧИСТА.

    • `lean` — бік, у який накручено сильніше (LONG / SHORT; рівно → None);
    • `lean_pp` — перевага натягу в пунктах (|pot_long − pot_short|);
    • `prob_long` — частка натягу вгору, % (pot_long / (pot_long + pot_short));
      нуль заряду → 50 (сторону не вигадуємо);
    • `move_long_pct` / `move_short_pct` — ймовірний хід від ПОТОЧНОЇ ціни до
      цілі «виміряного руху» (`tgt_long` = верх коробки + висота, `tgt_short`
      = низ − висота); висота не менша за MOVE_MIN_ATR × ATR. Після вистрілу
      ціль відкладається від рівня пробою.
    """
    w = max(float(box_h) - float(box_l), MOVE_MIN_ATR * float(atr or 0.0))
    hi, lo = float(box_h), float(box_l)
    if fired:
        lvl = float(fired.get('level') or (hi if fired.get('side') == 'LONG' else lo))
        if fired.get('side') == 'LONG':
            hi = lvl
        else:
            lo = lvl
    tl, ts = hi + w, max(lo - w, 0.0)
    pl, ps = float(pot_long or 0.0), float(pot_short or 0.0)
    tot = pl + ps
    lean = 'LONG' if pl > ps else ('SHORT' if ps > pl else None)
    p = float(price or 0.0)
    return {
        'lean': lean, 'lean_pp': round(abs(pl - ps), 1),
        'prob_long': round(100.0 * pl / tot, 1) if tot > 0 else 50.0,
        'tgt_long': tl, 'tgt_short': ts,
        'move_long_pct': round((tl - p) / p * 100.0, 2) if p > 0 else None,
        'move_short_pct': round((p - ts) / p * 100.0, 2) if p > 0 else None,
    }


def aggregate(recs: Dict[str, Dict], min_charge: float = BUILDING) -> Dict:
    """⚖ Куди накручено WATCHLIST загалом. ЧИСТА.

    Сума натягу LONG проти SHORT по монетах із зарядом ≥ `min_charge` (або
    вистрілених): плоска монета нічого не накручує і не має розмивати картину.
    `prob_long` — частка суми натягу вгору; `top_long`/`top_short` —
    найсильніше накручена монета кожного боку (з ймовірним ходом).
    """
    sl = ss = 0.0
    n_l = n_s = n = 0
    top = {'LONG': None, 'SHORT': None}
    for sym, r in (recs or {}).items():
        if not isinstance(r, dict) or not r.get('ok', True):
            continue
        if r.get('state') != 'fired' and float(r.get('charge') or 0.0) < min_charge:
            continue
        pl, ps = float(r.get('pot_long') or 0.0), float(r.get('pot_short') or 0.0)
        n += 1
        sl += pl
        ss += ps
        if pl > ps:
            n_l += 1
        elif ps > pl:
            n_s += 1
        for side, v, mv in (('LONG', pl, r.get('move_long_pct')), ('SHORT', ps, r.get('move_short_pct'))):
            if v > 0 and (top[side] is None or v > top[side]['pot']):
                top[side] = {'symbol': sym, 'pot': round(v, 1), 'move_pct': mv,
                             'state': r.get('state')}
    tot = sl + ss
    pl_ = round(100.0 * sl / tot, 1) if tot > 0 else 50.0
    lean = 'LONG' if pl_ - 50 > 5 else ('SHORT' if 50 - pl_ > 5 else None)
    return {'coins': n, 'n_long': n_l, 'n_short': n_s,
            'sum_long': round(sl, 1), 'sum_short': round(ss, 1),
            'prob_long': pl_, 'lean': lean,
            'top_long': top['LONG'], 'top_short': top['SHORT']}


def potential(rec: Optional[Dict], side: str) -> Optional[float]:
    """Потенціал монети в бік угоди (0-100) або None («невідомо»)."""
    if not isinstance(rec, dict) or not rec.get('ok'):
        return None
    sd = str(side or '').upper()
    if sd == 'LONG':
        return float(rec.get('pot_long') or 0.0)
    if sd == 'SHORT':
        return float(rec.get('pot_short') or 0.0)
    return None


def gate(rec: Optional[Dict], side: str, mode: str, min_pot: float) -> Tuple[bool, str]:
    """🚦 Чи відкривати угоду за потенціалом. ЧИСТА.

    • 'off'/'priority' — не блокує (priority лише міняє ЧЕРГОВІСТЬ);
    • 'require' — потенціал у бік угоди < `min_pot` → монета ЧЕКАЄ в черзі.
    ⚠️ Немає даних → НЕ блокуємо: «невідомо» ≠ «без потенціалу» (монета щойно
    в watchlist, сканер ще не дійшов) — інакше рестарт зупиняв би торгівлю.
    """
    if mode != 'require':
        return True, ''
    p = potential(rec, side)
    if p is None:
        return True, '🌀 даних «Пружини» ще немає — не блокуємо'
    try:
        need = float(min_pot or 0.0)
    except (TypeError, ValueError):
        need = 0.0
    if p >= need:
        return True, f'🌀 потенціал {side} {p:.0f} ≥ {need:.0f}'
    return False, (f'🌀 потенціал {side} {p:.0f} < {need:.0f} '
                   f'({rec.get("label", "")}, заряд {rec.get("charge", 0):.0f}) — чекає в черзі')


def priority_key(rec: Optional[Dict], side: str) -> float:
    """Ключ сортування черги: вищий потенціал — першим; невідомо — посередині
    (50), щоб монета без даних не опинялась ні першою, ні останньою."""
    p = potential(rec, side)
    return 50.0 if p is None else p
