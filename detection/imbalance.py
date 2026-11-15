"""📐 IMB — імбаланси (Fair Value Gap) на свічках (вимога 08.11).

Класичне визначення SMC на трьох свічках (бар i — третій):
    бичачий IMB:  low[i]  > high[i-2]  → розрив [high[i-2] .. low[i]]
    ведмежий IMB: high[i] < low[i-2]   → розрив [high[i] .. low[i-2]]
Середня свічка (i-1) — та, що «пробила» зону без зустрічної торгівлі.

ЗАПОВНЕННЯ (мітигація) — КАЛІБРОВКА «ЯК У TRADINGVIEW» (15.11, BTC 1H 11-15.11):
- зона ЗНИКАЄ, лише коли бар ЗАКРИВАЄТЬСЯ за її дальньою межею (бичачий —
  close < низу, ведмежий — close > верху). Ґніт, що проколов зону, її НЕ
  знімає (кейс: розрив 69 110–69 806 проколото ґнотом до 70 188 — у TV зона
  лишилась, у нас зникала);
- зона малюється в ПОВНИЙ розмір (`top`/`bottom` = початкові межі), а не
  «залишок»; глибину проколу видно з `filled_pct` (рахується за ґнотами).
`mitigation='wick'` — стара поведінка (зрізання ґнотами), лишена для тестів.

ШУМ: розрив менший за `min_atr` × ATR(14) (деф. 0.15) — не імбаланс, а дрібне
тремтіння ціни; без цього порогу на 1m/5m графік засипало б смужками.

ОБСЯГ:
- `vol` — обсяг ІМПУЛЬСНОЇ (середньої) свічки в МОНЕТАХ (як TV: «8.952K»
  BTC; звірено — бот 8.951K / 9.678K проти TV 8.952K / 9.702K). Біржа віддає
  обіг у $ (`v`), тож обсяг у монетах = обіг / типова ціна (h+l+c)/3.
- `sell_pct` / `buy_pct` — розклад обсягу ІМПУЛЬСНОЇ свічки за положенням
  закриття в діапазоні бару: продаж = (high − close)/(high − low), купівля —
  решта (15.11). Раніше брався КОЛІР трьох свічок → на кожному русі «100% / 0%»
  і бірюзова смуга не малювалась узагалі. Біржа taker buy/sell у свічках не
  віддає, тож це наближення: на 8 зонах BTC 1H середнє відхилення від TV ~11 п.п.
  (колір 3 свічок — ~40 п.п.). `split='color'` — стара формула.
- `pct` — частка БІЛЬШОЇ сторони (сумісність зі старим показом).

НАКЛАДАННЯ (15.11): розрив, що перекриває ЖИВУ зону того самого боку, не
створюється (`skip_overlap`) — так TV не малює 64 500–65 079 поверх 63 437–64 711.

ЛИШЕ ЧИСТІ ФУНКЦІЇ, без I/O. Бар: {t (мс), o, h, l, p|c|close, v}.
"""
from typing import Dict, List

MIN_ATR = 0.15         # мінімальний розмір розриву, частка ATR(14) (15.11: 0.10→0.15 — як TV, розрив 0.11·ATR там не малюється)
ATR_LEN = 14
MAX_SHOW = 12          # скільки найновіших незаповнених віддаємо


def _close(b):
    return float(b.get('p', b.get('c', b.get('close'))))


def _vol(b):
    """Обсяг бару в МОНЕТАХ. Є поле біржі `q` (обсяг у монетах, 15.11) → саме
    воно (TV показує рівно його: «9.245K»); інакше обіг ($) / типова ціна.
    Немає даних → 0."""
    try:
        q = b.get('q')
        if q is not None and float(q) > 0:
            return float(q)
        v = float(b.get('v', b.get('volume', 0)) or 0)
        if v <= 0:
            return 0.0
        tp = (float(b['h']) + float(b['l']) + _close(b)) / 3.0
        return v / tp if tp > 0 else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def _vol_view(bars, i, split='clv'):
    """(обсяг імпульсної свічки в монетах, % продажу, % купівлі) для розриву на барі i."""
    if split == 'clv':
        m = bars[i - 1]
        mid = _vol(m)
        if mid <= 0:
            return None, None, None
        try:
            hh, ll, cc = float(m['h']), float(m['l']), _close(m)
        except (KeyError, TypeError, ValueError):
            return mid, None, None
        sp = 50 if hh <= ll else round((hh - cc) / (hh - ll) * 100)
        sp = max(0, min(100, sp))
        return mid, sp, 100 - sp
    sell = buy = 0.0
    for k in (i - 2, i - 1, i):
        v = _vol(bars[k])
        try:
            o = float(bars[k].get('o', bars[k].get('open')))
            c = _close(bars[k])
        except (TypeError, ValueError):
            o = c = 0.0
        if c < o:
            sell += v
        elif c > o:
            buy += v
        else:
            sell += v / 2; buy += v / 2
    tot = sell + buy
    mid = _vol(bars[i - 1])
    if tot <= 0:
        return (mid or None), None, None
    sp = round(sell / tot * 100)
    return (mid or None), sp, 100 - sp


def _atr(h, l, c, n=ATR_LEN):
    """RMA true range (як ta.atr у Pine). None, поки історії замало."""
    out = [None] * len(h)
    s = 0.0
    for i in range(len(h)):
        tr = h[i] - l[i] if i == 0 else max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        if i < n:
            s += tr
            if i == n - 1:
                out[i] = s / n
        else:
            out[i] = (out[i - 1] * (n - 1) + tr) / n
    return out


def detect_imbalances(bars: List[Dict], min_atr: float = MIN_ATR,
                      max_show: int = MAX_SHOW, keep_filled: bool = False,
                      mitigation: str = 'close', split: str = 'clv',
                      skip_overlap: bool = True) -> List[Dict]:
    """Незаповнені IMB (найновіший першим):
    [{side: LONG|SHORT, top, bottom, top0, bottom0, t (сек, середня свічка),
      t_form (сек, бар, що підтвердив розрив), filled_pct, filled,
      vol (обсяг імпульсної свічки, монети), sell_pct/buy_pct, pct}]."""
    n = len(bars or [])
    if n < 3:
        return []
    try:
        h = [float(b['h']) for b in bars]
        l = [float(b['l']) for b in bars]
        c = [_close(b) for b in bars]
        t = [int(float(b['t']) // 1000 if float(b['t']) > 1e12 else float(b['t'])) for b in bars]
    except (KeyError, TypeError, ValueError):
        return []
    atr = _atr(h, l, c)
    live: List[Dict] = []
    for i in range(2, n):
        # спершу — заповнення наявних цим баром
        for z in live:
            if z['filled']:
                continue
            if z['side'] == 'LONG':
                if mitigation == 'close' and c[i] < z['bottom0']:
                    z['filled'] = True
                elif mitigation != 'close' and l[i] <= z['bottom']:
                    z['filled'] = True
                elif l[i] < z['rt']:
                    z['rt'] = max(l[i], z['bottom0'])
            else:
                if mitigation == 'close' and c[i] > z['top0']:
                    z['filled'] = True
                elif mitigation != 'close' and h[i] >= z['top']:
                    z['filled'] = True
                elif h[i] > z['rb']:
                    z['rb'] = min(h[i], z['top0'])
            if mitigation != 'close' and not z['filled']:
                z['top'], z['bottom'] = z['rt'], z['rb']
        a = atr[i] or atr[i - 1]
        thr = (a or 0.0) * float(min_atr)

        def _overlaps(side, lo, hi):
            # 15.11 як у TV: новий розрив, що накладається на ЖИВУ зону того
            # самого боку, не створюється (кейс BTC 1H 15.11: 64 500–65 079
            # всередині живої 63 437–64 711 — у TV її немає).
            if not skip_overlap:
                return False
            return any((not z['filled']) and z['side'] == side
                       and lo < z['top0'] and hi > z['bottom0'] for z in live)
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) > thr and not _overlaps('LONG', h[i - 2], l[i]):
            vol, sp, bp = _vol_view(bars, i, split)
            live.append({'vol': vol, 'sell_pct': sp, 'buy_pct': bp,
                         'pct': (None if sp is None else max(sp, bp)), 'side': 'LONG', 'top': l[i], 'bottom': h[i - 2],
                         'top0': l[i], 'bottom0': h[i - 2], 'rt': l[i], 'rb': h[i - 2],
                         't': t[i - 1], 't_form': t[i], 'filled': False})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) > thr and not _overlaps('SHORT', h[i], l[i - 2]):
            vol, sp, bp = _vol_view(bars, i, split)
            live.append({'vol': vol, 'sell_pct': sp, 'buy_pct': bp,
                         'pct': (None if sp is None else max(sp, bp)), 'side': 'SHORT', 'top': l[i - 2], 'bottom': h[i],
                         'top0': l[i - 2], 'bottom0': h[i], 'rt': l[i - 2], 'rb': h[i],
                         't': t[i - 1], 't_form': t[i], 'filled': False})
    out = []
    for z in reversed(live):
        if z['filled'] and not keep_filled:
            continue
        full = z['top0'] - z['bottom0']
        rest = z['rt'] - z['rb']
        z['filled_pct'] = 100 if z['filled'] else (
            round((1 - rest / full) * 100) if full > 0 else 0)
        out.append(z)
        if len(out) >= int(max_show):
            break
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 📐 «PRO Volumatic FVG» — 1:1 З PINE-СКРИПТУ КОРИСТУВАЧА (15.11)
#    «Smart Money Concepts [PRO BOT] v3.1», блок PRO_FVG_GROUP. Дефолти:
#    FVG Size Threshold 10% · Mitigation 'close' · Max FVGs 10 · бокс до
#    bar_index + 25. Розклад обсягу — свічки МОЛОДШОГО TF (ceil(TF/10) хв):
#    бичачий обсяг = сума обсягів зелених підсвічок, ведмежий — червоних
#    (доджі не рахуються); немає даних → частка за положенням закриття
#    (той самий фолбек, що в Pine). Обсяг зони = бичачий + ведмежий.
# ═══════════════════════════════════════════════════════════════════════════
PINE_SIZE_THR = 10          # proFVGSizeThreshold, % від 1000-барного максимуму
PINE_MAX_COUNT = 10         # proFVGMaxCount
PINE_RANK_LEN = 1000        # ta.percentile_nearest_rank(diff, 1000, 100)
PINE_EXT_BARS = 25          # box right = bar_index + 25


def pine_sub_minutes(bar_seconds: float) -> int:
    """proFVGSubTF = max(1, ceil(chartMin / 10)) хвилин."""
    import math
    chart_min = max(1.0, float(bar_seconds) / 60.0)
    return max(1, int(math.ceil(chart_min / 10.0)))


def split_from_subbars(subbars) -> tuple:
    """(бичачий, ведмежий) обсяг: сума обсягів зелених / червоних підсвічок."""
    bull = bear = 0.0
    for b in subbars or []:
        try:
            o = float(b['o']); c = _close(b); v = float(b['v'])
        except (KeyError, TypeError, ValueError):
            continue
        if c > o:
            bull += v
        elif c < o:
            bear += v
    return bull, bear


def _body_split(bar) -> tuple:
    """Pine-фолбек: обсяг × (close−low)/(high−low)."""
    try:
        h, l, c = float(bar['h']), float(bar['l']), _close(bar)
    except (KeyError, TypeError, ValueError):
        return 0.0, 0.0
    r = (c - l) / (h - l) if h > l else 0.5
    v = _vol(bar)
    return v * r, v * (1 - r)


def detect_volfvg(bars: List[Dict], split=None, size_thr: float = PINE_SIZE_THR,
                  max_count: int = PINE_MAX_COUNT, mitigation: str = 'close',
                  rank_len: int = PINE_RANK_LEN) -> List[Dict]:
    """Покроковий прогін Pine по барах (останній може бути живим).
    split(t_ms) → (bull, bear) обсяг підсвічок середньої свічки або None.
    → живі зони, найновіша першою: {side, top, bottom, t, t_form, mid_t,
    vol, buy_pct, sell_pct, pct, approx}."""
    from collections import deque
    n = len(bars or [])
    if n < 3:
        return []
    try:
        h = [float(b['h']) for b in bars]
        l = [float(b['l']) for b in bars]
        c = [_close(b) for b in bars]
        tms = [int(float(b['t'])) for b in bars]
    except (KeyError, TypeError, ValueError):
        return []
    sec = [t // 1000 if t > 10 ** 12 else t for t in tms]
    win = deque()                 # (i, diff) — монотонна черга для ковзного максимуму
    zones: List[Dict] = []        # індекс 0 — найновіша (як unshift у Pine)
    for i in range(2, n):
        bull_d = (l[i] - h[i - 2]) / l[i] * 100 if l[i] else 0.0
        bear_d = (l[i - 2] - h[i]) / h[i] * 100 if h[i] else 0.0
        d = max(bull_d, bear_d)
        while win and win[-1][1] <= d:
            win.pop()
        win.append((i, d))
        while win and win[0][0] <= i - rank_len:
            win.popleft()
        rank = win[0][1]
        pass_bull = rank != 0 and bull_d / rank * 100 > size_thr
        pass_bear = rank != 0 and bear_d / rank * 100 > size_thr
        new = []
        if h[i - 2] < l[i] and h[i - 2] < h[i - 1] and l[i - 2] < l[i] and pass_bull:
            new.append(('LONG', l[i], h[i - 2]))
        if l[i - 2] > h[i] and l[i - 2] > l[i - 1] and h[i - 2] > h[i] and pass_bear:
            new.append(('SHORT', l[i - 2], h[i]))
        for side, top, bot in new:
            sp = split(tms[i - 1]) if split else None
            approx = sp is None
            bull, bear = (sp if sp else _body_split(bars[i - 1]))
            tot = bull + bear
            bp = int(bull / tot * 100) if tot > 0 else 50
            zones.insert(0, {'side': side, 'top': top, 'bottom': bot,
                             'top0': top, 'bottom0': bot,
                             't': sec[i - 1], 't_form': sec[i], 'mid_t': tms[i - 1],
                             'vol': tot or None, 'buy_pct': bp, 'sell_pct': 100 - bp,
                             'pct': max(bp, 100 - bp), 'approx': approx,
                             'filled': False, 'filled_pct': 0})
        # мітигація
        lo_src = c[i] if mitigation == 'close' else l[i]
        hi_src = c[i] if mitigation == 'close' else h[i]
        for k in range(len(zones) - 1, -1, -1):
            z = zones[k]
            if (z['side'] == 'LONG' and lo_src < z['bottom']) or \
               (z['side'] == 'SHORT' and hi_src > z['top']):
                zones.pop(k)
        # накладки — рівно цикл Pine (межі циклів рахуються на їх старті)
        if len(zones) > 1:
            for a in range(len(zones) - 1, -1, -1):
                if a >= len(zones):
                    continue
                A = zones[a]
                for b in range(len(zones) - 1, -1, -1):
                    if b >= len(zones) or a == b or a >= len(zones):
                        continue
                    B = zones[b]
                    if A['bottom'] < B['top'] < A['top']:
                        zones.pop(a)
                        break
        if len(zones) > max_count:
            zones.pop()
    return zones
