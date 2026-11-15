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
