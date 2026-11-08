"""📐 IMB — імбаланси (Fair Value Gap) на свічках (вимога 08.11).

Класичне визначення SMC на трьох свічках (бар i — третій):
    бичачий IMB:  low[i]  > high[i-2]  → розрив [high[i-2] .. low[i]]
    ведмежий IMB: high[i] < low[i-2]   → розрив [high[i] .. low[i-2]]
Середня свічка (i-1) — та, що «пробила» зону без зустрічної торгівлі.

ЗАПОВНЕННЯ (мітигація) — по наступних барах:
- бичачий: ціна повертається ВНИЗ у розрив → верх зони опускається до
  мінімуму бару; торкнулась низу (або нижче) → IMB закрито, не показуємо;
- ведмежий — дзеркально.
Тобто показуємо РЕШТУ незаповненого розриву, як TradingView-індикатори FVG.

ШУМ: розрив менший за `min_atr` × ATR(14) (деф. 0.10) — не імбаланс, а дрібне
тремтіння ціни; без цього порогу на 1m/5m графік засипало б смужками.

ОБСЯГ (вимога 09.11, переробка 08.11 «як у TradingView»):
- `vol` — обсяг ІМПУЛЬСНОЇ (середньої) свічки в МОНЕТАХ (як TV: «11.461K»
  BTC). Біржа віддає обіг у $ (`v`), тож обсяг у монетах = обіг / типова ціна
  бару (h+l+c)/3. Раніше тут стояла сума 3 свічок у $ → «3.64B» замість «K».
- `sell_pct` / `buy_pct` — розклад обсягу трьох свічок розриву за НАПРЯМКОМ
  свічки (червона → продаж, зелена → купівля, доджі — навпіл). Bybit не віддає
  taker buy/sell у свічках, тож це наближення; звірено з TV на BTC 1H 05.11:
  98/2 · 72/28 у TV проти 100/0 · 77/23 у нас (дані різних бірж).
- `pct` — частка БІЛЬШОЇ сторони (сумісність зі старим показом).

ЛИШЕ ЧИСТІ ФУНКЦІЇ, без I/O. Бар: {t (мс), o, h, l, p|c|close, v}.
"""
from typing import Dict, List

MIN_ATR = 0.10         # мінімальний розмір розриву, частка ATR(14)
ATR_LEN = 14
MAX_SHOW = 12          # скільки найновіших незаповнених віддаємо


def _close(b):
    return float(b.get('p', b.get('c', b.get('close'))))


def _vol(b):
    """Обсяг бару в МОНЕТАХ: обіг ($) / типова ціна. Немає даних → 0."""
    try:
        v = float(b.get('v', b.get('volume', 0)) or 0)
        if v <= 0:
            return 0.0
        tp = (float(b['h']) + float(b['l']) + _close(b)) / 3.0
        return v / tp if tp > 0 else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def _vol_view(bars, i):
    """(обсяг імпульсної свічки в монетах, % продажу, % купівлі) для розриву на барі i."""
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
                      max_show: int = MAX_SHOW, keep_filled: bool = False) -> List[Dict]:
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
                if l[i] <= z['bottom']:
                    z['filled'] = True
                elif l[i] < z['top']:
                    z['top'] = l[i]
            else:
                if h[i] >= z['top']:
                    z['filled'] = True
                elif h[i] > z['bottom']:
                    z['bottom'] = h[i]
        a = atr[i] or atr[i - 1]
        thr = (a or 0.0) * float(min_atr)
        if l[i] > h[i - 2] and (l[i] - h[i - 2]) > thr:
            vol, sp, bp = _vol_view(bars, i)
            live.append({'vol': vol, 'sell_pct': sp, 'buy_pct': bp,
                         'pct': (None if sp is None else max(sp, bp)), 'side': 'LONG', 'top': l[i], 'bottom': h[i - 2],
                         'top0': l[i], 'bottom0': h[i - 2],
                         't': t[i - 1], 't_form': t[i], 'filled': False})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) > thr:
            vol, sp, bp = _vol_view(bars, i)
            live.append({'vol': vol, 'sell_pct': sp, 'buy_pct': bp,
                         'pct': (None if sp is None else max(sp, bp)), 'side': 'SHORT', 'top': l[i - 2], 'bottom': h[i],
                         'top0': l[i - 2], 'bottom0': h[i],
                         't': t[i - 1], 't_form': t[i], 'filled': False})
    out = []
    for z in reversed(live):
        if z['filled'] and not keep_filled:
            continue
        full = z['top0'] - z['bottom0']
        rest = z['top'] - z['bottom']
        z['filled_pct'] = 100 if z['filled'] else (
            round((1 - rest / full) * 100) if full > 0 else 0)
        out.append(z)
        if len(out) >= int(max_show):
            break
    return out
