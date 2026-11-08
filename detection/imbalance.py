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

ОБСЯГ (вимога 09.11, вигляд «як на скріні»: великий обсяг + %):
- `vol` — сумарний обсяг трьох свічок, що утворили розрив;
- `pct` — частка обсягу ІМПУЛЬСНОЇ (середньої) свічки в цій сумі: наскільки
  розрив «пробитий обсягом», а не тонким ринком. Немає обсягу → None.

ЛИШЕ ЧИСТІ ФУНКЦІЇ, без I/O. Бар: {t (мс), o, h, l, p|c|close, v}.
"""
from typing import Dict, List

MIN_ATR = 0.10         # мінімальний розмір розриву, частка ATR(14)
ATR_LEN = 14
MAX_SHOW = 12          # скільки найновіших незаповнених віддаємо


def _close(b):
    return float(b.get('p', b.get('c', b.get('close'))))


def _vol(b):
    try:
        v = float(b.get('v', b.get('volume', 0)) or 0)
        return v if v > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _vol_view(bars, i):
    """(сумарний обсяг 3 свічок, % імпульсної) для розриву на барі i."""
    v = [_vol(bars[i - 2]), _vol(bars[i - 1]), _vol(bars[i])]
    tot = sum(v)
    if tot <= 0:
        return None, None
    return tot, round(v[1] / tot * 100)


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
      vol (обсяг 3 свічок), pct (% обсягу імпульсної свічки)}]."""
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
            vol, pct = _vol_view(bars, i)
            live.append({'vol': vol, 'pct': pct, 'side': 'LONG', 'top': l[i], 'bottom': h[i - 2],
                         'top0': l[i], 'bottom0': h[i - 2],
                         't': t[i - 1], 't_form': t[i], 'filled': False})
        elif h[i] < l[i - 2] and (l[i - 2] - h[i]) > thr:
            vol, pct = _vol_view(bars, i)
            live.append({'vol': vol, 'pct': pct, 'side': 'SHORT', 'top': l[i - 2], 'bottom': h[i],
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
