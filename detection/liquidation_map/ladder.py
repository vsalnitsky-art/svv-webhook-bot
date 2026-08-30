"""💧 ДРАБИНА ЛІКВІДНОСТІ — рівні ліквідації, згорнуті в кроки по $N.

Навіщо: у каналах регулярно публікують зріз виду «на $82k — 4 150 стопів,
на $78k — 2 011». Бот уже рахує ТЕ САМЕ (`liquidation_map`: приріст OI →
плечі 25×/50×/100× → ціни ліквідації → кошики), але віддає це дрібною сіткою
0.25% і в доларах. Тут ми згортаємо ту саму сітку у КРОКИ ПО $N і переводимо
у **ВІДСОТКИ** від усієї ліквідності у вікні.

⚠️ ЧОМУ ВІДСОТКИ, А НЕ «КІЛЬКІСТЬ СТОПІВ» (принципово, не міняти без потреби):
реальних стоп-наказів трейдерів не публікує ЖОДНА централізована біржа. Число
«4 150 стопів» — це чужа оцінка: долари, поділені на припущений середній розмір
позиції. Два припущення поспіль, які неможливо перевірити. Частка ж від усієї
ліквідності — величина, яку ми справді знаємо зі своїх даних, і саме вона
відповідає на питання «куди тягне ціну», не вигадуючи точних цифр.

Модуль — ЧИСТІ функції без I/O: дані дає `liquidation_map.get_state()`.
"""

from typing import Dict, List, Optional


DEFAULT_STEP_USD = 1000.0
DEFAULT_TOP_N = 6
# Рівні, дрібніші за це, у драбину не потрапляють — інакше список забивається
# шумом, у якому не видно справжніх кластерів.
MIN_ROW_PCT = 1.0


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if x == x and x not in (float('inf'), float('-inf')) else None
    except (TypeError, ValueError):
        return None


def step_for(mark_price, step_usd=None) -> float:
    """Крок драбини. Явно заданий виграє; інакше — $1000 на BTC-масштабі і
    пропорційно менший на дешевших монетах (щоб на ETH не вийшла одна сходинка
    на весь графік)."""
    s = _f(step_usd)
    if s and s > 0:
        return s
    p = _f(mark_price) or 0.0
    if p <= 0:
        return DEFAULT_STEP_USD
    # ~1.25% ціни, округлено до «людського» кроку.
    raw = p * 0.0125
    for nice in (1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
                 1000.0, 2500.0, 5000.0):
        if raw <= nice:
            return nice
    return 10000.0


def build_ladder(levels: List[Dict], mark_price, *, step_usd=None,
                 top_n: int = DEFAULT_TOP_N,
                 min_row_pct: float = MIN_ROW_PCT) -> Dict:
    """Згорнути рівні ліквідації у сходинки по $step і перевести у відсотки.

    `levels` — список із `liquidation_map.get_state()['levels']`:
        {price, usd, side ('long'|'short'), age_min}
    `usd` там уже ЗГАСЛИЙ за часом — беремо саме його, бо стара побудова OI
    здебільшого вже не стоїть (позиції закрились, стопи пересунули).

    Повертає:
      {ok, mark_price, step, total_usd,
       above: {pct, usd},        # ліквідність ВИЩЕ ціни (переважно шорти)
       below: {pct, usd},        # НИЖЧЕ ціни (переважно лонги)
       pull: 'up'|'down'|'flat', # куди переважує маса
       pull_pct,                 # наскільки переважує (різниця часток)
       rows: [...]}              # сходинки, найбільша частка першою

    Кожен рядок: {price, price_hi, side, usd, pct, dist_pct, dir}
      • `price`    — нижня межа сходинки (те, що показуємо як «$82 000»);
      • `pct`      — частка від УСІЄЇ ліквідності у вікні;
      • `dist_pct` — відстань від поточної ціни у % (завжди додатна);
      • `dir`      — 'up' | 'down' щодо ціни.
    """
    mp = _f(mark_price)
    if not mp or mp <= 0:
        return {'ok': False, 'reason': 'немає ціни', 'rows': [],
                'above': {'pct': 0.0, 'usd': 0.0},
                'below': {'pct': 0.0, 'usd': 0.0},
                'pull': 'flat', 'pull_pct': 0.0, 'total_usd': 0.0}

    step = step_for(mp, step_usd)
    buckets: Dict[tuple, Dict] = {}
    total = 0.0
    up_usd = down_usd = 0.0

    for lv in (levels or []):
        # Вхід може прийти з чужого джерела — приймаємо ЛИШЕ словники, решту
        # мовчки пропускаємо (падати на сміттєвому елементі не можна: банер
        # тоді просто зникне з екрана).
        if not isinstance(lv, dict):
            continue
        price = _f(lv.get('price'))
        usd = _f(lv.get('usd'))
        if price is None or usd is None or price <= 0 or usd <= 0:
            continue
        side = str(lv.get('side') or '').lower()
        if side not in ('long', 'short'):
            continue
        lo = (int(price / step)) * step
        key = (lo, side)
        rec = buckets.get(key)
        if rec is None:
            rec = {'price': lo, 'price_hi': lo + step, 'side': side, 'usd': 0.0}
            buckets[key] = rec
        rec['usd'] += usd
        total += usd
        if price >= mp:
            up_usd += usd
        else:
            down_usd += usd

    if total <= 0:
        return {'ok': False, 'reason': 'ліквідність ще не набрана', 'rows': [],
                'above': {'pct': 0.0, 'usd': 0.0},
                'below': {'pct': 0.0, 'usd': 0.0},
                'pull': 'flat', 'pull_pct': 0.0,
                'mark_price': mp, 'step': step, 'total_usd': 0.0}

    rows = []
    for rec in buckets.values():
        pct = rec['usd'] / total * 100.0
        if pct < min_row_pct:
            continue
        # Відстань міряємо від СЕРЕДИНИ сходинки — так «$82 000» при ціні
        # $80 100 чесно дає ~2.4%, а не 2.37% від випадкової межі.
        mid = rec['price'] + step / 2.0
        rows.append({
            'price': round(rec['price'], 8),
            'price_hi': round(rec['price_hi'], 8),
            'side': rec['side'],
            'usd': round(rec['usd'], 0),
            'pct': round(pct, 1),
            'dist_pct': round(abs(mid - mp) / mp * 100.0, 2),
            'dir': 'up' if mid >= mp else 'down',
        })
    rows.sort(key=lambda r: (-r['pct'], r['dist_pct']))
    rows = rows[:max(1, int(top_n or DEFAULT_TOP_N))]

    up_pct = round(up_usd / total * 100.0, 1)
    down_pct = round(100.0 - up_pct, 1)
    diff = round(up_pct - down_pct, 1)
    # «Рівновага» — коли перевага в межах 10 п.п.: менша різниця не є
    # напрямком, і видавати її за сигнал було б перебільшенням.
    pull = 'up' if diff > 10 else ('down' if diff < -10 else 'flat')

    return {'ok': True, 'mark_price': mp, 'step': step,
            'total_usd': round(total, 0),
            'above': {'pct': up_pct, 'usd': round(up_usd, 0)},
            'below': {'pct': down_pct, 'usd': round(down_usd, 0)},
            'pull': pull, 'pull_pct': abs(diff), 'rows': rows}
