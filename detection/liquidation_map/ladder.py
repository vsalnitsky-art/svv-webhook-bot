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


# Словесна сила перекосу. Пороги ті самі, що вирішують `pull`: до 10 п.п. —
# рівновага (не напрямок), далі — за зростанням.
def _strength_word(diff_pp: float) -> str:
    d = abs(diff_pp)
    if d <= 10:
        return 'рівновага'
    if d <= 30:
        return 'помірно'
    if d <= 60:
        return 'виразно'
    return 'сильно'


def _fmt_price_ua(v) -> str:
    """$76 000 — нерозривний пробіл замість коми: так число читається одразу."""
    try:
        return f"${int(round(float(v))):,}".replace(',', ' ')
    except (TypeError, ValueError):
        return str(v)


def make_verdict(pull: str, pull_pct: float, above_pct: float,
                 below_pct: float, top_row: Optional[Dict]) -> Dict:
    """ВЕРДИКТ по драбині — висновок, а не опис даних.

    Змінюється РАЗОМ із даними: напрямок бере з `pull`, силу — з перекосу
    часток, а «магніт» — з найбільшої сходинки (саме до неї ціну тягне
    найдужче).

    Повертає {text, tone, strength, parts}:
      • `text`  — суцільний рядок (фолбек, лог, Telegram);
      • `parts` — ті самі дані ПО ШМАТКАХ, щоб UI розфарбував ЧИСЛА окремо
        від тексту і виніс магніт у ДРУГИЙ рядок. Розмітку тут не робимо:
        модуль лишається чистим, а кольори — справа фронта.

    ⚠️ Вердикт не радить входити — він каже, ДЕ лежить маса. При рівновазі
    так і пишемо, а не витискаємо напрямок із 52/48.
    """
    word = _strength_word(pull_pct)
    magnet = None
    magnet_txt = ''
    if top_row:
        _pstr = _fmt_price_ua(top_row.get('price'))
        _arrow = '↑' if top_row.get('dir') == 'up' else '↓'
        magnet = {'price': _pstr,
                  'pct': f"{top_row.get('pct')}%",
                  'dist': f"{_arrow}{top_row.get('dist_pct')}%",
                  'dir': top_row.get('dir')}
        magnet_txt = (f" · найбільший магніт {_pstr} "
                      f"({magnet['pct']}, {magnet['dist']})")

    if pull == 'down':
        parts = {'icon': '▼', 'lead': 'Маса ліквідності НИЖЧЕ ціни',
                 'lead_val': f'{below_pct}%', 'action': 'тягне ВНИЗ',
                 'strength': word, 'skew': f'+{pull_pct} п.п.', 'magnet': magnet}
        return {'tone': 'down', 'strength': word, 'parts': parts,
                'text': f"▼ Маса ліквідності НИЖЧЕ ціни ({below_pct}%) — "
                        f"тягне ВНИЗ, {word} (+{pull_pct} п.п.){magnet_txt}"}
    if pull == 'up':
        parts = {'icon': '▲', 'lead': 'Маса ліквідності ВИЩЕ ціни',
                 'lead_val': f'{above_pct}%', 'action': 'тягне ВГОРУ',
                 'strength': word, 'skew': f'+{pull_pct} п.п.', 'magnet': magnet}
        return {'tone': 'up', 'strength': word, 'parts': parts,
                'text': f"▲ Маса ліквідності ВИЩЕ ціни ({above_pct}%) — "
                        f"тягне ВГОРУ, {word} (+{pull_pct} п.п.){magnet_txt}"}
    parts = {'icon': '⚖', 'lead': 'Рівновага',
             'lead_val': f'{above_pct}% зверху / {below_pct}% знизу',
             'action': 'вираженого перекосу немає',
             'strength': word, 'skew': '', 'magnet': magnet}
    return {'tone': 'flat', 'strength': word, 'parts': parts,
            'text': f"⚖ Рівновага: {above_pct}% зверху / {below_pct}% знизу — "
                    f"вираженого перекосу немає{magnet_txt}"}


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
       rows: [...],              # сходинки, найбільша частка першою
       verdict: {text, tone, strength, parts}}  # висновок по цих даних
                                 # `parts` — по шматках, для розфарбування в UI

    Кожен рядок: {price, price_hi, side, usd, pct, dist_pct, dir}
      • `price`    — нижня межа сходинки (те, що показуємо як «$82 000»);
      • `pct`      — частка від УСІЄЇ ліквідності у вікні;
      • `dist_pct` — відстань від поточної ціни у % (завжди додатна);
      • `dir`      — 'up' | 'down' щодо ціни.
    """
    _empty_verdict = {'tone': 'none', 'strength': 'немає даних',
                      'text': '— даних ще немає',
                      'parts': {'icon': '—', 'lead': 'Даних ще немає',
                                'lead_val': '', 'action': '', 'strength': '',
                                'skew': '', 'magnet': None}}
    mp = _f(mark_price)
    if not mp or mp <= 0:
        return {'ok': False, 'reason': 'немає ціни', 'rows': [],
                'above': {'pct': 0.0, 'usd': 0.0},
                'below': {'pct': 0.0, 'usd': 0.0},
                'pull': 'flat', 'pull_pct': 0.0, 'total_usd': 0.0,
                'verdict': _empty_verdict}

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
        hi = lo + step
        # ⚠️ СХОДИНКА, ЩО НАКРИВАЄ ПОТОЧНУ ЦІНУ, ДІЛИТЬСЯ САМЕ ПО НІЙ.
        # Без цього вона діставала ОДИН напрямок (за своєю серединою), хоча
        # половина її обсягу лежить по інший бік ціни. На проді це дало пряме
        # протиріччя: підсумок «⬆50.5% / ⬇49.5%» (рахується по РІВНЯХ) проти
        # рядків, які сумарно давали 32%/68% — бо 16% усієї ліквідності сиділо
        # в сходинці $78 000, підписаній «↓», при ціні $78 563 усередині неї.
        if lo <= mp < hi:
            if price < mp:
                hi = mp
            else:
                lo = mp
        # Ключ — САМЕ ДІАПАЗОН, без сторони: після поділу по ціні кожен рядок
        # і так однорідний (вище ціни — ліквідації шортів, нижче — лонгів), а
        # ключ зі стороною давав ДВА рядки з однаковим підписом «$78 000».
        key = (lo, hi)
        rec = buckets.get(key)
        if rec is None:
            rec = {'price': lo, 'price_hi': hi, 'usd': 0.0,
                   'by_side': {'long': 0.0, 'short': 0.0}}
            buckets[key] = rec
        rec['usd'] += usd
        rec['by_side'][side] += usd
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
                'mark_price': mp, 'step': step, 'total_usd': 0.0,
                'verdict': _empty_verdict}

    rows = []
    for rec in buckets.values():
        pct = rec['usd'] / total * 100.0
        if pct < min_row_pct:
            continue
        # Відстань міряємо від СЕРЕДИНИ сходинки — так «$82 000» при ціні
        # $80 100 чесно дає ~2.4%, а не 2.37% від випадкової межі. Для
        # розділеної сходинки середина рахується від її РЕАЛЬНИХ меж.
        mid = (rec['price'] + rec['price_hi']) / 2.0
        _bs = rec['by_side']
        rows.append({
            'price': round(rec['price'], 8),
            'price_hi': round(rec['price_hi'], 8),
            # Переважна сторона — для підказки. Після поділу по ціні вона
            # практично завжди одна (шорти зверху, лонги знизу).
            'side': 'long' if _bs['long'] >= _bs['short'] else 'short',
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
            'pull': pull, 'pull_pct': abs(diff), 'rows': rows,
            'verdict': make_verdict(pull, abs(diff), up_pct, down_pct,
                                    rows[0] if rows else None)}
