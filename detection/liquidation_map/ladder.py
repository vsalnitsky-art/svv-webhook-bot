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

import math
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
    # ~1.25% ціни, округлено до «людського» кроку 1/2/5 × 10^k.
    # ⚠️ Раніше тут був СПИСОК готових кроків від $1 — і для дешевих монет
    # він давав $1 на монету за $0.47, тобто ВСЯ драбина сходилась в одну-дві
    # сходинки, а магніт підписувався «$0». Відколи драбина рахується не лише
    # по BTC (скан біржі, режим однієї монети), крок мусить масштабуватись
    # униз так само вільно, як і вгору.
    raw = p * 0.0125
    if raw <= 0:
        return DEFAULT_STEP_USD
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if raw <= mag * m:
            return mag * m
    return mag * 10.0


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
    """$76 000 — пробіл замість коми: так число читається одразу.

    ⚠️ Дешеві монети округляти до цілого НЕ МОЖНА: драбина тепер рахується не
    лише по BTC (скан біржі, режим однієї монети), і `int(round(0.42))` дав би
    магніт «$0». Тому нижче $10 показуємо знаки після коми — рівно стільки,
    скільки треба, щоб рівень лишався розрізненним."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(f)
    if a >= 10:
        return f"${int(round(f)):,}".replace(',', ' ')
    dec = 3 if a >= 1 else (5 if a >= 0.01 else 7)
    return f"${f:.{dec}f}"


def magnet_edge(row: Dict, side: str) -> Optional[float]:
    """Ближня межа сходинки для напрямку угоди.

    Сходинка — це СМУГА `[price .. price_hi]`, і ціна доходить до неї з одного
    боку: LONG зустрічає НИЖНЮ межу, SHORT — ВЕРХНЮ. Виходимо на ПЕРШОМУ дотику
    до кластера, а не сподіваємось, що ціна прошиє його наскрізь.
    """
    if not isinstance(row, dict):
        return None
    lo = _f(row.get('price'))
    if lo is None:
        return None
    hi = _f(row.get('price_hi'))
    if hi is None:
        hi = lo
    return lo if str(side).upper() == 'LONG' else hi


def pick_magnet_ahead(rows: List[Dict], ref_price, side: str,
                      min_pct: float = 0.0) -> Optional[Dict]:
    """🧲 НАЙБІЛЬША сходинка ПОПЕРЕДУ `ref_price` у бік `side`.

    ⚠️ НАВІЩО ОКРЕМЕ ПРАВИЛО, а не «найбільша сходинка драбини».
    Драбина симетрична навколо ціни й описує ВЕСЬ ринок: найтовща сходинка
    цілком може лежати ПОЗАДУ входу. Для БАНЕРА це правильна відповідь
    («куди тягне ціну взагалі»), для ЦІЛІ УГОДИ — ні: ціль за визначенням
    попереду. Реальний кейс: маса зверху 66.2%, а найбільша ОКРЕМА сходинка
    (14.2%) — знизу. Глобальний магніт для LONG опинявся позаду, і угода в
    напрямку, який ліквідність саме ПІДТРИМУЄ, лишалась без магнітного TP-2.

    Порядок вибору той самий, що в самій драбині: більша частка виграє,
    тайбрейк — ближча до `ref_price` (спрацює першою).

    `min_pct` — підлога частки (0 = будь-яка значуща сходинка; у драбині вони
    вже відфільтровані за `MIN_ROW_PCT`).
    Повертає САМ рядок (сирі числа) або None.
    """
    ref = _f(ref_price)
    sd = str(side).upper()
    if ref is None or ref <= 0 or sd not in ('LONG', 'SHORT'):
        return None
    try:
        floor_pct = float(min_pct or 0.0)
    except (TypeError, ValueError):
        floor_pct = 0.0

    best = None
    best_key = None
    for row in (rows or []):
        edge = magnet_edge(row, sd)
        if edge is None or edge <= 0:
            continue
        # Сходинка, ВСЕРЕДИНІ якої стоїть вхід, ціллю не є: ми вже в кластері,
        # «перший дотик» відбувся. Тому порівнюємо саме ближню межу.
        if not ((edge > ref) if sd == 'LONG' else (edge < ref)):
            continue
        pct = _f(row.get('pct')) or 0.0
        if pct < floor_pct:
            continue
        key = (-pct, abs(edge - ref))
        if best_key is None or key < best_key:
            best, best_key = row, key
    return dict(best) if best else None


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
       rows: [...],              # сходинки ЗА ЦІНОЮ, згори вниз
                                 # (відбір — за часткою, порядок — за ціною)
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
            # Сходинка впритул до ціни (одна з її меж — сама ціна): UI ставить
            # її поруч із лінією ціни і не дублює число в підписі.
            'at_price': abs(rec['price'] - mp) < 1e-9
                        or abs(rec['price_hi'] - mp) < 1e-9,
        })
    # ВІДБІР — за значущістю (найбільші частки), а ПОРЯДОК ПОКАЗУ — за ЦІНОЮ,
    # згори вниз: так драбина читається як стакан, і видно, що лежить над
    # ціною, а що під нею. «Магніт» для вердикту беремо ДО пересортування —
    # інакше в нього потрапив би просто найвищий за ціною рядок.
    rows.sort(key=lambda r: (-r['pct'], r['dist_pct']))
    rows = rows[:max(1, int(top_n or DEFAULT_TOP_N))]
    top_row = rows[0] if rows else None
    rows.sort(key=lambda r: -r['price'])

    up_pct = round(up_usd / total * 100.0, 1)
    down_pct = round(100.0 - up_pct, 1)
    diff = round(up_pct - down_pct, 1)
    # «Рівновага» — коли перевага в межах 10 п.п.: менша різниця не є
    # напрямком, і видавати її за сигнал було б перебільшенням.
    pull = 'up' if diff > 10 else ('down' if diff < -10 else 'flat')

    return {'ok': True, 'mark_price': mp, 'step': step,
            'total_usd': round(total, 0),
            # 🧲 САМА СХОДИНКА-МАГНІТ, сирими числами. Вердикт форматує її для
            # показу (`$78 000`), а споживачам логіки потрібне ЧИСЛО — інакше
            # довелось би парсити підпис або повторювати правило вибору
            # («найбільша частка, тайбрейк — ближча») і ризикувати розійтись.
            'magnet_row': (dict(top_row) if top_row else None),
            'above': {'pct': up_pct, 'usd': round(up_usd, 0)},
            'below': {'pct': down_pct, 'usd': round(down_usd, 0)},
            'pull': pull, 'pull_pct': abs(diff), 'rows': rows,
            'verdict': make_verdict(pull, abs(diff), up_pct, down_pct,
                                    top_row)}
