"""🎯 АВТОПІЛОТ УГОДИ — супровід відкритої позиції по ГРАФІКУ.

Мета (вимога користувача): «максимально вигідно тримати угоду і чітко вийти при
максимальному результаті». Тобто дві протилежні задачі одночасно:
  • НЕ вийти зарано, поки структура ще працює на нас;
  • НЕ віддати назад набраний прибуток і зафіксувати його там, де рух реально
    має зупинитись.

Відповідь — не таймер і не фіксований відсоток, а РЕАЛЬНІ ОБ'ЄКТИ ГРАФІКА, які
бот уже вміє рахувати:

  🎯 ЦІЛЬ (куди тягне ціну)
     • пули ліквідності попереду руху (`runway` з liq-map: `next` — найближчий
       великий кластер, де рух може сповільнитись; `main` — найбільший, ймовірна
       кінцева ціль);
     • екстремум ДИЛІНГ-ДІАПАЗОНУ попереду (swing high для LONG / low для SHORT)
       — той самий `_swing_trailing_range`, що живить PD-зону й лінії на графіку;
     • POC — ціна тяжіє до нього, тож він теж є магнітом/перепоною.

  🛡 СТОП (де структура ламається)
     • протилежний екстремум дилінг-діапазону + буфер — класичний структурний
       стоп: поки він цілий, рух у наш бік не спростовано;
     • РАТЧЕТ: стоп рухається ЛИШЕ в бік прибутку, ніколи назад.

Увесь модуль — ЧИСТІ функції без I/O: дані збирає викликач (TradeManager), тут
лише рішення. Це робить логіку повністю тестованою і не додає навантаження на
жоден мережевий шлях.
"""
from typing import Dict, List, Optional


# Дефолти автопілота. Свідомо консервативні: беремо ЗМІСТОВНІ цілі (не шум) і
# не чіпаємо стоп, поки він не покращується.
DEFAULTS = {
    'min_target_dist_pct': 0.30,   # ближче — це шум, не ціль
    'max_target_dist_pct': 15.0,   # далі — вже не «ціль цієї угоди»
    'target_tol_pct': 0.05,        # допуск «ціну досягнуто»
    'stop_buf_pct': 0.10,          # буфер за структурним рівнем
    'be_at_r': 1.0,                # з якого R підтягувати стоп щонайменше в БЗ
    'be_lock_pct': 0.05,           # скільки лишати «в плюсі» на беззбитку
}


def _f(v):
    try:
        x = float(v)
        return x if x == x and x not in (float('inf'), float('-inf')) else None
    except (TypeError, ValueError):
        return None


def _ahead(side: str, price: float, level: float) -> bool:
    """Рівень ПОПЕРЕДУ руху: для LONG — вище ціни, для SHORT — нижче."""
    return (level > price) if side == 'LONG' else (level < price)


def collect_targets(side: str, price: float, *, swing: Optional[Dict] = None,
                    runway: Optional[Dict] = None, poc=None,
                    cfg: Optional[Dict] = None) -> List[Dict]:
    """Усі ЗМІСТОВНІ об'єкти графіка ПОПЕРЕДУ руху, від найближчого до дальшого.

    Кожен елемент: {price, dist_pct, kind, label}. `kind` — щоб рішення можна
    було пояснити людською мовою, а не «бот вирішив»."""
    c = {**DEFAULTS, **(cfg or {})}
    price = _f(price) or 0.0
    if side not in ('LONG', 'SHORT') or price <= 0:
        return []
    lo = float(c['min_target_dist_pct'])
    hi = float(c['max_target_dist_pct'])
    out = []

    def _add(lvl, kind, label):
        lvl = _f(lvl)
        if lvl is None or lvl <= 0 or not _ahead(side, price, lvl):
            return
        d = abs(lvl - price) / price * 100.0
        if d < lo or d > hi:
            return
        out.append({'price': lvl, 'dist_pct': round(d, 3),
                    'kind': kind, 'label': label})

    # 1) Пули ліквідації попереду — головний магніт ціни.
    rw = runway or {}
    _nx = rw.get('next') or {}
    _mn = rw.get('main') or {}
    _add(_nx.get('price'), 'liq_next', 'найближчий пул ліквідності')
    _add(_mn.get('price'), 'liq_main', 'головний пул ліквідності')

    # 2) Екстремум дилінг-діапазону попереду руху (той самий, що на графіку).
    sw = swing or {}
    _sw_ahead = (sw.get('high') or {}) if side == 'LONG' else (sw.get('low') or {})
    _add(_sw_ahead.get('price'), 'swing',
         _sw_ahead.get('label') or 'екстремум дилінг-діапазону')

    # 3) POC — ціна тяжіє до нього; попереду руху він і магніт, і перепона.
    _add(poc, 'poc', 'POC (максимум обсягу)')

    out.sort(key=lambda x: x['dist_pct'])
    return out


def pick_objective(targets: List[Dict]) -> Optional[Dict]:
    """ГОЛОВНА ціль угоди — найдальший зі змістовних об'єктів попереду.

    Саме «найдальший», а не «найближчий»: завдання — витиснути з руху максимум.
    Найближчий об'єкт лишається як `next_obstacle` (місце ймовірної зупинки), і
    використовується для підтягування стопа, а не для виходу."""
    return targets[-1] if targets else None


def structure_stop(side: str, price: float, swing: Optional[Dict],
                   cfg: Optional[Dict] = None) -> Optional[Dict]:
    """🛡 Структурний стоп: ЗА протилежним екстремумом дилінг-діапазону + буфер.

    LONG  → під swing-low  (поки low цілий, висхідна структура жива)
    SHORT → над swing-high
    Повертає {price, label} або None. Рівень з НЕПРАВИЛЬНОГО боку від ціни
    (структура вже зламана) не повертається — вигадувати стоп не можна."""
    c = {**DEFAULTS, **(cfg or {})}
    price = _f(price) or 0.0
    sw = swing or {}
    if side not in ('LONG', 'SHORT') or price <= 0:
        return None
    src = (sw.get('low') or {}) if side == 'LONG' else (sw.get('high') or {})
    lvl = _f(src.get('price'))
    if lvl is None or lvl <= 0:
        return None
    buf = max(0.0, float(c['stop_buf_pct'])) / 100.0
    out = lvl * (1.0 - buf) if side == 'LONG' else lvl * (1.0 + buf)
    # Стоп мусить бути з ЗАХИСНОГО боку від ціни, інакше він спрацює миттєво.
    if (side == 'LONG' and out >= price) or (side == 'SHORT' and out <= price):
        return None
    return {'price': out,
            'label': f"структура: {src.get('label') or 'екстремум діапазону'}"}


def better_stop(side: str, new_stop, prev_stop) -> bool:
    """РАТЧЕТ: стоп рухається ЛИШЕ в бік прибутку. Ніколи не послаблюємо —
    інакше «супровід» перетворюється на відсування стопа під збиток."""
    n, p = _f(new_stop), _f(prev_stop)
    if n is None:
        return False
    if p is None:
        return True
    return (n > p) if side == 'LONG' else (n < p)


def breakeven_stop(side: str, entry: float, risk_pct: float,
                   cfg: Optional[Dict] = None) -> Optional[float]:
    """Рівень беззбитку з невеликим замком у плюс (щоб комісія не з'їла нуль)."""
    c = {**DEFAULTS, **(cfg or {})}
    e = _f(entry)
    if e is None or e <= 0:
        return None
    lock = max(0.0, float(c['be_lock_pct'])) / 100.0
    return e * (1.0 + lock) if side == 'LONG' else e * (1.0 - lock)


def plan(side: str, entry, price, *, swing=None, runway=None, poc=None,
         prev_stop=None, objective_lock: Optional[Dict] = None,
         cfg: Optional[Dict] = None) -> Dict:
    """Рішення автопілота на поточний момент.

    Повертає:
      {'action': 'hold' | 'trail' | 'take',
       'stop': float|None,           # новий рівень стопа (лише коли 'trail')
       'objective': dict|None,       # головна ціль угоди
       'next_obstacle': dict|None,   # найближча перепона попереду
       'targets': [...],             # усі змістовні об'єкти попереду
       'reasons': [str, ...]}        # ЛЮДСЬКОЮ мовою, чому саме так

    Логіка:
      1. Ціль ДОСЯГНУТО (ціна дійшла до головного об'єкта) → `take`: далі
         структурного приводу тягнути немає, фіксуємо максимум.
      2. Інакше рахуємо структурний стоп; коли він КРАЩИЙ за поточний → `trail`.
      3. Коли прибуток ≥ `be_at_r` × ризик, стоп не має бути гіршим за беззбиток.
      4. Нічого не покращилось → `hold` (це НЕ бездіяльність, а рішення тримати).
    """
    c = {**DEFAULTS, **(cfg or {})}
    reasons: List[str] = []
    e, p = _f(entry), _f(price)
    if side not in ('LONG', 'SHORT') or not p or p <= 0:
        return {'action': 'hold', 'stop': None, 'objective': None,
                'next_obstacle': None, 'targets': [],
                'reasons': ['немає ціни або напрямку']}

    targets = collect_targets(side, p, swing=swing, runway=runway, poc=poc, cfg=c)
    nearest = targets[0] if targets else None

    # 🔒 ЦІЛЬ ФІКСУЄТЬСЯ ОДИН РАЗ і далі не «тікає». Без цього була б класична
    # пастка рухомих воріт: щойно ціна підходить до цілі, та зникає зі списку
    # «попереду», найдальшою стає наступна — і угода ніколи не доходить до
    # фіксації. Тому викликач зберігає обрану ціль і повертає її сюди назад.
    _lock = objective_lock if (objective_lock or {}).get('price') else None
    objective = _lock or pick_objective(targets)

    # ── 1. Ціль досягнуто? ────────────────────────────────────────────────
    tol = max(0.0, float(c['target_tol_pct'])) / 100.0
    if objective:
        _tp = _f(objective.get('price'))
        reached = _tp is not None and (
            (p >= _tp * (1.0 - tol)) if side == 'LONG' else (p <= _tp * (1.0 + tol)))
        if reached:
            reasons.append(f"ціль досягнуто: {objective.get('label') or 'ціль'} "
                           f"@ {_tp:.8g} (ціна {p:.8g})")
            return {'action': 'take', 'stop': None, 'objective': objective,
                    'next_obstacle': nearest, 'targets': targets, 'reasons': reasons}

    # ── 2. Структурний стоп ───────────────────────────────────────────────
    cand, cand_label = None, ''
    st = structure_stop(side, p, swing, c)
    if st:
        cand, cand_label = st['price'], st['label']

    # ── 3. Беззбиток після досягнення be_at_r ─────────────────────────────
    if e and e > 0:
        move = ((p - e) / e * 100.0) if side == 'LONG' else ((e - p) / e * 100.0)
        risk = None
        _ps = _f(prev_stop)
        if _ps:
            risk = abs(e - _ps) / e * 100.0
        if risk and risk > 0 and move >= float(c['be_at_r']) * risk:
            be = breakeven_stop(side, e, risk, c)
            if be is not None and better_stop(side, be, cand):
                cand, cand_label = be, (f"беззбиток (+{c['be_lock_pct']}%) — "
                                        f"пройдено {move / risk:.1f}R")

    if cand is not None and better_stop(side, cand, prev_stop):
        reasons.append(f'стоп підтягнуто: {cand_label}')
        if objective:
            reasons.append(f"ціль: {objective.get('label') or 'ціль'} "
                           f"@ {_f(objective.get('price')):.8g}")
        if nearest and objective and nearest.get('price') != objective.get('price'):
            reasons.append(f"найближча перепона: {nearest['label']} "
                           f"(+{nearest['dist_pct']:.2f}%)")
        return {'action': 'trail', 'stop': cand, 'objective': objective,
                'next_obstacle': nearest, 'targets': targets, 'reasons': reasons}

    # ── 4. Тримаємо ───────────────────────────────────────────────────────
    if objective:
        _op = _f(objective.get('price'))
        _od = abs(_op - p) / p * 100.0 if _op else 0.0
        reasons.append(f"тримаємо до цілі: {objective.get('label') or 'ціль'} "
                       f"@ {_op:.8g} ({_od:+.2f}% ходу)")
    else:
        reasons.append('змістовних цілей попереду не видно — тримаємо за стопом')
    if cand is not None:
        reasons.append('структурний стоп не кращий за поточний — не чіпаємо')
    return {'action': 'hold', 'stop': None, 'objective': objective,
            'next_obstacle': nearest, 'targets': targets, 'reasons': reasons}
