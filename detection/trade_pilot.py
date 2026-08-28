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
    # 🎯 Мінімальний зазор для поділу на TP-1/TP-2, % ціни. Захищає від двох
    # безглуздих ситуацій: TP впритул до входу (комісія зʼїсть частковий вихід)
    # і два TP на тому самому місці (поділ без сенсу).
    'tp_min_gap_pct': 0.40,
    # 🎯 ВІКНО ДЛЯ TP-1, у % ШЛЯХУ від входу до TP-2 (а не у % ціни!).
    # Заміряно на проді: TP-1 ставився на НАЙБЛИЖЧОМУ обʼєкті, і на FIL це
    # означало закрити половину позиції через 1.5% руху при цілі 13.4% —
    # тобто 11% шляху. Половину позиції треба знімати там, де вже забрано
    # відчутну частину руху, але ще далеко до кінця.
    'tp1_min_path_pct': 30.0,
    'tp1_max_path_pct': 75.0,
    # Якщо у вікні НЕМАЄ жодного обʼєкта графіка — ставимо TP-1 на цій частці
    # шляху до цілі. Це не «рівень із повітря»: він ПОХІДНИЙ від власної цілі
    # автопілота, і в підписі так і сказано. 0 = не ставити (лишити порожнім).
    'tp1_fallback_path_pct': 50.0,
}

# Сила обʼєкта як місця ЙМОВІРНОЇ РЕАКЦІЇ ціни — щоб серед кількох придатних
# кандидатів TP-1 брати НАЙЗМІСТОВНІШИЙ, а не просто найближчий.
_TP1_WEIGHT = {'liq_next': 5, 'liq_main': 4, 'poc': 3, 'va': 3, 'eq': 2, 'swing': 1}


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
                    vah=None, val=None,
                    cfg: Optional[Dict] = None) -> List[Dict]:
    """Усі ЗМІСТОВНІ об'єкти графіка ПОПЕРЕДУ руху, від найближчого до дальшого.

    Кожен елемент: {price, dist_pct, kind, label}. `kind` — щоб рішення можна
    було пояснити людською мовою, а не «бот вирішив».

    ⚠️ ІНВЕНТАР НАВМИСНО ШИРОКИЙ — це головне для TP-1. Заміряно на проді:
    8 із 14 позицій не мали ЖОДНОГО обʼєкта між входом і ціллю, тож частковій
    фіксації просто не було де стати. Додано два джерела, які НЕ коштують
    жодного зайвого запиту:
      • **VAH/VAL** — межі зони вартості; `compute_poc` уже їх повертає разом
        із POC, ми їх просто не брали;
      • **EQ (рівновага)** — середина дилінг-діапазону (`(high+low)/2`). Це
        канонічний SMC-рівень, на якому PD-зона перемикається з премії на
        дисконт, і найчастіша точка реакції всередині діапазону."""
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

    # 4) Межі зони вартості — приходять із того самого `compute_poc`.
    _add(vah, 'va', 'VAH (верх зони вартості)')
    _add(val, 'va', 'VAL (низ зони вартості)')

    # 5) EQ — рівновага дилінг-діапазону (50%). Саме тут PD-зона перемикається
    #    між премією і дисконтом, тож рух через неї регулярно сповільнюється.
    _hi_p, _lo_p = _f((sw.get('high') or {}).get('price')), _f((sw.get('low') or {}).get('price'))
    if _hi_p and _lo_p and _hi_p > _lo_p:
        _add((_hi_p + _lo_p) / 2.0, 'eq', 'рівновага діапазону (50%)')

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


def progress(side: str, entry, price, objective) -> Optional[Dict]:
    """Скільки шляху до ЦІЛІ вже пройдено — для візуалізації в таблиці угод.

    Рахуємо від ВХОДУ до цілі: 0% — щойно відкрились, 100% — ціль досягнута.
    Повертає {pct, done_pct, left_pct, target} або None, коли рахувати нема з
    чого. `pct` обрізаний у [0..100], щоб смужка прогресу не «вилітала»;
    `done_pct`/`left_pct` — сирий рух і залишок у відсотках ЦІНИ (без обрізання),
    бо саме вони показують реальний стан, коли ціна пішла проти."""
    e, p = _f(entry), _f(price)
    t = _f((objective or {}).get('price')) if objective else None
    if e is None or p is None or t is None or e <= 0 or p <= 0 or t <= 0:
        return None
    span = abs(t - e)
    if span <= 0:
        return None
    done = (p - e) if side == 'LONG' else (e - p)
    pct = max(0.0, min(100.0, done / span * 100.0))
    return {'pct': round(pct, 1),
            'done_pct': round(done / e * 100.0, 3),
            'left_pct': round(abs(t - p) / p * 100.0, 3),
            'target': t}


def risk_reward(entry, stop, objective) -> Optional[float]:
    """R = (відстань ВХІД→ЦІЛЬ) / (відстань ВХІД→СТОП).

    Головне число угоди: R < 1 означає, що ціль ближча за стоп, тобто угода
    ризикує більшим заради меншого. Показуємо його в таблиці, щоб якість
    угоди була видна ОДРАЗУ, а не після закриття. Немає стопа або цілі —
    повертаємо None (вигадувати «умовний R» не можна)."""
    e, s, t = _f(entry), _f(stop), _f((objective or {}).get('price'))
    if not e or not s or not t or e <= 0 or s <= 0 or t <= 0:
        return None
    risk = abs(e - s)
    if risk <= 0:
        return None
    return round(abs(t - e) / risk, 2)


def plan_targets(side: str, entry, price, targets: List[Dict],
                 *, objective: Optional[Dict] = None, stop=None,
                 cfg: Optional[Dict] = None) -> Dict:
    """🎯 Розкласти цілі на ДВА рівні фіксації: TP-1 (частковий) і TP-2 (повний).

    АНАЛІТИКА (чому саме так):
      • **TP-2 = ГОЛОВНА ціль** — найдальший змістовний обʼєкт попереду. Це те,
        заради чого угода тримається; на ньому виходимо ПОВНІСТЮ.
      • **TP-1 = найближча ЗМІСТОВНА перепона** між входом і TP-2. Саме там рух
        статистично найчастіше сповільнюється (пул ліквідності / POC / межа
        діапазону), тож зафіксувати там частину — це забрати гроші в
        найімовірнішій точці реакції, лишивши «безкоштовний» хвіст до TP-2.

    ЖОРСТКІ ПЕРЕВІРКИ (без них поділ безглуздий):
      1. обидва рівні — СТРОГО попереду руху від ВХОДУ;
      2. TP-1 суворо МІЖ входом і TP-2;
      3. TP-1 віддалений від ВХОДУ щонайменше на `tp_min_gap_pct` — інакше
         комісія зʼїсть частковий вихід (зазор стосується ЛИШЕ поділу: TP-2
         виставляється завжди, коли ціль є попереду входу);
      4. TP-2 віддалений від TP-1 щонайменше на `tp_min_gap_pct` — два рівні
         впритул не мають сенсу, тоді лишається лише TP-2;
      5. якщо придатної проміжної цілі немає — TP-1 НЕ вигадуємо (None).

    ⚠️ РІВНІ НЕ ЗАЛЕЖАТЬ ВІД СТОПА. Стоп передається лише щоб ПОКАЗАТИ `r`
    (довідково, у лозі й у колонці) — він НЕ вирішує, ставити рівень чи ні.
    Була спроба відсікати цілі з малим R — користувач це СКАСУВАВ: пілот
    рахує рівні з обʼєктів графіка і виставляє їх, а не судить угоду.

    Повертає {'tp1', 'tp2', 'reasons'}, де tp1/tp2 = {price, label, kind,
    from_entry_pct, r} або None. `r` — кратність ризику (R), коли відомий стоп:
    саме вона дає зрозуміти, чи вартий рівень того, щоб на ньому виходити.
    """
    c = {**DEFAULTS, **(cfg or {})}
    reasons: List[str] = []
    e, p = _f(entry), _f(price)
    if side not in ('LONG', 'SHORT') or e is None or e <= 0:
        return {'tp1': None, 'tp2': None, 'reasons': ['немає входу або напрямку']}

    gap = max(0.0, float(c.get('tp_min_gap_pct', DEFAULTS['tp_min_gap_pct'])))
    _st = _f(stop)
    risk = (abs(e - _st) / e * 100.0) if (_st and _st > 0) else None

    def _pack(t: Dict) -> Dict:
        lvl = _f(t.get('price'))
        d = abs(lvl - e) / e * 100.0
        return {'price': lvl, 'label': t.get('label') or 'ціль',
                'kind': t.get('kind'), 'from_entry_pct': round(d, 3),
                'r': (round(d / risk, 2) if risk and risk > 0 else None)}

    # TP-2 — головна ціль (зафіксована для угоди, якщо вже обрана).
    _obj = objective or pick_objective(targets or [])
    if not _obj or not _f(_obj.get('price')):
        return {'tp1': None, 'tp2': None,
                'reasons': ['змістовної цілі попереду немає — TP не виставляємо']}
    tp2 = _pack(_obj)
    if not _ahead(side, e, tp2['price']):
        return {'tp1': None, 'tp2': None,
                'reasons': ['ціль опинилась позаду входу — TP не виставляємо']}
    reasons.append(f"TP-2 (повний вихід): {tp2['label']} @ {tp2['price']:.8g} "
                   f"(+{tp2['from_entry_pct']:.2f}% від входу"
                   + (f", {tp2['r']}R)" if tp2['r'] else ')'))

    # ── TP-1 ──────────────────────────────────────────────────────────────
    # Не «найближчий обʼєкт», а НАЙЗМІСТОВНІШИЙ у розумній частині шляху.
    span = tp2['from_entry_pct']
    lo_p = max(0.0, float(c.get('tp1_min_path_pct', DEFAULTS['tp1_min_path_pct'])))
    hi_p = max(lo_p, float(c.get('tp1_max_path_pct', DEFAULTS['tp1_max_path_pct'])))
    mid_p = (lo_p + hi_p) / 2.0

    cands = []
    for t in (targets or []):
        lvl = _f(t.get('price'))
        if lvl is None or not _ahead(side, e, lvl):
            continue
        cand = _pack(t)
        if cand['from_entry_pct'] < gap:
            continue                                   # надто близько до входу
        if not _ahead(side, lvl, tp2['price']):
            continue                                   # мусить бути ДО TP-2
        if abs(tp2['price'] - lvl) / lvl * 100.0 < gap:
            continue                                   # впритул до TP-2
        path = cand['from_entry_pct'] / span * 100.0 if span > 0 else 0.0
        if path < lo_p or path > hi_p:
            continue                                   # поза вікном шляху
        cand['path_pct'] = round(path, 1)
        cands.append(cand)

    tp1 = None
    if cands:
        # Сильніший обʼєкт виграє; за рівної сили — ближчий до середини вікна
        # (там частковий вихід найзбалансованіший).
        cands.sort(key=lambda x: (-_TP1_WEIGHT.get(x.get('kind'), 0),
                                  abs(x['path_pct'] - mid_p)))
        tp1 = cands[0]
        reasons.append(f"TP-1 (частковий): {tp1['label']} @ {tp1['price']:.8g} "
                       f"(+{tp1['from_entry_pct']:.2f}% від входу, "
                       f"{tp1['path_pct']:.0f}% шляху до цілі"
                       + (f", {tp1['r']}R)" if tp1['r'] else ')'))
    else:
        # Обʼєкта у вікні немає — беремо ПОХІДНИЙ рівень від власної цілі.
        fb = max(0.0, float(c.get('tp1_fallback_path_pct',
                                  DEFAULTS['tp1_fallback_path_pct'])))
        lvl = None
        if fb > 0 and span > 0:
            lvl = (e * (1.0 + span * fb / 100.0 / 100.0) if side == 'LONG'
                   else e * (1.0 - span * fb / 100.0 / 100.0))
        # ⚠️ Похідний рівень теж мусить лежати СТРОГО перед TP-2 — інакше
        # «частковий» вихід опинився б ДАЛІ за повний (частка шляху > 100%).
        if lvl and _ahead(side, e, lvl) and _ahead(side, lvl, tp2['price']) \
                and abs(lvl - e) / e * 100.0 >= gap \
                and abs(tp2['price'] - lvl) / lvl * 100.0 >= gap:
            tp1 = _pack({'price': lvl, 'kind': 'path',
                         'label': f'{fb:g}% шляху до цілі'})
            tp1['path_pct'] = round(fb, 1)
            reasons.append(f"TP-1 (частковий, похідний): {fb:g}% шляху до цілі "
                           f"@ {tp1['price']:.8g} (+{tp1['from_entry_pct']:.2f}% "
                           f"від входу) — обʼєкта графіка у вікні "
                           f"{lo_p:g}-{hi_p:g}% немає"
                           + (f", {tp1['r']}R" if tp1['r'] else ''))
        else:
            reasons.append('проміжного рівня немає — працюємо одним TP-2')
    return {'tp1': tp1, 'tp2': tp2, 'r': tp2['r'], 'reasons': reasons}


def plan(side: str, entry, price, *, swing=None, runway=None, poc=None,
         vah=None, val=None, prev_stop=None,
         objective_lock: Optional[Dict] = None,
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

    targets = collect_targets(side, p, swing=swing, runway=runway, poc=poc,
                              vah=vah, val=val, cfg=c)
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
