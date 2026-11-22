"""🧩 СЦЕНАРІЇ АНОМАЛІЙ ОБСЯГУ — комплекс реакцій, а не одна позначка (17.11).

Вимога користувача дослівно: «бот має аналізувати весь процес перед входом в
угоду по Аномалії, тобто не по одній реакції, а комплексно… комплекс реакцій,
які призведуть саме до найкращої точки входу. І таким самим чином
супроводжувати відкриту угоду… комплекс реакцій, які дадуть розуміння, що
потрібно закривати угоду».

Сценарій = ПОСЛІДОВНІСТЬ етапів у часі. Кожен етап — реакція обсягу (⚠ аномалія,
прочитана за правилами користувача в `ai_signals.read_anomalies`), CHoCH або
стан ціни. Етап зараховується, лише коли стався ПІСЛЯ попереднього. Сценарій:
• `complete` — усі етапи пройдено;
• `ready` — пройдено і ще СВІЖИЙ (останній етап ≤ ENTRY_FRESH_BARS свічок тому
  для входу) — це і є «найкраща точка входу»; запізнілий — `stale`;
• `broken` — зустрічна підтверджена реакція зламала сценарій (з причиною).

ЧИСТІ функції, жодного I/O. Вхід — `items` з `read_anomalies` (t, kind, tf,
dir, role, level, conf, price, o1, o4, climax_of), CHoCH {dir, t} і контекст
ціни (`ctx`: price, beyond, fired, charged).

ВХІД (бік d = 'up' для LONG):
  E1 🔁 Розворот після виснаження — 🔥 виснаження зустрічного руху / 🧽 поглинання
     на рівні → друга реакція в бік d (🧽 / ⚡ зміна ініціативи) → підтвердження
     (свічка в новий бік / CHoCH / ⚡ продовження) → ціна тримається, зустрічного
     тиску немає → ВХІД.
  E2 💥 Старт розряду Пружини — пружина заряджена → імпульс у бік d (🔥 старт /
     📈 сплеск / пробій коробки) → продовження (⚡ / підтверджена свічка / CHoCH)
     → пробій ніхто не поглинає (немає 🧽 проти) → ВХІД.
  E3 ↗ Продовження після відкату — підтверджений рух у бік d → відкат без
     зустрічного тиску → нова реакція в бік d НА РІВНІ → ВХІД.
ВИХІД (угода в бік d; лише реакції ПІСЛЯ входу):
  X1 🔥→🧽 Кульмінація руху угоди — 🔥 виснаження нашого руху → 🧽 проти на 1–3
     свічках → підтвердження → ВИХІД.
  X2 ⚡ Перехоплення ініціативи — реакція проти на рівні (⚡ зсув / 🧽) → друга
     реакція проти або CHoCH проти → ціна підтверджує (за рівнем першої реакції)
     → ВИХІД.
  X3 ⚖ Зусилля без результату — ≥2 реакції за угодою, після яких ціна не пішла
     (o4 < 0.2%) → зустрічна реакція або CHoCH проти → ВИХІД.
  X4 🧽🧽 Подвійне поглинання проти — два 🧽 проти угоди (другий не спростований)
     → ВИХІД.
  E4 🪤 Спринг (хибний пробій вниз) — ціна вибиває нижню межу діапазону → реакція
     покупців і повернення в діапазон → підтвердження → ВХІД.
  X5 🪤 Аптраст (хибний пробій угору) — дзеркально для виходу з LONG (і навпаки).

ДЖЕРЕЛА (звірено 17.11): Wyckoff — Selling Climax → Automatic Rally → Secondary
Test → Spring → Sign of Strength → Last Point of Support (E1, E3, E4), дзеркально
Buying Climax / UTAD / Sign of Weakness (X1, X5); VSA — stopping volume, no
supply / no demand, upthrust, Effort vs Result (X3); order flow — absorption,
exhaustion, initiative, «одна реакція — контекст, потрібне підтвердження»
(confirmation-етапи всіх сценаріїв).
⚠️ Правила — експертні; ваги реакцій калібрує `ai_signals.calibration` з
досліджень «⚠ Аномалії обсягу». Сценарії не бектестовано — стартовий набір.
"""
from typing import Dict, List, Optional

TF_SEC = {'15m': 900, '1h': 3600, '4h': 14400}
SCEN_LOOK_SEC = 12 * 3600       # вхід: увесь процес, що привів до точки (до 12 год)
SCEN_SPAN_BARS = 12             # етапи одного сценарію — у межах 12 свічок
COMBO_BARS = 3                  # 🧽 після 🔥 — на 1–3 свічках
ENTRY_FRESH_BARS = 2            # точка входу «свіжа» — останній етап ≤ 2 свічки тому
NO_PROGRESS_PCT = 0.2           # X3: рух через 4 свічки менший — «без результату»
HOLD_TOL_PCT = 0.35             # ціна «тримається» біля рівня реакції

ENTRY_NAMES = {'E1': '🔁 Розворот після виснаження', 'E2': '💥 Старт розряду Пружини',
               'E3': '↗ Продовження після відкату', 'E4': '🪤 Спринг (хибний пробій)'}
EXIT_NAMES = {'X1': '🔥→🧽 Кульмінація руху угоди', 'X2': '⚡ Перехоплення ініціативи',
              'X3': '⚖ Зусилля без результату', 'X4': '🧽🧽 Подвійне поглинання проти',
              'X5': '🪤 Аптраст (хибний пробій за угодою)'}


# 📘 ДОВІДНИК ПРАВИЛ (вимога 17.11: «заложи ШІ ці правила і алгоритм…»).
# Той самий порядок етапів, що в коді нижче; сторінка показує його як є.
CATALOG = {
    'entry': [
        {'key': 'E1', 'name': ENTRY_NAMES['E1'],
         'stages': ['🔥 виснаження зустрічного руху або 🧽 поглинання на рівні',
                    'друга реакція в бік входу (🧽 / ⚡ зміна ініціативи)',
                    'підтвердження: свічка закрилась у новий бік / CHoCH / ⚡ продовження',
                    'ціна тримається біля рівня реакції, зустрічного тиску немає'],
         'why': 'продавця (покупця) вичерпано і його обсяг поглинули — розворот із найкращою ціною входу',
         'break': 'підтверджена зустрічна реакція або ціна не втримала рівень'},
        {'key': 'E2', 'name': ENTRY_NAMES['E2'],
         'stages': ['🌀 пружина заряджена (стиснення)',
                    'імпульс у бік входу: 🔥 старт / 📈 сплеск / пробій коробки',
                    'продовження: ⚡ перекіс / підтверджена свічка / CHoCH',
                    'пробій ніхто не поглинає (немає 🧽 проти)'],
         'why': 'енергія стиснення вивільняється в один бік, і пробій приймають',
         'break': '🧽 поглинання проти пробою або пробій коробки в інший бік'},
        {'key': 'E3', 'name': ENTRY_NAMES['E3'],
         'stages': ['підтверджений рух у бік входу',
                    'відкат без зустрічного тиску',
                    'нова реакція в бік входу на рівні (OB / межа коробки)'],
         'why': 'тренд живий, відкат слабкий — вхід з рівня за трендом (Wyckoff LPS / VSA no supply)',
         'break': 'підтверджена зустрічна реакція під час відкату'},
        {'key': 'E4', 'name': ENTRY_NAMES['E4'],
         'stages': ['ціна вибила межу діапазону проти входу (збір стопів)',
                    'реакція в бік входу і повернення в діапазон',
                    'підтвердження: свічка в бік входу / CHoCH'],
         'why': 'Wyckoff Spring: стопи зібрано, продавця поглинули — сильний розворот із межі',
         'break': 'новий екстремум за межею — пробій справжній'},
    ],
    'exit': [
        {'key': 'X1', 'name': EXIT_NAMES['X1'],
         'stages': ['🔥 виснаження руху за угодою', '🧽 поглинання проти на 1–3 свічках',
                    'підтвердження: свічка проти / CHoCH проти'],
         'why': 'рух угоди вичерпано, його обсяг поглинули — розворот'},
        {'key': 'X2', 'name': EXIT_NAMES['X2'],
         'stages': ['реакція проти на рівні (⚡ зсув / 🧽)', 'друга реакція проти або CHoCH проти',
                    'ціна пішла за рівень першої реакції'],
         'why': 'ініціативу перехопила протилежна сторона'},
        {'key': 'X3', 'name': EXIT_NAMES['X3'],
         'stages': ['≥2 реакції за угодою без руху ціни', 'зустрічна реакція або CHoCH проти'],
         'why': 'обсяг є, а результату немає — рух угоди слабне'},
        {'key': 'X4', 'name': EXIT_NAMES['X4'],
         'stages': ['перше 🧽 проти угоди', 'друге 🧽 проти (не спростоване)'],
         'why': 'двічі поглинули рух угоди — сторона втрачає контроль'},
        {'key': 'X5', 'name': EXIT_NAMES['X5'],
         'stages': ['ціна вибила межу діапазону за угодою', 'реакція проти і повернення в діапазон',
                    'підтвердження: свічка проти / CHoCH проти'],
         'why': 'Wyckoff Upthrust / UTAD: пробій за угодою не прийняли — пастка, розворот'},
    ],
    'rules': ['Рахуються лише реакції, прочитані за правилами «⚠ Аномалій»: рівень, тренд до, підтвердження наступною свічкою або CHoCH.',
              'Вхід — лише коли сценарій пройдено ПОВНІСТЮ і точка свіжа (останній етап ≤ 2 свічки тому); запізнілий сценарій не торгуємо.',
              'Вихід судить лише реакції ПІСЛЯ входу; аномалії до входу — це причини входу, а не виходу.',
              'Свіже ⚡/📈 продовження за угодою тримає угоду, поки сценарій виходу не завершено.',
              'Одна позначка без рівня й контексту — лише «придивитись», не рішення.'],
}


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _bar(it) -> int:
    return TF_SEC.get((it or {}).get('tf'), 3600)


def _od(d):
    return 'dn' if d == 'up' else 'up'


def _side_dir(side) -> str:
    return 'up' if str(side).upper() == 'LONG' else 'dn'


def _st(k, label, it=None, t=None, why=''):
    """Етап: done, коли є реакція `it` або момент `t`."""
    tt = _f((it or {}).get('t')) if it else _f(t)
    return {'k': k, 'label': label, 'done': tt is not None, 't': tt,
            'px': _f((it or {}).get('price')) if it else None,
            'why': why or ((it or {}).get('why') or '')}


def _after(items, t, pred, span=None):
    for it in items:
        ti = _f(it.get('t'))
        if ti is None or ti <= t:
            continue
        if span is not None and ti - t > span:
            break
        if pred(it):
            return it
    return None


def _counter(items, t, d, roles=('reversal', 'shift', 'continuation', 'start', 'combo')):
    """Перша ПІДТВЕРДЖЕНА реакція проти d після t (злам сценарію)."""
    return _after(items, t, lambda x: x.get('dir') == _od(d) and x.get('role') in roles
                  and x.get('conf') is True)


def _price_holds(d, price, lvl, tol=HOLD_TOL_PCT) -> Optional[bool]:
    px, lv = _f(price), _f(lvl)
    if not px or not lv:
        return None
    return px >= lv * (1 - tol / 100) if d == 'up' else px <= lv * (1 + tol / 100)


def _finish(key, names, side, stages, now, broken='', fresh_bars=None):
    n = sum(1 for s in stages if s['done'])
    complete = n == len(stages) and not broken
    # 🐞 18.11: свіжість — за ПОДІЯМИ, а не за станом. Етап-стан («🌀 пружина
    # заряджена», t = now) робив будь-який старий сценарій «свіжим» назавжди
    # (кейс AVAX/TRUMP: імпульс 8 год тому → сигнал щопівгодини).
    last_t = max((s['t'] for s in stages if s['t'] is not None and not s.get('state')),
                 default=None)
    ready, stale = False, False
    if complete:
        if fresh_bars is None:
            ready = True
        else:
            span = fresh_bars * stages[-1].get('bar', 3600)
            ready = last_t is not None and now - last_t <= span
            stale = not ready
    return {'key': key, 'name': names[key], 'side': side, 'stages': stages,
            'done': n, 'total': len(stages), 'progress': round(n / len(stages), 3),
            'complete': complete, 'ready': ready, 'stale': stale,
            'broken': broken, 'last_t': last_t}


def _sorted(items, t0=None):
    out = [it for it in (items or []) if _f(it.get('t')) is not None
           and (t0 is None or _f(it.get('t')) >= t0)]
    return sorted(out, key=lambda x: _f(x['t']))


# ───────────────────────── ВХІД ─────────────────────────
def _e1(items, d, ctx, choch, now):
    stop = None
    for it in items:
        if it.get('dir') != d:
            continue
        if (it.get('kind') == 'climax' and it.get('role') == 'reversal') or \
                (it.get('kind') == 'absorb' and it.get('level')):
            stop = it
            break
    s1 = _st('stop', '🔥 виснаження / 🧽 поглинання зустрічного руху', stop)
    two = None
    if stop:
        span = SCEN_SPAN_BARS * _bar(stop)
        two = _after(items, _f(stop['t']), lambda x: x.get('dir') == d and x is not stop
                     and x.get('role') in ('reversal', 'shift', 'combo'), span)
    s2 = _st('react', 'друга реакція в бік входу (🧽 / ⚡ зміна ініціативи)', two)
    conf, cw = None, ''
    if two:
        t2 = _f(two['t'])
        if two.get('conf') is True:
            conf, cw = t2, 'свічка закрилась у новий бік'
        elif choch and choch.get('dir') == d and (_f(choch.get('t')) or 0) >= _f(stop['t']):
            conf, cw = _f(choch['t']), '🔀 CHoCH у бік входу'
        else:
            c = _after(items, t2, lambda x: x.get('dir') == d and x.get('conf') is True)
            if c:
                conf, cw = _f(c['t']), c.get('why') or 'підтверджена реакція'
    s3 = _st('confirm', 'підтвердження (свічка / CHoCH / ⚡ продовження)', t=conf, why=cw)
    broken, hold_t = '', None
    if two:
        ctr = _counter(items, _f(two['t']), d)
        if ctr:
            broken = 'зустрічна підтверджена реакція: ' + (ctr.get('why') or ctr.get('kind'))
        elif conf is not None:
            h = _price_holds(d, ctx.get('price'), two.get('price'))
            if h:
                hold_t = max(conf, _f(two['t']))
            elif h is False:
                broken = 'ціна не втримала рівень реакції'
    s4 = _st('hold', 'ціна тримається, зустрічного тиску немає', t=hold_t)
    st = [s1, s2, s3, s4]
    for s, it in zip(st, (stop, two, two, two)):
        s['bar'] = _bar(it)
    return st, broken


def _e2(items, d, ctx, choch, now):
    side = 'LONG' if d == 'up' else 'SHORT'
    s1 = _st('charged', '🌀 пружина заряджена', t=now if ctx.get('charged') else None)
    s1['state'] = True              # стан, а не подія — у свіжість не входить
    imp = next((it for it in items if it.get('dir') == d and (
        it.get('role') == 'start' or it.get('kind') == 'spike')), None)
    imp_t = _f(imp['t']) if imp else None
    if imp_t is None and (ctx.get('beyond') == side or ctx.get('fired') == side):
        # пробій живого бару: імпульс — від ВІДКРИТТЯ цієї 1h-свічки
        imp_t = _f(ctx.get('bar_t')) or (now - 3600)
    s2 = _st('impulse', 'імпульс у бік входу (🔥 старт / 📈 сплеск / пробій коробки)',
             t=imp_t, why=(imp or {}).get('why') or 'пробій коробки')
    fol, fw = None, ''
    if imp_t is not None:
        if imp and imp.get('conf') is True:
            fol, fw = _f(imp['t']), 'наступна свічка підтвердила імпульс'
        else:
            c = _after(items, imp_t - 1, lambda x: x.get('dir') == d and x is not imp and (
                x.get('role') == 'continuation' or x.get('conf') is True))
            if c:
                fol, fw = _f(c['t']), c.get('why') or 'продовження'
            elif choch and choch.get('dir') == d and (_f(choch.get('t')) or 0) >= imp_t - _bar(imp):
                fol, fw = _f(choch['t']), '🔀 CHoCH у бік входу'
    s3 = _st('follow', 'продовження (⚡ / підтверджена свічка / CHoCH)', t=fol, why=fw)
    broken, clean = '', None
    if imp_t is not None:
        ab = _after(items, imp_t - 1, lambda x: x.get('dir') == _od(d)
                    and x.get('kind') == 'absorb' and x.get('conf') is not False)
        if ab:
            broken = 'пробій поглинають: ' + (ab.get('why') or '🧽 проти')
        elif ctx.get('beyond') == ('SHORT' if d == 'up' else 'LONG'):
            broken = 'ціна пробила коробку в протилежний бік'
        elif fol is not None:
            clean = fol
    s4 = _st('clean', 'пробій ніхто не поглинає (немає 🧽 проти)', t=clean)
    st = [s1, s2, s3, s4]
    for s in st:
        s['bar'] = _bar(imp) if imp else 3600
    return st, broken


def _e3(items, d, ctx, choch, now):
    leg = next((it for it in items if it.get('dir') == d and it.get('conf') is True
                and it.get('role') in ('continuation', 'start')), None)
    s1 = _st('trend', 'підтверджений рух у бік входу', leg)
    quiet_t, broken, res = None, '', None
    if leg:
        t1 = _f(leg['t'])
        ctr = _counter(items, t1, d)
        if ctr:
            broken = 'відкат із зустрічним тиском: ' + (ctr.get('why') or ctr.get('kind'))
        res = _after(items, t1 + _bar(leg) - 1, lambda x: x.get('dir') == d and x is not leg
                     and x.get('level') and x.get('conf') is not False
                     and _price_holds(_od(d), x.get('price'), leg.get('price'), 0.0) is not False,
                     SCEN_SPAN_BARS * _bar(leg))
        if not broken and res:
            quiet_t = _f(res['t']) - 1
    s2 = _st('pullback', 'відкат без зустрічного тиску', t=quiet_t)
    s3 = _st('resume', 'нова реакція в бік входу на рівні', res if (res and not broken) else None)
    st = [s1, s2, s3]
    for s in st:
        s['bar'] = _bar(res or leg)
    return st, broken


def _false_break(items, d, ctx, choch, push):
    """Хибний пробій межі діапазону (Spring / Upthrust). `push` — бік пробою
    ('dn' = вниз під нижню межу), реакція і повернення — у бік `_od(push)`.
    → (етапи, злам)."""
    back = _od(push)
    lo, hi = _f(ctx.get('box_lo')), _f(ctx.get('box_hi'))
    edge = lo if push == 'dn' else hi
    out = (lambda p: p is not None and edge and (p < edge if push == 'dn' else p > edge))
    # пробій — ОКРЕМА реакція в бік пробою (агресія за межу), а не та сама свічка,
    # що потім повертає: комплекс = щонайменше дві різні реакції
    sweep = next((it for it in items if out(_f(it.get('price'))) and it.get('dir') == push),
                 None) if edge else None
    s1 = _st('sweep', 'ціна вибила межу діапазону', sweep,
             why=(f"закриття {sweep.get('price')} за межею {edge:g}" if sweep else ''))
    ret, rw, broken = None, '', ''
    if sweep:
        t1 = _f(sweep['t'])
        it = _after(items, t1, lambda x: x is not sweep and x.get('dir') == back and x.get('role') in (
            'reversal', 'shift', 'combo'), SCEN_SPAN_BARS * _bar(sweep))
        px = _f(ctx.get('price'))
        inside = px is not None and not out(px)
        if it and (inside or not out(_f(it.get('price')))):
            ret, rw = _f(it['t']), (it.get('why') or 'реакція') + ' · ціна знову в діапазоні'
        deeper = next((x for x in items if _f(x['t']) > t1 and _f(x.get('price')) is not None
                       and (_f(x['price']) < _f(sweep['price']) if push == 'dn'
                            else _f(x['price']) > _f(sweep['price']))
                       and x.get('dir') == push and x.get('conf') is True), None)
        if deeper:
            broken = 'новий екстремум за межею — пробій справжній'
    s2 = _st('back', 'реакція і повернення в діапазон', t=ret, why=rw)
    cf, cw = None, ''
    if ret is not None:
        c = _after(items, _f(sweep['t']), lambda x: x.get('dir') == back and x.get('conf') is True)
        if c:
            cf, cw = max(_f(c['t']), ret), 'свічка закрилась у бік повернення'
        elif choch and choch.get('dir') == back and (_f(choch.get('t')) or 0) >= _f(sweep['t']):
            cf, cw = max(_f(choch['t']), ret), '🔀 CHoCH'
    s3 = _st('confirm', 'підтвердження: свічка / CHoCH', t=cf, why=cw)
    st = [s1, s2, s3]
    for x in st:
        x['bar'] = _bar(sweep)
    return st, broken


def _e4(items, d, ctx, choch, now):
    return _false_break(items, d, ctx, choch, push=_od(d))


def _x5(items, d, ctx, choch, now):
    return _false_break(items, _od(d), ctx, choch, push=d)


def match_entry(items: List[Dict], side: str, ctx: Optional[Dict] = None,
                choch: Optional[Dict] = None, now: float = 0.0) -> List[Dict]:
    """Усі сценарії входу в бік `side` — стан кожного (етапи, готовність, злам)."""
    d = _side_dir(side)
    ctx = ctx or {}
    its = _sorted(items, now - SCEN_LOOK_SEC if now else None)
    out = []
    for key, fn in (('E1', _e1), ('E2', _e2), ('E3', _e3), ('E4', _e4)):
        out.append(_entry_anchored(key, fn, its, d, ctx, choch, now, str(side).upper()))
    return out


def _entry_anchored(key, fn, its, d, ctx, choch, now, side):
    """🐞 18.11: сценарій чіпляється за ПЕРШУ (найстарішу) реакцію вікна, тож
    старий завершений-і-запізнілий ланцюг затуляв свіжий. Пробуємо якорі від
    НАЙНОВІШОГО: перший готовий (свіжий, незламаний) — він; немає — як раніше,
    від початку вікна (прогрес / злам для показу)."""
    st, br = fn(its, d, ctx, choch, now)
    full = _finish(key, ENTRY_NAMES, side, st, now, br, fresh_bars=ENTRY_FRESH_BARS)
    if full['ready']:
        return full
    for i in range(len(its) - 1, 0, -1):
        if its[i].get('dir') != d and key != 'E4':
            continue
        st, br = fn(its[i:], d, ctx, choch, now)
        r = _finish(key, ENTRY_NAMES, side, st, now, br, fresh_bars=ENTRY_FRESH_BARS)
        if r['ready']:
            return r
    return full


# ───────────────────────── ВИХІД ─────────────────────────
def _x1(items, d, ctx, choch, now):
    cl = next((it for it in items if it.get('kind') == 'climax' and it.get('role') == 'reversal'
               and it.get('climax_of') == d), None)
    s1 = _st('climax', '🔥 виснаження руху за угодою', cl)
    ab = _after(items, _f(cl['t']), lambda x: x.get('kind') == 'absorb' and x.get('dir') == _od(d),
                COMBO_BARS * _bar(cl)) if cl else None
    s2 = _st('absorb', '🧽 поглинання проти на 1–3 свічках', ab)
    cf, cw = None, ''
    if ab:
        if ab.get('conf') is True:
            cf, cw = _f(ab['t']), 'свічка закрилась проти угоди'
        elif choch and choch.get('dir') == _od(d) and (_f(choch.get('t')) or 0) >= _f(cl['t']):
            cf, cw = _f(choch['t']), '🔀 CHoCH проти угоди'
    s3 = _st('confirm', 'підтвердження (свічка / CHoCH проти)', t=cf, why=cw)
    return [s1, s2, s3], ''


def _x2(items, d, ctx, choch, now):
    first = next((it for it in items if it.get('dir') == _od(d) and it.get('level') and (
        it.get('role') == 'shift' or it.get('kind') == 'absorb')), None)
    s1 = _st('shift', 'реакція проти на рівні (⚡ зсув / 🧽)', first)
    sec, sw = None, ''
    if first:
        it = _after(items, _f(first['t']), lambda x: x.get('dir') == _od(d)
                    and x.get('conf') is not False)
        if it:
            sec, sw = _f(it['t']), it.get('why') or 'друга реакція проти'
        elif choch and choch.get('dir') == _od(d) and (_f(choch.get('t')) or 0) >= _f(first['t']):
            sec, sw = _f(choch['t']), '🔀 CHoCH проти угоди'
    s2 = _st('second', 'друга реакція проти або CHoCH проти', t=sec, why=sw)
    pr = None
    if sec is not None:
        side = 'LONG' if d == 'up' else 'SHORT'
        lost = _price_holds(d, ctx.get('price'), first.get('price'), 0.0) is False
        if lost or ctx.get('beyond') == ('SHORT' if side == 'LONG' else 'LONG'):
            pr = max(sec, now)
    s3 = _st('price', 'ціна пішла за рівень першої реакції', t=pr)
    return [s1, s2, s3], ''


def _x3(items, d, ctx, choch, now):
    weak = [it for it in items if it.get('dir') == d and it.get('role') in ('continuation', 'start')
            and _f(it.get('o4')) is not None
            and _f(it['o4']) * (1 if d == 'up' else -1) < NO_PROGRESS_PCT]
    s1 = _st('effort', '≥2 реакції за угодою без руху ціни (зусилля без результату)',
             weak[1] if len(weak) >= 2 else None)
    nx, nw = None, ''
    if len(weak) >= 2:
        t1 = _f(weak[1]['t'])
        it = _after(items, t1, lambda x: x.get('dir') == _od(d) and x.get('conf') is not False)
        if it:
            nx, nw = _f(it['t']), it.get('why') or 'реакція проти'
        elif choch and choch.get('dir') == _od(d) and (_f(choch.get('t')) or 0) >= t1:
            nx, nw = _f(choch['t']), '🔀 CHoCH проти угоди'
    s2 = _st('turn', 'зустрічна реакція або CHoCH проти', t=nx, why=nw)
    return [s1, s2], ''


def _x4(items, d, ctx, choch, now):
    abs_ = [it for it in items if it.get('kind') == 'absorb' and it.get('dir') == _od(d)]
    a1 = abs_[0] if abs_ else None
    a2 = None
    if a1:
        a2 = _after(items, _f(a1['t']), lambda x: x.get('kind') == 'absorb'
                    and x.get('dir') == _od(d) and x.get('conf') is not False,
                    6 * _bar(a1))
    return [_st('a1', 'перше 🧽 проти угоди', a1),
            _st('a2', 'друге 🧽 проти (не спростоване)', a2)], ''


def match_exit(items: List[Dict], side: str, ctx: Optional[Dict] = None,
               choch: Optional[Dict] = None, now: float = 0.0,
               opened_at: Optional[float] = None) -> List[Dict]:
    """Сценарії виходу для угоди в бік `side`. Рахуються ЛИШЕ реакції після
    відкриття (до входу — причини входу, а не виходу); CHoCH — теж після входу."""
    d = _side_dir(side)
    ctx = ctx or {}
    oa = _f(opened_at)
    its = _sorted(items, oa)
    ch = choch if (choch and (oa is None or (_f(choch.get('t')) or 0) >= oa)) else None
    out = []
    for key, fn in (('X1', _x1), ('X2', _x2), ('X3', _x3), ('X4', _x4), ('X5', _x5)):
        st, br = fn(its, d, ctx, ch, now)
        out.append(_finish(key, EXIT_NAMES, str(side).upper(), st, now, br, fresh_bars=None))
    return out


def best(scens: List[Dict]) -> Optional[Dict]:
    """Головний сценарій: готовий → найбільший прогрес (незламаний) → None."""
    live = [s for s in scens or [] if not s.get('broken')]
    ready = [s for s in live if s.get('ready')]
    if ready:
        return max(ready, key=lambda s: (s['total'], s['last_t'] or 0))
    forming = [s for s in live if s.get('done')]
    return max(forming, key=lambda s: (s['progress'], s['last_t'] or 0)) if forming else None


def stages_text(s: Optional[Dict]) -> str:
    """«🔁 Розворот після виснаження 3/4: ✓ … · ✓ … · ⏳ …» — для логу й підказок."""
    if not s:
        return ''
    parts = [('✓ ' if x['done'] else '⏳ ') + x['label'] for x in s['stages']]
    tail = f" · ✗ злам: {s['broken']}" if s.get('broken') else (' · ⌛ запізно' if s.get('stale') else '')
    return f"{s['name']} {s['done']}/{s['total']}: " + ' · '.join(parts) + tail


def view(s: Optional[Dict]) -> Optional[Dict]:
    """Компактний знімок для UI / стану."""
    if not s:
        return None
    ev = [x for x in s['stages'] if x.get('t') is not None and not x.get('state')]
    last = max(ev, key=lambda x: x['t']) if ev else None
    return {'key': s['key'], 'name': s['name'], 'side': s['side'], 'done': s['done'],
            'total': s['total'], 'ready': s['ready'], 'stale': s['stale'],
            'broken': s['broken'], 'last_t': s.get('last_t'),
            # ціна на момент ОСТАННЬОЇ події сценарію (вимога 22.11: «ціна на момент
            # сигналу, дата і час»); немає ціни в реакції → None (не вигадуємо)
            'last_px': (last or {}).get('px'),
            'stages': [{'label': x['label'], 'done': x['done'], 'why': x.get('why') or '',
                        't': x.get('t'), 'px': x.get('px')}
                       for x in s['stages']]}
