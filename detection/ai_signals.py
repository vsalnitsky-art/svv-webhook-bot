"""🤖 ШІ: РОЗРЯД ЗАРЯДЖЕНОЇ «ПРУЖИНИ» — сигнал входу · вихід · епізоди (17.11).

Вимога: «Монети саме із "Пружин" мають брати участь у цьому дослідженні. Саме
монети, які "заряджені". Саме в ці моменти потрібно розуміння коли почнеться
момент розряду монети і має бути реакцією на вхід в угоду, а відповідно, коли
вже почались зворотні процеси… що вже потрібно виходити із угоди. Додай…
сигнал "Сигнали ШІ" і… (Вихід з угоди ШІ).»

⚠️ ЧЕСНО ПРО «ШІ». Навченої моделі ще немає — бракує даних. Тут ПРАВИЛА над
показниками, які бот уже рахує (заряд і натяг 🌀 Пружини, живий 1h-бар
«вистрілу», ⚠ аномалії обсягу), з ВАГАМИ, які підлаштовуються під факти:
множник кожного виду аномалії = її історична частка «ціна пішла в бік
підказки» (`vol_anomaly.stats`, ≥ CAL_MIN_N випадків). Плюс кожна заряджена
монета пише ЕПІЗОД (`sob_ai_episode`): як ішов заряд, коли й куди почався
розряд, що показували ворота в ту мить і чим закінчилось — це і є датасет,
на якому правила калібруються далі.

ЧИСТІ функції (без I/O): `box_pos`, `flow_score`, `entry_score`, `exit_score`,
`calibration`, `episode_*`. I/O — лише `AiEngine` (бари 1h монет у заряді,
читання таблиці аномалій, сигнал через СПІЛЬНІ ворота `_signal_allowed`).
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

# ── налаштування (ключі СКАНЕРА — там, де всі сигнали) ──
DEFAULTS = {'ai_signal_enabled': False, 'ai_entry_min': 70, 'ai_min_charge': 65}
TICK_SEC = 30               # такт рушія
LIVE_TTL = 60               # 1h-бари монети в заряді — не частіше раз на хвилину
WORKERS = 4
MAX_COINS = 40              # стеля монет, які ведемо одночасно
ANOM_LOOK_SEC = 2 * 3600    # аномалії, що враховуються «зараз»
SAMPLE_SEC = 300            # зріз епізоду
SAMPLES_MAX = 288
EVENTS_MAX = 60
EPISODE_IDLE_SEC = 1800     # не заряджена й без розряду 30 хв → епізод закрито
EPISODE_AFTER_SEC = 24 * 3600   # стежимо після розряду / сигналу
EPISODE_MAX_SEC = 72 * 3600
SIGNAL_RETRY_SEC = 1800     # ворота відхилили → не частіше раз на 30 хв
SIGNAL_TRIES = 3            # спроб сигналу в один бік на епізод
CAL_REFRESH_SEC = 3600
CAL_MIN_N = 30
CAL_DAYS = 30
DB_STATE = 'ai_engine_state_v1'

W_ENTRY = {'lean': 0.25, 'break': 0.30, 'impulse': 0.25, 'flow': 0.20}
W_EXIT = {'against': 0.30, 'flow': 0.20, 'climax': 0.20, 'target': 0.15, 'fade': 0.15}
FLOW_W = {'side': 1.0, 'absorb': 0.8}      # аномалії з напрямком
FLOW_VETO = -0.6
EXIT_MIN_PARTS = 2          # вихід — щонайменше два НЕЗАЛЕЖНІ свідчення
HOLD_PENALTY = 15           # продовження в бік угоди знижує бал виходу
COMBO_EXIT_SCORE = 90       # підтверджений 🔥→🧽 проти угоди
COMBO_WAIT_SCORE = 60       # 🔥→🧽 є, підтвердження ще немає
COMBO_ENTRY_SCORE = 85      # підтверджений 🔥→🧽 у бік розряду
PART_UA = {'lean': 'натяг', 'break': 'пробій коробки', 'impulse': 'імпульс',
           'flow': 'потік обсягу', 'against': 'розворот ціни', 'climax': 'кульмінація',
           'target': 'ціль Пружини', 'fade': 'віддача від піку'}


def _c(x, lo=0.0, hi=1.0):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return lo


def _opp(side):
    return 'SHORT' if str(side).upper() == 'LONG' else 'LONG'


# ───────────────────────── ЧИСТІ ФУНКЦІЇ ─────────────────────────
def box_pos(vw: Optional[Dict], side: str) -> Optional[float]:
    """Де ціна живого бару в 24-год коробці В БІК угоди: 0 = середина,
    1 = межа коробки, >1 — за межею. None — коробки немає."""
    vw = vw or {}
    try:
        hi, lo, px = float(vw['box_hi']), float(vw['box_lo']), float(vw['price'])
    except (KeyError, TypeError, ValueError):
        return None
    half = (hi - lo) / 2.0
    if half <= 0:
        return None
    v = (px - (hi + lo) / 2.0) / half
    return max(-3.0, min(3.0, v if str(side).upper() == 'LONG' else -v))


def flow_score(anoms: List[Dict], side: str, now: float, mult: Optional[Dict] = None,
               look: float = ANOM_LOOK_SEC):
    """Потік обсягу В БІК `side` за останні `look` секунд: −1…+1 + нотатки.
    ⚡ перекіс — бік тиску, 🧽 поглинання — бік розвороту; 🔥/📈 без напрямку
    не рахуються. Свіжіші за годину важать удвічі більше."""
    sd = str(side).upper()
    tot, notes = 0.0, []
    for a in anoms or []:
        k, h = a.get('kind'), a.get('hint')
        if k not in FLOW_W or h not in ('up', 'dn'):
            continue
        try:
            age = now - float(a.get('ts') or 0)
        except (TypeError, ValueError):
            continue
        if age < 0 or age > look:
            continue
        m = mult or {}
        w = FLOW_W[k] * float(m.get(f"{k}|{a.get('tf')}", m.get(k, 1.0))) * (1.0 if age <= 3600 else 0.5)
        sgn = 1.0 if (h == 'up') == (sd == 'LONG') else -1.0
        tot += sgn * w
        notes.append(f"{'+' if sgn > 0 else '−'}{k}{(' ' + str(a.get('tf'))) if a.get('tf') else ''}")
    return max(-1.0, min(1.0, tot)), notes


# ───── 📖 ЧИТАННЯ АНОМАЛІЙ (правила користувача 17.11) ─────
# • 🧽 Поглинання — НАЙСИЛЬНІШИЙ сигнал розвороту (бік — `hint`); найсильніше
#   на РІВНІ (межа діапазону, OB, High/Low, POC).
# • 🔥 Кульмінація — виснаження ЛИШЕ після ТРИВАЛОГО руху: розворот ПРОТИ
#   свічки кульмінації; посеред спокою / на пробої — навпаки СТАРТ руху в бік свічки.
# • ⚡ Перекіс і 📈 Сплеск — ПРОДОВЖЕННЯ в бік агресії; виняток — ⚡ ПРОТИ
#   тренду прямо на рівні: перша ознака зміни ініціативи (треба підтвердити).
# • 🔥 після довгого руху + 🧽 на наступних 1–3 свічках — найсильніший сценарій.
# • Підтвердження — наступна свічка закрилась у новий бік (`o1`) або CHoCH.
# • Одна позначка без рівня й контексту — лише «придивитись», не причина діяти.
TF_SEC = {'15m': 900, '1h': 3600, '4h': 14400}
TREND_PCT = {'15m': 0.8, '1h': 1.5, '4h': 3.0}   # «тривалий рух» за 8 свічок до
COMBO_BARS = 3              # 🧽 на 1–3 свічках після 🔥
LEVEL_TOL_PCT = 0.35        # «на рівні» — не далі (або 0.5·ATR)
CONF_O1_PCT = 0.05          # наступна свічка «закрилась у новий бік»
READ_BASE = {'absorb': 1.0, 'climax_rev': 0.8, 'climax_start': 0.5,
             'side': 0.6, 'side_shift': 0.5, 'spike': 0.4, 'combo': 1.0}
READ_ROLE_UA = {'reversal': 'розворот', 'continuation': 'продовження',
                'start': 'старт руху', 'shift': 'зміна ініціативи', 'combo': '🔥→🧽 розворот'}
KIND_ICON = {'absorb': '🧽', 'climax': '🔥', 'side': '⚡', 'spike': '📈'}


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def near_level(price, levels, tol_pct: float = LEVEL_TOL_PCT) -> Optional[str]:
    """Назва рівня, біля якого стоїть ціна (у межах tol_pct %), або None.
    `levels` = [(назва, ціна), …]."""
    px = _f(price)
    if not px or px <= 0:
        return None
    best, bd = None, None
    for nm, lv in levels or []:
        lv = _f(lv)
        if not lv or lv <= 0:
            continue
        d = abs(px / lv - 1) * 100
        if d <= tol_pct and (bd is None or d < bd):
            best, bd = nm, d
    return best


def last_choch(events) -> Optional[Dict]:
    """Останній CHoCH зі структури сканера → {'dir': 'up'|'dn', 't': сек}."""
    best = None
    for e in events or []:
        if str(e.get('tag') or '').upper() != 'CHOCH':
            continue
        t = _f(e.get('to_t') or e.get('t'))
        if t is None:
            continue
        t = t / 1000.0 if t > 1e12 else t
        d = str(e.get('dir') or '').lower()
        d = 'up' if d in ('bull', 'up', 'long') else ('dn' if d in ('bear', 'dn', 'short') else None)
        if d and (best is None or t > best['t']):
            best = {'dir': d, 't': t}
    return best


def classify_anomaly(a: Dict, levels=None, tol_pct: float = LEVEL_TOL_PCT) -> Optional[Dict]:
    """Одна аномалія → {dir, role, base, level, why} за правилами вище.
    `dir` — куди вона вказує ('up'/'dn'); None — напрямку немає."""
    k, tf = a.get('kind'), a.get('tf')
    pre = _f(a.get('pre_chg'))
    bd = a.get('bar_dir') if a.get('bar_dir') in ('up', 'dn') else None
    h = a.get('hint') if a.get('hint') in ('up', 'dn') else None
    thr = TREND_PCT.get(tf, 1.5)
    trend = None if pre is None or abs(pre) < thr else ('up' if pre > 0 else 'dn')
    lvl = near_level(a.get('price'), levels, tol_pct)
    if k == 'absorb' and h:
        return {'dir': h, 'role': 'reversal', 'base': READ_BASE['absorb'], 'level': lvl,
                'why': '🧽 поглинання' + (f' на рівні {lvl}' if lvl else ' (без рівня)')}
    if k == 'climax' and bd:
        if trend == bd:                       # після тривалого руху в бік свічки
            d = 'dn' if bd == 'up' else 'up'
            return {'dir': d, 'role': 'reversal', 'base': READ_BASE['climax_rev'], 'level': lvl,
                    'why': f'🔥 кульмінація після руху {pre:+.1f}% — виснаження', 'climax_of': bd}
        return {'dir': bd, 'role': 'start', 'base': READ_BASE['climax_start'], 'level': lvl,
                'why': '🔥 кульмінація без тривалого руху — старт / пробій'}
    if k == 'side' and h:
        if trend and trend != h and lvl:
            return {'dir': h, 'role': 'shift', 'base': READ_BASE['side_shift'], 'level': lvl,
                    'why': f'⚡ перекіс ПРОТИ тренду на рівні {lvl} — зміна ініціативи'}
        return {'dir': h, 'role': 'continuation', 'base': READ_BASE['side'], 'level': lvl,
                'why': '⚡ перекіс — продовження в бік агресії'}
    if k == 'spike' and bd:
        return {'dir': bd, 'role': 'continuation', 'base': READ_BASE['spike'], 'level': lvl,
                'why': '📈 сплеск — продовження'}
    return None


def read_anomalies(anoms: List[Dict], now: float, levels=None, choch: Optional[Dict] = None,
                   cal: Optional[Dict] = None, look: float = ANOM_LOOK_SEC,
                   tol_pct: float = LEVEL_TOL_PCT) -> Dict:
    """📖 Прочитати потік аномалій як людина. → {'up': {...}, 'dn': {...}, 'items': […]}.
    На кожен бік: `rev` — сила розвороту В цей бік (0..1), `cont` — продовження
    в цей бік, `start` — старт руху, `combo` — сценарій 🔥→🧽, `exhaust` —
    🔥-виснаження руху, що йшов У цей бік (для виходу), `notes`.
    Сила = база × рівень (без рівня ×0.6 для 🧽/⚡-зсуву) × підтвердження
    (наступна свічка в новий бік або CHoCH ×1.0, ще невідомо ×0.6, проти ×0.2)
    × калібровка з досліджень (`cal[kind|tf]` → `cal[kind]`)."""
    m = cal or {}
    items = []
    rows = []
    for a in anoms or []:
        t = _f(a.get('ts'))
        if t is None or now - t < 0 or now - t > look:
            continue
        rows.append((t, a))
    rows.sort(key=lambda x: x[0])
    for t, a in rows:
        c = classify_anomaly(a, levels, tol_pct)
        if not c:
            continue
        o1 = _f(a.get('o1'))
        conf = None
        if o1 is not None and abs(o1) >= CONF_O1_PCT:
            conf = (o1 > 0) == (c['dir'] == 'up')
        if not conf and choch and choch.get('dir') == c['dir'] and float(choch.get('t') or 0) >= t - 1:
            conf = True
        k, tf = a.get('kind'), a.get('tf')
        w = c['base'] * float(m.get(f'{k}|{tf}', m.get(k, 1.0)))
        if c['role'] in ('reversal', 'shift') and not c.get('level') and k != 'climax':
            w *= 0.6                          # без рівня — лише «придивитись»
        w *= 1.0 if conf else (0.2 if conf is False else 0.6)
        if now - t > 3600:
            w *= 0.7
        items.append(dict(c, t=t, kind=k, tf=tf, w=round(min(1.0, w), 3), conf=conf))
    # сценарій 🔥 (після руху) → 🧽 у бік розвороту на 1–3 свічках
    for i, x in enumerate(items):
        if x['kind'] != 'climax' or x['role'] != 'reversal':
            continue
        span = COMBO_BARS * TF_SEC.get(x['tf'], 3600)
        for y in items[i + 1:]:
            if (y['kind'] == 'absorb' and y['dir'] == x['dir']
                    and 0 < y['t'] - x['t'] <= span):
                items.append({'dir': x['dir'], 'role': 'combo', 'kind': 'combo', 'tf': x['tf'],
                              't': y['t'], 'level': y.get('level') or x.get('level'),
                              'conf': y['conf'] if y['conf'] is not None else x['conf'],
                              'w': round(min(1.0, READ_BASE['combo'] * (1.0 if (y['conf'] or x['conf']) else 0.8)), 3),
                              'why': '🔥→🧽 кульмінація + поглинання — найсильніший розворот'})
                break
    out = {}
    for d in ('up', 'dn'):
        side = {'rev': 0.0, 'cont': 0.0, 'start': 0.0, 'combo': 0.0, 'exhaust': 0.0,
                'combo_conf': False, 'notes': []}
        for x in items:
            if x['role'] == 'reversal' and x['kind'] == 'climax' and x.get('climax_of') == d:
                side['exhaust'] = max(side['exhaust'], x['w'])
            if x['dir'] != d:
                continue
            if x['role'] == 'combo' and x.get('conf'):
                side['combo_conf'] = True
            key = {'reversal': 'rev', 'shift': 'rev', 'combo': 'combo',
                   'continuation': 'cont', 'start': 'start'}[x['role']]
            side[key] = min(1.0, side[key] + x['w'] if key in ('cont',) else max(side[key], x['w']))
            side['notes'].append(x['why'] + ('' if x['conf'] is None else (' ✓' if x['conf'] else ' ✗')))
        side['rev'] = max(side['rev'], side['combo'])
        out[d] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in side.items()}
    out['items'] = items
    return out


def levels_of(vw: Optional[Dict], extra=None) -> List:
    """Рівні для «на рівні»: межі 24-год коробки живого бару + додаткові
    (OB, High/Low, POC — їх дає той, хто має доступ до сканера)."""
    vw = vw or {}
    lv = [('межа коробки', vw.get('box_hi')), ('межа коробки', vw.get('box_lo'))]
    return lv + list(extra or [])


def scanner_context(sc, symbol: str):
    """(рівні, останній CHoCH) монети з ПАМʼЯТІ сканера — без мережі:
    межі живих OB (internal + swing) і CHoCH внутрішньої структури головного TF.
    Монети немає в скані → ([], None) — тоді працює лише коробка Пружини."""
    lv, ch = [], None
    if sc is None:
        return lv, ch
    S = str(symbol or '').upper()
    try:
        z = (getattr(sc, '_ob_zones', {}) or {}).get(S) or {}
        tf = str(z.get('tf') or '').upper()
        for fam in ('internal', 'swing'):
            for ob in (z.get(fam) or [])[:4]:
                for k in ('bar_high', 'bar_low'):
                    if ob.get(k):
                        lv.append((f'OB {tf}', ob.get(k)))
    except Exception:
        pass
    try:
        an = ((getattr(sc, '_cache', {}) or {}).get(S) or {}).get('analysis') or {}
        ch = last_choch((an.get('internal') or {}).get('events'))
    except Exception:
        ch = None
    return lv, ch


def _tol_of(vw) -> float:
    vw = vw or {}
    atr, px = _f(vw.get('atr')), _f(vw.get('price'))
    if atr and px and px > 0:
        return max(LEVEL_TOL_PCT, 0.5 * atr / px * 100)
    return LEVEL_TOL_PCT


def _fired_side(rec, vw):
    f = (vw or {}).get('fired') or (rec or {}).get('fired')
    return f.get('side') if isinstance(f, dict) else None


def entry_score(side: str, rec: Optional[Dict], vw: Optional[Dict], anoms: List[Dict],
                now: float, mult: Optional[Dict] = None, levels=None,
                choch: Optional[Dict] = None) -> Dict:
    """Бал «розряд почався в бік `side`» 0…100.
    • натяг — частка натягу Пружини в цей бік (prob_long);
    • пробій — ціна живого бару біля / за межею коробки в цей бік;
    • імпульс — діапазон і обсяг живого бару відносно порогів вистрілу
      (лише коли ціна йде саме в цей бік);
    • потік — ⚠ аномалії обсягу в цей бік.
    `started` — розряд справді почався (не лише «накручено»); `veto` — ціна
    пробиває коробку В ІНШИЙ бік або сильний зустрічний потік."""
    sd = str(side).upper()
    rec, vw = rec or {}, vw or {}
    try:
        pl = float(rec.get('prob_long'))
    except (TypeError, ValueError):
        pl = 50.0
    p = pl / 100.0 if sd == 'LONG' else 1.0 - pl / 100.0
    fs = _fired_side(rec, vw)
    lean = 1.0 if fs == sd else _c((p - 0.5) * 2)
    pos = box_pos(vw, sd)
    if vw.get('beyond') == sd or fs == sd:
        brk = 1.0
    else:
        brk = _c(pos) * 0.6 if pos is not None else 0.0
    rx, vx = vw.get('range_x'), vw.get('vol_x')
    rn, vn = vw.get('range_need') or 1.8, vw.get('vol_need') or 1.8
    rpart = _c(float(rx) / float(rn)) if rx is not None else 0.0
    vpart = _c(float(vx) / float(vn)) if vx is not None else 0.0
    toward = (pos is not None and pos > 0) or vw.get('beyond') == sd or fs == sd
    imp = (0.5 * rpart + 0.5 * vpart) if toward else 0.0
    # 📖 Потік — як людина читає аномалії: розворот / старт / продовження В бік
    # угоди мінус те саме ПРОТИ неї (рівні, тренд до, підтвердження, 🔥→🧽).
    d = 'up' if sd == 'LONG' else 'dn'
    rd = read_anomalies(anoms, now, levels_of(vw, levels), choch,
                        _cal_part(mult, 'entry'), tol_pct=_tol_of(vw))
    me, ot = rd[d], rd['dn' if d == 'up' else 'up']
    pro = _c(max(me['rev'], 0.8 * me['start']) + 0.5 * me['cont'])
    con = _c(max(ot['rev'], 0.7 * ot['cont'], 0.7 * ot['start']))
    fl = max(-1.0, min(1.0, pro - con))
    notes = list(me['notes']) + [f'проти: {n}' for n in ot['notes']]
    parts = {'lean': round(lean, 3), 'break': round(brk, 3), 'impulse': round(imp, 3),
             'flow': round(max(0.0, fl), 3)}
    score = round(100.0 * sum(W_ENTRY[k] * parts[k] for k in W_ENTRY), 1)
    veto = ''
    if vw.get('beyond') == _opp(sd) or fs == _opp(sd):
        veto = 'ціна пробиває коробку в ПРОТИЛЕЖНИЙ бік'
    elif ot['combo'] >= 0.8:
        veto = '🔥→🧽 зустрічний сценарій розвороту'
    elif fl <= FLOW_VETO:
        veto = 'сильний зустрічний сигнал обсягу (розворот / тиск проти)'
    started = bool(vw.get('beyond') == sd or fs == sd
                   or (pos is not None and pos > 0.5 and (rpart >= 0.8 or vpart >= 0.8)))
    if me['combo_conf'] and started and not veto:
        # 🔥→🧽 розворот У бік угоди, підтверджений, і розряд почався
        score = max(score, COMBO_ENTRY_SCORE)
    return {'side': sd, 'score': score, 'parts': parts, 'flow_raw': round(fl, 3),
            'flow_notes': notes, 'started': started, 'veto': veto,
            'combo': me['combo'],
            'ready': bool(started and not veto), 'pos': None if pos is None else round(pos, 2)}


def exit_score(side: str, vw: Optional[Dict], anoms: List[Dict], now: float,
               exh: Optional[Dict] = None, pnl: Optional[float] = None,
               peak: Optional[float] = None, mult: Optional[Dict] = None,
               levels=None, choch: Optional[Dict] = None) -> Dict:
    """Бал «почались зворотні процеси» для угоди в бік `side` 0…100.
    • розворот — ціна пробиває коробку / вистрілила ПРОТИ угоди, або йде до
      протилежної межі; • потік — ⚠ аномалії проти угоди; • кульмінація —
      сплеск живого бару В БІК угоди (енергію витрачено); • ціль — пройдено
      ≥70% шляху до цілі Пружини; • віддача — угода віддала частину піку.
    `ready` — щонайменше EXIT_MIN_PARTS свідчень сильніші за 0.3."""
    sd = str(side).upper()
    op = _opp(sd)
    vw = vw or {}
    fs = _fired_side(None, vw)
    if vw.get('beyond') == op or fs == op:
        against = 1.0
    else:
        po = box_pos(vw, op)
        against = _c(po) * 0.6 if po is not None and po > 0 else 0.0
    xm = _cal_part(mult, 'exit')
    d = 'up' if sd == 'LONG' else 'dn'
    rd = read_anomalies(anoms, now, levels_of(vw, levels), choch, xm, tol_pct=_tol_of(vw))
    me, ot = rd[d], rd['dn' if d == 'up' else 'up']
    # потік ПРОТИ угоди: 🧽 / ⚡-зсув / 🔥→🧽 у протилежний бік + тиск проти
    fl = _c(ot['rev'] + 0.5 * ot['cont'] + 0.5 * ot['start'])
    notes = [f'проти: {n}' for n in ot['notes']]
    # кульмінація: сплеск живого бару В БІК угоди або 🔥 після тривалого руху
    # у бік угоди (виснаження), вага — частка розворотів після 🔥 з досліджень
    climax = max(1.0 if vw.get('climax') == sd else 0.0, me['exhaust'])
    if me['exhaust']:
        notes.append('🔥 виснаження руху за угодою')
    # продовження В бік угоди (⚡/📈 за угодою, підтверджене) — тримаємо
    hold = _c(me['cont'] + 0.5 * me['start'])
    if hold:
        notes.append('за угодою: ' + ', '.join(me['notes'][:2]))
    target = 0.0
    try:
        if exh and exh.get('pct') is not None:
            target = _c((float(exh['pct']) - 70.0) / 30.0)
    except (TypeError, ValueError):
        target = 0.0
    fade = 0.0
    try:
        if peak is not None and pnl is not None and float(peak) >= 0.6:
            share = (float(peak) - float(pnl)) / float(peak)
            fade = _c((share - 0.3) / 0.4)
    except (TypeError, ValueError, ZeroDivisionError):
        fade = 0.0
    parts = {'against': round(against, 3), 'flow': round(max(0.0, fl), 3),
             'climax': climax, 'target': round(target, 3), 'fade': round(fade, 3)}
    score = round(max(0.0, 100.0 * sum(W_EXIT[k] * parts[k] for k in W_EXIT)
                      - HOLD_PENALTY * hold), 1)
    strong = [k for k, v in parts.items() if v > 0.3]
    combo = ot['combo'] >= 0.8
    ready = len(strong) >= EXIT_MIN_PARTS and hold < 0.6
    if ot['combo_conf']:
        # 🔥→🧽 проти угоди, ПІДТВЕРДЖЕНИЙ свічкою / CHoCH — найсильніший
        # сценарій розвороту: виходимо, продовження вже не тримає.
        score, ready = max(score, COMBO_EXIT_SCORE), True
    elif combo:
        score = max(score, COMBO_WAIT_SCORE)   # сценарій є, чекаємо підтвердження
    return {'side': sd, 'score': score, 'parts': parts, 'flow_notes': notes,
            'strong': strong, 'hold': round(hold, 3), 'combo': combo,
            'combo_conf': ot['combo_conf'], 'ready': ready}


def parts_text(parts: Dict) -> str:
    return ' · '.join(f"{PART_UA.get(k, k)} {int(round(float(v) * 100))}%"
                      for k, v in (parts or {}).items() if float(v or 0) > 0)


def _cal_part(cal, part):
    """Частина калібровки ('entry'/'exit'); плаский словник (стара форма) — як є."""
    if not isinstance(cal, dict):
        return {}
    if 'entry' in cal or 'exit' in cal:
        return cal.get(part) or {}
    return cal


def calibration(stats: List[Dict]) -> Dict:
    """🎓 Ваги з ДОСЛІДЖЕНЬ «⚠ Аномалії обсягу» (`vol_anomaly.stats`):
    • entry[kind] і entry[kind|tf] — частка «ціна через 4 свічки пішла в бік
      підказки» / 50% (⚡ перекіс, 🧽 поглинання) — наскільки аномалія справді
      вказує бік розряду;
    • exit[…] — те саме для зустрічних аномалій + exit[climax] = частка
      «після 🔥 кульмінації рух розвернувся» / 50%;
    межі 0.5…1.5, випадків < CAL_MIN_N → 1.0 (не вигадуємо). `n` — обсяг вибірки."""
    ent: Dict = {}
    rev: Dict = {}
    for r in stats or []:
        k, tf = r.get('kind'), r.get('tf')
        n, pct = int(r.get('hint_n') or 0), r.get('hint_pct')
        if k in FLOW_W and pct is not None and n > 0:
            for key in (k, f'{k}|{tf}'):
                a = ent.setdefault(key, [0, 0.0])
                a[0] += n
                a[1] += n * float(pct)
        dn, rp = int(r.get('decided') or 0), r.get('reversal_pct')
        if k == 'climax' and rp is not None and dn > 0:
            for key in (k, f'{k}|{tf}'):
                a = rev.setdefault(key, [0, 0.0])
                a[0] += dn
                a[1] += dn * float(rp)

    def _m(acc):
        return {k: (round(_c((s / n) / 50.0, 0.5, 1.5), 3) if n >= CAL_MIN_N else 1.0)
                for k, (n, s) in acc.items()}
    em = _m(ent)
    xm = dict(em)
    xm.update(_m(rev))
    return {'entry': em, 'exit': xm,
            'n': {k: n for k, (n, _s) in {**ent, **rev}.items()}}


# ── епізоди: заряджена монета від заряду до розряду і наслідку ──
def _px(vw, rec):
    for src in (vw or {}, rec or {}):
        try:
            v = float(src.get('price'))
            if v > 0:
                return v
        except (TypeError, ValueError):
            continue
    return None


def episode_new(symbol: str, now: float, rec: Dict, vw: Optional[Dict], src: str) -> Dict:
    px = _px(vw, rec)
    return {'symbol': symbol, 'src': src, 'started_at': int(now), 'seen_at': int(now),
            'charge0': rec.get('charge'), 'price0': px, 'lean0': rec.get('lean'),
            'prob0': rec.get('prob_long'), 'hi': px, 'lo': px, 'charge_max': rec.get('charge'),
            'fire': None, 'signal': None, 'tries': {}, 's': [], 'ev': [], 'charged_at': int(now)}


def _ev(ep, now, kind, **kw):
    e = {'t': int(now), 'k': kind}
    e.update({k: v for k, v in kw.items() if v is not None})
    ep.setdefault('ev', []).append(e)
    if len(ep['ev']) > EVENTS_MAX:
        del ep['ev'][:len(ep['ev']) - EVENTS_MAX]


def episode_step(ep: Dict, now: float, rec: Dict, vw: Optional[Dict], el: Dict, es: Dict,
                 charged: bool) -> None:
    """Оновити епізод: екстремуми ціни, заряд, момент розряду, супровід сигналу."""
    px = _px(vw, rec)
    ep['seen_at'] = int(now)
    if charged:
        ep['charged_at'] = int(now)
    if px:
        ep['hi'] = max(ep.get('hi') or px, px)
        ep['lo'] = min(ep.get('lo') or px, px)
    try:
        ep['charge_max'] = max(float(ep.get('charge_max') or 0), float(rec.get('charge') or 0))
    except (TypeError, ValueError):
        pass
    fs = _fired_side(rec, vw)
    if fs and not ep.get('fire'):
        ep['fire'] = {'t': int(now), 'side': fs, 'px': px}
        _ev(ep, now, 'fire', side=fs, px=px)
    for sd, sc in (('LONG', el), ('SHORT', es)):
        k = 'start_' + sd
        if sc.get('started') and not sc.get('veto') and k not in ep:
            ep[k] = int(now)
            _ev(ep, now, 'discharge', side=sd, px=px, score=sc.get('score'),
                parts=sc.get('parts'))
    sig = ep.get('signal')
    if sig and px and sig.get('px'):
        m = (px / sig['px'] - 1) * 100 * (1 if sig['side'] == 'LONG' else -1)
        sig['best'] = round(max(float(sig.get('best', m)), m), 3)
        sig['worst'] = round(min(float(sig.get('worst', m)), m), 3)
        sig['last'] = round(m, 3)
    s = ep.setdefault('s', [])
    if not s or now - float(s[-1][0]) >= SAMPLE_SEC:
        s.append([int(now), px, rec.get('charge'), rec.get('state'), el.get('score'),
                  es.get('score'), (vw or {}).get('range_x'), (vw or {}).get('vol_x'),
                  (vw or {}).get('beyond'), (vw or {}).get('climax'), el.get('flow_raw')])
        if len(s) > SAMPLES_MAX:
            ep['s'] = s[::2]


SAMPLE_KEYS = ('t', 'px', 'charge', 'state', 'score_long', 'score_short', 'range_x',
               'vol_x', 'beyond', 'climax', 'flow')


def episode_done(ep: Dict, now: float, charged: bool) -> str:
    """Причина закриття епізоду або ''."""
    if now - float(ep.get('started_at') or now) >= EPISODE_MAX_SEC:
        return 'max'
    anchor = max(float((ep.get('fire') or {}).get('t') or 0),
                 float((ep.get('signal') or {}).get('t') or 0))
    if anchor:
        return 'after' if now - anchor >= EPISODE_AFTER_SEC else ''
    if not charged and now - float(ep.get('charged_at') or now) >= EPISODE_IDLE_SEC:
        return 'idle'
    return ''


def episode_outcome(ep: Dict) -> Dict:
    p0, hi, lo = ep.get('price0'), ep.get('hi'), ep.get('lo')
    last = None
    for row in reversed(ep.get('s') or []):
        if row[1]:
            last = row[1]
            break
    out = {}
    try:
        p0 = float(p0)
        out['max_up_pct'] = round((float(hi) / p0 - 1) * 100, 3)
        out['max_dn_pct'] = round((float(lo) / p0 - 1) * 100, 3)
        if last:
            out['move_pct'] = round((float(last) / p0 - 1) * 100, 3)
        out['discharge'] = ('LONG' if out['max_up_pct'] >= -out['max_dn_pct'] else 'SHORT')
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    sig = ep.get('signal') or {}
    if sig:
        out.update({'sig_best_pct': sig.get('best'), 'sig_worst_pct': sig.get('worst'),
                    'sig_end_pct': sig.get('last')})
    return out


# ───────────────────────── РУШІЙ ─────────────────────────
def _scanner():
    try:
        from detection.smc_scanner import get_smc_scanner
        return get_smc_scanner()
    except Exception:
        return None


def _spring_rows():
    import sys
    m = sys.modules.get('detection.spring_scan')
    eng = m.get_spring_scan() if m and hasattr(m, 'get_spring_scan') else None
    try:
        return list(eng.rows()) if eng else []
    except Exception:
        return []


class AiEngine:
    def __init__(self, db=None, md=None, scanner=None, tm=None, clock=time.time):
        self.db = db
        self._md = md
        self._sc = scanner
        self._tm = tm
        self._clock = clock
        self._lock = threading.Lock()
        self._eps: Dict[str, Dict] = {}
        self._live: Dict[str, tuple] = {}     # S → (ts, rec, vw)
        self._view: Dict[str, Dict] = {}      # S → останні бали (для UI)
        self._cal: Dict = {}
        self._cal_at = 0.0
        self._stat = {'last_at': 0, 'coins': 0, 'signals': 0, 'episodes_closed': 0,
                      'errors': 0, 'last_err': ''}
        self._dirty = False
        self._thread = None
        self._load()

    # джерела
    def _scn(self):
        return self._sc or _scanner()

    def settings(self) -> Dict:
        s = dict(DEFAULTS)
        try:
            st = self._scn().get_settings() or {}
            s.update({k: st[k] for k in DEFAULTS if k in st})
        except Exception:
            pass
        return s

    def _md_get(self):
        if self._md is None:
            from detection.market_data import get_market_data
            self._md = get_market_data()
        return self._md

    def _tm_get(self):
        if self._tm is not None:
            return self._tm
        try:
            from detection.trade_manager import get_trade_manager
            return get_trade_manager()
        except Exception:
            return None

    def calibration(self) -> Dict:
        return {k: dict(v) for k, v in self._cal.items()} if self._cal else {}

    def _refresh_cal(self, now):
        if now - self._cal_at < CAL_REFRESH_SEC or not self.db:
            return
        self._cal_at = now
        try:
            from detection.vol_anomaly import stats
            rows = self.db.get_vol_anomalies(since=now - CAL_DAYS * 86400, limit=50000) or []
            self._cal = calibration(stats(rows))
        except Exception as e:
            self._stat['last_err'] = f'калібровка: {e}'[:200]

    def candidates(self, min_charge: float) -> Dict[str, tuple]:
        """{S: (запис Пружини, джерело)} — заряджені монети обох Пружин +
        монети з відкритим епізодом (ведемо до наслідку)."""
        out: Dict[str, tuple] = {}
        sc = self._scn()
        try:
            uni = sc.scan_universe() if sc and hasattr(sc, 'scan_universe') else None
            recs = sc.spring_for(uni) if sc else {}
        except Exception:
            recs = {}
        for S, r in (recs or {}).items():
            if r.get('state') == 'charged' or float(r.get('charge') or 0) >= min_charge:
                out[S] = (r, 'wl')
        for r in _spring_rows():
            S = r.get('symbol')
            if S and S not in out and r.get('bybit') is not False:
                out[S] = (r, 'scan')
        for S, ep in list(self._eps.items()):
            if S not in out:
                out[S] = ({}, ep.get('src') or 'wl')
        ranked = sorted(out.items(), key=lambda kv: (kv[0] not in self._eps,
                                                      -float(kv[1][0].get('charge') or 0)))
        return dict(ranked[:MAX_COINS])

    def _fetch_live(self, S, now):
        hit = self._live.get(S)
        if hit and now - hit[0] < LIVE_TTL:
            return hit[1], hit[2]
        from detection import spring_coil
        kl = self._md_get().fetch_klines(S, limit=300, interval='1h') or []
        if len(kl) < 2:
            return (hit[1], hit[2]) if hit else (None, None)
        rec = spring_coil.analyze(kl[:-1])
        vw = spring_coil.live_fire_view(kl)
        rec = rec if rec.get('ok') else None
        self._live[S] = (now, rec, vw)
        return rec, vw

    def tick(self, now: Optional[float] = None) -> int:
        now = self._clock() if now is None else now
        s = self.settings()
        sc = self._scn()
        if sc is None or not (sc.get_settings() or {}).get('spring_enabled', True):
            return 0
        self._refresh_cal(now)
        min_charge = float(s.get('ai_min_charge') or 65)
        cand = self.candidates(min_charge)
        anoms: Dict[str, List[Dict]] = {}
        try:
            for a in (self.db.get_vol_anomalies(since=now - ANOM_LOOK_SEC, limit=5000)
                      if self.db else []) or []:
                anoms.setdefault(a.get('symbol'), []).append(a)
        except Exception:
            pass

        def load(S):
            try:
                return S, self._fetch_live(S, now)
            except Exception as e:
                self._stat['errors'] += 1
                self._stat['last_err'] = f'{S}: {e}'[:200]
                return S, (None, None)

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            lives = dict(ex.map(load, list(cand.keys())))
        view = {}
        for S, (rec0, src) in cand.items():
            rec, vw = lives.get(S) or (None, None)
            rec = rec or rec0 or {}
            if not rec:
                continue
            charged = rec.get('state') == 'charged' or float(rec.get('charge') or 0) >= min_charge
            al = anoms.get(S) or []
            lv, ch = scanner_context(self._scn(), S)
            el = entry_score('LONG', rec, vw, al, now, self._cal, lv, ch)
            es = entry_score('SHORT', rec, vw, al, now, self._cal, lv, ch)
            ep = self._eps.get(S)
            if ep is None and charged:
                ep = self._eps[S] = episode_new(S, now, rec, vw, src)
                _ev(ep, now, 'charged', charge=rec.get('charge'), lean=rec.get('lean'))
            if ep is not None:
                episode_step(ep, now, rec, vw, el, es, charged)
                self._dirty = True
            best = el if el['score'] >= es['score'] else es
            view[S] = {'symbol': S, 'src': src, 'charge': rec.get('charge'),
                       'state': rec.get('state'), 'lean': rec.get('lean'),
                       'prob_long': rec.get('prob_long'), 'long': el, 'short': es,
                       'best': best['side'], 'episode_age': (int(now - ep['started_at'])
                                                             if ep else None),
                       'signal': (ep or {}).get('signal')}
            if (s.get('ai_signal_enabled') and ep is not None and best['ready']
                    and best['score'] >= float(s.get('ai_entry_min') or 70)):
                self._try_signal(S, best, rec, vw, ep, now)
            if ep is not None:
                why = episode_done(ep, now, charged)
                if why:
                    self._close_episode(S, ep, now, why)
        with self._lock:
            self._view = view
        self._stat.update({'last_at': now, 'coins': len(view)})
        if self._dirty:
            self._persist()
        return len(view)

    def _try_signal(self, S, sc_res, rec, vw, ep, now):
        side = sc_res['side']
        if ep.get('signal'):
            return                          # один результативний сигнал на епізод
        tr = ep.setdefault('tries', {}).get(side) or [0, 0]
        if tr[0] >= SIGNAL_TRIES or now - tr[1] < SIGNAL_RETRY_SEC:
            return
        scn = self._scn()
        try:
            from detection.activity_log import log_activity
        except Exception:
            def log_activity(*a, **k):
                return None
        try:
            if scn is not None and hasattr(scn, '_dir_paused') and scn._dir_paused(S, side):
                return                      # ⏸ кнопка напрямку вимкнена — не опрацьовано
            ok, reason, detail = scn._signal_allowed(S, side, at_intake=True, origin='ai')
        except Exception as e:
            self._stat['last_err'] = f'ворота {S}: {e}'[:200]
            return
        ep['tries'][side] = [tr[0] + 1, int(now)]
        px = _px(vw, rec)
        head = (f"🤖 Сигнал ШІ {side} · бал {sc_res['score']} · розряд Пружини "
                f"(заряд {rec.get('charge')}) · {parts_text(sc_res['parts'])}")
        log_activity(S, 'signal', f'{head} · {detail}', side=side, source='AI',
                     extra={'ai_score': sc_res['score'], 'ai_parts': sc_res['parts']})
        if not ok:
            log_activity(S, 'rejected', reason, side=side, source='AI')
            _ev(ep, now, 'signal_rejected', side=side, score=sc_res['score'], reason=reason)
            return
        tm = self._tm_get()
        res = None
        if tm is not None:
            try:
                res = tm.on_signal(symbol=S, side=side, entry_price=px or 0, opened_by='ai')
            except Exception as e:
                self._stat['last_err'] = f'on_signal {S}: {e}'[:200]
        ep['signal'] = {'t': int(now), 'side': side, 'px': px, 'score': sc_res['score'],
                        'parts': sc_res['parts'], 'result': str(res)[:80] if res else None}
        _ev(ep, now, 'signal', side=side, px=px, score=sc_res['score'])
        self._stat['signals'] += 1

    def _close_episode(self, S, ep, now, why):
        self._eps.pop(S, None)
        ep['ended_at'] = int(now)
        ep['end_reason'] = why
        out = episode_outcome(ep)
        try:
            if self.db and hasattr(self.db, 'archive_ai_episode'):
                self.db.archive_ai_episode(ep, out)
        except Exception as e:
            self._stat['last_err'] = f'архів {S}: {e}'[:200]
        self._stat['episodes_closed'] += 1
        self._dirty = True

    # стан
    def _persist(self):
        self._dirty = False
        if not self.db:
            return
        try:
            self.db.set_setting(DB_STATE, {'eps': self._eps})
        except Exception:
            pass

    def _load(self):
        if not self.db:
            return
        try:
            st = self.db.get_setting(DB_STATE, None) or {}
            if isinstance(st, str):
                st = json.loads(st)
            eps = st.get('eps') or {}
            if isinstance(eps, dict):
                self._eps = {str(k).upper(): v for k, v in eps.items() if isinstance(v, dict)}
        except Exception:
            self._eps = {}

    def state(self) -> Dict:
        with self._lock:
            view = sorted(self._view.values(),
                          key=lambda v: -max(v['long']['score'], v['short']['score']))
        return {'ok': True, 'settings': self.settings(), 'stat': dict(self._stat),
                'calibration': self.calibration(), 'coins': view,
                'episodes_open': len(self._eps), 'server_now': self._clock(),
                'weights': {'entry': W_ENTRY, 'exit': W_EXIT}}

    def _loop(self):
        time.sleep(45)
        while True:
            try:
                self.tick()
            except Exception as e:
                self._stat['errors'] += 1
                self._stat['last_err'] = str(e)[:200]
                print(f'[AI] tick error: {e}')
            time.sleep(TICK_SEC)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name='ai-engine', daemon=True)
        self._thread.start()


_instance: Optional[AiEngine] = None


def init_ai_engine(db=None, start_loop: bool = True, **kw) -> AiEngine:
    global _instance
    if _instance is None:
        _instance = AiEngine(db=db, **kw)
        if start_loop:
            _instance.start()
    return _instance


def get_ai_engine() -> Optional[AiEngine]:
    return _instance
