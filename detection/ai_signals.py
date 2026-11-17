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
DEFAULTS = {'ai_signal_enabled': False, 'ai_min_charge': 65,
            'ai_professional_enabled': False, 'ai_professional_require_model': False,
            'ai_learning_enabled': True}
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
DB_PRO_LEARNING = 'ai_professional_learning_v1'
PRO_BAR_LIMIT = 300
PRO_CACHE_TTL = {'15m': 240, '1h': 900, '4h': 1800}
PRO_ANALYSIS_TTL = 45
LEARN_REFRESH_SEC = 900
LEARN_ROWS = 5000            # найновіших рядків на джерело (епізоди / дослідження)
PRO_TFS = ('15m', '1h', '4h')
PRO_OPEN_CAP = 60            # стеля монет відкритих угод для префетчу барів

W_ENTRY = {'lean': 0.25, 'break': 0.30, 'impulse': 0.25, 'flow': 0.20}
W_EXIT = {'against': 0.25, 'choch': 0.10, 'flow': 0.20, 'climax': 0.20,
          'target': 0.125, 'fade': 0.125}
FLOW_W = {'side': 1.0, 'absorb': 0.8}      # аномалії з напрямком
FLOW_VETO = -0.6
EXIT_MIN_PARTS = 2          # вихід — щонайменше два НЕЗАЛЕЖНІ свідчення
HOLD_PENALTY = 15           # продовження в бік угоди знижує бал виходу
COMBO_EXIT_SCORE = 90       # підтверджений 🔥→🧽 проти угоди
COMBO_WAIT_SCORE = 60       # 🔥→🧽 є, підтвердження ще немає
COMBO_ENTRY_SCORE = 85      # підтверджений 🔥→🧽 у бік розряду
# 🤖 Вихід під контролем ⚠ аномалій обсягу (вимога 17.11): без свідчення
# аномалій бот НЕ виходить; коли «кажуть закривати» ВСІ групи — максимум.
ANOM_GATE = 0.3             # мін. сила «аномалії кажуть закривати»
ALL_EXIT_SCORE = 95         # аномалії + ціна + прибуток — усі за вихід
SCENARIO_EXIT_SCORE = 92    # 🧩 сценарій виходу завершено (комплекс реакцій)
EXIT_LOOK_MAX = 24 * 3600   # аномалії від входу (сценарії виходу), не давніші за добу
HOLD_BLOCK = 0.6            # продовження за угодою ≥ цього — тримаємо
ANOM_PARTS = ('flow', 'climax')
PRICE_PARTS = ('against', 'choch')
PROFIT_PARTS = ('target', 'fade')
EXIT_VERDICT_UA = {'scenario': '🧩 сценарій виходу завершено',
                   'all': '✅ усі показники за вихід', 'combo': '🔥→🧽 підтверджений розворот',
                   'close': '⚠ аномалії + підтвердження — вихід',
                   'hold': '⚡ продовження за угодою — тримаємо',
                   'watch': '👁 аномалії є, підтвердження ще немає',
                   'wait': '⏳ ціна / прибуток кажуть «вийти», аномалії — ні',
                   'none': '— сигналів виходу немає'}
EXIT_CHECK_UA = {'flow': '📖 аномалії проти угоди', 'climax': '🔥 виснаження / кульмінація за угодою',
                 'against': '↩ ціна проти (коробка / вистріл)', 'choch': '🔀 CHoCH проти угоди',
                 'target': '🎯 ціль Пружини пройдена', 'fade': '📉 віддача від піку',
                 'hold': '⚡ продовження за угодою'}
PART_UA = {'lean': 'натяг', 'break': 'пробій коробки', 'impulse': 'імпульс',
           'flow': 'потік обсягу', 'against': 'розворот ціни', 'climax': 'кульмінація',
           'target': 'ціль Пружини', 'fade': 'віддача від піку', 'choch': 'CHoCH проти'}


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
        if now - t > 4 * 3600:
            w *= 0.7                          # вихід дивиться від входу — старі слабшають
        items.append(dict(c, t=t, kind=k, tf=tf, w=round(min(1.0, w), 3), conf=conf,
                          price=_f(a.get('price')), o1=o1, o4=_f(a.get('o4'))))
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


def _sib(name: str):
    """Сусідній модуль detection/<name>.py: пакетний імпорт, а якщо він
    недоступний (ізольовані тести без pybit) — напряму з файлу."""
    import sys
    full = f'detection.{name}'
    m = sys.modules.get(full)
    if m is None:
        try:
            import importlib
            m = importlib.import_module(full)
        except Exception:
            import importlib.util, os
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{name}.py')
            sp = importlib.util.spec_from_file_location(full, p)
            m = importlib.util.module_from_spec(sp)
            sys.modules[full] = m
            sp.loader.exec_module(m)
    return m


def _scen():
    return _sib('anomaly_scenarios')


def entry_score(side: str, rec: Optional[Dict], vw: Optional[Dict], anoms: List[Dict],
                now: float, mult: Optional[Dict] = None, levels=None,
                choch: Optional[Dict] = None, charged: Optional[bool] = None) -> Dict:
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
    # 🧩 КОМПЛЕКС РЕАКЦІЙ (17.11): весь процес до точки входу — сценарії за
    # аномаліями до 12 год (не одна позначка). Рішення про вхід — ЛИШЕ за
    # готовим сценарієм (`entry_decide`), бал — довідка.
    sm = _scen()
    if charged is None:
        charged = (rec.get('state') == 'charged')
    rs = read_anomalies(anoms, now, levels_of(vw, levels), choch, _cal_part(mult, 'entry'),
                        look=sm.SCEN_LOOK_SEC, tol_pct=_tol_of(vw))
    ctx = {'price': _f(vw.get('price')), 'beyond': vw.get('beyond'), 'fired': fs,
           'charged': bool(charged), 'bar_t': _f(vw.get('t')),
           'box_hi': _f(vw.get('box_hi')), 'box_lo': _f(vw.get('box_lo'))}
    scens = sm.match_entry(rs['items'], sd, ctx, choch, now)
    top = sm.best(scens)
    ready = bool(top and top.get('ready') and not veto)
    return {'side': sd, 'score': score, 'parts': parts, 'flow_raw': round(fl, 3),
            'flow_notes': notes, 'started': started, 'veto': veto,
            'combo': me['combo'],
            'ready': ready, 'pos': None if pos is None else round(pos, 2),
            'scenario': sm.view(top), 'scenario_text': sm.stages_text(top),
            'scenarios': [sm.view(x) for x in scens]}


def entry_decide(r: Dict) -> bool:
    """🤖 Вхід — лише коли КОМПЛЕКС реакцій склався в готовий сценарій
    (свіжа точка входу, без зламу) і немає вето. Порогу балу немає."""
    sc = (r or {}).get('scenario') or {}
    return bool(r and not r.get('veto') and sc.get('ready') and not sc.get('broken'))


def exit_score(side: str, vw: Optional[Dict], anoms: List[Dict], now: float,
               exh: Optional[Dict] = None, pnl: Optional[float] = None,
               peak: Optional[float] = None, mult: Optional[Dict] = None,
               levels=None, choch: Optional[Dict] = None,
               opened_at: Optional[float] = None, look: Optional[float] = None) -> Dict:
    """Бал «почались зворотні процеси» для угоди в бік `side` 0…100 — ПІД
    КОНТРОЛЕМ ⚠ АНОМАЛІЙ ОБСЯГУ (вимога 17.11). Три групи свідчень:
    • 📖 аномалії (`flow` — 🧽/⚡-зсув/🔥→🧽 і тиск ПРОТИ угоди; `climax` —
      🔥 виснаження руху за угодою або сплеск живого бару за угодою);
    • ціна (`against` — коробка пробита / пружина вистрілила проти, або ціна йде
      до протилежної межі; `choch` — CHoCH проти угоди ПІСЛЯ входу);
    • прибуток (`target` — ≥70% шляху до цілі Пружини; `fade` — віддача піку).
    `hold` — ⚡/📈 продовження ЗА угодою (свіжі 2 год) — тримаємо.
    Рішення (`verdict`): вихід лише коли аномалії кажуть «закривати»
    (≥ ANOM_GATE) І є підтвердження з ціни або прибутку, а продовження не
    тримає; ВСІ три групи за вихід → `all` (бал ≥ ALL_EXIT_SCORE);
    підтверджений 🔥→🧽 проти угоди → `combo` (вихід). Без аномалій ціна й
    прибуток лише `wait` — для них є інші правила виходу (SL, трейли).
    ⚠️ ЛИШЕ НОВЕ ПІСЛЯ ВХОДУ (`opened_at`): аномалії до входу і сплеск свічки,
    на якій угоду відкрито, — це те, ЧОМУ увійшли, а не причина виходити
    (інакше щойно відкрита угода тут же «мала б закриватись»). Вихід можливий
    не раніше, ніж зʼявиться нова аномалія — закриття наступної 15m/1h свічки."""
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
    od = 'dn' if d == 'up' else 'up'
    lk = max(ANOM_LOOK_SEC, float(look or ANOM_LOOK_SEC))
    oa = _f(opened_at)
    after = [a for a in (anoms or []) if oa is None or (_f(a.get('ts')) or 0) >= oa]
    rd = read_anomalies(after, now, levels_of(vw, levels), choch, xm, look=lk,
                        tol_pct=_tol_of(vw))
    me, ot = rd[d], rd[od]
    sm = _scen()
    xs = sm.match_exit(rd['items'], sd, {'price': _f(vw.get('price')), 'beyond': vw.get('beyond'),
                                         'box_hi': _f(vw.get('box_hi')),
                                         'box_lo': _f(vw.get('box_lo'))},
                       choch, now, opened_at=oa)
    xdone = [x for x in xs if x.get('complete')]
    xtop = max(xdone, key=lambda x: x['total']) if xdone else sm.best(xs)
    # продовження за угодою — СВІЖЕ (2 год, зокрема й до входу): воно тримає
    rr = read_anomalies(anoms, now, levels_of(vw, levels), choch, xm,
                        look=ANOM_LOOK_SEC, tol_pct=_tol_of(vw))
    mr = rr[d]
    fl = _c(max(ot['rev'], ot['combo']) + 0.5 * ot['cont'] + 0.5 * ot['start'])
    notes = [f'проти: {n}' for n in ot['notes']]
    # сплеск живого бару за угодою — лише на свічці, що ВІДКРИЛАСЬ після входу
    bt = _f(vw.get('t'))
    live_cl = vw.get('climax') == sd and (oa is None or (bt is not None and bt >= oa))
    climax = max(1.0 if live_cl else 0.0, me['exhaust'])
    if me['exhaust']:
        notes.append('🔥 виснаження руху за угодою')
    if live_cl:
        notes.append('💥 сплеск живого бару за угодою')
    hold = _c(mr['cont'] + 0.5 * mr['start'])
    if hold:
        notes.append('за угодою: ' + ', '.join(mr['notes'][:2]))
    ch = 0.0
    if choch and choch.get('dir') == od:
        t = _f(choch.get('t')) or 0.0
        if (opened_at is None and now - t <= lk) or (opened_at is not None and t >= float(opened_at)):
            ch = 1.0
            notes.append('🔀 CHoCH проти угоди')
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
    parts = {'against': round(against, 3), 'choch': ch, 'flow': round(max(0.0, fl), 3),
             'climax': round(climax, 3), 'target': round(target, 3), 'fade': round(fade, 3)}
    score = round(max(0.0, 100.0 * sum(W_EXIT[k] * parts[k] for k in W_EXIT)
                      - HOLD_PENALTY * hold), 1)
    strong = [k for k, v in parts.items() if v > 0.3]
    anom = max(parts[k] for k in ANOM_PARTS)
    anom_ok = anom >= ANOM_GATE
    price_ok = any(parts[k] > 0.3 for k in PRICE_PARTS)
    profit_ok = any(parts[k] > 0.3 for k in PROFIT_PARTS)
    combo = ot['combo'] >= 0.8
    held = hold >= HOLD_BLOCK
    if xdone:
        # 🧩 Сценарій виходу завершено — КОМПЛЕКС реакцій проти угоди склався.
        verdict, ready = 'scenario', True
        score = max(score, SCENARIO_EXIT_SCORE)
    elif ot['combo_conf']:
        # 🔥→🧽 проти угоди, ПІДТВЕРДЖЕНИЙ свічкою / CHoCH — найсильніший
        # сценарій розвороту: виходимо, продовження вже не тримає.
        verdict, ready = 'combo', True
        score = max(score, COMBO_EXIT_SCORE)
    elif anom_ok and price_ok and profit_ok and hold < ANOM_GATE:
        verdict, ready = 'all', True
        score = max(score, ALL_EXIT_SCORE)
    elif held:
        verdict, ready = 'hold', False
    elif anom_ok and (price_ok or profit_ok):
        verdict, ready = 'close', True
    elif anom_ok:
        verdict, ready = 'watch', False
        if combo:
            score = max(score, COMBO_WAIT_SCORE)   # сценарій є, чекаємо підтвердження
    elif price_ok or profit_ok:
        verdict, ready = 'wait', False
    else:
        verdict, ready = 'none', False
    checklist = [{'k': k, 'label': EXIT_CHECK_UA[k], 'v': parts[k],
                  'say': 'close' if parts[k] > 0.3 else ''} for k in
                 ('flow', 'climax', 'against', 'choch', 'target', 'fade')]
    checklist.append({'k': 'hold', 'label': EXIT_CHECK_UA['hold'], 'v': round(hold, 3),
                      'say': 'hold' if hold >= ANOM_GATE else ''})
    return {'side': sd, 'score': score, 'parts': parts, 'flow_notes': notes,
            'strong': strong, 'hold': round(hold, 3), 'combo': combo,
            'combo_conf': ot['combo_conf'], 'ready': ready, 'verdict': verdict,
            'verdict_ua': EXIT_VERDICT_UA[verdict], 'anom': round(anom, 3),
            'groups': {'anom': anom_ok, 'price': price_ok, 'profit': profit_ok},
            'checklist': checklist, 'scenario': sm.view(xtop),
            'scenario_text': sm.stages_text(xtop),
            'scenarios': [sm.view(x) for x in xs]}


AI_EXIT_MODES = ('confirmed', 'all')
AI_EXIT_MODE_UA = {'confirmed': 'аномалії + підтвердження',
                   'all': 'лише коли всі показники за вихід'}


def exit_decide(r: Dict, mode: str = 'confirmed') -> bool:
    """🤖 Рішення «закривати» — за ВЕРДИКТОМ аномалій, а не за порогом балу
    (вимога 17.11: «не просто з порогом бал ≥ 70»). Бал — лише довідка.
    • 'confirmed' (деф.): аномалії кажуть закривати + підтвердження ціною або
      прибутком (`close`), усі групи разом (`all`) або підтверджений 🔥→🧽;
    • 'all': лише коли ВСІ групи за вихід (`all`) або підтверджений 🔥→🧽."""
    v = (r or {}).get('verdict')
    if not (r or {}).get('ready'):
        return False
    if mode == 'all':
        return v in ('scenario', 'all', 'combo')
    return v in ('scenario', 'all', 'combo', 'close')


def exit_checklist_text(r: Dict) -> str:
    """«✓ аномалії · ✓ ціна · ✗ прибуток» — для рядка 🧾 Логу."""
    g = (r or {}).get('groups') or {}
    return ' · '.join(f"{'✓' if g.get(k) else '✗'} {nm}" for k, nm in
                      (('anom', 'аномалії'), ('price', 'ціна'), ('profit', 'прибуток')))


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
        self._pro = None                       # ProfessionalBrain, лінивий import
        self._learner = None                   # champion/challenger на архівах
        self._pro_lock = threading.RLock()
        self._pro_bars: Dict[str, Dict] = {}
        self._pro_analysis: Dict[str, tuple] = {}
        self._pro_open_syms: set = set()
        self._learn_at = 0.0
        self._learning_state: Dict = {'status': 'not_run'}
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

    # ── Професійне ядро: закриті 15m/1h/4h бари + архівне навчання ──
    def _pro_get(self):
        with self._pro_lock:
            if self._pro is None:
                self._pro = _sib('professional_ai').ProfessionalBrain()
            if self._learner is not None:
                self._pro.model = self._learner.model
            return self._pro

    @staticmethod
    def _pro_compact(a: Optional[Dict]) -> Optional[Dict]:
        if not isinstance(a, dict):
            return None
        out = {k: a.get(k) for k in ('ok', 'action', 'side', 'best_side', 'quality',
                                     'ts', 'data_issues', 'vetoes', 'regime', 'data',
                                     'features', 'limitations') if k in a}
        model = a.get('model') or {}
        out['model'] = {k: model.get(k) for k in ('version', 'ready', 'oos_samples',
                                                  'min_oos', 'brier', 'accuracy') if k in model}
        out['sides'] = {}
        for side, row in (a.get('sides') or {}).items():
            out['sides'][side] = {k: row.get(k) for k in (
                'quality', 'ready', 'model_probability', 'model_is_ready', 'vetoes',
                'components', 'risk_plan') if k in row}
        if 'risk_plan' in a:
            out['risk_plan'] = a.get('risk_plan')
        return out

    def _pro_fetch_bars(self, symbol: str, tf: str, now: float, fetch: bool = True):
        """Закриті бари для 🧠 мозку з власного кешу (`PRO_CACHE_TTL`).
        `fetch=False` — ЛИШЕ кеш (монітор угод у мережу не ходить: бари для
        відкритих угод заздалегідь тягне такт рушія, `_pro_prefetch_open`)."""
        with self._pro_lock:
            hit = (self._pro_bars.get(symbol) or {}).get(tf)
        if hit and (not fetch or now - hit[0] < PRO_CACHE_TTL[tf]):
            return hit[1]
        if not fetch:
            return []
        try:
            rows = self._md_get().fetch_klines(symbol, limit=PRO_BAR_LIMIT, interval=tf) or []
            if rows:
                with self._pro_lock:
                    self._pro_bars.setdefault(symbol, {})[tf] = (now, rows)
                return rows
        except Exception as e:
            self._stat['last_err'] = f'professional {symbol} {tf}: {e}'[:200]
        # Мережа тимчасово недоступна → старий кеш; analyze сам перевірить вік
        # останнього закритого бару й заборонить рішення на застарілих даних.
        return hit[1] if hit else []

    def _pro_market(self, symbol: str, now: float, fetch: bool = True) -> Dict:
        return {tf: self._pro_fetch_bars(symbol, tf, now, fetch) for tf in PRO_TFS}

    def _pro_exit_on(self) -> bool:
        """Чи увімкнено «🧠 Професійний вихід ШІ» у Trade Manager."""
        tm = self._tm_get()
        try:
            return bool((getattr(tm, '_settings', None) or {}).get('use_ai_professional_exit'))
        except Exception:
            return False

    def _pro_prefetch_open(self, now: float) -> int:
        """Бари 15m/1h/4h для монет ВІДКРИТИХ угод (real + paper) — паралельно,
        у такті рушія. Так монітор TM читає готовий кеш і не чекає біржу."""
        try:
            syms = _sib('sbr_signals')._open_trade_symbols()
        except Exception:
            syms = []
        syms = sorted(set(syms))[:PRO_OPEN_CAP]
        self._pro_open_syms = set(syms)
        if not syms:
            return 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(lambda S: self._pro_market(S, now), syms))
        return len(syms)

    def _pro_prune(self, keep_view) -> None:
        """Кеші 🧠 — лише для монет таблиці ШІ та відкритих угод (памʼять
        обмежена; монета, що вибула, рахується наново, коли повернеться)."""
        keep = set(keep_view) | set(getattr(self, '_pro_open_syms', ()) or ())
        with self._pro_lock:
            for S in [k for k in self._pro_bars if k not in keep]:
                self._pro_bars.pop(S, None)
        for S in [k for k in self._pro_analysis if k not in keep]:
            self._pro_analysis.pop(S, None)

    def _professional_snapshot(self, symbol: str, now: float) -> Dict:
        symbol = str(symbol or '').upper()
        hit = self._pro_analysis.get(symbol)
        if hit and now - hit[0] < PRO_ANALYSIS_TTL:
            return hit[1]
        market = self._pro_market(symbol, now)
        try:
            require = bool(self.settings().get('ai_professional_require_model'))
            with self._pro_lock:
                result = self._pro_get().analyze(market, now, require_model=require)
        except Exception as e:
            result = {'ok': False, 'action': 'WAIT', 'side': None,
                      'data_issues': [f'помилка аналізу: {e}'],
                      'vetoes': [f'помилка аналізу: {e}'], 'model': {},
                      'ts': now, 'sides': {}}
            self._stat['errors'] += 1
            self._stat['last_err'] = f'professional {symbol}: {e}'[:200]
        self._pro_analysis[symbol] = (now, result)
        return result

    def _learn_professional(self, now: float, force: bool = False) -> Optional[Dict]:
        """🎓 Champion/challenger на архівах (епізоди ШІ + 🔬 дослідження угод).
        Лише коли 🧠 справді задіяний (фільтр входу або професійний вихід):
        без нього в архівах немає ознак для навчання, а читання 2×5000
        рядків БД раз на 15 хв було б марною роботою."""
        s = self.settings()
        if not s.get('ai_learning_enabled') or not self.db:
            return None
        if not (s.get('ai_professional_enabled') or self._pro_exit_on()):
            return None
        if not force and now - self._learn_at < LEARN_REFRESH_SEC:
            return self._learning_state
        self._learn_at = now
        if not (hasattr(self.db, 'get_ai_episodes') and hasattr(self.db, 'get_research')):
            self._learning_state = {'status': 'archive_api_unavailable', 'promoted': False}
            return self._learning_state
        try:
            ArchiveLearner = _sib('professional_learning').ArchiveLearner
            # БД читаємо ПОЗА локом мозку: аналіз монет не чекає запиту.
            episodes = self.db.get_ai_episodes(limit=LEARN_ROWS) or []
            research = self.db.get_research(limit=LEARN_ROWS) or []
            with self._pro_lock:
                if self._learner is None:
                    self._learner = ArchiveLearner()
                report = self._learner.run(episodes, research)
                if self._pro is not None:
                    self._pro.model = self._learner.model
                self._learning_state = dict(report)
                state = self._learner.state()
            # Пишемо стан лише коли партію СПОЖИТО (promoted / rejected) —
            # «немає нових міток» нічого не міняє, а блоб великий.
            if report.get('status') in ('promoted', 'rejected_no_improvement'):
                self.db.set_setting(DB_PRO_LEARNING, state)
            return report
        except Exception as e:
            self._learning_state = {'status': 'learning_error', 'error': str(e)[:200],
                                    'promoted': False}
            self._stat['last_err'] = f'архівне навчання: {e}'[:200]
            return self._learning_state

    @staticmethod
    def _pro_exit_slim(d: Dict) -> Dict:
        """Те, що лягає на ПОЗИЦІЮ (і їде в /api/tm/state щополла): рішення й
        пояснення без повного аналізу (ознаки 3 TF — кілобайти на угоду)."""
        out = {k: d.get(k) for k in ('action', 'close', 'reason', 'explanation',
                                     'r_multiple', 'peak_r', 'risk_source',
                                     'current_price', 'monitor', 'ts', 'error') if k in d}
        er = d.get('entry_risk') or {}
        if er:
            out['plan'] = {k: er.get(k) for k in ('stop', 'tp1', 'tp2', 'risk_pct',
                                                  'net_rr_after_costs')}
        an = d.get('analysis') or {}
        if an:
            out['data_ok'] = bool(an.get('ok'))
            if an.get('data_issues'):
                out['data_issues'] = list(an.get('data_issues'))[:4]
        return out

    def professional_exit(self, symbol: str, side: str, entry_price: float,
                          opened_at: float, current_price: Optional[float] = None,
                          peak_price: Optional[float] = None, fetch: bool = False) -> Dict:
        """Супровід позиції для Trade Manager. Не виконує жодного ордера.
        За замовчуванням ЛИШЕ кеш барів (монітор TM у мережу не ходить)."""
        now = self._clock()
        try:
            market = self._pro_market(str(symbol).upper(), now, fetch)
            require = bool(self.settings().get('ai_professional_require_model'))
            with self._pro_lock:
                decision = self._pro_get().manage_position(
                    market, side, entry_price, opened_at, now,
                    current_price=current_price, peak_price=peak_price,
                    require_model=require)
            decision['analysis'] = self._pro_compact(decision.get('analysis'))
            return decision
        except Exception as e:
            self._stat['last_err'] = f'professional exit {symbol}: {e}'[:200]
            return {'action': 'HOLD', 'close': False, 'reason': 'professional_exit_error',
                    'error': str(e)[:200], 'ts': now}

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
        self._learn_professional(now)
        if self._pro_exit_on():
            try:
                self._pro_prefetch_open(now)
            except Exception as e:
                self._stat['last_err'] = f'professional prefetch: {e}'[:200]
        else:
            self._pro_open_syms = set()
        sc = self._scn()
        if sc is None or not (sc.get_settings() or {}).get('spring_enabled', True):
            return 0
        self._refresh_cal(now)
        min_charge = float(s.get('ai_min_charge') or 65)
        cand = self.candidates(min_charge)
        anoms: Dict[str, List[Dict]] = {}
        try:
            look = max(ANOM_LOOK_SEC, getattr(_scen(), 'SCEN_LOOK_SEC', ANOM_LOOK_SEC))
            for a in (self.db.get_vol_anomalies(since=now - look, limit=20000)
                      if self.db else []) or []:
                anoms.setdefault(a.get('symbol'), []).append(a)
        except Exception:
            pass

        pro_on = bool(s.get('ai_professional_enabled'))

        def load(S):
            try:
                if pro_on:                    # бари 🧠 — у тому самому пулі
                    self._pro_market(S, now)
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
            el = entry_score('LONG', rec, vw, al, now, self._cal, lv, ch, charged)
            es = entry_score('SHORT', rec, vw, al, now, self._cal, lv, ch, charged)
            ep = self._eps.get(S)
            if ep is None and charged:
                ep = self._eps[S] = episode_new(S, now, rec, vw, src)
                _ev(ep, now, 'charged', charge=rec.get('charge'), lean=rec.get('lean'))
            if ep is not None:
                episode_step(ep, now, rec, vw, el, es, charged)
                self._dirty = True
            # головний бік: готовий сценарій → більший прогрес сценарію → бал
            def _rk(x):
                sc = x.get('scenario') or {}
                return (1 if entry_decide(x) else 0, (sc.get('done') or 0) / max(1, sc.get('total') or 1),
                        x['score'])
            best = el if _rk(el) >= _rk(es) else es
            pro = self._professional_snapshot(S, now) if pro_on else None
            pro_side = ((pro or {}).get('sides') or {}).get(best['side']) or {}
            pro_allowed = (not s.get('ai_professional_enabled')) or bool(
                pro and pro.get('ok') and pro_side.get('ready'))
            view[S] = {'symbol': S, 'src': src, 'charge': rec.get('charge'),
                       'state': rec.get('state'), 'lean': rec.get('lean'),
                       'prob_long': rec.get('prob_long'), 'long': el, 'short': es,
                       'best': best['side'], 'episode_age': (int(now - ep['started_at'])
                                                             if ep else None),
                       'signal': (ep or {}).get('signal'),
                       'professional': self._pro_compact(pro),
                       'professional_allowed': pro_allowed}
            if ep is not None:
                for x in (el, es):
                    sc = x.get('scenario') or {}
                    if entry_decide(x):
                        k = f"scen_{x['side']}_{sc.get('key')}"
                        if k not in ep:
                            ep[k] = int(now)
                            _ev(ep, now, 'scenario', side=x['side'], key=sc.get('key'),
                                name=sc.get('name'), score=x.get('score'))
            if (s.get('ai_signal_enabled') and ep is not None and entry_decide(best)
                    and pro_allowed):
                self._try_signal(S, best, rec, vw, ep, now, professional=pro)
            if ep is not None:
                why = episode_done(ep, now, charged)
                if why:
                    self._close_episode(S, ep, now, why)
        with self._lock:
            self._view = view
        self._pro_prune(view.keys() if pro_on else ())
        self._stat.update({'last_at': now, 'coins': len(view)})
        if self._dirty:
            self._persist()
        return len(view)

    def _try_signal(self, S, sc_res, rec, vw, ep, now, professional=None):
        side = sc_res['side']
        if ep.get('signal'):
            return                          # один результативний сигнал на епізод
        pro_side = ((professional or {}).get('sides') or {}).get(side) or {}
        if self.settings().get('ai_professional_enabled') and not (
                professional and professional.get('ok') and pro_side.get('ready')):
            _ev(ep, now, 'professional_veto', side=side,
                reason='; '.join((professional or {}).get('vetoes') or ['немає аналізу']))
            return
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
        pro_compact = self._pro_compact(professional)
        head = (f"🤖 Сигнал ШІ {side} · 🧩 {sc_res.get('scenario_text') or ''} · "
                f"заряд Пружини {rec.get('charge')} · довідково бал {sc_res['score']}")
        if pro_side:
            head += f" · 🧠 графік {pro_side.get('quality')}/100"
        log_activity(S, 'signal', f'{head} · {detail}', side=side, source='AI',
                     extra={'ai_score': sc_res['score'], 'ai_parts': sc_res['parts'],
                            'ai_scenario': (sc_res.get('scenario') or {}).get('key'),
                            'professional_quality': pro_side.get('quality'),
                            'professional_model': (pro_compact or {}).get('model')})
        if not ok:
            log_activity(S, 'rejected', reason, side=side, source='AI')
            _ev(ep, now, 'signal_rejected', side=side, score=sc_res['score'], reason=reason)
            return
        tm = self._tm_get()
        res = None
        if tm is not None:
            try:
                res = tm.on_signal(symbol=S, side=side, entry_price=px or 0, opened_by='ai')
                attach = getattr(tm, 'set_ai_professional_context', None)
                if callable(attach) and pro_compact:
                    attach(S, side, pro_compact)
            except Exception as e:
                self._stat['last_err'] = f'on_signal {S}: {e}'[:200]
        ep['signal'] = {'t': int(now), 'side': side, 'px': px, 'score': sc_res['score'],
                        'parts': sc_res['parts'], 'result': str(res)[:80] if res else None,
                        'scenario': (sc_res.get('scenario') or {}).get('key'),
                        'scenario_name': (sc_res.get('scenario') or {}).get('name'),
                        'professional': pro_compact}
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
        try:
            pro_state = self.db.get_setting(DB_PRO_LEARNING, None)
            if isinstance(pro_state, str):
                pro_state = json.loads(pro_state)
            if pro_state:
                ArchiveLearner = _sib('professional_learning').ArchiveLearner
                self._learner = ArchiveLearner.from_state(pro_state)
                self._learning_state = dict(self._learner.last_report or
                                            {'status': self._learner.last_status})
        except Exception as e:
            # Пошкоджена модель НЕ завантажується частково і не використовується.
            self._learner = None
            self._learning_state = {'status': 'model_load_error', 'error': str(e)[:200]}

    def state(self) -> Dict:
        with self._lock:
            view = sorted(self._view.values(),
                          key=lambda v: -max(v['long']['score'], v['short']['score']))
        model = self._pro_get().model_state()
        learning = {k: v for k, v in (self._learning_state or {}).items() if k != 'model'}
        professional = {
            'enabled': bool(self.settings().get('ai_professional_enabled')),
            'require_model': bool(self.settings().get('ai_professional_require_model')),
            'learning_enabled': bool(self.settings().get('ai_learning_enabled')),
            'model': {k: model.get(k) for k in ('version', 'ready', 'train_samples',
                                                'oos_samples', 'min_oos', 'brier', 'accuracy')},
            'learning': learning,
            'processed_archive_records': len(self._learner.seen) if self._learner else 0,
            'monitored_symbols': len(self._pro_analysis),
        }
        return {'ok': True, 'settings': self.settings(), 'stat': dict(self._stat),
                'calibration': self.calibration(), 'coins': view,
                'professional': professional,
                'episodes_open': len(self._eps), 'server_now': self._clock(),
                'weights': {'entry': W_ENTRY, 'exit': W_EXIT},
                'catalog': getattr(_scen(), 'CATALOG', None)}

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
