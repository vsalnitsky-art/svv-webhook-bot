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
# 🧠 Окремих тумблерів «професійного фільтра / OOS-моделі» більше НЕМАЄ
# (17.11, «єдиний мозок»): графік 15m/1h/4h — частина рішення 🤖 завжди.
DEFAULTS = {'ai_signal_enabled': False, 'ai_min_charge': 65,
            'ai_learning_enabled': True}

# ── 🤖 ВЛАСНИЙ СЦЕНАРІЙ ШІ (вимога 22.11) ──────────────────────────────────
# Дослівно: «ШІ не мають зупиняти жоден із воріт, ШІ має працювати за своїм
# окремим сценарієм (дві пружини і окремі налаштування які стосуються лише
# роботи ШІ)… бот відкриває угоду ШІ і супроводжує лише свою відкриту угоду».
# Налаштування ШІ живуть ОКРЕМО (`ai_settings_v1`), а не в сканері / TM.
AI_DB_SETTINGS = 'ai_settings_v1'
AI_SCENARIOS = ('E1', 'E2', 'E3', 'E4')
AI_BOOKS = ('auto', 'paper', 'real')
AI_DEFAULTS = {
    'enabled': False,          # ШІ відкриває СВОЇ угоди (деф. ВИМК)
    'exit_enabled': True,      # ШІ супроводжує й закриває СВОЇ угоди
    'exit_mode': 'confirmed',  # 'confirmed' | 'all'
    'min_charge': 65,          # монета Пружини береться в роботу з цього заряду
    'scenarios': ['E2'],       # вхід лише на «💥 Старт розряду» (рішення 22.11)
    'allow_long': True,
    'allow_short': True,
    'book': 'auto',            # 'auto' (TRD · TM · Test Mode) | 'paper' | 'real'
    'max_open': 5,             # одночасно відкритих угод ШІ
    'max_per_30m': 3,          # нових угод ШІ за 30 хв (кластер = одна подія ринку)
    'flip_cooldown_min': 180,  # протилежний бік по тій самій монеті — не раніше
    'hold_pot': 40,            # 🌀 заряд у бік угоди ≥ → вихід «ще зарано» (0 = вимк)
    'sl_fallback_pct': 2.0,    # стоп, коли плану немає / джерело «%» (% від входу)
    # 🛑 «SL з» ШІ (22.11): plan — структура / ATR 15m мозку графіка · ob — OB
    # 1H → 15m → 5m (той самий ланцюг, що глобальне «🛑 SL з» бота) · pct — % від входу.
    'sl_source': 'ob',
    # 🛑 «SL з» ШІ — головний вимикач (23.11, деф. ВИМК): вимкнено → Manual SL на
    # угоду ШІ НЕ ставиться і стоп плану НЕ закриває угоду; стоп плану лишається
    # лише орієнтиром ризику (R, гейт R ≥ 1.3, база еталонів).
    'sl_enabled': False,
    'sl_min_pct': 0.8,         # стоп не ближче N % від входу (шум 15m не вибиває)
    'sl_max_pct': 4.0,         # стоп не далі N % (0 = без стелі)
    'min_conf': 40,            # 🎯 впевненість мозку ≥ N, інакше вхід не беремо (0 = вимк)
    # 🧩 Фільтр за блоком «МММ-Готовність» (23.11): off · banner (бік угоди =
    # банер) · coin (вкладка монети = бік) · both (обидва).
    'rdy_filter': 'off',
    # 💥 Підтвердження розряду (23.11): пробій коробки рахується лише ЗАКРИТОЮ
    # 1h-свічкою, а вхід — коли ще `discharge_bars` закритих свічок утримали
    # ціну за межею. Повернення в коробку = хибний старт.
    'discharge_confirm': True,
    'discharge_bars': 1,
    'discharge_tf': '1h',       # TF підтвердження розряду: '1h' | '15m' (рівень — коробка 1h)
    'discharge_bars_15m': 2,    # для 15m: закритих 15m-свічок утримали ціну за межею
    # 📦 Друге підтвердження розряду (24.11): САМИЙ ОСТАННІЙ VOB 1H на графіку
    # (той самий детектор і параметри «📦 Volumized OB Trend») мусить бути в бік,
    # куди стартував розряд, і не зламаний. Немає / проти / breaker → не відкриваємо.
    'discharge_vob1h': True,
    'false_start_exit': True,   # угода ШІ з E2: ціна повернулась у коробку → вихід
    # 🎯 Вихід на кульмінації (Виснаження) для угод ШІ (24.11): той самий
    # алгоритм і пороги, що в Exit Rules TM (`exh_exit_*`), незалежно від
    # глобального тумблера `use_exh_exit`.
    'exh_exit': True,
    'learning_enabled': True,  # 🎓 champion/challenger на архівах
    'etalon_enabled': True,    # 🏆 база еталонів відмінних угод (самопоповнення)
    'etalon_min_r': 1.5,       # «відмінна» = результат ≥ N R від стопа плану
    'etalon_min_sim': 0.35,    # мозок еталонів: найближчий досвід не далі за цю схожість
    'etalon_min_p': 0.6,       # мозок еталонів: частка «відмінних» серед схожих входів ≥
    'cleanup_enabled': True,   # 🧹 автоочистка таблиць навчання
}
AI_SL_SOURCES = ('plan', 'ob', 'pct')
AI_RDY_FILTERS = ('off', 'banner', 'coin', 'both')
AI_RDY_FILTER_UA = {'off': 'вимк', 'banner': 'за банером', 'coin': 'за вкладкою монети',
                    'both': 'банер + вкладка монети'}
DISCHARGE_LOOK = 6             # закритих 1h-свічок назад — шукаємо пробій коробки
DISCHARGE_LOOK_LTF = 16        # закритих 15m-свічок назад (4 год)
DISCHARGE_TFS = ('1h', '15m', '5m')  # TF підтвердження розряду (рівень — завжди коробка 1h)
DISCHARGE_TF_UA = {'1h': '1H закриття', '15m': '15m закриття', '5m': '5m CHoCH+BOS'}
# 📊 Тінь веде ще й «5m VOB» (24.11) — ЛИШЕ для порівняння, вибрати його для
# входу не можна (DISCHARGE_TFS не змінено).
# «5m VOB останній» (24.11) — те саме, але точка входу — НАЙНОВІШИЙ VOB 5m у бік
# розряду після його старту (кожен новий блок — нова точка), а не перший.
DIS_SHADOW_TFS = DISCHARGE_TFS + ('vob', 'vobl')
DIS_SHADOW_UA = dict(DISCHARGE_TF_UA, vob='5m VOB', vobl='5m VOB останній')
VOB_DEFAULTS = {'swing_length': 10, 'ob_end_method': 'Wick', 'max_atr_mult': 3.5}
STRUCT_LOOK_SEC = 4 * 3600     # 5m-структура: CHoCH+BOS шукаємо за останні 4 год
STRUCT_INTERNAL_SIZE = 5       # півоти внутрішньої структури (як у сканера за замовч.)
DIS_SHADOW_HOURS = 24          # тіньовий запис ведемо 24 год від підтвердження
DIS_SHADOW_MAX = 400
DISCHARGE_BOX = 24             # коробка перед пробоєм — 24 закриті 1h-свічки
DISCHARGE_STATE_UA = {'none': 'пробою немає', 'live': 'пробій лише на живій свічці',
                      'pending': 'пробій закрито, чекаємо підтвердження',
                      'confirmed': 'розряд підтверджено', 'false': 'хибний старт'}
AI_SL_SOURCE_UA = {'plan': 'структура / ATR 15m', 'ob': 'OB 1H → 15m → 5m', 'pct': '% від входу'}
ETALON_DB = 'ai_etalon_v1'
ETALON_MAX = 300
ETALON_MIN_N = 10              # поки еталонів менше — схожість лише довідка
ETALON_K = 5
ETALON_FALLBACK_PNL = 1.5      # «відмінна» без стопа плану: результат ≥ N %
FEAT_KEYS = ('charge', 'pot', 'prob', 'lean', 'break', 'impulse', 'flow', 'conf', 'rr')
CLEANUP_EVERY_SEC = 6 * 3600
CLEANUP_MARK_SEC = 3600        # 🏷 позначення «на видалення» — раз на годину (лише підрахунок)
CLEANUP_RULES = {'episode_nosignal_days': 7, 'episode_days': 0,
                 'anom_aux_days': 2, 'research_other_days': 30,
                 'research_ai_days': 0, 'anom_done_days': 60}
# 🏆🧠 РЕЗУЛЬТАТИ навчання (еталони, антибаза, дослідження угод ШІ, епізоди із
# сигналом) чистяться НЕ за датою, а за ЯКІСТЮ (вимога 22.11): мозок спершу
# оцінює всі записи, позначає найгірші й дублі кращих і видаляє лише їх, коли
# база переповнена. Службові дані (без результату) — за датою, як і раніше.
ETALON_KEEP = 240              # м'яка стеля бази: вище — позначаємо найслабші
ETALON_DUP_DIST = 0.04         # майже той самий вхід — лишається кращий із двох
RESEARCH_AI_KEEP = 600         # дослідження угод ШІ: тримаємо найінформативніші
EPISODE_SIG_KEEP = 2000        # епізоди із сигналом ШІ: тримаємо найінформативніші
CLEANUP_RULES_UA = [
    'епізоди заряду без сигналу ШІ — старші за 7 днів (результату в них немає)',
    'допоміжні аномалії (🧭 напрямок / 👤 підписки) — старші за 2 дні',
    'аномалії з повним наслідком — старші за 60 днів (калібровка бере 30)',
    'дослідження угод НЕ ШІ — старші за 30 днів',
    '🏆 еталони: дубль кращого входу і найслабші, коли базу переповнено (> %d)' % ETALON_KEEP,
    '❌ антибаза: дубль глибшого уроку і найменш збиткові, коли базу переповнено (> %d)' % ETALON_KEEP,
    '🔬 дослідження угод ШІ: найменш показові, коли їх > %d (угоди з еталонів захищені)' % RESEARCH_AI_KEEP,
    '📈 епізоди із сигналом: найменш показові, коли їх > %d' % EPISODE_SIG_KEEP,
]
OPEN_WINDOW_SEC = 1800
BLOCK_LOG_GAP = 900            # однакова відмова ШІ в лог — не частіше раз на 15 хв
HEARTBEAT_SEC = 900            # 🤖 «огляд» ШІ в 🧾 Лог — раз на 15 хв або на зміну точок входу


def ai_settings_norm(raw: Optional[Dict]) -> Dict:
    """🤖 Налаштування ШІ з валідацією. ЧИСТА. Невідоме / биті → дефолт."""
    s = dict(AI_DEFAULTS)
    r = raw if isinstance(raw, dict) else {}
    for k in ('enabled', 'exit_enabled', 'allow_long', 'allow_short', 'learning_enabled',
              'etalon_enabled', 'cleanup_enabled', 'sl_enabled', 'discharge_confirm',
              'false_start_exit', 'exh_exit', 'discharge_vob1h'):
        if k in r:
            s[k] = bool(r[k])
    if r.get('exit_mode') in ('confirmed', 'all'):
        s['exit_mode'] = r['exit_mode']
    if r.get('book') in AI_BOOKS:
        s['book'] = r['book']
    if r.get('sl_source') in AI_SL_SOURCES:
        s['sl_source'] = r['sl_source']
    if r.get('discharge_tf') in DISCHARGE_TFS:
        s['discharge_tf'] = r['discharge_tf']
    if r.get('rdy_filter') in AI_RDY_FILTERS:
        s['rdy_filter'] = r['rdy_filter']
    sc = r.get('scenarios')
    if isinstance(sc, (list, tuple)):
        got = [x for x in AI_SCENARIOS if x in {str(v).upper() for v in sc}]
        s['scenarios'] = got or ['E2']
    for k, lo, hi, cast in (('min_charge', 0, 100, float), ('max_open', 1, 50, int),
                            ('max_per_30m', 1, 50, int), ('flip_cooldown_min', 0, 1440, int),
                            ('hold_pot', 0, 100, float), ('sl_fallback_pct', 0.2, 20.0, float),
                            ('etalon_min_r', 0.5, 10.0, float), ('etalon_min_sim', 0.0, 1.0, float),
                            ('etalon_min_p', 0.3, 0.95, float), ('sl_min_pct', 0.0, 10.0, float),
                            ('sl_max_pct', 0.0, 20.0, float), ('min_conf', 0, 100, int),
                            ('discharge_bars', 1, 4, int), ('discharge_bars_15m', 1, 8, int)):
        if k in r:
            try:
                s[k] = cast(max(lo, min(hi, float(r[k]))))
            except (TypeError, ValueError):
                pass
    if s['sl_max_pct'] and s['sl_max_pct'] < s['sl_min_pct']:
        s['sl_max_pct'] = s['sl_min_pct']
    return s


def ai_settings_migrate(scanner_s: Optional[Dict], tm_s: Optional[Dict]) -> Dict:
    """Перший запуск: переносимо наявні ключі сканера / TM у власні. ЧИСТА."""
    a, t = scanner_s or {}, tm_s or {}
    raw = {}
    if 'ai_signal_enabled' in a:
        raw['enabled'] = a.get('ai_signal_enabled')
    if 'ai_min_charge' in a:
        raw['min_charge'] = a.get('ai_min_charge')
    if 'ai_learning_enabled' in a:
        raw['learning_enabled'] = a.get('ai_learning_enabled')
    if 'ai_exit_mode' in t:
        raw['exit_mode'] = t.get('ai_exit_mode')
    return ai_settings_norm(raw)


def ai_entry_block(ep: Optional[Dict], side: str, s: Dict, brain: Optional[Dict],
                   eb: Optional[Dict], open_n: int, opens: List[float],
                   last: Optional[tuple], now: float, rdy: Optional[Dict] = None,
                   discharge: Optional[Dict] = None, scen_key: str = '') -> str:
    """ЧОМУ готова точка входу НЕ стала угодою (22.11) — '' = відкриваємо. ЧИСТА.
    Те саме правило, що в `_try_signal`: показ у таблиці / огляді / тумблері
    пояснює кожен «тихий» пропуск, а не лише ті, що доходять до біржі."""
    s, ep, brain = s or {}, ep or {}, brain or {}
    sd = str(side).upper()
    if not brain.get('ok'):
        return '🧠 мозок не пустив: ' + (brain.get('reason') or 'чек-лист не пройдено')
    # 💥 E2 «Старт розряду»: лише ПІДТВЕРДЖЕНИЙ закритими свічками (23.11)
    if scen_key == 'E2' and s.get('discharge_confirm', True):
        ds = discharge or {}
        if ds.get('state') != 'confirmed':
            return '💥 розряд не підтверджено: ' + (ds.get('note') or
                                                   DISCHARGE_STATE_UA.get(ds.get('state'), 'немає даних'))
        if s.get('discharge_vob1h', True):
            v1 = ds.get('vob1h') or {}
            if not v1.get('ok'):
                return '📦 VOB 1H: ' + (v1.get('note') or 'останній VOB 1H не перевірено (немає даних)')
    rb = rdy_block(sd, rdy, s.get('rdy_filter') or 'off')
    if rb:
        return rb
    mc = float(s.get('min_conf') or 0)
    conf = brain.get('confidence')
    if mc > 0 and conf is not None and float(conf) < mc:
        return f'🎯 впевненість {int(conf)} < {int(mc)}'
    sig = ep.get('signal') or {}
    if sig.get('own'):
        return f"у цьому епізоді заряду вже була угода ШІ ({sig.get('side')})"
    tr = (ep.get('ai_tries') or {}).get(sd) or [0, 0]
    if tr[0] >= SIGNAL_TRIES:
        return f'{SIGNAL_TRIES} спроби відкриття в цьому епізоді не вдались'
    if tr[1] and now - float(tr[1]) < SIGNAL_RETRY_SEC:
        return f'повтор відкриття не раніше ніж через {int((SIGNAL_RETRY_SEC - (now - float(tr[1]))) // 60) + 1} хв'
    why = ai_limits_block(sd, s, open_n, opens, last, now)
    if why:
        return why
    if s.get('etalon_enabled') and eb and eb.get('mature') and eb.get('verdict') != 'take':
        return '🏆🧠 мозок еталонів: ' + (eb.get('reason') or 'ні')
    return ''


def ai_stop_for(side: str, price, plan: Optional[Dict], ob_stop=None, ob_label: str = '',
                s: Optional[Dict] = None) -> Dict:
    """🛑 «SL з» ШІ → план угоди з обраним стопом. ЧИСТА.
    plan — стоп плану (структура / ATR 15m) · ob — рівень OB (сходи 1H → 15m →
    5m), немає → стоп плану · pct — % від ціни. Потім межі `sl_min_pct` /
    `sl_max_pct` і перерахунок ризику / R; ціль «2.2R» тягнеться за новим стопом,
    ціль Пружини — ні (тож R може змінитись)."""
    s = s or {}
    sd = str(side).upper()
    sg = 1.0 if sd == 'LONG' else -1.0
    try:
        px = float(price)
    except (TypeError, ValueError):
        px = 0.0
    plan = dict(plan or {})
    if px <= 0:
        return plan
    src = s.get('sl_source') or 'ob'

    def _ok(v):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return v if v > 0 and sg * (px - v) > 0 else None

    stop, label = None, ''
    if src == 'ob':
        stop = _ok(ob_stop)
        label = ob_label or 'OB'
        if stop is None:
            stop, label = _ok(plan.get('stop')), 'структура / ATR 15m (OB не знайдено)'
    elif src == 'plan':
        stop, label = _ok(plan.get('stop')), 'структура / ATR 15m'
    if stop is None:
        pct = float(s.get('sl_fallback_pct') or 2.0)
        stop, label = px * (1 - sg * pct / 100.0), f'{pct}% від входу'
    dist = abs(px - stop) / px * 100.0
    lo, hi = float(s.get('sl_min_pct') or 0.0), float(s.get('sl_max_pct') or 0.0)
    clamp = ''
    if lo > 0 and dist < lo:
        dist, clamp = lo, f'відсунуто до мінімуму {lo}%'
    elif hi > 0 and dist > hi:
        dist, clamp = hi, f'підтягнуто до стелі {hi}%'
    stop = px * (1 - sg * dist / 100.0)
    cost = float(plan.get('cost_pct') or 0.0)
    tgt = plan.get('target')
    if plan.get('target_src') != 'spring' or not tgt:
        tgt = px + sg * 2.2 * abs(px - stop)
        plan['target_src'] = plan.get('target_src') or 'r'
    reward = abs(float(tgt) - px) / px * 100.0
    if sg * (float(tgt) - px) <= 0:
        reward = 0.0
    plan.update({'entry': round(px, 10), 'stop': round(stop, 10), 'target': round(float(tgt), 10),
                 'risk_pct': round(dist, 3), 'reward_pct': round(reward, 3),
                 'net_rr': round((reward - cost) / (dist + cost), 2) if dist > 0 else 0.0,
                 'stop_src': src, 'stop_label': label + (f' · {clamp}' if clamp else '')})
    return plan


def rdy_block(side: str, view: Optional[Dict], mode: str) -> str:
    """🧩 Фільтр ШІ за блоком «МММ-Готовність» (23.11). '' → можна. ЧИСТА.
    `view` = `ff.mm_gate_view(sym, book='rdy')` → {on, dir (банер), coin
    (вкладка монети LONG/SHORT/FLAT | None)}. Невідомо ≠ «за напрямком»:
    блок вимкнено / монети немає в таблиці → відмова з причиною."""
    if mode not in ('banner', 'coin', 'both'):
        return ''
    sd = str(side).upper()
    v = view or {}
    if not v.get('on'):
        return '🧩 МММ-Готовність вимкнено — фільтр ШІ не може перевірити напрямок'
    if mode in ('banner', 'both'):
        d = v.get('dir')
        if d not in ('LONG', 'SHORT'):
            return f'🧩 банер МММ-Готовність без напрямку (⚖), а вхід {sd}'
        if d != sd:
            return f'🧩 банер МММ-Готовність {d}, а вхід {sd}'
    if mode in ('coin', 'both'):
        c = v.get('coin')
        if c is None:
            return '🧩 монети немає в таблиці МММ-Готовність'
        if c != sd:
            return f"🧩 монета у вкладці {'⚖' if c == 'FLAT' else c} МММ-Готовність, а вхід {sd}"
    return ''


def rdy_align(side: str, rdy: Optional[Dict]) -> str:
    """Як угода стояла відносно банера «МММ-Готовність» на вході → 'with' /
    'against' / 'flat' (банер без напрямку) / 'off' (блок вимкнено) /
    'unknown' (знімка немає). ЧИСТА — для статистики угод."""
    r = rdy or {}
    if not r:
        return 'unknown'
    if not r.get('on'):
        return 'off'
    d = r.get('dir')
    if d not in ('LONG', 'SHORT'):
        return 'flat'
    return 'with' if d == str(side).upper() else 'against'


RDY_ALIGN_UA = {'with': '✅ за банером', 'against': '⛔ проти банера', 'flat': '⚖ банер без напрямку',
                'off': '🔌 блок вимкнено', 'unknown': '— без знімка'}


def rdy_trade_stats(trades: List[Dict]) -> Dict:
    """📊 Закриті угоди за положенням відносно банера «МММ-Готовність» на
    вході (`rdy_open`). ЧИСТА. → {groups: {align: {n, win, win_pct, avg,
    sum}}, n, ai: та сама розбивка лише для угод ШІ}."""
    def _grp(rows):
        g = {}
        for t in rows:
            a = rdy_align(t.get('side'), t.get('rdy_open'))
            try:
                pnl = float(t.get('pnl_pct'))
            except (TypeError, ValueError):
                continue
            x = g.setdefault(a, {'n': 0, 'win': 0, 'sum': 0.0})
            x['n'] += 1
            x['win'] += 1 if pnl > 0 else 0
            x['sum'] += pnl
        for x in g.values():
            x['win_pct'] = round(100.0 * x['win'] / x['n'], 1) if x['n'] else None
            x['avg'] = round(x['sum'] / x['n'], 3) if x['n'] else None
            x['sum'] = round(x['sum'], 3)
        return g
    rows = [t for t in (trades or []) if isinstance(t, dict)]
    ai = [t for t in rows if 'ai' == str(t.get('opened_by') or '').split(' ')[0]]
    return {'groups': _grp(rows), 'n': len(rows), 'ai': _grp(ai), 'ai_n': len(ai),
            'labels': RDY_ALIGN_UA}


def _bar_close(b):
    for k in ('p', 'c', 'close'):
        try:
            v = float((b or {}).get(k))
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    return None


def _bar_hl(b):
    try:
        return float(b.get('h', b.get('high'))), float(b.get('l', b.get('low')))
    except (TypeError, ValueError, AttributeError):
        return None, None


def _bar_sec(b):
    try:
        t = int(float((b or {}).get('t')))
    except (TypeError, ValueError):
        return None
    return t // 1000 if t > 10 ** 12 else t


def _discharge_walk(seq, side: str, live_price, need: int, box_now, tf: str) -> Dict:
    """Спільна машина станів розряду (23.11). `seq` — закриті свічки вікна
    [(t, close, box_hi, box_lo)] за часом; `box_now` — (hi, lo) коробки зараз.
    → {state, edge, box_hi, box_lo, t, held, need, tf, note}."""
    sd = str(side).upper()
    sg = 1.0 if sd == 'LONG' else -1.0
    out = {'state': 'none', 'edge': None, 'box_hi': None, 'box_lo': None, 't': None,
           'held': 0, 'need': int(need), 'tf': tf, 'note': ''}
    cur = None
    for t, c, hi, lo in seq:
        if cur is None or cur['state'] == 'false':
            edge = hi if sd == 'LONG' else lo
            if sg * (c - edge) > 0:
                cur = {'state': 'pending', 'edge': edge, 'box_hi': hi, 'box_lo': lo,
                       't': t, 'held': 0}
            continue
        if sg * (c - cur['edge']) <= 0:
            cur['state'] = 'false'
            cur['false_t'] = t
        else:
            cur['held'] += 1
    try:
        lp = float(live_price) if live_price is not None else None
    except (TypeError, ValueError):
        lp = None
    if cur is None:
        if box_now:
            hi, lo = box_now
            edge = hi if sd == 'LONG' else lo
            if lp and sg * (lp - edge) > 0:
                out.update(state='live', edge=edge, box_hi=hi, box_lo=lo,
                           note='пробій коробки лише на живій (незакритій) свічці')
                return out
        out['note'] = 'закритого пробою коробки немає'
        return out
    out.update({k: cur.get(k) for k in ('state', 'edge', 'box_hi', 'box_lo', 't', 'held')})
    if cur['state'] == 'false':
        out['note'] = f'свічка {tf} закрилась назад у коробці — хибний старт'
        out['false_t'] = cur.get('false_t')
    elif cur['held'] >= need:
        if lp is not None and sg * (lp - cur['edge']) <= 0:
            out['state'], out['note'] = 'pending', 'ціна зараз повернулась у коробку'
        else:
            out['state'], out['note'] = 'confirmed', f"пробій утримано {cur['held']} св. {tf}"
    else:
        out['note'] = f"пробій закрито, утримано {cur['held']}/{need} св. {tf}"
    return out


def _clean_bars(bars):
    return [b for b in (bars or []) if _bar_close(b) is not None and _bar_hl(b)[0] is not None
            and _bar_sec(b) is not None]


def _hl_box(bars):
    return max(_bar_hl(b)[0] for b in bars), min(_bar_hl(b)[1] for b in bars)


def discharge_state(closed: List[Dict], side: str, live_price=None, need: int = 1,
                    look: int = DISCHARGE_LOOK, box: int = DISCHARGE_BOX) -> Dict:
    """💥 Розряд Пружини справжній чи хибний старт (23.11). ЧИСТА.
    `closed` — ЗАКРИТІ 1h-свічки (живу відкинуто). Пробій = закриття свічки за
    межею коробки 24 попередніх свічок у бік `side`. Далі:
    • наступна закрита свічка закрилась НАЗАД у коробці → 'false' (хибний старт);
    • `need` закритих свічок після пробою втримали ціну за межею, і жива ціна
      теж за межею → 'confirmed';
    • пробій закрито, підтверджень ще замало (або жива ціна в коробці) → 'pending';
    • закритого пробою немає, лише жива свічка за межею → 'live'; нічого → 'none'.
    → {state, edge, box_lo, box_hi, t (пробій), held, need, tf, note}."""
    bars = _clean_bars(closed)
    n = len(bars)
    if n < box + 2:
        return {'state': 'none', 'edge': None, 'box_hi': None, 'box_lo': None, 't': None,
                'held': 0, 'need': int(need), 'tf': '1h', 'note': 'замало закритих свічок'}
    seq = []
    for k in range(max(box, n - look), n):
        hi, lo = _hl_box(bars[k - box:k])
        seq.append((_bar_sec(bars[k]), _bar_close(bars[k]), hi, lo))
    return _discharge_walk(seq, side, live_price, need, _hl_box(bars[n - box:n]), '1h')


def discharge_state_ltf(closed_1h: List[Dict], closed_ltf: List[Dict], side: str,
                        live_price=None, need: int = 2, look: int = DISCHARGE_LOOK_LTF,
                        box: int = DISCHARGE_BOX, ltf: str = '15m') -> Dict:
    """💥 Те саме підтвердження розряду на МОЛОДШОМУ TF (23.11). ЧИСТА.
    РІВЕНЬ — межа 24-год коробки 1h (як і Пружина): для кожної закритої
    свічки `ltf` коробка = 24 закриті 1h-свічки, що закінчились до її
    відкриття. ТАЙМІНГ — закриття `ltf`: пробій = закриття свічки за межею,
    `need` закритих свічок утримали → 'confirmed', назад у коробці → 'false'."""
    h = _clean_bars(closed_1h)
    lt = _clean_bars(closed_ltf)
    if len(h) < box or not lt:
        return {'state': 'none', 'edge': None, 'box_hi': None, 'box_lo': None, 't': None,
                'held': 0, 'need': int(need), 'tf': ltf, 'note': 'замало закритих свічок'}
    ends = [_bar_sec(b) + 3600 for b in h]
    seq = []
    for b in lt[-look:]:
        t = _bar_sec(b)
        j = 0
        while j < len(h) and ends[j] <= t:
            j += 1
        if j < box:
            continue
        hi, lo = _hl_box(h[j - box:j])
        seq.append((t, _bar_close(b), hi, lo))
    return _discharge_walk(seq, side, live_price, need, _hl_box(h[-box:]), ltf)


def dis_pick(both: Dict, tf: str) -> Dict:
    """Стан розряду ОБРАНОГО TF + `alts` (інші TF — для показу й тіні) + `vob1h`
    (друге підтвердження — останній VOB 1H). ЧИСТА."""
    tf = tf if tf in DISCHARGE_TFS else '1h'
    out = dict((both or {}).get(tf) or (both or {}).get('1h') or {})
    if (both or {}).get('vob1h') is not None:
        out['vob1h'] = both['vob1h']
    out['alts'] = [{k: ((both or {}).get(a) or {}).get(k) for k in ('state', 'tf', 'held', 'need', 't', 'note')}
                   for a in DIS_SHADOW_TFS if a != tf and (both or {}).get(a)]
    return out


def discharge_state_struct(closed_1h: List[Dict], closed_5m: List[Dict], side: str,
                           live_price=None, events=None, box: int = DISCHARGE_BOX,
                           look_sec: float = STRUCT_LOOK_SEC,
                           internal_size: int = STRUCT_INTERNAL_SIZE) -> Dict:
    """💥 Підтвердження розряду СТРУКТУРОЮ 5m (23.11). ЧИСТА.
    Напрямок і сигнал — внутрішня структура 5m (той самий `detect_smc_structure`,
    що в боті): після останньої зустрічної події — CHoCH у бік `side`, далі BOS
    у той самий бік, і свічка BOS ЗАКРИЛАСЬ ЗА МЕЖЕЮ 24-год коробки 1h (рівень
    Пружини). Без умови межі CHoCH усередині стиснутої коробки — шум.
    • CHoCH+BOS є, BOS за межею, зустрічної події після нього немає, жива ціна
      за межею → 'confirmed';
    • після BOS — зустрічна подія структури (CHoCH проти) → 'false';
    • CHoCH у бік є, BOS за межею ще немає / ціна в коробці → 'pending';
    • інакше → 'none'. `events` — готові внутрішні події (для тестів)."""
    sd = str(side).upper()
    sg = 1.0 if sd == 'LONG' else -1.0
    d = 'bull' if sd == 'LONG' else 'bear'
    out = {'state': 'none', 'edge': None, 'box_hi': None, 'box_lo': None, 't': None,
           'held': 0, 'need': 1, 'tf': '5m', 'note': ''}
    h = _clean_bars(closed_1h)
    m5 = _clean_bars(closed_5m)
    if len(h) < box or len(m5) < internal_size + 5:
        out['note'] = 'замало закритих свічок'
        return out
    if events is None:
        try:
            from detection.smc_structure import detect_smc_structure
        except Exception:
            try:
                detect_smc_structure = _sib('smc_structure').detect_smc_structure
            except Exception:
                out['note'] = 'немає детектора структури'
                return out
        an = detect_smc_structure(m5, internal_size=internal_size, swing_size=50)
        events = ((an or {}).get('internal') or {}).get('events') or []
    close_at = {_bar_sec(b): _bar_close(b) for b in m5}
    ends = [_bar_sec(b) + 3600 for b in h]

    def _box_at(t):
        j = 0
        while j < len(h) and ends[j] <= t:
            j += 1
        return _hl_box(h[j - box:j]) if j >= box else None

    t_end = _bar_sec(m5[-1]) or 0
    evs = []
    for e in events:
        t = _bar_sec({'t': e.get('to_t')})
        if t is None or t < t_end - look_sec:
            continue
        evs.append((t, str(e.get('tag') or ''), e.get('dir')))
    evs.sort()
    choch_t, bos, cand = None, None, None
    for t, tag, dr in evs:
        if dr != d:
            if bos:                       # структура розвернулась після сигналу
                out.update(state='false', t=bos['t'], edge=bos['edge'], box_hi=bos['hi'],
                           box_lo=bos['lo'], held=1,
                           note=f'після BOS — {tag} проти: структура 5m розвернулась (хибний старт)')
                out['false_t'] = t
                return out
            choch_t, cand = None, None
            continue
        if tag == 'CHoCH':
            choch_t = t
            continue
        if tag == 'BOS' and choch_t is not None:
            bx = _box_at(t)
            c = close_at.get(t)
            if bx and c is not None:
                edge = bx[0] if d == 'bull' else bx[1]
                cand = {'t': t, 'edge': edge, 'hi': bx[0], 'lo': bx[1], 'choch_t': choch_t}
                if sg * (c - edge) > 0 and not bos:
                    bos = cand
    if not bos:
        if choch_t is not None:
            bx = _box_at(t_end + 1)
            out.update(state='pending', t=choch_t,
                       note='CHoCH 5m у бік є, BOS за межею коробки 1h ще немає')
            if bx:
                out.update(edge=bx[0] if d == 'bull' else bx[1], box_hi=bx[0], box_lo=bx[1])
        else:
            out['note'] = 'CHoCH+BOS 5m у бік розряду немає'
        return out
    out.update(t=bos['t'], edge=bos['edge'], box_hi=bos['hi'], box_lo=bos['lo'], held=1,
               choch_t=bos['choch_t'])
    try:
        lp = float(live_price) if live_price is not None else None
    except (TypeError, ValueError):
        lp = None
    if lp is not None and sg * (lp - bos['edge']) <= 0:
        out.update(state='pending', note='CHoCH+BOS 5m є, але ціна зараз повернулась у коробку')
    else:
        out.update(state='confirmed', note='CHoCH+BOS 5m за межею коробки 1h')
    return out


def vob_detect(closed_5m: List[Dict], params: Optional[Dict] = None) -> Optional[Dict]:
    """Детектор Volumized OB (той самий, що бокс бота) над 5m-свічками; усі живі
    блоки без злиття. Немає детектора → None."""
    p = dict(VOB_DEFAULTS, **(params or {}))
    try:
        from detection.volumized_ob import detect_volumized_obs
    except Exception:
        try:
            detect_volumized_obs = _sib('volumized_ob').detect_volumized_obs
        except Exception:
            return None
    return detect_volumized_obs(_clean_bars(closed_5m), swing_length=int(p['swing_length']),
                                ob_end_method=str(p['ob_end_method']),
                                max_atr_mult=float(p['max_atr_mult']),
                                zone_count=str(p.get('zone_count') or 'High'),
                                combine_obs=bool(p.get('combine_obs', False)))


def vob1h_check(obs: Optional[Dict], side: str) -> Dict:
    """📦 Друге підтвердження розряду (24.11). ЧИСТА.
    `obs` — результат детектора Volumized OB над 1h-свічками (з ЖИВОЮ, як на
    графіку). Береться РІВНО `latest_ob` — самий останній блок, що бот малює
    останнім; старіший блок у бік НЕ підставляється.
    → {ok, side, t, top, bottom, breaker, note}: ok лише коли останній VOB 1H
    у бік `side` і не зламаний (breaker)."""
    sd = str(side).upper()
    out = {'ok': False, 'side': None, 't': None, 'top': None, 'bottom': None,
           'breaker': False, 'note': ''}
    if obs is None:
        out['note'] = 'VOB 1H не порахувався (немає закритих свічок 1h)'
        return out
    lo = (obs or {}).get('latest_ob')
    if not lo:
        out['note'] = 'останнього VOB 1H на графіку немає'
        return out
    vs = 'LONG' if lo.get('type') == 'Bull' else 'SHORT'
    try:
        out.update(top=float(lo.get('top')), bottom=float(lo.get('bottom')))
    except (TypeError, ValueError):
        pass
    out.update(side=vs, t=_bar_sec({'t': lo.get('formation_time') or lo.get('start_time')}),
               breaker=bool(lo.get('breaker')))
    if vs != sd:
        out['note'] = f'останній VOB 1H — {vs}, проти розряду {sd}'
    elif lo.get('breaker'):
        out['note'] = f'останній VOB 1H ({vs}) зламано — breaker'
    else:
        out.update(ok=True, note=f'останній VOB 1H — {vs}, у бік розряду')
    return out


def discharge_state_vob(closed_1h: List[Dict], closed_5m: List[Dict], side: str,
                        live_price=None, obs=None, box: int = DISCHARGE_BOX,
                        look_sec: float = STRUCT_LOOK_SEC, params: Optional[Dict] = None) -> Dict:
    """💥 Підтвердження розряду через 5m VOB (24.11, ЛИШЕ тінь). ЧИСТА.
    Той самий детектор Volumized OB, що й бокс у боті (`detect_volumized_obs`,
    параметри «📦 Volumized OB Trend»). VOB у бік `side` народжується на свічці,
    що ЗАКРИЛАСЬ за свінгом; підтвердження — ця свічка закрилась ще й ЗА МЕЖЕЮ
    24-год коробки 1h (рівень Пружини). Зона блоку — природний рівень стопа.
    • такий VOB є, не зламаний, жива ціна за межею → 'confirmed';
    • ціна пробила низ (LONG) / верх (SHORT) блоку — він став breaker → 'false';
    • VOB у бік є, але народився в коробці / ціна повернулась у коробку → 'pending';
    • інакше → 'none'. `obs` — готовий результат детектора (для тестів).
    → як інші стани + zone (рівень зламу), zone_top/zone_bot, vol_pct (частка
    переважної сторони обсягу блоку, %)."""
    sd = str(side).upper()
    sg = 1.0 if sd == 'LONG' else -1.0
    out = {'state': 'none', 'edge': None, 'box_hi': None, 'box_lo': None, 't': None,
           'held': 0, 'need': 1, 'tf': 'vob', 'note': ''}
    p = dict(VOB_DEFAULTS, **(params or {}))
    h = _clean_bars(closed_1h)
    m5 = _clean_bars(closed_5m)
    if len(h) < box or len(m5) < int(p['swing_length']) + 5:
        out['note'] = 'замало закритих свічок'
        return out
    if obs is None:
        obs = vob_detect(m5, p)
        if obs is None:
            out['note'] = 'немає детектора VOB'
            return out
    lst = (obs or {}).get('all_bullish' if sd == 'LONG' else 'all_bearish') or []
    close_at = {_bar_sec(b): _bar_close(b) for b in m5}
    ends = [_bar_sec(b) + 3600 for b in h]
    t_end = _bar_sec(m5[-1]) or 0

    def _box_at(t):
        j = 0
        while j < len(h) and ends[j] <= t:
            j += 1
        return _hl_box(h[j - box:j]) if j >= box else None

    cand, inside = [], None
    for ob in lst:
        ft = _bar_sec({'t': ob.get('formation_time') or ob.get('start_time')})
        if ft is None or ft < t_end - look_sec:
            continue
        bx = _box_at(ft)
        c = close_at.get(ft)
        if not bx or c is None:
            continue
        edge = bx[0] if sd == 'LONG' else bx[1]
        if sg * (c - edge) > 0:
            cand.append((ft, ob, bx, edge))
        elif inside is None or ft > inside[0]:
            inside = (ft, bx, edge)
    if not cand:
        if inside:
            out.update(state='pending', t=inside[0], edge=inside[2], box_hi=inside[1][0],
                       box_lo=inside[1][1], note='VOB 5m у бік є, але народився в коробці 1h')
        else:
            out['note'] = 'VOB 5m у бік розряду за межею коробки немає'
        return out
    cand.sort(key=lambda x: x[0])
    cur = None
    for ft, ob, bx, edge in cand:        # перший робочий; зламаний — беремо наступний
        bt = _bar_sec({'t': ob.get('break_time')}) if ob.get('breaker') else None
        if cur is None or (cur[4] is not None and cur[4] <= ft):
            cur = (ft, ob, bx, edge, bt)
    ft, ob, bx, edge, bt = cur
    zone = float(ob['bottom']) if sd == 'LONG' else float(ob['top'])
    vol = float(ob.get('ob_volume') or 0)
    dom = max(float(ob.get('ob_high_volume') or 0), float(ob.get('ob_low_volume') or 0))
    out.update(t=ft, edge=edge, box_hi=bx[0], box_lo=bx[1], held=1, zone=zone,
               zone_top=float(ob['top']), zone_bot=float(ob['bottom']),
               vol_pct=round(100.0 * dom / vol, 1) if vol > 0 else None)
    if bt is not None:
        out.update(state='false', note='ціна пробила зону VOB 5m — блок зламано (хибний старт)')
        out['false_t'] = bt
        return out
    try:
        lp = float(live_price) if live_price is not None else None
    except (TypeError, ValueError):
        lp = None
    if lp is not None and sg * (lp - edge) <= 0:
        out.update(state='pending', note='VOB 5m за межею є, але ціна зараз повернулась у коробку')
    else:
        out.update(state='confirmed', note='VOB 5m народився за межею коробки 1h')
    return out


def discharge_state_vob_last(closed_1h: List[Dict], closed_5m: List[Dict], side: str,
                             live_price=None, obs=None, box: int = DISCHARGE_BOX,
                             look_sec: float = STRUCT_LOOK_SEC, params: Optional[Dict] = None) -> Dict:
    """💥 Тінь «5m VOB останній» (24.11). ЧИСТА.
    Старт розряду — поточна серія закритих 5m-свічок ЗА МЕЖЕЮ 24-год коробки 1h
    (свічка назад у коробці серію обриває). У цей момент дивимось на ОСТАННІЙ VOB
    5m — САМИЙ ОСТАННІЙ блок, РІВНО той, що бот малює останнім (`latest_ob`
    детектора: параметри «📦 Volumized OB Trend» разом із Zone Count і Combine,
    свічки з ЖИВОЮ). Лише він: старіший блок у бік НЕ підставляється, якщо
    останній — проти. Він може народитись і ДО старту:
    • останній VOB у бік розряду, не зламаний, жива ціна за межею → 'confirmed'
      (`t` = пізніше з (старт серії, народження блоку): новий блок або нова
      серія — нова точка);
    • останній VOB у бік, але зламаний (breaker) → 'false';
    • останній VOB ПРОТИ розряду / VOB немає / ціна в коробці → 'pending';
    • серії за межею немає → 'none'.
    → як `discharge_state_vob` (+ `start_t`, `vob_t`, `vob_side`)."""
    sd = str(side).upper()
    sg = 1.0 if sd == 'LONG' else -1.0
    out = {'state': 'none', 'edge': None, 'box_hi': None, 'box_lo': None, 't': None,
           'held': 0, 'need': 1, 'tf': 'vobl', 'note': ''}
    p = dict(VOB_DEFAULTS, **(params or {}))
    h = _clean_bars(closed_1h)
    m5 = _clean_bars(closed_5m)
    if len(h) < box or len(m5) < int(p['swing_length']) + 5:
        out['note'] = 'замало закритих свічок'
        return out
    ends = [_bar_sec(b) + 3600 for b in h]
    t_end = _bar_sec(m5[-1]) or 0

    def _box_at(t):
        j = 0
        while j < len(h) and ends[j] <= t:
            j += 1
        return _hl_box(h[j - box:j]) if j >= box else None

    start, sbx, sedge = None, None, None      # поточна серія закриттів за межею
    for b in m5:
        t, c = _bar_sec(b), _bar_close(b)
        if t is None or c is None or t < t_end - look_sec:
            continue
        bx = _box_at(t)
        if not bx:
            continue
        edge = bx[0] if sd == 'LONG' else bx[1]
        if sg * (c - edge) > 0:
            if start is None:
                start, sbx, sedge = t, bx, edge
        else:
            start = None
    if start is None:
        out['note'] = 'розряду немає: закриття 5m у коробці 1h'
        return out
    out.update(edge=sedge, box_hi=sbx[0], box_lo=sbx[1], start_t=start)
    if obs is None:
        obs = vob_detect(m5, p)
        if obs is None:
            out['note'] = 'немає детектора VOB'
            return out
    last = None          # САМИЙ ОСТАННІЙ VOB = `latest_ob` детектора (той, що бот малює останнім)
    lo = (obs or {}).get('latest_ob')
    if lo:
        ft = _bar_sec({'t': lo.get('formation_time') or lo.get('start_time')})
        if ft is not None:
            last = (ft, 'LONG' if lo.get('type') == 'Bull' else 'SHORT', lo)
    else:
        for key, sd_ob in (('all_bullish', 'LONG'), ('all_bearish', 'SHORT')):
            for ob in (obs or {}).get(key) or []:
                ft = _bar_sec({'t': ob.get('formation_time') or ob.get('start_time')})
                if ft is not None and (last is None or ft > last[0]):
                    last = (ft, sd_ob, ob)
    if last is None:
        out.update(state='pending', note='розряд іде, VOB 5m на графіку немає')
        return out
    ft, vside, ob = last
    out.update(vob_t=ft, vob_side=vside)
    if vside != sd:
        out.update(state='pending', note=f'останній VOB 5m — {vside}, проти розряду {sd}')
        return out
    zone = float(ob['bottom']) if sd == 'LONG' else float(ob['top'])
    vol = float(ob.get('ob_volume') or 0)
    dom = max(float(ob.get('ob_high_volume') or 0), float(ob.get('ob_low_volume') or 0))
    out.update(t=max(ft, start), held=1, zone=zone, zone_top=float(ob['top']),
               zone_bot=float(ob['bottom']), vol_pct=round(100.0 * dom / vol, 1) if vol > 0 else None)
    if ob.get('breaker'):
        out.update(state='false', note='останній VOB 5m у бік розряду зламано')
        out['false_t'] = _bar_sec({'t': ob.get('break_time')})
        return out
    try:
        lp = float(live_price) if live_price is not None else None
    except (TypeError, ValueError):
        lp = None
    if lp is not None and sg * (lp - sedge) <= 0:
        out.update(state='pending', note='останній VOB 5m у бік є, але ціна зараз повернулась у коробку')
    else:
        out.update(state='confirmed', note='розряд іде, останній VOB 5m — у бік розряду'
                   + (' (народився до старту)' if ft < start else ''))
    return out


def dis_shadow_stats(recs: List[Dict]) -> Dict:
    """📊 Тіньове порівняння підтвердження розряду 1H проти 15m (23.11). ЧИСТА.
    Кожен запис — точка, де розряд став «підтвердженим» на своєму TF, і що ціна
    робила далі (у % у бік входу). → {tf: {n, done, win_pct, avg_last, avg_mfe,
    avg_mae, false_pct, avg_lead_min}}."""
    out = {}
    for tf in DIS_SHADOW_TFS:
        rs = [r for r in recs or [] if r.get('tf') == tf]
        n = len(rs)
        if not n:
            out[tf] = {'n': 0}
            continue
        def _avg(k):
            v = [float(r[k]) for r in rs if r.get(k) is not None]
            return round(sum(v) / len(v), 3) if v else None
        lead = []
        if tf != '1h':          # на скільки хвилин раніше за 1h той самий пробій
            for r in recs or []:
                if r.get('tf') != '1h':
                    continue
                v = (r.get('lead') or {}).get(tf)
                if v is None and tf == '15m':
                    v = r.get('lead_min')            # старий формат
                if v is not None:
                    lead.append(float(v))
        out[tf] = {'n': n, 'done': sum(1 for r in rs if r.get('done')),
                   'win_pct': round(100.0 * sum(1 for r in rs if (r.get('last') or 0) > 0) / n, 1),
                   'avg_last': _avg('last'), 'avg_mfe': _avg('mfe'), 'avg_mae': _avg('mae'),
                   'false_pct': round(100.0 * sum(1 for r in rs if r.get('false_t')) / n, 1),
                   'avg_lead_min': round(sum(lead) / len(lead), 1) if lead else None}
        zr = [r for r in rs if r.get('zone') is not None]
        if zr:      # 5m VOB: стоп за зоною блоку — як далеко і як часто вибивало б
            out[tf]['zone_n'] = len(zr)
            out[tf]['zone_hit_pct'] = round(100.0 * sum(1 for r in zr if r.get('zone_t')) / len(zr), 1)
            rk = [float(r['risk']) for r in zr if r.get('risk') is not None]
            out[tf]['avg_risk'] = round(sum(rk) / len(rk), 3) if rk else None
            rr = [float(r['mfe']) / float(r['risk']) for r in zr
                  if r.get('risk') and float(r['risk']) > 0 and r.get('mfe') is not None]
            out[tf]['avg_mfe_r'] = round(sum(rr) / len(rr), 2) if rr else None
    return out


def dis_shadow_update(rec: Dict, price, closed_1h, now: float,
                      horizon: float = None) -> Dict:
    """Оновити тіньовий запис ціною і закритими 1h-свічками. ЧИСТА (мутує rec).
    Рух — % від ціни підтвердження в бік входу; хибний старт — закрита після
    підтвердження 1h-свічка назад за межею пробою (те саме правило, що вихід)."""
    horizon = DIS_SHADOW_HOURS * 3600 if horizon is None else horizon
    if rec.get('done'):
        return rec
    sg = 1.0 if rec.get('side') == 'LONG' else -1.0
    try:
        px0, px = float(rec['px']), float(price)
        if px0 > 0 and px > 0:
            mv = sg * (px / px0 - 1) * 100.0
            rec['last'] = round(mv, 3)
            rec['mfe'] = round(max(float(rec.get('mfe') or 0.0), mv), 3)
            rec['mae'] = round(min(float(rec.get('mae') or 0.0), mv), 3)
            # 5m VOB: ціна дійшла до зони блоку (там стояв би стоп) — один раз.
            if rec.get('zone') is not None and not rec.get('zone_t') \
                    and sg * (px - float(rec['zone'])) <= 0:
                rec['zone_t'] = int(now)
                rec['zone_last'] = rec.get('last')
    except (TypeError, ValueError, KeyError):
        pass
    if not rec.get('false_t'):
        fs = false_start_hit(rec.get('side'), rec.get('edge'), closed_1h or [], rec.get('t'))
        if fs:
            rec['false_t'] = fs['t']
            rec['false_last'] = rec.get('last')
    if now - float(rec.get('t') or now) >= horizon:
        rec['done'] = True
    return rec


def false_start_hit(side: str, edge, closed: List[Dict], opened_at) -> Optional[Dict]:
    """🪤 Хибний старт ПІСЛЯ входу (23.11): закрита 1h-свічка, що закінчилась
    після відкриття угоди, закрилась назад за межею пробою (у коробці) → вихід.
    ЧИСТА. → {t, close, edge} | None."""
    try:
        e = float(edge)
        oa = float(opened_at)
    except (TypeError, ValueError):
        return None
    sg = 1.0 if str(side).upper() == 'LONG' else -1.0
    for b in closed or []:
        t, c = _bar_sec(b), _bar_close(b)
        if t is None or c is None or t + 3600 <= oa:
            continue
        if sg * (c - e) <= 0:
            return {'t': t, 'close': c, 'edge': e}
    return None


def scen_allowed(r: Optional[Dict], allowed) -> bool:
    """Готовий сценарій входу — серед дозволених у налаштуваннях ШІ. ЧИСТА."""
    key = ((r or {}).get('scenario') or {}).get('key')
    return bool(key) and key in set(allowed or ())


def ai_limits_block(side: str, s: Dict, open_n: int, opens: List[float],
                    last: Optional[tuple], now: float) -> str:
    """Власні правила ШІ (не ворота бота). '' → можна. ЧИСТА."""
    sd = str(side).upper()
    if sd == 'LONG' and not s.get('allow_long', True):
        return 'LONG вимкнено в налаштуваннях ШІ'
    if sd == 'SHORT' and not s.get('allow_short', True):
        return 'SHORT вимкнено в налаштуваннях ШІ'
    if open_n >= int(s.get('max_open') or 5):
        return f"відкрито угод ШІ {open_n} — ліміт {s.get('max_open')}"
    recent = [t for t in (opens or []) if now - t < OPEN_WINDOW_SEC]
    if len(recent) >= int(s.get('max_per_30m') or 3):
        return f"за 30 хв уже {len(recent)} нових угод ШІ — ліміт {s.get('max_per_30m')}"
    if last and last[0] != sd:
        cd = float(s.get('flip_cooldown_min') or 0) * 60
        if cd > 0 and now - float(last[1] or 0) < cd:
            left = int((cd - (now - float(last[1]))) // 60) + 1
            return f'протилежний сигнал по монеті був {int((now - float(last[1])) // 60)} хв тому — перевертання не раніше ніж через {left} хв'
    return ''


# ── 🏆 ЕТАЛОН ВІДМІННИХ УГОД (вимога 22.11) ─────────────────────────────────
# «створити базу еталон відмінних угод і щоб бот самостійно її поповнював для
# удосконалення самого себе». Кожна угода ШІ несе ОЗНАКИ ВХОДУ (знімок у
# момент сигналу); закрилась відмінно → ознаки йдуть у базу. Новий вхід
# порівнюється з еталонами (k найближчих) — схожість і є «досвід бота».
def entry_features(side: str, er: Optional[Dict], rec: Optional[Dict],
                   brain: Optional[Dict]) -> Dict:
    """Ознаки входу 0..1 (одна шкала для відстані). ЧИСТА."""
    sd = str(side).upper()
    er, rec, brain = er or {}, rec or {}, brain or {}
    parts = er.get('parts') or {}

    def _n(v, k=100.0):
        try:
            return _c(float(v) / k)
        except (TypeError, ValueError):
            return None
    pot = rec.get('pot_long' if sd == 'LONG' else 'pot_short')
    try:
        pl = float(rec.get('prob_long'))
        prob = pl if sd == 'LONG' else 100.0 - pl
    except (TypeError, ValueError):
        prob = None
    try:
        flow = (float(er.get('flow_raw')) + 1.0) / 2.0
    except (TypeError, ValueError):
        flow = None
    rr = (brain.get('plan') or {}).get('net_rr')
    out = {'charge': _n(rec.get('charge')), 'pot': _n(pot), 'prob': _n(prob),
           'lean': _n(parts.get('lean'), 1.0), 'break': _n(parts.get('break'), 1.0),
           'impulse': _n(parts.get('impulse'), 1.0), 'flow': _c(flow) if flow is not None else None,
           'conf': _n(brain.get('confidence')), 'rr': _n(rr, 5.0)}
    return {k: round(v, 4) for k, v in out.items() if v is not None}


def feat_distance(a: Dict, b: Dict) -> Optional[float]:
    """Середньоквадратична відстань по СПІЛЬНИХ ознаках (≥4). ЧИСТА."""
    ks = [k for k in FEAT_KEYS if k in (a or {}) and k in (b or {})]
    if len(ks) < 4:
        return None
    return (sum((float(a[k]) - float(b[k])) ** 2 for k in ks) / len(ks)) ** 0.5


def etalon_similarity(feat: Dict, base: List[Dict], side: Optional[str] = None,
                      k: int = ETALON_K) -> Optional[float]:
    """Схожість 0..1 на k найближчих еталонів (того самого боку, якщо є). ЧИСТА."""
    rows = [e for e in (base or []) if isinstance(e, dict) and e.get('feat')]
    if side:
        same = [e for e in rows if e.get('side') == str(side).upper()]
        rows = same if len(same) >= max(3, k) else rows
    ds = sorted(d for d in (feat_distance(feat, e['feat']) for e in rows) if d is not None)
    if not ds:
        return None
    top = ds[:max(1, k)]
    return round(sum(1.0 - min(1.0, d) for d in top) / len(top), 3)


def trade_r(closed: Dict) -> Optional[float]:
    """Результат угоди в R (від ризику плану). ЧИСТА."""
    try:
        pnl = float(closed.get('pnl_pct'))
        risk = float(((closed.get('ai_plan') or {}).get('risk_pct')))
        return round(pnl / risk, 2) if risk > 0 else None
    except (TypeError, ValueError):
        return None


def etalon_excellent(closed: Dict, s: Dict) -> bool:
    """Угода ШІ — ВІДМІННА: ≥ `etalon_min_r` R, а без стопа плану — результат
    ≥ ETALON_FALLBACK_PNL %. ЧИСТА."""
    if not (closed or {}).get('ai_feat'):
        return False
    r = trade_r(closed)
    if r is not None:
        return r >= float(s.get('etalon_min_r') or 1.5)
    try:
        return float(closed.get('pnl_pct')) >= ETALON_FALLBACK_PNL
    except (TypeError, ValueError):
        return False


ETALON_ANTI_DB = 'ai_etalon_anti_v1'   # ❌ антибаза: збиткові угоди ШІ (контраст для навчання)
ETALON_ANTI_R = -0.5                   # «невдала» = результат ≤ −0.5R (без плану — ≤ −1%)
ETALON_ANTI_PNL = -1.0
ETALON_EVENTS_MAX = 25


def etalon_failed(closed: Dict) -> bool:
    """Угода ШІ — НЕВДАЛА (в антибазу): ≤ ETALON_ANTI_R, без плану — ≤ ETALON_ANTI_PNL %. ЧИСТА."""
    if not (closed or {}).get('ai_feat'):
        return False
    r = trade_r(closed)
    if r is not None:
        return r <= ETALON_ANTI_R
    try:
        return float(closed.get('pnl_pct')) <= ETALON_ANTI_PNL
    except (TypeError, ValueError):
        return False


def _fnum(v, nd=4):
    try:
        f = float(v)
        return round(f, nd) if f == f else None
    except (TypeError, ValueError):
        return None


def ai_entry_snapshot(side: str, er: Optional[Dict], rec: Optional[Dict], brain: Optional[Dict],
                      sim: Optional[float] = None, px=None, now: Optional[float] = None,
                      adir: Optional[Dict] = None, anti_sim: Optional[float] = None) -> Dict:
    """📸 Повний знімок МОМЕНТУ ВХОДУ ШІ для бази еталонів: сценарій з етапами,
    складові балу, Пружина, чек-лист мозку, план, напрямок аномалій, схожість.
    Компактно (без сирих барів). ЧИСТА."""
    er, rec, brain = er or {}, rec or {}, brain or {}
    sc = er.get('scenario') or {}
    sp = {k: rec.get(k) for k in ('charge', 'pot_long', 'pot_short', 'prob_long', 'state', 'phase',
                                   'lean', 'lean_pp', 'box_hi', 'box_lo', 'atr', 'move_long_pct',
                                   'move_short_pct', 'tgt_long', 'tgt_short', 'move24_pct')
          if rec.get(k) is not None}
    ctx = brain.get('context') or {}
    return {
        't': int(now or time.time()), 'side': str(side).upper(), 'px': _fnum(px, 8),
        'scenario': {'key': sc.get('key'), 'name': sc.get('name'), 'done': sc.get('done'),
                     'total': sc.get('total'), 'last_t': sc.get('last_t'),
                     'stages': [{'label': x.get('label'), 'done': bool(x.get('done')), 't': x.get('t')}
                                for x in (sc.get('stages') or [])][:8]},
        'score': er.get('score'), 'parts': dict(er.get('parts') or {}),
        'spring': sp,
        'brain': {'ok': brain.get('ok'), 'confidence': brain.get('confidence'),
                  'checks': [{'label': c.get('label'), 'ok': c.get('ok'), 'note': c.get('note')}
                             for c in (brain.get('checks') or [])][:12],
                  'plan': brain.get('plan'),
                  'strict_ready': (brain.get('strict') or {}).get('ready'),
                  'strict_vetoes': list((brain.get('strict') or {}).get('vetoes') or [])[:6],
                  'context': ctx.get('note') if isinstance(ctx, dict) else None},
        'adir': ({k: (adir or {}).get(k) for k in ('dir', 'state', 'long', 'short')} if adir else None),
        'sim': sim, 'anti_sim': anti_sim,
    }


def etalon_lessons(closed: Dict) -> List[str]:
    """🎓 Уроки угоди людською мовою (для самовдосконалення). ЧИСТА."""
    out: List[str] = []
    pnl, peak = _fnum(closed.get('pnl_pct'), 2), _fnum(closed.get('peak_pnl_pct'), 2)
    mae = _fnum(closed.get('mae_pnl_pct'), 2)
    risk = _fnum((closed.get('ai_plan') or {}).get('risk_pct'), 4)
    if pnl is not None and peak and peak > 0:
        cap = round(100 * pnl / peak)
        out.append(f'взято {cap}% піку ({pnl:+.2f}% з {peak:+.2f}%)')
        if cap < 50:
            out.append('вихід запізнився: віддано більше половини руху')
    if mae is not None and risk:
        out.append(f'найглибший відкат {mae:+.2f}% = {round(mae / risk, 2)}R від стопа плану')
    summ = closed.get('research_summary') or {}
    for k, v in (summ.get('first') or {}).items() if isinstance(summ.get('first'), dict) else []:
        if isinstance(v, dict) and v.get('pnl') is not None and pnl is not None:
            better = float(v['pnl']) > pnl
            out.append(f"на першій події «{k}» було б {float(v['pnl']):+.2f}%" + (' — краще' if better else ''))
    ex = closed.get('ai_exit') or {}
    if ex.get('verdict'):
        out.append(f"вердикт виходу: {ex.get('verdict_ua') or ex.get('verdict')}")
    ent = closed.get('ai_entry') or {}
    if ent.get('sim') is not None:
        out.append(f"схожість на еталони при вході {ent['sim']}")
    return out[:10]


def etalon_record(closed: Dict) -> Dict:
    """🏆 Запис бази еталонів / антибази: ключі для k-NN (feat, side, scen, r) +
    УСЕ корисне для самовдосконалення — момент входу, план, шлях угоди, вихід,
    події дослідження, уроки. ЧИСТА."""
    try:
        dur = int(float(closed.get('closed_at')) - float(closed.get('opened_at')))
    except (TypeError, ValueError):
        dur = None
    plan = closed.get('ai_plan') or {}
    risk = _fnum(plan.get('risk_pct'), 4)
    peak, mae = _fnum(closed.get('peak_pnl_pct'), 3), _fnum(closed.get('mae_pnl_pct'), 3)
    pnl = _fnum(closed.get('pnl_pct'), 3)
    ex = closed.get('ai_exit') or {}
    rs = closed.get('research') if isinstance(closed.get('research'), dict) else {}
    evs = rs.get('ev') if isinstance(rs, dict) else None
    if evs is None and isinstance(closed.get('research'), list):
        evs = closed.get('research')
    t0 = _fnum(closed.get('opened_at'), 0) or 0
    events = []
    for e in (evs or [])[-ETALON_EVENTS_MAX:]:
        if isinstance(e, dict):
            events.append({'dt': int((_fnum(e.get('t'), 0) or t0) - t0), 'k': e.get('k') or e.get('kind'),
                           'n': e.get('note') or e.get('label'), 'pnl': e.get('pnl')})
    return {'sym': closed.get('symbol'), 'side': str(closed.get('side') or '').upper(),
            't': int(float(closed.get('closed_at') or time.time())),
            'scen': closed.get('ai_scenario'), 'feat': dict(closed.get('ai_feat') or {}),
            'pnl': pnl, 'peak': peak, 'mae': mae,
            'r': trade_r(closed), 'peak_r': round(peak / risk, 2) if (peak is not None and risk) else None,
            'mae_r': round(mae / risk, 2) if (mae is not None and risk) else None,
            'capture': round(pnl / peak, 2) if (pnl is not None and peak and peak > 0) else None,
            'reason': closed.get('reason'), 'dur_sec': dur,
            'paper': bool(closed.get('is_paper', closed.get('paper'))),
            'entry_px': _fnum(closed.get('entry_price'), 8), 'exit_px': _fnum(closed.get('exit_price'), 8),
            'plan': ({k: plan.get(k) for k in ('stop', 'target', 'risk_pct', 'reward_pct', 'net_rr',
                                                'target_src')} if plan else None),
            'entry': closed.get('ai_entry'),
            'spring_open': closed.get('spring_open'),
            'exit': ({'verdict': ex.get('verdict'), 'verdict_ua': ex.get('verdict_ua'),
                      'groups': ex.get('groups'), 'r': ex.get('r'), 'peak_r': ex.get('peak_r'),
                      'charge': ex.get('charge'),
                      'checklist': [{'label': c.get('label'), 'v': c.get('v'), 'say': c.get('say')}
                                    for c in (ex.get('checklist') or [])][:10]} if ex else None),
            'partials': list(closed.get('partial_closes_done') or [])[:6],
            'events': events,
            'summary': closed.get('research_summary'),
            'lessons': etalon_lessons(closed)}


ETALON_MATURE_GOOD = 10   # мозок еталонів вирішує, коли є ≥10 відмінних
ETALON_MATURE_ANTI = 5    # … і ≥5 невдалих угод (є з чим порівнювати)
ETALON_TP_CAPTURE = 0.8   # ціль = 0.8 × типовий пік схожих еталонів (реалістичний вихід)


def etalon_brain(feat: Dict, good: List[Dict], anti: List[Dict], side: Optional[str] = None,
                 s: Optional[Dict] = None, k: int = ETALON_K) -> Dict:
    """🏆🧠 ГОЛОВНИЙ МОЗОК ШІ — досвід самого ШІ (вимога 22.11: «база Еталон —
    основний мозок»). k найближчих угод з ОБОХ баз (відмінні + невдалі) того
    самого боку голосують зважено за близькістю:
      p_good — частка «відмінних» серед схожих входів;
      exp_r  — очікуваний результат у R; tp_r — типовий пік схожих відмінних.
    `mature` — бази зібрані: тоді вердикт `take`/`skip` є ОСТАТОЧНИМ рішенням
    на вхід; до того `learning` — рішення за правилами (вони й наповнюють базу).
    ЧИСТА."""
    s = s or {}
    sd = str(side or '').upper()

    def _pool(base, lab):
        rows = [e for e in (base or []) if isinstance(e, dict) and e.get('feat')]
        same = [e for e in rows if e.get('side') == sd] if sd else rows
        rows = same if len(same) >= 3 else rows
        return [(d, lab, e) for e in rows for d in [feat_distance(feat, e['feat'])] if d is not None]
    g_n = len([e for e in (good or []) if e.get('feat')])
    a_n = len([e for e in (anti or []) if e.get('feat')])
    mature = g_n >= ETALON_MATURE_GOOD and a_n >= ETALON_MATURE_ANTI
    out = {'n_good': g_n, 'n_anti': a_n, 'mature': mature, 'need_good': ETALON_MATURE_GOOD,
           'need_anti': ETALON_MATURE_ANTI, 'p_good': None, 'exp_r': None, 'tp_r': None,
           'nearest': [], 'verdict': 'learning', 'reason': ''}
    pool = sorted(_pool(good, 1) + _pool(anti, 0), key=lambda x: x[0])[:max(1, k)]
    if not pool:
        out['reason'] = 'досвіду ще немає — вирішують правила'
        return out
    ws = [max(0.02, 1.0 - min(1.0, d)) for d, _, _ in pool]
    W = sum(ws)
    out['p_good'] = round(sum(w for w, (_, lab, _) in zip(ws, pool) if lab) / W, 3)
    rs = [(w, e.get('r')) for w, (_, _, e) in zip(ws, pool) if isinstance(e.get('r'), (int, float))]
    if rs:
        out['exp_r'] = round(sum(w * r for w, r in rs) / sum(w for w, _ in rs), 2)
    pk = sorted(e.get('peak_r') for _, lab, e in pool if lab and isinstance(e.get('peak_r'), (int, float)))
    if len(pk) >= 3:
        out['tp_r'] = round(ETALON_TP_CAPTURE * pk[len(pk) // 2], 2)
    out['nearest'] = [{'sym': e.get('sym'), 'side': e.get('side'), 'good': bool(lab), 'r': e.get('r'),
                       'd': round(d, 3)} for d, lab, e in pool]
    if not mature:
        out['reason'] = (f'мозок еталонів вчиться: відмінних {g_n}/{ETALON_MATURE_GOOD}, '
                         f'невдалих {a_n}/{ETALON_MATURE_ANTI} — вирішують правила')
        return out
    p_min = float(s.get('etalon_min_p') or 0.6)
    sim_min = float(s.get('etalon_min_sim') or 0)
    near = 1.0 - pool[0][0]
    if near < sim_min:
        out['verdict'], out['reason'] = 'skip', f'немає схожого досвіду (найближчий {near:.2f} < {sim_min:.2f})'
    elif out['p_good'] < p_min:
        out['verdict'], out['reason'] = 'skip', (f"схожі входи частіше були невдалими "
                                                 f"(відмінних {round(out['p_good'] * 100)}% < {round(p_min * 100)}%)")
    elif out['exp_r'] is not None and out['exp_r'] <= 0:
        out['verdict'], out['reason'] = 'skip', f"очікуваний результат схожих угод {out['exp_r']}R"
    else:
        out['verdict'] = 'take'
        out['reason'] = (f"схожі входи: відмінних {round(out['p_good'] * 100)}%"
                         + (f" · очікувано {out['exp_r']}R" if out['exp_r'] is not None else ''))
    return out


def etalon_plan(plan: Optional[Dict], side: str, eb: Optional[Dict]) -> Optional[Dict]:
    """🎯 Ціль угоди з досвіду: зрілий мозок еталонів і типовий пік схожих
    відмінних (tp_r) → ціль = вхід ± tp_r × ризик плану. ЧИСТА."""
    if not plan or not eb or not eb.get('mature') or not eb.get('tp_r'):
        return plan
    try:
        stop, tgt = float(plan['stop']), float(plan['target'])
        risk = float(plan.get('risk_pct'))
    except (TypeError, ValueError, KeyError):
        return plan
    if risk <= 0:
        return plan
    sd = str(side).upper()
    ref = stop / (1 - risk / 100.0) if sd == 'LONG' else stop / (1 + risk / 100.0)
    rw = float(eb['tp_r']) * risk
    new_t = ref * (1 + rw / 100.0) if sd == 'LONG' else ref * (1 - rw / 100.0)
    out = dict(plan)
    out.update({'target': round(new_t, 10), 'reward_pct': round(rw, 3), 'target_src': 'etalon',
                'target_prev': tgt, 'net_rr': round(float(eb['tp_r']), 2)})
    return out


def etalon_quality(e: Dict, kind: str = 'good') -> float:
    """Якість запису бази. Еталон: результат у R (без плану — % / 1.5) + половина
    взятої частки піку. Антибаза: що глибший збиток, то цінніший урок. ЧИСТА."""
    r = (e or {}).get('r')
    pnl = (e or {}).get('pnl')
    if isinstance(r, (int, float)):
        base = float(r)
    elif isinstance(pnl, (int, float)):
        base = float(pnl) / ETALON_FALLBACK_PNL
    else:
        base = 0.0
    if kind == 'good':
        cap = (e or {}).get('capture')
        return round(base + 0.5 * (float(cap) if isinstance(cap, (int, float)) else 0.0), 4)
    return round(-base, 4)


def etalon_prune_plan(base: List[Dict], kind: str = 'good', keep: int = ETALON_KEEP,
                      dup: float = ETALON_DUP_DIST) -> List[tuple]:
    """🏷 Що позначити на видалення в базі: (index, причина). Спершу оцінка ВСІХ
    записів, потім (1) дубль кращого — майже той самий вхід того самого боку,
    лишається якісніший; (2) база понад `keep` — найслабші. Дата не важить. ЧИСТА."""
    rows = list(base or [])
    q = [etalon_quality(e, kind) for e in rows]
    order = sorted(range(len(rows)), key=lambda i: (-q[i], -float(rows[i].get('t') or 0)))
    kept, out = [], []
    for i in order:
        e = rows[i]
        twin = None
        for j in kept:
            if rows[j].get('side') != e.get('side'):
                continue
            d = feat_distance(e.get('feat') or {}, rows[j].get('feat') or {})
            if d is not None and d < dup:
                twin = j
                break
        if twin is not None:
            out.append((i, f"дубль кращого входу {rows[twin].get('sym') or ''} "
                           f"(якість {q[twin]} > {q[i]})"))
        else:
            kept.append(i)
    if len(kept) > keep:
        for i in kept[keep:]:
            out.append((i, f'найслабший у переповненій базі (якість {q[i]})'))
    return sorted(out)


def learning_info(row: Dict, kind: str) -> float:
    """Показовість результату для навчання: що сильніший рух (у будь-який бік) і
    що більший розрив пік ↔ підсумок, то більше вчить. ЧИСТА."""
    def _a(v):
        return abs(float(v)) if isinstance(v, (int, float)) else 0.0
    if kind == 'research':
        pnl, peak = row.get('pnl'), row.get('peak')
        give = max(0.0, float(peak) - float(pnl)) if isinstance(peak, (int, float)) and isinstance(pnl, (int, float)) else 0.0
        return round(_a(pnl) + 0.5 * give + 0.5 * _a(row.get('mae')), 4)
    return round(max(_a(row.get('best')), _a(row.get('worst'))) + _a(row.get('end')), 4)


def learning_prune_plan(rows: List[Dict], kind: str, keep: int, protected=()) -> List[int]:
    """id найменш показових записів понад `keep` (захищені — ніколи). ЧИСТА."""
    prot = set(protected or ())
    free = [r for r in (rows or []) if (r.get('sym'), int(float(r.get('t') or 0))) not in prot]
    room = max(0, int(keep) - (len(rows or []) - len(free)))
    if len(free) <= room:
        return []
    free.sort(key=lambda r: (-learning_info(r, kind), -float(r.get('t') or 0)))
    return [r['id'] for r in free[room:] if r.get('id') is not None]


def ai_readiness(s: Dict, spring_on: bool, tm_on: bool, test_mode: bool,
                 coins: int = 0, ready: int = 0, open_n: int = 0,
                 tick: Optional[Dict] = None, now: Optional[float] = None,
                 anom_on: bool = True) -> Dict:
    """🤖 ГОЛОВНИЙ ТУМБЛЕР ШІ (22.11): чи може ШІ ЗАРАЗ відкривати угоди і що
    саме заважає. ЧИСТА функція. → {state, blockers[], book_ua, text}.
    state: off (тумблер вимкнено) · blocked (є перешкода) · watching (працює,
    точок входу немає) · ready (є готова точка) · full (ліміт угод ШІ)."""
    s = s or {}
    bl = []
    if not spring_on:
        bl.append('🌀 Пружина вимкнена — немає заряджених монет')
    if not anom_on:
        bl.append('⚠ «Аномалії обсягу» вимкнено — сценаріям входу немає з чого скластись')
    book = str(s.get('book') or 'auto')
    if book == 'real' and not tm_on:
        bl.append('книга «реальна», а Trade Manager вимкнено')
    if book == 'paper' and not test_mode:
        bl.append('книга «paper», а Test Mode вимкнено')
    if book == 'auto' and not tm_on and not test_mode:
        bl.append('Trade Manager і Test Mode вимкнені — книги немає')
    if not (s.get('allow_long', True) or s.get('allow_short', True)):
        bl.append('обидва напрямки ШІ вимкнені')
    if not (s.get('scenarios') or []):
        bl.append('не обрано жодного сценарію')
    book_ua = {'real': '🏦 реальна', 'paper': '🧪 paper'}.get(
        book, '📒 авто (' + ('реальна/paper' if tm_on else ('paper' if test_mode else '—')) + ')')
    mx = int(s.get('max_open') or 0)
    if not s.get('enabled'):
        st = 'off'
    elif bl:
        st = 'blocked'
    elif mx and open_n >= mx:
        st = 'full'
    elif ready:
        st = 'ready'
    else:
        st = 'watching'
    tk = tick or {}
    last_at = float(tk.get('last_at') or 0)
    err = ''
    if tk.get('last_err') and float(tk.get('err_at') or 0) >= last_at - 60:
        err = str(tk.get('last_err'))
    if st in ('watching', 'ready') and not last_at:
        st = 'starting'
    txt = {'off': 'вимкнено — угод ШІ не буде',
           'starting': 'перший перебір ще йде (до кількох хвилин після запуску)',
           'blocked': 'не може відкривати: ' + '; '.join(bl),
           'full': f'ліміт угод ШІ ({open_n}/{mx})',
           'ready': f'готових точок входу: {ready}',
           'watching': f'перебирає {coins} заряджених — точки входу ще немає'}[st]
    cand = tk.get('cand')
    if st == 'watching' and cand is not None and int(cand) != int(coins or 0):
        txt += f' (кандидатів {int(cand)}, оцінено {int(coins or 0)})'
    if err:
        txt += ' · ⚠ ' + err
    age = (float(now) - last_at) if (now and last_at) else None
    return {'state': st, 'blockers': bl, 'book_ua': book_ua, 'text': txt, 'error': err,
            'last_at': last_at or None, 'age': None if age is None else int(age),
            'coins': int(coins or 0), 'ready': int(ready or 0), 'open_n': int(open_n or 0),
            'max_open': mx}


def ai_heartbeat_text(view: Dict, s: Dict, etalon: Dict, open_n: int = 0,
                      opens_30m: int = 0) -> tuple:
    """🤖 Рядок «огляду» ШІ для 🧾 Логу (вимога 22.11: «по логу не зрозуміло, чи
    взагалі відбувається перебір варіантів»): скільки монет перебрано, точки
    входу, що формується, хто відсіяв, стан мозку еталонів. → (текст, ключ
    змін — набір готових точок). ЧИСТА."""
    rows = list((view or {}).values())
    n_wl = sum(1 for r in rows if r.get('src') != 'scan')
    ready, forming, brain_no, et_no, blocked = [], [], 0, 0, []
    for r in rows:
        p = r.get('pick') or {}
        sc = p.get('scenario') or {}
        side = r.get('best') or p.get('side')
        arrow = '▲' if side == 'LONG' else '▼'
        name = str(r.get('symbol') or '').replace('USDT', '')
        if r.get('ready'):
            if r.get('block'):
                blocked.append(f"{name} {arrow} {sc.get('key') or ''} — {r.get('block')}")
            else:
                ready.append(f"{name} {arrow} {sc.get('key') or ''}".strip())
            continue
        if sc.get('ready') and not (r.get('brain') or {}).get('ok', True):
            brain_no += 1
        eb = r.get('etalon') or {}
        if sc.get('ready') and eb.get('mature') and eb.get('verdict') == 'skip':
            et_no += 1
        if sc.get('done') and not sc.get('broken') and not sc.get('stale'):
            forming.append((sc.get('done', 0) / max(1, sc.get('total') or 1),
                            f"{name} {arrow} {sc.get('key') or ''} {sc.get('done')}/{sc.get('total')}"))
    forming.sort(key=lambda x: -x[0])
    parts = [f"перебрано {len(rows)} монет (🌀 {n_wl} · 🔭 {len(rows) - n_wl})"]
    parts.append(('✅ точки входу: ' + ', '.join(ready[:6])) if ready else '✅ точок входу немає')
    if blocked:
        parts.append('⛔ не відкрито: ' + '; '.join(blocked[:4])
                     + (f' (+{len(blocked) - 4})' if len(blocked) > 4 else ''))
    if forming:
        parts.append('⏳ формується: ' + ', '.join(t for _, t in forming[:5])
                     + (f' (+{len(forming) - 5})' if len(forming) > 5 else ''))
    if brain_no:
        parts.append(f'🧠 мозок графіка відсіяв {brain_no}')
    if et_no:
        parts.append(f'🏆🧠 мозок еталонів відсіяв {et_no}')
    et = etalon or {}
    parts.append('🏆🧠 мозок еталонів ' + ('вирішує' if et.get('mature') else
                 f"вчиться 🏆 {et.get('n', 0)}/{et.get('need_good', ETALON_MATURE_GOOD)} · "
                 f"❌ {et.get('anti_n', 0)}/{et.get('need_anti', ETALON_MATURE_ANTI)}"))
    scen = ', '.join((s or {}).get('scenarios') or ['E2'])
    if (s or {}).get('enabled'):
        parts.append(f'угоди ШІ: відкрито {open_n} (за 30 хв {opens_30m}) · сценарії {scen}')
    else:
        parts.append('⏸ ШІ ВИМКНЕНО — лише нагляд, угоди не відкриває')
    return '🤖 ШІ · огляд: ' + ' · '.join(parts), tuple(sorted(ready))


def ai_rank(view: Dict, sim: Optional[float]) -> float:
    """Ранг рекомендації ШІ. Зрілий мозок еталонів — ГОЛОВНА складова
    (ймовірність «відмінна» серед схожих входів), до того — готовність
    сценарію, прогрес, впевненість правил і схожість. ЧИСТА."""
    best = (view or {}).get('pick') or {}
    sc = best.get('scenario') or {}
    prog = (sc.get('done') or 0) / max(1, sc.get('total') or 1)
    conf = float(((view or {}).get('brain') or {}).get('confidence') or 0) / 100.0
    ready = 1.0 if view.get('ready') else 0.0
    eb = (view or {}).get('etalon') or {}
    if eb.get('mature') and eb.get('p_good') is not None:
        return round(100 * (0.5 * float(eb['p_good']) + 0.3 * ready + 0.1 * prog + 0.1 * conf), 1)
    return round(100 * (0.45 * ready + 0.2 * prog + 0.2 * conf + 0.15 * (sim if sim is not None else 0.5)), 1)


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
PRO_CACHE_TTL = {'5m': 150, '15m': 240, '1h': 900, '4h': 1800}
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


# 🧭 НАПРЯМОК ЗА ⚠ АНОМАЛІЯМИ (вимога 18.11) — для воріт «Через Пружину».
# Дослівно: «напрямок руху, який пропонує таблиця пружин, часто не встигає
# оновитись і суперечить… на напрямок потрібно орієнтуватись саме по показниках
# аномалії, а саме в яку сторону відбувались певні рухи ціни».
# Пружина рахується по ЗАКРИТИХ 1h-барах (раз на годину), тож її бік відстає;
# аномалії пишуться на кожному закритті 15m/1h. Читання — ТЕ САМЕ
# `read_anomalies`, що й вхід ШІ (рівні, підтвердження, вік, калібровка).
DIR_LOOK_SEC = 6 * 3600        # рухи за останні 6 год (старші важать менше)
DIR_MIN = 0.3                  # сила бокового свідчення (= ANOM_GATE виходу)
DIR_MARGIN = 0.15              # перевага над протилежним боком
DIR_TTL = 120                  # кеш рушія вважається свіжим


def anomaly_direction(anoms: List[Dict], now: float, levels=None,
                      choch: Optional[Dict] = None, cal: Optional[Dict] = None,
                      look: float = DIR_LOOK_SEC) -> Dict:
    """→ {'dir': LONG|SHORT|None, 'state': dir|mixed|weak|none, 'up', 'dn',
    'margin', 'n', 'notes'}. Сила боку = max(розворот, 🔥→🧽, 0.8·старт) +
    0.5·продовження (як `entry_score`). Напрямок — лише коли переможець ≥
    DIR_MIN і випереджає інший бік на DIR_MARGIN; обидва сильні й близькі →
    `mixed` (суперечливі рухи), слабкі → `weak`, аномалій немає → `none`."""
    rd = read_anomalies(anoms, now, levels, choch, _cal_part(cal, 'entry'), look=look)
    sc = {}
    for d in ('up', 'dn'):
        x = rd[d]
        sc[d] = round(_c(max(x['rev'], x['combo'], 0.8 * x['start']) + 0.5 * x['cont']), 3)
    n = len(rd['items'])
    win = 'up' if sc['up'] >= sc['dn'] else 'dn'
    lose = 'dn' if win == 'up' else 'up'
    margin = round(sc[win] - sc[lose], 3)
    if not n:
        state, dr = 'none', None
    elif sc[win] >= DIR_MIN and margin >= DIR_MARGIN:
        state, dr = 'dir', ('LONG' if win == 'up' else 'SHORT')
    elif sc[win] >= DIR_MIN:
        state, dr = 'mixed', None
    else:
        state, dr = 'weak', None
    notes = [('▲ ' + t) for t in rd['up']['notes'][:3]] + [('▼ ' + t) for t in rd['dn']['notes'][:3]]
    last = max((x['t'] for x in rd['items']), default=None)
    return {'dir': dr, 'state': state, 'up': sc['up'], 'dn': sc['dn'], 'margin': margin,
            'n': n, 'notes': notes, 'last_t': last}


ADIR_UA = {'dir': 'напрямок', 'mixed': 'суперечливі рухи', 'weak': 'слабкі рухи',
           'none': 'аномалій немає'}


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


# ── ⚠ ОСОБИСТІ СПОВІЩЕННЯ «АНОМАЛІЇ ОБСЯГУ» (вимога 18.11) ────────────────
# Дослівно: «оповіщення в телеграм по вибору монети, сигнал стосовно "Аномалії
# обсягів"… максимально близько до моменту для входу в угоду». Найближча до
# входу точка, яку бот уміє визначити, — ГОТОВИЙ сценарій входу ШІ (E1–E4:
# комплекс реакцій обсягу склався, останній етап ≤ 2 свічки тому, без вето).
# Той самий `entry_score`/`entry_decide`, що й 🤖 сигнал ШІ — другого правила
# немає. Ворота бота (МММ, Пружини…) тут НЕ діють: це підказка людині.
ANOM_SUB_COOLDOWN = 3600       # одна монета й бік — не частіше раз на годину
ANOM_SENT_KEEP = 48 * 3600
# ⚠ позначки сигналів «Аномалії обсягу» на графіку кабінету (вимога 24.11)
ANOM_MARK_KEEP = 7 * 86400     # скільки днів тримаємо позначки
ANOM_MARK_MAX = 80             # позначок на монету
ANOM_MARK_COINS = 400          # монет у памʼяті


def anom_pick(rl: Dict, rs: Dict, adir: Optional[Dict] = None):
    """ЧИСТА: (бік, результат entry_score) або (None, причина). Готовий
    сценарій лише в ОДИН бік (обидва → суперечливо, тиша) і ⚠ напрямок за
    аномаліями (6 год) не проти нього."""
    ready = [r for r in (rl, rs) if r and entry_decide(r)]
    if not ready:
        return None, 'сценарію входу немає'
    if len(ready) > 1:
        return None, 'сценарії в обидва боки — суперечливо'
    r = ready[0]
    side = r.get('side')
    if adir and adir.get('state') == 'dir' and adir.get('dir') and adir['dir'] != side:
        return None, f"аномалії 6 год показують рух {adir['dir']}, а сценарій {side}"
    return side, r


def anom_tg_text(symbol: str, side: str, r: Dict, price=None, adir: Optional[Dict] = None) -> str:
    """Текст Telegram: бік, монета, сценарій, етапи, ціна, напрямок аномалій."""
    sc = (r or {}).get('scenario') or {}
    emo = '🟢' if side == 'LONG' else '🔴'
    out = ['🤖 <b>VSV·AI</b>', f"⚠️ <b>Аномалії обсягу</b> · {emo} <b>{side}</b>",
           f"<b>#{symbol}</b> · {sc.get('name') or 'сценарій входу'}"]
    for st in (sc.get('stages') or [])[:5]:
        out.append(('✓ ' if st.get('done') else '· ') + str(st.get('label') or ''))
    try:
        if price:
            from detection.sbr_signals import fmt_price
            out.append(f"💵 Ціна: <code>{fmt_price(float(price))}</code>")
    except Exception:
        pass
    if adir and adir.get('state') == 'dir' and adir.get('dir'):
        out.append(f"🧭 Аномалії 6 год: {'▲' if adir['dir'] == 'LONG' else '▼'} {adir['dir']}")
    return '\n'.join(out)


def anom_tg_short(symbol: str, side: str, r: Dict, price=None) -> str:
    """⚠ КОРОТКЕ повідомлення (вимога 24.11, деф. у кабінеті): лише головне —
    бік, монета, ціна, назва сценарію. Без етапів і пояснень."""
    sc = (r or {}).get('scenario') or {}
    emo = '🟢' if side == 'LONG' else '🔴'
    px = ''
    try:
        if price:
            from detection.sbr_signals import fmt_price
            px = f" · 💵 <code>{fmt_price(float(price))}</code>"
    except Exception:
        px = ''
    out = [f"⚠️ {emo} <b>{side}</b> · <b>#{symbol}</b>{px}"]
    if sc.get('name'):
        out.append(f"{sc['name']} · 🤖 <b>VSV·AI</b>")
    else:
        out.append('🤖 <b>VSV·AI</b>')
    return '\n'.join(out)


def _alarm_engine():
    import sys
    m = sys.modules.get('detection.sbr_signals')
    try:
        return m.get_sbr_engine() if m is not None else None
    except Exception:
        return None


def anom_watch_symbols() -> List[str]:
    """⚠ Монети, по яких шукаємо сигнал «Аномалії обсягу»: особисті підписки
    кабінету + монета ⏰ будильника (джерело ⚠). Їх ведуть монітор аномалій і
    рушій ШІ навіть поза Пружинами."""
    import sys
    out: List[str] = []
    ip = sys.modules.get('web.info_portal')
    try:
        out = [str(x).upper() for x in (ip.anom_symbols() or [])] if ip is not None else []
    except Exception:
        out = []
    eng = _alarm_engine()
    try:
        a = eng.alarm_anom_symbol() if eng is not None else None
    except Exception:
        a = None
    if a and a not in out:
        out.append(a)
    # 📈 графіки кабінету з увімкненим шаром ⚠ (24.11) — рахуємо, поки дивляться
    try:
        for S in (ip.anom_chart_symbols() or []) if ip is not None else []:
            S = str(S).upper()
            if S not in out:
                out.append(S)
    except Exception:
        pass
    # 🤖 монети торгівлі з кабінету (24.11) — сигнал рахується й без ⚠ підписки
    ut = sys.modules.get('web.user_trading')
    try:
        for S in (ut.trade_symbols() or []) if ut is not None else []:
            S = str(S).upper()
            if S not in out:
                out.append(S)
    except Exception:
        pass
    return out


_anom_sub_default = anom_watch_symbols


def _anom_alarm_default(symbol, side, scenario, price, now):
    eng = _alarm_engine()
    if eng is None or not hasattr(eng, 'alarm_from_anom'):
        return False
    return bool(eng.alarm_from_anom(symbol, side, scenario, price, now))


def _anom_trade_default(symbol, side, scenario, price, now):
    """🤖 Торгівля з кабінету (24.11): ⚠ сигнал → `web.user_trading.trade_execute`."""
    import sys
    ut = sys.modules.get('web.user_trading')
    if ut is None:
        return []
    return ut.trade_execute(symbol, side, note=f'⚠ {scenario}' if scenario else '⚠ Аномалії обсягу')


def _anom_send_default(symbol, text):
    import sys
    ip = sys.modules.get('web.info_portal')
    if ip is None:
        return 0, 0
    return ip.anom_personal_send(symbol, text)


class AiEngine:
    def __init__(self, db=None, md=None, scanner=None, tm=None, clock=time.time,
                 anom_subs=None, anom_send=None, anom_alarm=None, anom_trade=None):
        self._anom_subs = anom_subs or _anom_sub_default
        self._anom_trade = anom_trade or _anom_trade_default
        self._anom_send = anom_send or _anom_send_default
        self._anom_alarm = anom_alarm or _anom_alarm_default
        self._anom_sent: Dict[str, float] = {}   # 'S|side|key|last_t' → коли надіслано
        self._anom_marks: Dict[str, List[Dict]] = {}   # ⚠ позначки на графіку (24.11)
        self._anom_last: Dict[str, float] = {}   # 'S|side' → останнє надсилання
        self.db = db
        self._md = md
        self._sc = scanner
        self._tm = tm
        self._clock = clock
        self._lock = threading.Lock()
        self._eps: Dict[str, Dict] = {}
        self._live: Dict[str, tuple] = {}     # S → (ts, rec, vw)
        self._dis_shadow: List[Dict] = []     # 📊 тіньове порівняння розряду 1H / 15m
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
        self._adir: Dict[str, Dict] = {}       # 🧭 напрямок за аномаліями (усі монети)
        self._adir_at = 0.0
        self._stat = {'last_at': 0, 'coins': 0, 'signals': 0, 'episodes_closed': 0,
                      'errors': 0, 'last_err': ''}
        self._dirty = False
        self._thread = None
        # 🤖 власний сценарій ШІ (22.11)
        self._ai_s: Optional[Dict] = None
        self._ai_s_at = 0.0
        self._etalon: List[Dict] = []
        self._anti: List[Dict] = []
        self._ai_opens: List[float] = []          # часи відкриттів угод ШІ (ліміт 30 хв)
        self._ai_last: Dict[str, list] = {}       # S → [бік, час] останнього сигналу ШІ
        self._ai_block: Dict[str, list] = {}      # 'S|side' → [причина, час] (анти-флуд логу)
        self._cleanup_at = 0.0
        self._cleanup_mark: Dict = {}
        self._hb_at = 0.0
        self._hb_key: tuple = ()
        self._cleanup_stat: Dict = {}
        self._load()
        self._load_etalon()

    # джерела
    def _scn(self):
        return self._sc or _scanner()

    def ai_settings(self) -> Dict:
        """🤖 ВЛАСНІ налаштування ШІ (`ai_settings_v1`, 5с кеш). Перший запуск —
        міграція наявних ключів сканера / TM."""
        now = time.time()
        if self._ai_s is not None and now - self._ai_s_at < 5:
            return dict(self._ai_s)
        raw = None
        try:
            raw = self.db.get_setting(AI_DB_SETTINGS, None) if self.db else None
            if isinstance(raw, str):
                raw = json.loads(raw)
        except Exception:
            raw = None
        if isinstance(raw, dict):
            st = ai_settings_norm(raw)
        else:
            sc_s, tm_s = {}, {}
            try:
                sc_s = self._scn().get_settings() or {}
            except Exception:
                pass
            try:
                tm_s = getattr(self._tm_get(), '_settings', None) or {}
            except Exception:
                pass
            st = ai_settings_migrate(sc_s, tm_s)
            try:
                if self.db:
                    self.db.set_setting(AI_DB_SETTINGS, st)
            except Exception:
                pass
        self._ai_s, self._ai_s_at = st, now
        return dict(st)

    def update_ai_settings(self, patch: Dict) -> Dict:
        cur = self.ai_settings()
        cur.update({k: v for k, v in (patch or {}).items() if k in AI_DEFAULTS})
        st = ai_settings_norm(cur)
        if self.db:
            self.db.set_setting(AI_DB_SETTINGS, st)
        self._ai_s, self._ai_s_at = st, time.time()
        return dict(st)

    def settings(self) -> Dict:
        """Сумісність: старі ключі (`ai_signal_enabled` …) — з ВЛАСНИХ налаштувань ШІ."""
        a = self.ai_settings()
        return {'ai_signal_enabled': a['enabled'], 'ai_min_charge': a['min_charge'],
                'ai_learning_enabled': a['learning_enabled'], **a}

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

    def closed_bars(self, symbol: str, tf: str = '1h', now: Optional[float] = None) -> List[Dict]:
        """ЗАКРИТІ свічки `tf` монети з кешу рушія (жива відкинута). Без мережі."""
        now = self._clock() if now is None else now
        sec = {'5m': 300, '15m': 900, '1h': 3600, '4h': 14400}.get(tf, 3600)
        with self._pro_lock:
            hit = (self._pro_bars.get(str(symbol or '').upper()) or {}).get(tf)
        rows = list(hit[1]) if hit else []
        while rows and (_bar_sec(rows[-1]) or 0) + sec > now:
            rows.pop()
        return rows

    def closed_1h(self, symbol: str, now: Optional[float] = None) -> List[Dict]:
        return self.closed_bars(symbol, '1h', now)

    def discharge_both(self, symbol: str, side: str, now: Optional[float] = None,
                       live_price=None) -> Dict:
        """💥 Стан розряду в бік `side` на ОБОХ TF підтвердження {'1h', '15m'}
        (рівень — коробка 1h). Лише кеш рушія, без мережі."""
        s = self.ai_settings()
        h = self.closed_bars(symbol, '1h', now)
        m5 = self.closed_bars(symbol, '5m', now)
        out = {'1h': discharge_state(h, side, live_price, need=int(s.get('discharge_bars') or 1)),
               '15m': discharge_state_ltf(h, self.closed_bars(symbol, '15m', now), side, live_price,
                                          need=int(s.get('discharge_bars_15m') or 2)),
               '5m': discharge_state_struct(h, m5, side, live_price,
                                            internal_size=self._struct_isize())}
        try:        # 📊 лише тінь — збій детектора VOB не чіпає решту
            vp = self._vob_params()
            obs = vob_detect(m5, vp) if len(m5) >= int(vp['swing_length']) + 5 else None
            out['vob'] = discharge_state_vob(h, m5, side, live_price, obs=obs, params=vp)
            out['vobl'] = discharge_state_vob_last(h, m5, side, live_price,
                                                   obs=self._vob_chart(symbol, vp), params=vp)
        except Exception as e:
            for k in ('vob', 'vobl'):
                out.setdefault(k, {'state': 'none', 'tf': k, 'note': f'помилка VOB: {e}'[:120]})
        try:        # 📦 друге підтвердження: самий останній VOB 1H у бік розряду
            out['vob1h'] = vob1h_check(self._vob1h_obs(symbol), side)
        except Exception as e:
            out['vob1h'] = {'ok': False, 'note': f'помилка VOB 1H: {e}'[:120]}
        return out

    def _vob1h_obs(self, symbol: str) -> Optional[Dict]:
        """VOB 1H «як на графіку»: 1h-свічки кешу рушія РАЗОМ із живою, параметри
        «📦 Volumized OB Trend» разом із Zone Count / Combine. Без мережі."""
        vp = self._vob_params()
        with self._pro_lock:
            hit = (self._pro_bars.get(str(symbol or '').upper()) or {}).get('1h')
        rows = list(hit[1]) if hit else []
        if len(rows) < int(vp['swing_length']) + 5:
            return None
        return vob_detect(rows, dict(vp, zone_count=vp.get('chart_zone_count') or 'Low',
                                     combine_obs=vp.get('chart_combine', True)))

    def _vob_chart(self, symbol: str, vp: Dict) -> Optional[Dict]:
        """«Самий останній» VOB 5m — як на графіку бота: ті самі параметри разом із
        Zone Count / Combine, свічки з ЖИВОЮ (кеш рушія, без мережі)."""
        with self._pro_lock:
            hit = (self._pro_bars.get(str(symbol or '').upper()) or {}).get('5m')
        rows = list(hit[1]) if hit else []
        if len(rows) < int(vp['swing_length']) + 5:
            return None
        return vob_detect(rows, dict(vp, zone_count=vp.get('chart_zone_count') or 'Low',
                                     combine_obs=vp.get('chart_combine', True)))

    def _vob_params(self) -> Dict:
        """Параметри VOB = блок «📦 Volumized OB Trend» сканера (одне джерело)."""
        try:
            st = self._scn().get_settings() or {}
        except Exception:
            st = {}
        p = dict(VOB_DEFAULTS)
        try:
            p['swing_length'] = max(3, min(50, int(st.get('volumized_swing_length') or p['swing_length'])))
        except (TypeError, ValueError):
            pass
        if st.get('volumized_ob_end_method') in ('Wick', 'Close'):
            p['ob_end_method'] = st['volumized_ob_end_method']
        try:
            p['max_atr_mult'] = float(st.get('volumized_max_atr_mult') or p['max_atr_mult'])
        except (TypeError, ValueError):
            pass
        if st.get('volumized_zone_count') in ('Low', 'Medium', 'High', 'One'):
            p['chart_zone_count'] = st['volumized_zone_count']
        if 'volumized_combine_obs' in st:
            p['chart_combine'] = bool(st.get('volumized_combine_obs'))
        return p

    def _struct_isize(self) -> int:
        try:
            v = int((self._scn().get_settings() or {}).get('internal_size') or STRUCT_INTERNAL_SIZE)
            return max(2, min(20, v))
        except Exception:
            return STRUCT_INTERNAL_SIZE

    def discharge_for(self, symbol: str, side: str, now: Optional[float] = None,
                      live_price=None) -> Dict:
        """💥 Стан розряду на ОБРАНОМУ TF (`discharge_tf`); поруч — `alt`
        (інший TF, для показу й тіньового порівняння)."""
        tf = self.ai_settings().get('discharge_tf') or '1h'
        both = self.discharge_both(symbol, side, now, live_price)
        return dis_pick(both, tf)

    def _dis_shadow_tick(self, S: str, side: str, both: Dict, price, now: float) -> None:
        """📊 Тіньовий журнал (23.11): точка, де розряд став «підтвердженим» на
        КОЖНОМУ TF, і що ціна робила далі. Угод не відкриває. Для 1h рахуємо, на
        скільки хвилин раніше той самий пробій підтвердив 15m (`lead_min`)."""
        try:
            px = float(price)
        except (TypeError, ValueError):
            return
        if px <= 0:
            return
        for tf in DIS_SHADOW_TFS:
            d = both.get(tf) or {}
            if d.get('state') != 'confirmed' or not d.get('t'):
                continue
            key = f"{S}|{side}|{tf}|{int(d['t'])}"
            if any(r.get('key') == key for r in self._dis_shadow):
                continue
            rec = {'key': key, 'S': S, 'side': side, 'tf': tf, 't': int(now), 'brk_t': int(d['t']),
                   'px': px, 'edge': d.get('edge'), 'last': 0.0, 'mfe': 0.0, 'mae': 0.0}
            if d.get('zone') is not None:      # 5m VOB: зона блоку = рівень стопа
                try:
                    z = float(d['zone'])
                    rec['zone'] = z
                    rec['risk'] = round(abs(px - z) / px * 100.0, 3)
                    rec['vol_pct'] = d.get('vol_pct')
                except (TypeError, ValueError):
                    pass
            if tf == '1h':
                lead = {}
                for ltf in ('15m', '5m', 'vob', 'vobl'):
                    early = [r for r in self._dis_shadow if r.get('S') == S and r.get('side') == side
                             and r.get('tf') == ltf and 0 <= now - float(r.get('t') or 0) <= 4 * 3600]
                    if early:
                        lead[ltf] = round((now - float(early[-1]['t'])) / 60.0, 1)
                if lead:
                    rec['lead'] = lead
            self._dis_shadow.append(rec)
            self._dirty = True
        if len(self._dis_shadow) > DIS_SHADOW_MAX:
            self._dis_shadow = self._dis_shadow[-DIS_SHADOW_MAX:]

    def _dis_shadow_follow(self, prices: Dict, now: float) -> None:
        """Ведемо відкриті тіньові записи ціною монети (з такту) і 1h-свічками."""
        for r in self._dis_shadow:
            if r.get('done'):
                continue
            S = r.get('S')
            px = prices.get(S)
            if px is None:
                if now - float(r.get('t') or now) >= DIS_SHADOW_HOURS * 3600:
                    r['done'] = True
                continue
            dis_shadow_update(r, px, self.closed_1h(S, now), now)
            self._dirty = True

    def dis_shadow_state(self) -> Dict:
        recs = list(self._dis_shadow)
        return {'stats': dis_shadow_stats(recs), 'n': len(recs),
                'last': [{k: r.get(k) for k in ('S', 'side', 'tf', 't', 'px', 'last', 'mfe', 'mae',
                                                 'false_t', 'lead', 'done', 'zone', 'risk', 'zone_t')}
                         for r in recs[-12:]]}

    def false_start_check(self, symbol: str, side: str, edge, opened_at) -> Optional[Dict]:
        """🪤 Для виходу з угоди ШІ (TM): закрита після входу свічка повернулась
        у коробку за межу пробою. Лише кеш (монітор у мережу не ходить)."""
        return false_start_hit(side, edge, self.closed_1h(symbol), opened_at)

    def rdy_view(self, symbol: str) -> Optional[Dict]:
        """🧩 Що каже блок «МММ-Готовність» про монету (банер + вкладка)."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
            return ff.mm_gate_view(symbol, book='rdy') if ff else None
        except Exception:
            return None

    def _pro_exit_on(self) -> bool:
        """Чи увімкнено «🤖 Вихід з угоди ШІ» у Trade Manager — супровід угоди
        за планом мозку графіка тепер його частина (бари треба готувати)."""
        return bool(self.ai_settings().get('exit_enabled'))

    def _pro_prefetch_open(self, now: float) -> int:
        """Бари 15m/1h/4h для монет ВІДКРИТИХ угод (real + paper) — паралельно,
        у такті рушія. Так монітор TM читає готовий кеш і не чекає біржу."""
        # Лише ВЛАСНІ угоди ШІ: він супроводжує тільки їх (22.11).
        try:
            tm = self._tm_get()
            syms = [S for S, _b in (tm.ai_open_positions() if tm and hasattr(tm, 'ai_open_positions') else [])]
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
            with self._pro_lock:
                result = self._pro_get().analyze(market, now, require_model=False)
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
        Лише коли 🤖 справді задіяний (сигнали або вихід ШІ): без нього в
        архівах немає нових ознак, а читання 2×5000 рядків БД раз на 15 хв
        було б марною роботою."""
        s = self.settings()
        if not s.get('ai_learning_enabled') or not self.db:
            return None
        if not (s.get('ai_signal_enabled') or self._pro_exit_on()):
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
            with self._pro_lock:
                decision = self._pro_get().manage_position(
                    market, side, entry_price, opened_at, now,
                    current_price=current_price, peak_price=peak_price,
                    require_model=False)
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

    def _refresh_adir(self, anoms: Dict[str, List[Dict]], now: float) -> None:
        """🧭 Напрямок за аномаліями для КОЖНОЇ монети з рядками аномалій (ті самі
        рядки, що читає такт; лише памʼять сканера для рівнів / CHoCH)."""
        out = {}
        for S, al in (anoms or {}).items():
            if not S:
                continue
            try:
                lv, ch = scanner_context(self._scn(), S)
                out[str(S).upper()] = anomaly_direction(al, now, lv, ch, self._cal)
            except Exception as e:
                self._stat['last_err'] = f'напрямок аномалій {S}: {e}'[:200]
        with self._lock:
            self._adir, self._adir_at = out, now

    def anomaly_dir(self, symbol: str) -> Optional[Dict]:
        """🧭 Напрямок за аномаліями (кеш такту). Кеш застарів (рушій стоїть) →
        None: ворота тоді йдуть за вкладкою Пружини, а не за старими рухами.
        Свіжий кеш без рядків монети → `state: none` («аномалій немає»)."""
        now = self._clock()
        with self._lock:
            if not self._adir_at or now - self._adir_at > DIR_TTL:
                return None
            r = self._adir.get(str(symbol or '').upper())
        return dict(r) if r else {'dir': None, 'state': 'none', 'up': 0.0, 'dn': 0.0,
                                  'margin': 0.0, 'n': 0, 'notes': [], 'last_t': None}

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
        # ті самі 300 1h-барів — у кеш мозку графіка (другий запит не потрібен)
        with self._pro_lock:
            self._pro_bars.setdefault(S, {})['1h'] = (now, kl)
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
        spring_on = sc is not None and bool((sc.get_settings() or {}).get('spring_enabled', True))
        try:
            subs = [str(x).upper() for x in (self._anom_subs() or [])]
        except Exception:
            subs = []
        if not spring_on and not subs:
            with self._lock:
                self._view = {}
            self._stat.update({'last_at': now, 'coins': 0, 'cand': 0})
            if s.get('enabled') and now - self._hb_at >= HEARTBEAT_SEC:
                self._hb_at = now
                self._ai_log('', 'ai_scan', '🤖 ШІ · огляд: 🌀 Пружина вимкнена — '
                             'немає заряджених монет для перебору, угод ШІ не буде')
            return 0
        self._refresh_cal(now)
        min_charge = float(s.get('min_charge') or 65)
        allowed = s.get('scenarios') or ['E2']
        cand = self.candidates(min_charge) if spring_on else {}
        self._stat.update({'tick_at': now, 'cand': len(cand)})
        anoms: Dict[str, List[Dict]] = {}
        try:
            look = max(ANOM_LOOK_SEC, DIR_LOOK_SEC,
                       getattr(_scen(), 'SCEN_LOOK_SEC', ANOM_LOOK_SEC))
            for a in (self.db.get_vol_anomalies(since=now - look, limit=20000)
                      if self.db else []) or []:
                anoms.setdefault(a.get('symbol'), []).append(a)
        except Exception:
            pass
        self._refresh_adir(anoms, now)

        brain = _sib('ai_brain')

        def load(S):
            try:
                live = self._fetch_live(S, now)
                if S in cand:                 # графік 15m/1h/4h — лише кандидатам мозку
                    self._pro_market(S, now)
                    if s.get('discharge_confirm', True):
                        # 💥 5m-свічки для структурного підтвердження розряду (кеш 150с)
                        self._pro_fetch_bars(S, '5m', now)
                return S, live
            except Exception as e:
                self._stat['errors'] += 1
                self._stat['last_err'] = f'{S}: {e}'[:200]
                return S, (None, None)

        pool = list(cand.keys()) + [S for S in subs if S not in cand]
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            lives = dict(ex.map(load, pool))
        view = {}
        scored = {}
        try:
            _tm = self._tm_get()
            _open_n = len(_tm.ai_open_positions()) if _tm and hasattr(_tm, 'ai_open_positions') else 0
        except Exception:
            _open_n = 0
        for S, (rec0, src) in cand.items():
            try:
                rec, vw = lives.get(S) or (None, None)
                rec = rec or rec0 or {}
                if not rec:
                    continue
                charged = rec.get('state') == 'charged' or float(rec.get('charge') or 0) >= min_charge
                al = anoms.get(S) or []
                lv, ch = scanner_context(self._scn(), S)
                el = entry_score('LONG', rec, vw, al, now, self._cal, lv, ch, charged)
                es = entry_score('SHORT', rec, vw, al, now, self._cal, lv, ch, charged)
                scored[S] = (el, es, rec, vw)
                ep = self._eps.get(S)
                if ep is None and charged:
                    ep = self._eps[S] = episode_new(S, now, rec, vw, src)
                    _ev(ep, now, 'charged', charge=rec.get('charge'), lean=rec.get('lean'))
                if ep is not None:
                    episode_step(ep, now, rec, vw, el, es, charged)
                    self._dirty = True
                # головний бік: готовий ДОЗВОЛЕНИЙ сценарій (налаштування ШІ, деф.
                # лише E2 «💥 Старт розряду») → прогрес сценарію → бал
                def _rk(x):
                    sc = x.get('scenario') or {}
                    return (1 if (entry_decide(x) and scen_allowed(x, allowed)) else 0,
                            (sc.get('done') or 0) / max(1, sc.get('total') or 1), x['score'])
                best = el if _rk(el) >= _rk(es) else es
                ready = entry_decide(best) and scen_allowed(best, allowed)
                # 🤖🧠 ЄДИНИЙ МОЗОК: напрямок → старт → санітарні перевірки графіка → план
                pro = self._professional_snapshot(S, now)
                bc = brain.entry_check(best['side'], best, pro, rec)
                feat = entry_features(best['side'], best, rec, bc)
                sim = etalon_similarity(feat, self._etalon, best['side']) if self._etalon else None
                anti_sim = etalon_similarity(feat, self._anti, best['side']) if self._anti else None
                view[S] = {'symbol': S, 'src': src, 'charge': rec.get('charge'),
                           'state': rec.get('state'), 'lean': rec.get('lean'),
                           'prob_long': rec.get('prob_long'), 'long': el, 'short': es,
                           'best': best['side'], 'episode_age': (int(now - ep['started_at'])
                                                                 if ep else None),
                           'signal': (ep or {}).get('signal'),
                           'brain': bc, 'pick': best, 'ready': bool(ready and bc.get('ok')),
                           'feat': feat, 'sim': sim, 'anti_sim': anti_sim,
                           'etalon': etalon_brain(feat, self._etalon, self._anti, best['side'], s)}
                view[S]['rank'] = ai_rank(view[S], sim)
                view[S]['confidence'] = bc.get('confidence')
                if view[S]['etalon'].get('mature') and view[S]['etalon'].get('verdict') != 'take':
                    view[S]['ready'] = False          # 🏆🧠 головний мозок сказав «ні»
                _skey = (best.get('scenario') or {}).get('key') or ''
                ds = None
                if _skey == 'E2':
                    _both = self.discharge_both(S, best['side'], now, _px(vw, rec))
                    ds = dis_pick(_both, s.get('discharge_tf') or '1h')
                    self._dis_shadow_tick(S, best['side'], _both, _px(vw, rec), now)
                rv = self.rdy_view(S) if (s.get('rdy_filter') or 'off') != 'off' else None
                view[S]['discharge'] = ds
                if ready and bc.get('ok'):
                    # ЧОМУ готова точка не стане угодою (показ: таблиця · огляд · тумблер)
                    view[S]['block'] = ('ШІ вимкнено головним тумблером' if not s.get('enabled')
                                        else (f"заряд {rec.get('charge')} < {min_charge:g} — монета не в роботі ШІ"
                                              if ep is None else None)
                                        or ai_entry_block(ep, best['side'], s, bc, view[S]['etalon'],
                                                            _open_n, self._ai_opens,
                                                            self._ai_last.get(S), now, rdy=rv,
                                                            discharge=ds, scen_key=_skey))
                if ep is not None:
                    for x in (el, es):
                        sc = x.get('scenario') or {}
                        if entry_decide(x):
                            k = f"scen_{x['side']}_{sc.get('key')}"
                            if k not in ep:
                                ep[k] = int(now)
                                _ev(ep, now, 'scenario', side=x['side'], key=sc.get('key'),
                                    name=sc.get('name'), score=x.get('score'))
                if s.get('enabled') and ep is not None and ready:
                    self._try_signal(S, best, rec, vw, ep, now, professional=pro, brain=bc,
                                     feat=feat, sim=sim, anti_sim=anti_sim, eb=view[S]['etalon'],
                                     rdy=rv, discharge=ds)
                if ep is not None:
                    why = episode_done(ep, now, charged)
                    if why:
                        self._close_episode(S, ep, now, why)
            except Exception as e:      # одна монета не обнуляє весь перебір
                self._stat['errors'] += 1
                self._stat['err_at'] = now
                self._stat['last_err'] = f'{S}: {type(e).__name__}: {e}'[:200]
                print(f'[AI] {S}: {e}')

        try:
            self._dis_shadow_follow({S: _px(lv[1], lv[0]) for S, lv in lives.items()
                                     if lv and (lv[0] or lv[1])}, now)
        except Exception as e:
            self._stat['last_err'] = f'тінь розряду: {e}'[:200]
        try:
            self._anom_personal(subs, scored, lives, anoms, now)
        except Exception as e:
            self._stat['last_err'] = f'⚠ особисті: {e}'[:200]
        with self._lock:
            self._view = view
        try:
            self._heartbeat(view, s, now)
        except Exception as e:
            self._stat['last_err'] = f'огляд: {e}'[:200]
        try:
            self._cleanup(now)
        except Exception as e:
            self._stat['last_err'] = f'🧹 очистка: {e}'[:200]
        self._pro_prune(view.keys())
        self._stat.update({'last_at': now, 'coins': len(view)})
        if self._dirty:
            self._persist()
        return len(view)

    def _heartbeat(self, view: Dict, s: Dict, now: float) -> None:
        """🤖 Огляд перебору в 🧾 Лог: раз на HEARTBEAT_SEC або щойно змінився
        набір точок входу (щоб лог показував, що ШІ працює, але не флудив)."""
        tm = self._tm_get()
        try:
            open_n = len(tm.ai_open_positions()) if tm and hasattr(tm, 'ai_open_positions') else 0
        except Exception:
            open_n = 0
        opens = len([t for t in self._ai_opens if now - t < OPEN_WINDOW_SEC])
        text, key = ai_heartbeat_text(view, s, self.etalon_state(), open_n, opens)
        if key == self._hb_key and now - self._hb_at < HEARTBEAT_SEC:
            return
        self._hb_key, self._hb_at = key, now
        self._ai_log('', 'ai_scan', text, coins=len(view), ready=list(key))

    def _ai_log(self, S, event, text, side=None, **extra):
        try:
            from detection.activity_log import log_activity
            log_activity(S, event, text, side=side, source='AI', extra=extra or None)
        except Exception:
            pass

    @staticmethod
    def _ai_block_ev(ep: Dict, side: str, why: str, now: float) -> None:
        """Подія епізоду «ШІ не відкрив» — лише коли причина ЗМІНИЛАСЬ (такт 30с)."""
        last = ep.setdefault('block_last', {})
        if last.get(side) != why:
            last[side] = why
            _ev(ep, now, 'ai_block', side=side, reason=why)

    def _ai_block_note(self, S, side, why, now) -> None:
        """Власна відмова ШІ → 🧾 Лог ОДИН раз (та сама причина — раз на 15 хв)."""
        k = f'{S}|{side}'
        last = self._ai_block.get(k)
        if last and last[0] == why and now - float(last[1] or 0) < BLOCK_LOG_GAP:
            return
        self._ai_block[k] = [why, now]
        self._ai_log(S, 'rejected', f'🤖 ШІ {side}: {why}', side=side)

    def _ai_ob_stop(self, S: str, side: str, price) -> tuple:
        """🛑 «SL з: OB» — ТОЙ САМИЙ ланцюг, що глобальне «🛑 SL з» бота (1H OB →
        15m → 5m, буфер `queue3_vob_sl_buffer_pct`, рівень лише з правильного боку
        ціни). Кличеться ЛИШЕ в момент сигналу (Volumized може тягнути свічки).
        → (рівень | None, підпис)."""
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
        except Exception:
            ff = None
        if ff is None or not hasattr(ff, '_q4_pick_sl'):
            return None, ''
        try:
            fs = ff.get_settings() or {}
            buf = max(0.0, float(fs.get('queue3_vob_sl_buffer_pct', 0.10) or 0)) / 100.0
            chosen, _sk = ff._q4_pick_sl(S, str(side).upper(), '1h', buf, price)
        except Exception as e:
            self._stat['last_err'] = f'SL з OB {S}: {e}'[:200]
            return None, ''
        if not chosen:
            return None, ''
        return chosen[0], str(chosen[1] or 'OB')

    def _try_signal(self, S, sc_res, rec, vw, ep, now, professional=None, brain=None,
                    feat=None, sim=None, anti_sim=None, eb=None, rdy=None, discharge=None):
        """🤖 ВЛАСНИЙ СЦЕНАРІЙ ШІ (22.11): ворота бота (кнопки напрямку, МММ,
        Пружини, OB/PD/…, черги) НЕ діють — ШІ відкриває СВОЮ угоду за своїми
        правилами (налаштування ШІ) і далі супроводжує лише її."""
        side = sc_res['side']
        s = self.ai_settings()
        pro_side = ((professional or {}).get('sides') or {}).get(side) or {}
        if brain is None:
            brain = _sib('ai_brain').entry_check(side, sc_res, professional, rec)
        if not brain.get('ok'):
            # 🧠 мозок не дав входу: подія в епізоді — ЛИШЕ коли причина змінилась
            last = ep.setdefault('brain_veto', {})
            if last.get(side) != brain.get('reason'):
                last[side] = brain.get('reason')
                _ev(ep, now, 'brain_veto', side=side, reason=brain.get('reason'),
                    strict=(brain.get('strict') or {}).get('ready'))
            return
        tm = self._tm_get()
        if tm is None or not hasattr(tm, 'ai_open'):
            self._stat['last_err'] = 'Trade Manager без ai_open'
            return
        try:
            open_n = len(tm.ai_open_positions()) if hasattr(tm, 'ai_open_positions') else 0
        except Exception:
            open_n = 0
        # 🏆🧠 ГОЛОВНИЙ МОЗОК — база еталонів: коли досвід зібрано, саме вона
        # ухвалює остаточне рішення на вхід (k-NN за відмінними й невдалими угодами).
        if eb is None and feat is not None:
            eb = etalon_brain(feat, self._etalon, self._anti, side, s)
        _skey = (sc_res.get('scenario') or {}).get('key') or ''
        if discharge is None and _skey == 'E2':
            discharge = self.discharge_for(S, side, now, _px(vw, rec))
        if rdy is None:
            rdy = self.rdy_view(S)
        why = ai_entry_block(ep, side, s, brain, eb, open_n, self._ai_opens,
                             self._ai_last.get(S), now, rdy=rdy, discharge=discharge,
                             scen_key=_skey)
        if why:
            if not (ep.get('signal') or {}).get('own'):   # уже відкрита угода — не шумимо
                self._ai_block_note(S, side, why, now)
                self._ai_block_ev(ep, side, why, now)
            return
        # 🛑 «SL з» ШІ: стоп за обраним джерелом + межі мін / макс → перерахунок R.
        # Вимкнено (деф.) → стоп плану (структура / ATR 15m у межах мін / макс) —
        # ЛИШЕ орієнтир ризику: на угоду не ставиться і її не закриває.
        px0 = _px(vw, rec)
        sl_on = bool(s.get('sl_enabled'))
        ob_stop, ob_label = (self._ai_ob_stop(S, side, px0)
                             if sl_on and s.get('sl_source') == 'ob' else (None, ''))
        plan0 = ai_stop_for(side, px0, brain.get('plan') or {}, ob_stop, ob_label,
                            s if sl_on else dict(s, sl_source='plan'))
        if not sl_on:
            plan0['stop_off'] = True
            plan0['stop_label'] = 'SL вимкнено · орієнтир ризику ' + (plan0.get('stop_label') or '')
        brain = dict(brain, plan=plan0)
        mrr = getattr(_sib('ai_brain'), 'MIN_NET_RR', 1.3)
        if plan0.get('net_rr') is not None and plan0['net_rr'] < mrr:
            why = (f"🛑 стоп з «{plan0.get('stop_label')}» −{plan0.get('risk_pct')}% → "
                   f"R {plan0.get('net_rr')} < {mrr} до цілі")
            self._ai_block_note(S, side, why, now)
            self._ai_block_ev(ep, side, why, now)
            return
        tr = ep.setdefault('ai_tries', {}).get(side) or [0, 0]
        ep['ai_tries'][side] = [tr[0] + 1, int(now)]
        px = _px(vw, rec)
        pro_compact = self._pro_compact(professional)
        bm = _sib('ai_brain')
        plan = brain.get('plan') or {}
        if s.get('etalon_enabled'):
            plan = etalon_plan(plan, side, eb) or plan      # 🎯 ціль — з досвіду
        scen = sc_res.get('scenario') or {}
        head = (f"Сигнал ШІ {side} · 🧩 {sc_res.get('scenario_text') or ''} · "
                f"заряд Пружини {rec.get('charge')} · 🧠 {bm.checks_text(brain)}")
        if plan:
            head += (f" · план: стоп {plan.get('stop')} (−{plan.get('risk_pct')}%, {plan.get('stop_label') or 'план'}) · ціль "
                     f"{plan.get('target')} (+{plan.get('reward_pct')}%) · R {plan.get('net_rr')}")
        head += f" · впевненість {brain.get('confidence')}"
        if eb:
            head += (f" · 🏆🧠 мозок еталонів: {eb.get('reason')}"
                     + (f" · ціль з досвіду {eb.get('tp_r')}R" if plan.get('target_src') == 'etalon' else ''))
        elif sim is not None:
            head += f" · 🏆 схожість на еталони {sim:.2f} (база {len(self._etalon)})"
        self._ai_log(S, 'signal', head, side=side, ai_score=sc_res['score'],
                     ai_parts=sc_res['parts'], ai_scenario=scen.get('key'),
                     ai_plan=plan or None, ai_confidence=brain.get('confidence'),
                     ai_sim=sim, professional_quality=pro_side.get('quality'))
        res = None
        try:
            res = tm.ai_open(S, side, plan=plan or None, book=s.get('book') or 'auto',
                             sl_fallback_pct=s.get('sl_fallback_pct'), place_sl=sl_on,
                             feat=feat, scenario=scen.get('key'),
                             scenario_name=scen.get('name'), context=pro_compact or {},
                             entry={**ai_entry_snapshot(side, sc_res, rec, brain, sim, px, now,
                                                        self.anomaly_dir(S), anti_sim),
                                    'etalon': {k: (eb or {}).get(k) for k in
                                               ('mature', 'verdict', 'p_good', 'exp_r', 'tp_r', 'reason')},
                                    'plan_final': plan or None,
                                    'discharge': discharge, 'rdy': rdy})
        except Exception as e:
            res = {'ok': False, 'reason': f'помилка відкриття: {e}'}
            self._stat['last_err'] = f'ai_open {S}: {e}'[:200]
        ok = bool((res or {}).get('ok'))
        if not ok:
            self._ai_log(S, 'rejected', f"🤖 ШІ {side}: угоду НЕ відкрито — {(res or {}).get('reason')}",
                         side=side)
            _ev(ep, now, 'signal_rejected', side=side, score=sc_res['score'],
                reason=(res or {}).get('reason'))
            return
        self._ai_opens = [t for t in self._ai_opens if now - t < OPEN_WINDOW_SEC] + [now]
        self._ai_last[S] = [side, now]
        ep['signal'] = {'t': int(now), 'side': side, 'px': px, 'score': sc_res['score'],
                        'parts': sc_res['parts'], 'result': (res or {}).get('mode'), 'own': True,
                        'scenario': scen.get('key'), 'scenario_name': scen.get('name'),
                        'professional': pro_compact, 'feat': feat, 'sim': sim, 'anti_sim': anti_sim,
                        'etalon': {k: (eb or {}).get(k) for k in ('mature', 'verdict', 'p_good', 'exp_r', 'tp_r')},
                        'brain': {'checks': brain.get('checks'), 'plan': plan or None,
                                  'confidence': brain.get('confidence'),
                                  'strict': brain.get('strict'), 'context': brain.get('context')}}
        _ev(ep, now, 'signal', side=side, px=px, score=sc_res['score'])
        self._stat['signals'] += 1
        self._dirty = True

    # ── 🏆 ЕТАЛОН ──────────────────────────────────────────────────────────
    def _load_etalon(self):
        for key, attr in ((ETALON_DB, '_etalon'), (ETALON_ANTI_DB, '_anti')):
            try:
                raw = self.db.get_setting(key, None) if self.db else None
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, list):
                    setattr(self, attr, [e for e in raw if isinstance(e, dict) and e.get('feat')][-ETALON_MAX:])
            except Exception:
                setattr(self, attr, [])

    def etalon_consider(self, closed: Dict) -> bool:
        """Закрита угода ШІ: відмінна → у базу еталонів, невдала → в антибазу
        (контраст для навчання). У запис іде все корисне: вхід, план, шлях,
        вихід, події, уроки."""
        s = self.ai_settings()
        if not s.get('etalon_enabled'):
            return False
        if etalon_excellent(closed, s):
            attr, key, kind = '_etalon', ETALON_DB, 'good'
        elif etalon_failed(closed):
            attr, key, kind = '_anti', ETALON_ANTI_DB, 'bad'
        else:
            return False
        rec = etalon_record(closed)
        with self._lock:
            cur = getattr(self, attr)
            k2 = (rec['sym'], rec['t'])
            if any((e.get('sym'), e.get('t')) == k2 for e in cur):
                return False
            cur = cur + [rec]
            if len(cur) > ETALON_MAX:           # переповнено → геть найслабший, а не найстаріший
                qk = 'good' if kind == 'good' else 'bad'
                worst = min(range(len(cur)), key=lambda i: (etalon_quality(cur[i], qk),
                                                            float(cur[i].get('t') or 0)))
                cur.pop(worst)
            setattr(self, attr, cur)
            base = list(cur)
            added = any(e is rec for e in cur)
        try:
            if self.db:
                self.db.set_setting(key, base)
        except Exception:
            pass
        if not added:                           # новий запис слабший за всю повну базу
            return False
        res = f"результат {rec['pnl']}%" + (f" · {rec['r']}R" if rec['r'] is not None else '')
        if kind == 'good':
            msg = f"🏆 ШІ: угоду {rec['side']} додано в еталони ({res}) — база {len(base)}"
        else:
            msg = f"❌ ШІ: невдалу угоду {rec['side']} додано в антибазу ({res}) — база {len(base)}"
        if rec.get('lessons'):
            msg += ' · уроки: ' + '; '.join(rec['lessons'][:3])
        self._ai_log(rec['sym'], 'signal', msg, side=rec['side'])
        return True

    def etalon_state(self) -> Dict:
        base = list(self._etalon)
        rs = [e.get('r') for e in base if isinstance(e.get('r'), (int, float))]
        by_sc: Dict[str, int] = {}
        for e in base:
            by_sc[e.get('scen') or '?'] = by_sc.get(e.get('scen') or '?', 0) + 1
        anti = list(self._anti)
        caps = [e.get('capture') for e in base if isinstance(e.get('capture'), (int, float))]
        return {'n': len(base), 'min_n': ETALON_MIN_N, 'active': len(base) >= ETALON_MIN_N,
                'anti_n': len(anti), 'contrast': len(base) >= ETALON_MIN_N and len(anti) >= ETALON_MIN_N,
                'mature': len(base) >= ETALON_MATURE_GOOD and len(anti) >= ETALON_MATURE_ANTI,
                'need_good': ETALON_MATURE_GOOD, 'need_anti': ETALON_MATURE_ANTI,
                'avg_capture': round(sum(caps) / len(caps), 2) if caps else None,
                'anti_last': [{k: e.get(k) for k in ('sym', 'side', 'r', 'pnl', 'scen', 't')}
                              for e in anti[-5:][::-1]],
                'avg_r': round(sum(rs) / len(rs), 2) if rs else None,
                'long': sum(1 for e in base if e.get('side') == 'LONG'),
                'short': sum(1 for e in base if e.get('side') == 'SHORT'),
                'by_scenario': by_sc,
                'last': [{k: e.get(k) for k in ('sym', 'side', 'r', 'pnl', 'scen', 't', 'capture', 'lessons')}
                         for e in base[-8:][::-1]]}

    def etalon_clear(self) -> int:
        n = len(self._etalon) + len(self._anti)
        self._etalon, self._anti = [], []
        try:
            if self.db:
                self.db.set_setting(ETALON_DB, [])
                self.db.set_setting(ETALON_ANTI_DB, [])
        except Exception:
            pass
        return n

    # ── 🎯 ВИБІР ШІ ДЛЯ «Рекомендовані ботом» ──────────────────────────────
    def recommend(self) -> Dict[str, Dict]:
        """{S: {side, ready, rank, scenario, sim, confidence, plan}} — ТІ САМІ
        рішення, за якими ШІ відкриває свої угоди (лише памʼять)."""
        with self._lock:
            view = dict(self._view)
        out = {}
        for S, v in view.items():
            p = v.get('pick') or {}
            sc = p.get('scenario') or {}
            b = v.get('brain') or {}
            out[S] = {'side': p.get('side'), 'ready': bool(v.get('ready')), 'src': v.get('src'),
                      'rank': v.get('rank'), 'sim': v.get('sim'),
                      'etalon': {k: (v.get('etalon') or {}).get(k) for k in
                                 ('mature', 'verdict', 'p_good', 'exp_r', 'tp_r', 'reason')},
                      'scenario': sc.get('key'), 'scenario_name': sc.get('name'),
                      'stage': f"{sc.get('done') or 0}/{sc.get('total') or 0}" if sc else None,
                      'confidence': b.get('confidence'),
                      'plan': etalon_plan(b.get('plan'), p.get('side') or '', v.get('etalon')) or b.get('plan')}
        return out

    # ── 🧹 АВТООЧИСТКА ТАБЛИЦЬ НАВЧАННЯ ────────────────────────────────────
    def _quality_plan(self) -> Dict:
        """🏷 Позначення на видалення ЗА ЯКІСТЮ (а не за датою): еталони,
        антибаза, дослідження угод ШІ, епізоди із сигналом."""
        plan = {'etalon': [], 'anti': [], 'research': [], 'episodes': [], 'list': []}
        for attr, kind, key in (('_etalon', 'good', 'etalon'), ('_anti', 'bad', 'anti')):
            base = list(getattr(self, attr))
            for i, why in etalon_prune_plan(base, kind):
                e = base[i]
                plan[key].append((e.get('sym'), e.get('t')))
                plan['list'].append({'kind': key, 'sym': e.get('sym'), 'side': e.get('side'),
                                     'r': e.get('r'), 'pnl': e.get('pnl'), 'why': why})
        if self.db and hasattr(self.db, 'ai_learning_brief'):
            try:
                brief = self.db.ai_learning_brief() or {}
            except Exception:
                brief = {}
            prot = {(e.get('sym'), int(float(e.get('t') or 0)))
                    for e in list(self._etalon) + list(self._anti)}
            plan['research'] = learning_prune_plan(brief.get('research') or [], 'research',
                                                   RESEARCH_AI_KEEP, prot)
            plan['episodes'] = learning_prune_plan(brief.get('episodes') or [], 'episodes',
                                                   EPISODE_SIG_KEEP)
        return plan

    def _apply_quality_plan(self, plan: Dict) -> Dict:
        out = {'etalon': 0, 'research': 0, 'episodes': 0}
        for attr, key, dbk in (('_etalon', 'etalon', ETALON_DB), ('_anti', 'anti', ETALON_ANTI_DB)):
            drop = set(plan.get(key) or ())
            if not drop:
                continue
            with self._lock:
                cur = list(getattr(self, attr))
                keep = [e for e in cur if (e.get('sym'), e.get('t')) not in drop]
                setattr(self, attr, keep)
            out['etalon'] += len(cur) - len(keep)
            try:
                if self.db:
                    self.db.set_setting(dbk, keep)
            except Exception:
                pass
        if (plan.get('research') or plan.get('episodes')) and hasattr(self.db, 'delete_learning_ids'):
            try:
                r = self.db.delete_learning_ids(plan.get('research') or (), plan.get('episodes') or ()) or {}
                out['research'] += int(r.get('research') or 0)
                out['episodes'] += int(r.get('episodes') or 0)
            except Exception as e:
                print(f'[AI] 🧹 delete_learning_ids: {e}')
        return out

    @staticmethod
    def _plan_counts(plan: Dict) -> Dict:
        return {'etalon': len(plan.get('etalon') or ()) + len(plan.get('anti') or ()),
                'research_q': len(plan.get('research') or ()),
                'episodes_q': len(plan.get('episodes') or ())}

    def _cleanup(self, now: float, force: bool = False) -> Optional[Dict]:
        """🧹 Прибрати з таблиць навчання те, що лише заважає (вимоги 22.11).
        Раз на годину — ПОЗНАЧЕННЯ: мозок оцінює ВСІ результати й позначає
        найгірші та дублі (видно в панелі ШІ); раз на 6 год — видалення
        позначеного. Результати не видаляються за датою — лише за якістю."""
        if not force and not self.ai_settings().get('cleanup_enabled'):
            return None
        if not self.db or not hasattr(self.db, 'cleanup_ai_learning'):
            return None
        if not force and now - self._cleanup_at < CLEANUP_EVERY_SEC:
            if now - float((self._cleanup_mark or {}).get('at') or 0) >= CLEANUP_MARK_SEC:
                try:
                    m = self.db.cleanup_ai_learning(now + (CLEANUP_EVERY_SEC - (now - self._cleanup_at)),
                                                    dry=True, **CLEANUP_RULES) or {}
                except TypeError:
                    m = {}
                plan = self._quality_plan()
                pc = self._plan_counts(plan)
                m['etalon'] = pc['etalon']
                m['research'] = int(m.get('research') or 0) + pc['research_q']
                m['episodes'] = int(m.get('episodes') or 0) + pc['episodes_q']
                self._cleanup_mark = {'at': int(now), 'next_at': int(self._cleanup_at + CLEANUP_EVERY_SEC),
                                      'list': plan['list'][:12], **m}
            return None
        self._cleanup_at = now
        try:
            res = self.db.cleanup_ai_learning(now, **CLEANUP_RULES) or {}
        except TypeError:                       # старіший шар БД без нових правил
            res = self.db.cleanup_ai_learning(now) or {}
        q = self._apply_quality_plan(self._quality_plan())
        res['etalon'] = q['etalon']
        res['research'] = int(res.get('research') or 0) + q['research']
        res['episodes'] = int(res.get('episodes') or 0) + q['episodes']
        self._cleanup_stat = {'at': int(now), **res}
        self._cleanup_mark = {'at': int(now), 'next_at': int(now + CLEANUP_EVERY_SEC),
                              'episodes': 0, 'anomalies': 0, 'research': 0, 'etalon': 0, 'list': []}
        if any(v for k, v in res.items() if k != 'list'):
            print(f'[AI] 🧹 очистка таблиць навчання: {res}')
            self._ai_log('', 'signal', '🧹 ШІ: очистка навчання — видалено найгірше й службове: '
                         + ', '.join(f'{k} {v}' for k, v in res.items() if v))
        return res

    def _anom_personal(self, subs, scored, lives, anoms, now) -> int:
        """⚠ Особисті сповіщення: монета з підписки, сценарій входу ГОТОВИЙ →
        одне повідомлення підписникам (той самий сценарій не повторюється,
        монета+бік — не частіше раз на `ANOM_SUB_COOLDOWN`)."""
        sent = 0
        for k in [k for k, t in self._anom_sent.items() if now - t > ANOM_SENT_KEEP]:
            self._anom_sent.pop(k, None)
        subset = set(subs)
        # ⚠ позначки на графіку (24.11): той самий сигнал фіксуємо для ВСІХ
        # монет, які веде рушій (Пружини + підписки + графіки кабінету),
        # а надсилаємо / будимо / торгуємо — лише по монетах підписок.
        for S in list(dict.fromkeys(list(subs) + list(scored))):
            if S in scored:
                el, es, rec, vw = scored[S]
            else:
                rec, vw = lives.get(S) or (None, None)
                rec = rec or {}
                al = anoms.get(S) or []
                if not al:
                    continue
                lv, ch = scanner_context(self._scn(), S)
                charged = rec.get('state') == 'charged'
                el = entry_score('LONG', rec, vw, al, now, self._cal, lv, ch, charged)
                es = entry_score('SHORT', rec, vw, al, now, self._cal, lv, ch, charged)
            adir = self.anomaly_dir(S)
            side, r = anom_pick(el, es, adir)
            if not side:
                continue
            sc = r.get('scenario') or {}
            px = _px(vw, rec)
            try:
                self._anom_mark(S, side, r, px, adir, now)
            except Exception as e:
                self._stat['last_err'] = f'⚠ позначка {S}: {e}'[:200]
            if S not in subset:
                continue
            key = f"{S}|{side}|{sc.get('key')}|{int(float(sc.get('last_t') or 0))}"
            if key in self._anom_sent:
                continue
            self._anom_sent[key] = now
            self._dirty = True
            # ⏰ будильник (вимога 18.11): той самий сигнал — на КОЖЕН новий
            # сценарій (дія ідемпотентна: усе вже в потрібному стані → тиша).
            try:
                self._anom_alarm(S, side, sc.get('name') or '', px, now)
            except Exception as e:
                self._stat['last_err'] = f'⚠ будильник {S}: {e}'[:200]
            # 🤖 торгівля з кабінету (24.11): той самий НОВИЙ сценарій → одна дія
            # (немає угоди — відкрити · той самий бік — тримати · протилежний —
            # закрити й відкрити). Кулдаун Telegram на торгівлю не діє.
            try:
                self._anom_trade(S, side, sc.get('name') or '', px, now)
            except Exception as e:
                self._stat['last_err'] = f'🤖 торгівля {S}: {e}'[:200]
            if now - self._anom_last.get(f'{S}|{side}', 0) < ANOM_SUB_COOLDOWN:
                continue
            # два формати (24.11): кожен підписник отримує обраний у кабінеті
            text = {'short': anom_tg_short(S, side, r, px),
                    'full': anom_tg_text(S, side, r, px, adir)}
            try:
                ok, bad = self._anom_send(S, text)
            except Exception as e:
                ok, bad = 0, 1
                self._stat['last_err'] = f'⚠ TG {S}: {e}'[:200]
            if not ok and not bad:
                continue                  # підписників немає (монета лише будильника)
            self._anom_last[f'{S}|{side}'] = now
            if ok:
                sent += 1
                self._stat['anom_sent'] = self._stat.get('anom_sent', 0) + ok
            try:
                from detection.activity_log import log_activity
                log_activity(S, 'signal', f"⚠ Аномалії обсягу {side} · {sc.get('name') or ''} · "
                             f"👤 особисто: {ok}" + (f' · не доставлено {bad}' if bad else ''),
                             side=side, source='ANOM')
            except Exception:
                pass
        return sent

    def _anom_mark(self, S, side, r, px, adir, now) -> bool:
        """⚠ Запамʼятати сигнал «Аномалії обсягу» для графіка кабінету (24.11).
        Один запис на сценарій (бік · ключ · час останньої реакції)."""
        sc = (r or {}).get('scenario') or {}
        t = int(float(sc.get('last_t') or now))
        key = f"{side}|{sc.get('key')}|{t}"
        lst = self._anom_marks.setdefault(S, [])
        if any(m.get('k') == key for m in lst):
            return False
        lst.append({'k': key, 't': t, 'at': int(now), 'side': side, 'key': sc.get('key'),
                    'name': sc.get('name') or '', 'px': px, 'score': (r or {}).get('score'),
                    'stages': [str(x.get('label') or '') for x in (sc.get('stages') or [])
                               if x.get('done')][:6],
                    'adir': (adir or {}).get('dir') if (adir or {}).get('state') == 'dir' else None})
        lst.sort(key=lambda m: m.get('t') or 0)
        cut = now - ANOM_MARK_KEEP
        self._anom_marks[S] = [m for m in lst if (m.get('t') or 0) >= cut][-ANOM_MARK_MAX:]
        if len(self._anom_marks) > ANOM_MARK_COINS:
            old = sorted(self._anom_marks, key=lambda k: max([m.get('t') or 0 for m in self._anom_marks[k]] or [0]))
            for k in old[:len(self._anom_marks) - ANOM_MARK_COINS]:
                self._anom_marks.pop(k, None)
        self._dirty = True
        return True

    def anom_marks(self, symbol: str) -> List[Dict]:
        """⚠ Позначки сигналів по монеті (лише памʼять, для графіка кабінету)."""
        cut = self._clock() - ANOM_MARK_KEEP
        return [dict(m) for m in (self._anom_marks.get(str(symbol or '').upper()) or [])
                if (m.get('t') or 0) >= cut]

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
            self.db.set_setting(DB_STATE, {'eps': self._eps, 'anom_sent': self._anom_sent,
                                           'anom_marks': self._anom_marks,
                                           'anom_last': self._anom_last,
                                           'ai_opens': self._ai_opens[-50:],
                                           'ai_last': self._ai_last,
                                           'cleanup_at': self._cleanup_at,
                                           'dis_shadow': self._dis_shadow[-DIS_SHADOW_MAX:]})
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
            ao = st.get('ai_opens')
            if isinstance(ao, list):
                self._ai_opens = [float(x) for x in ao if isinstance(x, (int, float))]
            al = st.get('ai_last')
            if isinstance(al, dict):
                self._ai_last = {str(k).upper(): list(v) for k, v in al.items()
                                 if isinstance(v, (list, tuple)) and len(v) == 2}
            dsh = st.get('dis_shadow')
            if isinstance(dsh, list):
                self._dis_shadow = [r for r in dsh if isinstance(r, dict) and r.get('key')][-DIS_SHADOW_MAX:]
            try:
                self._cleanup_at = float(st.get('cleanup_at') or 0)
            except (TypeError, ValueError):
                self._cleanup_at = 0.0
            am = st.get('anom_marks')
            if isinstance(am, dict):
                self._anom_marks = {str(k).upper(): [m for m in v if isinstance(m, dict)]
                                    for k, v in am.items() if isinstance(v, list)}
            for k, attr in (('anom_sent', '_anom_sent'), ('anom_last', '_anom_last')):
                v = st.get(k)
                if isinstance(v, dict):
                    setattr(self, attr, {str(a): float(b) for a, b in v.items()
                                         if isinstance(b, (int, float))})
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

    def readiness(self) -> Dict:
        """Стан для ГОЛОВНОГО ТУМБЛЕРА (легкий: лише памʼять, без БД)."""
        s = self.ai_settings()
        sc = self._scn()
        try:
            spring_on = sc is not None and bool((sc.get_settings() or {}).get('spring_enabled', True))
        except Exception:
            spring_on = False
        tm = self._tm_get()
        try:
            tm_on = bool(tm and tm.is_enabled())
        except Exception:
            tm_on = False
        try:
            test_mode = bool(tm and (getattr(tm, '_settings', {}) or {}).get('test_mode', True))
        except Exception:
            test_mode = False
        try:
            own = list(tm.ai_open_positions()) if tm and hasattr(tm, 'ai_open_positions') else []
        except Exception:
            own = []
        import sys as _sys
        va = _sys.modules.get('detection.vol_anomaly')
        try:
            mon = va.get_vol_anomaly() if va and hasattr(va, 'get_vol_anomaly') else None
            anom_on = True if mon is None else bool((getattr(mon, '_settings', {}) or {}).get('enabled', True))
        except Exception:
            anom_on = True
        with self._lock:
            view = list(self._view.values())
        ready = len([v for v in view if v.get('ready') and not v.get('block')])
        blocked = [(v.get('symbol'), v.get('block')) for v in view if v.get('ready') and v.get('block')]
        r = ai_readiness(s, spring_on, tm_on, test_mode, len(view), ready, len(own),
                         tick=dict(self._stat), now=self._clock(), anom_on=anom_on)
        r['own'] = [{'symbol': S, 'paper': b} for S, b in own]
        r['blocked'] = [{'symbol': a, 'why': b} for a, b in blocked[:8]]
        r['enabled'] = bool(s.get('enabled'))
        return r

    def state(self) -> Dict:
        with self._lock:
            view = sorted(self._view.values(),
                          key=lambda v: -max(v['long']['score'], v['short']['score']))
        model = self._pro_get().model_state()
        learning = {k: v for k, v in (self._learning_state or {}).items() if k != 'model'}
        professional = {
            'enabled': True,                   # частина 🤖 рішення завжди (17.11)
            'learning_enabled': bool(self.settings().get('ai_learning_enabled')),
            'model': {k: model.get(k) for k in ('version', 'ready', 'train_samples',
                                                'oos_samples', 'min_oos', 'brier', 'accuracy')},
            'learning': learning,
            'processed_archive_records': len(self._learner.seen) if self._learner else 0,
            'monitored_symbols': len(self._pro_analysis),
        }
        tm = self._tm_get()
        try:
            own = [{'symbol': S, 'paper': b} for S, b in
                   (tm.ai_open_positions() if tm and hasattr(tm, 'ai_open_positions') else [])]
        except Exception:
            own = []
        try:
            rdy_stats = tm.rdy_stats() if tm and hasattr(tm, 'rdy_stats') else None
        except Exception:
            rdy_stats = None
        return {'ok': True, 'settings': self.settings(), 'ai': self.ai_settings(),
                'rdy_stats': rdy_stats, 'dis_shadow': self.dis_shadow_state(),
                'ai_defaults': AI_DEFAULTS, 'scenario_keys': list(AI_SCENARIOS),
                'etalon': self.etalon_state(), 'cleanup': dict(self._cleanup_stat),
                'cleanup_rules': CLEANUP_RULES, 'cleanup_rules_ua': CLEANUP_RULES_UA,
                'cleanup_mark': dict(self._cleanup_mark),
                'cleanup_next': int(self._cleanup_at + CLEANUP_EVERY_SEC) if self._cleanup_at else None,
                'own_trades': own,
                'readiness': self.readiness(),
                'opens_30m': len([t for t in self._ai_opens if self._clock() - t < OPEN_WINDOW_SEC]),
                'stat': dict(self._stat),
                'calibration': self.calibration(), 'coins': view,
                'professional': professional,
                'episodes_open': len(self._eps), 'server_now': self._clock(),
                'weights': {'entry': W_ENTRY, 'exit': W_EXIT},
                'catalog': getattr(_scen(), 'CATALOG', None),
                'brain_rules': getattr(_sib('ai_brain'), 'RULES', None)}

    def _loop(self):
        time.sleep(45)
        while True:
            try:
                self.tick()
            except Exception as e:
                self._stat['errors'] += 1
                self._stat['err_at'] = time.time()
                self._stat['last_err'] = f'такт: {type(e).__name__}: {e}'[:200]
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


def ai_settings_now() -> Dict:
    """Налаштування ШІ для інших модулів (немає рушія → дефолти)."""
    eng = get_ai_engine()
    try:
        return eng.ai_settings() if eng else dict(AI_DEFAULTS)
    except Exception:
        return dict(AI_DEFAULTS)


def on_ai_trade_closed(closed: Dict) -> bool:
    """TM: закрилась угода ШІ → кандидат у еталони."""
    eng = get_ai_engine()
    try:
        return bool(eng and eng.etalon_consider(closed))
    except Exception as e:
        print(f'[AI] etalon: {e}')
        return False
