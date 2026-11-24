"""👤 ІНФОСАЙТ — ОСОБИСТА СТОРІНКА КОРИСТУВАЧА (вимоги 08.11).

Дослівно: «Це буде особиста сторінка кожного користувача окремо… Для кожного
користувача в базі має бути окрема збережена інформація… вигляд як у боті
WATCHLIST із своїм списком монет і графік…» + «Графік відобрази всі дані, що
на скріні… POC також нанеси на графік. Додай вибір таймфрейму в шапці
графіку, як на біржі, і вибір таймфреймів для OB і VOB.»

ГОЛОВНЕ ПРАВИЛО НАВАНТАЖЕННЯ: вартість НЕ росте з кількістю користувачів.
Графік складається з НЕЗАЛЕЖНИХ частин, кожна зі своїм TF і СПІЛЬНИМ кешем
на (частина, монета, TF):
- `bars`  — свічки графіка + структура CHoCH/BOS (той самий
  `detect_smc_structure`, що в боті) на TF графіка;
- `ob`    — Order Blocks (обидві сім'ї, `detect_order_blocks`) на TF OB,
  закриті бари → кеш до закриття бару;
- `vob`   — Volumized OB (`detect_volumized_obs` з параметрами сканера) на
  TF VOB, кеш до закриття бару (≤ `VOB_MAX_TTL`);
- `pd`    — Premium/Discount на `pd_zone_timeframe` сканера (лише монети поза
  ботом; для монет бота — `scanner.get_pd_pct`).
Монета бота + TF, що збігається з TF сканера → частина береться з ПАМ'ЯТІ
сканера (нуль запитів). Інакше частину рахує ОДИН фоновий потік
(`_worker`), і запит сторінки біржу НЕ чекає («⏳ рахується»). Навантаження =
кількість УНІКАЛЬНИХ (монета, TF), а не людей; частину, яку ніхто не просив
`IDLE_SEC`, більше не рахуємо (стеля `WANT_MAX`). POC — той самий
`compute_poc`, кеш на монету `POC_TTL`, рахує лише потік.

Особисті дані (список монет, обрана монета, TF графіка/OB/VOB) — у
`User.prefs['info']`, тобто в БД ОКРЕМО на кожного користувача.
"""
import json
import threading
import time

IWL_MAX = 20                 # стеля особистого списку (вимога 08.11)
IWL_DEFAULT = ('BTCUSDT', 'ETHUSDT')
# 1m/5m прибрано з графіка, 1m — з VOB (вимога 08.11: надто шумно і важко).
CHART_TFS = ('15m', '30m', '1h', '4h', '1d')
OB_TFS = ('15m', '30m', '1h', '4h')
VOB_TFS = ('5m', '15m', '30m', '1h', '4h')
# 📈 HH / HL / LH / LL — свінг-структура на графіку кабінету (вимога 15.11):
# ті самі свінг-півоти `detect_smc_structure` (swing_size сканера), що дають
# swing-OB і High/Low; TF — свій, деф. вимкнено (новий шар).
HL_TFS = ('15m', '30m', '1h', '4h', '1d')
HL_TF_DEFAULT = '1h'
HL_SHOW = 40                 # скільки останніх півотів малюємо
OFF = 'off'
POC_TTL = 300.0              # спільний кеш POC на (монету, вікно, TF)
# ⚪ POC — СВОЇ параметри кабінету (вимога 08.11: «вибір POC параметрів, за
# замовчуванням 7 днів»). Ринок — як у бота (`poc_filter_market`).
POC_DAYS = (1, 2, 3, 5, 7, 14, 30)
POC_DAYS_DEFAULT = 7
POC_TFS = ('15m', '1h', '4h')
POC_TF_DEFAULT = '1h'
COIN_TTL = 10.0              # спільний кеш відповіді графіка
OHLC_BARS = 400              # скільки барів віддаємо на графік
EVENTS_MAX = 80              # скільки останніх CHoCH/BOS малюємо
BARS_TTL = 30.0              # свічки графіка (живий бар)
VOB_MAX_TTL = 300.0          # VOB рахується на живому барі — не рідше ніж раз на 5 хв
SHL_TTL = 120.0              # 🏁 High/Low: межі тягнуться за живим баром — раз на 2 хв
SETTLE_SEC = 3.0             # біржі треба кілька секунд, щоб віддати закритий бар
IDLE_SEC = 900.0             # ніхто не просив 15 хв → не рахуємо
WANT_MAX = 600               # стеля одночасно відстежуваних частин
RETRY_SEC = 120.0            # біржа не віддала / монети немає → повтор
OB_BARS = 700                # як у сканера (_update_smc_ob)
PD_BARS = 3000

_lock = threading.Lock()
_poc_cache = {}              # (SYM, днів, TF) → (ts, view|None)
_coin_cache = {}             # (SYM, tf, ob_tf, vob_tf) → (ts, payload)
_comp = {}                   # (kind, SYM, tf) → {'data','at','until','err','err_at'}
_want = {}                   # (kind, SYM, tf) → коли востаннє просили
_poc_want = {}               # (SYM, днів, TF) → коли востаннє просили POC
_thread = None


# ── ЧИСТІ функції ─────────────────────────────────────────────────────────
def norm_symbol(s):
    s = str(s or '').upper().strip().replace('/', '').replace('-', '').replace('_', '')
    if s and not s.endswith('USDT') and s.isalnum() and len(s) <= 12:
        s = s + 'USDT'
    return s


def valid_symbol(s):
    """Формат USDT-перпа (BTCUSDT, 1000PEPEUSDT). Наявність на біржі перевіряє
    перший розрахунок — монета, якої немає, отримає зрозумілу причину."""
    return bool(s) and s.endswith('USDT') and s.isalnum() and 6 <= len(s) <= 24


def norm_tf(v, allowed, default, allow_off=False):
    t = str(v or '').lower().strip()
    if allow_off and t == OFF:
        return OFF
    return t if t in allowed else default


def iwl_norm(lst, universe=None, cap=IWL_MAX):
    """Особистий список: верхній регістр, без дублів, стеля."""
    out, seen = [], set()
    uni = set(universe) if universe is not None else None
    for x in (lst or []):
        s = norm_symbol(x)
        if not valid_symbol(s) or s in seen:
            continue
        if uni is not None and s not in uni:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def pd_view(pct, premium_min, discount_max, tf):
    """{pct, zone: premium|discount|eq, tf} або None («ще не пораховано»)."""
    try:
        p = float(pct)
    except (TypeError, ValueError):
        return None
    zone = 'premium' if p >= float(premium_min) else (
        'discount' if p <= float(discount_max) else 'eq')
    return {'pct': round(p, 1), 'zone': zone, 'tf': str(tf or '').upper()}


def to_sec(t):
    try:
        t = float(t or 0)
    except (TypeError, ValueError):
        return 0
    return int(t // 1000) if t > 1e12 else int(t)


def bar_sec(tf):
    t = str(tf or '').lower()
    try:
        return int(t[:-1]) * {'m': 60, 'h': 3600, 'd': 86400}[t[-1]]
    except Exception:
        return 3600


def next_close(now, tf):
    b = bar_sec(tf)
    return (int(now // b) + 1) * b + SETTLE_SEC


OB_SHOW = {'swing': 3, 'internal': 5}   # скільки блоків сім'ї малювати
OB_OVERLAP_MAX = 0.7                     # перекриття з уже показаним → не малюємо


def _overlap_share(a_top, a_bot, b_top, b_bot):
    """ЧИСТА: частка перекриття двох цінових діапазонів (від МЕНШОГО з них)."""
    lo, hi = max(min(a_top, a_bot), min(b_top, b_bot)), min(max(a_top, a_bot), max(b_top, b_bot))
    span = min(abs(a_top - a_bot), abs(b_top - b_bot))
    if hi <= lo:
        return 0.0
    return 1.0 if span <= 0 else (hi - lo) / span


def ob_boxes(zones):
    """Знімок OB-сімей → бокси для графіка. Без дублів між сім'ями, не більше
    `OB_SHOW` найновіших на сімʼю і без блоків того самого боку, що майже
    повністю лежать в уже показаному (скрін 08.11: 4-5 боксів «OB 1H int»
    одне на одному в ±30 пунктах — графік ставав нечитабельним)."""
    out, seen = [], set()
    for fam in ('swing', 'internal'):
        n = 0
        for o in (zones or {}).get(fam) or []:
            if n >= OB_SHOW.get(fam, 5):
                break
            try:
                top, bot = float(o['bar_high']), float(o['bar_low'])
            except (KeyError, TypeError, ValueError):
                continue
            st = to_sec(o.get('bar_time'))
            key = (st, round(top, 10), round(bot, 10))
            if key in seen:
                continue
            seen.add(key)
            side = 'LONG' if o.get('bias') == 'BULLISH' else 'SHORT'
            if any(b['side'] == side and _overlap_share(top, bot, b['top'], b['bottom'])
                   >= OB_OVERLAP_MAX for b in out):
                continue
            n += 1
            out.append({'side': 'LONG' if o.get('bias') == 'BULLISH' else 'SHORT',
                        'top': top, 'bottom': bot, 'start': st, 'fam': fam,
                        'tag': str(o.get('created_by_tag') or '')})
    return out


def vob_boxes(zones):
    out = []
    for z in (zones or []):
        try:
            out.append({'side': 'LONG' if z.get('type') == 'Bull' else 'SHORT',
                        'top': float(z['top']), 'bottom': float(z['bottom']),
                        'start': to_sec(z.get('start_time_sec')),
                        'breaker': bool(z.get('breaker')), 'tf': str(z.get('tf') or '').upper(),
                        'volume': z.get('volume'), 'pct': z.get('pct'),
                        'hi_pct': z.get('hi_pct'), 'lo_pct': z.get('lo_pct')})
        except (KeyError, TypeError, ValueError):
            continue
    return out


VOB_CHOP_FLIPS = 2          # закриття перескочило зону туди-назад ≥ N разів → «пропиляна»


def vob_stale(v, ohlc):
    """Причина, чому VOB уже не відповідає ситуації (вимога 10.11), або ''.

    Лише ПОКАЗ кабінету — детектор і сигнали бота не чіпаємо.
    • `breaker` — детектор сам визнав зону знеціненою;
    • ціна вже ЗА зоною проти її сенсу (LONG — закриття нижче низу, SHORT —
      вище верху): детектор ще не встиг позначити breaker, а зона вже пробита;
    • «пропиляна»: після появи закриття перескакувало зону з одного боку на
      інший ≥ `VOB_CHOP_FLIPS` разів — рівень не тримає, ціна ходить крізь нього.
    """
    if v.get('breaker'):
        return 'breaker'
    try:
        top, bot = float(v['top']), float(v['bottom'])
    except (KeyError, TypeError, ValueError):
        return ''
    st = v.get('start') or 0
    after = [b for b in (ohlc or []) if (b.get('time') or 0) >= st]
    if not after:
        return ''
    last = float(after[-1]['close'])
    if v.get('side') == 'LONG' and last < bot:
        return 'below'
    if v.get('side') == 'SHORT' and last > top:
        return 'above'
    flips, prev = 0, None
    for b in after:
        c = float(b['close'])
        s = 1 if c > top else (-1 if c < bot else 0)
        if s == 0:
            continue
        if prev is not None and s != prev:
            flips += 1
        prev = s
    return 'chop' if flips >= VOB_CHOP_FLIPS else ''


def vob_fresh(boxes, ohlc):
    """VOB-бокси без тих, що вже не відповідають ситуації (`vob_stale`)."""
    return [v for v in (boxes or []) if not vob_stale(v, ohlc)]


def hl_points(pivots, n=HL_SHOW):
    """ЧИСТА: свінг-півоти → [{t, price, type, dir}] для HH/HL/LH/LL на графіку.
    Колір за напрямком структури: HH/HL — бичача (bull), LH/LL — ведмежа (bear)."""
    out = []
    for p in pivots or []:
        ty = str(p.get('type') or '').upper()
        if ty not in ('HH', 'HL', 'LH', 'LL'):
            continue
        try:
            out.append({'t': to_sec(p.get('t')), 'price': float(p['price']), 'type': ty,
                        'high': ty in ('HH', 'LH'),
                        'dir': 'bull' if ty in ('HH', 'HL') else 'bear'})
        except (KeyError, TypeError, ValueError):
            continue
    out.sort(key=lambda x: x['t'])
    return out[-int(n):]


def events_of(analysis, n=EVENTS_MAX):
    """CHoCH/BOS внутрішньої структури (як на графіку бота) → лінія від півота
    до бару пробою + мітка. Найновіші `n`."""
    out = []
    for e in ((analysis or {}).get('internal') or {}).get('events') or []:
        try:
            out.append({'from': to_sec(e.get('from_t')), 'to': to_sec(e.get('to_t')),
                        'level': float(e['level']), 'tag': str(e.get('tag') or ''),
                        'dir': 'bull' if e.get('dir') == 'bull' else 'bear'})
        except (KeyError, TypeError, ValueError):
            continue
    return out[-int(n):]


def ohlc_of(klines, n=OHLC_BARS):
    out = []
    for k in (klines or [])[-int(n):]:
        try:
            c = float(k.get('p', k.get('close', 0)))
            out.append({'time': to_sec(k.get('t')),
                        'open': float(k.get('o', c)), 'high': float(k.get('h', c)),
                        'low': float(k.get('l', c)), 'close': c,
                        'volume': bar_volume(k, c)})
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def bar_volume(k, close=None):
    """ЧИСТА: обсяг свічки в МОНЕТАХ (як TradingView): поле біржі `q`, інакше
    обіг $ / ціна закриття. Немає даних → 0."""
    try:
        q = float(k.get('q') or 0)
        if q > 0:
            return q
        c = float(close if close is not None else k.get('p', k.get('close', 0)) or 0)
        v = float(k.get('v') or 0)
        return (v / c) if (c > 0 and v > 0) else 0.0
    except (TypeError, ValueError, AttributeError):
        return 0.0


IMB_SHOW = 16               # скільки невідпрацьованих IMB малюємо (15.11: 12→16 — як у TV видно всю серію)
SBR_SHOW = 40               # скільки останніх сигналів SBR на графіку


IMB_MAX_FILLED = 100        # 15.11: як у TV — зону знімає лише ЗАКРИТТЯ за нею (детектор)
IMB_MIN_REST_ATR = 0.0      # 15.11: тонкі зони TV малює — не ховаємо


def _imb_atr(bars):
    """Останній ATR(14) барів графіка (та сама RMA, що в детекторі) або None."""
    try:
        from detection.imbalance import _atr, _close
        h = [float(b['h']) for b in bars]
        l = [float(b['l']) for b in bars]
        c = [_close(b) for b in bars]
        a = _atr(h, l, c)
        return next((x for x in reversed(a) if x), None)
    except Exception:
        return None


def imb_visible(z, atr):
    """ЧИСТА: чи малювати IMB. КАЛІБРОВКА 15.11 «як у TradingView»: зону
    відпрацьовано лише коли бар ЗАКРИВСЯ за нею — це вирішує детектор
    (`mitigation='close'`), тож тут лишилась тільки перевірка коректності
    (верх вище низу). Колишні пороги «≥80% заповнено» / «залишок < 0.25·ATR»
    ховали зони, які TV показує (проколоті ґнотом, тонкі)."""
    try:
        if float(z.get('filled_pct') or 0) > IMB_MAX_FILLED:
            return False
        rest = float(z['top']) - float(z['bottom'])
    except (KeyError, TypeError, ValueError):
        return False
    if atr and IMB_MIN_REST_ATR and rest < IMB_MIN_REST_ATR * float(atr):
        return False
    return rest > 0


IMB_BARS = 1500             # історія для Pine-фільтра розміру (1000-барний максимум)
_imb_split = {}             # (SYM, tf, t_ms) → (bull, bear) — закрита свічка не змінюється
_IMB_SPLIT_CAP = 4000
BB_KLINE_URL = 'https://api.bybit.com/v5/market/kline'


def sub_base_interval(sub_min):
    """Делегат `detection.imbalance.sub_base_interval` (одне джерело з ⚠ аномаліями)."""
    from detection.imbalance import sub_base_interval as _f
    return _f(sub_min)


def group_subbars(bars, base_min, sub_min, t0_ms):
    """Делегат `detection.imbalance.group_subbars`."""
    from detection.imbalance import group_subbars as _f
    return _f(bars, base_min, sub_min, t0_ms)


def _imb_sub_split(sym, tf, t_ms):
    """Розклад обсягу середньої свічки за ПІДСВІЧКАМИ — як
    request.security_lower_tf у Pine (TF/10). Кеш назавжди (свічка закрита)."""
    key = (sym, tf, int(t_ms))
    if key in _imb_split:
        return _imb_split[key]
    from detection.imbalance import pine_sub_minutes, split_from_subbars
    bs = bar_sec(tf)
    sub = pine_sub_minutes(bs)
    base = sub_base_interval(sub)
    try:
        from detection.market_data import get_market_data
        md = get_market_data()
        r = md._session.get(BB_KLINE_URL, params={
            'category': 'linear', 'symbol': sym, 'interval': str(base),
            'start': int(t_ms), 'end': int(t_ms) + bs * 1000 - 1, 'limit': 1000}, timeout=10)
        rows = (r.json().get('result') or {}).get('list') or [] if r.status_code == 200 else []
        bars = [{'t': int(x[0]), 'o': float(x[1]), 'p': float(x[4]), 'v': float(x[5])}
                for x in rows if int(t_ms) <= int(x[0]) < int(t_ms) + bs * 1000]
    except Exception:
        bars = []
    if len(bars) < 2:
        return None                       # Pine-фолбек (частка за закриттям)
    sp = split_from_subbars(group_subbars(bars, base, sub, int(t_ms)))
    if len(_imb_split) > _IMB_SPLIT_CAP:
        _imb_split.clear()
    _imb_split[key] = sp
    return sp


# ── 📊 ОБСЯГ: покупці / продавці в кожній свічці (16.11) ──────────────────
# Історія: обсяг свічки розкладається за ПІДСВІЧКАМИ Bybit (зелена підсвічка →
# покупці, червона → продавці) — той самий метод, що «Volume Delta» у
# TradingView і що вже дає відсотки IMB (`split_from_subbars`). Живу свічку
# браузер уточнює СПРАВЖНІМИ угодами (publicTrade Bybit, сторона тейкера).
# На 15m / 30m / 1h підсвічка = TF/10 (як у Pine → число збігається з IMB);
# на 4h / 1d — крупніша, щоб перше заповнення не коштувало десятків запитів.
VOL_SUB_MIN = {'4h': 15, '1d': 60}
VOL_TTL = 30.0               # жива свічка: оновлення розкладу
VOL_PAGE = 1000              # барів Bybit на запит
VOL_PAGES_PER_PASS = 4       # запитів за один прохід потоку (найновіші бари — першими)


def vol_sub_plan(tf):
    """ЧИСТА: (підсвічка хв, базовий інтервал Bybit хв) для TF графіка."""
    t = str(tf or '').lower()
    sub = VOL_SUB_MIN.get(t)
    if not sub:
        from detection.imbalance import pine_sub_minutes
        sub = pine_sub_minutes(bar_sec(t))
    return int(sub), int(sub_base_interval(int(sub)))


def _sig6(x):
    try:
        return float('%.6g' % float(x))
    except (TypeError, ValueError):
        return 0.0


def vol_split_rows(rows, tf, sub, base):
    """ЧИСТА: бари `base` хв → {t свічки графіка (сек): (покупці, продавці)}."""
    from detection.imbalance import split_from_subbars
    bs = bar_sec(tf)
    groups = {}
    for r in rows or []:
        try:
            groups.setdefault(int(r['t']) // 1000 // bs * bs, []).append(r)
        except (KeyError, TypeError, ValueError):
            continue
    out = {}
    for t, g in groups.items():
        b, s = split_from_subbars(group_subbars(g, base, sub, t * 1000))
        out[t] = (_sig6(b), _sig6(s))
    return out


def vol_step(prev, tf, now, fetch, pages=VOL_PAGES_PER_PASS, bars=OHLC_BARS, page=VOL_PAGE):
    """Один прохід розкладу обсягу. `fetch(base_min, end_ms)` → до `page`
    найновіших барів Bybit із t ≤ end_ms ([{t, o, p, v}], будь-який порядок).
    Заповнює ВІД НАЙНОВІШИХ: спершу жива свічка і свіжі закриті, далі історія
    вглиб — не більше `pages` запитів за прохід. Уже пораховані закриті свічки
    не перезапитуються (свічка закрита → розклад остаточний).
    → {'sub', 'sp': {t: (b, s)}, 'lo', 'hi', 'live': {t, b, s, asof}, 'done'}."""
    bs = bar_sec(tf)
    sub, base = vol_sub_plan(tf)
    t_live = int(now // bs) * bs
    t_from = t_live - (int(bars) - 1) * bs
    prev = prev or {}
    sp, lo, hi = {}, None, None
    if prev.get('sub') == sub and prev.get('lo') is not None and prev.get('hi') is not None \
            and int(prev['hi']) > t_from:
        lo, hi = max(int(prev['lo']), t_from), int(prev['hi'])
        sp = {int(t): tuple(v) for t, v in (prev.get('sp') or {}).items() if lo <= int(t) < hi}
    budget = [max(1, int(pages))]

    def fill(end_ms, floor_s):
        """Сторінки вниз від end_ms до межі floor_s. → (рядки, досягнута межа сек)."""
        rows, reached = [], None
        while budget[0] > 0:
            budget[0] -= 1
            pg = [r for r in (fetch(base, int(end_ms)) or []) if r.get('t') is not None]
            if not pg:
                return rows, floor_s                    # глибшої історії немає
            oldest = min(int(r['t']) for r in pg)
            if oldest <= floor_s * 1000 or len(pg) < page:
                rows += [r for r in pg if int(r['t']) >= floor_s * 1000]
                return rows, floor_s
            cut = -(-oldest // (bs * 1000)) * bs        # перша ПОВНІСТЮ покрита свічка
            if cut * 1000 > end_ms:
                return rows, floor_s                    # сторінка не вміщає й однієї свічки
            rows += [r for r in pg if int(r['t']) >= cut * 1000]
            reached, end_ms = cut, cut * 1000 - 1
        return rows, reached

    now_ms = int(now * 1000)
    live = prev.get('live') if (prev.get('live') or {}).get('t') == t_live else None
    # А) жива свічка + нові закриті (від «зараз» униз до вже покритого)
    rows, reached = fill(now_ms, hi if hi is not None else t_from)
    if reached is not None:
        got = vol_split_rows(rows, tf, sub, base)
        lv = got.pop(t_live, None)
        live = {'t': t_live, 'b': (lv or (0.0, 0.0))[0], 's': (lv or (0.0, 0.0))[1], 'asof': now_ms}
        if hi is None or reached > hi:                  # старе покриття не дістали → нове
            sp, lo = {}, reached
        sp.update({t: v for t, v in got.items() if t < t_live})
        hi = t_live
        # Б) історія вглиб, поки є бюджет запитів
        if lo > t_from and budget[0] > 0:
            rows, reached = fill(lo * 1000 - 1, t_from)
            if reached is not None:
                sp.update({t: v for t, v in vol_split_rows(rows, tf, sub, base).items()
                           if reached <= t < lo})
                lo = reached
    if lo is None:
        return {'sub': sub, 'sp': {}, 'lo': None, 'hi': None, 'live': None, 'done': False}
    return {'sub': sub, 'sp': sp, 'lo': lo, 'hi': hi, 'live': live, 'done': lo <= t_from}


def vol_view(data, ohlc):
    """ЧИСТА: розклад → на графік: `m` = [[t, покупці, продавці], …] лише для
    свічок графіка; `live` — жива свічка на момент `asof` (далі її веде браузер
    справжніми угодами); `fill` — історія ще дозаповнюється."""
    if not data or data.get('lo') is None:
        return None
    sp = data.get('sp') or {}
    m = []
    for b in ohlc or []:
        v = sp.get(b.get('time'))
        if v and (v[0] > 0 or v[1] > 0):
            m.append([b['time'], v[0], v[1]])
    return {'sub': data.get('sub'), 'm': m, 'live': data.get('live'),
            'fill': not data.get('done')}


def _vol_fetch(md):
    def fetch(base, end_ms):
        r = md._session.get(BB_KLINE_URL, params={
            'category': 'linear', 'symbol': fetch.sym, 'interval': str(int(base)),
            'end': int(end_ms), 'limit': VOL_PAGE}, timeout=10)
        if r.status_code != 200:
            raise ValueError('біржа не віддала підсвічки (HTTP %s)' % r.status_code)
        rows = (r.json().get('result') or {}).get('list') or []
        return [{'t': int(x[0]), 'o': float(x[1]), 'p': float(x[4]), 'v': float(x[5])}
                for x in rows]
    return fetch


def imb_of(klines, tf, now=None, sym=None, hist=None):
    """📐 IMB = «PRO Volumatic FVG» Pine-скрипту користувача (15.11, 1:1):
    той самий покроковий прогін (фільтр розміру від 1000-барного максимуму,
    мітигація закриттям, видалення накладок, ≤10 зон), відсотки — з підсвічок
    TF/10 (кеш `_imb_split`, тягне фоновий потік). `hist` — закриті бари
    глибокої історії (частина 'imb'); `klines` — бари графіка з живим."""
    try:
        from detection.imbalance import detect_volfvg
    except Exception:
        return []
    bars = [k for k in (hist or []) if k.get('t') is not None]
    tail = [k for k in (klines or []) if k.get('t') is not None]
    if bars and tail:
        last = int(bars[-1]['t'])
        bars = bars + [k for k in tail if int(k['t']) > last]
    elif not bars:
        bars = tail

    def _split(t):
        return _imb_split.get((sym, tf, int(t))) if sym else None
    out = []
    for z in detect_volfvg(bars, split=_split):
        out.append({'side': z['side'], 'top': z['top'], 'bottom': z['bottom'],
                    'start': z['t'], 'filled_pct': 0, 'vol': z.get('vol'),
                    'pct': z.get('pct'), 'sell_pct': z.get('sell_pct'),
                    'buy_pct': z.get('buy_pct'), 'approx': z.get('approx')})
    return out


SBR_TFS = ('1h', '4h')       # SBR — TF рушія бота («останній сигнал» у шапці графіка)
# 🔔 Особисті підписки — БУДЬ-ЯКИЙ TF рушія SBR (вимога 17.11). Дзеркало
# `sbr_signals.ALL_TFS` (тест звіряє): 15m/30m/45m/1d рушій рахує лише для
# монет, де цей TF обрано.
SBR_SUB_TFS = ('15m', '30m', '45m', '1h', '4h', '1d')
# 🔔 SBR НА ГРАФІКУ КАБІНЕТУ — на ВСІХ таймфреймах графіка (вимога 16.11).
# 'chart' = TF графіка; можна обрати й інший TF графіка (сигнали 4H на 15m-
# графіку тощо) — тоді його рахує фоновий потік (частина 'sbr'), кеш — до
# закриття бару цього TF.
SBR_CHART = 'chart'
SBR_CHART_TFS = CHART_TFS


def sbr_tf_norm(v):
    """ЧИСТА: TF сигналів SBR на графіку → 'chart' або один із TF графіка."""
    t = str(v or '').lower()
    return t if t in SBR_CHART_TFS else SBR_CHART


def sbr_of(klines, tf, now=None):
    """🔔 Сигнали Session Band Reversal на ЗАКРИТИХ барах — той самий
    `sbr_signals.compute_signals`, що рушій бота (1H/4H) і Telegram.
    Будь-який TF (вимога 16.11 скасувала «лише 1H / 4H» від 09.11)."""
    try:
        from detection.sbr_signals import compute_signals, closed_bars
    except Exception:
        return []
    now = time.time() if now is None else now
    b = bar_sec(tf)
    cb = closed_bars(klines or [], b, now)
    out = []
    for x in compute_signals(cb, b)[-SBR_SHOW:]:
        out.append({'side': x['side'], 't': to_sec(x['t']), 'ct': sig_close_sec(x, b), 'n': x.get('n'),
                    'trigger': x.get('trigger'), 'score': x.get('score'),
                    'risk': x.get('risk'), 'sl': x.get('sl'),
                    'time_kyiv': x.get('time_kyiv'), 'tf': str(tf).upper()})
    return out


def sig_of(payload):
    o = payload.get('ohlc') or []
    last = o[-1] if o else {}
    ev = payload.get('events') or [{}]
    return '|'.join(str(x) for x in (
        payload.get('tf'), payload.get('ob_tf'), payload.get('vob_tf'), payload.get('hl_tf'),
        len(payload.get('hl') or []), ((payload.get('hl') or [{}])[-1]).get('t'),
        len(o), last.get('time'), last.get('close'),
        len(payload.get('ob') or []), (payload.get('ob') or [{}])[0].get('start'),
        len(payload.get('vob') or []), (payload.get('vob') or [{}])[0].get('start'),
        len(payload.get('events') or []), ev[-1].get('to'),
        len(payload.get('imb') or []), sum(round(float(z.get('top') or 0), 8) for z in payload.get('imb') or []),
        sum(round(float(z.get('bottom') or 0), 8) for z in payload.get('imb') or []),
        len(payload.get('sbr') or []), ((payload.get('sbr') or [{}])[-1]).get('t'),
        (payload.get('pd') or {}).get('pct'), (payload.get('poc') or {}).get('poc'),
        ((payload.get('shl') or {}).get('high') or {}).get('price'),
        ((payload.get('shl') or {}).get('low') or {}).get('price'),
        *[((payload.get('sbr_last') or {}).get(t) or {}).get('t') for t in SBR_TFS],
        len((payload.get('vsplit') or {}).get('m') or []) if payload.get('vsplit') else '-',
        payload.get('shl_tf'), len(payload.get('anom_sig') or []),
        ((payload.get('anom_sig') or [{}])[-1]).get('t')))


def due(kind, rec, now):
    """ЧИСТА: чи пора перерахувати частину."""
    rec = rec or {}
    if rec.get('err') and now - float(rec.get('err_at') or 0) < RETRY_SEC:
        return False
    if not rec.get('at'):
        return True
    if kind == 'bars':
        return now - float(rec['at']) >= BARS_TTL
    return now >= float(rec.get('until') or 0)


# ── Доступ до бота (лише памʼять) ─────────────────────────────────────────
def _scanner():
    try:
        from detection.smc_scanner import get_smc_scanner
        return get_smc_scanner()
    except Exception:
        return None


def universe(sc=None):
    sc = sc or _scanner()
    try:
        return [str(s).upper() for s in (sc.get_watchlist() or [])] if sc else []
    except Exception:
        return []


def _in_bot(sc, sym):
    return sym in set(universe(sc))


def defaults(sc):
    st = getattr(sc, '_settings', {}) or {}
    try:
        tf = sc.get_timeframe()
    except Exception:
        tf = '15m'
    return {'tf': norm_tf(tf, CHART_TFS, '15m'),
            'ob_tf': norm_tf(st.get('ob_filter_timeframe'), OB_TFS, '1h'),
            'vob_tf': (norm_tf(st.get('volumized_timeframe'), VOB_TFS, '5m')
                       if st.get('use_volumized_ob', True) else OFF)}


def _bot_part(sc, kind, sym, tf):
    """Частина з ПАМ'ЯТІ сканера, коли монета в боті й TF збігається. None →
    сканер цю частину на цьому TF не рахує."""
    if not _in_bot(sc, sym):
        return None
    st = sc._settings
    if kind == 'bars':
        try:
            if tf != str(sc.get_timeframe()).lower():
                return None
        except Exception:
            return None
        with sc._lock:
            c = sc._cache.get(sym)
        if not c or not c.get('klines'):
            return None
        return {'kl': c['klines'], 'events': events_of(c.get('analysis'))}
    if kind == 'ob':
        if tf != str(st.get('ob_filter_timeframe', '1h')).lower():
            return None
        with sc._lock:
            z = dict((getattr(sc, '_ob_zones', {}) or {}).get(sym) or {})
        return {'zones': z} if z else None
    if kind == 'vob':
        if not st.get('use_volumized_ob', True) or tf != str(st.get('volumized_timeframe', '')).lower():
            return None
        vp = info_vob_params(st)
        if vp['own']:                 # свої параметри інфосайту → рахує фоновий потік
            return None
        try:
            return {'vz': sc._volumized_zones(sym), 'pk': vp['pk']}
        except Exception:
            return None
    return None


# 📦 Volumized OB — ОКРЕМІ параметри для графіка інфосайту (блок
# «📦 Volumized OB · інфосайт» у налаштуваннях бота). Ключ не задано → береться
# значення блоку бота «📦 Volumized OB Trend» (стара поведінка без міграції).
INFO_VOB_KEYS = (('swing', 'swing_length', 10), ('end', 'ob_end_method', 'Wick'),
                 ('atr', 'max_atr_mult', 3.5), ('zone', 'zone_count', 'Low'),
                 ('combine', 'combine_obs', True))


def info_vob_params(st):
    """ЧИСТА: {on, swing, end, atr, zone, combine, own, pk}. `own` — параметри
    інфосайту відрізняються від ботових (пам'ять сканера тоді не годиться);
    `pk` — підпис параметрів (кеш, порахований іншими, не показуємо)."""
    st = st or {}
    out, own = {}, False
    for k, name, d in INFO_VOB_KEYS:
        bot = st.get('volumized_' + name, d)
        v = st.get('info_vob_' + name)
        if v is None:
            v = bot
        out[k] = v
        own = own or v != bot
    try:
        out['swing'] = max(3, int(out['swing']))
    except (TypeError, ValueError):
        out['swing'] = 10
    try:
        out['atr'] = max(0.1, min(50.0, float(out['atr'])))
    except (TypeError, ValueError):
        out['atr'] = 3.5
    out['combine'] = bool(out['combine'])
    out['on'] = st.get('info_vob_enabled') is not False
    out['own'] = own
    out['pk'] = '%s|%s|%s|%s|%s' % (out['swing'], out['end'], out['atr'], out['zone'],
                                    int(out['combine']))
    return out


def _engine_sbr_last(sym, tf):
    """🔔 Останній сигнал SBR із ПАМ'ЯТІ рушія бота (той самий, що блок
    🔔 МММ-SBR і Telegram). None → рушій монету на цьому TF не веде."""
    try:
        from detection.sbr_signals import get_sbr_engine
        eng = get_sbr_engine()
        snap = (eng.snaps(sym) or {}).get(tf) if eng else None
    except Exception:
        return None
    if not snap:
        return None
    return {'sig': sbr_last_view(snap.get('last'), tf)}


def sig_close_sec(sig, b=0):
    """ЧИСТА: момент СПРАЦЮВАННЯ сигналу (закриття його бару), epoch с.
    Саме його показуємо в картці/бейджі — той самий момент, коли сигнал іде в
    Telegram (вимога 10.11). Немає `close_t` → відкриття бару + тривалість."""
    try:
        ct = float(sig.get('close_t') or 0)
    except (TypeError, ValueError):
        ct = 0
    if ct > 0:
        return to_sec(ct)
    t = to_sec(sig.get('t'))
    return t + int(b or 0) if t else 0


def sbr_last_view(sig, tf):
    """ЧИСТА: останній сигнал SBR → поля для бейджа кабінету."""
    if not sig or not sig.get('trigger'):
        return None
    return {'side': sig.get('side'), 't': to_sec(sig.get('t')),
            'ct': sig_close_sec(sig, bar_sec(tf)), 'n': sig.get('n'),
            'trigger': float(sig['trigger']), 'score': sig.get('score'),
            'time_kyiv': sig.get('time_kyiv'), 'tf': str(tf).upper()}


def sbr_move(side, trigger, price):
    """ЧИСТА: рух ціни від ціни сигналу (Trigger) до поточної, % → {pct, fav}.
    fav=True — ціна пішла В БІК сигналу (LONG угору, SHORT униз)."""
    try:
        tr, px = float(trigger), float(price)
    except (TypeError, ValueError):
        return None
    if tr <= 0 or px <= 0:
        return None
    pct = (px - tr) / tr * 100.0
    fav = pct > 0 if side == 'LONG' else (pct < 0 if side == 'SHORT' else None)
    if abs(pct) < 1e-9:
        fav = None
    return {'pct': round(pct, 3), 'fav': fav}


def _get(key):
    with _lock:
        return dict(_comp.get(key) or {})


def _kick():
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_worker, name='info-portal', daemon=True)
        _thread.start()


def want(key):
    """Сторінка попросила частину → поставити в роботу (без чекання)."""
    with _lock:
        if key not in _want and len(_want) >= WANT_MAX:
            return False
        _want[key] = time.time()
        _kick()
    return True


def poc_norm(days=None, tf=None):
    """ЧИСТА: параметри POC кабінету → (днів, TF) з дозволених значень."""
    try:
        d = int(float(days))
    except (TypeError, ValueError):
        d = POC_DAYS_DEFAULT
    if d not in POC_DAYS:
        d = POC_DAYS_DEFAULT
    t = str(tf or '').lower()
    return d, (t if t in POC_TFS else POC_TF_DEFAULT)


def poc_want(sym, days=None, tf=None):
    key = (sym,) + poc_norm(days, tf)
    with _lock:
        if key in _poc_want or len(_poc_want) < WANT_MAX:
            _poc_want[key] = time.time()
            _kick()


def part(sc, kind, sym, tf):
    """(дані|None, помилка). Пам'ять сканера, інакше спільний кеш + фоновий потік."""
    d = _engine_sbr_last(sym, tf) if kind == 'sbrlast' else _bot_part(sc, kind, sym, tf)
    if d is not None:
        return d, ''
    key = (kind, sym, tf)
    want(key)
    rec = _get(key)
    return rec.get('data'), rec.get('err') or ''


# ── 🛰 Фоновий розрахунок — ТИМИ САМИМИ функціями й параметрами ──────────
def _compute(sc, kind, sym, tf):
    from detection.market_data import get_market_data
    md = get_market_data()
    st = sc._settings
    isize = sc.get_internal_size()
    ssize = int(st.get('swing_size', 50))
    now = time.time()
    if kind == 'bars':
        from detection.smc_structure import detect_smc_structure
        kl = md.fetch_klines(sym, limit=OHLC_BARS, interval=tf)
        if not kl or len(kl) < 20:
            raise ValueError('монету не знайдено на біржі (немає барів)')
        an = detect_smc_structure(kl, internal_size=isize, swing_size=ssize)
        return {'kl': kl, 'events': events_of(an)}, now + BARS_TTL
    if kind == 'imb':
        # 📐 Глибока історія для Pine-фільтра + розклад обсягу живих зон.
        from detection.imbalance import detect_volfvg
        hk = md.fetch_klines(sym, limit=IMB_BARS, interval=tf)
        if not hk or len(hk) < 20:
            raise ValueError('немає барів ' + tf.upper())
        bs = bar_sec(tf)
        hk = [k for k in hk if (int(k['t']) // 1000 if int(k['t']) > 1e12 else int(k['t'])) + bs <= now]
        miss = 0
        for z in detect_volfvg(hk):
            if _imb_sub_split(sym, tf, z['mid_t']) is None:
                miss += 1
        if hk:
            _imb_sub_split(sym, tf, int(hk[-1]['t']))   # середня свічка для гепу на живому барі
        until = next_close(now, tf)
        return {'kl': hk}, (min(until, now + 120) if miss else until)
    if kind == 'vol':
        # 📊 Розклад обсягу свічок (покупці / продавці) за підсвічками Bybit.
        f = _vol_fetch(md)
        f.sym = sym
        d = vol_step(_get(('vol', sym, tf)).get('data'), tf, now, f)
        if d.get('lo') is None:
            raise ValueError('немає підсвічок ' + tf.upper())
        return d, (now + VOL_TTL if d.get('done') else now)
    if kind == 'ob':
        from detection.smc_structure import detect_smc_structure
        from detection.ob_detector import detect_order_blocks
        ok = md.fetch_klines(sym, limit=OB_BARS, interval=tf)
        zones = {'tf': tf}
        if ok and len(ok) > 60:
            cl = ok[:-1]
            res = detect_smc_structure(cl, internal_size=isize, swing_size=ssize)
            for fam in ('internal', 'swing'):
                f = res.get(fam, {}) or {}
                zones[fam] = detect_order_blocks(klines=cl, pivots=f.get('pivots', []),
                                                 events=f.get('events', []),
                                                 limit=getattr(sc, 'OB_ZONES_SHOW', 8))
        return {'zones': zones}, next_close(now, tf)
    if kind == 'vob':
        from detection.volumized_ob import detect_volumized_obs
        vk = md.fetch_klines(sym, limit=getattr(sc, 'VOB_KLINES_LIMIT', 3000), interval=tf)
        vz = []
        vp = info_vob_params(st)
        if vk and len(vk) > 20:
            r = detect_volumized_obs(
                vk, swing_length=vp['swing'], ob_end_method=vp['end'],
                max_atr_mult=vp['atr'], zone_count=vp['zone'],
                combine_obs=vp['combine'])
            obs = list(r.get('bullish_obs') or []) + list(r.get('bearish_obs') or [])
            obs.sort(key=lambda o: o.get('formation_time', o.get('start_time', 0)) or 0,
                     reverse=True)
            vz = [z for z in (sc._format_vob(o, tf) for o in obs) if z]
        return {'vz': vz, 'pk': vp['pk']}, min(next_close(now, tf), now + VOB_MAX_TTL)
    if kind == 'hl':
        # 📈 HH/HL/LH/LL — свінг-півоти на ЗАКРИТИХ барах свого TF (як OB).
        from detection.smc_structure import detect_smc_structure
        hk = md.fetch_klines(sym, limit=OB_BARS, interval=tf)
        pts = []
        if hk and len(hk) > 60:
            res = detect_smc_structure(hk[:-1], internal_size=isize, swing_size=ssize)
            pts = hl_points((res.get('swing') or {}).get('pivots') or [])
        return {'pts': pts}, next_close(now, tf)
    if kind == 'shl':
        # 🏁 Swing High / Low — ТА САМА функція бота, що лінії Strong/Weak на
        # графіку /smart-money (дилінг-діапазон = єдине джерело з PD-зоною).
        return {'levels': sc._swing_hl_for_tf(sym, tf)}, now + SHL_TTL
    if kind == 'sbr':
        # 🔔 SBR на ІНШОМУ TF, ніж графік (вибір у кабінеті): ті самі
        # compute_signals на закритих свічках свого TF, раз на закриття бару.
        from detection.sbr_signals import BARS as _SB
        sk = md.fetch_klines(sym, limit=_SB, interval=tf)
        if not sk or len(sk) < 30:
            raise ValueError('немає барів ' + tf.upper())
        return ({'sigs': sbr_of(sk, tf, now)}, next_close(now, tf) + SETTLE_SEC)
    if kind == 'sbrlast':
        # 🔔 Монети немає в рушії SBR → ТА САМА compute_signals на тих самих
        # 1000 закритих барах (як рушій), раз на закриття бару.
        from detection.sbr_signals import compute_signals, closed_bars, BARS
        sk = md.fetch_klines(sym, limit=BARS, interval=tf)
        if not sk or len(sk) < 30:
            raise ValueError('немає барів ' + tf.upper())
        sigs = compute_signals(closed_bars(sk, bar_sec(tf), now), bar_sec(tf))
        return ({'sig': sbr_last_view(sigs[-1] if sigs else None, tf)},
                next_close(now, tf) + SETTLE_SEC)
    if kind == 'pd':
        from detection.smc_structure import detect_smc_structure
        pk = md.fetch_klines(sym, limit=PD_BARS, interval=tf)
        pd_pct = None
        if pk and len(pk) > 60:
            pcl = pk[:-1]
            pres = detect_smc_structure(pcl, internal_size=isize, swing_size=ssize)
            pd_pct = sc._compute_pd_pct(pcl, (pres.get('swing', {}) or {}).get('pivots', []),
                                        pcl[-1].get('p'))
        return {'pd_pct': pd_pct}, next_close(now, tf)
    raise ValueError(kind)


def _worker():
    """ОДИН потік на всі частини — біржу не смикаємо паралельно."""
    global _thread
    while True:
        try:
            sc = _scanner()
            now = time.time()
            with _lock:
                for k in [k for k, t in _want.items() if now - t > IDLE_SEC]:
                    _want.pop(k, None)
                    _comp.pop(k, None)
                for k in [k for k, t in _poc_want.items() if now - t > IDLE_SEC]:
                    _poc_want.pop(k, None)
                keys = sorted(_want, key=lambda k: float((_comp.get(k) or {}).get('at') or 0))
                pwant = list(_poc_want)
                if not keys and not pwant:
                    _thread = None
                    return
            if sc is None:
                time.sleep(5)
                continue
            for key in keys:
                kind, sym, tf = key
                if (_engine_sbr_last(sym, tf) if kind == 'sbrlast'
                        else _bot_part(sc, kind, sym, tf)) is not None:
                    continue                      # бот уже веде цю частину
                if not due(kind, _get(key), time.time()):
                    continue
                try:
                    data, until = _compute(sc, kind, sym, tf)
                    upd = {'data': data, 'at': time.time(), 'until': until, 'err': ''}
                except Exception as e:
                    upd = {'err': str(e)[:160], 'err_at': time.time()}
                with _lock:
                    if key in _want:
                        _comp.setdefault(key, {}).update(upd)
                        _coin_drop(sym)
            for pk in pwant:
                sym, pdays, ptf = pk
                with _lock:
                    hit = _poc_cache.get(pk)
                if hit and time.time() - hit[0] < POC_TTL:
                    continue
                _poc(sc, sym, allow_compute=True, days=pdays, tf=ptf)
                with _lock:
                    _coin_drop(sym)
            time.sleep(1.0)
        except Exception as e:
            print(f"[INFO] portal worker: {e}")
            time.sleep(5)


def _coin_drop(sym):
    for k in [k for k in _coin_cache if k[0] == sym]:
        _coin_cache.pop(k, None)


def _price(sc, sym):
    d, _e = part(sc, 'bars', sym, defaults(sc)['tf'])
    kl = (d or {}).get('kl') or []
    try:
        return float(kl[-1].get('p') or 0) if kl else None
    except Exception:
        return None


def shl_tf(sc):
    """TF ліній High/Low — той самий, що в боті (`swing_hl_timeframe`, деф. 1H)."""
    try:
        return str(sc._settings.get('swing_hl_timeframe', '1h') or '1h').lower()
    except Exception:
        return '1h'


def shl_view(levels, tf):
    """ЧИСТА: {high:{price,label}, low:{price,label}} → для графіка (+ TF)."""
    out = {}
    for k in ('high', 'low'):
        v = (levels or {}).get(k) or {}
        try:
            p = float(v.get('price') or 0)
        except (TypeError, ValueError):
            p = 0.0
        if p > 0:
            out[k] = {'price': p, 'label': str(v.get('label') or k.title())}
    if not out:
        return None
    out['tf'] = str(tf).upper()
    return out


def _pd(sc, sym):
    try:
        prem, disc = sc.get_pd_thresholds()
        pd_tf = str(sc._settings.get('pd_zone_timeframe', '1h')).lower()
        if _in_bot(sc, sym):
            pct = sc.get_pd_pct(sym)
        else:
            d, _e = part(sc, 'pd', sym, pd_tf)
            pct = (d or {}).get('pd_pct')
        return pd_view(pct, prem, disc, pd_tf)
    except Exception:
        return None


def _poc_market(sc):
    s = getattr(sc, '_settings', {}) or {}
    return str(s.get('poc_filter_market', 'futures') or 'futures')


def _poc(sc, sym, allow_compute=True, price=None, days=None, tf=None):
    """POC через ТОЙ САМИЙ compute_poc, що бейдж графіка, зі спільним кешем.
    Вікно (днів) і TF — параметри кабінету (`poc_norm`, деф. 7 днів · 1H).
    Вікно прибите до години (to = поточна година) — ключ кешу стабільний."""
    days, tf = poc_norm(days, tf)
    ck = (sym, days, tf)
    now = time.time()
    with _lock:
        hit = _poc_cache.get(ck)
    if hit and now - hit[0] < POC_TTL:
        view = hit[1]
    elif not allow_compute:
        return ('pending', None)
    else:
        view = None
        try:
            from detection.volume_profile import compute_poc
            market = _poc_market(sc)
            to_s = int(now // 3600 * 3600)
            res = compute_poc(sym, from_sec=to_s - int(days * 86400), to_sec=to_s,
                              interval=tf, market=market)
            if res.get('ok') and res.get('poc'):
                view = {'poc': float(res['poc']), 'market': market.upper(),
                        'tf': tf.upper(), 'days': days, 'exchange': res.get('exchange')}
        except Exception as e:
            print(f"[INFO] POC {sym}: {e}")
        with _lock:
            _poc_cache[ck] = (now, view)
            if len(_poc_cache) > 600:
                _poc_cache.clear()
    if not view:
        return ('none', None)
    out = dict(view)
    if price is None and allow_compute is False:
        price = _price(sc, sym)
    try:
        from detection.volume_profile import price_vs_poc
        v = price_vs_poc(out['poc'], price)
        out['rel'], out['dist_pct'] = v.get('rel'), v.get('dist_pct')
    except Exception:
        out['rel'] = out['dist_pct'] = None
    return ('ok', out)


MM_TTL = 60.0                # кеш фолбеку МММ-бабло для монет поза знімком
MM_SYMBOLS = ('BTCUSDT',)    # 🧭 назва кольором МММ-бабло — лише BTC
_mm_cache = {}               # {SYM: (ts, mark|None)}


def mm_marks(symbols, ff=None, now=None):
    """🧭 «МММ-бабло» по монетах кабінету (вимога 08.11): напис монети фарбується
    так само, як рядок «МММ-бабло ↑ зверху → тягне в LONG / ↓ знизу → SHORT».

    Формула в боті ОДНА: (fa−fb)/den по liq-map, поріг ±0.1 — її рахує
    `FuelFilterDaemon._fuel_dir_legacy` (колонка «🧮 МММ LiQ»). Тут — ЛИШЕ
    читання: спершу готовий знімок двигуна (`mm_snapshot_for`), а для монет, яких
    у знімку немає, — той самий `_fuel_dir_legacy` над кешованим (20с) станом
    liq-map, з власним кешем MM_TTL. Біржу НЕ смикаємо і liq-map нових монет НЕ
    замовляємо: немає даних → кольору немає («невідомо» ≠ «рівновага»).
    → {SYM: {'dir': 'LONG'|'SHORT'|'FLAT', 'str': 0..100}}."""
    now = time.time() if now is None else now
    want = [norm_symbol(x) for x in (symbols or []) if x]
    want = [x for x in want if x]
    if not want:
        return {}
    if ff is None:
        try:
            from detection.fuel_filter import get_fuel_filter
            ff = get_fuel_filter()
        except Exception:
            ff = None
    if ff is None:
        return {}
    out = {}
    try:
        snap = ff.mm_snapshot_for(want) or {}
    except Exception:
        snap = {}
    for sym in want:
        v = snap.get(sym)
        if v is not None:
            st = v.get('mm')
            out[sym] = {'dir': st if st in ('LONG', 'SHORT') else 'FLAT',
                        'str': int(v.get('strength') or 0)}
            continue
        hit = _mm_cache.get(sym)
        if hit and now - hit[0] < MM_TTL:
            m = hit[1]
        else:
            try:
                m = ff._fuel_dir_legacy(sym)
            except Exception:
                m = None
            _mm_cache[sym] = (now, m)
            if len(_mm_cache) > 600:
                _mm_cache.clear()
        if m:
            st = m.get('status')
            out[sym] = {'dir': st if st in ('LONG', 'SHORT') else 'FLAT',
                        'str': int(m.get('strength') or 0)}
    return out


def rows_for(symbols, sc=None):
    """Рядки особистого списку: лише назва, ціна і стан (PD/POC тепер на
    графіку — вимога 08.11). Ціна — з тих самих барів, що графік за TF бота."""
    sc = sc or _scanner()
    uni = set(universe(sc))
    rows = []
    for sym in iwl_norm(symbols):
        src = 'bot' if sym in uni else 'ext'
        tf = defaults(sc)['tf'] if sc else '15m'
        d, err = part(sc, 'bars', sym, tf) if sc else (None, '')
        kl = (d or {}).get('kl') or []
        price = chg = None
        try:
            if kl:
                price = float(kl[-1].get('p') or 0) or None
                day = max(1, int(86400 // bar_sec(tf)))
                base = float(kl[-min(len(kl), day + 1)].get('p') or 0)
                if base > 0 and price:
                    chg = round((price / base - 1) * 100, 2)
        except Exception:
            pass
        rows.append({'symbol': sym, 'price': price, 'chg24': chg, 'src': src, 'err': err})
    # 🧭 Колір «МММ-бабло» — ЛИШЕ для BTC (вимога 08.11): решта монет без
    # кольору, тож і рахувати по них нічого.
    try:
        mm = mm_marks([r['symbol'] for r in rows if r['symbol'] in MM_SYMBOLS])
    except Exception:
        mm = {}
    for r in rows:
        r['mm'] = mm.get(r['symbol'])
    return rows


# ── ⚠ Сигнали «Аномалії обсягу» на графіку кабінету (вимога 24.11) ───────
# Той самий сигнал, що йде в Telegram (готовий сценарій входу ШІ). Рушій ШІ
# фіксує його по кожній монеті, яку веде (Пружини · підписки · графіки з
# увімкненим шаром ⚠). Монета з відкритого графіка ведеться ANOM_CHART_IDLE
# після останнього запиту; позначки зʼявляються з моменту, як її почали вести.
ANOM_CHART_IDLE = IDLE_SEC
ANOM_CHART_MAX = 60
_anom_chart = {}


def anom_chart_want(sym, now=None):
    now = time.time() if now is None else now
    S = norm_symbol(sym)
    if not valid_symbol(S):
        return
    with _lock:
        _anom_chart[S] = now
        if len(_anom_chart) > ANOM_CHART_MAX * 2:
            for k, _t in sorted(_anom_chart.items(), key=lambda x: x[1])[:len(_anom_chart) - ANOM_CHART_MAX]:
                _anom_chart.pop(k, None)


def anom_chart_symbols(now=None):
    """[SYM…] графіків кабінету з увімкненим шаром ⚠ (останні ANOM_CHART_IDLE)."""
    now = time.time() if now is None else now
    with _lock:
        live = [(k, t) for k, t in _anom_chart.items() if now - t <= ANOM_CHART_IDLE]
    live.sort(key=lambda x: -x[1])
    return [k for k, _t in live[:ANOM_CHART_MAX]]


def anom_marks_of(sym, ohlc=None):
    """⚠ позначки монети з рушія ШІ (памʼять), лише в межах графіка."""
    import sys
    m = sys.modules.get('detection.ai_signals')
    try:
        eng = m.get_ai_engine() if m is not None else None
        marks = eng.anom_marks(sym) if eng is not None else []
    except Exception:
        marks = []
    t0 = (ohlc or [{}])[0].get('time') if ohlc else None
    out = []
    for x in marks or []:
        try:
            t = int(x.get('t') or 0)
        except (TypeError, ValueError):
            continue
        if t0 is not None and int(x.get('at') or t) < t0:
            continue
        # ct — момент СПРАЦЮВАННЯ (коли рушій побачив готовий сценарій): позначка
        # стоїть на свічці графіка, де сигнал спрацював (як SBR, правило 10.11)
        out.append({'t': t, 'ct': int(x.get('at') or t), 'side': x.get('side'), 'name': x.get('name') or '',
                    'key': x.get('key') or '', 'px': x.get('px'), 'score': x.get('score'),
                    'stages': list(x.get('stages') or [])[:6], 'adir': x.get('adir')})
    return out


def coin_payload(sym, sc=None, tf=None, ob_tf=None, vob_tf=None, poc_days=None, poc_tf=None,
                 hl_tf=None, vol=False, sbr_tf=None, shl_tfv=None, anom=False):
    sc = sc or _scanner()
    sym = norm_symbol(sym)
    if not sc:
        return {'ok': False, 'reason': 'сканер бота недоступний'}
    if not valid_symbol(sym):
        return {'ok': False, 'reason': (sym or "?") + ' — некоректна назва монети'}
    dfl = defaults(sc)
    tf = norm_tf(tf, CHART_TFS, dfl['tf'])
    ob_tf = norm_tf(ob_tf, OB_TFS, dfl['ob_tf'], allow_off=True)
    vob_tf = norm_tf(vob_tf, VOB_TFS, dfl['vob_tf'], allow_off=True)
    poc_days, poc_tf = poc_norm(poc_days, poc_tf)
    hl_tf = norm_tf(hl_tf, HL_TFS, OFF, allow_off=True)
    in_bot = _in_bot(sc, sym)
    _vp = info_vob_params(getattr(sc, '_settings', None))
    vol = bool(vol)
    sbr_tf = sbr_tf_norm(sbr_tf)
    sbr_eff = tf if sbr_tf == SBR_CHART else sbr_tf       # TF, на якому рахуємо SBR
    # 🏁 High/Low — свій TF кабінету (24.11); без вибору — як у боті
    _shl_tf = norm_tf(shl_tfv, HL_TFS, shl_tf(sc))
    # ⚠ сигнали «Аномалії обсягу» на графіку (24.11): монету веде рушій ШІ, поки
    # графік відкрито з увімкненим шаром
    anom = bool(anom)
    if anom:
        anom_chart_want(sym)
    ck = (sym, tf, ob_tf, vob_tf, poc_days, poc_tf, hl_tf, _vp['on'], _vp['pk'], vol, sbr_eff,
          _shl_tf, anom)
    now = time.time()
    with _lock:
        hit = _coin_cache.get(ck)
    if hit and now - hit[0] < COIN_TTL:
        # частини все одно «просимо», інакше кеш потоку їх забуде
        for k, t in (('bars', tf), ('ob', ob_tf), ('vob', vob_tf if _vp['on'] else OFF),
                     ('shl', _shl_tf), ('imb', tf),
                     ('hl', hl_tf), ('vol', tf if vol else OFF),
                     ('sbr', sbr_eff if sbr_eff != tf else OFF)):
            if t != OFF and _bot_part(sc, k, sym, t) is None:
                want((k, sym, t))
        for t in SBR_TFS:
            if _engine_sbr_last(sym, t) is None:
                want(('sbrlast', sym, t))
        poc_want(sym, poc_days, poc_tf)
        return hit[1]
    bars, err = part(sc, 'bars', sym, tf)
    ohlc = ohlc_of((bars or {}).get('kl'))
    events = list((bars or {}).get('events') or [])
    ob, ob_wait = [], False
    if ob_tf != OFF:
        d, _e = part(sc, 'ob', sym, ob_tf)
        ob, ob_wait = ob_boxes((d or {}).get('zones')), d is None
    vob, vob_wait = [], False
    vp = info_vob_params(getattr(sc, '_settings', None))
    if vob_tf != OFF and vp['on']:
        d, _e = part(sc, 'vob', sym, vob_tf)
        if d is not None and d.get('pk') != vp['pk']:
            # пораховано ІНШИМИ параметрами (їх щойно змінили) → перерахунок
            with _lock:
                _comp.pop(('vob', sym, vob_tf), None)
            want(('vob', sym, vob_tf))
            d = None
        vob, vob_wait = vob_fresh(vob_boxes((d or {}).get('vz')), ohlc), d is None
    hl, hl_wait = [], False
    if hl_tf != OFF:
        d, _e = part(sc, 'hl', sym, hl_tf)
        hl, hl_wait = list((d or {}).get('pts') or []), d is None
    if sbr_eff == tf:
        sbr, sbr_wait = sbr_of((bars or {}).get('kl'), tf, now), False
    else:
        d, _e = part(sc, 'sbr', sym, sbr_eff)
        sbr, sbr_wait = list((d or {}).get('sigs') or []), d is None and not _e
    vsplit, vol_wait = None, False
    if vol and ohlc:
        d, _e = part(sc, 'vol', sym, tf)
        vsplit = vol_view(d, ohlc)
        vol_wait = vsplit is None and not _e
    if ohlc:
        reason = ''
    elif err:
        reason = err
    elif in_bot:
        reason = '⏳ завантаження барів ' + tf.upper()
    else:
        reason = '⏳ монети немає в боті — рахуємо окремо, кілька секунд'
    _shd, _e = part(sc, 'shl', sym, _shl_tf)
    shl = shl_view((_shd or {}).get('levels'), _shl_tf)
    sbr_last = {}
    for t in SBR_TFS:
        _sd, _e = part(sc, 'sbrlast', sym, t)
        # None = ще рахується · {'none': True} = сигналів немає · інакше сигнал
        sbr_last[t] = None if _sd is None else (_sd.get('sig') or {'none': True})
    poc_want(sym, poc_days, poc_tf)
    _st, poc = _poc(sc, sym, allow_compute=False,
                    price=(ohlc[-1]['close'] if ohlc else None),
                    days=poc_days, tf=poc_tf)
    payload = {
        'ok': bool(ohlc), 'symbol': sym, 'src': 'bot' if in_bot else 'ext', 'reason': reason,
        'tf': tf, 'ob_tf': ob_tf, 'vob_tf': vob_tf, 'hl_tf': hl_tf, 'hl': hl, 'hl_wait': hl_wait,
        'poc_days': poc_days, 'poc_tf': poc_tf,
        'ohlc': ohlc, 'events': events,
        'imb': imb_of((bars or {}).get('kl'), tf, now, sym=sym,
                      hist=(part(sc, 'imb', sym, tf)[0] or {}).get('kl')),
        'sbr': sbr, 'sbr_tf': sbr_tf, 'sbr_eff': sbr_eff, 'sbr_wait': sbr_wait,
        'sbr_last': sbr_last,
        'ob': ob, 'vob': vob, 'ob_wait': ob_wait, 'vob_wait': vob_wait,
        'pd': _pd(sc, sym), 'poc': poc, 'poc_state': _st, 'shl': shl, 'ts': now,
        'vsplit': vsplit, 'vol_wait': vol_wait, 'shl_tf': _shl_tf,
        'anom_sig': anom_marks_of(sym, ohlc) if anom else [],
    }
    payload['sig'] = sig_of(payload)
    if not ohlc or ob_wait or vob_wait or hl_wait or vol_wait or sbr_wait:
        return payload                 # «рахується» не кешуємо
    with _lock:
        _coin_cache[ck] = (now, payload)
        if len(_coin_cache) > 1200:
            _coin_cache.clear()
    return payload


# ── Особисті дані користувача ─────────────────────────────────────────────
def sbr_norm(raw, wl=None):
    """ЧИСТА: {SYM: ['1h','4h']} — персональні 🔔 SBR-підписки. Лише монети
    списку (якщо `wl` задано), лише TF із SBR_SUB_TFS (15M…1D); порожній вибір
    = немає підписки."""
    out = {}
    for k, v in (raw or {}).items() if isinstance(raw, dict) else []:
        S = norm_symbol(k)
        if not valid_symbol(S) or (wl is not None and S not in wl):
            continue
        tfs = [t for t in SBR_SUB_TFS if t in [str(x).lower() for x in (v or [])]]
        if tfs:
            out[S] = tfs
    return out


def anom_norm(raw, wl=None):
    """ЧИСТА: [SYM…] — особисті ⚠ сповіщення «Аномалії обсягу» (вимога 18.11).
    Лише валідні монети списку (якщо `wl` задано), без дублів."""
    out = []
    for k in (raw or []) if isinstance(raw, (list, tuple)) else []:
        S = norm_symbol(k)
        if valid_symbol(S) and (wl is None or S in wl) and S not in out:
            out.append(S)
    return out


def user_info(prefs, uni, dfl=None):
    dfl = dfl or {'tf': '15m', 'ob_tf': '1h', 'vob_tf': '5m'}
    info = (prefs or {}).get('info') or {}
    wl = info.get('wl')
    if wl is None:                       # перший вхід — стартовий набір
        wl = list(IWL_DEFAULT)
    wl = iwl_norm(wl)
    sel = norm_symbol(info.get('sel'))
    if sel not in wl:
        sel = wl[0] if wl else ''
    return {'wl': wl, 'sel': sel,
            'tf': norm_tf(info.get('tf'), CHART_TFS, dfl['tf']),
            'ob_tf': norm_tf(info.get('ob_tf'), OB_TFS, dfl['ob_tf'], allow_off=True),
            'vob_tf': norm_tf(info.get('vob_tf'), VOB_TFS, dfl['vob_tf'], allow_off=True),
            'hl_tf': norm_tf(info.get('hl_tf'), HL_TFS, OFF, allow_off=True),
            'sbr_tf': sbr_tf_norm(info.get('sbr_tf')),
            # що малювати на графіку (лише показ, рахується завжди)
            'show_imb': bool(info.get('show_imb', True)),
            'show_sbr': bool(info.get('show_sbr', True)),
            'show_shl': bool(info.get('show_shl', True)),
            'shl_tf': norm_tf(info.get('shl_tf'), HL_TFS, '1h'),
            # ⚠ сигнали «Аномалії обсягу» на графіку (24.11, деф. вимк.)
            'show_anom': bool(info.get('show_anom', False)),
            # ⚪ POC: вікно (днів) і TF — свої для кабінету (деф. 7 днів · 1H)
            'poc_days': poc_norm(info.get('poc_days'), info.get('poc_tf'))[0],
            'poc_tf': poc_norm(info.get('poc_days'), info.get('poc_tf'))[1],
            # 🔔 особисті SBR-сповіщення в Telegram (вимога 09.11)
            'sbr': sbr_norm(info.get('sbr'), wl),
            # ⚠ особисті сповіщення «Аномалії обсягу» (вимога 18.11)
            'anom': anom_norm(info.get('anom'), wl),
            # ⚠ вигляд повідомлення: коротке (деф.) | повне (вимога 24.11)
            'anom_fmt': anom_fmt_norm(info.get('anom_fmt'))}


ANOM_FMTS = ('short', 'full')


def anom_fmt_norm(v):
    """ЧИСТА: формат ⚠ сповіщення — 'short' (деф., лише головне) | 'full'."""
    v = str(v or '').strip().lower()
    return v if v in ANOM_FMTS else 'short'


def apply_change(cur, body, uni):
    """ЧИСТА: зміна особистих даних з тіла запиту {add|remove|set|sel|tf|ob_tf|vob_tf}."""
    out = dict(cur)
    wl = list(cur.get('wl') or [])
    sel = cur.get('sel') or ''
    err = ''
    if 'set' in body:
        wl = iwl_norm(body.get('set') or [])
    if body.get('add'):
        s = norm_symbol(body['add'])
        if not valid_symbol(s):
            err = (s or "?") + ' — некоректна назва (приклад: BTC або BTCUSDT)'
        elif s not in wl:
            if len(wl) >= IWL_MAX:
                err = f'максимум {IWL_MAX} монет'
            else:
                wl.append(s)
                sel = s
    if body.get('remove'):
        s = norm_symbol(body['remove'])
        wl = [x for x in wl if x != s]
    if 'sel' in body:
        s = norm_symbol(body.get('sel'))
        if s in wl:
            sel = s
    if sel not in wl:
        sel = wl[0] if wl else ''
    if 'tf' in body:
        out['tf'] = norm_tf(body.get('tf'), CHART_TFS, cur.get('tf') or '15m')
    if 'ob_tf' in body:
        out['ob_tf'] = norm_tf(body.get('ob_tf'), OB_TFS, cur.get('ob_tf') or '1h', allow_off=True)
    if 'vob_tf' in body:
        out['vob_tf'] = norm_tf(body.get('vob_tf'), VOB_TFS, cur.get('vob_tf') or '5m', allow_off=True)
    if 'hl_tf' in body:
        out['hl_tf'] = norm_tf(body.get('hl_tf'), HL_TFS, cur.get('hl_tf') or OFF, allow_off=True)
    if 'sbr_tf' in body:
        out['sbr_tf'] = sbr_tf_norm(body.get('sbr_tf'))
    if 'shl_tf' in body:
        out['shl_tf'] = norm_tf(body.get('shl_tf'), HL_TFS, cur.get('shl_tf') or '1h')
    for k in ('show_imb', 'show_sbr', 'show_shl', 'show_anom'):
        if k in body:
            out[k] = bool(body.get(k))
    if 'poc_days' in body or 'poc_tf' in body:
        out['poc_days'], out['poc_tf'] = poc_norm(
            body.get('poc_days', cur.get('poc_days')), body.get('poc_tf', cur.get('poc_tf')))
    sbr = dict(cur.get('sbr') or {})
    if isinstance(body.get('sbr'), dict):        # {symbol, tfs: ['1h','4h']}
        S = norm_symbol(body['sbr'].get('symbol'))
        tfs = sbr_norm({S: body['sbr'].get('tfs') or []}).get(S)
        if S in wl:
            if tfs:
                sbr[S] = tfs
            else:
                sbr.pop(S, None)
    out['sbr'] = sbr_norm(sbr, wl)               # прибрана монета → і підписка
    anom = list(cur.get('anom') or [])
    if isinstance(body.get('anom'), dict):       # {symbol, on}
        S = norm_symbol(body['anom'].get('symbol'))
        if S in wl:
            anom = [x for x in anom if x != S] + ([S] if body['anom'].get('on') else [])
    out['anom'] = anom_norm(anom, wl)
    if 'anom_fmt' in body:
        out['anom_fmt'] = anom_fmt_norm(body.get('anom_fmt'))
    out['wl'], out['sel'] = wl, sel
    return out, err


# ── 🔔 Особисті SBR-сповіщення (вимога 09.11) ────────────────────────────
# Користувач у кабінеті вмикає 🔔 біля монети й обирає 1H / 4H → сигнал іде
# ЙОМУ в особистий чат із головним ботом (не в групу). Розрахунок — той самий
# рушій `sbr_signals` (монети підписок додаються в його набір), тож сигнал у
# кабінеті, у групі і в особистому чаті — один і той самий.
SUBS_TTL = 30.0
SUBS_EXTRA_MAX = 300           # стеля монет поза WATCHLIST бота, які веде рушій
ANOM_SUB_MAX = 60              # стеля монет ⚠ сповіщень (рушій ШІ тягне їх 1h-бари)
_subs = {'at': 0.0, 'data': []}
_subs_lock = threading.Lock()


def sbr_subs(force=False):
    """[(chat_id, {SYM: [tfs]})] активних користувачів (кеш 30с)."""
    now = time.time()
    with _subs_lock:
        if not force and now - _subs['at'] < SUBS_TTL:
            return _subs['data']
    try:
        from web.auth import info_sbr_subscribers
        data = info_sbr_subscribers()
    except Exception:
        data = []
    with _subs_lock:
        _subs['at'], _subs['data'] = now, data
    return data


def sbr_subs_invalidate():
    with _subs_lock:
        _subs['at'] = 0.0


def sbr_extra_symbols():
    """Монети особистих підписок на 1H/4H — рушій SBR рахує їх навіть поза
    WATCHLIST бота. Підписка лише на інші TF (15M…1D) сюди не потрапляє: її
    веде `sbr_extra_tfs`, і 1H/4H по такій монеті не тягнуться даремно."""
    out = []
    for _c, sub in sbr_subs():
        for S, tfs in sbr_norm(sub).items():
            if S not in out and any(t in SBR_TFS for t in tfs):
                out.append(S)
    return out[:SUBS_EXTRA_MAX]


def sbr_extra_tfs():
    """{tf: [монети]} особистих підписок на ДОДАТКОВІ TF (15M/30M/45M/1D) —
    рушій рахує ці TF лише для цих монет (вимога 17.11)."""
    return sbr_tfs_map(sbr_subs())


def sbr_tfs_map(subs):
    """ЧИСТА: [(chat, {SYM: tfs})] → {tf: [SYM…]} для TF поза 1H/4H."""
    out = {}
    n = 0
    for _c, sub in subs or []:
        for S, tfs in sbr_norm(sub).items():
            for t in tfs:
                if t in SBR_TFS:
                    continue
                lst = out.setdefault(t, [])
                if S not in lst and n < SUBS_EXTRA_MAX:
                    lst.append(S)
                    n += 1
    return out


def sbr_recipients(subs, symbol, items):
    """ЧИСТА: {chat_id: [(tf, сигнал)…]} — хто що отримує з цих сигналів."""
    S = norm_symbol(symbol)
    out = {}
    for chat, sub in subs or []:
        tfs = sbr_norm(sub).get(S) or []
        got = [(tf, sig) for tf, sig in items if tf in tfs]
        if got:
            out[str(chat)] = got
    return out


def sbr_personal_send(symbol, items):
    """Розіслати свіжі сигнали монети підписникам. → (надіслано, помилок)."""
    rc = sbr_recipients(sbr_subs(), symbol, items)
    if not rc:
        return 0, 0
    try:
        from detection.sbr_signals import tg_text
        from web.tg_bot import tg_send
    except Exception:
        return 0, len(rc)
    ok = bad = 0
    for chat, got in rc.items():
        by_side = {}
        for tf, sig in got:
            by_side.setdefault(sig.get('side'), []).append((tf, sig))
        for grp in by_side.values():
            try:
                if tg_send(chat, '👤 <i>Ваш сигнал з кабінету</i>\n' + tg_text(norm_symbol(symbol), grp)):
                    ok += 1
                else:
                    bad += 1
            except Exception:
                bad += 1
    return ok, bad


# ── ⚠ Особисті сповіщення «Аномалії обсягу» (вимога 18.11) ──────────────
# Користувач у кабінеті вмикає ⚠ біля монети → коли по ній СКЛАВСЯ сценарій
# входу ШІ (комплекс реакцій обсягу, точка свіжа), повідомлення йде ЙОМУ в
# особистий чат. Рахує рушій ШІ (`ai_signals`), монети підписок ведуть монітор
# аномалій і рушій навіть поза Пружинами.
_asubs = {'at': 0.0, 'data': []}


def anom_subs(force=False):
    """[(chat_id, [SYM…])] активних користувачів (кеш 30с)."""
    now = time.time()
    with _subs_lock:
        if not force and now - _asubs['at'] < SUBS_TTL:
            return _asubs['data']
    try:
        from web.auth import info_anom_subscribers
        data = info_anom_subscribers()
    except Exception:
        data = []
    with _subs_lock:
        _asubs['at'], _asubs['data'] = now, data
    return data


def anom_subs_invalidate():
    with _subs_lock:
        _asubs['at'] = 0.0


def anom_symbols(subs=None):
    """ЧИСТА над підписками: монети ⚠ сповіщень (без дублів, стеля)."""
    out = []
    for x in (anom_subs() if subs is None else subs) or []:
        for S in anom_norm(x[1]):
            if S not in out:
                out.append(S)
    return out[:ANOM_SUB_MAX]


def anom_recipients(subs, symbol):
    """ЧИСТА: [chat_id…] — хто підписаний на ⚠ по цій монеті."""
    return [c for c, _f in anom_recipients_fmt(subs, symbol)]


def anom_recipients_fmt(subs, symbol):
    """ЧИСТА: [(chat_id, формат)…] — підписники монети з їхнім форматом
    повідомлення (запис без формату = коротке)."""
    S = norm_symbol(symbol)
    out = []
    for x in subs or []:
        c, sub = x[0], x[1]
        if S in anom_norm(sub):
            out.append((str(c), anom_fmt_norm(x[2] if len(x) > 2 else None)))
    return out


def anom_text_for(text, fmt):
    """ЧИСТА: текст під формат: `text` — рядок (один на всіх) або
    {'short': …, 'full': …}. Повне несе підпис «з кабінету»."""
    if isinstance(text, dict):
        if fmt == 'full':
            return '👤 <i>Ваш сигнал з кабінету</i>\n' + (text.get('full') or text.get('short') or '')
        return text.get('short') or text.get('full') or ''
    return '👤 <i>Ваш сигнал з кабінету</i>\n' + str(text or '')


def anom_personal_send(symbol, text):
    """Розіслати ⚠ сигнал монети підписникам (кожному — у його форматі).
    → (надіслано, помилок)."""
    rc = anom_recipients_fmt(anom_subs(), symbol)
    if not rc:
        return 0, 0
    try:
        from web.tg_bot import tg_send
    except Exception:
        return 0, len(rc)
    ok = bad = 0
    for chat, fmt in rc:
        try:
            if tg_send(chat, anom_text_for(text, fmt)):
                ok += 1
            else:
                bad += 1
        except Exception:
            bad += 1
    return ok, bad


# ── 🧪 ТЕСТОВІ (paper) угоди бота — лише читання (вимога 08.11) ─────────
PAPER_TTL = 2.0                 # спільний кеш знімка (сторінки опитують часто)
PAPER_CLOSED_SHOW = 50          # скільки останніх закритих paper-угод віддаємо
_PAPER_POS_KEYS = ('symbol', 'side', 'entry_price', 'current_price', 'pnl_pct',
                   'pnl_net', 'paper_size', 'opened_at', 'opened_by', 'manual_sl',
                   'manual_tp', 'manual_tp1', 'tp1_done', 'spring_open', 'mmd',
                   'spring_fire', 'exh_trail', 'exh', 'spring_open_tgt',
                   # 15.11: ті самі поля, що таблиця бота (Side, 📌, 🚨-трейл, SL-TF)
                   'ob4h', 'ob1h', 'manual_mode', 'manual_sl_tf', 'manual_sl_src',
                   'sl_breakeven', 'sbr_trail',
                   # ➕ усереднення (24.11)
                   'cab_avg_n', 'cab_tranches')
_PAPER_CLOSED_KEYS = ('symbol', 'side', 'entry_price', 'exit_price', 'opened_at',
                      'closed_at', 'pnl_pct', 'reason', 'opened_by', 'peak_pnl_pct',
                      'was_manual', 'cab_avg_n')
_paper_cache = {'at': 0.0, 'data': None, 'sig': None}


def paper_pos_view(p, spring=None):
    """ЧИСТА: відкрита paper-позиція → поля для кабінету (без службових)."""
    out = {k: p.get(k) for k in _PAPER_POS_KEYS if p.get(k) is not None}
    if spring:
        out['spring'] = {k: spring.get(k) for k in ('charge', 'state', 'label', 'dir', 'restored')}
    return out


def paper_closed_view(c):
    """ЧИСТА: закрита paper-угода → поля кабінету + віртуальний нетто $."""
    out = {k: c.get(k) for k in _PAPER_CLOSED_KEYS if c.get(k) is not None}
    pr = c.get('paper_result') or {}
    if pr.get('net_usd') is not None:
        out['net_usd'] = pr.get('net_usd')
        out['roi_pct'] = pr.get('roi_margin_pct')
    return out


def paper_book_sig(tm):
    """Дешевий підпис paper-книги: склад відкритих і закритих угод. Будь-яке
    видалення / закриття / відкриття міняє його → кеш знімка не віддає
    угоду, якої в боті вже немає (кейс 08.11: видалена угода ще висіла)."""
    try:
        cl = list(getattr(tm, '_shadow_closed', None) or [])
        op = dict(getattr(tm, '_shadow_positions', None) or {})
        last = cl[-1] if cl else {}
        return (len(cl), last.get('symbol'), last.get('closed_at'),
                tuple(sorted((k, (v or {}).get('opened_at')) for k, v in op.items())))
    except Exception:
        return None


def _paper_snapshot(tm):
    """Знімок paper-книги з TM (ТІ САМІ числа, що таблиця бота) + кеш.
    Кеш живе ≤ PAPER_TTL І лише поки склад книги не змінився."""
    now = time.time()
    sig = paper_book_sig(tm)
    with _lock:
        if (_paper_cache['data'] and now - _paper_cache['at'] < PAPER_TTL
                and sig is not None and sig == _paper_cache['sig']):
            return _paper_cache['data']
    st = tm.get_state()
    pos = st.get('shadow_positions') or []
    spr = {}
    try:
        sc = _scanner()
        if sc is not None and hasattr(sc, 'spring_for'):
            spr = sc.spring_for([p.get('symbol') for p in pos]) or {}
    except Exception:
        spr = {}
    # 🏷 OB 4H / 1H у полі Side — та сама памʼять сканера, що в таблиці бота.
    try:
        sc = _scanner()
        if sc is not None and hasattr(sc, 'ob_mark_for'):
            syms = [p.get('symbol') for p in pos]
            m4 = sc.ob_mark_for(syms) or {}
            try:
                m1 = sc.ob_mark_for(syms, tf='1h') or {}
            except TypeError:
                m1 = {}
            for p in pos:
                S = str(p.get('symbol') or '').upper()
                if m4.get(S):
                    p['ob4h'] = m4[S]
                if m1.get(S):
                    p['ob1h'] = m1[S]
    except Exception:
        pass
    # 🚨 Сигнал угоди (той самий `dec_trade_for`, що колонка 🚨 у боті).
    try:
        from detection.fuel_filter import get_fuel_filter, mmd_trade_key
        ff = get_fuel_filter()
        dm = ff.dec_trade_for([(p.get('symbol'), p.get('side'), p.get('opened_at'))
                               for p in pos]) if ff else {}
        for p in pos:
            k = mmd_trade_key(p.get('symbol'), p.get('side'), p.get('opened_at'))
            if k in (dm or {}):
                p['mmd'] = dm[k]
    except Exception:
        pass
    # 💥 Виснаження (процес вистрілу) — ЛИШЕ кеш TM, той самий, що колонка бота.
    for p in pos:
        try:
            sf = tm._spring_fire_view(p.get('symbol'), fetch=False)
        except Exception:
            sf = None
        if sf:
            p['spring_fire'] = sf
        try:
            ex = tm.trade_exh_for(p, sf)
        except Exception:
            ex = None
        if ex:
            p['exh'] = ex
    data = {
        'account': st.get('paper_account') or {},
        'stats_all': st.get('shadow_stats') or {},
        'positions': [paper_pos_view(p, spr.get(str(p.get('symbol') or '').upper())) for p in pos],
        'closed': [paper_closed_view(c) for c in (st.get('shadow_closed') or [])[-PAPER_CLOSED_SHOW:]][::-1],
        'test_mode': bool((st.get('settings') or {}).get('test_mode', True)),
    }
    with _lock:
        _paper_cache.update(at=now, data=data, sig=sig)
    return data


# ── 💧 ПРОФІЛЬ ЛІКВІДНОСТІ на графіку кабінету (вимога 09.11) ──────────────
# Розрахунок — ТОЙ САМИЙ `liq_scan.scan_one`, що режим однієї монети на
# 📡 Tickr (драбина + вердикт); відрізняється лише дрібніший крок сходинок,
# щоб профіль на графіку мав «рядки», як Volume Profile у TradingView.
# Рахується ЛИШЕ за кнопкою (через спільну чергу сканів біржі), спільний кеш.
LIQ_EXCHANGES = ('binance', 'bybit')
LIQ_BARS = (24, 48, 72, 120, 168, 240, 336, 504, 720, 1000)
LIQ_BARS_DEFAULT = 168
LIQ_RES = {'fine': 0.0025, 'mid': 0.005, 'coarse': 0.0125}
LIQ_RES_DEFAULT = 'mid'
LIQ_ROWS = 120
LIQ_MIN_ROW_PCT = 0.3        # профілю потрібні й дрібні сходинки (драбина Tickr — 1%)
LIQ_TTL = 120.0
# ⚖ НАВАНТАЖЕННЯ БІРЖІ (вимога 09.11). Запити кабінету йдуть з IP СЕРВЕРА —
# того самого, яким живе бот. Тому:
#  1) СИРІ дані (ціна · OI · свічки → рівні) кешуються на (монета, біржа,
#     глибина) — крок сходинок рахується з них БЕЗ мережі, тож сто кліків по
#     BTC з різним масштабом = один запит на `LIQ_TTL`;
#  2) кожен користувач може змусити сервер піти на біржу не частіше ніж раз
#     на `LIQ_USER_COOLDOWN` (кеш — без обмежень);
#  3) біржа відповіла лімітом / блоком IP → ПАУЗА всім запитам кабінету до
#     неї на `LIQ_PAUSE_SEC`, щоб не з'їсти ліміт бота;
#  4) основний шлях — БРАУЗЕР користувача тягне сирі дані сам (його IP) і
#     надсилає їх на розрахунок (`liq_from_client`); сервер — лише фолбек.
LIQ_USER_COOLDOWN = 30.0
LIQ_PAUSE_SEC = 300.0
LIQ_CALC_COOLDOWN = 2.0       # клієнтський шлях: лише арифметика (~10 мс)
LIQ_CLIENT_MAX_BARS = 1000
_liq_cache = {}               # (sym, ex, bars) → (ts, сирі дані + рівні)
_liq_user_at = {}             # користувач → час останнього походу на біржу
_liq_calc_at = {}             # користувач → час останнього розрахунку з браузера
_liq_pause = {}               # біржа → до коли пауза
_liq_lock = threading.Lock()


def liq_params(exchange=None, bars=None, res=None):
    """Нормалізувати параметри профілю: невідоме → дефолт."""
    ex = (exchange or '').lower()
    ex = ex if ex in LIQ_EXCHANGES else LIQ_EXCHANGES[0]
    try:
        b = int(bars)
    except (TypeError, ValueError):
        b = LIQ_BARS_DEFAULT
    b = b if b in LIQ_BARS else LIQ_BARS_DEFAULT
    r = res if res in LIQ_RES else LIQ_RES_DEFAULT
    return ex, b, r


def liq_view(out, sym, ex, bars, res):
    """Те, що потрібно графіку: сходинки + підсумок + вердикт (без зайвого)."""
    out = out or {}
    if not out.get('ok'):
        return {'ok': False, 'symbol': sym, 'exchange': ex, 'bars': bars, 'res': res,
                'reason': out.get('reason') or 'немає даних'}
    rows = []
    for r in out.get('rows') or []:
        try:
            lo, hi = float(r['price']), float(r.get('price_hi') or r['price'])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({'lo': lo, 'hi': hi, 'usd': float(r.get('usd') or 0),
                     'pct': float(r.get('pct') or 0), 'dir': r.get('dir'),
                     'dist': r.get('dist_pct')})
    # 🧲 магніт — найбільша сходинка (та сама, що «найбільший магніт» у
    # вердикті Tickr); сирі числа з `magnet_row`, а не форматований підпис.
    mag = None
    m = out.get('magnet_row') or {}
    try:
        if m.get('price') is not None:
            lo = float(m['price'])
            mag = {'lo': lo, 'hi': float(m.get('price_hi') or lo),
                   'pct': float(m.get('pct') or 0), 'usd': float(m.get('usd') or 0),
                   'dist': m.get('dist_pct'), 'dir': m.get('dir')}
    except (TypeError, ValueError):
        mag = None
    return {'ok': True, 'symbol': sym, 'exchange': out.get('exchange') or ex,
            'magnet': mag,
            'bars': bars, 'res': res, 'price': out.get('price'),
            'step': out.get('step'), 'total_usd': out.get('total_usd'),
            'above_pct': out.get('above_pct'), 'below_pct': out.get('below_pct'),
            'pull': out.get('pull'), 'verdict': out.get('verdict'),
            'verdict_parts': out.get('verdict_parts'),
            'oi_usd': out.get('oi_usd'), 'took_sec': out.get('took_sec'),
            'at': out.get('fetched_at') or time.time(), 'rows': rows}


def liq_step(v):
    """Явний крок сходинки (у ціні) від сторінки: число > 0, інакше None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return float('%.6g' % f) if f > 0 and f == f and f != float('inf') else None


def liq_pause_left(ex, now=None):
    """Скільки секунд ще триває пауза запитів кабінету до біржі (0 — немає)."""
    now = time.time() if now is None else now
    return max(0.0, _liq_pause.get(ex, 0.0) - now)


def _liq_render(raw, levels, sym, ex, b, r, st, now, src):
    """Драбина під потрібний крок із сирих даних — БЕЗ мережі."""
    from detection import liq_scan
    out = liq_scan.summarise_raw(raw, rows=LIQ_ROWS, step_pct=LIQ_RES[r],
                                 step_usd=st, min_row_pct=LIQ_MIN_ROW_PCT,
                                 levels=levels)
    v = liq_view(out, sym, ex, b, r)
    if v.get('ok'):
        v['src'] = src
        v['age'] = max(0, int(now - (raw.get('fetched_at') or now)))
    return v


def liq_profile(symbol, exchange=None, bars=None, res=None, now=None, runner=None,
                step=None, user=None):
    """Профіль ліквідності монети з сервера (фолбек браузерного шляху).

    Кеш — СИРІ дані на (монета, біржа, глибина), `LIQ_TTL`; крок (`step` з
    видимого діапазону графіка або `res` — % ціни) застосовується до них без
    нового запиту. Похід на біржу — не частіше `LIQ_USER_COOLDOWN` на
    користувача; блок біржі → пауза `LIQ_PAUSE_SEC`."""
    sym = norm_symbol(symbol)
    if not valid_symbol(sym):
        return {'ok': False, 'reason': 'невірна назва монети'}
    ex, b, r = liq_params(exchange, bars, res)
    st = liq_step(step)
    now = time.time() if now is None else now
    key = (sym, ex, b)
    hit = _liq_cache.get(key)
    if not (hit and now - hit[0] < LIQ_TTL):
        left = liq_pause_left(ex, now)
        if left > 0:
            return {'ok': False, 'paused': int(left) + 1,
                    'reason': f'{ex.upper()} тимчасово обмежила запити — '
                              f'пауза ще {int(left) + 1} с'}
        uk = str(user) if user is not None else None
        with _liq_lock:
            last = _liq_user_at.get(uk, 0.0) if uk else 0.0
            if uk and now - last < LIQ_USER_COOLDOWN:
                w = int(LIQ_USER_COOLDOWN - (now - last)) + 1
                return {'ok': False, 'retry_in': w,
                        'reason': f'зачекайте {w} с — запит до біржі не частіше '
                                  f'ніж раз на {int(LIQ_USER_COOLDOWN)} с'}
            if uk:
                _liq_user_at[uk] = now
                if len(_liq_user_at) > 2000:
                    for k in sorted(_liq_user_at, key=_liq_user_at.get)[:500]:
                        _liq_user_at.pop(k, None)
        if runner is None:
            from detection import liq_scan, scan_queue

            def runner(sym, ex, b):
                # Ім'я завдання БЕЗ кроку → однакові запити зливаються в черзі.
                return scan_queue.run(
                    f'info:liq:{sym}:{ex}:{b}',
                    lambda: liq_scan.fetch_one_raw(exchange=ex, symbol=sym, bars=b),
                    source='info', timeout=180.0)
        try:
            raw = runner(sym, ex, b)
        except Exception as e:
            raw = {'ok': False, 'reason': f'помилка: {str(e)[:100]}'}
        raw = raw or {}
        if raw.get('blocked'):
            _liq_pause[ex] = now + LIQ_PAUSE_SEC
            print(f'[INFO] 💧 {ex} відмовила ({raw.get("blocked")}) — пауза '
                  f'запитів кабінету {int(LIQ_PAUSE_SEC)} с')
            return {'ok': False, 'paused': int(LIQ_PAUSE_SEC),
                    'reason': f'{ex.upper()} обмежила запити — пауза '
                              f'{int(LIQ_PAUSE_SEC // 60)} хв'}
        if not raw.get('ok'):
            return liq_view(raw, sym, ex, b, r)
        from detection import liq_scan
        lv = liq_scan.build_levels(raw.get('klines') or [], raw['oi_usd'],
                                   raw['price'], symbol=sym)
        hit = (now, raw, lv)
        _liq_cache[key] = hit
        if len(_liq_cache) > 200:
            for k in sorted(_liq_cache, key=lambda k: _liq_cache[k][0])[:50]:
                _liq_cache.pop(k, None)
    return _liq_render(hit[1], hit[2], sym, ex, b, r, st, now, 'server')


def liq_client_raw(body):
    """Сирі дані, які браузер узяв у біржі сам → формат `fetch_one_raw`.

    Перевіряємо форму й межі; чужого кешу НЕ наповнюємо (дані від клієнта
    бачить лише він сам)."""
    body = body or {}
    try:
        price = float(body.get('price'))
        oi = float(body.get('oi_usd'))
    except (TypeError, ValueError):
        return None, 'немає ціни / OI'
    if not (price > 0 and oi > 0) or price != price or oi != oi \
            or price == float('inf') or oi == float('inf'):
        return None, 'невірні ціна / OI'
    kl = body.get('klines')
    if not isinstance(kl, list) or not (24 <= len(kl) <= LIQ_CLIENT_MAX_BARS):
        return None, 'невірна кількість свічок'
    bars = []
    for x in kl:
        if not isinstance(x, (list, tuple)) or len(x) < 4:
            continue
        try:
            h, l, c, v = (float(x[0]), float(x[1]), float(x[2]), float(x[3]))
        except (TypeError, ValueError):
            continue
        bars.append({'h': h, 'l': l, 'c': c, 'v': v})
    if len(bars) < 24:
        return None, 'замало свічок'
    return {'ok': True, 'price': price, 'oi_usd': oi, 'klines': bars,
            'took_sec': None, 'fetched_at': time.time()}, ''


def liq_from_client(body, user=None, now=None):
    """💧 Профіль із даних, завантажених БРАУЗЕРОМ (IP користувача)."""
    body = body or {}
    sym = norm_symbol(body.get('symbol') or '')
    if not valid_symbol(sym):
        return {'ok': False, 'reason': 'невірна назва монети'}
    ex, b, r = liq_params(body.get('exchange'), body.get('bars'), body.get('res'))
    now = time.time() if now is None else now
    uk = str(user) if user is not None else None
    if uk:
        with _liq_lock:
            if now - _liq_calc_at.get(uk, 0.0) < LIQ_CALC_COOLDOWN:
                return {'ok': False, 'retry_in': int(LIQ_CALC_COOLDOWN) + 1,
                        'reason': 'занадто часто — зачекайте секунду'}
            _liq_calc_at[uk] = now
            if len(_liq_calc_at) > 2000:
                for k in sorted(_liq_calc_at, key=_liq_calc_at.get)[:500]:
                    _liq_calc_at.pop(k, None)
    raw, why = liq_client_raw(body)
    if raw is None:
        return {'ok': False, 'reason': why}
    raw.update({'exchange': ex, 'symbol': sym, 'bars': len(raw['klines'])})
    try:
        return _liq_render(raw, None, sym, ex, b, r, liq_step(body.get('step')),
                           now, 'browser')
    except Exception as e:
        return {'ok': False, 'reason': f'помилка: {str(e)[:100]}'}


# ═══ 🤖 «РЕКОМЕНДОВАНІ БОТОМ» (вимога 10.11) ════════════════════════════════
# Монети 🌀 Пружини (WATCHLIST), що ПРОХОДЯТЬ її ворота у бік своєї вкладки, +
# усі монети таблиці 🔭 «Пружина — скан біржі». Лише читання памʼяті бота.
REC_TTL = 10.0
REC_KEYS = ('state', 'label', 'charge', 'dir', 'lean', 'pot_long', 'pot_short',
            'prob_long', 'price', 'tgt_long', 'tgt_short', 'move_long_pct',
            'move_short_pct', 'box_pct', 'squeeze_bars', 'fired', 'bar_t',
            'charge_in')
_rec_cache = {}


def recommended_rows(wl_recs, wl_pass, scan_rows):
    """ЧИСТА: рядки «Рекомендовані ботом» → [{symbol, src, side, …}].

    `wl_pass` — {SYM: {ok, side}} (`SMCScanner.spring_pass_map`); зі скану —
    усі видимі рядки таблиці, бік = вкладка (`spring_gates.tab_side`).
    Монета, що є в обох, береться один раз (WATCHLIST сильніший)."""
    try:
        from detection import spring_gates as _sg
    except Exception:
        _sg = None
    out, seen = [], set()
    for S, rec in (wl_recs or {}).items():
        g = (wl_pass or {}).get(S) or {}
        if not g.get('ok') or g.get('side') not in ('LONG', 'SHORT'):
            continue
        row = {k: (rec or {}).get(k) for k in REC_KEYS}
        row.update({'symbol': S, 'src': 'wl', 'side': g['side']})
        out.append(row)
        seen.add(S)
    for r in scan_rows or []:
        S = str((r or {}).get('symbol') or '').upper()
        if not S or S in seen:
            continue
        side = _sg.tab_side(r) if _sg else (r.get('dir') or r.get('lean'))
        row = {k: r.get(k) for k in REC_KEYS}
        row.update({'symbol': S, 'src': 'scan', 'side': side if side in ('LONG', 'SHORT') else None,
                    'vol_usd': r.get('vol_usd'), 'bybit': r.get('bybit'),
                    'added_at': r.get('added_at')})
        out.append(row)
        seen.add(S)

    out.sort(key=lambda x: (-(rec_prob(x) if rec_prob(x) is not None else -1), x['symbol']))
    return out


def rec_ai_merge(rows, ai_map, ai_recs=None):
    """ЧИСТА: 🤖 ВИБІР ШІ для «Рекомендовані ботом» (вимога 22.11) — ті самі
    рішення, за якими ШІ відкриває СВОЇ угоди (`AiEngine.recommend`).
    • рядок, який ШІ оцінив, несе `ai` {ready, rank, side, scenario, sim, …};
    • готовий вхід ШІ задає бік рядка (ШІ вирішує), бік таблиці лишається в `tab_side`;
    • монета, яку ШІ вважає готовою, але якої немає в таблицях, додається (`src='ai'`);
    • порядок: готові входи ШІ (за рангом) → решта, оцінена ШІ (за рангом) → інші."""
    ai_map = ai_map or {}
    seen = set()
    for r in rows:
        S = r.get('symbol')
        seen.add(S)
        a = ai_map.get(S)
        if not a:
            continue
        r['ai'] = dict(a)
        if a.get('ready') and a.get('side') in ('LONG', 'SHORT'):
            if r.get('side') != a['side']:
                r['tab_side'] = r.get('side')
            r['side'] = a['side']
    for S, a in ai_map.items():
        if S in seen or not a.get('ready') or a.get('side') not in ('LONG', 'SHORT'):
            continue
        rec = (ai_recs or {}).get(S)
        if not rec:
            continue
        row = {k: rec.get(k) for k in REC_KEYS}
        row.update({'symbol': S, 'src': 'ai', 'side': a['side'], 'ai': dict(a)})
        rows.append(row)

    def _k(x):
        a = x.get('ai') or {}
        rk = a.get('rank') if isinstance(a.get('rank'), (int, float)) else -1
        pr = rec_prob(x)
        return (0 if a.get('ready') else (1 if a else 2), -rk if a else 0,
                -(pr if pr is not None else -1), x.get('symbol') or '')
    rows.sort(key=_k)
    return rows


def rec_ai_only(rows, scan_ex=None):
    """ЧИСТА (вимога 22.11): «VSV SIGNALS» = ЛИШЕ угоди, рекомендовані ШІ —
    готова точка входу (`ai.ready`). Кожен рядок отримує біржу джерела `ex`:
    🌀 WATCHLIST → bybit (бари бота), 🔭 скан → біржа скану."""
    sx = str(scan_ex or 'binance').lower()
    out = []
    for r in rows or []:
        a = r.get('ai') or {}
        if not a.get('ready') or r.get('side') not in ('LONG', 'SHORT'):
            continue
        src = r.get('src')
        if src == 'ai':
            src = 'scan' if a.get('src') == 'scan' else 'wl'
        r['ex'] = 'bybit' if src == 'wl' else sx
        out.append(r)
    return out


def rec_prob(row):
    """ЧИСТА: ймовірність у бік рядка (%), None — немає даних. Сортування
    таблиці «Рекомендовані ботом» — за нею (вимога 10.11)."""
    try:
        pl = float(row.get('prob_long'))
    except (TypeError, ValueError):
        return None
    side = row.get('side')
    if side == 'LONG':
        return pl
    if side == 'SHORT':
        return 100.0 - pl
    return None


# ─── 🛰 VSV SIGNALS — ВІДБІР ЗА ЕНЕРГІЄЮ СТИСНЕННЯ, БЕЗ НАПРЯМКУ (24.11) ──
# Вимога: «Відображай всі монети із обох пружин, але відбирай найкращий заряд і
# високу ймовірність відпрацювання. Напрямок не вказуй (він не має 100%
# гарантії) — це має бути інформаційного характеру.» Джерела — ті самі записи
# 🌀 Пружини WATCHLIST і рядки 🔭 скану біржі (лише памʼять); оцінка — ЧИСТА
# `vsv_signal_view`, відбір — ЧИСТА `vsv_signal_rows`. Сторона (LONG/SHORT,
# ймовірність боку, цілі, стопи) у відповідь НЕ потрапляє взагалі.
# 🧭 25.11: окремим полем `hint` — ЙМОВІРНИЙ напрямок (зведення 🌀 натягу ·
# ⚠ order flow · 🤖 сетапу · руху 24 год) з власною впевненістю моделі і
# ФАКТИЧНОЮ точністю з журналу перевірки (`vsv_direction`). Сирі поля Пружини
# (prob_long, tgt_*, …) у рядок і далі не потрапляють; SL / цілі — теж.
VSV_SHOW = 20                 # скільки монет показуємо (найвищий VSV Score)
VSV_MIN_CHARGE = 65.0         # «🌀 заряджена» — та сама межа, що в Пружині
VSV_SQUEEZE_FULL = 24         # годин стиснення, що дають повний бал
VSV_W = {'charge': 0.45, 'squeeze': 0.15, 'volat': 0.15, 'volume': 0.10, 'ai': 0.15}
VSV_GRADES = ((80.0, 'A', 'висока'), (65.0, 'B', 'підвищена'), (0.0, 'C', 'помірна'))
VSV_ROW_KEYS = ('state', 'label', 'phase', 'charge', 'charge_in', 'squeeze_bars',
                'box_pct', 'vol_ratio', 'atr_rank', 'price', 'bar_t', 'unit_r',
                'move24_pct')


def _vsv_f(x, d=None):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return d
    return v if v == v else d


def vsv_grade(score):
    """ЧИСТА: VSV Score → (клас, слово)."""
    sc = _vsv_f(score, 0.0)
    for lo, g, w in VSV_GRADES:
        if sc >= lo:
            return g, w
    return 'C', 'помірна'


def vsv_signal_view(rec, ai=None):
    """ЧИСТА: запис Пружини (+ вибір ШІ) → оцінка готовності до імпульсу БЕЗ
    напрямку: {score, grade, grade_ua, parts, amp_pct, impulse, ai_ready,
    ai_scenario, notes}. None — запису немає / не порахований."""
    if not isinstance(rec, dict) or rec.get('ok') is False or rec.get('charge') is None:
        return None
    ch = max(0.0, min(100.0, _vsv_f(rec.get('charge'), 0.0)))
    sq_b = max(0, int(_vsv_f(rec.get('squeeze_bars'), 0) or 0))
    rank = _vsv_f(rec.get('atr_rank'))
    vr = _vsv_f(rec.get('vol_ratio'))
    a = ai if isinstance(ai, dict) else {}
    ai_ready = bool(a.get('ready'))
    parts = {
        'charge': ch / 100.0,
        'squeeze': min(1.0, sq_b / float(VSV_SQUEEZE_FULL)),
        'volat': (max(0.0, min(1.0, 1.0 - rank / 100.0)) if rank is not None else 0.5),
        'volume': (max(0.0, min(1.0, (1.3 - vr) / 0.9)) if vr is not None else 0.5),
        'ai': 1.0 if ai_ready else 0.0,
    }
    score = round(100.0 * sum(VSV_W[k] * parts[k] for k in VSV_W), 1)
    g, gw = vsv_grade(score)
    impulse = (rec.get('state') == 'fired') or bool(rec.get('fired'))
    ml, ms = _vsv_f(rec.get('move_long_pct')), _vsv_f(rec.get('move_short_pct'))
    amps = [abs(x) for x in (ml, ms) if x is not None]
    amp = round(sum(amps) / len(amps), 2) if amps else None
    notes = []
    if impulse:
        notes.append('💥 вихід із діапазону вже почався (expansion)')
    if ch >= 80:
        notes.append(f'висока енергія стиснення ({ch:.0f})')
    elif ch >= VSV_MIN_CHARGE:
        notes.append(f'накопичення завершується ({ch:.0f})')
    if sq_b >= 12:
        notes.append(f'squeeze {sq_b} год')
    if rank is not None and rank <= 25:
        notes.append(f'ATR у нижніх {rank:.0f}% історії')
    if vr is not None and vr <= 0.75:
        notes.append(f'обсяг висох ×{vr:.2f}')
    if ai_ready:
        notes.append('🤖 VSV·AI бачить сетап' + (f' · {a.get("scenario_name")}' if a.get('scenario_name') else ''))
    return {'score': score, 'grade': g, 'grade_ua': gw,
            'parts': {k: round(v * 100.0, 1) for k, v in parts.items()},
            'amp_pct': amp, 'impulse': impulse, 'ai_ready': ai_ready,
            'ai_scenario': a.get('scenario_name') if ai_ready else None,
            'notes': notes}


def vsv_signal_rows(wl_recs, scan_rows, ai_map=None, scan_ex=None, limit=VSV_SHOW,
                    adir_map=None):
    """ЧИСТА: монети ОБОХ Пружин → відбір «VSV SIGNALS» (найвищий VSV Score).

    Береться монета, яка «🌀 заряджена» (заряд ≥ `VSV_MIN_CHARGE`) або вже
    почала імпульс, або в якій VSV·AI бачить сетап. Дубль (є в обох) —
    один раз, WATCHLIST сильніший. Сирих полів напрямку Пружини в рядку НЕМАЄ;
    🧭 ЙМОВІРНИЙ напрямок (25.11) — окремим зведенням `hint` (vsv_direction)."""
    ai_map = ai_map or {}
    adir_map = adir_map or {}
    sx = str(scan_ex or 'binance').lower()
    out, seen = [], set()

    def _take(S, rec, src, extra=None):
        S = str(S or '').upper()
        if not S or S in seen:
            return
        v = vsv_signal_view(rec, ai_map.get(S))
        if v is None:
            return
        if not (_vsv_f(rec.get('charge'), 0.0) >= VSV_MIN_CHARGE or v['impulse'] or v['ai_ready']):
            return
        row = {k: rec.get(k) for k in VSV_ROW_KEYS}
        row.update(v)
        row.update({'symbol': S, 'src': src, 'ex': 'bybit' if src == 'wl' else sx})
        row.update(extra or {})
        try:
            row['hint'] = _vd().direction(rec, adir_map.get(S), ai_map.get(S))
        except Exception:
            row['hint'] = None
        out.append(row)
        seen.add(S)

    for S, rec in (wl_recs or {}).items():
        _take(S, rec, 'wl')
    for r in scan_rows or []:
        _take((r or {}).get('symbol'), r or {}, 'scan',
              {'vol_usd': (r or {}).get('vol_usd'), 'bybit': (r or {}).get('bybit'),
               'added_at': (r or {}).get('added_at')})
    out.sort(key=lambda x: (-(x.get('score') or 0), -(_vsv_f(x.get('charge'), 0.0)), x['symbol']))
    total = len(out)
    out = out[:max(1, int(limit or VSV_SHOW))]
    return out, {'total': total, 'shown': len(out),
                 'impulse': sum(1 for r in out if r.get('impulse')),
                 'ai': sum(1 for r in out if r.get('ai_ready')),
                 'charged': sum(1 for r in out if not r.get('impulse'))}


# ─── 🛑 SL рекомендованих монет — СХОДИ ORDER BLOCK 1H → 15m → 5m (10.11) ──
# Вимога: «пріоритет 1Н за напрямком, а якщо немає — шукаємо OB на меншому
# таймфреймі… визначати з алгоритму OB, а не з VOB». Блоки — ТІ САМІ, що бокси
# OB на графіку кабінету (`part(sc, 'ob', …)`: памʼять сканера для TF воріт,
# інакше фоновий потік тим самим `detect_order_blocks`, кеш до закриття бару).
# Запит сторінки біржу НЕ чекає: молодший TF питаємо, лише коли старший блоку
# в бік угоди не дав. Буфер / стеля / гарантія % — ті самі налаштування, що в
# угод (`queue3_vob_sl_buffer_pct`, `autosl_max_pct`, `autosl_fallback_pct`).
REC_SL_TFS = ('1h', '15m', '5m')
# 📊 Дві групи розрахунку (вимога 11.11): 1H — SL з OB 1H; 15m — SL з OB
# 15m (→ 5m). 🎯 TP СПІЛЬНИЙ для обох (ціль «Ймовірного ходу» Пружини 1h),
# тож групи відрізняються лише стопом і R.
REC_GROUP_TFS = {'1h': ('1h',), '15m': ('15m', '5m')}


def rec_ob_level(zones, side, price, buf=0.0):
    """ЧИСТА: стоп із OB-сімей одного TF → (рівень, блок) або None.

    LONG — найновіший БИЧАЧИЙ блок ПІД ціною (стоп під його низом), SHORT —
    найновіший ВЕДМЕЖИЙ НАД ціною (стоп над верхом). Спершу internal-блоки
    (ті, що живлять ворота бота), потім swing. Блок, через який ціна вже
    пройшла, стопом бути не може."""
    try:
        px = float(price)
    except (TypeError, ValueError):
        return None
    if px <= 0 or side not in ('LONG', 'SHORT'):
        return None
    want = 'BULLISH' if side == 'LONG' else 'BEARISH'
    for fam in ('internal', 'swing'):
        blocks = sorted((zones or {}).get(fam) or [],
                        key=lambda o: to_sec(o.get('bar_time')), reverse=True)
        for o in blocks:
            if o.get('bias') != want:
                continue
            try:
                top, bot = float(o['bar_high']), float(o['bar_low'])
            except (KeyError, TypeError, ValueError):
                continue
            lvl = bot * (1.0 - buf) if side == 'LONG' else top * (1.0 + buf)
            if (side == 'LONG' and lvl < px) or (side == 'SHORT' and lvl > px):
                return lvl, dict(o, fam=fam)
    return None


def rec_sl_from_obs(zs, side, price, s=None, tfs=REC_SL_TFS):
    """ЧИСТА: сходи TF → {'sl', 'pct', 'label', 'tf', 'fb'} або {'wait': True}.

    `zs` = {tf: zones | None}; None = блоки цього TF ще рахуються → чекаємо
    (молодший TF не підставляємо раніше за старший). Жоден TF не дав → гарантія
    «% від ціни» (якщо увімкнена), інакше sl=None з причиною."""
    s = s or {}
    try:
        px = float(price)
    except (TypeError, ValueError):
        px = 0.0
    if px <= 0 or side not in ('LONG', 'SHORT'):
        return {'sl': None, 'label': 'немає ціни'}
    try:
        buf = max(0.0, float(s.get('queue3_vob_sl_buffer_pct', 0.10) or 0)) / 100.0
    except (TypeError, ValueError):
        buf = 0.001
    try:
        mx = float(s.get('autosl_max_pct', 0) or 0)
    except (TypeError, ValueError):
        mx = 0.0
    lvl, label, tf_hit, fb = None, '', '', False
    for i, tf in enumerate(tfs):
        if tf not in zs or zs.get(tf) is None:
            return {'wait': True, 'label': f'рахується OB {tf.upper()}'}
        got = rec_ob_level(zs.get(tf), side, px, buf)
        if got:
            lvl, o = got
            tf_hit, fb = tf, i > 0
            label = f'OB {tf.upper()} {o.get("fam", "")}'.strip()
            break
    if lvl is not None and mx > 0 and abs(lvl - px) / px * 100.0 > mx:
        lvl = px * (1 - mx / 100.0) if side == 'LONG' else px * (1 + mx / 100.0)
        label += f' · стеля {mx:g}%'
    if lvl is None:
        if not s.get('autosl_fallback_on', True):
            return {'sl': None, 'label': 'OB у бік угоди немає на ' + '/'.join(t.upper() for t in tfs)}
        try:
            p = float(s.get('autosl_fallback_pct', 2.0) or 2.0)
        except (TypeError, ValueError):
            p = 2.0
        lvl = px * (1 - p / 100.0) if side == 'LONG' else px * (1 + p / 100.0)
        label, tf_hit, fb = f'{p:g}% від ціни (OB немає)', '%', True
    return {'sl': round(float(lvl), 10), 'pct': round((lvl - px) / px * 100.0, 2),
            'label': label, 'tf': tf_hit, 'fb': fb}


def rec_sl_get(sc, sym, side, price, s=None, tfs=REC_SL_TFS):
    """Збирає блоки сходами (лише поки старший TF не дав рівня) → SL. Без мережі:
    чого немає в кеші — ставиться у фоновий потік і чекає («⏳»)."""
    zs = {}
    for tf in tfs:
        d, err = part(sc, 'ob', sym, tf) if sc is not None else (None, 'немає сканера')
        if d is None:
            zs[tf] = {} if err else None
            if not err:
                break
            continue
        zs[tf] = d.get('zones') or {}
        if rec_ob_level(zs[tf], side, price):
            break
    return rec_sl_from_obs(zs, side, price, s, tfs)


def rec_levels(row, sl=None):
    """ЧИСТА: SL/TP/R рядка. TP = ціль «Ймовірний хід» у бік рядка."""
    side = row.get('side')
    up = side == 'LONG'
    tp = row.get('tgt_long' if up else 'tgt_short')
    mv = row.get('move_long_pct' if up else 'move_short_pct')
    out = {'tp': tp, 'tp_pct': mv, 'sl': None, 'sl_pct': None, 'sl_label': '',
           'sl_fb': False, 'rr': None}
    if side not in ('LONG', 'SHORT'):
        return out
    if sl and sl.get('sl'):
        out.update(sl=sl['sl'], sl_pct=sl.get('pct'), sl_label=sl.get('label') or '',
                   sl_fb=bool(sl.get('fb')), sl_tf=sl.get('tf') or '')
        try:
            risk, gain = abs(float(sl['pct'])), abs(float(mv))
            out['rr'] = round(gain / risk, 2) if risk > 0 else None
        except (TypeError, ValueError, KeyError):
            pass
    elif sl:
        out['sl_label'] = sl.get('label') or ''
    return out


# ⚖ БАНЕР «SMC Forecast» — рахує й веде БОТ (`detection/spring_banner.py`,
# вимога 11.11); тут лише делегування (інфосайт транслює показники бота).
def _vd():
    """🧭 detection/vsv_direction.py — один екземпляр на процес (без пакета)."""
    import sys
    m = sys.modules.get('detection.vsv_direction')
    if m is not None:
        return m
    import importlib.util
    import os
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'detection', 'vsv_direction.py')
    spec = importlib.util.spec_from_file_location('detection.vsv_direction', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['detection.vsv_direction'] = m
    spec.loader.exec_module(m)
    return m


def _sb():
    import importlib
    import sys
    m = sys.modules.get('detection.spring_banner')
    if m is not None:
        return m
    try:
        return importlib.import_module('detection.spring_banner')
    except Exception:
        import importlib.util
        import os
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'detection', 'spring_banner.py')
        spec = importlib.util.spec_from_file_location('detection.spring_banner', p)
        m = importlib.util.module_from_spec(spec)
        sys.modules['detection.spring_banner'] = m
        spec.loader.exec_module(m)
        return m


def spb_lean_of(agg):
    return _sb().lean_of(agg)


def spb_track(prev, lean, now, data_ts=None):
    return _sb().track(prev, lean, now, data_ts)


def spb_view(key, label, agg, st):
    return _sb().view(key, label, agg, st)


def spb_combined(leans, warm=False):
    return _sb().combined(leans, warm)


def spring_banners(sc, eng, now):
    return _sb().compute(sc, eng, now)


def recommended_payload(now=None):
    now = time.time() if now is None else now
    hit = _rec_cache.get('p')
    if hit and now - hit[0] < REC_TTL:
        return hit[1]
    sc = _scanner()
    wl_recs, scan_rows, scan_meta = {}, [], {}
    eng = None
    try:
        if sc is not None and hasattr(sc, 'spring_for'):
            wl_recs = sc.spring_for() or {}
    except Exception:
        pass
    try:
        from detection.spring_scan import get_spring_scan
        eng = get_spring_scan()
        if eng is not None:
            scan_rows = eng.rows()
            st = eng.state().get('result') or {}
            scan_meta = {'ts': st.get('ts'), 'exchange': st.get('exchange')}
    except Exception:
        pass
    # 🛰 VSV SIGNALS (24.11): обидві Пружини, відбір за VSV Score, БЕЗ напрямку.
    # 🤖 ВИБІР ШІ лише підсилює оцінку (бік сетапу на сторінку не йде).
    amap, _ai = {}, None
    try:
        from detection.ai_signals import get_ai_engine
        _ai = get_ai_engine()
        amap = (_ai.recommend() or {}) if _ai is not None else {}
    except Exception:
        amap = {}
    try:
        from detection import funding_spring as _fs
        wl_shaped = {S: dict(_fs.shape_row(rec), charge_in=(rec or {}).get('charge_in'))
                     for S, rec in (wl_recs or {}).items()}
    except Exception:
        wl_shaped = dict(wl_recs or {})
    adir = {}
    try:
        if _ai is not None and hasattr(_ai, 'anomaly_dir'):
            for S in list(wl_shaped) + [str((r or {}).get('symbol') or '').upper() for r in scan_rows]:
                if S and S not in adir:
                    adir[S] = _ai.anomaly_dir(S)
    except Exception:
        pass
    rows, counts = vsv_signal_rows(wl_shaped, scan_rows, amap, scan_meta.get('exchange'),
                                   adir_map=adir)
    dir_acc = vsv_dir_attach(rows, now)
    ai_n = counts.get('ai', 0)
    try:
        banners = spring_banners(sc, eng, now)
    except Exception:
        banners = []
    p = {'ok': True, 'rows': rows, 'banners': banners, 'counts': counts, 'dir_acc': dir_acc,
         'scan': scan_meta, 'wl_n': len(wl_recs), 'ai_ready': ai_n, 'ts': now}
    _rec_cache['p'] = (now, p)
    return p


# ─── 🧭 VSV SIGNALS: ЖУРНАЛ ТОЧНОСТІ НАПРЯМКУ + РОЗВОРОТИ (вимога 25.11) ────
# Кожен напрямок, що зʼявився в таблиці, перевіряє ринок (vsv_direction.journal_*):
# +поріг у бік прогнозу раніше за −поріг проти → ✅, навпаки → ❌. Журнал веде
# фоновий потік VSV раз на хвилину (живі ціни: стрічка Bybit / кеш сканера),
# незалежно від тумблера Telegram. Стан — БД `VSV_DIR_KEY` (переживає рестарт).
VSV_DIR_KEY = 'vsv_direction_journal_v1'
_vsv_dir = {'j': None, 'acc': None, 'flip': {}, 'saved': 0.0, 'dirty': False}
_vsv_dir_lock = threading.Lock()


def _vsv_dir_load():
    if _vsv_dir['j'] is not None:
        return
    j, fl = [], {}
    try:
        db = _vsv_db()
        raw = db.get_setting(VSV_DIR_KEY, None) if db is not None else None
        if isinstance(raw, dict):
            j = [e for e in (raw.get('j') or []) if isinstance(e, dict)]
            fl = {k: v for k, v in (raw.get('flip') or {}).items() if isinstance(v, dict)}
    except Exception:
        pass
    _vsv_dir['j'], _vsv_dir['flip'] = j, fl
    _vsv_dir['acc'] = _vd().accuracy(j)


def vsv_dir_attach(rows, now):
    """Рядкам — точність (кошик / загальна) і свіжий розворот напрямку.
    Повертає загальну точність для шапки."""
    try:
        vd = _vd()
        with _vsv_dir_lock:
            _vsv_dir_load()
            acc = _vsv_dir['acc'] or vd.accuracy([])
            fl = _vsv_dir['flip']
            for r in rows or []:
                h = r.get('hint')
                if not isinstance(h, dict):
                    continue
                S = r.get('symbol')
                rec = vd.track_flip(fl, S, h.get('side'), now)
                if rec != fl.get(S):
                    fl[S] = rec
                    _vsv_dir['dirty'] = True
                h['since'] = rec.get('since')
                h['flip'] = vd.flip_view(rec, now)
                h['acc'] = vd.row_accuracy(acc, h.get('prob')) if h.get('side') else None
        return {k: acc.get(k) for k in ('n', 'win', 'loss', 'flat', 'open', 'acc', 'min_n', 'buckets')}
    except Exception:
        return None


def _vsv_prices(syms):
    """Живі ціни монет журналу: стрічка Bybit (≤60с) → кеш сканера."""
    out = {}
    feed = None
    try:
        from detection.price_feed import get_price_feed
        feed = get_price_feed()
    except Exception:
        feed = None
    sc = _scanner()
    for S in syms:
        px = None
        if feed is not None:
            try:
                v = feed.price(S, max_age=60)
                px = v[0] if v else None
            except Exception:
                px = None
        if px is None and sc is not None and hasattr(sc, '_get_live_price'):
            try:
                px = sc._get_live_price(S)
            except Exception:
                px = None
        if px:
            out[S] = float(px)
    return out


def vsv_dir_tick(now=None, rows=None, prices=None):
    """Раз на хвилину: перевіряє відкриті прогнози і відкриває нові."""
    now = time.time() if now is None else now
    vd = _vd()
    if rows is None:
        try:
            rows = (recommended_payload(now) or {}).get('rows') or []
        except Exception:
            rows = []
    with _vsv_dir_lock:
        _vsv_dir_load()
        j = _vsv_dir['j']
    syms = sorted({e.get('S') for e in j if e.get('st') == 'open'}
                  | {str(r.get('symbol') or '').upper() for r in rows})
    px = prices if prices is not None else _vsv_prices([s for s in syms if s])
    with _vsv_dir_lock:
        j = vd.journal_update(_vsv_dir['j'], px, now)
        j = vd.journal_open(j, rows, now, px)
        changed = j != _vsv_dir['j']
        _vsv_dir['j'] = j
        _vsv_dir['acc'] = vd.accuracy(j)
        if changed:
            _vsv_dir['dirty'] = True
        save = _vsv_dir['dirty'] and now - _vsv_dir['saved'] >= 60
        blob = {'j': j, 'flip': dict(_vsv_dir['flip'])} if save else None
        if save:
            _vsv_dir['saved'], _vsv_dir['dirty'] = now, False
    if blob is not None:
        try:
            db = _vsv_db()
            if db is not None:
                db.set_setting(VSV_DIR_KEY, blob)
        except Exception:
            pass
    return _vsv_dir['acc']


# ─── 📨 VSV SIGNALS → Telegram «🔔 Сигнали» (вимога 25.11) ─────────────────
# Поява НОВОЇ монети у списку VSV SIGNALS (той самий `recommended_payload`, що
# малює сайт) → коротке повідомлення в спільну групу, тема «🔔 Сигнали»
# (категорія `sbr`). Без напрямку — як і на сайті. Тумблер — у боті (деф. УВІМК).
VSV_TG_SET_KEY = 'vsv_signals_tg_settings'   # {enabled}
VSV_TG_KEY = 'vsv_signals_tg_v1'             # {init, in, sent}
VSV_TG_TICK_SEC = 60
VSV_TG_WARMUP_SEC = 300       # після старту бота дані Пружин ще доїжджають
VSV_TG_COOLDOWN = 6 * 3600    # монета, що «блимає» на межі списку, — не частіше
VSV_TG_MAX_LINES = 15
_vsv_tg = {'boot': time.time(), 'thread': None, 'st': None, 'set': None,
           'last': None}


def _vsv_db():
    try:
        from storage.db_operations import get_db
        return get_db()
    except Exception:
        return None


def vsv_tg_settings():
    s = _vsv_tg.get('set')
    if s is None:
        s = {'enabled': True}
        db = _vsv_db()
        try:
            raw = db.get_setting(VSV_TG_SET_KEY, None) if db is not None else None
            if isinstance(raw, dict) and 'enabled' in raw:
                s['enabled'] = bool(raw.get('enabled'))
        except Exception:
            pass
        _vsv_tg['set'] = s
    return dict(s)


def vsv_tg_set(enabled):
    s = {'enabled': bool(enabled)}
    _vsv_tg['set'] = s
    db = _vsv_db()
    try:
        if db is not None:
            db.set_setting(VSV_TG_SET_KEY, s)
    except Exception:
        pass
    return dict(s)


def vsv_tg_new(rows, st, now, cooldown=VSV_TG_COOLDOWN):
    """ЧИСТА: які монети ЩОЙНО зʼявились у VSV SIGNALS → (нові рядки, новий стан).
    Перший прохід (стану ще немає) — тиха база; порожній список — «немає даних»,
    а не «усі пішли» (стан не чіпаємо); монета, про яку вже сказано менше ніж
    `cooldown` тому, не повторюється (блимання на межі топу)."""
    st = dict(st or {})
    syms = [str((r or {}).get('symbol') or '').upper() for r in rows or []]
    syms = [s for s in syms if s]
    if not syms:
        return [], st
    sent = {k: float(v) for k, v in (st.get('sent') or {}).items()}
    if not st.get('init'):
        return [], {'init': True, 'in': syms, 'sent': sent}
    prev = set(st.get('in') or [])
    new = []
    for r in rows:
        S = str((r or {}).get('symbol') or '').upper()
        if S and S not in prev and (S not in sent or now - sent[S] >= cooldown):
            new.append(r)
            sent[S] = now
    sent = {k: v for k, v in sent.items() if now - v < 2 * cooldown}
    if len(sent) > 300:
        sent = dict(sorted(sent.items(), key=lambda kv: kv[1])[-300:])
    return new, {'init': True, 'in': syms, 'sent': sent}


def vsv_tg_text(rows):
    """ЧИСТА: повідомлення для теми «🔔 Сигнали» — БЕЗ напрямку."""
    lines = ['🛰 <b>VSV SIGNALS</b> · нові інструменти в радарі волатильності']
    for r in (rows or [])[:VSV_TG_MAX_LINES]:
        S = str(r.get('symbol') or '').upper()
        ph = '💥' if r.get('impulse') else '🌀'
        bits = [f"{ph} #{S}", f"VSV <b>{float(r.get('score') or 0):.0f}</b> ({r.get('grade') or 'C'})"]
        if r.get('charge') is not None:
            bits.append(f"заряд {float(r['charge']):.0f}")
        sq = int(_vsv_f(r.get('squeeze_bars'), 0) or 0)
        if sq > 0:
            bits.append(f"squeeze {sq} год")
        if r.get('amp_pct') is not None:
            bits.append(f"±{float(r['amp_pct']):.2f}%")
        if r.get('ai_ready'):
            bits.append('🤖 сетап')
        lines.append(' · '.join(bits))
    if len(rows or []) > VSV_TG_MAX_LINES:
        lines.append(f"… і ще {len(rows) - VSV_TG_MAX_LINES}")
    lines.append('ℹ️ Напрямок не вказується: пробій діапазону можливий в обидва боки. '
                 'Інформаційний watchlist, не торгова рекомендація.')
    return '\n'.join(lines)


def vsv_tg_tick(now=None, send=None):
    """Один прохід: нові монети → Telegram. Повертає список надісланих символів."""
    now = time.time() if now is None else now
    if not vsv_tg_settings().get('enabled'):
        return []
    if now - _vsv_tg['boot'] < VSV_TG_WARMUP_SEC:
        return []
    st = _vsv_tg.get('st')
    if st is None:
        db = _vsv_db()
        try:
            raw = db.get_setting(VSV_TG_KEY, None) if db is not None else None
        except Exception:
            raw = None
        st = raw if isinstance(raw, dict) else {}
    rows = (recommended_payload(now) or {}).get('rows') or []
    new, st2 = vsv_tg_new(rows, st, now)
    if st2 != st:
        _vsv_tg['st'] = st2
        db = _vsv_db()
        try:
            if db is not None:
                db.set_setting(VSV_TG_KEY, st2)
        except Exception:
            pass
    else:
        _vsv_tg['st'] = st
    if not new:
        return []
    syms = [str(r.get('symbol') or '').upper() for r in new]
    ok, why = False, ''
    try:
        if send is not None:
            ok = bool(send(vsv_tg_text(new)))
        else:
            from web import tg_bot
            ok = bool(tg_bot.notify_category('sbr', vsv_tg_text(new)))
            if not ok:
                why = (getattr(tg_bot, 'last_send_error', lambda: '')() or '')
    except Exception as e:
        why = f'{type(e).__name__}: {e}'
    _vsv_tg['last'] = {'t': now, 'syms': syms, 'ok': ok, 'why': why}
    try:
        from detection.activity_log import log_activity
        txt = (f"🛰 VSV SIGNALS → Telegram «🔔 Сигнали»: {', '.join(syms)}" if ok
               else f"📨 Telegram НЕ прийняв VSV SIGNALS ({', '.join(syms)}): {why or 'тему вимкнено або не налаштовано'}")
        log_activity(syms[0] if len(syms) == 1 else 'VSV', 'signal', txt, source='VSV',
                     extra={'vsv': syms, 'ok': ok})
    except Exception:
        pass
    return syms if ok else []


def _vsv_tg_loop():
    while True:
        try:
            vsv_dir_tick()
        except Exception as e:
            print(f"[VSV-DIR] {type(e).__name__}: {e}")
        try:
            vsv_tg_tick()
        except Exception as e:
            print(f"[VSV-TG] {type(e).__name__}: {e}")
        time.sleep(VSV_TG_TICK_SEC)


def vsv_tg_start():
    t = _vsv_tg.get('thread')
    if t is None or not t.is_alive():
        t = threading.Thread(target=_vsv_tg_loop, name='vsv-signals-tg', daemon=True)
        _vsv_tg['thread'] = t
        t.start()


def register_info_portal(app):
    from flask import jsonify, request

    def _me():
        from web.auth import current_user
        return current_user()

    def _perms(u=None):
        from web.auth import _perms_mod
        return _perms_mod().of_user(_me() if u is None else u)

    def _no(key):
        """🛡 403, коли адміністратор не дав користувачу це право."""
        return jsonify({'ok': False, 'error': 'no_perm', 'perm': key,
                        'reason': 'Адміністратор не надав доступ до цього розділу'}), 403

    # 📨 VSV SIGNALS → Telegram «🔔 Сигнали»: тумблер живе в боті (шлях поза
    # /api/info і /api/me → гейт пускає лише користувачів із доступом до бота).
    @app.route('/api/vsv-signals/settings', methods=['GET', 'POST'])
    def api_vsv_signals_settings():
        if request.method == 'POST':
            body = request.get_json(silent=True) or {}
            vsv_tg_set(bool(body.get('enabled')))
        return jsonify({'ok': True, 'settings': vsv_tg_settings(), 'last': _vsv_tg.get('last'),
                        'in': len(((_vsv_tg.get('st') or {}).get('in')) or [])})

    try:
        vsv_tg_start()
    except Exception as e:
        print(f"[VSV-TG] start: {e}")

    @app.route('/api/info/universe')
    def api_info_universe():
        if not _perms().get('sec_terminal'):
            return _no('sec_terminal')
        """Монети WATCHLIST бота — підказка при додаванні (дані вже пораховані).
        Додати можна й будь-яку іншу USDT-монету — її порахує фоновий потік."""
        return jsonify({'ok': True, 'symbols': sorted(universe()),
                        'tfs': list(CHART_TFS), 'ob_tfs': list(OB_TFS),
                        'vob_tfs': list(VOB_TFS), 'hl_tfs': list(HL_TFS), 'sbr_chart_tfs': list(SBR_CHART_TFS), 'poc_days': list(POC_DAYS),
                        'poc_tfs': list(POC_TFS)})

    @app.route('/api/info/rows')
    def api_info_rows():
        if not _perms().get('sec_terminal'):
            return _no('sec_terminal')
        syms = [x for x in (request.args.get('symbols') or '').split(',') if x]
        return jsonify({'ok': True, 'rows': rows_for(syms), 'ts': time.time()})

    @app.route('/api/info/liq')
    def api_info_liq():
        """💧 Профіль ліквідності монети — ЛИШЕ за кнопкою (ходить на біржу)."""
        a = request.args
        u = _me()
        if not _perms(u).get('ly_liq'):
            return _no('ly_liq')
        return jsonify(liq_profile(a.get('symbol') or '', a.get('exchange'),
                                   a.get('bars'), a.get('res'), step=a.get('step'),
                                   user=getattr(u, 'id', None) if u else
                                   (request.remote_addr or None)))

    @app.route('/api/me/liq-calc', methods=['POST'])
    def api_info_liq_calc():
        """💧 Розрахунок профілю з сирих даних, які браузер узяв у біржі сам."""
        u = _me()
        if not _perms(u).get('ly_liq'):
            return _no('ly_liq')
        return jsonify(liq_from_client(request.get_json(silent=True) or {},
                                       user=getattr(u, 'id', None) if u else
                                       (request.remote_addr or None)))

    @app.route('/api/info/coin')
    def api_info_coin():
        from web.auth import _perms_mod
        user_perms = _perms_mod()
        pm = _perms()
        if not pm.get('sec_terminal'):
            return _no('sec_terminal')
        a = request.args
        p = coin_payload(a.get('symbol') or '', tf=a.get('tf'), ob_tf=a.get('ob_tf'),
                         vob_tf=a.get('vob_tf'), poc_days=a.get('poc_days'),
                         poc_tf=a.get('poc_tf'), hl_tf=a.get('hl_tf'), sbr_tf=a.get('sbr_tf'),
                         shl_tfv=a.get('shl_tf'),
                         anom=str(a.get('anom') or '') in ('1', 'true') and bool(pm.get('ly_anom')),
                         vol=str(a.get('vol') or '') in ('1', 'true') and bool(pm.get('ly_vol')))
        # 🛡 заборонені шари не віддаємо (кеш спільний — ріжемо копію); набір прав
        # входить у сигнатуру, щоб зміна прав одразу перемалювала графік
        p = user_perms.strip_coin(p, pm)
        if p.get('sig'):
            p = dict(p, sig=p['sig'] + '|' + user_perms.sig(pm))
        if p.get('ok') and a.get('sig') and a.get('sig') == p.get('sig'):
            return jsonify({'ok': True, 'same': True, 'sig': p['sig']})
        return jsonify(p)

    @app.route('/api/info/paper')
    def api_info_paper():
        """🧪 Тестові угоди бота: віртуальний рахунок, статистика за період
        (?from&to — межі дня рахує сторінка за годинником ПК), відкриті й
        останні закриті paper-угоди. Лише читання."""
        if not _perms().get('sec_paper'):
            return _no('sec_paper')
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Trade Manager недоступний'})
        d = dict(_paper_snapshot(tm))
        a = request.args
        if a.get('from') or a.get('to'):
            try:
                d['stats'] = tm.stats_for_period(a.get('from'), a.get('to'), shadow=True)
            except (TypeError, ValueError) as e:
                return jsonify({'ok': False, 'reason': f'некоректний період: {e}'})
        else:
            d['stats'] = d.get('stats_all') or {}
        d.pop('stats_all', None)
        r = jsonify({'ok': True, 'ts': time.time(), **d})
        r.headers['Cache-Control'] = 'no-store'
        return r

    @app.route('/api/info/recommended')
    def api_info_recommended():
        """🤖 «Рекомендовані ботом» — лише читання памʼяті бота."""
        if not _perms().get('sec_signals'):
            return _no('sec_signals')
        r = jsonify(recommended_payload())
        r.headers['Cache-Control'] = 'no-store'
        return r

    @app.route('/api/info/paper/days')
    def api_info_paper_days():
        """📅 Дні з закритими paper-угодами — для календаря."""
        if not _perms().get('sec_paper'):
            return _no('sec_paper')
        from detection.trade_manager import get_trade_manager
        tm = get_trade_manager()
        if not tm:
            return jsonify({'ok': False, 'reason': 'Trade Manager недоступний'})
        r = jsonify({'ok': True, 'items': tm.trade_days(shadow=True)})
        r.headers['Cache-Control'] = 'no-store'
        return r

    @app.route('/api/me/watchlist', methods=['GET', 'POST'])
    def api_me_watchlist():
        from web.auth import _load_prefs, _update_user
        u = _me()
        if u is None:
            return jsonify({'ok': False, 'error': 'auth_required'}), 401
        sc = _scanner()
        uni = universe(sc)
        prefs = _load_prefs(u)
        cur = user_info(prefs, uni, defaults(sc) if sc else None)
        err = ''
        pm = _perms(u)
        if request.method == 'POST':
            body = dict(request.get_json(silent=True) or {})
            # 🛡 підписка без права не вмикається (і не зберігається)
            for k, pk in (('sbr', 'ntf_sbr'), ('anom', 'ntf_anom')):
                if k in body and not pm.get(pk):
                    body.pop(k)
                    err = err or 'Адміністратор не надав доступ до цих сповіщень'
            cur, err2 = apply_change(cur, body, uni)
            err = err2 or err
            prefs['info'] = cur
            _update_user(u.id, prefs=json.dumps(prefs)[:20000])
            sbr_subs_invalidate()
            anom_subs_invalidate()
        return jsonify({'ok': not err, 'error': err, 'max': IWL_MAX, 'universe_n': len(uni),
                        'tg_linked': bool(getattr(u, 'telegram_chat_id', None)),
                        'sbr_tfs': list(SBR_SUB_TFS), **cur,
                        'perms': pm, 'is_admin': bool(getattr(u, 'is_admin', False))})
