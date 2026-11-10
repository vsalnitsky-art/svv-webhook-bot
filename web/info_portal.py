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
                        'volume': z.get('volume'), 'pct': z.get('pct')})
        except (KeyError, TypeError, ValueError):
            continue
    return out


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
                        'low': float(k.get('l', c)), 'close': c})
        except (TypeError, ValueError, AttributeError):
            continue
    return out


IMB_SHOW = 12               # скільки незаповнених IMB малюємо
SBR_SHOW = 40               # скільки останніх сигналів SBR на графіку


IMB_MAX_FILLED = 80         # заповнено ≥ N% → розрив відпрацьовано, не малюємо
IMB_MIN_REST_ATR = 0.25     # залишок зони < N·ATR(14) → лише смужка-шум


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
    """ЧИСТА: чи варто малювати IMB (вимога 09.11 — «мізерні полоси захламляють
    графік»). Ховаємо розрив, який ціна вже майже закрила (filled_pct ≥
    IMB_MAX_FILLED) або чий НЕЗАПОВНЕНИЙ залишок тонший за IMB_MIN_REST_ATR·ATR.
    ATR невідомий → судимо лише за заповненням."""
    try:
        if float(z.get('filled_pct') or 0) >= IMB_MAX_FILLED:
            return False
        rest = float(z['top']) - float(z['bottom'])
    except (KeyError, TypeError, ValueError):
        return False
    if atr and rest < IMB_MIN_REST_ATR * float(atr):
        return False
    return rest > 0


def imb_of(klines, tf, now=None):
    """📐 IMB (Fair Value Gap) — `detection.imbalance` на барах графіка
    РАЗОМ із ЖИВИМ (вимога 08.11, кейс BTC 1H: обвал 21:00 — TV уже малював
    розрив 74.7–75.7K, а кабінет мовчав, бо третя свічка ще не закрилась).
    Як у TradingView: зона живого бару може змінитись або зникнути до його
    закриття; закриті бари поводяться як раніше."""
    try:
        from detection.imbalance import detect_imbalances
    except Exception:
        return []
    closed = [k for k in (klines or []) if k.get('t') is not None]
    atr = _imb_atr(closed)
    out = []
    for z in detect_imbalances(closed, max_show=IMB_SHOW * 3):
        if not imb_visible(z, atr):
            continue                     # майже заповнений «ошматок» — не засмічує графік
        if len(out) >= IMB_SHOW:
            break
        out.append({'side': z['side'], 'top': z['top'], 'bottom': z['bottom'],
                    'start': z['t'], 'filled_pct': z.get('filled_pct', 0),
                    'vol': z.get('vol'), 'pct': z.get('pct'),
                    'sell_pct': z.get('sell_pct'), 'buy_pct': z.get('buy_pct')})
    return out


SBR_TFS = ('1h', '4h')       # SBR — лише ті TF, на яких його веде бот і TG


def sbr_of(klines, tf, now=None):
    """🔔 Сигнали Session Band Reversal на ЗАКРИТИХ барах ГРАФІКА — той самий
    `sbr_signals.compute_signals`, що рушій бота (1H/4H) і Telegram.
    Лише 1H / 4H (вимога 09.11): індикатор калібровано на цих TF, на інших
    позначки не малюємо."""
    if str(tf).lower() not in SBR_TFS:
        return []
    try:
        from detection.sbr_signals import compute_signals, closed_bars
    except Exception:
        return []
    now = time.time() if now is None else now
    b = bar_sec(tf)
    cb = closed_bars(klines or [], b, now)
    out = []
    for x in compute_signals(cb, b)[-SBR_SHOW:]:
        out.append({'side': x['side'], 't': to_sec(x['t']), 'n': x.get('n'),
                    'trigger': x.get('trigger'), 'score': x.get('score'),
                    'risk': x.get('risk'), 'sl': x.get('sl'),
                    'time_kyiv': x.get('time_kyiv'), 'tf': str(tf).upper()})
    return out


def sig_of(payload):
    o = payload.get('ohlc') or []
    last = o[-1] if o else {}
    ev = payload.get('events') or [{}]
    return '|'.join(str(x) for x in (
        payload.get('tf'), payload.get('ob_tf'), payload.get('vob_tf'),
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
        *[((payload.get('sbr_last') or {}).get(t) or {}).get('t') for t in SBR_TFS]))


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
        try:
            return {'vz': sc._volumized_zones(sym)}
        except Exception:
            return None
    return None


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


def sbr_last_view(sig, tf):
    """ЧИСТА: останній сигнал SBR → поля для бейджа кабінету."""
    if not sig or not sig.get('trigger'):
        return None
    return {'side': sig.get('side'), 't': to_sec(sig.get('t')), 'n': sig.get('n'),
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
        if vk and len(vk) > 20:
            r = detect_volumized_obs(
                vk, swing_length=int(st.get('volumized_swing_length', 10)),
                ob_end_method=st.get('volumized_ob_end_method', 'Wick'),
                max_atr_mult=float(st.get('volumized_max_atr_mult', 3.5)),
                zone_count=st.get('volumized_zone_count', 'Low'),
                combine_obs=bool(st.get('volumized_combine_obs', True)))
            obs = list(r.get('bullish_obs') or []) + list(r.get('bearish_obs') or [])
            obs.sort(key=lambda o: o.get('formation_time', o.get('start_time', 0)) or 0,
                     reverse=True)
            vz = [z for z in (sc._format_vob(o, tf) for o in obs) if z]
        return {'vz': vz}, min(next_close(now, tf), now + VOB_MAX_TTL)
    if kind == 'shl':
        # 🏁 Swing High / Low — ТА САМА функція бота, що лінії Strong/Weak на
        # графіку /smart-money (дилінг-діапазон = єдине джерело з PD-зоною).
        return {'levels': sc._swing_hl_for_tf(sym, tf)}, now + SHL_TTL
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


def coin_payload(sym, sc=None, tf=None, ob_tf=None, vob_tf=None, poc_days=None, poc_tf=None):
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
    in_bot = _in_bot(sc, sym)
    ck = (sym, tf, ob_tf, vob_tf, poc_days, poc_tf)
    now = time.time()
    with _lock:
        hit = _coin_cache.get(ck)
    if hit and now - hit[0] < COIN_TTL:
        # частини все одно «просимо», інакше кеш потоку їх забуде
        for k, t in (('bars', tf), ('ob', ob_tf), ('vob', vob_tf), ('shl', shl_tf(sc))):
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
    if vob_tf != OFF:
        d, _e = part(sc, 'vob', sym, vob_tf)
        vob, vob_wait = vob_boxes((d or {}).get('vz')), d is None
    if ohlc:
        reason = ''
    elif err:
        reason = err
    elif in_bot:
        reason = '⏳ завантаження барів ' + tf.upper()
    else:
        reason = '⏳ монети немає в боті — рахуємо окремо, кілька секунд'
    _shl_tf = shl_tf(sc)
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
        'tf': tf, 'ob_tf': ob_tf, 'vob_tf': vob_tf,
        'poc_days': poc_days, 'poc_tf': poc_tf,
        'ohlc': ohlc, 'events': events,
        'imb': imb_of((bars or {}).get('kl'), tf, now),
        'sbr': sbr_of((bars or {}).get('kl'), tf, now),
        'sbr_last': sbr_last,
        'ob': ob, 'vob': vob, 'ob_wait': ob_wait, 'vob_wait': vob_wait,
        'pd': _pd(sc, sym), 'poc': poc, 'poc_state': _st, 'shl': shl, 'ts': now,
    }
    payload['sig'] = sig_of(payload)
    if not ohlc or ob_wait or vob_wait:
        return payload                 # «рахується» не кешуємо
    with _lock:
        _coin_cache[ck] = (now, payload)
        if len(_coin_cache) > 1200:
            _coin_cache.clear()
    return payload


# ── Особисті дані користувача ─────────────────────────────────────────────
def sbr_norm(raw, wl=None):
    """ЧИСТА: {SYM: ['1h','4h']} — персональні 🔔 SBR-підписки. Лише монети
    списку (якщо `wl` задано), лише TF 1H/4H; порожній вибір = немає підписки."""
    out = {}
    for k, v in (raw or {}).items() if isinstance(raw, dict) else []:
        S = norm_symbol(k)
        if not valid_symbol(S) or (wl is not None and S not in wl):
            continue
        tfs = [t for t in SBR_TFS if t in [str(x).lower() for x in (v or [])]]
        if tfs:
            out[S] = tfs
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
            # що малювати на графіку (лише показ, рахується завжди)
            'show_imb': bool(info.get('show_imb', True)),
            'show_sbr': bool(info.get('show_sbr', True)),
            'show_shl': bool(info.get('show_shl', True)),
            # ⚪ POC: вікно (днів) і TF — свої для кабінету (деф. 7 днів · 1H)
            'poc_days': poc_norm(info.get('poc_days'), info.get('poc_tf'))[0],
            'poc_tf': poc_norm(info.get('poc_days'), info.get('poc_tf'))[1],
            # 🔔 особисті SBR-сповіщення в Telegram (вимога 09.11)
            'sbr': sbr_norm(info.get('sbr'), wl)}


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
    for k in ('show_imb', 'show_sbr', 'show_shl'):
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
    out['wl'], out['sel'] = wl, sel
    return out, err


# ── 🔔 Особисті SBR-сповіщення (вимога 09.11) ────────────────────────────
# Користувач у кабінеті вмикає 🔔 біля монети й обирає 1H / 4H → сигнал іде
# ЙОМУ в особистий чат із головним ботом (не в групу). Розрахунок — той самий
# рушій `sbr_signals` (монети підписок додаються в його набір), тож сигнал у
# кабінеті, у групі і в особистому чаті — один і той самий.
SUBS_TTL = 30.0
SUBS_EXTRA_MAX = 300           # стеля монет поза WATCHLIST бота, які веде рушій
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
    """Монети особистих підписок — рушій SBR рахує їх навіть поза WATCHLIST бота."""
    out = []
    for _c, sub in sbr_subs():
        for S in sub:
            if S not in out:
                out.append(S)
    return out[:SUBS_EXTRA_MAX]


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


# ── 🧪 ТЕСТОВІ (paper) угоди бота — лише читання (вимога 08.11) ─────────
PAPER_TTL = 2.0                 # спільний кеш знімка (сторінки опитують часто)
PAPER_CLOSED_SHOW = 50          # скільки останніх закритих paper-угод віддаємо
_PAPER_POS_KEYS = ('symbol', 'side', 'entry_price', 'current_price', 'pnl_pct',
                   'pnl_net', 'paper_size', 'opened_at', 'opened_by', 'manual_sl',
                   'manual_tp', 'manual_tp1', 'tp1_done', 'spring_open', 'mmd')
_PAPER_CLOSED_KEYS = ('symbol', 'side', 'entry_price', 'exit_price', 'opened_at',
                      'closed_at', 'pnl_pct', 'reason', 'opened_by')
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
            'move_short_pct', 'box_pct', 'squeeze_bars', 'fired', 'bar_t')
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

    def _pot(x):
        k = 'pot_long' if x.get('side') == 'LONG' else 'pot_short'
        try:
            return float(x.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0
    out.sort(key=lambda x: (-_pot(x), x['symbol']))
    return out


def recommended_payload(now=None):
    now = time.time() if now is None else now
    hit = _rec_cache.get('p')
    if hit and now - hit[0] < REC_TTL:
        return hit[1]
    sc = _scanner()
    wl_recs, wl_pass, scan_rows, scan_meta = {}, {}, [], {}
    try:
        if sc is not None and hasattr(sc, 'spring_for'):
            wl_recs = sc.spring_for() or {}
            wl_pass = sc.spring_pass_map() if hasattr(sc, 'spring_pass_map') else {}
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
    p = {'ok': True, 'rows': recommended_rows(wl_recs, wl_pass, scan_rows),
         'scan': scan_meta, 'wl_n': len(wl_recs), 'ts': now}
    _rec_cache['p'] = (now, p)
    return p


def register_info_portal(app):
    from flask import jsonify, request

    def _me():
        from web.auth import current_user
        return current_user()

    @app.route('/api/info/universe')
    def api_info_universe():
        """Монети WATCHLIST бота — підказка при додаванні (дані вже пораховані).
        Додати можна й будь-яку іншу USDT-монету — її порахує фоновий потік."""
        return jsonify({'ok': True, 'symbols': sorted(universe()),
                        'tfs': list(CHART_TFS), 'ob_tfs': list(OB_TFS),
                        'vob_tfs': list(VOB_TFS), 'poc_days': list(POC_DAYS),
                        'poc_tfs': list(POC_TFS)})

    @app.route('/api/info/rows')
    def api_info_rows():
        syms = [x for x in (request.args.get('symbols') or '').split(',') if x]
        return jsonify({'ok': True, 'rows': rows_for(syms), 'ts': time.time()})

    @app.route('/api/info/liq')
    def api_info_liq():
        """💧 Профіль ліквідності монети — ЛИШЕ за кнопкою (ходить на біржу)."""
        a = request.args
        u = _me()
        return jsonify(liq_profile(a.get('symbol') or '', a.get('exchange'),
                                   a.get('bars'), a.get('res'), step=a.get('step'),
                                   user=getattr(u, 'id', None) if u else
                                   (request.remote_addr or None)))

    @app.route('/api/me/liq-calc', methods=['POST'])
    def api_info_liq_calc():
        """💧 Розрахунок профілю з сирих даних, які браузер узяв у біржі сам."""
        u = _me()
        return jsonify(liq_from_client(request.get_json(silent=True) or {},
                                       user=getattr(u, 'id', None) if u else
                                       (request.remote_addr or None)))

    @app.route('/api/info/coin')
    def api_info_coin():
        a = request.args
        p = coin_payload(a.get('symbol') or '', tf=a.get('tf'), ob_tf=a.get('ob_tf'),
                         vob_tf=a.get('vob_tf'), poc_days=a.get('poc_days'),
                         poc_tf=a.get('poc_tf'))
        if p.get('ok') and a.get('sig') and a.get('sig') == p.get('sig'):
            return jsonify({'ok': True, 'same': True, 'sig': p['sig']})
        return jsonify(p)

    @app.route('/api/info/paper')
    def api_info_paper():
        """🧪 Тестові угоди бота: віртуальний рахунок, статистика за період
        (?from&to — межі дня рахує сторінка за годинником ПК), відкриті й
        останні закриті paper-угоди. Лише читання."""
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
        r = jsonify(recommended_payload())
        r.headers['Cache-Control'] = 'no-store'
        return r

    @app.route('/api/info/paper/days')
    def api_info_paper_days():
        """📅 Дні з закритими paper-угодами — для календаря."""
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
        if request.method == 'POST':
            cur, err = apply_change(cur, request.get_json(silent=True) or {}, uni)
            prefs['info'] = cur
            _update_user(u.id, prefs=json.dumps(prefs)[:20000])
            sbr_subs_invalidate()
        return jsonify({'ok': not err, 'error': err, 'max': IWL_MAX, 'universe_n': len(uni),
                        'tg_linked': bool(getattr(u, 'telegram_chat_id', None)),
                        'sbr_tfs': list(SBR_TFS), **cur})
