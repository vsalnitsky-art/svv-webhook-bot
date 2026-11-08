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
CHART_TFS = ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
OB_TFS = ('15m', '30m', '1h', '4h')
VOB_TFS = ('1m', '5m', '15m', '30m', '1h', '4h')
OFF = 'off'
POC_TTL = 300.0              # спільний кеш POC на монету
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
_poc_cache = {}              # SYM → (ts, view|None)
_coin_cache = {}             # (SYM, tf, ob_tf, vob_tf) → (ts, payload)
_comp = {}                   # (kind, SYM, tf) → {'data','at','until','err','err_at'}
_want = {}                   # (kind, SYM, tf) → коли востаннє просили
_poc_want = {}               # SYM → коли востаннє просили POC
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


def ob_boxes(zones):
    """Знімок OB-сімей → бокси для графіка (без дублів між сім'ями)."""
    out, seen = [], set()
    for fam in ('swing', 'internal'):
        for o in (zones or {}).get(fam) or []:
            try:
                top, bot = float(o['bar_high']), float(o['bar_low'])
            except (KeyError, TypeError, ValueError):
                continue
            st = to_sec(o.get('bar_time'))
            key = (st, round(top, 10), round(bot, 10))
            if key in seen:
                continue
            seen.add(key)
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


def imb_of(klines, tf, now=None):
    """📐 IMB (Fair Value Gap) на ЗАКРИТИХ барах графіка — `detection.imbalance`
    (той самий модуль бот може брати для своєї логіки)."""
    try:
        from detection.imbalance import detect_imbalances
    except Exception:
        return []
    now = time.time() if now is None else now
    b = bar_sec(tf)
    closed = [k for k in (klines or []) if to_sec(k.get('t')) + b <= now + 0.5]
    out = []
    for z in detect_imbalances(closed, max_show=IMB_SHOW):
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


def poc_want(sym):
    with _lock:
        if sym in _poc_want or len(_poc_want) < WANT_MAX:
            _poc_want[sym] = time.time()
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
            for sym in pwant:
                with _lock:
                    hit = _poc_cache.get(sym)
                if hit and time.time() - hit[0] < POC_TTL:
                    continue
                _poc(sc, sym, allow_compute=True)
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


def _poc_params(sc):
    s = getattr(sc, '_settings', {}) or {}
    try:
        days = float(s.get('poc_filter_window_days', 3) or 3)
    except (TypeError, ValueError):
        days = 3.0
    return (str(s.get('poc_filter_market', 'futures') or 'futures'),
            str(s.get('poc_filter_tf', '1h') or '1h'), days)


def _poc(sc, sym, allow_compute=True, price=None):
    """POC через ТОЙ САМИЙ compute_poc, що бейдж графіка, зі спільним кешем.
    Вікно прибите до години (to = поточна година) — ключ кешу стабільний."""
    now = time.time()
    with _lock:
        hit = _poc_cache.get(sym)
    if hit and now - hit[0] < POC_TTL:
        view = hit[1]
    elif not allow_compute:
        return ('pending', None)
    else:
        view = None
        try:
            from detection.volume_profile import compute_poc
            market, tf, days = _poc_params(sc)
            to_s = int(now // 3600 * 3600)
            res = compute_poc(sym, from_sec=to_s - int(days * 86400), to_sec=to_s,
                              interval=tf, market=market)
            if res.get('ok') and res.get('poc'):
                view = {'poc': float(res['poc']), 'market': market.upper(),
                        'tf': tf.upper(), 'days': days, 'exchange': res.get('exchange')}
        except Exception as e:
            print(f"[INFO] POC {sym}: {e}")
        with _lock:
            _poc_cache[sym] = (now, view)
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


def coin_payload(sym, sc=None, tf=None, ob_tf=None, vob_tf=None):
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
    in_bot = _in_bot(sc, sym)
    ck = (sym, tf, ob_tf, vob_tf)
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
        poc_want(sym)
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
    poc_want(sym)
    _st, poc = _poc(sc, sym, allow_compute=False,
                    price=(ohlc[-1]['close'] if ohlc else None))
    payload = {
        'ok': bool(ohlc), 'symbol': sym, 'src': 'bot' if in_bot else 'ext', 'reason': reason,
        'tf': tf, 'ob_tf': ob_tf, 'vob_tf': vob_tf,
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
                        'vob_tfs': list(VOB_TFS)})

    @app.route('/api/info/rows')
    def api_info_rows():
        syms = [x for x in (request.args.get('symbols') or '').split(',') if x]
        return jsonify({'ok': True, 'rows': rows_for(syms), 'ts': time.time()})

    @app.route('/api/info/coin')
    def api_info_coin():
        a = request.args
        p = coin_payload(a.get('symbol') or '', tf=a.get('tf'), ob_tf=a.get('ob_tf'),
                         vob_tf=a.get('vob_tf'))
        if p.get('ok') and a.get('sig') and a.get('sig') == p.get('sig'):
            return jsonify({'ok': True, 'same': True, 'sig': p['sig']})
        return jsonify(p)

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
