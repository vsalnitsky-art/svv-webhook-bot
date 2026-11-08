"""👤 ІНФОСАЙТ — ОСОБИСТА СТОРІНКА КОРИСТУВАЧА (вимога 08.11).

Дослівно: «Це буде особиста сторінка кожного користувача окремо… Для кожного
користувача в базі має бути окрема збережена інформація… вигляд як у боті
WATCHLIST із своїм списком монет і графік… (для початку лише назва монети,
🟢 Discount 18% 1H, POC), а на графіку структуру OB і VOB.»

ГОЛОВНЕ ПРАВИЛО НАВАНТАЖЕННЯ: вартість НЕ росте з кількістю користувачів.
- Монета з WATCHLIST бота (`universe`): усе, що показуємо, сканер УЖЕ рахує у
  своєму робочому такті (бари, OB 1H, VOB, PD). Ми лише ЧИТАЄМО памʼять.
  Тисяча користувачів = нуль додаткових запитів до біржі за цими даними.
- Монета, якої в боті НЕМАЄ (вимога 08.11: «все одно обробити»), — рахується
  ТИМИ САМИМИ функціями детекції з ТИМИ САМИМИ параметрами сканера, але в
  ОДНОМУ фоновому потоці (`_ext_worker`) зі спільним кешем на монету: бари
  головного TF раз на `EXT_FAST_TTL`, VOB — раз на закриття бару свого TF,
  OB 1H і PD — раз на закриття 1h-бару. Навантаження = кількість УНІКАЛЬНИХ
  таких монет, а не людей; монета, яку ніхто не відкривав `EXT_IDLE_SEC`,
  більше не рахується (стеля `EXT_MAX`). Запит сторінки біржу НЕ чекає —
  лише читає кеш («⏳ рахується» до першого результату).
- Єдине, чого сканер не тримає постійно, — POC. Він рахується через ТОЙ САМИЙ
  `volume_profile.compute_poc`, що бейдж графіка і POC-фільтр, але зі СПІЛЬНИМ
  кешем на монету (`POC_TTL`), і рахує його лише фоновий потік. Тобто на біржу
  йде ≤ 1 запит на монету раз на 5 хв — незалежно від того, скільки людей
  дивиться, і жоден HTTP-запит сторінки біржу не чекає.
- Відповідь графіка кешується на монету (`COIN_TTL`) і має сигнатуру `sig`:
  клієнт шле свою — незмінна відповідь іде порожньою (`same: true`).

Особисті дані (список монет і обрана монета) — у `User.prefs['info']`, тобто в
БД ОКРЕМО на кожного користувача. Писати може лише сам користувач (/api/me/…).
"""
import json
import threading
import time

IWL_MAX = 40                 # стеля особистого списку
IWL_DEFAULT = ('BTCUSDT', 'ETHUSDT')
POC_TTL = 300.0              # спільний кеш POC на монету
COIN_TTL = 10.0              # спільний кеш відповіді графіка на монету
OHLC_BARS = 400              # скільки барів віддаємо на графік
# ── монети ПОЗА ботом ──
EXT_FAST_TTL = 30.0          # бари головного TF (живий бар)
EXT_IDLE_SEC = 900.0         # ніхто не відкривав 15 хв → не рахуємо
EXT_MAX = 300                # стеля одночасно відстежуваних монет поза ботом
EXT_RETRY_SEC = 120.0        # біржа не віддала / монети немає → повтор
EXT_OB_BARS = 700            # як у сканера (_update_smc_ob)

_lock = threading.Lock()
_poc_cache = {}              # SYM → (ts, view|None)
_coin_cache = {}             # SYM → (ts, payload)


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


def iwl_norm(lst, universe=None, cap=IWL_MAX):
    """Особистий список: верхній регістр, без дублів, стеля. `universe` (якщо
    задано) — обмежити лише монетами бота; None → будь-який валідний символ."""
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


def ob_boxes(zones):
    """Знімок OB-сімей сканера → бокси для графіка (без дублів між сім'ями)."""
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
                        'pct': z.get('pct')})
        except (KeyError, TypeError, ValueError):
            continue
    return out


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


def sig_of(payload):
    o = payload.get('ohlc') or []
    last = o[-1] if o else {}
    return '|'.join(str(x) for x in (
        len(o), last.get('time'), last.get('close'),
        len(payload.get('ob') or []), (payload.get('ob') or [{}])[0].get('start'),
        len(payload.get('vob') or []), (payload.get('vob') or [{}])[0].get('start'),
        (payload.get('pd') or {}).get('pct'), (payload.get('poc') or {}).get('poc')))


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


def _price(sc, sym):
    try:
        if not _in_bot(sc, sym):
            kl = (_ext_get(sym).get('main') or [])
        else:
            with sc._lock:
                c = sc._cache.get(sym)
            kl = (c or {}).get('klines') or []
        return float(kl[-1].get('p') or 0) if kl else None
    except Exception:
        return None


def _pd(sc, sym):
    try:
        prem, disc = sc.get_pd_thresholds()
        pct = sc.get_pd_pct(sym) if _in_bot(sc, sym) else _ext_get(sym).get('pd_pct')
        return pd_view(pct, prem, disc, sc._settings.get('pd_zone_timeframe', '1h'))
    except Exception:
        return None


# ── 🛰 МОНЕТИ ПОЗА БОТОМ — власний розрахунок тими самими функціями ──────
_ext = {}                    # SYM → {'main','main_at','vz','vob_at','zones','pd_pct','slow_at','err','err_at'}
_ext_want = {}               # SYM → коли востаннє просили (монети поза ботом)
_poc_want = {}               # SYM → коли востаннє просили POC (будь-яка монета)
_ext_thread = None


def _bar_sec(tf):
    t = str(tf or '').lower()
    try:
        n = int(t[:-1])
        return n * {'m': 60, 'h': 3600, 'd': 86400}[t[-1]]
    except Exception:
        return 3600


def _ext_get(sym):
    with _lock:
        return dict(_ext.get(sym) or {})


def _kick():
    global _ext_thread
    if _ext_thread is None or not _ext_thread.is_alive():
        _ext_thread = threading.Thread(target=_ext_worker, name='info-ext', daemon=True)
        _ext_thread.start()


def poc_want(sym):
    with _lock:
        if sym in _poc_want or len(_poc_want) < EXT_MAX * 2:
            _poc_want[sym] = time.time()
            _kick()


def ext_want(sym):
    """Сторінка попросила монету поза ботом → поставити в роботу (без чекання)."""
    now = time.time()
    with _lock:
        if sym not in _ext_want and len(_ext_want) >= EXT_MAX:
            return False
        _ext_want[sym] = now
        _kick()
    return True


def ext_due(rec, now, vol_tf, slow_tf):
    """ЧИСТА: які частини монети пора перерахувати → set('main'|'vob'|'slow')."""
    rec = rec or {}
    if rec.get('err') and now - float(rec.get('err_at') or 0) < EXT_RETRY_SEC:
        return set()
    due = set()
    if now - float(rec.get('main_at') or 0) >= EXT_FAST_TTL:
        due.add('main')
    if int(now // _bar_sec(vol_tf)) != int(float(rec.get('vob_at') or 0) // _bar_sec(vol_tf)):
        due.add('vob')
    if int(now // _bar_sec(slow_tf)) != int(float(rec.get('slow_at') or 0) // _bar_sec(slow_tf)):
        due.add('slow')
    return due


def _ext_compute(sc, sym, due):
    """Рахуємо ТИМИ САМИМИ функціями й параметрами, що сканер."""
    from detection.market_data import get_market_data
    from detection.smc_structure import detect_smc_structure
    md = get_market_data()
    st = sc._settings
    isize = sc.get_internal_size()
    ssize = int(st.get('swing_size', 50))
    upd = {}
    now = time.time()
    if 'main' in due:
        kl = md.fetch_klines(sym, limit=OHLC_BARS, interval=sc.get_timeframe())
        if not kl or len(kl) < 20:
            raise ValueError('монету не знайдено на біржі (немає барів)')
        upd['main'], upd['main_at'] = kl, now
    if 'vob' in due:
        vz = []
        if st.get('use_volumized_ob', True):
            from detection.volumized_ob import detect_volumized_obs
            vtf = st.get('volumized_timeframe', '1h')
            vk = md.fetch_klines(sym, limit=sc.VOB_KLINES_LIMIT, interval=vtf)
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
                vz = [z for z in (sc._format_vob(o, vtf) for o in obs) if z]
        upd['vz'], upd['vob_at'] = vz, now
    if 'slow' in due:
        from detection.ob_detector import detect_order_blocks
        ob_tf = st.get('ob_filter_timeframe', '1h')
        ok = md.fetch_klines(sym, limit=EXT_OB_BARS, interval=ob_tf)
        zones = {}
        if ok and len(ok) > 60:
            cl = ok[:-1]
            res = detect_smc_structure(cl, internal_size=isize, swing_size=ssize)
            zones = {'tf': ob_tf}
            for fam in ('internal', 'swing'):
                f = res.get(fam, {}) or {}
                zones[fam] = detect_order_blocks(klines=cl, pivots=f.get('pivots', []),
                                                 events=f.get('events', []),
                                                 limit=getattr(sc, 'OB_ZONES_SHOW', 8))
        pd_tf = st.get('pd_zone_timeframe', '1h')
        pk = md.fetch_klines(sym, limit=3000, interval=pd_tf)
        pd_pct = None
        if pk and len(pk) > 60:
            pcl = pk[:-1]
            pres = detect_smc_structure(pcl, internal_size=isize, swing_size=ssize)
            pd_pct = sc._compute_pd_pct(pcl, (pres.get('swing', {}) or {}).get('pivots', []),
                                        pcl[-1].get('p'))
        upd['zones'], upd['pd_pct'], upd['slow_at'] = zones, pd_pct, now
    return upd


def _ext_worker():
    """ОДИН потік на всі монети поза ботом — біржу не смикаємо паралельно."""
    global _ext_thread
    while True:
        try:
            sc = _scanner()
            now = time.time()
            with _lock:
                for k in [k for k, t in _ext_want.items() if now - t > EXT_IDLE_SEC]:
                    _ext_want.pop(k, None)
                    _ext.pop(k, None)
                for k in [k for k, t in _poc_want.items() if now - t > EXT_IDLE_SEC]:
                    _poc_want.pop(k, None)
                want = list(_ext_want)
                pwant = list(_poc_want)
                if not want and not pwant:
                    _ext_thread = None
                    return
            if sc is None:
                time.sleep(5)
                continue
            uni = set(universe(sc))
            st = sc._settings
            vtf = st.get('volumized_timeframe', '1h')
            slow_tf = st.get('ob_filter_timeframe', '1h')
            for sym in want:
                if sym in uni:
                    continue                      # бот уже веде цю монету
                due = ext_due(_ext_get(sym), time.time(), vtf, slow_tf)
                if not due:
                    continue
                try:
                    upd = _ext_compute(sc, sym, due)
                    upd['err'] = ''
                except Exception as e:
                    upd = {'err': str(e)[:160], 'err_at': time.time()}
                with _lock:
                    if sym in _ext_want:
                        _ext.setdefault(sym, {}).update(upd)
                        _coin_cache.pop(sym, None)
            for sym in pwant:
                with _lock:
                    hit = _poc_cache.get(sym)
                if hit and time.time() - hit[0] < POC_TTL:
                    continue
                _poc(sc, sym, allow_compute=True)
                with _lock:
                    _coin_cache.pop(sym, None)
            time.sleep(2.0)
        except Exception as e:
            print(f"[INFO] ext worker: {e}")
            time.sleep(5)


def _poc_params(sc):
    s = getattr(sc, '_settings', {}) or {}
    try:
        days = float(s.get('poc_filter_window_days', 3) or 3)
    except (TypeError, ValueError):
        days = 3.0
    return (str(s.get('poc_filter_market', 'futures') or 'futures'),
            str(s.get('poc_filter_tf', '1h') or '1h'), days)


def _poc(sc, sym, allow_compute=True):
    """POC через ТОЙ САМИЙ compute_poc, що бейдж графіка, зі спільним кешем.
    Вікно прибите до години (to = поточна година) — ключ кешу стабільний."""
    now = time.time()
    computed = False
    with _lock:
        hit = _poc_cache.get(sym)
    if hit and now - hit[0] < POC_TTL:
        view = hit[1]
    elif not allow_compute:
        return ('pending', None, False)
    else:
        computed = True
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
        return ('none', None, computed)
    out = dict(view)
    price = _price(sc, sym)
    try:
        from detection.volume_profile import price_vs_poc
        v = price_vs_poc(out['poc'], price)
        out['rel'], out['dist_pct'] = v.get('rel'), v.get('dist_pct')
    except Exception:
        out['rel'] = out['dist_pct'] = None
    return ('ok', out, computed)


def rows_for(symbols, sc=None):
    sc = sc or _scanner()
    uni = set(universe(sc))
    rows = []
    for sym in iwl_norm(symbols):
        src = 'bot' if sym in uni else 'ext'
        ext_err = ""
        if src == 'ext':
            ext_want(sym)
            ext_err = _ext_get(sym).get('err') or ""
        poc_want(sym)
        st, poc, _c = _poc(sc, sym, allow_compute=False)
        rows.append({'symbol': sym, 'price': _price(sc, sym), 'pd': _pd(sc, sym),
                     'poc': poc, 'poc_state': st, 'src': src, 'err': ext_err})
    return rows


def coin_payload(sym, sc=None):
    sc = sc or _scanner()
    sym = norm_symbol(sym)
    if not sc:
        return {'ok': False, 'reason': 'сканер бота недоступний'}
    if not valid_symbol(sym):
        return {'ok': False, 'reason': (sym or "?") + ' — некоректна назва монети'}
    in_bot = _in_bot(sc, sym)
    if not in_bot:
        ext_want(sym)
    now = time.time()
    with _lock:
        hit = _coin_cache.get(sym)
    if hit and now - hit[0] < COIN_TTL:
        return hit[1]
    if in_bot:
        with sc._lock:
            c = sc._cache.get(sym)
            zones = dict(sc._ob_zones.get(sym) or {}) if hasattr(sc, '_ob_zones') else {}
        ohlc = ohlc_of((c or {}).get('klines'))
        try:
            vz = sc._volumized_zones(sym) if sc._settings.get('use_volumized_ob', True) else []
        except Exception:
            vz = []
        reason = "" if ohlc else "сканер ще не дійшов до монети — дані зʼявляться за хвилину"
    else:
        rec = _ext_get(sym)
        ohlc = ohlc_of(rec.get('main'))
        zones, vz = dict(rec.get('zones') or {}), list(rec.get('vz') or [])
        reason = (rec.get('err') or "") if not ohlc else ""
        if not ohlc and not reason:
            reason = "⏳ монети немає в боті — рахуємо окремо, кілька секунд"
    poc_want(sym)
    _st, poc, _c = _poc(sc, sym, allow_compute=False)
    try:
        tf_lbl = sc.get_display_label()
    except Exception:
        tf_lbl = ''
    payload = {
        'ok': bool(ohlc), 'symbol': sym, 'tf': tf_lbl, 'src': 'bot' if in_bot else 'ext',
        'reason': reason,
        'ohlc': ohlc,
        'ob': ob_boxes(zones), 'ob_tf': str(zones.get('tf') or '').upper(),
        'vob': vob_boxes(vz),
        'pd': _pd(sc, sym), 'poc': poc, 'ts': now,
    }
    payload['sig'] = sig_of(payload)
    if not ohlc:
        return payload                 # «рахується» не кешуємо
    with _lock:
        _coin_cache[sym] = (now, payload)
        if len(_coin_cache) > 600:
            _coin_cache.clear()
    return payload


# ── Особисті дані користувача ─────────────────────────────────────────────
def user_info(prefs, uni):
    info = (prefs or {}).get('info') or {}
    wl = info.get('wl')
    if wl is None:                       # перший вхід — стартовий набір
        wl = list(IWL_DEFAULT)
    wl = iwl_norm(wl)
    sel = norm_symbol(info.get('sel'))
    if sel not in wl:
        sel = wl[0] if wl else ''
    return {'wl': wl, 'sel': sel}


def apply_change(cur, body, uni):
    """ЧИСТА: зміна особистого списку з тіла запиту {add|remove|set|sel}."""
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
    return {'wl': wl, 'sel': sel}, err


def register_info_portal(app):
    from flask import jsonify, request

    def _me():
        from web.auth import current_user
        return current_user()

    @app.route('/api/info/universe')
    def api_info_universe():
        """Монети WATCHLIST бота — підказка при додаванні (дані вже пораховані).
        Додати можна й будь-яку іншу USDT-монету — її порахує фоновий потік."""
        return jsonify({'ok': True, 'symbols': sorted(universe())})

    @app.route('/api/info/rows')
    def api_info_rows():
        syms = [x for x in (request.args.get('symbols') or '').split(',') if x]
        return jsonify({'ok': True, 'rows': rows_for(syms), 'ts': time.time()})

    @app.route('/api/info/coin')
    def api_info_coin():
        p = coin_payload(request.args.get('symbol') or '')
        if p.get('ok') and request.args.get('sig') and request.args.get('sig') == p.get('sig'):
            return jsonify({'ok': True, 'same': True, 'sig': p['sig'],
                            'pd': p.get('pd'), 'poc': p.get('poc')})
        return jsonify(p)

    @app.route('/api/me/watchlist', methods=['GET', 'POST'])
    def api_me_watchlist():
        from web.auth import _load_prefs, _update_user
        u = _me()
        if u is None:
            return jsonify({'ok': False, 'error': 'auth_required'}), 401
        uni = universe()
        prefs = _load_prefs(u)
        cur = user_info(prefs, uni)
        err = ''
        if request.method == 'POST':
            cur, err = apply_change(cur, request.get_json(silent=True) or {}, uni)
            prefs['info'] = cur
            _update_user(u.id, prefs=json.dumps(prefs)[:20000])
        return jsonify({'ok': not err, 'error': err, 'wl': cur['wl'], 'sel': cur['sel'],
                        'max': IWL_MAX, 'universe_n': len(uni)})
