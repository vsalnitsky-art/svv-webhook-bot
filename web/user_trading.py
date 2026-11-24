"""🤖 ТОРГІВЛЯ З ОСОБИСТОГО КАБІНЕТУ (вимога 24.11).

Дослівно: «Організуй алгоритм відкриття угод за цими сигналами [⚠ Аномалії
обсягу · сценарій входу]. У особистому кабінеті користувача додай
налаштування стосовно торгівлі… На даному етапі це дозволено лише
адміністратору… Має бути контроль куди ідуть сигнали реал чи пепер. Якщо
сигнал був у лонг, а потім прилетів ще один в тому ж напрямку… якщо угода
відкрита, нічого не робимо, якщо прилітає протилежний сигнал, закриваємо
поточну угоду і відкриваємо нову… реєструвати апі ключі біржі із
шифруванням (ключі адміністратора бота перенеси в особистий кабінет).»

• Налаштування — `prefs['trade'] = {'enabled': bool, 'coins': {SYM: 'paper'|'real'}}`
  (ЧИСТА `trade_norm`). Монети — лише з WATCHLIST кабінету.
• Доступ — адміністратор завжди; користувач — лише на книги, які йому
  дозволив адміністратор (🛡 `trade_paper` / `trade_real`). Підроблені prefs
  нічого не дають — `auth.info_trade_subscribers` відкидає монети з
  недозволеною книгою.
• 👤 ХТО ТОРГУЄ (вимога 24.11): адміністратор = Trade Manager БОТА (ключі
  бота); кожен інший користувач — свій ОСОБИСТИЙ Trade Manager
  (`detection/user_tm.py`, дзеркало бота) з ВЛАСНИМИ ключами (`web.user_keys`).
  Ключі бота угоди користувача не бачать ніколи, і навпаки.
• Сигнал — той самий ⚠ «сценарій входу», що йде в Telegram (рушій ШІ
  `ai_signals._anom_personal`), одна дія на НОВИЙ сценарій. Виконання —
  `TradeManager.cab_signal_trade` (тримати / відкрити / реверс).
• Ключі біржі: адміністратор — `web.exchange_keys` (ключі бота);
  користувач — `web.user_keys` (лише свої, лише для свого TM).
"""
import json
import threading
import time
from collections import deque

BOOKS = ('paper', 'real')
BOOK_UA = {'paper': '🧪 paper', 'real': '🏦 реальна'}
SUBS_TTL = 30.0
LOG_MAX = 30

_lock = threading.Lock()
_subs = {'at': 0.0, 'data': []}
_log = deque(maxlen=LOG_MAX)          # останні дії торгівлі з кабінету (для показу)


def _norm_symbol(x):
    try:
        from web.info_portal import norm_symbol
        return norm_symbol(x)
    except Exception:
        s = str(x or '').upper().strip().replace('/', '').replace('-', '')
        return s if s.endswith('USDT') else (s + 'USDT' if s else '')


def trade_norm(raw, wl=None):
    """ЧИСТА: {'enabled': bool, 'coins': {SYM: 'paper'|'real'}}. Монети — лише
    зі списку `wl` (якщо задано), книга — лише paper / real (інше → paper)."""
    raw = raw if isinstance(raw, dict) else {}
    coins = {}
    src = raw.get('coins') if isinstance(raw.get('coins'), dict) else {}
    for k, v in src.items():
        S = _norm_symbol(k)
        if not S or (wl is not None and S not in wl):
            continue
        coins[S] = v if v in BOOKS else 'paper'
    return {'enabled': bool(raw.get('enabled')), 'coins': coins}


def trade_apply(cur, body, wl=None):
    """ЧИСТА: зміна налаштувань торгівлі з тіла запиту
    {enabled?, coin: {symbol, on, book?}}."""
    out = trade_norm(cur, wl)
    if 'enabled' in body:
        out['enabled'] = bool(body.get('enabled'))
    c = body.get('coin')
    if isinstance(c, dict):
        S = _norm_symbol(c.get('symbol'))
        if S and (wl is None or S in wl):
            if c.get('on'):
                b = c.get('book')
                out['coins'][S] = b if b in BOOKS else out['coins'].get(S, 'paper')
            else:
                out['coins'].pop(S, None)
    return trade_norm(out, wl)


def _perms_mod():
    """web.user_perms без важкого пакетного імпорту (`web/__init__` тягне
    flask_app): той самий модуль на процес (`sys.modules`)."""
    import sys
    m = sys.modules.get('web.user_perms')
    if m is not None:
        return m
    try:
        from web import user_perms as m
        return m
    except Exception:
        import importlib.util
        import os
        sp = importlib.util.spec_from_file_location(
            'web.user_perms', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_perms.py'))
        m = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(m)
        sys.modules['web.user_perms'] = m
        return m


def user_books(u):
    """🛡 Книги, у які користувачу ДОЗВОЛЕНО торгувати з кабінету (адмін — усі;
    користувачу — ті, що адміністратор увімкнув у «🛡 Керування користувачами»)."""
    if u is None:
        return ()
    try:
        P = _perms_mod()
        return P.books(P.of_user(u))
    except Exception:
        return BOOKS if getattr(u, 'is_admin', False) else ()


def trade_allowed(u) -> bool:
    """Торгівля з кабінету: адміністратор — завжди, користувач — якщо
    адміністратор надав право хоча б на одну книгу."""
    return bool(user_books(u))


def keys_allowed(u) -> bool:
    """🔐 Ключі БОТА — лише адміністратор."""
    return bool(u is not None and getattr(u, 'is_admin', False))


def keys_owner(u) -> str:
    """🔐 Чиї ключі керує цей користувач у кабінеті: 'bot' (адміністратор —
    ключі бота) | 'user' (власні, для свого TM; потрібне право на реальну
    книгу) | '' (ключів немає)."""
    if u is None:
        return ''
    if getattr(u, 'is_admin', False):
        return 'bot'
    # Trade Manager користувача без ключів не існує → ключі потрібні для будь-якої книги
    return 'user' if user_books(u) else ''


def is_admin_sub(t) -> bool:
    """Підписка адміністратора (торгує TM бота). Старі записи без прапорця —
    як і раніше (адміністратор)."""
    return bool((t or {}).get('admin', True))


def trade_limit_books(t, books):
    """ЧИСТА: прибрати монети з книгою, на яку немає права."""
    t = trade_norm(t)
    t['coins'] = {k: v for k, v in t['coins'].items() if v in (books or ())}
    return t


def trade_plan(subs, symbol):
    """ЧИСТА: {книга: [email…]} — хто торгує монетою і в яку книгу. Угоди
    йдуть через ОДИН Trade Manager бота, тож одна книга = одна дія, навіть
    якщо монету ввімкнули кілька адміністраторів."""
    S = _norm_symbol(symbol)
    out = {}
    for _uid, email, t in subs or []:
        if not is_admin_sub(t):
            continue          # 👤 користувач торгує СВОЇМ TM (`user_plan`)
        t = trade_norm(t)
        if not t['enabled'] or S not in t['coins']:
            continue
        out.setdefault(t['coins'][S], []).append(email)
    return out


def user_plan(subs, symbol):
    """ЧИСТА: [(uid, email, книга)] — користувачі (не адміністратори), чиї
    ОСОБИСТІ Trade Manager торгують монетою."""
    S = _norm_symbol(symbol)
    out = []
    for uid, email, t in subs or []:
        if is_admin_sub(t):
            continue
        tn = trade_norm(t)
        if tn['enabled'] and S in tn['coins']:
            out.append((uid, email, tn['coins'][S]))
    return out


def trade_subs(force=False):
    """[(uid, email, trade)] активних адміністраторів з увімкненою торгівлею
    (кеш 30с)."""
    now = time.time()
    with _lock:
        if not force and now - _subs['at'] < SUBS_TTL:
            return _subs['data']
    try:
        from web.auth import info_trade_subscribers
        data = info_trade_subscribers()
    except Exception:
        data = []
    with _lock:
        _subs['at'], _subs['data'] = now, data
    return data


def trade_subs_invalidate():
    with _lock:
        _subs['at'] = 0.0


def trade_symbols(subs=None):
    """ЧИСТА над підписками: монети, по яких увімкнено торгівлю."""
    out = []
    for _uid, _em, t in (trade_subs() if subs is None else subs) or []:
        t = trade_norm(t)
        if not t['enabled']:
            continue
        for S in t['coins']:
            if S not in out:
                out.append(S)
    return out


def _tm():
    try:
        from detection.trade_manager import get_trade_manager
        return get_trade_manager()
    except Exception:
        return None


def trade_execute(symbol, side, note='', subs=None, tm=None):
    """⚠ Сигнал «сценарій входу» → дія в кожній обраній книзі. → [результат].
    Кличе рушій ШІ на НОВОМУ сценарії (той самий, що йде в Telegram)."""
    subs = trade_subs() if subs is None else subs
    plan = trade_plan(subs, symbol)
    out = []
    for uid, email, book in user_plan(subs, symbol):
        try:
            from detection import user_tm
            r = user_tm.execute(uid, email, symbol, side, book, note=note)
        except Exception as e:
            r = {'ok': False, 'status': 'rejected', 'mode': book, 'reason': str(e)[:200]}
        r = dict(r or {}, symbol=_norm_symbol(symbol), side=side, book=book,
                 by=[email], ts=time.time(), note=note, own=True)
        out.append(r)
        with _lock:
            _log.appendleft(r)
    if not plan:
        return out
    tm = tm or _tm()
    for book in BOOKS:
        if book not in plan:
            continue
        if tm is None:
            r = {'ok': False, 'status': 'rejected', 'mode': book,
                 'reason': 'Trade Manager недоступний'}
        else:
            try:
                r = tm.cab_signal_trade(symbol, side, book, opened_by='anom', note=note)
            except Exception as e:
                r = {'ok': False, 'status': 'rejected', 'mode': book, 'reason': str(e)[:200]}
        r = dict(r or {}, symbol=_norm_symbol(symbol), side=side, book=book,
                 by=plan[book], ts=time.time(), note=note)
        out.append(r)
        with _lock:
            _log.appendleft(r)
    return out


def trade_log():
    with _lock:
        return list(_log)


def _tm_view(tm):
    if tm is None:
        return {'available': False}
    try:
        s = tm.get_settings() if hasattr(tm, 'get_settings') else getattr(tm, '_settings', {})
    except Exception:
        s = {}
    has_key = bool(getattr(getattr(tm, 'bybit', None), 'api_key', None))
    return {'available': True, 'enabled': bool(tm.is_enabled()),
            'test_mode': bool((s or {}).get('test_mode', False)), 'has_key': has_key,
            'max_open': (s or {}).get('max_open_positions')}


_POS_KEYS = ('symbol', 'side', 'entry_price', 'current_price', 'qty', 'pnl_pct',
             'opened_at', 'opened_by', 'manual_sl', 'manual_tp', 'manual_tp1',
             'sl_price', 'tp_price', 'leverage', 'liq_price', 'pnl_net', 'paper_size',
             'cab_avg_n', 'cab_tranches', 'mirror_bot', 'sl_breakeven', 'manual_sl_tf')
_CLOSED_KEYS = ('symbol', 'side', 'entry_price', 'exit_price', 'pnl_pct', 'pnl_usd',
                'opened_at', 'closed_at', 'reason', 'opened_by', 'paper_result',
                'cab_avg_n', 'pnl_source', 'peak_pnl_pct')


def _slim(rows, keys, limit=None):
    out = []
    for p in (rows or []):
        if isinstance(p, dict):
            out.append({k: p[k] for k in keys if k in p})
    return out[-limit:][::-1] if limit else out


def user_tm_view(t, light=False):
    """Стан ОСОБИСТОГО Trade Manager для кабінету (лише дані власника)."""
    from detection import user_tm as _u
    try:
        st = t.get_state()
    except Exception as e:
        st = {'error': str(e)[:200]}
    s = st.get('settings') or t.get_settings()
    v = {'ok': True, 'owner': 'user', 'enabled': bool(s.get('enabled')),
         'test_mode': bool(s.get('test_mode', False)), 'running': bool(st.get('running')),
         'has_key': t.bybit is not None, 'books': list(t.books()),
         'mirror_allowed': t.mirror_allowed(), 'meta': t.meta(),
         'settings': _u.personal_only(s),
         'positions': _slim(st.get('positions'), _POS_KEYS),
         'shadow_positions': _slim(st.get('shadow_positions'), _POS_KEYS),
         'stats': st.get('stats') or {}, 'shadow_stats': st.get('shadow_stats') or {},
         'paper_account': st.get('paper_account') or {}}
    if not light:
        v['closed'] = _slim(st.get('closed_trades'), _CLOSED_KEYS, 40)
        v['shadow_closed'] = _slim(st.get('shadow_closed'), _CLOSED_KEYS, 40)
        v['log'] = t.user_log(60)
        if t.bybit is not None:
            try:
                v['wallet'] = t.wallet_info()
            except Exception:
                v['wallet'] = {'ok': False}
    return v


def register_user_trading(app):
    from flask import jsonify, request

    def _me():
        from web.auth import current_user
        return current_user()

    def _deny(u):
        if u is None:
            return jsonify({'ok': False, 'error': 'auth_required'}), 401
        return jsonify({'ok': False, 'error': 'no_perm', 'allowed': False,
                        'reason': 'Адміністратор не надав доступ'}), 403

    def _wl(prefs):
        try:
            from web.info_portal import user_info, universe
            return user_info(prefs, universe()).get('wl') or []
        except Exception:
            return None

    def _utm(u, create=True):
        """👤 Особистий TM користувача (адміністратор — ні: його TM = бот)."""
        if u is None or getattr(u, 'is_admin', False):
            return None
        from detection import user_tm
        return user_tm.ensure(u.id, u.email) if create else user_tm.get(u.id)

    @app.route('/api/me/trading', methods=['GET', 'POST'])
    def api_me_trading():
        from web.auth import _load_prefs, _update_user
        u = _me()
        if u is None:
            return _deny(u)
        bks = user_books(u)
        if not bks:
            return jsonify({'ok': True, 'allowed': False,
                            'reason': 'Торгівлю з кабінету вмикає адміністратор '
                                      '(🛡 Керування користувачами в боті)'})
        prefs = _load_prefs(u)
        wl = _wl(prefs)
        cur = trade_limit_books(trade_norm(prefs.get('trade'), wl), bks)
        err = ''
        if request.method == 'POST':
            body = request.get_json(silent=True) or {}
            c = body.get('coin')
            if isinstance(c, dict) and c.get('on') and c.get('book') and c.get('book') not in bks:
                err = 'Адміністратор не дозволив книгу ' + BOOK_UA.get(c.get('book'), str(c.get('book')))
                body = {k: v for k, v in body.items() if k != 'coin'}
            cur = trade_limit_books(trade_apply(cur, body, wl), bks)
            prefs['trade'] = cur
            _update_user(u.id, prefs=json.dumps(prefs)[:20000])
            trade_subs_invalidate()
            try:
                from web.info_portal import anom_subs_invalidate
                anom_subs_invalidate()
            except Exception:
                pass
        adm = keys_allowed(u)
        tm = _tm() if adm else _utm(u)
        view = _tm_view(tm)
        if not adm and tm is None:
            view['need_keys'] = True
        view['owner'] = 'bot' if adm else 'user'
        return jsonify({'ok': not err, 'error': err, 'allowed': True, **cur, 'books': list(bks),
                        'keys_allowed': bool(keys_owner(u)), 'keys_owner': keys_owner(u),
                        'is_admin': adm, 'tm': view,
                        # журнал дій — адміністратор бачить усе, користувач — лише свої
                        'log': [r for r in trade_log() if adm or u.email in (r.get('by') or [])][:15]})

    # ── 🔐 ключі: адміністратор — ключі БОТА, користувач — ЛИШЕ свої ─────────
    def _keys_mod(u):
        own = keys_owner(u)
        if own == 'bot':
            from web import exchange_keys
            return own, exchange_keys
        if own == 'user':
            from web import user_keys
            return own, user_keys
        return '', None

    def _out(r, owner):
        return jsonify({**{k: v for k, v in r.items() if k != 'code'}, 'owner': owner}), r.get('code', 200)

    @app.route('/api/me/keys', methods=['GET', 'POST'])
    def api_me_keys():
        u = _me()
        own, m = _keys_mod(u)
        if m is None:
            return _deny(u)
        if request.method == 'POST':
            d = request.get_json(silent=True) or {}
            k, sc = d.get('api_key') or '', d.get('api_secret') or ''
            r = m.save(k, sc) if own == 'bot' else m.save(u.id, k, sc)
            return _out(r, own)
        return _out(m.status() if own == 'bot' else m.status(u.id), own)

    @app.route('/api/me/keys/clear', methods=['POST'])
    def api_me_keys_clear():
        u = _me()
        own, m = _keys_mod(u)
        if m is None:
            return _deny(u)
        return _out(m.clear() if own == 'bot' else m.clear(u.id), own)

    @app.route('/api/me/keys/test', methods=['POST'])
    def api_me_keys_test():
        u = _me()
        own, m = _keys_mod(u)
        if m is None:
            return _deny(u)
        return _out(m.test() if own == 'bot' else m.test(u.id), own)

    # ── 👤 МІЙ TRADE MANAGER ───────────────────────────────────────────────
    def _tm_guard():
        u = _me()
        if u is None:
            return None, None, _deny(u)
        if getattr(u, 'is_admin', False):
            return u, None, (jsonify({'ok': True, 'owner': 'bot',
                                      'reason': 'Ваш Trade Manager — це Trade Manager бота'}), 200)
        if not user_books(u):
            return u, None, _deny(u)
        t = _utm(u)
        if t is None:
            # 🔐 TM зʼявляється лише після підключення ключів API
            return u, None, (jsonify({'ok': True, 'owner': 'user', 'need_keys': True,
                                      'reason': 'Підключіть свої ключі API Bybit — після '
                                                'цього зʼявиться ваш Trade Manager'}), 200)
        return u, t, None

    @app.route('/api/me/tm', methods=['GET'])
    def api_me_tm():
        u, t, err = _tm_guard()
        if err:
            return err
        return jsonify(user_tm_view(t, request.args.get('light') == '1'))

    @app.route('/api/me/tm/settings', methods=['POST'])
    def api_me_tm_settings():
        u, t, err = _tm_guard()
        if err:
            return err
        d = request.get_json(silent=True) or {}
        msg = []
        meta = d.get('meta') if isinstance(d.get('meta'), dict) else {}
        if meta:
            meta = dict(meta)
            if meta.get('mirror_signals') and not t.mirror_allowed():
                meta.pop('mirror_signals', None)
                msg.append('🪞 Дублювання угод бота не дозволено адміністратором')
            if meta.get('mirror_book') and meta.get('mirror_book') not in t.books():
                meta.pop('mirror_book', None)
                msg.append('Книга дублювання не дозволена адміністратором')
            t.set_meta(meta)
        st = d.get('settings') if isinstance(d.get('settings'), dict) else {}
        if st:
            st = dict(st)
            if st.get('enabled') and 'real' not in t.books():
                st.pop('enabled', None)
                msg.append('Реальну торгівлю не дозволено адміністратором')
            if st.get('test_mode') and 'paper' not in t.books():
                st.pop('test_mode', None)
                msg.append('Paper-книгу не дозволено адміністратором')
            from detection.activity_log import owner_scope
            with owner_scope(t._log_sink):
                t.update_personal(st)
        return jsonify({**user_tm_view(t, True), 'note': ' · '.join(msg)})

    def _sym_side_book(d, t):
        S = _norm_symbol(d.get('symbol'))
        sd = str(d.get('side') or '').upper()
        bk = d.get('book')
        if not S or bk not in BOOKS:
            return None, None, None, 'невірні параметри'
        if bk not in t.books():
            return None, None, None, 'Адміністратор не дозволив книгу ' + BOOK_UA[bk]
        return S, sd, bk, ''

    @app.route('/api/me/tm/open', methods=['POST'])
    def api_me_tm_open():
        u, t, err = _tm_guard()
        if err:
            return err
        S, sd, bk, why = _sym_side_book(request.get_json(silent=True) or {}, t)
        if why or sd not in ('LONG', 'SHORT'):
            return jsonify({'ok': False, 'reason': why or 'side: LONG / SHORT'}), 400
        from detection import user_tm
        r = user_tm.hand_open(u.id, u.email, S, sd, bk)
        return jsonify(r), (200 if r.get('ok') else 400)

    @app.route('/api/me/tm/add', methods=['POST'])
    def api_me_tm_add():
        u, t, err = _tm_guard()
        if err:
            return err
        S, _sd, bk, why = _sym_side_book(request.get_json(silent=True) or {}, t)
        if why:
            return jsonify({'ok': False, 'reason': why}), 400
        from detection import user_tm
        r = user_tm.hand_add(u.id, u.email, S, bk)
        return jsonify(r), (200 if r.get('ok') else 400)

    @app.route('/api/me/tm/close', methods=['POST'])
    def api_me_tm_close():
        u, t, err = _tm_guard()
        if err:
            return err
        d = request.get_json(silent=True) or {}
        S = _norm_symbol(d.get('symbol'))
        bk = d.get('book')
        if not S or bk not in BOOKS:
            return jsonify({'ok': False, 'reason': 'невірні параметри'}), 400
        from detection import user_tm
        r = user_tm.hand_close(u.id, u.email, S, bk)
        return jsonify(r), (200 if r.get('ok') else 400)
