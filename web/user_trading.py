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
  дозволив адміністратор (🛡 `trade_paper` / `trade_real`, вимога 24.11).
  Угоди йдуть через Trade Manager БОТА (ключі й депозит бота). Підроблені
  prefs нічого не дають — `auth.info_trade_subscribers` відкидає монети з
  недозволеною книгою. 🔐 Ключі біржі — лише адміністратор.
• Сигнал — той самий ⚠ «сценарій входу», що йде в Telegram (рушій ШІ
  `ai_signals._anom_personal`), одна дія на НОВИЙ сценарій. Виконання —
  `TradeManager.cab_signal_trade` (тримати / відкрити / реверс).
• Ключі біржі — `web.exchange_keys` (Fernet → БД; ті самі, якими торгує бот).
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
    """🔐 Ключі біржі — це ключі БОТА, тож лише адміністратор."""
    return bool(u is not None and getattr(u, 'is_admin', False))


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
        t = trade_norm(t)
        if not t['enabled'] or S not in t['coins']:
            continue
        out.setdefault(t['coins'][S], []).append(email)
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
    plan = trade_plan(trade_subs() if subs is None else subs, symbol)
    if not plan:
        return []
    tm = tm or _tm()
    out = []
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
            'test_mode': bool((s or {}).get('test_mode', True)), 'has_key': has_key,
            'max_open': (s or {}).get('max_open_positions')}


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
        return jsonify({'ok': not err, 'error': err, 'allowed': True, **cur, 'books': list(bks),
                        'keys_allowed': adm, 'is_admin': adm,
                        'tm': _tm_view(_tm()),
                        # журнал дій — спільний для всіх; користувач бачить лише свої
                        'log': [r for r in trade_log() if adm or u.email in (r.get('by') or [])][:15]})

    @app.route('/api/me/keys', methods=['GET', 'POST'])
    def api_me_keys():
        u = _me()
        if not keys_allowed(u):
            return _deny(u)
        from web import exchange_keys
        if request.method == 'POST':
            d = request.get_json(silent=True) or {}
            r = exchange_keys.save(d.get('api_key') or '', d.get('api_secret') or '')
            return jsonify({k: v for k, v in r.items() if k != 'code'}), r.get('code', 200)
        return jsonify(exchange_keys.status())

    @app.route('/api/me/keys/clear', methods=['POST'])
    def api_me_keys_clear():
        u = _me()
        if not keys_allowed(u):
            return _deny(u)
        from web import exchange_keys
        r = exchange_keys.clear()
        return jsonify({k: v for k, v in r.items() if k != 'code'}), r.get('code', 200)

    @app.route('/api/me/keys/test', methods=['POST'])
    def api_me_keys_test():
        u = _me()
        if not keys_allowed(u):
            return _deny(u)
        from web import exchange_keys
        r = exchange_keys.test()
        return jsonify({k: v for k, v in r.items() if k != 'code'}), r.get('code', 200)
