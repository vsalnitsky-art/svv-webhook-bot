"""
Professional authentication & access control for the bot's web app.

Built entirely on the STDLIB + what Flask already ships (Werkzeug for password
hashing, itsdangerous for signed time-limited tokens, Flask signed-cookie
sessions) — NO extra dependencies.

Model (single shared bot, ACCESS CONTROL — not multi-tenant):
  • The whole site is gated: nothing is reachable until authenticated.
  • A new account becomes ACTIVE only when BOTH:
        email_confirmed == True   (proved email ownership via link)
        approved        == True   (an admin approved it)
    …unless the account is_admin (admins are always active).
  • Non-admin users have LIMITED rights: they may VIEW (GET) everything and
    manage their OWN profile/UI prefs, but every state-changing action
    (POST/PUT/DELETE/PATCH) is ADMIN-ONLY — no trading, no bot settings.
  • First admin is bootstrapped from env ADMIN_EMAIL (+ ADMIN_PASSWORD).
  • Email delivery via SMTP env vars; if SMTP is not configured, links are
    logged + surfaced to the admin panel (dev fallback) so nothing blocks.
"""
import os
import re
import json
import ssl
import time
import secrets
import smtplib
import threading
from datetime import datetime, timedelta
from email.message import EmailMessage

from flask import (session, request, redirect, url_for, jsonify, g,
                   render_template_string, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from storage.db_models import get_session, User

# ---- policy constants ----
MIN_PASSWORD_LEN = 8
DEFAULT_ACCESS_DAYS = 30          # auto access window granted on approval
CONFIRM_MAX_AGE = 24 * 3600      # email-confirm token valid 24h
RESET_MAX_AGE = 3600             # password-reset token valid 1h
LOGIN_MAX_FAILS = 8              # per email+ip within window
LOGIN_WINDOW = 900               # 15 min
_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

# Dev fallback: last confirm/reset link per email (shown to admin when SMTP off)
_pending_links = {}
_login_fails = {}                # {(email,ip): [timestamps]}
# Last SMTP attempt outcome — surfaced in the admin panel for diagnostics.
_smtp_last = {'ok': None, 'error': None, 'at': None, 'to': None, 'stage': None}
_lock = threading.Lock()


# ==================================================================
# Serializer / tokens
# ==================================================================
def _serializer(salt):
    from flask import current_app
    secret = current_app.secret_key or 'change-me'
    return URLSafeTimedSerializer(secret, salt=salt)


def _make_token(email, salt):
    return _serializer(salt).dumps(email.lower())


def _read_token(token, salt, max_age):
    try:
        return _serializer(salt).loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired, Exception):
        return None


# --- Telegram-link tokens: env-based secret so they also work in the poller
# thread (no Flask app context). AUTH_SECRET_KEY == app.secret_key here. ---
def _tg_secret():
    return (os.getenv('AUTH_SECRET_KEY') or os.getenv('FLASK_SECRET_KEY')
            or 'change-me')


def _make_tg_token(chat_id, username=None, name=None, lang=None, premium=None):
    return URLSafeTimedSerializer(_tg_secret(), salt='tglink').dumps(
        {'c': str(chat_id), 'u': username, 'n': name,
         'l': lang, 'p': bool(premium) if premium is not None else None})


def _read_tg_token(token, max_age=3600):
    """Returns {'c': chat_id, 'u': username, 'n': name, 'l': lang, 'p': premium}
    or None. Tolerates legacy tokens (bare chat_id string, or dicts w/o extras)."""
    try:
        v = URLSafeTimedSerializer(_tg_secret(), salt='tglink').loads(
            token, max_age=max_age)
        if isinstance(v, dict):
            return {'c': v.get('c'), 'u': v.get('u'), 'n': v.get('n'),
                    'l': v.get('l'), 'p': v.get('p')}
        return {'c': str(v), 'u': None, 'n': None, 'l': None, 'p': None}
    except Exception:
        return None


# ── Info-site auth token ────────────────────────────────────────────────────
# A signed, time-limited token that carries a user id. It lets the SEPARATELY
# HOSTED info-site authenticate cross-origin WITHOUT cookies: after the user
# logs in on the bot, the bot hands the site this token (URL hash); the site
# then sends it as the «X-Info-Token» header and the bot recognises the user.
_INFO_TOK_MAX_AGE = 7 * 24 * 3600     # 7 days


def _make_info_token(uid):
    return URLSafeTimedSerializer(_tg_secret(), salt='infotok').dumps({'u': int(uid)})


def _read_info_token(token, max_age=_INFO_TOK_MAX_AGE):
    v = _read_info_token_ts(token, max_age)
    return v[0] if v else None


def _read_info_token_ts(token, max_age=_INFO_TOK_MAX_AGE):
    """(uid, коли видано — epoch-секунди) або None."""
    try:
        v, ts = URLSafeTimedSerializer(_tg_secret(), salt='infotok').loads(
            token, max_age=max_age, return_timestamp=True)
        if not (isinstance(v, dict) and 'u' in v):
            return None
        try:
            if getattr(ts, 'tzinfo', None) is None:      # itsdangerous 1.x: naive UTC
                from datetime import timezone as _tz
                ts = ts.replace(tzinfo=_tz.utc)
            issued = ts.timestamp()
        except Exception:
            issued = 0.0
        return int(v['u']), float(issued)
    except Exception:
        return None


def info_token_revoked(prefs, issued):
    """ЧИСТА: токен інфосайту видано ДО виходу користувача → недійсний.
    `info_logout_at` пишеться на «Вихід» (вимога 08.11: «Вихід не відбувається
    … потім на сайт відбувається автоматично»)."""
    try:
        lo = float((prefs or {}).get('info_logout_at') or 0)
    except (TypeError, ValueError):
        lo = 0.0
    return lo > 0 and float(issued or 0) < lo


def _revoke_info_tokens(u):
    """Усі видані досі токени інфосайту цього користувача — недійсні."""
    try:
        prefs = json.loads(u.prefs or '{}')
    except Exception:
        prefs = {}
    prefs['info_logout_at'] = int(time.time()) + 1
    _update_user(u.id, prefs=json.dumps(prefs)[:20000])


# ==================================================================
# User store (thin, session-scoped)
# ==================================================================
def _norm_email(e):
    return (e or '').strip().lower()


def get_user_by_email(email):
    s = get_session()
    try:
        return s.query(User).filter_by(email=_norm_email(email)).first()
    finally:
        s.close()


def get_user_by_id(uid):
    s = get_session()
    try:
        return s.query(User).filter_by(id=int(uid)).first()
    except (TypeError, ValueError):
        return None
    finally:
        s.close()


def get_user_by_chat(chat_id):
    """Compact dict for the Telegram bot (avoids detached-instance issues)."""
    s = get_session()
    try:
        u = s.query(User).filter_by(telegram_chat_id=str(chat_id)).first()
        if not u:
            return None
        return {'id': u.id, 'email': u.email, 'active': _is_active(u),
                'approved': u.approved, 'is_admin': u.is_admin,
                'tg_user': getattr(u, 'telegram_username', None),
                'tg_name': getattr(u, 'telegram_name', None)}
    except Exception:
        return None
    finally:
        s.close()


def notify_new_user_to_admin(uid):
    """Ping the admin's Telegram with inline Approve/Reject for a new signup."""
    try:
        if not os.getenv('TELEGRAM_CHAT_ID'):
            return
        u = get_user_by_id(uid)
        if not u:
            return
        from web.tg_bot import notify_category
        tg_bits = ' '.join(x for x in [
            getattr(u, 'telegram_name', None),
            (f'@{u.telegram_username}' if getattr(u, 'telegram_username', None) else '')
        ] if x) or ('прив’язано' if u.telegram_chat_id else '—')
        # Richer Telegram profile captured at registration (lang / Premium / id).
        _tg = (_load_prefs(u).get('tg') or {})
        _extra = []
        if _tg.get('lang'):
            _extra.append(f"🌍 {_tg['lang']}")
        if _tg.get('premium'):
            _extra.append("⭐ Premium")
        if u.telegram_chat_id:
            _extra.append(f"id <code>{u.telegram_chat_id}</code>")
        extra_line = ("\n" + " · ".join(_extra)) if _extra else ""
        notify_category('register',
                f"🆕 <b>Нова реєстрація</b>\nЛогін: <code>{u.email}</code>\n"
                f"Telegram: {tg_bits}{extra_line}\n"
                f"Схвалити доступ?",
                buttons=[[{'text': '✓ Схвалити', 'callback_data': f'ap:{u.id}'},
                          {'text': '✗ Відхилити', 'callback_data': f'rj:{u.id}'}]])
    except Exception as e:
        print(f"[AUTH][TG] notify admin error: {e}")


def approve_user(uid, days=DEFAULT_ACCESS_DAYS):
    """Approve + enable a user, auto-grant a `days`-long access window (unless
    one is already set or the user is admin), then notify them via Telegram.
    Shared by the web panel and the Telegram Approve button."""
    u = get_user_by_id(uid)
    if not u:
        return False
    fields = {'approved': True, 'disabled': False}
    if not u.is_admin and days and not getattr(u, 'access_until', None):
        fields['access_until'] = datetime.utcnow() + timedelta(days=int(days))
    _update_user(uid, **fields)
    # Стан доступу не відкочуємо через збій Telegram. Повертаємо саме результат
    # доставки, щоб адмінська панель і callback не показували хибний успіх.
    return notify_user_approved(uid)


def sync_user_group(uid, force=False):
    """Привести членство користувача в Telegram-групі до його доступу на сайті
    (заблоковано → бан у кожному чаті бота; знову активний → зняти бан +
    запрошення). `force` — забути позначку стану й зробити дію наново (кнопка
    «🔄 Синхронізувати з групою»). Best-effort. → к-сть дій."""
    try:
        if force:
            u = get_user_by_id(uid)
            if u is not None and is_blocked(u):
                set_user_pref(uid, 'grp_state', None)        # → бан наново скрізь
            elif u is not None and _is_active(u) and u.telegram_chat_id:
                from web.tg_bot import group_unban
                group_unban(u.telegram_chat_id)              # залишковий бан геть
        from web.tg_bot import group_access_sync
        return group_access_sync(uid)
    except Exception as e:
        print(f"[AUTH] sync_user_group error: {e}")
        return 0


def tg_group_member_event(tg_id, ev):
    """НАВПАКИ (вимога 23.11): адміністратор Telegram-групи забанив / зняв бан
    → акаунт у боті вимикається / вмикається. `ev` = 'banned' | 'unbanned'
    (`tg_bot.tg_member_event`). Адміністратора бота не чіпаємо.
    → 'disabled' | 'enabled' | 'enabled_blocked' | None."""
    s = get_session()
    try:
        u = s.query(User).filter_by(telegram_chat_id=str(tg_id)).first()
        if u is None or u.is_admin:
            return None
        uid, lbl, was_off = u.id, u.email, bool(u.disabled)
    finally:
        s.close()
    from web.tg_bot import tg_send, notify_category
    from web.i18n import user_tr
    out = None
    if ev == 'banned':
        if was_off:
            sync_user_group(uid)                   # уже вимкнено — лише бан у решті чатів
            return None
        _update_user(uid, disabled=True, session_token=None)
        set_user_pref(uid, 'tg_banned_at', datetime.utcnow().isoformat())
        try:
            tg_send(tg_id, user_tr(uid, 'account_disabled'))
            notify_category('register', f"🚫 <b>{lbl}</b> — забанено в Telegram-групі "
                                        f"→ акаунт у боті ВИМКНЕНО.")
        except Exception:
            pass
        sync_user_group(uid)                       # бан і в решті чатів бота
        return 'disabled'
    if ev == 'unbanned':
        if was_off:
            _update_user(uid, disabled=False)
        set_user_pref(uid, 'tg_banned_at', None)
        u2 = get_user_by_id(uid)
        still = bool(u2 is not None and is_blocked(u2))
        if still:
            # доступ однаково не активний (строк минув / схвалення знято) →
            # бан повернеться; адміну — чому
            set_user_pref(uid, 'grp_state', None)
            out = 'enabled_blocked' if was_off else None
            try:
                notify_category('register', f"⚠️ <b>{lbl}</b> — бан у Telegram-групі знято, "
                                            f"але доступ не активний (строк минув або схвалення "
                                            f"знято) → у групі знову бан. Продовжте термін у "
                                            f"🛡 Керування користувачами.")
            except Exception:
                pass
        elif was_off:
            out = 'enabled'
            try:
                tg_send(tg_id, user_tr(uid, 'account_enabled'))
                notify_category('register', f"✅ <b>{lbl}</b> — бан у Telegram-групі знято "
                                            f"→ акаунт у боті УВІМКНЕНО.")
            except Exception:
                pass
        sync_user_group(uid)       # активний → розбан у решті чатів + запрошення
        return out
    return None


def _access_delivery(uid, decision, text, buttons=None):
    """Надіслати рішення про доступ особисто й залишити перевіряемий аудит.

    Повертає False, якщо Telegram не прив'язано або Bot API відхилив повідомлення.
    Адміністратор отримує окреме попередження — схвалення/відхилення більше не
    виглядає як успішно доставлене, коли користувач його не побачив.
    """
    u = get_user_by_id(uid)
    if not u:
        return False
    sent, error = False, ''
    if not u.telegram_chat_id:
        error = 'Telegram не привʼязано'
    else:
        try:
            from web.tg_bot import tg_send, last_send_error
            sent = bool(tg_send(u.telegram_chat_id, text, buttons=buttons))
            if not sent:
                error = last_send_error() or 'Telegram API не підтвердив відправку'
        except Exception as exc:
            error = str(exc)
    set_user_pref(uid, 'access_notice', {
        'decision': decision, 'sent': sent, 'at': int(time.time()),
        'error': error[:500],
    })
    if not sent:
        print(f"[AUTH][TG] access {decision} delivery failed for user {uid}: {error}")
        try:
            from web.tg_bot import notify_category
            notify_category(
                'register',
                f"⚠️ <b>{u.email}</b> — рішення «{decision}» збережено, "
                f"але повідомлення користувачу НЕ доставлено: {error}")
        except Exception as exc:
            print(f"[AUTH][TG] admin delivery warning error: {exc}")
    return sent


def notify_user_rejected(uid):
    """Повідомити про відхилення мовою, обраною під час реєстрації."""
    from web.i18n import user_tr
    return _access_delivery(uid, 'rejected', user_tr(uid, 'access_rejected'))


def notify_user_approved(uid, silent=False):
    """Tell the user (via their linked Telegram) that access is granted.

    `silent=True` — без повідомлення користувачу (зміна ТЕРМІНУ доступу,
    вимога 09.11: «повідомлень про продовження на 10/30 днів не потрібно»);
    позначка `ever_approved` і повернення в групу — як і раніше."""
    delivered = None
    try:
        u = get_user_by_id(uid)
        if not u:
            return False
        # «було схвалено» — щоб знятий доступ відрізнявся від «очікує схвалення»
        set_user_pref(uid, 'ever_approved', True)
        if not silent:
            from web.tg_bot import base_url
            from web.i18n import user_tr, language_for_user
            b = base_url()
            lang = language_for_user(uid)
            delivered = _access_delivery(
                uid, 'approved', user_tr(uid, 'access_approved'),
                buttons=[[{'text': user_tr(uid, 'login_short_button'),
                           'url': f"{b}/login?lang={lang}"}]] if b else None)
        if not u.telegram_chat_id:
            return delivered
        # 🎉 Auto-onboard the approved user into the community group (personal
        # one-time invite link — Bot API can't add a member directly).
        _st = (_load_prefs(u) or {}).get('grp_state')
        try:
            from web.tg_bot import invite_user_to_group
            # Функція приймає лише chat_id; зайвий `back` раніше переривав запрошення.
            invite_user_to_group(u.telegram_chat_id)
            set_user_pref(uid, 'grp_banned', None)
        except Exception as _e:
            print(f"[AUTH][TG] group invite error: {_e}")
        try:
            if _st == 'restricted':                    # обмежений версією 09.11 → права назад
                from web.tg_bot import group_unrestrict
                group_unrestrict(u.telegram_chat_id)
            if _st:
                set_user_pref(uid, 'grp_state', None)
            set_user_pref(uid, 'tg_banned_at', None)
        except Exception as _e:
            print(f"[AUTH][TG] group invite error: {_e}")
    except Exception as e:
        print(f"[AUTH][TG] notify user error: {e}")
        delivered = False
    return delivered


def create_user(email, password, is_admin=False, email_confirmed=False,
                approved=False, telegram_chat_id=None, telegram_username=None,
                telegram_name=None, bot_access=False, access_days=None):
    s = get_session()
    try:
        au = None
        if access_days and not is_admin:
            au = datetime.utcnow() + timedelta(days=int(access_days))
        u = User(email=_norm_email(email),
                 password_hash=generate_password_hash(password),
                 is_admin=bool(is_admin),
                 email_confirmed=bool(email_confirmed),
                 approved=bool(approved),
                 disabled=False, prefs='{}',
                 bot_access=bool(bot_access or is_admin),
                 access_until=au,
                 telegram_chat_id=(str(telegram_chat_id) if telegram_chat_id else None),
                 telegram_username=telegram_username,
                 telegram_name=telegram_name,
                 created_at=datetime.utcnow())
        s.add(u)
        s.commit()
        return u.id
    except Exception as e:
        s.rollback()
        print(f"[AUTH] create_user error: {e}")
        return None
    finally:
        s.close()


def _update_user(uid, **fields):
    s = get_session()
    try:
        u = s.query(User).filter_by(id=int(uid)).first()
        if not u:
            return False
        for k, v in fields.items():
            setattr(u, k, v)
        s.commit()
        return True
    except Exception as e:
        s.rollback()
        print(f"[AUTH] update_user error: {e}")
        return False
    finally:
        s.close()


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


_up = _perms_mod


def _perms_raw(u):
    try:
        return _up().norm(json.loads(u.prefs or '{}').get('perms'))
    except Exception:
        return {}


def _perms_eff(u):
    try:
        return _up().of_user(u)
    except Exception:
        return {}


def list_users():
    s = get_session()
    try:
        out = []
        for u in s.query(User).order_by(User.id.asc()).all():   # top→bottom
            try:
                log = json.loads(u.login_log or '[]')
            except Exception:
                log = []
            ips = sorted({e.get('ip') for e in log if isinstance(e, dict)})
            au = getattr(u, 'access_until', None)
            days_left = None
            if au:
                days_left = round((au - datetime.utcnow()).total_seconds() / 86400, 1)
            try:
                _pf = json.loads(u.prefs or '{}')
            except Exception:
                _pf = {}
            out.append({
                'id': u.id, 'email': u.email, 'is_admin': u.is_admin,
                'email_confirmed': u.email_confirmed, 'approved': u.approved,
                'disabled': u.disabled,
                'active': _is_active(u),
                'created_at': u.created_at.isoformat() if u.created_at else None,
                'last_login': u.last_login.isoformat() if u.last_login else None,
                'access_until': au.isoformat() if au else None,
                'days_left': days_left,
                'last_ip': getattr(u, 'last_ip', None),
                'distinct_ips': len(ips),
                'recent_ips': ips[:8],
                'login_count': len(log),
                'tg_linked': bool(getattr(u, 'telegram_chat_id', None)),
                'tg_user': getattr(u, 'telegram_username', None),
                'tg_name': getattr(u, 'telegram_name', None),
                'bot_access': bool(getattr(u, 'bot_access', False)) or u.is_admin,
                # 🛡 права інфосайту: збережені + що реально діє (адмін — усе)
                'perms': _perms_raw(u),
                'perms_eff': _perms_eff(u),
                # 🚫 Telegram-група: 'banned' | 'restricted' (09.11) | None;
                # tg_banned_at — забанив адміністратор САМОЇ групи (23.11)
                'grp_state': _pf.get('grp_state') or ('banned' if _pf.get('grp_banned') else None),
                'tg_banned_at': _pf.get('tg_banned_at'),
                'blocked': is_blocked(u),
            })
        return out
    finally:
        s.close()


def user_count():
    s = get_session()
    try:
        return s.query(User).count()
    finally:
        s.close()


def admin_pref(key, default=True):
    """The main admin's personal notification pref (notify_btc/notify_funding).
    Used to gate the Telegram GROUP topics: the admin's OWN cabinet toggle also
    controls what goes to the group — off in the cabinet → off in the group.
    Main admin = the account linked to TELEGRAM_CHAT_ID, else the first admin."""
    s = get_session()
    try:
        u = None
        adm_chat = os.getenv('TELEGRAM_CHAT_ID')
        if adm_chat:
            u = s.query(User).filter_by(telegram_chat_id=str(adm_chat)).first()
        if u is None:
            u = s.query(User).filter_by(is_admin=True).order_by(User.id.asc()).first()
        if u is None:
            return default
        prefs = json.loads(u.prefs or '{}')
        return bool(prefs.get(key, default))
    except Exception:
        return default
    finally:
        s.close()


def subscriber_chats(pref_key):
    """Telegram chat_ids of every ACTIVE user who opted into `pref_key`
    (e.g. 'notify_btc' / 'notify_funding') in their personal settings. Used by
    the bot to broadcast per-category alerts only to those who asked for them."""
    s = get_session()
    try:
        out = []
        for u in s.query(User).filter(User.telegram_chat_id.isnot(None)).all():
            if not _is_active(u):
                continue
            try:
                prefs = json.loads(u.prefs or '{}')
            except Exception:
                prefs = {}
            if prefs.get(pref_key):
                out.append(u.telegram_chat_id)
        return out
    except Exception:
        return []
    finally:
        s.close()


def info_sbr_subscribers():
    """🔔 Особисті SBR-підписки з кабінету (вимога 09.11):
    [(chat_id, {SYM: ['1h','4h']})] — ЛИШЕ АКТИВНІ користувачі з привʼязаним
    Telegram. Заблокований / протермінований / не схвалений — нічого не отримує."""
    s = get_session()
    try:
        out = []
        for u in s.query(User).filter(User.telegram_chat_id.isnot(None)).all():
            if not _is_active(u):
                continue
            try:
                prefs = json.loads(u.prefs or '{}')
            except Exception:
                prefs = {}
            if not _up().effective(u.is_admin, prefs.get('perms')).get('ntf_sbr'):
                continue                 # 🛡 адміністратор забрав це право
            sub = ((prefs.get('info') or {}).get('sbr')) or {}
            if isinstance(sub, dict) and sub:
                out.append((str(u.telegram_chat_id), sub))
        return out
    except Exception:
        return []
    finally:
        s.close()


def info_anom_subscribers():
    """⚠ Особисті сповіщення «Аномалії обсягу» з кабінету (вимога 18.11):
    [(chat_id, [SYM…])] — ЛИШЕ АКТИВНІ користувачі з привʼязаним Telegram
    (те саме правило доступу, що й для 🔔 SBR)."""
    s = get_session()
    try:
        out = []
        for u in s.query(User).filter(User.telegram_chat_id.isnot(None)).all():
            if not _is_active(u):
                continue
            try:
                prefs = json.loads(u.prefs or '{}')
            except Exception:
                prefs = {}
            if not _up().effective(u.is_admin, prefs.get('perms')).get('ntf_anom'):
                continue                 # 🛡 адміністратор забрав це право
            info = prefs.get('info') or {}
            sub = info.get('anom') or []
            if isinstance(sub, list) and sub:
                # 3-й елемент — формат повідомлення: short (деф.) | full (24.11)
                out.append((str(u.telegram_chat_id), list(sub),
                            'full' if info.get('anom_fmt') == 'full' else 'short'))
        return out
    except Exception:
        return []
    finally:
        s.close()


def info_trade_subscribers():
    """🤖 Торгівля з кабінету: [(uid, email, trade)] — АКТИВНІ користувачі з
    увімкненою торгівлею, яким адміністратор ДАВ право (🛡 `trade_paper` /
    `trade_real`; адмін — завжди). Монети з книгою, на яку права немає,
    відкидаються ТУТ — підроблені prefs нічого не дають."""
    s = get_session()
    try:
        out = []
        for u in s.query(User).all():
            if not _is_active(u):
                continue
            try:
                prefs = json.loads(u.prefs or '{}')
            except Exception:
                prefs = {}
            t = prefs.get('trade')
            if not (isinstance(t, dict) and t.get('enabled') and isinstance(t.get('coins'), dict)):
                continue
            bk = _up().books(_up().effective(u.is_admin, prefs.get('perms')))
            coins = {k: v for k, v in t['coins'].items() if v in bk}
            if coins:
                # 👤 admin → торгує Trade Manager БОТА (ключі бота); інші — свій TM
                out.append((u.id, u.email, dict(t, coins=coins, admin=bool(u.is_admin))))
        return out
    except Exception:
        return []
    finally:
        s.close()


def is_blocked(u):
    """ЧИСТА від I/O: акаунт ЗАБЛОКОВАНО (не плутати з «очікує схвалення»):
    вимкнено адміністратором · строк доступу минув · схвалення знято після
    того, як воно вже було. Адмін не блокується ніколи."""
    if u is None or getattr(u, 'is_admin', False):
        return False
    if getattr(u, 'disabled', False):
        return True
    if _is_active(u):
        return False
    try:
        prefs = json.loads(getattr(u, 'prefs', None) or '{}')
    except Exception:
        prefs = {}
    au = getattr(u, 'access_until', None)
    expired = bool(au and getattr(u, 'approved', False))
    return expired or bool(prefs.get('ever_approved'))


def group_members_state():
    """[(uid, chat_id, blocked, active, grp_state)] — для синхронізації
    членства в Telegram-групі з доступом на сайті."""
    s = get_session()
    try:
        out = []
        for u in s.query(User).filter(User.telegram_chat_id.isnot(None)).all():
            if u.is_admin:
                continue
            try:
                prefs = json.loads(u.prefs or '{}')
            except Exception:
                prefs = {}
            st = prefs.get('grp_state') or ('banned' if prefs.get('grp_banned') else None)
            out.append((u.id, str(u.telegram_chat_id), is_blocked(u),
                        _is_active(u), st))
        return out
    except Exception:
        return []
    finally:
        s.close()


def set_user_pref(uid, key, value):
    """Записати ОДИН ключ prefs, не чіпаючи решту (None — прибрати)."""
    u = get_user_by_id(uid)
    if not u:
        return
    prefs = _load_prefs(u)
    if value is None:
        prefs.pop(key, None)
    else:
        prefs[key] = value
    _update_user(uid, prefs=json.dumps(prefs)[:20000])


# ── 🔗 ПРИВʼЯЗКА TELEGRAM ДО ІСНУЮЧОГО АКАУНТА (вимога 08.11) ────────────────
# Акаунт, створений НЕ через бота (адмін, користувач з адмін-панелі), раніше не
# мав способу отримати Telegram: /start шле лише посилання на НОВУ реєстрацію.
# Тепер кабінет видає одноразовий код → deep-link `?start=lk_<код>` → бот
# привʼязує chat_id саме до цього акаунта.
TG_LINK_TTL_SEC = 30 * 60
TG_LINK_PREFIX = 'lk_'


def tg_link_code_ok(code):
    """ЧИСТА: формат коду (start-параметр Telegram: ≤64, [A-Za-z0-9_-])."""
    import re as _re
    return bool(_re.fullmatch(r'[A-Za-z0-9]{12,40}', str(code or '')))


def tg_link_issue(uid, now=None):
    """Видати новий одноразовий код (старий стає недійсним) → (code, until)."""
    import secrets as _sec
    now = time.time() if now is None else now
    code = _sec.token_hex(10)
    until = int(now + TG_LINK_TTL_SEC)
    set_user_pref(uid, 'tg_link', {'code': code, 'until': until})
    return code, until


def tg_link_consume(code, chat_id, username=None, name=None, now=None):
    """Привʼязати Telegram-чат до акаунта за кодом → (ok, повідомлення, uid|None).
    Код одноразовий і живе TG_LINK_TTL_SEC. Чат, уже привʼязаний до ІНШОГО
    акаунта, вдруге не привʼязується (один Telegram — один акаунт)."""
    now = time.time() if now is None else now
    if not tg_link_code_ok(code) or not chat_id:
        return False, 'Посилання недійсне.', None
    s = get_session()
    try:
        target = None
        for u in s.query(User).all():
            lk = (_load_prefs(u) or {}).get('tg_link') or {}
            if lk.get('code') == code:
                target = u
                break
        if target is None:
            return False, 'Посилання недійсне або вже використане.', None
        lk = (_load_prefs(target) or {}).get('tg_link') or {}
        if float(lk.get('until') or 0) < now:
            return False, ('Посилання прострочене — отримайте нове в особистому '
                           'кабінеті.'), None
        other = s.query(User).filter_by(telegram_chat_id=str(chat_id)).first()
        if other is not None and other.id != target.id:
            return False, (f'Цей Telegram уже привʼязаний до акаунта '
                           f'<b>{other.email}</b>.'), None
        prefs = _load_prefs(target)
        prefs.pop('tg_link', None)
        target.prefs = json.dumps(prefs)[:20000]
        target.telegram_chat_id = str(chat_id)
        if username:
            target.telegram_username = username
        if name and hasattr(target, 'telegram_name'):
            target.telegram_name = name
        s.commit()
        return True, target.email, target.id
    except Exception as e:
        s.rollback()
        print(f"[AUTH] tg_link_consume error: {e}")
        return False, 'Помилка привʼязки — спробуйте ще раз.', None
    finally:
        s.close()


_DB_BOT_CONTACTS = 'tg_bot_contacts'   # {chat_id: {username, name, language, first_seen, last_seen}}
_MAX_BOT_CONTACTS = 5000


def record_bot_contact(chat_id, username=None, name=None, language=None):
    """Remember EVERY chat that talks to the bot — even before (or without)
    registration — so the admin can SEE and reach all bot users, not just
    registered accounts. Telegram's locale is saved only on first detection;
    subsequent /start calls must not overwrite an established preference."""
    if not chat_id:
        return
    try:
        from storage.db_operations import get_db
        from web.i18n import normalize_language
        import time as _t
        db = get_db()
        data = db.get_setting(_DB_BOT_CONTACTS, {}) or {}
        if not isinstance(data, dict):
            data = {}
        cid = str(chat_id)
        rec = data.get(cid) or {}
        rec.setdefault('first_seen', _t.time())
        rec['last_seen'] = _t.time()
        if username:
            rec['username'] = username
        if name:
            rec['name'] = name
        if language and not rec.get('language'):
            rec['language'] = normalize_language(language)
            rec['language_source'] = 'telegram'
        data[cid] = rec
        if len(data) > _MAX_BOT_CONTACTS:      # keep the most-recent contacts
            data = dict(sorted(data.items(),
                               key=lambda kv: (kv[1] or {}).get('last_seen', 0),
                               reverse=True)[:_MAX_BOT_CONTACTS])
        db.set_setting(_DB_BOT_CONTACTS, data)
    except Exception as e:
        print(f"[AUTH] record_bot_contact error: {e}")


def forget_bot_contact(chat_id):
    """Прибрати чат зі списку контактів бота (користувача видалено)."""
    try:
        from storage.db_operations import get_db
        db = get_db()
        data = db.get_setting(_DB_BOT_CONTACTS, {}) or {}
        if isinstance(data, dict) and str(chat_id) in data:
            data.pop(str(chat_id), None)
            db.set_setting(_DB_BOT_CONTACTS, data)
    except Exception as e:
        print(f"[AUTH] forget_bot_contact error: {e}")


def bot_contacts():
    """{chat_id: {username, name, first_seen, last_seen}} — everyone who ever
    used the bot (for the admin subscriber list)."""
    try:
        from storage.db_operations import get_db
        d = get_db().get_setting(_DB_BOT_CONTACTS, {}) or {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def all_bot_chats(exclude_chat=None):
    """Audience for an ADMIN ANNOUNCEMENT = EVERYONE who uses the bot: all
    recorded bot contacts + registered users with a linked chat, MINUS accounts
    the admin explicitly disabled/rejected, minus `exclude_chat` (admin's own)."""
    ex = str(exclude_chat) if exclude_chat else None
    disabled, registered = set(), set()
    s = get_session()
    try:
        for u in s.query(User).filter(User.telegram_chat_id.isnot(None)).all():
            c = str(u.telegram_chat_id)
            # заблокований (вимкнено / строк минув / схвалення знято) — без розсилок
            (disabled if is_blocked(u) else registered).add(c)
    except Exception:
        pass
    finally:
        s.close()
    chats = {str(c) for c in bot_contacts().keys()} | registered
    chats -= disabled
    if ex:
        chats.discard(ex)
    return list(chats)


def pending_attention_count():
    """Users needing admin attention (new registrations awaiting approval)."""
    s = get_session()
    try:
        return s.query(User).filter(User.approved == False,   # noqa: E712
                                    User.is_admin == False,
                                    User.disabled == False).count()
    except Exception:
        return 0
    finally:
        s.close()


def _is_active(u):
    if u is None:
        return False
    if u.disabled:
        return False
    if u.is_admin:
        return True                    # admins never expire / never disabled
    if not (u.email_confirmed and u.approved):
        return False
    # Time-limited access: auto-block once the window has passed.
    au = getattr(u, 'access_until', None)
    if au:
        try:
            if au < datetime.utcnow():
                return False
        except Exception:
            pass
    return True


def _client_ip():
    fwd = (request.headers.get('X-Forwarded-For') or '').split(',')[0].strip()
    return fwd or request.remote_addr or '?'


def _migrate_user_columns():
    """Add any missing User columns to the EXISTING table (create_all won't
    ALTER). Dialect-agnostic: inspect existing columns, then ADD only the
    missing ones with a plain ADD COLUMN (works on Postgres AND sqlite; avoids
    the non-portable «IF NOT EXISTS»). Each ADD in its OWN transaction so one
    failure can't abort the rest."""
    want = [
        ('access_until', 'TIMESTAMP'),
        ('session_token', 'VARCHAR(64)'),
        ('last_ip', 'VARCHAR(64)'),
        ('login_log', 'TEXT'),
        ('telegram_chat_id', 'VARCHAR(32)'),
        ('telegram_username', 'VARCHAR(64)'),
        ('telegram_name', 'VARCHAR(128)'),
        ('bot_access', 'BOOLEAN DEFAULT FALSE'),
    ]
    try:
        from storage.db_models import engine
        from sqlalchemy import text, inspect
        try:
            existing = {c['name'] for c in inspect(engine).get_columns('sob_users')}
        except Exception:
            existing = set()
        for col, ddl in want:
            if col in existing:
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE sob_users ADD COLUMN {col} {ddl}"))
                print(f"[AUTH] migrated: added column {col}")
            except Exception as e:
                print(f"[AUTH] migrate skip {col}: {e}")
        print("[AUTH] user-columns migration ensured")
    except Exception as e:
        print(f"[AUTH] migrate error: {e}")


# ==================================================================
# Session helpers
# ==================================================================
def login_user(u):
    session.permanent = True
    session['uid'] = u.id
    session['is_admin'] = bool(u.is_admin)
    # Single-active-session (anti password-sharing): rotate the token so any
    # OTHER live session for this account is invalidated on its next request.
    tok = secrets.token_hex(16)
    session['stok'] = tok
    ip = _client_ip()
    try:
        log = json.loads(u.login_log or '[]')
        if not isinstance(log, list):
            log = []
    except Exception:
        log = []
    log.append({'t': int(time.time()), 'ip': ip})
    log = log[-15:]                      # keep last 15 logins
    _update_user(u.id, last_login=datetime.utcnow(), session_token=tok,
                 last_ip=ip, login_log=json.dumps(log))


def logout_user():
    session.pop('uid', None)
    session.pop('is_admin', None)


def current_user():
    """Cached-per-request current user object (or None). Auth sources, in order:
    the session cookie (same-origin) OR an «X-Info-Token» header (cross-origin
    info-site). `g._via_token` marks token auth so the single-session guard is
    skipped for it."""
    if getattr(g, '_cur_user_set', False):
        return g._cur_user
    u = None
    if 'uid' in session:
        u = get_user_by_id(session['uid'])
    else:
        try:
            tok = request.headers.get('X-Info-Token')
        except Exception:
            tok = None
        if tok:
            v = _read_info_token_ts(tok)
            if v:
                u = get_user_by_id(v[0])
                if u is not None:
                    try:
                        _pr = json.loads(u.prefs or '{}')
                    except Exception:
                        _pr = {}
                    if info_token_revoked(_pr, v[1]):
                        u = None
                if u:
                    g._via_token = True
    g._cur_user = u
    g._cur_user_set = True
    return u


# ==================================================================
# Mailer (SMTP via env, with dev fallback)
# ==================================================================
def _smtp_configured():
    return bool(os.getenv('SMTP_HOST'))


def _mail_provider():
    """Which mail transport is configured. HTTP APIs work on Render (port 443);
    SMTP ports (25/465/587) are BLOCKED on Render — so an API is preferred."""
    if os.getenv('RESEND_API_KEY'):
        return 'resend'
    if os.getenv('BREVO_API_KEY'):
        return 'brevo'
    if os.getenv('SENDGRID_API_KEY'):
        return 'sendgrid'
    if os.getenv('MAILJET_API_KEY') and os.getenv('MAILJET_SECRET_KEY'):
        return 'mailjet'
    if os.getenv('SMTP_HOST'):
        return 'smtp'
    return None


def _mail_from():
    return (os.getenv('MAIL_FROM') or os.getenv('SMTP_FROM')
            or os.getenv('SMTP_USER') or 'no-reply@bot.local')


def _http_post_json(url, headers, payload):
    import urllib.request
    import urllib.error
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, method='POST',
        headers={**headers, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read().decode(errors='replace')
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors='replace')


def _send_via_http_api(to, subject, html):
    """Send through an HTTP email API (works on Render — no SMTP ports needed).
    Returns (ok, info, provider) if a provider is configured, else None."""
    sender = _mail_from()
    if os.getenv('RESEND_API_KEY'):
        st, body = _http_post_json(
            'https://api.resend.com/emails',
            {'Authorization': 'Bearer ' + os.getenv('RESEND_API_KEY')},
            {'from': sender, 'to': [to], 'subject': subject, 'html': html})
        return (200 <= st < 300, f'resend HTTP {st}: {body[:200]}', 'resend')
    if os.getenv('BREVO_API_KEY'):
        st, body = _http_post_json(
            'https://api.brevo.com/v3/smtp/email',
            {'api-key': os.getenv('BREVO_API_KEY')},
            {'sender': {'email': sender}, 'to': [{'email': to}],
             'subject': subject, 'htmlContent': html})
        return (200 <= st < 300, f'brevo HTTP {st}: {body[:200]}', 'brevo')
    if os.getenv('SENDGRID_API_KEY'):
        st, body = _http_post_json(
            'https://api.sendgrid.com/v3/mail/send',
            {'Authorization': 'Bearer ' + os.getenv('SENDGRID_API_KEY')},
            {'personalizations': [{'to': [{'email': to}]}],
             'from': {'email': sender}, 'subject': subject,
             'content': [{'type': 'text/html', 'value': html}]})
        return (200 <= st < 300, f'sendgrid HTTP {st}: {body[:200]}', 'sendgrid')
    if os.getenv('MAILJET_API_KEY') and os.getenv('MAILJET_SECRET_KEY'):
        import base64
        auth = base64.b64encode(
            f"{os.getenv('MAILJET_API_KEY')}:{os.getenv('MAILJET_SECRET_KEY')}".encode()
        ).decode()
        st, body = _http_post_json(
            'https://api.mailjet.com/v3.1/send',
            {'Authorization': 'Basic ' + auth},
            {'Messages': [{'From': {'Email': sender}, 'To': [{'Email': to}],
                           'Subject': subject, 'HTMLPart': html}]})
        return (200 <= st < 300, f'mailjet HTTP {st}: {body[:200]}', 'mailjet')
    return None


def _tg_notify_link(to, subject, link):
    """Push a confirm/reset link to the admin's Telegram (uses the bot's existing
    TELEGRAM_BOT_TOKEN — HTTPS 443, not blocked on Render). Zero external signup:
    the admin gets every link and can forward it or just approve the user. OFF if
    AUTH_TG_LINKS=0 or Telegram isn't configured."""
    try:
        if not link or not os.getenv('TELEGRAM_BOT_TOKEN'):
            return False
        if str(os.getenv('AUTH_TG_LINKS', '1')).lower() in ('0', 'false', 'no'):
            return False
        from alerts.telegram_notifier import get_notifier
        n = get_notifier()
        if not n:
            return False
        txt = (f"🔐 <b>VSV Bot</b> · {subject}\n"
               f"Користувач: <code>{to}</code>\n{link}")
        return n.send_message(txt)
    except Exception as e:
        print(f"[AUTH][TG] link notify error: {e}")
        return False


def _rec_smtp(ok, to, error=None, stage=None):
    with _lock:
        _smtp_last.update({'ok': ok, 'error': (str(error)[:400] if error else None),
                           'at': time.time(), 'to': to, 'stage': stage})


def smtp_status():
    """Diagnostic snapshot of the mailer for the admin panel."""
    prov = _mail_provider()
    return {
        'configured': prov is not None,
        'provider': prov,                          # resend|brevo|sendgrid|smtp|None
        'is_api': prov in ('resend', 'brevo', 'sendgrid', 'mailjet'),
        'host': os.getenv('SMTP_HOST') or None,
        'port': os.getenv('SMTP_PORT', '587'),
        'user': os.getenv('SMTP_USER') or None,
        'from': _mail_from(),
        'tls': os.getenv('SMTP_TLS', '1'),
        'pass_set': bool(os.getenv('SMTP_PASS')),
        'last': dict(_smtp_last),
    }


def send_email(to, subject, html, link=None):
    """Send an email via SMTP env config. Returns True on send. When SMTP is
    NOT configured, stores the link for the admin panel + logs it (dev mode)
    and returns False. Every attempt records its outcome into `_smtp_last` so
    the admin panel can show EXACTLY why a mail failed (auth, connect, TLS…)."""
    if link:
        with _lock:
            _pending_links[_norm_email(to)] = {'link': link, 'subject': subject,
                                               'at': time.time()}
        # Always mirror the link to the admin's Telegram (existing bot) — a
        # zero-signup delivery channel independent of any email provider.
        _tg_notify_link(to, subject, link)
    # 1) HTTP email API first — works on Render (SMTP ports are blocked there).
    try:
        api = _send_via_http_api(to, subject, html)
    except Exception as e:
        api = (False, f'API error: {e}', _mail_provider())
    if api is not None:
        ok, info, prov = api
        _rec_smtp(ok, to, error=None if ok else info, stage=prov)
        print(f"[AUTH][MAIL] {prov} -> {'OK' if ok else 'FAIL'} to {to}: {info if not ok else ''}")
        return ok
    # 2) SMTP fallback (won't work on Render — kept for other hosts).
    if not _smtp_configured():
        print(f"[AUTH][MAIL-DEV] to={to} · {subject}\n  LINK: {link}")
        _rec_smtp(None, to, error='Пошта не налаштована (ні API-ключ, ні SMTP)',
                  stage='config')
        return False
    stage = 'init'
    try:
        host = os.getenv('SMTP_HOST')
        port = int(os.getenv('SMTP_PORT', '587') or 587)
        user = os.getenv('SMTP_USER', '')
        # Gmail app passwords are shown as «xxxx xxxx xxxx xxxx» — the spaces are
        # cosmetic; strip them so login doesn't fail on a pasted-with-spaces value.
        pwd = (os.getenv('SMTP_PASS', '') or '').replace(' ', '')
        sender = os.getenv('SMTP_FROM', user or 'no-reply@bot.local')
        use_tls = str(os.getenv('SMTP_TLS', '1')).lower() in ('1', 'true', 'yes')
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = to
        msg.set_content(re.sub('<[^<]+?>', '', html))   # plaintext fallback
        msg.add_alternative(html, subtype='html')
        if port == 465:
            stage = 'connect(SSL 465)'
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as srv:
                if user:
                    stage = 'login'
                    srv.login(user, pwd)
                stage = 'send'
                srv.send_message(msg)
        else:
            stage = f'connect({host}:{port})'
            with smtplib.SMTP(host, port, timeout=20) as srv:
                if use_tls:
                    stage = 'starttls'
                    srv.starttls(context=ssl.create_default_context())
                if user:
                    stage = 'login'
                    srv.login(user, pwd)
                stage = 'send'
                srv.send_message(msg)
        _rec_smtp(True, to, stage='sent')
        print(f"[AUTH][MAIL] sent to {to} · {subject}")
        return True
    except Exception as e:
        print(f"[AUTH][MAIL] send error to {to} @ stage={stage}: {e}")
        _rec_smtp(False, to, error=e, stage=stage)
        return False


def _base_url():
    # Honour an explicit public URL (Render) if set, else derive from request.
    ext = os.getenv('RENDER_EXTERNAL_URL') or os.getenv('PUBLIC_URL')
    if ext:
        return ext.rstrip('/')
    return request.host_url.rstrip('/')


# ==================================================================
# Admin bootstrap
# ==================================================================
def ensure_admin():
    """Create/mark the ADMIN_EMAIL account as an active admin on boot."""
    email = _norm_email(os.getenv('ADMIN_EMAIL', ''))
    if not email:
        print("[AUTH] ADMIN_EMAIL not set — no admin bootstrap. Set ADMIN_EMAIL "
              "(and ADMIN_PASSWORD) to create the first admin.")
        return
    u = get_user_by_email(email)
    if u is None:
        pwd = os.getenv('ADMIN_PASSWORD') or 'change-me-now'
        create_user(email, pwd, is_admin=True, email_confirmed=True, approved=True)
        print(f"[AUTH] bootstrapped admin {email} "
              f"(password from ADMIN_PASSWORD env{'' if os.getenv('ADMIN_PASSWORD') else ' — DEFAULT, CHANGE IT'}).")
    else:
        # Ensure the configured admin stays admin + active.
        if not (u.is_admin and u.email_confirmed and u.approved and not u.disabled):
            _update_user(u.id, is_admin=True, email_confirmed=True,
                         approved=True, disabled=False)
            print(f"[AUTH] promoted {email} to active admin.")


# ==================================================================
# Access gate
# ==================================================================
_PUBLIC_EXACT = {'/login', '/register', '/forgot', '/logout', '/pending',
                 '/api/health', '/api/version', '/favicon.ico'}
# '/info' is public so GUESTS can open the info-site shell (login/registration
# live ON it). The DATA APIs stay gated — a guest sees the auth hero, not state.
_PUBLIC_PREFIX = ('/static/', '/confirm/', '/reset/', '/auth/', '/info', '/api/i18n/')
_USER_MUTABLE_PREFIX = ('/api/me', '/logout', '/auth/')   # non-admin may POST here
# Info-site-only users (no bot_access) may reach ONLY these in THIS (bot) app:
# their cabinet, the bot-served info-site (/info) and the READ-ONLY APIs the
# info-site consumes. POSTs are still blocked by the mutating-method guard, so
# these give GET access only (except /api/me* which is the user's own account).
_USER_NO_BOT_ALLOWED = ('/cabinet', '/logout', '/api/me', '/pending',
                        '/api/health', '/api/version', '/info',
                        '/api/fuel-filter/state', '/api/stats',
                        '/api/sm/bias', '/api/sm/auto-gate', '/api/tm/state',
                        '/api/info/', '/api/i18n/')


_ADMIN_PREFIX = ('/admin/', '/api/admin/')


def is_admin_path(path):
    """🛡 Адмін-панель: `/admin/...` і `/api/admin/...` (+ точні `/admin`)."""
    path = path or ''
    return path == '/admin' or path.startswith(_ADMIN_PREFIX)


def admin_access_ok(u, via_token=False):
    """ЧИСТА (вимога 24.11): доступ до адмін-панелі — ЛИШЕ через сесію БОТА
    (вхід на сайті бота, cookie) і ЛИШЕ активному адміністратору. Токен
    інфосайту (X-Info-Token) і ключ INFO_API_KEY сюди НЕ пускають."""
    return bool(u is not None and getattr(u, 'is_admin', False)
                and not getattr(u, 'disabled', False) and not via_token)


def _is_public(path):
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIX)


def _wants_json(path):
    return path.startswith('/api/') or request.headers.get('Accept', '').find('application/json') >= 0


def install_auth_gate(app):
    # Dedicated auth secret — its OWN env var (AUTH_SECRET_KEY), so it does NOT
    # reuse FLASK_SECRET_KEY (which is used for other purposes). When set, it
    # signs BOTH the login session AND the email tokens. Falls back to the
    # app's existing secret_key when unset.
    _auth_secret = os.getenv('AUTH_SECRET_KEY')
    if _auth_secret:
        app.secret_key = _auth_secret
    elif not app.secret_key or app.secret_key == 'sleeper-ob-bot-secret-key-change-me':
        print("[AUTH] ⚠ AUTH_SECRET_KEY not set and app secret is the default — "
              "set AUTH_SECRET_KEY to a long random value for secure sessions/tokens.")
    # Cookie hardening.
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_SECURE=str(os.getenv('FLASK_COOKIE_SECURE', '')).lower()
        in ('1', 'true', 'yes'),
        PERMANENT_SESSION_LIFETIME=int(os.getenv('SESSION_LIFETIME_SEC', str(7 * 24 * 3600))),
    )

    @app.before_request
    def _auth_gate():
        path = request.path or '/'
        # CORS preflight (OPTIONS) is NEVER auth-gated: it carries no credentials
        # and, per spec, omits custom headers like X-API-Key — so the key bypass
        # below can't see it. Blocking it with 401 makes the browser refuse the
        # real cross-origin GET (→ «бот недоступний» on the info-site).
        if request.method == 'OPTIONS':
            return None
        if _is_public(path):
            return None
        # Controlled read-only bypass for the separate info-site: GET /api/* with
        # a matching X-API-Key. OFF unless INFO_API_KEY env is set (full lockdown
        # by default). Read-only (GET/HEAD) — never enables writes.
        _api_key = os.getenv('INFO_API_KEY')
        # ⚠️ Особистий кабінет (/api/me*, /api/info/*) ключ НЕ відкриває: там
        # дані конкретного користувача, і заблокований акаунт не має бачити
        # сторінку лише тому, що ключ лежить у config.js (вимога 09.11).
        if (_api_key and path.startswith('/api/')
                and not path.startswith(('/api/me', '/api/info/', '/api/admin'))
                and request.method in ('GET', 'HEAD', 'OPTIONS')
                and request.headers.get('X-API-Key') == _api_key):
            return None
        u = current_user()
        if u is None:
            if _wants_json(path):
                return jsonify({'ok': False, 'error': 'auth_required'}), 401
            return redirect(url_for('auth_login', next=path))
        # 🛡 Адмін-панель — лише сесія бота + адміністратор (до будь-яких
        # інших винятків гейта; токен інфосайту сюди не пускає).
        if is_admin_path(path) and not admin_access_ok(u, getattr(g, '_via_token', False)):
            if _wants_json(path):
                return jsonify({'ok': False, 'error': 'admin_only'}), 403
            return abort(403)
        if not _is_active(u):
            # Logged in but pending confirm/approval OR access expired → block.
            _expired = bool(getattr(u, 'access_until', None)
                            and not u.is_admin and u.email_confirmed and u.approved
                            and not u.disabled)
            if _wants_json(path):
                return jsonify({'ok': False,
                                'error': ('blocked' if (is_blocked(u) and not _expired)
                                          else ('expired' if _expired else 'not_active')),
                                'email_confirmed': bool(u.email_confirmed),
                                'approved': bool(u.approved)}), 403
            return redirect(url_for('auth_pending'))
        # Single-active-session enforcement (non-admins): if this session's token
        # no longer matches the account's current token, a newer login superseded
        # it → invalidate. Admins exempt; token-authed info-site requests exempt
        # (they carry no session cookie / stok by design).
        if not u.is_admin and not getattr(g, '_via_token', False):
            st = getattr(u, 'session_token', None)
            if st and session.get('stok') != st:
                logout_user()
                if _wants_json(path):
                    return jsonify({'ok': False, 'error': 'session_superseded'}), 401
                return redirect(url_for('auth_login', superseded='1'))
        # Active. Admins unrestricted.
        if u.is_admin:
            return None
        # Two-site model: without bot_access the user may use only the info-site
        # + their own cabinet — the BOT itself is off-limits until an admin grants
        # bot access. (The separate info-site shares these accounts.)
        if not getattr(u, 'bot_access', False):
            if not any(path.startswith(p) for p in _USER_NO_BOT_ALLOWED):
                if _wants_json(path):
                    return jsonify({'ok': False, 'error': 'no_bot_access'}), 403
                return _no_bot_access_page(u)
        # Non-admin: block mutating methods except own-profile endpoints.
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
            if not any(path.startswith(p) for p in _USER_MUTABLE_PREFIX):
                if _wants_json(path):
                    return jsonify({'ok': False, 'error': 'forbidden',
                                    'reason': 'read-only account'}), 403
                return abort(403)
        return None

    @app.context_processor
    def _inject_auth_user():
        # Expose the current user + pending-users count (admin badge) to templates.
        try:
            u = current_user()
            pend = 0
            if u and u.is_admin:
                try:
                    pend = pending_attention_count()
                except Exception:
                    pend = 0
            return {'auth_user': u, 'users_pending': pend}
        except Exception:
            return {'auth_user': None, 'users_pending': 0}

    @app.after_request
    def _inject_user_chip(resp):
        # Floating «кабінет · вийти» chip — ONLY for authenticated HTML pages that
        # DON'T already render the top-nav account chip (id="nav-user-chip", added
        # to base.html). Positioned TOP-LEFT. Skips auth pages / JSON / streamed.
        try:
            if (request.method != 'GET' or resp.is_streamed
                    or resp.status_code != 200
                    or 'text/html' not in (resp.content_type or '')
                    or _is_public(request.path or '/')):
                return resp
            u = current_user()
            if not u:
                return resp
            html = resp.get_data(as_text=True)
            if 'id="nav-user-chip"' in html or '</body>' not in html:
                return resp   # base.html already shows it in the top panel
            role = 'адмін' if u.is_admin else 'перегляд'
            adm = ('<a href="/admin/users" style="color:#fde68a;text-decoration:none">🛡</a> · '
                   if u.is_admin else '')
            chip = (
                '<div id="__authchip" style="position:fixed;left:12px;top:12px;z-index:99999;'
                'font:600 12px system-ui,sans-serif;background:#141922;color:#e5e7eb;'
                'border:1px solid rgba(255,255,255,.12);border-radius:20px;padding:6px 12px;'
                'box-shadow:0 4px 14px rgba(0,0,0,.5);opacity:.9">'
                f'👤 {u.email} <span style="color:#9aa3b5">· {role}</span> · '
                f'{adm}<a href="/cabinet" style="color:#60a5fa;text-decoration:none">кабінет</a> · '
                '<a href="/logout" style="color:#fca5a5;text-decoration:none">вийти</a></div>')
            resp.set_data(html.replace('</body>', chip + '</body>', 1))
        except Exception:
            pass
        return resp


# ==================================================================
# Page shell (inline templates — no extra files)
# ==================================================================
_SHELL = """<!doctype html><html lang="{{lang}}"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{title}} · VSV Bot</title>
<link rel="icon" type="image/png" href="/favicon.ico">
<link rel="apple-touch-icon" href="/favicon.ico"><style>
.brandlogo{display:block;width:76px;height:76px;margin:0 auto 16px;border-radius:18px;object-fit:cover;
 box-shadow:0 0 0 1px #26324a,0 10px 30px -12px rgba(0,0,0,.6)}
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
 background:radial-gradient(1200px 600px at 80% -10%,rgba(245,166,35,.06),transparent 60%),
  radial-gradient(900px 500px at -10% 10%,rgba(47,209,139,.05),transparent 55%),#0a0e17;
 color:#e8eef7;font-variant-numeric:tabular-nums;
 font-family:"Inter",ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.card{width:100%;max-width:400px;margin:20px;padding:28px 26px;border-radius:14px;
 background:linear-gradient(180deg,#131a26,#0d1220);border:1px solid #26324a;
 box-shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px -12px rgba(0,0,0,.55)}
h1{font-size:1.3rem;font-weight:900;letter-spacing:-.02em;margin:0 0 4px}
.sub{color:#aab6c8;font-size:.82rem;margin:0 0 18px}
label{display:block;font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#8695ac;margin:14px 0 5px}
input,select{width:100%;padding:11px 13px;border-radius:9px;border:1px solid #26324a;background:#1a2333;
 color:#e8eef7;font-size:.92rem}
input:focus,select:focus{outline:none;border-color:#f5a623;box-shadow:0 0 0 3px rgba(245,166,35,.2)}
.language-row{display:flex;align-items:center;gap:10px;margin:0 0 16px;padding:10px 12px;
 border:1px solid #2b3850;border-radius:10px;background:#171f2e}
.language-row label{margin:0;white-space:nowrap;color:#c4cfdf}.language-row select{padding:7px 9px}
.language-row .lang-icon{color:#f5c85f;font-size:1.15rem}
button{width:100%;margin-top:18px;padding:12px;border:0;border-radius:10px;font-weight:800;
 font-size:.92rem;cursor:pointer;background:linear-gradient(180deg,#ffc960,#f5a623);color:#1a1204;
 box-shadow:0 6px 18px -8px rgba(245,166,35,.6)}
button:hover{filter:brightness(1.05)}
.tgbtn{display:inline-block;padding:12px 22px;border-radius:10px;font-weight:800;text-decoration:none;
 background:linear-gradient(180deg,#ffc960,#f5a623);color:#1a1204;box-shadow:0 6px 18px -8px rgba(245,166,35,.6)}
.tgbtn:hover{filter:brightness(1.05)}
.msg{margin:12px 0;padding:10px 12px;border-radius:9px;font-size:.82rem}
.err{background:rgba(255,93,108,.12);border:1px solid rgba(255,93,108,.4);color:#ff9aa4}
.ok{background:rgba(47,209,139,.12);border:1px solid rgba(47,209,139,.4);color:#6ee7b7}
.links{margin-top:18px;font-size:.8rem;text-align:center;color:#8695ac}
.links a{color:#f5a623;text-decoration:none;font-weight:600} .links a:hover{text-decoration:underline}
/* cabinet rows + toggle switch */
.sect2{border-top:1px solid #26324a;margin-top:18px;padding-top:14px}
.seclbl{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;color:#8695ac;margin-bottom:8px}
.idrow{display:flex;justify-content:space-between;gap:12px;padding:7px 0;font-size:.88rem}
.idrow .k{color:#8695ac} .idrow .v{font-weight:700;text-align:right;word-break:break-all}
.crow{display:flex;justify-content:space-between;align-items:center;gap:14px;padding:12px 0;
 border-bottom:1px solid #1c2536} .crow:last-of-type{border-bottom:none}
.crow .t{font-size:.9rem;font-weight:600}
.sw{position:relative;width:48px;height:27px;flex:none;display:inline-block}
.sw input{position:absolute;opacity:0;width:100%;height:100%;margin:0;cursor:pointer;z-index:2}
.sw .tr{position:absolute;inset:0;border-radius:999px;background:#1a2333;border:1px solid #26324a;transition:.2s}
.sw .tr:before{content:"";position:absolute;top:2px;left:2px;width:21px;height:21px;border-radius:50%;
 background:#7d8ba1;transition:.2s}
.sw input:checked ~ .tr{background:rgba(47,209,139,.35);border-color:#2fd18b}
.sw input:checked ~ .tr:before{transform:translateX(21px);background:#2fd18b}
.sw input:disabled{cursor:not-allowed} .sw input:disabled ~ .tr{opacity:.45}
table{width:100%;border-collapse:collapse;font-size:.8rem} th,td{padding:7px 9px;text-align:left;
 border-bottom:1px solid rgba(255,255,255,.06)} th{color:#9aa3b5;font-weight:600}
.badge{padding:2px 8px;border-radius:20px;font-size:.68rem;font-weight:700}
.b-on{background:rgba(34,197,94,.15);color:#86efac} .b-off{background:rgba(148,163,184,.15);color:#cbd5e1}
.b-adm{background:rgba(250,204,21,.15);color:#fde68a}
.actbtn{padding:4px 9px;font-size:.72rem;border-radius:6px;border:1px solid #2a3140;background:#1a2130;
 color:#e5e7eb;cursor:pointer;margin:1px;text-decoration:none;display:inline-block}
.actbtn:hover{background:#232c3d}
body.full{align-items:flex-start;justify-content:flex-start}
.fullcard{max-width:1560px!important;width:calc(100vw - 32px);margin:16px auto!important}
</style></head><body class="{{ 'full' if full else '' }}">
<div class="card {{ 'fullcard' if full else '' }}" style="max-width:{{width or 400}}px">
{% if not full %}<img class="brandlogo" src="/favicon.ico" alt="VSV" onerror="this.style.display='none'">{% endif %}
{{body|safe}}</div>
<script>{{script|safe}}</script></body></html>"""


_AUTH_LANG_SCRIPT = r"""
(function(){
  var select=document.getElementById('profile-language');
  function apply(lang){
    if(!window.VSV_AUTH_STRINGS[lang]) lang='uk';
    document.documentElement.lang=lang;
    if(select) select.value=lang;
    document.querySelectorAll('[data-i18n]').forEach(function(el){
      var value=window.VSV_AUTH_STRINGS[lang][el.getAttribute('data-i18n')];
      if(value!=null) el.innerHTML=value;
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function(el){
      var value=window.VSV_AUTH_STRINGS[lang][el.getAttribute('data-i18n-placeholder')];
      if(value!=null) el.setAttribute('placeholder',value);
    });
    document.querySelectorAll('[data-lang-link]').forEach(function(a){
      try{var url=new URL(a.href,location.href);url.searchParams.set('lang',lang);a.href=url.href;}catch(e){}
    });
  }
  if(select){
    var explicit='';
    try{explicit=new URLSearchParams(location.search).get('lang')||'';}catch(e){}
    if(!explicit){try{var saved=localStorage.getItem('vsv-info-language');
      if(saved&&window.VSV_AUTH_STRINGS[saved]) apply(saved);}catch(e){}}
    select.addEventListener('change',function(){
      try{localStorage.setItem('vsv-info-language',select.value);}catch(e){}
      apply(select.value);
    });
  }
  window.VSV_AUTH_LANGUAGE={apply:apply};
})();
"""


def _page(title, body, script='', width=400, full=False, lang=None):
    from web.i18n import normalize_language, client_dictionaries
    try:
        requested = request.values.get('lang') or request.form.get('language')
    except Exception:
        requested = None
    lang = normalize_language(lang or requested)
    lang_script = ('window.VSV_AUTH_STRINGS=' + client_dictionaries() + ';'
                   + _AUTH_LANG_SCRIPT + script)
    return render_template_string(_SHELL, title=title, body=body, script=lang_script,
                                  width=width, full=full, lang=lang)


def _throttled(email):
    ip = request.remote_addr or '?'
    key = (email, ip)
    now = time.time()
    with _lock:
        hist = [t for t in _login_fails.get(key, []) if now - t < LOGIN_WINDOW]
        _login_fails[key] = hist
        return len(hist) >= LOGIN_MAX_FAILS


def _record_fail(email):
    ip = request.remote_addr or '?'
    key = (email, ip)
    with _lock:
        _login_fails.setdefault(key, []).append(time.time())


def cabinet_home(bot_ok, from_info, info_url):
    """ЧИСТА: головна сторінка авторизованого користувача — куди веде «← На
    головну» і куди повертає прямий вхід у кабінет. Користувач без доступу до
    бота (або прийшов з інфосайту) → інфосайт, інакше → дашборд бота."""
    if from_info or not bot_ok:
        return (info_url + '/') if info_url else '/info/'
    return '/'


def cabinet_entry_ok(referer, sec_fetch_site, host, info_url):
    """ЧИСТА (вимога 24.11): кабінет відкривається ЛИШЕ переходом зі сторінки
    авторизованого користувача — бота (той самий хост) або інфосайту
    (`INFO_SITE_URL`). Прямий вхід (набраний / збережений URL, посилання
    ззовні) → False → редирект на головну сторінку."""
    if (sec_fetch_site or '').lower() == 'same-origin':
        return True
    try:
        from urllib.parse import urlparse
        ref = urlparse(referer or '')
    except Exception:
        return False
    if not ref.netloc:
        return False
    if ref.netloc.lower() == (host or '').lower():
        return True
    try:
        inf = urlparse(info_url or '').netloc.lower()
    except Exception:
        inf = ''
    return bool(inf) and ref.netloc.lower() == inf


# ==================================================================
# Routes
# ==================================================================
def register_auth_routes(app):

    @app.route('/login', methods=['GET', 'POST'])
    def auth_login():
        from web.i18n import normalize_language, tr
        nxt = request.args.get('next') or request.form.get('next') or '/smart-money'
        # The bot always starts at Smart Money. A protected deep link must not
        # become the post-login destination; the only exception is the separate
        # info-site's signed-token handoff, which is not a bot-page navigation.
        destination = '/info-login' if nxt == '/info-login' else '/smart-money'
        if current_user() and _is_active(current_user()):
            return redirect(destination)
        lang = normalize_language(request.values.get('lang') or request.form.get('language'))
        if request.method == 'POST':
            email = _norm_email(request.form.get('email'))
            pw = request.form.get('password') or ''
            u = get_user_by_email(email)
            _is_adm = bool(u and u.is_admin)
            # Admins are EXEMPT from login throttling / IP-blocking — so an
            # admin account can never lock itself out.
            if not _is_adm and _throttled(email):
                return _page(tr(lang, 'login_title'), _login_form(
                    nxt, err=tr(lang, 'error_throttled'), lang=lang), lang=lang)
            if not u or not check_password_hash(u.password_hash, pw):
                if not _is_adm:
                    _record_fail(email)
                return _page(tr(lang, 'login_title'), _login_form(
                    nxt, err=tr(lang, 'error_credentials'), lang=lang), lang=lang)
            if u.disabled:
                return _page(tr(lang, 'login_title'), _login_form(
                    nxt, err=tr(lang, 'error_disabled'), lang=lang), lang=lang)
            # Explicit auth-page choice becomes the durable profile preference.
            if request.form.get('language'):
                set_user_pref(u.id, 'language', lang)
            login_user(u)
            if not _is_active(u):
                return redirect(url_for('auth_pending'))
            return redirect(destination)
        _info = tr(lang, 'session_superseded') if request.args.get('superseded') else ''
        return _page(tr(lang, 'login_title'), _login_form(nxt, err=_info, lang=lang), lang=lang)

    @app.route('/register', methods=['GET', 'POST'])
    def auth_register():
        from web.i18n import normalize_language, tr
        # A `tg` token (from the Telegram bot's /start link) ties this signup to
        # a Telegram chat → email is auto-confirmed (Telegram verified the user),
        # and the admin is notified in Telegram with Approve/Reject buttons.
        tg_raw = request.values.get('tg') or ''
        _tg = _read_tg_token(tg_raw) if tg_raw else None
        tg_chat = _tg.get('c') if _tg else None
        tg_user = _tg.get('u') if _tg else None
        tg_name = _tg.get('n') if _tg else None
        tg_lang = _tg.get('l') if _tg else None
        tg_premium = _tg.get('p') if _tg else None
        lang = normalize_language(request.values.get('lang') or
                                  request.form.get('language') or tg_lang)
        if not tg_chat:
            return _page(tr(lang, 'register_title'), _tg_register_prompt(lang), lang=lang)
        if request.method == 'POST':
            login = _norm_email(request.form.get('login') or request.form.get('email'))
            pw = request.form.get('password') or ''
            pw2 = request.form.get('password2') or ''
            if not login or len(login) < 3:
                return _page(tr(lang, 'register_title'), _register_form(
                    err=tr(lang, 'error_login_min'), tg=tg_raw, lang=lang), lang=lang)
            email = login
            if len(pw) < MIN_PASSWORD_LEN:
                return _page(tr(lang, 'register_title'), _register_form(
                    err=tr(lang, 'error_password_min'), tg=tg_raw, lang=lang), lang=lang)
            if pw != pw2:
                return _page(tr(lang, 'register_title'), _register_form(
                    err=tr(lang, 'error_password_match'), tg=tg_raw, lang=lang), lang=lang)
            if get_user_by_email(email):
                return _page(tr(lang, 'register_title'), _register_form(
                    err=tr(lang, 'error_login_taken'), tg=tg_raw, lang=lang), lang=lang)
            uid = create_user(email, pw, is_admin=False, email_confirmed=True,
                              approved=False, telegram_chat_id=tg_chat,
                              telegram_username=tg_user, telegram_name=tg_name)
            try:
                if uid:
                    import time as _t
                    _u = get_user_by_id(uid)
                    _p = _load_prefs(_u) if _u else {}
                    # The explicit form choice — not Telegram's device locale — is
                    # authoritative for the website and all personal notifications.
                    _p['language'] = lang
                    _p['tg'] = {'username': tg_user, 'name': tg_name,
                                'chat_id': str(tg_chat), 'lang': tg_lang,
                                'premium': bool(tg_premium), 'linked_at': _t.time()}
                    _update_user(uid, prefs=json.dumps(_p)[:20000])
            except Exception as _e:
                print(f"[AUTH] save tg profile error: {_e}")
            notify_new_user_to_admin(uid)
            try:
                from web.tg_bot import tg_send
                tg_send(tg_chat, tr(lang, 'access_pending', login=email))
            except Exception:
                pass
            return _page(tr(lang, 'register_title'), _msg_box(
                tr(lang, 'ready'), 'ok', tr(lang, 'registration_saved'),
                extra=f'<div class="links"><a href="/login?lang={lang}" data-i18n="to_login">{tr(lang, "to_login")}</a></div>'),
                lang=lang)
        return _page(tr(lang, 'register_title'), _register_form(tg=tg_raw, lang=lang), lang=lang)

    @app.route('/confirm/<token>')
    def auth_confirm(token):
        email = _read_token(token, 'confirm', CONFIRM_MAX_AGE)
        if not email:
            return _page('Підтвердження', _msg_box(
                'Недійсне посилання', 'err',
                'Посилання підтвердження недійсне або протерміноване.',
                extra='<div class="links"><a href="/login">До входу</a></div>'))
        u = get_user_by_email(email)
        if u:
            _update_user(u.id, email_confirmed=True)
            with _lock:
                _pending_links.pop(_norm_email(email), None)
        return _page('Підтвердження', _msg_box(
            'Email підтверджено', 'ok',
            'Дякуємо! Тепер акаунт має схвалити адміністратор. '
            'Щойно він це зробить — ви зможете увійти.',
            extra='<div class="links"><a href="/login">До входу</a></div>'))

    @app.route('/forgot', methods=['GET', 'POST'])
    def auth_forgot():
        if request.method == 'POST':
            email = _norm_email(request.form.get('email'))
            u = get_user_by_email(email)
            # Always show success (don't leak which emails exist).
            if u:
                token = _make_token(email, 'reset')
                link = f"{_base_url()}/reset/{token}"
                send_email(email, 'Скидання пароля',
                           f'<p>Щоб скинути пароль, перейдіть за посиланням:</p>'
                           f'<p><a href="{link}">{link}</a></p>'
                           f'<p>Дійсне 1 год. Якщо це не ви — проігноруйте.</p>',
                           link=link)
            return _page('Скидання пароля', _msg_box(
                'Перевірте пошту', 'ok',
                'Якщо такий email існує, ми надіслали посилання для скидання пароля.',
                extra='<div class="links"><a href="/login">← До входу</a></div>'))
        return _page('Скидання пароля', _forgot_form())

    @app.route('/reset/<token>', methods=['GET', 'POST'])
    def auth_reset(token):
        email = _read_token(token, 'reset', RESET_MAX_AGE)
        if not email:
            return _page('Скидання пароля', _msg_box(
                'Недійсне посилання', 'err',
                'Посилання скидання недійсне або протерміноване.',
                extra='<div class="links"><a href="/forgot">Надіслати знову</a></div>'))
        if request.method == 'POST':
            pw = request.form.get('password') or ''
            pw2 = request.form.get('password2') or ''
            if len(pw) < MIN_PASSWORD_LEN:
                return _page('Скидання пароля', _reset_form(
                    token, err=f'Пароль мінімум {MIN_PASSWORD_LEN} символів.'))
            if pw != pw2:
                return _page('Скидання пароля', _reset_form(
                    token, err='Паролі не збігаються.'))
            u = get_user_by_email(email)
            if u:
                _update_user(u.id, password_hash=generate_password_hash(pw))
            return _page('Скидання пароля', _msg_box(
                'Пароль змінено', 'ok', 'Тепер увійдіть з новим паролем.',
                extra='<div class="links"><a href="/login">До входу</a></div>'))
        return _page('Скидання пароля', _reset_form(token))

    @app.route('/logout', methods=['GET', 'POST'])
    def auth_logout():
        u = current_user()
        if u is not None:
            _revoke_info_tokens(u)          # вихід = вихід і з інфосайту
        logout_user()
        if request.args.get('next') == 'info':
            # Вихід з інфосайту: сесію бота закрито (інакше «Вхід» миттєво
            # видавав новий токен без пароля) → назад на сайт.
            site = os.getenv('INFO_SITE_URL')
            return redirect(site or '/info/')
        return redirect(url_for('auth_login'))

    @app.route('/api/me/logout', methods=['POST'])
    def api_me_logout():
        """Вихід з інфосайту: відкликати ВСІ токени користувача (токен живе 7
        діб і лежить у localStorage — без відкликання його копія лишалась би
        дійсною) і закрити сесію бота, якщо запит прийшов із cookie."""
        u = current_user()
        if u is not None:
            _revoke_info_tokens(u)
        logout_user()
        return jsonify({'ok': True})

    @app.route('/favicon.ico')
    def favicon():
        from flask import send_from_directory, current_app
        sd = current_app.static_folder
        # Tolerate both the clean name and the accidental double extension
        # (a file uploaded as vsv-logo.png can land as vsv-logo.png.png).
        for name in ('vsv-logo.png', 'vsv-logo.png.png'):
            if sd and os.path.exists(os.path.join(sd, name)):
                return send_from_directory(sd, name, mimetype='image/png')
        return ('', 404)

    # ---- info-site served by the bot itself (same domain → auth + API work) ----
    _INFOSITE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'info')

    @app.route('/info')
    def info_site_root():
        return redirect('/info/')

    @app.route('/info/')
    def info_site_index():
        from flask import send_from_directory
        return send_from_directory(_INFOSITE_DIR, 'index.html')

    @app.route('/info/<path:fname>')
    def info_site_asset(fname):
        from flask import send_from_directory
        return send_from_directory(_INFOSITE_DIR, fname)

    @app.route('/info-login')
    def info_login_bounce():
        """After a normal login, hand the SEPARATELY-HOSTED info-site a signed
        token (URL hash) so it can authenticate cross-origin without cookies.
        The «Вхід» button on that site points here via /login?next=/info-login."""
        u = current_user()
        if not u or not _is_active(u):
            return redirect(url_for('auth_login', next='/info-login'))
        site = os.getenv('INFO_SITE_URL')
        tok = _make_info_token(u.id)
        if site:
            sep = '&' if '#' in site else '#'
            return redirect(f"{site.rstrip('/')}{sep}it={tok}")
        # No external site configured → bot-served info-site (same origin).
        return redirect(f"/info/#it={tok}")

    @app.route('/pending')
    def auth_pending():
        u = current_user()
        if not u:
            return redirect(url_for('auth_login'))
        if _is_active(u):
            return redirect('/')
        au = getattr(u, 'access_until', None)
        if au and u.email_confirmed and u.approved and not u.disabled and au < datetime.utcnow():
            txt = ('Термін вашого доступу завершився. Зверніться до адміністратора '
                   'для продовження.')
        elif u.disabled:
            txt = 'Акаунт вимкнено адміністратором.'
        elif not u.email_confirmed:
            txt = ('Підтвердіть свій email за посиланням із листа. '
                   'Після цього акаунт має схвалити адміністратор.')
        else:
            txt = ('Email підтверджено. Акаунт очікує схвалення адміністратора — '
                   'щойно він схвалить, ви отримаєте повний доступ.')
        return _page('Очікування', _msg_box('Акаунт очікує активації', 'ok', txt,
                     extra=f'<div class="links">{u.email} · '
                           f'<a href="/logout">Вийти</a></div>'))

    # ---- 👤 personal cabinet (any active user) ----
    # Вимога 24.11: «Особистий кабінет користувача зроби повною сторінкою і з
    # доступом лише із головної сторінки авторизованого користувача.»
    # Повна сторінка — `admin_pages.render_cabinet`; вхід — лише переходом зі
    # сторінки авторизованого користувача (бот / інфосайт), інакше — на головну.
    @app.route('/cabinet')
    def auth_cabinet():
        u = current_user()
        _info_url = (os.getenv('INFO_SITE_URL') or '').rstrip('/')
        _bot_ok = bool(u.is_admin or getattr(u, 'bot_access', False))
        home = cabinet_home(_bot_ok, request.args.get('from') == 'info', _info_url)
        if not cabinet_entry_ok(request.headers.get('Referer', ''),
                                request.headers.get('Sec-Fetch-Site', ''),
                                request.host, _info_url):
            return redirect(home)
        tgu = getattr(u, 'telegram_username', None)
        tgn = getattr(u, 'telegram_name', None)
        tgc = getattr(u, 'telegram_chat_id', None)
        _prefs = _load_prefs(u)
        au = getattr(u, 'access_until', None)
        if u.is_admin or not au:
            access = 'безлімітний'
        else:
            left = (au - datetime.utcnow()).total_seconds() / 86400
            access = (f'до {au.strftime("%d.%m.%Y")} · ще {left:.0f} дн' if left > 0
                      else f'завершився {au.strftime("%d.%m.%Y")}')
        _pm = _perms_eff(u)
        trade_href = (_info_url or '/info') + '/#trade-card'
        from web.admin_pages import render_cabinet
        ctx = {
            'email': u.email, 'is_admin': bool(u.is_admin),
            'role': 'Адміністратор' if u.is_admin else 'Користувач',
            'tg': ((' '.join(x for x in [tgn, (f'@{tgu}' if tgu else '')] if x) or 'прив’язано')
                   if tgc else '—'),
            'tg_linked': bool(tgc), 'bot_ok': _bot_ok,
            'bot': 'так' if _bot_ok else 'ні (лише інфосайт)', 'access': access,
            'ntf_btc': bool(_prefs.get('notify_btc')), 'ntf_fnd': bool(_prefs.get('notify_funding')),
            'trade': bool(u.is_admin or _pm.get('trade_paper') or _pm.get('trade_real')),
            'trade_href': trade_href, 'home': home,
            'home_lbl': '← На головну' if home != '/' else '← На дашборд',
        }
        return render_cabinet(ctx, (_info_url + '/') if _info_url else '/info/')

    @app.route('/api/me')
    def api_me():
        u = current_user()
        return jsonify({'ok': True, 'email': u.email, 'is_admin': u.is_admin,
                        'prefs': _load_prefs(u),
                        # 🛡 що показувати / дозволяти (сервер перевіряє сам)
                        'perms': _perms_eff(u), 'perm_defs': _up().defs()})

    @app.route('/api/me/password', methods=['POST'])
    def api_me_password():
        u = current_user()
        d = request.get_json(silent=True) or {}
        pw, pw2 = d.get('password') or '', d.get('password2') or ''
        if len(pw) < MIN_PASSWORD_LEN:
            return jsonify({'ok': False, 'error': f'Мінімум {MIN_PASSWORD_LEN} символів'})
        if pw != pw2:
            return jsonify({'ok': False, 'error': 'Паролі не збігаються'})
        _update_user(u.id, password_hash=generate_password_hash(pw))
        return jsonify({'ok': True})

    @app.route('/api/me/prefs', methods=['GET', 'POST'])
    def api_me_prefs():
        u = current_user()
        if request.method == 'POST':
            # 🛡 Користувач пише ЛИШЕ свій простір `ui`: права (`perms`),
            # торгівля, підписки, позначки групи / виходу — серверні ключі, і
            # раніше цей маршрут дозволяв їх перезаписати.
            d = request.get_json(silent=True) or {}
            prefs = _load_prefs(u)
            prefs['ui'] = d if isinstance(d, dict) else {}
            _update_user(u.id, prefs=json.dumps(prefs)[:20000])
            return jsonify({'ok': True})
        return jsonify({'ok': True, 'prefs': _load_prefs(u)})

    @app.route('/api/me/notify', methods=['GET', 'POST'])
    def api_me_notify():
        """Per-user Telegram notification opt-ins (₿ BTCUSDT, 💰 Funding).
        MERGES into prefs (does not clobber other keys). Requires a linked
        Telegram chat to actually receive anything."""
        u = current_user()
        prefs = _load_prefs(u)
        keys = ('notify_btc', 'notify_funding', 'notify_opportunity',
                'notify_spike', 'notify_trades')
        if request.method == 'POST':
            d = request.get_json(silent=True) or {}
            for k in keys:
                if k in d:
                    prefs[k] = bool(d[k])
            _update_user(u.id, prefs=json.dumps(prefs)[:20000])
        out = {'ok': True, 'tg_linked': bool(getattr(u, 'telegram_chat_id', None))}
        for k in keys:
            out[k] = bool(prefs.get(k))
        return jsonify(out)

    @app.route('/api/me/tg-link', methods=['GET', 'POST'])
    def api_me_tg_link():
        """🔗 Привʼязати Telegram до ЦЬОГО акаунта: POST видає одноразове
        посилання на бота (`?start=lk_<код>`, живе 30 хв); GET — лише стан."""
        u = current_user()
        out = {'ok': True, 'tg_linked': bool(getattr(u, 'telegram_chat_id', None)),
               'tg_user': getattr(u, 'telegram_username', None)}
        if request.method == 'POST':
            try:
                from web.tg_bot import _bot_username
                bun = _bot_username()
            except Exception:
                bun = None
            if not bun:
                return jsonify({'ok': False, 'error': 'Бот Telegram недоступний '
                                '(не вдалось дізнатись його імʼя).'})
            code, until = tg_link_issue(u.id)
            out.update(url=f"https://t.me/{bun}?start={TG_LINK_PREFIX}{code}",
                       until=until, bot=bun)
        return jsonify(out)

    # ---- admin ----
    @app.route('/admin/users')
    def admin_users_page():
        u = current_user()
        if not admin_access_ok(u, getattr(g, '_via_token', False)):
            return abort(403)
        from web.admin_pages import render_list
        return render_list(u.email, os.getenv('INFO_SITE_URL'))

    @app.route('/admin/users/<int:uid>')
    def admin_user_page(uid):
        """🛡 «Керувати» — окрема сторінка користувача (24.11), не модальне вікно."""
        u = current_user()
        if not admin_access_ok(u, getattr(g, '_via_token', False)):
            return abort(403)
        from web.admin_pages import render_user
        return render_user(uid, u.email, os.getenv('INFO_SITE_URL'))

    @app.route('/api/admin/users')
    def api_admin_users():
        if not current_user().is_admin:
            return jsonify({'ok': False, 'error': 'forbidden'}), 403
        with _lock:
            links = {k: v for k, v in _pending_links.items()}
        return jsonify({'ok': True, 'users': list_users(),
                        'perm_defs': _up().defs(),
                        'smtp': _smtp_configured(),
                        'smtp_status': smtp_status(),
                        'pending': pending_attention_count(),
                        'pending_links': links})

    @app.route('/api/admin/smtp-test', methods=['POST'])
    def api_admin_smtp_test():
        if not current_user().is_admin:
            return jsonify({'ok': False, 'error': 'forbidden'}), 403
        d = request.get_json(silent=True) or {}
        to = _norm_email(d.get('to')) or current_user().email
        ok = send_email(to, 'VSV Bot — тест пошти',
                        '<p>✅ Тестовий лист від VSV Bot. Якщо ви це бачите — '
                        'пошта налаштована правильно.</p>')
        # Also exercise the Telegram link channel so the admin can confirm it.
        tg = _tg_notify_link(to, 'Тест Telegram (з панелі)',
                             _base_url() + '/admin/users')
        return jsonify({'ok': ok, 'to': to, 'tg': tg,
                        'tg_configured': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
                        'status': smtp_status()})

    @app.route('/api/admin/users/<int:uid>', methods=['POST'])
    def api_admin_user_action(uid):
        me = current_user()
        if not me.is_admin:
            return jsonify({'ok': False, 'error': 'forbidden'}), 403
        d = request.get_json(silent=True) or {}
        action = d.get('action')
        target = get_user_by_id(uid)
        if not target:
            return jsonify({'ok': False, 'error': 'not found'})
        # Notify BOTH sides about the change: the user (their Telegram) + the
        # admin (audit line to the admin's private chat via the 'register' cat).
        _tgc = getattr(target, 'telegram_chat_id', None)
        _lbl = target.email

        notification_sent = None

        def _notify_change(user_text=None, admin_text=None):
            nonlocal notification_sent
            from web.tg_bot import notify_category
            from web.i18n import user_tr
            if user_text:
                # Existing call sites stay readable; personal delivery is always
                # translated from the recipient's saved registration preference.
                keys = {
                    "⚠️ Ваш доступ призупинено адміністратором.": 'access_suspended',
                    "🤖 Вам надано доступ до бота.": 'bot_granted',
                    "🤖 Доступ до бота вимкнено.": 'bot_revoked',
                    "⛔ Ваш акаунт вимкнено адміністратором.": 'account_disabled',
                    "✅ Ваш акаунт знову активний.": 'account_enabled',
                    "🛡 Вам надано права адміністратора.": 'admin_granted',
                    "🛡 Права адміністратора знято.": 'admin_revoked',
                    "🔑 Адміністратор змінив ваш пароль. Використовуйте новий пароль для входу.": 'password_changed',
                    "🔒 Ваші сесії завершено — увійдіть знову.": 'sessions_closed',
                    "🗑 Ваш акаунт видалено адміністратором.": 'account_deleted',
                }
                key = keys.get(user_text)
                text = user_tr(uid, key) if key else user_text
                notification_sent = _access_delivery(uid, 'account_update', text)
            if admin_text:
                try:
                    notify_category('register', admin_text)
                except Exception:
                    pass
            return notification_sent

        if action == 'approve':
            notification_sent = approve_user(uid)
            status = "користувача повідомлено" if notification_sent else "повідомлення НЕ доставлено"
            _notify_change(admin_text=f"✅ <b>{_lbl}</b> — схвалено (30 днів); {status}.")
        elif action == 'revoke':
            _update_user(uid, approved=False)
            _notify_change("⚠️ Ваш доступ призупинено адміністратором.",
                           f"⚠️ <b>{_lbl}</b> — схвалення знято.")
            sync_user_group(uid)
        elif action == 'grant_bot':
            _update_user(uid, bot_access=True)
            _notify_change("🤖 Вам надано доступ до бота.",
                           f"🤖 <b>{_lbl}</b> — надано доступ до бота.")
        elif action == 'revoke_bot':
            _update_user(uid, bot_access=False)
            _notify_change("🤖 Доступ до бота вимкнено.",
                           f"🤖 <b>{_lbl}</b> — доступ до бота знято.")
        elif action == 'confirm':          # admin can confirm email manually
            _update_user(uid, email_confirmed=True)
        elif action == 'disable':
            if target.id == me.id:
                return jsonify({'ok': False, 'error': 'не можна вимкнути себе'})
            _update_user(uid, disabled=True, session_token=None)
            _notify_change("⛔ Ваш акаунт вимкнено адміністратором.",
                           f"⛔ <b>{_lbl}</b> — вимкнено.")
            sync_user_group(uid)
        elif action == 'enable':
            _update_user(uid, disabled=False)
            _notify_change("✅ Ваш акаунт знову активний.",
                           f"✅ <b>{_lbl}</b> — увімкнено.")
            sync_user_group(uid)
        elif action == 'make_admin':
            _update_user(uid, is_admin=True, approved=True, email_confirmed=True)
            _notify_change("🛡 Вам надано права адміністратора.",
                           f"🛡 <b>{_lbl}</b> → адміністратор.")
        elif action == 'remove_admin':
            if target.id == me.id:
                return jsonify({'ok': False, 'error': 'не можна зняти себе'})
            _update_user(uid, is_admin=False)
            _notify_change("🛡 Права адміністратора знято.",
                           f"🛡 <b>{_lbl}</b> — знято адміна.")
        elif action == 'set_password':
            pw = d.get('password') or ''
            if len(pw) < MIN_PASSWORD_LEN:
                return jsonify({'ok': False, 'error': f'Мінімум {MIN_PASSWORD_LEN} символів'})
            _update_user(uid, password_hash=generate_password_hash(pw))
            _notify_change("🔑 Адміністратор змінив ваш пароль. "
                           "Використовуйте новий пароль для входу.",
                           f"🔑 <b>{_lbl}</b> — пароль змінено.")
        elif action == 'set_access':
            # Time-limited access: days>0 → active for N days then auto-block;
            # days==0 → unlimited. Also approves + enables the account.
            try:
                days = int(d.get('days') or 0)
            except (TypeError, ValueError):
                days = 0
            au = (datetime.utcnow() + timedelta(days=days)) if days > 0 else None
            _update_user(uid, access_until=au, approved=True, disabled=False)
            # 🔕 Користувачу про зміну/продовження терміну НЕ пишемо (вимога
            # 09.11); групу й позначку `ever_approved` відновлюємо як і раніше.
            notify_user_approved(uid, silent=True)
            _dtxt = f"{days} дн." if days > 0 else "без обмеження"
            _notify_change(admin_text=f"⏳ <b>{_lbl}</b> — доступ {_dtxt}.")
        elif action == 'set_perms':
            # 🛡 що користувачу показувати / дозволяти на інфосайті (24.11)
            if target.is_admin:
                return jsonify({'ok': False, 'error': 'адміністратор має доступ до всього'})
            _tp = _load_prefs(target)
            _tp['perms'] = _up().apply(_tp.get('perms'), d)
            _update_user(uid, prefs=json.dumps(_tp)[:20000])
            for _inv in ('sbr_subs_invalidate', 'anom_subs_invalidate'):
                try:
                    import web.info_portal as _ip
                    getattr(_ip, _inv)()
                except Exception:
                    pass
            try:
                from web.user_trading import trade_subs_invalidate
                trade_subs_invalidate()
            except Exception:
                pass
            return jsonify({'ok': True, 'perms': _tp['perms'],
                            'perms_eff': _up().effective(False, _tp['perms'])})
        elif action == 'group_sync':
            # 🔄 привести Telegram-групу до доступу НАНОВО (бан не пройшов —
            # бот не був адміном тощо)
            n = sync_user_group(uid, force=True)
            return jsonify({'ok': True, 'n': n})
        elif action == 'kick':
            # Invalidate all live sessions of this user (force re-login).
            _update_user(uid, session_token=None)
            _notify_change("🔒 Ваші сесії завершено — увійдіть знову.",
                           f"🔒 <b>{_lbl}</b> — сесії скинуто.")
        elif action == 'resend':
            token = _make_token(target.email, 'confirm')
            link = f"{_base_url()}/confirm/{token}"
            send_email(target.email, 'Підтвердження реєстрації',
                       f'<p>Підтвердіть email для доступу до бота:</p>'
                       f'<p><a href="{link}">{link}</a></p>', link=link)
        elif action == 'delete':
            if target.id == me.id:
                return jsonify({'ok': False, 'error': 'не можна видалити себе'})
            # Notify the user BEFORE removing them (their chat link is gone after).
            _notify_change("🗑 Ваш акаунт видалено адміністратором.",
                           f"🗑 <b>{_lbl}</b> — акаунт видалено.")
            try:                       # ВИДАЛЕНО: бан у групі + слід у боті (ДО видалення —
                from web.tg_bot import forget_bot_user   # потім chat_id вже немає)
                if target.telegram_chat_id:
                    forget_bot_user(target.telegram_chat_id)
            except Exception:
                pass
            _delete_user(uid)
        else:
            return jsonify({'ok': False, 'error': 'unknown action'})
        return jsonify({'ok': True, 'notification_sent': notification_sent})

    @app.route('/api/admin/users/create', methods=['POST'])
    def api_admin_user_create():
        if not current_user().is_admin:
            return jsonify({'ok': False, 'error': 'forbidden'}), 403
        d = request.get_json(silent=True) or {}
        # Registration is Telegram-only (login + password, NO email) — the admin
        # create form matches that: a LOGIN (display name ≥ 3 chars), not an email.
        login = _norm_email(d.get('login') or d.get('email'))
        pw = d.get('password') or ''
        if not login or len(login) < 3:
            return jsonify({'ok': False, 'error': 'Логін — мінімум 3 символи'})
        if len(pw) < MIN_PASSWORD_LEN:
            return jsonify({'ok': False, 'error': f'Пароль мінімум {MIN_PASSWORD_LEN} символів'})
        if get_user_by_email(login):
            return jsonify({'ok': False, 'error': 'Такий логін уже існує'})
        _tgc = (str(d.get('telegram_chat_id') or '').strip() or None)
        _tgu = ((d.get('telegram_username') or '').lstrip('@').strip() or None)
        # Admin-created accounts are ACTIVE immediately (approved), get a 30-day
        # window, bot access per the checkbox, and optional Telegram linking.
        uid = create_user(login, pw, is_admin=bool(d.get('is_admin')),
                          email_confirmed=True, approved=True,
                          bot_access=bool(d.get('bot_access')),
                          access_days=DEFAULT_ACCESS_DAYS,
                          telegram_chat_id=_tgc, telegram_username=_tgu)
        # If linked to Telegram, tell the new user their account is ready.
        if uid and _tgc:
            try:
                notify_user_approved(uid)
            except Exception:
                pass
        return jsonify({'ok': True})


def _delete_user(uid):
    # 🔐 ключі біржі й особистий Trade Manager користувача — разом із ним
    try:
        from web import user_keys
        user_keys.purge(uid)
    except Exception:
        pass
    try:
        from detection import user_tm
        user_tm.drop(uid)
    except Exception:
        pass
    s = get_session()
    try:
        u = s.query(User).filter_by(id=int(uid)).first()
        if u:
            s.delete(u)
            s.commit()
    except Exception:
        s.rollback()
    finally:
        s.close()


def _load_prefs(u):
    try:
        return json.loads(u.prefs or '{}')
    except Exception:
        return {}


# ---- form/body builders ----
def _msg_box(title, cls, text, extra=''):
    return f'<h1>{title}</h1><div class="msg {cls}">{text}</div>{extra}'


def _no_bot_access_page(u):
    # Info-only users land straight on the bot-served info-site (or an external
    # one if INFO_SITE_URL is set). No dead-end "no access" page.
    return redirect(os.getenv('INFO_SITE_URL') or '/info/')


def _bot_link():
    return os.getenv('TELEGRAM_BOT_LINK') or ''


def _language_field(lang='uk'):
    from web.i18n import language_options, tr, normalize_language
    lang = normalize_language(lang)
    return ('<div class="language-row"><span class="lang-icon" aria-hidden="true">◎</span>'
            f'<label for="profile-language" data-i18n="language">{tr(lang, "language")}</label>'
            f'<select id="profile-language" name="language" aria-label="Language">'
            f'{language_options(lang)}</select></div>')


def _tg_register_prompt(lang='uk'):
    """Registration is Telegram-only — no email form. Point the user at the bot."""
    from web.i18n import tr
    bl = _bot_link()
    btn = (f'<a class="tgbtn" href="{bl}" data-i18n="open_telegram">{tr(lang, "open_telegram")}</a>' if bl else
           f'<div class="msg ok" data-i18n="find_bot">{tr(lang, "find_bot")}</div>')
    return (f'<h1>📝 <span data-i18n="register_title">{tr(lang, "register_title")}</span></h1>'
            f'{_language_field(lang)}'
            f'<p class="sub" data-i18n="registration_only_telegram">{tr(lang, "registration_only_telegram")}</p>'
            f'<div style="text-align:center;margin:8px 0 4px">{btn}</div>'
            f'<p class="sub" style="text-align:center;margin-top:14px" data-i18n="start_bot_hint">'
            f'{tr(lang, "start_bot_hint")}</p>'
            f'<div class="links"><a href="/login" data-lang-link data-i18n="have_account">{tr(lang, "have_account")}</a></div>')


def _login_form(nxt='/', err='', lang='uk'):
    from markupsafe import escape
    from web.i18n import tr
    e = f'<div class="msg err">{escape(err)}</div>' if err else ''
    bl = _bot_link()
    reg = (f'<a href="{bl}" data-i18n="register_telegram">{tr(lang, "register_telegram")}</a>' if bl
           else f'<a href="/register" data-lang-link data-i18n="register_title">{tr(lang, "register_title")}</a>')
    return (f'<h1>🔐 <span data-i18n="login_title">{tr(lang, "login_title")}</span></h1>'
            f'<p class="sub" data-i18n="login_subtitle">{tr(lang, "login_subtitle")}</p>{e}'
            f'<form method="post">{_language_field(lang)}'
            f'<input type="hidden" name="next" value="{escape(nxt)}">'
            f'<label data-i18n="login_name">{tr(lang, "login_name")}</label>'
            f'<input name="email" type="text" required autofocus autocomplete="username" '
            f'data-i18n-placeholder="login_placeholder" placeholder="{tr(lang, "login_placeholder")}">'
            f'<label data-i18n="password">{tr(lang, "password")}</label>'
            f'<input name="password" type="password" required autocomplete="current-password">'
            f'<button type="submit" data-i18n="login">{tr(lang, "login")}</button></form>'
            f'<div class="links">{reg}</div>')


def _register_form(err='', tg='', lang='uk'):
    from markupsafe import escape
    from web.i18n import tr
    e = f'<div class="msg err">{escape(err)}</div>' if err else ''
    tg_hidden = f'<input type="hidden" name="tg" value="{escape(tg)}">' if tg else ''
    tg_note = (f'<div class="msg ok" style="font-size:.78rem" data-i18n="telegram_confirmed">'
               f'{tr(lang, "telegram_confirmed")}</div>')
    return (f'<h1>📝 <span data-i18n="register_title">{tr(lang, "register_title")}</span></h1>'
            f'<p class="sub" data-i18n="register_subtitle">{tr(lang, "register_subtitle")}</p>{e}{tg_note}'
            f'<form method="post">{tg_hidden}{_language_field(lang)}'
            f'<label data-i18n="register_name">{tr(lang, "register_name")}</label>'
            f'<input name="login" type="text" required autofocus autocomplete="username" '
            f'data-i18n-placeholder="register_placeholder" placeholder="{tr(lang, "register_placeholder")}">'
            f'<label data-i18n="password_min">{tr(lang, "password_min")}</label>'
            f'<input name="password" type="password" required autocomplete="new-password">'
            f'<label data-i18n="password_repeat">{tr(lang, "password_repeat")}</label>'
            f'<input name="password2" type="password" required autocomplete="new-password">'
            f'<button type="submit" data-i18n="register">{tr(lang, "register")}</button></form>'
            f'<div class="links"><a href="/login" data-lang-link data-i18n="have_account">{tr(lang, "have_account")}</a></div>')


def _forgot_form(err=''):
    e = f'<div class="msg err">{err}</div>' if err else ''
    return (f'<h1>🔑 Скидання пароля</h1><p class="sub">Надішлемо посилання на пошту</p>{e}'
            f'<form method="post">'
            f'<label>Email</label><input name="email" type="email" required autofocus>'
            f'<button type="submit">Надіслати посилання</button></form>'
            f'<div class="links"><a href="/login">← До входу</a></div>')


def _reset_form(token, err=''):
    e = f'<div class="msg err">{err}</div>' if err else ''
    return (f'<h1>🔑 Новий пароль</h1>{e}'
            f'<form method="post">'
            f'<label>Новий пароль (мін. 8)</label><input name="password" type="password" required autofocus>'
            f'<label>Повторіть пароль</label><input name="password2" type="password" required>'
            f'<button type="submit">Зберегти пароль</button></form>')


# ==================================================================
# One-call installer
# ==================================================================
def init_auth(app):
    """Wire auth into a Flask app: routes + gate + admin bootstrap."""
    register_auth_routes(app)
    install_auth_gate(app)
    try:
        _migrate_user_columns()
    except Exception as e:
        print(f"[AUTH] migrate error: {e}")
    try:
        ensure_admin()
    except Exception as e:
        print(f"[AUTH] ensure_admin error: {e}")
    try:
        from web.tg_bot import start_tg_bot
        start_tg_bot()
    except Exception as e:
        print(f"[AUTH] tg-bot start error: {e}")
