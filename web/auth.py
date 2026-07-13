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
import smtplib
import threading
from datetime import datetime
from email.message import EmailMessage

from flask import (session, request, redirect, url_for, jsonify, g,
                   render_template_string, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from storage.db_models import get_session, User

# ---- policy constants ----
MIN_PASSWORD_LEN = 8
CONFIRM_MAX_AGE = 24 * 3600      # email-confirm token valid 24h
RESET_MAX_AGE = 3600             # password-reset token valid 1h
LOGIN_MAX_FAILS = 8              # per email+ip within window
LOGIN_WINDOW = 900               # 15 min
_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

# Dev fallback: last confirm/reset link per email (shown to admin when SMTP off)
_pending_links = {}
_login_fails = {}                # {(email,ip): [timestamps]}
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


def create_user(email, password, is_admin=False, email_confirmed=False,
                approved=False):
    s = get_session()
    try:
        u = User(email=_norm_email(email),
                 password_hash=generate_password_hash(password),
                 is_admin=bool(is_admin),
                 email_confirmed=bool(email_confirmed),
                 approved=bool(approved),
                 disabled=False, prefs='{}',
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


def list_users():
    s = get_session()
    try:
        return [{
            'id': u.id, 'email': u.email, 'is_admin': u.is_admin,
            'email_confirmed': u.email_confirmed, 'approved': u.approved,
            'disabled': u.disabled,
            'active': _is_active(u),
            'created_at': u.created_at.isoformat() if u.created_at else None,
            'last_login': u.last_login.isoformat() if u.last_login else None,
        } for u in s.query(User).order_by(User.created_at.desc()).all()]
    finally:
        s.close()


def user_count():
    s = get_session()
    try:
        return s.query(User).count()
    finally:
        s.close()


def _is_active(u):
    if u is None:
        return False
    if u.disabled:
        return False
    if u.is_admin:
        return True
    return bool(u.email_confirmed and u.approved)


# ==================================================================
# Session helpers
# ==================================================================
def login_user(u):
    session.permanent = True
    session['uid'] = u.id
    session['is_admin'] = bool(u.is_admin)
    _update_user(u.id, last_login=datetime.utcnow())


def logout_user():
    session.pop('uid', None)
    session.pop('is_admin', None)


def current_user():
    """Cached-per-request current user object (or None)."""
    if 'uid' not in session:
        return None
    if getattr(g, '_cur_user', None) is not None:
        return g._cur_user
    u = get_user_by_id(session['uid'])
    g._cur_user = u
    return u


# ==================================================================
# Mailer (SMTP via env, with dev fallback)
# ==================================================================
def _smtp_configured():
    return bool(os.getenv('SMTP_HOST'))


def send_email(to, subject, html, link=None):
    """Send an email via SMTP env config. Returns True on send. When SMTP is
    NOT configured, stores the link for the admin panel + logs it (dev mode)
    and returns False (caller still succeeds — the flow isn't blocked)."""
    if link:
        with _lock:
            _pending_links[_norm_email(to)] = {'link': link, 'subject': subject,
                                               'at': time.time()}
    if not _smtp_configured():
        print(f"[AUTH][MAIL-DEV] to={to} · {subject}\n  LINK: {link}")
        return False
    try:
        host = os.getenv('SMTP_HOST')
        port = int(os.getenv('SMTP_PORT', '587') or 587)
        user = os.getenv('SMTP_USER', '')
        pwd = os.getenv('SMTP_PASS', '')
        sender = os.getenv('SMTP_FROM', user or 'no-reply@bot.local')
        use_tls = str(os.getenv('SMTP_TLS', '1')).lower() in ('1', 'true', 'yes')
        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = to
        msg.set_content(re.sub('<[^<]+?>', '', html))   # plaintext fallback
        msg.add_alternative(html, subtype='html')
        if port == 465:
            ctx = ssl.create_default_context()
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=20) as srv:
                if user:
                    srv.login(user, pwd)
                srv.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as srv:
                if use_tls:
                    srv.starttls(context=ssl.create_default_context())
                if user:
                    srv.login(user, pwd)
                srv.send_message(msg)
        return True
    except Exception as e:
        print(f"[AUTH][MAIL] send error to {to}: {e}")
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
                 '/api/health', '/favicon.ico'}
_PUBLIC_PREFIX = ('/static/', '/confirm/', '/reset/', '/auth/')
_USER_MUTABLE_PREFIX = ('/api/me', '/logout', '/auth/')   # non-admin may POST here


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
        if _is_public(path):
            return None
        # Controlled read-only bypass for the separate info-site: GET /api/* with
        # a matching X-API-Key. OFF unless INFO_API_KEY env is set (full lockdown
        # by default). Read-only (GET/HEAD) — never enables writes.
        _api_key = os.getenv('INFO_API_KEY')
        if (_api_key and path.startswith('/api/')
                and request.method in ('GET', 'HEAD', 'OPTIONS')
                and request.headers.get('X-API-Key') == _api_key):
            return None
        u = current_user()
        if u is None:
            if _wants_json(path):
                return jsonify({'ok': False, 'error': 'auth_required'}), 401
            return redirect(url_for('auth_login', next=path))
        if not _is_active(u):
            # Logged in but pending confirm/approval → block everything else.
            if _wants_json(path):
                return jsonify({'ok': False, 'error': 'not_active',
                                'email_confirmed': bool(u.email_confirmed),
                                'approved': bool(u.approved)}), 403
            return redirect(url_for('auth_pending'))
        # Active. Admins unrestricted.
        if u.is_admin:
            return None
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
        # Expose the current user to Jinja templates (base.html nav chip).
        try:
            return {'auth_user': current_user()}
        except Exception:
            return {'auth_user': None}

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
_SHELL = """<!doctype html><html lang="uk"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{title}} · VSV Bot</title>
<link rel="icon" type="image/png" href="/favicon.ico">
<link rel="apple-touch-icon" href="/favicon.ico"><style>
.brandlogo{display:block;width:76px;height:76px;margin:0 auto 14px;border-radius:16px;object-fit:cover}
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
 background:#0b0e14;color:#e5e7eb;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.card{width:100%;max-width:400px;margin:20px;padding:26px 24px;border-radius:14px;
 background:#141922;border:1px solid rgba(255,255,255,0.08);box-shadow:0 10px 40px rgba(0,0,0,.5)}
h1{font-size:1.15rem;margin:0 0 4px} .sub{color:#9aa3b5;font-size:.82rem;margin:0 0 18px}
label{display:block;font-size:.75rem;color:#cbd5e1;margin:12px 0 5px}
input{width:100%;padding:10px 12px;border-radius:8px;border:1px solid #2a3140;background:#0e1219;
 color:#e5e7eb;font-size:.9rem}
button{width:100%;margin-top:18px;padding:11px;border:0;border-radius:8px;font-weight:700;
 font-size:.9rem;cursor:pointer;background:#2563eb;color:#fff}
button:hover{background:#1d4ed8}
.msg{margin:12px 0;padding:10px 12px;border-radius:8px;font-size:.82rem}
.err{background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.4);color:#fca5a5}
.ok{background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.4);color:#86efac}
.links{margin-top:16px;font-size:.8rem;text-align:center;color:#9aa3b5}
.links a{color:#60a5fa;text-decoration:none} .links a:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;font-size:.8rem} th,td{padding:7px 9px;text-align:left;
 border-bottom:1px solid rgba(255,255,255,.06)} th{color:#9aa3b5;font-weight:600}
.badge{padding:2px 8px;border-radius:20px;font-size:.68rem;font-weight:700}
.b-on{background:rgba(34,197,94,.15);color:#86efac} .b-off{background:rgba(148,163,184,.15);color:#cbd5e1}
.b-adm{background:rgba(250,204,21,.15);color:#fde68a}
.actbtn{padding:4px 9px;font-size:.72rem;border-radius:6px;border:1px solid #2a3140;background:#1a2130;
 color:#e5e7eb;cursor:pointer;margin:1px}
</style></head><body><div class="card" style="max-width:{{width or 400}}px">
<img class="brandlogo" src="/favicon.ico" alt="VSV" onerror="this.style.display='none'">
{{body|safe}}</div>
<script>{{script|safe}}</script></body></html>"""


def _page(title, body, script='', width=400):
    return render_template_string(_SHELL, title=title, body=body, script=script,
                                  width=width)


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


# ==================================================================
# Routes
# ==================================================================
def register_auth_routes(app):

    @app.route('/login', methods=['GET', 'POST'])
    def auth_login():
        if current_user() and _is_active(current_user()):
            return redirect('/')
        nxt = request.args.get('next') or request.form.get('next') or '/'
        if request.method == 'POST':
            email = _norm_email(request.form.get('email'))
            pw = request.form.get('password') or ''
            if _throttled(email):
                return _page('Вхід', _login_form(nxt,
                             err='Забагато спроб. Спробуйте за кілька хвилин.'))
            u = get_user_by_email(email)
            if not u or not check_password_hash(u.password_hash, pw):
                _record_fail(email)
                return _page('Вхід', _login_form(nxt, err='Невірний email або пароль.'))
            if u.disabled:
                return _page('Вхід', _login_form(nxt, err='Акаунт вимкнено адміністратором.'))
            login_user(u)
            if not _is_active(u):
                return redirect(url_for('auth_pending'))
            return redirect(nxt if nxt.startswith('/') else '/')
        return _page('Вхід', _login_form(nxt))

    @app.route('/register', methods=['GET', 'POST'])
    def auth_register():
        if request.method == 'POST':
            email = _norm_email(request.form.get('email'))
            pw = request.form.get('password') or ''
            pw2 = request.form.get('password2') or ''
            if not _EMAIL_RE.match(email):
                return _page('Реєстрація', _register_form(err='Невірний email.'))
            if len(pw) < MIN_PASSWORD_LEN:
                return _page('Реєстрація', _register_form(
                    err=f'Пароль мінімум {MIN_PASSWORD_LEN} символів.'))
            if pw != pw2:
                return _page('Реєстрація', _register_form(err='Паролі не збігаються.'))
            if get_user_by_email(email):
                return _page('Реєстрація', _register_form(
                    err='Такий email уже зареєстрований.'))
            create_user(email, pw, is_admin=False, email_confirmed=False,
                        approved=False)
            token = _make_token(email, 'confirm')
            link = f"{_base_url()}/confirm/{token}"
            send_email(email, 'Підтвердження реєстрації',
                       f'<p>Підтвердіть email для доступу до бота:</p>'
                       f'<p><a href="{link}">{link}</a></p>'
                       f'<p>Посилання дійсне 24 год.</p>', link=link)
            return _page('Реєстрація', _msg_box(
                'Майже готово', 'ok',
                'Ми надіслали лист із підтвердженням на вашу пошту. '
                'Після підтвердження email акаунт має схвалити адміністратор — '
                'тоді ви зможете увійти.',
                extra='<div class="links"><a href="/login">← До входу</a></div>'))
        return _page('Реєстрація', _register_form())

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
        logout_user()
        return redirect(url_for('auth_login'))

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

    @app.route('/pending')
    def auth_pending():
        u = current_user()
        if not u:
            return redirect(url_for('auth_login'))
        if _is_active(u):
            return redirect('/')
        if not u.email_confirmed:
            txt = ('Підтвердіть свій email за посиланням із листа. '
                   'Після цього акаунт має схвалити адміністратор.')
        else:
            txt = ('Email підтверджено. Акаунт очікує схвалення адміністратора — '
                   'щойно він схвалить, ви отримаєте повний доступ.')
        return _page('Очікування', _msg_box('Акаунт очікує активації', 'ok', txt,
                     extra=f'<div class="links">{u.email} · '
                           f'<a href="/logout">Вийти</a></div>'))

    # ---- personal cabinet (any active user) ----
    @app.route('/cabinet')
    def auth_cabinet():
        u = current_user()
        role = 'Адміністратор' if u.is_admin else 'Користувач (перегляд)'
        body = (f'<h1>👤 Особистий кабінет</h1>'
                f'<p class="sub">{u.email} · {role}</p>'
                f'<label>Змінити пароль</label>'
                f'<input id="np" type="password" placeholder="Новий пароль (мін. 8)">'
                f'<input id="np2" type="password" placeholder="Повторіть пароль" style="margin-top:8px">'
                f'<button onclick="chpw()">Змінити пароль</button>'
                f'<div id="m" class="msg" style="display:none"></div>'
                f'<div class="links">'
                + ('<a href="/admin/users">🛡 Адмін-панель</a> · ' if u.is_admin else '')
                + '<a href="/">← На дашборд</a> · <a href="/logout">Вийти</a></div>')
        script = """
        async function chpw(){
          const p=document.getElementById('np').value, p2=document.getElementById('np2').value;
          const m=document.getElementById('m'); m.style.display='block';
          const r=await fetch('/api/me/password',{method:'POST',headers:{'Content-Type':'application/json'},
            body:JSON.stringify({password:p,password2:p2})});
          const d=await r.json();
          m.className='msg '+(d.ok?'ok':'err'); m.textContent=d.ok?'Пароль змінено.':(d.error||'Помилка');
          if(d.ok){document.getElementById('np').value='';document.getElementById('np2').value='';}
        }"""
        return _page('Кабінет', body, script=script, width=440)

    @app.route('/api/me')
    def api_me():
        u = current_user()
        return jsonify({'ok': True, 'email': u.email, 'is_admin': u.is_admin,
                        'prefs': _load_prefs(u)})

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
            d = request.get_json(silent=True) or {}
            _update_user(u.id, prefs=json.dumps(d)[:20000])
            return jsonify({'ok': True})
        return jsonify({'ok': True, 'prefs': _load_prefs(u)})

    # ---- admin ----
    @app.route('/admin/users')
    def admin_users_page():
        u = current_user()
        if not u.is_admin:
            return abort(403)
        return _page('Користувачі', _admin_users_body(), script=_admin_script(),
                     width=1120)

    @app.route('/api/admin/users')
    def api_admin_users():
        if not current_user().is_admin:
            return jsonify({'ok': False, 'error': 'forbidden'}), 403
        with _lock:
            links = {k: v for k, v in _pending_links.items()}
        return jsonify({'ok': True, 'users': list_users(),
                        'smtp': _smtp_configured(), 'pending_links': links})

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
        if action == 'approve':
            _update_user(uid, approved=True)
        elif action == 'revoke':
            _update_user(uid, approved=False)
        elif action == 'confirm':          # admin can confirm email manually
            _update_user(uid, email_confirmed=True)
        elif action == 'disable':
            if target.id == me.id:
                return jsonify({'ok': False, 'error': 'не можна вимкнути себе'})
            _update_user(uid, disabled=True)
        elif action == 'enable':
            _update_user(uid, disabled=False)
        elif action == 'make_admin':
            _update_user(uid, is_admin=True, approved=True, email_confirmed=True)
        elif action == 'remove_admin':
            if target.id == me.id:
                return jsonify({'ok': False, 'error': 'не можна зняти себе'})
            _update_user(uid, is_admin=False)
        elif action == 'set_password':
            pw = d.get('password') or ''
            if len(pw) < MIN_PASSWORD_LEN:
                return jsonify({'ok': False, 'error': f'Мінімум {MIN_PASSWORD_LEN} символів'})
            _update_user(uid, password_hash=generate_password_hash(pw))
        elif action == 'resend':
            token = _make_token(target.email, 'confirm')
            link = f"{_base_url()}/confirm/{token}"
            send_email(target.email, 'Підтвердження реєстрації',
                       f'<p>Підтвердіть email для доступу до бота:</p>'
                       f'<p><a href="{link}">{link}</a></p>', link=link)
        elif action == 'delete':
            if target.id == me.id:
                return jsonify({'ok': False, 'error': 'не можна видалити себе'})
            _delete_user(uid)
        else:
            return jsonify({'ok': False, 'error': 'unknown action'})
        return jsonify({'ok': True})

    @app.route('/api/admin/users/create', methods=['POST'])
    def api_admin_user_create():
        if not current_user().is_admin:
            return jsonify({'ok': False, 'error': 'forbidden'}), 403
        d = request.get_json(silent=True) or {}
        email = _norm_email(d.get('email'))
        pw = d.get('password') or ''
        if not _EMAIL_RE.match(email):
            return jsonify({'ok': False, 'error': 'Невірний email'})
        if len(pw) < MIN_PASSWORD_LEN:
            return jsonify({'ok': False, 'error': f'Пароль мінімум {MIN_PASSWORD_LEN} символів'})
        if get_user_by_email(email):
            return jsonify({'ok': False, 'error': 'Такий email уже існує'})
        # Admin-created accounts are ACTIVE immediately (email confirmed + approved).
        create_user(email, pw, is_admin=bool(d.get('is_admin')),
                    email_confirmed=True, approved=True)
        return jsonify({'ok': True})


def _delete_user(uid):
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


def _login_form(nxt='/', err=''):
    e = f'<div class="msg err">{err}</div>' if err else ''
    return (f'<h1>🔐 Вхід</h1><p class="sub">Авторизуйтесь для доступу до бота</p>{e}'
            f'<form method="post">'
            f'<input type="hidden" name="next" value="{nxt}">'
            f'<label>Email</label><input name="email" type="email" required autofocus>'
            f'<label>Пароль</label><input name="password" type="password" required>'
            f'<button type="submit">Увійти</button></form>'
            f'<div class="links"><a href="/forgot">Забули пароль?</a> · '
            f'<a href="/register">Реєстрація</a></div>')


def _register_form(err=''):
    e = f'<div class="msg err">{err}</div>' if err else ''
    return (f'<h1>📝 Реєстрація</h1><p class="sub">Доступ — після підтвердження '
            f'email та схвалення адміністратора</p>{e}'
            f'<form method="post">'
            f'<label>Email</label><input name="email" type="email" required autofocus>'
            f'<label>Пароль (мін. 8)</label><input name="password" type="password" required>'
            f'<label>Повторіть пароль</label><input name="password2" type="password" required>'
            f'<button type="submit">Зареєструватися</button></form>'
            f'<div class="links"><a href="/login">← Вже маю акаунт</a></div>')


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


def _admin_users_body():
    return (
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">'
        '<div><h1 style="margin:0">🛡 Керування користувачами</h1>'
        '<p class="sub" style="margin:4px 0 0">Новий акаунт активний лише після '
        'підтвердження email ТА схвалення адміністратора</p></div>'
        '<div><a class="actbtn" href="/cabinet">👤 Кабінет</a> '
        '<a class="actbtn" href="/">📊 Дашборд</a> '
        '<a class="actbtn" href="/logout" style="color:#fca5a5">Вийти</a></div></div>'
        # stat cards
        '<div id="stats" style="display:flex;gap:10px;flex-wrap:wrap;margin:14px 0"></div>'
        '<div id="smtp" class="msg" style="display:none"></div>'
        # toolbar: search + create
        '<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:8px 0 4px">'
        '<input id="q" placeholder="🔎 пошук за email…" style="max-width:260px" oninput="render()">'
        '<select id="flt" onchange="render()" style="max-width:200px">'
        '<option value="all">Усі</option>'
        '<option value="pending">Очікують схвалення</option>'
        '<option value="active">Активні</option>'
        '<option value="admin">Адміни</option>'
        '<option value="disabled">Вимкнені</option></select>'
        '<button class="actbtn" style="margin-left:auto" onclick="toggleCreate()">➕ Створити користувача</button>'
        '</div>'
        # create form (hidden)
        '<div id="createbox" style="display:none;margin:6px 0;padding:12px;border:1px dashed #2a3140;border-radius:10px">'
        '<div style="display:flex;gap:8px;flex-wrap:wrap;align-items:end">'
        '<div><label>Email</label><input id="c_email" type="email" style="width:220px"></div>'
        '<div><label>Пароль (мін. 8)</label><input id="c_pw" type="text" style="width:180px"></div>'
        '<label style="display:flex;align-items:center;gap:6px;margin:0"><input id="c_adm" type="checkbox" style="width:auto"> адмін</label>'
        '<button class="actbtn" onclick="createUser()">Створити (одразу активний)</button>'
        '</div><div id="c_msg" style="font-size:.75rem;margin-top:6px;color:#fca5a5"></div></div>'
        # table
        '<div style="overflow-x:auto"><table id="tbl"><thead><tr>'
        '<th>#</th><th>Email</th><th>Роль</th><th>Email</th><th>Схвалення</th>'
        '<th>Активність</th><th>Створено</th><th>Останній вхід</th><th>Дії</th>'
        '</tr></thead><tbody></tbody></table></div>'
        '<p id="empty" class="sub" style="display:none;text-align:center;margin:16px 0">Нічого не знайдено</p>')


def _admin_script():
    return """
    let USERS=[], LINKS={}, ME=null;
    async function load(){
      const d=await (await fetch('/api/admin/users')).json();
      USERS=d.users||[]; LINKS=d.pending_links||{};
      const s=document.getElementById('smtp');
      if(!d.smtp){ s.style.display='block'; s.className='msg err';
        s.innerHTML='⚠ SMTP не налаштований — листи не надсилаються. Посилання підтвердження показані в таблиці (dev-режим). Задайте SMTP_HOST/PORT/USER/PASS/FROM у env для авто-розсилки.'; }
      else { s.style.display='none'; }
      stats(); render();
    }
    function stats(){
      const t=USERS.length, adm=USERS.filter(u=>u.is_admin).length,
            act=USERS.filter(u=>u.active).length,
            pend=USERS.filter(u=>!u.approved&&!u.is_admin).length,
            dis=USERS.filter(u=>u.disabled).length;
      const card=(n,l,c)=>`<div style="flex:1;min-width:120px;background:#0e1219;border:1px solid rgba(255,255,255,.07);border-radius:10px;padding:10px 14px"><div style="font-size:1.4rem;font-weight:800;color:${c}">${n}</div><div class="sub" style="margin:0">${l}</div></div>`;
      document.getElementById('stats').innerHTML=
        card(t,'усього','#e5e7eb')+card(pend,'очікують схвалення','#fbbf24')+
        card(act,'активні','#4ade80')+card(adm,'адміни','#fde68a')+card(dis,'вимкнені','#9aa3b5');
    }
    function render(){
      const q=(document.getElementById('q').value||'').toLowerCase();
      const flt=document.getElementById('flt').value;
      const tb=document.querySelector('#tbl tbody'); tb.innerHTML='';
      let list=USERS.filter(u=>u.email.toLowerCase().includes(q));
      if(flt==='pending') list=list.filter(u=>!u.approved&&!u.is_admin);
      else if(flt==='active') list=list.filter(u=>u.active);
      else if(flt==='admin') list=list.filter(u=>u.is_admin);
      else if(flt==='disabled') list=list.filter(u=>u.disabled);
      document.getElementById('empty').style.display=list.length?'none':'block';
      for(const u of list){
        const pl=LINKS[u.email]?LINKS[u.email].link:null;
        const role=u.is_admin?'<span class="badge b-adm">адмін</span>':'<span class="badge b-off">користувач</span>';
        const em=u.email_confirmed?'<span class="badge b-on">✓ підтв.</span>':'<span class="badge b-off">✗ ні</span>';
        const ap=u.approved?'<span class="badge b-on">✓ схвалено</span>':'<span class="badge b-off">очікує</span>';
        const ac=u.disabled?'<span class="badge b-off">вимкнено</span>':(u.active?'<span class="badge b-on">активний</span>':'<span class="badge b-off">неактивний</span>');
        let act='';
        act+= u.approved?btn(u.id,'revoke','⨯ Відкликати'):btn(u.id,'approve','✓ Схвалити');
        if(!u.email_confirmed){ act+=btn(u.id,'confirm','✉ Підтв. email'); act+=btn(u.id,'resend','↻ Лист'); }
        act+= u.disabled?btn(u.id,'enable','▶ Увімкнути'):btn(u.id,'disable','⏸ Вимкнути');
        act+= u.is_admin?btn(u.id,'remove_admin','− Адмін'):btn(u.id,'make_admin','+ Адмін');
        act+=`<button class="actbtn" onclick="setpw(${u.id})">🔑 Пароль</button>`;
        act+=`<button class="actbtn" style="color:#fca5a5" onclick="act(${u.id},'delete')">🗑</button>`;
        const tr=document.createElement('tr');
        tr.innerHTML=`<td style="color:#9aa3b5">${u.id}</td>`+
          `<td>${u.email}${pl?`<div style="font-size:.64rem;color:#60a5fa;word-break:break-all">🔗 <a href="${pl}" style="color:#60a5fa">${pl}</a></div>`:''}</td>`+
          `<td>${role}</td><td>${em}</td><td>${ap}</td><td>${ac}</td>`+
          `<td style="font-size:.68rem;color:#9aa3b5;white-space:nowrap">${fmt(u.created_at)}</td>`+
          `<td style="font-size:.68rem;color:#9aa3b5;white-space:nowrap">${fmt(u.last_login)||'—'}</td>`+
          `<td style="white-space:nowrap">${act}</td>`;
        tb.appendChild(tr);
      }
    }
    function fmt(s){return s?String(s).slice(0,16).replace('T',' '):'';}
    function btn(id,a,label){return `<button class="actbtn" onclick="act(${id},'${a}')">${label}</button>`}
    async function act(id,a,extra){
      if(a==='delete'&&!confirm('Видалити користувача назавжди?'))return;
      const body=Object.assign({action:a},extra||{});
      const r=await fetch('/api/admin/users/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      const d=await r.json(); if(!d.ok)alert(d.error||'Помилка'); load();
    }
    function setpw(id){ const p=prompt('Новий пароль для користувача (мін. 8 символів):'); if(!p)return; act(id,'set_password',{password:p}); }
    function toggleCreate(){ const b=document.getElementById('createbox'); b.style.display=b.style.display==='none'?'block':'none'; }
    async function createUser(){
      const email=document.getElementById('c_email').value, pw=document.getElementById('c_pw').value,
            adm=document.getElementById('c_adm').checked, m=document.getElementById('c_msg');
      const r=await fetch('/api/admin/users/create',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({email,password:pw,is_admin:adm})});
      const d=await r.json();
      if(!d.ok){ m.textContent=d.error||'Помилка'; return; }
      m.textContent=''; document.getElementById('c_email').value=''; document.getElementById('c_pw').value=''; document.getElementById('c_adm').checked=false;
      toggleCreate(); load();
    }
    load(); setInterval(load, 15000);"""


# ==================================================================
# One-call installer
# ==================================================================
def init_auth(app):
    """Wire auth into a Flask app: routes + gate + admin bootstrap."""
    register_auth_routes(app)
    install_auth_gate(app)
    try:
        ensure_admin()
    except Exception as e:
        print(f"[AUTH] ensure_admin error: {e}")
