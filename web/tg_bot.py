"""
Telegram bot receiver — registration + approval + user notifications, all via
Telegram (no email needed). Long-polling daemon (getUpdates) so it works on any
host (Render / VPS) without a public webhook.

Flow (login stays password-based on the web; Telegram handles onboarding):
  • User opens t.me/<bot> → /start → bot sends a registration link
    (<site>/register?tg=<signed chat_id>) — the link ties the account to their
    Telegram chat. Telegram having accepted /start = the user is reachable.
  • User sets email+password on the web form → account is created linked to that
    chat, email_confirmed=True (Telegram-verified), approved=False.
  • Admin gets a Telegram message with inline «✓ Схвалити / ✗ Відхилити».
  • On approve → the USER is notified in Telegram that access is granted.

Uses stdlib urllib only. Single poller (safe with gunicorn -w 1).
"""
import os
import json
import time
import threading
import urllib.request
import urllib.error

_started = False
_lock = threading.Lock()
# Maps a bot message shown in the ADMIN chat → the user chat it came from, so
# the admin can just «Reply» to it and the bot routes the answer back. In-memory
# (bounded); on restart, admins can still use «/reply <chat_id> <text>».
_reply_map = {}
_map_lock = threading.Lock()


def _full_name(frm):
    """Human name from a Telegram `from` object (first + last). Always present
    even when the user has no @username set."""
    parts = [(frm or {}).get('first_name'), (frm or {}).get('last_name')]
    return ' '.join(p for p in parts if p).strip() or None


def _who_label(frm, info=None):
    """Best display label: «Ім'я (@handle)» / «@handle» / «Ім'я» / fallback.
    `info` = optional dict from auth.get_user_by_chat (email/tg_name/tg_user)."""
    info = info or {}
    name = _full_name(frm) or info.get('tg_name')
    uname = (frm or {}).get('username') or info.get('tg_user')
    handle = f'@{uname}' if uname else ''
    label = ' '.join(x for x in [name, handle] if x).strip()
    return label or 'невідомий користувач'


def _remember(admin_msg_id, user_chat):
    with _map_lock:
        _reply_map[int(admin_msg_id)] = str(user_chat)
        if len(_reply_map) > 1000:
            for k in list(_reply_map)[:200]:
                _reply_map.pop(k, None)


def _token():
    return os.getenv('TELEGRAM_BOT_TOKEN')


def _admin_chat():
    return os.getenv('TELEGRAM_CHAT_ID')


# ── Категорії повідомлень ──────────────────────────────────────────────────
# Щоб не все валилось в одну групу: кожна категорія може мати власний чат/групу
# (env TELEGRAM_CHAT_<CAT>) і/або тему форуму (env TELEGRAM_TOPIC_<CAT>). Якщо
# власного чату немає — шле в головний чат адміна з категорійним заголовком.
_CAT_LABEL = {
    'funding':  '💰 Funding',
    'btc':      '₿ BTCUSDT',
    'trades':   '📈 Угоди',
    'register': '📝 Реєстрація',
    'support':  '💬 Підтримка',
}
_CAT_ENV = {
    'funding':  ('TELEGRAM_CHAT_FUNDING',  'TELEGRAM_TOPIC_FUNDING'),
    'btc':      ('TELEGRAM_CHAT_BTC',      'TELEGRAM_TOPIC_BTC'),
    'trades':   ('TELEGRAM_CHAT_TRADES',   'TELEGRAM_TOPIC_TRADES'),
    'register': ('TELEGRAM_CHAT_REGISTER', 'TELEGRAM_TOPIC_REGISTER'),
    'support':  ('TELEGRAM_CHAT_SUPPORT',  'TELEGRAM_TOPIC_SUPPORT'),
}
# Category hashtag — prepended to EVERY message so that even in ONE chat (the
# private bot chat, where Telegram has no topics) you can tap the tag to filter
# by category. Distinct from the per-symbol tags (#BTCUSDT) so they don't clash.
_CAT_TAG = {
    'funding':  '#Funding',
    'btc':      '#BTC_сеанс',
    'trades':   '#Угода',
    'register': '#Реєстрація',
    'support':  '#Підтримка',
}
_PREF_TAG = {'notify_funding': _CAT_TAG['funding'], 'notify_btc': _CAT_TAG['btc']}

# ── Аудиторія категорії ─────────────────────────────────────────────────────
# PUBLIC  — ринкові сигнали, можна у спільну групу/тему (усі учасники бачать):
#           funding, btc, trades.
# ADMIN   — службово/персональне (реєстрації, звернення) → ЛИШЕ у приватний чат
#           адміна (TELEGRAM_CHAT_ID), НІКОЛИ у спільну групу.
_ADMIN_ONLY_CATS = {'register', 'support'}


_forum_topics_cache = None    # {category: thread_id} for TELEGRAM_FORUM_CHAT


def _forum_thread(category):
    """Get/auto-create a forum TOPIC for `category` inside TELEGRAM_FORUM_CHAT —
    one supergroup, a topic per category (💰/₿/📈/📝/💬). Thread ids persist in
    DB so topics aren't recreated on restart. Returns (chat_id, thread_id) or
    (None, None) when no forum is configured / creation failed (falls back)."""
    chat = os.getenv('TELEGRAM_FORUM_CHAT')
    if not chat:
        return None, None
    global _forum_topics_cache
    if _forum_topics_cache is None:
        try:
            from storage.db_operations import get_db
            saved = get_db().get_setting('tg_forum_topics', {}) or {}
            _forum_topics_cache = saved.get(str(chat), {}) if isinstance(saved, dict) else {}
        except Exception:
            _forum_topics_cache = {}
    tid = _forum_topics_cache.get(category)
    if tid:
        return chat, tid
    res = _api('createForumTopic', {'chat_id': chat, 'name': _CAT_LABEL.get(category, category)})
    tid = (res.get('result') or {}).get('message_thread_id') if res.get('ok') else None
    if tid:
        _forum_topics_cache[category] = tid
        try:
            from storage.db_operations import get_db
            db = get_db()
            saved = db.get_setting('tg_forum_topics', {}) or {}
            if not isinstance(saved, dict):
                saved = {}
            saved.setdefault(str(chat), {})[category] = tid
            db.set_setting('tg_forum_topics', saved)
        except Exception:
            pass
    return chat, tid


def _cat_chat(category):
    """(chat_id, thread_id) for a category. Priority: forum-topic supergroup →
    per-category env chat/topic → main admin chat.
    ADMIN-only categories (реєстрації/підтримка) ALWAYS go to the admin's
    PRIVATE chat — never a shared group/topic — so other users can't see them."""
    if category in _ADMIN_ONLY_CATS:
        return _admin_chat(), None
    fchat, fthread = _forum_thread(category)
    if fchat and fthread:
        return fchat, str(fthread)
    cenv, tenv = _CAT_ENV.get(category, (None, None))
    chat = (os.getenv(cenv) if cenv else None) or _admin_chat()
    thread = os.getenv(tenv) if tenv else None
    return chat, thread


def _cat_enabled(category):
    """Whether a GROUP category is on. 💰 Funding / ₿ BTCUSDT follow the admin's
    OWN cabinet toggle (notify_funding/notify_btc) — turning it off there stops
    the group too. Other categories default ON."""
    key = {'funding': 'notify_funding', 'btc': 'notify_btc'}.get(category)
    if not key:
        return True
    try:
        from web.auth import admin_pref
        return admin_pref(key, True)
    except Exception:
        return True


def notify_category(category, text, buttons=None):
    """Send an ADMIN-facing message routed by category. When the category has no
    dedicated chat/topic, prefixes a category header so one chat stays sorted.
    Skips entirely when the admin has switched this category off."""
    if not _cat_enabled(category):
        return False
    chat, thread = _cat_chat(category)
    if not chat:
        return False
    tag = _CAT_TAG.get(category, '')
    body = f"{tag}\n{text}" if tag else text
    p = {'chat_id': chat, 'text': body, 'parse_mode': 'HTML'}
    if thread:
        try:
            p['message_thread_id'] = int(thread)
        except (TypeError, ValueError):
            pass
    if buttons:
        p['reply_markup'] = {'inline_keyboard': buttons}
    return bool(_api('sendMessage', p).get('ok'))


def cat_tag(category):
    """Category hashtag for a category key (for callers that send directly)."""
    return _CAT_TAG.get(category, '')


def broadcast_to_subscribers(pref_key, text):
    """Send `text` to every ACTIVE user who opted into `pref_key` (in their
    personal Telegram chat). Prepends the category hashtag so each user can
    filter by category in their own bot chat. Best-effort; returns count sent."""
    try:
        from web.auth import subscriber_chats
        chats = subscriber_chats(pref_key)
    except Exception:
        chats = []
    tag = _PREF_TAG.get(pref_key, '')
    body = f"{tag}\n{text}" if tag else text
    n = 0
    for cid in chats:
        try:
            if tg_send(cid, body):
                n += 1
        except Exception:
            pass
    return n


def base_url():
    """Request-free public URL (poller has no Flask request context)."""
    return (os.getenv('RENDER_EXTERNAL_URL') or os.getenv('PUBLIC_URL')
            or os.getenv('BASE_URL') or '').rstrip('/')


def _api(method, payload=None, timeout=35):
    tok = _token()
    if not tok:
        return {'ok': False, 'error': 'no token'}
    url = f"https://api.telegram.org/bot{tok}/{method}"
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=data, method='POST',
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode())
        except Exception:
            return {'ok': False, 'error': f'HTTP {e.code}'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def tg_send(chat_id, text, buttons=None):
    """Send an HTML message; `buttons` = list of rows of {text, url|callback_data}."""
    p = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML',
         'disable_web_page_preview': False}
    if buttons:
        p['reply_markup'] = {'inline_keyboard': buttons}
    return bool(_api('sendMessage', p).get('ok'))


def _send_get_id(chat_id, text, buttons=None):
    """Send and return the new message_id (or None)."""
    p = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    if buttons:
        p['reply_markup'] = {'inline_keyboard': buttons}
    return (_api('sendMessage', p).get('result') or {}).get('message_id')


def _answer_cb(cb_id, text=''):
    _api('answerCallbackQuery', {'callback_query_id': cb_id, 'text': text})


def _edit(chat_id, msg_id, text):
    _api('editMessageText', {'chat_id': chat_id, 'message_id': msg_id,
                             'text': text, 'parse_mode': 'HTML'})


# ---- handlers -------------------------------------------------------------
def _site_url():
    """Where the info-site lives (for one-tap login links)."""
    return (os.getenv('INFO_SITE_URL') or base_url() or '').rstrip('/')


def _handle_start(chat_id, username=None, name=None):
    from web.auth import get_user_by_chat, _make_tg_token
    b = base_url()
    existing = get_user_by_chat(str(chat_id))
    if existing:
        btns = None
        if existing['active']:
            # One-tap login: hand a signed token straight to the info-site.
            try:
                from web.auth import _make_info_token
                site = _site_url()
                if site:
                    tok = _make_info_token(existing['id'])
                    btns = [[{'text': '🔓 Увійти на сайт', 'url': f"{site}#it={tok}"}]]
            except Exception:
                btns = None
        tg_send(chat_id,
                f"👋 Ви зареєстровані як <b>{existing['email']}</b>.\n"
                + ("✅ Акаунт активний — тисніть «Увійти на сайт»."
                   if existing['active'] else
                   "⏳ Акаунт очікує схвалення адміністратора.")
                + "\n\n💬 Питання чи труднощі? Напишіть сюди — адміністратор відповість.",
                buttons=btns)
        return
    if not b:
        tg_send(chat_id, "⚠️ Сервіс тимчасово недоступний (не задано публічну "
                         "адресу). Зверніться до адміністратора.")
        return
    tok = _make_tg_token(str(chat_id), username, name)
    link = f"{b}/register?tg={tok}"
    tg_send(chat_id,
            "👋 <b>Вітаю у VSV Bot!</b>\n\n"
            "Щоб отримати доступ:\n"
            "1️⃣ Натисніть «Реєстрація на сайті» і задайте <b>лише пароль</b> "
            "(email не потрібен — вас підтверджує Telegram).\n"
            "2️⃣ Адміністратор схвалить ваш акаунт.\n"
            "3️⃣ Ви отримаєте сповіщення тут — і зможете увійти.\n\n"
            "🔔 Після входу в кабінеті/на інфо-сайті можна ввімкнути сповіщення "
            "₿ BTCUSDT і 💰 Funding — вони приходитимуть сюди.\n\n"
            "💬 Виникли труднощі? Просто напишіть повідомлення сюди — "
            "я передам його адміністратору, і він відповість тут.",
            buttons=[[{'text': '📝 Реєстрація на сайті', 'url': link}]])


def _handle_callback(cb):
    from web.auth import (get_user_by_id, _update_user, approve_user)
    data = cb.get('data', '') or ''
    cb_id = cb.get('id')
    msg = cb.get('message', {}) or {}
    chat_id = str((msg.get('chat', {}) or {}).get('id', ''))
    msg_id = msg.get('message_id')
    from_id = str((cb.get('from', {}) or {}).get('id', ''))
    # Only the configured admin chat may approve/reject.
    if _admin_chat() and from_id != str(_admin_chat()) and chat_id != str(_admin_chat()):
        _answer_cb(cb_id, 'Лише адміністратор')
        return
    if ':' not in data:
        _answer_cb(cb_id)
        return
    act, uid = data.split(':', 1)
    u = get_user_by_id(uid)
    if not u:
        _answer_cb(cb_id, 'Користувача нема')
        return
    if act == 'ap':
        approve_user(u.id)            # approve + 30-day access + notify user
        _answer_cb(cb_id, 'Схвалено ✓')
        _edit(chat_id, msg_id, f"✅ <b>{u.email}</b> — схвалено (30 днів).")
    elif act == 'rj':
        _update_user(u.id, approved=False, disabled=True)
        _answer_cb(cb_id, 'Відхилено ✗')
        _edit(chat_id, msg_id, f"⛔ <b>{u.email}</b> — відхилено.")
    else:
        _answer_cb(cb_id)


def _handle_message(m):
    """Route a plain message: user → admin (support), admin reply → user."""
    text = (m.get('text') or '').strip()
    cid = (m.get('chat', {}) or {}).get('id')
    if not cid:
        return
    cid_s = str(cid)
    frm = (m.get('from', {}) or {})
    uname = frm.get('username')
    fname = _full_name(frm)
    admin = _admin_chat()
    is_admin_chat = bool(admin) and cid_s == str(admin)

    # /id — reply with this chat's id (+ topic thread id). Works in groups too
    # (commands reach the bot even with privacy mode on). Used to configure
    # TELEGRAM_FORUM_CHAT for the categorised topics.
    if text.startswith('/id'):
        thr = m.get('message_thread_id')
        ctype = (m.get('chat', {}) or {}).get('type', '')
        extra = f"\nthread_id: <code>{thr}</code>" if thr else ""
        tg_send(cid, f"🆔 chat_id: <code>{cid_s}</code>\nтип: {ctype}{extra}\n\n"
                     f"Для тем-груп додай на боті env "
                     f"<code>TELEGRAM_FORUM_CHAT={cid_s}</code>")
        return

    if is_admin_chat:
        # 1) Admin swipe-replies to a forwarded user message.
        target = None
        rt = m.get('reply_to_message') or {}
        if rt.get('message_id') is not None:
            with _map_lock:
                target = _reply_map.get(int(rt['message_id']))
        # 2) Or explicit «/reply <chat_id> <text>».
        if text.startswith('/reply'):
            parts = text.split(None, 2)
            if len(parts) >= 3:
                target, text = parts[1], parts[2]
            else:
                tg_send(cid, "Формат: <code>/reply &lt;chat_id&gt; текст</code>")
                return
        if target:
            ok = tg_send(target, f"💬 <b>Відповідь адміністратора:</b>\n{text}")
            tg_send(cid, "✅ Надіслано користувачу." if ok
                    else "⚠️ Не вдалося надіслати (можливо, користувач не почав чат).")
            return
        if text.startswith('/start'):
            _handle_start(cid, uname, fname)
            return
        return   # admin typed something that isn't a reply/command → ignore

    # ---- user side ----
    if text.startswith('/start'):
        _handle_start(cid, uname, fname)
        return
    if not text:
        return
    if not admin:
        tg_send(cid, "⚠️ Підтримка тимчасово недоступна.")
        return
    # Any other text from a user = a support message → forward to the admin.
    try:
        from web.auth import get_user_by_chat
        info = get_user_by_chat(cid_s) or {}
    except Exception:
        info = {}
    # Friendly identity line: name/@handle (always) + email (if registered).
    who = _who_label(frm, info)
    email_line = f"\n✉️ {info['email']}" if info.get('email') else ""
    header = (f"{_CAT_TAG.get('support', '')}\n"
              f"✉️ <b>Повідомлення від користувача</b>\n👤 {who}{email_line}\n"
              f"chat_id: <code>{cid_s}</code>\n\n{text}\n\n"
              f"<i>Відповісти: свайп-Reply на це повідомлення або "
              f"/reply {cid_s} текст</i>")
    mid = _send_get_id(admin, header)
    if mid:
        _remember(mid, cid_s)
    tg_send(cid, "✅ Ваше повідомлення надіслано адміністратору. "
                 "Відповідь прийде сюди.")


def _poll_loop():
    # Ensure no webhook is set (getUpdates and webhook are mutually exclusive).
    try:
        _api('deleteWebhook', {'drop_pending_updates': False}, timeout=15)
    except Exception:
        pass
    offset = None
    print("[TG-BOT] long-polling started")
    while True:
        try:
            payload = {'timeout': 30, 'allowed_updates': ['message', 'callback_query']}
            if offset is not None:
                payload['offset'] = offset
            res = _api('getUpdates', payload, timeout=40)
            if not res.get('ok'):
                time.sleep(5)
                continue
            for upd in res.get('result', []):
                offset = upd['update_id'] + 1
                try:
                    if 'callback_query' in upd:
                        _handle_callback(upd['callback_query'])
                    elif 'message' in upd:
                        _handle_message(upd['message'])
                except Exception as e:
                    print(f"[TG-BOT] handler error: {e}")
        except Exception as e:
            print(f"[TG-BOT] poll error: {e}")
            time.sleep(5)


def start_tg_bot():
    """Launch the poller once, only if a bot token is configured."""
    global _started
    with _lock:
        if _started:
            return
        if not _token():
            print("[TG-BOT] TELEGRAM_BOT_TOKEN not set — Telegram onboarding off.")
            return
        _started = True
    threading.Thread(target=_poll_loop, daemon=True, name='tg-bot-poll').start()
