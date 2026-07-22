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
_PREF_TAG = {
    'notify_funding': _CAT_TAG['funding'],
    'notify_btc': _CAT_TAG['btc'],
    'notify_trades': _CAT_TAG['trades'],
    'notify_opportunity': '#Рекомендація',
    'notify_spike': '#Аномалія',
}

# ── Аудиторія категорії ─────────────────────────────────────────────────────
# PUBLIC  — ринкові сигнали, можна у спільну групу/тему (усі учасники бачать):
#           funding, btc, trades.
# ADMIN   — службово/персональне (реєстрації, звернення) → ЛИШЕ у приватний чат
#           адміна (TELEGRAM_CHAT_ID), НІКОЛИ у спільну групу.
_ADMIN_ONLY_CATS = {'register', 'support'}

# Working market topics whose bot posts are PROTECTED: no forwarding / copying /
# saving (protect_content). Members can still WRITE in the topic. NOTE: Telegram
# cannot block screenshots in a normal group (only secret chats) — protect_content
# is the strongest the Bot API offers and disables forward/copy/save.
_PROTECTED_CATS = {'funding', 'btc', 'trades'}


_forum_topics_cache = None    # {category: thread_id} for TELEGRAM_FORUM_CHAT


def _forum_thread(category):
    """Get/auto-create a forum TOPIC for `category` inside TELEGRAM_FORUM_CHAT —
    one supergroup, a topic per category (💰/₿/📈/📝/💬). Thread ids persist in
    DB so topics aren't recreated on restart. Returns (chat_id, thread_id) or
    (None, None) when no forum is configured / creation failed (falls back)."""
    chat = os.getenv('TELEGRAM_FORUM_CHAT')
    if not chat:
        return None, None
    # ADMIN-only categories (реєстрація/підтримка) must NEVER become a group
    # topic that members can see — they always go to the admin's private chat.
    if category in _ADMIN_ONLY_CATS:
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
    # Protect the working market topics: no forward / copy / save of these posts.
    if category in _PROTECTED_CATS:
        p['protect_content'] = True
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


def _group_chat():
    """The community group id (forum supergroup, or a dedicated group env)."""
    return os.getenv('TELEGRAM_FORUM_CHAT') or os.getenv('TELEGRAM_GROUP_CHAT')


def invite_user_to_group(chat_id):
    """Auto-onboard an approved user into the community group. The Telegram Bot
    API CANNOT add a member directly, so we create a PERSONAL one-time invite
    link and DM it — one tap and they're in. Also works if the group requires
    join approval (see _handle_join_request auto-approve). Best-effort; needs the
    bot to be a group admin with «invite users» rights. Returns True if sent."""
    group = _group_chat()
    if not group or not chat_id:
        return False
    try:
        link = None
        # creates_join_request=True → EVERY join via this link needs approval,
        # and the bot auto-approves ONLY registered+active users
        # (_handle_join_request). This enforces «group access via site
        # registration only» even if the link leaks — nobody gets in unapproved.
        r = _api('createChatInviteLink', {'chat_id': group,
                                          'name': f'reg {chat_id}',
                                          'creates_join_request': True})
        if r.get('ok'):
            link = (r.get('result') or {}).get('invite_link')
        if not link:   # fallback: the group's primary link
            r2 = _api('exportChatInviteLink', {'chat_id': group})
            link = r2.get('result') if r2.get('ok') else None
        if not link:
            print(f"[TG-BOT] group invite: no link (bot admin + invite rights?)")
            return False
        return bool(tg_send(
            chat_id,
            "🎉 <b>Вас схвалено!</b>\nПриєднайтесь до нашої групи спільноти:",
            buttons=[[{'text': '➡️ Приєднатися до групи', 'url': link}]]))
    except Exception as e:
        print(f"[TG-BOT] invite_user_to_group error: {e}")
        return False


def base_url():
    """Request-free public URL (poller has no Flask request context).

    Пріоритет: явні BASE_URL / PUBLIC_URL / BOT_PUBLIC_URL перебивають legacy
    RENDER_EXTERNAL_URL. Це важливо після переїзду з Render на власний VDS:
    якщо стара змінна RENDER_EXTERNAL_URL випадково лишилась у середовищі,
    вона НЕ повинна тягнути Telegram-кнопки на старий домен. Задай на боті
    BASE_URL=https://bot.vsv-trade.com.ua — і всі посилання будуть новими.
    """
    return (os.getenv('BASE_URL') or os.getenv('PUBLIC_URL')
            or os.getenv('BOT_PUBLIC_URL') or os.getenv('RENDER_EXTERNAL_URL')
            or '').rstrip('/')


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


def _handle_start(chat_id, username=None, name=None, lang=None, premium=None):
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
    tok = _make_tg_token(str(chat_id), username, name, lang, premium)
    link = f"{b}/register?tg={tok}"
    tg_send(chat_id,
            "👋 <b>Вітаю у VSV Bot!</b>\n\n"
            "Потрапити до групи можна <b>лише через реєстрацію на інфо-сайті</b> — "
            "інакше доступу до групи немає (нікого не додають вручну).\n\n"
            "Щоб отримати доступ:\n"
            "1️⃣ Натисніть «Реєстрація на сайті» і задайте <b>лише пароль</b> "
            "(email не потрібен — вас підтверджує Telegram).\n"
            "2️⃣ Адміністратор підтвердить вашу реєстрацію.\n"
            "3️⃣ Ринкові сигнали (угоди, ₿ BTCUSDT, 💰 Funding) приходитимуть "
            "в <b>окрему групу</b> — посилання на вхід у групу надійде сюди "
            "<b>автоматично, після підтвердження реєстрації адміністратором</b>.\n\n"
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


def _handle_join_request(jr):
    """Group access is REGISTRATION-GATED: auto-approve a join request ONLY from a
    user who registered on the info-site AND was approved by the admin (active).
    Everyone else is LEFT PENDING for the admin/owner to decide — nobody gets in
    without site registration; nobody is added by anyone but admin/owner."""
    chat = (jr.get('chat', {}) or {}).get('id')
    uid_chat = str(((jr.get('from', {}) or {}).get('id')) or '')
    if not chat or not uid_chat:
        return
    try:
        from web.auth import get_user_by_chat
        info = get_user_by_chat(uid_chat) or {}
        active = bool(info.get('active'))
    except Exception:
        active = False
    if active:
        try:
            _api('approveChatJoinRequest', {'chat_id': chat,
                                            'user_id': int(uid_chat)})
            print(f"[TG-BOT] auto-approved join request from {uid_chat} (registered)")
        except Exception as e:
            print(f"[TG-BOT] approve join error: {e}")
    else:
        # Not a registered+approved user → leave the request PENDING for the
        # admin/owner (do NOT auto-approve; do NOT auto-decline).
        print(f"[TG-BOT] join request from {uid_chat} left pending (not registered)")


# Any non-text content Telegram may carry — so the bot forwards photos, videos,
# documents, voice, stickers… exactly like a normal chat, not just text.
_MEDIA_KEYS = ('photo', 'video', 'document', 'animation', 'voice', 'audio',
               'video_note', 'sticker')


def _has_media(m):
    return any(k in m for k in _MEDIA_KEYS)


def _copy_message(to_chat, from_chat, message_id):
    """Copy ANY message (media incl. its caption) to another chat. Returns the
    new message_id or None. `copyMessage` re-sends content with no «forwarded
    from» header — clean for support relaying both ways."""
    try:
        r = _api('copyMessage', {'chat_id': to_chat, 'from_chat_id': from_chat,
                                 'message_id': message_id})
        return (r.get('result') or {}).get('message_id') if r.get('ok') else None
    except Exception:
        return None


def _admin_broadcast(m, admin_cid):
    """📢 Admin ANNOUNCEMENT — copy the admin's message (text OR media, keeping
    caption) to EVERY active bot subscriber. Reports the delivered count back."""
    try:
        from web.auth import all_bot_chats
        chats = all_bot_chats(exclude_chat=admin_cid)
    except Exception:
        chats = []
    n = 0
    for c in chats:
        try:
            if _copy_message(c, admin_cid, m.get('message_id')):
                n += 1
        except Exception:
            pass
    tg_send(admin_cid, f"📢 Розіслано підписникам: <b>{n}</b>.")


def _handle_message(m):
    """Route a message: user → admin (support), admin reply → user. Forwards ANY
    content type (text / photo / video / document / voice / sticker …)."""
    text = (m.get('text') or '').strip()
    cid = (m.get('chat', {}) or {}).get('id')
    if not cid:
        return
    cid_s = str(cid)
    frm = (m.get('from', {}) or {})
    uname = frm.get('username')
    fname = _full_name(frm)
    flang = frm.get('language_code')      # richer Telegram profile (saved at reg)
    fprem = frm.get('is_premium')
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

    # ── Only the PRIVATE bot chat is handled. The GROUP is a normal group: the
    # bot does NOT forward group messages to the admin and posts NO confirmation
    # there — members chat and react as usual. Support/forwarding happens ONLY
    # when a user writes to the bot in their PRIVATE chat. ──
    ctype = (m.get('chat', {}) or {}).get('type', '')
    if ctype and ctype != 'private':
        return

    # Remember EVERY private chat as a bot contact — so the admin sees ALL bot
    # users (even those who pressed /start but never registered) and can reach them.
    try:
        from web.auth import record_bot_contact
        record_bot_contact(cid_s, uname, fname)
    except Exception:
        pass

    if is_admin_chat:
        # 👥 /subs — list the bot's subscribers (admin only).
        if text.startswith('/subs'):
            try:
                from web.auth import bot_contacts
                c = bot_contacts()
                top = sorted(c.items(), key=lambda kv: (kv[1] or {}).get('last_seen', 0),
                             reverse=True)[:50]
                lines = []
                for cid_k, r in top:
                    r = r or {}
                    nm = r.get('name') or ''
                    un = f"@{r['username']}" if r.get('username') else ''
                    lines.append(f"• {(' '.join(x for x in [nm, un] if x)) or '—'} "
                                 f"<code>{cid_k}</code>")
                tg_send(cid, f"👥 Підписників бота: <b>{len(c)}</b>"
                             + (("\n" + "\n".join(lines)) if lines else "")
                             + ("\n…" if len(c) > 50 else ""))
            except Exception as e:
                tg_send(cid, f"⚠️ Помилка: {e}")
            return
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
            if _has_media(m):
                # Admin replied with media → label + copy it to the user.
                tg_send(target, "💬 <b>Відповідь адміністратора:</b>")
                ok = bool(_copy_message(target, cid, m.get('message_id')))
            else:
                ok = tg_send(target, f"💬 <b>Відповідь адміністратора:</b>\n{text}")
            tg_send(cid, "✅ Надіслано користувачу." if ok
                    else "⚠️ Не вдалося надіслати (можливо, користувач не почав чат).")
            return
        if text.startswith('/start'):
            _handle_start(cid, uname, fname, flang, fprem)
            return
        # A plain admin message (not a reply, not a command) = an ANNOUNCEMENT —
        # broadcast it (text OR media) to EVERY bot subscriber, like a channel.
        if text.startswith('/'):
            return   # unknown command → ignore
        if not text and not _has_media(m):
            return
        _admin_broadcast(m, cid_s)
        return

    # ---- user side ----
    if text.startswith('/start'):
        _handle_start(cid, uname, fname, flang, fprem)
        return
    has_media = _has_media(m)
    if not text and not has_media:
        return   # nothing forwardable (e.g. a service/system message)
    if not admin:
        tg_send(cid, "⚠️ Підтримка тимчасово недоступна.")
        return
    # A user message (text OR media) = a support message → forward to the admin.
    try:
        from web.auth import get_user_by_chat
        info = get_user_by_chat(cid_s) or {}
    except Exception:
        info = {}
    # Friendly identity line: name/@handle (always) + email (if registered).
    who = _who_label(frm, info)
    email_line = f"\n✉️ {info['email']}" if info.get('email') else ""
    # 🌐 Registration status on the info-site — so the admin instantly knows who
    # is writing: a registered (active / pending) account, or a stranger.
    if not info:
        reg_line = "\n🌐 Реєстрація: <b>❌ НЕ зареєстрований на сайті</b>"
    elif info.get('active'):
        reg_line = "\n🌐 Реєстрація: <b>✅ зареєстрований · активний</b>"
    elif info.get('approved'):
        reg_line = "\n🌐 Реєстрація: <b>⛔ вимкнено / протерміновано</b>"
    else:
        reg_line = "\n🌐 Реєстрація: <b>⏳ очікує схвалення</b>"
    # Header carries identity; the plain-text body is inlined only for text
    # messages. Media is copied AFTER (it keeps its own caption).
    header = (f"{_CAT_TAG.get('support', '')}\n"
              f"✉️ <b>Повідомлення від користувача</b>\n👤 {who}{email_line}{reg_line}\n"
              f"chat_id: <code>{cid_s}</code>"
              + (f"\n\n{text}" if (text and not has_media) else ""))
    mid = _send_get_id(admin, header)
    if mid:
        _remember(mid, cid_s)   # swipe-Reply on the header routes back to the user
    if has_media:
        # Copy the actual photo/video/document/voice/… (with its caption) so the
        # admin sees everything, exactly like a normal chat. Swipe-Reply on the
        # media works too.
        mid2 = _copy_message(admin, cid, m.get('message_id'))
        if mid2:
            _remember(mid2, cid_s)
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
            payload = {'timeout': 30,
                       'allowed_updates': ['message', 'callback_query',
                                           'chat_join_request']}
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
                    elif 'chat_join_request' in upd:
                        _handle_join_request(upd['chat_join_request'])
                    elif 'message' in upd:
                        _handle_message(upd['message'])
                except Exception as e:
                    print(f"[TG-BOT] handler error: {e}")
        except Exception as e:
            print(f"[TG-BOT] poll error: {e}")
            time.sleep(5)


def _purge_admin_only_topics():
    """One-shot on startup: delete any STALE admin-only forum topics
    (📝 Реєстрація / 💬 Підтримка) left in the group by older builds, so members
    never see them. Registration/support always go to the admin's private chat."""
    chat = os.getenv('TELEGRAM_FORUM_CHAT')
    if not chat:
        return
    try:
        from storage.db_operations import get_db
        db = get_db()
        saved = db.get_setting('tg_forum_topics', {}) or {}
        if not isinstance(saved, dict):
            return
        cmap = saved.get(str(chat), {}) or {}
        changed = False
        for cat in list(_ADMIN_ONLY_CATS):
            tid = cmap.get(cat)
            if tid:
                try:
                    _api('deleteForumTopic', {'chat_id': chat,
                                              'message_thread_id': int(tid)})
                    print(f"[TG-BOT] removed stale group topic '{cat}' (admin-only)")
                except Exception:
                    pass
                cmap.pop(cat, None)
                changed = True
        if changed:
            saved[str(chat)] = cmap
            db.set_setting('tg_forum_topics', saved)
            global _forum_topics_cache
            _forum_topics_cache = None   # force reload without the purged ids
    except Exception as e:
        print(f"[TG-BOT] purge admin-only topics error: {e}")


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
    try:
        _purge_admin_only_topics()   # tidy stale 📝/💬 topics from the group
    except Exception:
        pass
    threading.Thread(target=_poll_loop, daemon=True, name='tg-bot-poll').start()
