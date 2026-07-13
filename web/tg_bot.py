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
def _handle_start(chat_id, username=None):
    from web.auth import get_user_by_chat, _make_tg_token
    b = base_url()
    existing = get_user_by_chat(str(chat_id))
    if existing:
        tg_send(chat_id,
                f"👋 Ви вже зареєстровані як <b>{existing['email']}</b>.\n"
                + ("✅ Акаунт активний — входьте на сайті."
                   if existing['active'] else
                   "⏳ Акаунт очікує схвалення адміністратора.")
                + "\n\n💬 Питання чи труднощі? Напишіть сюди — адміністратор відповість.",
                buttons=[[{'text': '🔐 Увійти', 'url': f"{b}/login"}]] if b else None)
        return
    if not b:
        tg_send(chat_id, "⚠️ Сервіс тимчасово недоступний (не задано публічну "
                         "адресу). Зверніться до адміністратора.")
        return
    tok = _make_tg_token(str(chat_id), username)
    link = f"{b}/register?tg={tok}"
    tg_send(chat_id,
            "👋 <b>Вітаю у VSV Bot!</b>\n\n"
            "Щоб отримати доступ:\n"
            "1️⃣ Натисніть «Реєстрація» і задайте email + пароль.\n"
            "2️⃣ Адміністратор схвалить ваш акаунт.\n"
            "3️⃣ Ви отримаєте сповіщення тут — і зможете увійти.\n\n"
            "💬 Виникли труднощі? Просто напишіть повідомлення сюди — "
            "я передам його адміністратору, і він відповість тут.",
            buttons=[[{'text': '📝 Реєстрація', 'url': link}]])


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
    admin = _admin_chat()
    is_admin_chat = bool(admin) and cid_s == str(admin)

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
            _handle_start(cid, uname)
            return
        return   # admin typed something that isn't a reply/command → ignore

    # ---- user side ----
    if text.startswith('/start'):
        _handle_start(cid, uname)
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
    who = info.get('email') or (('@' + uname) if uname else 'невідомий користувач')
    uref = f" (@{uname})" if (uname and info.get('email')) else ""
    header = (f"✉️ <b>Повідомлення від користувача</b>\n{who}{uref}\n"
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
