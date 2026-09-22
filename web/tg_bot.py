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
import re
import json
import time
import threading
import urllib.request
import urllib.error

_started = False
_lock = threading.Lock()
# Maps a bot message shown in the ADMIN chat → the user chat it came from, so
# the admin can just «Reply» to it and the bot routes the answer back. Bounded
# in-memory cache MIRRORED to DB (tg_reply_map) so it survives restarts; even if
# the link is missing, the reply falls back to parsing the chat_id from the
# forwarded header — and NEVER to a broadcast. Mass sends need explicit /announce.
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
        _persist_reply_map()


def _persist_reply_map():
    """Persist _reply_map to DB so a swipe-Reply on a forwarded support message
    still routes to the RIGHT user AFTER a restart/redeploy — instead of the link
    being lost and the reply falling through to an accidental broadcast. Called
    under _map_lock. Best-effort."""
    try:
        from storage.db_operations import get_db
        get_db().set_setting('tg_reply_map',
                             {str(k): v for k, v in _reply_map.items()})
    except Exception:
        pass


def _load_reply_map():
    """Restore _reply_map from DB on boot (survives redeploys)."""
    try:
        from storage.db_operations import get_db
        saved = get_db().get_setting('tg_reply_map', {}) or {}
    except Exception:
        saved = {}
    if not isinstance(saved, dict):
        return
    with _map_lock:
        for k, v in saved.items():
            try:
                _reply_map[int(k)] = str(v)
            except (TypeError, ValueError):
                pass


def _chat_id_from_text(text):
    """Extract the user's chat_id from a forwarded support header
    («chat_id: <code>7659029832</code>»). Robust fallback for a swipe-Reply when
    the reply-map link is gone — so we NEVER misroute a reply into a broadcast.
    Returns a str chat_id or None."""
    if not text:
        return None
    mt = re.search(r'chat_id\D{0,16}(\d{5,})', str(text))
    return mt.group(1) if mt else None


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
    # 🧮 ТЕМА ПЕРЕЙМЕНОВАНА (вимога 21.09): головне в ній тепер — зміни банера
    # «🧮 МММ-монітор» і 🔻 корекція, а ₿-сеанс лишається тут же (він і був).
    # ⚠️ КЛЮЧ категорії лишається `btc` — його знають `TELEGRAM_CHAT_BTC` /
    # `TELEGRAM_TOPIC_BTC`, збережені id тем (`tg_forum_topics`) і налаштування
    # адміна `notify_btc`. Перейменувати ключ означало б загубити і тему, і
    # тумблер (той самий прецедент, що з `q2_auto_ob_sl*`).
    'btc':      '🧮 МММ-монітор',
    'trades':   '📈 Угоди',
    'signal':   '🎯 Напрямок',
    'register': '📝 Реєстрація',
    'support':  '💬 Підтримка',
}
_CAT_ENV = {
    'funding':  ('TELEGRAM_CHAT_FUNDING',  'TELEGRAM_TOPIC_FUNDING'),
    'btc':      ('TELEGRAM_CHAT_BTC',      'TELEGRAM_TOPIC_BTC'),
    'trades':   ('TELEGRAM_CHAT_TRADES',   'TELEGRAM_TOPIC_TRADES'),
    'signal':   ('TELEGRAM_CHAT_SIGNAL',   'TELEGRAM_TOPIC_SIGNAL'),
    'register': ('TELEGRAM_CHAT_REGISTER', 'TELEGRAM_TOPIC_REGISTER'),
    'support':  ('TELEGRAM_CHAT_SUPPORT',  'TELEGRAM_TOPIC_SUPPORT'),
}
# Category hashtag — prepended to EVERY message so that even in ONE chat (the
# private bot chat, where Telegram has no topics) you can tap the tag to filter
# by category. Distinct from the per-symbol tags (#BTCUSDT) so they don't clash.
_CAT_TAG = {
    'funding':  '#Funding',
    'btc':      '#МММ_монітор',
    'trades':   '#Угода',
    'signal':   '#Напрямок',
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


_forum_rename_err = ''        # причина, чому тему не перейменували
_last_send_err = ''           # причина останньої невдалої відправки
_forum_names_cache = None     # {(chat, category): остання назва, яку ми ставили}


def _forum_rename_if_needed(chat, category, tid):
    """Перейменувати ВЖЕ СТВОРЕНУ тему форуму, якщо підпис категорії змінився.

    Тема створюється ОДИН раз, а її `message_thread_id` персиститься — тож
    зміна `_CAT_LABEL` сама по собі НЕ міняє назву в групі: у Telegram і далі
    висіло б «₿ BTCUSDT» над повідомленнями про 🧮 МММ-монітор. Тобто без цього
    перейменування вимога «зміни в групі назву» виконана б НЕ була.

    ⚠️ Назву, яку ми поставили, ЗАПАМʼЯТОВУЄМО в БД: інакше `editForumTopic`
    смикався б на КОЖНЕ повідомлення (зайвий запит до API на рівному місці).
    ⚠️ Невдача НЕ ламає відправку — тема лишається зі старою назвою, а
    повідомлення йде як ішло (best-effort, як і створення теми). ПРИЧИНУ
    відмови памʼятаємо в `_forum_rename_err`, щоб «назва не змінилась» можна
    було ПОБАЧИТИ в перевірці теми, а не лише в stdout.
    ⚠️ Кеш ключований ПАРОЮ (чат, категорія): теми різних категорій можуть
    жити в РІЗНИХ чатах (форум-група проти `TELEGRAM_CHAT_<CAT>`), і спільний
    ключ писав би назву однієї теми в запис іншої.
    """
    global _forum_names_cache, _forum_rename_err
    want = _CAT_LABEL.get(category)
    if not want:
        return
    if _forum_names_cache is None:
        try:
            from storage.db_operations import get_db
            saved = get_db().get_setting('tg_forum_topic_names', {}) or {}
            _forum_names_cache = {}
            if isinstance(saved, dict):
                for _c, _m in saved.items():
                    if isinstance(_m, dict):
                        for _cat, _nm in _m.items():
                            _forum_names_cache[(str(_c), _cat)] = _nm
        except Exception:
            _forum_names_cache = {}
    ckey = (str(chat), category)
    if _forum_names_cache.get(ckey) == want:
        return
    try:
        res = _api('editForumTopic', {'chat_id': chat, 'message_thread_id': int(tid),
                                      'name': want})
    except Exception as e:
        _forum_rename_err = f'{category}: {e}'
        print(f"[TG] rename topic {category} error: {e}")
        return
    if not (res or {}).get('ok'):
        _forum_rename_err = (f"{category}: "
                             f"{(res or {}).get('description') or (res or {}).get('error') or 'відмова'}")
        print(f"[TG] rename topic {category} refused: {_forum_rename_err}")
        return
    _forum_rename_err = ''
    _forum_names_cache[ckey] = want
    try:
        from storage.db_operations import get_db
        db = get_db()
        saved = db.get_setting('tg_forum_topic_names', {}) or {}
        if not isinstance(saved, dict):
            saved = {}
        saved.setdefault(str(chat), {})[category] = want
        db.set_setting('tg_forum_topic_names', saved)
    except Exception:
        pass
    print(f"[TG] forum topic renamed: {category} → {want}")


def sync_topic_names():
    """🏷 Привести назви ВЖЕ СТВОРЕНИХ тем до `_CAT_LABEL` — НЕЗАЛЕЖНО від того,
    чи шлемо ми зараз повідомлення і чи увімкнений тумблер категорії.

    **Скарга 22.09 дослівно:** «Тема в телеграм групі знову називається чомусь
    "₿ BTCUSDT" і повідомлень про "✅ КОРЕКЦІЯ ЗАВЕРШИЛАСЬ" не було.»

    ⚠️ **ОБИДВА СИМПТОМИ МАЛИ ОДИН КОРІНЬ, і це моя помилка проєктування.**
    Перейменування жило ВСЕРЕДИНІ `_cat_chat`, тобто виконувалось лише як
    ПОБІЧНИЙ ЕФЕКТ відправки. А `notify_category` перевіряє тумблер кабінету
    (`_cat_enabled`) **ПЕРШИМ РЯДКОМ** і виходить ДО `_cat_chat`. Отже:
      • немає повідомлень (тумблер `notify_btc` вимкнено, або просто тиша) →
        `_cat_chat` не викликається → назва НІКОЛИ не оновиться;
      • і навпаки: стара назва теми — це ОЗНАКА того, що повідомлення не йдуть,
        а не окрема проблема.
    Тобто «назва не змінилась» і «сповіщень немає» ззовні виглядали як дві
    різні поломки, хоча це одна.

    ⚠️ **НІЧОГО НЕ СТВОРЮЄМО.** Беремо лише ВЖЕ ВІДОМІ теми — з персистованої
    мапи `tg_forum_topics` і з явних env `TELEGRAM_CHAT_<CAT>` +
    `TELEGRAM_TOPIC_<CAT>`. Створювати тему для вимкненої категорії було б
    протилежною помилкою: тумблер має гасити групу, а не наповнювати її.
    ⚠️ Назва теми — це ВЛАСТИВІСТЬ теми, а не сповіщення, тож тумблер її НЕ
    стосується. Викликається на СТАРТІ бота і з 🩺 перевірки теми.
    """
    done = 0
    try:
        fchat = os.getenv('TELEGRAM_FORUM_CHAT')
        if fchat:
            try:
                from storage.db_operations import get_db
                saved = get_db().get_setting('tg_forum_topics', {}) or {}
                cmap = saved.get(str(fchat), {}) if isinstance(saved, dict) else {}
            except Exception:
                cmap = {}
            for cat, tid in list((cmap or {}).items()):
                if tid and cat not in _ADMIN_ONLY_CATS:
                    _forum_rename_if_needed(fchat, cat, tid)
                    done += 1
        for cat, (cenv, tenv) in _CAT_ENV.items():
            if cat in _ADMIN_ONLY_CATS:
                continue
            chat = os.getenv(cenv) if cenv else None
            thread = os.getenv(tenv) if tenv else None
            if chat and thread:
                _forum_rename_if_needed(chat, cat, thread)
                done += 1
    except Exception as e:
        print(f"[TG] sync topic names error: {e}")
    return done


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
        _forum_rename_if_needed(chat, category, tid)
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
    # ⚠️ ПЕРЕЙМЕНУВАННЯ ПОТРІБНЕ І НА ЦЬОМУ ШЛЯХУ (скарга 21.09 «назва теми
    # залишається старою»). Тему можна задати не лише автостворенням у
    # `TELEGRAM_FORUM_CHAT`, а й напряму — `TELEGRAM_CHAT_BTC` +
    # `TELEGRAM_TOPIC_BTC`. Тоді `_forum_thread` виходить ПЕРШИМ рядком
    # (форум-чату немає), і виклик, що стояв лише там, не спрацьовував НІКОЛИ.
    if chat and thread:
        _forum_rename_if_needed(chat, category, thread)
    return chat, thread


def _cat_enabled(category):
    """Whether a GROUP category is on. 💰 Funding / 🧮 МММ-монітор follow the admin's
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


def category_check(category, send_test=False):
    """🩺 КУДИ САМЕ піде повідомлення категорії — і чому воно НЕ йде.

    Скарга 21.09 «сповіщень немає» була нерозвʼязною з боку користувача: усі
    чотири причини мовчазні — вимкнений тумблер кабінету, не налаштований
    чат/тема, бот не адмін теми, відмова Telegram. Тут вони НАЗВАНІ.
    `send_test=True` ще й реально шле пробне повідомлення В ТУ САМУ тему тим
    самим шляхом (`notify_category`), тож перевірка не «схожа на відправку», а
    і Є відправкою.
    """
    label = _CAT_LABEL.get(category, category)
    out = {'category': category, 'label': label,
           'enabled': bool(_cat_enabled(category)),
           'token': bool(_token()), 'chat': None, 'thread': None,
           'route': 'none', 'rename_err': _forum_rename_err,
           'send_err': _last_send_err, 'reason': ''}
    if not out['token']:
        out['reason'] = 'не задано TELEGRAM_BOT_TOKEN — бот не може писати нікуди'
        return out
    # 🏷 Назву теми лагодимо ДО перевірки тумблера: вона до сповіщень стосунку
    # не має, а саме на цьому місці «₿ BTCUSDT» і зависало назавжди, коли
    # повідомлення не йшли (скарга 22.09). Нічого не створюємо — лише
    # перейменовуємо вже відомі теми.
    try:
        sync_topic_names()
        out['rename_err'] = _forum_rename_err
    except Exception:
        pass
    if not out['enabled']:
        out['reason'] = ('вимкнено майстер-тумблер теми в кабінеті адміна '
                         '(«📢 Групові теми» → notify_btc)')
        return out
    chat, thread = _cat_chat(category)
    out['chat'], out['thread'] = chat, thread
    if not chat:
        out['reason'] = 'не налаштовано жодного чату (ні форум-група, ні TELEGRAM_CHAT_*)'
        return out
    if os.getenv('TELEGRAM_FORUM_CHAT') and str(chat) == str(os.getenv('TELEGRAM_FORUM_CHAT')):
        out['route'] = 'forum'
    elif _CAT_ENV.get(category) and os.getenv(_CAT_ENV[category][0]):
        out['route'] = 'env'
    else:
        out['route'] = 'admin'
        out['reason'] = ('для цієї категорії немає власного чату — повідомлення '
                         'йдуть у ПРИВАТНИЙ чат адміна, а не в тему групи')
    if not thread:
        out['reason'] = (out['reason'] or
                         'теми немає — повідомлення йде в сам чат, без теми')
    if send_test:
        ok = notify_category(category, f'🩺 Перевірка теми «{label}» — цей рядок '
                                       'надіслано з налаштувань бота.')
        out['sent'] = bool(ok)
        out['send_err'] = _last_send_err
        out['rename_err'] = _forum_rename_err       # спроба була саме зараз
        if not ok:
            out['reason'] = (f'Telegram не прийняв: {_last_send_err}' if _last_send_err
                             else (out['reason'] or 'Telegram не прийняв повідомлення'))
    return out


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
    # ⚠️ Причину відмови ЗБЕРІГАЄМО: раніше `notify_category` віддавав голий
    # bool, і «Telegram не прийняв» було не відрізнити від «ми не слали».
    global _last_send_err
    res = _api('sendMessage', p) or {}
    if res.get('ok'):
        _last_send_err = ''
        return True
    _last_send_err = str(res.get('description') or res.get('error') or 'відмова')
    print(f"[TG] send to {category} failed: {_last_send_err}")
    return False


def last_send_error():
    """Причина ОСТАННЬОЇ невдалої відправки в групову тему.

    Потрібна викликачам (`fuel_filter._broadcast_users`), щоб «повідомлення не
    пішло» можна було НАПИСАТИ В 🧾 ЛОГ дослівною причиною, а не лишати в
    stdout, якого на проді не видно (скарга 22.09).
    """
    return _last_send_err


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
    # ⚠️ Причину відмови ЗБЕРІГАЄМО: раніше `notify_category` віддавав голий
    # bool, і «Telegram не прийняв» було не відрізнити від «ми не слали».
    global _last_send_err
    res = _api('sendMessage', p) or {}
    if res.get('ok'):
        _last_send_err = ''
        return True
    _last_send_err = str(res.get('description') or res.get('error') or 'відмова')
    print(f"[TG] send to {category} failed: {_last_send_err}")
    return False


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
            "3️⃣ Ринкові сигнали (угоди, 🧮 МММ-монітор, 💰 Funding) приходитимуть "
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


def _admin_broadcast(m, admin_cid, body_override=None):
    """📢 Admin ANNOUNCEMENT — deliver to EVERY active bot subscriber. Text goes
    as `body_override` (the text after /announce); media is copied keeping its
    caption. Reports the delivered count back. Triggered ONLY by an explicit
    /announce|/broadcast — never by a plain message or a failed reply."""
    try:
        from web.auth import all_bot_chats
        chats = all_bot_chats(exclude_chat=admin_cid)
    except Exception:
        chats = []
    has_media = _has_media(m)
    n = 0
    for c in chats:
        try:
            if not has_media and body_override:
                ok = tg_send(c, f"📢 <b>Оголошення:</b>\n{body_override}")
            else:
                ok = _copy_message(c, admin_cid, m.get('message_id'))
            if ok:
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
        is_reply = rt.get('message_id') is not None
        if is_reply:
            with _map_lock:
                target = _reply_map.get(int(rt['message_id']))
            # Fallback: link gone (restart / eviction) → parse the user's chat_id
            # straight from the forwarded header text. Keeps swipe-Reply reliable
            # and, crucially, stops a reply ever leaking into a broadcast.
            if not target:
                target = _chat_id_from_text(rt.get('text') or rt.get('caption') or '')
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
        # A swipe-Reply we could NOT link to a user → NEVER broadcast; ask for the
        # explicit form instead (prevents an accidental mass send).
        if is_reply:
            tg_send(cid, "⚠️ Не вдалося визначити, кому відповісти (звʼязок втрачено). "
                         "Відповідай через <code>/reply &lt;chat_id&gt; текст</code>.")
            return
        if text.startswith('/start'):
            _handle_start(cid, uname, fname, flang, fprem)
            return
        # 📢 ANNOUNCEMENT to ALL subscribers — ONLY via an EXPLICIT command, so a
        # plain message or a failed reply can NEVER become an accidental mass send.
        if text.startswith('/announce') or text.startswith('/broadcast'):
            _body = text.split(None, 1)
            _body = _body[1].strip() if len(_body) > 1 else ''
            if not _body and not _has_media(m):
                tg_send(cid, "Формат: <code>/announce текст</code> "
                             "(або /announce у підписі до фото/відео).")
                return
            _admin_broadcast(m, cid_s, body_override=(_body or None))
            return
        if text.startswith('/'):
            return   # unknown command → ignore
        # Plain admin message that is neither a reply nor a command → a HINT, not a
        # broadcast (this is the change that stops internal replies going to all).
        tg_send(cid,
                "ℹ️ <b>Кому це надіслати?</b>\n"
                "• Відповісти користувачу — свайп-<i>Reply</i> на його повідомлення "
                "або <code>/reply &lt;chat_id&gt; текст</code>.\n"
                "• Оголошення ВСІМ підписникам — <code>/announce текст</code>.")
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
        _load_reply_map()            # restore support-reply links (survive redeploys)
    except Exception:
        pass
    try:
        _purge_admin_only_topics()   # tidy stale 📝/💬 topics from the group
    except Exception:
        pass
    try:
        # 🏷 Назви тем — НА СТАРТІ, а не «коли щось надішлеться». Інакше
        # вимкнений тумблер категорії (або просто тиша) назавжди лишав тему зі
        # старою назвою — саме так «₿ BTCUSDT» і повернулось на очі (22.09).
        sync_topic_names()
    except Exception:
        pass
    threading.Thread(target=_poll_loop, daemon=True, name='tg-bot-poll').start()
