"""Language-aware personal Telegram delivery.

The website dictionary in BotSetting is reused for trading terminology. Account
and support flows have complete local translations so a new chat can use the
Telegram locale before a profile exists; after registration the profile wins.
"""
import re

from web.i18n import normalize_language


_EXTRA = {
    'uk': {
        'login_site': '🔓 Увійти на сайт', 'register_site': '📝 Реєстрація на сайті',
        'start_registered': '👋 Ви зареєстровані як <b>{email}</b>.\n{status}\n\n💬 Питання чи труднощі? Напишіть сюди — адміністратор відповість.',
        'status_active': '✅ Акаунт активний — тисніть «Увійти на сайт».',
        'status_pending': '⏳ Акаунт очікує схвалення адміністратора.',
        'service_unavailable': '⚠️ Сервіс тимчасово недоступний (не задано публічну адресу). Зверніться до адміністратора.',
        'start_new': '👋 <b>Вітаю у VSV Bot!</b>\n\nПотрапити до групи можна <b>лише через реєстрацію на інфо-сайті</b> — інакше доступу до групи немає (нікого не додають вручну).\n\nЩоб отримати доступ:\n1️⃣ Натисніть «Реєстрація на сайті» і задайте <b>лише пароль</b> (email не потрібен — вас підтверджує Telegram).\n2️⃣ Адміністратор підтвердить вашу реєстрацію.\n3️⃣ Ринкові сигнали (угоди, 🧮 VSV-монітор, 💰 Funding) приходитимуть в <b>окрему групу</b> — посилання на вхід у групу надійде сюди <b>автоматично, після підтвердження реєстрації адміністратором</b>.\n\n🔗 <b>Уже маєте акаунт</b> (створений адміністратором)? Не реєструйтесь вдруге — увійдіть на сайт і в особистому кабінеті натисніть «🔗 Привʼязати Telegram».\n\n💬 Виникли труднощі? Просто напишіть повідомлення сюди — я передам його адміністратору, і він відповість тут.',
        'link_failed': '⚠️ Не вдалось привʼязати Telegram: {reason}',
        'link_ok': '✅ Telegram привʼязано до акаунта <b>{login}</b>.\n{status}',
        'link_active': 'Сповіщення й доступ до групи — як у зареєстрованого користувача.',
        'admin_reply': '💬 <b>Відповідь адміністратора:</b>',
        'support_sent': '✅ Ваше повідомлення надіслано адміністратору. Відповідь прийде сюди.',
        'support_unavailable': '⚠️ Підтримка тимчасово недоступна.',
        'personal_signal': 'Ваш сигнал з кабінету', 'user_tm': 'Мій Trade Manager',
    },
    'en': {
        'login_site': '🔓 Sign in to the website', 'register_site': '📝 Register on the website',
        'start_registered': '👋 You are registered as <b>{email}</b>.\n{status}\n\n💬 Questions or issues? Write here and the administrator will reply.',
        'status_active': '✅ Your account is active — select “Sign in to the website”.',
        'status_pending': '⏳ Your account is awaiting administrator approval.',
        'service_unavailable': '⚠️ The service is temporarily unavailable (no public address is configured). Contact the administrator.',
        'start_new': '👋 <b>Welcome to VSV Bot!</b>\n\nYou can join the group <b>only by registering on the info site</b> — no one is added manually.\n\nTo get access:\n1️⃣ Select “Register on the website” and choose <b>only a password</b> (no email is needed — Telegram verifies you).\n2️⃣ The administrator will approve your registration.\n3️⃣ Market signals (trades, 🧮 VSV monitor, 💰 Funding) will be posted to a <b>separate group</b> — your invitation will arrive here <b>automatically after administrator approval</b>.\n\n🔗 <b>Already have an account</b> created by the administrator? Do not register again — sign in and select “🔗 Link Telegram” in your account.\n\n💬 Need help? Send a message here — I will forward it to the administrator, who will reply in this chat.',
        'link_failed': '⚠️ Could not link Telegram: {reason}',
        'link_ok': '✅ Telegram has been linked to account <b>{login}</b>.\n{status}',
        'link_active': 'Notifications and group access are enabled as for a registered user.',
        'admin_reply': '💬 <b>Administrator reply:</b>',
        'support_sent': '✅ Your message has been sent to the administrator. The reply will arrive here.',
        'support_unavailable': '⚠️ Support is temporarily unavailable.',
        'personal_signal': 'Your account signal', 'user_tm': 'My Trade Manager',
    },
    'ru': {
        'login_site': '🔓 Войти на сайт', 'register_site': '📝 Регистрация на сайте',
        'start_registered': '👋 Вы зарегистрированы как <b>{email}</b>.\n{status}\n\n💬 Вопросы или трудности? Напишите сюда — администратор ответит.',
        'status_active': '✅ Аккаунт активен — нажмите «Войти на сайт».',
        'status_pending': '⏳ Аккаунт ожидает одобрения администратора.',
        'service_unavailable': '⚠️ Сервис временно недоступен (не указан публичный адрес). Обратитесь к администратору.',
        'start_new': '👋 <b>Добро пожаловать в VSV Bot!</b>\n\nПопасть в группу можно <b>только после регистрации на инфосайте</b> — вручную никого не добавляют.\n\nКак получить доступ:\n1️⃣ Нажмите «Регистрация на сайте» и задайте <b>только пароль</b> (email не нужен — вас подтверждает Telegram).\n2️⃣ Администратор подтвердит регистрацию.\n3️⃣ Рыночные сигналы (сделки, 🧮 VSV-монитор, 💰 Funding) будут публиковаться в <b>отдельной группе</b> — приглашение придёт сюда <b>автоматически после одобрения администратором</b>.\n\n🔗 <b>Уже есть аккаунт</b>, созданный администратором? Не регистрируйтесь повторно — войдите на сайт и выберите «🔗 Привязать Telegram» в личном кабинете.\n\n💬 Нужна помощь? Напишите сюда — я передам сообщение администратору, и он ответит в этом чате.',
        'link_failed': '⚠️ Не удалось привязать Telegram: {reason}',
        'link_ok': '✅ Telegram привязан к аккаунту <b>{login}</b>.\n{status}',
        'link_active': 'Уведомления и доступ к группе включены как для зарегистрированного пользователя.',
        'admin_reply': '💬 <b>Ответ администратора:</b>',
        'support_sent': '✅ Ваше сообщение отправлено администратору. Ответ придёт сюда.',
        'support_unavailable': '⚠️ Поддержка временно недоступна.',
        'personal_signal': 'Ваш сигнал из кабинета', 'user_tm': 'Мой Trade Manager',
    },
    'pl': {
        'login_site': '🔓 Zaloguj się na stronie', 'register_site': '📝 Zarejestruj się na stronie',
        'start_registered': '👋 Jesteś zarejestrowany jako <b>{email}</b>.\n{status}\n\n💬 Masz pytanie lub problem? Napisz tutaj — administrator odpowie.',
        'status_active': '✅ Konto jest aktywne — wybierz „Zaloguj się na stronie”.',
        'status_pending': '⏳ Konto oczekuje na zatwierdzenie przez administratora.',
        'service_unavailable': '⚠️ Usługa jest tymczasowo niedostępna (nie skonfigurowano adresu publicznego). Skontaktuj się z administratorem.',
        'start_new': '👋 <b>Witaj w VSV Bot!</b>\n\nDo grupy można dołączyć <b>wyłącznie po rejestracji na stronie informacyjnej</b> — nikt nie jest dodawany ręcznie.\n\nAby uzyskać dostęp:\n1️⃣ Wybierz „Zarejestruj się na stronie” i ustaw <b>tylko hasło</b> (adres e-mail nie jest potrzebny — Telegram potwierdza Twoją tożsamość).\n2️⃣ Administrator zatwierdzi rejestrację.\n3️⃣ Sygnały rynkowe (transakcje, 🧮 monitor VSV, 💰 Funding) będą publikowane w <b>osobnej grupie</b> — zaproszenie przyjdzie tutaj <b>automatycznie po zatwierdzeniu przez administratora</b>.\n\n🔗 <b>Masz już konto</b> utworzone przez administratora? Nie rejestruj się ponownie — zaloguj się i wybierz „🔗 Połącz Telegram” w swoim koncie.\n\n💬 Potrzebujesz pomocy? Napisz tutaj — przekażę wiadomość administratorowi, który odpowie w tym czacie.',
        'link_failed': '⚠️ Nie udało się połączyć Telegrama: {reason}',
        'link_ok': '✅ Telegram został połączony z kontem <b>{login}</b>.\n{status}',
        'link_active': 'Powiadomienia i dostęp do grupy są aktywne jak dla zarejestrowanego użytkownika.',
        'admin_reply': '💬 <b>Odpowiedź administratora:</b>',
        'support_sent': '✅ Wiadomość została wysłana do administratora. Odpowiedź przyjdzie tutaj.',
        'support_unavailable': '⚠️ Wsparcie jest tymczasowo niedostępne.',
        'personal_signal': 'Twój sygnał z konta', 'user_tm': 'Mój Trade Manager',
    },
    'de': {
        'login_site': '🔓 Auf der Website anmelden', 'register_site': '📝 Auf der Website registrieren',
        'start_registered': '👋 Sie sind als <b>{email}</b> registriert.\n{status}\n\n💬 Fragen oder Probleme? Schreiben Sie hier — der Administrator antwortet.',
        'status_active': '✅ Ihr Konto ist aktiv — wählen Sie „Auf der Website anmelden“.',
        'status_pending': '⏳ Ihr Konto wartet auf die Freigabe durch den Administrator.',
        'service_unavailable': '⚠️ Der Dienst ist vorübergehend nicht verfügbar (keine öffentliche Adresse konfiguriert). Kontaktieren Sie den Administrator.',
        'start_new': '👋 <b>Willkommen bei VSV Bot!</b>\n\nSie können der Gruppe <b>nur nach einer Registrierung auf der Info-Website</b> beitreten — niemand wird manuell hinzugefügt.\n\nSo erhalten Sie Zugang:\n1️⃣ Wählen Sie „Auf der Website registrieren“ und vergeben Sie <b>nur ein Passwort</b> (keine E-Mail erforderlich — Telegram bestätigt Ihre Identität).\n2️⃣ Der Administrator gibt Ihre Registrierung frei.\n3️⃣ Marktsignale (Trades, 🧮 VSV-Monitor, 💰 Funding) werden in einer <b>separaten Gruppe</b> veröffentlicht — Ihre Einladung kommt <b>automatisch nach der Freigabe</b> hier an.\n\n🔗 <b>Sie haben bereits ein Konto</b> vom Administrator? Registrieren Sie sich nicht erneut — melden Sie sich an und wählen Sie „🔗 Telegram verknüpfen“ in Ihrem Konto.\n\n💬 Benötigen Sie Hilfe? Schreiben Sie hier — ich leite die Nachricht an den Administrator weiter, der in diesem Chat antwortet.',
        'link_failed': '⚠️ Telegram konnte nicht verknüpft werden: {reason}',
        'link_ok': '✅ Telegram wurde mit dem Konto <b>{login}</b> verknüpft.\n{status}',
        'link_active': 'Benachrichtigungen und Gruppenzugang sind wie für registrierte Nutzer aktiviert.',
        'admin_reply': '💬 <b>Antwort des Administrators:</b>',
        'support_sent': '✅ Ihre Nachricht wurde an den Administrator gesendet. Die Antwort kommt hier an.',
        'support_unavailable': '⚠️ Der Support ist vorübergehend nicht verfügbar.',
        'personal_signal': 'Ihr Kontosignal', 'user_tm': 'Mein Trade Manager',
    },
    'es': {
        'login_site': '🔓 Iniciar sesión en el sitio', 'register_site': '📝 Registrarse en el sitio',
        'start_registered': '👋 Estás registrado como <b>{email}</b>.\n{status}\n\n💬 ¿Dudas o problemas? Escribe aquí y el administrador responderá.',
        'status_active': '✅ Tu cuenta está activa: pulsa «Iniciar sesión en el sitio».',
        'status_pending': '⏳ Tu cuenta espera la aprobación del administrador.',
        'service_unavailable': '⚠️ El servicio no está disponible temporalmente (no hay una dirección pública configurada). Contacta con el administrador.',
        'start_new': '👋 <b>¡Bienvenido a VSV Bot!</b>\n\nSolo puedes unirte al grupo <b>registrándote en el sitio de información</b>; nadie se añade manualmente.\n\nPara obtener acceso:\n1️⃣ Pulsa «Registrarse en el sitio» y elige <b>solo una contraseña</b> (no necesitas correo: Telegram verifica tu identidad).\n2️⃣ El administrador aprobará tu registro.\n3️⃣ Las señales de mercado (operaciones, 🧮 monitor VSV, 💰 Funding) se publicarán en un <b>grupo independiente</b>; la invitación llegará aquí <b>automáticamente tras la aprobación</b>.\n\n🔗 <b>¿Ya tienes una cuenta</b> creada por el administrador? No te registres de nuevo: inicia sesión y pulsa «🔗 Vincular Telegram» en tu cuenta.\n\n💬 ¿Necesitas ayuda? Escribe aquí: enviaré el mensaje al administrador y responderá en este chat.',
        'link_failed': '⚠️ No se pudo vincular Telegram: {reason}',
        'link_ok': '✅ Telegram se ha vinculado a la cuenta <b>{login}</b>.\n{status}',
        'link_active': 'Las notificaciones y el acceso al grupo están activados como para un usuario registrado.',
        'admin_reply': '💬 <b>Respuesta del administrador:</b>',
        'support_sent': '✅ Tu mensaje se ha enviado al administrador. La respuesta llegará aquí.',
        'support_unavailable': '⚠️ El soporte no está disponible temporalmente.',
        'personal_signal': 'Tu señal de la cuenta', 'user_tm': 'Mi Trade Manager',
    },
}


def telegram_tr(language, key, **variables):
    lang = normalize_language(language)
    value = _EXTRA.get(lang, {}).get(key, _EXTRA['en'].get(key, _EXTRA['uk'].get(key, key)))
    for name, replacement in variables.items():
        value = value.replace('{' + name + '}', str(replacement))
    return value


def language_for_chat(chat_id):
    try:
        from web.auth import get_user_by_chat, _load_prefs
        from web.i18n import language_from_prefs
        user = get_user_by_chat(str(chat_id))
        return language_from_prefs(_load_prefs(user)) if user else None
    except Exception:
        return None


def _replace_phrase(text, source, translated):
    if not source or source.startswith('_') or source == translated:
        return text
    try:
        return re.sub(r'(?<!\w)' + re.escape(source) + r'(?!\w)',
                      lambda _: translated, text)
    except re.error:
        return text


def translate_message(text, language):
    """Translate HTML without altering market values, symbols or valid markup."""
    lang = normalize_language(language)
    if lang == 'uk' or not text:
        return text
    raw = str(text)
    for values in (_EXTRA.get(lang, {}),):
        for key, value in values.items():
            source = _EXTRA['uk'].get(key)
            if source and ('{' not in source):
                if raw == source:
                    return value
                raw = _replace_phrase(raw, source, value)
    try:
        from web.i18n_store import get_dictionary
        result = get_dictionary(lang) or {}
        strings = result.get('strings') or {}
        if raw in strings:
            return strings[raw]
        for source in sorted((x for x in strings if isinstance(x, str) and not x.startswith('_')),
                             key=len, reverse=True):
            translated = strings.get(source)
            if isinstance(translated, str) and len(source) >= 3:
                raw = _replace_phrase(raw, source, translated)
    except Exception:
        pass
    return raw


def translate_chat_message(chat_id, text, buttons=None):
    lang = language_for_chat(chat_id)
    if not lang or normalize_language(lang) == 'uk':
        return text, buttons
    translated = translate_message(text, lang)
    if buttons:
        buttons = [[dict(button, text=translate_message(button.get('text', ''), lang))
                    for button in row] for row in buttons]
    return translated, buttons
