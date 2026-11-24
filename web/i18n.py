"""Profile language and server-side translations for VSV Bot and InfoSite.

The profile preference is authoritative after authentication.  Browser storage is
only a pre-login convenience; Telegram language_code is only an initial hint.
"""
import json

SUPPORTED_LANGUAGES = ('uk', 'en', 'ru', 'pl', 'de', 'es')
LANGUAGE_NAMES = {
    'uk': 'Українська', 'en': 'English', 'ru': 'Русский',
    'pl': 'Polski', 'de': 'Deutsch', 'es': 'Español',
}

_STRINGS = {
    'uk': {
        'language': 'Мова', 'login': 'Увійти', 'register': 'Зареєструватися',
        'login_title': 'Вхід', 'register_title': 'Реєстрація',
        'login_subtitle': 'Ваш логін та пароль, створені при реєстрації в Telegram',
        'register_subtitle': 'Через Telegram — логін і пароль',
        'login_name': 'Логін (ваше ім’я для входу)',
        'register_name': 'Логін (ім’я для входу)',
        'login_placeholder': 'ім’я, яке ви задали при реєстрації',
        'register_placeholder': 'напр. myname', 'password': 'Пароль',
        'password_min': 'Пароль (мін. 8)', 'password_repeat': 'Повторіть пароль',
        'have_account': '← Вже маю акаунт', 'to_login': '← До входу',
        'register_telegram': 'Реєстрація через Telegram',
        'telegram_confirmed': '🔗 Telegram підтверджено. Придумайте <b>логін</b> (ім’я для входу) і пароль — далі схвалення адміністратора.',
        'registration_only_telegram': 'Реєстрація — лише через Telegram, без email.',
        'open_telegram': '📨 Відкрити Telegram-бота',
        'start_bot_hint': 'Натисніть <b>/start</b> у боті — він надішле персональне посилання для завершення реєстрації.',
        'find_bot': 'Знайдіть нашого бота в Telegram і натисніть <b>/start</b>.',
        'ready': 'Готово ✓',
        'registration_saved': 'Акаунт створено й прив’язано до вашого Telegram. Після схвалення адміністратором ви отримаєте сповіщення в Telegram і зможете увійти.',
        'login_short': 'Увійти',
        'access_approved': '✅ <b>Ваш акаунт активовано!</b>\nТепер увійдіть на сайті своїм <b>логіном</b> і паролем.',
        'access_rejected': '⛔ <b>Доступ до VSV відхилено адміністратором.</b>\nЯкщо вважаєте це помилкою, напишіть адміністратору в цьому чаті.',
        'access_pending': '✅ Акаунт <b>{login}</b> створено. Очікуйте схвалення адміністратора — я повідомлю тут.',
        'access_suspended': '⚠️ Ваш доступ призупинено адміністратором.',
        'bot_granted': '🤖 Вам надано доступ до бота.',
        'bot_revoked': '🤖 Доступ до бота вимкнено.',
        'account_disabled': '⛔ Ваш акаунт вимкнено адміністратором.',
        'account_enabled': '✅ Ваш акаунт знову активний.',
        'admin_granted': '🛡 Вам надано права адміністратора.',
        'admin_revoked': '🛡 Права адміністратора знято.',
        'password_changed': '🔑 Адміністратор змінив ваш пароль. Використовуйте новий пароль для входу.',
        'sessions_closed': '🔒 Ваші сесії завершено — увійдіть знову.',
        'account_deleted': '🗑 Ваш акаунт видалено адміністратором.',
        'login_short_button': '🔐 Увійти',
    },
    'en': {
        'language': 'Language', 'login': 'Sign in', 'register': 'Create account',
        'login_title': 'Sign in', 'register_title': 'Registration',
        'login_subtitle': 'Use the login and password you created during Telegram registration',
        'register_subtitle': 'Via Telegram — login and password',
        'login_name': 'Login (your sign-in name)', 'register_name': 'Login (sign-in name)',
        'login_placeholder': 'the name you chose during registration',
        'register_placeholder': 'e.g. myname', 'password': 'Password',
        'password_min': 'Password (min. 8)', 'password_repeat': 'Confirm password',
        'have_account': '← I already have an account', 'to_login': '← Back to sign in',
        'register_telegram': 'Register via Telegram',
        'telegram_confirmed': '🔗 Telegram verified. Create a <b>login</b> (sign-in name) and password — then wait for administrator approval.',
        'registration_only_telegram': 'Registration is available only through Telegram; no email is required.',
        'open_telegram': '📨 Open Telegram bot',
        'start_bot_hint': 'Send <b>/start</b> to the bot — it will provide your personal registration link.',
        'find_bot': 'Find our Telegram bot and send <b>/start</b>.',
        'ready': 'Done ✓',
        'registration_saved': 'Your account has been created and linked to Telegram. You will receive a Telegram notification after administrator approval and can then sign in.',
        'login_short': 'Sign in',
        'access_approved': '✅ <b>Your account is active!</b>\nYou can now sign in with your <b>login</b> and password.',
        'access_rejected': '⛔ <b>Your access to VSV was declined by the administrator.</b>\nIf you believe this is a mistake, contact the administrator in this chat.',
        'access_pending': '✅ Account <b>{login}</b> has been created. Please wait for administrator approval — I will notify you here.',
        'access_suspended': '⚠️ Your access has been suspended by the administrator.',
        'bot_granted': '🤖 You have been granted access to the bot.',
        'bot_revoked': '🤖 Your access to the bot has been disabled.',
        'account_disabled': '⛔ Your account has been disabled by the administrator.',
        'account_enabled': '✅ Your account is active again.',
        'admin_granted': '🛡 You have been granted administrator privileges.',
        'admin_revoked': '🛡 Your administrator privileges have been revoked.',
        'password_changed': '🔑 The administrator changed your password. Use the new password to sign in.',
        'sessions_closed': '🔒 Your sessions have ended — please sign in again.',
        'account_deleted': '🗑 Your account has been deleted by the administrator.',
        'login_short_button': '🔐 Sign in',
    },
    'ru': {
        'language': 'Язык', 'login': 'Войти', 'register': 'Зарегистрироваться',
        'login_title': 'Вход', 'register_title': 'Регистрация',
        'login_subtitle': 'Логин и пароль, созданные при регистрации в Telegram',
        'register_subtitle': 'Через Telegram — логин и пароль',
        'login_name': 'Логин (ваше имя для входа)', 'register_name': 'Логин (имя для входа)',
        'login_placeholder': 'имя, указанное при регистрации', 'register_placeholder': 'например, myname',
        'password': 'Пароль', 'password_min': 'Пароль (мин. 8)', 'password_repeat': 'Повторите пароль',
        'have_account': '← У меня уже есть аккаунт', 'to_login': '← Ко входу',
        'register_telegram': 'Регистрация через Telegram',
        'telegram_confirmed': '🔗 Telegram подтверждён. Придумайте <b>логин</b> (имя для входа) и пароль — затем дождитесь одобрения администратора.',
        'registration_only_telegram': 'Регистрация — только через Telegram, без email.',
        'open_telegram': '📨 Открыть Telegram-бота',
        'start_bot_hint': 'Нажмите <b>/start</b> в боте — он отправит персональную ссылку для завершения регистрации.',
        'find_bot': 'Найдите нашего бота в Telegram и нажмите <b>/start</b>.', 'ready': 'Готово ✓',
        'registration_saved': 'Аккаунт создан и привязан к вашему Telegram. После одобрения администратором вы получите уведомление в Telegram и сможете войти.',
        'login_short': 'Войти',
        'access_approved': '✅ <b>Ваш аккаунт активирован!</b>\nТеперь войдите на сайт, используя <b>логин</b> и пароль.',
        'access_rejected': '⛔ <b>Администратор отклонил ваш доступ к VSV.</b>\nЕсли вы считаете это ошибкой, напишите администратору в этом чате.',
        'access_pending': '✅ Аккаунт <b>{login}</b> создан. Ожидайте одобрения администратора — я сообщу здесь.',
        'access_suspended': '⚠️ Ваш доступ приостановлен администратором.',
        'bot_granted': '🤖 Вам предоставлен доступ к боту.', 'bot_revoked': '🤖 Доступ к боту отключён.',
        'account_disabled': '⛔ Ваш аккаунт отключён администратором.', 'account_enabled': '✅ Ваш аккаунт снова активен.',
        'admin_granted': '🛡 Вам предоставлены права администратора.', 'admin_revoked': '🛡 Права администратора отозваны.',
        'password_changed': '🔑 Администратор изменил ваш пароль. Используйте новый пароль для входа.',
        'sessions_closed': '🔒 Ваши сеансы завершены — войдите снова.', 'account_deleted': '🗑 Ваш аккаунт удалён администратором.',
        'login_short_button': '🔐 Войти',
    },
    'pl': {
        'language': 'Język', 'login': 'Zaloguj się', 'register': 'Utwórz konto',
        'login_title': 'Logowanie', 'register_title': 'Rejestracja',
        'login_subtitle': 'Użyj loginu i hasła utworzonego podczas rejestracji w Telegramie',
        'register_subtitle': 'Przez Telegram — login i hasło',
        'login_name': 'Login (nazwa do logowania)', 'register_name': 'Login (nazwa do logowania)',
        'login_placeholder': 'nazwa wybrana podczas rejestracji', 'register_placeholder': 'np. myname',
        'password': 'Hasło', 'password_min': 'Hasło (min. 8)', 'password_repeat': 'Powtórz hasło',
        'have_account': '← Mam już konto', 'to_login': '← Powrót do logowania',
        'register_telegram': 'Rejestracja przez Telegram',
        'telegram_confirmed': '🔗 Telegram zweryfikowany. Utwórz <b>login</b> i hasło, a następnie poczekaj na zatwierdzenie przez administratora.',
        'registration_only_telegram': 'Rejestracja jest dostępna wyłącznie przez Telegram; adres e-mail nie jest wymagany.',
        'open_telegram': '📨 Otwórz bota Telegram',
        'start_bot_hint': 'Wyślij <b>/start</b> do bota — otrzymasz osobisty link rejestracyjny.',
        'find_bot': 'Znajdź naszego bota Telegram i wyślij <b>/start</b>.', 'ready': 'Gotowe ✓',
        'registration_saved': 'Konto zostało utworzone i połączone z Telegramem. Po zatwierdzeniu przez administratora otrzymasz powiadomienie i będziesz mógł się zalogować.',
        'login_short': 'Zaloguj się',
        'access_approved': '✅ <b>Twoje konto jest aktywne!</b>\nMożesz teraz zalogować się przy użyciu <b>loginu</b> i hasła.',
        'access_rejected': '⛔ <b>Administrator odrzucił Twój dostęp do VSV.</b>\nJeśli uważasz, że to pomyłka, skontaktuj się z administratorem w tym czacie.',
        'access_pending': '✅ Konto <b>{login}</b> zostało utworzone. Poczekaj na zatwierdzenie przez administratora — powiadomię Cię tutaj.',
        'access_suspended': '⚠️ Twój dostęp został zawieszony przez administratora.',
        'bot_granted': '🤖 Przyznano Ci dostęp do bota.', 'bot_revoked': '🤖 Twój dostęp do bota został wyłączony.',
        'account_disabled': '⛔ Twoje konto zostało wyłączone przez administratora.', 'account_enabled': '✅ Twoje konto jest ponownie aktywne.',
        'admin_granted': '🛡 Przyznano Ci uprawnienia administratora.', 'admin_revoked': '🛡 Twoje uprawnienia administratora zostały cofnięte.',
        'password_changed': '🔑 Administrator zmienił Twoje hasło. Użyj nowego hasła podczas logowania.',
        'sessions_closed': '🔒 Twoje sesje zostały zakończone — zaloguj się ponownie.', 'account_deleted': '🗑 Twoje konto zostało usunięte przez administratora.',
        'login_short_button': '🔐 Zaloguj się',
    },
    'de': {
        'language': 'Sprache', 'login': 'Anmelden', 'register': 'Konto erstellen',
        'login_title': 'Anmeldung', 'register_title': 'Registrierung',
        'login_subtitle': 'Verwenden Sie den bei der Telegram-Registrierung erstellten Login und das Passwort',
        'register_subtitle': 'Über Telegram — Login und Passwort',
        'login_name': 'Login (Ihr Anmeldename)', 'register_name': 'Login (Anmeldename)',
        'login_placeholder': 'der bei der Registrierung gewählte Name', 'register_placeholder': 'z. B. myname',
        'password': 'Passwort', 'password_min': 'Passwort (mind. 8)', 'password_repeat': 'Passwort bestätigen',
        'have_account': '← Ich habe bereits ein Konto', 'to_login': '← Zur Anmeldung',
        'register_telegram': 'Registrierung über Telegram',
        'telegram_confirmed': '🔗 Telegram bestätigt. Erstellen Sie einen <b>Login</b> und ein Passwort — anschließend erfolgt die Freigabe durch den Administrator.',
        'registration_only_telegram': 'Die Registrierung ist nur über Telegram möglich; keine E-Mail erforderlich.',
        'open_telegram': '📨 Telegram-Bot öffnen',
        'start_bot_hint': 'Senden Sie <b>/start</b> an den Bot — Sie erhalten Ihren persönlichen Registrierungslink.',
        'find_bot': 'Suchen Sie unseren Telegram-Bot und senden Sie <b>/start</b>.', 'ready': 'Fertig ✓',
        'registration_saved': 'Ihr Konto wurde erstellt und mit Telegram verknüpft. Nach der Freigabe durch den Administrator erhalten Sie eine Telegram-Benachrichtigung und können sich anmelden.',
        'login_short': 'Anmelden',
        'access_approved': '✅ <b>Ihr Konto ist aktiv!</b>\nSie können sich jetzt mit Ihrem <b>Login</b> und Passwort anmelden.',
        'access_rejected': '⛔ <b>Ihr Zugang zu VSV wurde vom Administrator abgelehnt.</b>\nWenn Sie dies für einen Fehler halten, kontaktieren Sie den Administrator in diesem Chat.',
        'access_pending': '✅ Konto <b>{login}</b> wurde erstellt. Bitte warten Sie auf die Freigabe durch den Administrator — ich benachrichtige Sie hier.',
        'access_suspended': '⚠️ Ihr Zugang wurde vom Administrator vorübergehend gesperrt.',
        'bot_granted': '🤖 Sie haben Zugriff auf den Bot erhalten.', 'bot_revoked': '🤖 Ihr Bot-Zugriff wurde deaktiviert.',
        'account_disabled': '⛔ Ihr Konto wurde vom Administrator deaktiviert.', 'account_enabled': '✅ Ihr Konto ist wieder aktiv.',
        'admin_granted': '🛡 Sie haben Administratorrechte erhalten.', 'admin_revoked': '🛡 Ihre Administratorrechte wurden entzogen.',
        'password_changed': '🔑 Der Administrator hat Ihr Passwort geändert. Verwenden Sie zur Anmeldung das neue Passwort.',
        'sessions_closed': '🔒 Ihre Sitzungen wurden beendet — melden Sie sich erneut an.', 'account_deleted': '🗑 Ihr Konto wurde vom Administrator gelöscht.',
        'login_short_button': '🔐 Anmelden',
    },
    'es': {
        'language': 'Idioma', 'login': 'Iniciar sesión', 'register': 'Crear cuenta',
        'login_title': 'Inicio de sesión', 'register_title': 'Registro',
        'login_subtitle': 'Usa el usuario y la contraseña creados durante el registro en Telegram',
        'register_subtitle': 'A través de Telegram: usuario y contraseña',
        'login_name': 'Usuario (nombre de acceso)', 'register_name': 'Usuario (nombre de acceso)',
        'login_placeholder': 'el nombre elegido durante el registro', 'register_placeholder': 'p. ej., myname',
        'password': 'Contraseña', 'password_min': 'Contraseña (mín. 8)', 'password_repeat': 'Confirmar contraseña',
        'have_account': '← Ya tengo una cuenta', 'to_login': '← Volver a iniciar sesión',
        'register_telegram': 'Registrarse mediante Telegram',
        'telegram_confirmed': '🔗 Telegram verificado. Crea un <b>usuario</b> y una contraseña; después espera la aprobación del administrador.',
        'registration_only_telegram': 'El registro solo está disponible mediante Telegram; no se necesita correo electrónico.',
        'open_telegram': '📨 Abrir el bot de Telegram',
        'start_bot_hint': 'Envía <b>/start</b> al bot: recibirás tu enlace personal de registro.',
        'find_bot': 'Busca nuestro bot de Telegram y envía <b>/start</b>.', 'ready': 'Listo ✓',
        'registration_saved': 'Tu cuenta se ha creado y vinculado a Telegram. Recibirás una notificación cuando el administrador la apruebe y podrás iniciar sesión.',
        'login_short': 'Iniciar sesión',
        'access_approved': '✅ <b>¡Tu cuenta está activa!</b>\nYa puedes iniciar sesión con tu <b>usuario</b> y contraseña.',
        'access_rejected': '⛔ <b>El administrador ha rechazado tu acceso a VSV.</b>\nSi crees que se trata de un error, escribe al administrador en este chat.',
        'access_pending': '✅ La cuenta <b>{login}</b> se ha creado. Espera la aprobación del administrador; te avisaré aquí.',
        'access_suspended': '⚠️ El administrador ha suspendido tu acceso.',
        'bot_granted': '🤖 Se te ha concedido acceso al bot.', 'bot_revoked': '🤖 Tu acceso al bot se ha desactivado.',
        'account_disabled': '⛔ El administrador ha desactivado tu cuenta.', 'account_enabled': '✅ Tu cuenta vuelve a estar activa.',
        'admin_granted': '🛡 Se te han concedido privilegios de administrador.', 'admin_revoked': '🛡 Se han revocado tus privilegios de administrador.',
        'password_changed': '🔑 El administrador ha cambiado tu contraseña. Utiliza la nueva contraseña para iniciar sesión.',
        'sessions_closed': '🔒 Tus sesiones han finalizado; vuelve a iniciar sesión.', 'account_deleted': '🗑 El administrador ha eliminado tu cuenta.',
        'login_short_button': '🔐 Iniciar sesión',
    },
}

_AUTH_ERRORS = {
    'uk': {
        'error_throttled': 'Забагато спроб. Спробуйте за кілька хвилин.',
        'error_credentials': 'Невірний логін або пароль.',
        'error_disabled': 'Акаунт вимкнено адміністратором.',
        'error_login_min': 'Логін — мінімум 3 символи.',
        'error_password_min': 'Пароль — мінімум 8 символів.',
        'error_password_match': 'Паролі не збігаються.',
        'error_login_taken': 'Такий логін уже зайнятий — виберіть інший.',
        'session_superseded': 'Сесію завершено: у цей акаунт увійшли з іншого пристрою.',
    },
    'en': {
        'error_throttled': 'Too many attempts. Please try again in a few minutes.',
        'error_credentials': 'Incorrect login or password.',
        'error_disabled': 'Your account has been disabled by the administrator.',
        'error_login_min': 'Login must contain at least 3 characters.',
        'error_password_min': 'Password must contain at least 8 characters.',
        'error_password_match': 'Passwords do not match.',
        'error_login_taken': 'This login is already taken — please choose another one.',
        'session_superseded': 'Your session ended because this account was signed in on another device.',
    },
    'ru': {
        'error_throttled': 'Слишком много попыток. Повторите через несколько минут.',
        'error_credentials': 'Неверный логин или пароль.',
        'error_disabled': 'Аккаунт отключён администратором.',
        'error_login_min': 'Логин — минимум 3 символа.',
        'error_password_min': 'Пароль — минимум 8 символов.',
        'error_password_match': 'Пароли не совпадают.',
        'error_login_taken': 'Этот логин уже занят — выберите другой.',
        'session_superseded': 'Сеанс завершён: в аккаунт вошли с другого устройства.',
    },
    'pl': {
        'error_throttled': 'Zbyt wiele prób. Spróbuj ponownie za kilka minut.',
        'error_credentials': 'Nieprawidłowy login lub hasło.',
        'error_disabled': 'Twoje konto zostało wyłączone przez administratora.',
        'error_login_min': 'Login musi mieć co najmniej 3 znaki.',
        'error_password_min': 'Hasło musi mieć co najmniej 8 znaków.',
        'error_password_match': 'Hasła nie są zgodne.',
        'error_login_taken': 'Ten login jest już zajęty — wybierz inny.',
        'session_superseded': 'Sesja zakończona: konto zostało otwarte na innym urządzeniu.',
    },
    'de': {
        'error_throttled': 'Zu viele Versuche. Bitte versuchen Sie es in einigen Minuten erneut.',
        'error_credentials': 'Login oder Passwort ist falsch.',
        'error_disabled': 'Ihr Konto wurde vom Administrator deaktiviert.',
        'error_login_min': 'Der Login muss mindestens 3 Zeichen enthalten.',
        'error_password_min': 'Das Passwort muss mindestens 8 Zeichen enthalten.',
        'error_password_match': 'Die Passwörter stimmen nicht überein.',
        'error_login_taken': 'Dieser Login ist bereits vergeben — bitte wählen Sie einen anderen.',
        'session_superseded': 'Sitzung beendet: Das Konto wurde auf einem anderen Gerät angemeldet.',
    },
    'es': {
        'error_throttled': 'Demasiados intentos. Inténtalo de nuevo en unos minutos.',
        'error_credentials': 'Usuario o contraseña incorrectos.',
        'error_disabled': 'El administrador ha desactivado tu cuenta.',
        'error_login_min': 'El usuario debe tener al menos 3 caracteres.',
        'error_password_min': 'La contraseña debe tener al menos 8 caracteres.',
        'error_password_match': 'Las contraseñas no coinciden.',
        'error_login_taken': 'Este usuario ya está ocupado; elige otro.',
        'session_superseded': 'La sesión finalizó porque la cuenta se abrió en otro dispositivo.',
    },
}
for _lang, _values in _AUTH_ERRORS.items():
    _STRINGS[_lang].update(_values)


def normalize_language(value, default='uk'):
    code = str(value or '').strip().lower().replace('_', '-').split('-')[0]
    return code if code in SUPPORTED_LANGUAGES else default


def language_from_prefs(prefs, default='uk'):
    try:
        if isinstance(prefs, str):
            prefs = json.loads(prefs or '{}')
        return normalize_language((prefs or {}).get('language'), default)
    except Exception:
        return normalize_language(default)


def language_for_user(uid, default='uk'):
    try:
        from web.auth import get_user_by_id, _load_prefs
        user = get_user_by_id(uid)
        return language_from_prefs(_load_prefs(user), default) if user else normalize_language(default)
    except Exception:
        return normalize_language(default)


def tr(language, key, **variables):
    lang = normalize_language(language)
    value = _STRINGS.get(lang, {}).get(key)
    if value is None:
        value = _STRINGS['en'].get(key, _STRINGS['uk'].get(key, key))
    if variables:
        for name, replacement in variables.items():
            value = value.replace('{' + name + '}', str(replacement))
    return value


def user_tr(uid, key, **variables):
    return tr(language_for_user(uid), key, **variables)


def language_options(selected='uk'):
    selected = normalize_language(selected)
    return ''.join(
        '<option value="{code}"{selected}>{name}</option>'.format(
            code=code, name=LANGUAGE_NAMES[code],
            selected=' selected' if code == selected else '')
        for code in SUPPORTED_LANGUAGES)


def client_dictionaries():
    """Trusted local strings for instant auth-page language switching."""
    return json.dumps(_STRINGS, ensure_ascii=False).replace('</', '<\\/')
