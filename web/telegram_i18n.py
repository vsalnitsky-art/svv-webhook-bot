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


# Нейтральні персональні сповіщення про аномалії обсягу. Текст формується один
# раз на монету, а `translate_message` замінює цілі фрази мовою отримувача.
# Терміни LONG/SHORT навмисно відсутні: спостереження не видається за прогноз.
_ANOMALY_EXTRA = {
    'uk': {
        'personal_observation': 'Ваше спостереження з кабінету',
        'anom_title_short': 'VSV · Аномалія обсягу',
        'anom_title_full': 'Аномалії обсягу · ринковий контекст',
        'anom_context_default': 'Зафіксовано незвичну послідовність обсягу та реакції ціни: баланс ринкової активності може змінюватися.',
        'anom_context_e1': 'Послідовність кульмінації й абсорбції: ринок перевіряє зміну балансу ініціативи біля рівня.',
        'anom_context_e2': 'Після стиснення зафіксовано імпульс на обсязі: ринок перевіряє прийняття ціни за межею діапазону.',
        'anom_context_e3': 'Після імпульсу сформовано відкат і нову реакцію біля рівня: перевіряється стійкість поточної структури.',
        'anom_context_e4': 'Межу діапазону проколото з подальшим поверненням: ринок перевіряє, чи був це збір ліквідності, а не справжній пробій.',
        'anom_price_observed': 'Ціна в момент спостереження:',
        'anom_attention': 'На що звернути увагу на графіку',
        'anom_check_level': 'Реакція ціни на найближчий рівень або межу діапазону.',
        'anom_check_close': 'Чи підтверджує наступна свічка реакцію, а не одразу скасовує її.',
        'anom_check_absorption': 'Чи не з’являється поглинання імпульсу та повернення в попередній діапазон.',
        'anom_watch_short': 'На графіку перевірте реакцію біля рівня та закриття наступної свічки.',
        'anom_disclaimer_full': 'Напрямок не прогнозується. Це спостереження за зміною ринкової активності, а не сигнал до входу — рішення залишається за вашою стратегією.',
        'anom_disclaimer_short': 'Напрямок не прогнозується. Це спостереження, а не сигнал до входу.',
    },
    'en': {
        'personal_observation': 'Your account observation',
        'anom_title_short': 'VSV · Volume anomaly',
        'anom_title_full': 'Volume anomalies · Market context',
        'anom_context_default': 'An unusual sequence of volume and price reactions has been detected: the balance of market activity may be shifting.',
        'anom_context_e1': 'A climax-and-absorption sequence: the market is testing whether control is changing hands near a level.',
        'anom_context_e2': 'After compression, a volume-backed impulse has been detected: the market is testing acceptance beyond the range.',
        'anom_context_e3': 'An impulse was followed by a pullback and another reaction near a level: the current structure is being tested.',
        'anom_context_e4': 'The range boundary was swept and price returned inside: the market is testing whether this was liquidity collection rather than a genuine breakout.',
        'anom_price_observed': 'Price at observation:',
        'anom_attention': 'What to check on the chart',
        'anom_check_level': 'How price reacts to the nearest level or range boundary.',
        'anom_check_close': 'Whether the next candle confirms the reaction instead of immediately invalidating it.',
        'anom_check_absorption': 'Whether the impulse is absorbed and price returns to the previous range.',
        'anom_watch_short': 'On the chart, check the reaction near the level and the next candle close.',
        'anom_disclaimer_full': 'No direction is being forecast. This is an observation of changing market activity, not an entry signal — the decision remains with your strategy.',
        'anom_disclaimer_short': 'No direction is being forecast. This is an observation, not an entry signal.',
    },
    'ru': {
        'personal_observation': 'Ваше наблюдение из кабинета',
        'anom_title_short': 'VSV · Аномалия объёма',
        'anom_title_full': 'Аномалии объёма · Рыночный контекст',
        'anom_context_default': 'Зафиксирована необычная последовательность объёма и реакции цены: баланс рыночной активности может меняться.',
        'anom_context_e1': 'Последовательность кульминации и абсорбции: рынок проверяет смену баланса инициативы возле уровня.',
        'anom_context_e2': 'После сжатия зафиксирован импульс на объёме: рынок проверяет принятие цены за границей диапазона.',
        'anom_context_e3': 'После импульса сформирован откат и новая реакция возле уровня: проверяется устойчивость текущей структуры.',
        'anom_context_e4': 'Граница диапазона была проколота с последующим возвратом: рынок проверяет, был ли это сбор ликвидности, а не истинный пробой.',
        'anom_price_observed': 'Цена в момент наблюдения:',
        'anom_attention': 'На что обратить внимание на графике',
        'anom_check_level': 'Реакция цены на ближайший уровень или границу диапазона.',
        'anom_check_close': 'Подтверждает ли следующая свеча реакцию, а не сразу отменяет её.',
        'anom_check_absorption': 'Не возникает ли поглощение импульса и возврат в предыдущий диапазон.',
        'anom_watch_short': 'На графике проверьте реакцию возле уровня и закрытие следующей свечи.',
        'anom_disclaimer_full': 'Направление не прогнозируется. Это наблюдение за изменением рыночной активности, а не сигнал для входа — решение остаётся за вашей стратегией.',
        'anom_disclaimer_short': 'Направление не прогнозируется. Это наблюдение, а не сигнал для входа.',
    },
    'pl': {
        'personal_observation': 'Twoja obserwacja z konta',
        'anom_title_short': 'VSV · Anomalia wolumenu',
        'anom_title_full': 'Anomalie wolumenu · Kontekst rynkowy',
        'anom_context_default': 'Wykryto nietypową sekwencję wolumenu i reakcji ceny: równowaga aktywności rynkowej może się zmieniać.',
        'anom_context_e1': 'Sekwencja klimaksu i absorpcji: rynek sprawdza, czy dochodzi do zmiany inicjatywy przy poziomie.',
        'anom_context_e2': 'Po kompresji zarejestrowano impuls na wolumenie: rynek sprawdza akceptację ceny poza zakresem.',
        'anom_context_e3': 'Po impulsie powstała korekta i kolejna reakcja przy poziomie: sprawdzana jest trwałość obecnej struktury.',
        'anom_context_e4': 'Granica zakresu została przebita, po czym cena wróciła do zakresu: rynek sprawdza, czy było to zebranie płynności, a nie prawdziwe wybicie.',
        'anom_price_observed': 'Cena w momencie obserwacji:',
        'anom_attention': 'Co sprawdzić na wykresie',
        'anom_check_level': 'Reakcja ceny na najbliższy poziom lub granicę zakresu.',
        'anom_check_close': 'Czy następna świeca potwierdza reakcję, zamiast natychmiast ją unieważniać.',
        'anom_check_absorption': 'Czy impuls nie zostaje zaabsorbowany, a cena nie wraca do poprzedniego zakresu.',
        'anom_watch_short': 'Na wykresie sprawdź reakcję przy poziomie i zamknięcie następnej świecy.',
        'anom_disclaimer_full': 'Kierunek nie jest prognozowany. To obserwacja zmieniającej się aktywności rynkowej, a nie sygnał wejścia — decyzja należy do Twojej strategii.',
        'anom_disclaimer_short': 'Kierunek nie jest prognozowany. To obserwacja, a nie sygnał wejścia.',
    },
    'de': {
        'personal_observation': 'Ihre Konto-Beobachtung',
        'anom_title_short': 'VSV · Volumenanomalie',
        'anom_title_full': 'Volumenanomalien · Marktkontext',
        'anom_context_default': 'Eine ungewöhnliche Sequenz aus Volumen und Kursreaktionen wurde festgestellt: Das Gleichgewicht der Marktaktivität kann sich verändern.',
        'anom_context_e1': 'Eine Sequenz aus Klimax und Absorption: Der Markt prüft einen Wechsel der Initiative nahe einem Niveau.',
        'anom_context_e2': 'Nach der Kompression wurde ein volumengestützter Impuls festgestellt: Der Markt prüft die Akzeptanz des Kurses außerhalb der Spanne.',
        'anom_context_e3': 'Auf einen Impuls folgten ein Rücksetzer und eine erneute Reaktion nahe einem Niveau: Die Stabilität der aktuellen Struktur wird geprüft.',
        'anom_context_e4': 'Die Bereichsgrenze wurde überwunden, gefolgt von einer Rückkehr in den Bereich: Der Markt prüft, ob dies Liquiditätssammlung statt eines echten Ausbruchs war.',
        'anom_price_observed': 'Kurs im Moment der Beobachtung:',
        'anom_attention': 'Worauf im Chart zu achten ist',
        'anom_check_level': 'Die Reaktion des Kurses auf das nächste Niveau oder die Bereichsgrenze.',
        'anom_check_close': 'Ob die nächste Kerze die Reaktion bestätigt, statt sie sofort zu entkräften.',
        'anom_check_absorption': 'Ob der Impuls absorbiert wird und der Kurs in den vorherigen Bereich zurückkehrt.',
        'anom_watch_short': 'Prüfen Sie im Chart die Reaktion nahe dem Niveau und den Schluss der nächsten Kerze.',
        'anom_disclaimer_full': 'Es wird keine Richtung prognostiziert. Dies ist eine Beobachtung der sich verändernden Marktaktivität, kein Einstiegssignal — die Entscheidung bleibt Ihrer Strategie überlassen.',
        'anom_disclaimer_short': 'Es wird keine Richtung prognostiziert. Dies ist eine Beobachtung, kein Einstiegssignal.',
    },
    'es': {
        'personal_observation': 'Tu observación de la cuenta',
        'anom_title_short': 'VSV · Anomalía de volumen',
        'anom_title_full': 'Anomalías de volumen · Contexto de mercado',
        'anom_context_default': 'Se detectó una secuencia inusual de volumen y reacción del precio: el equilibrio de la actividad del mercado puede estar cambiando.',
        'anom_context_e1': 'Una secuencia de clímax y absorción: el mercado comprueba si el equilibrio de la iniciativa cambia cerca de un nivel.',
        'anom_context_e2': 'Tras la compresión se detectó un impulso respaldado por volumen: el mercado comprueba la aceptación del precio fuera del rango.',
        'anom_context_e3': 'Después del impulso se formó un retroceso y otra reacción cerca de un nivel: se está comprobando la solidez de la estructura actual.',
        'anom_context_e4': 'El límite del rango fue perforado y el precio volvió al interior: el mercado comprueba si fue una captura de liquidez y no una ruptura real.',
        'anom_price_observed': 'Precio en el momento de la observación:',
        'anom_attention': 'Qué comprobar en el gráfico',
        'anom_check_level': 'La reacción del precio al nivel más cercano o al límite del rango.',
        'anom_check_close': 'Si la siguiente vela confirma la reacción en lugar de invalidarla de inmediato.',
        'anom_check_absorption': 'Si el impulso es absorbido y el precio vuelve al rango anterior.',
        'anom_watch_short': 'En el gráfico, comprueba la reacción cerca del nivel y el cierre de la siguiente vela.',
        'anom_disclaimer_full': 'No se pronostica una dirección. Es una observación del cambio en la actividad del mercado, no una señal de entrada: la decisión corresponde a tu estrategia.',
        'anom_disclaimer_short': 'No se pronostica una dirección. Es una observación, no una señal de entrada.',
    },
}

# Ймовірний напрямок уже визначений готовим сценарієм E1–E4 у `anom_pick`.
# Окремі статичні фрази дозволяють перекладати повідомлення цілком, не чіпаючи
# LONG/SHORT, ціну, символ монети або HTML-розмітку.
_ANOMALY_DIRECTION_EXTRA = {
    'uk': {
        'anom_direction_label': 'Ймовірний напрямок:',
        'anom_up': 'угору', 'anom_down': 'униз',
        'anom_reason_e1_long': 'Поглинання тиску продавців може передати ініціативу покупцям. Підтвердження — утримання зони реакції й закриття наступної свічки вище неї.',
        'anom_reason_e1_short': 'Поглинання купівель може передати ініціативу продавцям. Підтвердження — утримання зони реакції й закриття наступної свічки нижче неї.',
        'anom_reason_e2_long': 'Імпульс на підвищеному обсязі може означати прийняття ціни вище діапазону. Підтвердження — закриття над межею без швидкого повернення всередину.',
        'anom_reason_e2_short': 'Імпульс на підвищеному обсязі може означати прийняття ціни нижче діапазону. Підтвердження — закриття під межею без швидкого повернення всередину.',
        'anom_reason_e3_long': 'Відкат може бути перевіркою підтримки перед продовженням угору. Підтвердження — утримання рівня й поновлення купівель.',
        'anom_reason_e3_short': 'Відкат може бути перевіркою опору перед продовженням униз. Підтвердження — утримання рівня й поновлення продажів.',
        'anom_reason_e4_long': 'Прокол нижньої межі з поверненням може означати збір ліквідності перед рухом угору. Підтвердження — закриття назад над межею.',
        'anom_reason_e4_short': 'Прокол верхньої межі з поверненням може означати збір ліквідності перед рухом униз. Підтвердження — закриття назад під межею.',
        'anom_reason_default_long': 'Сукупність реакцій обсягу вказує на можливу перевагу покупців. Підтвердження — утримання зони реакції й подальше закриття вище неї.',
        'anom_reason_default_short': 'Сукупність реакцій обсягу вказує на можливу перевагу продавців. Підтвердження — утримання зони реакції й подальше закриття нижче неї.',
        'anom_disclaimer_full': 'Ймовірний сценарій, а не гарантія чи сигнал до входу — використовуйте його лише після підтвердження на графіку.',
        'anom_disclaimer_short': 'Ймовірний сценарій, а не гарантія чи сигнал до входу.',
    },
    'en': {
        'anom_direction_label': 'Probable direction:',
        'anom_up': 'upward', 'anom_down': 'downward',
        'anom_reason_e1_long': 'Absorption of selling pressure may hand the initiative to buyers. Confirmation requires the reaction zone to hold and the next candle to close above it.',
        'anom_reason_e1_short': 'Absorption of buying pressure may hand the initiative to sellers. Confirmation requires the reaction zone to hold and the next candle to close below it.',
        'anom_reason_e2_long': 'A high-volume impulse may indicate price acceptance above the range. Confirmation is a close above the boundary without a quick return inside.',
        'anom_reason_e2_short': 'A high-volume impulse may indicate price acceptance below the range. Confirmation is a close below the boundary without a quick return inside.',
        'anom_reason_e3_long': 'The pullback may be a support test before further upside. Confirmation requires the level to hold and buying to resume.',
        'anom_reason_e3_short': 'The pullback may be a resistance test before further downside. Confirmation requires the level to hold and selling to resume.',
        'anom_reason_e4_long': 'A sweep below the range followed by a return may indicate liquidity collection before an upward move. Confirmation is a close back above the boundary.',
        'anom_reason_e4_short': 'A sweep above the range followed by a return may indicate liquidity collection before a downward move. Confirmation is a close back below the boundary.',
        'anom_reason_default_long': 'The combination of volume reactions suggests that buyers may be gaining control. Confirmation requires the reaction zone to hold and price to close above it.',
        'anom_reason_default_short': 'The combination of volume reactions suggests that sellers may be gaining control. Confirmation requires the reaction zone to hold and price to close below it.',
        'anom_disclaimer_full': 'This is a probable scenario, not a guarantee or an entry signal — use it only after confirmation on the chart.',
        'anom_disclaimer_short': 'This is a probable scenario, not a guarantee or an entry signal.',
    },
    'ru': {
        'anom_direction_label': 'Вероятное направление:',
        'anom_up': 'вверх', 'anom_down': 'вниз',
        'anom_reason_e1_long': 'Поглощение давления продавцов может передать инициативу покупателям. Подтверждение — удержание зоны реакции и закрытие следующей свечи выше неё.',
        'anom_reason_e1_short': 'Поглощение покупок может передать инициативу продавцам. Подтверждение — удержание зоны реакции и закрытие следующей свечи ниже неё.',
        'anom_reason_e2_long': 'Импульс на повышенном объёме может означать принятие цены выше диапазона. Подтверждение — закрытие над границей без быстрого возврата внутрь.',
        'anom_reason_e2_short': 'Импульс на повышенном объёме может означать принятие цены ниже диапазона. Подтверждение — закрытие под границей без быстрого возврата внутрь.',
        'anom_reason_e3_long': 'Откат может быть проверкой поддержки перед продолжением роста. Подтверждение — удержание уровня и возобновление покупок.',
        'anom_reason_e3_short': 'Откат может быть проверкой сопротивления перед продолжением снижения. Подтверждение — удержание уровня и возобновление продаж.',
        'anom_reason_e4_long': 'Прокол нижней границы с возвратом может означать сбор ликвидности перед движением вверх. Подтверждение — закрытие обратно над границей.',
        'anom_reason_e4_short': 'Прокол верхней границы с возвратом может означать сбор ликвидности перед движением вниз. Подтверждение — закрытие обратно под границей.',
        'anom_reason_default_long': 'Совокупность реакций объёма указывает на возможное преимущество покупателей. Подтверждение — удержание зоны реакции и дальнейшее закрытие выше неё.',
        'anom_reason_default_short': 'Совокупность реакций объёма указывает на возможное преимущество продавцов. Подтверждение — удержание зоны реакции и дальнейшее закрытие ниже неё.',
        'anom_disclaimer_full': 'Это вероятный сценарий, а не гарантия или сигнал для входа — используйте его только после подтверждения на графике.',
        'anom_disclaimer_short': 'Это вероятный сценарий, а не гарантия или сигнал для входа.',
    },
    'pl': {
        'anom_direction_label': 'Prawdopodobny kierunek:',
        'anom_up': 'w górę', 'anom_down': 'w dół',
        'anom_reason_e1_long': 'Absorpcja presji sprzedaży może przekazać inicjatywę kupującym. Potwierdzeniem jest utrzymanie strefy reakcji i zamknięcie następnej świecy powyżej niej.',
        'anom_reason_e1_short': 'Absorpcja zakupów może przekazać inicjatywę sprzedającym. Potwierdzeniem jest utrzymanie strefy reakcji i zamknięcie następnej świecy poniżej niej.',
        'anom_reason_e2_long': 'Impuls na zwiększonym wolumenie może oznaczać akceptację ceny powyżej zakresu. Potwierdzeniem jest zamknięcie nad granicą bez szybkiego powrotu do środka.',
        'anom_reason_e2_short': 'Impuls na zwiększonym wolumenie może oznaczać akceptację ceny poniżej zakresu. Potwierdzeniem jest zamknięcie pod granicą bez szybkiego powrotu do środka.',
        'anom_reason_e3_long': 'Korekta może być testem wsparcia przed dalszym wzrostem. Potwierdzeniem jest utrzymanie poziomu i wznowienie zakupów.',
        'anom_reason_e3_short': 'Korekta może być testem oporu przed dalszym spadkiem. Potwierdzeniem jest utrzymanie poziomu i wznowienie sprzedaży.',
        'anom_reason_e4_long': 'Wybicie poniżej zakresu i powrót mogą oznaczać zebranie płynności przed ruchem w górę. Potwierdzeniem jest zamknięcie z powrotem nad granicą.',
        'anom_reason_e4_short': 'Wybicie powyżej zakresu i powrót mogą oznaczać zebranie płynności przed ruchem w dół. Potwierdzeniem jest zamknięcie z powrotem pod granicą.',
        'anom_reason_default_long': 'Układ reakcji wolumenu wskazuje na możliwą przewagę kupujących. Potwierdzeniem jest utrzymanie strefy reakcji i kolejne zamknięcie powyżej niej.',
        'anom_reason_default_short': 'Układ reakcji wolumenu wskazuje na możliwą przewagę sprzedających. Potwierdzeniem jest utrzymanie strefy reakcji i kolejne zamknięcie poniżej niej.',
        'anom_disclaimer_full': 'To prawdopodobny scenariusz, a nie gwarancja ani sygnał wejścia — wykorzystaj go dopiero po potwierdzeniu na wykresie.',
        'anom_disclaimer_short': 'To prawdopodobny scenariusz, a nie gwarancja ani sygnał wejścia.',
    },
    'de': {
        'anom_direction_label': 'Wahrscheinliche Richtung:',
        'anom_up': 'aufwärts', 'anom_down': 'abwärts',
        'anom_reason_e1_long': 'Die Absorption des Verkaufsdrucks kann die Initiative an Käufer übergeben. Bestätigt wird dies durch das Halten der Reaktionszone und einen Schluss der nächsten Kerze darüber.',
        'anom_reason_e1_short': 'Die Absorption der Käufe kann die Initiative an Verkäufer übergeben. Bestätigt wird dies durch das Halten der Reaktionszone und einen Schluss der nächsten Kerze darunter.',
        'anom_reason_e2_long': 'Ein Impuls mit erhöhtem Volumen kann auf eine Akzeptanz des Kurses oberhalb der Spanne hindeuten. Bestätigung ist ein Schluss über der Grenze ohne schnelle Rückkehr in die Spanne.',
        'anom_reason_e2_short': 'Ein Impuls mit erhöhtem Volumen kann auf eine Akzeptanz des Kurses unterhalb der Spanne hindeuten. Bestätigung ist ein Schluss unter der Grenze ohne schnelle Rückkehr in die Spanne.',
        'anom_reason_e3_long': 'Der Rücksetzer kann eine Unterstützungsprüfung vor einer weiteren Aufwärtsbewegung sein. Bestätigung sind gehaltenes Niveau und erneute Käufe.',
        'anom_reason_e3_short': 'Der Rücksetzer kann eine Widerstandsprüfung vor einer weiteren Abwärtsbewegung sein. Bestätigung sind gehaltenes Niveau und erneute Verkäufe.',
        'anom_reason_e4_long': 'Das Unterschreiten der unteren Grenze mit anschließender Rückkehr kann auf Liquiditätssammlung vor einer Aufwärtsbewegung hindeuten. Bestätigung ist ein Schluss zurück über der Grenze.',
        'anom_reason_e4_short': 'Das Überschreiten der oberen Grenze mit anschließender Rückkehr kann auf Liquiditätssammlung vor einer Abwärtsbewegung hindeuten. Bestätigung ist ein Schluss zurück unter der Grenze.',
        'anom_reason_default_long': 'Die Kombination der Volumenreaktionen deutet auf einen möglichen Vorteil der Käufer hin. Bestätigung sind das Halten der Reaktionszone und ein weiterer Schluss darüber.',
        'anom_reason_default_short': 'Die Kombination der Volumenreaktionen deutet auf einen möglichen Vorteil der Verkäufer hin. Bestätigung sind das Halten der Reaktionszone und ein weiterer Schluss darunter.',
        'anom_disclaimer_full': 'Dies ist ein wahrscheinliches Szenario, keine Garantie und kein Einstiegssignal — nutzen Sie es erst nach Bestätigung im Chart.',
        'anom_disclaimer_short': 'Dies ist ein wahrscheinliches Szenario, keine Garantie und kein Einstiegssignal.',
    },
    'es': {
        'anom_direction_label': 'Dirección probable:',
        'anom_up': 'al alza', 'anom_down': 'a la baja',
        'anom_reason_e1_long': 'La absorción de la presión vendedora puede transferir la iniciativa a los compradores. La confirmación exige mantener la zona de reacción y cerrar la siguiente vela por encima.',
        'anom_reason_e1_short': 'La absorción de las compras puede transferir la iniciativa a los vendedores. La confirmación exige mantener la zona de reacción y cerrar la siguiente vela por debajo.',
        'anom_reason_e2_long': 'Un impulso con mayor volumen puede indicar aceptación del precio por encima del rango. La confirmación es un cierre sobre el límite sin un retorno rápido al interior.',
        'anom_reason_e2_short': 'Un impulso con mayor volumen puede indicar aceptación del precio por debajo del rango. La confirmación es un cierre bajo el límite sin un retorno rápido al interior.',
        'anom_reason_e3_long': 'El retroceso puede ser una prueba de soporte antes de continuar al alza. La confirmación exige mantener el nivel y reanudar las compras.',
        'anom_reason_e3_short': 'El retroceso puede ser una prueba de resistencia antes de continuar a la baja. La confirmación exige mantener el nivel y reanudar las ventas.',
        'anom_reason_e4_long': 'La perforación del límite inferior con retorno puede indicar captura de liquidez antes de subir. La confirmación es un cierre de nuevo sobre el límite.',
        'anom_reason_e4_short': 'La perforación del límite superior con retorno puede indicar captura de liquidez antes de bajar. La confirmación es un cierre de nuevo bajo el límite.',
        'anom_reason_default_long': 'El conjunto de reacciones del volumen sugiere una posible ventaja de los compradores. La confirmación exige mantener la zona de reacción y cerrar después por encima.',
        'anom_reason_default_short': 'El conjunto de reacciones del volumen sugiere una posible ventaja de los vendedores. La confirmación exige mantener la zona de reacción y cerrar después por debajo.',
        'anom_disclaimer_full': 'Es un escenario probable, no una garantía ni una señal de entrada: utilízalo solo tras confirmarlo en el gráfico.',
        'anom_disclaimer_short': 'Es un escenario probable, no una garantía ni una señal de entrada.',
    },
}
for _translations in (_ANOMALY_EXTRA, _ANOMALY_DIRECTION_EXTRA):
    for _lang, _values in _translations.items():
        _EXTRA.setdefault(_lang, {}).update(_values)


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
        if user:
            return language_from_prefs(_load_prefs(user))
        # /start also works before a website profile exists: reuse the locale
        # captured from the first Telegram contact instead of reverting to uk.
        from web.i18n import normalize_language
        from web.auth import bot_contacts
        contact = bot_contacts().get(str(chat_id)) or {}
        return normalize_language(contact.get('language'), default='') if contact.get('language') else None
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
        # Спочатку довгі фрази: короткі «угору»/«униз» не повинні змінювати
        # текст усередині ще не перекладеного пояснення сценарію.
        for key, value in sorted(values.items(), key=lambda item: len(str(_EXTRA['uk'].get(item[0], ''))), reverse=True):
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
