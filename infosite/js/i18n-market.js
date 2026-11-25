/* Professional translations for live market tables and server-generated evidence. */
(function () {
  "use strict";
  var dictionaries = window.VSV_I18N_DICTIONARIES || (window.VSV_I18N_DICTIONARIES = {});
  var patterns = window.VSV_I18N_PATTERNS || (window.VSV_I18N_PATTERNS = {});
  var text = {
    en: {
      "🌀 натяг Пружини": "Spring bias", "⚠ order flow 6 год": "Order flow · 6h", "📈 рух за 24 год": "24h price context", "🤖 сетап VSV·AI": "VSV·AI setup",
      "Потенціал «Пружини»": "Spring bias", "Order flow · 6 год": "Order flow · 6h", "Сценарій VSV·AI": "VSV·AI setup", "Контекст руху · 24 год": "24h price context",
      "суперечливі реакції обсягу (обидва боки сильні)": "conflicting volume reactions — aggression is elevated on both sides",
      "конфлікт агресії: обидва боки активні, переваги немає": "conflicting aggression — both sides are active, with no clear edge",
      "покупці домінують": "buyers hold the initiative", "продавці домінують": "sellers hold the initiative",
      "покупці домінують (слабко)": "buyers hold a weak initiative", "продавці домінують (слабко)": "sellers hold a weak initiative",
      "ініціатива покупців": "buy-side initiative", "ініціатива продавців": "sell-side initiative",
      "ініціатива покупців · слабке підтвердження": "weak buy-side initiative", "ініціатива продавців · слабке підтвердження": "weak sell-side initiative",
      "паритет агресії": "aggression is balanced", "рівновага": "balanced order flow",
      "джерела врівноважують одне одного — ⚖ напрямок не визначено": "signals offset one another — no confirmed directional edge",
      "докази компенсують одне одного — ⚖ ринкової переваги немає": "signals offset one another — no confirmed market edge",
      "недостатньо даних (немає ні натягу Пружини, ні order flow)": "insufficient confirmation — neither Spring bias nor order flow is available",
      "недостатньо підтверджень: потрібен потенціал «Пружини» або order flow": "insufficient confirmation — Spring bias or order-flow evidence is required",
      "Напрямок перераховується постійно і може змінитися: хибний пробій, збір ліквідності, новий учасник чи рух BTC розвертають картину. Не 100% і не остаточний — див. «ⓘ Що це?».": "Directional bias updates with new market data. A false breakout, liquidity sweep, order-flow shift or BTC impulse can invalidate the current scenario. This is a probabilistic estimate, not a guarantee or an entry recommendation — see ‘ⓘ What is this?’.",
      "Оцінка напрямку оновлюється з новими ринковими даними. Хибний пробій, збір ліквідності, зміна order flow або імпульс BTC можуть скасувати поточний сценарій. Це ймовірнісна оцінка, а не гарантія чи рекомендація до входу; підтверджуйте власною структурою ринку та ризик-менеджментом.": "Directional bias updates with new market data. A false breakout, liquidity sweep, order-flow shift or BTC impulse can invalidate the current scenario. This is a probabilistic estimate, not a guarantee or an entry recommendation; confirm it with your own market-structure analysis and risk management.",
      "Час": "Time", "Монета": "Asset", "Сигнал": "Signal", "Книга": "Book", "Дія": "Action", "Деталі": "Details", "Останні дії": "Recent activity",
      "Відкриті угоди": "Open trades", "Закриті угоди": "Closed trades", "Відкритих угод немає": "No open trades", "Закритих угод немає": "No closed trades", "Подій ще немає": "No events yet",
      "Дій ще не було — угода зʼявиться на першому новому сигналі.": "No activity yet — a trade will appear on the next new signal."
    },
    ru: {
      "🌀 натяг Пружини": "Перевес «Пружины»", "⚠ order flow 6 год": "Order flow · 6 ч", "📈 рух за 24 год": "Контекст цены · 24 ч", "🤖 сетап VSV·AI": "Сетап VSV·AI",
      "Потенціал «Пружини»": "Перевес «Пружины»", "Order flow · 6 год": "Order flow · 6 ч", "Сценарій VSV·AI": "Сценарий VSV·AI", "Контекст руху · 24 год": "Контекст цены · 24 ч",
      "суперечливі реакції обсягу (обидва боки сильні)": "конфликт реакций объёма — агрессия повышена с обеих сторон",
      "конфлікт агресії: обидва боки активні, переваги немає": "конфликт агрессии — обе стороны активны, явного перевеса нет",
      "покупці домінують": "инициатива у покупателей", "продавці домінують": "инициатива у продавцов",
      "покупці домінують (слабко)": "слабая инициатива покупателей", "продавці домінують (слабко)": "слабая инициатива продавцов",
      "ініціатива покупців": "инициатива покупателей", "ініціатива продавців": "инициатива продавцов",
      "ініціатива покупців · слабке підтвердження": "слабая инициатива покупателей", "ініціатива продавців · слабке підтвердження": "слабая инициатива продавцов",
      "паритет агресії": "паритет агрессии", "рівновага": "баланс order flow",
      "джерела врівноважують одне одного — ⚖ напрямок не визначено": "сигналы компенсируют друг друга — подтверждённого направленного перевеса нет",
      "докази компенсують одне одного — ⚖ ринкової переваги немає": "сигналы компенсируют друг друга — подтверждённого рыночного перевеса нет",
      "недостатньо даних (немає ні натягу Пружини, ні order flow)": "недостаточно подтверждений — нет данных ни о перевесе «Пружины», ни об order flow",
      "недостатньо підтверджень: потрібен потенціал «Пружини» або order flow": "недостаточно подтверждений — нужен перевес «Пружины» или подтверждение order flow",
      "Напрямок перераховується постійно і може змінитися: хибний пробій, збір ліквідності, новий учасник чи рух BTC розвертають картину. Не 100% і не остаточний — див. «ⓘ Що це?».": "Направленный перевес обновляется с поступлением рыночных данных. Ложный пробой, сбор ликвидности, смена order flow или импульс BTC могут отменить текущий сценарий. Это вероятностная оценка, а не гарантия или рекомендация к входу — см. «ⓘ Что это?».",
      "Оцінка напрямку оновлюється з новими ринковими даними. Хибний пробій, збір ліквідності, зміна order flow або імпульс BTC можуть скасувати поточний сценарій. Це ймовірнісна оцінка, а не гарантія чи рекомендація до входу; підтверджуйте власною структурою ринку та ризик-менеджментом.": "Направленный перевес обновляется с поступлением рыночных данных. Ложный пробой, сбор ликвидности, смена order flow или импульс BTC могут отменить текущий сценарий. Это вероятностная оценка, а не гарантия или рекомендация к входу; подтверждайте её собственным анализом структуры рынка и риск-менеджментом.",
      "Час": "Время", "Монета": "Актив", "Сигнал": "Сигнал", "Книга": "Книга", "Дія": "Действие", "Деталі": "Детали", "Останні дії": "Последние действия",
      "Відкриті угоди": "Открытые сделки", "Закриті угоди": "Закрытые сделки", "Відкритих угод немає": "Нет открытых сделок", "Закритих угод немає": "Нет закрытых сделок", "Подій ще немає": "Событий пока нет",
      "Дій ще не було — угода зʼявиться на першому новому сигналі.": "Действий пока не было — сделка появится при следующем новом сигнале."
    },
    pl: {
      "🌀 натяг Пружини": "Przewaga Spring", "⚠ order flow 6 год": "Order flow · 6 godz.", "📈 рух за 24 год": "Kontekst ceny · 24 godz.", "🤖 сетап VSV·AI": "Setup VSV·AI",
      "Потенціал «Пружини»": "Przewaga Spring", "Order flow · 6 год": "Order flow · 6 godz.", "Сценарій VSV·AI": "Scenariusz VSV·AI", "Контекст руху · 24 год": "Kontekst ceny · 24 godz.",
      "суперечливі реакції обсягу (обидва боки сильні)": "sprzeczne reakcje wolumenu — agresja jest podwyższona po obu stronach",
      "конфлікт агресії: обидва боки активні, переваги немає": "konflikt agresji — obie strony są aktywne, bez wyraźnej przewagi",
      "покупці домінують": "inicjatywa kupujących", "продавці домінують": "inicjatywa sprzedających",
      "покупці домінують (слабко)": "słaba inicjatywa kupujących", "продавці домінують (слабко)": "słaba inicjatywa sprzedających",
      "ініціатива покупців": "inicjatywa kupujących", "ініціатива продавців": "inicjatywa sprzedających",
      "ініціатива покупців · слабке підтвердження": "słaba inicjatywa kupujących", "ініціатива продавців · слабке підтвердження": "słaba inicjatywa sprzedających",
      "паритет агресії": "równowaga agresji", "рівновага": "równowaga order flow",
      "джерела врівноважують одне одного — ⚖ напрямок не визначено": "sygnały równoważą się — brak potwierdzonej przewagi kierunkowej",
      "докази компенсують одне одного — ⚖ ринкової переваги немає": "sygnały równoważą się — brak potwierdzonej przewagi rynkowej",
      "недостатньо даних (немає ні натягу Пружини, ні order flow)": "niewystarczające potwierdzenie — brak przewagi Spring i danych order flow",
      "недостатньо підтверджень: потрібен потенціал «Пружини» або order flow": "niewystarczające potwierdzenie — wymagana przewaga Spring lub sygnał order flow",
      "Напрямок перераховується постійно і може змінитися: хибний пробій, збір ліквідності, новий учасник чи рух BTC розвертають картину. Не 100% і не остаточний — див. «ⓘ Що це?».": "Przewaga kierunkowa jest aktualizowana wraz z nowymi danymi. Fałszywe wybicie, zbieranie płynności, zmiana order flow lub impuls BTC mogą unieważnić scenariusz. To ocena probabilistyczna, nie gwarancja ani rekomendacja wejścia — zob. „ⓘ Co to jest?”.",
      "Оцінка напрямку оновлюється з новими ринковими даними. Хибний пробій, збір ліквідності, зміна order flow або імпульс BTC можуть скасувати поточний сценарій. Це ймовірнісна оцінка, а не гарантія чи рекомендація до входу; підтверджуйте власною структурою ринку та ризик-менеджментом.": "Przewaga kierunkowa jest aktualizowana wraz z nowymi danymi. Fałszywe wybicie, zbieranie płynności, zmiana order flow lub impuls BTC mogą unieważnić scenariusz. To ocena probabilistyczna, nie gwarancja ani rekomendacja wejścia; potwierdzaj ją własną analizą struktury rynku i zarządzaniem ryzykiem.",
      "Час": "Czas", "Монета": "Aktywo", "Сигнал": "Sygnał", "Книга": "Księga", "Дія": "Akcja", "Деталі": "Szczegóły", "Останні дії": "Ostatnia aktywność",
      "Відкриті угоди": "Otwarte transakcje", "Закриті угоди": "Zamknięte transakcje", "Відкритих угод немає": "Brak otwartych transakcji", "Закритих угод немає": "Brak zamkniętych transakcji", "Подій ще немає": "Brak zdarzeń",
      "Дій ще не було — угода зʼявиться на першому новому сигналі.": "Brak aktywności — transakcja pojawi się przy kolejnym nowym sygnale."
    },
    de: {
      "🌀 натяг Пружини": "Spring-Tendenz", "⚠ order flow 6 год": "Order Flow · 6 Std.", "📈 рух за 24 год": "Kurskontext · 24 Std.", "🤖 сетап VSV·AI": "VSV·AI-Setup",
      "Потенціал «Пружини»": "Spring-Tendenz", "Order flow · 6 год": "Order Flow · 6 Std.", "Сценарій VSV·AI": "VSV·AI-Szenario", "Контекст руху · 24 год": "Kurskontext · 24 Std.",
      "суперечливі реакції обсягу (обидва боки сильні)": "widersprüchliche Volumenreaktionen — erhöhte Aggression auf beiden Seiten",
      "конфлікт агресії: обидва боки активні, переваги немає": "widersprüchliche Aggression — beide Seiten sind aktiv, ohne klaren Vorteil",
      "покупці домінують": "Käufer haben die Initiative", "продавці домінують": "Verkäufer haben die Initiative",
      "покупці домінують (слабко)": "schwache Käuferinitiative", "продавці домінують (слабко)": "schwache Verkäuferinitiative",
      "ініціатива покупців": "Käuferinitiative", "ініціатива продавців": "Verkäuferinitiative",
      "ініціатива покупців · слабке підтвердження": "schwache Käuferinitiative", "ініціатива продавців · слабке підтвердження": "schwache Verkäuferinitiative",
      "паритет агресії": "ausgeglichene Aggression", "рівновага": "ausgeglichener Order Flow",
      "джерела врівноважують одне одного — ⚖ напрямок не визначено": "Signale heben sich gegenseitig auf — kein bestätigter Richtungsvorteil",
      "докази компенсують одне одного — ⚖ ринкової переваги немає": "Signale heben sich gegenseitig auf — kein bestätigter Marktvorteil",
      "недостатньо даних (немає ні натягу Пружини, ні order flow)": "unzureichende Bestätigung — weder Spring-Tendenz noch Order Flow verfügbar",
      "недостатньо підтверджень: потрібен потенціал «Пружини» або order flow": "unzureichende Bestätigung — Spring-Tendenz oder Order-Flow-Signale erforderlich",
      "Напрямок перераховується постійно і може змінитися: хибний пробій, збір ліквідності, новий учасник чи рух BTC розвертають картину. Не 100% і не остаточний — див. «ⓘ Що це?».": "Die Richtungstendenz wird mit neuen Marktdaten aktualisiert. Fehlausbruch, Liquiditäts-Sweep, Order-Flow-Wechsel oder BTC-Impuls können das Szenario entkräften. Dies ist eine Wahrscheinlichkeitseinschätzung, keine Garantie oder Einstiegsempfehlung — siehe „ⓘ Was ist das?“.",
      "Оцінка напрямку оновлюється з новими ринковими даними. Хибний пробій, збір ліквідності, зміна order flow або імпульс BTC можуть скасувати поточний сценарій. Це ймовірнісна оцінка, а не гарантія чи рекомендація до входу; підтверджуйте власною структурою ринку та ризик-менеджментом.": "Die Richtungstendenz wird mit neuen Marktdaten aktualisiert. Fehlausbruch, Liquiditäts-Sweep, Order-Flow-Wechsel oder BTC-Impuls können das Szenario entkräften. Dies ist eine Wahrscheinlichkeitseinschätzung, keine Garantie oder Einstiegsempfehlung; bestätigen Sie sie mit eigener Marktstrukturanalyse und Risikomanagement.",
      "Час": "Zeit", "Монета": "Asset", "Сигнал": "Signal", "Книга": "Buch", "Дія": "Aktion", "Деталі": "Details", "Останні дії": "Letzte Aktivität",
      "Відкриті угоди": "Offene Trades", "Закриті угоди": "Geschlossene Trades", "Відкритих угод немає": "Keine offenen Trades", "Закритих угод немає": "Keine geschlossenen Trades", "Подій ще немає": "Noch keine Ereignisse",
      "Дій ще не було — угода зʼявиться на першому новому сигналі.": "Noch keine Aktivität — ein Trade erscheint beim nächsten neuen Signal."
    },
    es: {
      "🌀 натяг Пружини": "Sesgo de Spring", "⚠ order flow 6 год": "Order flow · 6 h", "📈 рух за 24 год": "Contexto del precio · 24 h", "🤖 сетап VSV·AI": "Configuración VSV·AI",
      "Потенціал «Пружини»": "Sesgo de Spring", "Order flow · 6 год": "Order flow · 6 h", "Сценарій VSV·AI": "Escenario VSV·AI", "Контекст руху · 24 год": "Contexto del precio · 24 h",
      "суперечливі реакції обсягу (обидва боки сильні)": "reacciones de volumen contradictorias: agresión elevada en ambos lados",
      "конфлікт агресії: обидва боки активні, переваги немає": "agresión contradictoria: ambos lados están activos, sin ventaja clara",
      "покупці домінують": "iniciativa compradora", "продавці домінують": "iniciativa vendedora",
      "покупці домінують (слабко)": "iniciativa compradora débil", "продавці домінують (слабко)": "iniciativa vendedora débil",
      "ініціатива покупців": "iniciativa compradora", "ініціатива продавців": "iniciativa vendedora",
      "ініціатива покупців · слабке підтвердження": "iniciativa compradora débil", "ініціатива продавців · слабке підтвердження": "iniciativa vendedora débil",
      "паритет агресії": "agresión equilibrada", "рівновага": "order flow equilibrado",
      "джерела врівноважують одне одного — ⚖ напрямок не визначено": "las señales se compensan: no hay ventaja direccional confirmada",
      "докази компенсують одне одного — ⚖ ринкової переваги немає": "las señales se compensan: no hay ventaja de mercado confirmada",
      "недостатньо даних (немає ні натягу Пружини, ні order flow)": "confirmación insuficiente: no hay sesgo de Spring ni datos de order flow",
      "недостатньо підтверджень: потрібен потенціал «Пружини» або order flow": "confirmación insuficiente: se requiere sesgo de Spring o evidencia de order flow",
      "Напрямок перераховується постійно і може змінитися: хибний пробій, збір ліквідності, новий учасник чи рух BTC розвертають картину. Не 100% і не остаточний — див. «ⓘ Що це?».": "El sesgo direccional se actualiza con nuevos datos. Una ruptura falsa, un barrido de liquidez, un cambio de order flow o un impulso de BTC pueden invalidar el escenario. Es una estimación probabilística, no una garantía ni una recomendación de entrada; consulta «ⓘ ¿Qué es?».",
      "Оцінка напрямку оновлюється з новими ринковими даними. Хибний пробій, збір ліквідності, зміна order flow або імпульс BTC можуть скасувати поточний сценарій. Це ймовірнісна оцінка, а не гарантія чи рекомендація до входу; підтверджуйте власною структурою ринку та ризик-менеджментом.": "El sesgo direccional se actualiza con nuevos datos. Una ruptura falsa, un barrido de liquidez, un cambio de order flow o un impulso de BTC pueden invalidar el escenario. Es una estimación probabilística, no una garantía ni una recomendación de entrada; confírmala con tu propio análisis de estructura y gestión de riesgo.",
      "Час": "Hora", "Монета": "Activo", "Сигнал": "Señal", "Книга": "Libro", "Дія": "Acción", "Деталі": "Detalles", "Останні дії": "Actividad reciente",
      "Відкриті угоди": "Operaciones abiertas", "Закриті угоди": "Operaciones cerradas", "Відкритих угод немає": "No hay operaciones abiertas", "Закритих угод немає": "No hay operaciones cerradas", "Подій ще немає": "Aún no hay eventos",
      "Дій ще не було — угода зʼявиться на першому новому сигналі.": "Aún no hay actividad: la operación aparecerá con la próxima señal nueva."
    }
  };
  var dynamic = {
    en: {
      "натяг вгору {}%": "LONG bias {}%", "натяг вниз {}%": "SHORT bias {}%", "оцінка LONG: {}%": "LONG bias {}%", "оцінка SHORT: {}%": "SHORT bias {}%",
      "імпульс підтверджує LONG": "impulse confirms LONG", "імпульс підтверджує SHORT": "impulse confirms SHORT", "підтверджений сценарій LONG": "confirmed LONG setup", "підтверджений сценарій SHORT": "confirmed SHORT setup",
      "+{}% за {} год": "+{}% over {}h", "-{}% за {} год": "−{}% over {}h", "зміна +{}%": "change +{}%", "зміна -{}%": "change −{}%",
      "{} з {} джерел за LONG": "{} of {} signals favour LONG", "{} з {} джерел за SHORT": "{} of {} signals favour SHORT", "{} з {} джерел за LONG, {} проти": "{} of {} signals favour LONG; {} oppose", "{} з {} джерел за SHORT, {} проти": "{} of {} signals favour SHORT; {} oppose",
      "{} з {} джерел підтверджують LONG": "{} of {} signals confirm LONG", "{} з {} джерел підтверджують SHORT": "{} of {} signals confirm SHORT", "{} з {} джерел підтверджують LONG, {} спростовують": "{} of {} signals confirm LONG; {} contradict", "{} з {} джерел підтверджують SHORT, {} спростовують": "{} of {} signals confirm SHORT; {} contradict"
    },
    ru: {
      "натяг вгору {}%": "перевес LONG {}%", "натяг вниз {}%": "перевес SHORT {}%", "оцінка LONG: {}%": "перевес LONG {}%", "оцінка SHORT: {}%": "перевес SHORT {}%",
      "імпульс підтверджує LONG": "импульс подтверждает LONG", "імпульс підтверджує SHORT": "импульс подтверждает SHORT", "підтверджений сценарій LONG": "подтверждённый сценарий LONG", "підтверджений сценарій SHORT": "подтверждённый сценарий SHORT",
      "+{}% за {} год": "+{}% за {} ч", "-{}% за {} год": "−{}% за {} ч", "зміна +{}%": "изменение +{}%", "зміна -{}%": "изменение −{}%",
      "{} з {} джерел за LONG": "{} из {} источников за LONG", "{} з {} джерел за SHORT": "{} из {} источников за SHORT", "{} з {} джерел за LONG, {} проти": "{} из {} источников за LONG; {} против", "{} з {} джерел за SHORT, {} проти": "{} из {} источников за SHORT; {} против",
      "{} з {} джерел підтверджують LONG": "{} из {} источников подтверждают LONG", "{} з {} джерел підтверджують SHORT": "{} из {} источников подтверждают SHORT", "{} з {} джерел підтверджують LONG, {} спростовують": "{} из {} источников подтверждают LONG; {} противоречат", "{} з {} джерел підтверджують SHORT, {} спростовують": "{} из {} источников подтверждают SHORT; {} противоречат"
    },
    pl: {
      "натяг вгору {}%": "przewaga LONG {}%", "натяг вниз {}%": "przewaga SHORT {}%", "оцінка LONG: {}%": "przewaga LONG {}%", "оцінка SHORT: {}%": "przewaga SHORT {}%",
      "імпульс підтверджує LONG": "impuls potwierdza LONG", "імпульс підтверджує SHORT": "impuls potwierdza SHORT", "підтверджений сценарій LONG": "potwierdzony scenariusz LONG", "підтверджений сценарій SHORT": "potwierdzony scenariusz SHORT",
      "+{}% за {} год": "+{}% w {} godz.", "-{}% за {} год": "−{}% w {} godz.", "зміна +{}%": "zmiana +{}%", "зміна -{}%": "zmiana −{}%",
      "{} з {} джерел за LONG": "{} z {} sygnałów za LONG", "{} з {} джерел за SHORT": "{} z {} sygnałów za SHORT", "{} з {} джерел за LONG, {} проти": "{} z {} sygnałów za LONG; {} przeciw", "{} з {} джерел за SHORT, {} проти": "{} z {} sygnałów za SHORT; {} przeciw",
      "{} з {} джерел підтверджують LONG": "{} z {} sygnałów potwierdza LONG", "{} з {} джерел підтверджують SHORT": "{} z {} sygnałów potwierdza SHORT", "{} з {} джерел підтверджують LONG, {} спростовують": "{} z {} sygnałów potwierdza LONG; {} przeczy", "{} з {} джерел підтверджують SHORT, {} спростовують": "{} z {} sygnałów potwierdza SHORT; {} przeczy"
    },
    de: {
      "натяг вгору {}%": "LONG-Tendenz {}%", "натяг вниз {}%": "SHORT-Tendenz {}%", "оцінка LONG: {}%": "LONG-Tendenz {}%", "оцінка SHORT: {}%": "SHORT-Tendenz {}%",
      "імпульс підтверджує LONG": "Impuls bestätigt LONG", "імпульс підтверджує SHORT": "Impuls bestätigt SHORT", "підтверджений сценарій LONG": "bestätigtes LONG-Szenario", "підтверджений сценарій SHORT": "bestätigtes SHORT-Szenario",
      "+{}% за {} год": "+{}% in {} Std.", "-{}% за {} год": "−{}% in {} Std.", "зміна +{}%": "Veränderung +{}%", "зміна -{}%": "Veränderung −{}%",
      "{} з {} джерел за LONG": "{} von {} Signalen für LONG", "{} з {} джерел за SHORT": "{} von {} Signalen für SHORT", "{} з {} джерел за LONG, {} проти": "{} von {} Signalen für LONG; {} dagegen", "{} з {} джерел за SHORT, {} проти": "{} von {} Signalen für SHORT; {} dagegen",
      "{} з {} джерел підтверджують LONG": "{} von {} Signalen bestätigen LONG", "{} з {} джерел підтверджують SHORT": "{} von {} Signalen bestätigen SHORT", "{} з {} джерел підтверджують LONG, {} спростовують": "{} von {} Signalen bestätigen LONG; {} widersprechen", "{} з {} джерел підтверджують SHORT, {} спростовують": "{} von {} Signalen bestätigen SHORT; {} widersprechen"
    },
    es: {
      "натяг вгору {}%": "sesgo LONG {}%", "натяг вниз {}%": "sesgo SHORT {}%", "оцінка LONG: {}%": "sesgo LONG {}%", "оцінка SHORT: {}%": "sesgo SHORT {}%",
      "імпульс підтверджує LONG": "el impulso confirma LONG", "імпульс підтверджує SHORT": "el impulso confirma SHORT", "підтверджений сценарій LONG": "escenario LONG confirmado", "підтверджений сценарій SHORT": "escenario SHORT confirmado",
      "+{}% за {} год": "+{}% en {} h", "-{}% за {} год": "−{}% en {} h", "зміна +{}%": "cambio +{}%", "зміна -{}%": "cambio −{}%",
      "{} з {} джерел за LONG": "{} de {} señales favorecen LONG", "{} з {} джерел за SHORT": "{} de {} señales favorecen SHORT", "{} з {} джерел за LONG, {} проти": "{} de {} señales favorecen LONG; {} en contra", "{} з {} джерел за SHORT, {} проти": "{} de {} señales favorecen SHORT; {} en contra",
      "{} з {} джерел підтверджують LONG": "{} de {} señales confirman LONG", "{} з {} джерел підтверджують SHORT": "{} de {} señales confirman SHORT", "{} з {} джерел підтверджують LONG, {} спростовують": "{} de {} señales confirman LONG; {} contradicen", "{} з {} джерел підтверджують SHORT, {} спростовують": "{} de {} señales confirman SHORT; {} contradicen"
    }
  };
  /* Фактичні title-рядки, зібрані з DOM усіх таблиць термінала. */
  var renderedText = {
    en: {
      "Енергія стиснення «Пружини»": "Spring compression energy",
      "при потраплянні в список": "on entering the list",
      "Measured move у будь-який бік": "Measured move in either direction",
      "діапазон 24 год": "24h range"
    },
    ru: {
      "Енергія стиснення «Пружини»": "Энергия сжатия «Пружины»",
      "при потраплянні в список": "при входе в список",
      "Measured move у будь-який бік": "Measured move в любом направлении",
      "діапазон 24 год": "диапазон 24 ч"
    },
    pl: {
      "Енергія стиснення «Пружини»": "Energia kompresji Spring",
      "при потраплянні в список": "przy wejściu na listę",
      "Measured move у будь-який бік": "Measured move w dowolnym kierunku",
      "діапазон 24 год": "zakres 24 godz."
    },
    de: {
      "Енергія стиснення «Пружини»": "Kompressionsenergie der Spring",
      "при потраплянні в список": "bei Aufnahme in die Liste",
      "Measured move у будь-який бік": "Measured Move in beide Richtungen",
      "діапазон 24 год": "24-Std.-Spanne"
    },
    es: {
      "Енергія стиснення «Пружини»": "Energía de compresión de Spring",
      "при потраплянні в список": "al entrar en la lista",
      "Measured move у будь-який бік": "Measured move en cualquier dirección",
      "діапазон 24 год": "rango de 24 h"
    }
  };
  var renderedDynamic = {
    en: { "Графік {} (USDT-перпетуал) у TradingView": "{} chart (USDT perpetual) on TradingView" },
    ru: { "Графік {} (USDT-перпетуал) у TradingView": "График {} (USDT-перпетуал) в TradingView" },
    pl: { "Графік {} (USDT-перпетуал) у TradingView": "Wykres {} (kontrakt perpetual USDT) w TradingView" },
    de: { "Графік {} (USDT-перпетуал) у TradingView": "{}-Chart (USDT-Perpetual) auf TradingView" },
    es: { "Графік {} (USDT-перпетуал) у TradingView": "Gráfico de {} (perpetuo USDT) en TradingView" }
  };
  Object.keys(text).forEach(function (lang) {
    Object.assign(text[lang], renderedText[lang]);
    Object.assign(dynamic[lang], renderedDynamic[lang]);
    dictionaries[lang] = Object.assign(dictionaries[lang] || {}, text[lang]);
    patterns[lang] = Object.assign(patterns[lang] || {}, dynamic[lang]);
  });
})();
