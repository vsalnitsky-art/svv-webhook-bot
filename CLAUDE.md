# CLAUDE.md — пам'ять проєкту svv-webhook-bot

> Цей файл Claude Code читає АВТОМАТИЧНО на старті кожної сесії.
> Він — єдиний міст контексту між різними чатами (сесії НЕ діляться пам'яттю).
> Тримай його актуальним: коротко, фактологічно, без води.

## Що це за проєкт

Крипто-трейдинг бот (Bybit/ф'ючерси) зі Smart-Money-логікою. Працює як
веб-застосунок Flask + кілька фонових daemon-потоків. Відповіді користувачу —
**українською**.

## ДИЗАЙН UI — ЗАВЖДИ ПРОФЕСІЙНО (правило назавжди)

Будь-яка сторінка/блок/таблиця — тільки **професійний, охайний вигляд**: єдина
система відступів і типографіки, стримані кольори з акцентами (не строкато),
1px-бордери, вирівнювання в стовпчик, картки однакової висоти, без «стрибання»
елементів, uppercase-підписи для метрик, monospace для чисел. Стат-картки —
в один ряд (горизонтальний скрол на вузькому). НЕ аматорський підхід.

## Як запускати

- Точка входу: `main_bot.py` (для gunicorn: `gunicorn main_bot:app`).
- Flask-фабрика: `web/flask_app.py` → `create_app()`; об'єкт `app`.
- Локально dev: `python web/flask_app.py` → `0.0.0.0:5000`.
- Прод-порт береться з env `PORT` (дефолт 10000), хост через `RENDER_EXTERNAL_URL`.
- Перевірка синтаксису після правок: `python -m py_compile <файл>`.

## Архітектура (головне)

- **`web/flask_app.py`** — усі HTTP-маршрути та JSON API. Великий файл (~5600 рядків).
- **`detection/fuel_filter.py`** — ядро стратегії «FF» (liquidation-fuel + черга на
  вхід + двигун відкриття). Daemon-клас `FuelFilterDaemon`. Синглтон:
  `get_fuel_filter()`, ініт `init_fuel_filter(...)`. Стан читається через
  `ff.get_state()`.
- **`detection/trade_manager.py`** — відкриття/закриття позицій, перехоплення
  сигналів у чергу FF, формування Reason.
- **`detection/funding_monitor.py`** — сканер funding-ставок (окрема таблиця 💰).
- **`detection/smc_scanner.py` / `smc_analyzer.py`** — детекція CHoCH / CHoCH+BOS
  (TF 15m). ТІЛЬКИ свіжий CHoCH/CHoCH+BOS за напрямком увімкненої кнопки потрапляє
  в чергу FF.
- **`storage/db_operations.py`** — шар БД. `get_db()`, методи
  `db.get_setting(key, default)` / `db.set_setting(key, value)` (значення —
  JSON-серіалізовні).
- **`templates/smart_money.html`** — головна UI-сторінка стратегії FF.

## Ключові конвенції стратегії FF (не зламати!)

- **СЕАНСИ ММ (головна логіка черги).** Сеанс = зафіксований напрямок ММ BTC
  (LONG/SHORT), тримається в `_btc_verdict_dir`/`_btc_verdict_since`
  (`_update_btc_verdict`). Правила:
  - **WAIT (ML збалансований, |dir| ≤ 0.1) = ПАУЗА:** напрямок сеансу,
    таймер і черга ЗБЕРІГАЮТЬСЯ; `_btc_paused=True`; таймер продовжує лічити;
    банер показує «напрямок · ⏸ ПАУЗА»; двигун під час паузи НЕ відкриває.
  - **Той самий напрямок знову** (LONG→WAIT→LONG) = той самий сеанс.
  - **Протилежний фліп** (LONG↔SHORT) = НОВИЙ сеанс: скид таймера + **чистка
    всієї черги** (`_pending={}`) + скид `engine_attempts`. Це ЄДИНИЙ випадок,
    коли ММ чистить чергу (замість старої per-flip OP-4 чистки протилежних).
- **Черга `_pending` ПЕРСИСТЕНТНА, відновлюється ЗА СЕАНСОМ.** Зберігається в
  БД і відновлюється на старті. Стале не тягнеться, бо на першому тіку після
  старту, якщо live ML протилежний відновленому сеансу → чергу чистить фліп
  сеансу; якщо той самий напрямок/WAIT → монети валідні й чекають далі.
- Монета входить у чергу ЛИШЕ зі свіжого CHoCH/CHoCH+BOS за напрямком увімкненої
  кнопки (`intercept`). Показ у таблиці НЕ фільтрується кнопками (кнопки керують
  лише відкриттям) — інакше при WAIT черга «зникала».
- Паливо (ММ-напрямок) рахується для монет у черзі; на BTC START (сеанс тримає
  напрямок ≥ поріг, не пауза) двигун відкриває угоду, якщо паливо монети
  співпадає з напрямком сигналу.
- 9 операцій видалення з черги пронумеровані й керуються `_QUEUE_OPS_ALLOWED`
  (всі зараз `True`). OP-4 = чистка черги на фліпі сеансу.
- Банер ₿ BTCUSDT = напрямок СЕАНСУ, а джерело live-напрямку — **МММ-БАБЛО**
  (`_fuel_dir_smoothed('BTCUSDT')['status']`), ТОЧНО як комірка «МММ» монет:
  LONG / SHORT / None(`|dir| ≤ 0.1` = ⚖ рівновага «напрямку немає» → WAIT/пауза).
  Контраріанська liqmap-модель (за запитом). Сила банера = |fuel_dir|×100.
  Сеанс-механіка (пауза на WAIT, фліп на протилежний, START після
  `start_signal_minutes`) — без змін; `_btc_trend_dir()` більше не використовується.

## Черга-3 «🎯 Готовність» (стратегія на grade_setup)

Третя, ОКРЕМА черга/двигун (поряд із Чергою-1 і Чергою-2, усі незалежні тумблери).
Живиться тими самими CHoCH/CHoCH+BOS через `intercept()` (гілка `q3` → `_pending3`).

- **Тригер входу:** двигун `_engine_tick_readiness` відкриває монету у напрямку
  кнопки, коли її SMC-«готовність» `grade_setup(...)` **HOT АБО `score ≥
  queue3_open_min_score`** (дефолт **43**; 0 = лише HOT). Строгий HOT
  (score ≥ 70 + усі блоки, без вето) на реальному потоці майже не спрацьовує
  (спостережений max ~66), тож поріг дає стратегії реальні угоди. КАЛІБРУВАННЯ
  53→43: розворотні CHoCH+BOS системно занижуються блоком таймінгу CTR (свіжий
  розворот б'є проти CTR), і хороші угоди падали в «СЕРЕДНІЙ» (38–52) під 53. **БЕЗ ₿ START /
  сеансів ММ** — весь контекст ₿/CTR/зона/ліквідність уже ВСЕРЕДИНІ `grade_setup`
  (див. `detection/setup_grader.py`). Санітарні ворота: кнопки LONG/SHORT, дедуп,
  вже-в-угодах, ціна; `_open` тримає власну стелю виснаженості.
- **CTR більше НЕ ріже розвороти (2 ворота прибрано).** Було: (а) вето «CTR проти
  входу» в `grade_setup` капало бал ≤43; (б) `safeguard_ctr` у `_open` давав хард-
  відмову «CTR проти». Свіжий CHoCH+BOS за визначенням проти CTR → ці ворота
  різали КОЖЕН розворот. Тепер: `setup_ctr_mode` (дефолт **soft**) — CTR у грейді
  партіал (0.2) без вето; `queue3_ignore_ctr` (дефолт **True**) — Q3 пропускає
  CTR-перевірку safeguard (МММ/виснаженість лишаються). `_soft_safeguard(...,
  skip_ctr=)`, `_open(..., skip_ctr_safeguard=)`. Режими CTR: soft/normal/off.
- Читає готовий кеш `self._setup_cache[sym]` (той самий, що й колонка
  «Готовність»); `_pending3` додано в `targets` для `_refresh_setup_cache`.
- **Повне логування** кожного рішення (opened/hold/skipped + розклад блоків) →
  таблиця `sob_readiness_log` + рядок у Лог роботи бота (`activity_log`, source
  `Q3`). Тумблер `readiness_log_enabled` (дефолт ON). Анти-флуд: незмінний
  hold/skip пишеться не частіше ніж раз на `READINESS_LOG_MIN_GAP` (300с).
- Налаштування: `queue3_enabled` (дефолт OFF), `queue3_open_min_score` (43),
  `queue3_ignore_ctr` (True), `queue3_ttl_hours` (6 — протермінування, щоб монети
  не висіли годинами), `setup_ctr_mode` (soft), `readiness_log_enabled`.
  Q3-лог у `activity_log` несе пласкі `su_*` блоки
  (structure/poi/zone/liq/mm/timing/context) — щоб CSV-експорт був аналізовним.
- API: `GET /api/fuel-filter/readiness-log?limit&symbol&outcome`;
  `POST /api/fuel-filter/queue3/{delete,clear}`. UI: тумблер + таблиця «Черга-3»
  на `/smart-money`. get_state → `timers3`/`pending3_visible`/`queue3_enabled`.
- Тести: `test_readiness_strategy.py`.

## ⚡ Скальперська «Готовність» для funding-монет

Окремий SMC-грейд ЛИШЕ для монет таблиці «💰 Funding — МММ» (`_anomalies`) на
швидкому TF. `_compute_setup` параметризовано за TF (`base_tf`/`htf_tf`/
`write_exit`); скальп-варіант зве його з `base_tf='5m'`, `htf_tf='15m'`,
`write_exit=False` (не чіпає 1H-«Готовність виходу»). Окремий кеш
`_setup_scalp_cache` + `_refresh_setup_scalp_cache` (власний TTL/cap, тільки
funding-монети). Той самий `grade_setup` — інші свічки; на швидких TF
структура/зона шумніші → трактувати як ОКРЕМИЙ показник, не 1:1 з 1H.
- Налаштування: `funding_setup_scalp_on` (OFF), `funding_setup_tf` (5m),
  `funding_setup_htf` (15m), `funding_setup_ttl` (45с), `funding_setup_max_per_cycle` (8).
- get_state → funding-рядки несуть `setup_scalp`; UI — колонка «⚡ Скальп» у
  funding-таблиці + блок налаштувань у акордеоні. Інфосайт дзеркалить колонку.
- Навантаження мінімальне: klines кешуються (TTL), тротл cap/цикл; funding-
  підмножина мала (одиниці–десятки монет).
- TG на «хороший сигнал» по колонці Скальп: `scalp_tg_on`/`scalp_tg_min_score`
  (HOT або score≥поріг) / `scalp_tg_cooldown_min` / `scalp_tg_dir`. Метод
  `_scalp_setup_alert` (edge-trigger+кулдаун) у `_run_alerts` → топік 💰 funding.
- Панель `/api/fuel-filter/panel/<sym>` віддає `setup_scalp`; TradingView-оверлей
  (`tools/tradingview_mm_overlay.user.js`, v1.7.3) показує ⚡ Скальп-бейдж.

## 🎯 Шаровий конфлюенс (1..5) для funding-монет + консолідований TG

`_funding_layers(sym, a)` рахує 5 шарів У БІК напрямку монети: 1) МММ ≥ легкий
(сила ≥10) + тренд сили АСИМЕТРИЧНО за напрямком: **SHORT — НЕ слабшає (↑ або →;
гасне ЛИШЕ на ↓)**, бо на встановленому спаді сила часто на плато (→) і строге ↑
гасило шар майже завжди; **LONG — СТРОГО РОСТЕ (↑)**, як було раніше · 2) SCORE ≥
СЕРЕДНІЙ (≥40) · 3) Готовність(1H) ≥ СЕРЕДНІЙ (≥38) · 4) Скальп ≥ СЕРЕДНІЙ (≥38) ·
**5) ЦІНА у бік — свіжий рух (~15 хв) з 💰 Funding Rate Scanner: LONG=РОСТЕ (up),
SHORT=СПАДАЄ (down)**. Кожен шар несе `dir` (напрямок монети, КОЛИ засвічений) →
у колонці кольори за напрямком: **усі зелені (LONG) або всі червоні (SHORT)**.
Повертає {count, `base`(=усі 5), base4, layers[]}. get_state → funding-рядки несуть
`layers`; UI — колонка «🎯 Шари» (`ffLayersCell` фарбує крапки за `l.dir`).
- **5-й шар «Ціна» — ПОСТІЙНИЙ (світиться в колонці)**, живиться
  `self._funding_price` (кеш з `_get_funding_price_dirs()` → `FundingMonitor.
  get_price_dirs()`; оновлюється в тіку поряд з `_funding_trends`). Джерело —
  вікно `PRICE_WINDOW`=15хв, мертва зона `PRICE_DEADZONE`=0.10% (funding_monitor).
- **VOB (1m) = ОКРЕМИЙ ОДНОРАЗОВИЙ ТРИГЕР сигналу (не колонка-шар).**
  `_funding_vob(sym,d)` ПОСТІЙНО моніторить 1m-OB для КОЖНОЇ funding-монети (кеш
  8с) → `detect_volumized_obs(swing=5, ob_end_method='Wick', max_atr_mult=3.5,
  zone_count='Low')` → найновіший НЕ-breaker OB у бік. На НОВОМУ OB
  (`formation_time` ≠ `_vob_seen`) перевіряємо всі 5 шарів САМЕ ЗАРАЗ: якщо
  `base ≥ layer_tg_min` (cap 5) → сигнал; інакше чекаємо наступний OB.
- На НОВОМУ OB (інший formation_time) + кулдаун → ОДИН TG «🎯 Рекомендація бота»
  (`_layer_signal_alert`/`_send_layer_alert`, топік 💰 funding). Це ЗАМІНЯЄ старе
  «рекомендована ботом» повідомлення.
- Налаштування: `layer_tg_on` (OFF), `layer_tg_min` (скільки з 5, дефолт 5),
  `layer_tg_cooldown_min` (30). VOB-параметри — константи `_VOB_*` (1m/5/Wick/3.5/Low).
- Тести: `test_readiness_strategy.py` (funding_layers 5-шарів + ЦІНА-напрямок +
  VOB-alert + all-layers-gate).
- **Сортування таблиці «💰 Funding — МММ» — ЗА ШАРАМИ** (get_state): більше
  засвічених шарів вище; тайбрейк — довше тримається (`held_sec`).

## 🎯 Черга-3: авто-відкриття за «VOB + 5 шарів» (funding-монети)

Той самий моніторинг VOB у `_layer_signal_alert` живить ДВА незалежні виходи на
НОВОМУ Volumized OB (edge за formation_time): (1) TG-сигнал (`layer_tg_on`);
(2) авто-відкриття угоди `queue3_vob_open` (дефолт **ON**). Логіка —
`_vob_open_or_trail(sym,a,d,ob,lay,s,now)`:
- **Нова угода:** усі 5 шарів зійшлись + монети ще немає в угоді → `_open(...,
  opened_by='Q3-VOB(funding)', skip_ctr_safeguard=True)` за напрямком. **SL = межа
  блоку OB + буфер**: SHORT → `top*(1+buf)` (над верхом), LONG → `bottom*(1-buf)`
  (під низом); `buf = queue3_vob_sl_buffer_pct/100` (дефолт 0.10%). SL ставиться
  через `tm.update_manual_sl_tp(sym, manual_sl=, is_shadow=)`.
- **Повторний VOB** по вже відкритій ботом монеті → угоду НЕ відкриваємо, лише
  ПЕРЕСУВАЄМО SL на новий блок. Трекер `self._vob_trade` {sym→{side,sl,ftime,
  entry,mode}}; чиститься, коли монета зникла І позиції вже нема.
- Усе (open / sl_moved) пишеться в 🧾 Лог роботи бота (`activity_log`, source
  `Q3-VOB`) з міткою «монета з фандингу».
- Налаштування: `queue3_vob_open` (True), `queue3_vob_sl_buffer_pct` (0.10).
  UI — у гармошці **«⚙️ Налаштування (Черга-3)»** (`templates/smart_money.html`),
  куди перенесено ВСІ параметри Черги-3 (SCORE≥/TTL/ignore-CTR/CTR-mode/Лог +
  VOB-open/буфер). Заголовок колонки шарів — лише «🎯» (без слова «Шари»).
- Тести: `test_vob_open_*` у `test_readiness_strategy.py`.

## 💰 Funding Rate Scanner — колонка «Price» (dashboard)

Таблиця «💰 Funding Rate Scanner» (`templates/dashboard.html`, дані з
`FundingMonitor.get_watchlist()`) у колонці **Price** показує ЧІТКИЙ свіжий
напрямок ЦІНИ: `price_dir` (up/down/flat) + `price_chg_recent` (% за вікно
`PRICE_WINDOW`≈15хв) — **▲ росте (зелений) / ▼ спадає (червоний) / ▬ рівно**.
Це окремо від `price_change` (сумарно від старту стеження, тепер у підказці).
`_recent_price_move(rates)` рахує рух по останніх ~15 семплах (мертва зона
`PRICE_DEADZONE`=0.10%). Ту саму метрику віддає `get_price_dirs()` для 5-го шару.

## Лог Черги-3: рух у черзі + виснаженість

`_log_readiness(..., move_pct=, exhaustion=)` — у `sob_readiness_log` і в UI-лозі
несе **рух ціни від входу в чергу** (`move_pct`, + = у наш бік; `_pending3[sym]
['added_price']`) і **виснаженість** — щоб було видно, що монета вже відпрацювала,
поки стояла в черзі. У рядку логу: «… · рух +2.3% · вичерп 72% · …».

## Telegram-бот (`web/tg_bot.py`) — БЕЗПЕКА розсилки

Відповідь адміна юзеру НІКОЛИ не повинна ставати масовою розсилкою:
- Свайп-Reply на шапку «Повідомлення від користувача» → лише тому юзеру. Звʼязок
  `_reply_map` **персиститься в БД** (`tg_reply_map`, `_load_reply_map` на старті)
  → переживає рестарти. Якщо звʼязок усе одно втрачено — `chat_id` парситься з
  тексту шапки (`_chat_id_from_text`).
- Reply без звʼязку → підказка `/reply <chat_id>`, **не розсилка**.
- Масова розсилка ЛИШЕ за явним `/announce <текст>` (`_admin_broadcast`). Звичайне
  повідомлення адміна → підказка, ніколи не broadcast.
- Тести: `test_tg_reply.py`.

## БД-ключі (storage)

- `fuel_filter_settings` — налаштування FF (вкл. `queue3_enabled`,
  `readiness_log_enabled`).
- `fuel_filter_state` — JSON-блоб стану: timers, fuel_managed, anomalies(=funding),
  engine_attempts, fuel_ema, fuel_hyst, btc_verdict_dir/since, pending, pending2,
  **pending3**.
- `fuel_filter_scan_list` — дозволені для сканування символи.
- Таблиця `sob_readiness_log` — пер-рішення лог Черги-3 «Готовність». Префікс
  `sob_` → автоматично в аналізі Database Administration; входить у
  `_SERVICE_TABLES_TIME` → її ЧИСТИТЬ і ручна «🗄️ Службові», і DB-autoclean.

## Основні JSON API (read-only — для інфо-сайту)

- `GET /api/fuel-filter/state` → весь стан FF (`ff.get_state()`): черга, таймери,
  funding-таблиця, банер BTC, статуси. **Головне джерело для інфо-сайту.**
- `GET /api/stats`, `GET /api/trades`, `GET /api/signals`, `GET /api/sleepers`,
  `GET /api/orderblocks`, `GET /api/health`.
- `GET /api/scheduler/status`, `GET /api/events`.
- Сторінки: `/` (головна), `/smart-money` (FF), `/tickr`, `/trades`, `/settings`.

> Маршрути, що ЗМІНЮЮТЬ стан (POST: toggle, scan, delete-timer, force-open,
> settings, trade/close…) — інфо-сайту НЕ потрібні й не повинні викликатись.

## Git / деплой (важливо для кожної сесії)

- Робоча гілка: `claude/analyze-project-01Gopju9D7AHv4pgvetccBeB`. Розробляти й
  комітити сюди.
- **PUSH ЗАБЛОКОВАНИЙ:** git-relay повертає 403 (egress-політика середовища). Це
  НЕ мережевий збій — retry не допомагає, обходити заборонено.
- **ФОРМАТ ВІДДАЧІ ЗМІН (завжди!):** віддавати користувачу САМІ ЗМІНЕНІ ФАЙЛИ
  (не патч, не bundle) зі збереженням структури папок, запаковані в **ZIP**
  архів. Тобто `detection/trade_manager.py` лежить у zip за шляхом
  `detection/trade_manager.py`. Користувач розпаковує в корінь репозиторію й
  пушить локально.
- Не створювати PR без явного прохання.

## Окрема сесія для інфо-сайту

Інфо-сайт розробляється в ОКРЕМОМУ чаті. Бриф: див. `INFO_SITE.md`.
Принцип «в парі»: бот пише стан → інфо-сайт лише ЧИТАЄ його через JSON API.
Жодного спільного коду, жодних кнопок керування на інфо-сайті.
