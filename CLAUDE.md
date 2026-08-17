# CLAUDE.md — пам'ять проєкту svv-webhook-bot

> Цей файл Claude Code читає АВТОМАТИЧНО на старті кожної сесії.
> Він — єдиний міст контексту між різними чатами (сесії НЕ діляться пам'яттю).
> Тримай його актуальним: коротко, фактологічно, без води.

## Що це за проєкт

Крипто-трейдинг бот (Bybit/ф'ючерси) зі Smart-Money-логікою. Працює як
веб-застосунок Flask + кілька фонових daemon-потоків. Відповіді користувачу —
**українською**.

## ПРОФЕСІЙНО — ЗАВЖДИ (правило назавжди, весь проєкт)

Проєкт **професійний, а не аматорський** — і в UI, і в логіці/даних.

**UI.** Будь-яка сторінка/блок/таблиця — тільки **професійний, охайний вигляд**:
єдина система відступів і типографіки, стримані кольори з акцентами (не строкато),
1px-бордери, вирівнювання в стовпчик, картки однакової висоти, без «стрибання»
елементів, uppercase-підписи для метрик, monospace для чисел. Стат-картки —
в один ряд (горизонтальний скрол на вузькому). НЕ аматорський підхід.

**Логіка/дані.** Жодних «на око достатньо» зрізань кутів. Та сама метрика,
показана у двох місцях, ЗАВЖДИ рахується з ОДНОГО джерела правди — інакше
з'являються протиріччя (див. кейс PD-зони нижче). Кожен показник рахується на
задекларованому TF/вікні; коротке ролінг-вікно НЕ видавати за структурний
діапазон. Фікси — з тестом, що фіксує поведінку.

**Приклад-урок (PD Premium/Discount).** Банер рішення показував «PD Premium 89%»,
тоді як бейдж графіка — «Discount 17% (1H)» для тієї ж монети. Причина:
`position_evaluator._score_pd_zone` рахував Premium/Discount з ролінг-вікна
20 барів 15m, а бейдж/фільтр — з дилінг-діапазону `pd_zone_timeframe` (1H).
Фікс: `evaluate_entry`/`_score_pd_zone` приймають `pd_pct`(+пороги) і, коли задано,
класифікують САМЕ його (єдине джерело — `scanner.get_pd_pct()` +
`get_pd_thresholds()`); ролінг-вікно лишилось лише як fallback для Health-шляху.
Тест: `test_pd_zone_source.py`.

## VOB-алерт: ВАЛІДНІСТЬ = СВІЖІСТЬ (не зламати!)

Volumized-OB-сигнал (`vob_alert_enabled`) фаєриться в `smc_scanner._scan` на
НОВОМУ OB (edge за `formation_time`). **Природа свінг-OB:** OB стає валідним лише
через ~`volumized_swing_length` барів після своєї свічки (свінг мусить
підтвердитись) — тому бокс на графіку стоїть «назад у часі», хоча сигнал свіжий.
Це НЕ баг.

- **Гейт свіжості** (`vob_alert_max_age_bars`, 0=авто=`swing_length+2`): фаєримо
  ЛИШЕ якщо OB щойно підтвердився (вік `_vob_age_bars` ≤ поріг). Застарілий/
  фантомний OB → **ПОВНИЙ no-op**: НЕ рухаємо базу (`_vob_alert_seen`), НЕ
  логуємо, НЕ реагуємо, ці дані НЕ використовуємо. Реакція — тільки на НОВИЙ
  свіжоутворений OB. (База рухається лише при свіжому OB — і на first-sight,
  тихо, без сигналу.)
- **Швидкість:** детекція лишається на ЖИВОМУ барі (`vol_data['klines']`) — НЕ
  переходити на закриті бари (це додало б до 1 TF затримки). Сигнал іде в ту ж
  мить, щойно OB став валідним.
- Далі — `_signal_allowed` (OB/PD/Forecast) → `on_signal` → черга/угода.
- Тест: `test_vob_freshness.py`.

## SMC-чарт (/smart-money): налаштування відображення

Гармошка **⚙️ Settings** керує чартом. Ключові фічі:

- **🏁 Swing High/Low (Display, `show_swing_hl_enabled`/`swing_hl_timeframe`,
  дефолт OFF/1H).** Дві горизонтальні лінії — ТОЧНО як у TradingView/LuxAlgo.
  Рівні = trailing swing-екстремуми ДИЛІНГ-ДІАПАЗОНУ (`SMCScanner.
  _swing_trailing_range` — ЄДИНЕ ДЖЕРЕЛО з PD-зоною, тому лінії й бейдж
  Premium/Discount ніколи не розходяться). Назви за swing-трендом
  (`_swing_hl_labels`): бичачий → «Weak High»(зверху)+«Strong Low»(знизу);
  ведмежий → «Strong High»+«Weak Low». НЕ брати «останній HH/LH та HL/LL півот»
  — це давало проміжні хибні рівні (кейс ATOM: 1.3937/1.3252 замість 1.566/1.33).
  Стан: `swing_levels`={high:{price,label},low:{price,label}}|None, `show_swing_hl`,
  `swing_hl_tf`. Фронт малює з `swing_levels` (динамічні підписи).
  Тест: `test_swing_hl_levels.py`.
- **🔮 Forecast min-strength (`forecast_min_strength`: off/moderate/strong).**
  Поріг СИЛИ (впевненості) прогнозу ДО увімкнених 1H/4H-фільтрів напрямку: збіг
  напрямку зі слабшою за поріг впевненістю → 'weak' (не-збіг, ріже сигнал).
  strong ≥66%, moderate ≥40%. У `_forecast_filter_allows` (smc_scanner).
- **📦 Volumized OB TF** тепер включає **5m** (ALLOWED_VOL_TFS).
- **Два OB-бокси на графіку одночасно.** `volObPrim` (Volumized OB, зелений/
  червоний суцільний) і `obMatchPrim` (Require OB Match, teal/orange пунктир) —
  ДВА незалежні `VolOBPrimitive`, малюються обидва, коли увімкнені відповідні
  тумблери. **Критично:** старт боксу прив'язувати через `_snapSecToChart(sec,
  ohlc)` — `timeToCoordinate()` повертає null для часу МІЖ барами, тож OB на TF,
  дрібнішому за TF графіка (Volumized 5m на графіку 15m), інакше НЕ малюється.

## ФІЛЬТРИ ВХОДУ — СПІЛЬНІ ВОРОТА для ВСІХ сигналів (не зламати!)

`SMCScanner._signal_allowed(symbol, side)` — ЄДИНІ ворота УСІХ фільтрів, кожен
НЕЗАЛЕЖНИЙ зі своїм тумблером. Викликаються з ОБОХ шляхів входу:
1. **CHoCH/BOS** → `_send_alert` (як і було);
2. **Volumized OB alert** (`vob_alert_enabled`) → перед `_tm.on_signal(...,
   opened_by='vob_alert')` у скан-тіку.

Ланцюг незалежних фільтрів (кожен за своїм тумблером):
- **OB** (`ob_filter_enabled`) · **PD** (`use_pd_zone_filter`) ·
- **Forecast НАПРЯМОК** (1H/4H match + AND/OR) — `_forecast_filter_allows`, ЛИШЕ
  напрямок (силу НЕ враховує — `min_conf=0`);
- **Forecast МІН.СИЛА** — ОКРЕМИЙ `forecast_strength_filter_enabled` +
  `forecast_min_strength` (`_forecast_strength_allows`: пропускає лише якщо є
  прогноз у бік сигналу з впевненістю ≥ поріг на 1H або 4H);
- **Decision-вердикт (осн. напрямок)** — `decision_filter_enabled`
  (`_decision_gate`/`_decision_filter_allows`): банер Decision Center має
  рекомендувати ТОЙ САМИЙ напрямок (LONG/SHORT), що й сигнал (достатньо бути в
  осн. напрямку); NEUTRAL/протилежний → блок. 10с-кеш; at_intake — блок лише на
  протилежний, строго при відкритті (`_open` re-gate). ЦЕ — про банер «LONG 80%
  СИЛЬНИЙ», НЕ про Forecast-бейджі 1H/4H (їх гейтить Мін.сила).
- **POC «краще LONG/SHORT»** — `poc_filter_enabled` (`_poc_filter_allows`:
  напрямковий — ціна нижче POC → LONG, вище → SHORT, на POC → нейтрально). Рахує
  через ТОЙ САМИЙ `detection.volume_profile.compute_poc`/`price_vs_poc`, що й бейдж
  чарту, з ТИМИ САМИМИ параметрами: `poc_filter_market`/`poc_filter_tf`/
  `poc_filter_window_days` (дефолти FUTURES/1H/3д) → вердикт фільтра 1:1 з чартом.
  `compute_poc` має власний TTL-кеш (не б'є біржу на кожен сигнал).
Тести: `test_independent_filters.py`.

**Прозорість + «чекати вердикт»:**
- `_signal_allowed` повертає **(allowed, reason, detail)**; `detail` — РОЗКЛАД
  ЗНАЧЕНЬ кожного увімкненого фільтра (✓/✗ + числа: прогноз 1H/4H сторона%,
  Мін.сила поріг, PD %, OB tf, POC). Пишеться у 🧾 Лог РАЗОМ із сигналом
  (`Свіжий сигнал … · <detail>`) — щоб було видно, що КОЖЕН фільтр відпрацював
  (`_forecast_pair` дає сирі значення прогнозу).
- **Forecast «чекати вердикт», а не викидати:** `_signal_allowed(at_intake=True)`
  на ІНТЕЙКУ по прогнозу блокує ЛИШЕ явну протилежність; nodata/neutral →
  пускаємо в чергу (сигнал ЧЕКАЄ). Строга перевірка (match + Мін.сила) — при
  ВІДКРИТТІ: `fuel_filter._open` re-gate кличе `_forecast_filter_allows`/
  `_forecast_strength_allows` (строго) для scanner-опенів (не funding) → якщо
  вердикту ще нема / проти → `return False` (монета лишається в черзі, ретрай).
- Telegram open/close повідомлення БІЛЬШЕ не несуть рядок `🏷 opened_by` (прибрано
  на прохання) — мітка лишається в таблицях/лозі/модалці, але не в TG.

**НІКОЛИ не відкривати сигнал в обхід `_signal_allowed`.** Був дефект: VOB-alert
кликав `on_signal` напряму й ОБХОДИВ усі фільтри — бот відкрив ASTERUSDT SHORT,
попри `Require Forecast 1H+4H match · AND · Сильний` з прогнозом LONG (фільтри
живуть лише у `_send_alert`, а VOB туди не заходив). Урок: КОЖЕН новий шлях
відкриття мусить проходити ці спільні ворота, інакше налаштування користувача —
фікція. Заблокований сигнал → `log_activity('rejected', reason)`.
Тест: `test_signal_gate_unified.py`.

## 🏷️ Мітки угод «Сигнал → Двигун» (не зламати!)

Кожна угода несе мітку ПОХОДЖЕННЯ у форматі **"<signal> → <engine>"**, щоб було
видно, ВІД ЯКОГО СИГНАЛУ вона пішла (раніше двигун черги штампував свою мітку, а
оригінальний сигнал губився — напр. ASTER SHORT від Volumized-OB показувався як
«Fuel Auto-Filter»).

- **Єдине джерело правди:** `detection/signal_labels.py` — `compose(signal,
  engine)`, `pretty_opened_by(raw)`, `signal_code_of(raw)`, мапи
  `SIGNAL_BADGES`/`ENGINE_BADGES`. **JS-дзеркало** ОБОВʼЯЗКОВЕ і має бути
  синхронним: у `templates/smart_money.html` та `infosite/app.js`
  (`prettyOpenedBy`, `signalIconHtml`, `SIGNAL_ICON_JS`).
- **Зберігаємо МАШИННІ коди** у `position['opened_by']` (напр. `vob_alert → Q4`),
  а бейджі (🟪 Volumized OB → 🎯 Черга-4) робимо лише ПРИ ПОКАЗІ. Так підрядки
  `funding`/`POC-сетап`/`external`/`choch` лишаються для логіки (substring-
  перевірки, `signal_code_of` для порівнянь).
- **Кожен двигун FF** тепер відкриває з `opened_by=_ob_compose(info.get('kind'),
  '<ENGINE>')` (Q1='EXH', Q2, Q3, Q4, Q3-VOB(funding)); `kind` = оригінальний
  сигнал із черги. Прямі опени (FF off) лишають сирий код сигналу (без двигуна).
- **Показ СКРІЗЬ** (вимога користувача — «скрізь означає скрізь»): іконка-
  «картинка» сигналу СТОЇТЬ ПОРЯД ІЗ НАЗВОЮ МОНЕТИ у КОЖНІЙ таблиці:
  • угоди — `histSymOpen`/`histSymCell`/архів/інфосайт;
  • черги Черга-1/2/3 — `ffQueueRowHTML` (з `t.kind`); Черга-4 — власний рендер
    (backend timers4 несе `kind`);
  • 🎯 POC-сетап — константний бейдж 🎯 (`signalIconHtml('POC-сетап')`);
  • 💰 Funding — МММ — уже має 💰 біля символу (джерело — funding-сканер).
  Плюс повна мітка у 🧾 Лозі (відкриття+закриття), у Telegram (рядок `🏷`), у
  модалці історії («🏷 Джерело»). DB-колонка `opened_by` розширена 40→80.
  ⚠️ Черги несуть СИРИЙ `kind` (сигнал), а не `opened_by` — двигуна ще нема
  (угоду ще не відкрито); `signalIconHtml` працює і на сирому коді.
- Коди сигналів: `choch`/`choch_bos` (smc_scanner on_signal), `vob_alert`
  (Volumized-OB alert), `vob` (funding-VOB), `poc` (POC-сетап → черга FF),
  `opp` (реверс), `external`, `manual`.
- **ФАНТОМНИЙ «CHoCH» — виправлено.** `poc_setup._route_to_ff` жорстко ставив
  `intercept(kind='choch')` → запис POC-сетапу в черзі показувався як «CHoCH»,
  хоча CHoCH-алерти вимкнені (`choch_alerts_enabled=False`; `_process_alerts`
  чесно виходить рано — реальний CHoCH НЕ фаєриться). Тепер `kind='poc'` →
  правильний бейдж 🎯 POC-сетап скрізь. `intercept._kind_lbl` теж знає 'poc'/'opp'.
  Урок: НЕ використовувати чужий код сигналу як дефолт при маршрутизації в чергу.
- Тести: `test_signal_labels.py`.

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
  вже-в-угодах, ціна.
- **Запобіжники Черги-3 (калібровано по activity_log).** Дані показали, що з 116
  «_open відхилив» більшість — жорсткі вето виснаженості (>80%) і слабкого МММ на
  сетапах, що вже пішли в наш бік (напр. NEARUSDT SHORT SCORE 53 ХОРОШИЙ, рух +1%,
  вичерп 81-89% → різалось). Тому:
  - `queue3_ignore_exhaustion` (дефолт **True**) — Q3/Q3-VOB пропускають ЖОРСТКЕ
    вето виснаженості у `_open` (`skip_exhaustion=`); виснаженість ЛИШАЄТЬСЯ м'яко
    в `grade_setup`/SCORE. Знімає і `max_exhaustion_pct`-гейт, і safeguard-вето.
  - `safeguard_mm_price_override` (дефолт **True**, tf `safeguard_mm_price_tf`=15m,
    ГЛОБАЛЬНО) — слабкий МММ (<`safeguard_mm_min`) НЕ ріже вхід, якщо `_candle_
    momentum(tf)` = напрямок угоди (ціна чітко йде в бік). Контраріанський МММ
    часто «мовчить» на чистому тренді.
  - `queue3_require_ob_match` (дефолт **True**) — `_ob_match_ok(sym,side,s)`: при
    увімкненому OB-фільтрі сканера НЕ відкривати ПРОТИ OB на `ob_filter_timeframe`
    (ріже лише ЯВНИЙ контр-OB; немає OB → дозволяємо, щоб не вбити Q3-VOB). Гейт у
    Q3 (`_engine_tick_readiness`, OB міг фліпнути за час очікування) і в Q3-VOB
    (`_vob_open_or_trail` — funding-опен взагалі не проходить сканерний `_send_alert`).
  - Лог Q3 «_open відхилив» тепер несе КОНКРЕТНУ причину з `_engine_skip`.
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

## 🎯 Черга-4: пер-шарові тумблери (економія сервера)

Кожен із 4 шарів Черги-4 має ВЛАСНИЙ тумблер: `queue4_new_mm_on`,
`queue4_old_mm_on`, `queue4_setup_on`, `queue4_runway_on` (усі дефолт True =
стара поведінка). **ВИМКНЕНИЙ шар:** (1) НЕ рахується — `_queue4_layers` не
смикає `_fuel_dir_smoothed`/`_fuel_dir_legacy`, не читає setup-кеш, а
`_refresh_setup_cache` НЕ додає pending4 (важкий grade_setup) → менше load;
(2) колонка ховається у таблиці (класи `q4c-new/old/setup/runway` на th+td,
`_ffApplyQ4Cols()`). `_queue4_layers` повертає `required` = к-сть УВІМКНЕНИХ;
двигун відкриває коли `base ≥ required` (усі увімкнені збіглись; required=0 →
не відкриває). `Новий МММ` і `Запас` беруть дані з того самого `fn`
(`_fuel_dir_smoothed`) — рахуємо `fn` лише якщо хоч один із них увімкнено.

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
- **Ворота «проти загального тренду»** (`queue3_vob_block_against_trend`, дефолт
  **ON**): НОВУ угоду не відкриваємо, якщо напрямок проти ЗАГАЛЬНОГО тренду монети
  (`dir_overall` ≈2 год з `_funding_price`): LONG у загальному ↓ / SHORT у ↑ →
  skip (лог `Q3-VOB`). 'flat' — дозволяємо. Трейл SL — без цих воріт.
- **TG «Рекомендація бота»** тепер надсилає `_vob_open_or_trail` (не окремий
  alert-branch, щоб не дублювати), формат мінімальний (`_send_layer_alert(sym,a,
  d,sl,mode)`): `🎯 Рекомендація бота · #SYM 🔴 SHORT` + `🛑 SL: <ціна>` (mode
  'open') / `♻️ Угода вже відкрита — змінено лише SL: <ціна>` (mode 'trail').
  Якщо `queue3_vob_open` OFF, а `layer_tg_on` ON — падає на просту «Рекомендацію»
  (mode 'signal', без SL) за старим порогом `layer_tg_min`.
- Усе (open / sl_moved / skipped) пишеться в 🧾 Лог роботи бота (`activity_log`,
  source `Q3-VOB`) з міткою «монета з фандингу».
- Налаштування: `queue3_vob_open` (True), `queue3_vob_sl_buffer_pct` (0.10),
  `queue3_vob_block_against_trend` (True). UI — у гармошці **«⚙️ Налаштування
  (Черга-3)»** (`templates/smart_money.html`), куди перенесено ВСІ параметри
  Черги-3. Заголовок колонки шарів — лише «🎯».
- **💰-мітка у відкритих угодах** (`histSymOpen` у smart_money + `renderTrades` в
  infosite): монета з фандингу (є в `anomalies` АБО `opened_by` містить funding).
- Тести: `test_vob_open_*` (+ ворота тренду) у `test_readiness_strategy.py`.

## ✦ «Золотий funding» TG — формат + затримка-підтвердження

`_send_gold_alert(sym,a,step)` формат (топік 💰 funding): `✦ FUNDING 🟢LONG /
#SYM / 💰 Funding: -2.000%`. Показуємо СНЕПНУТЕ чисте значення (`step` зі знаком
ставки) → завжди точне (−0.500/−1.500/−2.000%), а не сире (−0.495/−1.503%).
Перед відправкою рівень має протриматись `funding_gold_confirm_sec` (дефолт 30с,
«чітке визначення»); далі повтор раз на `funding_gold_cooldown_min` (60хв), поки
той самий рівень тримається. Тест: `test_gold_funding_confirm_then_repeat`.

## 💰 Funding Rate Scanner — колонка «Price» (dashboard)

Таблиця «💰 Funding Rate Scanner» (`templates/dashboard.html`, дані з
`FundingMonitor.get_watchlist()`) у колонці **Price** показує ЧІТКИЙ свіжий
напрямок ЦІНИ: `price_dir` (up/down/flat) + `price_chg_recent` (% за вікно
`PRICE_WINDOW`≈15хв) — **▲ росте (зелений) / ▼ спадає (червоний) / ▬ рівно**.
Плюс ЗАГАЛЬНИЙ тренд монети (`price_dir_overall`/`price_chg_overall`,
`PRICE_WINDOW_LONG`≈2год, deadzone 0.30%) — рядок «загалом ▲/▼ X%», щоб не входити
проти загального тренду (той самий `dir_overall` живить ворота VOB-відкриття).
Цей самий блок (`price` у get_state anomaly-рядку = `_funding_price[sym]`)
показується ДРУГИМ РЯДКОМ у таблиці «💰 Funding — МММ» (bot + infosite):
«💹 Ціна: ▲ +X% 15хв · 📈 Загалом: ▲ +Y% ~2год», вирівняно в стовпчик.
Це окремо від `price_change` (сумарно від старту стеження, тепер у підказці).
`_recent_price_move(rates)` рахує рух по останніх ~15 семплах (мертва зона
`PRICE_DEADZONE`=0.10%). Ту саму метрику віддає `get_price_dirs()` для 5-го шару.

## Закриті угоди: постійний архів у БД + швидкість закриття

- **Кожна закрита угода (real+paper) БЕЗУМОВНО пишеться в постійну таблицю
  `sob_trade_archive`** (`_archive_closed` → `db.archive_trade`, гейт по
  прихованому `archive_trades` ПРИБРАНО). Ролінг-блоб `tm_closed_trades` (кеп
  `CLOSED_TRADES_LIMIT`=2000) лишається для живої стрічки. get_state віддає ВЕСЬ
  блоб (не 50) — таблиці рендерять усе з захистом-сигнатурою (не «стрибають» під
  час скролу).
- **Таблиці «📜 Recent Closed» / «Recent Paper Closes» — прокрутка + копіювання +
  «за весь час (БД)».** Клас `.tm-vhscroll` (max-height, sticky-заголовок);
  `copyTableToClipboard()` → TSV у буфер; `toggleArchiveView('closed'|'shadow')`
  тягне ВСЮ історію з `/api/trade-archive/export?is_paper=…` і показує read-only
  (жива стрічка тоді не перемальовує — `window._tmArchiveMode`).
- **Швидкість закриття:** цикл монітора виходів — налаштовний
  `monitor_interval_secs` (дефолт **4с**, було жорстко 10с; clamp [2,30]).
  Ціна береться з кешу сканера (без API-хіта), тож частіше = швидше закриття
  без навантаження. Реконсіляція з Bybit — на власному `reconcile_interval_secs`
  (n_ticks рахується від ефективного інтервалу монітора). Сам close-ордер
  reduce-only шлеться синхронно, важке (real-PnL/архів/notify) — у фоні
  (`_finalize_close_async`), тож не блокує монітор.

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
