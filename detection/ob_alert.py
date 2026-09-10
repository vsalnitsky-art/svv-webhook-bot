"""🆕 АЛЕРТ «НОВИЙ OB НА ГРАФІКУ» — ЧИСТІ функції (без I/O, без мережі).

**Вимога користувача (09.09), дослівно:** «Мені потрібно моментальна реакція на
появу на графіку нового OB 1H і моментальна відправка повідомлення в "Лог роботи
бота" про появу нового OB в який саме час і поточна ціна монети на графіку. І ще
один додатковий нюанс — потрібно вияснити який на даний момент останній і
актуальний OB 4H і також записати цю інформацію до інформації про OB 1H.»

Тут живуть РІШЕННЯ («чи це НОВИЙ блок і чи він щойно зʼявився») і ФОРМАТ —
тобто рівно те, що перевіряється юніт-тестом. Мережа, БД і запис у лог
лишаються у `smc_scanner`.

──────────────────────────────────────────────────────────────────────────────
⚠️ ЧОМУ «ЗʼЯВИВСЯ» = `created_at_t + тривалість бару`, А НЕ `bar_time`
──────────────────────────────────────────────────────────────────────────────
У блоку ДВА різні часи, і плутати їх не можна:
  • `bar_time`     — час СВІЧКИ самого блоку. Вона лежить «назад у часі»: блок
                     малюється на тій свічці, що була ДО пробою структури.
  • `created_at_t` — час бару, на якому спрацював BOS/CHoCH, тобто момент, коли
                     блок узагалі ВИНИК.
Детекція працює по ЗАКРИТИХ барах (`klines_closed`), тож блок стає ВИДИМИМ на
графіку не в `created_at_t`, а коли ТОЙ бар ЗАКРИВСЯ: `created_at_t + tf`.
Саме це число і є «зʼявився» — те, що користувач бачить на графіку.

⚠️ `НОВИЙ` = ЩЕ НЕ ОПРАЦЬОВАНИЙ, а не «bar_time більший» (пастка breaker).
Той самий урок, що вже задокументований для VOB: коли поточний OB стає
breaker, він випадає зі списку і «останнім» стає СТАРІШИЙ блок із МЕНШИМ
`bar_time`. Умова `bt > prev` тоді не виконається НІКОЛИ, і реальна зміна
блоку на графіку лишиться невидимою. Тому тримаємо СПИСОК опрацьованих
`bar_time` (`SEEN_CAP`), а новим вважаємо будь-який, якого в списку немає.

⚠️ СТАРИЙ БЛОК НЕ «ЗʼЯВИВСЯ» → тихо беремо за базу, у лог НЕ пишемо.
Інакше кожен рестарт (а `botupdate` робиться часто) виливав би в лог поточний
OB усіх монет watchlist — тобто СТАН, а не ПОДІЮ. Це рівно той флуд, через
який лог VOB уже довелось чистити. Вікно свіжості — один бар TF
(`max_lag_for`): у межах поточного бару новішого блоку фізично бути не може.

──────────────────────────────────────────────────────────────────────────────
🆕 ДРУГИЙ TF (ЗБІГ) + ДЕДУП «1 НА ТРЕНД» — вимога користувача 10.09
──────────────────────────────────────────────────────────────────────────────
Дослівно: «якщо вибрано другий таймфрейм, то Новий OB має тепер спрацювати,
коли зійдуться обидва таймфрейми (1Н і 4Н тільки тоді вважається, що Новий OB
зʼявився), інакше якщо вибрано основний таймфрейм, а це 1Н, то мають іти всі
Нові OB 1Н» + «якщо вибрано лише основний таймфрейм 1Н то SHORT має право
перебити лише протилежний Новий OB LONG. А якщо задіяно два таймфрейми
наприклад 1Н+4Н то 1Н+4Н SHORT має право перебити лише 1Н+4Н LONG».

Обидва рішення — ЧИСТІ функції `converge` і `dedup_allows`, а «лінію тренду»
для дедупу назначає `combo_label` (`'1H'` або `'1H+4H'`). Саме тому дедуп
знає, В ЯКІЙ комбінації TF стояла попередня позначка: перемкнули другий TF —
це ІНШИЙ за визначенням сигнал, і стара позначка його не глушить.
"""
from typing import Dict, List, Optional

# Скільки опрацьованих `bar_time` тримаємо на монету. 16 — із запасом: блок
# змінюється раз на кілька барів, а переповнення лише «забуде» найдавніший.
SEEN_CAP = 16


def tf_secs(tf) -> int:
    """'5m'→300, '1h'→3600, '4h'→14400. Невідоме → 3600 (година)."""
    t = str(tf or '').lower().strip()
    try:
        if t.endswith('m'):
            return int(t[:-1]) * 60
        if t.endswith('h'):
            return int(t[:-1]) * 3600
        if t.endswith('d'):
            return int(t[:-1]) * 86400
    except (TypeError, ValueError):
        pass
    return 3600


def _to_sec(ts) -> Optional[float]:
    """Час бару приходить і в секундах, і в мілісекундах — зводимо до секунд."""
    try:
        v = float(ts)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v / 1000.0 if v > 1e12 else v


def close_of(bar) -> Optional[float]:
    """Ціна закриття бару.

    ⚠️ КЛЮЧ — `p`, А НЕ `close`. `market_data.fetch_klines` віддає
    `[{p, v, b, s, h, l, o, t}, …]` (див. його докстрінг). Перша версія читала
    `bar.get('close')` — і в КОЖНОМУ рядку логу замість ціни стояв прочерк
    (`⚡ виявлено за 4хв 46с · —`), хоча самі бари були на місці.
    `close`/`c` лишені як фолбек: у `get_chart_data` ohlc уже сконвертований і
    несе саме `close`, тож функція має працювати з обома формами."""
    if not isinstance(bar, dict):
        return None
    for k in ('p', 'close', 'c'):
        try:
            v = float(bar.get(k))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    return None


def side_of(bias) -> Optional[str]:
    """BULLISH→LONG, BEARISH→SHORT, решта→None (напрямку немає)."""
    b = str(bias or '').upper().strip()
    if b == 'BULLISH':
        return 'LONG'
    if b == 'BEARISH':
        return 'SHORT'
    return None


def appeared_at(ob: Optional[Dict], tf) -> Optional[float]:
    """Коли блок став ВИДИМИМ на графіку (секунди, UTC).

    = закриття бару, на якому спрацював BOS/CHoCH (`created_at_t` + тривалість
    бару), бо детекція йде по ЗАКРИТИХ барах. Немає `created_at_t` → None:
    час появи не вигадуємо."""
    if not isinstance(ob, dict):
        return None
    base = _to_sec(ob.get('created_at_t'))
    if base is None:
        return None
    return base + tf_secs(tf)


def is_processed(seen, bar_time) -> bool:
    """Чи цей блок уже опрацьований (є у списку `seen`).

    ⚠️ Порівнюємо ЦІЛІ СЕКУНДИ, бо `seen_add` зберігає саме `int(секунди)`.
    Перша версія звіряла з допуском `abs(a-b) < 0.5` — і дробовий `bar_time`
    (1788976171.524) НІКОЛИ не знаходив свій же записаний 1788976171, тож той
    самий блок щоразу виглядав «новим» і писався в лог повторно. Справжні бари
    приходять рівними секундами, тож у проді це не стріляло б — але залежати
    від цього не можна. Тест-замок: `test_new_ob_logs_exactly_once…`."""
    bt = _to_sec(bar_time)
    if bt is None:
        return False
    key = int(bt)
    for v in (seen or []):
        s = _to_sec(v)
        if s is not None and int(s) == key:
            return True
    return False


def seen_add(seen, bar_time, cap: int = SEEN_CAP) -> List[int]:
    """Позначити блок опрацьованим. Повертає НОВИЙ список (без мутації входу),
    обрізаний до `cap` — найдавніші забуваються першими."""
    bt = _to_sec(bar_time)
    out = [int(_to_sec(v)) for v in (seen or []) if _to_sec(v) is not None]
    if bt is not None and not is_processed(out, bt):
        out.append(int(bt))
    c = max(1, int(cap or SEEN_CAP))
    return out[-c:]


def max_lag_for(tf, cfg_sec=0) -> float:
    """Вікно свіжості: наскільки давнім може бути блок, щоб вважатись «щойно
    зʼявився».

    `cfg_sec > 0` — явне значення користувача; **0 = АВТО = один бар TF**
    (той самий принцип «0 = авто», що у `vob_alert_max_age_bars`). У межах
    поточного бару новішого блоку фізично існувати не може, тож один бар — це
    рівно «найсвіжіший можливий стан», а не довільне число."""
    try:
        c = float(cfg_sec or 0)
    except (TypeError, ValueError):
        c = 0.0
    return c if c > 0 else float(tf_secs(tf))


def outcome(seen, ob: Optional[Dict], tf, now: float, cfg_max_lag=0) -> str:
    """ЄДИНЕ рішення: що робити з поточним блоком.

      'no_ob'     — блоку немає / немає напрямку → нічого;
      'duplicate' — уже опрацьований → нічого (це СТАН, не подія);
      'stale'     — новий для нас, але зʼявився давно → тихо беремо за базу;
      'new'       — новий І щойно зʼявився → ПОДІЯ, пишемо в лог.

    ⚠️ Час появи невідомий → 'stale', а не 'new': сказати «зʼявився», не знаючи
    коли, означало б збрехати в самому рядку логу."""
    if not isinstance(ob, dict) or side_of(ob.get('bias')) is None:
        return 'no_ob'
    bt = ob.get('bar_time')
    if _to_sec(bt) is None:
        return 'no_ob'
    if is_processed(seen, bt):
        return 'duplicate'
    app = appeared_at(ob, tf)
    if app is None:
        return 'stale'
    try:
        lag = float(now) - app
    except (TypeError, ValueError):
        return 'stale'
    return 'new' if lag <= max_lag_for(tf, cfg_max_lag) else 'stale'


# ══════════ 🆕 ДРУГИЙ TF: ЗБІГ = «OB ЗʼЯВИВСЯ» ════════════════════════════

def combo_label(tf1, tf2=None, htf_on: bool = False) -> str:
    """Підпис АКТИВНОЇ комбінації TF — і він же «лінія тренду» для дедупу.

    `'1H'` (лише основний) або `'1H+4H'` (увімкнено другий). Дедуп порівнює
    саме цей рядок, тож «1H+4H SHORT перебиває лише 1H+4H LONG» виконується
    буквально, а перемикання другого TF не глушить новий тип сигналу старою
    позначкою."""
    a = str(tf1 or '').upper().strip() or '?'
    if not htf_on:
        return a
    b = str(tf2 or '').upper().strip()
    return f'{a}+{b}' if b else a


def converge(side1, side2, htf_on: bool):
    """Чи вважається, що «НОВИЙ OB ЗʼЯВИВСЯ» → `(ok, note)`.

      • другий TF ВИМКНЕНО → вирішує ОДИН основний TF: ідуть ВСІ нові OB;
      • УВІМКНЕНО → потрібен ЗБІГ напрямків обох TF, і лише тоді це подія.

    ⚠️ **«Немає OB на старшому TF» ≠ «узгоджено».** Якби відсутність даних
    пропускала сигнал, увімкнений другий TF мовчки НЕ робив би нічого на тих
    монетах, де 4H-блок ще не порахувався — тобто налаштування виглядало б
    працюючим, а насправді не діяло (урок «будь-яка невизначеність = блок»,
    той самий, що в OB-фільтрі з рядком без тега).
    ⚠️ Це НЕ fail-open, як у 💧 фільтра ліквідності: там джерело ЗОВНІШНЄ
    (біржа блокує IP), а тут дані НАШІ власні — блок порахується наступним
    баром старшого TF."""
    if not side1:
        return False, 'немає напрямку основного TF'
    if not htf_on:
        return True, ''
    if not side2:
        return False, 'немає OB на старшому TF — збігу нема'
    if str(side1) != str(side2):
        return False, f'розбіг TF: {side1} проти {side2}'
    return True, f'збіг обох TF ({side1})'


def dedup_allows(prev, side, combo):
    """«Deduplicate Signals (1 per trend)» для 🆕 Нового OB → `(ok, note)`.

    `prev` — позначка останнього ВИСТРІЛЕНОГО сигналу монети
    (`{'side': 'SHORT', 'combo': '1H+4H'}`) або None.

    Правило користувача: той самий напрямок у тій самій комбінації TF —
    глушиться; перебити може ЛИШЕ протилежний напрямок у ТІЙ САМІЙ комбінації.
    ⚠️ Інша комбінація TF = інший за визначенням сигнал → стара позначка його
    не стосується (інакше після вмикання другого TF перший же 1H+4H-сигнал
    мовчки загубився б через позначку від часів «лише 1H»)."""
    if not isinstance(prev, dict):
        return True, ''
    p_side = str(prev.get('side') or '').upper().strip()
    p_combo = str(prev.get('combo') or '').strip()
    if not p_side:
        return True, ''
    _combo = str(combo or '').strip()
    if p_combo and p_combo != _combo:
        return True, f'інша комбінація TF (було {p_combo})'
    if p_side == str(side or '').upper().strip():
        return False, (f'дедуп 1 на тренд: {p_side} на {p_combo or _combo} уже '
                       f'був — перебити може лише протилежний')
    return True, f'перебито протилежним ({p_side} → {side})'


# ─────────────────────────── ФОРМАТ ───────────────────────────────────────

def fmt_utc(ts) -> str:
    """'09.09.26 о 21:00' (UTC). Порожньо → '—'."""
    s = _to_sec(ts)
    if s is None:
        return '—'
    import time as _t
    return _t.strftime('%d.%m.%y о %H:%M', _t.gmtime(s))


def fmt_price(p) -> str:
    """'$78,539.90' — ДЗЕРКАЛО `fmtPriceJS` зі `smart_money.html`.

    ⚠️ Пороги і кількість знаків мусять збігатися з JS один-в-один: та сама
    ціна в 🧾 Лозі й у таблиці не має писатись двома різними числами (урок
    PD-зони). Тест-замок звіряє ці числа з шаблоном."""
    try:
        v = float(p)
    except (TypeError, ValueError):
        return '—'
    if v <= 0:
        return '—'
    if v < 0.0001:
        return f'${v:.8f}'
    if v < 0.01:
        return f'${v:.6f}'
    if v < 1:
        return f'${v:.5f}'
    if v < 100:
        return f'${v:.4f}'
    return f'${v:,.2f}'


def fmt_lag(sec) -> str:
    """'8с' / '3хв 20с' / '—'. Відʼємне (годинник зсунувся) → '0с'."""
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return '—'
    if s < 0:
        s = 0.0
    s = int(round(s))
    if s < 60:
        return f'{s}с'
    m, r = divmod(s, 60)
    if m < 60:
        return f'{m}хв {r}с' if r else f'{m}хв'
    h, m2 = divmod(m, 60)
    return f'{h}г {m2}хв' if m2 else f'{h}г'


def build_parts(symbol: str, tf1, side1, tag1, tf4, side4, price,
                appeared, now, htf_note: str = '', first: bool = False,
                bar_time=None, combo: str = '') -> Dict:
    """ШМАТКИ повідомлення — чисті ДАНІ, без розмітки.

    ⚠️ Той самий прийом, що у вердикті драбини ліквідності (`verdict.parts`
    + `text`): UI фарбує ЧИСЛА окремо від слів, тож рядок не зливається в
    одну сіру смугу. Розмітку модуль НЕ робить — кольори лишаються справою
    фронта, а `build_text` дає той самий зміст пласким текстом (для TG/CSV).
    Тест-замок не дає `parts` і `text` розійтись."""
    try:
        lag = max(0.0, float(now) - float(appeared)) if appeared else None
    except (TypeError, ValueError):
        lag = None
    return {
        'symbol': (symbol or '').upper().strip(),
        'tf1': str(tf1 or '').upper(),
        'side1': side1,
        # ⚠️ Тег НЕ перегинаємо у верхній регістр: детектор віддає саме «CHoCH»
        # і «BOS», і в усьому проєкті вони пишуться так. «CHOCH» виглядало б як
        # інша сутність.
        'tag1': (str(tag1).strip() or None) if tag1 else None,
        'tf4': str(tf4 or '').upper() or None,
        'side4': side4,
        # Активна комбінація TF — те саме, чим дедуп міряє «тренд».
        'combo': str(combo or '').upper() or None,
        'htf_note': str(htf_note or ''),
        'appeared': appeared,
        'appeared_txt': fmt_utc(appeared),
        # ⚠️ СВІЧКА САМОГО БЛОКУ — та, НА ЯКІЙ намальовано бокс. Вона лежить
        # «назад у часі» відносно появи, і саме через це виникає питання
        # «блок на графіку з 12:00, а бот побачив о 21:00?» (кейс LITUSDT).
        # Обидва часи в рядку — і питання відпадає само.
        'bar_time': _to_sec(bar_time),
        'bar_time_txt': fmt_utc(bar_time) if _to_sec(bar_time) else None,
        'price': price,
        'price_txt': fmt_price(price),
        'lag_sec': lag,
        'lag_txt': (fmt_lag(lag) if lag is not None else '—'),
        # ⚠️ ПЕРШИЙ ПОКАЗ ПІСЛЯ СТАРТУ — це НЕ наша реакція.
        # На проді після рестарту 6 монет дали «⚡ виявлено за 4хв 43с»: бот
        # просто піднявся через 4хв після закриття бару, а число читалось як
        # «бот думав 4 хвилини». Тут `lag` міряє вік блоку, а не швидкість
        # бота, тож підпис мусить бути ІНШИЙ.
        'first': bool(first),
    }


def build_text(parts: Dict) -> str:
    """Плаский рядок для 🧾 Логу — у порядку, який задав користувач:
    `LONG OB 1H (CHoCH) · LONG OB 4H · зʼявився 09.09.26 о 21:00 UTC ·
     ⚡ виявлено за 8с · $78,539.90`"""
    p = parts or {}
    bits = []
    _s1 = p.get('side1') or '—'
    _t1 = f"{_s1} OB {p.get('tf1') or '?'}"
    if p.get('tag1'):
        _t1 += f" ({p['tag1']})"
    bits.append(_t1)
    if p.get('tf4'):
        if p.get('side4'):
            bits.append(f"{p['side4']} OB {p['tf4']}")
        else:
            bits.append(f"OB {p['tf4']}: немає")
    if p.get('htf_note'):
        bits.append(p['htf_note'])
    # Свічка блоку — ПЕРЕД появою: спершу «де намальовано», потім «коли виник».
    if p.get('bar_time_txt'):
        bits.append(f"свічка блоку {p['bar_time_txt']} UTC")
    bits.append(f"зʼявився {p.get('appeared_txt') or '—'} UTC")
    if p.get('first'):
        # Перший показ після старту: чесно кажемо, що це вік блоку, а не
        # швидкість реакції бота.
        bits.append(f"👀 перший показ після старту (вік {p.get('lag_txt')})")
    elif p.get('lag_sec') is not None:
        bits.append(f"⚡ виявлено за {p.get('lag_txt')}")
    bits.append(p.get('price_txt') or '—')
    return ' · '.join(bits)
