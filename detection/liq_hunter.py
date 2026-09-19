"""
liq_hunter — 💧 СКАНЕР ЛІКВІДНОСТІ: власний рушій відбору монет під напрямок
банера «🧮 МММ-монітор» (вимога 18.09).

**Дослівно:** «організуємо сканер із своїм вікном і основним тумблером і
гармошкою налаштувань, під вікном "МММ-монітор", який буде відбирати монети за
певним фільтром і додавати в свою таблицю. За основу беремо алгоритм [скан
ліквідності 📡 Tickr]… Додай до налаштувань періодичність сканування (за
замовчуванням 15хв) і фільтр саме які монети беремо (за замовчуванням 65%) —
"Маса ліквідності НИЖЧЕ ціни 65% · тягне ВНИЗ" за напрямком, який вказує на
даний момент банер "МММ-монітор". Кнопку "Сканувати" залиш, для ручного
сканування і оновлення даних. Дані при кожному скані оновлюються, але потрібно
[перевірити] чи немає вже конкретної монети в роботі (відкрита угода або
черга), в такому випадку додавати повторно монету не потрібно.»

**АЛГОРИТМ ВЗЯТО ГОТОВИЙ — `detection/liq_scan.scan_liquidity`**, той самий, що
живить блок 💧 на сторінці 📡 Tickr, із тими самими параметрами (біржа, список,
кількість монет, глибина історії, пороги обігу/OI, сортування) і ТИМИ САМИМИ
дефолтами, що стоять на тій сторінці. Своєї копії розрахунку тут немає і бути
не може: інакше «маса 68% вниз» у сканері й на 📡 Tickr означали б різні числа
(урок PD-зони).

**ЩО САМЕ ВІДБИРАЄМО.** Напрямок задає банер МММ-монітора — не сканер:
  • банер 🟢 LONG → беремо монети, де маса ліквідності ВИЩЕ ціни ≥ порогу
    (`above_pct`, тобто «тягне ВГОРУ»);
  • банер 🔴 SHORT → маса НИЖЧЕ ціни ≥ порогу (`below_pct`, «тягне ВНИЗ»).
Це ТА САМА конвенція, що в 💧 фільтрі входу (`liq_filter`): «за напрямком» =
частка маси, яка тягне ціну ТУДИ, куди ми входимо. Переплутати боки означало б
відбирати рівно протилежні монети.

⚠️ **БАНЕР ⚖ БЕЗ НАПРЯМКУ → БІРЖУ НЕ ЧІПАЄМО ВЗАГАЛІ.** Відбирати «у бік, якого
немає» нічого; тягнути при цьому сотні запитів — марне навантаження. Таблиця
чиститься, а причина пишеться в статус: мовчазна порожня таблиця читалась би як
поломка.
⚠️ **ФЛІП БАНЕРА → ПЕРЕСКАН** (вимога 3): монети, відібрані під старий
напрямок, після розвороту не просто застарілі — вони СУПЕРЕЧАТЬ банеру. Тому на
зміні напрямку таблиця перебудовується. Спаму тут немає за побудовою: банер уже
перемикається ПОВІЛЬНО (гістерезис + вікно підтвердження у `_mm_track_bias`), а
понад це діє власний мінімальний проміжок `MIN_GAP_SEC`.
⚠️ **КОЖЕН СКАН ІДЕ ЧЕРЕЗ СПІЛЬНУ ЧЕРГУ** `detection/scan_queue.py` — за раз
працює рівно один скан біржі на весь бот (вимога 4).
⚠️ **ЦЕ СПОСТЕРЕЖЕННЯ, А НЕ ТОРГІВЛЯ.** Рушій нічого не відкриває і нікуди не
сигналить САМ: він веде список кандидатів і ВІДПОВІДАЄ на питання «чи є ця
монета в таблиці й у той самий бік». Відкриття угод лишається за чергами,
сигналами і ✋ ручними діями.

**💧 VOB + БАНЕР + ТАБЛИЦЯ → СИГНАЛ (вимога 19.09), дослівно:** «Якщо увімкнено
📦 Volumized OB Trend і увімкнено "Сканер ліквідності", то кожен новоутворений
VOB звіряємо з таблицею "Сканер ліквідності", і якщо співпадає напрямок VOB +
напрямок банера "МММ-монітор" і є така ж монета у таблиці — одразу маємо сигнал
і відправляємо монету далі по алгоритму в Чергу (якщо активно) або відразу
відкриваємо угоду, перед цим перевіривши, чи не існує вже відкрита така угода.»
- Саму ПОДІЮ (новий VOB) ловить сканер — там, де VOB і рахується (`smc_scanner`,
  той самий блок, що малює бокс на графіку). Тут живе лише ВІДПОВІДЬ про
  таблицю: `vob_match(symbol, side)` над ЧИСТОЮ `vob_confluence(...)`.
- ⚠️ Другого списку монет не заводимо: звіряємось РІВНО з тими рядками, що
  людина бачить у таблиці 💧 Сканера.
"""

import json
import threading
import time
from typing import Callable, Dict, List, Optional

CHECK_SECS = 10.0          # як часто прокидаємось (фліп банера ловимо тут)
MIN_GAP_SEC = 60.0         # мінімальний проміжок МІЖ сканами (антиспам)
SCAN_JOB = 'liq-hunter'    # імʼя завдання у спільній черзі сканів
_DB_SETTINGS = 'liq_hunter_settings'
_DB_STATE = 'liq_hunter_state'

# Дефолти скану — РІВНО ті, що стоять у блоці 💧 на сторінці 📡 Tickr
# («за замовчуванням як на скріні»): Binance · Топ біржі · 40 монет · 168 год ·
# за перекосом · обіг ≥ $20M · OI ≥ $5M.
DEFAULTS = {
    'enabled': False,               # майстер-тумблер (новий рушій не вмикаємо
                                    # мовчки — та сама причина, що у нових
                                    # типів сигналу)
    'exchange': 'binance',
    'universe': 'top',              # top | watchlist
    'top_n': 40,
    'bars': 168,
    'sort_by': 'pull',
    'min_vol_usd': 20_000_000,
    'min_oi_usd': 5_000_000,
    'interval_min': 15,             # періодичність скану, хв (вимога)
    'min_mass_pct': 65.0,           # поріг «Маса ліквідності» у бік банера, %
    # 🧲 Мінімальна ВІДСТАНЬ до найсильнішого магніту, % від поточної ціни
    # (вимога 18.09): магніт за 0.5% — це не ціль, а шум, і брати таку монету
    # в таблицю немає сенсу. 0 = не фільтрувати.
    'min_magnet_dist_pct': 3.0,
    'rescan_on_flip': True,         # перескан на зміні напрямку банера
    # 💧 VOB + банер + таблиця → СИГНАЛ (вимога 19.09). Дефолт **УВІМКНЕНО**:
    # умову ввімкнення користувач задав САМИМИ наявними тумблерами («якщо
    # увімкнено 📦 Volumized OB Trend і увімкнено Сканер»), тож третій тумблер
    # із дефолтом OFF означав би, що вимога не працює, доки його не знайдуть.
    # ⚠️ Потік угод це мовчки не розширює: сам 💧 Сканер дефолтом ВИМКНЕНИЙ,
    # тобто шлях оживає лише після свідомого вмикання сканера. Тумблер існує,
    # щоб можна було лишити таблицю і вимкнути САМЕ сигнали.
    'vob_signal_on': True,
}

# Скільки останніх рішень шляху «VOB + таблиця» тримаємо для показу.
SIGNAL_LOG_CAP = 12
# TTL кешу налаштувань. Малий СВІДОМО: зміна з UI має діяти одразу, а кеш тут
# лише щоб не читати блоб із БД сотні разів за скан-цикл.
SETTINGS_TTL = 5.0


def mass_pct(row: Dict, side: str):
    """Частка маси ліквідності, що тягне ціну В БІК `side`.

    ЧИСТА функція і ЄДИНЕ місце, де боки зіставляються з полями драбини:
    **LONG → `above_pct`** (маса ВИЩЕ ціни тягне вгору), **SHORT →
    `below_pct`**. Та сама конвенція, що в 💧 фільтрі входу і в кольорах
    драбини; переплутати її = відбирати протилежні монети.
    """
    key = 'above_pct' if side == 'LONG' else 'below_pct'
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def magnet_dist_pct(row: Dict):
    """Відстань до найсильнішого магніту у ВІДСОТКАХ — СИРИМ числом.

    ⚠️ `magnet_dist` — це ФОРМАТОВАНИЙ рядок («↓5.58%»), і парсити його заради
    числа НЕ МОЖНА (задокументована пастка магніта). Беремо сирий рядок
    сходинки `magnet_row['dist_pct']` — той самий, з якого малюється підпис.
    Немає даних → `None` (це НЕ «нуль»).
    """
    mr = row.get('magnet_row') or {}
    try:
        return abs(float(mr.get('dist_pct')))
    except (TypeError, ValueError):
        return None


def pick_rows(rows: List[Dict], side: str, min_pct: float,
              in_work=None, min_magnet_dist: float = 0.0) -> (List[Dict], Dict):
    """ЧИСТИЙ відбір монет під напрямок банера.

    Повертає `(відібрані, розклад)`. Розклад називає КОЖНУ причину відсіву —
    «монет мало» ніколи не має бути здогадкою:
      • `weak`    — маса в потрібний бік нижча за поріг;
      • `near`    — 🧲 магніт ЗАБЛИЗЬКО (менше `min_magnet_dist` % від ціни);
      • `in_work` — монета ВЖЕ в роботі (відкрита угода або черга), тож
        повторно її не додаємо (дослівна вимога);
      • `nodata`  — по монеті скан не дав чисел.

    ⚠️ Магніт БЕЗ ВІДСТАНІ (драбина не дала сходинки) фільтром `near` НЕ
    ріжеться: «невідомо» — це не «близько», і вигадувати відмову ми не маємо
    (той самий принцип, що fail-open у 💧 фільтра). Такий рядок просто покаже
    «—» у колонці магніту.
    """
    work = {str(x).upper() for x in (in_work or set())}
    out, st = [], {'weak': 0, 'near': 0, 'in_work': 0, 'nodata': 0,
                   'scanned': len(rows or [])}
    for r in (rows or []):
        sym = str(r.get('symbol') or '').upper()
        if not r.get('ok'):
            st['nodata'] += 1
            continue
        p = mass_pct(r, side)
        if p is None:
            st['nodata'] += 1
            continue
        if p < float(min_pct or 0):
            st['weak'] += 1
            continue
        _md = magnet_dist_pct(r)
        if float(min_magnet_dist or 0) > 0 and _md is not None \
                and _md < float(min_magnet_dist):
            st['near'] += 1
            continue
        # ⚠️ Перевірка «в роботі» стоїть ПІСЛЯ порога навмисно: у розкладі має
        # бути видно, скільки монет ПІДІЙШЛИ і були пропущені саме тому, що
        # вже в роботі, а не змішано зі слабкими.
        if sym in work:
            st['in_work'] += 1
            continue
        out.append(r)
    out.sort(key=lambda r: -(mass_pct(r, side) or 0))
    return out, st


def vob_confluence(row: Optional[Dict], vob_side: str,
                   bias_dir: Optional[str]) -> (bool, str):
    """ЧИСТЕ правило «новий VOB → сигнал?» (вимога 19.09).

    Три умови РАЗОМ і в цьому порядку:
      1. банер 🧮 МММ-монітора має НАПРЯМОК (⚖ рівновага — не напрямок);
      2. напрямок VOB ЗБІГАЄТЬСЯ з напрямком банера;
      3. монета Є в таблиці 💧 Сканера, і рядок зібрано в ТОЙ САМИЙ бік.

    ⚠️ Пункт 3 перевіряє ще й `row['side']`, а не лише присутність символу.
    Одразу після розвороту банера таблиця ще стара (перескан попереду), і її
    рядки зібрані під ПРОТИЛЕЖНИЙ бік — узяти такий рядок означало б назвати
    «збігом» пряме протиріччя. Замість цього чесно кажемо, що чекаємо перескан.

    Повертає `(ok, причина/розклад)`. Причина потрібна ЗАВЖДИ: «сигналу немає»
    без пояснення читається як поломка.
    """
    side = (vob_side or '').upper().strip()
    bias = (bias_dir or '').upper().strip()
    if side not in ('LONG', 'SHORT'):
        return False, 'напрямок VOB невідомий'
    if bias not in ('LONG', 'SHORT'):
        return False, '⚖ банер 🧮 МММ-монітора без напрямку'
    if side != bias:
        return False, f'VOB {side} ПРОТИ банера {bias}'
    if not row:
        return False, 'монети немає в таблиці 💧 Сканера ліквідності'
    r_side = str(row.get('side') or '').upper()
    if r_side and r_side != side:
        return False, (f'рядок таблиці зібрано під {r_side} — '
                       f'чекаємо перескан під {side}')
    _bits = [f'VOB {side} = банер {bias}']
    try:
        _bits.append(f"💧 маса {float(row.get('mass_pct')):.1f}% у бік {side}")
    except (TypeError, ValueError):
        pass
    if row.get('magnet_price'):
        _d = row.get('magnet_dist_pct')
        _bits.append('🧲 магніт ' + str(row.get('magnet_price'))
                     + (f' ({float(_d):.2f}%)' if _d is not None else ''))
    return True, ' · '.join(_bits)


class LiqHunterDaemon:
    def __init__(self, db, get_watchlist: Optional[Callable] = None,
                 scan_fn: Optional[Callable] = None,
                 get_fuel_filter: Optional[Callable] = None):
        self._db = db
        self._get_watchlist = get_watchlist
        # Скан і доступ до FF інʼєктуються — так їх підміняють тести, а в
        # проді беруться справжні модулі (ліниво, щоб не тягнути пакет
        # `detection/__init__` у ізольовані тести).
        self._scan_fn = scan_fn
        self._get_ff = get_fuel_filter
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._rows: List[Dict] = []
        self._since: Dict[str, float] = {}   # sym → коли вперше зʼявився
        self._ts: float = 0.0                # час останнього УСПІШНОГО скану
        self._dir: Optional[str] = None      # напрямок, під який зібрано список
        self._last_dir_seen: Optional[str] = None
        self._status: str = 'Ще не сканували'
        self._last: Dict = {}                # розклад останнього скану
        self._scanning: bool = False
        self._next_at: float = 0.0
        # 💧 VOB + таблиця: ОСТАННІ рішення шляху (сигнал / відсіяно фільтрами /
        # пропущено — угода вже відкрита). Без цього шлях був би невидимий, а
        # «нічого не відбувається» читалось би як поломка (урок «де сигнали?»).
        self._signals: List[Dict] = []
        self._sig_count: int = 0
        # Кеш налаштувань (див. `get_settings`): шлях «VOB + таблиця» питає їх
        # по кожній монеті кожного скан-циклу.
        self._s_cache: Dict = {}
        self._s_cache_at: float = 0.0

    # ── налаштування ────────────────────────────────────────────────────
    def get_settings(self) -> Dict:
        """Налаштування сканера з КОРОТКИМ кешем.

        ⚠️ Кеш тут ОБОВʼЯЗКОВИЙ, а не «оптимізація»: відколи шлях «VOB +
        таблиця» питає `vob_signal_on()` по КОЖНІЙ монеті КОЖНОГО скан-циклу,
        читання блоба з БД коштувало б сотні сесій SQLAlchemy на цикл (той
        самий урок, що з `get_mm_settings`). `SETTINGS_TTL` малий, а
        `update_settings` кладе свіже значення в кеш одразу, тож зміна з UI
        діє миттєво.
        """
        _now = time.time()
        _c = self._s_cache
        if _c and (_now - self._s_cache_at) < SETTINGS_TTL:
            return dict(_c)
        s = dict(DEFAULTS)
        try:
            raw = self._db.get_setting(_DB_SETTINGS, {}) or {}
            if isinstance(raw, str):
                raw = json.loads(raw or '{}')
            if isinstance(raw, dict):
                for k in DEFAULTS:
                    if k in raw:
                        s[k] = raw[k]
        except Exception:
            pass
        s = self._validate(s)
        self._s_cache, self._s_cache_at = dict(s), _now
        return s

    @staticmethod
    def _validate(s: Dict) -> Dict:
        s['enabled'] = bool(s.get('enabled'))
        s['rescan_on_flip'] = bool(s.get('rescan_on_flip', True))
        s['vob_signal_on'] = bool(s.get('vob_signal_on', True))
        s['exchange'] = str(s.get('exchange') or 'binance').lower()
        s['universe'] = 'watchlist' if str(s.get('universe')) == 'watchlist' \
            else 'top'
        s['sort_by'] = str(s.get('sort_by') or 'pull')

        def _i(k, dflt, lo, hi):
            try:
                s[k] = max(lo, min(hi, int(float(s.get(k, dflt)))))
            except (TypeError, ValueError):
                s[k] = dflt

        _i('top_n', 40, 5, 200)
        # Стеля глибини — 1000 барів на один запит (найтісніший ліміт серед
        # бірж, Bybit). Та сама межа, що в `liq_scan`.
        _i('bars', 168, 24, 1000)
        _i('interval_min', 15, 1, 720)
        _i('min_vol_usd', 20_000_000, 0, 10_000_000_000)
        _i('min_oi_usd', 5_000_000, 0, 10_000_000_000)
        try:
            s['min_mass_pct'] = max(0.0, min(100.0,
                                             float(s.get('min_mass_pct', 65))))
        except (TypeError, ValueError):
            s['min_mass_pct'] = 65.0
        try:
            s['min_magnet_dist_pct'] = max(0.0, min(100.0, float(
                s.get('min_magnet_dist_pct', 3))))
        except (TypeError, ValueError):
            s['min_magnet_dist_pct'] = 3.0
        return s

    def update_settings(self, patch: Dict) -> Dict:
        s = self.get_settings()
        if isinstance(patch, dict):
            for k in DEFAULTS:
                if k in patch:
                    s[k] = patch[k]
        s = self._validate(s)
        # ⚠️ Кеш оновлюємо ОДРАЗУ: інакше до `SETTINGS_TTL` секунд бот працював
        # би за старим значенням, а UI вже показував би нове.
        self._s_cache, self._s_cache_at = dict(s), time.time()
        try:
            self._db.set_setting(_DB_SETTINGS, s)
        except Exception as e:
            print(f"[LiqHunter] settings persist error: {e}")
        if s.get('enabled'):
            self.start()
        else:
            # ⚠️ Вимкнений сканер ЧИСТИТЬ таблицю: «заморожені» рядки, які вже
            # ніхто не оновлює, виглядають як живі (той самий принцип, що з
            # банером монітора при вимкненому тумблері).
            with self._lock:
                self._rows, self._since, self._ts = [], {}, 0.0
                self._status = 'Сканер вимкнено'
            self._persist_state()
        return s

    def is_enabled(self) -> bool:
        return bool(self.get_settings().get('enabled'))

    def set_enabled(self, on: bool) -> Dict:
        return self.update_settings({'enabled': bool(on)})

    # ── персист (таблиця переживає рестарт) ─────────────────────────────
    def _persist_state(self):
        try:
            with self._lock:
                blob = {'rows': self._rows, 'ts': self._ts, 'dir': self._dir,
                        'since': self._since, 'status': self._status,
                        'last': self._last}
            self._db.set_setting(_DB_STATE, blob)
        except Exception as e:
            print(f"[LiqHunter] persist error: {e}")

    def _restore_state(self):
        try:
            raw = self._db.get_setting(_DB_STATE, {}) or {}
            if isinstance(raw, str):
                raw = json.loads(raw or '{}')
            if not isinstance(raw, dict):
                return
            with self._lock:
                self._rows = list(raw.get('rows') or [])
                self._ts = float(raw.get('ts') or 0.0)
                self._dir = raw.get('dir') or None
                self._since = {k: float(v) for k, v
                               in (raw.get('since') or {}).items()}
                self._status = str(raw.get('status') or 'Відновлено після рестарту')
                self._last = dict(raw.get('last') or {})
            print(f"[LiqHunter] restored {len(self._rows)} rows from DB")
        except Exception as e:
            print(f"[LiqHunter] restore error: {e}")

    # ── джерела ─────────────────────────────────────────────────────────
    def _ff(self):
        if self._get_ff:
            return self._get_ff()
        try:
            from detection.fuel_filter import get_fuel_filter
            return get_fuel_filter()
        except Exception:
            return None

    def bias_dir(self) -> Optional[str]:
        """Напрямок банера «🧮 МММ-монітор» — ПІДТВЕРДЖЕНИЙ (`mm_bias`)."""
        try:
            ff = self._ff()
            if not ff:
                return None
            d = (ff.mm_bias() or {}).get('dir')
            return d if d in ('LONG', 'SHORT') else None
        except Exception:
            return None

    def _in_work(self) -> set:
        """Монети «в роботі» — ЄДИНИМ джерелом `ff.symbols_in_work()`."""
        try:
            ff = self._ff()
            if ff and hasattr(ff, 'symbols_in_work'):
                return set(ff.symbols_in_work() or set())
        except Exception:
            pass
        return set()

    def _watchlist(self) -> List[str]:
        try:
            return [str(x).upper() for x in (self._get_watchlist() or [])] \
                if self._get_watchlist else []
        except Exception:
            return []

    def _run_scan_fn(self, s: Dict) -> Dict:
        """Сам скан — ГОТОВИЙ `liq_scan.scan_liquidity` із налаштуваннями."""
        if self._scan_fn:
            return self._scan_fn(s)
        from detection import liq_scan
        return liq_scan.scan_liquidity(
            exchange=s['exchange'], top_n=int(s['top_n']),
            min_vol_usd=float(s['min_vol_usd']),
            min_oi_usd=float(s['min_oi_usd']),
            bars=int(s['bars']), sort_by=s['sort_by'],
            universe=s['universe'],
            symbols=(self._watchlist() if s['universe'] == 'watchlist' else None))

    # ── 💧 VOB + БАНЕР + ТАБЛИЦЯ (відповідь для сканера) ────────────────
    def row_for(self, symbol: str) -> Optional[Dict]:
        """Рядок таблиці по монеті або None. РІВНО те, що видно на сторінці."""
        sym = str(symbol or '').upper().strip()
        if not sym:
            return None
        with self._lock:
            for r in self._rows:
                if str(r.get('symbol') or '').upper() == sym:
                    return dict(r)
        return None

    def vob_signal_on(self) -> bool:
        """Чи ПРАЦЮЄ шлях «VOB + таблиця» зараз: сканер увімкнено І сигнали не
        вимкнені окремим тумблером. ОКРЕМИЙ метод, щоб сканер не вгадував це з
        тексту причини (розбір рядків — це не перевірка стану)."""
        s = self.get_settings()
        return bool(s.get('enabled')) and bool(s.get('vob_signal_on', True))

    def vob_match(self, symbol: str, vob_side: str) -> (bool, str, Optional[Dict]):
        """Чи дає НОВИЙ VOB по монеті сигнал за правилом «VOB + банер + таблиця».

        Повертає `(ok, причина, рядок_таблиці)`. Сам сигнал НЕ шлемо — його шле
        сканер (там, де VOB і виник), через ті самі спільні ворота
        `_signal_allowed`, що й решта сигналів.
        """
        s = self.get_settings()
        if not s.get('enabled'):
            return False, '💧 Сканер ліквідності вимкнено', None
        if not s.get('vob_signal_on', True):
            return False, 'сигнали «VOB + таблиця» вимкнено в налаштуваннях сканера', None
        row = self.row_for(symbol)
        ok, note = vob_confluence(row, vob_side, self.bias_dir())
        return ok, note, row

    def note_vob_signal(self, symbol: str, side: str, status: str,
                        detail: str = ''):
        """Записати РІШЕННЯ шляху «VOB + таблиця» (для показу в панелі).

        `status`: 'signal' (пішов далі по алгоритму) · 'rejected' (спільні
        ворота не пропустили) · 'in_trade' (угода по монеті вже відкрита) ·
        'dup' (цей самий блок уже пішов як 🟪 VOB-алерт).
        ⚠️ Звичайні «не збіглось» СЮДИ НЕ пишемо: це СТАН більшості монет
        щотакту, і він залив би панель (той самий урок, що з логом VOB).
        """
        rec = {'symbol': str(symbol or '').upper(), 'side': side,
               'status': status, 'detail': detail, 'at': time.time()}
        with self._lock:
            self._signals.insert(0, rec)
            del self._signals[SIGNAL_LOG_CAP:]
            if status == 'signal':
                self._sig_count += 1

    # ── скан ────────────────────────────────────────────────────────────
    def scan(self, reason: str = 'manual') -> Dict:
        """Один скан: біржа → фільтр за напрямком банера → таблиця.

        ⚠️ Іде ЧЕРЕЗ СПІЛЬНУ ЧЕРГУ сканів: два скани біржі одночасно не
        працюють ніколи, а повторний запит того самого скану, поки перший ще
        в черзі, чекає на ТОЙ САМИЙ результат (дедуп за іменем завдання).
        """
        s = self.get_settings()
        side = self.bias_dir()
        now = time.time()
        if side not in ('LONG', 'SHORT'):
            # ⚖ Банер без напрямку — відбирати нема в який бік. Біржу НЕ
            # чіпаємо, а стару таблицю ЧИСТИМО: монети відбирались під
            # напрямок, якого вже немає.
            with self._lock:
                self._rows, self._since = [], {}
                self._dir = None
                self._status = ('⚖ Банер «🧮 МММ-монітор» без напрямку — '
                                'скан не запускаємо, відбирати нема в який бік')
                self._last = {'reason': reason, 'at': now, 'skipped': 'flat'}
                self._next_at = now + max(60.0, float(s['interval_min']) * 60.0)
            self._persist_state()
            return {'ok': False, 'reason': 'flat', 'dir': None,
                    'status': self._status}
        with self._lock:
            self._scanning = True
        try:
            from detection import scan_queue
            res = scan_queue.run(SCAN_JOB, lambda: self._run_scan_fn(s),
                                 source=reason, timeout=600.0)
        except Exception as e:
            res = {'ok': False, 'reason': f'{type(e).__name__}: {e}'}
        finally:
            with self._lock:
                self._scanning = False
        now = time.time()
        if not isinstance(res, dict) or not res.get('ok'):
            why = (res or {}).get('reason') if isinstance(res, dict) else 'збій'
            with self._lock:
                self._status = f'⚠️ Скан не вдався: {why}'
                self._last = {'reason': reason, 'at': now, 'error': why}
                self._next_at = now + max(60.0, float(s['interval_min']) * 60.0)
            self._persist_state()
            return {'ok': False, 'reason': why, 'dir': side}
        picked, st = pick_rows(res.get('rows') or [], side,
                               float(s['min_mass_pct']), self._in_work(),
                               float(s['min_magnet_dist_pct']))
        rows = []
        with self._lock:
            old_since = dict(self._since)
        since = {}
        for r in picked:
            sym = str(r.get('symbol') or '').upper()
            # Монета, що ТРИМАЄТЬСЯ у списку між сканами, не «зʼявляється»
            # заново — інакше таймер у таблиці обнулявся б кожні 15 хв і не
            # показував би нічого корисного.
            since[sym] = old_since.get(sym, now)
            rows.append({
                'symbol': sym,
                'side': side,
                'mass_pct': round(mass_pct(r, side) or 0.0, 1),
                'mass_dir': 'up' if side == 'LONG' else 'down',
                'pull': r.get('pull'), 'pull_pct': r.get('pull_pct'),
                'price': r.get('price'),
                # 🧲 НАЙСИЛЬНІШИЙ МАГНІТ — з ТОГО САМОГО рядка скану (вимога:
                # «Відображай… Найсильніший магніт — напрямок»).
                'magnet_price': r.get('magnet_price'),
                'magnet_pct': r.get('magnet_pct'),
                'magnet_dist': r.get('magnet_dist'),
                # СИРЕ число відстані — фронт малює ним рівну колонку і колір;
                # формальний підпис (`magnet_dist`) лишається для тултипа.
                'magnet_dist_pct': magnet_dist_pct(r),
                'magnet_dir': r.get('magnet_dir'),
                'exchange': r.get('exchange') or s['exchange'],
                'fallback': bool(r.get('fallback')),
                'since': since[sym],
                'ts': now,
            })
        _q = res.get('queue_wait_sec')
        status = (f"✅ {len(rows)} монет у бік {side} · маса ≥ {s['min_mass_pct']:.0f}% · "
                  f"переглянуто {st['scanned']} · слабких {st['weak']} · "
                  f"у роботі {st['in_work']}"
                  + (f" · магніт ближче {s['min_magnet_dist_pct']:.0f}%: "
                     f"{st['near']}" if st['near'] else '')
                  + (f" · без даних {st['nodata']}" if st['nodata'] else '')
                  + (f" · черга {_q}с" if _q else ''))
        with self._lock:
            self._rows, self._since, self._ts = rows, since, now
            self._dir, self._last_dir_seen = side, side
            self._status = status
            self._last = {'reason': reason, 'at': now, 'dir': side,
                          'took_sec': res.get('took_sec'),
                          'queue_wait_sec': _q, **st, 'kept': len(rows)}
            self._next_at = now + float(s['interval_min']) * 60.0
        self._persist_state()
        return {'ok': True, 'dir': side, 'rows': len(rows), 'status': status,
                **st}

    # ── цикл ────────────────────────────────────────────────────────────
    def _run(self):
        self._stop.wait(20)                  # дати боту піднятись
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[LiqHunter] tick error: {e}")
            self._stop.wait(CHECK_SECS)

    def _tick(self):
        s = self.get_settings()
        if not s.get('enabled'):
            return
        now = time.time()
        side = self.bias_dir()
        with self._lock:
            last_seen = self._last_dir_seen
            ts = self._ts
            nxt = self._next_at
        # 🔁 ФЛІП БАНЕРА → ПЕРЕСКАН (вимога 3). Банер уже перемикається
        # повільно (гістерезис + підтвердження), тож це РЕАЛЬНИЙ розворот, а
        # не шум; понад це тримаємо власний мінімальний проміжок.
        flip = bool(s.get('rescan_on_flip')) and side != last_seen
        due = (not ts) or (now >= nxt)
        if not (flip or due):
            return
        if flip and ts and (now - ts) < MIN_GAP_SEC:
            # Надто рано — не пропускаємо подію, а зсуваємо скан на кінець
            # проміжку (наступний тік його підхопить як `due`).
            with self._lock:
                self._next_at = ts + MIN_GAP_SEC
            return
        with self._lock:
            self._last_dir_seen = side
        self.scan('flip' if flip else 'schedule')

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='liq-hunter')
        self._thread.start()
        print("[LiqHunter] daemon started")

    # ── стан для UI ─────────────────────────────────────────────────────
    def get_state(self) -> Dict:
        s = self.get_settings()
        with self._lock:
            rows = [dict(r) for r in self._rows]
            out = {'ok': True, 'enabled': bool(s.get('enabled')),
                   'settings': s, 'rows': rows, 'count': len(rows),
                   'ts': self._ts, 'dir': self._dir,
                   'status': self._status, 'last': dict(self._last),
                   'scanning': bool(self._scanning),
                   'next_at': self._next_at,
                   # 💧 VOB + таблиця: скільки сигналів пішло від старту і що
                   # саме сталось останнім — шлях мусить бути ВИДИМИЙ.
                   'signals': [dict(x) for x in self._signals],
                   'signal_count': self._sig_count,
                   'running': bool(self._thread and self._thread.is_alive())}
        # Живий напрямок банера — щоб було видно РОЗБІЖНІСТЬ між тим, під який
        # бік зібрано таблицю, і тим, куди банер дивиться зараз.
        out['bias_dir'] = self.bias_dir()
        try:
            from detection import scan_queue
            q = scan_queue.state()
            out['queue'] = {'depth': q.get('depth'),
                            'running': (q.get('running') or {}).get('name'),
                            'last': (q.get('last') or {}).get(SCAN_JOB)}
        except Exception:
            out['queue'] = {}
        return out


_instance: Optional[LiqHunterDaemon] = None


def init_liq_hunter(db, get_watchlist=None, scan_fn=None,
                    get_fuel_filter=None) -> LiqHunterDaemon:
    global _instance
    if _instance is None:
        _instance = LiqHunterDaemon(db, get_watchlist, scan_fn, get_fuel_filter)
        try:
            _instance._restore_state()
        except Exception:
            pass
        try:
            if _instance.is_enabled():
                _instance.start()
                print("[LiqHunter] restored ON state from DB — loop running")
        except Exception:
            pass
    return _instance


def get_liq_hunter() -> Optional[LiqHunterDaemon]:
    return _instance
