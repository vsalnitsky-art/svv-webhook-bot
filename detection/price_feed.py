"""⚡ Швидка стрічка цін для ВІДКРИТИХ угод (вимога 25.10).

Навіщо. Монітор виходів (TP/SL/TP-1) тікає раз на 4с, але раніше брав ціну
з кешу SMC-сканера — закриття останнього бару на момент ОСТАННЬОГО скану
монети. Скан проходить ~230 монет послідовно, тож ціна могла бути на хвилини
старшою, ніж здавалось, а ґноти між сканами монітор не бачив узагалі (кейс
XMRUSDT: у таблиці $590.36, на графіку $610.67).

Як. ОДИН bulk-запит Bybit `tickers?category=linear` (усі монети за раз) раз на
`POLL_SEC`, і лише поки є що стерегти (`watch`). Кожен семпл ціни монети, що
стережеться, пишеться в коротку історію, з якої споживач (книга угоди) бере
ДІАПАЗОН між своїми тактами — так TP/SL бачать і ґноти, а не лише точку.

Модуль без залежностей від решти бота: `fetch_all` подається ззовні
(`{SYMBOL: price}`), тож ізольовані тести підкладають свою функцію.
"""
import threading
import time
from collections import deque
from typing import Callable, Dict, Iterable, Optional, Tuple

POLL_SEC = 2.0          # такт стрічки (один запит на ВСІ монети)
STALE_SEC = 10.0        # старша ціна — не «поточна», беремо інше джерело
RANGE_MAX_SEC = 30.0    # діапазон між тактами не довший (рестарт / пауза)
HIST_SEC = 120.0        # скільки семплів памʼятаємо на монету


class PriceFeed:
    def __init__(self, fetch_all: Callable[[], Dict[str, float]],
                 poll_sec: float = POLL_SEC, stale_sec: float = STALE_SEC):
        self._fetch = fetch_all
        self.poll_sec = float(poll_sec)
        self.stale_sec = float(stale_sec)
        self._lock = threading.Lock()
        self._px: Dict[str, Tuple[float, float]] = {}     # SYM → (ціна, ts)
        self._hist: Dict[str, deque] = {}                 # SYM → [(ts, ціна)]
        self._taken: Dict[str, float] = {}                # ключ → ts останнього взяття
        self._watch: set = set()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_ok = 0.0
        self.last_err = ''
        self.polls = 0
        self.errors = 0

    # ── керування ────────────────────────────────────────────────────────
    def watch(self, symbols: Iterable[str]) -> None:
        """Набір монет, які стережемо (відкриті угоди). Порожній → стрічка
        біржу не смикає взагалі."""
        ws = {str(s).upper() for s in (symbols or []) if s}
        with self._lock:
            self._watch = ws
            for s in list(self._hist):
                if s not in ws:
                    self._hist.pop(s, None)
        if ws:
            self.start()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name='price-feed')
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                busy = bool(self._watch)
            if busy:
                self.poll()
            self._stop.wait(self.poll_sec)

    # ── дані ─────────────────────────────────────────────────────────────
    def poll(self, now: Optional[float] = None) -> bool:
        try:
            data = self._fetch() or {}
        except Exception as e:
            self.errors += 1
            self.last_err = f'{type(e).__name__}: {e}'[:200]
            return False
        now = time.time() if now is None else now
        if not data:
            self.errors += 1
            self.last_err = 'біржа віддала порожній список'
            return False
        self.ingest(data, now)
        return True

    def ingest(self, data: Dict[str, float], now: float) -> None:
        with self._lock:
            for sym, p in data.items():
                try:
                    p = float(p)
                except (TypeError, ValueError):
                    continue
                if p <= 0:
                    continue
                sym = str(sym).upper()
                self._px[sym] = (p, now)
                if sym in self._watch:
                    h = self._hist.setdefault(sym, deque())
                    h.append((now, p))
                    while h and now - h[0][0] > HIST_SEC:
                        h.popleft()
            self.polls += 1
            self.last_ok = now
            self.last_err = ''

    def price(self, symbol: str, now: Optional[float] = None,
              max_age: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """(ціна, ts) або None, коли ціни немає чи вона старша за `max_age`."""
        now = time.time() if now is None else now
        lim = self.stale_sec if max_age is None else float(max_age)
        with self._lock:
            v = self._px.get(str(symbol).upper())
        if not v or now - v[1] > lim:
            return None
        return v

    def take_range(self, key: str, symbol: str,
                   now: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """(мін, макс) ціни монети з моменту попереднього взяття цим `key`.

        Ключ — КНИГА угоди (`SYM|R` / `SYM|S`): дві книги по одній монеті
        беруть діапазон незалежно, інакше одна «зʼїдала» б ґніт іншої.
        Перше взяття ключа = лише останній семпл: що було ДО того, як угоду
        почали стерегти, рішенням не є. Вікно обрізане `RANGE_MAX_SEC`."""
        now = time.time() if now is None else now
        sym = str(symbol).upper()
        with self._lock:
            h = list(self._hist.get(sym) or ())
            last = self._taken.get(key)
            if not h:
                return None
            newest = h[-1][0]
            self._taken[key] = newest
        if last is None:
            p = h[-1][1]
            return (p, p)
        lo_t = max(last, now - RANGE_MAX_SEC)
        pts = [p for (t, p) in h if t > lo_t]
        if not pts:
            pts = [h[-1][1]]
        return (min(pts), max(pts))

    def forget(self, key: str) -> None:
        with self._lock:
            self._taken.pop(key, None)

    def health(self, now: Optional[float] = None) -> Dict:
        now = time.time() if now is None else now
        with self._lock:
            watched = len(self._watch)
        age = (now - self.last_ok) if self.last_ok else None
        return {
            'watched': watched,
            'ok': bool(age is not None and age <= self.stale_sec),
            'age': round(age, 1) if age is not None else None,
            'err': self.last_err,
            'polls': self.polls,
            'errors': self.errors,
        }


def bybit_fetch_all(connector) -> Callable[[], Dict[str, float]]:
    """`{SYMBOL: lastPrice}` для всіх лінійних перпів одним запитом. Іде повз
    `get_tickers` конектора, бо той друкує рядок на кожен виклик (раз на 2с
    це засмічувало б stdout)."""
    def _fetch() -> Dict[str, float]:
        sess = getattr(connector, 'session', None)
        if sess is None:
            return {}
        r = sess.get_tickers(category='linear') or {}
        if int(r.get('retCode', 0) or 0) != 0:
            raise RuntimeError(f"retCode {r.get('retCode')}: {r.get('retMsg')}")
        out = {}
        for t in (r.get('result') or {}).get('list') or []:
            s = t.get('symbol')
            try:
                p = float(t.get('lastPrice') or 0)
            except (TypeError, ValueError):
                continue
            if s and p > 0:
                out[s] = p
        return out
    return _fetch


_instance: Optional[PriceFeed] = None
_inst_lock = threading.Lock()


def get_price_feed(connector=None) -> Optional[PriceFeed]:
    """Синглтон. Створюється з конектора Bybit при першому виклику."""
    global _instance
    if _instance is not None or connector is None:
        return _instance
    with _inst_lock:
        if _instance is None:
            _instance = PriceFeed(bybit_fetch_all(connector))
    return _instance
