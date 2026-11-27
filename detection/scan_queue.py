"""
scan_queue — ОДНА СПІЛЬНА ЧЕРГА НА ВСІ СКАНИ БІРЖІ (вимога 18.09).

**Дослівно:** «Організуй всі скани біржі по графіку, щоб не навантажувати
біржу, кожен скан в свою чергу.»

Проблема була не в окремому скані, а в ЇХ ЗБІГУ. Кожен із них — це десятки
HTTP за секунди (скан ліквідності: 1 запит тікерів + 1-2 на монету × до 200
монет), і коли два таких стартували одночасно — ручний зі сторінки 📡 Tickr і
періодичний 💧 Сканера, — біржа отримувала подвійний залп, а ми ризикували
418/429 рівно тоді, коли дані потрібні найбільше.

**Правило: за раз працює РІВНО ОДИН скан.** Решта стоїть у черзі й заходить
по порядку, з паузою `MIN_GAP_SEC` між сусідніми — щоб навіть безперервний
потік завдань не перетворювався на суцільний залп.

Ключові рішення:
  • **ДЕДУП ЗА ІМЕНЕМ.** Другий запит того самого скану, поки перший ще чекає
    АБО ВЖЕ ПРАЦЮЄ, НЕ стає в чергу окремо — він чекає на ТОЙ САМИЙ результат.
    Інакше нетерплячий подвійний клік по «Сканувати» коштував би подвійного
    залпу, а дав би ті самі числа.
  • **ЧЕРГА НІЧОГО НЕ ЗНАЄ ПРО СКАНИ.** Вона приймає `name` + функцію; хто і
    що саме качає — справа виклику. Тому сюди однаково лягають і скан
    ліквідності, і будь-який майбутній.
  • **ВИНЯТОК НЕ ЛАМАЄ ЧЕРГУ.** Він ловиться, віддається тому, хто чекав, і
    записується в історію; наступне завдання йде як звичайно.
  • **СТАН ВИДИМИЙ** (`state()`): що працює зараз, скільки чекає, чим
    закінчився кожен скан. Мовчазна черга читалась би як «кнопка не працює»
    (той самий урок, що з мовчазними clamp-ами у скані ліквідності).

⚠️ Сюди СВІДОМО не заводяться безперервні цикли — SMC-сканер і демон liq-map.
Вони не «скани на вимогу», а робочий такт бота: поставити їх у спільну чергу
означало б віддати ЇМ у заручники торгову логіку (ціни для закриття позицій
беруться саме з того циклу). Черга для одноразових і періодичних сканів.
"""

import threading
import time
from typing import Callable, Dict, List, Optional

# Пауза МІЖ сусідніми сканами: навіть коли черга повна, біржа отримує роботу
# порціями, а не суцільним потоком.
MIN_GAP_SEC = 5.0
# Скільки чекати на результат за замовчуванням (скан 200 монет × 1000 барів
# реально йде до кількох хвилин).
DEFAULT_TIMEOUT = 300.0
_HISTORY_CAP = 25


class _Job:
    """Одне завдання в черзі. `wait()` віддає результат тому, хто його чекає."""

    __slots__ = ('name', 'fn', 'source', 'queued_at', 'started_at', 'done_at',
                 'result', 'error', 'done')

    def __init__(self, name: str, fn: Callable, source: str = ''):
        self.name = name
        self.fn = fn
        self.source = source
        self.queued_at = time.time()
        self.started_at = 0.0
        self.done_at = 0.0
        self.result = None
        self.error: Optional[str] = None
        self.done = threading.Event()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self.done.wait(timeout)

    def info(self) -> Dict:
        return {'name': self.name, 'source': self.source,
                'queued_at': self.queued_at, 'started_at': self.started_at,
                'done_at': self.done_at, 'error': self.error,
                'waited_sec': round((self.started_at or time.time())
                                    - self.queued_at, 1),
                'took_sec': round((self.done_at - self.started_at), 1)
                if self.done_at and self.started_at else None}


class ScanQueue:
    def __init__(self):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._q: List[_Job] = []
        self._cur: Optional[_Job] = None
        self._last: Dict[str, Dict] = {}     # name → чим закінчився останній
        self._hist: List[Dict] = []
        self._last_end = 0.0
        self._thread: Optional[threading.Thread] = None
        self._total = 0

    # ── публічне ────────────────────────────────────────────────────────
    def submit(self, name: str, fn: Callable, source: str = '') -> _Job:
        """Поставити скан у чергу. Той самий `name`, що ВЖЕ чекає **або вже
        ПРАЦЮЄ**, другим завданням не стає — повертається той самий job.

        ⚠️ Дедуп мусить накривати і ТОГО, ХТО ПРАЦЮЄ. Спершу він дивився лише
        на чергу — і подвійний клік по «Сканувати» встигав проскочити: перший
        запит воркер уже забрав, у черзі порожньо, другий ставав НОВИМ
        завданням, тобто давав рівно той подвійний залп, від якого черга й
        заводилась (упіймано тестом).
        """
        with self._cv:
            cur = self._cur
            if cur is not None and cur.name == name and not cur.done.is_set():
                return cur
            for j in self._q:
                if j.name == name:
                    return j
            job = _Job(name, fn, source)
            self._q.append(job)
            self._cv.notify_all()
        self._ensure_worker()
        return job

    def run(self, name: str, fn: Callable, source: str = '',
            timeout: float = DEFAULT_TIMEOUT) -> Dict:
        """Синхронний виклик для кнопок і маршрутів: стати в чергу й дочекатись.

        Повертає результат самої функції (як є) або
        `{'ok': False, 'reason': ...}`, якщо скан упав чи не вклався в час.
        ⚠️ Відповідь ЗАВЖДИ каже, скільки довелось чекати черги — інакше
        «кнопка думає 40 секунд» виглядало б як гальма самого скану.
        """
        job = self.submit(name, fn, source)
        if not job.wait(timeout):
            return {'ok': False, 'queued': True,
                    'reason': f'скан «{name}» не вклався у {int(timeout)}с — '
                              f'він лишається в черзі, спробуйте пізніше'}
        if job.error:
            return {'ok': False, 'reason': job.error,
                    'queue_wait_sec': job.info().get('waited_sec')}
        res = job.result
        if isinstance(res, dict):
            res = dict(res)
            res['queue_wait_sec'] = job.info().get('waited_sec')
        return res

    def busy(self) -> bool:
        with self._lock:
            return bool(self._cur) or bool(self._q)

    def state(self) -> Dict:
        with self._lock:
            cur = self._cur.info() if self._cur else None
            return {
                'running': cur,
                'queued': [j.info() for j in self._q],
                'depth': len(self._q) + (1 if self._cur else 0),
                'last': {k: dict(v) for k, v in self._last.items()},
                'history': list(self._hist),
                'total': self._total,
                'min_gap_sec': MIN_GAP_SEC,
            }

    # ── воркер ──────────────────────────────────────────────────────────
    def _ensure_worker(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name='scan-queue')
            self._thread.start()

    def _loop(self):
        while True:
            with self._cv:
                while not self._q:
                    # Порожня черга — воркер просто спить; він єдиний і живе
                    # весь час процесу (створюється лише раз).
                    self._cv.wait(30)
                job = self._q.pop(0)
                self._cur = job
                gap = MIN_GAP_SEC - (time.time() - self._last_end)
            if gap > 0:
                time.sleep(gap)
            job.started_at = time.time()
            try:
                job.result = job.fn()
            except Exception as e:                    # скан упав — черга живе
                job.error = f'{type(e).__name__}: {e}'
                print(f"[ScanQueue] {job.name} error: {job.error}")
            job.done_at = time.time()
            with self._lock:
                self._cur = None
                self._last_end = job.done_at
                self._total += 1
                info = job.info()
                self._last[job.name] = info
                self._hist.insert(0, info)
                del self._hist[_HISTORY_CAP:]
            job.done.set()


_instance: Optional[ScanQueue] = None
_init_lock = threading.Lock()


def get_scan_queue() -> ScanQueue:
    """Синглтон. Черга ОДНА на процес — інакше «за раз один скан» не
    гарантувалось би нічим."""
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = ScanQueue()
    return _instance


def run(name: str, fn: Callable, source: str = '',
        timeout: float = DEFAULT_TIMEOUT) -> Dict:
    """Короткий шлях: `scan_queue.run('tickr:liq', lambda: ...)`."""
    return get_scan_queue().run(name, fn, source, timeout)


def submit(name: str, fn: Callable, source: str = '') -> _Job:
    return get_scan_queue().submit(name, fn, source)


def state() -> Dict:
    return get_scan_queue().state()
