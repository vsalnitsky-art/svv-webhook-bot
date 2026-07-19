"""Периодичний лог використання пам'яті процесу (RSS).

Навіщо: на Render (free-tier 512 МБ) процес періодично вбивається OOM-кілером.
OOM = SIGKILL ззовні, його НЕ видно в лозі застосунку (ні traceback, ні
MemoryError). Цей демон раз на інтервал друкує реальний RSS, щоб було видно
РОСТ пам'яті перед кілом і хто/коли її роздуває.

Без нових залежностей: RSS читається з /proc/self/status (Linux = Render),
fallback на stdlib `resource`. Топ-споживачі (tracemalloc) — лише коли явно
ввімкнено env MEM_TRACE=1 (сам tracemalloc додає overhead, тому за замовч. off).

Керування через env:
  MEM_LOG_INTERVAL  секунд між записами (дефолт 60)
  MEM_WARN_MB       поріг попередження, МБ (дефолт 450 — близько до 512)
  MEM_TRACE         '1' → вмикає tracemalloc + топ-10 алокацій
  MEM_TRACE_EVERY   друкувати топ раз на N записів (дефолт 5)

Префікс усіх рядків — [MEM], щоб легко грепати в логах Render.
"""

import os
import sys
import time
import threading

_started = False
_started_pid = None
_lock = threading.Lock()


def _p(msg):
    """print із примусовим flush. Під gunicorn stdout блок-буферизований —
    без flush рядки [MEM] застрягають у буфері й ГУБЛЯТЬСЯ при OOM-кілі
    (SIGKILL не флашить буфер). Саме тому в попередньому лозі бачили лише
    2 перші читання, а перед крахом — жодного. Flush це лікує."""
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _read_rss_mb():
    """Поточний RSS у МБ. /proc (Linux) → psutil → resource(peak). None якщо ніяк."""
    # 1) Linux /proc — точний поточний RSS, найдешевше й без залежностей.
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0   # kB → MB
    except Exception:
        pass
    # 2) psutil, якщо раптом встановлено.
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
    except Exception:
        pass
    # 3) resource — ru_maxrss це ПІК (kB на Linux), не поточне, але краще ніж нічого.
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def _read_cgroup_mem():
    """Памʼять КОНТЕЙНЕРА (cgroup) у МБ: (usage, limit). Саме це міряє Render і
    за цим вбиває OOM. ВАЖЛИВО: cgroup-usage включає page cache (файли, БД,
    mmap) — те, чого VmRSS процесу НЕ показує. Тому RSS може бути 124 МБ, а
    контейнер — 512 МБ. cgroup v2 → v1. None якщо недоступно."""
    # cgroup v2 (сучасний Render)
    try:
        with open('/sys/fs/cgroup/memory.current') as f:
            usage = int(f.read().strip())
        limit = None
        try:
            with open('/sys/fs/cgroup/memory.max') as f:
                v = f.read().strip()
                limit = None if v == 'max' else int(v)
        except Exception:
            pass
        return (usage / 1048576.0, (limit / 1048576.0 if limit else None))
    except Exception:
        pass
    # cgroup v1
    try:
        with open('/sys/fs/cgroup/memory/memory.usage_in_bytes') as f:
            usage = int(f.read().strip())
        limit = None
        try:
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f:
                limit = int(f.read().strip())
                if limit > (1 << 62):     # «безлімітний» sentinel
                    limit = None
        except Exception:
            pass
        return (usage / 1048576.0, (limit / 1048576.0 if limit else None))
    except Exception:
        return (None, None)


def _read_cgroup_cache_mb():
    """Скільки з cgroup-usage — це page cache (файли/БД). cgroup v2: memory.stat
    'file'; v1: 'cache'. Допомагає відрізнити «БД у кеші» від «heap». None якщо ні."""
    for path, key in (('/sys/fs/cgroup/memory.stat', 'file '),
                      ('/sys/fs/cgroup/memory/memory.stat', 'cache ')):
        try:
            with open(path) as f:
                for line in f:
                    if line.startswith(key):
                        return int(line.split()[1]) / 1048576.0
        except Exception:
            continue
    return None


def _proc_rss_table(top=6):
    """Усі процеси cgroup за RSS (МБ) + сумарний RSS. КЛЮЧОВЕ: self-RSS може
    бути малим (126 МБ), а сусідній процес — тримати сотні МБ. Це показує, ЯКИЙ
    саме процес зʼїдає памʼять. Читає /proc/[pid]/status(VmRSS) + /proc/[pid]/comm.
    Повертає (total_rss_mb, [(pid, rss_mb, name), ...top]). (None, []) якщо ні."""
    rows = []
    total = 0.0
    try:
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                rss = None
                with open(f'/proc/{pid}/status') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            rss = int(line.split()[1]) / 1024.0
                            break
                if rss is None:
                    continue
                try:
                    with open(f'/proc/{pid}/comm') as f:
                        name = f.read().strip()
                except Exception:
                    name = '?'
                total += rss
                rows.append((int(pid), rss, name))
            except Exception:
                continue
    except Exception:
        return (None, [])
    rows.sort(key=lambda r: -r[1])
    return (total, rows[:top])


def _gc_summary():
    """Короткий стан gc: к-сть об'єктів + лічильники поколінь."""
    try:
        import gc
        counts = gc.get_count()
        n = len(gc.get_objects())
        return f"gc(obj {n}, coll {counts[0]}/{counts[1]}/{counts[2]})"
    except Exception:
        return "gc(?)"


def _trace_top(limit=10):
    """Топ-N місць коду за розміром алокацій (лише якщо tracemalloc активний)."""
    lines = []
    try:
        import tracemalloc
        if not tracemalloc.is_tracing():
            return lines
        snap = tracemalloc.take_snapshot()
        stats = snap.statistics('lineno')[:limit]
        for i, st in enumerate(stats, 1):
            fr = st.traceback[0]
            mb = st.size / (1024.0 * 1024.0)
            lines.append(f"[MEM]   #{i} {fr.filename}:{fr.lineno}  {mb:.1f} MB ({st.count} блоків)")
    except Exception:
        pass
    return lines


def _monitor_loop(interval, warn_mb, trace, trace_every):
    start_rss = _read_rss_mb()
    cg0, cg_limit = _read_cgroup_mem()
    peak_rss = start_rss or 0.0
    last = start_rss
    tick = 0

    lim_txt = f"{cg_limit:.0f} MB" if cg_limit else "?"
    _p(f"[MEM] monitor старт · RSS {start_rss:.1f} MB · "
       f"cgroup {cg0:.1f}/{lim_txt}" if (start_rss is not None and cg0 is not None)
       else f"[MEM] monitor старт · RSS {start_rss} · cgroup {cg0}")
    _p(f"[MEM] інтервал {interval}s · під-семпл 5s (ловить сплески) · "
       f"поріг {warn_mb} MB · trace={'on' if trace else 'off'}")

    # Під-семплування: заміряємо часто (5s), щоб зловити КОРОТКИЙ сплеск, а
    # друкуємо раз на `interval`, показуючи МАКСИМУМ за вікно. Без цього
    # 60-секундний знімок пропускав різкий стрибок памʼяті перед OOM.
    sub = min(interval, 5)
    win_peak_rss = start_rss or 0.0
    win_peak_cg = cg0 or 0.0
    elapsed = 0

    while True:
        try:
            time.sleep(sub)
            elapsed += sub

            rss = _read_rss_mb()
            cg_usage, _cl = _read_cgroup_mem()
            if _cl:
                cg_limit = _cl
            if rss is not None:
                win_peak_rss = max(win_peak_rss, rss)
                peak_rss = max(peak_rss, rss)
            if cg_usage is not None:
                win_peak_cg = max(win_peak_cg, cg_usage)

            # Раннє попередження ще ДО планового друку — якщо в під-семплі
            # cgroup підскочив до порога, друкуємо негайно (перед можливим кілом).
            near = (cg_usage is not None and cg_limit
                    and cg_usage >= 0.88 * cg_limit)
            if elapsed < interval and not near:
                continue
            elapsed = 0

            if rss is None:
                _p("[MEM] RSS недоступний (немає /proc, psutil, resource)")
                continue
            d_last = (rss - last) if last is not None else 0.0
            d_start = (rss - start_rss) if start_rss is not None else 0.0
            last = rss

            # cgroup — головна метрика (за нею вбиває Render)
            cg_txt = ''
            if cg_usage is not None:
                lim_s = f"/{cg_limit:.0f}" if cg_limit else ""
                cache = _read_cgroup_cache_mb()
                cache_s = f" (cache {cache:.0f})" if cache is not None else ""
                cg_txt = (f" · CGROUP {cg_usage:.1f}{lim_s} MB"
                          f" · cg-peak {win_peak_cg:.1f}{cache_s}")

            flag = ''
            watch = cg_usage if cg_usage is not None else rss
            watch_lim = cg_limit if (cg_usage is not None and cg_limit) else 512
            if watch >= 0.88 * watch_lim:
                flag = f'  ⚠️ БЛИЗЬКО ДО ЛІМІТУ {watch_lim:.0f} MB — імовірний OOM'

            _p(f"[MEM] RSS {rss:.1f} MB · Δ {d_last:+.1f} · Δstart {d_start:+.1f} · "
               f"rss-peak {win_peak_rss:.1f}{cg_txt} · "
               f"{_gc_summary()} · threads {threading.active_count()}{flag}")

            # Хто тримає памʼять: перелік процесів cgroup за RSS. Показує, чи є
            # інший процес (напр. другий gunicorn-worker/subprocess) на сотні МБ,
            # якого self-RSS не бачить. Це закриває питання «де ті ~350 МБ».
            tot, procs = _proc_rss_table()
            if procs:
                plist = ' · '.join(f"{nm}[{pid}] {r:.0f}" for pid, r, nm in procs)
                _p(f"[MEM] procs: сума-RSS {tot:.1f} MB · {plist}")

            win_peak_rss = rss
            win_peak_cg = cg_usage or 0.0
            tick += 1

            if trace and (tick % trace_every == 0):
                top = _trace_top()
                if top:
                    _p("[MEM] топ алокацій Python-heap (tracemalloc):")
                    for ln in top:
                        _p(ln)
        except Exception as e:
            _p(f"[MEM] monitor помилка: {e}")
            time.sleep(sub)


def start_mem_monitor():
    """Запустити демон-монітор пам'яті (ідемпотентно ПО ПРОЦЕСУ). Викликати на
    старті І з worker-процесу (напр. gunicorn post-fork / перший запит).

    ВАЖЛИВО про fork: якщо майстер стартував монітор до fork, дочірній worker
    успадкує `_started=True`, АЛЕ потік монітора у fork НЕ переноситься (живе
    лише в майстрі). Тому звіряємось за PID: у новому процесі стартуємо заново —
    інакше worker (де саме тече памʼять) лишався б без монітора."""
    global _started, _started_pid
    pid = os.getpid()
    with _lock:
        if _started and _started_pid == pid:
            return
        _started = True
        _started_pid = pid

    try:
        interval = max(5, int(os.environ.get('MEM_LOG_INTERVAL', '60')))
    except (TypeError, ValueError):
        interval = 60
    try:
        warn_mb = float(os.environ.get('MEM_WARN_MB', '450'))
    except (TypeError, ValueError):
        warn_mb = 450.0
    trace = os.environ.get('MEM_TRACE', '') in ('1', 'true', 'yes', 'on')
    try:
        trace_every = max(1, int(os.environ.get('MEM_TRACE_EVERY', '5')))
    except (TypeError, ValueError):
        trace_every = 5

    if trace:
        try:
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start(25)   # глибина стеку 25 кадрів
        except Exception:
            trace = False

    # faulthandler: якщо процес падає на жорсткому сигналі (SIGSEGV/SIGABRT/
    # SIGFPE/SIGBUS) — у stderr впаде traceback УСІХ потоків. Це НЕ ловить
    # SIGKILL від OOM (його не ловить ніхто), але покаже C-рівневий креш чи
    # dead-lock, якщо падіння НЕ через памʼять. Дешево, stdlib.
    try:
        import faulthandler
        if not faulthandler.is_enabled():
            faulthandler.enable()
    except Exception:
        pass

    t = threading.Thread(
        target=_monitor_loop,
        args=(interval, warn_mb, trace, trace_every),
        daemon=True,
        name='mem-monitor',
    )
    t.start()
