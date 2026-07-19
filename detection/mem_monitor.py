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
import time
import threading

_started = False
_lock = threading.Lock()


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
    peak = start_rss or 0.0
    last = start_rss
    tick = 0
    print(f"[MEM] monitor старт · RSS {start_rss:.1f} MB · інтервал {interval}s · "
          f"поріг {warn_mb} MB · trace={'on' if trace else 'off'}"
          if start_rss is not None else "[MEM] monitor старт · RSS невідомий")

    while True:
        try:
            time.sleep(interval)
            tick += 1
            rss = _read_rss_mb()
            if rss is None:
                print("[MEM] RSS недоступний (немає /proc, psutil, resource)")
                continue
            if rss > peak:
                peak = rss
            d_last = (rss - last) if last is not None else 0.0
            d_start = (rss - start_rss) if start_rss is not None else 0.0
            last = rss

            flag = ''
            if rss >= warn_mb:
                flag = '  ⚠️ БЛИЗЬКО ДО ЛІМІТУ 512 MB — імовірний OOM'

            print(f"[MEM] RSS {rss:.1f} MB · Δ{interval}s {d_last:+.1f} · "
                  f"Δstart {d_start:+.1f} · peak {peak:.1f} · "
                  f"{_gc_summary()} · threads {threading.active_count()}{flag}")

            if trace and (tick % trace_every == 0):
                top = _trace_top()
                if top:
                    print("[MEM] топ алокацій (tracemalloc):")
                    for ln in top:
                        print(ln)
        except Exception as e:
            try:
                print(f"[MEM] monitor помилка: {e}")
            except Exception:
                pass
            time.sleep(interval)


def start_mem_monitor():
    """Запустити демон-монітор пам'яті (ідемпотентно). Викликати один раз на старті."""
    global _started
    with _lock:
        if _started:
            return
        _started = True

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

    t = threading.Thread(
        target=_monitor_loop,
        args=(interval, warn_mb, trace, trace_every),
        daemon=True,
        name='mem-monitor',
    )
    t.start()
