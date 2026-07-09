"""
🧾 Activity log — a single, bounded, in-memory feed of what the bot DID with
every signal: received → queued (Q1/Q2) → dropped/ejected/skipped → opened →
closed. It answers "чому бот це пропустив/відхилив?" without reading the code.

Design:
  • ONE process-wide singleton (get_activity_log()), safe from all threads.
  • In-memory ring buffer (deque, capped) — cheap, no DB churn. It RESETS on a
    process restart (Render redeploy); only the on/off flag is persisted.
  • Every hook calls .log(...) but nothing is stored while disabled — so the
    monitor has ZERO cost when the operator turns it off.
"""
import time
import threading
from collections import deque
from typing import Optional, List, Dict

_MAX_EVENTS = 1500                       # ring-buffer cap (oldest dropped)
_DB_ENABLED = 'activity_log_enabled'     # persisted on/off flag
_DB_EVENTS = 'activity_log_events'       # persisted event buffer (survives restart)
_FLUSH_EVERY_SEC = 30                    # throttle DB writes (bounded single row)


class ActivityLog:
    def __init__(self):
        self._events = deque(maxlen=_MAX_EVENTS)
        self._lock = threading.RLock()
        self._enabled = None             # lazy-loaded from DB on first touch
        self._db = None
        self._loaded = False             # events restored from DB yet?
        self._dirty = False              # unsaved events since last flush?
        self._last_flush = 0.0
        self._next_id = 1                # monotonic id per event (for delete)

    # ---- DB (lazy) ----
    def _get_db(self):
        if self._db is None:
            try:
                from storage.db_operations import get_db
                self._db = get_db()
            except Exception:
                self._db = None
        return self._db

    def _ensure_loaded(self):
        """Restore the persisted event buffer ONCE (survives a redeploy)."""
        if self._loaded:
            return
        self._loaded = True
        db = self._get_db()
        if not db:
            return
        try:
            raw = db.get_setting(_DB_EVENTS, []) or []
            if isinstance(raw, list):
                with self._lock:
                    for e in raw[-_MAX_EVENTS:]:
                        if isinstance(e, dict):
                            self._events.append(e)
                    # Continue the id sequence past whatever was restored.
                    self._next_id = max((int(e.get('id') or 0) for e in self._events),
                                        default=0) + 1
        except Exception:
            pass

    def _flush(self, force: bool = False):
        """Persist the buffer to DB — throttled, and only when there are new
        events (one bounded row; no unbounded growth)."""
        if not (self._dirty or force):
            return
        now = time.time()
        if not force and (now - self._last_flush) < _FLUSH_EVERY_SEC:
            return
        db = self._get_db()
        if not db:
            return
        self._last_flush = now
        self._dirty = False
        try:
            with self._lock:
                snap = list(self._events)
            db.set_setting(_DB_EVENTS, snap)
        except Exception:
            pass

    def is_enabled(self) -> bool:
        if self._enabled is None:
            db = self._get_db()
            try:
                self._enabled = bool(db.get_setting(_DB_ENABLED, False)) if db else False
            except Exception:
                self._enabled = False
            self._ensure_loaded()
        return bool(self._enabled)

    def set_enabled(self, on: bool) -> bool:
        self._enabled = bool(on)
        self._ensure_loaded()
        db = self._get_db()
        if db:
            try:
                db.set_setting(_DB_ENABLED, self._enabled)
            except Exception:
                pass
        self._flush(force=True)   # persist current buffer on any toggle
        return self._enabled

    # ---- write ----
    def log(self, symbol: str, event: str, detail: str = '',
            side: Optional[str] = None, source: str = '',
            ts: Optional[float] = None) -> None:
        """Append ONE activity event (no-op while disabled).
        event: short key — 'signal' | 'queued' | 'dropped' | 'ejected' |
               'opened' | 'skipped' | 'rejected' | 'closed' | 'passthrough'.
        source: which stage — 'scanner' | 'intercept' | 'Q1' | 'Q2' | 'engine'
               | 'TM'. detail: human reason (UA)."""
        if not self.is_enabled():
            return
        try:
            with self._lock:
                self._events.append({
                    'id': self._next_id,
                    't': ts if ts is not None else time.time(),
                    'symbol': (symbol or '').upper().strip(),
                    'side': side,
                    'event': str(event or ''),
                    'detail': str(detail or '')[:400],
                    'source': str(source or ''),
                })
                self._next_id += 1
                self._dirty = True
            self._flush()   # throttled persist (survives restart)
        except Exception:
            pass   # logging must NEVER break the trading path

    # ---- read ----
    def get(self, limit: int = 400, symbol: Optional[str] = None,
            event: Optional[str] = None) -> List[Dict]:
        """Newest-first list, optionally filtered by symbol / event."""
        self._ensure_loaded()
        with self._lock:
            items = list(self._events)
        if symbol:
            su = symbol.upper()
            items = [e for e in items if e.get('symbol') == su]
        if event:
            items = [e for e in items if e.get('event') == event]
        items = items[-int(limit or 400):]
        items.reverse()
        return items

    def delete(self, ids) -> int:
        """Remove events whose `id` is in `ids`. Returns how many were removed."""
        try:
            idset = {int(i) for i in (ids or [])}
        except (TypeError, ValueError):
            return 0
        if not idset:
            return 0
        removed = 0
        with self._lock:
            kept = [e for e in self._events if int(e.get('id') or -1) not in idset]
            removed = len(self._events) - len(kept)
            if removed:
                self._events.clear()
                self._events.extend(kept)
                self._dirty = True
        if removed:
            self._flush(force=True)
        return removed

    def clear(self) -> int:
        with self._lock:
            n = len(self._events)
            self._events.clear()
            self._dirty = False
        db = self._get_db()
        if db:
            try:
                db.set_setting(_DB_EVENTS, [])
            except Exception:
                pass
        return n

    def count(self) -> int:
        with self._lock:
            return len(self._events)


_INSTANCE: Optional[ActivityLog] = None


def get_activity_log() -> ActivityLog:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ActivityLog()
    return _INSTANCE


def log_activity(symbol: str, event: str, detail: str = '',
                 side: Optional[str] = None, source: str = '') -> None:
    """Module-level convenience wrapper (safe, never raises)."""
    try:
        get_activity_log().log(symbol, event, detail, side=side, source=source)
    except Exception:
        pass
