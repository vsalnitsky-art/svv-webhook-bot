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


class ActivityLog:
    def __init__(self):
        self._events = deque(maxlen=_MAX_EVENTS)
        self._lock = threading.RLock()
        self._enabled = None             # lazy-loaded from DB on first touch
        self._db = None

    # ---- DB (lazy) ----
    def _get_db(self):
        if self._db is None:
            try:
                from storage.db_operations import get_db
                self._db = get_db()
            except Exception:
                self._db = None
        return self._db

    def is_enabled(self) -> bool:
        if self._enabled is None:
            db = self._get_db()
            try:
                self._enabled = bool(db.get_setting(_DB_ENABLED, False)) if db else False
            except Exception:
                self._enabled = False
        return bool(self._enabled)

    def set_enabled(self, on: bool) -> bool:
        self._enabled = bool(on)
        db = self._get_db()
        if db:
            try:
                db.set_setting(_DB_ENABLED, self._enabled)
            except Exception:
                pass
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
                    't': ts if ts is not None else time.time(),
                    'symbol': (symbol or '').upper().strip(),
                    'side': side,
                    'event': str(event or ''),
                    'detail': str(detail or '')[:400],
                    'source': str(source or ''),
                })
        except Exception:
            pass   # logging must NEVER break the trading path

    # ---- read ----
    def get(self, limit: int = 400, symbol: Optional[str] = None,
            event: Optional[str] = None) -> List[Dict]:
        """Newest-first list, optionally filtered by symbol / event."""
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

    def clear(self) -> int:
        with self._lock:
            n = len(self._events)
            self._events.clear()
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
