"""
auto_gate — server-side driver for the Smart Money "Auto-gate" toggle.

When enabled, a background thread periodically computes the trade-bias
verdict (LONG / SHORT / WAIT) for the configured symbol and applies it to
the Trade Manager side gates:
    LONG  → allow_long_entries=True,  allow_short_entries=False
    SHORT → allow_short_entries=True, allow_long_entries=False
    WAIT  → both False (no new entries while direction is unclear)

This makes Auto-gate work 24/7 without a browser open: the toggle state
and the chosen symbol are persisted in DB, restored on boot, and the loop
re-applies the verdict on every cycle.

The verdict itself is computed by web.flask_app's bias logic, exposed via
compute_bias() so both the HTTP endpoint and this loop share one source
of truth. The watchlist-consensus component is a FRONTEND-pushed cache
(markers live in the browser); when the browser is closed the loop uses
the last cached consensus if fresh, otherwise omits that vote.

DB keys:
  sm_auto_gate_enabled : 'true' | 'false'
  sm_bias_symbol       : symbol for the verdict (default BTCUSDT)
  sm_wl_consensus_cache: JSON {src, n_long, n_short, n_flat, ts} (frontend)
"""

import time
import json
import threading
from typing import Optional, Callable

CYCLE_SECS = 60          # match the UI refresh cadence
WL_CACHE_TTL = 300       # frontend consensus older than 5 min is ignored

_DB_ENABLED = 'sm_auto_gate_enabled'
_DB_SYMBOL = 'sm_bias_symbol'
_DB_WL_CACHE = 'sm_wl_consensus_cache'
_DB_CLOSE_ON_WAIT = 'sm_auto_gate_close_on_wait'


class AutoGateDaemon:
    def __init__(self, db, compute_bias: Callable, get_trade_manager: Callable):
        self._db = db
        self._compute_bias = compute_bias          # (symbol, wl_consensus) -> dict
        self._get_tm = get_trade_manager
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last = {'verdict': None, 'applied_at': 0, 'ts': 0}
        self._lock = threading.Lock()

    # --- persisted state ---
    def is_enabled(self) -> bool:
        return str(self._db.get_setting(_DB_ENABLED, 'false')).lower() in ('true', '1')

    def set_enabled(self, on: bool):
        self._db.set_setting(_DB_ENABLED, 'true' if on else 'false')
        if on:
            self.start()
            try:
                self._tick()
            except Exception as e:
                print(f"[AutoGate] immediate apply error: {e}")

    def is_close_on_wait(self) -> bool:
        return str(self._db.get_setting(_DB_CLOSE_ON_WAIT, 'false')).lower() in ('true', '1')

    def set_close_on_wait(self, on: bool):
        self._db.set_setting(_DB_CLOSE_ON_WAIT, 'true' if on else 'false')

    def get_symbol(self) -> str:
        return (self._db.get_setting(_DB_SYMBOL, 'BTCUSDT') or 'BTCUSDT').upper()

    def _wl_consensus(self) -> Optional[dict]:
        """Frontend-pushed consensus, if fresh enough to trust."""
        raw = self._db.get_setting(_DB_WL_CACHE, '')
        if not raw:
            return None
        try:
            c = json.loads(raw)
            if time.time() - c.get('ts', 0) > WL_CACHE_TTL:
                return None      # stale — browser closed too long ago
            return c
        except Exception:
            return None

    # --- gate application ---
    def _apply(self, verdict: str):
        tm = self._get_tm()
        if not tm:
            return
        if verdict == 'LONG':
            allow_long, allow_short = True, False
        elif verdict == 'SHORT':
            allow_long, allow_short = False, True
        else:                       # WAIT → both off
            allow_long, allow_short = False, False
        # Only POST when something actually changes (avoid log/DB churn)
        s = tm.get_settings() if hasattr(tm, 'get_settings') else {}
        cur_l = bool(s.get('allow_long_entries', True))
        cur_s = bool(s.get('allow_short_entries', True))
        if cur_l == allow_long and cur_s == allow_short:
            return
        tm.update_settings({'allow_long_entries': allow_long,
                            'allow_short_entries': allow_short})
        print(f"[AutoGate] verdict={verdict} → LONG={allow_long} SHORT={allow_short}")

    def _tick(self):
        if not self.is_enabled():
            return
        symbol = self.get_symbol()
        wl = self._wl_consensus()
        verdict_data = self._compute_bias(symbol, wl)
        verdict = verdict_data.get('verdict', 'WAIT')
        with self._lock:
            prev_verdict = self._last.get('verdict')
            self._last = {'verdict': verdict, 'ts': time.time(),
                          'applied_at': time.time(), 'symbol': symbol,
                          'confidence': verdict_data.get('confidence', 0)}
        self._apply(verdict)
        # Close-on-WAIT: only on the TRANSITION into WAIT (not every tick
        # while already waiting), and only when the toggle is on. Closing
        # uses the throttled queue so the exchange isn't overwhelmed.
        if (verdict == 'WAIT' and prev_verdict not in (None, 'WAIT')
                and self.is_close_on_wait()):
            try:
                tm = self._get_tm()
                if tm and hasattr(tm, 'close_all_with_queue'):
                    res = tm.close_all_with_queue(reason='auto_gate_wait')
                    print(f"[AutoGate] WAIT transition → closed "
                          f"real={len(res.get('closed_real', []))} "
                          f"shadow={len(res.get('closed_shadow', []))}")
            except Exception as e:
                print(f"[AutoGate] close-on-WAIT error: {e}")

    def _run(self):
        # small initial delay so other singletons finish booting
        self._stop.wait(8)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[AutoGate] tick error: {e}")
            self._stop.wait(CYCLE_SECS)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='auto-gate')
        self._thread.start()
        print("[AutoGate] daemon started")

    def status(self) -> dict:
        with self._lock:
            last = dict(self._last)
        return {'enabled': self.is_enabled(), 'symbol': self.get_symbol(),
                'close_on_wait': self.is_close_on_wait(),
                'running': bool(self._thread and self._thread.is_alive()),
                'last': last}


_instance: Optional[AutoGateDaemon] = None


def init_auto_gate(db, compute_bias, get_trade_manager) -> AutoGateDaemon:
    """Create the singleton and auto-start the loop if the persisted
    toggle is ON (so it survives bot restarts)."""
    global _instance
    if _instance is None:
        _instance = AutoGateDaemon(db, compute_bias, get_trade_manager)
        if _instance.is_enabled():
            _instance.start()
            print("[AutoGate] restored ON state from DB — loop running")
    return _instance


def get_auto_gate() -> Optional[AutoGateDaemon]:
    return _instance
