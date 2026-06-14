"""
tickr_opportunity_daemon — optional daily auto-rotation of the SMC
watchlist from the Tickr opportunity scan.

When enabled, once per day it:
  1) captures a fresh OI baseline,
  2) (after the configured interval, on the NEXT run) scores coins with
     the coiled-spring opportunity_scan using that baseline as the
     OI-acceleration reference,
  3) replaces the SMC watchlist with the top-N opportunities.

The two-step baseline→score is why OI acceleration only becomes
meaningful from the second daily run onward; the first run seeds it.

State persisted in DB so it survives restarts and auto-starts on boot.
DB keys:
  tickr_opp_auto_enabled : 'true' | 'false'
  tickr_opp_auto_topn    : int
  tickr_opp_auto_last     : last-run unix ts
"""

import time
import json
import threading
from typing import Optional, Callable

DAY_SECS = 24 * 3600
CHECK_SECS = 1800        # wake every 30 min, act when a day has passed

_DB_ENABLED = 'tickr_opp_auto_enabled'
_DB_TOPN = 'tickr_opp_auto_topn'
_DB_LAST = 'tickr_opp_auto_last'
_DB_OI = 'tickr_opp_oi_baseline'


class OpportunityDaemon:
    def __init__(self, db, scan_fn: Callable, oi_snapshot_fn: Callable,
                 get_scanner: Callable):
        self._db = db
        self._scan = scan_fn               # (params, oi_baseline) -> result
        self._oi_snap = oi_snapshot_fn     # () -> {sym: oi_usd}
        self._get_scanner = get_scanner
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_result = {}

    def is_enabled(self) -> bool:
        return str(self._db.get_setting(_DB_ENABLED, 'false')).lower() in ('true', '1')

    def set_enabled(self, on: bool):
        self._db.set_setting(_DB_ENABLED, 'true' if on else 'false')
        if on:
            self.start()

    def top_n(self) -> int:
        try:
            return max(5, min(50, int(self._db.get_setting(_DB_TOPN, '20'))))
        except Exception:
            return 20

    def _run_rotation(self):
        # 1) read existing OI baseline (seeded on a prior run)
        raw = self._db.get_setting(_DB_OI, '')
        oi_baseline = {}
        if raw:
            try:
                oi_baseline = json.loads(raw).get('oi', {})
            except Exception:
                oi_baseline = {}
        # 2) score with coiled-spring logic
        res = self._scan({'top_n': self.top_n()}, oi_baseline)
        if not res.get('ok'):
            print(f"[OppDaemon] scan failed: {res.get('reason')}")
            return
        symbols = [r['symbol'] for r in res.get('symbols', [])]
        # 3) replace watchlist
        s = self._get_scanner()
        if s and symbols:
            try:
                for cur in list(s.get_watchlist()):
                    if cur not in symbols:
                        s.remove_symbol(cur)
                for sym in symbols:
                    s.add_symbol(sym)
                print(f"[OppDaemon] watchlist rotated → {len(symbols)} coins")
            except Exception as e:
                print(f"[OppDaemon] watchlist update error: {e}")
        # 4) refresh OI baseline for the NEXT run's acceleration calc
        try:
            oi = self._oi_snap()
            self._db.set_setting(_DB_OI, json.dumps({'oi': oi, 'ts': time.time()}))
        except Exception:
            pass
        self._db.set_setting(_DB_LAST, str(time.time()))
        with self._lock:
            self._last_result = {'ts': time.time(), 'count': len(symbols),
                                 'symbols': symbols}

    def _run(self):
        self._stop.wait(12)
        while not self._stop.is_set():
            try:
                if self.is_enabled():
                    last = float(self._db.get_setting(_DB_LAST, '0') or 0)
                    if time.time() - last >= DAY_SECS:
                        self._run_rotation()
            except Exception as e:
                print(f"[OppDaemon] loop error: {e}")
            self._stop.wait(CHECK_SECS)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='tickr-opp')
        self._thread.start()
        print("[OppDaemon] started")

    def run_now(self) -> dict:
        """Manual trigger (ignores the daily timer)."""
        self._run_rotation()
        with self._lock:
            return dict(self._last_result)

    def status(self) -> dict:
        with self._lock:
            last_result = dict(self._last_result)
        last_ts = float(self._db.get_setting(_DB_LAST, '0') or 0)
        return {'enabled': self.is_enabled(), 'top_n': self.top_n(),
                'running': bool(self._thread and self._thread.is_alive()),
                'last_run_ts': last_ts, 'last_result': last_result}


_instance: Optional[OpportunityDaemon] = None


def init_opp_daemon(db, scan_fn, oi_snapshot_fn, get_scanner) -> OpportunityDaemon:
    global _instance
    if _instance is None:
        _instance = OpportunityDaemon(db, scan_fn, oi_snapshot_fn, get_scanner)
        if _instance.is_enabled():
            _instance.start()
            print("[OppDaemon] restored ON state from DB")
    return _instance


def get_opp_daemon() -> Optional[OpportunityDaemon]:
    return _instance
