"""
tickr_opportunity_daemon — continuous Tickr→Watchlist auto-pipeline.

When enabled, every `interval_min` minutes it:
  1) scores Bybit swap-USDT (active-only) coins with the coiled-spring
     opportunity_scan, using the previous run's OI snapshot as the
     OI-acceleration baseline,
  2) any coin scoring >= `min_score` "fires" and is pushed into the SMC
     watchlist with source='tickr' and its score,
  3) coins already in the watchlist as tickr that fire again get their
     24h TTL refreshed (added_ts reset) and score updated,
  4) manual coins are never touched (manual has priority, no TTL),
  5) when near MAX_WATCHLIST, a firing coin stronger than the weakest
     tickr coin evicts that weakest coin,
  6) a separate sweep removes tickr coins whose 24h TTL has elapsed.

Per-coin TTL (each coin lives exactly TTL_SECS from its own added_ts)
is enforced by the scanner (expire_tickr_symbols); this daemon just
calls it each tick.

State persisted in DB so it survives restarts and auto-starts on boot.
DB keys:
  tickr_opp_auto_enabled  : 'true' | 'false'
  tickr_opp_auto_interval : scan interval, minutes
  tickr_opp_auto_minscore : score threshold to fire (0..100)
  tickr_opp_auto_last     : last-run unix ts
  tickr_opp_oi_baseline   : {oi:{sym:usd}, ts} OI snapshot for accel calc
"""

import time
import json
import threading
from typing import Optional, Callable

TTL_SECS = 24 * 3600     # how long a tickr coin lives in the watchlist
CHECK_SECS = 60          # daemon wakes this often; acts when interval elapsed
SWEEP_SECS = 300         # TTL-expiry sweep cadence (independent of scans)

# Tunable bounds (UI sliders clamp to these)
INTERVAL_MIN_DEFAULT = 15
INTERVAL_MIN_LO, INTERVAL_MIN_HI = 5, 120
MINSCORE_DEFAULT = 75.0
MINSCORE_LO, MINSCORE_HI = 50.0, 95.0

_DB_ENABLED = 'tickr_opp_auto_enabled'
_DB_INTERVAL = 'tickr_opp_auto_interval'
_DB_MINSCORE = 'tickr_opp_auto_minscore'
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
        self._last_sweep = 0.0

    # ---- settings ----

    def is_enabled(self) -> bool:
        return str(self._db.get_setting(_DB_ENABLED, 'false')).lower() in ('true', '1')

    def set_enabled(self, on: bool):
        self._db.set_setting(_DB_ENABLED, 'true' if on else 'false')
        if on:
            self.start()

    def interval_min(self) -> int:
        try:
            v = int(float(self._db.get_setting(_DB_INTERVAL, INTERVAL_MIN_DEFAULT)))
        except Exception:
            v = INTERVAL_MIN_DEFAULT
        return max(INTERVAL_MIN_LO, min(INTERVAL_MIN_HI, v))

    def set_interval_min(self, minutes: int):
        v = max(INTERVAL_MIN_LO, min(INTERVAL_MIN_HI, int(minutes)))
        self._db.set_setting(_DB_INTERVAL, v)

    def min_score(self) -> float:
        try:
            v = float(self._db.get_setting(_DB_MINSCORE, MINSCORE_DEFAULT))
        except Exception:
            v = MINSCORE_DEFAULT
        return max(MINSCORE_LO, min(MINSCORE_HI, v))

    def set_min_score(self, score: float):
        v = max(MINSCORE_LO, min(MINSCORE_HI, float(score)))
        self._db.set_setting(_DB_MINSCORE, v)

    # ---- core ----

    def _read_oi_baseline(self) -> dict:
        raw = self._db.get_setting(_DB_OI, '')
        if not raw:
            return {}
        try:
            return json.loads(raw).get('oi', {})
        except Exception:
            return {}

    def _refresh_oi_baseline(self):
        try:
            oi = self._oi_snap()
            if oi:
                self._db.set_setting(_DB_OI, json.dumps({'oi': oi, 'ts': time.time()}))
        except Exception:
            pass

    def _sweep_expired(self):
        """Remove tickr coins past their 24h TTL. Independent of scans so
        coins expire on time even if scanning is sparse."""
        s = self._get_scanner()
        if not s:
            return []
        try:
            return s.expire_tickr_symbols(TTL_SECS)
        except Exception as e:
            print(f"[OppDaemon] sweep error: {e}")
            return []

    def _run_scan(self):
        """One opportunity scan → fire coins over the threshold into the
        watchlist (add / refresh-TTL / evict-weakest as needed)."""
        min_score = self.min_score()
        oi_baseline = self._read_oi_baseline()
        # Scan a generous top slice; the threshold (not top_n) decides who
        # actually fires, so ask for enough headroom.
        res = self._scan({'top_n': 100}, oi_baseline)
        if not res.get('ok'):
            print(f"[OppDaemon] scan failed: {res.get('reason')}")
            self._refresh_oi_baseline()
            self._db.set_setting(_DB_LAST, str(time.time()))
            return

        s = self._get_scanner()
        fired = [r for r in res.get('symbols', [])
                 if r.get('opportunity', 0) >= min_score]
        added, refreshed, evicted, skipped = [], [], [], []

        if s:
            try:
                from detection.smc_scanner import MAX_WATCHLIST
            except Exception:
                MAX_WATCHLIST = 100
            for r in fired:
                sym = r['symbol']
                score = float(r.get('opportunity', 0))
                wl = s.get_watchlist()
                src = s.get_watchlist_sources()
                if sym in wl:
                    if src.get(sym) == 'tickr':
                        # already tracked → extend TTL + update score
                        s.touch_tickr_symbol(sym, score)
                        refreshed.append(sym)
                    else:
                        # manual coin already covers it → leave it alone
                        skipped.append(sym)
                    continue
                # New coin. If at capacity, try to evict the weakest tickr
                # coin that is strictly weaker than this candidate.
                if len(wl) >= MAX_WATCHLIST:
                    ev = s.evict_lowest_tickr(below_score=score)
                    if not ev:
                        skipped.append(sym)   # nobody weaker → can't add
                        continue
                    evicted.append(ev)
                r_add = s.add_symbol(sym, source='tickr', score=score)
                if r_add.get('ok'):
                    added.append(sym)
                else:
                    skipped.append(sym)

        # Refresh OI baseline for the next scan's acceleration calc.
        self._refresh_oi_baseline()
        self._db.set_setting(_DB_LAST, str(time.time()))
        with self._lock:
            self._last_result = {
                'ts': time.time(), 'min_score': min_score,
                'fired': len(fired), 'added': added, 'refreshed': refreshed,
                'evicted': evicted, 'skipped': len(skipped),
                'scanned': res.get('count_in', 0),
            }
        if added or refreshed or evicted:
            print(f"[OppDaemon] scan: +{len(added)} added, "
                  f"{len(refreshed)} refreshed, {len(evicted)} evicted "
                  f"(threshold {min_score}, {len(fired)} fired)")

    def _run(self):
        self._stop.wait(12)
        while not self._stop.is_set():
            try:
                if self.is_enabled():
                    now = time.time()
                    # TTL sweep on its own cadence (so coins expire on time)
                    if now - self._last_sweep >= SWEEP_SECS:
                        self._sweep_expired()
                        self._last_sweep = now
                    # Opportunity scan on the configured interval
                    last = float(self._db.get_setting(_DB_LAST, '0') or 0)
                    if now - last >= self.interval_min() * 60:
                        self._run_scan()
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
        """Manual trigger (ignores the interval timer)."""
        self._sweep_expired()
        self._run_scan()
        with self._lock:
            return dict(self._last_result)

    def status(self) -> dict:
        with self._lock:
            last_result = dict(self._last_result)
        last_ts = float(self._db.get_setting(_DB_LAST, '0') or 0)
        # Surface the live tickr watchlist with TTL remaining for the UI.
        active = []
        s = self._get_scanner()
        if s:
            try:
                meta = s.get_watchlist_meta()   # {sym: {ts, score}}
                now = time.time()
                for sym, m in meta.items():
                    ttl_left = max(0, TTL_SECS - int(now - float(m.get('ts', now))))
                    active.append({
                        'symbol': sym,
                        'score': round(float(m.get('score', 0)), 1),
                        'added_ts': float(m.get('ts', 0)),
                        'ttl_left_secs': ttl_left,
                    })
                active.sort(key=lambda a: a['score'], reverse=True)
            except Exception:
                active = []
        return {
            'enabled': self.is_enabled(),
            'interval_min': self.interval_min(),
            'min_score': self.min_score(),
            'ttl_hours': TTL_SECS // 3600,
            'running': bool(self._thread and self._thread.is_alive()),
            'last_run_ts': last_ts,
            'last_result': last_result,
            'active': active,
            'active_count': len(active),
        }


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
