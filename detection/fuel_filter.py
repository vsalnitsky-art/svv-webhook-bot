"""
fuel_filter — server-side automated trading filter driven by liquidation
"fuel" direction.

Idea (operator spec):
  • Scan every WATCHLIST coin.
  • When a coin shows "Паливо зверху (тягне в LONG)" (fuel_dir > +0.1) or
    the opposite "Паливо знизу (тягне в SHORT)" (fuel_dir < -0.1) a per-coin
    TIMER starts.
  • If that same status holds continuously for ≥ duration_minutes
    (default 5) we OPEN a position in the fuel direction.
  • The position stays open until EITHER:
       – the fuel status changes / disappears (timer resets) → CLOSE, or
       – (optional, toggled) the move "Потенціал"/exhaustion reaches
         potential_threshold_pct (default 95%) → CLOSE (take the move
         before it reverses).

Safety / exchange-friendliness:
  • Fuel direction is read from the *cached* liquidation map state
    (refreshed by its own daemon every 60 s) — this module makes ZERO extra
    exchange calls to compute fuel.
  • IMPORTANT: the liq-map daemon only tracks BACKGROUND_SYMBOLS (BTC/ETH)
    plus symbols requested on-demand in the last 30 min. So each tick we
    call lm.request_symbol() for every WATCHLIST coin — this is what makes
    the daemon actually scan ALL watchlist coins (not just the ones a user
    happens to be viewing in the UI). First-seen coins need ~1-2 liq-map
    ticks (≤2 min) before fuel data becomes meaningful.
  • Exhaustion ("Потенціал") needs klines, so it is computed ONLY for coins
    that already have an OPEN fuel position (a handful at most), never for
    the whole watchlist. Results are cached per-symbol with a short TTL.
  • The scan cycle is 30 s — twice per liq-map refresh, which is plenty.

Persistence / recovery:
  • All live state (per-coin timers, open positions, recent closes) and all
    settings are persisted to the DB as JSON settings, mirroring how the
    Trade Manager persists its book. On boot the daemon restores everything
    so timers and positions survive restarts/redeploys.

Execution modes:
  • Positions are opened via Trade Manager (real Bybit) or Test Mode (paper)
    based on their toggle states — this filter doesn't maintain its own mode.
  • If TM is enabled → real position via manual_open()
  • If Test Mode is enabled → paper position via _open_shadow()
  • If both are on, real TM takes precedence.
"""

import time
import threading
from typing import Optional, Callable, Dict, List

CYCLE_SECS = 30                 # scan cadence (twice per liq-map refresh)
EXHAUSTION_TTL = 120            # cache exhaustion per symbol for 2 min
BIAS_TTL = 10                   # cache compute_bias result per symbol (sec)
FUEL_LONG_THR = 0.1            # fuel_dir > +0.1 → LONG bias
FUEL_SHORT_THR = -0.1          # fuel_dir < -0.1 → SHORT bias
CLOSED_LIMIT = 100             # keep last N closes for the UI
# Grace period before closing on FUEL FADE (status → neutral/None). Without
# this, a single transient liq-map data gap or a brief dip into the ±0.1
# neutral zone would slam the position shut the very next tick. We only honour
# a *sustained* loss of fuel. A clear FLIP to the opposite side still closes
# immediately (that's a real reversal, not a data gap).
FUEL_FADE_GRACE_SEC = 180      # 3 min of continuous neutral before close
# Minimum time a freshly-opened position is held before ANY auto-exit
# (flip / fade / wait / exhaustion) may fire. A manual force-open takes the
# timer's side regardless of where fuel currently points, so without this
# grace the very next tick could see fuel on the opposite side and close the
# trade instantly ("opened → immediately in Recent Closed Trades").
MIN_HOLD_AFTER_OPEN_SEC = 90

_DB_SETTINGS = 'fuel_filter_settings'
_DB_STATE = 'fuel_filter_state'
_DB_SCAN_LIST = 'fuel_filter_scan_list'   # which symbols FF is allowed to scan

# ─── queue (❤️ Черга на вхід) removal operations ───
# Every place that REMOVES a coin from _pending is numbered and guarded. These
# were temporarily frozen (all False) to diagnose coins vanishing from the
# queue. The real culprit was the queue being RESTORED from the DB on boot
# (stale coins like AAVEUSDT reappearing without a fresh signal) — now fixed by
# making the queue ephemeral in _load_state. All removal ops are verified
# correct, so they are re-enabled (True). Flip a value to False to re-freeze a
# single op for future debugging.
#   1 = position closed (_close)
#   2 = engine opened the trade (_engine_tick success)
#   3 = engine: TM already holds the coin (_engine_tick)
#   4 = ММ-flip purge of opposite-direction entries (_update_btc_verdict)
#   5 = manual delete ✕ (delete_timer)
#   6 = manual "Очистити всі" (clear_all_timers)
#   7 = remove_pending()
#   8 = clear_pending()
#   9 = manual force-open (force_open_timer success)
_QUEUE_OPS_ALLOWED = {1: True, 2: True, 3: True, 4: True, 5: True,
                      6: True, 7: True, 8: True, 9: True}


def _q_allowed(op: int) -> bool:
    """True if queue-removal op #op is currently allowed (debug gate)."""
    if _QUEUE_OPS_ALLOWED.get(op, True):
        return True
    print(f"[FF-QUEUE] op#{op} BLOCKED — removal suppressed (coin kept in queue)")
    return False

DEFAULT_SETTINGS = {
    'enabled': False,
    'duration_minutes': 5,        # min duration to show in table (threshold filter)
    'potential_threshold_pct': 95,  # exhaustion ≥ this → close
    'use_potential_exit': True,   # toggle the exhaustion exit on/off
    'max_exhaustion_pct': 75,     # engine/open GATE: skip a coin if its move is
                                  # more exhausted than this % (0..100). Active.
    'skip_wait_coins': False,     # (legacy) not used for auto-open anymore
    'manage_open_positions': True,  # if True, FF closes positions it opened
    'direction_smoothing_min': 0,   # EMA window (min) for ММ direction; 0 = OFF (raw)
    'anomaly_hours': 10,            # fuel held longer than this → "anomaly" list
    'start_signal_minutes': 5,      # BTC ММ held ≥ this → START signal (else STOP)
    # ── BTC-START auto-engine (banner toggle) ──
    'start_engine_enabled': False,        # master: auto-open basket on BTC START
    'start_engine_independent': False,    # auto-open independent of BTC (own dir)
    'start_engine_use_anomalies': True,   # source: 🜂 anomalies table
    'start_engine_use_timers': True,      # source: ⏱️ active timers table
    'start_engine_scan_secs': 15,         # engine scan cadence (sec)
    'start_engine_include_funding': False,# include 💰 funding-marked coins
    'engine_candle_confirm': True,        # only open when candles confirm dir (2/2)
    'engine_candle_tf': '5m',             # timeframe for the candle confirmation
    'engine_require_strong_hold': False,  # only open when SCORE=STRONG HOLD & dir matches
    'start_signal_tg_alerts': False,      # Telegram alert on BTC START/STOP change
    'funding_duration_minutes': 0,        # separate show-threshold for 💰 funding coins
    'funding_tg_alerts': False,           # Telegram alert when a funding coin enters the table
}


class FuelFilterDaemon:
    def __init__(self, db, get_trade_manager: Callable,
                 get_watchlist: Callable):
        self._db = db
        self._get_tm = get_trade_manager
        self._get_watchlist = get_watchlist
        self._thread: Optional[threading.Thread] = None
        self._engine_thread: Optional[threading.Thread] = None
        self._alert_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        # live state (restored from DB on boot)
        self._timers: Dict[str, Dict] = {}      # symbol -> {dir, since}
        # Tracking dict: which symbols fuel filter opened (not full position data)
        self._fuel_managed: Dict[str, Dict] = {}  # symbol -> {opened_at, side, fuel_dir}
        # exhaustion cache: symbol -> {ts, exhaustion, side}
        self._exh_cache: Dict[str, Dict] = {}
        # compute_bias cache: symbol -> {ts, data}. Shared by exhaustion,
        # verdict and panel-status so we call the (heavy) shared bias
        # computation at most once per symbol per BIAS_TTL window.
        self._bias_cache: Dict[str, Dict] = {}
        # Aggregate scan stats for the panel header: how many scan-targets were
        # scanned, watchlist size, and average move exhaustion (LONG/SHORT).
        # Recomputed each tick.
        self._scan_stats: Dict = {}
        # Pre-computed SCORE verdicts {sym: dict}, refreshed in the background
        # _tick so the UI endpoints (FF state + TM state) read them instantly
        # instead of doing per-row liq-map / kline-fetch work in the request
        # path (that made the page slow to open).
        self._score_cache: Dict[str, Dict] = {}
        # Fuel STRENGTH (0..100) per symbol — current cycle and previous cycle,
        # so the UI can show the value + a rising/falling trend. Refreshed once
        # per cycle in _refresh_score_cache (cheap; read by both tables).
        self._fuel_str: Dict[str, int] = {}
        self._fuel_str_prev: Dict[str, int] = {}
        # Symbols pulled in from the 💰 Funding Rate Scanner (when it's enabled).
        # They get fuel timers + a row in the ❤️ table, flagged distinctly, but
        # are MONITOR-ONLY (no auto-open / management). Refreshed each tick.
        self._funding_syms: set = set()
        # {symbol: current funding %} for funding-sourced coins (UI display).
        self._funding_rates: Dict[str, float] = {}
        # {symbol: nextFundingTime ms} — countdown to next funding settlement.
        self._funding_next: Dict[str, int] = {}
        # {symbol: 24h volume USD} for funding-sourced coins (UI display).
        self._funding_vols: Dict[str, float] = {}
        # Short-TTL klines cache for candle confirmation at a custom TF:
        # {(symbol, tf): (ts, klines)}.
        self._candle_cache: Dict = {}
        # {symbol: count} — how many times the engine tried to open this coin but
        # the candle confirmation failed (it stayed in the table waiting).
        self._engine_attempts: Dict[str, int] = {}
        # Direction smoothing state (anti-twitch): EMA of the raw liq-fuel
        # imbalance per symbol, plus the hysteresis direction latch. Advanced
        # ONCE per scan cycle in _tick; everyone else reads it. Persisted so
        # smoothing survives a restart instead of snapping back to raw.
        self._fuel_ema: Dict[str, float] = {}
        self._fuel_hyst: Dict[str, Optional[str]] = {}  # sym → 'LONG'/'SHORT'/None
        # Last BTC START/STOP state we sent a Telegram alert for (None until set).
        self._btc_start_last_alert: Optional[str] = None
        # Last live BTC direction — reported in the STOP message.
        self._btc_last_dir: Optional[str] = None
        # Funding coins already TG-alerted as "entered the table" (alert once).
        self._funding_alerted: set = set()
        # Latest BTC fuel snapshot — pinned permanently at the top of the table.
        self._btc_state: Dict = {}
        # ── BTC ММ "session" ──────────────────────────────────────────────
        # A session = a committed BTC ММ direction (LONG/SHORT). WAIT (ML
        # балансований) is a PAUSE — it keeps the session, its start time and
        # the queue. Only an OPPOSITE flip (LONG↔SHORT) ends the session, resets
        # the timer and CLEARS the queue. `_btc_verdict_dir`/`_btc_verdict_since`
        # hold the SESSION (not the live blip); `_btc_paused` is True while the
        # live ML is WAIT inside a live session (banner shows "напрямок · пауза").
        self._btc_verdict_dir: Optional[str] = None   # session dir 'LONG'/'SHORT'/None
        self._btc_verdict_since: float = 0.0           # session start (persists through pause)
        self._btc_paused: bool = False                 # live ML is WAIT within the session
        # 💰 Funding fuel table (repurposed from the old anomalies storage):
        # coins from the 💰 Funding Rate Scanner that currently show fuel. Same
        # dict shape so the existing table/endpoints keep working. {symbol:
        # {dir, started_at, start_price, holding, last_price, last_held_sec,
        # ended_at, end_price}}
        self._anomalies: Dict[str, Dict] = {}
        # ❤️ FF "база" — intercepted coins WAITING to open. Populated by
        # intercept() (every coin the main LONG/SHORT flow would open while FF
        # is on), direction = the signal side (self-overwrites on re-entry).
        # On ₿ START the engine fuel-checks these and opens the matching ones.
        # {symbol: {dir, added_at}}. Persisted.
        self._pending: Dict[str, Dict] = {}
        self._last_tick_ts = 0
        self._load_state()

    def _get_funding_symbols(self) -> List[str]:
        """Tracked coins from the 💰 Funding Rate Scanner — but only while that
        module is enabled and running. Returns [] otherwise. These are merged
        into fuel scanning for monitoring only."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if not fm or not fm.is_enabled() or not getattr(fm, '_running', False):
                return []
            if hasattr(fm, 'get_symbols'):
                return [s.upper() for s in fm.get_symbols()]
            return []
        except Exception:
            return []

    def _get_funding_rates(self) -> Dict[str, float]:
        """{SYMBOL: current funding %} from the 💰 Funding Rate Scanner."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm and hasattr(fm, 'get_rates'):
                return {str(k).upper(): v for k, v in fm.get_rates().items()}
        except Exception:
            pass
        return {}

    def _get_funding_next(self) -> Dict[str, int]:
        """{SYMBOL: nextFundingTime ms} from the 💰 Funding Rate Scanner."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm and hasattr(fm, 'get_next_funding'):
                return {str(k).upper(): int(v) for k, v in fm.get_next_funding().items()}
        except Exception:
            pass
        return {}

    def _get_funding_volumes(self) -> Dict[str, float]:
        """{SYMBOL: 24h volume (USD)} from the 💰 Funding Rate Scanner."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm and hasattr(fm, 'get_volumes'):
                return {str(k).upper(): float(v) for k, v in fm.get_volumes().items()}
        except Exception:
            pass
        return {}

    def _get_funding_threshold(self) -> float:
        """The 💰 Funding Rate Scanner 'Entry ≤' threshold (negative %), used as
        the START of the funding progress bar (entry → -4)."""
        try:
            from detection.funding_monitor import get_funding_monitor
            fm = get_funding_monitor()
            if fm is not None:
                return float(getattr(fm, '_entry_threshold', -1.0))
        except Exception:
            pass
        return -1.0

    # ------------------------------------------------------------------
    # settings (DB-persisted JSON)
    # ------------------------------------------------------------------
    def get_settings(self) -> Dict:
        try:
            stored = self._db.get_setting(_DB_SETTINGS, {}) or {}
        except Exception:
            stored = {}
        s = dict(DEFAULT_SETTINGS)
        if isinstance(stored, dict):
            s.update(stored)
        # sanitize
        s['duration_minutes'] = max(0, int(s.get('duration_minutes', 5) or 0))
        s['potential_threshold_pct'] = max(
            1, min(100, int(s.get('potential_threshold_pct', 95) or 95)))
        s['max_exhaustion_pct'] = max(
            1, min(100, int(s.get('max_exhaustion_pct', 75) or 75)))
        s['use_potential_exit'] = bool(s.get('use_potential_exit', True))
        s['skip_wait_coins'] = bool(s.get('skip_wait_coins', False))
        s['enabled'] = bool(s.get('enabled', False))
        try:
            s['direction_smoothing_min'] = max(0, min(600,
                int(s.get('direction_smoothing_min', 45))))
        except (TypeError, ValueError):
            s['direction_smoothing_min'] = 45
        # Remove obsolete keys
        s.pop('mode', None)
        s.pop('max_positions', None)
        return s

    def update_settings(self, patch: Dict) -> Dict:
        s = self.get_settings()
        if isinstance(patch, dict):
            for k in DEFAULT_SETTINGS:
                if k in patch:
                    s[k] = patch[k]
            # The two engine-mode toggles are mutually exclusive: turning one ON
            # forces the other OFF. Both may be OFF (engine idle). When a patch
            # tries to enable both, the BTC mode wins by convention.
            if patch.get('start_engine_enabled') is True:
                s['start_engine_independent'] = False
            elif patch.get('start_engine_independent') is True:
                s['start_engine_enabled'] = False
            # Safety net: never persist both ON.
            if s.get('start_engine_enabled') and s.get('start_engine_independent'):
                s['start_engine_independent'] = False
        try:
            self._db.set_setting(_DB_SETTINGS, s)
        except Exception as e:
            print(f"[FuelFilter] settings persist error: {e}")
        # validated copy
        s2 = self.get_settings()
        if s2.get('enabled'):
            self.start()
        return s2

    def is_enabled(self) -> bool:
        return self.get_settings().get('enabled', False)

    def set_enabled(self, on: bool):
        self.update_settings({'enabled': bool(on)})
        if on:
            self.start()
            try:
                self._tick()
            except Exception as e:
                print(f"[FuelFilter] immediate tick error: {e}")

    # ------------------------------------------------------------------
    # ❤️ FF "база" — intercepted coins waiting to open
    # ------------------------------------------------------------------
    def intercept(self, symbol: str, side: str) -> bool:
        """Catch a coin the main LONG/SHORT flow would have opened. We ONLY
        accumulate FRESH signals whose direction matches an ENABLED main button
        (LONG button on → only LONG signals queue, and vice versa). Direction
        self-overwrites on re-entry. The engine opens it later on ₿ START when
        its fuel matches. Returns True if queued, False if dropped."""
        sym = (symbol or '').upper().strip()
        side = (side or '').upper().strip()
        if not sym or side not in ('LONG', 'SHORT'):
            return False
        # Direction gate: the matching main button must be ON.
        allow_long, allow_short = self._entry_gates()
        if (side == 'LONG' and not allow_long) or (side == 'SHORT' and not allow_short):
            return False
        with self._lock:
            prev = self._pending.get(sym) or {}
            self._pending[sym] = {
                'dir': side,
                'added_at': prev.get('added_at') if (prev.get('dir') == side and prev.get('added_at'))
                            else time.time(),
            }
            self._persist_state()
        print(f"[FuelFilter] intercepted {sym} {side} → queued in ❤️ FF")
        return True

    def remove_pending(self, symbol: str) -> bool:
        """Drop a coin from the waiting base (user ✕)."""
        if not _q_allowed(7):   # OP 7: remove_pending
            return False
        sym = (symbol or '').upper().strip()
        with self._lock:
            existed = self._pending.pop(sym, None) is not None
            if existed:
                self._persist_state()
        return existed

    def clear_pending(self) -> int:
        if not _q_allowed(8):   # OP 8: clear_pending
            return 0
        with self._lock:
            n = len(self._pending)
            self._pending = {}
            self._persist_state()
        return n

    def _entry_gates(self) -> tuple:
        """(allow_long, allow_short) from TM's main directional buttons — used
        both to FILTER the FF table display and to select open candidates."""
        try:
            tm = self._get_tm() if self._get_tm else None
            ts = tm.get_settings() if tm and hasattr(tm, 'get_settings') else {}
            return (bool(ts.get('allow_long_entries', True)),
                    bool(ts.get('allow_short_entries', True)))
        except Exception:
            return (True, True)

    # ------------------------------------------------------------------
    # scan-list (whitelist) — which WATCHLIST coins FF is allowed to scan.
    # Lets the operator pick a handful of coins instead of hammering every
    # coin on the board (load control). Empty list = scan NOTHING (opt-in).
    # ------------------------------------------------------------------
    def get_scan_list(self) -> List[str]:
        try:
            lst = self._db.get_setting(_DB_SCAN_LIST, []) or []
        except Exception:
            lst = []
        if not isinstance(lst, list):
            return []
        return [str(s).upper() for s in lst]

    def set_scan(self, symbol: str, on: bool) -> List[str]:
        """Add/remove a symbol from the scan-list. Returns the new list."""
        sym = (symbol or '').upper().strip()
        if not sym:
            return self.get_scan_list()
        lst = self.get_scan_list()
        if on:
            if sym not in lst:
                lst.append(sym)
        else:
            lst = [s for s in lst if s != sym]
            # No longer scanning → drop any pending timer for it (an open
            # position stays managed until it exits on its own rules).
            self._timers.pop(sym, None)
        try:
            self._db.set_setting(_DB_SCAN_LIST, lst)
        except Exception as e:
            print(f"[FuelFilter] scan-list persist error: {e}")
        return lst

    def is_scanned(self, symbol: str) -> bool:
        return (symbol or '').upper() in self.get_scan_list()

    # ------------------------------------------------------------------
    # state persistence
    # ------------------------------------------------------------------
    def _load_state(self):
        try:
            st = self._db.get_setting(_DB_STATE, {}) or {}
        except Exception:
            st = {}
        if isinstance(st, dict):
            self._fuel_managed = st.get('fuel_managed', {}) or {}
            # Timers belong to ACTIVE managed positions only. Drop any orphan
            # timer left over from an old strategy / previous session whose coin
            # is no longer a tracked position — otherwise the table shows ghost
            # rows for coins that never fired a fresh signal this session.
            _timers_raw = st.get('timers', {}) or {}
            self._timers = {s: v for s, v in _timers_raw.items()
                            if s in self._fuel_managed}
            # Anomalies: coins that held fuel longer than anomaly_hours. They
            # live in their OWN table, persist across fuel loss, and are removed
            # only by the user (manual delete / clear). {symbol: {...}}
            self._anomalies = st.get('anomalies', {}) or {}
            # Candle-confirm attempt counters per coin — restored so the
            # "🕯️ Спроби" column and the "Opened by" attempt number survive
            # a bot restart instead of resetting to 0.
            ea = st.get('engine_attempts', {}) or {}
            if isinstance(ea, dict):
                self._engine_attempts = {str(k).upper(): int(v)
                                         for k, v in ea.items()
                                         if str(v).lstrip('-').isdigit()}
            # Direction-smoothing state.
            em = st.get('fuel_ema', {}) or {}
            if isinstance(em, dict):
                self._fuel_ema = {str(k).upper(): float(v) for k, v in em.items()}
            hy = st.get('fuel_hyst', {}) or {}
            if isinstance(hy, dict):
                self._fuel_hyst = {str(k).upper(): (v if v in ('LONG', 'SHORT') else None)
                                   for k, v in hy.items()}
            # Restore the BTC ММ SESSION first (direction + start time).
            bvd = st.get('btc_verdict_dir')
            self._btc_verdict_dir = bvd if bvd in ('LONG', 'SHORT') else None
            try:
                self._btc_verdict_since = float(st.get('btc_verdict_since') or 0.0)
            except (TypeError, ValueError):
                self._btc_verdict_since = 0.0
            self._btc_paused = False
            # Restore the entry queue PER SESSION. We persist the queue now and
            # bring it back on boot, tied to the session it belonged to. The
            # session-flip logic in _update_btc_verdict handles staleness: on the
            # first tick, if the live ММ is OPPOSITE the restored session, the
            # queue is cleared (session mismatch → fresh start). If the ML is the
            # same direction (or WAIT/pause), the queued coins are still valid and
            # keep waiting. This replaces the old blanket-ephemeral behaviour.
            pend = st.get('pending', {}) or {}
            if isinstance(pend, dict):
                self._pending = {str(k).upper(): v for k, v in pend.items()
                                 if isinstance(v, dict) and v.get('dir') in ('LONG', 'SHORT')}
            if (self._fuel_managed or self._anomalies or self._engine_attempts
                    or self._timers or self._pending):
                print(f"[FuelFilter] restored {len(self._pending)} queued "
                      f"(session={self._btc_verdict_dir or '—'}), "
                      f"{len(self._fuel_managed)} tracked position(s), "
                      f"{len(self._timers)} timer(s), "
                      f"{len(self._anomalies)} anomaly(ies), "
                      f"{len(self._engine_attempts)} attempt-counter(s) from DB")

    def _persist_state(self):
        try:
            self._db.set_setting(_DB_STATE, {
                'timers': self._timers,
                'fuel_managed': self._fuel_managed,
                'anomalies': self._anomalies,
                'engine_attempts': self._engine_attempts,
                'fuel_ema': self._fuel_ema,
                'fuel_hyst': self._fuel_hyst,
                'btc_verdict_dir': self._btc_verdict_dir,
                'btc_verdict_since': self._btc_verdict_since,
                'pending': self._pending,
            })
        except Exception as e:
            print(f"[FuelFilter] state persist error: {e}")

    # ------------------------------------------------------------------
    # per-coin SCORE for the active-timers table
    # ------------------------------------------------------------------
    @staticmethod
    def _score_label(score: float):
        """Map a 0..100 hold-score to a (label, color) verdict."""
        if score >= 72:
            return ('STRONG HOLD', '#16a34a')   # green
        if score >= 55:
            return ('HOLD', '#84cc16')           # lime
        if score >= 40:
            return ('NEUTRAL', '#eab308')        # amber
        if score >= 25:
            return ('WEAK', '#f97316')           # orange
        return ('EXHAUSTED', '#ef4444')          # red

    def _candle_momentum(self, symbol, tf='5m'):
        """LIVE price direction from the last 2 closed candles (close vs open).
        Returns (dir, strength): dir ∈ {'LONG','SHORT',None}, strength ∈ {0,1}.
        Both bars must agree (15m × 2 = ~30-min impulse); a split → neutral.
        Uses the cached kline helper (no extra exchange load on a warm cache)."""
        try:
            kl = self._candle_klines(symbol, tf)
        except Exception:
            kl = None
        if not kl or len(kl) < 3:
            return (None, 0.0)
        last2 = kl[:-1][-2:]
        if len(last2) < 2:
            return (None, 0.0)
        ups = sum(1 for k in last2 if float(k.get('p', 0)) > float(k.get('o', 0)))
        downs = sum(1 for k in last2 if float(k.get('p', 0)) < float(k.get('o', 0)))
        if ups == 2:
            return ('LONG', 1.0)
        if downs == 2:
            return ('SHORT', 1.0)
        return (None, 0.0)

    def _timer_score_for(self, symbol, direction, held_sec, exhaustion,
                         dur_sec, tf='5m'):
        """Per-coin hold quality + its OWN live direction.

        CRITICAL: the displayed direction follows ACTUAL PRICE ACTION (recent
        candles), NOT just the liq-fuel bias. Fuel tells where liquidity is
        stacked; price tells where the coin is going RIGHT NOW. If a coin is
        dumping while fuel is long-biased, the SCORE must say SHORT (and flag a
        conflict) — not 'STRONG HOLD 🟢'. Direction priority:
            price momentum → fuel status → timer direction.

        Magnitude blends, all relative to that live direction:
          • room     (30%) — 1 − exhaustion (how much of the move is left)
          • hold     (15%) — ММ hold duration vs the show threshold (~3× sat.)
          • fuel     (25%) — liq-fuel imbalance aligned with the direction
          • momentum (30%) — candle momentum aligned with the direction
        When fuel bias and price momentum DISAGREE the score is hard-capped
        (→ WEAK at best) and `conflict` is set, because the ММ setup is being
        violated by price. Returns {score,label,color,dir,conflict}."""
        # Fuel bias (where liquidity sits) — SMOOTHED + hysteresis, read-only.
        fuel_dir = None
        signed = None
        try:
            fd = self._fuel_dir_smoothed(symbol)
            if fd:
                if fd.get('status') in ('LONG', 'SHORT'):
                    fuel_dir = fd['status']
                if fd.get('dir') is not None:
                    signed = float(fd['dir'])
        except Exception:
            pass

        # Price momentum (what the coin is actually doing now).
        price_dir, pstrength = self._candle_momentum(symbol, tf)

        # Displayed direction: price action wins, then fuel, then timer.
        live_dir = price_dir or fuel_dir or direction
        conflict = bool(fuel_dir and price_dir and fuel_dir != price_dir)

        # Exhaustion (room) for the live direction. Compute it whenever it
        # wasn't supplied (the queue passes None) OR when the live direction
        # differs from the signal — otherwise `exf` stayed None and the
        # "Виснаженість" column always showed "—".
        ex = exhaustion
        if ex is None or live_dir != direction:
            try:
                ex = self._exhaustion(symbol, live_dir)
            except Exception:
                ex = exhaustion
        try:
            exf = float(ex) if ex is not None else None
        except (TypeError, ValueError):
            exf = None
        room = 0.5 if exf is None else max(0.0, min(1.0, (100.0 - exf) / 100.0))

        # Hold conviction.
        ratio = (held_sec / dur_sec) if dur_sec and dur_sec > 0 else 1.0
        hold = max(0.0, min(1.0, ratio / 3.0))

        # Fuel magnitude aligned with the live direction (against → 0).
        fmag = 0.0
        if signed is not None:
            aligned = signed if live_dir == 'LONG' else -signed
            # /0.35 (was /0.5): the liq-fuel imbalance rarely reaches 0.5, so
            # the harsher divisor kept this term tiny and dragged SCORE down.
            fmag = max(0.0, min(1.0, aligned / 0.35))

        # Momentum aligned with the live direction.
        mom = pstrength if (price_dir and price_dir == live_dir) else 0.0

        # Weights. For a QUEUED coin (held_sec == 0) the 'hold' term is not
        # meaningful yet — it would otherwise sit at 0 and drag every ❤️ queue
        # SCORE down into WEAK/EXHAUSTED (capping the max at ~85). Redistribute
        # its weight across the live components so the queue SCORE uses the full
        # range and stays sensitive. Open positions (held>0) keep all four.
        w_room, w_hold, w_fuel, w_mom = 0.30, 0.15, 0.25, 0.30
        if not held_sec or held_sec <= 0:
            _tw = w_room + w_fuel + w_mom
            w_room, w_fuel, w_mom, w_hold = w_room / _tw, w_fuel / _tw, w_mom / _tw, 0.0
        score = 100.0 * (w_room * room + w_hold * hold + w_fuel * fmag + w_mom * mom)

        # Conflict: price is fighting the fuel setup → not a healthy hold.
        if conflict:
            score = min(score, 38)        # → WEAK at best
        # Exhaustion override (FF exits on exhaustion).
        if exf is not None:
            if exf >= 90:
                score = min(score, 22)    # → EXHAUSTED
            elif exf >= 80:
                score = min(score, 38)    # → WEAK
        label, color = self._score_label(score)
        # Fuel STRENGTH 0..100 = |fuel imbalance| × 100 (how lopsided the
        # liquidity is). |dir| ≤ 0.1 (≤10%) → no direction; higher = stronger.
        fuel_strength = int(round(abs(signed) * 100)) if signed is not None else None
        return {'score': int(round(score)), 'label': label, 'color': color,
                'dir': live_dir, 'conflict': conflict, 'exh': exf,
                # Per-coin ММ (liq-fuel) direction — shown in its own column in
                # the ❤️ queue table. LONG / SHORT / None(=збалансований).
                'fuel_dir': fuel_dir,
                'fuel_strength': fuel_strength}

    def score_dict(self, symbol: str) -> Optional[Dict]:
        """Full SCORE verdict dict for `symbol` RIGHT NOW — same shape the
        ⏱️ Active Timers rows carry ({score,label,color,dir,conflict}) — so the
        open-positions tables (real + paper) can render the IDENTICAL badge.
        Pulls held/dir/exhaustion from the live timer when present, else scores
        with held=0 and lets the score derive its own live direction.

        Prefers the background score cache (fast, no per-call liq-map/kline work)
        so TM's get_state — called on every UI poll — stays cheap. Only falls
        back to a live compute for symbols not in the cache (e.g. manual
        positions FF doesn't scan), which are few."""
        try:
            sym = (symbol or '').upper().strip()
            if not sym:
                return None
            with self._lock:
                cached = self._score_cache.get(sym)
            if cached:
                return cached
            s = self.get_settings()
            tf = s.get('engine_candle_tf', '5m')
            with self._lock:
                t = self._timers.get(sym)
                if t:
                    held = time.time() - t.get('since', time.time())
                    tdir = t.get('dir') or 'LONG'
                    exh = t.get('exhaustion')
                else:
                    held, tdir, exh = 0.0, 'LONG', None
            dur = float(s.get('duration_minutes', 5) or 5) * 60
            return self._timer_score_for(sym, tdir, held, exh, dur, tf)
        except Exception:
            return None

    def score_snapshot(self, symbol: str) -> Optional[str]:
        """Compact, human SCORE string for `symbol` RIGHT NOW, e.g.
        'STRONG HOLD 🟢▲ 79'. Used to stamp a position at open and at close."""
        sc = self.score_dict(symbol)
        if not sc:
            return None
        arrow = '🟢▲' if sc.get('dir') == 'LONG' else (
            '🔴▼' if sc.get('dir') == 'SHORT' else '')
        warn = ' ⚠' if sc.get('conflict') else ''
        return f"{sc['label']} {arrow} {sc['score']}{warn}".strip()

    # ------------------------------------------------------------------
    # fuel / exhaustion measurement (cached sources only)
    # ------------------------------------------------------------------
    def _fuel_dir(self, symbol: str) -> Optional[Dict]:
        """Replicate compute_bias()'s fuel-direction math off the CACHED
        liquidation-map state (no exchange calls). Returns
        {dir, mark_price, status} or None when data is unavailable."""
        try:
            from detection.liquidation_map.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            mark = None
            lst = None
            if lm:
                try:
                    prof = self._db.get_setting('liqmap_decay_profile', 'tori')
                except Exception:
                    prof = 'tori'
                lst = lm.get_state(symbol, lookback_hours=24, profile=prof)
                mark = lst.get('mark_price') if lst else None

            # Fallback: if liquidation map doesn't have mark_price, get it from market_data
            if not mark:
                try:
                    from detection.market_data import get_market_data
                    md = get_market_data()
                    if md:
                        ticker = md.get_ticker(symbol)
                        mark = ticker.get('last') if ticker else None
                except Exception:
                    pass

            if not mark or mark <= 0:
                return None

            fa = fb = 0.0
            for lev in (lst.get('levels') or []) if lst else []:
                dist = abs(lev['price'] - mark) / mark * 100.0
                if dist > 15:
                    continue
                w = lev['usd'] / (1.0 + dist / 2.0)
                if lev['price'] > mark:
                    fa += w
                else:
                    fb += w
            den = fa + fb
            fuel_dir = (fa - fb) / den if den > 0 else 0.0
            if den <= 0:
                status = None
            elif fuel_dir > FUEL_LONG_THR:
                status = 'LONG'
            elif fuel_dir < FUEL_SHORT_THR:
                status = 'SHORT'
            else:
                status = None
            return {'dir': round(fuel_dir, 3), 'mark_price': mark,
                    'status': status}
        except Exception as e:
            print(f"[FuelFilter] fuel calc error {symbol}: {e}")
            return None

    def _fuel_dir_smoothed(self, symbol: str, update: bool = False) -> Optional[Dict]:
        """Anti-twitch wrapper over _fuel_dir. Returns the SAME shape but with a
        time-smoothed `dir` (EMA over `direction_smoothing_min`) and a hysteresis
        `status` (enter ±0.15 / exit <±0.05, sticky in between).

        update=True  → advance the EMA + hysteresis latch (call ONCE per scan
                       cycle, from _tick). update=False → read-only (UI/score).
        `raw_dir`/`raw_status` carry the instantaneous values for reference."""
        fd = self._fuel_dir(symbol)
        if not fd:
            return None
        raw = fd.get('dir', 0.0)
        sym = (symbol or '').upper()
        W = float(self.get_settings().get('direction_smoothing_min', 0) or 0)
        if W <= 0:
            # Smoothing OFF → raw fuel, instant (no lag, matches the source the
            # main window reads). This is the default.
            return {'dir': raw, 'mark_price': fd.get('mark_price'),
                    'status': fd.get('status'), 'raw_dir': raw,
                    'raw_status': fd.get('status')}
        if update:
            N = max(1.0, W * 60.0 / CYCLE_SECS)      # samples in the window
            alpha = 2.0 / (N + 1.0)                   # EMA smoothing factor
            prev = self._fuel_ema.get(sym)
            ema = raw if prev is None else (alpha * raw + (1.0 - alpha) * prev)
            self._fuel_ema[sym] = ema
            # Hysteresis latch on the SMOOTHED value.
            cur = self._fuel_hyst.get(sym)
            if ema > 0.15:
                cur = 'LONG'
            elif ema < -0.15:
                cur = 'SHORT'
            elif abs(ema) < 0.05:
                cur = None
            # else: keep the current latch (sticky 0.05..0.15 zone)
            self._fuel_hyst[sym] = cur
            status = cur
        else:
            ema = self._fuel_ema.get(sym, raw)
            # Read the latch; if smoothing hasn't initialised yet, fall back to
            # the raw status so a fresh coin is still usable immediately.
            status = self._fuel_hyst.get(sym) if sym in self._fuel_hyst \
                else fd.get('status')
        return {'dir': round(ema, 3), 'mark_price': fd.get('mark_price'),
                'status': status, 'raw_dir': raw, 'raw_status': fd.get('status')}

    def _update_btc_verdict(self):
        """BTC ММ *session* tracker (drives the ₿ banner + START engine + queue).

        A SESSION = a committed BTC ММ direction. The live ММ comes from the
        MAIN-WINDOW indicator (compute_bias fuel, ±0.1): dir > +0.1 → LONG,
        < −0.1 → SHORT, |dir| ≤ 0.1 → WAIT (ML збалансований).

        Session rules (per user's "сеанси"):
          • WAIT / data gap → PAUSE: keep the session direction, keep its start
            time (timer keeps counting), keep the queue. `_btc_paused = True`.
          • Same direction (LONG→WAIT→LONG) → SAME session, resumes.
          • OPPOSITE direction (LONG↔SHORT) → NEW session: reset the start time,
            CLEAR the whole ❤️ queue, start counting the other way.
        So the queue is cleared ONLY on a genuine session flip — never on a
        transient WAIT. Called once per cycle."""
        try:
            from web.flask_app import compute_bias
            d = compute_bias(self._db, 'BTCUSDT', None)
            fuel = ((d or {}).get('components') or {}).get('fuel') or {}
            fdir = fuel.get('dir')
            if fdir is None:
                live = None                 # data gap → treat as WAIT (pause)
            elif fdir > FUEL_LONG_THR:      # > +0.1 → LONG (як головне вікно)
                live = 'LONG'
            elif fdir < FUEL_SHORT_THR:     # < -0.1 → SHORT
                live = 'SHORT'
            else:
                live = None                 # |dir| ≤ 0.1 → збалансований → WAIT
        except Exception as e:
            print(f"[FuelFilter] BTC ММ calc error: {e}")
            return
        now = time.time()
        sess = self._btc_verdict_dir        # current session direction

        # WAIT / balanced → pause the session (keep dir, timer, queue).
        if live is None:
            if not self._btc_paused and sess:
                self._btc_paused = True
                self._persist_state()
            return

        # No session yet → start one (nothing to clear).
        if sess is None:
            self._btc_verdict_dir = live
            self._btc_verdict_since = now
            self._btc_paused = False
            self._persist_state()
            return

        # Same direction → session continues (resume from pause if needed).
        if live == sess:
            if self._btc_paused:
                self._btc_paused = False
                self._persist_state()
            return

        # OPPOSITE direction → NEW session: reset timer + CLEAR the whole queue.
        self._btc_verdict_dir = live
        self._btc_verdict_since = now
        self._btc_paused = False
        if _q_allowed(4):   # OP 4: session flip → clear the queue
            with self._lock:
                n = len(self._pending)
                self._pending = {}
                self._engine_attempts.clear()
            if n:
                print(f"[FuelFilter] сеанс {sess}→{live}: чергу очищено ({n} монет)")
        self._persist_state()

    def _bias(self, symbol: str) -> Dict:
        """Cached FF-specific bias computation. Uses compute_bias_for_ff —
        a SOFTER variant that issues LONG/SHORT more readily than the strict
        dashboard version. This reduces the time FF spends in WAIT state.
        All FF consumers (exhaustion, verdict, panel-status) go through here
        so the numbers are consistent."""
        now = time.time()
        c = self._bias_cache.get(symbol)
        if c and (now - c.get('ts', 0)) < BIAS_TTL:
            return c.get('data') or {}
        from web.flask_app import compute_bias_for_ff
        data = compute_bias_for_ff(self._db, symbol) or {}
        self._bias_cache[symbol] = {'ts': now, 'data': data}
        return data

    def _is_wait_verdict(self, symbol: str) -> bool:
        """Check if the given symbol has a WAIT verdict (unclear direction).
        Returns True if verdict is WAIT, False otherwise (or on error)."""
        try:
            verdict = self._bias(symbol).get('verdict', 'WAIT')
            return verdict == 'WAIT'
        except Exception as e:
            print(f"[FuelFilter] verdict check error {symbol}: {e}")
            return False  # on error, don't block the trade

    def get_coin_indicators(self, symbol: str) -> Dict:
        """Collect key indicators for a symbol (used by FF UI second row).
        Returns: {forecast_1h, forecast_4h, fuel_status} or partial dict on error.
        Forecast: {side, pct, confidence} or None. Fuel: 'LONG'|'SHORT'|None."""
        result = {}
        # --- Forecast 1H & 4H ---
        try:
            from detection.forecast_engine import get_forecast_engine
            fe = get_forecast_engine()
            cached = fe.get(symbol) if fe else None
            if cached:
                f1 = cached.get('forecast_1h') or {}
                f4 = cached.get('forecast_4h') or {}
                result['forecast_1h'] = {'side': f1.get('side', 0),
                                          'pct': f1.get('pct', 0),
                                          'confidence': f1.get('confidence', 0)} if f1.get('side') else None
                result['forecast_4h'] = {'side': f4.get('side', 0),
                                          'pct': f4.get('pct', 0),
                                          'confidence': f4.get('confidence', 0)} if f4.get('side') else None
            else:
                result['forecast_1h'] = None
                result['forecast_4h'] = None
        except Exception:
            result['forecast_1h'] = None
            result['forecast_4h'] = None
        # --- Fuel direction (smoothed + hysteresis, read-only) ---
        try:
            fuel_data = self._fuel_dir_smoothed(symbol)
            result['fuel_status'] = fuel_data.get('status') if fuel_data else None
        except Exception:
            result['fuel_status'] = None
        return result

    def get_panel_status(self, symbol: str) -> Dict:
        """Full FF decision-state for ONE symbol — used by the chart panel's
        second status row. Heavier than get_coin_indicators (computes verdict +
        exhaustion), but only ever called for the single currently-open symbol.

        Returns everything the operator needs to understand WHY FF is or isn't
        acting on this coin:
          enabled         — FF master toggle
          in_scan_list    — is this coin in FF's scan whitelist
          fuel_status     — liq-fuel direction (the timer trigger): LONG/SHORT/None
          exhaustion      — move exhaustion % for fuel-status side (entry gate)
          max_exhaustion  — entry gate threshold
          exhausted       — exhaustion > max (entry would be rejected)
          verdict         — compute_bias verdict (LONG/SHORT/WAIT)
          skip_wait       — skip_wait_coins setting
          wait_blocked    — verdict==WAIT AND skip_wait (entry/position blocked)
          timer           — {progress_pct, held_sec, dir} or None
          holding         — FF currently manages an open position on this coin
        """
        symbol = (symbol or '').upper()
        settings = self.get_settings()
        out = {
            'enabled': bool(settings.get('enabled', False)),
            'in_scan_list': symbol in set(self.get_scan_list()),
            'skip_wait': bool(settings.get('skip_wait_coins', False)),
            'max_exhaustion': settings.get('max_exhaustion_pct', 75),
        }
        # Fuel direction (timer trigger) — smoothed + hysteresis, read-only
        try:
            fuel_data = self._fuel_dir_smoothed(symbol)
            out['fuel_status'] = fuel_data.get('status') if fuel_data else None
        except Exception:
            out['fuel_status'] = None
        # Exhaustion for the fuel-status side (entry gate input)
        exh = None
        try:
            side = out.get('fuel_status')
            if side in ('LONG', 'SHORT'):
                exh = self._exhaustion(symbol, side)
        except Exception:
            exh = None
        out['exhaustion'] = exh
        out['exhausted'] = (exh is not None and exh > out['max_exhaustion'])
        # Verdict (WAIT gate input) — uses wl=None (watchlist NOT considered)
        try:
            out['verdict'] = self._bias(symbol).get('verdict', 'WAIT')
        except Exception:
            out['verdict'] = None
        out['wait_blocked'] = (out['skip_wait'] and out.get('verdict') == 'WAIT')
        # Timer / holding state. NEW STRATEGY: a timer exists only for an OPEN
        # FF position (starts at open, runs while the position is open). Return
        # it so the panel can show '⏱ Таймер' ONLY when there is a live trade.
        with self._lock:
            holding = symbol in self._fuel_managed
            t = self._timers.get(symbol)
            timer = None
            if t:
                held = time.time() - t.get('since', time.time())
                timer = {'dir': t.get('dir'), 'held_sec': int(held)}
        out['holding'] = holding
        out['timer'] = timer
        return out

    def _exhaustion(self, symbol: str, side: str) -> Optional[float]:
        """Move exhaustion (0..100) for `side`. Reads the EXACT value the
        dashboard's "Потенціал LONG/SHORT" panel shows — i.e. compute_bias's
        move_long['exhaustion'] / move_short['exhaustion'] — so FF's numbers
        match the panel byte-for-byte (no separate re-computation that drifts).
        Returns None on insufficient data."""
        try:
            data = self._bias(symbol)
            mv = data.get('move_long') if side == 'LONG' else data.get('move_short')
            if mv and mv.get('ok'):
                return mv.get('exhaustion')
            return None
        except Exception as e:
            print(f"[FuelFilter] exhaustion calc error {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # open / close (pure delegation to TradeManager)
    # ------------------------------------------------------------------
    def _open(self, symbol: str, side: str, fuel: Dict, settings: Dict,
              opened_by: Optional[str] = None):
        """Trigger position open via TradeManager/TestMode. Fuel filter does NOT
        store position data — it only tracks which symbols it opened and delegates
        the actual position to TM. Positions appear in Trade Manager or Test Mode
        tables based on toggle states.

        opened_by: optional label stored on the position's "Opened by" field
        (e.g. the candle-confirm attempt the auto-engine opened on). When None,
        TM uses its default ('manual_ui')."""
        print(f"[FuelFilter] _open CALLED for {symbol} {side} (timer reached 100%)")

        entry_price = fuel.get('mark_price')
        if not entry_price or entry_price <= 0:
            print(f"[FuelFilter] {symbol}: no entry price — skip open")
            return False

        # CHECK EXHAUSTION BEFORE OPENING: don't enter exhausted moves
        max_exh = settings.get('max_exhaustion_pct', 75)
        exh = self._exhaustion(symbol, side)
        if exh is not None and exh > max_exh:
            print(f"[FuelFilter] {symbol}: exhaustion {exh:.1f}% > {max_exh}% — "
                  f"rejecting open (too exhausted)")
            return False

        # CHECK WAIT VERDICT: if enabled, don't open coins in WAIT state
        if settings.get('skip_wait_coins', False):
            if self._is_wait_verdict(symbol):
                print(f"[FuelFilter] {symbol}: verdict is WAIT — "
                      f"rejecting open (skip_wait_coins enabled)")
                return False

        tm = self._get_tm() if self._get_tm else None
        if not tm:
            print(f"[FuelFilter] {symbol}: no TradeManager available — skip open")
            return False

        # Check which mode is active (TM real or Test Mode paper)
        tm_settings = tm.get_settings() if hasattr(tm, 'get_settings') and callable(tm.get_settings) else {}
        tm_enabled = tm_settings.get('enabled', False)
        test_mode = tm_settings.get('test_mode', True)

        print(f"[FuelFilter] {symbol}: TM settings: enabled={tm_enabled}, test_mode={test_mode}")

        if not tm_enabled and not test_mode:
            print(f"[FuelFilter] {symbol}: neither TM nor Test Mode enabled — skip open")
            return False

        # Prefer real TM if both are on
        is_real = tm_enabled
        mode = 'real' if is_real else 'paper'
        print(f"[FuelFilter] {symbol}: opening in {mode} mode")

        # Don't double-open: if TM already holds this symbol (real or shadow),
        # just adopt it into tracking instead of trying to open again.
        if self._tm_has_position(symbol, is_real):
            print(f"[FuelFilter] {symbol}: TM already has a position — adopting into tracking")
        else:
            print(f"[FuelFilter] {symbol}: attempting to open {side} position via TM...")
            try:
                if is_real:
                    # Real position via TM — bypass LONG/SHORT gates
                    # Fuel Filter operates independently from manual trade signals
                    print(f"[FuelFilter] {symbol}: calling tm.manual_open({symbol}, {side}, bypass_gates=True)")
                    res = tm.manual_open(symbol, side, bypass_gates=True,
                                         opened_by=opened_by)
                    print(f"[FuelFilter] {symbol}: manual_open returned: {res}")
                    if not res or not res.get('ok'):
                        reason = (res or {}).get('reason', 'unknown')
                        print(f"[FuelFilter] {symbol}: real open rejected: {reason}")
                        return False
                    # manual_open may DOWNGRADE to a paper (shadow) position when
                    # max_open_positions is reached. Respect the mode it ACTUALLY
                    # used — otherwise we'd verify/track against the wrong book
                    # (real vs shadow), the verification below would fail, and the
                    # tick loop would immediately drop the marker / close it.
                    actual_mode = res.get('mode', 'real')
                    is_real = (actual_mode == 'real')
                    mode = 'real' if is_real else 'paper'
                    if not is_real:
                        print(f"[FuelFilter] {symbol}: manual_open downgraded to "
                              f"paper (max positions reached) — tracking as paper")
                else:
                    # Paper position via Test Mode (shadow) — bypass LONG/SHORT gates
                    # Fuel Filter operates independently from manual trade signals
                    if hasattr(tm, '_open_shadow') and callable(tm._open_shadow):
                        sh_tag = opened_by or 'fuel_filter'
                        print(f"[FuelFilter] {symbol}: calling tm._open_shadow({symbol}, {side}, {entry_price}, {sh_tag!r}, bypass_gates=True)")
                        tm._open_shadow(symbol, side, entry_price, sh_tag, bypass_gates=True)
                        print(f"[FuelFilter] {symbol}: _open_shadow call completed")
                    else:
                        print(f"[FuelFilter] {symbol}: Test Mode enabled but _open_shadow not available")
                        return False
            except Exception as e:
                print(f"[FuelFilter] {symbol}: open error ({mode}): {e}")
                import traceback
                traceback.print_exc()
                return False

            # VERIFY the open actually landed before tracking. _open_shadow can
            # silently return early (e.g. LONG/SHORT entries gated off), and a
            # real open can be rejected at the order layer. If nothing landed,
            # do NOT mark _fuel_managed — otherwise the timer disappears but no
            # position exists ("trades vanish, nothing happens"). Returning
            # False makes the caller reset the timer so we retry next DURATION
            # window instead of hammering the failing open every single cycle.
            has_pos = self._tm_has_position(symbol, is_real)
            print(f"[FuelFilter] {symbol}: verification check — _tm_has_position={has_pos}")
            if not has_pos:
                print(f"[FuelFilter] {symbol}: open did not land in TM "
                      f"(gated/rejected) — NOT tracking, timer will restart")
                return False

        # Track that fuel filter opened this position (for exit condition monitoring)
        with self._lock:
            self._fuel_managed[symbol] = {
                'opened_at': time.time(),
                'side': side,
                'fuel_dir': fuel.get('dir'),
                'mode': mode,
            }
            self._persist_state()
        exh_str = f"{exh:.1f}%" if exh is not None else 'N/A'
        print(f"[FuelFilter] OPEN SUCCESS: {mode} {side} {symbol} @ {entry_price} "
              f"(fuel {fuel.get('dir')}, exhaustion {exh_str})")
        return True

    def _tm_has_position(self, symbol: str, is_real: bool) -> bool:
        """True if TradeManager currently holds a position for this symbol in
        the relevant book (real → _positions, paper → _shadow_positions)."""
        tm = self._get_tm() if self._get_tm else None
        if not tm:
            return False
        try:
            if is_real:
                book = getattr(tm, '_positions', {}) or {}
            else:
                book = getattr(tm, '_shadow_positions', {}) or {}
            return symbol in book
        except Exception:
            return False

    def _close(self, symbol: str, exit_price: float, reason: str, is_real: bool):
        """Trigger position close via TM. Removes fuel tracking.

        For SHADOW positions a valid exit_price is required (it drives the paper
        PnL). If we don't have one we abort the close and keep tracking so the
        next tick — with a fresh price — can do it cleanly. A bad price would
        otherwise record a garbage (huge) PnL. Real closes go through Bybit so
        the price arg is irrelevant there.
        """
        tm = self._get_tm() if self._get_tm else None
        if not tm:
            return

        if not is_real and (not exit_price or exit_price <= 0):
            # Try once more to get a price for the paper close
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                if md:
                    ticker = md.get_ticker(symbol)
                    exit_price = ticker.get('last') if ticker else None
            except Exception:
                exit_price = None
            if not exit_price or exit_price <= 0:
                print(f"[FuelFilter] {symbol}: no price for paper close — "
                      f"deferring (reason={reason})")
                return  # keep tracking; retry next tick

        try:
            if is_real:
                # Real position — TM will close via Bybit
                if hasattr(tm, 'manual_close') and callable(tm.manual_close):
                    tm.manual_close(symbol, reason=reason)
                else:
                    print(f"[FuelFilter] TM has no manual_close method")
            else:
                # Shadow position — TM will close internally (needs exit_price)
                if hasattr(tm, '_close_shadow') and callable(tm._close_shadow):
                    tm._close_shadow(symbol, exit_price, reason)
                else:
                    print(f"[FuelFilter] TM has no _close_shadow method")
        except Exception as e:
            print(f"[FuelFilter] close error {symbol}: {e}")
            import traceback
            traceback.print_exc()

        # Remove fuel tracking. NEW STRATEGY: the timer = position lifetime, so
        # it is reset (popped) on close. The coin is gone from the base too.
        with self._lock:
            self._fuel_managed.pop(symbol, None)
            self._timers.pop(symbol, None)
            if _q_allowed(1):   # OP 1: remove from queue on position close
                self._pending.pop(symbol, None)
            self._persist_state()
        print(f"[FuelFilter] CLOSE trigger {symbol} reason={reason}")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def _run(self):
        self._stop.wait(10)  # let other singletons boot
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[FuelFilter] tick error: {e}")
            self._stop.wait(CYCLE_SECS)

    def _register_with_liqmap(self, symbols: List[str]):
        """Register every watchlist symbol with the liquidation-map daemon so
        it actually SCANS them. This is the root fix for "not all WATCHLIST
        coins scanned": the liq map only tracks BACKGROUND_SYMBOLS (BTC/ETH)
        plus on-demand symbols requested in the last 30 min. Without this call,
        coins nobody is actively viewing in the UI have zero OI data, so
        fuel_dir is always neutral and they never trigger. We re-request every
        tick (30 s) which keeps them well inside the 30-min on-demand TTL."""
        try:
            from detection.liquidation_map.liquidation_map import get_liquidation_map
            lm = get_liquidation_map()
            if not lm:
                return
            for s in symbols:
                try:
                    lm.request_symbol(s)
                except Exception:
                    pass
        except Exception as e:
            print(f"[FuelFilter] liqmap register error: {e}")

    def _tick(self):
        settings = self.get_settings()
        self._last_tick_ts = time.time()
        if not settings.get('enabled'):
            return
        # BTC banner direction = main-window ММ indicator (compute_bias fuel).
        self._update_btc_verdict()
        now = time.time()

        # 💰 Funding scanner membership / rates — it has its OWN table now.
        funding_syms = self._get_funding_symbols()
        with self._lock:
            self._funding_syms = set(funding_syms)
            self._funding_rates = self._get_funding_rates()
            self._funding_next = self._get_funding_next()
            self._funding_vols = self._get_funding_volumes()
            managed = list(self._fuel_managed.keys())
            pending = list(self._pending.keys())

        # NEW STRATEGY: fuel is computed ONLY for the few relevant coins —
        # open FF positions + the waiting base (_pending) + funding-scanner
        # coins + BTC. No more whole-WATCHLIST scan.
        relevant = list(dict.fromkeys(
            ['BTCUSDT'] + managed + pending + list(funding_syms)))
        self._register_with_liqmap(relevant)

        # Advance EMA + read fuel once per cycle for each relevant coin.
        fuels = {}
        for sym in relevant:
            fuels[sym] = self._fuel_dir_smoothed(sym, update=True)

        # BTC table-row snapshot (fuel-based, like the other coins).
        bfuel = fuels.get('BTCUSDT')
        bstatus = bfuel.get('status') if bfuel else None
        with self._lock:
            self._btc_state = {
                'dir': bstatus,
                'exhaustion': (self._exhaustion('BTCUSDT', bstatus)
                               if bstatus in ('LONG', 'SHORT') else None),
                'held_sec': 0,
                'managed': 'BTCUSDT' in self._fuel_managed,
            }

        # Manage open FF positions (exit rules).
        if settings.get('manage_open_positions', True):
            for sym in managed:
                try:
                    self._manage_position(sym, settings, now, fuels.get(sym))
                except Exception as e:
                    print(f"[FuelFilter] manage error {sym}: {e}")

        # 💰 Funding fuel table — its OWN scan (separate from the base).
        try:
            self._scan_funding(settings, now, fuels)
        except Exception as e:
            print(f"[FuelFilter] funding scan error: {e}")

        # Header stats (now only over the relevant set).
        with self._lock:
            self._scan_stats = {
                'targets': len(pending),
                'scanned': sum(1 for f in fuels.values() if f and f.get('mark_price')),
                'pending': len(pending),
                'managed': len(managed),
            }
            self._persist_state()

        # Background SCORE cache (pending + managed) — keeps the UI fast.
        self._refresh_score_cache(settings)

    def _manage_position(self, symbol, settings, now, fuel):
        """Run the exit rules for one open FF-managed position. Extracted from
        the old per-symbol tick loop; behaviour unchanged. The timer is popped
        by _close on exit."""
        with self._lock:
            track = self._fuel_managed.get(symbol)
        if not track:
            return
        side = track['side']
        is_real = track.get('mode') == 'real'
        opposite = 'SHORT' if side == 'LONG' else 'LONG'
        status = fuel.get('status') if fuel else None
        mark = fuel.get('mark_price') if fuel else None
        if not mark:
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                if md:
                    tk = md.get_ticker(symbol)
                    mark = tk.get('last') if tk else None
            except Exception:
                pass

        # Position vanished in TM (manual close / SL / TP) → drop marker + timer.
        if not self._tm_has_position(symbol, is_real):
            with self._lock:
                self._fuel_managed.pop(symbol, None)
                self._timers.pop(symbol, None)
            return

        exh = self._exhaustion(symbol, side)
        with self._lock:
            if exh is not None:
                track['exhaustion'] = exh
                t = self._timers.get(symbol)
                if t is not None:
                    t['exhaustion'] = exh

        opened_at = track.get('opened_at', 0)
        if opened_at and (now - opened_at) < MIN_HOLD_AFTER_OPEN_SEC:
            return

        # exit: clear flip to the opposite side.
        if status == opposite:
            self._close(symbol, mark or 0.0, reason='fuel_flipped', is_real=is_real)
            return
        # exit: WAIT verdict (optional).
        if settings.get('skip_wait_coins', False) and self._is_wait_verdict(symbol):
            self._close(symbol, mark or 0.0, reason='wait_verdict', is_real=is_real)
            return
        # exit: fuel fade (sustained), with grace.
        if status != side:
            faded = track.get('faded_since')
            if not faded:
                with self._lock:
                    track['faded_since'] = now
            elif (now - faded) >= FUEL_FADE_GRACE_SEC:
                self._close(symbol, mark or 0.0, reason='fuel_faded', is_real=is_real)
                return
        else:
            if track.get('faded_since'):
                with self._lock:
                    track.pop('faded_since', None)
        # exit: exhaustion reached (optional).
        if settings.get('use_potential_exit') and exh is not None \
                and exh >= settings.get('potential_threshold_pct', 95):
            self._close(symbol, mark or 0.0, reason='potential_reached', is_real=is_real)
            return
        with self._lock:
            self._persist_state()

    def _scan_funding(self, settings, now, fuels):
        """💰 Funding fuel table (own scan): for each 💰 Funding Rate Scanner
        coin, show ONLY while it currently holds fuel (no fuel → row removed).
        Stores the live funding rate, next-funding time, previous rate (for the
        rising/falling trend) and the entry threshold so the UI can draw a
        progress bar from 'Entry ≤' → −4%. Telegram entry/exit alerts on
        transitions. Stored in self._anomalies (existing table/endpoints)."""
        funding = set(self._funding_syms)
        thr = self._get_funding_threshold()
        notifier = None
        if settings.get('funding_tg_alerts', False):
            tm = self._get_tm() if self._get_tm else None
            notifier = getattr(tm, 'notifier', None) if tm else None
        btc_line = self._btc_status_text(settings, now)
        with self._lock:
            for sym in funding:
                fuel = fuels.get(sym) or self._fuel_dir_smoothed(sym)
                status = fuel.get('status') if fuel else None
                mark = fuel.get('mark_price') if fuel else None
                rate = self._funding_rates.get(sym)
                nf = self._funding_next.get(sym)
                vol = self._funding_vols.get(sym)
                a = self._anomalies.get(sym)
                if status in ('LONG', 'SHORT'):
                    if not a or not a.get('holding'):
                        self._anomalies[sym] = {
                            'symbol': sym, 'dir': status, 'started_at': now,
                            'start_price': mark, 'holding': True,
                            'last_price': mark, 'last_held_sec': 0,
                            'funding': True, 'rate': rate, 'prev_rate': rate,
                            'next_funding': nf, 'entry_threshold': thr,
                            'vol24h': vol,
                        }
                        self._notify_funding(notifier, sym, status, mark, btc_line,
                                             settings, now, entered=True)
                    else:
                        a['holding'] = True
                        a['dir'] = status
                        if mark:
                            a['last_price'] = mark
                        a['last_held_sec'] = int(now - a.get('started_at', now))
                        a['prev_rate'] = a.get('rate')
                        a['rate'] = rate
                        a['next_funding'] = nf
                        a['entry_threshold'] = thr
                        a['vol24h'] = vol
                        a['funding'] = True
                else:
                    # Fuel ended → exit alert + remove the row.
                    if a is not None:
                        if a.get('holding'):
                            self._notify_funding(notifier, sym, a.get('dir'),
                                                 a.get('last_price'), btc_line,
                                                 settings, now, entered=False)
                        self._anomalies.pop(sym, None)
            # A funding coin that LEFT the scanner while still holding fuel must
            # ALSO fire an exit alert (this path used to remove it silently —
            # the missing "вихід" message).
            for sym in list(self._anomalies.keys()):
                a = self._anomalies[sym]
                if a.get('funding') and sym not in funding:
                    if a.get('holding'):
                        self._notify_funding(notifier, sym, a.get('dir'),
                                             a.get('last_price'), btc_line,
                                             settings, now, entered=False)
                    self._anomalies.pop(sym, None)
            self._persist_state()

    def _notify_funding(self, notifier, sym, d, price, btc_line, settings, now, entered):
        if not notifier:
            return
        try:
            rate = self._funding_rates.get(sym)
            rtxt = (f"funding {rate:+.3f}% · " if rate is not None else '')
            etxt = ''
            if d in ('LONG', 'SHORT'):
                exh = self._exhaustion(sym, d)
                if exh is not None:
                    etxt = f"🔥 {exh:.1f}% · "
            line2 = f"{rtxt}{etxt}{btc_line}"
            if entered:
                dtxt = '🟢 LONG' if d == 'LONG' else ('🔴 SHORT' if d == 'SHORT' else '')
                ptxt = (f" · {self._fmt_price(price)}" if price else '')
                notifier.send_message(f"💰 <b>{sym}</b> {dtxt}{ptxt}\n{line2}")
            else:
                ptxt = (f" · {self._fmt_price(price)}" if price else '')
                notifier.send_message(f"💰 <b>{sym}</b> ⛔ вихід{ptxt}\n{line2}")
        except Exception as e:
            print(f"[FuelFilter] funding TG error {sym}: {e}")


    def _refresh_score_cache(self, settings: Dict):
        """Background SCORE computation for the displayed rows — the waiting
        base (_pending) + open FF positions (_timers) + funding-fuel coins —
        off the request path so the UI never blocks on liq-map/kline work."""
        try:
            dur = float(settings.get('duration_minutes', 5) or 0) * 60
            tf = settings.get('engine_candle_tf', '5m')
            now = time.time()
            with self._lock:
                pending = list(self._pending.items())   # (sym, {dir, added_at})
                timers = list(self._timers.items())      # open positions
                anomalies = [(s, a.get('dir')) for s, a in self._anomalies.items()]
            # symbol -> (dir, held_sec)
            targets = {}
            for sym, info in pending:
                targets[sym] = (info.get('dir'), 0.0)   # waiting → no fuel-hold
            for sym, t in timers:
                targets[sym] = (t.get('dir'), now - t.get('since', now))
            for sym, d in anomalies:
                targets.setdefault(sym, (d, 0.0))
            cache = {}
            for sym, (d, held) in targets.items():
                try:
                    sc = self._timer_score_for(sym, d, held, None, dur, tf)
                    if sc:
                        cache[sym] = sc
                except Exception:
                    pass
            # Fuel-strength trend: snapshot this cycle's strengths and keep the
            # previous cycle so the UI can draw a rising/falling arrow.
            new_str = {s: sc.get('fuel_strength') for s, sc in cache.items()
                       if sc.get('fuel_strength') is not None}
            with self._lock:
                self._score_cache = cache
                self._fuel_str_prev = self._fuel_str
                self._fuel_str = new_str
        except Exception as e:
            print(f"[FuelFilter] score cache error: {e}")

    @staticmethod
    def _fmt_dur(sec) -> str:
        """Compact duration: 'Hг MMхв' / 'Mхв SSс' / 'Sс'."""
        sec = max(0, int(sec))
        h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
        if h:
            return f"{h}г {m:02d}хв"
        if m:
            return f"{m}хв {s:02d}с"
        return f"{s}с"

    @staticmethod
    def _fmt_price(p) -> str:
        """Compact price formatter for Telegram messages."""
        try:
            p = float(p)
        except (TypeError, ValueError):
            return '—'
        if p <= 0:
            return '—'
        if p >= 100:
            return f"{p:,.2f}"
        if p >= 1:
            return f"{p:.4f}"
        return f"{p:.6g}"

    def _funding_alert(self, s: Dict, now: float):
        """Telegram alerts for 💰 funding coins, evaluated LIVE so they fire
        within the fast alert cadence (not the 30s scan tick):
          • ENTRY when a funding coin is in the table (timer ≥ threshold AND ММ
            still holds its direction) — with direction, price and funding %;
          • EXIT when ММ no longer holds (live) and it leaves the table.
        Each fires once; re-armed on the opposite transition. Two-line layout."""
        if not s.get('funding_tg_alerts', False):
            self._funding_alerted.clear()
            return
        fdur = float(s.get('funding_duration_minutes', 0) or 0) * 60
        tm = self._get_tm() if self._get_tm else None
        notifier = getattr(tm, 'notifier', None) if tm else None
        # Snapshot eligible funding timers under the lock; do the LIVE fuel
        # checks (and Telegram) outside it.
        with self._lock:
            candidates = []
            for sym in list(self._funding_syms):
                if sym in self._fuel_managed or sym in self._anomalies:
                    continue
                t = self._timers.get(sym)
                if not t or (now - t.get('since', now)) < fdur:
                    continue
                candidates.append((sym, t.get('dir')))
            alerted = set(self._funding_alerted)
            managed_or_anom = set(self._fuel_managed) | set(self._anomalies)
        # Live membership: in the table only if ММ still holds the direction.
        current, dir_of, price_of = set(), {}, {}
        for sym, d in candidates:
            fuel = self._fuel_dir_smoothed(sym)
            # A clear opposite/neutral reading (with data) means ММ ended →
            # not in the table. A data gap (fuel is None) keeps it (no flapping).
            if fuel is not None and fuel.get('status') != d:
                continue
            current.add(sym)
            dir_of[sym] = d
            price_of[sym] = fuel.get('mark_price') if fuel else None
        new_entries = [sym for sym in current if sym not in alerted]
        # Exit = was alerted, now out of the table for ММ reasons (not because it
        # moved to a position / anomalies table).
        left = [sym for sym in (alerted - current) if sym not in managed_or_anom]
        with self._lock:
            self._funding_alerted = set(current)
        if not notifier:
            return
        btc_line = self._btc_status_text(s, now)   # current ₿ status

        def _line2(sym, d=None):
            rate = self._funding_rates.get(sym)
            rtxt = (f"funding {rate:+.3f}% · " if rate is not None else '')
            etxt = ''
            if d in ('LONG', 'SHORT'):
                exh = self._exhaustion(sym, d)
                if exh is not None:
                    etxt = f"🔥 {exh:.1f}% · "
            nf = self._funding_next.get(sym)
            ntxt = ''
            if nf:
                rem = int(nf / 1000 - now)
                if rem > 0:
                    ntxt = f" · ⏳ {self._fmt_dur(rem)} до funding"
            return f"{rtxt}{etxt}{btc_line}{ntxt}"

        for sym in new_entries:
            d = dir_of.get(sym)
            dtxt = '🟢 LONG' if d == 'LONG' else ('🔴 SHORT' if d == 'SHORT' else '')
            ptxt = (f" · {self._fmt_price(price_of.get(sym))}" if price_of.get(sym) else '')
            try:
                notifier.send_message(f"💰 <b>{sym}</b> {dtxt}{ptxt}\n{_line2(sym, d)}")
            except Exception as e:
                print(f"[FuelFilter] funding TG send error: {e}")
        for sym in left:
            fuel = self._fuel_dir_smoothed(sym)
            price = fuel.get('mark_price') if fuel else None
            d = fuel.get('status') if fuel else None
            ptxt = (f" · {self._fmt_price(price)}" if price else '')
            try:
                notifier.send_message(f"💰 <b>{sym}</b> ⛔ вихід{ptxt}\n{_line2(sym, d)}")
            except Exception as e:
                print(f"[FuelFilter] funding exit TG send error: {e}")

    def _btc_status_text(self, s: Dict, now: float) -> str:
        """Compact current ₿ BTCUSDT status for messages as a direction:
        '₿ 🟢 LONG' / '₿ 🔴 SHORT' when START is active, else '₿ ⚪ WAIT'
        (no confirmed start: counting, or no BTC timer)."""
        period = float(s.get('start_signal_minutes', 5) or 5) * 60
        vdir = self._btc_verdict_dir
        if vdir in ('LONG', 'SHORT') and self._btc_verdict_since:
            held = now - self._btc_verdict_since
            if held >= period:
                return '₿ 🟢 LONG' if vdir == 'LONG' else '₿ 🔴 SHORT'
        return '₿ ⚪ WAIT'

    def _btc_start_alert(self, s: Dict, now: float):
        """Send a Telegram message when BTC flips START↔STOP (if enabled).
        Only START and STOP are alerted (COUNTING is intermediate)."""
        if not s.get('start_signal_tg_alerts', False):
            return
        period = float(s.get('start_signal_minutes', 5) or 5) * 60
        direction = None
        if self._btc_verdict_dir in ('LONG', 'SHORT') and self._btc_verdict_since:
            direction = self._btc_verdict_dir
            held = now - self._btc_verdict_since
            state = 'START' if held >= period else 'COUNTING'
            # Remember the live direction so a later STOP can report which side
            # it stopped on (at STOP time there's no timer/direction left).
            self._btc_last_dir = direction
        else:
            state = 'STOP'
        alertable = state if state in ('START', 'STOP') else None
        if not alertable:
            return
        prev = self._btc_start_last_alert
        if prev is None:
            # First observation — record without alerting (avoid startup spam).
            self._btc_start_last_alert = alertable
            return
        if alertable == prev:
            return
        self._btc_start_last_alert = alertable
        tm = self._get_tm() if self._get_tm else None
        notifier = getattr(tm, 'notifier', None) if tm else None
        if not notifier:
            return
        def _dtxt(dd):
            return '🟢 LONG' if dd == 'LONG' else ('🔴 SHORT' if dd == 'SHORT' else '')
        if alertable == 'START':
            msg = f"🟢 <b>BTCUSDT START</b> {_dtxt(direction)}"
        else:
            # STOP — show the direction it was running in before stopping.
            msg = f"⛔ <b>BTCUSDT STOP</b> {_dtxt(self._btc_last_dir)}".rstrip()
        try:
            notifier.send_message(msg)
            print(f"[FuelFilter] BTC TG alert sent: {alertable}")
        except Exception as e:
            print(f"[FuelFilter] BTC TG send error: {e}")

    # ------------------------------------------------------------------
    # public API for the dashboard
    # ------------------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='fuel-filter')
        self._thread.start()
        # Auto-engine loop (BTC-START basket opener) — separate cadence.
        if not (self._engine_thread and self._engine_thread.is_alive()):
            self._engine_thread = threading.Thread(target=self._run_engine,
                                                   daemon=True, name='fuel-engine')
            self._engine_thread.start()
        # Fast alert loop — near-instant Telegram on threshold crossings.
        if not (self._alert_thread and self._alert_thread.is_alive()):
            self._alert_thread = threading.Thread(target=self._run_alerts,
                                                  daemon=True, name='fuel-alerts')
            self._alert_thread.start()
        print("[FuelFilter] daemon started")

    def stop(self):
        self._stop.set()

    def _run_alerts(self):
        """Fire BTC START/STOP and funding entry/exit Telegram alerts on a fast
        cadence (~3s) so they're near-instant — instead of waiting up to a full
        30s scan tick. Reads the live timer state (since is fixed; held grows),
        so threshold crossings are detected within seconds."""
        self._stop.wait(8)
        while not self._stop.is_set():
            try:
                s = self.get_settings()
                if s.get('enabled'):
                    now = time.time()
                    self._btc_start_alert(s, now)
                    # Funding entry/exit TG now fires from _scan_funding on
                    # transitions (the old _funding_alert membership-rescan is
                    # retired together with the watchlist scan).
            except Exception as e:
                print(f"[FuelFilter] alert loop error: {e}")
            self._stop.wait(3)

    # ------------------------------------------------------------------
    # BTC-START auto-engine: when ON and BTC banner == START, open the basket
    # of coins from the chosen tables (anomalies / active timers) in the BANNER
    # direction, via FF._open (→ _fuel_managed → exhaustion-exit + control).
    # ------------------------------------------------------------------
    def _run_engine(self):
        self._stop.wait(15)
        while not self._stop.is_set():
            try:
                self._engine_tick()
            except Exception as e:
                print(f"[FF-Engine] tick error: {e}")
            secs = 15
            try:
                secs = max(5, int(self.get_settings().get('start_engine_scan_secs', 15)))
            except Exception:
                pass
            self._stop.wait(secs)

    def _candle_klines(self, symbol: str, tf: str):
        """Klines for candle confirmation at the configured TF. Uses the
        scanner's cached klines for free when the TF matches the scanner's TF;
        otherwise fetches a few bars via market_data with a short-TTL cache."""
        tm = self._get_tm() if self._get_tm else None
        scanner = getattr(tm, 'scanner', None) if tm else None
        # Free path: scanner already holds klines at its own TF.
        try:
            if scanner and hasattr(scanner, 'get_timeframe') \
                    and scanner.get_timeframe() == tf:
                kl = scanner._get_cached_klines(symbol)
                if kl:
                    return kl
        except Exception:
            pass
        # Custom TF → fetch (cached ~20s to avoid hammering on each engine tick).
        now = time.time()
        key = (symbol, tf)
        c = self._candle_cache.get(key)
        if c and (now - c[0]) < 20:
            return c[1]
        kl = None
        try:
            from detection.market_data import get_market_data
            md = get_market_data()
            if md:
                kl = md.fetch_klines(symbol, limit=6, interval=tf)
        except Exception:
            kl = None
        if kl:
            self._candle_cache[key] = (now, kl)
        return kl

    def _candle_confirms(self, symbol: str, side: str, tf: str = '5m'):
        """Pre-entry candle confirmation: is recent price action moving in
        `side`? Uses the last 2 CLOSED bars at the given TF — BOTH must move in
        the direction (15m × 2 = ~30-min impulse window). Returns:
          True  — confirmed (both last 2 closed bars in the direction)
          False — against / mixed
          None  — no kline data yet (caller treats as not-confirmed → waits)."""
        kl = self._candle_klines(symbol, tf)
        if not kl or len(kl) < 3:
            return None
        last2 = kl[:-1][-2:]   # drop the still-forming bar, take last 2 closed
        if len(last2) < 2:
            return None
        ups = sum(1 for k in last2 if float(k.get('p', 0)) > float(k.get('o', 0)))
        downs = sum(1 for k in last2 if float(k.get('p', 0)) < float(k.get('o', 0)))
        if side == 'LONG':
            return ups >= 2
        if side == 'SHORT':
            return downs >= 2
        return False

    def _engine_tick(self):
        s = self.get_settings()
        # Snapshot the attempt counters so we only hit the DB when they actually
        # change this tick (they're persisted so the "🕯️ Спроби" column and the
        # "Opened by" attempt number survive a bot restart).
        _att_before = dict(self._engine_attempts)

        def _persist_attempts_if_changed():
            if self._engine_attempts != _att_before:
                self._persist_state()

        # NEW STRATEGY: candidates come from the intercepted base (_pending),
        # NOT from a watchlist scan. Two mutually-exclusive modes:
        #   • BTC mode  (start_engine_enabled)     — only act while ₿ BTCUSDT is
        #     START (its ММ held ≥ start_signal_minutes); the trigger, not a
        #     direction filter.
        #   • Indep mode(start_engine_independent) — act immediately, no BTC.
        # In BOTH: which coins are eligible is decided by the MAIN LONG/SHORT
        # buttons, and a coin opens only when its live fuel matches its OWN
        # signal direction. Both OFF → engine idle.
        btc_mode = bool(s.get('start_engine_enabled'))
        indep_mode = bool(s.get('start_engine_independent'))
        if not s.get('enabled') or not (btc_mode or indep_mode):
            self._engine_attempts.clear()   # engine off → reset counters
            _persist_attempts_if_changed()
            return
        now = time.time()

        if btc_mode:
            # ₿ START gate: the SESSION must hold a direction ≥ threshold.
            if self._btc_verdict_dir not in ('LONG', 'SHORT') or not self._btc_verdict_since:
                self._engine_attempts.clear()
                _persist_attempts_if_changed()
                return
            # Do NOT open while the session is PAUSED (live ML is WAIT/balanced).
            # The session, its timer and the queue stay alive — opening simply
            # waits until the ML resumes the session direction.
            if self._btc_paused:
                _persist_attempts_if_changed()
                return
            period = float(s.get('start_signal_minutes', 5) or 5) * 60
            if (now - self._btc_verdict_since) < period:
                self._engine_attempts.clear()
                _persist_attempts_if_changed()
                return

        # Candidates = waiting base, filtered by the MAIN LONG/SHORT buttons.
        allow_long, allow_short = self._entry_gates()
        cand = {}   # sym -> (waited_sec, signal_dir)
        with self._lock:
            for sym, info in self._pending.items():
                d = info.get('dir')
                if d not in ('LONG', 'SHORT'):
                    continue
                if d == 'LONG' and not allow_long:
                    continue
                if d == 'SHORT' and not allow_short:
                    continue
                cand[sym] = (now - info.get('added_at', now), d)

        mode_lbl = "BTC-START" if btc_mode else "НЕЗАЛЕЖНИЙ"
        if not cand:
            self._engine_attempts.clear()
            _persist_attempts_if_changed()
            print(f"[FF-Engine] {mode_lbl} · 0 кандидатів (кнопки L={allow_long} S={allow_short})")
            return

        dur = float(s.get('duration_minutes', 5)) * 60
        max_exh = float(s.get('max_exhaustion_pct', 75) or 75)
        tf = s.get('engine_candle_tf', '5m')
        trace = []

        # Open oldest-waiting first.
        for sym, (held, d) in sorted(cand.items(), key=lambda kv: -kv[1][0]):
            if sym in self._fuel_managed:
                trace.append(f"{sym}:managed")
                continue   # already managed (no duplicate)
            # Skip if TM already holds this coin (real OR paper) — don't dup.
            if self._tm_has_position(sym, True) or self._tm_has_position(sym, False):
                trace.append(f"{sym}:вже-в-угодах")
                if _q_allowed(3):   # OP 3: TM already holds the coin
                    with self._lock:
                        self._pending.pop(sym, None)
                continue
            fuel = self._fuel_dir_smoothed(sym)
            if not fuel or not fuel.get('mark_price'):
                trace.append(f"{sym}:немає-ціни")
                continue
            # GATE: fuel must be present AND match the signal direction.
            if fuel.get('status') != d:
                trace.append(f"{sym}:паливо≠{d}({fuel.get('status')})")
                continue
            # Exhaustion gate (same as _open) — surfaced HERE so it's visible and
            # does NOT silently waste a candle check. Too-exhausted coins are
            # skipped without counting an attempt (the move is just too far gone).
            exh = self._exhaustion(sym, d)
            if exh is not None and exh > max_exh:
                trace.append(f"{sym}:виснаж{exh:.0f}%>{max_exh:.0f}%")
                continue
            # (Candle-confirmation gate retired — opens trigger on the fuel↔
            # direction match itself.)
            # SCORE gate: only open when the coin's SCORE is STRONG HOLD AND its
            # SCORE direction matches the candidate direction.
            if s.get('engine_require_strong_hold', False):
                sc = self._timer_score_for(sym, d, held, exh, dur, tf)
                if sc.get('label') != 'STRONG HOLD' or sc.get('dir') != d:
                    trace.append(f"{sym}:score={sc.get('label')}/{sc.get('dir')}≠STRONG·{d}")
                    continue
            try:
                # _open routes through TM (bypass gates, like Alerts but ignoring
                # the LONG/SHORT master + SMC filters) AND registers the position
                # in _fuel_managed so exhaustion-exit + control manage it.
                # The "Opened by" field records which candle-confirm attempt the
                # engine opened on (failed checks bump _engine_attempts; opening
                # on the first check is attempt #1) AND the coin's ММ timer value
                # at the moment of opening.
                # "Opened by" records the EXHAUSTION at the moment of entry.
                opened = self._open(
                    sym, d, fuel, s,
                    opened_by=(f"🔥 Exhaust {exh:.1f}%" if exh is not None else "🔥 FF"))
                if opened:
                    # Timer starts NOW (at open) and runs while the position is
                    # open; _close resets it. Coin leaves the waiting base.
                    with self._lock:
                        self._timers[sym] = {'dir': d, 'since': now,
                                             'start_price': fuel.get('mark_price')}
                        if _q_allowed(2):   # OP 2: engine opened the trade
                            self._pending.pop(sym, None)
                    self._engine_attempts.pop(sym, None)   # opened → reset
                    trace.append(f"{sym}:✅ВІДКРИТО {d}")
                    print(f"[FF-Engine] opened {d} {sym} ({mode_lbl}, exh="
                          f"{('%.1f%%' % exh) if exh is not None else '—'})")
                else:
                    trace.append(f"{sym}:_open-відхилив")
            except Exception as e:
                trace.append(f"{sym}:помилка")
                print(f"[FF-Engine] open error {sym}: {e}")

        # Prune attempt counters for coins that are no longer candidates.
        considered = set(cand)
        for k in list(self._engine_attempts):
            if k not in considered:
                self._engine_attempts.pop(k, None)

        # Persist the counters to DB if they changed this tick (survives restart).
        _persist_attempts_if_changed()

        print(f"[FF-Engine] {mode_lbl} · {len(cand)} канд · "
              f"паливо-гейт · "
              + ' '.join(trace))

    def get_state(self) -> Dict:
        """Snapshot for the UI: settings + live timers + active tracking.
        NEW STRATEGY: shows only timers held >= duration_minutes (threshold).
        No auto-open. Position management (close) is optional via setting."""
        with self._lock:
            settings = self.get_settings()
            duration_sec = settings['duration_minutes'] * 60
            funding_dur = float(settings.get('funding_duration_minutes', 0) or 0) * 60
            now = time.time()
            # The table = the waiting BASE (_pending). Show EVERY queued coin
            # (that isn't already an open position), REGARDLESS of the current
            # LONG/SHORT button state. Interception already gated entry by the
            # enabled button at ADD time — re-filtering here by the live buttons
            # made the WHOLE ❤️ queue vanish the moment the verdict flipped the
            # buttons to WAIT (both off), wiping coins that legitimately waited.
            # Buttons control OPENING (the engine), not visibility. The count
            # equals the rows shown.
            visible_pending = 0
            timers = []
            for sym, info in self._pending.items():
                if sym in self._fuel_managed:
                    continue  # already opened
                d = info.get('dir')
                visible_pending += 1
                waited = now - info.get('added_at', now)
                timers.append({
                    'symbol': sym, 'dir': d,
                    # No timer while waiting — it starts at open. Show how long
                    # the coin has been queued instead.
                    'held_sec': int(waited),
                    'waiting': True,
                    # Real move-exhaustion for the coin (0 fresh → 100 exhausted).
                    # NB: the score dict stores it under 'exh' — the old '_exh'
                    # key never existed, so this column always showed "—".
                    'exhaustion': (self._score_cache.get(sym) or {}).get('exh'),
                    'score': self._score_cache.get(sym),
                    # Per-coin ММ (liq-fuel) direction for the ММ column —
                    # LONG / SHORT / None(=збалансований). The UI compares it
                    # with `dir` (the signal side) to show ✓ збіг / ✗ проти.
                    'mm': (self._score_cache.get(sym) or {}).get('fuel_dir'),
                    # Fuel STRENGTH 0..100 (+ previous cycle for the trend arrow).
                    'mm_str': self._fuel_str.get(sym),
                    'mm_str_prev': self._fuel_str_prev.get(sym),
                    'funding': False,
                    'funding_rate': None,
                    'funding_next_ms': None,
                    'engine_attempts': self._engine_attempts.get(sym, 0),
                })
            all_timers = sorted(timers, key=lambda x: -x['held_sec'])
            bs = self._btc_state or {}
            # BTC START/STOP signal: progress counts up while the MAIN-WINDOW
            # verdict (compute_bias) holds LONG/SHORT, reaching START at
            # start_signal_minutes; STOP when the verdict is WAIT/None. The
            # banner is therefore 1:1 with 🎯 Smart Money Concepts (NOT liq-fuel).
            ssm = float(settings.get('start_signal_minutes', 5) or 5)
            period_sec = ssm * 60
            # SESSION direction (persists through WAIT). The timer keeps counting
            # through a pause, so START can still be reached while paused; the
            # `paused` flag lets the banner show "напрямок · ⏸ пауза".
            b_dir = self._btc_verdict_dir
            if b_dir in ('LONG', 'SHORT') and self._btc_verdict_since:
                b_held = now - self._btc_verdict_since
            else:
                b_dir = None
                b_held = 0
            has_btc = b_dir in ('LONG', 'SHORT')
            if not has_btc:
                btc_status, btc_prog = 'STOP', 0
            elif b_held >= period_sec:
                btc_status, btc_prog = 'START', 100
            else:
                btc_status = 'COUNTING'
                btc_prog = round(b_held / period_sec * 100, 1) if period_sec else 0
            btc_start = {
                'status': btc_status,
                'progress': btc_prog,
                'held_sec': int(b_held),
                'period_sec': int(period_sec),
                'dir': b_dir,
                'paused': bool(self._btc_paused and has_btc),
            }
            # 💰 Funding table — only coins currently holding fuel. Carries the
            # live funding rate + next-funding time + trend (prev_rate) + the
            # entry threshold so the UI can draw the entry→−4% progress bar.
            anomalies = []
            for sym, a in self._anomalies.items():
                held = int(now - a.get('started_at', now))
                anomalies.append({
                    'symbol': sym,
                    'dir': a.get('dir'),
                    'holding': True,
                    'held_sec': held,
                    'start_price': a.get('start_price'),
                    'current_price': a.get('last_price'),
                    'funding': True,
                    'funding_rate': a.get('rate'),
                    'funding_prev_rate': a.get('prev_rate'),
                    'funding_next_ms': a.get('next_funding'),
                    'entry_threshold': a.get('entry_threshold'),
                    'vol24h': a.get('vol24h'),
                })
            anomalies.sort(key=lambda x: -x['held_sec'])
        return {
            'ok': True,
            'settings': settings,
            'running': bool(self._thread and self._thread.is_alive()),
            'last_tick_ts': self._last_tick_ts,
            'timers': all_timers,
            'btc_start': btc_start,
            'anomalies': anomalies,
            'active_symbols': list(self._fuel_managed.keys()),
            'tracked_count': len(self._fuel_managed),
            'scan_list': [],   # retired (FF no longer scans the WATCHLIST)
            'pending_count': len(self._pending),
            'pending_visible': visible_pending,   # rows actually shown (= header)
            'scan_stats': dict(self._scan_stats),
        }

    def active_symbols(self) -> List[str]:
        """Symbols with an open fuel-managed position — used to draw the ❤ marker
        in the watchlist."""
        with self._lock:
            return list(self._fuel_managed.keys())

    def is_in_table(self, symbol: str, side: str) -> bool:
        """Return True if `symbol` with direction `side` is currently shown in
        the ❤️ Fuel Auto-Filter table (i.e. its fuel timer has held for at least
        the configured `duration_minutes` threshold). Used as a confirmation gate
        for trade opens: a trade is only allowed if its coin+direction shows
        sustained fuel here."""
        symbol = (symbol or '').upper().strip()
        side = (side or '').upper().strip()
        with self._lock:
            settings = self.get_settings()
            duration_sec = settings.get('duration_minutes', 5) * 60
            now = time.time()
            t = self._timers.get(symbol)
            if not t:
                return False
            if t.get('dir') != side:
                return False
            held = now - t.get('since', now)
            return held >= duration_sec

    def get_exhaustion_map(self) -> Dict[str, float]:
        """Get exhaustion values for all fuel-managed positions (for UI merge).
        Returns {symbol: exhaustion_pct, ...}."""
        with self._lock:
            return {sym: track.get('exhaustion')
                    for sym, track in self._fuel_managed.items()
                    if track.get('exhaustion') is not None}

    def get_fuel_strength_map(self) -> Dict[str, Dict]:
        """Fuel STRENGTH (0..100) + previous-cycle value + direction, per symbol
        we track (read from the pre-computed score cache — CHEAP, no per-request
        compute). For the open-position tables' 'Паливо' column.
        Returns {symbol: {'now': int, 'prev': int|None, 'dir': 'LONG'|'SHORT'|None}}."""
        with self._lock:
            out = {}
            for sym, sc in self._score_cache.items():
                st = sc.get('fuel_strength')
                if st is not None:
                    out[sym] = {'now': st,
                                'prev': self._fuel_str_prev.get(sym),
                                'dir': sc.get('fuel_dir')}
            return out

    def delete_timer(self, symbol: str) -> bool:
        """Remove a coin from the FF table — the waiting base (_pending) and/or
        a running timer. Returns True if anything was removed."""
        symbol = symbol.upper()
        with self._lock:
            removed = False
            if _q_allowed(5):   # OP 5: manual delete ✕
                removed = (self._pending.pop(symbol, None) is not None)
            if symbol in self._timers:
                self._timers.pop(symbol)
                removed = True
            if removed:
                self._persist_state()
                print(f"[FuelFilter] removed from FF table: {symbol}")
            return removed

    def force_open_timer(self, symbol: str) -> Dict:
        """Manually trigger position open for a running timer, bypassing the
        duration requirement (progress % doesn't matter).

        Returns: {'ok': bool, 'reason': str, 'opened': bool}
        """
        symbol = symbol.upper()
        settings = self.get_settings()
        if not settings.get('enabled'):
            return {'ok': False, 'reason': 'Fuel Auto-Filter is disabled'}

        with self._lock:
            timer = self._timers.get(symbol)
            pend = self._pending.get(symbol)

        # Check if already holding a position
        if symbol in self._fuel_managed:
            return {'ok': False, 'reason': f'Already holding position for {symbol}'}

        # Direction comes from the waiting base (signal side) or, if already
        # opened-and-timing, from the timer. Force-open ignores the ₿ START /
        # fuel gate — the operator explicitly wants in.
        side = (pend or {}).get('dir') or (timer or {}).get('dir')
        if side not in ('LONG', 'SHORT'):
            return {'ok': False, 'reason': f'{symbol} не у черзі FF'}

        # Use the SAME fuel helper the tick loop uses (smoothed, read-only). It
        # returns {dir, mark_price, status} and has its own market_data fallback
        # for mark_price, so even with neutral/faded fuel we still get a price.
        fuel = self._fuel_dir_smoothed(symbol)
        if not fuel or not fuel.get('mark_price'):
            # Last-ditch: try a direct price fetch so a liq-map gap doesn't
            # block a manual open the user explicitly asked for.
            price = None
            try:
                from detection.market_data import get_market_data
                md = get_market_data()
                if md:
                    ticker = md.get_ticker(symbol)
                    price = ticker.get('last') if ticker else None
            except Exception:
                price = None
            if not price or price <= 0:
                return {'ok': False, 'reason': f'Не вдалося отримати ціну для {symbol}'}
            fuel = {'dir': (fuel.get('dir') if fuel else 0.0),
                    'mark_price': price,
                    'status': (fuel.get('status') if fuel else None)}

        try:
            opened = self._open(symbol, side, fuel, settings)
        except Exception as e:
            print(f"[FuelFilter] force_open_timer error for {symbol}: {e}")
            return {'ok': False, 'reason': f'Помилка: {str(e)}'}

        if opened:
            # Timer starts now (position lifetime); coin leaves the waiting base.
            with self._lock:
                self._timers[symbol] = {'dir': side, 'since': time.time(),
                                        'start_price': fuel.get('mark_price')}
                if _q_allowed(9):   # OP 9: manual force-open
                    self._pending.pop(symbol, None)
                self._persist_state()
            return {'ok': True, 'reason': f'Позицію {side} відкрито вручну', 'opened': True}
        else:
            # _open() returns False on its own gates (exhaustion too high,
            # WAIT verdict if skip_wait on, no entry price, qty below min, …).
            return {'ok': False, 'opened': False,
                    'reason': 'Відкриття відхилено (виснаженість/вердикт/розмір — див. лог)'}

    def clear_all_timers(self) -> int:
        """Clear the whole FF table — waiting base + any timers."""
        with self._lock:
            count = len(self._timers)
            if _q_allowed(6):   # OP 6: manual "Очистити всі"
                count += len(self._pending)
                self._pending.clear()
            self._timers.clear()
            self._persist_state()
            print(f"[FuelFilter] Cleared {count} timers")
            return count

    def delete_anomaly(self, symbol: str) -> bool:
        """Remove ONE coin from the anomalies table (user action)."""
        symbol = (symbol or '').upper()
        with self._lock:
            if symbol in self._anomalies:
                self._anomalies.pop(symbol, None)
                self._persist_state()
                print(f"[FuelFilter] Anomaly deleted: {symbol}")
                return True
            return False

    def clear_anomalies(self) -> int:
        """Clear the whole anomalies table. Returns count removed."""
        with self._lock:
            count = len(self._anomalies)
            self._anomalies.clear()
            self._persist_state()
            print(f"[FuelFilter] Cleared {count} anomalies")
            return count


_instance: Optional[FuelFilterDaemon] = None


def init_fuel_filter(db, get_trade_manager, get_watchlist) -> FuelFilterDaemon:
    """Create the singleton and auto-start the loop if the persisted toggle
    is ON (so it survives restarts)."""
    global _instance
    if _instance is None:
        _instance = FuelFilterDaemon(db, get_trade_manager, get_watchlist)
        if _instance.is_enabled():
            _instance.start()
            print("[FuelFilter] restored ON state from DB — loop running")
    return _instance


def get_fuel_filter() -> Optional[FuelFilterDaemon]:
    return _instance
