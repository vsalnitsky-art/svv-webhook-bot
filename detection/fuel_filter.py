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

DEFAULT_SETTINGS = {
    'enabled': False,
    'duration_minutes': 5,        # min duration to show in table (threshold filter)
    'potential_threshold_pct': 95,  # exhaustion ≥ this → close
    'use_potential_exit': True,   # toggle the exhaustion exit on/off
    'max_exhaustion_pct': 75,     # (legacy) not used for auto-open anymore
    'skip_wait_coins': False,     # (legacy) not used for auto-open anymore
    'manage_open_positions': True,  # if True, FF closes positions it opened
    'anomaly_hours': 10,            # fuel held longer than this → "anomaly" list
    'start_signal_minutes': 5,      # BTC ММ held ≥ this → START signal (else STOP)
    # ── BTC-START auto-engine (banner toggle) ──
    'start_engine_enabled': False,        # master: auto-open basket on BTC START
    'start_engine_use_anomalies': True,   # source: 🜂 anomalies table
    'start_engine_use_timers': True,      # source: ⏱️ active timers table
    'start_engine_scan_secs': 15,         # engine scan cadence (sec)
    'start_engine_include_funding': False,# include 💰 funding-marked coins
}


class FuelFilterDaemon:
    def __init__(self, db, get_trade_manager: Callable,
                 get_watchlist: Callable):
        self._db = db
        self._get_tm = get_trade_manager
        self._get_watchlist = get_watchlist
        self._thread: Optional[threading.Thread] = None
        self._engine_thread: Optional[threading.Thread] = None
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
        # Symbols pulled in from the 💰 Funding Rate Scanner (when it's enabled).
        # They get fuel timers + a row in the ❤️ table, flagged distinctly, but
        # are MONITOR-ONLY (no auto-open / management). Refreshed each tick.
        self._funding_syms: set = set()
        # Latest BTC fuel snapshot — pinned permanently at the top of the table.
        self._btc_state: Dict = {}
        # Anomalies: coins that held fuel longer than anomaly_hours. Own table,
        # survive fuel loss, user-deleted only. {symbol: {dir, started_at,
        # start_price, holding, last_price, last_held_sec, ended_at, end_price}}
        self._anomalies: Dict[str, Dict] = {}
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
            self._timers = st.get('timers', {}) or {}
            self._fuel_managed = st.get('fuel_managed', {}) or {}
            # Anomalies: coins that held fuel longer than anomaly_hours. They
            # live in their OWN table, persist across fuel loss, and are removed
            # only by the user (manual delete / clear). {symbol: {...}}
            self._anomalies = st.get('anomalies', {}) or {}
            if self._fuel_managed or self._anomalies:
                print(f"[FuelFilter] restored {len(self._fuel_managed)} tracked "
                      f"position(s), {len(self._timers)} timer(s), "
                      f"{len(self._anomalies)} anomaly(ies) from DB")

    def _persist_state(self):
        try:
            self._db.set_setting(_DB_STATE, {
                'timers': self._timers,
                'fuel_managed': self._fuel_managed,
                'anomalies': self._anomalies,
            })
        except Exception as e:
            print(f"[FuelFilter] state persist error: {e}")

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
        # --- Fuel direction ---
        try:
            fuel_data = self._fuel_dir(symbol)
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
        # Fuel direction (timer trigger)
        try:
            fuel_data = self._fuel_dir(symbol)
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
        # Timer / holding state
        with self._lock:
            holding = symbol in self._fuel_managed
            t = self._timers.get(symbol)
            duration_sec = settings['duration_minutes'] * 60
            timer = None
            if t and not holding:
                held = time.time() - t.get('since', time.time())
                timer = {
                    'dir': t.get('dir'),
                    'held_sec': int(held),
                    'progress_pct': (round(min(100.0, held / duration_sec * 100.0), 1)
                                     if duration_sec > 0 else 100.0),
                }
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
    def _open(self, symbol: str, side: str, fuel: Dict, settings: Dict):
        """Trigger position open via TradeManager/TestMode. Fuel filter does NOT
        store position data — it only tracks which symbols it opened and delegates
        the actual position to TM. Positions appear in Trade Manager or Test Mode
        tables based on toggle states."""
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
                    res = tm.manual_open(symbol, side, bypass_gates=True)
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
                        print(f"[FuelFilter] {symbol}: calling tm._open_shadow({symbol}, {side}, {entry_price}, 'fuel_filter', bypass_gates=True)")
                        tm._open_shadow(symbol, side, entry_price, 'fuel_filter', bypass_gates=True)
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

        # Remove fuel tracking
        with self._lock:
            self._fuel_managed.pop(symbol, None)
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
        try:
            watchlist = self._get_watchlist() or []
        except Exception:
            watchlist = []
        watchlist = [s.upper() for s in watchlist]

        # Pull in coins from the 💰 Funding Rate Scanner (when enabled). They are
        # monitored for fuel exactly like watchlist coins and fully actionable
        # (force-open, exhaustion-exit, position management). The 💰 badge marks
        # ONLY coins that are NOT already in the SMC watchlist — a coin present
        # in the watchlist is treated as a normal watchlist coin (no badge).
        funding_syms = self._get_funding_symbols()
        watchlist_set = set(watchlist)
        with self._lock:
            self._funding_syms = set(funding_syms) - watchlist_set

        # SCAN all watchlist coins for data/stats. FF flag still controls which
        # coins can auto-open positions (checked later in the loop).
        # BTCUSDT is ALWAYS scanned + pinned in the table (see _btc_state).
        scan_targets = list(dict.fromkeys(watchlist + funding_syms))
        symbols = list(dict.fromkeys(
            ['BTCUSDT'] + scan_targets + list(self._fuel_managed.keys())))

        # CRITICAL: register ALL symbols with liq-map so they all have fuel data.
        # SMC scanner already does this for the watchlist every 60s, but funding
        # coins may be outside the watchlist — re-register here so they too get
        # liq-map coverage.
        self._register_with_liqmap(symbols)

        # FF scan-list: which coins get a fuel timer (+ show in the table).
        # Funding-scanner coins are added so they get monitored; BTCUSDT is
        # always included so its timer/held value is tracked for the pinned row.
        scan_list = set(self.get_scan_list()) | set(funding_syms) | {'BTCUSDT'}

        # Get TM settings to know which mode positions live in
        tm = self._get_tm() if self._get_tm else None
        tm_settings = tm.get_settings() if tm and hasattr(tm, 'get_settings') else {}
        tm_enabled = tm_settings.get('enabled', False)

        duration_sec = settings['duration_minutes'] * 60
        anomaly_sec = float(settings.get('anomaly_hours', 10)) * 3600
        now = time.time()

        # Aggregate scan-stats accumulators (for the panel header). Now covers
        # ALL watchlist coins (not just FF-flagged), so user sees full picture.
        scan_set = set(scan_targets)
        _scanned_ok = 0
        _max_long = _max_short = None
        _cnt_long = _cnt_short = 0

        for symbol in symbols:
            fuel = self._fuel_dir(symbol)
            status = fuel.get('status') if fuel else None
            fuel_managed = symbol in self._fuel_managed

            # Snapshot BTC state every tick so the table can pin a permanent
            # BTC row regardless of whether its timer has held past the
            # threshold (or exists at all).
            if symbol == 'BTCUSDT':
                _bt = self._timers.get('BTCUSDT')
                _bheld = int(now - _bt['since']) if _bt and _bt.get('since') else 0
                _bexh = (self._exhaustion('BTCUSDT', status)
                         if status in ('LONG', 'SHORT') else None)
                with self._lock:
                    self._btc_state = {
                        'dir': status,
                        'exhaustion': _bexh,
                        'held_sec': _bheld,
                        'managed': fuel_managed,
                    }

            # --- accumulate panel scan stats (ALL watchlist coins now).
            # Runs before any branch/continue so every target is counted.
            # Track MAXIMUM exhaustion for each side (how close the most exhausted
            # coin got to 100%), filtering by status so LONG-moving coins contribute
            # to LONG stat, SHORT-moving coins to SHORT stat.
            if symbol in scan_set:
                if fuel and fuel.get('mark_price'):
                    _scanned_ok += 1
                if status == 'LONG':
                    _el = self._exhaustion(symbol, 'LONG')
                    if _el is not None:
                        _max_long = max(_max_long, _el) if _max_long is not None else _el
                        _cnt_long += 1
                elif status == 'SHORT':
                    _es = self._exhaustion(symbol, 'SHORT')
                    if _es is not None:
                        _max_short = max(_max_short, _es) if _max_short is not None else _es
                        _cnt_short += 1

            # ---- manage positions fuel filter opened ----
            # Only if manage_open_positions is enabled (can be toggled off)
            if fuel_managed and settings.get('manage_open_positions', True):
                track = self._fuel_managed[symbol]
                side = track['side']
                is_real = track.get('mode') == 'real'
                opposite = 'SHORT' if side == 'LONG' else 'LONG'
                mark = fuel.get('mark_price') if fuel else None

                # If no mark price available, try to get current price
                if not mark:
                    try:
                        from detection.market_data import get_market_data
                        md = get_market_data()
                        if md:
                            ticker = md.get_ticker(symbol)
                            mark = ticker.get('last') if ticker else None
                    except Exception:
                        pass

                # --- SYNC: if the position no longer exists in TM (user closed
                # it manually, or it was never actually opened), drop our marker
                # so the timer can run again. Prevents "ghost" tracked symbols
                # that hide the timer but have no real position. ---
                if not self._tm_has_position(symbol, is_real):
                    print(f"[FuelFilter] {symbol}: tracked but no TM position — "
                          f"dropping marker")
                    with self._lock:
                        self._fuel_managed.pop(symbol, None)
                    self._timers.pop(symbol, None)
                    continue

                # Compute exhaustion (for UI display + exit condition)
                exh = self._exhaustion(symbol, side)
                if exh is not None:
                    with self._lock:
                        track['exhaustion'] = exh

                # MIN-HOLD GRACE: don't run ANY auto-exit for the first
                # MIN_HOLD_AFTER_OPEN_SEC after opening. A manual force-open
                # uses the timer's side even if fuel currently points the other
                # way; without this the flip exit below would close it on the
                # very next tick. UI display (exhaustion) is already updated above.
                opened_at = track.get('opened_at', 0)
                if opened_at and (now - opened_at) < MIN_HOLD_AFTER_OPEN_SEC:
                    with self._lock:
                        self._persist_state()
                    continue

                # exit 1a: CLEAR FLIP to the opposite side → close immediately.
                # This is a genuine reversal, not a data gap.
                if status == opposite:
                    self._close(symbol, mark or 0.0, reason='fuel_flipped',
                                is_real=is_real)
                    self._timers.pop(symbol, None)
                    continue

                # exit 1a-WAIT: WAIT verdict → close position immediately and clear timer
                # WAIT means unclear direction, position should not be held
                if settings.get('skip_wait_coins', False) and self._is_wait_verdict(symbol):
                    print(f"[FuelFilter] {symbol}: WAIT verdict detected — closing position")
                    self._close(symbol, mark or 0.0, reason='wait_verdict',
                                is_real=is_real)
                    self._timers.pop(symbol, None)
                    continue

                # exit 1b: FUEL FADE (status neutral/None) → only close after a
                # sustained grace period. Transient gaps must NOT close us.
                if status != side:
                    faded_since = track.get('faded_since')
                    if not faded_since:
                        with self._lock:
                            track['faded_since'] = now
                        # first observed fade — keep position, wait it out
                    elif (now - faded_since) >= FUEL_FADE_GRACE_SEC:
                        self._close(symbol, mark or 0.0, reason='fuel_faded',
                                    is_real=is_real)
                        self._timers.pop(symbol, None)
                        continue
                    # else: still inside grace window — hold
                else:
                    # fuel back to our side → reset the fade timer
                    if track.get('faded_since'):
                        with self._lock:
                            track.pop('faded_since', None)

                # exit 2: exhaustion reached (optional)
                if settings['use_potential_exit'] and exh is not None:
                    if exh >= settings['potential_threshold_pct']:
                        self._close(symbol, mark or 0.0, reason='potential_reached', is_real=is_real)
                        self._timers.pop(symbol, None)
                        continue

                with self._lock:
                    self._persist_state()
                continue

            # ---- no position: run the timer ----
            if status in ('LONG', 'SHORT'):
                # FF FLAG CHECK: timer runs only for FF-flagged coins.
                # All watchlist coins get scanned for stats, but only flagged ones
                # get timers (and show in the table).
                if symbol not in scan_list:
                    # Not FF-flagged → skip timer logic (but still collected stats above)
                    if symbol in self._timers:
                        self._timers.pop(symbol, None)
                    continue

                _mk = fuel.get('mark_price') if fuel else None
                t = self._timers.get(symbol)
                if not t or t.get('dir') != status:
                    # new status (or direction flip) → (re)start timer.
                    # Capture the price at timer start (for the anomaly table).
                    self._timers[symbol] = {'dir': status, 'since': now,
                                            'start_price': _mk}
                    t = self._timers[symbol]
                # Exhaustion for the timer row — shows how much room is left.
                exh = self._exhaustion(symbol, status)
                if exh is not None:
                    t['exhaustion'] = exh

                # NEW STRATEGY: timer runs continuously while fuel exists.
                # No auto-open. Table shows only timers held >= duration_minutes
                # (filtering happens in get_state). Timer keeps running until
                # fuel disappears (status becomes None/'WAIT').

                # ── Anomaly tracking: fuel held longer than anomaly_hours ──
                # Such coins move to their OWN table (excluded from the FF table
                # in get_state), survive fuel loss, and are user-deleted only.
                held = now - t.get('since', now)
                if held >= anomaly_sec:
                    a = self._anomalies.get(symbol)
                    if not a:
                        self._anomalies[symbol] = {
                            'symbol': symbol, 'dir': status,
                            'started_at': t.get('since', now),
                            'start_price': t.get('start_price') or _mk,
                            'holding': True,
                            'last_price': _mk, 'last_held_sec': int(held),
                            'ended_at': None, 'end_price': None,
                        }
                    else:
                        a['holding'] = True
                        a['dir'] = status
                        if _mk:
                            a['last_price'] = _mk
                        a['last_held_sec'] = int(held)
                        # Fuel came back after an earlier end → clear end marks.
                        a['ended_at'] = None
                        a['end_price'] = None
            else:
                # status off → reset any running timer
                if symbol in self._timers:
                    self._timers.pop(symbol, None)

        # Drop orphaned timers for symbols that are no longer eligible for a
        # timer (e.g. a funding coin after the 💰 scanner is turned off, or a
        # coin removed from the watchlist / scan-list). Without this their row
        # would linger in the ❤️ table forever since the loop above never visits
        # symbols outside `symbols`.
        with self._lock:
            for sym in list(self._timers.keys()):
                if sym not in scan_list and sym not in self._fuel_managed:
                    self._timers.pop(sym, None)

            # Safety net: record an anomaly for ANY timer past the threshold —
            # even symbols the main loop didn't visit this tick (e.g. FF-flagged
            # but out of the current watchlist). This guarantees a long-held coin
            # always moves to the anomalies table (and out of the FF table).
            for sym, t in self._timers.items():
                held = now - t.get('since', now)
                if held >= anomaly_sec and sym not in self._anomalies:
                    self._anomalies[sym] = {
                        'symbol': sym, 'dir': t.get('dir'),
                        'started_at': t.get('since', now),
                        'start_price': t.get('start_price'),
                        'holding': True,
                        'last_price': t.get('start_price'),
                        'last_held_sec': int(held),
                        'ended_at': None, 'end_price': None,
                    }

            # Anomaly end-detection: an anomaly still flagged "holding" but with
            # no active timer means its fuel just ended — freeze end time/price
            # but KEEP the row (user deletes it). Done after timer cleanup so
            # the timer set is final.
            active_timers = set(self._timers.keys())
            for sym, a in self._anomalies.items():
                if a.get('holding') and sym not in active_timers:
                    a['holding'] = False
                    if not a.get('ended_at'):
                        a['ended_at'] = now
                        a['end_price'] = a.get('last_price')

        # Store aggregate scan-stats for the panel header.
        with self._lock:
            self._scan_stats = {
                'targets': len(scan_targets),
                'scanned': _scanned_ok,
                'watchlist_total': len(watchlist),
                'max_exh_long': (round(_max_long, 1) if _max_long is not None else None),
                'max_exh_short': (round(_max_short, 1) if _max_short is not None else None),
                'exh_long_count': _cnt_long,
                'exh_short_count': _cnt_short,
            }
            self._persist_state()

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
        print("[FuelFilter] daemon started")

    def stop(self):
        self._stop.set()

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

    def _engine_tick(self):
        s = self.get_settings()
        if not s.get('start_engine_enabled') or not s.get('enabled'):
            return
        now = time.time()
        # BTC banner must be START (BTC ММ held ≥ start_signal_minutes).
        btc_t = self._timers.get('BTCUSDT')
        if not btc_t or btc_t.get('dir') not in ('LONG', 'SHORT'):
            return
        period = float(s.get('start_signal_minutes', 5) or 5) * 60
        if (now - btc_t.get('since', now)) < period:
            return
        direction = btc_t['dir']     # open the basket in the BANNER direction

        # Gather candidates from the chosen source tables. Only coins whose own
        # ММ direction matches the BTC banner direction — opening against it
        # would just be flip-closed by the manage branch (needless churn).
        dur = float(s.get('duration_minutes', 5)) * 60
        cands = set()
        with self._lock:
            if s.get('start_engine_use_timers', True):
                for sym, t in self._timers.items():
                    if sym == 'BTCUSDT' or sym in self._fuel_managed or sym in self._anomalies:
                        continue
                    if t.get('dir') != direction:
                        continue
                    if (now - t.get('since', now)) >= dur:
                        cands.add(sym)
            if s.get('start_engine_use_anomalies', True):
                for sym, a in self._anomalies.items():
                    if (a.get('holding') and a.get('dir') == direction
                            and sym not in self._fuel_managed):
                        cands.add(sym)
            funding = set(self._funding_syms)
        if not s.get('start_engine_include_funding', False):
            cands = {c for c in cands if c not in funding}
        if not cands:
            return

        for sym in cands:
            if sym in self._fuel_managed:
                continue   # already managed (no duplicate)
            # Skip if TM already holds this coin (real OR paper) — don't dup.
            if self._tm_has_position(sym, True) or self._tm_has_position(sym, False):
                continue
            fuel = self._fuel_dir(sym)
            if not fuel or not fuel.get('mark_price'):
                continue
            try:
                # _open routes through TM (bypass gates, like Alerts but ignoring
                # the LONG/SHORT master + SMC filters) AND registers the position
                # in _fuel_managed so exhaustion-exit + control manage it.
                opened = self._open(sym, direction, fuel, s)
                if opened:
                    print(f"[FF-Engine] opened {direction} {sym} (BTC START)")
            except Exception as e:
                print(f"[FF-Engine] open error {sym}: {e}")

    def get_state(self) -> Dict:
        """Snapshot for the UI: settings + live timers + active tracking.
        NEW STRATEGY: shows only timers held >= duration_minutes (threshold).
        No auto-open. Position management (close) is optional via setting."""
        with self._lock:
            settings = self.get_settings()
            duration_sec = settings['duration_minutes'] * 60
            now = time.time()
            timers = []
            for sym, t in self._timers.items():
                if sym in self._fuel_managed:
                    continue  # already opened, don't show timer
                if sym in self._anomalies:
                    continue  # moved to the anomalies table — hide here
                held = now - t.get('since', now)
                # FILTER: only show timers that exceeded the threshold
                if held < duration_sec:
                    continue
                timers.append({
                    'symbol': sym, 'dir': t.get('dir'),
                    'held_sec': int(held),
                    'exhaustion': t.get('exhaustion'),
                    # Flag rows that come from the 💰 Funding Rate Scanner so the
                    # UI can distinguish them (monitor-only).
                    'funding': sym in self._funding_syms,
                    # progress_pct removed - no longer needed
                })
            # BTC is now a normal row (its dedicated visual lives in the
            # START/STOP banner). No pinning.
            all_timers = sorted(timers, key=lambda x: -x['held_sec'])
            bs = self._btc_state or {}
            scan_list = self.get_scan_list()
            # BTC START/STOP signal: a progress that counts up while BTC holds
            # ММ, reaching START at start_signal_minutes; STOP when no BTC timer.
            ssm = float(settings.get('start_signal_minutes', 5) or 5)
            period_sec = ssm * 60
            # Read BTC held DIRECTLY from the live timer (same source the table
            # row uses) so the banner and the table never disagree. _btc_state
            # is only a per-tick snapshot and can lag by up to a scan cycle.
            btc_t = self._timers.get('BTCUSDT')
            if btc_t and btc_t.get('dir') in ('LONG', 'SHORT'):
                b_dir = btc_t.get('dir')
                b_held = now - btc_t.get('since', now)
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
            }
            # Anomalies — own table. Live "held" while still holding fuel,
            # frozen at end otherwise.
            anomalies = []
            for sym, a in self._anomalies.items():
                holding = bool(a.get('holding'))
                if holding:
                    t = self._timers.get(sym)
                    held = int(now - t.get('since', a.get('started_at', now))) \
                        if t else int(a.get('last_held_sec', 0))
                else:
                    held = int(a.get('last_held_sec', 0))
                anomalies.append({
                    'symbol': sym,
                    'dir': a.get('dir'),
                    'holding': holding,
                    'held_sec': held,
                    'started_at': a.get('started_at'),
                    'start_price': a.get('start_price'),
                    'ended_at': a.get('ended_at'),
                    'end_price': a.get('end_price'),
                    'current_price': a.get('last_price'),
                    'funding': sym in self._funding_syms,
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
            'scan_list': scan_list,
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

    def delete_timer(self, symbol: str) -> bool:
        """Delete a specific timer. Returns True if deleted, False if not found."""
        symbol = symbol.upper()
        with self._lock:
            if symbol in self._timers:
                self._timers.pop(symbol)
                self._persist_state()
                print(f"[FuelFilter] Timer deleted: {symbol}")
                return True
            return False

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

        if not timer:
            return {'ok': False, 'reason': f'No active timer for {symbol}'}

        # Check if already holding a position
        if symbol in self._fuel_managed:
            return {'ok': False, 'reason': f'Already holding position for {symbol}'}

        # Direction comes from the timer (set when the timer started). For a
        # manual force-open we don't require fuel to STILL point that way —
        # the operator explicitly wants in. We only need a valid entry price.
        # NOTE: timers store the direction under 'dir' (see tick loop /
        # get_state), NOT 'side' — reading 'side' raised KeyError: 'side'.
        side = timer.get('dir')
        if side not in ('LONG', 'SHORT'):
            return {'ok': False, 'reason': f'Таймер {symbol} без напрямку'}

        # Use the SAME fuel helper the tick loop uses. It returns
        # {dir, mark_price, status} and has its own market_data fallback for
        # mark_price, so even with neutral/faded fuel we still get a price.
        fuel = self._fuel_dir(symbol)
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
            # Remove timer since we opened successfully
            with self._lock:
                self._timers.pop(symbol, None)
                self._persist_state()
            return {'ok': True, 'reason': f'Позицію {side} відкрито вручну', 'opened': True}
        else:
            # _open() returns False on its own gates (exhaustion too high,
            # WAIT verdict if skip_wait on, no entry price, qty below min, …).
            return {'ok': False, 'opened': False,
                    'reason': 'Відкриття відхилено (виснаженість/вердикт/розмір — див. лог)'}

    def clear_all_timers(self) -> int:
        """Clear all timers. Returns count of timers cleared."""
        with self._lock:
            count = len(self._timers)
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
