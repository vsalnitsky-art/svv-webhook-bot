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
FUEL_LONG_THR = 0.1            # fuel_dir > +0.1 → LONG bias
FUEL_SHORT_THR = -0.1          # fuel_dir < -0.1 → SHORT bias
CLOSED_LIMIT = 100             # keep last N closes for the UI
# Grace period before closing on FUEL FADE (status → neutral/None). Without
# this, a single transient liq-map data gap or a brief dip into the ±0.1
# neutral zone would slam the position shut the very next tick. We only honour
# a *sustained* loss of fuel. A clear FLIP to the opposite side still closes
# immediately (that's a real reversal, not a data gap).
FUEL_FADE_GRACE_SEC = 180      # 3 min of continuous neutral before close

_DB_SETTINGS = 'fuel_filter_settings'
_DB_STATE = 'fuel_filter_state'
_DB_SCAN_LIST = 'fuel_filter_scan_list'   # which symbols FF is allowed to scan

DEFAULT_SETTINGS = {
    'enabled': False,
    'duration_minutes': 5,        # status must hold this long before open
    'potential_threshold_pct': 95,  # exhaustion ≥ this → close
    'use_potential_exit': True,   # toggle the exhaustion exit on/off
    'max_exhaustion_pct': 75,     # don't open if exhaustion > this (entry filter)
    'skip_wait_coins': False,     # don't open if coin verdict is WAIT
}


class FuelFilterDaemon:
    def __init__(self, db, get_trade_manager: Callable,
                 get_watchlist: Callable):
        self._db = db
        self._get_tm = get_trade_manager
        self._get_watchlist = get_watchlist
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()
        # live state (restored from DB on boot)
        self._timers: Dict[str, Dict] = {}      # symbol -> {dir, since}
        # Tracking dict: which symbols fuel filter opened (not full position data)
        self._fuel_managed: Dict[str, Dict] = {}  # symbol -> {opened_at, side, fuel_dir}
        # exhaustion cache: symbol -> {ts, exhaustion, side}
        self._exh_cache: Dict[str, Dict] = {}
        self._last_tick_ts = 0
        self._load_state()

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
            if self._fuel_managed:
                print(f"[FuelFilter] restored {len(self._fuel_managed)} tracked "
                      f"position(s), {len(self._timers)} timer(s) from DB")

    def _persist_state(self):
        try:
            self._db.set_setting(_DB_STATE, {
                'timers': self._timers,
                'fuel_managed': self._fuel_managed,
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

    def _is_wait_verdict(self, symbol: str) -> bool:
        """Check if the given symbol has a WAIT verdict (unclear direction).
        Returns True if verdict is WAIT, False otherwise (or on error)."""
        try:
            # Import compute_bias from flask_app (shared bias computation)
            from web.flask_app import compute_bias
            verdict_data = compute_bias(self._db, symbol, wl=None)
            verdict = verdict_data.get('verdict', 'WAIT')
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

    def _exhaustion(self, symbol: str, side: str) -> Optional[float]:
        """Move exhaustion (0..100) for an OPEN position only. Uses the SAME
        data source as the dashboard's "Потенціал LONG/SHORT" panel (scanner's
        cached klines) so percentages match. Cached with a short TTL. Returns
        None on insufficient data."""
        now = time.time()
        c = self._exh_cache.get(symbol)
        if c and c.get('side') == side and (now - c.get('ts', 0)) < EXHAUSTION_TTL:
            return c.get('exhaustion')
        try:
            from detection.move_potential import analyze_move_potential
            # Use scanner's cached klines (same source as dashboard panel)
            mp_klines = None
            bars_per_day = 96
            try:
                from detection.smc_scanner import get_smc_scanner
                _sc = get_smc_scanner()
                if _sc:
                    mp_klines = _sc._get_cached_klines(symbol)
            except Exception:
                pass
            # Fallback to 1h klines if scanner cache unavailable
            if not mp_klines or len(mp_klines) < 20:
                try:
                    from detection.market_data import get_market_data
                    md = get_market_data()
                    if md:
                        mp_klines = md.fetch_klines(symbol, interval='1h', limit=200)
                        bars_per_day = 24
                except Exception:
                    pass
            if not mp_klines or len(mp_klines) < 20:
                return None
            mp = analyze_move_potential(side=side, klines=mp_klines,
                                        bars_per_day=bars_per_day)
            exh = mp.get('exhaustion') if mp and mp.get('ok') else None
            self._exh_cache[symbol] = {'ts': now, 'exhaustion': exh, 'side': side}
            return exh
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
            return

        # CHECK EXHAUSTION BEFORE OPENING: don't enter exhausted moves
        max_exh = settings.get('max_exhaustion_pct', 75)
        exh = self._exhaustion(symbol, side)
        if exh is not None and exh > max_exh:
            print(f"[FuelFilter] {symbol}: exhaustion {exh:.1f}% > {max_exh}% — "
                  f"rejecting open (too exhausted)")
            return

        # CHECK WAIT VERDICT: if enabled, don't open coins in WAIT state
        if settings.get('skip_wait_coins', False):
            if self._is_wait_verdict(symbol):
                print(f"[FuelFilter] {symbol}: verdict is WAIT — "
                      f"rejecting open (skip_wait_coins enabled)")
                return

        tm = self._get_tm() if self._get_tm else None
        if not tm:
            print(f"[FuelFilter] {symbol}: no TradeManager available — skip open")
            return

        # Check which mode is active (TM real or Test Mode paper)
        tm_settings = tm.get_settings() if hasattr(tm, 'get_settings') and callable(tm.get_settings) else {}
        tm_enabled = tm_settings.get('enabled', False)
        test_mode = tm_settings.get('test_mode', True)

        print(f"[FuelFilter] {symbol}: TM settings: enabled={tm_enabled}, test_mode={test_mode}")

        if not tm_enabled and not test_mode:
            print(f"[FuelFilter] {symbol}: neither TM nor Test Mode enabled — skip open")
            return

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
                        return
                else:
                    # Paper position via Test Mode (shadow) — bypass LONG/SHORT gates
                    # Fuel Filter operates independently from manual trade signals
                    if hasattr(tm, '_open_shadow') and callable(tm._open_shadow):
                        print(f"[FuelFilter] {symbol}: calling tm._open_shadow({symbol}, {side}, {entry_price}, 'fuel_filter', bypass_gates=True)")
                        tm._open_shadow(symbol, side, entry_price, 'fuel_filter', bypass_gates=True)
                        print(f"[FuelFilter] {symbol}: _open_shadow call completed")
                    else:
                        print(f"[FuelFilter] {symbol}: Test Mode enabled but _open_shadow not available")
                        return
            except Exception as e:
                print(f"[FuelFilter] {symbol}: open error ({mode}): {e}")
                import traceback
                traceback.print_exc()
                return

            # VERIFY the open actually landed before tracking. _open_shadow can
            # silently return early (e.g. LONG/SHORT entries gated off), and a
            # real open can be rejected at the order layer. If nothing landed,
            # do NOT mark _fuel_managed — otherwise the timer disappears but no
            # position exists ("trades vanish, nothing happens"). Leaving it
            # untracked lets the timer keep running and retry next cycle.
            has_pos = self._tm_has_position(symbol, is_real)
            print(f"[FuelFilter] {symbol}: verification check — _tm_has_position={has_pos}")
            if not has_pos:
                print(f"[FuelFilter] {symbol}: open did not land in TM "
                      f"(gated/rejected) — NOT tracking, timer continues")
                return

        # Track that fuel filter opened this position (for exit condition monitoring)
        with self._lock:
            self._fuel_managed[symbol] = {
                'opened_at': time.time(),
                'side': side,
                'fuel_dir': fuel.get('dir'),
                'mode': mode,
            }
            self._persist_state()
        print(f"[FuelFilter] OPEN SUCCESS: {mode} {side} {symbol} @ {entry_price} "
              f"(fuel {fuel.get('dir')}, exhaustion {exh:.1f}% if exh else 'N/A')")

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

        # SCAN-LIST (whitelist): only scan coins the operator explicitly marked
        # with the "FF" flag — keeps load off the bot instead of scanning every
        # coin on the board. Empty list = scan nothing (pure opt-in). We still
        # always manage coins we already opened (so an open position is never
        # abandoned even if it's later un-flagged).
        scan_list = set(self.get_scan_list())
        scan_targets = [s for s in watchlist if s in scan_list]
        symbols = list(dict.fromkeys(
            scan_targets + list(self._fuel_managed.keys())))

        # CRITICAL: register scanned symbols with the liq map so it scans them.
        # Otherwise only BTC/ETH + UI-viewed coins have fuel data. We only
        # register the small FF whitelist now (not the whole watchlist) — the
        # whole point of the FF flag is to keep this set small.
        self._register_with_liqmap(symbols)

        # Get TM settings to know which mode positions live in
        tm = self._get_tm() if self._get_tm else None
        tm_settings = tm.get_settings() if tm and hasattr(tm, 'get_settings') else {}
        tm_enabled = tm_settings.get('enabled', False)

        duration_sec = settings['duration_minutes'] * 60
        now = time.time()

        for symbol in symbols:
            fuel = self._fuel_dir(symbol)
            status = fuel.get('status') if fuel else None
            fuel_managed = symbol in self._fuel_managed

            # ---- manage positions fuel filter opened ----
            if fuel_managed:
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
                # WAIT check: if skip_wait_coins enabled and verdict is WAIT, reset timer
                if settings.get('skip_wait_coins', False) and self._is_wait_verdict(symbol):
                    print(f"[FuelFilter] {symbol}: WAIT verdict — resetting timer")
                    if symbol in self._timers:
                        self._timers.pop(symbol, None)
                    continue

                t = self._timers.get(symbol)
                if not t or t.get('dir') != status:
                    # new status (or direction flip) → (re)start timer
                    self._timers[symbol] = {'dir': status, 'since': now}
                    t = self._timers[symbol]
                # Exhaustion for the timer row — lets the operator SEE the
                # coin's state (how much room is left) BEFORE a position opens.
                exh = self._exhaustion(symbol, status)
                if exh is not None:
                    t['exhaustion'] = exh

                # ENTRY EXHAUSTION GATE: if exhaustion already exceeds the max
                # entry threshold, the position can NEVER open (it would be
                # rejected in _open). Running the timer is pointless and confusing
                # ("100% exhaustion but still counting"). Reset it so the coin is
                # not shown as a running timer. It restarts fresh if exhaustion
                # later drops back below the threshold.
                max_exh = settings.get('max_exhaustion_pct', 75)
                if exh is not None and exh > max_exh:
                    print(f"[FuelFilter] {symbol}: exhaustion {exh:.1f}% > {max_exh}% "
                          f"— resetting timer (too exhausted to enter)")
                    self._timers.pop(symbol, None)
                    continue

                held = now - t.get('since', now)
                if held >= duration_sec:
                    self._open(symbol, status, fuel, settings)
            else:
                # status off → reset any running timer
                if symbol in self._timers:
                    self._timers.pop(symbol, None)

        with self._lock:
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
        print("[FuelFilter] daemon started")

    def stop(self):
        self._stop.set()

    def get_state(self) -> Dict:
        """Snapshot for the UI: settings + live timers + active tracking.
        Actual position data lives in TM/Test Mode and is shown in their tables."""
        with self._lock:
            settings = self.get_settings()
            duration_sec = settings['duration_minutes'] * 60
            now = time.time()
            timers = []
            for sym, t in self._timers.items():
                if sym in self._fuel_managed:
                    continue  # already opened, don't show timer
                held = now - t.get('since', now)
                timer_data = {
                    'symbol': sym, 'dir': t.get('dir'),
                    'held_sec': int(held),
                    'exhaustion': t.get('exhaustion'),
                    'progress_pct': (round(min(100.0, held / duration_sec * 100.0), 1)
                                     if duration_sec > 0 else 100.0),
                }
                # Attach indicators (forecast + fuel) for UI display
                indicators = self.get_coin_indicators(sym)
                timer_data['indicators'] = indicators
                timers.append(timer_data)
            scan_list = self.get_scan_list()
        return {
            'ok': True,
            'settings': settings,
            'running': bool(self._thread and self._thread.is_alive()),
            'last_tick_ts': self._last_tick_ts,
            'timers': sorted(timers, key=lambda x: -x['held_sec']),
            'active_symbols': list(self._fuel_managed.keys()),
            'tracked_count': len(self._fuel_managed),
            'scan_list': scan_list,
        }

    def active_symbols(self) -> List[str]:
        """Symbols with an open fuel-managed position — used to draw the ❤ marker
        in the watchlist."""
        with self._lock:
            return list(self._fuel_managed.keys())

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

    def clear_all_timers(self) -> int:
        """Clear all timers. Returns count of timers cleared."""
        with self._lock:
            count = len(self._timers)
            self._timers.clear()
            self._persist_state()
            print(f"[FuelFilter] Cleared {count} timers")
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
