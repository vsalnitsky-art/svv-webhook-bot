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
  • Exhaustion ("Потенціал") needs klines, so it is computed ONLY for coins
    that already have an OPEN fuel position (a handful at most), never for
    the whole watchlist. Results are cached per-symbol with a short TTL.
  • The scan cycle is 30 s — twice per liq-map refresh, which is plenty.

Persistence / recovery:
  • All live state (per-coin timers, open positions, recent closes) and all
    settings are persisted to the DB as JSON settings, mirroring how the
    Trade Manager persists its book. On boot the daemon restores everything
    so timers and positions survive restarts/redeploys.

Execution modes (DB-persisted `mode`):
  • 'paper' (default) — virtual book only, PnL marked to live price. No real
    orders, zero risk while the strategy is validated.
  • 'real'            — delegates the actual open/close to the existing
    TradeManager (manual_open / close) so all real-order logic stays in one
    place; this module only decides WHEN.
"""

import time
import threading
from typing import Optional, Callable, Dict, List

CYCLE_SECS = 30                 # scan cadence (twice per liq-map refresh)
EXHAUSTION_TTL = 120            # cache exhaustion per symbol for 2 min
FUEL_LONG_THR = 0.1            # fuel_dir > +0.1 → LONG bias
FUEL_SHORT_THR = -0.1          # fuel_dir < -0.1 → SHORT bias
CLOSED_LIMIT = 100             # keep last N closes for the UI

_DB_SETTINGS = 'fuel_filter_settings'
_DB_STATE = 'fuel_filter_state'

DEFAULT_SETTINGS = {
    'enabled': False,
    'duration_minutes': 5,        # status must hold this long before open
    'potential_threshold_pct': 95,  # exhaustion ≥ this → close
    'use_potential_exit': True,   # toggle the exhaustion exit on/off
    'mode': 'paper',              # 'paper' | 'real'
    'max_positions': 0,           # 0 = unlimited
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
        self._positions: Dict[str, Dict] = {}   # symbol -> position dict
        self._closed: List[Dict] = []           # recent closes (newest last)
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
        s['use_potential_exit'] = bool(s.get('use_potential_exit', True))
        s['enabled'] = bool(s.get('enabled', False))
        s['mode'] = 'real' if str(s.get('mode', 'paper')).lower() == 'real' else 'paper'
        s['max_positions'] = max(0, int(s.get('max_positions', 0) or 0))
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
    # state persistence
    # ------------------------------------------------------------------
    def _load_state(self):
        try:
            st = self._db.get_setting(_DB_STATE, {}) or {}
        except Exception:
            st = {}
        if isinstance(st, dict):
            self._timers = st.get('timers', {}) or {}
            self._positions = st.get('positions', {}) or {}
            self._closed = st.get('closed', []) or []
            if self._positions:
                print(f"[FuelFilter] restored {len(self._positions)} open "
                      f"position(s), {len(self._timers)} timer(s) from DB")

    def _persist_state(self):
        try:
            self._db.set_setting(_DB_STATE, {
                'timers': self._timers,
                'positions': self._positions,
                'closed': self._closed[-CLOSED_LIMIT:],
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
            if lm is None:
                return None
            try:
                prof = self._db.get_setting('liqmap_decay_profile', 'tori')
            except Exception:
                prof = 'tori'
            lst = lm.get_state(symbol, lookback_hours=24, profile=prof)
            mark = lst.get('mark_price')
            if not mark:
                return None
            fa = fb = 0.0
            for lev in (lst.get('levels') or []):
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

    def _exhaustion(self, symbol: str, side: str) -> Optional[float]:
        """Move exhaustion (0..100) for an OPEN position only. Cached with a
        short TTL so we fetch klines at most every EXHAUSTION_TTL seconds per
        coin. Returns None on insufficient data."""
        now = time.time()
        c = self._exh_cache.get(symbol)
        if c and c.get('side') == side and (now - c.get('ts', 0)) < EXHAUSTION_TTL:
            return c.get('exhaustion')
        try:
            from detection.market_data import get_market_data
            from detection.move_potential import analyze_move_potential
            md = get_market_data()
            kl = md.fetch_klines(symbol, interval='15m', limit=200) if md else None
            bars_per_day = 96
            if not kl or len(kl) < 96:
                kl = md.fetch_klines(symbol, interval='1h', limit=200) if md else None
                bars_per_day = 24
            if not kl or len(kl) < 20:
                return None
            mp = analyze_move_potential(side=side, klines=kl,
                                        bars_per_day=bars_per_day)
            exh = mp.get('exhaustion') if mp and mp.get('ok') else None
            self._exh_cache[symbol] = {'ts': now, 'exhaustion': exh, 'side': side}
            return exh
        except Exception as e:
            print(f"[FuelFilter] exhaustion calc error {symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # open / close (paper native, real delegated to TradeManager)
    # ------------------------------------------------------------------
    def _open(self, symbol: str, side: str, fuel: Dict, settings: Dict):
        mode = settings.get('mode', 'paper')
        entry_price = fuel.get('mark_price')
        if not entry_price or entry_price <= 0:
            return
        pos = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'opened_at': time.time(),
            'fuel_dir_at_entry': fuel.get('dir'),
            'mode': mode,
            'is_paper': mode != 'real',
            'last_price': entry_price,
            'pnl_pct': 0.0,
            'exhaustion': None,
        }
        if mode == 'real':
            tm = self._get_tm() if self._get_tm else None
            if not tm:
                print(f"[FuelFilter] real mode but no TradeManager — skip {symbol}")
                return
            try:
                res = tm.manual_open(symbol, side)
            except Exception as e:
                print(f"[FuelFilter] real open error {symbol}: {e}")
                return
            if not res or not res.get('ok'):
                reason = (res or {}).get('reason', 'unknown')
                print(f"[FuelFilter] real open rejected {symbol}: {reason}")
                return
            rp = res.get('entry_price') or res.get('position', {}).get('entry_price')
            if rp:
                pos['entry_price'] = rp
                pos['last_price'] = rp
        with self._lock:
            self._positions[symbol] = pos
            self._persist_state()
        print(f"[FuelFilter] OPEN {mode} {side} {symbol} @ {pos['entry_price']} "
              f"(fuel {fuel.get('dir')})")

    def _close(self, symbol: str, exit_price: float, reason: str):
        with self._lock:
            pos = self._positions.pop(symbol, None)
        if not pos:
            return
        if pos.get('mode') == 'real':
            tm = self._get_tm() if self._get_tm else None
            if tm:
                try:
                    # Route through whichever close helper the TM exposes.
                    if hasattr(tm, 'manual_close'):
                        tm.manual_close(symbol, reason=reason)
                    elif hasattr(tm, 'close_position'):
                        tm.close_position(symbol, reason=reason)
                    elif hasattr(tm, '_close_position'):
                        tm._close_position(symbol, exit_price, reason)
                except Exception as e:
                    print(f"[FuelFilter] real close error {symbol}: {e}")
        side = pos.get('side')
        entry = pos.get('entry_price') or exit_price
        if entry and exit_price:
            sign = 1 if side == 'LONG' else -1
            pnl_pct = (exit_price - entry) / entry * 100.0 * sign
        else:
            pnl_pct = 0.0
        rec = {
            'symbol': symbol, 'side': side,
            'entry_price': entry, 'exit_price': exit_price,
            'opened_at': pos.get('opened_at'),
            'closed_at': time.time(),
            'pnl_pct': round(pnl_pct, 3),
            'reason': reason,
            'mode': pos.get('mode', 'paper'),
        }
        with self._lock:
            self._closed.append(rec)
            self._closed = self._closed[-CLOSED_LIMIT:]
            self._persist_state()
        print(f"[FuelFilter] CLOSE {symbol} @ {exit_price} "
              f"({pnl_pct:+.2f}%) reason={reason}")

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

    def _tick(self):
        settings = self.get_settings()
        self._last_tick_ts = time.time()
        if not settings.get('enabled'):
            return
        try:
            watchlist = self._get_watchlist() or []
        except Exception:
            watchlist = []
        # union of watchlist + symbols we already hold (so we keep managing a
        # position even if its coin drops off the watchlist)
        symbols = list(dict.fromkeys(
            [s.upper() for s in watchlist] + list(self._positions.keys())))

        duration_sec = settings['duration_minutes'] * 60
        now = time.time()

        for symbol in symbols:
            fuel = self._fuel_dir(symbol)
            status = fuel.get('status') if fuel else None
            has_pos = symbol in self._positions

            # ---- manage an OPEN position ----
            if has_pos:
                pos = self._positions[symbol]
                mark = fuel.get('mark_price') if fuel else None
                if mark:
                    pos['last_price'] = mark
                    sign = 1 if pos['side'] == 'LONG' else -1
                    pos['pnl_pct'] = round(
                        (mark - pos['entry_price']) / pos['entry_price']
                        * 100.0 * sign, 3)
                # exit 1: fuel status changed / disappeared / flipped
                if status != pos['side']:
                    self._close(symbol, mark or pos['entry_price'],
                                reason='fuel_status_changed')
                    self._timers.pop(symbol, None)
                    continue
                # exit 2: exhaustion reached (optional)
                if settings['use_potential_exit']:
                    exh = self._exhaustion(symbol, pos['side'])
                    if exh is not None:
                        pos['exhaustion'] = exh
                        if exh >= settings['potential_threshold_pct']:
                            self._close(symbol, mark or pos['entry_price'],
                                        reason='potential_reached')
                            self._timers.pop(symbol, None)
                            continue
                with self._lock:
                    self._persist_state()
                continue

            # ---- no position: run the timer ----
            if status in ('LONG', 'SHORT'):
                t = self._timers.get(symbol)
                if not t or t.get('dir') != status:
                    # new status (or direction flip) → (re)start timer
                    self._timers[symbol] = {'dir': status, 'since': now}
                    continue
                held = now - t.get('since', now)
                if held >= duration_sec:
                    # capacity check
                    maxp = settings['max_positions']
                    if maxp and len(self._positions) >= maxp:
                        continue
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
        """Snapshot for the UI: settings + live timers + open positions +
        recent closes."""
        with self._lock:
            settings = self.get_settings()
            duration_sec = settings['duration_minutes'] * 60
            now = time.time()
            timers = []
            for sym, t in self._timers.items():
                if sym in self._positions:
                    continue
                held = now - t.get('since', now)
                timers.append({
                    'symbol': sym, 'dir': t.get('dir'),
                    'held_sec': int(held),
                    'progress_pct': (round(min(100.0, held / duration_sec * 100.0), 1)
                                     if duration_sec > 0 else 100.0),
                })
            positions = []
            for sym, p in self._positions.items():
                positions.append({
                    'symbol': sym, 'side': p.get('side'),
                    'entry_price': p.get('entry_price'),
                    'last_price': p.get('last_price'),
                    'pnl_pct': p.get('pnl_pct'),
                    'exhaustion': p.get('exhaustion'),
                    'fuel_dir_at_entry': p.get('fuel_dir_at_entry'),
                    'opened_at': p.get('opened_at'),
                    'mode': p.get('mode', 'paper'),
                    'age_sec': int(now - (p.get('opened_at') or now)),
                })
            closed = list(reversed(self._closed[-50:]))
        return {
            'ok': True,
            'settings': settings,
            'running': bool(self._thread and self._thread.is_alive()),
            'last_tick_ts': self._last_tick_ts,
            'timers': sorted(timers, key=lambda x: -x['held_sec']),
            'positions': positions,
            'closed': closed,
            'active_symbols': list(self._positions.keys()),
        }

    def active_symbols(self) -> List[str]:
        """Symbols with an open fuel position — used to draw the ❤ marker in
        the watchlist."""
        with self._lock:
            return list(self._positions.keys())

    def clear_closed(self):
        with self._lock:
            self._closed = []
            self._persist_state()


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
