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

Two modes (DB key sm_auto_gate_mode):
  'simple' (legacy) — mirrors the raw verdict (LONG/SHORT/WAIT) 1:1.
  'smart'  (new)    — runs the smart_direction algorithm that considers
                       4H trend exhaustion + confluence for optimal direction.

DB keys:
  sm_auto_gate_enabled : 'true' | 'false'
  sm_auto_gate_mode    : 'simple' | 'smart'  (default 'simple' for compat)
  sm_bias_symbol       : symbol for the verdict (default BTCUSDT)
  sm_wl_consensus_cache: JSON {src, n_long, n_short, n_flat, ts} (frontend)
"""

import time
import json
import threading
from typing import Optional, Callable

CYCLE_SECS = 60          # match the UI refresh cadence
WAIT_HYSTERESIS_DEFAULT = 3   # consecutive WAIT ticks before close-on-WAIT fires
WL_CACHE_TTL = 300       # frontend consensus older than 5 min is ignored

_DB_ENABLED = 'sm_auto_gate_enabled'
_DB_MODE = 'sm_auto_gate_mode'
_DB_SYMBOL = 'sm_bias_symbol'
_DB_WL_CACHE = 'sm_wl_consensus_cache'
_DB_CLOSE_ON_WAIT = 'sm_auto_gate_close_on_wait'
_DB_TREND_ALERT = 'sm_4h_trend_alert_enabled'
_DB_TREND_ALERT_1H = 'sm_1h_trend_alert_enabled'
_DB_WAIT_HYSTERESIS = 'sm_auto_gate_wait_hysteresis'


class AutoGateDaemon:
    def __init__(self, db, compute_bias: Callable, get_trade_manager: Callable):
        self._db = db
        self._compute_bias = compute_bias          # (symbol, wl_consensus) -> dict
        self._get_tm = get_trade_manager
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last = {'verdict': None, 'applied_at': 0, 'ts': 0}
        self._lock = threading.Lock()
        # Hysteresis state for close-on-WAIT: count consecutive WAIT ticks so
        # a brief LONG→WAIT→LONG flicker doesn't liquidate the whole book.
        self._wait_streak = 0
        self._wait_closed_this_streak = False
        # Last known trend per (symbol, tf), for change detection → Telegram.
        self._last_trend = {}

    def is_trend_alert(self) -> bool:
        """Telegram alert on 4H trend change (independent of auto-gate)."""
        return str(self._db.get_setting(_DB_TREND_ALERT, 'false')).lower() in ('true', '1')

    def set_trend_alert(self, on: bool):
        self._db.set_setting(_DB_TREND_ALERT, 'true' if on else 'false')
        if on:
            self.start()

    def is_trend_alert_1h(self) -> bool:
        """Telegram alert on 1H trend change."""
        return str(self._db.get_setting(_DB_TREND_ALERT_1H, 'false')).lower() in ('true', '1')

    def set_trend_alert_1h(self, on: bool):
        self._db.set_setting(_DB_TREND_ALERT_1H, 'true' if on else 'false')
        if on:
            self.start()

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

    def get_mode(self) -> str:
        """'simple' (legacy raw verdict mirror) or 'smart' (new algorithm)."""
        return (self._db.get_setting(_DB_MODE, 'simple') or 'simple').lower()

    def set_mode(self, mode: str):
        """Set 'simple' or 'smart'. Changes take effect on next tick."""
        m = (mode or 'simple').lower()
        if m not in ('simple', 'smart'):
            m = 'simple'
        self._db.set_setting(_DB_MODE, m)
        # Immediate apply if the gate is already running
        if self.is_enabled():
            try:
                self._tick()
            except Exception as e:
                print(f"[AutoGate] mode-change apply error: {e}")

    def is_close_on_wait(self) -> bool:
        return str(self._db.get_setting(_DB_CLOSE_ON_WAIT, 'false')).lower() in ('true', '1')

    def set_close_on_wait(self, on: bool):
        self._db.set_setting(_DB_CLOSE_ON_WAIT, 'true' if on else 'false')

    def wait_hysteresis(self) -> int:
        """How many consecutive WAIT ticks must occur before close-on-WAIT
        fires. 1 = close on the first WAIT (old behaviour); higher values
        ride out brief WAIT flickers."""
        try:
            v = int(self._db.get_setting(_DB_WAIT_HYSTERESIS,
                                         str(WAIT_HYSTERESIS_DEFAULT)))
            return max(1, min(20, v))
        except Exception:
            return WAIT_HYSTERESIS_DEFAULT

    def set_wait_hysteresis(self, n: int):
        try:
            self._db.set_setting(_DB_WAIT_HYSTERESIS, str(max(1, min(20, int(n)))))
        except Exception:
            pass

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
    def _apply(self, verdict: str, verdict_data: Optional[Dict] = None):
        """Apply the direction gates. In 'simple' mode, `verdict` is mirrored
        1:1 (LONG/SHORT/WAIT). In 'smart' mode, `verdict_data` is run through
        the smart_direction algorithm which may produce different gates
        (e.g. WAIT verdict but 4H exhausted → opens reversal side)."""
        tm = self._get_tm()
        if not tm:
            return

        mode = self.get_mode()
        if mode == 'smart' and verdict_data:
            try:
                from detection.smart_direction import compute_smart_direction
                sd = compute_smart_direction(
                    verdict_data,
                    allow_early_reversal=True,
                    require_confidence=60,
                )
                allow_long = sd.get('allow_long', False)
                allow_short = sd.get('allow_short', False)
                reason = sd.get('reason', '')
                log_msg = f"smart: {sd.get('mode')} ({reason})"
            except Exception as e:
                print(f"[AutoGate] smart_direction error: {e} — falling back to simple")
                mode = 'simple'

        if mode == 'simple':
            # Legacy 1:1 verdict mirror
            if verdict == 'LONG':
                allow_long, allow_short = True, False
            elif verdict == 'SHORT':
                allow_long, allow_short = False, True
            else:
                allow_long, allow_short = False, False
            log_msg = f"simple: {verdict}"

        # Only POST when something actually changes (avoid log/DB churn)
        s = tm.get_settings() if hasattr(tm, 'get_settings') else {}
        cur_l = bool(s.get('allow_long_entries', True))
        cur_s = bool(s.get('allow_short_entries', True))
        if cur_l == allow_long and cur_s == allow_short:
            return
        tm.update_settings({'allow_long_entries': allow_long,
                            'allow_short_entries': allow_short})
        print(f"[AutoGate] {log_msg} → LONG={allow_long} SHORT={allow_short}")

    def _run(self):
        # small initial delay so other singletons finish booting
        self._stop.wait(8)
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as e:
                print(f"[AutoGate] tick error: {e}")
            self._stop.wait(CYCLE_SECS)

    def _tick(self):
        # Trend-change alert runs independently of the auto-gate enable flag,
        # so it must be checked even when auto-gate itself is off.
        gate_on = self.is_enabled()
        alert_4h = self.is_trend_alert()
        alert_1h = self.is_trend_alert_1h()
        if not gate_on and not alert_4h and not alert_1h:
            return
        symbol = self.get_symbol()
        wl = self._wl_consensus()
        verdict_data = self._compute_bias(symbol, wl)

        # --- Trend-change Telegram alerts (independent of gate) ---
        if alert_4h:
            try:
                self._check_trend_change(symbol, verdict_data, 'reversal', '4H')
            except Exception as e:
                print(f"[AutoGate] 4H trend-alert error: {e}")
        if alert_1h:
            try:
                self._check_trend_change(symbol, verdict_data, 'reversal_1h', '1H')
            except Exception as e:
                print(f"[AutoGate] 1H trend-alert error: {e}")

        if not gate_on:
            return
        self._gate_logic(symbol, verdict_data)

    def _check_trend_change(self, symbol: str, verdict_data: dict,
                            rev_key: str = 'reversal', tf: str = '4H'):
        """Detect a trend flip on the given TF vs last seen and notify Telegram."""
        rev = (verdict_data or {}).get(rev_key) or {}
        trend = rev.get('trend_4h')   # field name is generic in the module
        if trend not in ('LONG', 'SHORT'):
            return  # ignore NEUTRAL transitions (too noisy)
        state_key = f'{symbol}:{tf}'
        prev = self._last_trend.get(state_key)
        self._last_trend[state_key] = trend
        if prev is None or prev == trend:
            return  # first observation or no change
        # Real flip LONG↔SHORT
        try:
            from alerts.telegram_notifier import get_notifier
            tg = get_notifier()
            if tg:
                arrow = '🟢 ВГОРУ (LONG)' if trend == 'LONG' else '🔴 ВНИЗ (SHORT)'
                was = '🟢 LONG' if prev == 'LONG' else '🔴 SHORT'
                price = verdict_data.get('price')
                px = f"\nЦіна: {price}" if price else ''
                tg.send_message(
                    f"🔄 <b>Зміна {tf} тренду — {symbol}</b>\n"
                    f"Було: {was} → Стало: {arrow}{px}\n"
                    f"<i>Smart Money · Схильність до розвороту ({tf})</i>")
                print(f"[AutoGate] {tf} trend change {symbol}: {prev}→{trend}, alerted")
        except Exception as e:
            print(f"[AutoGate] trend telegram error: {e}")

    def _gate_logic(self, symbol: str, verdict_data: dict):
        verdict = verdict_data.get('verdict', 'WAIT')
        with self._lock:
            self._last = {'verdict': verdict, 'ts': time.time(),
                          'applied_at': time.time(), 'symbol': symbol,
                          'confidence': verdict_data.get('confidence', 0),
                          'mode': self.get_mode()}
        self._apply(verdict, verdict_data)
        # Close-on-WAIT with HYSTERESIS (see history): require N consecutive
        # WAIT ticks and close only ONCE per sustained streak.
        if verdict == 'WAIT':
            self._wait_streak += 1
            need = self.wait_hysteresis()
            if (self._wait_streak >= need
                    and not self._wait_closed_this_streak
                    and self.is_close_on_wait()):
                try:
                    tm = self._get_tm()
                    if tm and hasattr(tm, 'close_all_with_queue'):
                        res = tm.close_all_with_queue(reason='auto_gate_wait')
                        print(f"[AutoGate] WAIT sustained {self._wait_streak}× "
                              f"(≥{need}) → closed "
                              f"real={len(res.get('closed_real', []))} "
                              f"shadow={len(res.get('closed_shadow', []))}")
                        self._wait_closed_this_streak = True
                except Exception as e:
                    print(f"[AutoGate] close-on-WAIT error: {e}")
        else:
            if self._wait_streak:
                print(f"[AutoGate] WAIT streak reset by {verdict} "
                      f"(was {self._wait_streak})")
            self._wait_streak = 0
            self._wait_closed_this_streak = False

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
        return {'enabled': self.is_enabled(), 'mode': self.get_mode(),
                'symbol': self.get_symbol(),
                'close_on_wait': self.is_close_on_wait(),
                'trend_alert': self.is_trend_alert(),
                'trend_alert_1h': self.is_trend_alert_1h(),
                'wait_hysteresis': self.wait_hysteresis(),
                'wait_streak': self._wait_streak,
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
